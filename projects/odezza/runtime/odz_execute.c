/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "odz_internal.h"
#include <float.h>
static Session *session(OdzRuntime *r,OdzJob *j,uint32_t slots,size_t systems,uint64_t configs_per_system) {
    uint32_t module=j->options.module;
    if(j->options.auto_module) {
        /* Measured 5080 knee: short screens favor 32; substantial rollout work
         * benefits from 64, especially on one GPU. This controls module size,
         * not registers or thread shape; specialization checks those separately. */
        int long_rollout=j->rk4.steps_per_configuration>=32&&configs_per_system>=128;
        size_t parallel=(size_t)j->options.workers*(long_rollout?1:j->options.slots);
        size_t target=(systems+parallel-1)/parallel;
        uint32_t ceiling=long_rollout?64:32;
        if(ceiling>j->options.module)ceiling=j->options.module;
        /* Keep the dispatch in compiled templates: a one-system shape can
         * have its dispatch elided by NVRTC and become uninspectable. */
        module=2;
        while(module<target&&module<ceiling)module*=2;
    }
    size_t i;
    Session *s=NULL;
    OdrScoringOptions o= {
        0
    };
    for(i=0;i<ODZ_SESSIONS;i++)if(r->sessions[i].handle&&r->sessions[i].slots==slots&&r->sessions[i].module==module) {
        s=&r->sessions[i];
        break;
    }
    if(s) {
        s->used=++r->clock;
        return s;
    }
    for(i=0;i<ODZ_SESSIONS;i++)if(!r->sessions[i].handle) {
        s=&r->sessions[i];
        break;
    }
    if(!s) {
        s=&r->sessions[0];
        for(i=1;i<ODZ_SESSIONS;i++)if(r->sessions[i].used<s->used)s=&r->sessions[i];
        if(odr_scoring_destroy(s->handle)) {
            odz_fail(j,"could not retire scoring session");
            r->fatal=1;
            return NULL;
        }
        memset(s,0,sizeof(*s));
    }
    o.cache_directory=r->cache;
    o.shape.sm_version=r->sm;
    o.shape.state_count=o.shape.state_capacity=j->fixed->state_count;
    o.shape.constant_count=slots;
    o.shape.constant_capacity=(slots+7)/8*8;
    if(!o.shape.constant_capacity)o.shape.constant_capacity=1;
    if(o.shape.constant_capacity>253-2*j->fixed->state_count)o.shape.constant_capacity=253-2*j->fixed->state_count;
    o.shape.system_capacity=module;
    o.shape.shared_patch_capacity=64;
    o.shape.system_patch_capacity=j->options.patch;
    o.shape.worker_count=j->options.workers;
    o.shape.cubin_slots_per_worker=j->options.slots;
    if(odr_scoring_create(j->fixed,&o,&s->handle)) {
        odz_fail(j,"scoring preparation: %s",odr_scoring_error(s->handle));
        return NULL;
    }
    s->slots=slots;s->module=module;
    s->used=++r->clock;
    {
        OdrScoringStats stats;
        odr_scoring_stats(s->handle,&stats);
        pthread_mutex_lock(&j->mutex);
        j->cache_hits+=stats.cache_hits;
        j->cache_misses+=stats.cache_misses;
        j->template_evictions+=stats.cache_evictions;
        j->template_invalidations+=stats.cache_invalidations;
        j->template_oversized+=stats.cache_oversized;
        j->growths+=stats.capacity_growths;
        j->template_seconds+=stats.template_prepare_seconds+stats.prespecialize_seconds;
        j->nvrtc_seconds+=stats.nvrtc_seconds;
        if(module>j->devices[r->worker_index].module_capacity)j->devices[r->worker_index].module_capacity=module;
        if(!j->devices[r->worker_index].minimum_module_capacity||module<j->devices[r->worker_index].minimum_module_capacity)j->devices[r->worker_index].minimum_module_capacity=module;
        pthread_mutex_unlock(&j->mutex);
    }
    return s;
}

/* Separate archives preserve raw rows and exact numeric-distinct retention,
 * including tags attached after the originating page has been recycled. */
static void archive(OdzJob *j,Kept *k,int raw) {
    FamilyRun *f=&j->families[k->family];
    Kept **head;
    if(raw?!f->row_archive:!f->archive)return;
    head=raw?&f->row_archive[k->ast]:&f->archive[k->ast];
    Kept **p=head,**worst=NULL;
    size_t count=0;
    unsigned cap=raw?j->tag_row_k:j->tag_local_k;
    while(*p) {
        Kept *old=*p;
        if(old->permutation==k->permutation) {
            if(raw?old->bank==k->bank:odz_same_numeric(old,k)) {
                if(odz_rank_order(old,k)<=0)return;
                *p=raw?old->row_archive_next:old->archive_next;
                odz_keep_unref(j,old);
                continue;
            }
            ++count;
            if(!worst||odz_rank_order(*worst,old)<0)worst=p;
        }
        p=raw?&(*p)->row_archive_next:&(*p)->archive_next;
    }
    if(count>=cap) {
        Kept *old;
        if(!worst||odz_rank_order(*worst,k)<=0)return;
        old=*worst;
        *worst=raw?old->row_archive_next:old->archive_next;
        odz_keep_unref(j,old);
    }
    ++k->refs;
    if(raw)k->row_archive_next=*head;else k->archive_next=*head;
    *head=k;
}

int odz_score_tile(OdzRuntime *r,OdzJob *j,Page *page) {
    const OdrCandidate **items=NULL;
    size_t *indices=NULL,n=0,i,s,first=0,work_bytes=0,alignment=0;
    void *work=NULL,*packed=NULL;
    const OdezzaScoringSystem *systems;
    OdrBatch selected= {
        0
    };
    OdrCandidate *copies=NULL;
    OdrError error;
    OdezzaScoringLaunch launch= {
        0
    };
    OdezzaScoringRunReport report={0};
    OdezzaScoreReductionSize reduction;
    OdezzaScoreReductionReport rr;
    OdrScoringStats before,after;
    Session *pipeline=NULL;
    uint32_t slots,bits,banks;
    uint32_t gpu_k=j->local_k>1?(j->local_k<16?16:j->local_k):1;
    uint32_t retained_k=j->local_k+j->row_k;
    /* Producer pages contain exactly one family. Raw family/global rankings
     * can reduce the entire tile on device. AST-specific tags and distinctness
     * require finer groups so their eligible winners are not discarded. */
    OdezzaScoreGrouping grouping=!j->local_k&&!j->tag_count?
        ODEZZA_SCORE_GLOBAL:ODEZZA_SCORE_BY_SYSTEM_PERMUTATION;
    uint64_t base,left,perms,groups=0,chunk=j->options.chunk;
    OdzSizing *sizing=&j->devices[r->worker_index].sizing;
    int memory_limited=0,balanced=0,available_limited=0;
    size_t result_sizes[5]={0},result_bytes=0,reserved_host_bytes=0;
    int result=0;
    OdezzaScoreWinner *winners=NULL;
    OdezzaScoreCounts *counts=NULL;
    float *values=NULL;
    Kept **local=NULL;
    uint32_t *local_counts=NULL;
    int first_pass=1,more;
    double begin;
    if(j->row_k>gpu_k)gpu_k=j->row_k;
    if(odezza_score_reducer_set_k(r->reducer,gpu_k))return odz_fail(j,"invalid reducer k");
    while(first<page->batch.count&&page->rows[first]>=page->batch.candidates[first].bank_count)++first;
    if(first==page->batch.count)return 0;
    slots=page->batch.candidates[first].slot_count;
    bits=page->batch.candidates[first].toggle_bits;
    perms=UINT64_C(1)<<bits;
    base=page->rows[first];
    left=page->batch.candidates[first].bank_count-base;
    /* A cold template or a tiny tail must not teach the next family to launch
     * tiny tiles. Timing is local to this job/family/compatible numeric layout. */
    if(r->rate_family!=page->batch.candidates[first].family_index||r->rate_slots!=slots||r->rate_bits!=bits) {
        r->configs_per_second=0;
        r->rate_family=page->batch.candidates[first].family_index;
        r->rate_slots=slots;r->rate_bits=bits;
    }
    chunk=odz_sizing_chunk(sizing,chunk,j->options.auto_chunk,r->configs_per_second,j->options.tile_seconds);
    /* Preserve complete toggle products, but never exceed a hard ceiling. */
    if(j->options.auto_chunk&&chunk<perms&&perms<=j->options.chunk&&perms<=sizing->work_configurations)chunk=perms;
    if(perms>chunk)return odz_fail(j,"one toggle product exceeds tile integration/configuration limit; reduce toggles per AST");
    items=odz_host_alloc(j,page->batch.count*sizeof(*items),ODZ_MEM_TILE);
    indices=odz_host_alloc(j,page->batch.count*sizeof(*indices),ODZ_MEM_TILE);
    copies=odz_host_alloc(j,page->batch.count*sizeof(*copies),ODZ_MEM_TILE);
    if(!items||!indices||!copies)result=odz_fail(j,"tile allocation failed");
    for(i=0;i<page->batch.count&&!result;i++) {
        const OdrCandidate *c=&page->batch.candidates[i];
        if(c->slot_count==slots&&c->toggle_bits==bits&&n<chunk/perms&&page->rows[i]==base&&page->rows[i]<c->bank_count) {
            if(c->bank_count-base<left)left=c->bank_count-base;
            items[n]=c;
            copies[n]=*c;
            indices[n++]=i;
        }
    }
    if(grouping==ODEZZA_SCORE_GLOBAL)for(i=1;i<n;i++) {
        if(items[i]->family_index!=items[0]->family_index||items[i]->index<=items[i-1]->index) {
            grouping=ODEZZA_SCORE_BY_SYSTEM_PERMUTATION;
            break;
        }
    }
    if(!result&&(!n||n*perms>chunk))result=odz_fail(j,"one toggle product/group exceeds max_chunk_configurations; reduce batch_variants or increase ceiling");
    if(!result)available_limited=n<=chunk/perms/left;
    if(!result&&j->options.auto_chunk) {
        /* Spend a module load on a substantial bank before adding more ASTs.
         * Shrinking rows across a full page repeats all module loads and can
         * create a positive feedback loop in the measured-cost controller. */
        uint64_t target=left<chunk/perms?left:chunk/perms;
        uint64_t systems_per_tile;
        if(target>UINT32_MAX)target=UINT32_MAX;
        banks=(uint32_t)odz_sizing_banks(left,target);
        balanced=banks<target;
        systems_per_tile=chunk/((uint64_t)banks*perms);
        if(n>systems_per_tile)n=(size_t)systems_per_tile;
    }
    else banks=!result?(uint32_t)fmin((double)UINT32_MAX,fmin((double)left,(double)(chunk/(n*perms)))):0;
    while(!result) {
        uint64_t need;size_t x;
        launch.profile_timing=j->options.profile_timing;
        launch.constant_bank_count=banks;
        launch.active_toggle_count=bits;
        if(odezza_score_reduction_requirements(&launch,n,grouping,gpu_k,&reduction)) {
            result=odz_fail(j,"invalid reduction layout");
            break;
        }
        groups=reduction.group_count;
        need=n*(uint64_t)banks*(perms+slots)*4+reduction.workspace_bytes+reduction.winner_bytes+reduction.count_bytes+groups*gpu_k*slots*4;
        result_sizes[0]=reduction.winner_bytes;
        result_sizes[1]=reduction.count_bytes;
        result_sizes[2]=(size_t)groups*gpu_k*slots*4+1;
        result_sizes[3]=(size_t)groups*retained_k*sizeof(*local)+1;
        result_sizes[4]=(size_t)groups*sizeof(*local_counts);
        result_bytes=0;
        for(x=0;x<5;x++) {
            size_t z=odz_aligned_bytes(result_sizes[x]);
            if(z>SIZE_MAX-result_bytes) {result=odz_fail(j,"reduction scratch size overflow");break;}
            result_bytes+=z;
        }
        if(result)break;
        if(need<=j->options.device_bytes&&(!r->pooled_buffers||
            (need+6*255<=r->tile_slab.size&&result_bytes<=r->result_capacity))) {
            break;
        }
        if(n>1&&(j->options.auto_chunk||r->pooled_buffers)) {
            memory_limited=1;
            n=(n+1)/2;
            continue;
        }
        if(banks<=1) {
            result=odz_fail(j,"one row and reduction exceed max_device_bytes or reserved tile pools");
            break;
        }
        memory_limited=1;
        banks/=2;
        if(j->options.auto_chunk) {
            uint32_t target=banks;
            banks=(uint32_t)odz_sizing_banks(left,banks);
            if(banks<target)balanced=1;
        }
    }
    if(!result) {
        DeviceBuffer *views[]={&r->coefficients,&r->scores,&r->workspace,&r->winners,&r->counts,&r->gather};
        size_t sizes[]={n*(size_t)banks*slots*4,n*(size_t)banks*(size_t)perms*4,
            reduction.workspace_bytes,reduction.winner_bytes,reduction.count_bytes,(size_t)groups*gpu_k*slots*4};
        if(r->pooled_buffers) {
            if(odz_device_views(&r->tile_slab,views,sizes,6))result=odz_fail(j,"reserved tile pool layout failed");
        } else {
            uint64_t resident=0;size_t x;
            for(x=0;x<6;x++)resident+=views[x]->size>sizes[x]?views[x]->size:sizes[x];
            if(resident>j->options.device_bytes)for(x=0;x<6;x++)if(views[x]->size>sizes[x]&&odz_gpu_free(r,views[x])) {
                result=odz_fail(j,"could not trim retained tile buffers");break;
            }
        }
    }
    if(!result) {
        pipeline=session(r,j,slots,n,(uint64_t)banks*perms);
        if(!pipeline)result=1;
    }
    odz_memory_sample(r,j,0);
    if(!result) {
        selected.candidates=copies;
        selected.count=n;
        if(odr_scoring_pack_systems(&selected,NULL,0,&work_bytes,&systems,&error))result=odz_fail(j,"native descriptor measure failed");
        else {
            packed=odz_host_alloc(j,work_bytes,ODZ_MEM_TILE);
            if(!packed||odr_scoring_pack_systems(&selected,packed,work_bytes,&work_bytes,&systems,&error))result=odz_fail(j,"native descriptor allocation failed");
        }
    }
    if(!result&&(odz_gpu_buffer(r,&r->coefficients,n*(size_t)banks*slots*4)||odz_gpu_buffer(r,&r->scores,n*(size_t)banks*(size_t)perms*4)||odz_gpu_buffer(r,&r->workspace,reduction.workspace_bytes)||odz_gpu_buffer(r,&r->winners,reduction.winner_bytes)||odz_gpu_buffer(r,&r->counts,reduction.count_bytes)||odz_gpu_buffer(r,&r->gather,(size_t)groups*gpu_k*slots*4)))result=odz_fail(j,"%s",r->error);
    if(!result)result=odz_gpu_bind(r,j,items,n,slots);
    if(!result)result=odz_gpu_prepare(r,j,slots,(uint32_t)n,base,banks);
    if(!result) {
        launch.constant_banks_device=r->coefficients.ptr;
        launch.trajectory_offsets_device=r->offsets.ptr;
        launch.trajectory_times_device=r->times.ptr;
        launch.reference_data_device=r->reference.ptr;
        launch.trajectory_count=j->trajectories->trajectory_count;
        launch.trajectory_point_count=j->trajectories->point_count;
        launch.steps_per_observation=j->rk4.steps_per_observation;
        launch.allow_missing_observations=j->trajectories->allow_missing_observations;
        launch.mse_output_device=r->scores.ptr;
        launch.input_ready_event=r->ready;
        odr_scoring_stats(pipeline->handle,&before);
        for(;;) {
            OdezzaResult code;
            if(odr_scoring_workspace(pipeline->handle,&work_bytes,&alignment)) {
                result=odz_fail(j,"pipeline workspace query failed");
                break;
            }
            odz_host_free(j,work);
            work=odz_host_alloc(j,work_bytes,ODZ_MEM_PIPELINE);
            if(!work) {
                result=odz_fail(j,"pipeline workspace allocation failed");
                break;
            }
            code=odr_scoring_run(pipeline->handle,systems,n,&launch,work,work_bytes,&work_bytes,&report);
            if(code==ODEZZA_ERROR_INSUFFICIENT_BUFFER)continue;
            if(code) {
                if(code==ODEZZA_ERROR_CUDA_UNFENCED)r->fatal=1;
                result=odz_fail(j,"scoring failed: %s",odr_scoring_error(pipeline->handle));
            }
            break;
        }
        odr_scoring_stats(pipeline->handle,&after);
        pthread_mutex_lock(&j->mutex);
        j->cache_hits+=after.cache_hits-before.cache_hits;
        j->cache_misses+=after.cache_misses-before.cache_misses;
        j->template_evictions+=after.cache_evictions-before.cache_evictions;
        j->template_invalidations+=after.cache_invalidations-before.cache_invalidations;
        j->template_oversized+=after.cache_oversized-before.cache_oversized;
        j->growths+=after.capacity_growths-before.capacity_growths;
        j->retries+=after.retry_requested_configurations-before.retry_requested_configurations;
        j->nvrtc_seconds+=after.nvrtc_seconds-before.nvrtc_seconds;
        j->template_seconds+=after.template_prepare_seconds+after.prespecialize_seconds-before.template_prepare_seconds-before.prespecialize_seconds;
        if(!result) {
            OdezzaScoringRunReport *p=&j->devices[r->worker_index].pipeline;
            j->pipeline_seconds+=report.total_seconds;
#define ADD(field) p->field+=report.field
            ADD(system_count);ADD(module_count);ADD(configuration_count);ADD(total_seconds);
            ADD(queue_wait_seconds);ADD(cubin_reset_seconds);ADD(specialization_seconds);
            ADD(module_load_seconds);ADD(function_lookup_seconds);ADD(launch_seconds);
            ADD(completion_wait_seconds);ADD(module_unload_seconds);ADD(gpu_sum_seconds);
            ADD(gpu_active_seconds);ADD(gpu_span_seconds);ADD(profiled_modules);
#undef ADD
            if(report.maximum_register_count>p->maximum_register_count)p->maximum_register_count=report.maximum_register_count;
            if(report.maximum_shared_memory_bytes>p->maximum_shared_memory_bytes)p->maximum_shared_memory_bytes=report.maximum_shared_memory_bytes;
            if(after.actual_shape.system_patch_capacity>j->devices[r->worker_index].maximum_patch_capacity)j->devices[r->worker_index].maximum_patch_capacity=after.actual_shape.system_patch_capacity;
        }
        pthread_mutex_unlock(&j->mutex);
    }
    if(!result) {
        pthread_mutex_lock(&j->mutex);
        if(odz_host_claim(j,result_bytes,ODZ_MEM_TILE))result=1;
        else {
            reserved_host_bytes=result_bytes;
            if(r->pooled_buffers) {
                unsigned char *p=r->result_slab;
                winners=(OdezzaScoreWinner *)p;p+=odz_aligned_bytes(result_sizes[0]);
                counts=(OdezzaScoreCounts *)p;p+=odz_aligned_bytes(result_sizes[1]);
                values=(float *)p;p+=odz_aligned_bytes(result_sizes[2]);
                local=(Kept **)p;p+=odz_aligned_bytes(result_sizes[3]);
                local_counts=(uint32_t *)p;
                memset(local,0,result_sizes[3]);memset(local_counts,0,result_sizes[4]);
            } else {
                winners=malloc(reduction.winner_bytes);
                counts=malloc(reduction.count_bytes);
                values=malloc((size_t)groups*gpu_k*slots*4+1);
                local=calloc((size_t)groups*retained_k+1,sizeof(*local));
                local_counts=calloc((size_t)groups,sizeof(*local_counts));
            }
            if(!winners||!counts||!values||!local||!local_counts)result=odz_fail(j,"reduction result allocation failed");
        }
        pthread_mutex_unlock(&j->mutex);
    }
    do {
        uint64_t g;
        more=0;
        if(result)break;
        begin=odz_now();
        if(odezza_score_reducer_run(r->reducer,&launch,n,grouping,r->workspace.ptr,reduction.workspace_bytes,r->winners.ptr,reduction.winner_bytes,r->counts.ptr,reduction.count_bytes,&rr)||(slots&&odezza_score_reducer_gather(r->reducer,&launch,n,grouping,slots,r->winners.ptr,reduction.winner_bytes,r->gather.ptr,(size_t)groups*gpu_k*slots*4))) {
            r->fatal=1;
            result=odz_fail(j,"GPU reduction/gather failed; runtime quarantined");
            break;
        }
        {
            double copy_begin=odz_now();
            if(cuMemcpyDtoH(winners,r->winners.ptr,reduction.winner_bytes)||cuMemcpyDtoH(counts,r->counts.ptr,reduction.count_bytes)||(slots&&cuMemcpyDtoH(values,r->gather.ptr,(size_t)groups*gpu_k*slots*4))) {
                r->fatal=1;result=odz_fail(j,"GPU result transfer failed; runtime quarantined");break;
            }
            pthread_mutex_lock(&j->mutex);
            j->devices[r->worker_index].result_copy_seconds+=odz_now()-copy_begin;
            j->devices[r->worker_index].result_d2h_bytes+=reduction.winner_bytes+reduction.count_bytes+(uint64_t)groups*gpu_k*slots*4;
            pthread_mutex_unlock(&j->mutex);
        }
        pthread_mutex_lock(&j->mutex);
        j->reduction_seconds+=odz_now()-begin;
        begin=odz_now();
        for(g=0;g<groups&&!result;g++) {
            const OdrCandidate *c=items[g/perms];
            unsigned rank;
            if(first_pass) {
                FamilyRun *f=&j->families[c->family_index];
                f->valid+=counts[g].valid;
                f->invalid+=counts[g].invalid;
                j->valid+=counts[g].valid;
                j->invalid+=counts[g].invalid;
            }
            for(rank=0;rank<gpu_k;rank++) {
                size_t wi=(size_t)g*gpu_k+rank;
                OdezzaScoreWinner *w=&winners[wi];
                OdezzaScoreIndex ix;
                Kept *k;
                int duplicate=0;
                if(w->score_index==UINT64_MAX)continue;
                if(local_counts[g]>=j->local_k&&(!first_pass||rank>=j->row_k))continue;
                if(odezza_score_decode_index(&launch,n,w->score_index,&ix)) {
                    result=odz_fail(j,"reducer index decode failed");
                    break;
                }
                c=items[ix.system_index];
                result=odz_offer(j,c,ix.permutation,base+ix.bank_index,w->mse,values+wi*slots,&k);
                if(result)break;
                if(first_pass&&rank<j->row_k) {
                    ++k->refs;
                    local[(size_t)g*retained_k+j->local_k+rank]=k;
                }
                for(s=0;s<local_counts[g];s++)if(odz_same_numeric(k,local[(size_t)g*retained_k+s]))duplicate=1;
                if(duplicate||local_counts[g]>=j->local_k) {
                    odz_keep_unref(j,k);
                    continue;
                }
                local[(size_t)g*retained_k+local_counts[g]++]=k;
            }
            if(local_counts[g]<j->local_k&&counts[g].valid>gpu_k)more=1;
        }
        ++j->reduction_passes;
        if(!first_pass)++j->reduction_continuations;
        j->reduction_kernel_seconds+=rr.kernel_seconds;
        j->retention_seconds+=odz_now()-begin;
        pthread_mutex_unlock(&j->mutex);
        if(first_pass&&!result) {
            pthread_mutex_lock(&j->mutex);
            j->configurations+=n*(uint64_t)banks*perms;
            ++j->chunks;
            for(i=0;i<n;i++)j->families[items[i]->family_index].scored+=(uint64_t)banks*perms;
            pthread_mutex_unlock(&j->mutex);
        }
        first_pass=0;
        pthread_mutex_lock(&j->mutex);
        if(more&&(j->cancelled||odz_now()-j->started>=j->options.seconds)) {
            j->retention_incomplete=1;
            more=0;
        }
        pthread_mutex_unlock(&j->mutex);
        if(more&&!result) {
            uint64_t total=groups*gpu_k;
            unsigned blocks=(unsigned)((total+255)/256);
            void *args[]= {
                &r->scores.ptr,&r->winners.ptr,&total
            };
            if(blocks>65535)blocks=65535;
            if(cuLaunchKernel(r->consume,blocks,1,1,256,1,1,0,r->stream,args,NULL)||cuEventRecord(r->ready,r->stream)||cuEventSynchronize(r->ready)) {r->fatal=1;result=odz_fail(j,"reduction continuation failed; runtime quarantined");}
        }
    }
    while(more&&!result);
    pthread_mutex_lock(&j->mutex);
    if(local&&local_counts) {
        uint64_t g;
        for(g=0;g<groups;g++)for(i=0;i<retained_k;i++) {
            Kept *k=local[(size_t)g*retained_k+i];
            int raw=i>=j->local_k;
            if(!k)continue;
            if(!result) {
                if((j->global.unit==3)==raw)odz_rank_offer(j,&j->global,k);
                if((j->families[k->family].ranking.unit==3)==raw)odz_rank_offer(j,&j->families[k->family].ranking,k);
                for(s=0;s<j->tag_count;s++)if((j->tags[s].ranking.unit==3)==raw&&(k->shared->provenance->tags&(UINT64_C(1)<<s)))odz_rank_offer(j,&j->tags[s].ranking,k);
                if(j->tag_count)archive(j,k,raw);
            }
            odz_keep_unref(j,k);
        }
    }
    if(!result) {
        uint64_t configurations=n*(uint64_t)banks*perms;
        DeviceStats *d=&j->devices[r->worker_index];
        d->configurations+=configurations;
        ++d->chunks;
        if(configurations>d->max_tile)d->max_tile=configurations;
        odz_sizing_record(sizing,n,banks,perms,memory_limited,
            !j->options.auto_chunk&&j->options.chunk<sizing->target_configurations,
            available_limited,balanced,report.total_seconds,j->options.tile_seconds);
        /* Use successful prepared scoring-call time, excluding request parsing,
         * retention and cold template growth. Ignore undersized tails. The
         * parallel floor still applies when the duration target is unattainable. */
        if(report.total_seconds>0&&configurations>=sizing->wave_configurations&&
           after.capacity_growths==before.capacity_growths) {
            double rate=configurations/report.total_seconds;
            r->configs_per_second=r->configs_per_second>0?.5*(r->configs_per_second+rate):rate;
        }
        for(i=0;i<n;i++) {
            FamilyRun *f=&j->families[items[i]->family_index];
            page->rows[indices[i]]+=banks;
            ++f->chunks;
        }
    }
    pthread_mutex_unlock(&j->mutex);
    if(r->fatal) {
        pthread_mutex_lock(&j->mutex);
        j->quarantined=1;
        pthread_mutex_unlock(&j->mutex);
        return result;
    }
    if(!r->pooled_buffers) {
        free(local);
        free(local_counts);
        free(winners);
        free(counts);
        free(values);
    }
    odz_host_release(j,reserved_host_bytes,ODZ_MEM_TILE);
    odz_host_free(j,work);
    odz_host_free(j,packed);
    odz_host_free(j,copies);
    odz_host_free(j,indices);
    odz_host_free(j,items);
    odz_memory_sample(r,j,0);
    return result;
}
