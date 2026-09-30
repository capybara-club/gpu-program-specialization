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
typedef struct Writer {
    char *out;
    size_t cap,used;
    int error;
}

Writer;

static void writef(Writer *w,const char *format,...) {
    va_list args;
    int n;
    va_start(args,format);
    n=vsnprintf(w->out&&w->used<w->cap?w->out+w->used:NULL,w->used<w->cap?w->cap-w->used:0,format,args);
    va_end(args);
    if(n<0||SIZE_MAX-w->used<(size_t)n) {
        w->error=1;
        return;
    }
    w->used+=(size_t)n;
}

static void device_timing(Writer *w,const DeviceStats *d,int profiled) {
    const OdezzaScoringRunReport *p=&d->pipeline;
    const OdzSizing *s=&d->sizing;
    writef(w,",\"sizing\":{\"policy\":\"" ODZ_TILE_SIZING_POLICY "\",\"scope\":\"total tile blocks across modules; not simultaneous blocks or measured occupancy; latency target is advisory\",\"multiprocessors\":%u,\"threads_per_sm\":%u,\"block_threads\":%u",
        s->sms,s->threads_per_sm,ODZ_SCORING_BLOCK_THREADS);
#define SIZE_VALUE(name,field) writef(w,",\"" name "\":%llu",(unsigned long long)s->field)
    SIZE_VALUE("hardware_wave_configurations",wave_configurations);
    SIZE_VALUE("parallel_target_configurations",target_configurations);
    SIZE_VALUE("work_ceiling_configurations",work_configurations);
    SIZE_VALUE("smallest_tile_configurations",minimum_configurations);
    SIZE_VALUE("maximum_tile_systems",maximum_systems);
    SIZE_VALUE("minimum_tile_blocks",minimum_blocks);SIZE_VALUE("maximum_tile_blocks",maximum_blocks);
    SIZE_VALUE("underfilled_tiles",underfilled_tiles);SIZE_VALUE("underfilled_configurations",underfilled_configurations);
    SIZE_VALUE("memory_limited_tiles",memory_limited_tiles);SIZE_VALUE("explicit_underfill_limit_tiles",explicit_limited_tiles);
    SIZE_VALUE("available_work_limited_tiles",available_limited_tiles);SIZE_VALUE("balanced_bank_tiles",balanced_tiles);
    SIZE_VALUE("scoring_calls_over_duration_target",over_target_seconds);
#undef SIZE_VALUE
    writef(w,",\"underfilled_scoring_call_seconds\":%.9g,\"active_fraction_of_block_capacity\":%.9g,\"mean_blocks_per_loaded_module\":%.9g",s->underfilled_call_seconds,
        s->total_blocks?(double)d->configurations/((double)s->total_blocks*ODZ_SCORING_BLOCK_THREADS):0,
        p->module_count?(double)s->total_blocks/p->module_count:0);
    writef(w,",\"duration_scope\":\"successful scoring-call host wall time; histogram percentile upper bounds\",\"maximum_scoring_call_seconds\":%.9g,\"p50_scoring_call_seconds_upper_bound\":%.9g,\"p95_scoring_call_seconds_upper_bound\":%.9g}",
        s->maximum_call_seconds,odz_sizing_percentile(s,50),odz_sizing_percentile(s,95));
    writef(w,",\"pipeline_breakdown\":{\"scope\":\"successful scoring calls; per-module host sums overlap; CUDA union per call is not SM occupancy\",\"modules\":%zu",p->module_count);
#define SECONDS(field) writef(w,",\"" #field "\":%.9g",p->field)
    SECONDS(queue_wait_seconds);SECONDS(cubin_reset_seconds);SECONDS(specialization_seconds);
    SECONDS(module_load_seconds);SECONDS(function_lookup_seconds);SECONDS(launch_seconds);
    SECONDS(completion_wait_seconds);SECONDS(module_unload_seconds);
#undef SECONDS
    writef(w,",\"profile_enabled\":%s,\"profiled_modules\":%llu,\"gpu_sum_seconds\":",profiled?"true":"false",(unsigned long long)p->profiled_modules);
    if(profiled)writef(w,"%.9g",p->gpu_sum_seconds);else writef(w,"null");
    writef(w,",\"gpu_active_seconds\":");if(profiled)writef(w,"%.9g",p->gpu_active_seconds);else writef(w,"null");
    writef(w,",\"gpu_span_seconds\":");if(profiled)writef(w,"%.9g",p->gpu_span_seconds);else writef(w,"null");
    writef(w,",\"gpu_gap_seconds\":");if(profiled)writef(w,"%.9g",fmax(0,p->gpu_span_seconds-p->gpu_active_seconds));else writef(w,"null");
    writef(w,",\"maximum_registers\":%u,\"maximum_dynamic_shared_bytes\":%u,\"maximum_module_capacity\":%u,\"minimum_module_capacity\":%u,\"maximum_patch_capacity\":%u},\"transfers\":{\"trajectory_h2d_bytes\":%llu,\"numeric_h2d_bytes\":%llu,\"result_d2h_bytes\":%llu,\"trajectory_upload_host_seconds\":%.9g,\"result_copy_host_seconds\":%.9g},\"worker_wall_seconds\":%.9g,\"busy_scope\":\"host tile execution including waits; not GPU utilization\"",
        p->maximum_register_count,p->maximum_shared_memory_bytes,d->module_capacity,d->minimum_module_capacity,d->maximum_patch_capacity,
        (unsigned long long)d->trajectory_h2d_bytes,(unsigned long long)d->numeric_h2d_bytes,(unsigned long long)d->result_d2h_bytes,
        d->upload_seconds,d->result_copy_seconds,d->worker_started?fmax(0,d->worker_finished-d->worker_started):0);
}

static void string(Writer *w,const char *s) {
    const unsigned char *p=(const unsigned char *)(s?s:"");
    writef(w,"\"");
    while(*p) {
        unsigned c=*p++;
        if(c=='"'||c=='\\')writef(w,"\\%c",c);
        else if(c<32)writef(w,"\\u%04x",c);
        else writef(w,"%c",c);
    }
    writef(w,"\"");
}

static void id(Writer *w,const Kept *k) {
    writef(w,"\"%u:%llu:%llu:%u\"",k->family,(unsigned long long)k->ast,(unsigned long long)k->bank,k->permutation);
}

static void ranking(Writer *w,const Ranking *r) {
    size_t i;
    writef(w,"[");
    for(i=0;i<r->count;i++) {
        if(i)writef(w,",");
        id(w,r->rows[i]);
    }
    writef(w,"]");
}

static void programs(Writer *w,const unsigned char *data,size_t count) {
    size_t at=0;
    int first=1;
    writef(w,"[");
    while(at<count) {
        size_t i,n=(size_t)data[at+1]|(size_t)data[at+2]<<8|(size_t)data[at+3]<<16|(size_t)data[at+4]<<24;
        at+=5;
        if(!first)writef(w,",");
        first=0;
        writef(w,"\"");
        for(i=0;i<n;i++)writef(w,"%02x",data[at+i]);
        writef(w,"\"");
        at+=n;
    }
    writef(w,"]");
}

static void candidate(Writer *w,const Kept *k,const OdzJob *j) {
    size_t i;
    OdrCandidate c= {
        0
    };
    c.family=j->grammar->families[k->family].name;
    c.derivation_index=k->derivation;
    writef(w,"{\"mse\":%.9g,\"origin\":{\"family_index\":%u,\"ast_index\":\"%llu\",\"derivation_index\":\"%llu\",\"variant_index\":\"%llu\",\"bank_index\":\"%llu\",\"permutation\":%u},\"family\":",k->mse,k->family,(unsigned long long)k->ast,(unsigned long long)k->derivation,(unsigned long long)k->variant,(unsigned long long)k->bank,k->permutation);
    string(w,c.family);
    writef(w,",\"values\":[");
    for(i=0;i<k->nslots;i++) {
        if(i)writef(w,",");
        if(isfinite(k->values[i]))writef(w,"%.9g",k->values[i]);
        else writef(w,"null");
    }
    writef(w,"],\"value_bits\":[");
    for(i=0;i<k->nslots;i++) {
        uint32_t bits;
        memcpy(&bits,&k->values[i],4);
        if(i)writef(w,",");
        writef(w,"\"%08x\"",bits);
    }
    writef(w,"],\"slots\":[");
    for(i=0;i<k->nslots;i++) {
        const OdrAxis *a=&k->slots[i];
        uint64_t seed=0,stream=0;
        if(i)writef(w,",");
        writef(w,"{\"name\":");
        string(w,a->name);
        writef(w,",\"instance\":%u,\"kind\":%u,\"axis_index\":\"%llu\"",a->instance,a->kind,(unsigned long long)(a->kind?(k->bank/a->numeric_stride)%a->count:0));
        if(a->kind>=2) {
            odz_bank_address(&c,a,&seed,&stream);
            writef(w,",\"seed\":\"%llu\",\"stream\":\"%llu\",\"count\":\"%llu\",\"transform\":%u,\"scale\":%.9g,\"shift\":%.9g,\"scale_slot\":%d,\"shift_slot\":%d",(unsigned long long)seed,(unsigned long long)stream,(unsigned long long)a->count,a->transform,a->scale,a->shift,a->scale_slot,a->shift_slot);
        }
        writef(w,"}");
    }
    writef(w,"],\"tags\":[");
    for(i=0;i<k->shared->provenance->count;i++) {
        if(i)writef(w,",");
        string(w,k->shared->provenance->names[i]);
    }
    writef(w,"],\"resolved_programs\":");
    programs(w,k->numeric,k->numeric_size);
    writef(w,"}");
}

static int same_id(const Kept *a,const Kept *b) {
    return a->family==b->family&&a->ast==b->ast&&a->bank==b->bank&&a->permutation==b->permutation;
}

static Ranking *rank_at(OdzJob *j,size_t i) {
    if(!i)return &j->global;
    if(i<=j->family_count)return j->families?&j->families[i-1].ranking:NULL;
    return &j->tags[i-1-j->family_count].ranking;
}

/* Preserve first-seen order with a bounded, preflight-reserved identity set. */
static int earlier_candidate(OdzJob *j,const Kept *candidate) {
    size_t at;
    if(!j->report_seen_capacity)return 0;
    at=(size_t)odr_mix(candidate->ast^(candidate->bank*UINT64_C(0x9e3779b97f4a7c15))^
        ((uint64_t)candidate->family<<32)^candidate->permutation)&(j->report_seen_capacity-1);
    while(j->report_seen[at]) {
        if(same_id(j->report_seen[at],candidate))return 1;
        at=(at+1)&(j->report_seen_capacity-1);
    }
    j->report_seen[at]=(Kept *)candidate;return 0;
}

static void memory(Writer *w,OdzJob *j) {
    static const char *names[]={"request","arenas","retained","tile_scratch","pipeline_workspace"};
    HostMemory m=odz_host_snapshot(j);
    size_t i;
    writef(w,",\"memory\":{\"host_budget_bytes\":%llu,\"worker_host_ceiling_bytes\":%llu,\"charged_live_bytes\":%llu,\"charged_peak_bytes\":%llu,\"denied_reservations\":%llu,\"job_arena_bytes\":%llu,\"retained_bytes\":%llu,\"classes\":{",
        (unsigned long long)j->options.host_bytes,(unsigned long long)j->host_ceiling,(unsigned long long)m.total,(unsigned long long)m.peak_total,
        (unsigned long long)m.denied,(unsigned long long)m.live[ODZ_MEM_ARENA],(unsigned long long)m.live[ODZ_MEM_RETAINED]);
    for(i=0;i<ODZ_MEM_CLASSES;i++)writef(w,"%s\"%s\":{\"live_bytes\":%llu,\"peak_bytes\":%llu}",i?",":"",names[i],(unsigned long long)m.live[i],(unsigned long long)m.peak[i]);
    writef(w,"},\"scope\":\"shared job reservations across all device workers, including used borrowed buffers; not total RSS\","
        "\"reserved_trajectory_cpu_bytes\":%zu,\"reserved_result_cpu_bytes\":%zu,"
        "\"rss_sample_valid\":%s,\"process_rss_bytes\":%llu,\"job_peak_sampled_process_rss_bytes\":%llu,\"process_lifetime_peak_rss_bytes\":%llu,"
        "\"opaque_allocations\":\"compiler, core handles, modules, driver and allocator overhead are included only in process/device samples, not the host byte budget\"}",
        j->borrowed_inputs?j->trajectory_capacity:0,j->tile_host_capacity*j->device_count,
        j->rss_valid?"true":"false",(unsigned long long)j->rss_bytes,(unsigned long long)j->peak_sampled_rss_bytes,(unsigned long long)j->process_peak_rss_bytes);
}

int odz_job_report(OdzJob *j,char *output,size_t capacity,size_t *required) {
    static const char *states[]= {
        "queued","preparing","running","complete","cancelled","failed","timeout"
    };
    Writer w= {
        output,capacity,0,0
    };
    size_t count=0,i,k;
    double elapsed;
    if(!j||!required)return -1;
    pthread_mutex_lock(&j->mutex);
    if(j->report_seen)memset(j->report_seen,0,j->report_seen_capacity*sizeof(*j->report_seen));
    elapsed=j->started?(j->finished?j->finished:odz_now())-j->started:0;
    writef(&w,"{\"status\":");
    string(&w,states[j->state]);
    writef(&w,",\"delivery\":{\"max_report_bytes\":%llu,\"conservative_bound_bytes\":%llu,\"preflight_rejected\":%s}",
        (unsigned long long)(j->options.report_bytes?j->options.report_bytes:ODZ_MAX_REPORT_BYTES),
        (unsigned long long)j->report_bound,j->report_rejected?"true":"false");
    if(j->report_rejected) {
        writef(&w,",\"counts\":{\"completed_configurations\":0,\"valid\":0,\"invalid\":0,\"completed_chunks\":0},\"error\":");
        string(&w,j->error);writef(&w,"}");*required=w.used+1;
        pthread_mutex_unlock(&j->mutex);return w.error?-1:output&&capacity<=w.used?1:0;
    }
    writef(&w,",\"numeric_input_scope\":\"%s\"",j->attempt_inputs?"attempt":"runtime_cache");
    writef(&w,",\"integration_limits\":{\"points_per_configuration\":%u,\"steps_per_observation\":%u,\"steps_per_configuration\":%llu,\"base_work_units_per_tile\":%llu,\"maximum_parallel_target_configurations\":%llu,\"tile_policy\":\"" ODZ_TILE_SIZING_POLICY "\"}",
        ODZ_MAX_POINTS_PER_CONFIGURATION,ODZ_MAX_STEPS_PER_OBSERVATION,(unsigned long long)ODZ_MAX_STEPS_PER_CONFIGURATION,(unsigned long long)ODZ_BASE_TILE_WORK_UNITS,(unsigned long long)ODZ_MAX_PARALLEL_CONFIGURATIONS);
    writef(&w,",\"buffer_storage\":{\"trajectory_cpu\":\"%s\",\"scoring\":\"%s\",\"trajectory_device_bytes_per_gpu\":%zu,\"tile_device_bytes_per_gpu\":%zu,\"tile_host_bytes_per_gpu\":%zu}",
        j->borrowed_inputs?"borrowed_arena":"job_allocation",j->pooled_buffers?"reserved_slabs":"grow_on_demand",
        j->trajectory_device_capacity,j->tile_device_capacity,j->tile_host_capacity);
    writef(&w,",\"retention_complete\":%s,\"runtime_quarantined\":%s",j->retention_incomplete?"false":"true",j->quarantined?"true":"false");
    writef(&w,",\"backend\":\"odezza.c99-runtime.v1\",\"sampling_profile\":");
    string(&w,odr_grammar_sampling_profile());
    writef(&w,",\"rng_profile\":\"" ODZ_RNG_PROFILE "\",\"error\":");
    if(j->error[0])string(&w,j->error);
    else writef(&w,"null");
    writef(&w,",\"counts\":{\"completed_configurations\":%llu,\"valid\":%llu,\"invalid\":%llu,\"completed_chunks\":%llu},\"timing\":{\"total_seconds\":%.9g,\"parse_seconds\":%.9g,\"generation_seconds\":%.9g,\"pipeline_seconds\":%.9g,\"rng_seconds\":%.9g,\"prelude_seconds\":%.9g,\"reduction_gather_seconds\":%.9g,\"retention_seconds\":%.9g,\"template_seconds\":%.9g,\"nvrtc_seconds\":%.9g,\"runtime_initialization_nvrtc_seconds\":%.9g},\"cache\":{\"hits\":%llu,\"misses\":%llu,\"capacity_growths\":%llu,\"retry_requested_configurations\":%llu,\"numeric_pool_evictions\":%llu}",(unsigned long long)j->configurations,(unsigned long long)j->valid,(unsigned long long)j->invalid,(unsigned long long)j->chunks,elapsed,j->parse_seconds,j->generation_seconds,j->pipeline_seconds,j->rng_seconds,j->prelude_seconds,j->reduction_seconds,j->retention_seconds,j->template_seconds,j->nvrtc_seconds,j->auxiliary_nvrtc_seconds,(unsigned long long)j->cache_hits,(unsigned long long)j->cache_misses,(unsigned long long)j->growths,(unsigned long long)j->retries,(unsigned long long)j->pool_evictions);
    writef(&w,",\"critical_path\":{\"parse_and_reserve_seconds\":%.9g,\"execute_and_finalize_seconds\":%.9g,\"scope\":\"serial wall phases; generation, GPU workers and finalization are contained in execute\"}",j->parse_seconds,j->execution_started?fmax(0,j->execution_finished-j->execution_started):0);
    writef(&w,",\"template_cache\":{\"backend\":\"sqlite\",\"evictions\":%llu,\"invalidations\":%llu,\"oversized_uncached\":%llu}",
        (unsigned long long)j->template_evictions,(unsigned long long)j->template_invalidations,(unsigned long long)j->template_oversized);
    writef(&w,",\"reduction\":{\"backend\":\"cub_block_hierarchy\",\"passes\":%llu,\"continuation_passes\":%llu,\"kernel_seconds\":%.9g,\"local_distinct_k\":%u,\"local_row_k\":%u,\"runtime_initialization_seconds\":%.9g}",
        (unsigned long long)j->reduction_passes,(unsigned long long)j->reduction_continuations,j->reduction_kernel_seconds,j->local_k,j->row_k,j->reducer_setup_seconds);
    memory(&w,j);
    writef(&w,",\"families\":[");
    if(j->families)for(i=0;i<j->family_count;i++) {
        FamilyRun *f=&j->families[i];
        if(i)writef(&w,",");
        writef(&w,"{\"id\":");
        string(&w,j->grammar->families[i].name);
        writef(&w,",\"generated_asts\":%llu,\"allocated_configurations\":%llu,\"reserved_configurations\":%llu,\"completed_configurations\":%llu,\"valid\":%llu,\"invalid\":%llu,\"batch_variants\":%u,\"max_configurations_per_skeleton\":%llu,\"configuration_limited_derivations\":%llu,\"generation_stop\":%u,\"duplicates\":%llu,\"pruned\":%llu}",(unsigned long long)f->generated,(unsigned long long)j->grammar->families[i].max_configs,(unsigned long long)f->last.configurations_reserved,(unsigned long long)f->scored,(unsigned long long)f->valid,(unsigned long long)f->invalid,f->batch_size,(unsigned long long)j->grammar->families[i].max_configs_per_skeleton,(unsigned long long)f->last.configuration_limited_derivations,f->last.stop_reason,(unsigned long long)f->last.duplicates,(unsigned long long)f->last.pruned);
    }
    writef(&w,"],\"execution\":{\"automatic_ast_pages\":%s,\"automatic_module_packing\":%s,\"module_capacity\":%u,\"automatic_tile_sizing\":%s,\"configuration_ceiling\":%llu,\"target_tile_seconds\":%.9g,\"devices\":[",j->options.auto_batch?"true":"false",j->options.auto_module?"true":"false",j->options.module,j->options.auto_chunk?"true":"false",(unsigned long long)j->options.chunk,j->options.tile_seconds);
    for(i=0;i<j->device_count;i++) {
        const DeviceStats *d=&j->devices[i];
        if(i)writef(&w,",");
        writef(&w,"{\"device\":%u,\"completed_configurations\":%llu,\"completed_chunks\":%llu,\"largest_tile_configurations\":%llu,\"busy_seconds\":%.9g,\"page_wait_seconds\":%.9g,\"memory\":{\"allocated_device_bytes\":%llu,\"peak_allocated_device_bytes\":%llu,\"numeric_pool_bytes\":%llu,\"peak_numeric_pool_bytes\":%llu,\"device_sample_valid\":%s,\"device_total_bytes\":%llu,\"device_used_bytes\":%llu,\"peak_sampled_device_used_bytes\":%llu,\"usage_scope\":\"explicit allocations belong to this runtime; device-used samples include all processes\"}",d->device,(unsigned long long)d->configurations,(unsigned long long)d->chunks,(unsigned long long)d->max_tile,d->busy_seconds,d->wait_seconds,(unsigned long long)d->allocated_bytes,(unsigned long long)d->peak_allocated_bytes,(unsigned long long)d->numeric_pool_bytes,(unsigned long long)d->peak_numeric_pool_bytes,d->device_sample_valid?"true":"false",(unsigned long long)d->device_total_bytes,(unsigned long long)d->device_used_bytes,(unsigned long long)d->peak_device_used_bytes);
        device_timing(&w,d,j->options.profile_timing);writef(&w,"}");
    }
    writef(&w,"],\"stage_timing_scope\":\"sum across workers; overlaps generation and other devices\"},\"states\":[");
    if(j->fixed)for(i=0;i<j->fixed->state_count;i++) {
        if(i)writef(&w,",");
        string(&w,j->fixed->states[i]);
    }
    writef(&w,"],\"trajectories\":[");
    if(j->trajectories)for(i=0;i<j->trajectories->trajectory_count;i++) {
        const OdrTrajectoryInfo *t=&j->trajectories->info[i];
        if(i)writef(&w,",");
        writef(&w,"{\"points\":%u,\"observed_scalars\":%u,\"uniform_spacing\":%s,\"min_dt\":%.9g,\"max_dt\":%.9g}",t->point_count,t->observed_count,t->uniform_spacing?"true":"false",t->min_dt,t->max_dt);
    }
    writef(&w,"],\"integration\":{\"method\":\"rk4\",\"steps_per_observation\":%u,\"steps_per_configuration\":%llu},\"leaderboards\":{\"global\":",j->rk4.steps_per_observation,(unsigned long long)j->rk4.steps_per_configuration);
    ranking(&w,&j->global);
    writef(&w,",\"families\":{");
    if(j->families)for(i=0;i<j->family_count;i++) {
        if(i)writef(&w,",");
        string(&w,j->grammar->families[i].name);
        writef(&w,":");
        ranking(&w,&j->families[i].ranking);
    }
    writef(&w,"},\"tags\":{");
    for(i=0;i<j->tag_count;i++) {
        if(i)writef(&w,",");
        string(&w,j->tags[i].name);
        writef(&w,":");
        ranking(&w,&j->tags[i].ranking);
    }
    writef(&w,"}},\"candidates\":{");
    for(i=0;i<1+j->family_count+j->tag_count;i++) {
        Ranking *r=rank_at(j,i);
        if(!r)continue;
        for(k=0;k<r->count;k++)if(!earlier_candidate(j,r->rows[k])) {
            if(count++)writef(&w,",");
            id(&w,r->rows[k]);writef(&w,":");candidate(&w,r->rows[k],j);
        }
    }
    writef(&w,"}}");
    *required=w.used+1;
    pthread_mutex_unlock(&j->mutex);
    return w.error?-1:output&&capacity<=w.used?1:0;
}
