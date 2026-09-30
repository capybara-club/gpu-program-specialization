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
#include <errno.h>
#include <sys/stat.h>
double odz_now(void) {
    struct timespec t= {
        0,0
    };
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

int odz_fail(OdzJob *j,const char *format,...) {
    va_list args;
    pthread_mutex_lock(&j->mutex);
    if(!j->error[0]) {
        va_start(args,format);
        vsnprintf(j->error,sizeof(j->error),format,args);
        va_end(args);
    }
    pthread_mutex_unlock(&j->mutex);
    return 1;
}

static int option(OdrJson root,const char *name,uint64_t fallback,uint64_t max,uint64_t *out,OdzJob *j) {
    OdrJson v;
    *out=fallback;
    if(jget(root,name,&v)&&(!ju64(v,out)||!*out))return odz_fail(j,"%s must be a positive integer",name);
    if(*out>max)return odz_fail(j,"%s exceeds native service ceiling",name);
    return 0;
}

int odz_options(OdzJob *j) {
    OdrJson root= {
        j->json,j->json_size
    },e= {
        NULL,0
    },v;
    OdrError error;
    uint64_t n;
    Options *o=&j->options;
    if(jvalidate(root,&error))return odz_fail(j,"%s",error.message);
    if(!jkeys(root,"problem,grammar,execution",&error))return odz_fail(j,"%s",error.message);
    (void)jget(root,"execution",&e);
    if(e.data&&!jkeys(e,"batch_variants,module_systems,patch_capacity,worker_count,cubin_slots_per_worker,max_chunk_configurations,target_tile_seconds,max_device_bytes,max_bank_bytes,max_host_bytes,dedup_bytes_per_family,max_seconds,max_report_bytes,profile_timing",&error))return odz_fail(j,"%s",error.message);
#define OPT(field,key,def,max) do{if(option(e,key,def,max,&n,j))return 1;o->field=n;}while(0)
    o->auto_batch=!jget(e,"batch_variants",&v);
    o->auto_module=!jget(e,"module_systems",&v);
    o->profile_timing=0;
    if(jget(e,"profile_timing",&v)) {
        if(jtype(v)!='t'&&jtype(v)!='f')return odz_fail(j,"profile_timing must be boolean");
        o->profile_timing=jtype(v)=='t';
    }
    OPT(batch,"batch_variants",1024,4096);
    OPT(module,"module_systems",64,256);
    OPT(patch,"patch_capacity",384,4096);
    OPT(workers,"worker_count",2,32);
    OPT(slots,"cubin_slots_per_worker",2,32);
    o->auto_chunk=!jget(e,"max_chunk_configurations",&v);
    OPT(chunk,"max_chunk_configurations",16777216,UINT64_C(4294967296));
    OPT(device_bytes,"max_device_bytes",268435456,UINT64_C(8589934592));
    OPT(pool_bytes,"max_bank_bytes",268435456,UINT64_C(2147483648));
    OPT(host_bytes,"max_host_bytes",UINT64_C(2147483648),UINT64_C(17179869184));
    OPT(report_bytes,"max_report_bytes",ODZ_MAX_REPORT_BYTES,ODZ_MAX_REPORT_BYTES);
    if(o->report_bytes<4096)return odz_fail(j,"max_report_bytes must be at least 4096");
    OPT(dedup_bytes,"dedup_bytes_per_family",67108864,UINT64_C(2147483648));
#undef OPT
    if(j->host_ceiling&&o->host_bytes>j->host_ceiling)
        return odz_fail(j,"max_host_bytes exceeds worker host ceiling (%llu bytes)",(unsigned long long)j->host_ceiling);
    o->tile_seconds=.05;
    if(jget(e,"target_tile_seconds",&v)) {
        float f;
        if(!jfloat(v,&f)||f<=0||f>1)return odz_fail(j,"target_tile_seconds must be in (0,1]");
        o->tile_seconds=f;
    }
    o->seconds=600;
    if(jget(e,"max_seconds",&v)) {
        float f;
        if(!jfloat(v,&f)||f<=0)return odz_fail(j,"max_seconds must be positive");
        o->seconds=f;
    }
    return 0;
}

static int create_job(const char *json,size_t bytes,void *arena,size_t capacity,int borrowed,OdzJob **out) {
    OdzJob *j;
    pthread_mutexattr_t attr;
    if(!out||!json||!bytes||bytes>64*1024*1024)return 1;
    *out=NULL;
    j=calloc(1,sizeof(*j));
    if(!j)return 1;
    j->borrowed_inputs=borrowed;
    j->ta=arena;j->trajectory_capacity=capacity;
    j->json=borrowed?(char *)json:malloc(bytes+1);
    if(!j->json) {
        free(j);
        return 1;
    }
    if(!borrowed) {
        memcpy(j->json,json,bytes);
        j->json[bytes]=0;
    }
    j->json_size=bytes;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr,PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&j->mutex,&attr);
    pthread_mutexattr_destroy(&attr);
    pthread_mutex_init(&j->snapshot_mutex,NULL);
    pthread_mutex_init(&j->memory_mutex,NULL);
    pthread_cond_init(&j->condition,NULL);
    *out=j;
    return 0;
}

int odz_job_create(const char *json,size_t bytes,OdzJob **out) {
    return create_job(json,bytes,NULL,0,0,out);
}

int odz_job_create_borrowed(const char *json,size_t bytes,void *arena,size_t capacity,OdzJob **out) {
    if(out)*out=NULL;
    if(!arena||!capacity||(uintptr_t)arena%16)return 1;
    return create_job(json,bytes,arena,capacity,1,out);
}

int odz_runtime_create(unsigned device,const char *cache,OdzRuntime **out) {
    OdzRuntime *r;
    int major,minor,sms,threads;
    const char *loading=getenv("CUDA_MODULE_LOADING");
    if(!out||!cache)return 1;
    *out=NULL;
    r=calloc(1,sizeof(*r));
    if(!r)return 1;
    *out=r;
    r->device_count=1;
    if(!loading||strcmp(loading,"EAGER")) {
        snprintf(r->error,sizeof(r->error),"CUDA_MODULE_LOADING=EAGER is required before initialization");
        return 1;
    }
    if(strlen(cache)>=sizeof(r->cache))return 1;
    strcpy(r->cache,cache);
    if(mkdir(cache,0700)&&errno!=EEXIST) {
        snprintf(r->error,sizeof(r->error),"cache directory creation failed");
        return 1;
    }
    if(cuInit(0)||cuDeviceGet(&r->device,(int)device)||cuDevicePrimaryCtxRetain(&r->context,r->device)||cuCtxSetCurrent(r->context)||cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,r->device)||cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,r->device))return 1;
    r->sm=(uint32_t)(10*major+minor);
    if(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,r->device)||
       cuDeviceGetAttribute(&threads,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,r->device)||
       sms<=0||threads<=0||(uint64_t)sms*(uint64_t)threads>ODZ_MAX_PARALLEL_CONFIGURATIONS/2) {
        snprintf(r->error,sizeof(r->error),"unsupported device parallel-capacity description");
        return 1;
    }
    r->multiprocessors=(uint32_t)sms;r->threads_per_sm=(uint32_t)threads;
    return odz_gpu_init(r);
}

const char *odz_runtime_error(const OdzRuntime *r) {
    return r?r->error:"invalid runtime";
}

int odz_runtime_host_limit(OdzRuntime *r,size_t bytes) {
    if(!r||!bytes||r->submitted||r->fatal)return 1;
    r->host_ceiling=bytes;
    return 0;
}

void odz_runtime_destroy(OdzRuntime *r) {
    if(!r)return;
    odz_workers_close(r);
    if(r->fatal)return;
    if(r->context) {
        cuCtxSetCurrent(r->context);
        odz_gpu_close(r);
        cuDevicePrimaryCtxRelease(r->device);
    }
    free(r);
}

void odz_job_cancel(OdzJob *j) {
    if(j) {
        pthread_mutex_lock(&j->snapshot_mutex);
        j->cancel_signal=1;
        pthread_mutex_unlock(&j->snapshot_mutex);
    }
}

static void cancellation(OdzJob *j) {
    pthread_mutex_lock(&j->snapshot_mutex);
    if(j->cancel_signal)j->cancelled=1;
    pthread_mutex_unlock(&j->snapshot_mutex);
}

int odz_stopped(OdzJob *j) {
    cancellation(j);
    return j->cancelled||j->error[0]||odz_now()-j->started>=j->options.seconds;
}

static int parse(OdzJob *j) {
    OdrJson root= {
        j->json,j->json_size
    };
    OdrError e;
    size_t n,i,k;
    OdrResult result;
    if(odz_options(j))return 1;
    if(odz_host_claim(j,sizeof(*j)+j->json_size+1,ODZ_MEM_REQUEST))return 1;
#define PARSE(fn,out,arena) do{result=fn(root,NULL,0,&n,&j->out,&e);if(result)return odz_fail(j,"%s",e.message);j->arena=odz_alloc(j,n);if(!j->arena)return 1;result=fn(root,j->arena,n,&n,&j->out,&e);if(result)return odz_fail(j,"%s",e.message);}while(0)
    if(j->borrowed_inputs) {
        result=odr_trajectories_parse(root,NULL,0,&n,&j->trajectories,&e);
        if(result)return odz_fail(j,"%s",e.message);
        if(n>j->trajectory_capacity)return odz_fail(j,"trajectory CPU pool too small: need %zu, reserved %zu bytes",n,j->trajectory_capacity);
        if(odz_host_claim(j,n,ODZ_MEM_ARENA))return 1;
        result=odr_trajectories_parse(root,j->ta,j->trajectory_capacity,&n,&j->trajectories,&e);
        if(result)return odz_fail(j,"%s",e.message);
    } else {PARSE(odr_trajectories_parse,trajectories,ta);}
    if(j->trajectories->point_count>ODZ_MAX_POINTS_PER_CONFIGURATION)
        return odz_fail(j,"integration_work_limit: %u trajectory points per configuration (limit %u)",
            j->trajectories->point_count,ODZ_MAX_POINTS_PER_CONFIGURATION);
    if(j->pooled_buffers) {
        const OdrTrajectories *t=j->trajectories;
        uint64_t bytes=odz_aligned_bytes(((size_t)t->trajectory_count+1)*4)+
            odz_aligned_bytes((size_t)t->point_count*4)+odz_aligned_bytes((size_t)t->state_count*t->point_count*4);
        if(bytes>j->trajectory_device_capacity)return odz_fail(j,"trajectory GPU pool too small: need %llu, reserved %zu bytes",(unsigned long long)bytes,j->trajectory_device_capacity);
    }
    PARSE(odr_static_parse,fixed,sa);
#undef PARSE
    result=odr_grammar_parse(root,j->fixed,NULL,0,&n,&j->grammar,&e);
    if(result)return odz_fail(j,"%s",e.message);
    j->ga=odz_alloc(j,n);
    if(!j->ga)return 1;
    result=odr_grammar_parse(root,j->fixed,j->ga,n,&n,&j->grammar,&e);
    if(result)return odz_fail(j,"%s",e.message);
    if(j->grammar->info.max_seconds&&j->grammar->info.max_seconds<j->options.seconds)j->options.seconds=j->grammar->info.max_seconds;
    if(odr_trajectories_rk4(j->trajectories,j->grammar->info.rk4_max_dt,j->grammar->info.rk4_max_steps,&j->rk4,&e))return odz_fail(j,"%s",e.message);
    if(j->rk4.steps_per_observation>ODZ_MAX_STEPS_PER_OBSERVATION||
       j->rk4.steps_per_configuration>ODZ_MAX_STEPS_PER_CONFIGURATION)
        return odz_fail(j,"integration_work_limit: %u steps per observation (limit %u), %llu per configuration (limit %llu)",
            j->rk4.steps_per_observation,ODZ_MAX_STEPS_PER_OBSERVATION,
            (unsigned long long)j->rk4.steps_per_configuration,(unsigned long long)ODZ_MAX_STEPS_PER_CONFIGURATION);
    j->family_count=j->grammar->family_count;
    j->families=odz_alloc(j,j->family_count*sizeof(*j->families));
    if(!j->families||odz_retention_parse(j)||odz_report_preflight(j))return 1;
    for(i=0;i<j->family_count;i++) {
        OdrProducerOptions o= {
            0
        };
        FamilyRun *f=&j->families[i];
        o.family_index=(uint32_t)i;
        o.dedup_capacity=j->options.dedup_bytes;
        o.max_attempts_per_batch=4096;
        if(j->tag_count) {
            uint64_t count=j->grammar->families[i].max_variants;
            if(count>SIZE_MAX/sizeof(Kept*))return odz_fail(j,"family retention index overflow");
            if(j->tag_local_k)f->archive=odz_alloc(j,(size_t)count*sizeof(Kept*));
            if(j->tag_row_k)f->row_archive=odz_alloc(j,(size_t)count*sizeof(Kept*));
            if((j->tag_local_k&&!f->archive)||(j->tag_row_k&&!f->row_archive))return 1;
        }
        {
            void **pair=odz_alloc(j,2*sizeof(void*));
            f->alias_user=pair;
            if(!pair)return 1;
            pair[0]=j;
            pair[1]=(void *)(uintptr_t)i;
            o.on_duplicate=odz_alias;
            o.provenance_user=pair;
        }
        if(odr_producer_create(j->grammar,&o,NULL,0,&n,&f->producer,&e))return odz_fail(j,"%s",e.message);
        f->bytes=n;
        f->arena=odz_alloc(j,n);
        if(!f->arena)return 1;
        if(odr_producer_create(j->grammar,&o,f->arena,n,&n,&f->producer,&e))return odz_fail(j,"%s",e.message);
        f->batch_size=j->options.batch;
        if(j->options.auto_batch) {
            uint64_t budget=j->options.host_bytes/(8*j->family_count*j->page_count);
            uint64_t page_ceiling=j->grammar->families[i].max_variants>=65536?UINT64_C(67108864):
                j->rk4.steps_per_configuration>=32?UINT64_C(33554432):UINT64_C(16777216);
            if(budget>page_ceiling)budget=page_ceiling;
            if(f->batch_size>j->grammar->families[i].max_variants)f->batch_size=(uint32_t)j->grammar->families[i].max_variants;
            for(;;) {
                if(odr_batch_requirements(f->producer,f->batch_size,&n))return odz_fail(j,"automatic page capacity overflow");
                if(n<=budget||f->batch_size==1)break;
                f->batch_size=(f->batch_size+1)/2;
            }
        }
        for(k=0;k<j->page_count;k++) {
            Page *p=&f->pages[k];
            if(odr_batch_requirements(f->producer,f->batch_size,&p->capacity))return odz_fail(j,"batch capacity overflow");
            p->arena=odz_alloc(j,p->capacity);
            p->rows=odz_alloc(j,f->batch_size*sizeof(uint64_t));
            if(!p->arena||!p->rows)return 1;
        }
    }
    return 0;
}

static void *generate(void *argument) {
    OdzJob *j=argument;
    size_t turn=0;
    for(;;) {
        size_t i,k,fi=0;
        Page *page=NULL;
        FamilyRun *f=NULL;
        int all_done=1;
        OdrBatch batch;
        OdrError error;
        size_t required;
        double begin;
        OdrResult result;
        pthread_mutex_lock(&j->mutex);
        for(i=0;i<j->family_count;i++) {
            size_t index=(turn+i)%j->family_count;
            FamilyRun *candidate=&j->families[index];
            if(candidate->done)continue;
            all_done=0;
            for(k=0;k<j->page_count;k++)if(!candidate->pages[k].state) {
                page=&candidate->pages[k];
                f=candidate;
                fi=index;
                break;
            }
            if(page)break;
        }
        if(odz_stopped(j)||all_done) {
            j->generator_done=1;
            pthread_cond_broadcast(&j->condition);
            pthread_mutex_unlock(&j->mutex);
            break;
        }
        if(!page) {
            pthread_cond_wait(&j->condition,&j->mutex);
            pthread_mutex_unlock(&j->mutex);
            continue;
        }
        page->state=1;
        turn=(fi+1)%j->family_count;
        pthread_mutex_unlock(&j->mutex);
        begin=odz_now();
        result=odr_producer_next(f->producer,f->generated,f->batch_size,page->arena,page->capacity,&required,&batch,&error);
        pthread_mutex_lock(&j->mutex);
        j->generation_seconds+=odz_now()-begin;
        if(result)odz_fail(j,"family %s: %s",j->grammar->families[fi].name,error.message);
        f->last=batch;
        f->generated=batch.next_index;
        f->done=batch.exhausted||result;
        page->batch=batch;
        memset(page->rows,0,f->batch_size*sizeof(uint64_t));
        page->state=batch.count?2:0;
        odz_publish(j);
        pthread_cond_broadcast(&j->condition);
        pthread_mutex_unlock(&j->mutex);
    }
    return NULL;
}

static int upload(OdzRuntime *r,OdzJob *j) {
    const OdrTrajectories *t=j->trajectories;
    size_t a=((size_t)t->trajectory_count+1)*4,b=(size_t)t->point_count*4,c=(size_t)t->state_count*t->point_count*4;
    if(r->pooled_buffers) {
        DeviceBuffer *views[]={&r->offsets,&r->times,&r->reference};
        size_t sizes[]={a,b,c};
        if(odz_device_views(&r->trajectory_slab,views,sizes,3))return odz_fail(j,"trajectory GPU pool layout failed");
    }
    int failed;double start=odz_now();
    failed=odz_gpu_buffer(r,&r->offsets,a)||odz_gpu_buffer(r,&r->times,b)||odz_gpu_buffer(r,&r->reference,c)||cuMemcpyHtoD(r->offsets.ptr,t->offsets,a)||cuMemcpyHtoD(r->times.ptr,t->times,b)||cuMemcpyHtoD(r->reference.ptr,t->values,c);
    pthread_mutex_lock(&j->mutex);
    j->devices[r->worker_index].upload_seconds+=odz_now()-start;
    if(!failed)j->devices[r->worker_index].trajectory_h2d_bytes+=a+b+c;
    pthread_mutex_unlock(&j->mutex);
    return failed;
}

/* The immutable grammar and compact finalists survive; generation arenas do not. */ static void release_work(OdzJob *j) {
    size_t i,k,x;
    if(!j->families)return;
    for(i=0;i<j->family_count;i++) {
        FamilyRun *f=&j->families[i];
        if(f->archive) {
            size_t count=(size_t)j->grammar->families[i].max_variants;
            for(x=0;x<count;x++) {
                Kept *p=f->archive[x];
                while(p) {
                    Kept *next=p->archive_next;
                    odz_keep_unref(j,p);
                    p=next;
                }
            }
            odz_host_free(j,f->archive);
            f->archive=NULL;

        }
        if(f->row_archive) {
            size_t count=(size_t)j->grammar->families[i].max_variants;
            for(x=0;x<count;x++) {
                Kept *p=f->row_archive[x];
                while(p) {Kept *next=p->row_archive_next;odz_keep_unref(j,p);p=next;}
            }
            odz_host_free(j,f->row_archive);
            f->row_archive=NULL;
        }
        odz_aliases_close(j,f);
        if(f->alias_user) {
            odz_host_free(j,f->alias_user);
            f->alias_user=NULL;

        }
        if(f->arena) {
            odz_host_free(j,f->arena);
            f->arena=NULL;
            f->producer=NULL;

        }
        for(k=0;k<j->page_count;k++) {
            Page *p=&f->pages[k];
            if(p->arena) {
                odz_host_free(j,p->arena);
                p->arena=NULL;

            }
            if(p->rows) {
                odz_host_free(j,p->rows);
                p->rows=NULL;

            }
        }
    }
}

/* Each worker leases a page exclusively. Other pages can generate/execute
 * concurrently; neither CUDA synchronization nor specialization holds job mutex. */
int odz_device_run(OdzRuntime *r,OdzJob *j) {
    size_t i,k;
    int result=0;
    pthread_mutex_lock(&j->mutex);
    j->devices[r->worker_index].worker_started=odz_now();
    result=odz_sizing_init(&j->devices[r->worker_index].sizing,r->multiprocessors,r->threads_per_sm,
        j->rk4.steps_per_configuration>j->trajectories->point_count?
        j->rk4.steps_per_configuration:j->trajectories->point_count);
    pthread_mutex_unlock(&j->mutex);
    if(result)return odz_fail(j,"invalid tile sizing work envelope");
    ++r->job_epoch;
    r->configs_per_second=0;
    if(cuCtxSetCurrent(r->context)) {r->fatal=1;result=odz_fail(j,"CUDA context selection failed");}
    if(!result)result=odz_gpu_job_begin(r,j);
    if(!result)odz_memory_sample(r,j,1);
    if(!result&&upload(r,j)) {r->fatal=1;result=odz_fail(j,"trajectory upload failed");}
    while(!result) {
        Page *page=NULL;
        int complete=1;
        double begin;
        pthread_mutex_lock(&j->mutex);
        if(odz_stopped(j)) {
            pthread_mutex_unlock(&j->mutex);
            break;
        }
        for(i=0;i<j->family_count;i++) {
            size_t index=(j->turn+i)%j->family_count;
            FamilyRun *f=&j->families[index];
            Page *earliest=NULL;
            for(k=0;k<j->page_count;k++) {
                Page *p=&f->pages[k];
                if(p->state)complete=0;
                if(p->state==2&&(!earliest||p->batch.start<earliest->batch.start))earliest=p;
            }
            if(!f->done)complete=0;
            if(earliest&&!page) {
                page=earliest;
                j->turn=(index+1)%j->family_count;
                break;
            }
        }
        if(!page) {
            if(complete&&j->generator_done) {
                pthread_mutex_unlock(&j->mutex);
                break;
            }
            begin=odz_now();
            pthread_cond_wait(&j->condition,&j->mutex);
            j->devices[r->worker_index].wait_seconds+=odz_now()-begin;
            pthread_mutex_unlock(&j->mutex);
            continue;
        }
        page->state=3;
        pthread_mutex_unlock(&j->mutex);
        begin=odz_now();
        result=odz_score_tile(r,j,page);
        pthread_mutex_lock(&j->mutex);
        j->devices[r->worker_index].busy_seconds+=odz_now()-begin;
        {
            int finished=1;
            for(i=0;i<page->batch.count;i++)if(page->rows[i]<page->batch.candidates[i].bank_count)finished=0;
            page->state=finished?0:2;
        }
        if(result&&!j->error[0])odz_fail(j,"native execution failed");
        odz_publish(j);
        pthread_cond_broadcast(&j->condition);
        pthread_mutex_unlock(&j->mutex);
    }
    pthread_mutex_lock(&j->mutex);
    j->devices[r->worker_index].worker_finished=odz_now();
    pthread_mutex_unlock(&j->mutex);
    odz_memory_sample(r,j,1);
    for(i=0;i<ODZ_SESSIONS;i++)if(r->sessions[i].handle) {
        if(odr_scoring_destroy(r->sessions[i].handle)) {
            r->fatal=1;
            odz_fail(j,"scoring cleanup could not fence work");
        }
        memset(&r->sessions[i],0,sizeof(r->sessions[i]));
    }
    if(r->attempt_inputs)for(i=0;i<ODZ_POOLS;i++) {
        r->pools[i].valid=0;
        r->pools[i].host_source=NULL;
        r->pools[i].host_job=0;
        memset(r->pools[i].key,0,sizeof(r->pools[i].key));
    }
    odz_memory_sample(r,j,1);
    pthread_mutex_lock(&j->mutex);
    if(r->fatal)j->quarantined=1;
    pthread_cond_broadcast(&j->condition);
    pthread_mutex_unlock(&j->mutex);
    return result;
}

int odz_job_run(OdzRuntime *r,OdzJob *j) {
    int result;
    double start;
    if(!r||!j)return 1;
    pthread_mutex_lock(&j->mutex);
    if(j->state) {
        pthread_mutex_unlock(&j->mutex);
        return 1;
    }
    j->started=odz_now();
    r->submitted=1;
    j->host_ceiling=r->host_ceiling;
    j->state=1;
    j->device_count=r->device_count;
    j->attempt_inputs=r->attempt_inputs;
    j->pooled_buffers=r->pooled_buffers;
    j->trajectory_device_capacity=r->trajectory_slab.size;
    j->tile_device_capacity=r->tile_slab.size;
    j->tile_host_capacity=r->result_capacity;
    j->page_count=2*r->device_count;
    j->devices[0].device=(unsigned)r->device;
    j->auxiliary_nvrtc_seconds=r->auxiliary_nvrtc_seconds;
    j->reducer_setup_seconds=r->reducer_setup_seconds;
    odz_publish(j);
    start=odz_now();
    cancellation(j);
    result=r->fatal?odz_fail(j,"runtime quarantined after unfenced CUDA failure"):r->reservation_failed?odz_fail(j,"runtime pool reservation failed; destroy runtime"):j->cancelled?1:parse(j);
    j->parse_seconds=odz_now()-start;
    if(!result) {
        j->state=2;
        j->execution_started=odz_now();
        if(pthread_create(&j->generator,NULL,generate,j))result=odz_fail(j,"generator thread creation failed");
        else j->generator_started=1;
    }
    odz_publish(j);
    pthread_mutex_unlock(&j->mutex);
    if(!result) {
        result=odz_workers_start(r,j);
        if(!result)result=odz_device_run(r,j);
        odz_workers_wait(r,j);
    }
    pthread_mutex_lock(&j->mutex);
    if(result&&!j->error[0]&&!j->cancelled)odz_fail(j,"native execution failed");
    pthread_cond_broadcast(&j->condition);
    pthread_mutex_unlock(&j->mutex);
    if(j->generator_started)pthread_join(j->generator,NULL);
    odz_memory_sample(r,j,1);
    pthread_mutex_lock(&j->mutex);
    if(r->fatal)j->quarantined=1;
    if(j->quarantined)r->fatal=1;
    if(!j->quarantined)release_work(j);
    cancellation(j);
    j->execution_finished=odz_now();
    j->finished=j->execution_finished;
    j->state=j->error[0]?5:j->cancelled?4:j->finished-j->started>=j->options.seconds?6:3;
    odz_publish(j);
    pthread_mutex_unlock(&j->mutex);
    return j->state==5;
}

void odz_job_destroy(OdzJob *j) {
    size_t i,k;
    if(!j||j->quarantined)return;
    if(j->families) {
        odz_retention_close(j);
        for(i=0;i<j->family_count;i++) {
            FamilyRun *f=&j->families[i];
            odz_host_free(j,f->alias_user);
            for(k=0;k<j->page_count;k++) {
                odz_host_free(j,f->pages[k].rows);
                odz_host_free(j,f->pages[k].arena);
            }
            odz_host_free(j,f->arena);
        }
    }
    odz_host_free(j,j->families);
    odz_host_free(j,j->ga);
    odz_host_free(j,j->sa);
    if(!j->borrowed_inputs) {
        odz_host_free(j,j->ta);
        free(j->json);
    }
    pthread_cond_destroy(&j->condition);
    pthread_mutex_destroy(&j->memory_mutex);
    pthread_mutex_destroy(&j->snapshot_mutex);
    pthread_mutex_destroy(&j->mutex);
    free(j);
}
