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
#include "numeric_source.h"
#include <nvrtc.h>
#include "odr_cache.h"
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/file.h>
#include <fcntl.h>
static void hash(const void *data,size_t size,unsigned char out[32]) {
    OSha256 s;
    o_sha256_init(&s);
    o_sha256_update(&s,data,size);
    o_sha256_final(&s,out);
}

static void pool_hash(unsigned kind,const void *data,size_t size,unsigned char out[32]) {
    OSha256 s;
    o_sha256_init(&s);
    o_sha256_update(&s,&kind,sizeof(kind));
    o_sha256_update(&s,data,size);
    o_sha256_final(&s,out);
}

static uint64_t name_hash(uint64_t h,const char *s) {
    if(s)while(*s)h=(h^(unsigned char)*s++)*UINT64_C(1099511628211);
    return (h^255)*UINT64_C(1099511628211);
}

void odz_bank_address(const OdrCandidate *c,const OdrAxis *a,uint64_t *seed,uint64_t *stream) {
    uint64_t h=name_hash(UINT64_C(14695981039346656037),ODZ_RNG_PROFILE);
    h=name_hash(name_hash(h,a->bank),a->stream);
    h=odr_mix(h^a->instance);
    if(a->skeleton_scope)h=odr_mix(name_hash(h,c->family)^c->derivation_index);
    *seed=a->seed;
    *stream=h&INT64_MAX;
}

int odz_gpu_buffer(OdzRuntime *r,DeviceBuffer *b,size_t bytes) {
    CUdeviceptr p;
    CUresult result;
    if(bytes<=b->size)return 0;
    if(b->borrowed) {
        snprintf(r->error,sizeof(r->error),"reserved device view is too small (%zu > %zu bytes)",bytes,b->size);
        return 1;
    }
    /* Callers resize only after the previous tile/input use has been fenced.
     * Retire first so growth does not briefly retain both old and new buffers. */
    if(b->ptr&&odz_gpu_free(r,b)) {
        snprintf(r->error,sizeof(r->error),"CUDA buffer retirement failed");
        return 1;
    }
    result=cuMemAlloc(&p,bytes);
    if(result!=CUDA_SUCCESS) {
        if(result!=CUDA_ERROR_OUT_OF_MEMORY)r->fatal=1;
        snprintf(r->error,sizeof(r->error),"CUDA allocation failed (%zu bytes)",bytes);
        return 1;
    }
    r->allocated_bytes+=bytes;
    if(r->allocated_bytes>r->peak_allocated_bytes)r->peak_allocated_bytes=r->allocated_bytes;
    b->ptr=p;
    b->size=bytes;
    return 0;
}

int odz_gpu_free(OdzRuntime *r,DeviceBuffer *b) {
    if(!b->ptr)return 0;
    if(!b->borrowed) {
        if(cuMemFree(b->ptr)!=CUDA_SUCCESS) {r->fatal=1;return 1;}
        if(b->size>r->allocated_bytes)abort();
        r->allocated_bytes-=b->size;
    }
    memset(b,0,sizeof(*b));
    return 0;
}

static int compile_numeric(OdzRuntime *r) {
    size_t size=1,i,n=0,bytes=0;char *source=NULL,*cubin=NULL,key[256],hex[65],arch[64];
    unsigned char digest[32];nvrtcProgram program=NULL;nvrtcResult result;
    OdrCache *cache=NULL;double begin,seconds=0;int major=0,minor=0,failed=1,hit;
    for(i=0;i<sizeof(numeric_source)/sizeof(*numeric_source);i++)size+=strlen(numeric_source[i]);
    source=malloc(size);if(!source)return 1;
    for(i=0;i<sizeof(numeric_source)/sizeof(*numeric_source);i++) {
        size_t l=strlen(numeric_source[i]);memcpy(source+n,numeric_source[i],l);n+=l;
    }
    source[n]=0;hash(source,n,digest);
    for(i=0;i<32;i++)snprintf(hex+i*2,3,"%02x",digest[i]);
    if(nvrtcVersion(&major,&minor)!=NVRTC_SUCCESS)goto done;
    snprintf(key,sizeof(key),"numeric-v1:c++17:fmad0:nvrtc-%d.%d:sm%u:%s",major,minor,r->sm,hex);
    if(odr_cache_open(r->cache,&cache,r->error,sizeof(r->error)))goto done;
    if(odr_cache_lock(cache,key))goto cache_error;
    hit=odr_cache_get(cache,key,(void **)&cubin,&bytes);if(hit<0)goto cache_error;
    if(hit) {
        const char *options[]={"--std=c++17",arch,"--fmad=false"};
        snprintf(arch,sizeof(arch),"--gpu-architecture=sm_%u",r->sm);
        result=nvrtcCreateProgram(&program,source,"odezza_numeric.cu",0,NULL,NULL);begin=odz_now();
        if(!result)result=nvrtcCompileProgram(program,3,options);
        seconds=odz_now()-begin;r->auxiliary_nvrtc_seconds+=seconds;
        if(result) {
            size_t len=0;
            if(program&&!nvrtcGetProgramLogSize(program,&len)&&len) {
                char *log=malloc(len);
                if(log){nvrtcGetProgramLog(program,log);snprintf(r->error,sizeof(r->error),"%s",log);free(log);}
            }
            goto done;
        }
        if(nvrtcGetCUBINSize(program,&bytes)||(cubin=malloc(bytes))==NULL||nvrtcGetCUBIN(program,cubin))goto done;
        if(odr_cache_put(cache,key,cubin,bytes,seconds)!=0) {
            snprintf(r->error,sizeof(r->error),"numeric template exceeds cache budget or storage failed: %s",odr_cache_error(cache));goto done;
        }
    }
    odr_cache_unlock(cache);
    if(cuModuleLoadData(&r->numeric_module,cubin)!=CUDA_SUCCESS)goto done;
    failed=cuModuleGetFunction(&r->rng,r->numeric_module,"grammar_rng")||
        cuModuleGetFunction(&r->prelude,r->numeric_module,"grammar_prelude")||
        cuModuleGetFunction(&r->consume,r->numeric_module,"grammar_consume");
    goto done;
cache_error:
    snprintf(r->error,sizeof(r->error),"%s",odr_cache_error(cache));
done:
    if(program)nvrtcDestroyProgram(&program);
    odr_cache_close(cache);free(cubin);free(source);return failed;
}

int odz_gpu_init(OdzRuntime *r) {
    double begin;
    OdezzaResult result;
    if(cuStreamCreate(&r->stream,CU_STREAM_NON_BLOCKING)||cuEventCreate(&r->ready,CU_EVENT_DISABLE_TIMING)||cuEventCreate(&r->uploaded,CU_EVENT_DISABLE_TIMING)||cuEventCreate(&r->start,CU_EVENT_DEFAULT)||cuEventCreate(&r->end,CU_EVENT_DEFAULT)||compile_numeric(r))return 1;
    begin=odz_now();
    result=odezza_score_reducer_create(r->sm,1,&r->reducer);
    r->reducer_setup_seconds=odz_now()-begin;
    if(result) {
        char error[2048]={0};
        if(r->reducer)odezza_score_reducer_write_error(r->reducer,error,sizeof(error));
        snprintf(r->error,sizeof(r->error),"CUB reducer initialization failed (%d): %.400s",(int)result,error);
    }
    return result!=ODEZZA_SUCCESS;
}

static int timed_end(OdzRuntime *r,double *seconds) {
    float ms=0;
    if(cuEventRecord(r->end,r->stream)||cuEventSynchronize(r->end)||cuEventElapsedTime(&ms,r->start,r->end))return 1;
    *seconds+=ms*.001;
    return 0;
}

/* Binding happens after the previous tile's completion. Current-epoch entries
 * are borrowed by descriptors being assembled and must never be evicted. */
static Pool *evict_pool(OdzRuntime *r,OdzJob *j) {
    Pool *old=NULL;
    size_t x;
    for(x=0;x<ODZ_POOLS;x++)if(r->pools[x].data.ptr&&r->pools[x].used!=r->epoch&&(!old||r->pools[x].used<old->used))old=&r->pools[x];
    if(!old)return NULL;
    {
        size_t bytes=old->data.size;
        if(odz_gpu_free(r,&old->data)) {
            r->fatal=1;
            odz_fail(j,"numeric pool retirement failed");
            return NULL;
        }
        r->pool_bytes-=bytes;
    }
    memset(old,0,sizeof(*old));
    pthread_mutex_lock(&j->mutex);
    ++j->pool_evictions;
    pthread_mutex_unlock(&j->mutex);
    return old;
}

int odz_gpu_job_begin(OdzRuntime *r,OdzJob *j) {
    DeviceBuffer *tiles[]={&r->coefficients,&r->scores,&r->workspace,&r->winners,&r->counts,&r->gather};
    size_t i;uint64_t total=0;
    ++r->epoch;
    while(r->pool_bytes>j->options.pool_bytes)if(!evict_pool(r,j))
        return odz_fail(j,"cannot trim retained numeric pools to max_bank_bytes");
    /* Fixed startup slabs have a separate resident-memory contract. Unpooled
     * callers must not retain a previous request's larger tile allocations. */
    if(!r->pooled_buffers) {
        for(i=0;i<sizeof(tiles)/sizeof(*tiles);i++)total+=tiles[i]->size;
        if(total>j->options.device_bytes)for(i=0;i<sizeof(tiles)/sizeof(*tiles);i++)if(odz_gpu_free(r,tiles[i]))
            return odz_fail(j,"could not retire oversized tile buffers");
    }
    r->peak_allocated_bytes=r->allocated_bytes;
    r->peak_pool_bytes=r->pool_bytes;
    return 0;
}

int odz_gpu_bind(OdzRuntime *r,OdzJob *j,const OdrCandidate **c,size_t count,uint32_t slots) {
    size_t i,s,n=count*slots;
    NumericSlot *desc=odz_host_alloc(j,(n?n:1)*sizeof(*desc),ODZ_MEM_TILE);
    int failed=0;
    if(!desc)return odz_fail(j,"numeric descriptor allocation failed");
    ++r->epoch;
    /* Retained capacity is real VRAM even on a cache hit or a zero-slot job.
     * The previous binding is fenced before entry, so all old epochs can retire. */
    while(r->pool_bytes>j->options.pool_bytes)if(!evict_pool(r,j)) {
        failed=odz_fail(j,"cannot trim numeric pools to max_bank_bytes");break;
    }
    for(i=0;i<count&&!failed;i++)for(s=0;s<slots;s++) {
        const OdrAxis *a=&c[i]->slots[s];
        NumericSlot *d=&desc[i*slots+s];
        unsigned char key[32];
        Pool *pool=NULL;
        size_t x;
        uint64_t seed=0,stream=0;
        d->a=a->scale_slot;
        d->b=a->shift_slot;
        d->v0=a->scale;
        d->v1=a->shift;
        d->divisor=a->numeric_stride;
        d->count=a->count;
        if(!a->kind) {
            d->v0=a->values[0];
            continue;
        }
        if(a->kind==1) {
            for(x=0;x<ODZ_POOLS;x++)if(r->pools[x].valid&&r->pools[x].host_job==r->job_epoch&&r->pools[x].host_source==a->values&&r->pools[x].data.size==a->count*4) {
                pool=&r->pools[x];
                break;
            }
            if(!pool)pool_hash(1,a->values,(size_t)a->count*4,key);
            d->kind=1;
        }
        else {
            uint64_t address[4];
            odz_bank_address(c[i],a,&seed,&stream);
            address[0]=seed;
            address[1]=stream;
            address[2]=a->kind;
            address[3]=a->count;
            pool_hash(2,address,sizeof(address),key);
            d->kind=a->transform==0?1:a->transform==1?2:a->transform==2?3:a->transform==3?4:5;
        }
        if(!a->count||a->count>SIZE_MAX/4||a->count*4>j->options.pool_bytes) {
            failed=odz_fail(j,"one numeric pool exceeds max_bank_bytes");
            break;
        }
        for(x=0;!pool&&x<ODZ_POOLS;x++)if(r->pools[x].valid&&!memcmp(r->pools[x].key,key,32)&&r->pools[x].data.size==a->count*4) {
            pool=&r->pools[x];
            break;
        }
        if(!pool) {
            /* Reuse capacity, never data, from a completed earlier attempt. */
            for(x=0;x<ODZ_POOLS;x++)if(!r->pools[x].valid&&r->pools[x].data.size==a->count*4) {
                pool=&r->pools[x];
                break;
            }
            while(r->pool_bytes+(pool?0:a->count*4)>j->options.pool_bytes) {
                if(pool)pool->used=r->epoch;
                if(!evict_pool(r,j)) {
                    failed=odz_fail(j,"active numeric pools exceed max_bank_bytes; reduce batch_variants");
                    break;
                }
            }
            if(failed)break;
            for(x=0;!pool&&x<ODZ_POOLS;x++)if(!r->pools[x].data.ptr) {
                pool=&r->pools[x];
                break;
            }
            if(!pool)pool=evict_pool(r,j);
            if(!pool) {
                failed=odz_fail(j,"active numeric pool cache entry limit exceeded; reduce batch_variants");
                break;
            }
            if(!pool->data.ptr&&odz_gpu_buffer(r,&pool->data,(size_t)a->count*4)) {
                failed=odz_fail(j,"%s",r->error);
                break;
            }
            if(!pool->used)r->pool_bytes+=pool->data.size;
            if(r->pool_bytes>r->peak_pool_bytes)r->peak_pool_bytes=r->pool_bytes;
            if(a->kind==1) {
                if(cuMemcpyHtoD(pool->data.ptr,a->values,pool->data.size)) {
                    r->fatal=1;
                    failed=odz_fail(j,"grid upload failed");
                    break;
                }
                pthread_mutex_lock(&j->mutex);
                j->devices[r->worker_index].numeric_h2d_bytes+=pool->data.size;
                pthread_mutex_unlock(&j->mutex);
            }
            else {
                double seconds=0;
                unsigned normal=a->kind==3;
                uint64_t samples=a->count;
                unsigned blocks=(unsigned)((samples+1023)/1024);
                void *args[]= {
                    &pool->data.ptr,&samples,&seed,&stream,&normal
                };
                if(blocks>65535)blocks=65535;
                cuEventRecord(r->start,r->stream);
                if(cuLaunchKernel(r->rng,blocks,1,1,256,1,1,0,r->stream,args,NULL)||timed_end(r,&seconds)) {
                    r->fatal=1;
                    failed=odz_fail(j,"Philox generation failed");
                    break;
                }
                pthread_mutex_lock(&j->mutex);
                j->rng_seconds+=seconds;
                pthread_mutex_unlock(&j->mutex);
            }
            memcpy(pool->key,key,32);
            pool->valid=1;
        }
        if(a->kind==1) {
            pool->host_source=a->values;
            pool->host_job=r->job_epoch;
        }
        pool->used=r->epoch;
        d->data=pool->data.ptr;
    }
    if(!failed&&n&&(odz_gpu_buffer(r,&r->descriptors,n*sizeof(*desc))||cuMemcpyHtoD(r->descriptors.ptr,desc,n*sizeof(*desc)))) {r->fatal=1;failed=odz_fail(j,"numeric descriptor upload failed");}
    if(!failed&&n) {
        pthread_mutex_lock(&j->mutex);
        j->devices[r->worker_index].numeric_h2d_bytes+=n*sizeof(*desc);
        pthread_mutex_unlock(&j->mutex);
    }
    /* Pageable synchronous HtoD may return after host staging, before DMA ends.
     * Explicitly join those legacy-stream copies to our nonblocking stream. */
    if(!failed&&(cuEventRecord(r->uploaded,CU_STREAM_LEGACY)||cuStreamWaitEvent(r->stream,r->uploaded,0))) {r->fatal=1;failed=odz_fail(j,"numeric upload dependency failed");}
    odz_host_free(j,desc);
    return failed;
}

int odz_gpu_prepare(OdzRuntime *r,OdzJob *j,uint32_t slots,uint32_t count,uint64_t start,uint32_t banks) {
    uint64_t items=(uint64_t)count*banks;
    unsigned blocks=(unsigned)((items+255)/256);
    void *args[]= {
        &r->descriptors.ptr,&r->coefficients.ptr,&slots,&count,&start,&banks
    };
    if(blocks>65535)blocks=65535;
    if(slots) {
        double seconds=0;
        cuEventRecord(r->start,r->stream);
        if(cuLaunchKernel(r->prelude,blocks,1,1,256,1,1,0,r->stream,args,NULL)||timed_end(r,&seconds)) {
            r->fatal=1;
            return odz_fail(j,"numeric prelude failed");
        }
        pthread_mutex_lock(&j->mutex);
        j->prelude_seconds+=seconds;
        pthread_mutex_unlock(&j->mutex);
    }
    if(cuEventRecord(r->ready,r->stream)) {r->fatal=1;return odz_fail(j,"input-ready event failed");}
    return 0;
}

void odz_gpu_close(OdzRuntime *r) {
    size_t i;
    DeviceBuffer *buffers[]= {
        &r->descriptors,&r->coefficients,&r->scores,&r->workspace,&r->winners,&r->counts,&r->gather,&r->offsets,&r->times,&r->reference,&r->trajectory_slab,&r->tile_slab
    };
    if(r->stream&&r->ready) {
        cuEventRecord(r->ready,r->stream);
        if(cuEventSynchronize(r->ready))return;
    }
    for(i=0;i<ODZ_SESSIONS;i++)if(r->sessions[i].handle)odr_scoring_destroy(r->sessions[i].handle);
    if(r->reducer)odezza_score_reducer_destroy(r->reducer);
    for(i=0;i<ODZ_POOLS;i++)if(odz_gpu_free(r,&r->pools[i].data))return;
    for(i=0;i<sizeof(buffers)/sizeof(*buffers);i++)if(odz_gpu_free(r,buffers[i]))return;
    free(r->result_slab);
    if(r->numeric_module)cuModuleUnload(r->numeric_module);
    if(r->start)cuEventDestroy(r->start);
    if(r->end)cuEventDestroy(r->end);
    if(r->ready)cuEventDestroy(r->ready);
    if(r->uploaded)cuEventDestroy(r->uploaded);
    if(r->stream)cuStreamDestroy(r->stream);
}
