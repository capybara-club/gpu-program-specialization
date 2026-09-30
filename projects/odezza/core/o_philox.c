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
#include "o_philox.h"
#include "o_odezza_internal.h"
#include "o_philox_source.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct OdezzaRngPool {
    OdezzaRngPoolView view;
    CUmodule module;
    CUstream stream;
    CUevent begin,end;
    int ready;
    char error[2048];
};
OdezzaResult odezza_philox4x32(const uint32_t counter[4],const uint32_t key[2],uint32_t out[4]) {
    uint32_t c[4],k0,k1,r;
    if(!counter||!key||!out) return ODEZZA_ERROR_INVALID_ARGUMENT;
    memcpy(c,counter,sizeof(c));k0=key[0];k1=key[1];
    for(r=0;r<10;++r) {
        uint64_t a=UINT64_C(0xd2511f53)*c[0],b=UINT64_C(0xcd9e8d57)*c[2];
        uint32_t next[4]={(uint32_t)(b>>32)^c[1]^k0,(uint32_t)b,(uint32_t)(a>>32)^c[3]^k1,(uint32_t)a};
        memcpy(c,next,sizeof(c));k0+=0x9e3779b9u;k1+=0xbb67ae85u;
    }
    memcpy(out,c,sizeof(c));return ODEZZA_SUCCESS;
}
OdezzaResult odezza_rng_uniform(uint64_t seed,uint64_t stream,uint64_t sample,float *out) {
    uint64_t block=sample/4,domain=stream<<1;
    uint32_t counter[4]={(uint32_t)block,(uint32_t)(block>>32),(uint32_t)domain,(uint32_t)(domain>>32)};
    uint32_t key[2]={(uint32_t)seed,(uint32_t)(seed>>32)},words[4];
    if(!out||stream>INT64_MAX) return ODEZZA_ERROR_INVALID_ARGUMENT;
    (void)odezza_philox4x32(counter,key,words);
    *out=((float)(words[sample%4]>>9)+0.5f)*0x1p-23f;return ODEZZA_SUCCESS;
}
static int valid_info(const OdezzaRngPoolInfo *p) {
    return p&&p->size&&p->size<=SIZE_MAX/sizeof(float)&&p->sample_base<=UINT64_MAX-p->size&&
        p->stream<=INT64_MAX&&p->distributions>=1u&&p->distributions<=3u;
}
OdezzaResult odezza_sampled_parameter_validate(const OdezzaRngPoolInfo *pool,uint32_t banks,const OdezzaSampledParameter *p) {
    if(!valid_info(pool)||!banks||!p||p->distribution>2||!isfinite(p->scale)||!isfinite(p->shift)||
        !isfinite(p->lower)||!isfinite(p->upper)||p->lower>p->upper) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if(!p->distribution) return p->shift>=p->lower&&p->shift<=p->upper?ODEZZA_SUCCESS:ODEZZA_ERROR_INVALID_ARGUMENT;
    if(!(pool->distributions&p->distribution)||!p->stride||p->offset>=pool->size||
        (uint64_t)(banks-1)*p->stride>=pool->size-p->offset) return ODEZZA_ERROR_INVALID_ARGUMENT;
    return ODEZZA_SUCCESS;
}
static OdezzaResult gpu(OdezzaRngPool *p,CUresult result) {
    const char *message="unknown CUDA error";
    if(result==CUDA_SUCCESS)return ODEZZA_SUCCESS;
    (void)cuGetErrorString(result,&message);snprintf(p->error,sizeof(p->error),"%s",message);return ODEZZA_ERROR_CUDA;
}
#define GPU(x) do { OdezzaResult result_=gpu(p,(x));if(result_)return result_; } while(0)
OdezzaResult odezza_rng_pool_destroy(OdezzaRngPool *p) {
    int failed=0;
    if(!p)return ODEZZA_SUCCESS;
    if(p->stream&&cuStreamSynchronize(p->stream)!=CUDA_SUCCESS)failed=1;
    if(p->view.uniform_device&&cuMemFree(p->view.uniform_device)!=CUDA_SUCCESS)failed=1;
    if(p->view.normal_device&&cuMemFree(p->view.normal_device)!=CUDA_SUCCESS)failed=1;
    if(p->begin&&cuEventDestroy(p->begin)!=CUDA_SUCCESS)failed=1;
    if(p->end&&cuEventDestroy(p->end)!=CUDA_SUCCESS)failed=1;
    if(p->stream&&cuStreamDestroy(p->stream)!=CUDA_SUCCESS)failed=1;
    if(p->module&&cuModuleUnload(p->module)!=CUDA_SUCCESS)failed=1;
    free(p);return failed?ODEZZA_ERROR_CUDA:ODEZZA_SUCCESS;
}
OdezzaResult odezza_rng_pool_view(const OdezzaRngPool *p,OdezzaRngPoolView *out) {
    if(!p||!p->ready||!out)return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out=p->view;return ODEZZA_SUCCESS;
}
OdezzaResult odezza_rng_pool_write_error(const OdezzaRngPool *p,char *out,size_t capacity) {
    if(!p||!out)return ODEZZA_ERROR_INVALID_ARGUMENT;
    if(capacity<=strlen(p->error))return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    strcpy(out,p->error);return ODEZZA_SUCCESS;
}
OdezzaResult odezza_rng_pool_create(uint32_t sm,const OdezzaRngPoolInfo *info,OdezzaRngPool **out) {
    OdezzaRngPool *p;
    OdezzaNvrtcCompilation *compilation=NULL;
    OdezzaResult result;
    char arch[64],*source,*cubin=NULL;
    const char *options[]={arch,"--std=c++11","--fmad=false"};
    size_t i,length=1,position=0,bytes=0,log_bytes=0;
    unsigned distribution;
    if(!out)return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out=NULL;
    if(!valid_info(info)||sm<70||sm>999)return ODEZZA_ERROR_INVALID_ARGUMENT;
    p=calloc(1,sizeof(*p));if(!p)return ODEZZA_ERROR_ALLOCATION;
    *out=p;p->view.info=*info;
    for(i=0;i<sizeof(o_philox_source)/sizeof(*o_philox_source);++i)length+=strlen(o_philox_source[i]);
    source=malloc(length);if(!source)return ODEZZA_ERROR_ALLOCATION;
    for(i=0;i<sizeof(o_philox_source)/sizeof(*o_philox_source);++i) {
        size_t n=strlen(o_philox_source[i]);memcpy(source+position,o_philox_source[i],n);position+=n;
    }
    source[position]=0;snprintf(arch,sizeof(arch),"--gpu-architecture=sm_%u",sm);
    result=odezza_nvrtc_compilation_create(source,"philox_pool.cu",options,3,&compilation);free(source);
    if(!result) { OdezzaResult compiled;result=odezza_nvrtc_compilation_result(compilation,&compiled);if(!result)result=compiled; }
    if(compilation&&!odezza_nvrtc_compilation_log_size(compilation,&log_bytes)&&log_bytes) {
        char *log=malloc(log_bytes);
        if(log) { if(!odezza_nvrtc_compilation_write_log(compilation,log,log_bytes))snprintf(p->error,sizeof(p->error),"%s",log);free(log); }
    }
    if(!result)result=odezza_nvrtc_compilation_cubin_size(compilation,&bytes);
    if(!result) { cubin=malloc(bytes);if(!cubin)result=ODEZZA_ERROR_ALLOCATION; }
    if(!result)result=odezza_nvrtc_compilation_write_cubin(compilation,cubin,bytes);
    if(!result)result=gpu(p,cuModuleLoadData(&p->module,cubin));
    free(cubin);if(compilation)(void)odezza_nvrtc_compilation_destroy(compilation);
    if(result)return result;
    GPU(cuStreamCreate(&p->stream,CU_STREAM_NON_BLOCKING));GPU(cuEventCreate(&p->begin,CU_EVENT_DEFAULT));GPU(cuEventCreate(&p->end,CU_EVENT_DEFAULT));
    for(distribution=1;distribution<=2;++distribution) if(info->distributions&distribution) {
        CUfunction function;
        CUdeviceptr *ptr=distribution==1?&p->view.uniform_device:&p->view.normal_device;
        uint64_t size=info->size,seed=info->seed,stream=info->stream,base=info->sample_base;
        void *args[]={ptr,&size,&seed,&stream,&base};
        uint64_t needed_blocks=1+(size-1)/1024;
        unsigned blocks=(unsigned)(needed_blocks>65535?65535:needed_blocks);float ms;
        GPU(cuMemAlloc(ptr,(size_t)size*sizeof(float)));
        GPU(cuModuleGetFunction(&function,p->module,distribution==1?"fill_uniform":"fill_normal"));
        GPU(cuEventRecord(p->begin,p->stream));
        GPU(cuLaunchKernel(function,blocks,1,1,256,1,1,0,p->stream,args,NULL));
        GPU(cuEventRecord(p->end,p->stream));GPU(cuEventSynchronize(p->end));GPU(cuEventElapsedTime(&ms,p->begin,p->end));
        if(distribution==1)p->view.uniform_seconds=ms*.001;else p->view.normal_seconds=ms*.001;
    }
    p->ready=1;return ODEZZA_SUCCESS;
}
