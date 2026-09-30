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
#include "lm_internal.h"
#include "secant_sr_lm_source.h"
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <time.h>

struct SRLMCudaImpl {
  SRLMBatch batch;
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction statistics[4], step;
  CUstream stream;
  CUevent begin, done;
  CUdeviceptr input, target, programs, states, trial_stats, best_stats, indices;
  void *host_programs, *host_states;
  unsigned *host_indices;
  size_t rows;
  int retained, pending, failed;
  SRLMStats resources;
};
static double now(void) {
  struct timespec t;
  return clock_gettime(CLOCK_MONOTONIC,&t) ? 0 : t.tv_sec+t.tv_nsec*1e-9;
}
static SecantResult wait_owned(SRLMCuda s) {
  if (!s->pending) return SECANT_SUCCESS;
  if (cuEventRecord(s->done,s->stream)!=CUDA_SUCCESS || cuEventSynchronize(s->done)!=CUDA_SUCCESS) {
    /* Only the error-recovery path falls back to draining this owned stream. */
    if (cuStreamSynchronize(s->stream)!=CUDA_SUCCESS) {
      s->failed=1;
      return SECANT_ERROR_COMPLETION_UNKNOWN;
    }
    s->pending=0;
    return SECANT_ERROR_DRIVER_FAILED;
  }
  s->pending=0;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_lm_cuda_destroy(SRLMCuda s) {
  CUcontext current;
  SecantResult r;
  if (!s) return SECANT_SUCCESS;
  if (s->context && (cuCtxGetCurrent(&current)!=CUDA_SUCCESS || current!=s->context))
    return SECANT_ERROR_INVALID_STATE;
  s->failed=1;
  r=wait_owned(s);
  if (r) return r;
#define RELEASE(h,fn) do { if(h) { if(fn(h)!=CUDA_SUCCESS) return SECANT_ERROR_DRIVER_FAILED; h=0; } } while(0)
  RELEASE(s->module,cuModuleUnload);
  RELEASE(s->begin,cuEventDestroy);
  RELEASE(s->done,cuEventDestroy);
  RELEASE(s->stream,cuStreamDestroy);
  RELEASE(s->input,cuMemFree);
  RELEASE(s->target,cuMemFree);
  RELEASE(s->programs,cuMemFree);
  RELEASE(s->states,cuMemFree);
  RELEASE(s->trial_stats,cuMemFree);
  RELEASE(s->best_stats,cuMemFree);
  RELEASE(s->indices,cuMemFree);
  RELEASE(s->host_indices,cuMemFreeHost);
  RELEASE(s->host_programs,cuMemFreeHost);
  RELEASE(s->host_states,cuMemFreeHost);
#undef RELEASE
  if (s->retained && cuDevicePrimaryCtxRelease(s->device)!=CUDA_SUCCESS)
    return SECANT_ERROR_DRIVER_FAILED;
  sr_lm_batch_free(&s->batch);
  free(s);
  return SECANT_SUCCESS;
}
static SecantResult compile_module(SRLMCuda s,int major,int minor) {
  nvrtcProgram p;
  nvrtcResult e;
  char arch[64],stack[64];
  const char *headers[]={sr_lm_layout}, *names[]={"lm_layout.h"};
  const char *options[]={arch,stack,"--std=c++14","--use_fast_math","--fmad=false"};
  void *cubin=NULL;
  size_t size=0;
  double t=now();
  SecantResult r=SECANT_ERROR_COMPILE_FAILED;
  snprintf(arch,sizeof(arch),"--gpu-architecture=sm_%d%d",major,minor);
  {
    unsigned depth=s->batch.config.max_depth+1,nodes=(s->batch.config.max_nodes+1)/2;
    unsigned capacity=depth<nodes?depth:nodes;
    if(capacity<2)capacity=2; /* binary operand locations remain well-formed */
    snprintf(stack,sizeof(stack),"-DSR_LM_STACK=%u",capacity);
  }
  if(nvrtcCreateProgram(&p,sr_lm_source,"secant_sr_lm.cu",1,headers,names)!=NVRTC_SUCCESS) return r;
  e=nvrtcCompileProgram(p,5,options);
  if(e!=NVRTC_SUCCESS && nvrtcGetProgramLogSize(p,&size)==NVRTC_SUCCESS) {
    char *log=malloc(size);
    if(log) { if(nvrtcGetProgramLog(p,log)==NVRTC_SUCCESS) fprintf(stderr,"%s\n",log); free(log); }
  }
  if(e==NVRTC_SUCCESS) e=nvrtcGetCUBINSize(p,&size);
  if(e==NVRTC_SUCCESS) {
    cubin=malloc(size);
    if(!cubin) r=SECANT_ERROR_ALLOCATION_FAILED;
    else if(nvrtcGetCUBIN(p,cubin)==NVRTC_SUCCESS)r=SECANT_SUCCESS;
  }
  nvrtcDestroyProgram(&p);
  s->resources.nvrtc_seconds=now()-t;
  if(r==SECANT_SUCCESS && cuModuleLoadData(&s->module,cubin)!=CUDA_SUCCESS)r=SECANT_ERROR_DRIVER_FAILED;
  free(cubin);
  return r;
}
#define CUDA(call) do { CUresult e=(call); if(e!=CUDA_SUCCESS) { \
  const char *message=NULL; cuGetErrorString(e,&message); \
  fprintf(stderr,"LM %s: %s\n",#call,message?message:"CUDA failure"); \
  r=SECANT_ERROR_DRIVER_FAILED; goto fail; } } while(0)
SecantResult secant_sr_lm_cuda_create(const SRConfig *c,const SRLMOptions *o,
    const float *input,const float *target,size_t rows,SRLMCuda *out,SRLMStats *stats) {
  SRLMCuda s;
  SecantResult r;
  size_t inputs,programs,states,program_bytes,state_bytes,stat_bytes,i;
  int major,minor;
  double t=now();
  if(!out) return SECANT_ERROR_INVALID_VALUE;
  *out=NULL;
  if(stats) memset(stats,0,sizeof(*stats));
  if(!c||!input||!target||!rows||!sr_mul(rows,c->num_inputs,&inputs)||inputs>SIZE_MAX/sizeof(float))
    return SECANT_ERROR_INVALID_VALUE;
  s=calloc(1,sizeof(*s));
  if(!s) return SECANT_ERROR_ALLOCATION_FAILED;
  r=sr_lm_batch_create(c,o,&s->batch);
  if(r) goto fail;
  for(i=0;i<inputs;++i) if(!isfinite(input[i])) {r=SECANT_ERROR_INVALID_VALUE;goto fail;}
  for(i=0;i<rows;++i) if(!isfinite(target[i])) {r=SECANT_ERROR_INVALID_VALUE;goto fail;}
  programs=o->capacity*o->bindings;states=programs*o->starts;
  program_bytes=programs*sizeof(SRLMProgram);state_bytes=states*sizeof(SRLMState);
  if(!sr_mul(states,SR_LM_STATISTICS*sizeof(double),&stat_bytes)||states>INT_MAX) {r=SECANT_ERROR_OVERFLOW;goto fail;}
  s->rows=rows;
  CUDA(cuInit(0));
  CUDA(cuDeviceGet(&s->device,o->device));
  CUDA(cuDevicePrimaryCtxRetain(&s->context,s->device));s->retained=1;
  CUDA(cuCtxSetCurrent(s->context));
  CUDA(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,s->device));
  CUDA(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,s->device));
  r=compile_module(s,major,minor);if(r) goto fail;
  CUDA(cuModuleGetFunction(&s->step,s->module,"lm_step"));
  {
    int step_local;size_t shape;
    CUDA(cuFuncGetAttribute(&step_local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,s->step));
    if(step_local) {
      fprintf(stderr,"LM register solve rejected: local=%d bytes\n",step_local);
      r=SECANT_ERROR_REGISTER_OVERFLOW;goto fail;
    }
    for(shape=0;shape<4;++shape) {
      char name[64];int registers,local,blocks;
      snprintf(name,sizeof(name),"lm_statistics_%u",1u<<(unsigned)shape);
      CUDA(cuModuleGetFunction(s->statistics+shape,s->module,name));
      CUDA(cuFuncGetAttribute(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,s->statistics[shape]));
      CUDA(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,s->statistics[shape]));
      if(local){fprintf(stderr,"LM %s rejected: %d bytes local storage\n",name,local);r=SECANT_ERROR_REGISTER_OVERFLOW;goto fail;}
      CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,s->statistics[shape],(int)o->threads,o->threads/32*SR_LM_STATISTICS*sizeof(double)));
      s->resources.registers_by_shape[shape]=registers;s->resources.blocks_by_shape[shape]=blocks;
      if(registers>s->resources.registers)s->resources.registers=registers;
    }
  }
  s->resources.shared_bytes=(int)(o->threads/32*SR_LM_STATISTICS*sizeof(double));
  s->resources.active_blocks_per_sm=s->resources.blocks_by_shape[3];
  CUDA(cuDeviceGetAttribute(&s->resources.multiprocessors,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,s->device));
  CUDA(cuStreamCreate(&s->stream,CU_STREAM_NON_BLOCKING));
  CUDA(cuEventCreate(&s->begin,CU_EVENT_DEFAULT));
  CUDA(cuEventCreate(&s->done,CU_EVENT_DEFAULT));
  CUDA(cuMemAlloc(&s->input,inputs*sizeof(float)));
  CUDA(cuMemAlloc(&s->target,rows*sizeof(float)));
  CUDA(cuMemAlloc(&s->programs,program_bytes));
  CUDA(cuMemAlloc(&s->states,state_bytes));
  CUDA(cuMemAlloc(&s->trial_stats,stat_bytes));
  CUDA(cuMemAlloc(&s->best_stats,stat_bytes));
  CUDA(cuMemAlloc(&s->indices,states*sizeof(unsigned)));
  CUDA(cuMemHostAlloc((void **)&s->host_indices,states*sizeof(unsigned),0));
  CUDA(cuMemHostAlloc(&s->host_programs,program_bytes,0));
  CUDA(cuMemHostAlloc(&s->host_states,state_bytes,0));
  CUDA(cuMemcpyHtoD(s->input,input,inputs*sizeof(float)));
  CUDA(cuMemcpyHtoD(s->target,target,rows*sizeof(float)));
  s->resources.setup_seconds=now()-t;
  if(stats) *stats=s->resources;
  *out=s;
  return SECANT_SUCCESS;
fail:
  { SecantResult cleanup=secant_sr_lm_cuda_destroy(s);
    if(cleanup) {*out=s;return cleanup;}
  }
  return r;
}
SecantResult secant_sr_lm_cuda_fit(SRLMCuda s,const SRModel *models,size_t count,
    const float *banks,size_t elements,uint64_t sequence,double max_seconds,
    SecantAstProgramSet *out,SRLMStats *stats) {
  SecantResult r;
  CUcontext current;
  size_t i,program_bytes,state_bytes,shape;
  unsigned offsets[4],counts[4],at=0;
  unsigned starts,n,iteration,propose;
  unsigned long long rows;
  double t=now(),device_seconds=0;
  if(stats) memset(stats,0,sizeof(*stats));
  if(out) memset(out,0,sizeof(*out));
  if(!s||!out||!isfinite(max_seconds)||max_seconds<=0) return SECANT_ERROR_INVALID_VALUE;
  if(s->failed||cuCtxGetCurrent(&current)!=CUDA_SUCCESS||current!=s->context) return SECANT_ERROR_INVALID_STATE;
  if((uint64_t)s->batch.options.capacity*s->batch.options.bindings*s->batch.options.starts > UINT64_MAX/s->rows/(s->batch.options.iterations+1))
    return SECANT_ERROR_OVERFLOW;
  r=sr_lm_batch_prepare(&s->batch,models,count,banks,elements,sequence);
  if(r) return r;
  program_bytes=s->batch.program_count*sizeof(SRLMProgram);state_bytes=s->batch.state_count*sizeof(SRLMState);
  memcpy(s->host_programs,s->batch.programs,program_bytes);
  memcpy(s->host_states,s->batch.states,state_bytes);
  for(shape=0;shape<4;++shape) {
    offsets[shape]=at;
    for(i=0;i<s->batch.state_count;++i) {
      unsigned parameters=s->batch.programs[i/s->batch.options.starts].parameters;
      unsigned group=parameters<=1?0:parameters<=2?1:parameters<=4?2:3;
      if(group==shape)s->host_indices[at++]=(unsigned)i;
    }
    counts[shape]=at-offsets[shape];s->batch.stats.states_by_shape[shape]=counts[shape];
  }
  s->pending=1;
  CUDA(cuMemcpyHtoDAsync(s->programs,s->host_programs,program_bytes,s->stream));
  CUDA(cuMemcpyHtoDAsync(s->states,s->host_states,state_bytes,s->stream));
  CUDA(cuMemcpyHtoDAsync(s->indices,s->host_indices,s->batch.state_count*sizeof(unsigned),s->stream));
  starts=(unsigned)s->batch.options.starts;n=(unsigned)s->batch.state_count;rows=s->rows;
  for(i=0;i<=s->batch.options.iterations;++i) {
    void *b[]={&s->programs,&s->states,&starts,&n,&s->trial_stats,&s->best_stats,&iteration,&propose};
    float ms;
    iteration=(unsigned)i;propose=i<s->batch.options.iterations;
    s->pending=1;
    CUDA(cuEventRecord(s->begin,s->stream));
    for(shape=0;shape<4;++shape) if(counts[shape]) {
      void *a[]={&s->programs,&s->states,&starts,&s->input,&s->target,&rows,&s->trial_stats,&s->indices,offsets+shape};
      CUDA(cuLaunchKernel(s->statistics[shape],counts[shape],1,1,s->batch.options.threads,1,1,
          s->batch.options.threads/32*SR_LM_STATISTICS*sizeof(double),s->stream,a,NULL));
    }
    CUDA(cuLaunchKernel(s->step,(n+127)/128,1,1,128,1,1,0,s->stream,b,NULL));
    r=wait_owned(s);if(r) goto fail;
    CUDA(cuEventElapsedTime(&ms,s->begin,s->done));device_seconds+=ms*.001;
    s->batch.stats.statistics_evaluations+=n;
    s->batch.stats.row_evaluations+=(uint64_t)n*rows;
    s->batch.stats.iterations=i;
    if(now()-t>=max_seconds) break;
  }
  s->pending=1;
  CUDA(cuMemcpyDtoHAsync(s->host_states,s->states,state_bytes,s->stream));
  r=wait_owned(s);if(r) goto fail;
  memcpy(s->batch.states,s->host_states,state_bytes);
  r=sr_lm_batch_finish(&s->batch,out);if(r) goto fail;
  s->batch.stats.seconds=now()-t;s->batch.stats.device_seconds=device_seconds;
  s->batch.stats.registers=s->resources.registers;s->batch.stats.local_bytes=s->resources.local_bytes;
  s->batch.stats.shared_bytes=s->resources.shared_bytes;
  s->batch.stats.active_blocks_per_sm=s->resources.active_blocks_per_sm;
  s->batch.stats.multiprocessors=s->resources.multiprocessors;
  memcpy(s->batch.stats.registers_by_shape,s->resources.registers_by_shape,sizeof(s->resources.registers_by_shape));
  memcpy(s->batch.stats.blocks_by_shape,s->resources.blocks_by_shape,sizeof(s->resources.blocks_by_shape));
  if(stats) *stats=s->batch.stats;
  return SECANT_SUCCESS;
fail:
  s->failed=1;s->batch.count=0;
  { SecantResult drain=wait_owned(s);if(drain) return drain; }
  return r;
}
#undef CUDA
SecantResult secant_sr_lm_cuda_model(SRLMCuda s,size_t index,SRModel *out) {
  const SRCandidate *m;
  if(!s||!out||s->failed||index>=s->batch.count) return SECANT_ERROR_INVALID_VALUE;
  m=s->batch.models+index;
  *out=(SRModel){m->nodes,m->count,m->score,s->batch.generations[index],m->fingerprint};
  return SECANT_SUCCESS;
}
