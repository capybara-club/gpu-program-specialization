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
#define _POSIX_C_SOURCE 200809L
#include "o_philox.h"
#include "o_score_reduce.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define T(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#define C(x) T((x)==CUDA_SUCCESS)
#define O(x) T((x)==ODEZZA_SUCCESS)
static void scoring(uint32_t sm,const OdezzaRngPoolView *view) {
    OdezzaScoringPipelineCreateInfo info={0};OdezzaScoringPipeline *pipeline=NULL;
    OdezzaScoringLaunch launch={0},explicit_launch;OdezzaScoringRunReport report;
    OdezzaScoreReducer *reducer=NULL;OdezzaScoreReductionSize size;
    OdezzaSampledParameter parameters[3*3];
    OdezzaScoringRhs rhs[3];OdezzaScoringSystem systems[3];
    uint8_t program[]={0x82,0,0x82,1,0x84,0,0x82,2,0x90,0x80};
    const unsigned banks=1031,bits=1,constants=3,n=3*banks*2;
    uint32_t offsets[]={0,3};float times[]={0,.5f,1},reference[]={0,.2f,.4f};
    CUdeviceptr rows,winners,workspace,counts;
    OdezzaScoreWinner *indices=malloc((size_t)n*sizeof(*indices));
    float *a=malloc((size_t)n*4),*b=malloc((size_t)n*4),*values=malloc((size_t)n*constants*4),*bank_rows=malloc((size_t)3*banks*constants*4);
    void *host_workspace;size_t host_bytes,alignment;unsigned i,j;
    T(indices&&a&&b&&values&&bank_rows);
    info.sm_version=sm;info.state_count=info.state_capacity=1;info.constant_count=info.constant_capacity=constants;
    info.system_capacity=1;info.shared_patch_capacity=64;info.system_patch_capacity=128;info.worker_count=1;info.cubin_slots_per_worker=2;
    { OdezzaResult result=odezza_scoring_pipeline_create(&info,&pipeline);if(result){char error[2048];size_t bytes;odezza_scoring_pipeline_write_error(pipeline,error,sizeof(error),&bytes);fprintf(stderr,"pipeline: %s\n",error);}O(result); }O(odezza_scoring_pipeline_workspace_requirements(pipeline,&host_bytes,&alignment));
    host_workspace=malloc(host_bytes);T(host_workspace&&(uintptr_t)host_workspace%alignment==0);
    for(i=0;i<3;++i) {
        parameters[i*3]=(OdezzaSampledParameter){i*30000,3,1,(float)(i+1),-.5f,-.5f,4};
        parameters[i*3+1]=(OdezzaSampledParameter){i*30000+1,3,2,.15f,.2f,-.2f,.6f};
        parameters[i*3+2]=(OdezzaSampledParameter){0,0,0,0,.1f,.1f,.1f};
        for(j=0;j<3;++j)O(odezza_sampled_parameter_validate(&view->info,banks,parameters+i*3+j));
        rhs[i].state_index=0;rhs[i].program.bytes=program;rhs[i].program.byte_count=sizeof(program);systems[i].rhs=rhs+i;systems[i].rhs_count=1;
    }
    C(cuEventCreate(&launch.input_ready_event,CU_EVENT_DISABLE_TIMING));
    launch.constant_bank_count=banks;launch.active_toggle_count=bits;launch.trajectory_count=1;launch.trajectory_point_count=3;launch.steps_per_observation=2;
    launch.rng_pool_size=view->info.size;launch.uniform_pool_device=view->uniform_device;launch.normal_pool_device=view->normal_device;
    C(cuMemAlloc(&launch.sampled_parameters_device,sizeof(parameters)));C(cuMemcpyHtoD(launch.sampled_parameters_device,parameters,sizeof(parameters)));
    C(cuMemAlloc(&launch.trajectory_offsets_device,sizeof(offsets)));C(cuMemcpyHtoD(launch.trajectory_offsets_device,offsets,sizeof(offsets)));
    C(cuMemAlloc(&launch.trajectory_times_device,sizeof(times)));C(cuMemcpyHtoD(launch.trajectory_times_device,times,sizeof(times)));
    C(cuMemAlloc(&launch.reference_data_device,sizeof(reference)));C(cuMemcpyHtoD(launch.reference_data_device,reference,sizeof(reference)));
    C(cuMemAlloc(&launch.mse_output_device,n*4));
    C(cuEventRecord(launch.input_ready_event,NULL));
    O(odezza_scoring_pipeline_run(pipeline,systems,3,&launch,host_workspace,host_bytes,&report));C(cuMemcpyDtoH(a,launch.mse_output_device,n*4));
    O(odezza_score_reducer_create(sm,1,&reducer));
    O(odezza_score_reduction_requirements(&launch,3,ODEZZA_SCORE_BY_SYSTEM_PERMUTATION,1,&size));
    C(cuMemAlloc(&workspace,size.workspace_bytes));C(cuMemAlloc(&counts,size.count_bytes));C(cuMemAlloc(&winners,(size_t)n*sizeof(*indices)));
    O(odezza_score_reducer_run(reducer,&launch,3,ODEZZA_SCORE_BY_SYSTEM_PERMUTATION,workspace,size.workspace_bytes,winners,size.winner_bytes,counts,size.count_bytes,NULL));
    /* Gather one configuration per system for each bank, using original indices. */
    C(cuMemAlloc(&rows,3*constants*4));
    for(j=0;j<banks;++j) {
        for(i=0;i<3;++i)indices[i]=(OdezzaScoreWinner){0,0,(uint64_t)i*banks*2+j*2};
        C(cuMemcpyHtoD(winners,indices,3*sizeof(*indices)));C(cuEventRecord(launch.input_ready_event,NULL));
        O(odezza_score_reducer_gather(reducer,&launch,3,ODEZZA_SCORE_BY_SYSTEM,constants,winners,3*sizeof(*indices),rows,3*constants*4));
        C(cuMemcpyDtoH(values,rows,3*constants*4));
        for(i=0;i<3;++i)memcpy(bank_rows+((size_t)i*banks+j)*constants,values+i*constants,constants*4);
    }
    explicit_launch=launch;explicit_launch.sampled_parameters_device=0;
    C(cuMemAlloc(&explicit_launch.constant_banks_device,(size_t)3*banks*constants*4));C(cuMemcpyHtoD(explicit_launch.constant_banks_device,bank_rows,(size_t)3*banks*constants*4));
    C(cuEventRecord(launch.input_ready_event,NULL));
    O(odezza_scoring_pipeline_run(pipeline,systems,3,&explicit_launch,host_workspace,host_bytes,&report));C(cuMemcpyDtoH(b,launch.mse_output_device,n*4));
    T(memcmp(a,b,n*4)==0);
    for(i=0;i<n;++i)T(isfinite(a[i])&&a[i]<FLT_MAX);
    C(cuMemFree(explicit_launch.constant_banks_device));C(cuMemFree(rows));C(cuMemFree(workspace));C(cuMemFree(counts));C(cuMemFree(winners));
    C(cuMemFree(launch.sampled_parameters_device));C(cuMemFree(launch.trajectory_offsets_device));C(cuMemFree(launch.trajectory_times_device));C(cuMemFree(launch.reference_data_device));C(cuMemFree(launch.mse_output_device));
    C(cuEventDestroy(launch.input_ready_event));
    O(odezza_score_reducer_destroy(reducer));O(odezza_scoring_pipeline_destroy(pipeline));
    free(host_workspace);free(indices);free(a);free(b);free(values);free(bank_rows);
}
int main(int argc,char **argv) {
    uint32_t zero[4]={0},key[2]={0},words[4],expected[]={0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8};
    CUdevice device;CUcontext context;int major,minor;OdezzaRngPool *pool=NULL,*slice=NULL;OdezzaRngPoolView view,part;
    OdezzaRngPoolInfo info={20260906,37,7,1048583,3};
    float *uniform=malloc((size_t)info.size*4),*normal=malloc((size_t)info.size*4),small[39];uint64_t i;double mean=0,variance=0;
    O(odezza_philox4x32(zero,key,words));T(!memcmp(words,expected,sizeof(words)));
    T(setenv("CUDA_MODULE_LOADING","EAGER",1)==0);
    C(cuInit(0));C(cuDeviceGet(&device,argc>1?atoi(argv[1]):0));C(cuDevicePrimaryCtxRetain(&context,device));C(cuCtxSetCurrent(context));
    C(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));C(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
    { OdezzaResult result=odezza_rng_pool_create((uint32_t)(major*10+minor),&info,&pool);if(result){char error[2048];odezza_rng_pool_write_error(pool,error,sizeof(error));fprintf(stderr,"%s\n",error);}O(result); }
    O(odezza_rng_pool_view(pool,&view));C(cuMemcpyDtoH(uniform,view.uniform_device,(size_t)info.size*4));C(cuMemcpyDtoH(normal,view.normal_device,(size_t)info.size*4));
    for(i=0;i<info.size;++i) {float expected_uniform;O(odezza_rng_uniform(info.seed,info.stream,info.sample_base+i,&expected_uniform));T(uniform[i]==expected_uniform&&uniform[i]>0&&uniform[i]<1);T(isfinite(normal[i]));mean+=normal[i];variance+=(double)normal[i]*normal[i];}
    mean/=info.size;variance=variance/info.size-mean*mean;T(fabs(mean)<.006&&fabs(variance-1)<.015);
    info.sample_base+=13;info.size=39;O(odezza_rng_pool_create((uint32_t)(major*10+minor),&info,&slice));O(odezza_rng_pool_view(slice,&part));
    C(cuMemcpyDtoH(small,part.uniform_device,sizeof(small)));T(!memcmp(small,uniform+13,sizeof(small)));
    C(cuMemcpyDtoH(small,part.normal_device,sizeof(small)));T(!memcmp(small,normal+13,sizeof(small)));
    scoring((uint32_t)(major*10+minor),&view);
    printf("{\"device\":%d,\"samples_per_plane\":%llu,\"uniform_ms\":%.6f,\"normal_ms\":%.6f,\"normal_mean\":%.9f,\"normal_variance\":%.9f,\"sampled_vs_explicit_scores\":\"bitwise_equal\"}\n",argc>1?atoi(argv[1]):0,(unsigned long long)view.info.size,view.uniform_seconds*1000,view.normal_seconds*1000,mean,variance);
    O(odezza_rng_pool_destroy(slice));O(odezza_rng_pool_destroy(pool));C(cuDevicePrimaryCtxRelease(device));free(uniform);free(normal);return 0;
}
