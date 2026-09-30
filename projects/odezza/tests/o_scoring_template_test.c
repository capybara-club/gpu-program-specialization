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
#include "odezza.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"template check failed at %d: %s\n",__LINE__,#x);return 1; } } while (0)
int main(void) {
    CUdevice device;CUcontext context;int major,minor;
    OdezzaScoringTemplateInfo shape={0};OdezzaScoringTemplate *a=NULL,*b=NULL,*bad=NULL;
    OdezzaScoringPipelineCreateInfo info={0};OdezzaScoringPipeline *p=NULL,*q=NULL;
    unsigned char *artifact;size_t bytes,need,alignment;void *workspace;double compile_seconds;
    unsigned char program[]={ODEZZA_AST_CONSTANT_F32,0,ODEZZA_AST_STATE_F32,0,ODEZZA_AST_MUL_F32,ODEZZA_AST_RETURN_F32};
    OdezzaScoringRhs rhs={0,{program,sizeof(program)}};OdezzaScoringSystem system={&rhs,1};
    OdezzaScoringLaunch launch={0};OdezzaScoringRunReport report;
    uint32_t offsets[]={0,2};float times[]={0,.1f},reference[]={1,1.105170185f},bank[]={0,1},scores[2];
    CHECK(cuInit(0)==CUDA_SUCCESS && cuDeviceGet(&device,0)==CUDA_SUCCESS);
    CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device)==CUDA_SUCCESS);
    CHECK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device)==CUDA_SUCCESS);
    shape.sm_version=(uint32_t)(10*major+minor);shape.state_capacity=2;shape.constant_capacity=1;
    shape.system_capacity=4;shape.shared_patch_capacity=32;shape.system_patch_capacity=32;
    /* The compiler can elide the final branch for a one-system template.
     * Both cold preparation and artifact read must accept its physical arena. */
    {
        OdezzaScoringTemplateInfo one={shape.sm_version,1,3,1,64,128};
        OdezzaScoringTemplate *single=NULL,*loaded=NULL;
        void *data;size_t n;
        CHECK(odezza_scoring_template_create(&one,&single)==ODEZZA_SUCCESS);
        CHECK(odezza_scoring_template_write(single,NULL,0,&n)==ODEZZA_SUCCESS);
        data=malloc(n);CHECK(data);
        CHECK(odezza_scoring_template_write(single,data,n,&n)==ODEZZA_SUCCESS);
        CHECK(odezza_scoring_template_read(&one,data,n,&loaded)==ODEZZA_SUCCESS);
        CHECK(odezza_scoring_template_destroy(single)==ODEZZA_SUCCESS);
        CHECK(odezza_scoring_template_destroy(loaded)==ODEZZA_SUCCESS);
        free(data);
    }
    /* Template creation/round-trip needs no CUDA context. */
    CHECK(odezza_scoring_template_create(&shape,&a)==ODEZZA_SUCCESS);
    CHECK(odezza_scoring_template_nvrtc_seconds(a,&compile_seconds)==ODEZZA_SUCCESS && compile_seconds>0);
    CHECK(odezza_scoring_template_write(a,NULL,0,&bytes)==ODEZZA_SUCCESS);
    artifact=malloc(bytes);CHECK(artifact);
    CHECK(odezza_scoring_template_write(a,artifact,bytes-1,&need)==ODEZZA_ERROR_INSUFFICIENT_BUFFER && need==bytes);
    CHECK(odezza_scoring_template_write(a,artifact,bytes,&need)==ODEZZA_SUCCESS);
    CHECK(odezza_scoring_template_read(&shape,artifact,bytes,&b)==ODEZZA_SUCCESS);
    CHECK(odezza_scoring_template_nvrtc_seconds(b,&compile_seconds)==ODEZZA_SUCCESS && compile_seconds==0);
    artifact[bytes-1]^=1;
    CHECK(odezza_scoring_template_read(&shape,artifact,bytes,&bad)==ODEZZA_ERROR_FORMAT);
    CHECK(odezza_scoring_template_write(bad,NULL,0,&need)==ODEZZA_ERROR_BUSY);
    CHECK(odezza_scoring_template_destroy(bad)==ODEZZA_SUCCESS);bad=NULL;artifact[bytes-1]^=1;
    CHECK(odezza_scoring_template_read(&shape,artifact,bytes-1,&bad)==ODEZZA_ERROR_FORMAT);
    CHECK(odezza_scoring_template_destroy(bad)==ODEZZA_SUCCESS);bad=NULL;
    shape.constant_capacity=2;
    CHECK(odezza_scoring_template_read(&shape,artifact,bytes,&bad)==ODEZZA_ERROR_FORMAT);
    CHECK(odezza_scoring_template_destroy(bad)==ODEZZA_SUCCESS);shape.constant_capacity=1;
    free(artifact);
    CHECK(cuDevicePrimaryCtxRetain(&context,device)==CUDA_SUCCESS && cuCtxSetCurrent(context)==CUDA_SUCCESS);
    info.sm_version=shape.sm_version;info.state_count=1;info.state_capacity=2;
    info.constant_count=info.constant_capacity=1;info.system_capacity=4;
    info.shared_patch_capacity=info.system_patch_capacity=32;info.worker_count=1;info.cubin_slots_per_worker=2;
    { OdezzaResult result=odezza_scoring_pipeline_create_with_template(&info,b,&p);
      if(result){char error[1024];size_t size;odezza_scoring_pipeline_write_error(p,error,sizeof(error),&size);fprintf(stderr,"prepare: %s\n",error);}
      CHECK(result==ODEZZA_SUCCESS); }
    info.state_capacity=1;
    CHECK(odezza_scoring_pipeline_create_with_template(&info,b,&q)==ODEZZA_ERROR_INVALID_ARGUMENT);
    CHECK(odezza_scoring_pipeline_destroy(q)==ODEZZA_SUCCESS);
    CHECK(odezza_scoring_template_destroy(a)==ODEZZA_SUCCESS && odezza_scoring_template_destroy(b)==ODEZZA_SUCCESS);
    CHECK(odezza_scoring_pipeline_workspace_requirements(p,&need,&alignment)==ODEZZA_SUCCESS);
    workspace=malloc(need);CHECK(workspace);
#define UPLOAD(field,values) CHECK(cuMemAlloc(&launch.field,sizeof(values))==CUDA_SUCCESS && cuMemcpyHtoD(launch.field,values,sizeof(values))==CUDA_SUCCESS)
    UPLOAD(trajectory_offsets_device,offsets);UPLOAD(trajectory_times_device,times);
    UPLOAD(reference_data_device,reference);UPLOAD(constant_banks_device,bank);
    CHECK(cuMemAlloc(&launch.mse_output_device,sizeof(scores))==CUDA_SUCCESS);
    launch.trajectory_count=1;launch.trajectory_point_count=2;launch.constant_bank_count=2;launch.steps_per_observation=4;
    CHECK(odezza_scoring_pipeline_run(p,&system,1,&launch,workspace,need,&report)==ODEZZA_SUCCESS);
    CHECK(cuMemcpyDtoH(scores,launch.mse_output_device,sizeof(scores))==CUDA_SUCCESS);
    CHECK(fabs(scores[0]-(reference[1]-1)*(reference[1]-1))<1e-6 && scores[1]<1e-10);
    CHECK(report.configuration_count==2);
    CHECK(cuMemFree(launch.trajectory_offsets_device)==CUDA_SUCCESS && cuMemFree(launch.trajectory_times_device)==CUDA_SUCCESS);
    CHECK(cuMemFree(launch.reference_data_device)==CUDA_SUCCESS && cuMemFree(launch.constant_banks_device)==CUDA_SUCCESS);
    CHECK(cuMemFree(launch.mse_output_device)==CUDA_SUCCESS);
    free(workspace);CHECK(odezza_scoring_pipeline_destroy(p)==ODEZZA_SUCCESS);
    CHECK(cuDevicePrimaryCtxRelease(device)==CUDA_SUCCESS);
    puts("opaque template round-trip, corruption/shape rejection, independent lifetime and GPU replay: verified");return 0;
}
