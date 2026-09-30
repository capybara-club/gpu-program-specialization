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
#include "odezza_scoring_request.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"check failed line %d: %s\n",__LINE__,#x);return 1;} } while(0)
#define PARSE(fn,arg,var,storage) do {size_t bytes;CHECK(fn(arg,NULL,0,&bytes,&var,&error)==ODR_OK);storage=malloc(bytes);CHECK(storage);CHECK(fn(arg,storage,bytes,&bytes,&var,&error)==ODR_OK);}while(0)
int main(int argc,char **argv) {
    FILE *file;
    long length;
    char *text;
    OdrJson input;
    OdrError error= {
        0
    };
    const OdrTrajectories *trajectories;
    const OdrStatic *fixed;
    const OdrGrammar *grammar;
    OdrProducer *producer;
    OdrProducerOptions po= {
        0
    };
    OdrBatch batch;
    size_t bytes,need,align;
    void *td,*sd,*gd,*pd,*bd,*workspace=NULL;
    OdrScoringSession *session=NULL;
    OdrScoringOptions options= {
        0
    };
    OdrScoringStats stats;
    const OdezzaScoringSystem *systems;
    void *systems_arena;
    OdrGrammarInfo grammar_info;
    OdrRk4Layout rk4;
    OdezzaScoringLaunch launch= {
        0
    };
    OdezzaScoringRunReport report;
    CUdevice device;
    CUcontext context;
    int major,minor;
    unsigned pass;
    const float *coefficients;
    float scores[3];
    uint64_t grown=0;
    CHECK(argc==3);
    file=fopen(argv[1],"rb");
    CHECK(file);
    CHECK(fseek(file,0,SEEK_END)==0);
    length=ftell(file);
    CHECK(length>0);
    rewind(file);
    text=malloc((size_t)length+1);
    CHECK(text);
    CHECK(fread(text,1,(size_t)length,file)==(size_t)length);
    fclose(file);
    text[length]=0;
    input.data=text;
    input.size=(size_t)length;
    PARSE(odr_trajectories_parse,input,trajectories,td);
    PARSE(odr_static_parse,input,fixed,sd);
    CHECK(odr_grammar_parse(input,fixed,NULL,0,&bytes,&grammar,&error)==ODR_OK);
    gd=malloc(bytes);
    CHECK(odr_grammar_parse(input,fixed,gd,bytes,&need,&grammar,&error)==ODR_OK);
    po.max_asts=1;
    CHECK(odr_producer_create(grammar,&po,NULL,0,&bytes,&producer,&error)==ODR_OK);
    pd=malloc(bytes);
    CHECK(odr_producer_create(grammar,&po,pd,bytes,&need,&producer,&error)==ODR_OK);
    CHECK(odr_batch_requirements(producer,1,&bytes)==ODR_OK);
    bd=malloc(bytes);
    CHECK(odr_producer_next(producer,0,1,bd,bytes,&need,&batch,&error)==ODR_OK&&batch.count==1);
    CHECK(batch.candidates[0].rhs_count==1&&batch.candidates[0].slot_count==1&&batch.candidates[0].numeric_count==3);
    CHECK(odr_scoring_pack_systems(&batch,NULL,0,&bytes,&systems,&error)==ODR_OK);
    systems_arena=malloc(bytes);
    CHECK(systems_arena);
    CHECK(odr_scoring_pack_systems(&batch,systems_arena,bytes,&need,&systems,&error)==ODR_OK);
    coefficients=batch.candidates[0].slots[0].values;
    CHECK(coefficients);
    CHECK(odr_grammar_info(grammar,&grammar_info)==ODR_OK);
    CHECK(odr_trajectories_rk4(trajectories,grammar_info.rk4_max_dt,grammar_info.rk4_max_steps,&rk4,&error)==ODR_OK);
    CHECK(cuInit(0)==CUDA_SUCCESS&&cuDeviceGet(&device,0)==CUDA_SUCCESS);
    CHECK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device)==CUDA_SUCCESS&&cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device)==CUDA_SUCCESS);
    CHECK(cuDevicePrimaryCtxRetain(&context,device)==CUDA_SUCCESS&&cuCtxSetCurrent(context)==CUDA_SUCCESS);
#define UPLOAD(field,ptr,len) CHECK(cuMemAlloc(&launch.field,(len))==CUDA_SUCCESS&&cuMemcpyHtoD(launch.field,(ptr),(len))==CUDA_SUCCESS)
    UPLOAD(trajectory_offsets_device,trajectories->offsets,((size_t)trajectories->trajectory_count+1)*4);
    UPLOAD(trajectory_times_device,trajectories->times,(size_t)trajectories->point_count*4);
    UPLOAD(reference_data_device,trajectories->values,(size_t)trajectories->state_count*trajectories->point_count*4);
    UPLOAD(constant_banks_device,coefficients,3*sizeof(float));
    CHECK(cuMemAlloc(&launch.mse_output_device,sizeof(scores))==CUDA_SUCCESS);
    launch.trajectory_count=trajectories->trajectory_count;
    launch.trajectory_point_count=trajectories->point_count;
    launch.allow_missing_observations=trajectories->allow_missing_observations;
    launch.constant_bank_count=(uint32_t)batch.candidates[0].bank_count;
    launch.steps_per_observation=rk4.steps_per_observation;
    (void)mkdir(argv[2],0700);
    options.cache_directory=argv[2];
    options.maximum_patch_capacity=512;
    options.shape.sm_version=(uint32_t)(10*major+minor);
    options.shape.state_count=options.shape.state_capacity=2;
    options.shape.constant_count=options.shape.constant_capacity=1;
    options.shape.system_capacity=4;
    options.shape.shared_patch_capacity=2;
    options.shape.system_patch_capacity=2;
    options.shape.worker_count=2;
    options.shape.cubin_slots_per_worker=2;
    for(pass=0;pass<2;pass++) {
        OdezzaResult r=odr_scoring_create(fixed,&options,&session);
        if(r)fprintf(stderr,"prepare %d: %s\n",r,odr_scoring_error(session));
        CHECK(r==ODEZZA_SUCCESS);
        for(;;) {
            CHECK(odr_scoring_workspace(session,&bytes,&align)==ODEZZA_SUCCESS);
            free(workspace);
            workspace=malloc(bytes);
            CHECK(workspace);
            r=odr_scoring_run(session,systems,1,&launch,workspace,bytes,&need,&report);
            if(r==ODEZZA_ERROR_INSUFFICIENT_BUFFER)continue;
            if(r)fprintf(stderr,"run %d: %s\n",r,odr_scoring_error(session));
            CHECK(r==ODEZZA_SUCCESS);
            break;
        }
        CHECK(cuMemcpyDtoH(scores,launch.mse_output_device,sizeof(scores))==CUDA_SUCCESS);
        CHECK(scores[1]<1e-10f&&scores[0]>scores[1]&&scores[2]>scores[1]);
        CHECK(report.configuration_count==3);
        CHECK(odr_scoring_stats(session,&stats)==ODEZZA_SUCCESS);
        printf("{\"pass\":%u,\"mse\":%.12g,\"cache_hits\":%llu,\"cache_misses\":%llu,\"growths\":%llu,\"nvrtc_seconds\":%.9f,\"template_seconds\":%.9f,\"prespecialize_seconds\":%.9f,\"system_patch_capacity\":%u}\n",pass,scores[1],(unsigned long long)stats.cache_hits,(unsigned long long)stats.cache_misses,(unsigned long long)stats.capacity_growths,stats.nvrtc_seconds,stats.template_prepare_seconds,stats.prespecialize_seconds,stats.actual_shape.system_patch_capacity);
        CHECK(stats.capacity_growths>0);
        if(pass) {
            CHECK(stats.cache_hits>0&&stats.cache_misses==0&&stats.nvrtc_seconds==0&&stats.capacity_growths==grown);
        }
        grown=stats.capacity_growths;
        CHECK(odr_scoring_destroy(session)==ODEZZA_SUCCESS);
        session=NULL;
    }
    CHECK(cuMemFree(launch.trajectory_offsets_device)==CUDA_SUCCESS&&cuMemFree(launch.trajectory_times_device)==CUDA_SUCCESS&&cuMemFree(launch.reference_data_device)==CUDA_SUCCESS&&cuMemFree(launch.constant_banks_device)==CUDA_SUCCESS&&cuMemFree(launch.mse_output_device)==CUDA_SUCCESS);
    CHECK(cuDevicePrimaryCtxRelease(device)==CUDA_SUCCESS);
    free(workspace);
    free(systems_arena);
    free(bd);
    free(pd);
    free(gd);
    free(sd);
    free(td);
    free(text);
    return 0;
}
