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
#include "o_score_reduce.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define T(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); exit(1); } } while (0)
#define C(x) T((x)==CUDA_SUCCESS)
#define O(x) T((x)==ODEZZA_SUCCESS)
static unsigned pattern;
static int compare(const void *a, const void *b) {
    const OdezzaScoreWinner *x=a,*y=b;
    if(x->mse!=y->mse) return x->mse<y->mse?-1:1;
    return x->score_index<y->score_index?-1:x->score_index>y->score_index;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void test(OdezzaScoreReducer *r, unsigned k, unsigned systems, unsigned banks, unsigned bits, unsigned mode) {
    OdezzaScoringLaunch l={0}; OdezzaScoreReductionSize s; OdezzaScoreReductionReport report;
    CUdeviceptr workspace,winners,counts,rows;
    OdezzaScoreWinner *actual,*expected;
    OdezzaScoreCounts *actual_counts;
    float *scores,*constants,*gathered;
    size_t n, i, g, j, elements; double download=0, reduced=0, start;
    l.constant_bank_count=banks; l.active_toggle_count=bits;
    O(odezza_score_reduction_requirements(&l,systems,(OdezzaScoreGrouping)mode,k,&s));
    n=(size_t)s.score_count; elements=n/(size_t)s.group_count;
    scores=malloc(n*4); constants=malloc((size_t)systems*banks*2*4);
    actual=malloc(s.winner_bytes); expected=malloc(elements*sizeof(*expected));
    actual_counts=malloc(s.count_bytes); gathered=malloc(s.group_count*k*2*4);
    T(scores&&constants&&actual&&expected&&actual_counts&&gathered);
    for(i=0;i<n;++i) {
        scores[i]=(float)((i*177+31)%10007);
        if(i%37==0) scores[i]=NAN;
        if(i%41==0) scores[i]=INFINITY;
        if(i%43==0) scores[i]=FLT_MAX;
        if(i%47==0) scores[i]=-1;
        if(i%29==0) scores[i]=-0.0f;
        if(pattern==1)scores[i]=i%2?-0.f:0.f;
        if(pattern==2)scores[i]=(float)(n-i);
        if(i<(size_t)banks*(1u<<bits)) scores[i]=NAN; /* all-invalid system */
    }
    for(i=0;i<(size_t)systems*banks*2;++i) constants[i]=(float)i/16;
    C(cuMemAlloc(&l.mse_output_device,n*4)); C(cuMemcpyHtoD(l.mse_output_device,scores,n*4));
    C(cuMemAlloc(&l.constant_banks_device,(size_t)systems*banks*8)); C(cuMemcpyHtoD(l.constant_banks_device,constants,(size_t)systems*banks*8));
    C(cuMemAlloc(&workspace,s.workspace_bytes)); C(cuMemAlloc(&winners,s.winner_bytes)); C(cuMemAlloc(&counts,s.count_bytes));
    C(cuMemAlloc(&rows,s.group_count*k*8));
    T(odezza_score_reducer_run(r,&l,systems,(OdezzaScoreGrouping)mode,workspace,s.workspace_bytes-1,winners,s.winner_bytes,counts,s.count_bytes,NULL)==ODEZZA_ERROR_INSUFFICIENT_BUFFER);
    T(odezza_score_reducer_run(r,&l,systems,(OdezzaScoreGrouping)mode,workspace,s.workspace_bytes,workspace,s.winner_bytes,counts,s.count_bytes,NULL)==ODEZZA_ERROR_INVALID_ARGUMENT);
    O(odezza_score_reducer_run(r,&l,systems,(OdezzaScoreGrouping)mode,workspace,s.workspace_bytes,winners,s.winner_bytes,counts,s.count_bytes,&report));
    C(cuMemcpyDtoH(actual,winners,s.winner_bytes)); C(cuMemcpyDtoH(actual_counts,counts,s.count_bytes));
    O(odezza_score_reducer_gather(r,&l,systems,(OdezzaScoreGrouping)mode,2,winners,s.winner_bytes,rows,s.group_count*k*8));
    C(cuMemcpyDtoH(gathered,rows,s.group_count*k*8));
    for(g=0;g<s.group_count;++g) {
        size_t valid=0,invalid=0,negative=0;
        for(j=0;j<elements;++j) {
            size_t index=mode==2?j:mode==0?g*s.configuration_count+j:(g/(1u<<bits))*s.configuration_count+j*(1u<<bits)+g%(1u<<bits);
            float f=scores[index];
            if(isfinite(f)&&f>=0&&f<FLT_MAX) { expected[valid].mse=f;expected[valid].reserved=0;expected[valid++].score_index=index; }
            else { ++invalid; if(isfinite(f)&&f<0) ++negative; }
        }
        qsort(expected,valid,sizeof(*expected),compare);
        T(actual_counts[g].valid==valid && actual_counts[g].invalid==invalid && actual_counts[g].negative==negative);
        for(j=0;j<k;++j) {
            OdezzaScoreWinner w=actual[g*k+j]; OdezzaScoreIndex decoded;
            if(j>=valid) { T(w.mse==FLT_MAX&&w.score_index==UINT64_MAX&&isnan(gathered[(g*k+j)*2])); continue; }
            T(memcmp(&w,&expected[j],sizeof(w))==0);
            O(odezza_score_decode_index(&l,systems,w.score_index,&decoded));
            T(decoded.system_index*s.configuration_count+(decoded.bank_index<<bits)+decoded.permutation==w.score_index);
            T(gathered[(g*k+j)*2]==constants[(decoded.system_index*banks+decoded.bank_index)*2]);
            T(gathered[(g*k+j)*2+1]==constants[(decoded.system_index*banks+decoded.bank_index)*2+1]);
        }
    }
    if(n>1000000) {
        for(i=0;i<10;++i) { start=now(); C(cuMemcpyDtoH(scores,l.mse_output_device,n*4)); download+=now()-start;
            start=now(); O(odezza_score_reducer_run(r,&l,systems,(OdezzaScoreGrouping)mode,workspace,s.workspace_bytes,winners,s.winner_bytes,counts,s.count_bytes,&report));
            C(cuMemcpyDtoH(actual,winners,s.winner_bytes)); C(cuMemcpyDtoH(actual_counts,counts,s.count_bytes)); reduced+=now()-start; }
        printf("{\"scores\":%zu,\"k\":%u,\"grouping\":%u,\"raw_bytes\":%zu,\"reduced_bytes\":%zu,\"raw_download_ms\":%.6f,\"reduce_and_download_ms\":%.6f,\"last_kernel_ms\":%.6f}\n",n,k,mode,n*4,s.winner_bytes+s.count_bytes,download*100,reduced*100,report.kernel_seconds*1000);
    }
    C(cuMemFree(rows)); C(cuMemFree(counts)); C(cuMemFree(winners)); C(cuMemFree(workspace)); C(cuMemFree(l.constant_banks_device)); C(cuMemFree(l.mse_output_device));
    free(gathered);free(actual_counts);free(expected);free(actual);free(constants);free(scores);
}
int main(int argc,char **argv) {
    CUdevice d; CUcontext c; int major,minor; unsigned k,mode; OdezzaScoreReducer *r=NULL;
    OdezzaScoringLaunch l={0}; OdezzaScoreIndex index; OdezzaScoreReductionSize size;
    l.constant_bank_count=3;l.active_toggle_count=32;
    O(odezza_score_decode_index(&l,2,(UINT64_C(5)<<32)+0xffffffff,&index));
    T(index.system_index==1&&index.bank_index==2&&index.permutation==UINT32_MAX);
    T(odezza_score_reduction_requirements(&l,SIZE_MAX,ODEZZA_SCORE_GLOBAL,1,&size)==ODEZZA_ERROR_OVERFLOW);
    l.active_toggle_count=33;T(odezza_score_decode_index(&l,1,0,&index)==ODEZZA_ERROR_INVALID_ARGUMENT);
    C(cuInit(0)); C(cuDeviceGet(&d,argc>1?atoi(argv[1]):0)); C(cuDevicePrimaryCtxRetain(&c,d)); C(cuCtxSetCurrent(c));
    C(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,d)); C(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,d));
    {
        unsigned ks[]={1,3,4,16,17,64,255,256};size_t x;
        OdezzaResult result=odezza_score_reducer_create((unsigned)(major*10+minor),1,&r);
        if(result) {char error[2048];odezza_score_reducer_write_error(r,error,sizeof(error));fprintf(stderr,"compile: %s\n",error);} O(result);
        T(odezza_score_reducer_set_k(r,0)==ODEZZA_ERROR_INVALID_ARGUMENT);
        T(odezza_score_reducer_set_k(r,257)==ODEZZA_ERROR_INVALID_ARGUMENT);
        for(x=0;x<sizeof(ks)/sizeof(*ks);x++) {
            k=ks[x];O(odezza_score_reducer_set_k(r,k));
            for(mode=0;mode<3;++mode) {
                test(r,k,3,3,0,mode);test(r,k,33,1027,2,mode);
                test(r,k,3,2049,1,mode);test(r,k,2,65537,1,mode);
            }
            // Multiple merge levels; index ties and exact negative-zero bits.
            for(pattern=1;pattern<=2;pattern++)test(r,k,2,131073,0,2);
            pattern=0;
        }
        test(r,256,2,1048577,0,2);
        O(odezza_score_reducer_destroy(r));
    }
    C(cuDevicePrimaryCtxRelease(d)); puts("score reducer checks passed");return 0;
}
