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
#include "o_score_reduce.h"
#include "o_odezza_internal.h"
#include "o_score_reduce_source.h"
#include "o_sampled_constants_source.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef ODEZZA_CUDA_INCLUDE_DIR
#define ODEZZA_CUDA_INCLUDE_DIR "/usr/local/cuda/include"
#endif

struct OdezzaScoreReducer {
    uint32_t k;
    int ready;
    CUmodule module;
    CUfunction reduce[6], merge[6], gather;
    CUstream stream;
    CUevent start, stop;
    char error[2048];
};
static OdezzaResult cuda_result(OdezzaScoreReducer *r, CUresult result) {
    const char *message = "unknown CUDA error";
    if (result == CUDA_SUCCESS) return ODEZZA_SUCCESS;
    (void)cuGetErrorString(result, &message);
    snprintf(r->error, sizeof(r->error), "%s", message);
    return ODEZZA_ERROR_CUDA;
}
#define GPU(call) do { OdezzaResult result_ = cuda_result(r, (call)); if (result_) return result_; } while (0)
static int mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (b && a > UINT64_MAX / b) return 0;
    *out = a * b; return 1;
}
static OdezzaResult layout(const OdezzaScoringLaunch *l, size_t systems, uint64_t *configs, uint64_t *scores) {
    if (!l || !systems || !l->constant_bank_count || l->active_toggle_count > 32) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *configs = (uint64_t)l->constant_bank_count << l->active_toggle_count;
    if (!mul(systems, *configs, scores)) return ODEZZA_ERROR_OVERFLOW;
    return ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_reduction_requirements(const OdezzaScoringLaunch *l, size_t systems,
    OdezzaScoreGrouping mode, uint32_t k, OdezzaScoreReductionSize *out) {
    OdezzaScoreReductionSize s = {0};
    uint64_t elements, tickets, bytes, counts;
    OdezzaResult result;
    if (!out || k < 1 || k > 256 || mode < 0 || mode > ODEZZA_SCORE_GLOBAL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    result = layout(l, systems, &s.configuration_count, &s.score_count);
    if (result) return result;
    if (s.score_count > SIZE_MAX / sizeof(float)) return ODEZZA_ERROR_OVERFLOW;
    s.group_count = mode == ODEZZA_SCORE_GLOBAL ? 1 : systems;
    if (mode == ODEZZA_SCORE_BY_SYSTEM_PERMUTATION &&
        !mul(systems, UINT64_C(1) << l->active_toggle_count, &s.group_count)) return ODEZZA_ERROR_OVERFLOW;
    elements = s.score_count / s.group_count;
    if (1 + (elements - 1) / 2048 > UINT32_MAX) return ODEZZA_ERROR_OVERFLOW;
    s.tiles_per_group = (uint32_t)(1 + (elements - 1) / 2048);
    if (!mul(s.group_count, s.tiles_per_group, &tickets) || tickets > INT32_MAX) return ODEZZA_ERROR_OVERFLOW;
    if (!mul(tickets, k * sizeof(OdezzaScoreWinner), &bytes) ||
        !mul(tickets, sizeof(OdezzaScoreCounts), &counts) || bytes > SIZE_MAX || counts > SIZE_MAX - bytes)
        return ODEZZA_ERROR_OVERFLOW;
    if (bytes + counts > SIZE_MAX / 2) return ODEZZA_ERROR_OVERFLOW;
    /* Two reusable levels; retain the nonzero workspace contract for all callers.
     * A one-tile run writes final outputs directly and needs no merge launch. */
    s.workspace_bytes = (size_t)(bytes + counts) * (s.tiles_per_group > 1 ? 2 : 1);
    s.winner_bytes = (size_t)(bytes / s.tiles_per_group);
    s.count_bytes = (size_t)(counts / s.tiles_per_group);
    *out = s; return ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_decode_index(const OdezzaScoringLaunch *l, size_t systems, uint64_t index, OdezzaScoreIndex *out) {
    uint64_t configs, scores;
    OdezzaResult result;
    if (!out) return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    result = layout(l, systems, &configs, &scores);
    if (result) return result;
    if (index >= scores) return ODEZZA_ERROR_INVALID_ARGUMENT;
    out->system_index = index / configs;
    out->configuration_index = index % configs;
    out->bank_index = out->configuration_index >> l->active_toggle_count;
    out->permutation = (uint32_t)(out->configuration_index & ((UINT64_C(1) << l->active_toggle_count) - 1));
    return ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_reducer_write_error(const OdezzaScoreReducer *r, char *out, size_t capacity) {
    if (!r || !out) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (capacity <= strlen(r->error)) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    memcpy(out, r->error, strlen(r->error) + 1); return ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_reducer_destroy(OdezzaScoreReducer *r) {
    int failed = 0;
    if (!r) return ODEZZA_SUCCESS;
    if (r->stream && cuStreamSynchronize(r->stream) != CUDA_SUCCESS) failed = 1;
    if (r->start && cuEventDestroy(r->start) != CUDA_SUCCESS) failed = 1;
    if (r->stop && cuEventDestroy(r->stop) != CUDA_SUCCESS) failed = 1;
    if (r->stream && cuStreamDestroy(r->stream) != CUDA_SUCCESS) failed = 1;
    if (r->module && cuModuleUnload(r->module) != CUDA_SUCCESS) failed = 1;
    free(r); return failed ? ODEZZA_ERROR_CUDA : ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_reducer_create(uint32_t sm, uint32_t k, OdezzaScoreReducer **out) {
    OdezzaScoreReducer *r;
    OdezzaNvrtcCompilation *compilation = NULL;
    OdezzaResult result;
    char arch[64], *source, *cubin = NULL;
    const char *options[] = {arch, "--std=c++17",
        "--include-path=" ODEZZA_CUDA_INCLUDE_DIR "/cccl",
        "--include-path=" ODEZZA_CUDA_INCLUDE_DIR};
    size_t i, length = 64, position, bytes = 0, log_bytes = 0;
    if (!out) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out = NULL;
    if (k < 1 || k > 256 || sm < 70 || sm > 999) return ODEZZA_ERROR_INVALID_ARGUMENT;
    r = calloc(1, sizeof(*r));
    if (!r) return ODEZZA_ERROR_ALLOCATION;
    *out = r; r->k = k;
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%u", sm);
    for (i = 0; i < sizeof(o_reduce_source)/sizeof(*o_reduce_source); ++i) length += strlen(o_reduce_source[i]);
    for (i=0;i<sizeof(o_sampled_source)/sizeof(*o_sampled_source);++i) length+=strlen(o_sampled_source[i]);
    source = malloc(length);
    if (!source) return ODEZZA_ERROR_ALLOCATION;
    position = 0;
    for (i=0;i<sizeof(o_sampled_source)/sizeof(*o_sampled_source);++i) {
        size_t n=strlen(o_sampled_source[i]);memcpy(source+position,o_sampled_source[i],n);position+=n;
    }
    for (i = 0; i < sizeof(o_reduce_source)/sizeof(*o_reduce_source); ++i) {
        size_t n = strlen(o_reduce_source[i]); memcpy(source + position, o_reduce_source[i], n); position += n;
    }
    source[position] = 0;
    result = odezza_nvrtc_compilation_create(source, "score_reduce.cu", options, 4, &compilation);
    free(source);
    if (!result) { OdezzaResult compiled; result = odezza_nvrtc_compilation_result(compilation, &compiled); if (!result) result = compiled; }
    if (compilation && odezza_nvrtc_compilation_log_size(compilation, &log_bytes) == ODEZZA_SUCCESS && log_bytes) {
        char *log = malloc(log_bytes);
        if (log) { if (!odezza_nvrtc_compilation_write_log(compilation, log, log_bytes)) snprintf(r->error, sizeof(r->error), "%s", log); free(log); }
    }
    if (!result) result = odezza_nvrtc_compilation_cubin_size(compilation, &bytes);
    if (!result) { cubin = malloc(bytes); if (!cubin) result = ODEZZA_ERROR_ALLOCATION; }
    if (!result) result = odezza_nvrtc_compilation_write_cubin(compilation, cubin, bytes);
    if (!result) result = cuda_result(r, cuModuleLoadData(&r->module, cubin));
    free(cubin);
    (void)odezza_nvrtc_compilation_destroy(compilation);
    if (result) return result;
    for (i=0;i<5;++i) {
        char name[64]; unsigned items=i?1u<<(i-1):0;
        snprintf(name,sizeof(name),"reduce_scores_%u",items);
        GPU(cuModuleGetFunction(&r->reduce[i],r->module,name));
        snprintf(name,sizeof(name),"merge_scores_%u",items);
        GPU(cuModuleGetFunction(&r->merge[i],r->module,name));
    }
    GPU(cuModuleGetFunction(&r->reduce[5],r->module,"reduce_scores_small"));
    GPU(cuModuleGetFunction(&r->merge[5],r->module,"merge_scores_small"));
    GPU(cuModuleGetFunction(&r->gather, r->module, "gather_constants"));
    GPU(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));
    GPU(cuEventCreate(&r->start, CU_EVENT_DEFAULT));
    GPU(cuEventCreate(&r->stop, CU_EVENT_DEFAULT));
    r->ready = 1; r->error[0] = 0; return ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_reducer_set_k(OdezzaScoreReducer *r,uint32_t k) {
    if (!r || !r->ready || k<1 || k>256) return ODEZZA_ERROR_INVALID_ARGUMENT;
    r->k=k;
    return ODEZZA_SUCCESS;
}
static unsigned shape(uint64_t elements,uint32_t k) {
    unsigned index=1;
    if (k==1) return 0;
    if (k<=4) return 5;
    while (elements>256 && index<4) { elements=(elements+1)/2; ++index; }
    return index;
}
static int range(CUdeviceptr p, size_t bytes, unsigned alignment) {
    return p && p % alignment == 0 && bytes <= UINT64_MAX - p;
}
static int overlap(CUdeviceptr a, size_t na, CUdeviceptr b, size_t nb) {
    return a < b + nb && b < a + na;
}
OdezzaResult odezza_score_reducer_run(OdezzaScoreReducer *r, const OdezzaScoringLaunch *l,
    size_t systems, OdezzaScoreGrouping grouping, CUdeviceptr workspace, size_t workspace_bytes,
    CUdeviceptr winners, size_t winner_bytes, CUdeviceptr counts, size_t count_bytes,
    OdezzaScoreReductionReport *report) {
    OdezzaScoreReductionSize s;
    OdezzaResult result;
    uint64_t permutations, elements;
    CUdeviceptr level_winners[2],level_counts[2],out_winners,out_counts;
    unsigned mode = (unsigned)grouping;
    float ms;
    void *first[9];
    uint32_t previous,next,fanout;
    unsigned level=0;
    CUdeviceptr pointers[4]; size_t sizes[4], i, j;
    if (report) memset(report, 0, sizeof(*report));
    if (!r || !r->ready) return ODEZZA_ERROR_INVALID_ARGUMENT;
    result = odezza_score_reduction_requirements(l, systems, grouping, r->k, &s);
    if (result) return result;
    if (workspace_bytes < s.workspace_bytes || winner_bytes < s.winner_bytes || count_bytes < s.count_bytes)
        return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    pointers[0]=l->mse_output_device; pointers[1]=workspace; pointers[2]=winners; pointers[3]=counts;
    sizes[0]=(size_t)s.score_count*4; sizes[1]=s.workspace_bytes; sizes[2]=s.winner_bytes; sizes[3]=s.count_bytes;
    for (i=0;i<4;++i) {
        if (!range(pointers[i],sizes[i],i?8:4)) return ODEZZA_ERROR_INVALID_ARGUMENT;
        for (j=0;j<i;++j) if (overlap(pointers[i],sizes[i],pointers[j],sizes[j])) return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    permutations=UINT64_C(1)<<l->active_toggle_count; elements=s.score_count/s.group_count;
    level_winners[0]=workspace;
    level_counts[0]=workspace+s.winner_bytes*s.tiles_per_group;
    level_winners[1]=level_counts[0]+s.count_bytes*s.tiles_per_group;
    level_counts[1]=level_winners[1]+s.winner_bytes*s.tiles_per_group;
    out_winners=s.tiles_per_group==1?winners:level_winners[0];
    out_counts=s.tiles_per_group==1?counts:level_counts[0];
    first[0]=(void *)&l->mse_output_device; first[1]=&s.configuration_count; first[2]=&permutations; first[3]=&elements;
    first[4]=&s.tiles_per_group; first[5]=&mode; first[6]=&r->k; first[7]=&out_winners; first[8]=&out_counts;
    if(l->input_ready_event)GPU(cuStreamWaitEvent(r->stream,l->input_ready_event,0));
    GPU(cuEventRecord(r->start,r->stream));
    GPU(cuLaunchKernel(r->reduce[shape(elements,r->k)],(unsigned)(s.group_count*s.tiles_per_group),1,1,256,1,1,0,r->stream,first,NULL));
    previous=s.tiles_per_group;
    fanout=2048/r->k;
    while(previous>1) {
        void *args[]={ (void *)&l->mse_output_device,&level_winners[level],&level_counts[level],
            &previous,&next,&fanout,&r->k,&out_winners,&out_counts };
        next=1+(previous-1)/fanout;
        out_winners=next==1?winners:level_winners[1-level];
        out_counts=next==1?counts:level_counts[1-level];
        GPU(cuLaunchKernel(r->merge[shape((uint64_t)(previous<fanout?previous:fanout)*r->k,r->k)],
            (unsigned)(s.group_count*next),1,1,256,1,1,0,r->stream,args,NULL));
        previous=next;level=1-level;
    }
    GPU(cuEventRecord(r->stop,r->stream)); GPU(cuEventSynchronize(r->stop));
    GPU(cuEventElapsedTime(&ms,r->start,r->stop));
    if (report) { report->kernel_seconds=ms*0.001; report->group_count=s.group_count; report->score_count=s.score_count; }
    return ODEZZA_SUCCESS;
}
OdezzaResult odezza_score_reducer_gather(OdezzaScoreReducer *r, const OdezzaScoringLaunch *l,
    size_t systems, OdezzaScoreGrouping grouping, uint32_t constants, CUdeviceptr winners,
    size_t winner_bytes, CUdeviceptr rows, size_t row_bytes) {
    OdezzaScoreReductionSize s;
    OdezzaResult result;
    uint64_t entries, values, bank_values, banks;
    unsigned blocks;
    void *args[13];
    if (!r || !r->ready) return ODEZZA_ERROR_INVALID_ARGUMENT;
    result=odezza_score_reduction_requirements(l,systems,grouping,r->k,&s);
    if (result) return result;
    if (!constants) return ODEZZA_SUCCESS;
    entries=s.group_count*r->k; banks=l->constant_bank_count;
    if (!mul(entries,constants,&values) || !mul(systems,banks,&bank_values) || !mul(bank_values,constants,&bank_values) ||
        values>SIZE_MAX/4 || bank_values>SIZE_MAX/4) return ODEZZA_ERROR_OVERFLOW;
    if (winner_bytes<s.winner_bytes || row_bytes<values*4) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    if (!range(winners,s.winner_bytes,8) || !range(rows,(size_t)values*4,4) || (!l->sampled_parameters_device&&!range(l->constant_banks_device,(size_t)bank_values*4,4)) ||
        overlap(winners,s.winner_bytes,rows,(size_t)values*4) || (!l->sampled_parameters_device&&overlap(l->constant_banks_device,(size_t)bank_values*4,rows,(size_t)values*4)))
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if(l->sampled_parameters_device) {
        uint64_t descriptors;
        if(!mul(systems,constants,&descriptors)||descriptors>SIZE_MAX/sizeof(OdezzaSampledParameter)||
            !l->rng_pool_size||l->rng_pool_size>SIZE_MAX/4||(!l->uniform_pool_device&&!l->normal_pool_device))return ODEZZA_ERROR_INVALID_ARGUMENT;
        if(!range(l->sampled_parameters_device,(size_t)descriptors*sizeof(OdezzaSampledParameter),8)||
            overlap(l->sampled_parameters_device,(size_t)descriptors*sizeof(OdezzaSampledParameter),rows,(size_t)values*4))return ODEZZA_ERROR_INVALID_ARGUMENT;
        if(l->uniform_pool_device&&(!range(l->uniform_pool_device,(size_t)l->rng_pool_size*4,4)||overlap(l->uniform_pool_device,(size_t)l->rng_pool_size*4,rows,(size_t)values*4)))return ODEZZA_ERROR_INVALID_ARGUMENT;
        if(l->normal_pool_device&&(!range(l->normal_pool_device,(size_t)l->rng_pool_size*4,4)||overlap(l->normal_pool_device,(size_t)l->rng_pool_size*4,rows,(size_t)values*4)))return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    blocks=values>65535*256?65535:(unsigned)(1+(values-1)/256);
    args[0]=&winners; args[1]=(void *)&l->constant_banks_device; args[2]=&entries; args[3]=&s.configuration_count;
    args[4]=&banks; args[5]=(void *)&l->active_toggle_count; args[6]=&constants; args[7]=&s.score_count; args[8]=&rows;
    args[9]=(void *)&l->sampled_parameters_device;args[10]=(void *)&l->uniform_pool_device;
    args[11]=(void *)&l->normal_pool_device;args[12]=(void *)&l->rng_pool_size;
    if(l->input_ready_event)GPU(cuStreamWaitEvent(r->stream,l->input_ready_event,0));
    GPU(cuLaunchKernel(r->gather,blocks,1,1,256,1,1,0,r->stream,args,NULL));
    GPU(cuEventRecord(r->stop,r->stream));GPU(cuEventSynchronize(r->stop));return ODEZZA_SUCCESS;
}
