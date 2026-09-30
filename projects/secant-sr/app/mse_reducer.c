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

#include "mse_reducer.h"

#include <nvrtc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SECANT_SR_MSE_REDUCER_THREADS 256u
#define SECANT_SR_MSE_REDUCER_HISTOGRAM_BINS 256u
#define SECANT_SR_MSE_REDUCER_MAX_BLOCKS 4096u
#define SECANT_SR_MSE_REDUCER_ARTIFACT_VERSION 1u

static const char secant_sr_mse_reducer_artifact_name[] = "mse_reducer";

static double
secant_sr_mse_reducer_seconds_get(void) {
    struct timespec value;

    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static const char secant_sr_mse_reducer_source[] =
    "struct Entry { float mse; float robustness; unsigned int ast_idx; unsigned int setting_idx; };\n"
    "__device__ unsigned long long entry_key(Entry value) {\n"
    "    const unsigned int bits = __float_as_uint(value.mse);\n"
    "    if ((bits & 0x80000000u) != 0u || (bits & 0x7f800000u) == 0x7f800000u) {\n"
    "        return 0xffffffff00000000ull | (unsigned long long)value.ast_idx;\n"
    "    }\n"
    "    return ((unsigned long long)bits << 32u) | (unsigned long long)value.ast_idx;\n"
    "}\n"
    "extern \"C\" __global__ void reduce_settings(\n"
    "    const float* sse, unsigned long long num_asts, unsigned long long num_settings,\n"
    "    unsigned long long leading_dimension, float inverse_num_rows, float inverse_target_ssd, Entry* output) {\n"
    "    __shared__ float values[256];\n"
    "    __shared__ float robustness[256];\n"
    "    __shared__ unsigned int settings[256];\n"
    "    const unsigned long long ast = (unsigned long long)blockIdx.x;\n"
    "    float best = __int_as_float(0x7f800000);\n"
    "    float robustness_sum = 0.0f;\n"
    "    unsigned int best_setting = 0xffffffffu;\n"
    "    if (ast < num_asts) {\n"
    "        for (unsigned long long setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"
    "            const float candidate = sse[ast * leading_dimension + setting];\n"
    "            const unsigned int bits = __float_as_uint(candidate);\n"
    "            if ((bits & 0x80000000u) == 0u && (bits & 0x7f800000u) != 0x7f800000u &&\n"
    "                (candidate < best || (candidate == best && setting < best_setting))) {\n"
    "                best = candidate;\n"
    "                best_setting = (unsigned int)setting;\n"
    "            }\n"
    "            if ((bits & 0x80000000u) == 0u && (bits & 0x7f800000u) != 0x7f800000u) {\n"
    "                robustness_sum += fminf(fmaxf(1.0f - candidate * inverse_target_ssd, 0.0f), 1.0f);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    values[threadIdx.x] = best;\n"
    "    robustness[threadIdx.x] = robustness_sum;\n"
    "    settings[threadIdx.x] = best_setting;\n"
    "    __syncthreads();\n"
    "    for (unsigned int offset = 128u; offset != 0u; offset >>= 1u) {\n"
    "        if (threadIdx.x < offset) {\n"
    "            const float rhs = values[threadIdx.x + offset];\n"
    "            const unsigned int rhs_setting = settings[threadIdx.x + offset];\n"
    "            if (rhs < values[threadIdx.x] ||\n"
    "                (rhs == values[threadIdx.x] && rhs_setting < settings[threadIdx.x])) {\n"
    "                values[threadIdx.x] = rhs;\n"
    "                settings[threadIdx.x] = rhs_setting;\n"
    "            }\n"
    "            robustness[threadIdx.x] += robustness[threadIdx.x + offset];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (threadIdx.x == 0u && ast < num_asts) {\n"
    "        Entry result;\n"
    "        result.mse = values[0] * inverse_num_rows;\n"
    "        result.robustness = robustness[0] / (float)num_settings;\n"
    "        result.ast_idx = (unsigned int)ast;\n"
    "        result.setting_idx = settings[0] == 0xffffffffu ? 0u : settings[0];\n"
    "        output[ast] = result;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void histogram_keys(\n"
    "    const Entry* entries, unsigned long long num_entries, unsigned int shift,\n"
    "    unsigned long long prefix, unsigned long long prefix_mask, unsigned long long* histogram) {\n"
    "    const unsigned long long begin =\n"
    "        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;\n"
    "    for (unsigned long long idx = begin; idx < num_entries; idx += stride) {\n"
    "        const unsigned long long key = entry_key(entries[idx]);\n"
    "        if ((key & prefix_mask) == prefix) {\n"
    "            atomicAdd(histogram + ((key >> shift) & 255ull), 1ull);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void gather_keys(\n"
    "    const Entry* entries, unsigned long long num_entries, unsigned long long threshold,\n"
    "    Entry* output, unsigned int* output_count) {\n"
    "    const unsigned long long begin =\n"
    "        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;\n"
    "    for (unsigned long long idx = begin; idx < num_entries; idx += stride) {\n"
    "        const Entry value = entries[idx];\n"
    "        if (entry_key(value) <= threshold) {\n"
    "            output[atomicAdd(output_count, 1u)] = value;\n"
    "        }\n"
    "    }\n"
    "}\n";

static int
secant_sr_mse_entry_compare(const void* lhs_pointer, const void* rhs_pointer) {
    const SecantSRMSEEntry* lhs = lhs_pointer;
    const SecantSRMSEEntry* rhs = rhs_pointer;

    if (lhs->mse < rhs->mse) {
        return -1;
    }
    if (lhs->mse > rhs->mse) {
        return 1;
    }
    if (lhs->ast_idx < rhs->ast_idx) {
        return -1;
    }
    return lhs->ast_idx > rhs->ast_idx;
}

static int
secant_sr_mse_reducer_cubin_compile(
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char architecture[64];
    const char* options[4];
    nvrtcProgram program = NULL;
    nvrtcResult result;
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d", major, minor) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    result = nvrtcCreateProgram(&program, secant_sr_mse_reducer_source, "secant_sr_mse_reducer.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, 4, options);
    }
    if (result == NVRTC_SUCCESS) {
        result = nvrtcGetCUBINSize(program, &cubin_size);
    }
    if (result == NVRTC_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin != NULL && nvrtcGetCUBIN(program, (char*)cubin) == NVRTC_SUCCESS) {
            success = 1;
        }
    }
    if (!success && program != NULL) {
        size_t log_size = 0u;

        if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1u) {
            char* log = malloc(log_size);

            if (log != NULL) {
                if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    fprintf(stderr, "%s\n", log);
                }
                free(log);
            }
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    if (!success) {
        free(cubin);
        return 0;
    }
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

static int
secant_sr_mse_reducer_module_load(SecantSRMSEReducer* reducer, const unsigned char* cubin) {
    if (cuModuleLoadData(&reducer->module, cubin) == CUDA_SUCCESS &&
        cuModuleGetFunction(&reducer->reduce_settings_function, reducer->module, "reduce_settings") == CUDA_SUCCESS &&
        cuModuleGetFunction(&reducer->histogram_function, reducer->module, "histogram_keys") == CUDA_SUCCESS &&
        cuModuleGetFunction(&reducer->gather_function, reducer->module, "gather_keys") == CUDA_SUCCESS) {
        return 1;
    }
    if (reducer->module != NULL) {
        (void)cuModuleUnload(reducer->module);
    }
    reducer->module = NULL;
    reducer->reduce_settings_function = NULL;
    reducer->histogram_function = NULL;
    reducer->gather_function = NULL;
    return 0;
}

int
secant_sr_mse_reducer_create(
    size_t ast_capacity,
    size_t top_k_capacity,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRMSEReducer* reducer
) {
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;
    size_t best_bytes;
    size_t selected_bytes;
    double stored_compile_seconds = 0.0;
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    int cache_hit = 0;
    int compiled = 0;

    if (reducer == NULL) {
        return 0;
    }
    memset(reducer, 0, sizeof(*reducer));
    if (session == NULL || session->context == NULL || ast_capacity == 0u || ast_capacity > UINT32_MAX ||
        top_k_capacity == 0u || top_k_capacity > ast_capacity ||
        ast_capacity > SIZE_MAX / sizeof(SecantSRMSEEntry) ||
        top_k_capacity > SIZE_MAX / sizeof(SecantSRMSEEntry)) {
        return 0;
    }
    best_bytes = ast_capacity * sizeof(SecantSRMSEEntry);
    selected_bytes = top_k_capacity * sizeof(SecantSRMSEEntry);
    reducer->context = session->context;
    if (nvrtcVersion(&nvrtc_major, &nvrtc_minor) != NVRTC_SUCCESS) {
        return 0;
    }
    if (cache != NULL) {
        const double begin = secant_sr_mse_reducer_seconds_get();

        if (!secant_sr_cubin_cache_artifact_lookup(
                cache,
                secant_sr_mse_reducer_artifact_name,
                SECANT_SR_MSE_REDUCER_ARTIFACT_VERSION,
                session->compute_capability_major,
                session->compute_capability_minor,
                nvrtc_major,
                nvrtc_minor,
                1,
                1,
                &cubin,
                &cubin_size,
                &stored_compile_seconds,
                &cache_hit)) {
            return 0;
        }
        reducer->cache_lookup_seconds = secant_sr_mse_reducer_seconds_get() - begin;
        if (cache_hit) {
            reducer->cache_hits = 1u;
            reducer->estimated_uncached_compile_seconds = stored_compile_seconds;
        } else {
            reducer->cache_misses = 1u;
        }
    }
    if (cache_hit && !secant_sr_mse_reducer_module_load(reducer, cubin)) {
        free(cubin);
        cubin = NULL;
        cubin_size = 0u;
        reducer->cache_hits = 0u;
        reducer->cache_misses = 1u;
        reducer->cache_invalidations = 1u;
        cache_hit = 0;
    }
    if (!cache_hit) {
        const double begin = secant_sr_mse_reducer_seconds_get();

        if (!secant_sr_mse_reducer_cubin_compile(
                session->compute_capability_major, session->compute_capability_minor, &cubin, &cubin_size)) {
            return 0;
        }
        reducer->compile_seconds = secant_sr_mse_reducer_seconds_get() - begin;
        reducer->estimated_uncached_compile_seconds = reducer->compile_seconds;
        stored_compile_seconds = reducer->compile_seconds;
        compiled = 1;
        if (!secant_sr_mse_reducer_module_load(reducer, cubin)) {
            free(cubin);
            return 0;
        }
    }
    if (compiled && cache != NULL) {
        const double begin = secant_sr_mse_reducer_seconds_get();

        if (!secant_sr_cubin_cache_artifact_store(
                cache,
                secant_sr_mse_reducer_artifact_name,
                SECANT_SR_MSE_REDUCER_ARTIFACT_VERSION,
                session->compute_capability_major,
                session->compute_capability_minor,
                nvrtc_major,
                nvrtc_minor,
                1,
                1,
                cubin,
                cubin_size,
                stored_compile_seconds)) {
            free(cubin);
            secant_sr_mse_reducer_destroy(reducer);
            return 0;
        }
        reducer->cache_store_seconds = secant_sr_mse_reducer_seconds_get() - begin;
    }
    if (cuStreamCreate(&reducer->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
        cuMemAlloc(&reducer->best_entries, best_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&reducer->histogram, SECANT_SR_MSE_REDUCER_HISTOGRAM_BINS * sizeof(uint64_t)) != CUDA_SUCCESS ||
        cuMemAlloc(&reducer->selected_entries, selected_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&reducer->selected_count, sizeof(uint32_t)) != CUDA_SUCCESS) {
        free(cubin);
        secant_sr_mse_reducer_destroy(reducer);
        return 0;
    }
    free(cubin);
    reducer->ast_capacity = ast_capacity;
    reducer->top_k_capacity = top_k_capacity;
    return 1;
}

static int
secant_sr_mse_reducer_launch(
    SecantSRMSEReducer* reducer,
    CUdeviceptr sse,
    size_t num_asts,
    size_t num_settings,
    size_t sse_leading_dimension,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    unsigned long long ast_count = num_asts;
    unsigned long long setting_count = num_settings;
    unsigned long long leading_dimension = sse_leading_dimension;
    const float inverse_num_rows = 1.0f / (float)num_rows;
    const float inverse_target_ssd = (float)(1.0 / target_sum_squared_deviation);
    void* arguments[7];

    if (reducer == NULL || reducer->module == NULL || sse == 0u || num_asts == 0u ||
        num_asts > reducer->ast_capacity || num_settings == 0u || sse_leading_dimension < num_settings ||
        num_rows == 0u || target_sum_squared_deviation <= 0.0 || num_asts > UINT32_MAX) {
        return 0;
    }
    arguments[0] = &sse;
    arguments[1] = &ast_count;
    arguments[2] = &setting_count;
    arguments[3] = &leading_dimension;
    arguments[4] = (void*)&inverse_num_rows;
    arguments[5] = (void*)&inverse_target_ssd;
    arguments[6] = &reducer->best_entries;
    return cuLaunchKernel(
               reducer->reduce_settings_function,
               (unsigned int)num_asts,
               1u,
               1u,
               SECANT_SR_MSE_REDUCER_THREADS,
               1u,
               1u,
               0u,
               reducer->stream,
               arguments,
               NULL) == CUDA_SUCCESS;
}

int
secant_sr_mse_reducer_best_get(
    SecantSRMSEReducer* reducer,
    CUdeviceptr sse,
    size_t num_asts,
    size_t num_settings,
    size_t sse_leading_dimension,
    size_t num_rows,
    double target_sum_squared_deviation,
    SecantSRMSEEntry* entries,
    size_t entry_capacity
) {
    if (entries == NULL || entry_capacity < num_asts ||
        !secant_sr_mse_reducer_launch(
            reducer, sse, num_asts, num_settings, sse_leading_dimension, num_rows, target_sum_squared_deviation)) {
        return 0;
    }
    return cuMemcpyDtoHAsync(
               entries,
               reducer->best_entries,
               num_asts * sizeof(*entries),
               reducer->stream) == CUDA_SUCCESS &&
        cuStreamSynchronize(reducer->stream) == CUDA_SUCCESS;
}

int
secant_sr_mse_reducer_top_k_get(
    SecantSRMSEReducer* reducer,
    CUdeviceptr sse,
    size_t num_asts,
    size_t num_settings,
    size_t sse_leading_dimension,
    size_t num_rows,
    double target_sum_squared_deviation,
    size_t top_k,
    SecantSRMSEEntry* entries,
    size_t entry_capacity,
    size_t* num_entries_ret
) {
    uint64_t histogram[SECANT_SR_MSE_REDUCER_HISTOGRAM_BINS];
    uint64_t prefix = 0u;
    uint64_t prefix_mask = 0u;
    uint64_t threshold;
    size_t rank;
    size_t effective_top_k;
    unsigned int blocks;
    int shift;

    if (num_entries_ret == NULL) {
        return 0;
    }
    *num_entries_ret = 0u;
    if (reducer == NULL || entries == NULL || top_k == 0u || top_k > reducer->top_k_capacity ||
        entry_capacity < top_k ||
        !secant_sr_mse_reducer_launch(
            reducer, sse, num_asts, num_settings, sse_leading_dimension, num_rows, target_sum_squared_deviation)) {
        return 0;
    }
    blocks = (unsigned int)((num_asts + SECANT_SR_MSE_REDUCER_THREADS - 1u) / SECANT_SR_MSE_REDUCER_THREADS);
    if (blocks > SECANT_SR_MSE_REDUCER_MAX_BLOCKS) {
        blocks = SECANT_SR_MSE_REDUCER_MAX_BLOCKS;
    }
    effective_top_k = top_k < num_asts ? top_k : num_asts;
    rank = effective_top_k - 1u;
    for (shift = 56; shift >= 0; shift -= 8) {
        unsigned int shift_value = (unsigned int)shift;
        unsigned long long entry_count = num_asts;
        unsigned long long prefix_value = prefix;
        unsigned long long prefix_mask_value = prefix_mask;
        void* arguments[] = {
            &reducer->best_entries,
            &entry_count,
            &shift_value,
            &prefix_value,
            &prefix_mask_value,
            &reducer->histogram
        };
        size_t bin;

        if (cuMemsetD8Async(
                reducer->histogram,
                0u,
                SECANT_SR_MSE_REDUCER_HISTOGRAM_BINS * sizeof(uint64_t),
                reducer->stream) != CUDA_SUCCESS ||
            cuLaunchKernel(
                reducer->histogram_function,
                blocks,
                1u,
                1u,
                SECANT_SR_MSE_REDUCER_THREADS,
                1u,
                1u,
                0u,
                reducer->stream,
                arguments,
                NULL) != CUDA_SUCCESS ||
            cuMemcpyDtoHAsync(
                histogram,
                reducer->histogram,
                sizeof(histogram),
                reducer->stream) != CUDA_SUCCESS ||
            cuStreamSynchronize(reducer->stream) != CUDA_SUCCESS) {
            return 0;
        }
        if (shift == 56) {
            const uint64_t invalid = histogram[255u];
            const size_t valid_entries = invalid <= num_asts ? num_asts - (size_t)invalid : 0u;

            effective_top_k = top_k < valid_entries ? top_k : valid_entries;
            if (effective_top_k == 0u) {
                return 1;
            }
            rank = effective_top_k - 1u;
        }
        for (bin = 0u; bin < SECANT_SR_MSE_REDUCER_HISTOGRAM_BINS; ++bin) {
            if (rank < histogram[bin]) {
                prefix |= (uint64_t)bin << shift;
                prefix_mask |= UINT64_C(255) << shift;
                break;
            }
            rank -= (size_t)histogram[bin];
        }
        if (bin == SECANT_SR_MSE_REDUCER_HISTOGRAM_BINS) {
            return 0;
        }
    }
    threshold = prefix;
    {
        unsigned long long entry_count = num_asts;
        unsigned long long threshold_value = threshold;
        void* arguments[] = {
            &reducer->best_entries,
            &entry_count,
            &threshold_value,
            &reducer->selected_entries,
            &reducer->selected_count
        };
        uint32_t selected_count = 0u;

        if (cuMemsetD8Async(reducer->selected_count, 0u, sizeof(uint32_t), reducer->stream) != CUDA_SUCCESS ||
            cuLaunchKernel(
                reducer->gather_function,
                blocks,
                1u,
                1u,
                SECANT_SR_MSE_REDUCER_THREADS,
                1u,
                1u,
                0u,
                reducer->stream,
                arguments,
                NULL) != CUDA_SUCCESS ||
            cuMemcpyDtoHAsync(
                &selected_count,
                reducer->selected_count,
                sizeof(selected_count),
                reducer->stream) != CUDA_SUCCESS ||
            cuStreamSynchronize(reducer->stream) != CUDA_SUCCESS ||
            selected_count != effective_top_k || selected_count > entry_capacity ||
            cuMemcpyDtoHAsync(
                entries,
                reducer->selected_entries,
                selected_count * sizeof(*entries),
                reducer->stream) != CUDA_SUCCESS ||
            cuStreamSynchronize(reducer->stream) != CUDA_SUCCESS) {
            return 0;
        }
        qsort(entries, selected_count, sizeof(*entries), secant_sr_mse_entry_compare);
        *num_entries_ret = selected_count;
    }
    return 1;
}

void
secant_sr_mse_reducer_destroy(SecantSRMSEReducer* reducer) {
    if (reducer == NULL) {
        return;
    }
    if (reducer->selected_count != 0u) {
        (void)cuMemFree(reducer->selected_count);
    }
    if (reducer->selected_entries != 0u) {
        (void)cuMemFree(reducer->selected_entries);
    }
    if (reducer->histogram != 0u) {
        (void)cuMemFree(reducer->histogram);
    }
    if (reducer->best_entries != 0u) {
        (void)cuMemFree(reducer->best_entries);
    }
    if (reducer->stream != NULL) {
        (void)cuStreamDestroy(reducer->stream);
    }
    if (reducer->module != NULL) {
        (void)cuModuleUnload(reducer->module);
    }
    memset(reducer, 0, sizeof(*reducer));
}
