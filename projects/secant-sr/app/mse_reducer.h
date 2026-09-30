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
#ifndef SECANT_SR_APP_MSE_REDUCER_H_INCLUDED
#define SECANT_SR_APP_MSE_REDUCER_H_INCLUDED

#include "cubin_evaluator.h"

typedef struct SecantSRMSEEntry {
    float mse;
    float robustness;
    uint32_t ast_idx;
    uint32_t setting_idx;
} SecantSRMSEEntry;

typedef struct SecantSRMSEReducer {
    CUcontext context;
    CUmodule module;
    CUfunction reduce_settings_function;
    CUfunction histogram_function;
    CUfunction gather_function;
    CUstream stream;
    CUdeviceptr best_entries;
    CUdeviceptr histogram;
    CUdeviceptr selected_entries;
    CUdeviceptr selected_count;
    size_t ast_capacity;
    size_t top_k_capacity;
    size_t cache_hits;
    size_t cache_misses;
    size_t cache_invalidations;
    double cache_lookup_seconds;
    double cache_store_seconds;
    double compile_seconds;
    double estimated_uncached_compile_seconds;
} SecantSRMSEReducer;

int secant_sr_mse_reducer_create(
    size_t ast_capacity,
    size_t top_k_capacity,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRMSEReducer* reducer
);

int secant_sr_mse_reducer_best_get(
    SecantSRMSEReducer* reducer,
    CUdeviceptr sse,
    size_t num_asts,
    size_t num_settings,
    size_t sse_leading_dimension,
    size_t num_rows,
    double target_sum_squared_deviation,
    SecantSRMSEEntry* entries,
    size_t entry_capacity
);

int secant_sr_mse_reducer_top_k_get(
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
);

void secant_sr_mse_reducer_destroy(SecantSRMSEReducer* reducer);

#endif /* SECANT_SR_APP_MSE_REDUCER_H_INCLUDED */
