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
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "implicit_sindy.h"
#include "stack_ptx_harness_common.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    IFRSTD_FEATURES = 32,
    IFRSTD_SETTINGS = 2,
    IFRSTD_RHS = 2,
    IFRSTD_SWEEPS = 3,
    IFRSTD_TRAIN_ROWS = 193,
    IFRSTD_VALIDATION_ROWS = 127,
    IFRSTD_STLSQ_MAX_ITERATIONS = 32
};

typedef struct {
    float train_gram[IFRSTD_SETTINGS * IFRSTD_FEATURES * IFRSTD_FEATURES];
    float train_x_sum[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    float train_xty[IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    float train_y_sum[IFRSTD_SETTINGS * IFRSTD_RHS];
    float train_yy[IFRSTD_SETTINGS * IFRSTD_RHS];
    float validation_gram[IFRSTD_SETTINGS * IFRSTD_FEATURES * IFRSTD_FEATURES];
    float validation_x_sum[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    float validation_xty[IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    float validation_y_sum[IFRSTD_SETTINGS * IFRSTD_RHS];
    float validation_yy[IFRSTD_SETTINGS * IFRSTD_RHS];
    float alphas[IFRSTD_SWEEPS];
    float thresholds[IFRSTD_SWEEPS];
    float posv_x_mean[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    float posv_x_scale[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    float posv_y_mean[IFRSTD_SETTINGS * IFRSTD_RHS];
    float posv_beta[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    int32_t posv_solve_info[IFRSTD_SWEEPS * IFRSTD_SETTINGS];
    float posv_mse[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    float stlsq_x_mean[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    float stlsq_x_scale[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    float stlsq_y_mean[IFRSTD_SETTINGS * IFRSTD_RHS];
    float stlsq_beta[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    uint32_t stlsq_active_masks[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    int32_t stlsq_active_counts[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    int32_t stlsq_iteration_counts[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    int32_t stlsq_solve_info[IFRSTD_SWEEPS * IFRSTD_SETTINGS];
    float stlsq_mse[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
} IfrstdHost;

typedef struct {
    CUdeviceptr train_gram;
    CUdeviceptr train_x_sum;
    CUdeviceptr train_xty;
    CUdeviceptr train_y_sum;
    CUdeviceptr validation_gram;
    CUdeviceptr validation_x_sum;
    CUdeviceptr validation_xty;
    CUdeviceptr validation_y_sum;
    CUdeviceptr validation_yy;
    CUdeviceptr alphas;
    CUdeviceptr thresholds;
    CUdeviceptr posv_x_mean;
    CUdeviceptr posv_x_scale;
    CUdeviceptr posv_y_mean;
    CUdeviceptr posv_beta;
    CUdeviceptr posv_solve_info;
    CUdeviceptr posv_mse;
    CUdeviceptr stlsq_x_mean;
    CUdeviceptr stlsq_x_scale;
    CUdeviceptr stlsq_y_mean;
    CUdeviceptr stlsq_beta;
    CUdeviceptr stlsq_active_masks;
    CUdeviceptr stlsq_active_counts;
    CUdeviceptr stlsq_iteration_counts;
    CUdeviceptr stlsq_solve_info;
    CUdeviceptr stlsq_mse;
} IfrstdDevice;

typedef struct {
    IfrstdHost host;
    IfrstdDevice device;
    ImplicitFeatureRidgeSolve* solve;
} IfrstdState;

typedef struct {
    double x_mean[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    double x_scale[IFRSTD_SETTINGS * IFRSTD_FEATURES];
    double y_mean[IFRSTD_SETTINGS * IFRSTD_RHS];
    double normalized_gram[IFRSTD_SETTINGS * IFRSTD_FEATURES * IFRSTD_FEATURES];
    double normalized_rhs[IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    double beta[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    int32_t solve_info[IFRSTD_SWEEPS * IFRSTD_SETTINGS];
    double mse[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    double stlsq_beta[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES];
    uint32_t stlsq_active_masks[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    int32_t stlsq_active_counts[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    int32_t stlsq_iteration_counts[IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS];
    int32_t stlsq_solve_info[IFRSTD_SWEEPS * IFRSTD_SETTINGS];
} IfrstdCpuReference;

static float*
ifrstd_device_f32(
    CUdeviceptr ptr
) {
    return (float*)(uintptr_t)ptr;
}

static int32_t*
ifrstd_device_i32(
    CUdeviceptr ptr
) {
    return (int32_t*)(uintptr_t)ptr;
}

static uint32_t*
ifrstd_device_u32(
    CUdeviceptr ptr
) {
    return (uint32_t*)(uintptr_t)ptr;
}

static size_t
ifrstd_x_index(
    int setting,
    int feature
) {
    return (size_t)setting * IFRSTD_FEATURES + (size_t)feature;
}

static size_t
ifrstd_rhs_index(
    int setting,
    int rhs
) {
    return (size_t)setting * IFRSTD_RHS + (size_t)rhs;
}

static size_t
ifrstd_gram_index(
    int setting,
    int row,
    int col
) {
    return
        (size_t)setting * IFRSTD_FEATURES * IFRSTD_FEATURES +
        (size_t)row +
        (size_t)col * IFRSTD_FEATURES;
}

static size_t
ifrstd_xty_index(
    int setting,
    int rhs,
    int feature
) {
    return
        ((size_t)setting * IFRSTD_RHS + (size_t)rhs) * IFRSTD_FEATURES +
        (size_t)feature;
}

static size_t
ifrstd_beta_index(
    int sweep,
    int setting,
    int rhs,
    int feature
) {
    return
        (((size_t)sweep * IFRSTD_SETTINGS + (size_t)setting) * IFRSTD_RHS + (size_t)rhs) *
        IFRSTD_FEATURES +
        (size_t)feature;
}

static size_t
ifrstd_sweep_setting_index(
    int sweep,
    int setting
) {
    return (size_t)sweep * IFRSTD_SETTINGS + (size_t)setting;
}

static size_t
ifrstd_sweep_rhs_index(
    int sweep,
    int setting,
    int rhs
) {
    return ((size_t)sweep * IFRSTD_SETTINGS + (size_t)setting) * IFRSTD_RHS + (size_t)rhs;
}

static int
ifrstd_cu(
    CUresult result,
    const char* label
) {
    if (result == CUDA_SUCCESS) return 1;
    fprintf(
        stderr,
        "%s failed: %s %s\n",
        label,
        stack_ptx_test_cu_name(result),
        stack_ptx_test_cu_string(result)
    );
    return 0;
}

static int
ifrstd_solve_check(
    ImplicitFeatureRidgeSolveResult result,
    const char* label
) {
    if (result == IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) return 1;
    fprintf(
        stderr,
        "%s failed: %s\n",
        label,
        implicit_feature_ridge_solve_result_to_string(result)
    );
    return 0;
}

static uint32_t
ifrstd_mix_u32(
    uint32_t x
) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float
ifrstd_unit_value(
    int setting,
    int row,
    int feature,
    int validation
) {
    uint32_t seed =
        0x9e3779b9u ^
        (uint32_t)(setting + 1) * 0x85ebca6bu ^
        (uint32_t)(row + 17) * 0xc2b2ae35u ^
        (uint32_t)(feature + 5) * 0x27d4eb2fu ^
        (uint32_t)(validation + 3) * 0x165667b1u;
    uint32_t mixed = ifrstd_mix_u32(seed);
    float uniform = (float)(mixed & 0x00ffffffu) * (1.0f / 16777216.0f);
    float trend = 0.0025f * (float)((row + feature * 3 + setting * 7) % 23);
    return uniform - 0.5f + trend;
}

static float
ifrstd_beta_raw(
    int rhs,
    int feature
) {
    static const float rhs0[] = {
        1.25f, -0.80f, 0.55f, 0.00f, 0.35f, 0.00f, -0.45f, 0.00f
    };
    static const float rhs1[] = {
        0.00f, 0.65f, -1.10f, 0.75f, 0.00f, -0.40f, 0.00f, 0.30f
    };
    if (feature >= 8) return 0.0f;
    return rhs == 0 ? rhs0[feature] : rhs1[feature];
}

static void
ifrstd_accumulate_stats(
    int validation,
    int rows,
    float* gram,
    float* x_sum,
    float* xty,
    float* y_sum,
    float* yy
) {
    int setting;
    int row;
    int p;
    int q;
    int rhs;

    memset(gram, 0, IFRSTD_SETTINGS * IFRSTD_FEATURES * IFRSTD_FEATURES * sizeof(float));
    memset(x_sum, 0, IFRSTD_SETTINGS * IFRSTD_FEATURES * sizeof(float));
    memset(xty, 0, IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES * sizeof(float));
    memset(y_sum, 0, IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float));
    memset(yy, 0, IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float));

    for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
        for (row = 0; row < rows; ++row) {
            float x[IFRSTD_FEATURES];
            float y[IFRSTD_RHS];

            for (p = 0; p < IFRSTD_FEATURES; ++p) {
                x[p] = ifrstd_unit_value(
                    setting,
                    row,
                    p,
                    validation
                );
                x_sum[setting * IFRSTD_FEATURES + p] += x[p];
            }
            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                float value = rhs == 0 ? 0.125f : -0.075f;
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    value += x[p] * ifrstd_beta_raw(rhs, p);
                }
                y[rhs] = value;
                y_sum[setting * IFRSTD_RHS + rhs] += value;
                yy[setting * IFRSTD_RHS + rhs] += value * value;
            }
            for (q = 0; q < IFRSTD_FEATURES; ++q) {
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    gram[
                        setting * IFRSTD_FEATURES * IFRSTD_FEATURES +
                        p +
                        q * IFRSTD_FEATURES
                    ] += x[p] * x[q];
                }
            }
            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    xty[
                        (setting * IFRSTD_RHS + rhs) * IFRSTD_FEATURES +
                        p
                    ] += x[p] * y[rhs];
                }
            }
        }
    }
}

static void
ifrstd_make_inputs(
    IfrstdHost* host
) {
    host->alphas[0] = 1.0e-4f;
    host->alphas[1] = 1.0e-2f;
    host->alphas[2] = 1.0e-1f;
    host->thresholds[0] = 4.0e-2f;
    host->thresholds[1] = 8.0e-2f;
    host->thresholds[2] = 1.6e-1f;

    ifrstd_accumulate_stats(
        0,
        IFRSTD_TRAIN_ROWS,
        host->train_gram,
        host->train_x_sum,
        host->train_xty,
        host->train_y_sum,
        host->train_yy
    );
    ifrstd_accumulate_stats(
        1,
        IFRSTD_VALIDATION_ROWS,
        host->validation_gram,
        host->validation_x_sum,
        host->validation_xty,
        host->validation_y_sum,
        host->validation_yy
    );
}

static int
ifrstd_alloc_device_array(
    CUdeviceptr* out,
    size_t bytes,
    const char* label
) {
    if (out == NULL || bytes == 0u) return 0;
    return ifrstd_cu(cuMemAlloc(out, bytes), label);
}

static int
ifrstd_alloc_device(
    IfrstdDevice* device
) {
    const size_t gram_bytes = IFRSTD_SETTINGS * IFRSTD_FEATURES * IFRSTD_FEATURES * sizeof(float);
    const size_t x_sum_bytes = IFRSTD_SETTINGS * IFRSTD_FEATURES * sizeof(float);
    const size_t xty_bytes = IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES * sizeof(float);
    const size_t rhs_bytes = IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float);
    const size_t sweep_bytes = IFRSTD_SWEEPS * sizeof(float);
    const size_t beta_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES * sizeof(float);
    const size_t solve_info_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * sizeof(int32_t);
    const size_t sweep_rhs_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float);
    const size_t sweep_rhs_u32_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(uint32_t);
    const size_t sweep_rhs_i32_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(int32_t);

    return
        ifrstd_alloc_device_array(&device->train_gram, gram_bytes, "cuMemAlloc train_gram") &&
        ifrstd_alloc_device_array(&device->train_x_sum, x_sum_bytes, "cuMemAlloc train_x_sum") &&
        ifrstd_alloc_device_array(&device->train_xty, xty_bytes, "cuMemAlloc train_xty") &&
        ifrstd_alloc_device_array(&device->train_y_sum, rhs_bytes, "cuMemAlloc train_y_sum") &&
        ifrstd_alloc_device_array(&device->validation_gram, gram_bytes, "cuMemAlloc validation_gram") &&
        ifrstd_alloc_device_array(&device->validation_x_sum, x_sum_bytes, "cuMemAlloc validation_x_sum") &&
        ifrstd_alloc_device_array(&device->validation_xty, xty_bytes, "cuMemAlloc validation_xty") &&
        ifrstd_alloc_device_array(&device->validation_y_sum, rhs_bytes, "cuMemAlloc validation_y_sum") &&
        ifrstd_alloc_device_array(&device->validation_yy, rhs_bytes, "cuMemAlloc validation_yy") &&
        ifrstd_alloc_device_array(&device->alphas, sweep_bytes, "cuMemAlloc alphas") &&
        ifrstd_alloc_device_array(&device->thresholds, sweep_bytes, "cuMemAlloc thresholds") &&
        ifrstd_alloc_device_array(&device->posv_x_mean, x_sum_bytes, "cuMemAlloc posv_x_mean") &&
        ifrstd_alloc_device_array(&device->posv_x_scale, x_sum_bytes, "cuMemAlloc posv_x_scale") &&
        ifrstd_alloc_device_array(&device->posv_y_mean, rhs_bytes, "cuMemAlloc posv_y_mean") &&
        ifrstd_alloc_device_array(&device->posv_beta, beta_bytes, "cuMemAlloc posv_beta") &&
        ifrstd_alloc_device_array(&device->posv_solve_info, solve_info_bytes, "cuMemAlloc posv_solve_info") &&
        ifrstd_alloc_device_array(&device->posv_mse, sweep_rhs_bytes, "cuMemAlloc posv_mse") &&
        ifrstd_alloc_device_array(&device->stlsq_x_mean, x_sum_bytes, "cuMemAlloc stlsq_x_mean") &&
        ifrstd_alloc_device_array(&device->stlsq_x_scale, x_sum_bytes, "cuMemAlloc stlsq_x_scale") &&
        ifrstd_alloc_device_array(&device->stlsq_y_mean, rhs_bytes, "cuMemAlloc stlsq_y_mean") &&
        ifrstd_alloc_device_array(&device->stlsq_beta, beta_bytes, "cuMemAlloc stlsq_beta") &&
        ifrstd_alloc_device_array(&device->stlsq_active_masks, sweep_rhs_u32_bytes, "cuMemAlloc stlsq_active_masks") &&
        ifrstd_alloc_device_array(&device->stlsq_active_counts, sweep_rhs_i32_bytes, "cuMemAlloc stlsq_active_counts") &&
        ifrstd_alloc_device_array(&device->stlsq_iteration_counts, sweep_rhs_i32_bytes, "cuMemAlloc stlsq_iteration_counts") &&
        ifrstd_alloc_device_array(&device->stlsq_solve_info, solve_info_bytes, "cuMemAlloc stlsq_solve_info") &&
        ifrstd_alloc_device_array(&device->stlsq_mse, sweep_rhs_bytes, "cuMemAlloc stlsq_mse");
}

static int
ifrstd_copy_htod(
    CUdeviceptr dst,
    const void* src,
    size_t bytes,
    const char* label
) {
    return ifrstd_cu(cuMemcpyHtoD(dst, src, bytes), label);
}

static int
ifrstd_copy_dtoh(
    void* dst,
    CUdeviceptr src,
    size_t bytes,
    const char* label
) {
    return ifrstd_cu(cuMemcpyDtoH(dst, src, bytes), label);
}

static int
ifrstd_copy_inputs_to_device(
    IfrstdState* state
) {
    const IfrstdHost* host = &state->host;
    const IfrstdDevice* device = &state->device;
    const size_t gram_bytes = IFRSTD_SETTINGS * IFRSTD_FEATURES * IFRSTD_FEATURES * sizeof(float);
    const size_t x_sum_bytes = IFRSTD_SETTINGS * IFRSTD_FEATURES * sizeof(float);
    const size_t xty_bytes = IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES * sizeof(float);
    const size_t rhs_bytes = IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float);
    const size_t sweep_bytes = IFRSTD_SWEEPS * sizeof(float);

    return
        ifrstd_copy_htod(device->train_gram, host->train_gram, gram_bytes, "cuMemcpyHtoD train_gram") &&
        ifrstd_copy_htod(device->train_x_sum, host->train_x_sum, x_sum_bytes, "cuMemcpyHtoD train_x_sum") &&
        ifrstd_copy_htod(device->train_xty, host->train_xty, xty_bytes, "cuMemcpyHtoD train_xty") &&
        ifrstd_copy_htod(device->train_y_sum, host->train_y_sum, rhs_bytes, "cuMemcpyHtoD train_y_sum") &&
        ifrstd_copy_htod(device->validation_gram, host->validation_gram, gram_bytes, "cuMemcpyHtoD validation_gram") &&
        ifrstd_copy_htod(device->validation_x_sum, host->validation_x_sum, x_sum_bytes, "cuMemcpyHtoD validation_x_sum") &&
        ifrstd_copy_htod(device->validation_xty, host->validation_xty, xty_bytes, "cuMemcpyHtoD validation_xty") &&
        ifrstd_copy_htod(device->validation_y_sum, host->validation_y_sum, rhs_bytes, "cuMemcpyHtoD validation_y_sum") &&
        ifrstd_copy_htod(device->validation_yy, host->validation_yy, rhs_bytes, "cuMemcpyHtoD validation_yy") &&
        ifrstd_copy_htod(device->alphas, host->alphas, sweep_bytes, "cuMemcpyHtoD alphas") &&
        ifrstd_copy_htod(device->thresholds, host->thresholds, sweep_bytes, "cuMemcpyHtoD thresholds");
}

static int
ifrstd_run_solver(
    IfrstdState* state
) {
    IfrstdDevice* device = &state->device;
    const float scale_epsilon = 1.0e-3f;

    if (!ifrstd_solve_check(
            implicit_feature_ridge_solve_create(&state->solve),
            "implicit_feature_ridge_solve_create")) return 0;

    if (!ifrstd_solve_check(
            implicit_feature_ridge_solve_posv_sweep(
                state->solve,
                NULL,
                IFRSTD_TRAIN_ROWS,
                IFRSTD_SETTINGS,
                IFRSTD_RHS,
                IFRSTD_SWEEPS,
                scale_epsilon,
                ifrstd_device_f32(device->train_gram),
                IFRSTD_FEATURES,
                IFRSTD_FEATURES * IFRSTD_FEATURES,
                ifrstd_device_f32(device->train_x_sum),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->train_xty),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_f32(device->train_y_sum),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->alphas),
                1,
                ifrstd_device_f32(device->posv_x_mean),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->posv_x_scale),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->posv_y_mean),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->posv_beta),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_i32(device->posv_solve_info),
                IFRSTD_SETTINGS,
                1
            ),
            "implicit_feature_ridge_solve_posv_sweep")) return 0;

    if (!ifrstd_solve_check(
            implicit_feature_ridge_solve_stlsq_sweep(
                state->solve,
                NULL,
                IFRSTD_TRAIN_ROWS,
                IFRSTD_SETTINGS,
                IFRSTD_RHS,
                IFRSTD_SWEEPS,
                scale_epsilon,
                ifrstd_device_f32(device->train_gram),
                IFRSTD_FEATURES,
                IFRSTD_FEATURES * IFRSTD_FEATURES,
                ifrstd_device_f32(device->train_x_sum),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->train_xty),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_f32(device->train_y_sum),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->alphas),
                1,
                ifrstd_device_f32(device->thresholds),
                1,
                ifrstd_device_f32(device->stlsq_x_mean),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->stlsq_x_scale),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->stlsq_y_mean),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->stlsq_beta),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_u32(device->stlsq_active_masks),
                1,
                IFRSTD_RHS,
                IFRSTD_SETTINGS * IFRSTD_RHS,
                ifrstd_device_i32(device->stlsq_active_counts),
                1,
                IFRSTD_RHS,
                IFRSTD_SETTINGS * IFRSTD_RHS,
                ifrstd_device_i32(device->stlsq_iteration_counts),
                1,
                IFRSTD_RHS,
                IFRSTD_SETTINGS * IFRSTD_RHS,
                ifrstd_device_i32(device->stlsq_solve_info),
                IFRSTD_SETTINGS,
                1
            ),
            "implicit_feature_ridge_solve_stlsq_sweep")) return 0;

    if (!ifrstd_solve_check(
            implicit_feature_ridge_score_validation_mse(
                state->solve,
                NULL,
                IFRSTD_VALIDATION_ROWS,
                IFRSTD_SETTINGS,
                IFRSTD_RHS,
                IFRSTD_SWEEPS,
                ifrstd_device_f32(device->validation_gram),
                IFRSTD_FEATURES,
                IFRSTD_FEATURES * IFRSTD_FEATURES,
                ifrstd_device_f32(device->validation_x_sum),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->validation_xty),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_f32(device->validation_y_sum),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->validation_yy),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->posv_beta),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_f32(device->posv_x_mean),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->posv_x_scale),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->posv_y_mean),
                1,
                IFRSTD_RHS,
                ifrstd_device_i32(device->posv_solve_info),
                IFRSTD_SETTINGS,
                1,
                ifrstd_device_f32(device->posv_mse),
                1,
                IFRSTD_RHS,
                IFRSTD_SETTINGS * IFRSTD_RHS
            ),
            "implicit_feature_ridge_score_validation_mse posv")) return 0;

    if (!ifrstd_solve_check(
            implicit_feature_ridge_score_validation_mse(
                state->solve,
                NULL,
                IFRSTD_VALIDATION_ROWS,
                IFRSTD_SETTINGS,
                IFRSTD_RHS,
                IFRSTD_SWEEPS,
                ifrstd_device_f32(device->validation_gram),
                IFRSTD_FEATURES,
                IFRSTD_FEATURES * IFRSTD_FEATURES,
                ifrstd_device_f32(device->validation_x_sum),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->validation_xty),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_f32(device->validation_y_sum),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->validation_yy),
                1,
                IFRSTD_RHS,
                ifrstd_device_f32(device->stlsq_beta),
                IFRSTD_FEATURES,
                IFRSTD_RHS * IFRSTD_FEATURES,
                IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES,
                ifrstd_device_f32(device->stlsq_x_mean),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->stlsq_x_scale),
                IFRSTD_FEATURES,
                ifrstd_device_f32(device->stlsq_y_mean),
                1,
                IFRSTD_RHS,
                ifrstd_device_i32(device->stlsq_solve_info),
                IFRSTD_SETTINGS,
                1,
                ifrstd_device_f32(device->stlsq_mse),
                1,
                IFRSTD_RHS,
                IFRSTD_SETTINGS * IFRSTD_RHS
            ),
            "implicit_feature_ridge_score_validation_mse stlsq")) return 0;

    return ifrstd_cu(cuCtxSynchronize(), "cuCtxSynchronize");
}

static int
ifrstd_copy_outputs_to_host(
    IfrstdState* state
) {
    IfrstdHost* host = &state->host;
    const IfrstdDevice* device = &state->device;
    const size_t x_sum_bytes = IFRSTD_SETTINGS * IFRSTD_FEATURES * sizeof(float);
    const size_t rhs_bytes = IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float);
    const size_t beta_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * IFRSTD_FEATURES * sizeof(float);
    const size_t solve_info_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * sizeof(int32_t);
    const size_t sweep_rhs_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(float);
    const size_t sweep_rhs_u32_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(uint32_t);
    const size_t sweep_rhs_i32_bytes = IFRSTD_SWEEPS * IFRSTD_SETTINGS * IFRSTD_RHS * sizeof(int32_t);

    return
        ifrstd_copy_dtoh(host->posv_x_mean, device->posv_x_mean, x_sum_bytes, "cuMemcpyDtoH posv_x_mean") &&
        ifrstd_copy_dtoh(host->posv_x_scale, device->posv_x_scale, x_sum_bytes, "cuMemcpyDtoH posv_x_scale") &&
        ifrstd_copy_dtoh(host->posv_y_mean, device->posv_y_mean, rhs_bytes, "cuMemcpyDtoH posv_y_mean") &&
        ifrstd_copy_dtoh(host->posv_beta, device->posv_beta, beta_bytes, "cuMemcpyDtoH posv_beta") &&
        ifrstd_copy_dtoh(host->posv_solve_info, device->posv_solve_info, solve_info_bytes, "cuMemcpyDtoH posv_solve_info") &&
        ifrstd_copy_dtoh(host->posv_mse, device->posv_mse, sweep_rhs_bytes, "cuMemcpyDtoH posv_mse") &&
        ifrstd_copy_dtoh(host->stlsq_x_mean, device->stlsq_x_mean, x_sum_bytes, "cuMemcpyDtoH stlsq_x_mean") &&
        ifrstd_copy_dtoh(host->stlsq_x_scale, device->stlsq_x_scale, x_sum_bytes, "cuMemcpyDtoH stlsq_x_scale") &&
        ifrstd_copy_dtoh(host->stlsq_y_mean, device->stlsq_y_mean, rhs_bytes, "cuMemcpyDtoH stlsq_y_mean") &&
        ifrstd_copy_dtoh(host->stlsq_beta, device->stlsq_beta, beta_bytes, "cuMemcpyDtoH stlsq_beta") &&
        ifrstd_copy_dtoh(host->stlsq_active_masks, device->stlsq_active_masks, sweep_rhs_u32_bytes, "cuMemcpyDtoH stlsq_active_masks") &&
        ifrstd_copy_dtoh(host->stlsq_active_counts, device->stlsq_active_counts, sweep_rhs_i32_bytes, "cuMemcpyDtoH stlsq_active_counts") &&
        ifrstd_copy_dtoh(host->stlsq_iteration_counts, device->stlsq_iteration_counts, sweep_rhs_i32_bytes, "cuMemcpyDtoH stlsq_iteration_counts") &&
        ifrstd_copy_dtoh(host->stlsq_solve_info, device->stlsq_solve_info, solve_info_bytes, "cuMemcpyDtoH stlsq_solve_info") &&
        ifrstd_copy_dtoh(host->stlsq_mse, device->stlsq_mse, sweep_rhs_bytes, "cuMemcpyDtoH stlsq_mse");
}

static int
ifrstd_cholesky_posv(
    const double* a_in,
    const double* b,
    double* x,
    int n
) {
    double a[IFRSTD_FEATURES * IFRSTD_FEATURES];
    double y[IFRSTD_FEATURES];
    int i;
    int j;
    int k;

    memcpy(a, a_in, (size_t)n * (size_t)n * sizeof(double));
    for (i = 0; i < n; ++i) {
        for (j = 0; j <= i; ++j) {
            double sum = a[(size_t)i * n + (size_t)j];
            for (k = 0; k < j; ++k) {
                sum -= a[(size_t)i * n + (size_t)k] * a[(size_t)j * n + (size_t)k];
            }
            if (i == j) {
                if (!(sum > 1.0e-14)) return 0;
                a[(size_t)i * n + (size_t)j] = sqrt(sum);
            } else {
                a[(size_t)i * n + (size_t)j] = sum / a[(size_t)j * n + (size_t)j];
            }
        }
        for (j = i + 1; j < n; ++j) {
            a[(size_t)i * n + (size_t)j] = 0.0;
        }
    }

    for (i = 0; i < n; ++i) {
        double sum = b[i];
        for (k = 0; k < i; ++k) {
            sum -= a[(size_t)i * n + (size_t)k] * y[k];
        }
        y[i] = sum / a[(size_t)i * n + (size_t)i];
    }
    for (i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (k = i + 1; k < n; ++k) {
            sum -= a[(size_t)k * n + (size_t)i] * x[k];
        }
        x[i] = sum / a[(size_t)i * n + (size_t)i];
    }
    return 1;
}

static int32_t
ifrstd_popcount_u32(
    uint32_t value
) {
    int32_t count = 0;
    while (value != 0u) {
        count += (int32_t)(value & 1u);
        value >>= 1;
    }
    return count;
}

static void
ifrstd_active_system(
    const IfrstdCpuReference* ref,
    int setting,
    int rhs,
    double alpha,
    uint32_t active_mask,
    double* a,
    double* b
) {
    int p;
    int q;

    for (p = 0; p < IFRSTD_FEATURES; ++p) {
        const int p_active = ((active_mask >> p) & 1u) != 0u;
        b[p] = p_active ? ref->normalized_rhs[ifrstd_xty_index(setting, rhs, p)] : 0.0;
        for (q = 0; q < IFRSTD_FEATURES; ++q) {
            const int q_active = ((active_mask >> q) & 1u) != 0u;
            double value;
            if (p == q) {
                value = p_active ? 1.0 + alpha : 1.0;
            } else if (p_active && q_active) {
                value = ref->normalized_gram[ifrstd_gram_index(setting, p, q)];
            } else {
                value = 0.0;
            }
            a[(size_t)p * IFRSTD_FEATURES + (size_t)q] = value;
        }
    }
}

static uint32_t
ifrstd_threshold_mask(
    const double* solution,
    uint32_t active_mask,
    double threshold
) {
    uint32_t next_mask = 0u;
    int p;

    for (p = 0; p < IFRSTD_FEATURES; ++p) {
        if (((active_mask >> p) & 1u) != 0u && fabs(solution[p]) >= threshold) {
            next_mask |= 1u << p;
        }
    }
    return next_mask;
}

static void
ifrstd_build_cpu_posv_reference(
    const IfrstdHost* host,
    IfrstdCpuReference* ref
) {
    const double inv_train_rows = 1.0 / (double)IFRSTD_TRAIN_ROWS;
    const double inv_validation_rows = 1.0 / (double)IFRSTD_VALIDATION_ROWS;
    const double scale_floor = 1.0e-6;
    int setting;
    int sweep;
    int rhs;
    int p;
    int q;

    memset(ref, 0, sizeof(*ref));

    for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
        for (p = 0; p < IFRSTD_FEATURES; ++p) {
            const double sum_p = (double)host->train_x_sum[ifrstd_x_index(setting, p)];
            double variance =
                ((double)host->train_gram[ifrstd_gram_index(setting, p, p)] -
                 sum_p * sum_p * inv_train_rows) *
                inv_train_rows;
            if (variance < scale_floor) variance = scale_floor;
            ref->x_mean[ifrstd_x_index(setting, p)] = sum_p * inv_train_rows;
            ref->x_scale[ifrstd_x_index(setting, p)] = sqrt(variance);
        }

        for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
            ref->y_mean[ifrstd_rhs_index(setting, rhs)] =
                (double)host->train_y_sum[ifrstd_rhs_index(setting, rhs)] * inv_train_rows;
        }

        for (p = 0; p < IFRSTD_FEATURES; ++p) {
            for (q = 0; q < IFRSTD_FEATURES; ++q) {
                double value;
                if (p == q) {
                    value = 1.0;
                } else {
                    const double centered =
                        (double)host->train_gram[ifrstd_gram_index(setting, p, q)] -
                        (double)host->train_x_sum[ifrstd_x_index(setting, p)] *
                            (double)host->train_x_sum[ifrstd_x_index(setting, q)] *
                            inv_train_rows;
                    value =
                        centered /
                        (ref->x_scale[ifrstd_x_index(setting, p)] *
                         ref->x_scale[ifrstd_x_index(setting, q)]) *
                        inv_train_rows;
                }
                ref->normalized_gram[ifrstd_gram_index(setting, p, q)] = value;
            }
        }

        for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
            const double mean = ref->y_mean[ifrstd_rhs_index(setting, rhs)];
            for (p = 0; p < IFRSTD_FEATURES; ++p) {
                const double centered_rhs =
                    (double)host->train_xty[ifrstd_xty_index(setting, rhs, p)] -
                    (double)host->train_x_sum[ifrstd_x_index(setting, p)] * mean;
                ref->normalized_rhs[ifrstd_xty_index(setting, rhs, p)] =
                    centered_rhs / ref->x_scale[ifrstd_x_index(setting, p)] * inv_train_rows;
            }
        }
    }

    for (sweep = 0; sweep < IFRSTD_SWEEPS; ++sweep) {
        for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
            double a[IFRSTD_FEATURES * IFRSTD_FEATURES];
            int setting_ok = 1;

            for (p = 0; p < IFRSTD_FEATURES; ++p) {
                for (q = 0; q < IFRSTD_FEATURES; ++q) {
                    double value = ref->normalized_gram[ifrstd_gram_index(setting, p, q)];
                    if (p == q) value += (double)host->alphas[sweep];
                    a[(size_t)p * IFRSTD_FEATURES + (size_t)q] = value;
                }
            }

            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                double b[IFRSTD_FEATURES];
                double x[IFRSTD_FEATURES];
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    b[p] = ref->normalized_rhs[ifrstd_xty_index(setting, rhs, p)];
                    x[p] = 0.0;
                }
                if (!ifrstd_cholesky_posv(a, b, x, IFRSTD_FEATURES)) {
                    setting_ok = 0;
                    break;
                }
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    ref->beta[ifrstd_beta_index(sweep, setting, rhs, p)] = x[p];
                }
            }
            ref->solve_info[ifrstd_sweep_setting_index(sweep, setting)] = setting_ok ? 0 : 1;
        }
    }

    for (sweep = 0; sweep < IFRSTD_SWEEPS; ++sweep) {
        const double alpha = (double)host->alphas[sweep];
        const double threshold = (double)host->thresholds[sweep];
        for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
            int setting_info = 0;
            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                uint32_t active_mask = 0xffffffffu;
                int iteration_count = 0;
                double solution[IFRSTD_FEATURES];

                memset(solution, 0, sizeof(solution));
                for (int iteration = 0; iteration < IFRSTD_STLSQ_MAX_ITERATIONS; ++iteration) {
                    double a[IFRSTD_FEATURES * IFRSTD_FEATURES];
                    double b[IFRSTD_FEATURES];
                    uint32_t next_mask;

                    ifrstd_active_system(ref, setting, rhs, alpha, active_mask, a, b);
                    if (!ifrstd_cholesky_posv(a, b, solution, IFRSTD_FEATURES)) {
                        setting_info = 1;
                        active_mask = 0u;
                        iteration_count = iteration + 1;
                        break;
                    }

                    next_mask = ifrstd_threshold_mask(solution, active_mask, threshold);
                    iteration_count = iteration + 1;
                    if (next_mask == active_mask || next_mask == 0u) {
                        active_mask = next_mask;
                        break;
                    }
                    active_mask = next_mask;
                }

                if (setting_info == 0 && active_mask != 0u) {
                    double a[IFRSTD_FEATURES * IFRSTD_FEATURES];
                    double b[IFRSTD_FEATURES];
                    ifrstd_active_system(ref, setting, rhs, alpha, active_mask, a, b);
                    if (!ifrstd_cholesky_posv(a, b, solution, IFRSTD_FEATURES)) {
                        setting_info = 1;
                    }
                } else {
                    memset(solution, 0, sizeof(solution));
                }

                if (setting_info == 0) {
                    for (p = 0; p < IFRSTD_FEATURES; ++p) {
                        ref->stlsq_beta[ifrstd_beta_index(sweep, setting, rhs, p)] = solution[p];
                    }
                } else {
                    for (p = 0; p < IFRSTD_FEATURES; ++p) {
                        ref->stlsq_beta[ifrstd_beta_index(sweep, setting, rhs, p)] = 0.0;
                    }
                }

                ref->stlsq_active_masks[ifrstd_sweep_rhs_index(sweep, setting, rhs)] = active_mask;
                ref->stlsq_active_counts[ifrstd_sweep_rhs_index(sweep, setting, rhs)] =
                    ifrstd_popcount_u32(active_mask);
                ref->stlsq_iteration_counts[ifrstd_sweep_rhs_index(sweep, setting, rhs)] = iteration_count;
            }
            ref->stlsq_solve_info[ifrstd_sweep_setting_index(sweep, setting)] = setting_info;
        }
    }

    for (sweep = 0; sweep < IFRSTD_SWEEPS; ++sweep) {
        for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                double beta_raw[IFRSTD_FEATURES];
                double intercept = ref->y_mean[ifrstd_rhs_index(setting, rhs)];
                double xty_proj = 0.0;
                double xsum_proj = 0.0;
                double quad = 0.0;
                double sse;

                if (ref->solve_info[ifrstd_sweep_setting_index(sweep, setting)] != 0) {
                    ref->mse[ifrstd_sweep_rhs_index(sweep, setting, rhs)] = INFINITY;
                    continue;
                }

                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    beta_raw[p] =
                        ref->beta[ifrstd_beta_index(sweep, setting, rhs, p)] /
                        ref->x_scale[ifrstd_x_index(setting, p)];
                    intercept -= ref->x_mean[ifrstd_x_index(setting, p)] * beta_raw[p];
                    xty_proj += beta_raw[p] * (double)host->validation_xty[ifrstd_xty_index(setting, rhs, p)];
                    xsum_proj += beta_raw[p] * (double)host->validation_x_sum[ifrstd_x_index(setting, p)];
                }
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    for (q = 0; q < IFRSTD_FEATURES; ++q) {
                        quad +=
                            beta_raw[p] *
                            (double)host->validation_gram[ifrstd_gram_index(setting, p, q)] *
                            beta_raw[q];
                    }
                }

                sse =
                    (double)host->validation_yy[ifrstd_rhs_index(setting, rhs)] -
                    2.0 * xty_proj +
                    quad -
                    2.0 * intercept * (double)host->validation_y_sum[ifrstd_rhs_index(setting, rhs)] +
                    2.0 * intercept * xsum_proj +
                    (double)IFRSTD_VALIDATION_ROWS * intercept * intercept;
                ref->mse[ifrstd_sweep_rhs_index(sweep, setting, rhs)] =
                    sse > 0.0 ? sse * inv_validation_rows : 0.0;
            }
        }
    }
}

static int
ifrstd_check_close(
    const char* name,
    double actual,
    double expected,
    double atol,
    double rtol
) {
    const double diff = fabs(actual - expected);
    const double limit = atol + rtol * fabs(expected);
    if (diff <= limit) return 1;
    fprintf(
        stderr,
        "%s mismatch: actual=%.9g expected=%.9g diff=%.9g limit=%.9g\n",
        name,
        actual,
        expected,
        diff,
        limit
    );
    return 0;
}

static double
ifrstd_cpu_validation_mse(
    const IfrstdHost* host,
    const float* beta_standardized,
    const float* x_mean,
    const float* x_scale,
    const float* y_mean,
    const int32_t* solve_info,
    int sweep,
    int setting,
    int rhs
) {
    const double inv_validation_rows = 1.0 / (double)IFRSTD_VALIDATION_ROWS;
    double beta_raw[IFRSTD_FEATURES];
    double intercept;
    double xty_proj = 0.0;
    double xsum_proj = 0.0;
    double quad = 0.0;
    double sse;
    int p;
    int q;

    if (solve_info[ifrstd_sweep_setting_index(sweep, setting)] != 0) return INFINITY;

    intercept = (double)y_mean[ifrstd_rhs_index(setting, rhs)];
    for (p = 0; p < IFRSTD_FEATURES; ++p) {
        beta_raw[p] =
            (double)beta_standardized[ifrstd_beta_index(sweep, setting, rhs, p)] /
            (double)x_scale[ifrstd_x_index(setting, p)];
        intercept -= (double)x_mean[ifrstd_x_index(setting, p)] * beta_raw[p];
        xty_proj += beta_raw[p] * (double)host->validation_xty[ifrstd_xty_index(setting, rhs, p)];
        xsum_proj += beta_raw[p] * (double)host->validation_x_sum[ifrstd_x_index(setting, p)];
    }
    for (p = 0; p < IFRSTD_FEATURES; ++p) {
        for (q = 0; q < IFRSTD_FEATURES; ++q) {
            quad +=
                beta_raw[p] *
                (double)host->validation_gram[ifrstd_gram_index(setting, p, q)] *
                beta_raw[q];
        }
    }

    sse =
        (double)host->validation_yy[ifrstd_rhs_index(setting, rhs)] -
        2.0 * xty_proj +
        quad -
        2.0 * intercept * (double)host->validation_y_sum[ifrstd_rhs_index(setting, rhs)] +
        2.0 * intercept * xsum_proj +
        (double)IFRSTD_VALIDATION_ROWS * intercept * intercept;
    return sse > 0.0 ? sse * inv_validation_rows : 0.0;
}

static int
ifrstd_check_mse_kernel_reference(
    const IfrstdHost* host,
    const char* label,
    const float* beta_standardized,
    const float* x_mean,
    const float* x_scale,
    const float* y_mean,
    const int32_t* solve_info,
    const float* mse
) {
    int ok = 1;
    int sweep;
    int setting;
    int rhs;

    for (sweep = 0; sweep < IFRSTD_SWEEPS; ++sweep) {
        for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                const double expected = ifrstd_cpu_validation_mse(
                    host,
                    beta_standardized,
                    x_mean,
                    x_scale,
                    y_mean,
                    solve_info,
                    sweep,
                    setting,
                    rhs
                );
                const double actual = (double)mse[ifrstd_sweep_rhs_index(sweep, setting, rhs)];
                if (isinf(expected) && isinf(actual)) continue;
                ok = ifrstd_check_close(label, actual, expected, 4.0e-5, 2.0e-4) && ok;
            }
        }
    }

    return ok;
}

static int
ifrstd_check_cpu_posv_reference(
    const IfrstdHost* host
) {
    IfrstdCpuReference ref;
    int ok = 1;
    int setting;
    int sweep;
    int rhs;
    int p;

    ifrstd_build_cpu_posv_reference(host, &ref);

    ok = ifrstd_check_mse_kernel_reference(
        host,
        "posv_mse",
        host->posv_beta,
        host->posv_x_mean,
        host->posv_x_scale,
        host->posv_y_mean,
        host->posv_solve_info,
        host->posv_mse
    ) && ok;
    ok = ifrstd_check_mse_kernel_reference(
        host,
        "stlsq_mse",
        host->stlsq_beta,
        host->stlsq_x_mean,
        host->stlsq_x_scale,
        host->stlsq_y_mean,
        host->stlsq_solve_info,
        host->stlsq_mse
    ) && ok;

    for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
        for (p = 0; p < IFRSTD_FEATURES; ++p) {
            ok = ifrstd_check_close(
                "posv_x_mean",
                (double)host->posv_x_mean[ifrstd_x_index(setting, p)],
                ref.x_mean[ifrstd_x_index(setting, p)],
                2.0e-6,
                2.0e-6
            ) && ok;
            ok = ifrstd_check_close(
                "posv_x_scale",
                (double)host->posv_x_scale[ifrstd_x_index(setting, p)],
                ref.x_scale[ifrstd_x_index(setting, p)],
                2.0e-6,
                2.0e-6
            ) && ok;
            ok = ifrstd_check_close(
                "stlsq_x_mean",
                (double)host->stlsq_x_mean[ifrstd_x_index(setting, p)],
                ref.x_mean[ifrstd_x_index(setting, p)],
                2.0e-6,
                2.0e-6
            ) && ok;
            ok = ifrstd_check_close(
                "stlsq_x_scale",
                (double)host->stlsq_x_scale[ifrstd_x_index(setting, p)],
                ref.x_scale[ifrstd_x_index(setting, p)],
                2.0e-6,
                2.0e-6
            ) && ok;
        }
        for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
            ok = ifrstd_check_close(
                "posv_y_mean",
                (double)host->posv_y_mean[ifrstd_rhs_index(setting, rhs)],
                ref.y_mean[ifrstd_rhs_index(setting, rhs)],
                2.0e-6,
                2.0e-6
            ) && ok;
            ok = ifrstd_check_close(
                "stlsq_y_mean",
                (double)host->stlsq_y_mean[ifrstd_rhs_index(setting, rhs)],
                ref.y_mean[ifrstd_rhs_index(setting, rhs)],
                2.0e-6,
                2.0e-6
            ) && ok;
        }
    }

    for (sweep = 0; sweep < IFRSTD_SWEEPS; ++sweep) {
        for (setting = 0; setting < IFRSTD_SETTINGS; ++setting) {
            const size_t info_index = ifrstd_sweep_setting_index(sweep, setting);
            if (host->posv_solve_info[info_index] != ref.solve_info[info_index]) {
                fprintf(
                    stderr,
                    "posv_solve_info mismatch: sweep=%d setting=%d actual=%d expected=%d\n",
                    sweep,
                    setting,
                    (int)host->posv_solve_info[info_index],
                    (int)ref.solve_info[info_index]
                );
                ok = 0;
            }
            if (host->stlsq_solve_info[info_index] != ref.stlsq_solve_info[info_index]) {
                fprintf(
                    stderr,
                    "stlsq_solve_info mismatch: sweep=%d setting=%d actual=%d expected=%d\n",
                    sweep,
                    setting,
                    (int)host->stlsq_solve_info[info_index],
                    (int)ref.stlsq_solve_info[info_index]
                );
                ok = 0;
            }
            for (rhs = 0; rhs < IFRSTD_RHS; ++rhs) {
                const size_t sweep_rhs_index = ifrstd_sweep_rhs_index(sweep, setting, rhs);
                ok = ifrstd_check_close(
                    "posv_mse",
                    (double)host->posv_mse[sweep_rhs_index],
                    ref.mse[sweep_rhs_index],
                    4.0e-5,
                    2.0e-4
                ) && ok;
                if (host->stlsq_active_masks[sweep_rhs_index] != ref.stlsq_active_masks[sweep_rhs_index]) {
                    fprintf(
                        stderr,
                        "stlsq_active_masks mismatch: sweep=%d setting=%d rhs=%d actual=0x%08x expected=0x%08x\n",
                        sweep,
                        setting,
                        rhs,
                        host->stlsq_active_masks[sweep_rhs_index],
                        ref.stlsq_active_masks[sweep_rhs_index]
                    );
                    ok = 0;
                }
                if (host->stlsq_active_counts[sweep_rhs_index] != ref.stlsq_active_counts[sweep_rhs_index]) {
                    fprintf(
                        stderr,
                        "stlsq_active_counts mismatch: sweep=%d setting=%d rhs=%d actual=%d expected=%d\n",
                        sweep,
                        setting,
                        rhs,
                        (int)host->stlsq_active_counts[sweep_rhs_index],
                        (int)ref.stlsq_active_counts[sweep_rhs_index]
                    );
                    ok = 0;
                }
                if (host->stlsq_iteration_counts[sweep_rhs_index] != ref.stlsq_iteration_counts[sweep_rhs_index]) {
                    fprintf(
                        stderr,
                        "stlsq_iteration_counts mismatch: sweep=%d setting=%d rhs=%d actual=%d expected=%d\n",
                        sweep,
                        setting,
                        rhs,
                        (int)host->stlsq_iteration_counts[sweep_rhs_index],
                        (int)ref.stlsq_iteration_counts[sweep_rhs_index]
                    );
                    ok = 0;
                }
                for (p = 0; p < IFRSTD_FEATURES; ++p) {
                    ok = ifrstd_check_close(
                        "posv_beta",
                        (double)host->posv_beta[ifrstd_beta_index(sweep, setting, rhs, p)],
                        ref.beta[ifrstd_beta_index(sweep, setting, rhs, p)],
                        2.5e-4,
                        2.5e-4
                    ) && ok;
                    ok = ifrstd_check_close(
                        "stlsq_beta",
                        (double)host->stlsq_beta[ifrstd_beta_index(sweep, setting, rhs, p)],
                        ref.stlsq_beta[ifrstd_beta_index(sweep, setting, rhs, p)],
                        2.5e-4,
                        2.5e-4
                    ) && ok;
                }
            }
        }
    }

    if (ok) printf("PASS implicit_feature_ridge_solve_cpu_reference_test\n");
    return ok;
}

static void
ifrstd_destroy(
    IfrstdState* state
) {
    IfrstdDevice* device;
    if (state == NULL) return;
    device = &state->device;
    if (state->solve != NULL) (void)implicit_feature_ridge_solve_destroy(state->solve);
    if (device->train_gram != 0) (void)cuMemFree(device->train_gram);
    if (device->train_x_sum != 0) (void)cuMemFree(device->train_x_sum);
    if (device->train_xty != 0) (void)cuMemFree(device->train_xty);
    if (device->train_y_sum != 0) (void)cuMemFree(device->train_y_sum);
    if (device->validation_gram != 0) (void)cuMemFree(device->validation_gram);
    if (device->validation_x_sum != 0) (void)cuMemFree(device->validation_x_sum);
    if (device->validation_xty != 0) (void)cuMemFree(device->validation_xty);
    if (device->validation_y_sum != 0) (void)cuMemFree(device->validation_y_sum);
    if (device->validation_yy != 0) (void)cuMemFree(device->validation_yy);
    if (device->alphas != 0) (void)cuMemFree(device->alphas);
    if (device->thresholds != 0) (void)cuMemFree(device->thresholds);
    if (device->posv_x_mean != 0) (void)cuMemFree(device->posv_x_mean);
    if (device->posv_x_scale != 0) (void)cuMemFree(device->posv_x_scale);
    if (device->posv_y_mean != 0) (void)cuMemFree(device->posv_y_mean);
    if (device->posv_beta != 0) (void)cuMemFree(device->posv_beta);
    if (device->posv_solve_info != 0) (void)cuMemFree(device->posv_solve_info);
    if (device->posv_mse != 0) (void)cuMemFree(device->posv_mse);
    if (device->stlsq_x_mean != 0) (void)cuMemFree(device->stlsq_x_mean);
    if (device->stlsq_x_scale != 0) (void)cuMemFree(device->stlsq_x_scale);
    if (device->stlsq_y_mean != 0) (void)cuMemFree(device->stlsq_y_mean);
    if (device->stlsq_beta != 0) (void)cuMemFree(device->stlsq_beta);
    if (device->stlsq_active_masks != 0) (void)cuMemFree(device->stlsq_active_masks);
    if (device->stlsq_active_counts != 0) (void)cuMemFree(device->stlsq_active_counts);
    if (device->stlsq_iteration_counts != 0) (void)cuMemFree(device->stlsq_iteration_counts);
    if (device->stlsq_solve_info != 0) (void)cuMemFree(device->stlsq_solve_info);
    if (device->stlsq_mse != 0) (void)cuMemFree(device->stlsq_mse);
}

int
main(
    int argc,
    char** argv
) {
    IfrstdState state;
    CUdevice device;
    int owns_primary = 0;
    int ok;

    if (argc != 1) {
        fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }

    memset(&state, 0, sizeof(state));
    if (!stack_ptx_test_init_context(&device, &owns_primary)) return 1;

    ifrstd_make_inputs(&state.host);
    ok =
        ifrstd_alloc_device(&state.device) &&
        ifrstd_copy_inputs_to_device(&state) &&
        ifrstd_run_solver(&state) &&
        ifrstd_copy_outputs_to_host(&state);

    if (ok) ok = ifrstd_check_cpu_posv_reference(&state.host);

    ifrstd_destroy(&state);
    stack_ptx_test_release_context(device, owns_primary);
    if (!ok) return 1;

    return 0;
}
