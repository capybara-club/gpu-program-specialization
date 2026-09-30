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
#include "secant_sindy.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    CAPACITY = 32,
    ROWS = 256,
    STATS = CAPACITY + CAPACITY * CAPACITY + CAPACITY
};

static int cuda_check(CUresult result, const char* operation) {
    const char* name = NULL;
    const char* description = NULL;

    if (result == CUDA_SUCCESS) {
        return 1;
    }
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    fprintf(stderr, "%s failed: %s: %s\n", operation, name != NULL ? name : "unknown",
            description != NULL ? description : "unknown");
    return 0;
}

static void* file_read(const char* path) {
    FILE* file = fopen(path, "rb");
    long length;
    void* data;

    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) fclose(file);
        return NULL;
    }
    data = malloc((size_t)length);
    if (data == NULL || fread(data, 1u, (size_t)length, file) != (size_t)length || fclose(file) != 0) {
        free(data);
        return NULL;
    }
    return data;
}

static int parity_u32(uint32_t value) {
    value ^= value >> 16u;
    value ^= value >> 8u;
    value ^= value >> 4u;
    value &= 0x0fu;
    return (0x6996u >> value) & 1u;
}

int main(int argc, char** argv) {
    float statistics[STATS] = {0};
    float target_stats[2] = {0};
    float alpha = 0.0f;
    float threshold = 0.5f;
    float coefficients_cpu[CAPACITY] = {0};
    float coefficients_gpu[CAPACITY] = {0};
    float intercept_cpu = 0.0f;
    float intercept_gpu = 0.0f;
    float sse_cpu = 0.0f;
    float sse_gpu = 0.0f;
    int32_t info_cpu = 0;
    int32_t info_gpu = 0;
    uint32_t mask_cpu = 0u;
    uint32_t mask_gpu = 0u;
    int32_t count_cpu = 0;
    int32_t count_gpu = 0;
    int32_t iterations_cpu = 0;
    int32_t iterations_gpu = 0;
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();
    void* fatbin = NULL;
    CUdevice device;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function;
    CUdeviceptr d_statistics = 0u;
    CUdeviceptr d_target_stats = 0u;
    CUdeviceptr d_alpha = 0u;
    CUdeviceptr d_threshold = 0u;
    CUdeviceptr d_coefficients = 0u;
    CUdeviceptr d_intercept = 0u;
    CUdeviceptr d_sse = 0u;
    CUdeviceptr d_info = 0u;
    CUdeviceptr d_mask = 0u;
    CUdeviceptr d_count = 0u;
    CUdeviceptr d_iterations = 0u;
    size_t feature;
    size_t row;
    int ok = 1;

    if (argc != 2 || (fatbin = file_read(argv[1])) == NULL) {
        fprintf(stderr, "usage: %s solver.fatbin\n", argv[0]);
        return 2;
    }
    for (feature = 0u; feature < CAPACITY; ++feature) {
        for (row = 0u; row < ROWS; ++row) {
            const float value = parity_u32((uint32_t)row & (uint32_t)(feature + 1u)) != 0 ? -1.0f : 1.0f;
            const float target = 3.0f +
                (parity_u32((uint32_t)row & 1u) != 0 ? -1.0f : 1.0f) +
                2.0f * (parity_u32((uint32_t)row & 32u) != 0 ? -1.0f : 1.0f);
            statistics[feature] += value;
            statistics[CAPACITY + CAPACITY * CAPACITY + feature] += value * target;
            if (feature == 0u) {
                target_stats[0] += target;
                target_stats[1] += target * target;
            }
        }
        {
            size_t rhs;
            for (rhs = 0u; rhs < CAPACITY; ++rhs) {
                for (row = 0u; row < ROWS; ++row) {
                    const float lhs_value = parity_u32((uint32_t)row & (uint32_t)(feature + 1u)) != 0 ? -1.0f : 1.0f;
                    const float rhs_value = parity_u32((uint32_t)row & (uint32_t)(rhs + 1u)) != 0 ? -1.0f : 1.0f;
                    statistics[CAPACITY + feature * CAPACITY + rhs] += lhs_value * rhs_value;
                }
            }
        }
    }

    run.feature_capacity = CAPACITY;
    run.num_asts = CAPACITY;
    run.num_rows = ROWS;
    run.num_targets = 1u;
    run.num_sweeps = 1u;
    run.max_iterations = CAPACITY;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, STATS};
    run.statistics_leading_dimension = STATS;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){&alpha, 1u};
    run.thresholds = (SecantSindyConstHostSpanF32){&threshold, 1u};
    run.coefficients = (SecantSindyHostSpanF32){coefficients_cpu, CAPACITY};
    run.intercepts = (SecantSindyHostSpanF32){&intercept_cpu, 1u};
    run.sse = (SecantSindyHostSpanF32){&sse_cpu, 1u};
    run.solve_info = (SecantSindyHostSpanS32){&info_cpu, 1u};
    run.active_masks = (SecantSindyHostSpanU32){&mask_cpu, 1u};
    run.active_counts = (SecantSindyHostSpanS32){&count_cpu, 1u};
    run.iteration_counts = (SecantSindyHostSpanS32){&iterations_cpu, 1u};
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_SUCCESS) {
        ok = 0;
        goto cleanup;
    }

    if (!cuda_check(cuInit(0), "cuInit") || !cuda_check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
        !cuda_check(cuModuleLoadData(&module, fatbin), "cuModuleLoadData") ||
        !cuda_check(cuModuleGetFunction(&function, module, "secant_sindy_cuda_stlsq_solve_f32"), "get STLSQ") ||
        !cuda_check(cuMemAlloc(&d_statistics, sizeof(statistics)), "allocate statistics") ||
        !cuda_check(cuMemAlloc(&d_target_stats, sizeof(target_stats)), "allocate target stats") ||
        !cuda_check(cuMemAlloc(&d_alpha, sizeof(alpha)), "allocate alpha") ||
        !cuda_check(cuMemAlloc(&d_threshold, sizeof(threshold)), "allocate threshold") ||
        !cuda_check(cuMemAlloc(&d_coefficients, sizeof(coefficients_gpu)), "allocate coefficients") ||
        !cuda_check(cuMemAlloc(&d_intercept, sizeof(intercept_gpu)), "allocate intercept") ||
        !cuda_check(cuMemAlloc(&d_sse, sizeof(sse_gpu)), "allocate SSE") ||
        !cuda_check(cuMemAlloc(&d_info, sizeof(info_gpu)), "allocate info") ||
        !cuda_check(cuMemAlloc(&d_mask, sizeof(mask_gpu)), "allocate mask") ||
        !cuda_check(cuMemAlloc(&d_count, sizeof(count_gpu)), "allocate count") ||
        !cuda_check(cuMemAlloc(&d_iterations, sizeof(iterations_gpu)), "allocate iterations") ||
        !cuda_check(cuMemcpyHtoD(d_statistics, statistics, sizeof(statistics)), "copy statistics") ||
        !cuda_check(cuMemcpyHtoD(d_target_stats, target_stats, sizeof(target_stats)), "copy target stats") ||
        !cuda_check(cuMemcpyHtoD(d_alpha, &alpha, sizeof(alpha)), "copy alpha") ||
        !cuda_check(cuMemcpyHtoD(d_threshold, &threshold, sizeof(threshold)), "copy threshold")) {
        ok = 0;
        goto cleanup;
    }
    {
        size_t statistics_ld = STATS;
        size_t target_stats_ld = 2u;
        size_t num_rows = ROWS;
        size_t num_asts = CAPACITY;
        size_t num_targets = 1u;
        size_t num_sweeps = 1u;
        size_t max_iterations = CAPACITY;
        float scale_epsilon = 1.0e-6f;
        void* arguments[] = {&d_statistics, &statistics_ld, &d_target_stats, &target_stats_ld, &num_rows, &num_asts,
                             &num_targets, &num_sweeps, &d_alpha, &d_threshold, &max_iterations, &scale_epsilon,
                             &d_coefficients, &d_intercept, &d_sse, &d_info, &d_mask, &d_count, &d_iterations};
        ok &= cuda_check(cuLaunchKernel(function, 1u, 1u, 1u, 32u, 1u, 1u, 0u, 0, arguments, NULL), "launch STLSQ");
    }
    ok &= cuda_check(cuCtxSynchronize(), "synchronize STLSQ");
    ok &= cuda_check(cuMemcpyDtoH(coefficients_gpu, d_coefficients, sizeof(coefficients_gpu)), "copy coefficients");
    ok &= cuda_check(cuMemcpyDtoH(&intercept_gpu, d_intercept, sizeof(intercept_gpu)), "copy intercept");
    ok &= cuda_check(cuMemcpyDtoH(&sse_gpu, d_sse, sizeof(sse_gpu)), "copy SSE");
    ok &= cuda_check(cuMemcpyDtoH(&info_gpu, d_info, sizeof(info_gpu)), "copy info");
    ok &= cuda_check(cuMemcpyDtoH(&mask_gpu, d_mask, sizeof(mask_gpu)), "copy mask");
    ok &= cuda_check(cuMemcpyDtoH(&count_gpu, d_count, sizeof(count_gpu)), "copy count");
    ok &= cuda_check(cuMemcpyDtoH(&iterations_gpu, d_iterations, sizeof(iterations_gpu)), "copy iterations");
    for (feature = 0u; feature < CAPACITY && ok; ++feature) {
        if (fabsf(coefficients_gpu[feature] - coefficients_cpu[feature]) > 3.0e-4f) {
            ok = 0;
        }
    }
    if (ok && (fabsf(intercept_gpu - intercept_cpu) > 3.0e-4f || fabsf(sse_gpu - sse_cpu) > 5.0e-2f ||
               info_gpu != info_cpu || mask_gpu != 0x80000001u || mask_gpu != mask_cpu || count_gpu != count_cpu ||
               iterations_gpu != iterations_cpu)) {
        fprintf(stderr, "capacity-32 GPU mismatch: mask=0x%08x count=%d iterations=%d info=%d\n",
                mask_gpu, count_gpu, iterations_gpu, info_gpu);
        ok = 0;
    }

cleanup:
#define FREE_DEVICE(pointer_) do { if ((pointer_) != 0u) (void)cuMemFree(pointer_); } while (0)
    FREE_DEVICE(d_iterations);
    FREE_DEVICE(d_count);
    FREE_DEVICE(d_mask);
    FREE_DEVICE(d_info);
    FREE_DEVICE(d_sse);
    FREE_DEVICE(d_intercept);
    FREE_DEVICE(d_coefficients);
    FREE_DEVICE(d_threshold);
    FREE_DEVICE(d_alpha);
    FREE_DEVICE(d_target_stats);
    FREE_DEVICE(d_statistics);
#undef FREE_DEVICE
    if (module != NULL) (void)cuModuleUnload(module);
    if (context != NULL) (void)cuDevicePrimaryCtxRelease(device);
    free(fatbin);
    if (!ok) return 1;
    printf("secant-sindy capacity-32 CUDA STLSQ matches CPU and preserves active-mask bit 31\n");
    return 0;
}
