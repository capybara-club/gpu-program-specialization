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
#include <cuda.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURE_CAPACITY = 32,
    MAX_TARGETS = 4
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

static int size_parse(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static void* file_read(const char* path, size_t* size_ret) {
    FILE* file = fopen(path, "rb");
    long length;
    void* data;

    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return NULL;
    }
    data = malloc((size_t)length);
    if (data == NULL || fread(data, 1u, (size_t)length, file) != (size_t)length || fclose(file) != 0) {
        free(data);
        return NULL;
    }
    *size_ret = (size_t)length;
    return data;
}

int main(int argc, char** argv) {
    const char* mode;
    size_t cohorts = 1024u;
    size_t num_targets = 4u;
    size_t num_sweeps = 4u;
    size_t timed_iterations = 20u;
    size_t warmup_iterations = 3u;
    size_t max_stlsq_iterations = FEATURE_CAPACITY;
    const size_t num_rows = 4096u;
    const size_t target_stats_leading_dimension = 2u;
    const float scale_epsilon = 1.0e-6f;
    size_t statistics_leading_dimension;
    size_t statistics_count;
    size_t output_count;
    size_t coefficient_count;
    size_t num_asts;
    float* statistics = NULL;
    float* target_stats = NULL;
    float* alphas = NULL;
    float* thresholds = NULL;
    void* fatbin = NULL;
    size_t fatbin_size = 0u;
    CUdevice device;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUevent begin = NULL;
    CUevent end = NULL;
    CUdeviceptr d_statistics = 0u;
    CUdeviceptr d_target_stats = 0u;
    CUdeviceptr d_alphas = 0u;
    CUdeviceptr d_thresholds = 0u;
    CUdeviceptr d_coefficients = 0u;
    CUdeviceptr d_intercepts = 0u;
    CUdeviceptr d_sse = 0u;
    CUdeviceptr d_info = 0u;
    CUdeviceptr d_masks = 0u;
    CUdeviceptr d_counts = 0u;
    CUdeviceptr d_iteration_counts = 0u;
    float elapsed_ms = 0.0f;
    size_t cohort;
    size_t sweep;
    size_t iteration;
    int stlsq;
    int ok = 1;

    if (argc < 3 || argc > 9) {
        fprintf(stderr, "usage: %s solver.fatbin <ridge|stlsq> [cohorts targets sweeps iterations warmups max-stlsq]\n",
                argv[0]);
        return 2;
    }
    mode = argv[2];
    stlsq = strcmp(mode, "stlsq") == 0;
    if (!stlsq && strcmp(mode, "ridge") != 0) {
        return 2;
    }
    if ((argc > 3 && !size_parse(argv[3], &cohorts)) || (argc > 4 && !size_parse(argv[4], &num_targets)) ||
        (argc > 5 && !size_parse(argv[5], &num_sweeps)) ||
        (argc > 6 && !size_parse(argv[6], &timed_iterations)) ||
        (argc > 7 && !size_parse(argv[7], &warmup_iterations)) ||
        (argc > 8 && !size_parse(argv[8], &max_stlsq_iterations)) || cohorts == 0u || num_targets == 0u ||
        num_targets > MAX_TARGETS || num_sweeps == 0u || timed_iterations == 0u || max_stlsq_iterations == 0u ||
        max_stlsq_iterations > FEATURE_CAPACITY) {
        return 2;
    }

    statistics_leading_dimension = FEATURE_CAPACITY + FEATURE_CAPACITY * FEATURE_CAPACITY +
        FEATURE_CAPACITY * num_targets;
    statistics_count = cohorts * statistics_leading_dimension;
    output_count = num_sweeps * cohorts * num_targets;
    coefficient_count = output_count * FEATURE_CAPACITY;
    num_asts = cohorts * FEATURE_CAPACITY;
    statistics = (float*)calloc(statistics_count, sizeof(*statistics));
    target_stats = (float*)calloc(num_targets * 2u, sizeof(*target_stats));
    alphas = (float*)malloc(num_sweeps * sizeof(*alphas));
    thresholds = (float*)malloc(num_sweeps * sizeof(*thresholds));
    fatbin = file_read(argv[1], &fatbin_size);
    (void)fatbin_size;
    if (statistics == NULL || target_stats == NULL || alphas == NULL || thresholds == NULL || fatbin == NULL) {
        ok = 0;
        goto cleanup;
    }
    for (cohort = 0u; cohort < cohorts; ++cohort) {
        float* stats = statistics + cohort * statistics_leading_dimension;
        size_t feature;
        size_t target;
        for (feature = 0u; feature < FEATURE_CAPACITY; ++feature) {
            stats[FEATURE_CAPACITY + feature * FEATURE_CAPACITY + feature] = (float)num_rows;
            for (target = 0u; target < num_targets; ++target) {
                const int coefficient_code = (int)((feature * 5u + target * 3u) % 11u) - 5;
                stats[FEATURE_CAPACITY + FEATURE_CAPACITY * FEATURE_CAPACITY + feature * num_targets + target] =
                    (float)num_rows * 0.075f * (float)coefficient_code;
            }
        }
    }
    for (sweep = 0u; sweep < num_sweeps; ++sweep) {
        alphas[sweep] = 0.01f + 0.02f * (float)sweep;
        thresholds[sweep] = 0.05f + 0.025f * (float)sweep;
    }
    for (cohort = 0u; cohort < num_targets; ++cohort) {
        target_stats[cohort * 2u + 1u] = (float)num_rows * 128.0f;
    }

    if (!cuda_check(cuInit(0), "cuInit") || !cuda_check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
        !cuda_check(cuModuleLoadData(&module, fatbin), "cuModuleLoadData") ||
        !cuda_check(cuModuleGetFunction(&function, module, stlsq ? "secant_sindy_cuda_stlsq_solve_f32" :
                                       "secant_sindy_cuda_ridge_solve_f32"), "cuModuleGetFunction") ||
        !cuda_check(cuMemAlloc(&d_statistics, statistics_count * sizeof(float)), "statistics allocation") ||
        !cuda_check(cuMemAlloc(&d_target_stats, num_targets * 2u * sizeof(float)), "target stats allocation") ||
        !cuda_check(cuMemAlloc(&d_alphas, num_sweeps * sizeof(float)), "alphas allocation") ||
        !cuda_check(cuMemAlloc(&d_thresholds, num_sweeps * sizeof(float)), "thresholds allocation") ||
        !cuda_check(cuMemAlloc(&d_coefficients, coefficient_count * sizeof(float)), "coefficients allocation") ||
        !cuda_check(cuMemAlloc(&d_intercepts, output_count * sizeof(float)), "intercepts allocation") ||
        !cuda_check(cuMemAlloc(&d_sse, output_count * sizeof(float)), "SSE allocation") ||
        !cuda_check(cuMemAlloc(&d_info, output_count * sizeof(int32_t)), "info allocation") ||
        !cuda_check(cuMemAlloc(&d_masks, output_count * sizeof(uint32_t)), "masks allocation") ||
        !cuda_check(cuMemAlloc(&d_counts, output_count * sizeof(int32_t)), "counts allocation") ||
        !cuda_check(cuMemAlloc(&d_iteration_counts, output_count * sizeof(int32_t)), "iterations allocation") ||
        !cuda_check(cuMemcpyHtoD(d_statistics, statistics, statistics_count * sizeof(float)), "copy statistics") ||
        !cuda_check(cuMemcpyHtoD(d_target_stats, target_stats, num_targets * 2u * sizeof(float)), "copy target stats") ||
        !cuda_check(cuMemcpyHtoD(d_alphas, alphas, num_sweeps * sizeof(float)), "copy alphas") ||
        !cuda_check(cuMemcpyHtoD(d_thresholds, thresholds, num_sweeps * sizeof(float)), "copy thresholds") ||
        !cuda_check(cuEventCreate(&begin, CU_EVENT_DEFAULT), "begin event") ||
        !cuda_check(cuEventCreate(&end, CU_EVENT_DEFAULT), "end event")) {
        ok = 0;
        goto cleanup;
    }

    for (iteration = 0u; iteration < warmup_iterations + timed_iterations; ++iteration) {
        void* ridge_args[] = {&d_statistics, &statistics_leading_dimension, &d_target_stats,
                              (void*)&target_stats_leading_dimension, (void*)&num_rows, &num_asts, &num_targets,
                              &num_sweeps, &d_alphas, (void*)&scale_epsilon, &d_coefficients, &d_intercepts, &d_sse,
                              &d_info};
        void* stlsq_args[] = {&d_statistics, &statistics_leading_dimension, &d_target_stats,
                              (void*)&target_stats_leading_dimension, (void*)&num_rows, &num_asts, &num_targets,
                              &num_sweeps, &d_alphas, &d_thresholds, &max_stlsq_iterations, (void*)&scale_epsilon,
                              &d_coefficients, &d_intercepts, &d_sse, &d_info, &d_masks, &d_counts,
                              &d_iteration_counts};
        if (iteration == warmup_iterations && !cuda_check(cuEventRecord(begin, 0), "record begin")) {
            ok = 0;
            goto cleanup;
        }
        if (!cuda_check(cuLaunchKernel(function, (unsigned int)cohorts, 1u, 1u, 32u, 1u, 1u, 0u, 0,
                                       stlsq ? stlsq_args : ridge_args, NULL), "solver launch")) {
            ok = 0;
            goto cleanup;
        }
    }
    if (!cuda_check(cuEventRecord(end, 0), "record end") || !cuda_check(cuEventSynchronize(end), "wait end") ||
        !cuda_check(cuEventElapsedTime(&elapsed_ms, begin, end), "elapsed time")) {
        ok = 0;
        goto cleanup;
    }
    {
        const double seconds = (double)elapsed_ms * 1.0e-3;
        const double output_solves = (double)timed_iterations * (double)output_count;
        printf("mode=%s feature_capacity=%d cohorts=%zu targets=%zu sweeps=%zu warmups=%zu iterations=%zu "
               "max_stlsq_iterations=%zu seconds=%.9f cohort_launches_per_second=%.3f "
               "output_models_per_second=%.3f preprocessing_timed=0 module_load_timed=0 transfers_timed=0\n",
               mode, FEATURE_CAPACITY, cohorts, num_targets, num_sweeps, warmup_iterations, timed_iterations,
               max_stlsq_iterations, seconds, (double)timed_iterations * (double)cohorts / seconds,
               output_solves / seconds);
    }

cleanup:
#define FREE_DEVICE(pointer_) do { if ((pointer_) != 0u) (void)cuMemFree(pointer_); } while (0)
    if (end != NULL) (void)cuEventDestroy(end);
    if (begin != NULL) (void)cuEventDestroy(begin);
    FREE_DEVICE(d_iteration_counts);
    FREE_DEVICE(d_counts);
    FREE_DEVICE(d_masks);
    FREE_DEVICE(d_info);
    FREE_DEVICE(d_sse);
    FREE_DEVICE(d_intercepts);
    FREE_DEVICE(d_coefficients);
    FREE_DEVICE(d_thresholds);
    FREE_DEVICE(d_alphas);
    FREE_DEVICE(d_target_stats);
    FREE_DEVICE(d_statistics);
#undef FREE_DEVICE
    if (module != NULL) (void)cuModuleUnload(module);
    if (context != NULL) (void)cuDevicePrimaryCtxRelease(device);
    free(fatbin);
    free(thresholds);
    free(alphas);
    free(target_stats);
    free(statistics);
    return ok ? 0 : 1;
}
