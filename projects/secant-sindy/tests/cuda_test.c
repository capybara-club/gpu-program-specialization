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
#include "secant.h"
#include "secant_sindy.h"

#include <cuda.h>
#include <nvrtc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    CUDA_TEST_CAPACITY = 4,
    CUDA_TEST_FEATURES = 3,
    CUDA_TEST_TARGETS = 2,
    CUDA_TEST_SWEEPS = 2,
    CUDA_TEST_ROWS = 257,
    CUDA_TEST_STATS = CUDA_TEST_CAPACITY + CUDA_TEST_CAPACITY * CUDA_TEST_CAPACITY +
        CUDA_TEST_CAPACITY * CUDA_TEST_TARGETS,
    CUDA_TEST_OUTPUTS = CUDA_TEST_TARGETS * CUDA_TEST_SWEEPS
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

static void* file_read(const char* path, size_t* size_ret) {
    FILE* file = fopen(path, "rb");
    long length;
    void* data;

    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
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

static int close_enough(float actual, float expected, float absolute, float relative, const char* label) {
    const float tolerance = absolute + relative * fabsf(expected);
    if (fabsf(actual - expected) <= tolerance) {
        return 1;
    }
    fprintf(stderr, "%s mismatch: gpu=%.9g cpu=%.9g tolerance=%.9g\n", label, actual, expected, tolerance);
    return 0;
}

static const SecantAstInstruction gram_input_0[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction gram_input_1[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction gram_input_2[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const gram_asts[CUDA_TEST_FEATURES] = {
    gram_input_0,
    gram_input_1,
    gram_input_2
};

typedef struct GramRunnerState {
    unsigned char* cubin;
    void* plan_storage;
    SecantCubinRunner runner;
} GramRunnerState;

static int gram_cubin_compile(const char* source, int major, int minor, unsigned char** cubin_ret,
                              size_t* cubin_size_ret) {
    char architecture[64];
    const char* options[3];
    nvrtcProgram program = NULL;
    nvrtcResult result;
    size_t cubin_size = 0u;
    size_t log_size = 0u;
    int ok = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d", major, minor) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    result = nvrtcCreateProgram(&program, source, "secant_sindy_gram.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, 3, options);
    }
    if (result != NVRTC_SUCCESS) {
        if (program != NULL && nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size != 0u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL && nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                fprintf(stderr, "%s\n", log);
            }
            free(log);
        }
    } else if (nvrtcGetCUBINSize(program, &cubin_size) == NVRTC_SUCCESS && cubin_size != 0u) {
        unsigned char* cubin = (unsigned char*)malloc(cubin_size);
        if (cubin != NULL && nvrtcGetCUBIN(program, (char*)cubin) == NVRTC_SUCCESS) {
            *cubin_ret = cubin;
            *cubin_size_ret = cubin_size;
            ok = 1;
        } else {
            free(cubin);
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    return ok;
}

static int gram_runner_create(int major, int minor, GramRunnerState* state) {
    SecantCubinGramStatsRecipe recipe = secant_cubin_gram_stats_recipe_init();
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    SecantCubinPlan* plan = NULL;
    char* source = NULL;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    SecantResult result;
    int ok = 0;

    memset(state, 0, sizeof(*state));
    recipe.num_kernels = 1u;
    recipe.asts_per_kernel = 4u;
    recipe.num_inputs = CUDA_TEST_FEATURES;
    recipe.num_targets = CUDA_TEST_TARGETS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = 128u;
    result = secant_cubin_source_size(&recipe.header, &source_size);
    if (result != SECANT_SUCCESS) {
        goto cleanup;
    }
    source = (char*)malloc(source_size);
    if (source == NULL || secant_cubin_source_write(&recipe.header, source, source_size) != SECANT_SUCCESS ||
        !gram_cubin_compile(source, major, minor, &state->cubin, &cubin_size)) {
        goto cleanup;
    }
    result = secant_cubin_plan_storage_size(&recipe.header, state->cubin, cubin_size, &plan_storage_size);
    if (result != SECANT_SUCCESS || plan_storage_size == 0u) {
        goto cleanup;
    }
    state->plan_storage = malloc(plan_storage_size);
    if (state->plan_storage == NULL ||
        secant_cubin_plan_init(&recipe.header, state->cubin, cubin_size, state->plan_storage, plan_storage_size,
                              &plan) != SECANT_SUCCESS) {
        goto cleanup;
    }
    options.num_workers = 1u;
    options.num_streams = 1u;
    if (secant_cubin_runner_create(plan, state->cubin, cubin_size, &options, &state->runner) != SECANT_SUCCESS) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(source);
    if (!ok) {
        if (state->runner != NULL) {
            (void)secant_cubin_runner_destroy(state->runner);
        }
        free(state->plan_storage);
        free(state->cubin);
        memset(state, 0, sizeof(*state));
    }
    return ok;
}

static int gram_runner_run(GramRunnerState* state, CUdeviceptr input, CUdeviceptr targets,
                           CUdeviceptr statistics) {
    SecantCubinGramStatsRun run = secant_cubin_gram_stats_run_init();

    run.programs.asts.items = gram_asts;
    run.programs.asts.count = CUDA_TEST_FEATURES;
    run.input = (SecantDeviceMatrixF32){(uintptr_t)input, CUDA_TEST_FEATURES * CUDA_TEST_ROWS, CUDA_TEST_ROWS};
    run.targets = (SecantDeviceMatrixF32){(uintptr_t)targets, CUDA_TEST_TARGETS * CUDA_TEST_ROWS, CUDA_TEST_ROWS};
    run.num_rows = CUDA_TEST_ROWS;
    run.num_targets = CUDA_TEST_TARGETS;
    run.statistics = (SecantDeviceMatrixF32){(uintptr_t)statistics, CUDA_TEST_STATS, CUDA_TEST_STATS};
    return secant_cubin_runner_run_gram_stats(state->runner, &run, NULL) == SECANT_SUCCESS;
}

static int gram_runner_destroy(GramRunnerState* state) {
    int ok = 1;

    if (state->runner != NULL && secant_cubin_runner_destroy(state->runner) != SECANT_SUCCESS) {
        ok = 0;
    }
    free(state->plan_storage);
    free(state->cubin);
    memset(state, 0, sizeof(*state));
    return ok;
}

int main(int argc, char** argv) {
    float features[CUDA_TEST_FEATURES][CUDA_TEST_ROWS];
    float targets[CUDA_TEST_TARGETS][CUDA_TEST_ROWS];
    float statistics[CUDA_TEST_STATS] = {0};
    float statistics_gpu[CUDA_TEST_STATS] = {0};
    float target_stats_cpu[CUDA_TEST_TARGETS][2] = {{0}};
    float target_stats_gpu[CUDA_TEST_TARGETS][2] = {{0}};
    float alpha[CUDA_TEST_SWEEPS] = {0.0f, 0.05f};
    float threshold[CUDA_TEST_SWEEPS] = {0.5f, 1.0f};
    float coefficients_cpu[CUDA_TEST_OUTPUTS * CUDA_TEST_CAPACITY] = {0};
    float intercepts_cpu[CUDA_TEST_OUTPUTS] = {0};
    float sse_cpu[CUDA_TEST_OUTPUTS] = {0};
    int32_t info_cpu[CUDA_TEST_OUTPUTS] = {0};
    float coefficients_gpu[CUDA_TEST_OUTPUTS * CUDA_TEST_CAPACITY] = {0};
    float intercepts_gpu[CUDA_TEST_OUTPUTS] = {0};
    float sse_gpu[CUDA_TEST_OUTPUTS] = {0};
    int32_t info_gpu[CUDA_TEST_OUTPUTS] = {0};
    uint32_t masks_cpu[CUDA_TEST_OUTPUTS] = {0};
    uint32_t masks_gpu[CUDA_TEST_OUTPUTS] = {0};
    int32_t counts_cpu[CUDA_TEST_OUTPUTS] = {0};
    int32_t counts_gpu[CUDA_TEST_OUTPUTS] = {0};
    int32_t iterations_cpu[CUDA_TEST_OUTPUTS] = {0};
    int32_t iterations_gpu[CUDA_TEST_OUTPUTS] = {0};
    SecantSindyCpuTargetStatsRun target_run = secant_sindy_cpu_target_stats_run_init();
    SecantSindyCpuRidgeRun ridge = secant_sindy_cpu_ridge_run_init();
    SecantSindyCpuSTLSQRun stlsq = secant_sindy_cpu_stlsq_run_init();
    CUdevice device;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction target_function;
    CUfunction ridge_function;
    CUfunction stlsq_function;
    GramRunnerState gram_state = {0};
    int device_major = 0;
    int device_minor = 0;
    CUdeviceptr d_features = 0;
    CUdeviceptr d_targets = 0;
    CUdeviceptr d_statistics = 0;
    CUdeviceptr d_target_stats = 0;
    CUdeviceptr d_alphas = 0;
    CUdeviceptr d_thresholds = 0;
    CUdeviceptr d_coefficients = 0;
    CUdeviceptr d_intercepts = 0;
    CUdeviceptr d_sse = 0;
    CUdeviceptr d_info = 0;
    CUdeviceptr d_masks = 0;
    CUdeviceptr d_counts = 0;
    CUdeviceptr d_iterations = 0;
    void* fatbin = NULL;
    size_t fatbin_size = 0u;
    size_t row;
    size_t lhs;
    size_t rhs_idx;
    int ok = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s solver.fatbin\n", argv[0]);
        return 2;
    }
    for (row = 0u; row < CUDA_TEST_ROWS; ++row) {
        const float centered = (float)row - 128.0f;
        features[0][row] = centered * (1.0f / 53.0f);
        features[1][row] = sinf((float)row * 0.137f) + (float)(row % 7u) * 0.031f;
        features[2][row] = cosf((float)row * 0.073f) - (float)(row % 11u) * 0.019f;
        targets[0][row] = 2.0f + 3.0f * features[0][row] - 1.5f * features[1][row];
        targets[1][row] = -4.0f + 2.0f * features[1][row] - 3.5f * features[2][row];
    }
    for (lhs = 0u; lhs < CUDA_TEST_FEATURES; ++lhs) {
        for (row = 0u; row < CUDA_TEST_ROWS; ++row) {
            statistics[lhs] += features[lhs][row];
        }
        for (rhs_idx = 0u; rhs_idx < CUDA_TEST_FEATURES; ++rhs_idx) {
            for (row = 0u; row < CUDA_TEST_ROWS; ++row) {
                statistics[CUDA_TEST_CAPACITY + lhs * CUDA_TEST_CAPACITY + rhs_idx] +=
                    features[lhs][row] * features[rhs_idx][row];
            }
        }
        for (rhs_idx = 0u; rhs_idx < CUDA_TEST_TARGETS; ++rhs_idx) {
            for (row = 0u; row < CUDA_TEST_ROWS; ++row) {
                statistics[CUDA_TEST_CAPACITY + CUDA_TEST_CAPACITY * CUDA_TEST_CAPACITY +
                           lhs * CUDA_TEST_TARGETS + rhs_idx] += features[lhs][row] * targets[rhs_idx][row];
            }
        }
    }
    target_run.targets.data = &targets[0][0];
    target_run.targets.num_elements = CUDA_TEST_TARGETS * CUDA_TEST_ROWS;
    target_run.targets_leading_dimension = CUDA_TEST_ROWS;
    target_run.num_rows = CUDA_TEST_ROWS;
    target_run.num_targets = CUDA_TEST_TARGETS;
    target_run.target_stats.data = &target_stats_cpu[0][0];
    target_run.target_stats.num_elements = CUDA_TEST_TARGETS * 2u;
    target_run.target_stats_leading_dimension = 2u;
    if (secant_sindy_cpu_target_stats_run(&target_run) != SECANT_SINDY_SUCCESS) {
        return 1;
    }

    ridge.feature_capacity = CUDA_TEST_CAPACITY;
    ridge.num_asts = CUDA_TEST_FEATURES;
    ridge.num_rows = CUDA_TEST_ROWS;
    ridge.num_targets = CUDA_TEST_TARGETS;
    ridge.num_sweeps = CUDA_TEST_SWEEPS;
    ridge.statistics = (SecantSindyConstHostSpanF32){statistics, CUDA_TEST_STATS};
    ridge.statistics_leading_dimension = CUDA_TEST_STATS;
    ridge.target_stats = (SecantSindyConstHostSpanF32){&target_stats_cpu[0][0], CUDA_TEST_TARGETS * 2u};
    ridge.target_stats_leading_dimension = 2u;
    ridge.alphas = (SecantSindyConstHostSpanF32){alpha, CUDA_TEST_SWEEPS};
    ridge.coefficients = (SecantSindyHostSpanF32){coefficients_cpu, CUDA_TEST_OUTPUTS * CUDA_TEST_CAPACITY};
    ridge.intercepts = (SecantSindyHostSpanF32){intercepts_cpu, CUDA_TEST_OUTPUTS};
    ridge.sse = (SecantSindyHostSpanF32){sse_cpu, CUDA_TEST_OUTPUTS};
    ridge.solve_info = (SecantSindyHostSpanS32){info_cpu, CUDA_TEST_OUTPUTS};
    if (secant_sindy_cpu_ridge_run(&ridge) != SECANT_SINDY_SUCCESS) {
        return 1;
    }

    fatbin = file_read(argv[1], &fatbin_size);
    (void)fatbin_size;
    if (fatbin == NULL || !cuda_check(cuInit(0), "cuInit") || !cuda_check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
        !cuda_check(cuDeviceGetAttribute(&device_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                    "compute capability major") ||
        !cuda_check(cuDeviceGetAttribute(&device_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                    "compute capability minor") ||
        !cuda_check(cuModuleLoadData(&module, fatbin), "cuModuleLoadData") ||
        !cuda_check(cuModuleGetFunction(&target_function, module, "secant_sindy_cuda_target_stats_f32"),
                    "target function") ||
        !cuda_check(cuModuleGetFunction(&ridge_function, module, "secant_sindy_cuda_ridge_solve_f32"),
                    "ridge function") ||
        !cuda_check(cuModuleGetFunction(&stlsq_function, module, "secant_sindy_cuda_stlsq_solve_f32"),
                    "stlsq function")) {
        ok = 0;
        goto cleanup;
    }
    if (!gram_runner_create(device_major, device_minor, &gram_state)) {
        fprintf(stderr, "Secant Gram runner creation failed\n");
        ok = 0;
        goto cleanup;
    }
#define CUDA_TEST_ALLOC(pointer_, bytes_) \
    do { if (!cuda_check(cuMemAlloc(&(pointer_), (bytes_)), "cuMemAlloc")) { ok = 0; goto cleanup; } } while (0)
    CUDA_TEST_ALLOC(d_features, sizeof(features));
    CUDA_TEST_ALLOC(d_targets, sizeof(targets));
    CUDA_TEST_ALLOC(d_statistics, sizeof(statistics));
    CUDA_TEST_ALLOC(d_target_stats, sizeof(target_stats_gpu));
    CUDA_TEST_ALLOC(d_alphas, sizeof(alpha));
    CUDA_TEST_ALLOC(d_thresholds, sizeof(threshold));
    CUDA_TEST_ALLOC(d_coefficients, sizeof(coefficients_gpu));
    CUDA_TEST_ALLOC(d_intercepts, sizeof(intercepts_gpu));
    CUDA_TEST_ALLOC(d_sse, sizeof(sse_gpu));
    CUDA_TEST_ALLOC(d_info, sizeof(info_gpu));
    CUDA_TEST_ALLOC(d_masks, sizeof(masks_gpu));
    CUDA_TEST_ALLOC(d_counts, sizeof(counts_gpu));
    CUDA_TEST_ALLOC(d_iterations, sizeof(iterations_gpu));
#undef CUDA_TEST_ALLOC
    ok &= cuda_check(cuMemcpyHtoD(d_features, features, sizeof(features)), "copy features");
    ok &= cuda_check(cuMemcpyHtoD(d_targets, targets, sizeof(targets)), "copy targets");
    ok &= cuda_check(cuMemcpyHtoD(d_alphas, alpha, sizeof(alpha)), "copy alphas");
    ok &= cuda_check(cuMemcpyHtoD(d_thresholds, threshold, sizeof(threshold)), "copy thresholds");
    ok &= cuda_check(cuMemsetD8(d_target_stats, 0u, sizeof(target_stats_gpu)), "clear target stats");
    if (!ok) {
        goto cleanup;
    }
    {
        size_t targets_ld = CUDA_TEST_ROWS;
        size_t num_rows = CUDA_TEST_ROWS;
        size_t num_targets = CUDA_TEST_TARGETS;
        size_t stats_ld = 2u;
        void* args[] = {&d_targets, &targets_ld, &num_rows, &num_targets, &d_target_stats, &stats_ld};
        ok &= cuda_check(cuLaunchKernel(target_function, 4u, CUDA_TEST_TARGETS, 1u, 128u, 1u, 1u, 0u, 0, args, NULL),
                         "launch target stats");
    }
    ok &= cuda_check(cuCtxSynchronize(), "target stats synchronize");
    ok &= cuda_check(cuMemcpyDtoH(target_stats_gpu, d_target_stats, sizeof(target_stats_gpu)), "copy target stats");
    for (lhs = 0u; lhs < CUDA_TEST_TARGETS * 2u; ++lhs) {
        ok &= close_enough((&target_stats_gpu[0][0])[lhs], (&target_stats_cpu[0][0])[lhs], 2.0e-2f, 2.0e-5f,
                           "target stats");
    }
    if (ok && !gram_runner_run(&gram_state, d_features, d_targets, d_statistics)) {
        fprintf(stderr, "Secant Gram runner execution failed\n");
        ok = 0;
    }
    ok &= cuda_check(cuMemcpyDtoH(statistics_gpu, d_statistics, sizeof(statistics_gpu)), "copy Gram statistics");
    for (lhs = 0u; lhs < CUDA_TEST_STATS; ++lhs) {
        ok &= close_enough(statistics_gpu[lhs], statistics[lhs], 5.0e-2f, 5.0e-4f, "Gram statistics");
    }
    {
        size_t stats_ld = CUDA_TEST_STATS;
        size_t target_stats_ld = 2u;
        size_t num_rows = CUDA_TEST_ROWS;
        size_t num_asts = CUDA_TEST_FEATURES;
        size_t num_targets = CUDA_TEST_TARGETS;
        size_t num_sweeps = CUDA_TEST_SWEEPS;
        float scale_epsilon = 1.0e-6f;
        void* args[] = {&d_statistics, &stats_ld, &d_target_stats, &target_stats_ld, &num_rows, &num_asts,
                        &num_targets, &num_sweeps, &d_alphas, &scale_epsilon, &d_coefficients, &d_intercepts,
                        &d_sse, &d_info};
        ok &= cuda_check(cuLaunchKernel(ridge_function, 1u, 1u, 1u, 32u, 1u, 1u, 0u, 0, args, NULL),
                         "launch ridge");
    }
    ok &= cuda_check(cuCtxSynchronize(), "ridge synchronize");
    ok &= cuda_check(cuMemcpyDtoH(coefficients_gpu, d_coefficients, sizeof(coefficients_gpu)), "copy coefficients");
    ok &= cuda_check(cuMemcpyDtoH(intercepts_gpu, d_intercepts, sizeof(intercepts_gpu)), "copy intercepts");
    ok &= cuda_check(cuMemcpyDtoH(sse_gpu, d_sse, sizeof(sse_gpu)), "copy sse");
    ok &= cuda_check(cuMemcpyDtoH(info_gpu, d_info, sizeof(info_gpu)), "copy info");
    for (lhs = 0u; lhs < CUDA_TEST_OUTPUTS * CUDA_TEST_CAPACITY; ++lhs) {
        ok &= close_enough(coefficients_gpu[lhs], coefficients_cpu[lhs], 2.0e-4f, 2.0e-4f, "ridge coefficient");
    }
    for (lhs = 0u; lhs < CUDA_TEST_OUTPUTS; ++lhs) {
        ok &= close_enough(intercepts_gpu[lhs], intercepts_cpu[lhs], 2.0e-4f, 2.0e-4f, "ridge intercept");
        ok &= close_enough(sse_gpu[lhs], sse_cpu[lhs], 7.5e-2f, 5.0e-3f, "ridge sse");
        ok &= info_gpu[lhs] == info_cpu[lhs];
    }

    stlsq.feature_capacity = CUDA_TEST_CAPACITY;
    stlsq.num_asts = CUDA_TEST_FEATURES;
    stlsq.num_rows = CUDA_TEST_ROWS;
    stlsq.num_targets = CUDA_TEST_TARGETS;
    stlsq.num_sweeps = CUDA_TEST_SWEEPS;
    stlsq.max_iterations = CUDA_TEST_CAPACITY;
    stlsq.statistics = ridge.statistics;
    stlsq.statistics_leading_dimension = CUDA_TEST_STATS;
    stlsq.target_stats = ridge.target_stats;
    stlsq.target_stats_leading_dimension = 2u;
    stlsq.alphas = ridge.alphas;
    stlsq.thresholds = (SecantSindyConstHostSpanF32){threshold, CUDA_TEST_SWEEPS};
    stlsq.coefficients = ridge.coefficients;
    stlsq.intercepts = ridge.intercepts;
    stlsq.sse = ridge.sse;
    stlsq.solve_info = ridge.solve_info;
    stlsq.active_masks = (SecantSindyHostSpanU32){masks_cpu, CUDA_TEST_OUTPUTS};
    stlsq.active_counts = (SecantSindyHostSpanS32){counts_cpu, CUDA_TEST_OUTPUTS};
    stlsq.iteration_counts = (SecantSindyHostSpanS32){iterations_cpu, CUDA_TEST_OUTPUTS};
    if (secant_sindy_cpu_stlsq_run(&stlsq) != SECANT_SINDY_SUCCESS) {
        ok = 0;
        goto cleanup;
    }
    {
        size_t stats_ld = CUDA_TEST_STATS;
        size_t target_stats_ld = 2u;
        size_t num_rows = CUDA_TEST_ROWS;
        size_t num_asts = CUDA_TEST_FEATURES;
        size_t num_targets = CUDA_TEST_TARGETS;
        size_t num_sweeps = CUDA_TEST_SWEEPS;
        size_t max_iterations = CUDA_TEST_CAPACITY;
        float scale_epsilon = 1.0e-6f;
        void* args[] = {&d_statistics, &stats_ld, &d_target_stats, &target_stats_ld, &num_rows, &num_asts,
                        &num_targets, &num_sweeps, &d_alphas, &d_thresholds, &max_iterations, &scale_epsilon,
                        &d_coefficients, &d_intercepts, &d_sse, &d_info, &d_masks, &d_counts, &d_iterations};
        ok &= cuda_check(cuLaunchKernel(stlsq_function, 1u, 1u, 1u, 32u, 1u, 1u, 0u, 0, args, NULL),
                         "launch stlsq");
    }
    ok &= cuda_check(cuCtxSynchronize(), "stlsq synchronize");
    ok &= cuda_check(cuMemcpyDtoH(coefficients_gpu, d_coefficients, sizeof(coefficients_gpu)), "copy stlsq beta");
    ok &= cuda_check(cuMemcpyDtoH(intercepts_gpu, d_intercepts, sizeof(intercepts_gpu)), "copy stlsq intercept");
    ok &= cuda_check(cuMemcpyDtoH(sse_gpu, d_sse, sizeof(sse_gpu)), "copy stlsq sse");
    ok &= cuda_check(cuMemcpyDtoH(info_gpu, d_info, sizeof(info_gpu)), "copy stlsq info");
    ok &= cuda_check(cuMemcpyDtoH(masks_gpu, d_masks, sizeof(masks_gpu)), "copy stlsq masks");
    ok &= cuda_check(cuMemcpyDtoH(counts_gpu, d_counts, sizeof(counts_gpu)), "copy stlsq counts");
    ok &= cuda_check(cuMemcpyDtoH(iterations_gpu, d_iterations, sizeof(iterations_gpu)), "copy stlsq iterations");
    for (lhs = 0u; lhs < CUDA_TEST_OUTPUTS * CUDA_TEST_CAPACITY; ++lhs) {
        ok &= close_enough(coefficients_gpu[lhs], coefficients_cpu[lhs], 2.0e-4f, 2.0e-4f, "stlsq coefficient");
    }
    for (lhs = 0u; lhs < CUDA_TEST_OUTPUTS; ++lhs) {
        ok &= close_enough(intercepts_gpu[lhs], intercepts_cpu[lhs], 2.0e-4f, 2.0e-4f, "stlsq intercept");
        ok &= close_enough(sse_gpu[lhs], sse_cpu[lhs], 7.5e-2f, 5.0e-3f, "stlsq sse");
        ok &= info_gpu[lhs] == info_cpu[lhs] && masks_gpu[lhs] == masks_cpu[lhs] &&
            counts_gpu[lhs] == counts_cpu[lhs] && iterations_gpu[lhs] == iterations_cpu[lhs];
    }

cleanup:
#define CUDA_TEST_FREE(pointer_) do { if ((pointer_) != 0) cuMemFree(pointer_); } while (0)
    CUDA_TEST_FREE(d_iterations);
    CUDA_TEST_FREE(d_counts);
    CUDA_TEST_FREE(d_masks);
    CUDA_TEST_FREE(d_info);
    CUDA_TEST_FREE(d_sse);
    CUDA_TEST_FREE(d_intercepts);
    CUDA_TEST_FREE(d_coefficients);
    CUDA_TEST_FREE(d_thresholds);
    CUDA_TEST_FREE(d_alphas);
    CUDA_TEST_FREE(d_target_stats);
    CUDA_TEST_FREE(d_statistics);
    CUDA_TEST_FREE(d_targets);
    CUDA_TEST_FREE(d_features);
#undef CUDA_TEST_FREE
    if (!gram_runner_destroy(&gram_state)) {
        ok = 0;
    }
    if (module != NULL) {
        cuModuleUnload(module);
    }
    if (context != NULL) {
        cuDevicePrimaryCtxRelease(device);
    }
    free(fatbin);
    if (!ok) {
        return 1;
    }
    printf("secant-sindy generated CUDA solver matches CPU oracle\n");
    return 0;
}
