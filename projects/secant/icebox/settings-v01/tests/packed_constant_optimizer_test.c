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

#include <cuda.h>
#include <nvrtc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_KERNELS 1u
#define TEST_ASTS_PER_KERNEL 4u
#define TEST_ASTS 5u
#define TEST_COLUMNS 2u
#define TEST_CONSTANTS 2u
#define TEST_ROWS 257u
#define TEST_SETTINGS 257u
#define TEST_ITERATIONS 3u
#define TEST_STATE_WIDTH (1u + 4u * TEST_CONSTANTS)

static const SecantAstInstruction test_scale_bias_routine[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_ast0[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_constant_f32_bits(0x3e800000u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_ast1[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_ast2[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_ast3[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_sub_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_ast4[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_invalid_dynamic_column_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_column_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_invalid_dynamic_constant_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(TEST_CONSTANTS),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static int
test_source_compile(
    const char* source,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char architecture[64];
    const char* options[3];
    nvrtcProgram program = NULL;
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;
    nvrtcResult result;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d", major, minor) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    result = nvrtcCreateProgram(&program, source, "packed_constant_optimizer_test.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, 3, options);
    }
    if (result == NVRTC_SUCCESS && nvrtcGetCUBINSize(program, &cubin_size) == NVRTC_SUCCESS && cubin_size != 0u) {
        cubin = (unsigned char*)malloc(cubin_size);
        if (cubin == NULL || nvrtcGetCUBIN(program, (char*)cubin) != NVRTC_SUCCESS) {
            free(cubin);
            cubin = NULL;
        }
    } else {
        size_t log_size = 0u;

        if (program != NULL && nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size != 0u) {
            char* log = (char*)malloc(log_size);

            if (log != NULL && nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                fprintf(stderr, "%s\n", log);
            }
            free(log);
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    if (cubin == NULL) {
        return 0;
    }
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

static int
test_near(float actual, float expected) {
    return fabsf(actual - expected) <= 3.0e-4f * (1.0f + fabsf(expected));
}

int
main(void) {
    static const SecantAstInstruction* const asts[TEST_ASTS] = {
        test_ast0, test_ast1, test_ast2, test_ast3, test_ast4
    };
    static const SecantAstInstruction* const routines[] = {test_scale_bias_routine};
    SecantCubinPackedConstantOptimizerSSERecipe recipe =
        secant_cubin_packed_constant_optimizer_sse_recipe_init();
    SecantCubinPackedConstantOptimizerSSERun gpu_run =
        secant_cubin_packed_constant_optimizer_sse_run_init();
    SecantCpuPackedConstantOptimizerSSERun cpu_run = secant_cpu_packed_constant_optimizer_sse_run_init();
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_target = 0u;
    CUdeviceptr device_sse = 0u;
    CUdeviceptr device_best = 0u;
    unsigned char* cubin = NULL;
    char* source = NULL;
    void* plan_storage = NULL;
    float input[TEST_COLUMNS * TEST_ROWS];
    float target[TEST_ROWS];
    float cpu_sse[TEST_ASTS * TEST_SETTINGS];
    float gpu_sse[TEST_ASTS * TEST_SETTINGS];
    float cpu_best[TEST_ASTS * TEST_STATE_WIDTH];
    float gpu_best[TEST_ASTS * TEST_STATE_WIDTH];
    float cpu_constants[TEST_ASTS * TEST_CONSTANTS] = {
        2.0f, 1.0f,
        0.5f, 1.5f,
        0.0f, -1.0f,
        1.25f, 0.25f,
        1.5f, 0.75f
    };
    float gpu_constants[TEST_ASTS * TEST_CONSTANTS];
    float cpu_scales[TEST_ASTS * TEST_CONSTANTS] = {
        1.25f, 0.50f,
        0.75f, 1.50f,
        0.25f, 2.00f,
        1.00f, 0.625f,
        1.75f, 0.375f
    };
    float gpu_scales[TEST_ASTS * TEST_CONSTANTS];
    float cpu_velocities[TEST_ASTS * TEST_CONSTANTS] = {
        0.10f, -0.20f,
        -0.15f, 0.25f,
        0.30f, -0.10f,
        0.40f, -0.35f,
        -0.25f, 0.15f
    };
    float gpu_velocities[TEST_ASTS * TEST_CONSTANTS];
    float cpu_current_sse[TEST_ASTS] = {INFINITY, INFINITY, INFINITY, 0.0f, INFINITY};
    float gpu_current_sse[TEST_ASTS];
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t plan_storage_size = 0u;
    int major = 0;
    int minor = 0;
    int success = 0;
    size_t idx;

    if (cuInit(0u) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS || cuCtxSetCurrent(context) != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialization failed\n");
        goto cleanup;
    }
    for (idx = 0u; idx < TEST_ROWS; ++idx) {
        const float x = ((float)(idx % 67u) - 33.0f) * 0.125f;

        input[idx] = x;
        input[TEST_ROWS + idx] = 0.25f * x * x - 0.5f;
        target[idx] = 2.0f * x + 1.0f;
    }
    recipe.num_kernels = TEST_KERNELS;
    recipe.asts_per_kernel = TEST_ASTS_PER_KERNEL;
    recipe.num_input_columns = TEST_COLUMNS;
    recipe.num_input_constants = TEST_CONSTANTS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = 256u;
    if (secant_cubin_source_size(&recipe.header, &source_size) != SECANT_SUCCESS ||
        (source = (char*)malloc(source_size)) == NULL ||
        secant_cubin_source_write(&recipe.header, source, source_size) != SECANT_SUCCESS ||
        strstr(source, "secant_cuda_packed_constant_optimizer_reduce_f32") == NULL ||
        strstr(source, "float perturbation_scale") != NULL ||
        strstr(source, "float scale_learning_rate") == NULL ||
        !test_source_compile(source, major, minor, &cubin, &cubin_size) ||
        secant_cubin_plan_storage_size(&recipe.header, cubin, cubin_size, &plan_storage_size) != SECANT_SUCCESS ||
        (plan_storage = malloc(plan_storage_size)) == NULL ||
        secant_cubin_plan_init(
            &recipe.header, cubin, cubin_size, plan_storage, plan_storage_size, &plan) != SECANT_SUCCESS) {
        fprintf(stderr, "packed optimizer source compilation or inspection failed\n");
        goto cleanup;
    }
    options.num_workers = 2u;
    options.num_streams = 4u;
    if (secant_cubin_runner_create(plan, cubin, cubin_size, &options, &runner) != SECANT_SUCCESS ||
        cuMemAlloc(&device_input, sizeof(input)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_target, sizeof(target)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_sse, sizeof(gpu_sse)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_best, sizeof(gpu_best)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_input, input, sizeof(input)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_target, target, sizeof(target)) != CUDA_SUCCESS) {
        fprintf(stderr, "packed optimizer runner or device setup failed\n");
        goto cleanup;
    }
    memset(cpu_sse, 0, sizeof(cpu_sse));
    memset(cpu_best, 0, sizeof(cpu_best));
    memcpy(gpu_constants, cpu_constants, sizeof(gpu_constants));
    memcpy(gpu_scales, cpu_scales, sizeof(gpu_scales));
    memcpy(gpu_velocities, cpu_velocities, sizeof(gpu_velocities));
    memcpy(gpu_current_sse, cpu_current_sse, sizeof(gpu_current_sse));
    cpu_run.programs.asts.items = asts;
    cpu_run.programs.asts.count = TEST_ASTS;
    cpu_run.programs.routines.items = routines;
    cpu_run.programs.routines.count = 1u;
    cpu_run.programs.current_constants.data = cpu_constants;
    cpu_run.programs.current_constants.num_elements = TEST_ASTS * TEST_CONSTANTS;
    cpu_run.programs.current_constants.leading_dimension = TEST_CONSTANTS;
    cpu_run.programs.current_constant_scales.data = cpu_scales;
    cpu_run.programs.current_constant_scales.num_elements = TEST_ASTS * TEST_CONSTANTS;
    cpu_run.programs.current_constant_scales.leading_dimension = TEST_CONSTANTS;
    cpu_run.num_input_columns = TEST_COLUMNS;
    cpu_run.num_input_constants = TEST_CONSTANTS;
    cpu_run.input.data = input;
    cpu_run.input.num_elements = TEST_COLUMNS * TEST_ROWS;
    cpu_run.input.leading_dimension = TEST_ROWS;
    cpu_run.target.data = target;
    cpu_run.target.num_elements = TEST_ROWS;
    cpu_run.target.leading_dimension = TEST_ROWS;
    cpu_run.num_rows = TEST_ROWS;
    cpu_run.num_settings = TEST_SETTINGS;
    cpu_run.num_iterations = TEST_ITERATIONS;
    cpu_run.seed = UINT64_C(0x123456789abcdef0);
    cpu_run.generation = UINT64_C(19);
    cpu_run.iteration = UINT64_C(7);
    cpu_run.current_constant_velocities.data = cpu_velocities;
    cpu_run.current_constant_velocities.num_elements = TEST_ASTS * TEST_CONSTANTS;
    cpu_run.current_constant_velocities.leading_dimension = TEST_CONSTANTS;
    cpu_run.current_sse.data = cpu_current_sse;
    cpu_run.current_sse.num_elements = TEST_ASTS;
    cpu_run.momentum = 0.7f;
    cpu_run.scale_learning_rate = 0.35f;
    cpu_run.scale_failure_decay = 0.6f;
    cpu_run.minimum_scale = 0.01f;
    cpu_run.maximum_scale = 2.0f;
    cpu_run.sse.data = cpu_sse;
    cpu_run.sse.num_elements = TEST_ASTS * TEST_SETTINGS;
    cpu_run.sse.leading_dimension = TEST_SETTINGS;
    cpu_run.best.data = cpu_best;
    cpu_run.best.num_elements = TEST_ASTS * TEST_STATE_WIDTH;
    cpu_run.best.leading_dimension = TEST_STATE_WIDTH;
    {
        const SecantResult result = secant_cpu_run_packed_constant_optimizer_sse(&cpu_run);

        if (result != SECANT_SUCCESS) {
            fprintf(stderr, "packed optimizer CPU oracle failed: %s\n", secant_result_to_string(result));
            goto cleanup;
        }
    }
    gpu_run.programs.asts.items = asts;
    gpu_run.programs.asts.count = TEST_ASTS;
    gpu_run.programs.routines.items = routines;
    gpu_run.programs.routines.count = 1u;
    gpu_run.programs.current_constants.data = gpu_constants;
    gpu_run.programs.current_constants.num_elements = TEST_ASTS * TEST_CONSTANTS;
    gpu_run.programs.current_constants.leading_dimension = TEST_CONSTANTS;
    gpu_run.programs.current_constant_scales.data = gpu_scales;
    gpu_run.programs.current_constant_scales.num_elements = TEST_ASTS * TEST_CONSTANTS;
    gpu_run.programs.current_constant_scales.leading_dimension = TEST_CONSTANTS;
    gpu_run.input.address = (uintptr_t)device_input;
    gpu_run.input.num_elements = TEST_COLUMNS * TEST_ROWS;
    gpu_run.input.leading_dimension = TEST_ROWS;
    gpu_run.target.address = (uintptr_t)device_target;
    gpu_run.target.num_elements = TEST_ROWS;
    gpu_run.target.leading_dimension = TEST_ROWS;
    gpu_run.num_rows = TEST_ROWS;
    gpu_run.num_settings = TEST_SETTINGS;
    gpu_run.num_iterations = TEST_ITERATIONS;
    gpu_run.seed = cpu_run.seed;
    gpu_run.generation = cpu_run.generation;
    gpu_run.iteration = cpu_run.iteration;
    gpu_run.current_constant_velocities.data = gpu_velocities;
    gpu_run.current_constant_velocities.num_elements = TEST_ASTS * TEST_CONSTANTS;
    gpu_run.current_constant_velocities.leading_dimension = TEST_CONSTANTS;
    gpu_run.current_sse.data = gpu_current_sse;
    gpu_run.current_sse.num_elements = TEST_ASTS;
    gpu_run.momentum = cpu_run.momentum;
    gpu_run.scale_learning_rate = cpu_run.scale_learning_rate;
    gpu_run.scale_failure_decay = cpu_run.scale_failure_decay;
    gpu_run.minimum_scale = cpu_run.minimum_scale;
    gpu_run.maximum_scale = cpu_run.maximum_scale;
    gpu_run.sse.address = (uintptr_t)device_sse;
    gpu_run.sse.num_elements = TEST_ASTS * TEST_SETTINGS;
    gpu_run.sse.leading_dimension = TEST_SETTINGS;
    gpu_run.best.address = (uintptr_t)device_best;
    gpu_run.best.num_elements = TEST_ASTS * TEST_STATE_WIDTH;
    gpu_run.best.leading_dimension = TEST_STATE_WIDTH;
    if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_SUCCESS ||
        stats.modules_loaded != 2u * TEST_ITERATIONS ||
        cuMemcpyDtoH(gpu_sse, device_sse, sizeof(gpu_sse)) != CUDA_SUCCESS ||
        cuMemcpyDtoH(gpu_best, device_best, sizeof(gpu_best)) != CUDA_SUCCESS) {
        fprintf(stderr, "packed optimizer GPU execution failed\n");
        goto cleanup;
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_SETTINGS; ++idx) {
        if (!test_near(gpu_sse[idx], cpu_sse[idx])) {
            fprintf(stderr, "packed optimizer SSE mismatch at %zu: gpu=%g cpu=%g\n", idx, gpu_sse[idx], cpu_sse[idx]);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_STATE_WIDTH; ++idx) {
        if (!test_near(gpu_best[idx], cpu_best[idx])) {
            fprintf(stderr, "packed optimizer winner mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_best[idx], cpu_best[idx]);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_CONSTANTS; ++idx) {
        if (!test_near(gpu_constants[idx], cpu_constants[idx])) {
            fprintf(stderr, "packed optimizer constant mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_constants[idx], cpu_constants[idx]);
            goto cleanup;
        }
        if (!test_near(gpu_scales[idx], cpu_scales[idx])) {
            fprintf(stderr, "packed optimizer scale mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_scales[idx], cpu_scales[idx]);
            goto cleanup;
        }
        if (!test_near(gpu_velocities[idx], cpu_velocities[idx])) {
            fprintf(stderr, "packed optimizer velocity mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_velocities[idx], cpu_velocities[idx]);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS; ++idx) {
        if (!test_near(gpu_current_sse[idx], cpu_current_sse[idx])) {
            fprintf(stderr, "packed optimizer incumbent SSE mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_current_sse[idx], cpu_current_sse[idx]);
            goto cleanup;
        }
    }
    if (!test_near(gpu_constants[3u * TEST_CONSTANTS], 1.25f) ||
        !test_near(gpu_constants[3u * TEST_CONSTANTS + 1u], 0.25f) ||
        !test_near(gpu_scales[3u * TEST_CONSTANTS], 1.0f * 0.6f * 0.6f * 0.6f) ||
        !test_near(gpu_scales[3u * TEST_CONSTANTS + 1u], 0.625f * 0.6f * 0.6f * 0.6f) ||
        !test_near(gpu_velocities[3u * TEST_CONSTANTS], 0.4f * 0.7f * 0.7f * 0.7f) ||
        !test_near(gpu_velocities[3u * TEST_CONSTANTS + 1u], -0.35f * 0.7f * 0.7f * 0.7f) ||
        gpu_current_sse[3] != 0.0f) {
        fprintf(stderr, "packed optimizer rejected-step state policy mismatch\n");
        goto cleanup;
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_CONSTANTS; ++idx) {
        cpu_velocities[idx] = 0.0f;
        gpu_velocities[idx] = 0.0f;
    }
    cpu_run.update_mode = SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION;
    cpu_run.num_elites = 8u;
    cpu_run.momentum = 0.0f;
    cpu_run.num_iterations = 2u;
    cpu_run.iteration = 101u;
    gpu_run.update_mode = cpu_run.update_mode;
    gpu_run.num_elites = cpu_run.num_elites;
    gpu_run.momentum = cpu_run.momentum;
    gpu_run.num_iterations = cpu_run.num_iterations;
    gpu_run.iteration = cpu_run.iteration;
    if (secant_cpu_run_packed_constant_optimizer_sse(&cpu_run) != SECANT_SUCCESS ||
        secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_SUCCESS ||
        stats.modules_loaded != 2u * cpu_run.num_iterations ||
        cuMemcpyDtoH(gpu_sse, device_sse, sizeof(gpu_sse)) != CUDA_SUCCESS ||
        cuMemcpyDtoH(gpu_best, device_best, sizeof(gpu_best)) != CUDA_SUCCESS) {
        fprintf(stderr, "packed optimizer elite-distribution execution failed\n");
        goto cleanup;
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_SETTINGS; ++idx) {
        if (!test_near(gpu_sse[idx], cpu_sse[idx])) {
            fprintf(stderr, "packed optimizer elite SSE mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_sse[idx], cpu_sse[idx]);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_STATE_WIDTH; ++idx) {
        if (!test_near(gpu_best[idx], cpu_best[idx])) {
            fprintf(stderr, "packed optimizer elite state mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_best[idx], cpu_best[idx]);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_CONSTANTS; ++idx) {
        if (!test_near(gpu_constants[idx], cpu_constants[idx]) ||
            !test_near(gpu_scales[idx], cpu_scales[idx]) ||
            !test_near(gpu_velocities[idx], cpu_velocities[idx])) {
            fprintf(stderr, "packed optimizer elite host state mismatch at %zu\n", idx);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS; ++idx) {
        if (!test_near(gpu_current_sse[idx], cpu_current_sse[idx])) {
            fprintf(stderr, "packed optimizer elite incumbent mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_current_sse[idx], cpu_current_sse[idx]);
            goto cleanup;
        }
    }
    {
        static const SecantAstInstruction* const invalid_asts[] = {test_invalid_dynamic_column_ast};
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;
        float constants_before[TEST_ASTS * TEST_CONSTANTS];

        memcpy(constants_before, gpu_constants, sizeof(constants_before));
        invalid.programs.asts.items = invalid_asts;
        invalid.programs.asts.count = 1u;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
                SECANT_ERROR_BAD_PROGRAM ||
            memcmp(constants_before, gpu_constants, sizeof(constants_before)) != 0) {
            fprintf(stderr, "packed optimizer accepted a dynamic column or modified centers on validation failure\n");
            goto cleanup;
        }
    }
    {
        static const SecantAstInstruction* const invalid_asts[] = {test_invalid_dynamic_constant_ast};
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;

        invalid.programs.asts.items = invalid_asts;
        invalid.programs.asts.count = 1u;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_BAD_PROGRAM) {
            fprintf(stderr, "packed optimizer accepted an out-of-range dynamic constant index\n");
            goto cleanup;
        }
    }
    {
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;

        invalid.num_elites = 17u;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "packed optimizer accepted too many elites\n");
            goto cleanup;
        }
    }
    {
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;

        invalid.momentum = 0.25f;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "packed optimizer accepted momentum in elite-distribution mode\n");
            goto cleanup;
        }
    }
    {
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;

        invalid.best.address = invalid.sse.address;
        invalid.best.num_elements = invalid.sse.num_elements;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "packed optimizer accepted overlapping outputs\n");
            goto cleanup;
        }
    }
    {
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;
        float invalid_scales[TEST_ASTS * TEST_CONSTANTS];

        memcpy(invalid_scales, gpu_scales, sizeof(invalid_scales));
        invalid_scales[0] = 0.0f;
        invalid.programs.current_constant_scales.data = invalid_scales;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "packed optimizer accepted a non-positive coordinate scale\n");
            goto cleanup;
        }
    }
    {
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;
        float invalid_sse[TEST_ASTS];

        memcpy(invalid_sse, gpu_current_sse, sizeof(invalid_sse));
        invalid_sse[0] = NAN;
        invalid.current_sse.data = invalid_sse;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "packed optimizer accepted a NaN incumbent SSE\n");
            goto cleanup;
        }
    }
    {
        SecantCubinPackedConstantOptimizerSSERun invalid = gpu_run;

        invalid.current_constant_velocities.data = invalid.programs.current_constants.data;
        invalid.current_constant_velocities.num_elements = invalid.programs.current_constants.num_elements;
        if (secant_cubin_runner_run_packed_constant_optimizer_sse(runner, &invalid, &stats) !=
            SECANT_ERROR_INVALID_VALUE) {
            fprintf(stderr, "packed optimizer accepted overlapping host state\n");
            goto cleanup;
        }
    }
    printf("packed_constant_optimizer asts=%u settings=%u rows=%u winner_iterations=%u "
        "elite_iterations=%u module_loads=%u status=pass\n",
        TEST_ASTS, TEST_SETTINGS, TEST_ROWS, TEST_ITERATIONS, 2u,
        2u * TEST_ITERATIONS + 2u * 2u);
    success = 1;

cleanup:
    if (runner != NULL) (void)secant_cubin_runner_destroy(runner);
    if (device_best != 0u) (void)cuMemFree(device_best);
    if (device_sse != 0u) (void)cuMemFree(device_sse);
    if (device_target != 0u) (void)cuMemFree(device_target);
    if (device_input != 0u) (void)cuMemFree(device_input);
    free(plan_storage);
    free(cubin);
    free(source);
    if (context != NULL) {
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return success ? 0 : 1;
}
