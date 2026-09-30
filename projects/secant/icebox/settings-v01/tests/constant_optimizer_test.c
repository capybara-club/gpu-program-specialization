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

#define TEST_KERNELS 3u
#define TEST_ASTS 5u
#define TEST_COLUMNS 2u
#define TEST_CONSTANTS 2u
#define TEST_ROWS 257u
#define TEST_SETTINGS 257u
#define TEST_ITERATIONS 3u

static const SecantAstInstruction test_ast0[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
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

typedef struct TestPhilox4x32 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
} TestPhilox4x32;

static TestPhilox4x32
test_philox_round(TestPhilox4x32 c, uint32_t key0, uint32_t key1) {
    const uint64_t product0 = UINT64_C(0xd2511f53) * c.x;
    const uint64_t product1 = UINT64_C(0xcd9e8d57) * c.z;
    TestPhilox4x32 result;

    result.x = (uint32_t)(product1 >> 32u) ^ c.y ^ key0;
    result.y = (uint32_t)product1;
    result.z = (uint32_t)(product0 >> 32u) ^ c.w ^ key1;
    result.w = (uint32_t)product0;
    return result;
}

static TestPhilox4x32
test_random4(
    uint64_t setting,
    uint64_t ast_index,
    uint64_t seed,
    uint64_t generation,
    uint64_t iteration,
    uint32_t group
) {
    TestPhilox4x32 counter;
    uint32_t key0 = (uint32_t)seed ^ (uint32_t)generation * UINT32_C(0x9e3779b9) ^
        (uint32_t)(iteration >> 32u) * UINT32_C(0x85ebca6b) ^ group * UINT32_C(0x27d4eb2d);
    uint32_t key1 = (uint32_t)(seed >> 32u) ^ (uint32_t)(generation >> 32u) * UINT32_C(0xbb67ae85) ^
        (uint32_t)iteration * UINT32_C(0xc2b2ae35) ^ group * UINT32_C(0x165667b1);
    size_t round;

    counter.x = (uint32_t)setting;
    counter.y = (uint32_t)(setting >> 32u);
    counter.z = (uint32_t)ast_index;
    counter.w = (uint32_t)(ast_index >> 32u);
    for (round = 0u; round < 10u; ++round) {
        counter = test_philox_round(counter, key0, key1);
        key0 += UINT32_C(0x9e3779b9);
        key1 += UINT32_C(0xbb67ae85);
    }
    return counter;
}

static float
test_delta(uint32_t word) {
    return (float)(word >> 8u) * 0x1.0p-23f - 1.0f;
}

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
    nvrtcResult nvrtc_result;
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d", major, minor) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    nvrtc_result = nvrtcCreateProgram(&program, source, "constant_optimizer_test.cu", 0, NULL, NULL);
    if (nvrtc_result == NVRTC_SUCCESS) {
        nvrtc_result = nvrtcCompileProgram(program, 3, options);
    }
    if (nvrtc_result == NVRTC_SUCCESS &&
        nvrtcGetCUBINSize(program, &cubin_size) == NVRTC_SUCCESS && cubin_size != 0u) {
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
    return fabsf(actual - expected) <= 2.0e-5f * (1.0f + fabsf(expected));
}

static int
test_cpu_optimize(
    const SecantAstInstruction* const* asts,
    const float* input,
    const float* target,
    uint64_t seed,
    uint64_t generation,
    uint64_t iteration_base,
    uint64_t ast_index_base,
    float perturbation_scale,
    float perturbation_decay,
    float* constants,
    float* output
) {
    SecantCpuConstantOptimizerSSERun run = secant_cpu_constant_optimizer_sse_run_init();
    size_t iteration_idx;
    float scale = perturbation_scale;

    run.programs.asts.items = asts;
    run.programs.asts.count = TEST_ASTS;
    run.num_input_columns = TEST_COLUMNS;
    run.num_input_constants = TEST_CONSTANTS;
    run.input.data = input;
    run.input.num_elements = TEST_COLUMNS * TEST_ROWS;
    run.input.leading_dimension = TEST_ROWS;
    run.current_constants.num_elements = TEST_ASTS * TEST_CONSTANTS;
    run.current_constants.leading_dimension = TEST_CONSTANTS;
    run.target.data = target;
    run.target.num_elements = TEST_ROWS;
    run.target.leading_dimension = TEST_ROWS;
    run.num_rows = TEST_ROWS;
    run.num_settings = TEST_SETTINGS;
    run.seed = seed;
    run.generation = generation;
    run.ast_index_base = ast_index_base;
    run.output.data = output;
    run.output.num_elements = TEST_ASTS * TEST_SETTINGS;
    run.output.leading_dimension = TEST_SETTINGS;

    for (iteration_idx = 0u; iteration_idx < TEST_ITERATIONS; ++iteration_idx) {
        size_t ast_idx;

        memset(output, 0, TEST_ASTS * TEST_SETTINGS * sizeof(*output));
        run.current_constants.data = constants;
        run.iteration = iteration_base + iteration_idx;
        run.perturbation_scale = scale;
        if (secant_cpu_run_constant_optimizer_sse(&run) != SECANT_SUCCESS) {
            return 0;
        }
        for (ast_idx = 0u; ast_idx < TEST_ASTS; ++ast_idx) {
            size_t best_setting = 0u;
            size_t setting;
            size_t constant_idx;

            for (setting = 1u; setting < TEST_SETTINGS; ++setting) {
                if (output[ast_idx * TEST_SETTINGS + setting] <
                    output[ast_idx * TEST_SETTINGS + best_setting]) {
                    best_setting = setting;
                }
            }
            if (best_setting == 0u) {
                continue;
            }
            for (constant_idx = 0u; constant_idx < TEST_CONSTANTS; ++constant_idx) {
                const TestPhilox4x32 random = test_random4(
                    best_setting,
                    ast_index_base + ast_idx,
                    seed,
                    generation,
                    iteration_base + iteration_idx,
                    (uint32_t)(constant_idx / 4u));
                const uint32_t words[4] = {random.x, random.y, random.z, random.w};

                constants[ast_idx * TEST_CONSTANTS + constant_idx] +=
                    scale * test_delta(words[constant_idx % 4u]);
            }
        }
        scale *= perturbation_decay;
    }
    return 1;
}

int
main(void) {
    static const SecantAstInstruction* const asts[TEST_ASTS] = {
        test_ast0, test_ast1, test_ast0, test_ast1, test_ast0
    };
    SecantCubinConstantOptimizerSSERecipe recipe = secant_cubin_constant_optimizer_sse_recipe_init();
    SecantCubinConstantOptimizerSSERun gpu_run = secant_cubin_constant_optimizer_sse_run_init();
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantCubinPlan* plan = NULL;
    SecantCubinRunner runner = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_constants = 0u;
    CUdeviceptr device_target = 0u;
    CUdeviceptr device_output = 0u;
    unsigned char* cubin = NULL;
    unsigned char* incomplete_cubin = NULL;
    char* source = NULL;
    void* plan_storage = NULL;
    float input[TEST_COLUMNS * TEST_ROWS];
    float target[TEST_ROWS];
    float initial_constants[TEST_ASTS * TEST_CONSTANTS] = {
        1.25f, 0.25f,
        -0.5f, 1.5f,
        0.75f, -1.25f,
        2.0f, 0.5f,
        -1.5f, 0.875f
    };
    float reduced_constants[TEST_ASTS * TEST_CONSTANTS];
    float expected_constants[TEST_ASTS * TEST_CONSTANTS];
    float cpu_output[TEST_ASTS * TEST_SETTINGS];
    float gpu_output[TEST_ASTS * TEST_SETTINGS];
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t incomplete_cubin_size = 0u;
    size_t plan_storage_size = 0u;
    int major = 0;
    int minor = 0;
    const uint64_t seed = UINT64_C(0x0123456789abcdef);
    const uint64_t generation = UINT64_C(17);
    const uint64_t iteration = UINT64_C(3);
    const uint64_t ast_index_base = UINT64_C(91);
    const float perturbation_scale = 2.0f;
    const float perturbation_decay = 0.75f;
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
        input[idx] = ((float)(int)(idx % 29u) - 14.0f) * 0.125f;
        input[TEST_ROWS + idx] = ((float)(int)(idx % 17u) - 8.0f) * 0.2f;
        target[idx] = 2.0f * input[idx] + 1.0f;
    }
    memset(cpu_output, 0, sizeof(cpu_output));
    memset(gpu_output, 0, sizeof(gpu_output));
    memcpy(expected_constants, initial_constants, sizeof(expected_constants));
    if (!test_cpu_optimize(
            asts,
            input,
            target,
            seed,
            generation,
            iteration,
            ast_index_base,
            perturbation_scale,
            perturbation_decay,
            expected_constants,
            cpu_output)) {
        fprintf(stderr, "CPU optimizer oracle failed\n");
        goto cleanup;
    }

    recipe.num_kernels = TEST_KERNELS;
    recipe.num_input_columns = TEST_COLUMNS;
    recipe.num_input_constants = TEST_CONSTANTS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = 192u;
    if (secant_cubin_source_size(&recipe.header, &source_size) != SECANT_SUCCESS) {
        goto cleanup;
    }
    source = (char*)malloc(source_size);
    if (source == NULL || secant_cubin_source_write(&recipe.header, source, source_size) != SECANT_SUCCESS) {
        fprintf(stderr, "optimizer source generation failed\n");
        goto cleanup;
    }
    {
        char* reducer = strstr(source, "extern \"C\" __global__\nvoid secant_cuda_constant_optimizer_reduce_f32");
        char saved;

        if (reducer == NULL) {
            fprintf(stderr, "embedded optimizer reducer was not generated\n");
            goto cleanup;
        }
        saved = *reducer;
        *reducer = '\0';
        if (!test_source_compile(source, major, minor, &incomplete_cubin, &incomplete_cubin_size)) {
            fprintf(stderr, "incomplete optimizer fixture compilation failed\n");
            *reducer = saved;
            goto cleanup;
        }
        *reducer = saved;
        if (secant_cubin_plan_storage_size(
                &recipe.header,
                incomplete_cubin,
                incomplete_cubin_size,
                &plan_storage_size) == SECANT_SUCCESS) {
            fprintf(stderr, "optimizer CUBIN without its reducer was accepted\n");
            goto cleanup;
        }
    }
    if (!test_source_compile(source, major, minor, &cubin, &cubin_size) ||
        secant_cubin_plan_storage_size(&recipe.header, cubin, cubin_size, &plan_storage_size) != SECANT_SUCCESS) {
        fprintf(stderr, "optimizer source compilation or inspection failed\n");
        goto cleanup;
    }
    plan_storage = malloc(plan_storage_size);
    if (plan_storage == NULL ||
        secant_cubin_plan_init(
            &recipe.header, cubin, cubin_size, plan_storage, plan_storage_size, &plan) != SECANT_SUCCESS) {
        fprintf(stderr, "optimizer plan initialization failed\n");
        goto cleanup;
    }
    options.num_workers = 2u;
    options.num_streams = 2u;
    if (secant_cubin_runner_create(plan, cubin, cubin_size, &options, &runner) != SECANT_SUCCESS ||
        cuMemAlloc(&device_input, sizeof(input)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_constants, sizeof(initial_constants)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_target, sizeof(target)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_output, sizeof(gpu_output)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_input, input, sizeof(input)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_constants, initial_constants, sizeof(initial_constants)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_target, target, sizeof(target)) != CUDA_SUCCESS) {
        fprintf(stderr, "optimizer runner setup failed\n");
        goto cleanup;
    }

    gpu_run.programs.asts.items = asts;
    gpu_run.programs.asts.count = TEST_ASTS;
    gpu_run.input.address = (uintptr_t)device_input;
    gpu_run.input.num_elements = TEST_COLUMNS * TEST_ROWS;
    gpu_run.input.leading_dimension = TEST_ROWS;
    gpu_run.current_constants.address = (uintptr_t)device_constants;
    gpu_run.current_constants.num_elements = TEST_ASTS * TEST_CONSTANTS;
    gpu_run.current_constants.leading_dimension = TEST_CONSTANTS;
    gpu_run.target.address = (uintptr_t)device_target;
    gpu_run.target.num_elements = TEST_ROWS;
    gpu_run.target.leading_dimension = TEST_ROWS;
    gpu_run.num_rows = TEST_ROWS;
    gpu_run.num_settings = TEST_SETTINGS;
    gpu_run.num_iterations = TEST_ITERATIONS;
    gpu_run.seed = seed;
    gpu_run.generation = generation;
    gpu_run.iteration = iteration;
    gpu_run.ast_index_base = ast_index_base;
    gpu_run.perturbation_scale = perturbation_scale;
    gpu_run.perturbation_decay = perturbation_decay;
    gpu_run.output.address = (uintptr_t)device_output;
    gpu_run.output.num_elements = TEST_ASTS * TEST_SETTINGS;
    gpu_run.output.leading_dimension = TEST_SETTINGS;
    if (secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_SUCCESS ||
        stats.modules_loaded != (TEST_ASTS + TEST_KERNELS - 1u) / TEST_KERNELS ||
        cuMemcpyDtoH(gpu_output, device_output, sizeof(gpu_output)) != CUDA_SUCCESS ||
        cuMemcpyDtoH(reduced_constants, device_constants, sizeof(reduced_constants)) != CUDA_SUCCESS) {
        fprintf(stderr, "optimizer GPU execution failed\n");
        goto cleanup;
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_SETTINGS; ++idx) {
        if (!test_near(gpu_output[idx], cpu_output[idx])) {
            fprintf(stderr, "proposal SSE mismatch at %zu: gpu=%g cpu=%g\n", idx, gpu_output[idx], cpu_output[idx]);
            goto cleanup;
        }
    }
    gpu_run.perturbation_scale = -1.0f;
    if (secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "negative optimizer scale was accepted\n");
        goto cleanup;
    }
    gpu_run.perturbation_scale = perturbation_scale;
    gpu_run.perturbation_decay = 0.0f;
    if (secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "zero optimizer decay was accepted\n");
        goto cleanup;
    }
    gpu_run.perturbation_decay = 1.01f;
    if (secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "optimizer decay above one was accepted\n");
        goto cleanup;
    }
    gpu_run.perturbation_decay = perturbation_decay;
    gpu_run.num_iterations = 0u;
    if (secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "zero optimizer iterations were accepted\n");
        goto cleanup;
    }
    gpu_run.num_iterations = TEST_ITERATIONS;
    for (idx = 0u; idx < TEST_ASTS * TEST_CONSTANTS; ++idx) {
        if (!test_near(reduced_constants[idx], expected_constants[idx])) {
            fprintf(stderr, "optimizer reconstructed constant mismatch at %zu: gpu=%g cpu=%g\n",
                idx, reduced_constants[idx], expected_constants[idx]);
            goto cleanup;
        }
    }
    memset(cpu_output, 0, sizeof(cpu_output));
    memset(gpu_output, 0, sizeof(gpu_output));
    memcpy(expected_constants, initial_constants, sizeof(expected_constants));
    if (!test_cpu_optimize(
            asts,
            input,
            target,
            seed,
            generation,
            iteration,
            ast_index_base,
            perturbation_scale,
            1.0f,
            expected_constants,
            cpu_output) ||
        cuMemcpyHtoD(device_constants, initial_constants, sizeof(initial_constants)) != CUDA_SUCCESS) {
        fprintf(stderr, "constant-scale optimizer oracle setup failed\n");
        goto cleanup;
    }
    gpu_run.perturbation_decay = 1.0f;
    if (secant_cubin_runner_run_constant_optimizer_sse(runner, &gpu_run, &stats) != SECANT_SUCCESS ||
        stats.modules_loaded != (TEST_ASTS + TEST_KERNELS - 1u) / TEST_KERNELS ||
        cuMemcpyDtoH(gpu_output, device_output, sizeof(gpu_output)) != CUDA_SUCCESS ||
        cuMemcpyDtoH(reduced_constants, device_constants, sizeof(reduced_constants)) != CUDA_SUCCESS) {
        fprintf(stderr, "constant-scale optimizer GPU execution failed\n");
        goto cleanup;
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_SETTINGS; ++idx) {
        if (!test_near(gpu_output[idx], cpu_output[idx])) {
            fprintf(stderr, "constant-scale SSE mismatch at %zu: gpu=%g cpu=%g\n",
                idx, gpu_output[idx], cpu_output[idx]);
            goto cleanup;
        }
    }
    for (idx = 0u; idx < TEST_ASTS * TEST_CONSTANTS; ++idx) {
        if (!test_near(reduced_constants[idx], expected_constants[idx])) {
            fprintf(stderr, "constant-scale value mismatch at %zu: gpu=%g cpu=%g\n",
                idx, reduced_constants[idx], expected_constants[idx]);
            goto cleanup;
        }
    }
    {
        const SecantCubinRunHeader* batch[] = {&gpu_run.header, &gpu_run.header};

        if (secant_cubin_runner_run_batch(runner, batch, 2u, &stats) != SECANT_ERROR_UNSUPPORTED_SHAPE) {
            fprintf(stderr, "constant optimizer accepted ambiguous batch execution\n");
            goto cleanup;
        }
    }
    {
        SecantCubinConstantOptimizerSSERecipe tile_256_recipe = recipe;
        SecantCubinPlan* tile_256_plan = NULL;
        unsigned char* tile_256_cubin = NULL;
        char* tile_256_source = NULL;
        void* tile_256_plan_storage = NULL;
        size_t tile_256_source_size = 0u;
        size_t tile_256_cubin_size = 0u;
        size_t tile_256_plan_storage_size = 0u;

        tile_256_recipe.tile_rows = 256u;
        tile_256_recipe.threads_per_block = 256u;
        if (secant_cubin_source_size(&tile_256_recipe.header, &tile_256_source_size) != SECANT_SUCCESS ||
            (tile_256_source = (char*)malloc(tile_256_source_size)) == NULL ||
            secant_cubin_source_write(
                &tile_256_recipe.header, tile_256_source, tile_256_source_size) != SECANT_SUCCESS ||
            !test_source_compile(
                tile_256_source, major, minor, &tile_256_cubin, &tile_256_cubin_size) ||
            secant_cubin_plan_storage_size(
                &tile_256_recipe.header,
                tile_256_cubin,
                tile_256_cubin_size,
                &tile_256_plan_storage_size) != SECANT_SUCCESS ||
            (tile_256_plan_storage = malloc(tile_256_plan_storage_size)) == NULL ||
            secant_cubin_plan_init(
                &tile_256_recipe.header,
                tile_256_cubin,
                tile_256_cubin_size,
                tile_256_plan_storage,
                tile_256_plan_storage_size,
                &tile_256_plan) != SECANT_SUCCESS) {
            fprintf(stderr, "256-row optimizer source compilation or inspection failed\n");
            free(tile_256_plan_storage);
            free(tile_256_cubin);
            free(tile_256_source);
            goto cleanup;
        }
        free(tile_256_plan_storage);
        free(tile_256_cubin);
        free(tile_256_source);
    }
    printf("constant_optimizer asts=%u settings=%u rows=%u status=pass\n", TEST_ASTS, TEST_SETTINGS, TEST_ROWS);
    success = 1;

cleanup:
    if (runner != NULL) (void)secant_cubin_runner_destroy(runner);
    if (device_output != 0u) (void)cuMemFree(device_output);
    if (device_target != 0u) (void)cuMemFree(device_target);
    if (device_constants != 0u) (void)cuMemFree(device_constants);
    if (device_input != 0u) (void)cuMemFree(device_input);
    free(plan_storage);
    free(incomplete_cubin);
    free(cubin);
    free(source);
    if (context != NULL) {
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return success ? 0 : 1;
}
