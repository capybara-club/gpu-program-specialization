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
#include "secant_cuda_runner.h"
#include "secant_ptx.h"
#include "secant_ptx_runner.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ROWS 257u
#define TEST_COLUMNS 3u
#define TEST_TARGETS 2u
#define TEST_SETTINGS 17u
#define TEST_CONSTANTS 2u
#define TEST_ASTS_PER_MODULE 4u
#define TEST_ASTS 5u
#define TEST_OPTIMIZER_ITERATIONS 3u
#define TEST_STATE_WIDTH (1u + 4u * TEST_CONSTANTS)
#define TEST_COMPILE_SCRATCH (16u * 1024u * 1024u)

typedef enum TestBackend {
    TEST_BACKEND_CUDA = 0,
    TEST_BACKEND_PTX = 1
} TestBackend;

static const SecantAstInstruction test_dynamic_ast0[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_ast1[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_dynamic_constant_or_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_dynamic_ast2[] = {
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_or_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_optimizer_ast0[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_optimizer_ast1[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_add_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_optimizer_ast2[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_sub_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const test_dynamic_asts[] = {
    test_dynamic_ast0,
    test_dynamic_ast1,
    test_dynamic_ast2
};

static const SecantAstInstruction* const test_optimizer_asts[] = {
    test_optimizer_ast0,
    test_optimizer_ast1,
    test_optimizer_ast2,
    test_optimizer_ast0,
    test_optimizer_ast1
};

typedef struct TestCUDA {
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    int major;
    int minor;
    int retained;
} TestCUDA;

static uint32_t
test_f32_bits(float value) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int
test_close(float actual, float expected, float tolerance) {
    if (isinf(actual) || isinf(expected)) {
        return isinf(actual) && isinf(expected) && signbit(actual) == signbit(expected);
    }
    return isfinite(actual) && isfinite(expected) &&
        fabsf(actual - expected) <= tolerance * (1.0f + fabsf(expected));
}

static int
test_compare(const char* label, const float* actual, const float* expected, size_t count, float tolerance) {
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        if (!test_close(actual[idx], expected[idx], tolerance)) {
            fprintf(stderr, "%s mismatch at %zu: %.9g != %.9g\n", label, idx, actual[idx], expected[idx]);
            return 0;
        }
    }
    return 1;
}

static int
test_cuda_create(TestCUDA* cuda) {
    memset(cuda, 0, sizeof(*cuda));
    if (cuInit(0u) != CUDA_SUCCESS || cuDeviceGet(&cuda->device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuda->context, cuda->device) != CUDA_SUCCESS) {
        return 0;
    }
    cuda->retained = 1;
    return cuCtxGetCurrent(&cuda->previous_context) == CUDA_SUCCESS &&
        cuCtxSetCurrent(cuda->context) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&cuda->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda->device) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&cuda->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda->device) == CUDA_SUCCESS;
}

static void
test_cuda_destroy(TestCUDA* cuda) {
    (void)cuCtxSetCurrent(cuda->previous_context);
    if (cuda->retained) {
        (void)cuDevicePrimaryCtxRelease(cuda->device);
    }
}

static int
test_dynamic_leaf_backend(
    TestBackend backend,
    const TestCUDA* cuda,
    const float* input,
    const float* targets
) {
    static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
    static const char* const nvptx_options[] = { "--opt-level=1" };
    uint32_t leaf_masks[TEST_SETTINGS];
    uint32_t leaf_words[TEST_SETTINGS * 2u];
    float expected[3u * TEST_TARGETS * TEST_SETTINGS];
    float actual[3u * TEST_TARGETS * TEST_SETTINGS];
    SecantCpuDynamicLeafSSERun cpu_run = secant_cpu_dynamic_leaf_sse_run_init();
    SecantCUDARunner cuda_runner = NULL;
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXRunner ptx_runner = NULL;
    SecantRunnerStats stats = secant_runner_stats_init();
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_targets = 0u;
    CUdeviceptr device_masks = 0u;
    CUdeviceptr device_words = 0u;
    CUdeviceptr device_output = 0u;
    size_t setting;
    size_t log_size = 0u;
    char log[16384] = { 0 };
    int success = 1;
    SecantResult result;

    for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
        const uint32_t mask = (uint32_t)(setting & 3u);

        leaf_masks[setting] = mask;
        leaf_words[setting * 2u] = (mask & 1u) != 0u
            ? (uint32_t)(setting % TEST_COLUMNS)
            : test_f32_bits(-0.4f + 0.07f * (float)setting);
        leaf_words[setting * 2u + 1u] = (mask & 2u) != 0u
            ? (uint32_t)((setting + 1u) % TEST_COLUMNS)
            : test_f32_bits(0.2f - 0.03f * (float)setting);
    }
    memset(expected, 0, sizeof(expected));
    cpu_run.programs.asts.items = test_dynamic_asts;
    cpu_run.programs.asts.count = 3u;
    cpu_run.num_dynamic_leaves = 2u;
    cpu_run.num_input_columns = TEST_COLUMNS;
    cpu_run.num_static_input_columns = TEST_COLUMNS;
    cpu_run.num_targets = TEST_TARGETS;
    cpu_run.input.data = input;
    cpu_run.input.num_elements = TEST_COLUMNS * TEST_ROWS;
    cpu_run.input.leading_dimension = TEST_ROWS;
    cpu_run.leaf_masks.data = leaf_masks;
    cpu_run.leaf_masks.num_elements = TEST_SETTINGS;
    cpu_run.leaf_words.data = leaf_words;
    cpu_run.leaf_words.num_elements = TEST_SETTINGS * 2u;
    cpu_run.leaf_words.leading_dimension = 2u;
    cpu_run.targets.data = targets;
    cpu_run.targets.num_elements = TEST_TARGETS * TEST_ROWS;
    cpu_run.targets.leading_dimension = TEST_ROWS;
    cpu_run.num_rows = TEST_ROWS;
    cpu_run.num_settings = TEST_SETTINGS;
    cpu_run.output.data = expected;
    cpu_run.output.num_elements = 3u * TEST_TARGETS * TEST_SETTINGS;
    cpu_run.output.leading_dimension = TEST_SETTINGS;
    if (secant_cpu_run_dynamic_leaf_sse(&cpu_run) != SECANT_SUCCESS) {
        return 0;
    }
    if (cuMemAlloc(&device_input, sizeof(float) * TEST_COLUMNS * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemAlloc(&device_targets, sizeof(float) * TEST_TARGETS * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemAlloc(&device_masks, sizeof(leaf_masks)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_words, sizeof(leaf_words)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_output, sizeof(actual)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_input, input, sizeof(float) * TEST_COLUMNS * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_targets, targets, sizeof(float) * TEST_TARGETS * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_masks, leaf_masks, sizeof(leaf_masks)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_words, leaf_words, sizeof(leaf_words)) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success && backend == TEST_BACKEND_CUDA) {
        result = secant_cuda_dynamic_leaf_sse_runner_create(
            1u, TEST_ASTS_PER_MODULE, TEST_COLUMNS, TEST_COLUMNS, 2u, TEST_TARGETS, 128u, 128u,
            (uint32_t)cuda->major, (uint32_t)cuda->minor, nvrtc_options, 2u,
            TEST_COMPILE_SCRATCH, 2u, 2u, &cuda_runner);
    } else if (success) {
        const uint32_t source_major = cuda->major >= 8 ? 8u : (uint32_t)cuda->major;
        const uint32_t source_minor = cuda->major >= 8 ? 0u : (uint32_t)cuda->minor;
        SecantPTXResult ptx_result = secant_ptx_dynamic_leaf_sse_create(
            1u, TEST_ASTS_PER_MODULE, TEST_COLUMNS, TEST_COLUMNS, 2u, TEST_TARGETS, 128u, 128u,
            source_major, source_minor, nvrtc_options, 2u, true, log, sizeof(log), &log_size, &ptx_handle);

        result = ptx_result == SECANT_PTX_SUCCESS
            ? secant_ptx_dynamic_leaf_sse_runner_create(
                ptx_handle, (uint32_t)cuda->major, (uint32_t)cuda->minor, nvptx_options, 1u,
                TEST_COMPILE_SCRATCH, 2u, 2u, &ptx_runner)
            : SECANT_ERROR_COMPILE_FAILED;
    }
    if (success && result != SECANT_SUCCESS) {
        fprintf(stderr, "%s dynamic-leaf create failed: %s\n",
            backend == TEST_BACKEND_CUDA ? "CUDA" : "PTX", secant_result_to_string(result));
        if (log_size != 0u) fprintf(stderr, "%s\n", log);
        success = 0;
    }
    if (success && backend == TEST_BACKEND_CUDA) {
        result = secant_cuda_dynamic_leaf_sse_runner_run_all(
            cuda_runner, NULL, 0u, NULL, test_dynamic_asts, 3u,
            (uintptr_t)device_input, TEST_COLUMNS * TEST_ROWS, TEST_COLUMNS, TEST_ROWS,
            (uintptr_t)device_masks, TEST_SETTINGS, (uintptr_t)device_words, TEST_SETTINGS * 2u, 2u,
            TEST_SETTINGS, 5u, TEST_TARGETS, (uintptr_t)device_targets, TEST_TARGETS * TEST_ROWS, TEST_ROWS,
            TEST_ROWS, (uintptr_t)device_output, 3u * TEST_TARGETS * TEST_SETTINGS, TEST_SETTINGS, &stats);
    } else if (success) {
        result = secant_ptx_dynamic_leaf_sse_runner_run_all(
            ptx_runner, NULL, 0u, test_dynamic_asts, 3u,
            (uintptr_t)device_input, TEST_COLUMNS * TEST_ROWS, TEST_COLUMNS, TEST_ROWS,
            (uintptr_t)device_masks, TEST_SETTINGS, (uintptr_t)device_words, TEST_SETTINGS * 2u, 2u,
            TEST_SETTINGS, 5u, TEST_TARGETS, (uintptr_t)device_targets, TEST_TARGETS * TEST_ROWS, TEST_ROWS,
            TEST_ROWS, (uintptr_t)device_output, 3u * TEST_TARGETS * TEST_SETTINGS, TEST_SETTINGS, &stats);
    }
    if (success && (result != SECANT_SUCCESS || cuMemcpyDtoH(actual, device_output, sizeof(actual)) != CUDA_SUCCESS ||
        !test_compare(backend == TEST_BACKEND_CUDA ? "CUDA dynamic leaf" : "PTX dynamic leaf",
            actual, expected, 3u * TEST_TARGETS * TEST_SETTINGS, 1.0e-3f))) {
        success = 0;
    }
    if (ptx_runner != NULL) (void)secant_ptx_runner_destroy(ptx_runner);
    if (ptx_handle != NULL) (void)secant_ptx_handle_destroy(ptx_handle);
    if (cuda_runner != NULL) (void)secant_cuda_runner_destroy(cuda_runner);
    if (device_output != 0u) (void)cuMemFree(device_output);
    if (device_words != 0u) (void)cuMemFree(device_words);
    if (device_masks != 0u) (void)cuMemFree(device_masks);
    if (device_targets != 0u) (void)cuMemFree(device_targets);
    if (device_input != 0u) (void)cuMemFree(device_input);
    return success;
}

static int
test_packed_backend(
    TestBackend backend,
    const TestCUDA* cuda,
    const float* input,
    const float* target
) {
    static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
    static const char* const nvptx_options[] = { "--opt-level=1" };
    static const float initial_constants[TEST_ASTS * TEST_CONSTANTS] = {
        0.5f, -0.2f, 0.8f, 0.3f, -0.4f, 0.7f, 1.1f, -0.6f, 0.2f, 1.0f
    };
    static const float initial_scales[TEST_ASTS * TEST_CONSTANTS] = {
        0.7f, 0.4f, 0.5f, 0.8f, 0.6f, 0.3f, 0.9f, 0.5f, 0.4f, 0.7f
    };
    float expected_constants[TEST_ASTS * TEST_CONSTANTS];
    float expected_scales[TEST_ASTS * TEST_CONSTANTS];
    float expected_velocities[TEST_ASTS * TEST_CONSTANTS] = { 0.0f };
    float expected_current_sse[TEST_ASTS];
    float expected_sse[TEST_ASTS * TEST_SETTINGS];
    float expected_best[TEST_ASTS * TEST_STATE_WIDTH];
    float actual_constants[TEST_ASTS * TEST_CONSTANTS];
    float actual_scales[TEST_ASTS * TEST_CONSTANTS];
    float actual_velocities[TEST_ASTS * TEST_CONSTANTS] = { 0.0f };
    float actual_current_sse[TEST_ASTS];
    float actual_sse[TEST_ASTS * TEST_SETTINGS];
    float actual_best[TEST_ASTS * TEST_STATE_WIDTH];
    SecantCpuPackedConstantOptimizerSSERun cpu_run = secant_cpu_packed_constant_optimizer_sse_run_init();
    SecantCubinPackedConstantOptimizerSSERun gpu_run =
        secant_cubin_packed_constant_optimizer_sse_run_init();
    SecantCUDARunner cuda_runner = NULL;
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXRunner ptx_runner = NULL;
    SecantRunnerStats stats = secant_runner_stats_init();
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_target = 0u;
    CUdeviceptr device_sse = 0u;
    CUdeviceptr device_best = 0u;
    size_t log_size = 0u;
    char log[16384] = { 0 };
    size_t idx;
    int success = 1;
    SecantResult result;

    memcpy(expected_constants, initial_constants, sizeof(initial_constants));
    memcpy(expected_scales, initial_scales, sizeof(initial_scales));
    for (idx = 0u; idx < TEST_ASTS; ++idx) expected_current_sse[idx] = INFINITY;
    memset(expected_sse, 0, sizeof(expected_sse));
    memset(expected_best, 0, sizeof(expected_best));
    cpu_run.programs.asts.items = test_optimizer_asts;
    cpu_run.programs.asts.count = TEST_ASTS;
    cpu_run.programs.current_constants.data = expected_constants;
    cpu_run.programs.current_constants.num_elements = TEST_ASTS * TEST_CONSTANTS;
    cpu_run.programs.current_constants.leading_dimension = TEST_CONSTANTS;
    cpu_run.programs.current_constant_scales.data = expected_scales;
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
    cpu_run.num_iterations = TEST_OPTIMIZER_ITERATIONS;
    cpu_run.seed = UINT64_C(0x123456789abcdef0);
    cpu_run.generation = 11u;
    cpu_run.iteration = 7u;
    cpu_run.update_mode = SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER;
    cpu_run.num_elites = 1u;
    cpu_run.current_constant_velocities.data = expected_velocities;
    cpu_run.current_constant_velocities.num_elements = TEST_ASTS * TEST_CONSTANTS;
    cpu_run.current_constant_velocities.leading_dimension = TEST_CONSTANTS;
    cpu_run.current_sse.data = expected_current_sse;
    cpu_run.current_sse.num_elements = TEST_ASTS;
    cpu_run.momentum = 0.25f;
    cpu_run.scale_learning_rate = 0.2f;
    cpu_run.scale_failure_decay = 0.6f;
    cpu_run.minimum_scale = 1.0e-4f;
    cpu_run.maximum_scale = 4.0f;
    cpu_run.sse.data = expected_sse;
    cpu_run.sse.num_elements = TEST_ASTS * TEST_SETTINGS;
    cpu_run.sse.leading_dimension = TEST_SETTINGS;
    cpu_run.best.data = expected_best;
    cpu_run.best.num_elements = TEST_ASTS * TEST_STATE_WIDTH;
    cpu_run.best.leading_dimension = TEST_STATE_WIDTH;
    if (secant_cpu_run_packed_constant_optimizer_sse(&cpu_run) != SECANT_SUCCESS) return 0;

    memcpy(actual_constants, initial_constants, sizeof(initial_constants));
    memcpy(actual_scales, initial_scales, sizeof(initial_scales));
    for (idx = 0u; idx < TEST_ASTS; ++idx) actual_current_sse[idx] = INFINITY;
    if (cuMemAlloc(&device_input, sizeof(float) * TEST_COLUMNS * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemAlloc(&device_target, sizeof(float) * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemAlloc(&device_sse, sizeof(actual_sse)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_best, sizeof(actual_best)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_input, input, sizeof(float) * TEST_COLUMNS * TEST_ROWS) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_target, target, sizeof(float) * TEST_ROWS) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success && backend == TEST_BACKEND_CUDA) {
        result = secant_cuda_packed_constant_optimizer_sse_runner_create(
            1u, TEST_ASTS_PER_MODULE, TEST_COLUMNS, TEST_CONSTANTS, 128u, 128u,
            (uint32_t)cuda->major, (uint32_t)cuda->minor, nvrtc_options, 2u,
            TEST_COMPILE_SCRATCH, 2u, 2u, &cuda_runner);
    } else if (success) {
        const uint32_t source_major = cuda->major >= 8 ? 8u : (uint32_t)cuda->major;
        const uint32_t source_minor = cuda->major >= 8 ? 0u : (uint32_t)cuda->minor;
        SecantPTXResult ptx_result = secant_ptx_packed_constant_optimizer_sse_create(
            1u, TEST_ASTS_PER_MODULE, TEST_COLUMNS, TEST_CONSTANTS, 128u, 128u,
            source_major, source_minor, nvrtc_options, 2u, true, log, sizeof(log), &log_size, &ptx_handle);

        result = ptx_result == SECANT_PTX_SUCCESS
            ? secant_ptx_packed_constant_optimizer_sse_runner_create(
                ptx_handle, (uint32_t)cuda->major, (uint32_t)cuda->minor, nvptx_options, 1u,
                TEST_COMPILE_SCRATCH, 2u, 2u, &ptx_runner)
            : SECANT_ERROR_COMPILE_FAILED;
    }
    if (success && result != SECANT_SUCCESS) {
        fprintf(stderr, "%s packed create failed: %s\n%s\n",
            backend == TEST_BACKEND_CUDA ? "CUDA" : "PTX", secant_result_to_string(result), log);
        success = 0;
    }

    gpu_run.programs.asts.items = test_optimizer_asts;
    gpu_run.programs.asts.count = TEST_ASTS;
    gpu_run.programs.current_constants.data = actual_constants;
    gpu_run.programs.current_constants.num_elements = TEST_ASTS * TEST_CONSTANTS;
    gpu_run.programs.current_constants.leading_dimension = TEST_CONSTANTS;
    gpu_run.programs.current_constant_scales.data = actual_scales;
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
    gpu_run.num_iterations = TEST_OPTIMIZER_ITERATIONS;
    gpu_run.seed = cpu_run.seed;
    gpu_run.generation = cpu_run.generation;
    gpu_run.iteration = cpu_run.iteration;
    gpu_run.update_mode = cpu_run.update_mode;
    gpu_run.num_elites = cpu_run.num_elites;
    gpu_run.current_constant_velocities.data = actual_velocities;
    gpu_run.current_constant_velocities.num_elements = TEST_ASTS * TEST_CONSTANTS;
    gpu_run.current_constant_velocities.leading_dimension = TEST_CONSTANTS;
    gpu_run.current_sse.data = actual_current_sse;
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
    if (success) {
        result = backend == TEST_BACKEND_CUDA
            ? secant_cuda_packed_constant_optimizer_sse_runner_run(cuda_runner, &gpu_run, NULL, &stats)
            : secant_ptx_packed_constant_optimizer_sse_runner_run(ptx_runner, &gpu_run, &stats);
        if (result != SECANT_SUCCESS ||
            cuMemcpyDtoH(actual_sse, device_sse, sizeof(actual_sse)) != CUDA_SUCCESS ||
            cuMemcpyDtoH(actual_best, device_best, sizeof(actual_best)) != CUDA_SUCCESS) {
            fprintf(stderr, "%s packed optimizer failed: %s\n",
                backend == TEST_BACKEND_CUDA ? "CUDA" : "PTX", secant_result_to_string(result));
            success = 0;
        }
    }
    if (success) {
        const char* label = backend == TEST_BACKEND_CUDA ? "CUDA packed" : "PTX packed";

        success = test_compare(label, actual_sse, expected_sse, TEST_ASTS * TEST_SETTINGS, 2.0e-3f) &&
            test_compare(label, actual_best, expected_best, TEST_ASTS * TEST_STATE_WIDTH, 2.0e-3f) &&
            test_compare(label, actual_constants, expected_constants, TEST_ASTS * TEST_CONSTANTS, 2.0e-3f) &&
            test_compare(label, actual_scales, expected_scales, TEST_ASTS * TEST_CONSTANTS, 2.0e-3f) &&
            test_compare(label, actual_velocities, expected_velocities, TEST_ASTS * TEST_CONSTANTS, 2.0e-3f) &&
            test_compare(label, actual_current_sse, expected_current_sse, TEST_ASTS, 2.0e-3f);
    }
    if (ptx_runner != NULL) (void)secant_ptx_runner_destroy(ptx_runner);
    if (ptx_handle != NULL) (void)secant_ptx_handle_destroy(ptx_handle);
    if (cuda_runner != NULL) (void)secant_cuda_runner_destroy(cuda_runner);
    if (device_best != 0u) (void)cuMemFree(device_best);
    if (device_sse != 0u) (void)cuMemFree(device_sse);
    if (device_target != 0u) (void)cuMemFree(device_target);
    if (device_input != 0u) (void)cuMemFree(device_input);
    return success;
}

int
main(void) {
    float input[TEST_COLUMNS * TEST_ROWS];
    float targets[TEST_TARGETS * TEST_ROWS];
    TestCUDA cuda;
    size_t row;
    int success;

    for (row = 0u; row < TEST_ROWS; ++row) {
        const float x = ((float)(row % 67u) - 33.0f) * 0.03125f;

        input[row] = x;
        input[TEST_ROWS + row] = 0.25f * x * x - 0.3f;
        input[2u * TEST_ROWS + row] = cosf(0.7f * x);
        targets[row] = 1.4f * x - 0.2f;
        targets[TEST_ROWS + row] = sinf(x + 0.3f);
    }
    if (!test_cuda_create(&cuda)) return 1;
    success = test_dynamic_leaf_backend(TEST_BACKEND_CUDA, &cuda, input, targets);
    success = test_dynamic_leaf_backend(TEST_BACKEND_PTX, &cuda, input, targets) && success;
    success = test_packed_backend(TEST_BACKEND_CUDA, &cuda, input, targets) && success;
    success = test_packed_backend(TEST_BACKEND_PTX, &cuda, input, targets) && success;
    test_cuda_destroy(&cuda);
    return success ? 0 : 1;
}
