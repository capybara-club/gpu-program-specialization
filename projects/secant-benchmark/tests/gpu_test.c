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
#include "secant_cuda.h"
#include "secant_cuda_runner.h"
#include "secant_ptx.h"
#include "secant_ptx_runner.h"
#include "support/cuda_runtime.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ROWS 257u
#define TEST_INPUTS 4u
#define TEST_TARGETS 2u
#define TEST_ASTS 4u
#define TEST_RUNNER_MODULES 6u
#define TEST_RUNNER_ASTS (TEST_RUNNER_MODULES * TEST_ASTS)
#define TEST_SCRATCH_SIZE (8u * 1024u * 1024u)

typedef enum TestBackend {
    TEST_BACKEND_CUDA = 0,
    TEST_BACKEND_PTX = 1
} TestBackend;

static unsigned char test_scratch[TEST_SCRATCH_SIZE];
static char test_log[16384];

static const SecantAstInstruction test_safe_div[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32(1.0e-6f),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_add[] = {
    secant_ast_encode_input_f32(0u),
    secant_ast_encode_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sin[] = {
    secant_ast_encode_input_f32(2u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_fma[] = {
    secant_ast_encode_input_f32(0u),
    secant_ast_encode_input_f32(2u),
    secant_ast_encode_input_f32(3u),
    secant_ast_encode_fma_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_div[] = {
    secant_ast_encode_input_f32(3u),
    secant_ast_encode_input_f32(1u),
    secant_ast_encode_routine_f32(0u, 2u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const test_routines[] = { test_safe_div };
static const char* const test_routine_names[] = { "safe_div" };
static const SecantAstInstruction* const test_asts[] = {
    test_add, test_sin, test_fma, test_div
};

typedef struct TestCUDA {
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    CUstream stream;
    CUdeviceptr input;
    CUdeviceptr targets;
    CUdeviceptr output;
    int major;
    int minor;
    int retained;
} TestCUDA;

static int
test_cuda_create(TestCUDA* cuda) {
    memset(cuda, 0, sizeof(*cuda));
    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&cuda->device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuda->context, cuda->device) != CUDA_SUCCESS) {
        return 0;
    }
    cuda->retained = 1;
    if (cuCtxGetCurrent(&cuda->previous_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(cuda->context) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &cuda->major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            cuda->device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &cuda->minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            cuda->device) != CUDA_SUCCESS ||
        cuStreamCreate(&cuda->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

static void
test_cuda_destroy(TestCUDA* cuda) {
    if (cuda->stream != NULL) {
        (void)cuStreamDestroy(cuda->stream);
    }
    if (cuda->input != 0u) {
        (void)cuMemFree(cuda->input);
    }
    if (cuda->targets != 0u) {
        (void)cuMemFree(cuda->targets);
    }
    if (cuda->output != 0u) {
        (void)cuMemFree(cuda->output);
    }
    (void)cuCtxSetCurrent(cuda->previous_context);
    if (cuda->retained) {
        (void)cuDevicePrimaryCtxRelease(cuda->device);
    }
}

static int
test_close(float actual, float expected, float tolerance) {
    return isfinite(actual) && isfinite(expected) &&
        fabsf(actual - expected) <= tolerance * (1.0f + fabsf(expected));
}

static int
test_compare(const float* actual, const float* expected, size_t count, float tolerance) {
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        if (!test_close(actual[idx], expected[idx], tolerance)) {
            fprintf(stderr, "mismatch at %zu: %.9g != %.9g\n",
                idx, actual[idx], expected[idx]);
            return 0;
        }
    }
    return 1;
}

static int
test_compile_and_run_materialize(
    TestBackend backend,
    TestCUDA* cuda,
    const float* input,
    const float* expected,
    float* actual
) {
    static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
    static const char* const nvptx_options[] = { "--opt-level=1" };
    const size_t input_bytes = TEST_INPUTS * TEST_ROWS * sizeof(float);
    const size_t output_bytes = TEST_ASTS * TEST_ROWS * sizeof(float);
    SecantCUDACompiled cuda_compiled = NULL;
    void* cuda_module = NULL;
    void* cuda_functions[1];
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXCompiled ptx_compiled = NULL;
    void* ptx_module = NULL;
    void* ptx_functions[1];
    const void* binary = NULL;
    size_t binary_size = 0u;
    size_t log_size = 0u;
    int success = 1;

    if (cuMemAlloc(&cuda->input, input_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&cuda->output, output_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoDAsync(cuda->input, input, input_bytes, cuda->stream) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success && backend == TEST_BACKEND_CUDA) {
        const SecantCUDAResult result = secant_cuda_materialize_compile(
            1u,
            TEST_ASTS,
            TEST_INPUTS,
            test_routines,
            1u,
            test_routine_names,
            test_asts,
            (uint32_t)cuda->major,
            (uint32_t)cuda->minor,
            nvrtc_options,
            2u,
            false,
            test_scratch,
            sizeof(test_scratch),
            test_log,
            sizeof(test_log),
            &log_size,
            &cuda_compiled);

        if (result != SECANT_CUDA_SUCCESS) {
            fprintf(stderr, "CUDA compile failed: %s\n%s\n",
                secant_cuda_result_to_string(result), test_log);
            success = 0;
        }
        if (success &&
            (secant_cuda_compiled_binary_get(
                 cuda_compiled,
                 &binary,
                 &binary_size) != SECANT_CUDA_SUCCESS ||
             secant_test_cuda_module_load(
                 binary,
                 binary_size,
                 1u,
                 &cuda_module,
                 cuda_functions) != SECANT_CUDA_SUCCESS)) {
            success = 0;
        }
        if (success &&
            secant_test_cuda_run_static_column_materialize(
                cuda_functions[0],
                TEST_INPUTS,
                TEST_ASTS,
                (const float*)(uintptr_t)cuda->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                cuda->stream,
                (float*)(uintptr_t)cuda->output,
                TEST_ASTS * TEST_ROWS,
                TEST_ROWS) != SECANT_CUDA_SUCCESS) {
            success = 0;
        }
    } else if (success) {
        const uint32_t source_major = cuda->major >= 8 ? 8u : (uint32_t)cuda->major;
        const uint32_t source_minor = cuda->major >= 8 ? 0u : (uint32_t)cuda->minor;
        SecantPTXResult result = secant_ptx_materialize_create(
            1u,
            TEST_ASTS,
            TEST_INPUTS,
            source_major,
            source_minor,
            nvrtc_options,
            2u,
            false,
            test_log,
            sizeof(test_log),
            &log_size,
            &ptx_handle);

        if (result == SECANT_PTX_SUCCESS) {
            result = secant_ptx_compile(
                ptx_handle,
                test_routines,
                1u,
                test_asts,
                (uint32_t)cuda->major,
                (uint32_t)cuda->minor,
                nvptx_options,
                1u,
                false,
                test_scratch,
                sizeof(test_scratch),
                test_log,
                sizeof(test_log),
                &log_size,
                &ptx_compiled);
        }
        if (result != SECANT_PTX_SUCCESS) {
            fprintf(stderr, "PTX compile failed: %s\n%s\n",
                secant_ptx_result_to_string(result), test_log);
            success = 0;
        }
        if (success &&
            (secant_ptx_compiled_binary_get(
                 ptx_compiled,
                 &binary,
                 &binary_size) != SECANT_PTX_SUCCESS ||
             secant_test_cuda_module_load(
                 binary,
                 binary_size,
                 1u,
                 &ptx_module,
                 ptx_functions) != SECANT_CUDA_SUCCESS)) {
            success = 0;
        }
        if (success &&
            secant_test_cuda_run_static_column_materialize(
                ptx_functions[0],
                TEST_INPUTS,
                TEST_ASTS,
                (const float*)(uintptr_t)cuda->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                cuda->stream,
                (float*)(uintptr_t)cuda->output,
                TEST_ASTS * TEST_ROWS,
                TEST_ROWS) != SECANT_CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success &&
        (cuMemcpyDtoHAsync(actual, cuda->output, output_bytes, cuda->stream) != CUDA_SUCCESS ||
         cuStreamSynchronize(cuda->stream) != CUDA_SUCCESS ||
         !test_compare(actual, expected, TEST_ASTS * TEST_ROWS, 2.0e-4f))) {
        success = 0;
    }

    if (ptx_module != NULL &&
        cuModuleUnload((CUmodule)ptx_module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (ptx_compiled != NULL &&
        secant_ptx_compiled_destroy(ptx_compiled) != SECANT_PTX_SUCCESS) {
        success = 0;
    }
    if (ptx_handle != NULL && secant_ptx_handle_destroy(ptx_handle) != SECANT_PTX_SUCCESS) {
        success = 0;
    }
    if (cuda_module != NULL &&
        cuModuleUnload((CUmodule)cuda_module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (cuda_compiled != NULL &&
        secant_cuda_compiled_destroy(cuda_compiled) != SECANT_CUDA_SUCCESS) {
        success = 0;
    }
    (void)cuMemFree(cuda->output);
    (void)cuMemFree(cuda->input);
    cuda->output = 0u;
    cuda->input = 0u;
    return success;
}

static int
test_compile_and_run_sse(
    TestBackend backend,
    TestCUDA* cuda,
    const float* input,
    const float* targets,
    const float* expected,
    float* actual
) {
    static const char* const nvrtc_options[] = { "--restrict", "--no-cache" };
    static const char* const nvptx_options[] = { "--opt-level=1" };
    const size_t input_bytes = TEST_INPUTS * TEST_ROWS * sizeof(float);
    const size_t target_bytes = TEST_TARGETS * TEST_ROWS * sizeof(float);
    const size_t output_bytes = TEST_ASTS * TEST_TARGETS * sizeof(float);
    SecantCUDACompiled cuda_compiled = NULL;
    void* cuda_module = NULL;
    void* cuda_functions[1];
    SecantPTXHandle ptx_handle = NULL;
    SecantPTXCompiled ptx_compiled = NULL;
    void* ptx_module = NULL;
    void* ptx_functions[1];
    const void* binary = NULL;
    size_t binary_size = 0u;
    size_t log_size = 0u;
    int success = 1;

    memcpy(actual, expected, output_bytes);
    if (cuMemAlloc(&cuda->input, input_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&cuda->targets, target_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&cuda->output, output_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoDAsync(cuda->input, input, input_bytes, cuda->stream) != CUDA_SUCCESS ||
        cuMemcpyHtoDAsync(cuda->targets, targets, target_bytes, cuda->stream) != CUDA_SUCCESS ||
        cuMemsetD8Async(cuda->output, 0u, output_bytes, cuda->stream) != CUDA_SUCCESS) {
        success = 0;
    }
    if (success && backend == TEST_BACKEND_CUDA) {
        SecantCUDAResult result = secant_cuda_sse_compile(
            1u,
            TEST_ASTS,
            TEST_INPUTS,
            TEST_TARGETS,
            128u,
            128u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC,
            test_routines,
            1u,
            test_routine_names,
            test_asts,
            (uint32_t)cuda->major,
            (uint32_t)cuda->minor,
            nvrtc_options,
            2u,
            false,
            test_scratch,
            sizeof(test_scratch),
            test_log,
            sizeof(test_log),
            &log_size,
            &cuda_compiled);

        if (result != SECANT_CUDA_SUCCESS) {
            fprintf(stderr, "CUDA SSE compile failed: %s\n%s\n",
                secant_cuda_result_to_string(result), test_log);
            success = 0;
        }
        if (success &&
            (secant_cuda_compiled_binary_get(
                 cuda_compiled,
                 &binary,
                 &binary_size) != SECANT_CUDA_SUCCESS ||
             secant_test_cuda_module_load(
                 binary,
                 binary_size,
                 1u,
                 &cuda_module,
                 cuda_functions) != SECANT_CUDA_SUCCESS)) {
            success = 0;
        }
        if (success &&
            secant_test_cuda_run_static_column_sse(
                cuda_functions[0],
                TEST_INPUTS,
                TEST_ASTS,
                TEST_TARGETS,
                128u,
                128u,
                (const float*)(uintptr_t)cuda->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                (const float*)(uintptr_t)cuda->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                cuda->stream,
                (float*)(uintptr_t)cuda->output,
                TEST_ASTS * TEST_TARGETS,
                TEST_TARGETS) != SECANT_CUDA_SUCCESS) {
            success = 0;
        }
    } else if (success) {
        const uint32_t source_major = cuda->major >= 8 ? 8u : (uint32_t)cuda->major;
        const uint32_t source_minor = cuda->major >= 8 ? 0u : (uint32_t)cuda->minor;
        SecantPTXResult result = secant_ptx_sse_create(
            1u,
            TEST_ASTS,
            TEST_INPUTS,
            TEST_TARGETS,
            128u,
            128u,
            SECANT_SSE_REDUCTION_MODE_ATOMIC,
            source_major,
            source_minor,
            nvrtc_options,
            2u,
            false,
            test_log,
            sizeof(test_log),
            &log_size,
            &ptx_handle);

        if (result == SECANT_PTX_SUCCESS) {
            result = secant_ptx_compile(
                ptx_handle,
                test_routines,
                1u,
                test_asts,
                (uint32_t)cuda->major,
                (uint32_t)cuda->minor,
                nvptx_options,
                1u,
                false,
                test_scratch,
                sizeof(test_scratch),
                test_log,
                sizeof(test_log),
                &log_size,
                &ptx_compiled);
        }
        if (result != SECANT_PTX_SUCCESS) {
            fprintf(stderr, "PTX SSE compile failed: %s\n%s\n",
                secant_ptx_result_to_string(result), test_log);
            success = 0;
        }
        if (success &&
            (secant_ptx_compiled_binary_get(
                 ptx_compiled,
                 &binary,
                 &binary_size) != SECANT_PTX_SUCCESS ||
             secant_test_cuda_module_load(
                 binary,
                 binary_size,
                 1u,
                 &ptx_module,
                 ptx_functions) != SECANT_CUDA_SUCCESS)) {
            success = 0;
        }
        if (success &&
            secant_test_cuda_run_static_column_sse(
                ptx_functions[0],
                TEST_INPUTS,
                TEST_ASTS,
                TEST_TARGETS,
                128u,
                128u,
                (const float*)(uintptr_t)cuda->input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                (const float*)(uintptr_t)cuda->targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                cuda->stream,
                (float*)(uintptr_t)cuda->output,
                TEST_ASTS * TEST_TARGETS,
                TEST_TARGETS) != SECANT_CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success &&
        (cuMemcpyDtoHAsync(actual, cuda->output, output_bytes, cuda->stream) != CUDA_SUCCESS ||
         cuStreamSynchronize(cuda->stream) != CUDA_SUCCESS ||
         !test_compare(actual, expected, TEST_ASTS * TEST_TARGETS, 5.0e-4f))) {
        success = 0;
    }

    if (ptx_module != NULL &&
        cuModuleUnload((CUmodule)ptx_module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (ptx_compiled != NULL &&
        secant_ptx_compiled_destroy(ptx_compiled) != SECANT_PTX_SUCCESS) {
        success = 0;
    }
    if (ptx_handle != NULL && secant_ptx_handle_destroy(ptx_handle) != SECANT_PTX_SUCCESS) {
        success = 0;
    }
    if (cuda_module != NULL &&
        cuModuleUnload((CUmodule)cuda_module) != CUDA_SUCCESS) {
        success = 0;
    }
    if (cuda_compiled != NULL &&
        secant_cuda_compiled_destroy(cuda_compiled) != SECANT_CUDA_SUCCESS) {
        success = 0;
    }
    (void)cuMemFree(cuda->output);
    (void)cuMemFree(cuda->targets);
    (void)cuMemFree(cuda->input);
    cuda->output = 0u;
    cuda->targets = 0u;
    cuda->input = 0u;
    return success;
}

static int
test_runner(
    TestBackend backend,
    int sse,
    TestCUDA* cuda,
    const float* input,
    const float* targets,
    const float* expected_materialize,
    const float* expected_sse
) {
    static const char* const nvrtc_options[] = {
        "--restrict",
        "--no-cache"
    };
    static const char* const nvptx_options[] = {
        "--opt-level=1"
    };
    const size_t input_bytes =
        TEST_INPUTS * TEST_ROWS * sizeof(float);
    const size_t target_bytes =
        TEST_TARGETS * TEST_ROWS * sizeof(float);
    const size_t output_elements = sse
        ? TEST_RUNNER_ASTS * TEST_TARGETS
        : TEST_RUNNER_ASTS * TEST_ROWS;
    const size_t output_bytes = output_elements * sizeof(float);
    const SecantAstInstruction* runner_asts[TEST_RUNNER_ASTS];
    float* expected = NULL;
    float* actual = NULL;
    SecantCUDARunner cuda_runner = NULL;
    SecantPTXRunner ptx_runner = NULL;
    SecantPTXHandle ptx_handle = NULL;
    SecantRunnerStats stats;
    size_t log_size = 0u;
    size_t ast_idx;
    int success = 1;

    for (ast_idx = 0u; ast_idx < TEST_RUNNER_ASTS; ++ast_idx) {
        runner_asts[ast_idx] = test_asts[ast_idx % TEST_ASTS];
    }
    expected = (float*)malloc(output_bytes);
    actual = (float*)malloc(output_bytes);
    if (expected == NULL || actual == NULL) {
        success = 0;
    }
    if (success) {
        const size_t module_output_elements = sse
            ? TEST_ASTS * TEST_TARGETS
            : TEST_ASTS * TEST_ROWS;
        const float* source = sse
            ? expected_sse
            : expected_materialize;
        size_t module_idx;

        for (module_idx = 0u;
             module_idx < TEST_RUNNER_MODULES;
             ++module_idx) {
            memcpy(
                expected + module_idx * module_output_elements,
                source,
                module_output_elements * sizeof(float));
        }
    }
    if (success &&
        (cuMemAlloc(&cuda->input, input_bytes) != CUDA_SUCCESS ||
         cuMemAlloc(&cuda->output, output_bytes) != CUDA_SUCCESS ||
         cuMemcpyHtoD(
             cuda->input,
             input,
             input_bytes) != CUDA_SUCCESS ||
         (sse &&
          (cuMemAlloc(&cuda->targets, target_bytes) != CUDA_SUCCESS ||
           cuMemcpyHtoD(
               cuda->targets,
               targets,
               target_bytes) != CUDA_SUCCESS)))) {
        success = 0;
    }
    if (success && backend == TEST_BACKEND_CUDA) {
        SecantResult result;

        result = sse
            ? secant_cuda_sse_runner_create(
                1u,
                TEST_ASTS,
                TEST_INPUTS,
                TEST_TARGETS,
                128u,
                128u,
                (uint32_t)cuda->major,
                (uint32_t)cuda->minor,
                nvrtc_options,
                2u,
                TEST_SCRATCH_SIZE,
                2u,
                2u,
                &cuda_runner)
            : secant_cuda_materialize_runner_create(
                1u,
                TEST_ASTS,
                TEST_INPUTS,
                (uint32_t)cuda->major,
                (uint32_t)cuda->minor,
                nvrtc_options,
                2u,
                TEST_SCRATCH_SIZE,
                2u,
                2u,
                &cuda_runner);
        if (result != SECANT_SUCCESS) {
            fprintf(
                stderr,
                "CUDA runner create failed: %s\n",
                secant_result_to_string(result));
            success = 0;
        }
    }
    if (success && backend == TEST_BACKEND_PTX) {
        const uint32_t source_major =
            cuda->major >= 8 ? 8u : (uint32_t)cuda->major;
        const uint32_t source_minor =
            cuda->major >= 8 ? 0u : (uint32_t)cuda->minor;
        SecantPTXResult ptx_result;
        SecantResult result;

        ptx_result = sse
            ? secant_ptx_sse_create(
                1u,
                TEST_ASTS,
                TEST_INPUTS,
                TEST_TARGETS,
                128u,
                128u,
                SECANT_SSE_REDUCTION_MODE_ATOMIC,
                source_major,
                source_minor,
                nvrtc_options,
                2u,
                false,
                test_log,
                sizeof(test_log),
                &log_size,
                &ptx_handle)
            : secant_ptx_materialize_create(
                1u,
                TEST_ASTS,
                TEST_INPUTS,
                source_major,
                source_minor,
                nvrtc_options,
                2u,
                false,
                test_log,
                sizeof(test_log),
                &log_size,
                &ptx_handle);
        if (ptx_result != SECANT_PTX_SUCCESS) {
            fprintf(
                stderr,
                "PTX runner template create failed: %s\n%s\n",
                secant_ptx_result_to_string(ptx_result),
                test_log);
            success = 0;
        }
        if (success) {
            result = sse
                ? secant_ptx_sse_runner_create(
                    ptx_handle,
                    (uint32_t)cuda->major,
                    (uint32_t)cuda->minor,
                    nvptx_options,
                    1u,
                    TEST_SCRATCH_SIZE,
                    2u,
                    2u,
                    &ptx_runner)
                : secant_ptx_materialize_runner_create(
                    ptx_handle,
                    (uint32_t)cuda->major,
                    (uint32_t)cuda->minor,
                    nvptx_options,
                    1u,
                    TEST_SCRATCH_SIZE,
                    2u,
                    2u,
                    &ptx_runner);
            if (result != SECANT_SUCCESS) {
                fprintf(
                    stderr,
                    "PTX runner create failed: %s\n",
                    secant_result_to_string(result));
                success = 0;
            }
        }
    }
    if (success) {
        SecantResult result;

        if (!sse) {
            const size_t module_span = TEST_ASTS * TEST_ROWS;

            result = backend == TEST_BACKEND_CUDA
                ? secant_cuda_materialize_runner_run_all(
                    cuda_runner,
                    test_routines,
                    1u,
                    test_routine_names,
                    runner_asts,
                    TEST_RUNNER_ASTS,
                    (uintptr_t)cuda->input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->output,
                    output_elements,
                    TEST_ROWS,
                    module_span - 1u,
                    &stats)
                : secant_ptx_materialize_runner_run_all(
                    ptx_runner,
                    test_routines,
                    1u,
                    runner_asts,
                    TEST_RUNNER_ASTS,
                    (uintptr_t)cuda->input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->output,
                    output_elements,
                    TEST_ROWS,
                    module_span - 1u,
                    &stats);
            if (result != SECANT_ERROR_INVALID_VALUE) {
                fprintf(
                    stderr,
                    "%s runner overlap result: %s\n",
                    backend == TEST_BACKEND_CUDA ? "CUDA" : "PTX",
                    secant_result_to_string(result));
                success = 0;
            }
        }
    }
    if (success) {
        SecantResult result;

        if (backend == TEST_BACKEND_CUDA) {
            result = sse
                ? secant_cuda_sse_runner_run_all(
                    cuda_runner,
                    test_routines,
                    1u,
                    test_routine_names,
                    runner_asts,
                    TEST_RUNNER_ASTS,
                    (uintptr_t)cuda->input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->targets,
                    TEST_TARGETS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->output,
                    output_elements,
                    TEST_TARGETS,
                    &stats)
                : secant_cuda_materialize_runner_run_all(
                    cuda_runner,
                    test_routines,
                    1u,
                    test_routine_names,
                    runner_asts,
                    TEST_RUNNER_ASTS,
                    (uintptr_t)cuda->input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->output,
                    output_elements,
                    TEST_ROWS,
                    TEST_ASTS * TEST_ROWS,
                    &stats);
        } else {
            result = sse
                ? secant_ptx_sse_runner_run_all(
                    ptx_runner,
                    test_routines,
                    1u,
                    runner_asts,
                    TEST_RUNNER_ASTS,
                    (uintptr_t)cuda->input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->targets,
                    TEST_TARGETS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->output,
                    output_elements,
                    TEST_TARGETS,
                    &stats)
                : secant_ptx_materialize_runner_run_all(
                    ptx_runner,
                    test_routines,
                    1u,
                    runner_asts,
                    TEST_RUNNER_ASTS,
                    (uintptr_t)cuda->input,
                    TEST_INPUTS * TEST_ROWS,
                    TEST_ROWS,
                    TEST_ROWS,
                    (uintptr_t)cuda->output,
                    output_elements,
                    TEST_ROWS,
                    TEST_ASTS * TEST_ROWS,
                    &stats);
        }
        if (result != SECANT_SUCCESS ||
            stats.num_modules != TEST_RUNNER_MODULES ||
            stats.modules_loaded != TEST_RUNNER_MODULES ||
            stats.compile_critical_seconds <= 0.0 ||
            stats.compile_critical_seconds >
                stats.compile_work_seconds) {
            fprintf(
                stderr,
                "%s %s runner failed: %s\n",
                backend == TEST_BACKEND_CUDA ? "CUDA" : "PTX",
                sse ? "SSE" : "materialize",
                secant_result_to_string(result));
            success = 0;
        }
    }
    if (success &&
        (cuMemcpyDtoH(actual, cuda->output, output_bytes) !=
             CUDA_SUCCESS ||
         !test_compare(
             actual,
             expected,
             output_elements,
             sse ? 5.0e-4f : 3.0e-5f))) {
        success = 0;
    }
    if (ptx_runner != NULL &&
        secant_ptx_runner_destroy(ptx_runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (cuda_runner != NULL &&
        secant_cuda_runner_destroy(cuda_runner) != SECANT_SUCCESS) {
        success = 0;
    }
    if (ptx_handle != NULL &&
        secant_ptx_handle_destroy(ptx_handle) != SECANT_PTX_SUCCESS) {
        success = 0;
    }
    if (cuda->output != 0u) {
        (void)cuMemFree(cuda->output);
        cuda->output = 0u;
    }
    if (cuda->targets != 0u) {
        (void)cuMemFree(cuda->targets);
        cuda->targets = 0u;
    }
    if (cuda->input != 0u) {
        (void)cuMemFree(cuda->input);
        cuda->input = 0u;
    }
    free(actual);
    free(expected);
    return success;
}

int
main(void) {
    float input[TEST_INPUTS * TEST_ROWS];
    float targets[TEST_TARGETS * TEST_ROWS];
    float expected_materialize[TEST_ASTS * TEST_ROWS];
    float actual_materialize[TEST_ASTS * TEST_ROWS];
    float expected_sse[TEST_ASTS * TEST_TARGETS] = { 0.0f };
    float actual_sse[TEST_ASTS * TEST_TARGETS];
    TestCUDA cuda;
    SecantResult cpu_result;
    size_t row;
    int success;

    for (row = 0u; row < TEST_ROWS; ++row) {
        input[row] = 0.5f + 0.002f * (float)row;
        input[TEST_ROWS + row] = 1.1f + 0.001f * (float)row;
        input[2u * TEST_ROWS + row] = -0.4f + 0.0015f * (float)row;
        input[3u * TEST_ROWS + row] = 0.8f - 0.0005f * (float)row;
        targets[row] = 0.3f + 0.001f * (float)row;
        targets[TEST_ROWS + row] = -0.2f + 0.0007f * (float)row;
    }
    cpu_result = secant_cpu_run_static_column_materialize(
        TEST_INPUTS,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        input,
        TEST_INPUTS * TEST_ROWS,
        TEST_ROWS,
        TEST_ROWS,
        expected_materialize,
        TEST_ASTS * TEST_ROWS,
        TEST_ROWS);
    if (cpu_result != SECANT_SUCCESS) {
        return 1;
    }
    cpu_result = secant_cpu_run_static_column_sse(
        TEST_INPUTS,
        TEST_TARGETS,
        test_routines,
        1u,
        test_asts,
        TEST_ASTS,
        input,
        TEST_INPUTS * TEST_ROWS,
        TEST_ROWS,
        targets,
        TEST_TARGETS * TEST_ROWS,
        TEST_ROWS,
        TEST_ROWS,
        expected_sse,
        TEST_ASTS * TEST_TARGETS,
        TEST_TARGETS);
    if (cpu_result != SECANT_SUCCESS || !test_cuda_create(&cuda)) {
        return 1;
    }

    success = test_compile_and_run_materialize(
        TEST_BACKEND_CUDA, &cuda, input, expected_materialize, actual_materialize);
    success = test_compile_and_run_materialize(
        TEST_BACKEND_PTX, &cuda, input, expected_materialize, actual_materialize) && success;
    success = test_compile_and_run_sse(
        TEST_BACKEND_CUDA, &cuda, input, targets, expected_sse, actual_sse) && success;
    success = test_compile_and_run_sse(
        TEST_BACKEND_PTX, &cuda, input, targets, expected_sse, actual_sse) && success;
    success = test_runner(
        TEST_BACKEND_CUDA,
        0,
        &cuda,
        input,
        targets,
        expected_materialize,
        expected_sse) && success;
    success = test_runner(
        TEST_BACKEND_PTX,
        0,
        &cuda,
        input,
        targets,
        expected_materialize,
        expected_sse) && success;
    success = test_runner(
        TEST_BACKEND_CUDA,
        1,
        &cuda,
        input,
        targets,
        expected_materialize,
        expected_sse) && success;
    success = test_runner(
        TEST_BACKEND_PTX,
        1,
        &cuda,
        input,
        targets,
        expected_materialize,
        expected_sse) && success;

    test_cuda_destroy(&cuda);
    return success ? 0 : 1;
}
