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
#include "support/cuda_runtime.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_ROWS 257u
#define TEST_INPUT_COLUMNS 4u
#define TEST_INPUT_CONSTANTS 3u
#define TEST_INPUTS (TEST_INPUT_COLUMNS + TEST_INPUT_CONSTANTS)
#define TEST_TARGETS 2u
#define TEST_SETTINGS 131u
#define TEST_KERNELS 2u
#define TEST_ASTS_PER_KERNEL 3u
#define TEST_ASTS (TEST_KERNELS * TEST_ASTS_PER_KERNEL)
#define TEST_TILE_ROWS 128u
#define TEST_THREADS 128u
#define TEST_SCRATCH_SIZE (8u * 1024u * 1024u)

static unsigned char test_scratch[TEST_SCRATCH_SIZE];
static char test_log[16384];

static const SecantAstInstruction test_mul_add[] = {
    secant_ast_encode_input_f32(0u),
    secant_ast_encode_input_f32(4u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sin[] = {
    secant_ast_encode_input_f32(2u),
    secant_ast_encode_input_f32(5u),
    secant_ast_encode_add_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_max_mul[] = {
    secant_ast_encode_input_f32(3u),
    secant_ast_encode_input_f32(6u),
    secant_ast_encode_max_f32,
    secant_ast_encode_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const test_asts[TEST_ASTS] = {
    test_mul_add,
    test_sin,
    test_max_mul,
    test_max_mul,
    test_mul_add,
    test_sin
};

typedef struct TestCUDA {
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    CUstream stream;
    CUdeviceptr input;
    CUdeviceptr constants;
    CUdeviceptr targets;
    CUdeviceptr output;
    CUmodule module;
    SecantCUDACompiled compiled;
    int major;
    int minor;
    int retained;
} TestCUDA;

static int
test_cuda_create(TestCUDA* cuda) {
    memset(cuda, 0, sizeof(*cuda));
    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&cuda->device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(
            &cuda->context,
            cuda->device) != CUDA_SUCCESS) {
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
        cuStreamCreate(
            &cuda->stream,
            CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

static void
test_cuda_destroy(TestCUDA* cuda) {
    if (cuda->module != NULL) {
        (void)cuModuleUnload(cuda->module);
    }
    if (cuda->compiled != NULL) {
        (void)secant_cuda_compiled_destroy(cuda->compiled);
    }
    if (cuda->output != 0u) {
        (void)cuMemFree(cuda->output);
    }
    if (cuda->targets != 0u) {
        (void)cuMemFree(cuda->targets);
    }
    if (cuda->constants != 0u) {
        (void)cuMemFree(cuda->constants);
    }
    if (cuda->input != 0u) {
        (void)cuMemFree(cuda->input);
    }
    if (cuda->stream != NULL) {
        (void)cuStreamDestroy(cuda->stream);
    }
    (void)cuCtxSetCurrent(cuda->previous_context);
    if (cuda->retained) {
        (void)cuDevicePrimaryCtxRelease(cuda->device);
    }
}

static int
test_expected_generate(
    const float* input,
    const float* constant_settings,
    const float* targets,
    float* expected
) {
    float combined_input[TEST_INPUTS * TEST_ROWS];
    float setting_sse[TEST_ASTS * TEST_TARGETS];
    size_t setting;

    memcpy(
        combined_input,
        input,
        TEST_INPUT_COLUMNS * TEST_ROWS * sizeof(float));
    for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
        size_t constant_idx;
        size_t output_idx;

        for (constant_idx = 0u;
             constant_idx < TEST_INPUT_CONSTANTS;
             ++constant_idx) {
            const float value =
                constant_settings[
                    constant_idx * TEST_SETTINGS + setting];
            size_t row;

            for (row = 0u; row < TEST_ROWS; ++row) {
                combined_input[
                    (TEST_INPUT_COLUMNS + constant_idx) *
                        TEST_ROWS +
                    row] = value;
            }
        }
        memset(setting_sse, 0, sizeof(setting_sse));
        if (secant_cpu_run_static_column_sse(
                TEST_INPUTS,
                TEST_TARGETS,
                NULL,
                0u,
                test_asts,
                TEST_ASTS,
                combined_input,
                TEST_INPUTS * TEST_ROWS,
                TEST_ROWS,
                targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                setting_sse,
                TEST_ASTS * TEST_TARGETS,
                TEST_TARGETS) != SECANT_SUCCESS) {
            return 0;
        }
        for (output_idx = 0u;
             output_idx < TEST_ASTS * TEST_TARGETS;
             ++output_idx) {
            expected[output_idx * TEST_SETTINGS + setting] =
                setting_sse[output_idx];
        }
    }
    return 1;
}

static int
test_compare(
    const float* actual,
    const float* expected,
    size_t count
) {
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        const float error = fabsf(actual[idx] - expected[idx]);
        const float tolerance = 5.0e-4f * (1.0f + fabsf(expected[idx]));

        if (!isfinite(actual[idx]) || !isfinite(expected[idx]) ||
            error > tolerance) {
            fprintf(
                stderr,
                "mismatch at %zu: %.9g != %.9g "
                "(error %.9g, tolerance %.9g)\n",
                idx,
                actual[idx],
                expected[idx],
                error,
                tolerance);
            return 0;
        }
    }
    return 1;
}

int
main(void) {
    static const char* const nvrtc_options[] = {
        "--restrict",
        "--no-cache"
    };
    float input[TEST_INPUT_COLUMNS * TEST_ROWS];
    float constant_settings[TEST_INPUT_CONSTANTS * TEST_SETTINGS];
    float targets[TEST_TARGETS * TEST_ROWS];
    float expected[TEST_ASTS * TEST_TARGETS * TEST_SETTINGS];
    float actual[TEST_ASTS * TEST_TARGETS * TEST_SETTINGS];
    void* functions[TEST_KERNELS] = { NULL };
    void* loaded_module = NULL;
    const void* cubin = NULL;
    size_t cubin_size = 0u;
    size_t log_size = 0u;
    size_t idx;
    TestCUDA cuda;
    int success;

    for (idx = 0u; idx < TEST_ROWS; ++idx) {
        input[idx] = 0.5f + 0.002f * (float)idx;
        input[TEST_ROWS + idx] = 1.1f + 0.001f * (float)idx;
        input[2u * TEST_ROWS + idx] =
            -0.4f + 0.0015f * (float)idx;
        input[3u * TEST_ROWS + idx] =
            0.8f - 0.0005f * (float)idx;
        targets[idx] = 0.3f + 0.001f * (float)idx;
        targets[TEST_ROWS + idx] =
            -0.2f + 0.0007f * (float)idx;
    }
    for (idx = 0u;
         idx < TEST_INPUT_CONSTANTS * TEST_SETTINGS;
         ++idx) {
        const size_t constant_idx = idx / TEST_SETTINGS;
        const size_t setting = idx % TEST_SETTINGS;

        constant_settings[idx] =
            -0.7f +
            0.4f * (float)constant_idx +
            0.005f * (float)setting;
    }

    success = test_expected_generate(
        input,
        constant_settings,
        targets,
        expected);
    memset(&cuda, 0, sizeof(cuda));
    if (success) {
        success = test_cuda_create(&cuda);
    }
    if (success) {
        const SecantCUDAResult result =
            secant_cuda_dynamic_constant_sse_compile(
                TEST_KERNELS,
                TEST_ASTS_PER_KERNEL,
                TEST_INPUT_COLUMNS,
                TEST_INPUT_CONSTANTS,
                TEST_TARGETS,
                TEST_TILE_ROWS,
                TEST_THREADS,
                SECANT_SSE_REDUCTION_MODE_ATOMIC,
                NULL,
                0u,
                NULL,
                test_asts,
                (uint32_t)cuda.major,
                (uint32_t)cuda.minor,
                nvrtc_options,
                2u,
                false,
                test_scratch,
                sizeof(test_scratch),
                test_log,
                sizeof(test_log),
                &log_size,
                &cuda.compiled);

        if (result != SECANT_CUDA_SUCCESS) {
            fprintf(
                stderr,
                "dynamic SSE compile failed: %s\n%s\n",
                secant_cuda_result_to_string(result),
                test_log);
            success = 0;
        }
    }
    if (success &&
        (secant_cuda_compiled_binary_get(
             cuda.compiled,
             &cubin,
             &cubin_size) != SECANT_CUDA_SUCCESS ||
         secant_test_cuda_module_load(
             cubin,
             cubin_size,
             TEST_KERNELS,
             &loaded_module,
             functions) != SECANT_CUDA_SUCCESS)) {
        success = 0;
    }
    if (success) {
        cuda.module = (CUmodule)loaded_module;
    }
    if (success) {
        const size_t input_bytes = sizeof(input);
        const size_t constants_bytes = sizeof(constant_settings);
        const size_t target_bytes = sizeof(targets);
        const size_t output_bytes = sizeof(actual);

        if (cuMemAlloc(&cuda.input, input_bytes) != CUDA_SUCCESS ||
            cuMemAlloc(&cuda.constants, constants_bytes) != CUDA_SUCCESS ||
            cuMemAlloc(&cuda.targets, target_bytes) != CUDA_SUCCESS ||
            cuMemAlloc(&cuda.output, output_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoDAsync(
                cuda.input,
                input,
                input_bytes,
                cuda.stream) != CUDA_SUCCESS ||
            cuMemcpyHtoDAsync(
                cuda.constants,
                constant_settings,
                constants_bytes,
                cuda.stream) != CUDA_SUCCESS ||
            cuMemcpyHtoDAsync(
                cuda.targets,
                targets,
                target_bytes,
                cuda.stream) != CUDA_SUCCESS ||
            cuMemsetD8Async(
                cuda.output,
                0u,
                output_bytes,
                cuda.stream) != CUDA_SUCCESS) {
            success = 0;
        }
    }
    for (idx = 0u; idx < TEST_KERNELS && success; ++idx) {
        const size_t kernel_output_elements =
            TEST_ASTS_PER_KERNEL * TEST_TARGETS * TEST_SETTINGS;
        const CUdeviceptr output =
            cuda.output + idx * kernel_output_elements * sizeof(float);

        if (secant_test_cuda_run_dynamic_constant_sse(
                functions[idx],
                TEST_INPUT_COLUMNS,
                TEST_INPUT_CONSTANTS,
                TEST_ASTS_PER_KERNEL,
                TEST_TARGETS,
                TEST_TILE_ROWS,
                TEST_THREADS,
                (const float*)(uintptr_t)cuda.input,
                TEST_INPUT_COLUMNS * TEST_ROWS,
                TEST_ROWS,
                (const float*)(uintptr_t)cuda.constants,
                TEST_INPUT_CONSTANTS * TEST_SETTINGS,
                TEST_SETTINGS,
                TEST_SETTINGS,
                (const float*)(uintptr_t)cuda.targets,
                TEST_TARGETS * TEST_ROWS,
                TEST_ROWS,
                TEST_ROWS,
                cuda.stream,
                (float*)(uintptr_t)output,
                kernel_output_elements,
                TEST_SETTINGS) != SECANT_CUDA_SUCCESS) {
            success = 0;
        }
    }
    if (success &&
        (cuMemcpyDtoHAsync(
             actual,
             cuda.output,
             sizeof(actual),
             cuda.stream) != CUDA_SUCCESS ||
         cuStreamSynchronize(cuda.stream) != CUDA_SUCCESS ||
         !test_compare(
             actual,
             expected,
             TEST_ASTS * TEST_TARGETS * TEST_SETTINGS))) {
        success = 0;
    }

    test_cuda_destroy(&cuda);
    return success ? 0 : 1;
}
