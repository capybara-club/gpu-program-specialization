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
#include "hip_runtime.h"

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define _SECANT_HIP_COMMON_NAME_BYTES 256u
#define _SECANT_HIP_COMMON_SSE_MAX_WAVES 16u

#define _SECANT_HIP_COMMON_ERROR_RET(ans) do { SecantResult _secant_hip_common_result = (ans); return _secant_hip_common_result; } while (0)

const char*
secant_test_hip_result_to_string(SecantResult result) {
    return secant_result_to_string(result);
}

static int
_secant_hip_common_checked_add(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static int
_secant_hip_common_checked_mul(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
_secant_hip_common_span_required(
    size_t outer_count,
    size_t leading_dimension,
    size_t inner_count,
    size_t* required_ret
) {
    size_t offset;

    return outer_count != 0u &&
        inner_count != 0u &&
        _secant_hip_common_checked_mul(
            outer_count - 1u,
            leading_dimension,
            &offset) &&
        _secant_hip_common_checked_add(offset, inner_count, required_ret);
}

SecantResult
secant_test_hip_module_load(
    const void* hsaco,
    size_t hsaco_size,
    const char* function_name_prefix,
    size_t num_functions,
    void** module_ret,
    void** functions
) {
    hipModule_t module = NULL;
    SecantResult result = SECANT_SUCCESS;
    size_t function_idx;

    if (hsaco == NULL || hsaco_size == 0u ||
        function_name_prefix == NULL ||
        function_name_prefix[0] == '\0' ||
        num_functions == 0u ||
        module_ret == NULL || functions == NULL) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *module_ret = NULL;
    memset(functions, 0, num_functions * sizeof(*functions));
    if (hipModuleLoadData(&module, hsaco) != hipSuccess) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    for (function_idx = 0u;
         function_idx < num_functions && result == SECANT_SUCCESS;
         ++function_idx) {
        char name[_SECANT_HIP_COMMON_NAME_BYTES];
        hipFunction_t function = NULL;
        const int name_bytes = snprintf(
            name,
            sizeof(name),
            "%s_%03zu",
            function_name_prefix,
            function_idx);

        if (name_bytes < 0) {
            result = SECANT_ERROR_FORMAT;
        } else if ((size_t)name_bytes >= sizeof(name)) {
            result = SECANT_ERROR_OVERFLOW;
        } else if (hipModuleGetFunction(
                &function,
                module,
                name) != hipSuccess) {
            result = SECANT_ERROR_DRIVER_FAILED;
        } else {
            functions[function_idx] = (void*)function;
        }
    }
    if (result != SECANT_SUCCESS) {
        (void)hipModuleUnload(module);
        memset(functions, 0, num_functions * sizeof(*functions));
        _SECANT_HIP_COMMON_ERROR_RET(result);
    }
    *module_ret = (void*)module;
    return SECANT_SUCCESS;
}

SecantResult
secant_test_hip_run_static_column_materialize(
    void* function,
    size_t num_inputs,
    size_t asts_per_kernel,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_rows,
    void* native_stream,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    const unsigned int block_x = 256u;
    const float* device_input = input;
    float* device_output = output;
    hipStream_t stream = (hipStream_t)native_stream;
    size_t required_input;
    size_t required_output;
    unsigned int grid_x;
    void* args[5];

    if (function == NULL || num_inputs == 0u ||
        asts_per_kernel == 0u || num_rows == 0u ||
        input_leading_dimension < num_rows ||
        output_leading_dimension < num_rows ||
        !_secant_hip_common_span_required(
            num_inputs,
            input_leading_dimension,
            num_rows,
            &required_input) ||
        !_secant_hip_common_span_required(
            asts_per_kernel,
            output_leading_dimension,
            num_rows,
            &required_output) ||
        required_input > input_num_elements ||
        required_output > output_num_elements) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (num_rows > (size_t)UINT_MAX * block_x) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    grid_x = (unsigned int)((num_rows + block_x - 1u) / block_x);
    args[0] = &device_input;
    args[1] = &input_leading_dimension;
    args[2] = &num_rows;
    args[3] = &device_output;
    args[4] = &output_leading_dimension;
    if (hipModuleLaunchKernel(
            (hipFunction_t)function,
            grid_x, 1u, 1u,
            block_x, 1u, 1u,
            0u, stream, args, NULL) != hipSuccess) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    return SECANT_SUCCESS;
}

SecantResult
secant_test_hip_run_static_column_sse(
    void* function,
    size_t num_inputs,
    size_t asts_per_kernel,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    void* native_stream,
    float* output_sse,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    const float* device_input = input;
    const float* device_targets = targets;
    float* device_output = output_sse;
    hipStream_t stream = (hipStream_t)native_stream;
    const size_t num_tiles =
        tile_rows != 0u
            ? num_rows / tile_rows +
                (num_rows % tile_rows != 0u ? 1u : 0u)
            : 0u;
    size_t required_input;
    size_t required_targets;
    size_t required_output;
    int max_threads_per_block;
    void* args[7];

    if (function == NULL || num_inputs == 0u ||
        asts_per_kernel == 0u || num_targets == 0u ||
        tile_rows == 0u || threads_per_block < 64u ||
        threads_per_block > tile_rows ||
        threads_per_block >
            _SECANT_HIP_COMMON_SSE_MAX_WAVES * 64u ||
        threads_per_block % 64u != 0u ||
        num_rows == 0u ||
        input_leading_dimension < num_rows ||
        targets_leading_dimension < num_rows ||
        output_leading_dimension < num_targets ||
        !_secant_hip_common_span_required(
            num_inputs,
            input_leading_dimension,
            num_rows,
            &required_input) ||
        !_secant_hip_common_span_required(
            num_targets,
            targets_leading_dimension,
            num_rows,
            &required_targets) ||
        !_secant_hip_common_span_required(
            asts_per_kernel,
            output_leading_dimension,
            num_targets,
            &required_output) ||
        required_input > input_num_elements ||
        required_targets > targets_num_elements ||
        required_output > output_num_elements) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (num_tiles > UINT_MAX || threads_per_block > UINT_MAX ||
        hipFuncGetAttribute(
            &max_threads_per_block,
            HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            (hipFunction_t)function) != hipSuccess ||
        threads_per_block > (size_t)max_threads_per_block) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    args[0] = &device_input;
    args[1] = &input_leading_dimension;
    args[2] = &device_targets;
    args[3] = &targets_leading_dimension;
    args[4] = &num_rows;
    args[5] = &device_output;
    args[6] = &output_leading_dimension;
    if (hipModuleLaunchKernel(
            (hipFunction_t)function,
            (unsigned int)num_tiles, 1u, 1u,
            (unsigned int)threads_per_block, 1u, 1u,
            0u, stream, args, NULL) != hipSuccess) {
        _SECANT_HIP_COMMON_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    return SECANT_SUCCESS;
}

#undef _SECANT_HIP_COMMON_ERROR_RET
