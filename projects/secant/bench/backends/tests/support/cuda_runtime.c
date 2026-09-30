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
#include "cuda_runtime.h"

#include <cuda.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define _SECANT_CUDA_COMMON_ERROR_RET(ans) do { \
    SecantCUDAResult _secant_cuda_common_result = (ans); \
    return _secant_cuda_common_result; \
} while (0)

const char*
secant_test_cuda_result_to_string(SecantCUDAResult result) {
    static const char* const strings[SECANT_CUDA_RESULT_NUM_ENUMS] = {
        "SECANT_CUDA_SUCCESS",
        "SECANT_CUDA_ERROR_INVALID_VALUE",
        "SECANT_CUDA_ERROR_OVERFLOW",
        "SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_CUDA_ERROR_FORMAT",
        "SECANT_CUDA_ERROR_BAD_PROGRAM",
        "SECANT_CUDA_ERROR_STACK_OVERFLOW",
        "SECANT_CUDA_ERROR_STACK_UNDERFLOW",
        "SECANT_CUDA_ERROR_TOO_MANY_ARGS",
        "SECANT_CUDA_ERROR_UNSUPPORTED_OP",
        "SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS",
        "SECANT_CUDA_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS",
        "SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED",
        "SECANT_CUDA_ERROR_ALLOCATION_FAILED",
        "SECANT_CUDA_ERROR_COMPILE_FAILED",
        "SECANT_CUDA_ERROR_INVALID_STATE",
        "SECANT_CUDA_ERROR_UNSUPPORTED_SHAPE"
    };

    if (result == SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED) {
        return "SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED";
    }
    if (result == SECANT_TEST_CUDA_ERROR_RUN_FAILED) {
        return "SECANT_TEST_CUDA_ERROR_RUN_FAILED";
    }
    return result < SECANT_CUDA_RESULT_NUM_ENUMS
        ? strings[result]
        : "SECANT_TEST_CUDA_ERROR_UNKNOWN";
}

static int
_secant_cuda_common_checked_add(
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
_secant_cuda_common_checked_mul(
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
_secant_cuda_common_span_required(
    size_t outer_count,
    size_t leading_dimension,
    size_t inner_count,
    size_t* required_ret
) {
    size_t offset;

    return outer_count != 0u &&
        inner_count != 0u &&
        _secant_cuda_common_checked_mul(
            outer_count - 1u,
            leading_dimension,
            &offset) &&
        _secant_cuda_common_checked_add(offset, inner_count, required_ret);
}

static int
_secant_cuda_common_function_index(
    const char* name,
    size_t num_functions,
    size_t* function_idx_ret
) {
    const char* suffix;
    size_t function_idx = 0u;

    if (name == NULL || function_idx_ret == NULL) {
        return 0;
    }
    suffix = strrchr(name, '_');
    if (suffix == NULL || suffix[1] == '\0') {
        return 0;
    }
    ++suffix;
    while (*suffix != '\0') {
        const unsigned int digit = (unsigned int)(unsigned char)*suffix - '0';

        if (digit > 9u ||
            function_idx > (SIZE_MAX - digit) / 10u) {
            return 0;
        }
        function_idx = function_idx * 10u + digit;
        ++suffix;
    }
    if (function_idx >= num_functions) {
        return 0;
    }
    *function_idx_ret = function_idx;
    return 1;
}

static SecantCUDAResult
_secant_cuda_common_module_load_current(
    const void* cubin,
    size_t num_functions,
    CUmodule* module_ret,
    void** functions
) {
    CUfunction* enumerated_functions = NULL;
    CUmodule module = NULL;
    CUmoduleLoadingMode loading_mode;
    unsigned int function_count;
    SecantCUDAResult result = SECANT_CUDA_SUCCESS;
    size_t function_idx;

    if (num_functions > UINT_MAX) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    if (cuModuleLoadData(&module, cubin) != CUDA_SUCCESS ||
        cuModuleGetLoadingMode(&loading_mode) != CUDA_SUCCESS ||
        cuModuleGetFunctionCount(&function_count, module) != CUDA_SUCCESS ||
        function_count < (unsigned int)num_functions) {
        result = SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED;
    }
    if (result == SECANT_CUDA_SUCCESS) {
        enumerated_functions = (CUfunction*)malloc(
            function_count * sizeof(*enumerated_functions));
        if (enumerated_functions == NULL) {
            result = SECANT_CUDA_ERROR_ALLOCATION_FAILED;
        }
    }
    if (result == SECANT_CUDA_SUCCESS &&
        cuModuleEnumerateFunctions(
            enumerated_functions,
            function_count,
            module) != CUDA_SUCCESS) {
        result = SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED;
    }
    memset(functions, 0, num_functions * sizeof(*functions));
    for (function_idx = 0u;
         function_idx < (size_t)function_count &&
             result == SECANT_CUDA_SUCCESS;
         ++function_idx) {
        const char* name = NULL;
        size_t parsed_idx;

        if (cuFuncGetName(
                &name,
                enumerated_functions[function_idx]) != CUDA_SUCCESS) {
            result = SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED;
        } else if (_secant_cuda_common_function_index(
                name,
                num_functions,
                &parsed_idx)) {
            if (functions[parsed_idx] != NULL ||
                (loading_mode != CU_MODULE_EAGER_LOADING &&
                 cuFuncLoad(enumerated_functions[function_idx]) !=
                    CUDA_SUCCESS)) {
                result = SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED;
            } else {
                functions[parsed_idx] =
                    (void*)enumerated_functions[function_idx];
            }
        }
    }
    for (function_idx = 0u;
         function_idx < num_functions && result == SECANT_CUDA_SUCCESS;
         ++function_idx) {
        if (functions[function_idx] == NULL) {
            result = SECANT_TEST_CUDA_ERROR_MODULE_LOAD_FAILED;
        }
    }
    free(enumerated_functions);
    if (result != SECANT_CUDA_SUCCESS) {
        if (module != NULL) {
            (void)cuModuleUnload(module);
        }
        memset(functions, 0, num_functions * sizeof(*functions));
        _SECANT_CUDA_COMMON_ERROR_RET(result);
    }
    *module_ret = module;
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_test_cuda_module_load(
    const void* cubin,
    size_t cubin_size,
    size_t num_functions,
    void** module_ret,
    void** functions
) {
    CUmodule module = NULL;
    SecantCUDAResult result;

    if (cubin == NULL || cubin_size == 0u || num_functions == 0u ||
        module_ret == NULL || functions == NULL) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *module_ret = NULL;
    memset(functions, 0, num_functions * sizeof(*functions));
    result = _secant_cuda_common_module_load_current(
        cubin,
        num_functions,
        &module,
        functions);
    if (result != SECANT_CUDA_SUCCESS) {
        _SECANT_CUDA_COMMON_ERROR_RET(result);
    }
    *module_ret = (void*)module;
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_test_cuda_run_static_column_materialize(
    void* function,
    size_t num_inputs,
    size_t asts_per_kernel,
    int active_counts_launch_abi,
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
    CUdeviceptr device_input = (CUdeviceptr)(uintptr_t)input;
    CUdeviceptr device_output = (CUdeviceptr)(uintptr_t)output;
    CUstream stream = (CUstream)native_stream;
    size_t required_input;
    size_t required_output;
    unsigned int grid_x;
    void* args[6];

    if (function == NULL || num_inputs == 0u ||
        asts_per_kernel == 0u || num_rows == 0u ||
        (active_counts_launch_abi != 0 && active_counts_launch_abi != 1) ||
        input_leading_dimension < num_rows ||
        output_leading_dimension < num_rows ||
        !_secant_cuda_common_span_required(
            num_inputs,
            input_leading_dimension,
            num_rows,
            &required_input) ||
        !_secant_cuda_common_span_required(
            asts_per_kernel,
            output_leading_dimension,
            num_rows,
            &required_output) ||
        required_input > input_num_elements ||
        required_output > output_num_elements) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (num_rows > (size_t)UINT_MAX * block_x) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    grid_x = (unsigned int)((num_rows + block_x - 1u) / block_x);
    args[0] = &device_input;
    args[1] = &input_leading_dimension;
    args[2] = &num_rows;
    if (active_counts_launch_abi) {
        args[3] = &asts_per_kernel;
        args[4] = &device_output;
        args[5] = &output_leading_dimension;
    } else {
        args[3] = &device_output;
        args[4] = &output_leading_dimension;
    }
    if (cuLaunchKernel(
            (CUfunction)function,
            grid_x, 1u, 1u,
            block_x, 1u, 1u,
            0u, stream, args, NULL) != CUDA_SUCCESS) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_TEST_CUDA_ERROR_RUN_FAILED);
    }
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_test_cuda_run_static_column_sse(
    void* function,
    size_t num_inputs,
    size_t asts_per_kernel,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    int active_counts_launch_abi,
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
    CUdeviceptr device_input = (CUdeviceptr)(uintptr_t)input;
    CUdeviceptr device_targets = (CUdeviceptr)(uintptr_t)targets;
    CUdeviceptr device_output = (CUdeviceptr)(uintptr_t)output_sse;
    CUstream stream = (CUstream)native_stream;
    const size_t num_tiles =
        tile_rows != 0u
            ? num_rows / tile_rows +
                (num_rows % tile_rows != 0u ? 1u : 0u)
            : 0u;
    size_t required_input;
    size_t required_targets;
    size_t required_output;
    int max_threads_per_block;
    void* args[9];

    if (function == NULL || num_inputs == 0u ||
        asts_per_kernel == 0u || num_targets == 0u ||
        (active_counts_launch_abi != 0 && active_counts_launch_abi != 1) ||
        tile_rows == 0u || threads_per_block < 32u ||
        threads_per_block > tile_rows ||
        threads_per_block >
            SECANT_CUDA_SSE_MAX_WARPS * 32u ||
        threads_per_block % 32u != 0u ||
        num_rows == 0u ||
        input_leading_dimension < num_rows ||
        targets_leading_dimension < num_rows ||
        output_leading_dimension < num_targets ||
        !_secant_cuda_common_span_required(
            num_inputs,
            input_leading_dimension,
            num_rows,
            &required_input) ||
        !_secant_cuda_common_span_required(
            num_targets,
            targets_leading_dimension,
            num_rows,
            &required_targets) ||
        !_secant_cuda_common_span_required(
            asts_per_kernel,
            output_leading_dimension,
            num_targets,
            &required_output) ||
        required_input > input_num_elements ||
        required_targets > targets_num_elements ||
        required_output > output_num_elements) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (num_tiles > UINT_MAX || threads_per_block > UINT_MAX ||
        cuFuncGetAttribute(
            &max_threads_per_block,
            CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            (CUfunction)function) != CUDA_SUCCESS ||
        threads_per_block > (size_t)max_threads_per_block) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_TEST_CUDA_ERROR_RUN_FAILED);
    }
    args[0] = &device_input;
    args[1] = &input_leading_dimension;
    args[2] = &device_targets;
    args[3] = &targets_leading_dimension;
    args[4] = &num_rows;
    if (active_counts_launch_abi) {
        args[5] = &asts_per_kernel;
        args[6] = &num_targets;
        args[7] = &device_output;
        args[8] = &output_leading_dimension;
    } else {
        args[5] = &device_output;
        args[6] = &output_leading_dimension;
    }
    if (cuLaunchKernel(
            (CUfunction)function,
            (unsigned int)num_tiles, 1u, 1u,
            (unsigned int)threads_per_block, 1u, 1u,
            0u, stream, args, NULL) != CUDA_SUCCESS) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_TEST_CUDA_ERROR_RUN_FAILED);
    }
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_test_cuda_run_dynamic_constant_sse(
    void* function,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t asts_per_kernel,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    int active_counts_launch_abi,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* constant_settings,
    size_t constant_settings_num_elements,
    size_t constants_leading_dimension,
    size_t num_settings,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    void* native_stream,
    float* output_sse,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    CUdeviceptr device_input = (CUdeviceptr)(uintptr_t)input;
    CUdeviceptr device_constant_settings = (CUdeviceptr)(uintptr_t)constant_settings;
    CUdeviceptr device_targets = (CUdeviceptr)(uintptr_t)targets;
    CUdeviceptr device_output = (CUdeviceptr)(uintptr_t)output_sse;
    CUstream stream = (CUstream)native_stream;
    const size_t num_tiles =
        tile_rows != 0u
            ? num_rows / tile_rows +
                (num_rows % tile_rows != 0u ? 1u : 0u)
            : 0u;
    size_t num_output_series;
    size_t required_input;
    size_t required_constants;
    size_t required_targets;
    size_t required_output;
    int max_threads_per_block;
    void* args[12];

    if (function == NULL || num_input_columns == 0u ||
        num_input_columns >
            SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants >
            SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        asts_per_kernel == 0u || num_targets == 0u ||
        (active_counts_launch_abi != 0 && active_counts_launch_abi != 1) ||
        tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u || num_settings == 0u ||
        num_rows == 0u ||
        input_leading_dimension < num_rows ||
        constants_leading_dimension < num_settings ||
        targets_leading_dimension < num_rows ||
        output_leading_dimension < num_settings ||
        !_secant_cuda_common_checked_mul(
            asts_per_kernel,
            num_targets,
            &num_output_series) ||
        !_secant_cuda_common_span_required(
            num_input_columns,
            input_leading_dimension,
            num_rows,
            &required_input) ||
        !_secant_cuda_common_span_required(
            num_input_constants,
            constants_leading_dimension,
            num_settings,
            &required_constants) ||
        !_secant_cuda_common_span_required(
            num_targets,
            targets_leading_dimension,
            num_rows,
            &required_targets) ||
        !_secant_cuda_common_span_required(
            num_output_series,
            output_leading_dimension,
            num_settings,
            &required_output) ||
        required_input > input_num_elements ||
        required_constants > constant_settings_num_elements ||
        required_targets > targets_num_elements ||
        required_output > output_num_elements) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (num_tiles > UINT_MAX || threads_per_block > UINT_MAX ||
        cuFuncGetAttribute(
            &max_threads_per_block,
            CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            (CUfunction)function) != CUDA_SUCCESS ||
        threads_per_block > (size_t)max_threads_per_block) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_TEST_CUDA_ERROR_RUN_FAILED);
    }

    args[0] = &device_input;
    args[1] = &input_leading_dimension;
    args[2] = &device_constant_settings;
    args[3] = &constants_leading_dimension;
    args[4] = &num_settings;
    args[5] = &device_targets;
    args[6] = &targets_leading_dimension;
    args[7] = &num_rows;
    if (active_counts_launch_abi) {
        args[8] = &asts_per_kernel;
        args[9] = &num_targets;
        args[10] = &device_output;
        args[11] = &output_leading_dimension;
    } else {
        args[8] = &device_output;
        args[9] = &output_leading_dimension;
    }
    if (cuLaunchKernel(
            (CUfunction)function,
            (unsigned int)num_tiles, 1u, 1u,
            (unsigned int)threads_per_block, 1u, 1u,
            0u, stream, args, NULL) != CUDA_SUCCESS) {
        _SECANT_CUDA_COMMON_ERROR_RET(SECANT_TEST_CUDA_ERROR_RUN_FAILED);
    }
    return SECANT_CUDA_SUCCESS;
}

#undef _SECANT_CUDA_COMMON_ERROR_RET
