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
#ifndef SECANT_TEST_HIP_RUNTIME_H_INCLUDED
#define SECANT_TEST_HIP_RUNTIME_H_INCLUDED

#include "secant_hip.h"

#include <stddef.h>

#define SECANT_TEST_HIP_ERROR_MODULE_LOAD_FAILED \
    ((SecantHIPResult)SECANT_HIP_RESULT_NUM_ENUMS)
#define SECANT_TEST_HIP_ERROR_RUN_FAILED \
    ((SecantHIPResult)(SECANT_HIP_RESULT_NUM_ENUMS + 1))

const char* secant_test_hip_result_to_string(SecantHIPResult result);

SecantHIPResult secant_test_hip_module_load(
    const void* hsaco,
    size_t hsaco_size,
    const char* function_name_prefix,
    size_t num_functions,
    void** module_ret,
    void** functions
);

SecantHIPResult secant_test_hip_run_static_column_materialize(
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
);

SecantHIPResult secant_test_hip_run_static_column_sse(
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
);

SecantHIPResult secant_test_hip_run_dynamic_constant_sse(
    void* function,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t asts_per_kernel,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
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
);

#endif
