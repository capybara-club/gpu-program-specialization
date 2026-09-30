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
#include "generate_internal.h"

SecantResult
secant_hsaco_materialize_source_generate(
    const SecantHsacoMaterializeRecipe* recipe,
    char* hip_source,
    size_t hip_source_size,
    size_t* hip_source_size_ret
) {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t patch_capacity_instructions;
    size_t marker_count;
    size_t kernel_idx;
    size_t offset = 0u;

    if (recipe == NULL || hip_source_size_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    num_kernels = recipe->num_kernels;
    asts_per_kernel = recipe->asts_per_kernel;
    num_inputs = recipe->num_inputs;
    patch_capacity_instructions =
        recipe->patch_capacity_instructions;
    *hip_source_size_ret = 0u;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_validate_generate_args(
        _SECANT_HSACO_SHAPE_MATERIALIZE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        patch_capacity_instructions,
        &marker_count));
    (void)marker_count;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        hip_source,
        hip_source_size,
        &offset,
        "#include <stddef.h>\n"
        "#include <stdint.h>\n\n"
        "#include <hip/hip_runtime.h>\n\n"));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        size_t input_idx;
        size_t ast_idx;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_hsaco_materialize_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t input_leading_dimension,\n"
            "    size_t num_rows,\n"
            "    float* __restrict__ output,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (row >= num_rows) {\n"
            "        return;\n"
            "    }\n\n",
            kernel_idx));
        for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "    const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
                input_idx,
                input_idx));
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"
            "    {\n"));
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_skeleton(
            _SECANT_HSACO_SHAPE_MATERIALIZE,
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            0u,
            patch_capacity_instructions,
            hip_source,
            hip_source_size,
            &offset));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        output[(size_t)%zu * output_leading_dimension + row] = result%zu;\n",
                ast_idx,
                ast_idx));
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            kernel_idx + 1u == num_kernels
                ? "    }\n}\n"
                : "    }\n}\n\n"));
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_write_finish(
        hip_source,
        hip_source_size,
        offset,
        hip_source_size_ret));
}
