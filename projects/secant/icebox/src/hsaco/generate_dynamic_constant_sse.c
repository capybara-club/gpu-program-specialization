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

#define _SECANT_HSACO_DYNAMIC_MAX_INPUT_COLUMNS 32u
#define _SECANT_HSACO_DYNAMIC_MAX_INPUT_CONSTANTS 32u

SecantResult
secant_hsaco_dynamic_constant_sse_source_generate(
    const SecantHsacoDynamicConstantSSERecipe* recipe,
    char* hip_source,
    size_t hip_source_size,
    size_t* hip_source_size_ret
) {
    size_t num_inputs;
    size_t marker_count;
    size_t row_tile_elements;
    size_t kernel_idx;
    size_t offset = 0u;

    if (recipe == NULL || hip_source_size_ret == NULL ||
        recipe->num_input_columns == 0u ||
        recipe->num_input_columns >
            _SECANT_HSACO_DYNAMIC_MAX_INPUT_COLUMNS ||
        recipe->num_input_constants == 0u ||
        recipe->num_input_constants >
            _SECANT_HSACO_DYNAMIC_MAX_INPUT_CONSTANTS ||
        recipe->num_targets == 0u ||
        recipe->tile_rows == 0u ||
        recipe->threads_per_block == 0u ||
        recipe->threads_per_block > 1024u ||
        (recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
        recipe->num_input_columns >
            SIZE_MAX - recipe->num_input_constants) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    num_inputs =
        recipe->num_input_columns + recipe->num_input_constants;
    *hip_source_size_ret = 0u;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_validate_generate_args(
        _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        num_inputs,
        recipe->num_targets,
        recipe->patch_capacity_instructions,
        &marker_count));
    if (recipe->num_input_columns >
            SIZE_MAX - recipe->num_targets ||
        !_secant_hsaco_checked_mul(
            recipe->num_input_columns + recipe->num_targets,
            recipe->tile_rows,
            &row_tile_elements)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    (void)marker_count;

    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        hip_source,
        hip_source_size,
        &offset,
        "#include <stddef.h>\n"
        "#include <stdint.h>\n\n"
        "#include <hip/hip_runtime.h>\n\n"));
    for (kernel_idx = 0u;
         kernel_idx < recipe->num_kernels;
         ++kernel_idx) {
        size_t ast_idx;
        size_t constant_idx;
        size_t input_idx;
        size_t target_idx;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_hsaco_dynamic_constant_sse_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t input_leading_dimension,\n"
            "    const float* __restrict__ constant_settings,\n"
            "    size_t constants_leading_dimension,\n"
            "    size_t num_settings,\n"
            "    const float* __restrict__ targets,\n"
            "    size_t targets_leading_dimension,\n"
            "    size_t num_rows,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    __shared__ float row_tile[%zu];\n"
            "    float* const input_tile = row_tile;\n"
            "    float* const target_tile = row_tile + %zu;\n"
            "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
            "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
            "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n"
            "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
            "        const size_t row = tile_begin + tile_row;\n",
            kernel_idx,
            row_tile_elements,
            recipe->num_input_columns * recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows));
        for (input_idx = 0u;
             input_idx < recipe->num_input_columns;
             ++input_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        input_tile[(size_t)%zu * %zuu + tile_row] = input[(size_t)%zu * input_leading_dimension + row];\n",
                input_idx,
                recipe->tile_rows,
                input_idx));
        }
        for (target_idx = 0u;
             target_idx < recipe->num_targets;
             ++target_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        target_tile[(size_t)%zu * %zuu + tile_row] = targets[(size_t)%zu * targets_leading_dimension + row];\n",
                target_idx,
                recipe->tile_rows,
                target_idx));
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "    }\n\n"
            "    __syncthreads();\n\n"
            "    for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
        for (constant_idx = 0u;
             constant_idx < recipe->num_input_constants;
             ++constant_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        const float input%zu = constant_settings[(size_t)%zu * constants_leading_dimension + setting];\n",
                recipe->num_input_columns + constant_idx,
                constant_idx));
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"));
        for (ast_idx = 0u;
             ast_idx < recipe->asts_per_kernel;
             ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < recipe->num_targets;
                 ++target_idx) {
                _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                    hip_source,
                    hip_source_size,
                    &offset,
                    "        float sse_%03zu_%03zu = 0.0f;\n",
                    ast_idx,
                    target_idx));
            }
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"
            "        #pragma unroll 1\n"
            "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
        for (input_idx = 0u;
             input_idx < recipe->num_input_columns;
             ++input_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
                input_idx,
                input_idx,
                recipe->tile_rows));
        }
        for (target_idx = 0u;
             target_idx < recipe->num_targets;
             ++target_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "            const float target%zu = target_tile[(size_t)%zu * %zuu + eval_row];\n",
                target_idx,
                target_idx,
                recipe->tile_rows));
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"
            "            {\n"));
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_skeleton(
            _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE,
            kernel_idx,
            recipe->asts_per_kernel,
            num_inputs,
            recipe->num_targets,
            recipe->patch_capacity_instructions,
            hip_source,
            hip_source_size,
            &offset));
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "            }\n"
            "        }\n\n"));
        if (recipe->reduction_mode ==
            SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        const size_t workspace_num_tiles = (num_rows + %zuu - 1u) / %zuu;\n",
                recipe->tile_rows,
                recipe->tile_rows));
        }
        for (ast_idx = 0u;
             ast_idx < recipe->asts_per_kernel;
             ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < recipe->num_targets;
                 ++target_idx) {
                const size_t pair_idx =
                    ast_idx * recipe->num_targets + target_idx;

                if (recipe->reduction_mode ==
                    SECANT_SSE_REDUCTION_MODE_ATOMIC) {
                    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                        hip_source,
                        hip_source_size,
                        &offset,
                        "        atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + setting, sse_%03zu_%03zu);\n",
                        pair_idx,
                        ast_idx,
                        target_idx));
                } else {
                    const size_t global_pair =
                        kernel_idx *
                            recipe->asts_per_kernel *
                            recipe->num_targets +
                        pair_idx;

                    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                        hip_source,
                        hip_source_size,
                        &offset,
                        "        output_sse[((size_t)%zu * num_settings + setting) * workspace_num_tiles + blockIdx.x] = sse_%03zu_%03zu;\n",
                        global_pair,
                        ast_idx,
                        target_idx));
                }
            }
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            kernel_idx + 1u == recipe->num_kernels &&
                recipe->reduction_mode ==
                    SECANT_SSE_REDUCTION_MODE_ATOMIC
                ? "    }\n}\n"
                : "    }\n}\n\n"));
    }
    if (recipe->reduction_mode ==
        SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_hsaco_dynamic_constant_sse_reduce(\n"
            "    const float* __restrict__ workspace,\n"
            "    size_t num_tiles,\n"
            "    size_t num_settings,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    const size_t result = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    const size_t num_results = %zuu * num_settings;\n"
            "    if (result < num_results) {\n"
            "        float sum = 0.0f;\n"
            "        for (size_t tile = 0u; tile < num_tiles; ++tile) {\n"
            "            sum += workspace[result * num_tiles + tile];\n"
            "        }\n"
            "        const size_t pair = result / num_settings;\n"
            "        const size_t setting = result %% num_settings;\n"
            "        output_sse[pair * output_leading_dimension + setting] = sum;\n"
            "    }\n"
            "}\n",
            recipe->num_kernels *
                recipe->asts_per_kernel *
                recipe->num_targets));
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_write_finish(
        hip_source,
        hip_source_size,
        offset,
        hip_source_size_ret));
}

#undef _SECANT_HSACO_DYNAMIC_MAX_INPUT_CONSTANTS
#undef _SECANT_HSACO_DYNAMIC_MAX_INPUT_COLUMNS
