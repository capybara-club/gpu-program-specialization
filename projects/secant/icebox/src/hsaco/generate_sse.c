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
secant_hsaco_sse_source_generate(
    const SecantHsacoSSERecipe* recipe,
    char* hip_source,
    size_t hip_source_size,
    size_t* hip_source_size_ret
) {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
    size_t marker_count;
    size_t num_pairs;
    size_t num_partials;
    size_t num_results;
    size_t kernel_idx;
    size_t offset = 0u;

    if (recipe == NULL || hip_source_size_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    num_kernels = recipe->num_kernels;
    asts_per_kernel = recipe->asts_per_kernel;
    num_inputs = recipe->num_inputs;
    num_targets = recipe->num_targets;
    tile_rows = recipe->tile_rows;
    threads_per_block = recipe->threads_per_block;
    patch_capacity_instructions =
        recipe->patch_capacity_instructions;
    *hip_source_size_ret = 0u;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_validate_generate_args(
        _SECANT_HSACO_SHAPE_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        patch_capacity_instructions,
        &marker_count));
    if ((recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
        tile_rows == 0u || threads_per_block < 64u ||
        threads_per_block > tile_rows ||
        threads_per_block > _SECANT_HSACO_MAX_WAVES * 64u ||
        threads_per_block % 64u != 0u ||
        !_secant_hsaco_checked_mul(asts_per_kernel, num_targets, &num_pairs) ||
        !_secant_hsaco_checked_mul(
            num_pairs,
            _SECANT_HSACO_MAX_WAVES,
            &num_partials) ||
        !_secant_hsaco_checked_mul(
            num_kernels,
            num_pairs,
            &num_results)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
    }
    (void)marker_count;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        hip_source,
        hip_source_size,
        &offset,
        "#include <stddef.h>\n"
        "#include <stdint.h>\n\n"
        "#include <hip/hip_runtime.h>\n\n"));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        size_t ast_idx;
        size_t target_idx;
        size_t input_idx;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_hsaco_sse_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t input_leading_dimension,\n"
            "    const float* __restrict__ targets,\n"
            "    size_t targets_leading_dimension,\n"
            "    size_t num_rows,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    __shared__ float partial_sse[%zu];\n"
            "    const unsigned int lane = threadIdx.x %% warpSize;\n"
            "    const unsigned int wave = threadIdx.x / warpSize;\n"
            "    const unsigned int num_waves = blockDim.x / warpSize;\n"
            "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
            "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
            "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n",
            kernel_idx,
            num_partials,
            tile_rows,
            tile_rows,
            tile_rows));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                    hip_source,
                    hip_source_size,
                    &offset,
                    "    float sse_%03zu_%03zu = 0.0f;\n",
                    ast_idx,
                    target_idx));
            }
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"
            "    #pragma unroll 1\n"
            "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
            "        const size_t row = tile_begin + tile_row;\n"));
        for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
                input_idx,
                input_idx));
        }
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                hip_source,
                hip_source_size,
                &offset,
                "        const float target%zu = targets[(size_t)%zu * targets_leading_dimension + row];\n",
                target_idx,
                target_idx));
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"
            "        {\n"));
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_skeleton(
            _SECANT_HSACO_SHAPE_SSE,
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            num_targets,
            patch_capacity_instructions,
            hip_source,
            hip_source_size,
            &offset));
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "        }\n"
            "    }\n\n"));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                const size_t pair_idx =
                    ast_idx * num_targets + target_idx;

                _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                    hip_source,
                    hip_source_size,
                    &offset,
                    "    for (unsigned int delta = warpSize / 2u; delta != 0u; delta >>= 1u) {\n"
                    "        sse_%03zu_%03zu += __shfl_down(sse_%03zu_%03zu, delta, warpSize);\n"
                    "    }\n"
                    "    if (lane == 0u) {\n"
                    "        partial_sse[(size_t)%zu * %uu + wave] = sse_%03zu_%03zu;\n"
                    "    }\n",
                    ast_idx,
                    target_idx,
                    ast_idx,
                    target_idx,
                    pair_idx,
                    _SECANT_HSACO_MAX_WAVES,
                    ast_idx,
                    target_idx));
            }
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "\n"
            "    __syncthreads();\n"
            "    if (wave == 0u) {\n"));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                const size_t pair_idx =
                    ast_idx * num_targets + target_idx;

                _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                    hip_source,
                    hip_source_size,
                    &offset,
                    "        float block_sse_%03zu_%03zu = lane < num_waves ? partial_sse[(size_t)%zu * %uu + lane] : 0.0f;\n"
                    "        for (unsigned int delta = warpSize / 2u; delta != 0u; delta >>= 1u) {\n"
                    "            block_sse_%03zu_%03zu += __shfl_down(block_sse_%03zu_%03zu, delta, warpSize);\n"
                    "        }\n"
                    "        if (lane == 0u) {\n",
                    ast_idx,
                    target_idx,
                    pair_idx,
                    _SECANT_HSACO_MAX_WAVES,
                    ast_idx,
                    target_idx,
                    ast_idx,
                    target_idx));
                if (recipe->reduction_mode ==
                    SECANT_SSE_REDUCTION_MODE_ATOMIC) {
                    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                        hip_source,
                        hip_source_size,
                        &offset,
                        "            atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + %zuu, block_sse_%03zu_%03zu);\n",
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx));
                } else {
                    const size_t global_pair =
                        kernel_idx * num_pairs + pair_idx;

                    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                        hip_source,
                        hip_source_size,
                        &offset,
                        "            const size_t num_tiles = (num_rows + %zuu - 1u) / %zuu;\n"
                        "            output_sse[(size_t)%zu * num_tiles + blockIdx.x] = block_sse_%03zu_%03zu;\n",
                        tile_rows,
                        tile_rows,
                        global_pair,
                        ast_idx,
                        target_idx));
                }
                _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                    hip_source,
                    hip_source_size,
                    &offset,
                    "        }\n"));
            }
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "    }\n}\n\n"));
    }
    if (recipe->reduction_mode ==
        SECANT_SSE_REDUCTION_MODE_ATOMIC) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_hsaco_sse_compact(\n"
            "    const float* __restrict__ padded_sse,\n"
            "    size_t padded_leading_dimension,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    const size_t result = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (result < %zuu) {\n"
            "        const size_t series = result / %zuu;\n"
            "        const size_t target = result %% %zuu;\n"
            "        output_sse[series * output_leading_dimension + target] = padded_sse[series * padded_leading_dimension + target];\n"
            "    }\n"
            "}\n",
            num_results,
            num_targets,
            num_targets));
    } else {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            hip_source,
            hip_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_hsaco_sse_reduce(\n"
            "    const float* __restrict__ workspace,\n"
            "    size_t num_tiles,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    const size_t result = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (result < %zuu) {\n"
            "        float sum = 0.0f;\n"
            "        for (size_t tile = 0u; tile < num_tiles; ++tile) {\n"
            "            sum += workspace[result * num_tiles + tile];\n"
            "        }\n"
            "        const size_t ast = result / %zuu;\n"
            "        const size_t target = result %% %zuu;\n"
            "        output_sse[ast * output_leading_dimension + target] = sum;\n"
            "    }\n"
            "}\n",
            num_results,
            num_targets,
            num_targets));
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_write_finish(
        hip_source,
        hip_source_size,
        offset,
        hip_source_size_ret));
}
