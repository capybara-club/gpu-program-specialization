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
#include "s_cubin_internal.h"


#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#define _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS 16u
#define _SECANT_CUBIN_SSE_GENERATE_MAX_REGISTERS 256u
#define _SECANT_CUBIN_SSE_GENERATE_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_sse_generate_result = (ans); \
    return _secant_cubin_sse_generate_result; \
} while (0)
#define _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_sse_generate_check_result = (ans); \
    if (_secant_cubin_sse_generate_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(_secant_cubin_sse_generate_check_result); \
    } \
} while (0)

static int
_secant_cubin_sse_generate_checked_add(
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
_secant_cubin_sse_generate_checked_mul(
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

static SecantResult
_secant_cubin_sse_generate_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    va_list measure_args;
    int required;

    if (offset == NULL || format == NULL) {
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    va_copy(measure_args, args);
    required = vsnprintf(NULL, 0u, format, measure_args);
    va_end(measure_args);
    if (required < 0) {
        va_end(args);
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
            SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        va_end(args);
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
            SECANT_ERROR_OVERFLOW);
    }
    if (buffer != NULL) {
        if (*offset > buffer_size || (size_t)required >= buffer_size - *offset) {
            va_end(args);
            _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        if (vsnprintf(
                buffer + *offset,
                buffer_size - *offset,
                format,
                args) != required) {
            va_end(args);
            _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
                SECANT_ERROR_FORMAT);
        }
    }
    va_end(args);
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_sse_generate_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_sse_generate_reduction(
    size_t first_operand,
    size_t num_operands,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    size_t* result_operand_ret
) {
    size_t left_operands;
    size_t left;
    size_t right;

    if (num_operands == 1u) {
        *result_operand_ret = first_operand;
        return SECANT_SUCCESS;
    }
    left_operands = num_operands / 2u;
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_reduction(
            first_operand,
            left_operands,
            buffer,
            buffer_size,
            offset,
            &left));
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_reduction(
            first_operand + left_operands,
            num_operands - left_operands,
            buffer,
            buffer_size,
            offset,
            &right));
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "        \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            left,
            left,
            right));
    *result_operand_ret = left;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_sse_generate_skeleton(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t num_sources = num_inputs + num_targets;
    const size_t num_outputs = asts_per_kernel * num_targets;
    const size_t marker_count = num_sources + num_outputs;
    const uint32_t marker = _SECANT_CUBIN_SSE_GENERATE_FIRST_MARKER_BITS + (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_sources + num_outputs;
    const size_t source_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t source_idx;
    size_t ast_idx;
    size_t target_idx;

    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                buffer,
                buffer_size,
                offset,
                "        float marked%zu __attribute__((unused));\n",
                source_idx));
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "        float keepalive __attribute__((unused));\n"
            "        asm volatile(\n"
            "            \"{\\n\\t\"\n"
            "            \".reg .u32 keepalive_address;\\n\\t\"\n"
            "            \"brkpt;\\n\\t\"\n"));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
                source_idx,
                source_operand_offset + source_idx,
                marker + (uint32_t)source_idx));
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"brkpt;\\n\\t\"\n"));
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_reduction(
            0u,
            num_sources,
            buffer,
            buffer_size,
            offset,
            &reduction_operand));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t output_idx = ast_idx * num_targets + target_idx;
            const size_t output_operand = num_sources + output_idx;

            _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                _secant_cubin_sse_generate_write(
                    buffer,
                    buffer_size,
                    offset,
                    "            \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
                    output_operand,
                    output_operand,
                    marker +
                        (uint32_t)(num_sources + output_idx)));
        }
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"brkpt;\\n\\t\"\n"
            "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            reduction_operand,
            source_operand_offset));
    for (source_idx = 1u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
                keepalive_operand,
                keepalive_operand,
                source_operand_offset + source_idx));
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
            "            \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
            keepalive_operand));
    for (source_idx = 0u;
         source_idx < patch_capacity_instructions;
         ++source_idx) {
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"brkpt;\\n\\t\"\n"));
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"}\\n\\t\"\n"
            "            : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                buffer,
                buffer_size,
                offset,
                "%s\"=&f\"(marked%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                _secant_cubin_sse_generate_write(
                    buffer,
                    buffer_size,
                    offset,
                    ", \"+f\"(sse_%03zu_%03zu)",
                    ast_idx,
                    target_idx));
        }
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            ", \"=&f\"(keepalive)\n"
            "            : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        if (source_idx < num_inputs) {
            _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                _secant_cubin_sse_generate_write(
                    buffer,
                    buffer_size,
                    offset,
                    "%s\"f\"(input%zu)",
                    source_idx == 0u ? "" : ", ",
                    source_idx));
        } else {
            _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                _secant_cubin_sse_generate_write(
                    buffer,
                    buffer_size,
                    offset,
                    "%s\"f\"(target%zu)",
                    source_idx == 0u ? "" : ", ",
                    source_idx - num_inputs));
        }
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_write(
            buffer,
            buffer_size,
            offset,
            "\n"
            "            : \"memory\");\n"));
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_sse_generate_validate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    size_t* num_pairs_ret
) {
    size_t num_pairs;
    size_t marker_count;
    size_t total_markers;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        asts_per_kernel > _SECANT_CUBIN_SSE_GENERATE_MAX_REGISTERS ||
        num_inputs == 0u ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        num_targets == 0u || patch_capacity_instructions == 0u ||
        !_secant_cubin_sse_generate_checked_mul(
            asts_per_kernel,
            num_targets,
            &num_pairs) ||
        !_secant_cubin_sse_generate_checked_add(
            num_inputs,
            num_targets,
            &marker_count) ||
        !_secant_cubin_sse_generate_checked_add(
            marker_count,
            num_pairs,
            &marker_count) ||
        marker_count > _SECANT_CUBIN_SSE_GENERATE_MAX_REGISTERS ||
        !_secant_cubin_sse_generate_checked_mul(
            num_kernels,
            marker_count,
            &total_markers) ||
        total_markers - 1u >
            0x7fffffffu -
                _SECANT_CUBIN_SSE_GENERATE_FIRST_MARKER_BITS) {
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    *num_pairs_ret = num_pairs;
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_sse_recipe_validate(
    const SecantCubinSSERecipe* recipe
) {
    size_t num_pairs;
    size_t num_partials;

    if (recipe == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
        _secant_cubin_sse_generate_validate(
            recipe->num_kernels,
            recipe->asts_per_kernel,
            recipe->num_inputs,
            recipe->num_targets,
            recipe->patch_capacity_instructions,
            &num_pairs));
    if (recipe->tile_rows == 0u ||
        recipe->threads_per_block < 32u ||
        recipe->threads_per_block > recipe->tile_rows ||
        recipe->threads_per_block > _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS * 32u ||
        recipe->threads_per_block % 32u != 0u ||
        !_secant_cubin_sse_generate_checked_mul(
            num_pairs,
            _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS,
            &num_partials)) {
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_sse_source_generate(
    const SecantCubinSSERecipe* recipe,
    char* cuda_source,
    size_t cuda_source_size,
    size_t* cuda_source_size_ret
) {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
    size_t num_pairs;
    size_t num_partials;
    size_t num_warps;
    size_t kernel_idx;
    size_t offset = 0u;

    if (recipe == NULL || cuda_source_size_ret == NULL) {
        _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    num_kernels = recipe->num_kernels;
    asts_per_kernel = recipe->asts_per_kernel;
    num_inputs = recipe->num_inputs;
    num_targets = recipe->num_targets;
    tile_rows = recipe->tile_rows;
    threads_per_block = recipe->threads_per_block;
    patch_capacity_instructions = recipe->patch_capacity_instructions;
    *cuda_source_size_ret = 0u;
    _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(_secant_cubin_sse_recipe_validate(recipe));
    num_pairs = asts_per_kernel * num_targets;
    num_partials = num_pairs * _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS;
    num_warps = threads_per_block / 32u;

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        size_t ast_idx;
        size_t target_idx;
        size_t input_idx;

        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "extern \"C\" __global__\n"
                "void secant_cubin_sse_%03zu(\n"
                "    const float* __restrict__ input,\n"
                "    size_t input_leading_dimension,\n"
                "    const float* __restrict__ targets,\n"
                "    size_t targets_leading_dimension,\n"
                "    size_t num_rows,\n"
                "    size_t num_asts,\n"
                "    size_t num_targets,\n"
                "    float* __restrict__ output_sse,\n"
                "    size_t output_leading_dimension\n"
                ") {\n"
                "    __shared__ float partial_sse[%zu];\n"
                "    const unsigned int lane = threadIdx.x & 31u;\n"
                "    const unsigned int warp = threadIdx.x >> 5u;\n"
                "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
                "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
                "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n",
                kernel_idx,
                num_partials,
                tile_rows,
                tile_rows,
                tile_rows));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < num_targets;
                 ++target_idx) {
                _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                    _secant_cubin_sse_generate_write(
                        cuda_source,
                        cuda_source_size,
                        &offset,
                        "    float sse_%03zu_%03zu = 0.0f;\n",
                        ast_idx,
                        target_idx));
            }
        }
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "\n"
                "    #pragma unroll 1\n"
                "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
                "        const size_t row = tile_begin + tile_row;\n"));
        for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
            _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                _secant_cubin_sse_generate_write(
                    cuda_source,
                    cuda_source_size,
                    &offset,
                    "        const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
                    input_idx,
                    input_idx));
        }
        for (target_idx = 0u;
             target_idx < num_targets;
             ++target_idx) {
            _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                _secant_cubin_sse_generate_write(
                    cuda_source,
                    cuda_source_size,
                    &offset,
                    "        const float target%zu = targets[(size_t)(%zuu < num_targets ? %zuu : 0u) * "
                    "targets_leading_dimension + row];\n",
                    target_idx,
                    target_idx,
                    target_idx));
        }
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "\n"
                "        {\n"));
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_skeleton(
                kernel_idx,
                asts_per_kernel,
                num_inputs,
                num_targets,
                patch_capacity_instructions,
                cuda_source,
                cuda_source_size,
                &offset));
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "        }\n"
                "    }\n\n"));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < num_targets;
                 ++target_idx) {
                const size_t pair_idx = ast_idx * num_targets + target_idx;

                _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                    _secant_cubin_sse_generate_write(
                        cuda_source,
                        cuda_source_size,
                        &offset,
                        "    if (%zuu < num_asts && %zuu < num_targets) {\n"
                        "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                        "            sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, sse_%03zu_%03zu, delta);\n"
                        "        }\n"
                        "        if (lane == 0u) {\n"
                        "            partial_sse[(size_t)%zu * %uu + warp] = sse_%03zu_%03zu;\n"
                        "        }\n"
                        "    }\n",
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx,
                        pair_idx,
                        _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS,
                        ast_idx,
                        target_idx));
            }
        }
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "\n"
                "    __syncthreads();\n"
                "    if (warp == 0u) {\n"));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < num_targets;
                 ++target_idx) {
                const size_t pair_idx = ast_idx * num_targets + target_idx;

                _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                    _secant_cubin_sse_generate_write(
                        cuda_source,
                        cuda_source_size,
                        &offset,
                        "        if (%zuu < num_asts && %zuu < num_targets) {\n"
                        "            float block_sse_%03zu_%03zu = lane < %zuu ? "
                        "partial_sse[(size_t)%zu * %uu + lane] : 0.0f;\n"
                        "            for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                        "                block_sse_%03zu_%03zu += "
                        "__shfl_down_sync(0xffffffffu, block_sse_%03zu_%03zu, delta);\n"
                        "            }\n"
                        "            if (lane == 0u) {\n",
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx,
                        num_warps,
                        pair_idx,
                        _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS,
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx));
                _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                    _secant_cubin_sse_generate_write(
                        cuda_source,
                        cuda_source_size,
                        &offset,
                        "                atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + %zuu, "
                        "block_sse_%03zu_%03zu);\n",
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx));
                _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
                    _secant_cubin_sse_generate_write(
                        cuda_source,
                        cuda_source_size,
                        &offset,
                        "            }\n"
                        "        }\n"));
            }
        }
        _SECANT_CUBIN_SSE_GENERATE_CHECK_RET(
            _secant_cubin_sse_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                kernel_idx + 1u == num_kernels
                    ? "    }\n}\n"
                    : "    }\n}\n\n"));
    }
    _SECANT_CUBIN_SSE_GENERATE_ERROR_RET(
        _secant_cubin_sse_generate_finish(
            cuda_source,
            cuda_source_size,
            offset,
            cuda_source_size_ret));
}

#undef _SECANT_CUBIN_SSE_GENERATE_CHECK_RET
#undef _SECANT_CUBIN_SSE_GENERATE_ERROR_RET
#undef _SECANT_CUBIN_SSE_GENERATE_FIRST_MARKER_BITS
#undef _SECANT_CUBIN_SSE_GENERATE_MAX_REGISTERS
#undef _SECANT_CUBIN_SSE_GENERATE_MAX_WARPS
