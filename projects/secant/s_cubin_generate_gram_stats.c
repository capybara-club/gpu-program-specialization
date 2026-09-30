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

#define _SECANT_CUBIN_GRAM_STATS_MAX_FEATURES 32u
#define _SECANT_CUBIN_GRAM_STATS_MAX_SHARED_BYTES (48u * 1024u)
#define _SECANT_CUBIN_GRAM_STATS_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_CUBIN_GRAM_STATS_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_gram_stats_result = (ans); \
    return _secant_cubin_gram_stats_result; \
} while (0)
#define _SECANT_CUBIN_GRAM_STATS_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_gram_stats_check_result = (ans); \
    if (_secant_cubin_gram_stats_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_GRAM_STATS_ERROR_RET(_secant_cubin_gram_stats_check_result); \
    } \
} while (0)

static int
_secant_cubin_gram_stats_checked_add(
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
_secant_cubin_gram_stats_checked_mul(
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
_secant_cubin_gram_stats_write(
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
        _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    va_copy(measure_args, args);
    required = vsnprintf(NULL, 0u, format, measure_args);
    va_end(measure_args);
    if (required < 0) {
        va_end(args);
        _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        va_end(args);
        _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    if (buffer != NULL) {
        if (*offset > buffer_size || (size_t)required >= buffer_size - *offset) {
            va_end(args);
            _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        if (vsnprintf(buffer + *offset, buffer_size - *offset, format, args) != required) {
            va_end(args);
            _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_FORMAT);
        }
    }
    va_end(args);
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_gram_stats_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_gram_stats_reduction(
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
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_reduction(
        first_operand,
        left_operands,
        buffer,
        buffer_size,
        offset,
        &left));
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_reduction(
        first_operand + left_operands,
        num_operands - left_operands,
        buffer,
        buffer_size,
        offset,
        &right));
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        left,
        left,
        right));
    *result_operand_ret = left;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_gram_stats_skeleton(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t patch_capacity_instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t marker_count = num_inputs + asts_per_kernel;
    const uint32_t marker = _SECANT_CUBIN_GRAM_STATS_FIRST_MARKER_BITS +
        (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_inputs + asts_per_kernel;
    const size_t input_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t input_idx;
    size_t ast_idx;

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "        float marked%zu __attribute__((unused));\n",
            input_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "        float result%zu;\n",
            ast_idx));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "        float keepalive __attribute__((unused));\n"
        "        asm volatile(\n"
        "            \"{\\n\\t\"\n"
        "            \".reg .u32 keepalive_address;\\n\\t\"\n"
        "            \"brkpt;\\n\\t\"\n"));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "            \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            input_idx,
            input_operand_offset + input_idx,
            marker + (uint32_t)input_idx));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "            \"brkpt;\\n\\t\"\n"));
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_reduction(
        0u,
        num_inputs,
        buffer,
        buffer_size,
        offset,
        &reduction_operand));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "            \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            num_inputs + ast_idx,
            reduction_operand,
            marker + (uint32_t)(num_inputs + ast_idx)));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "            \"brkpt;\\n\\t\"\n"));
    if (asts_per_kernel == 1u) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            num_inputs,
            input_operand_offset));
        input_idx = 1u;
    } else {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            num_inputs,
            num_inputs + 1u));
        for (ast_idx = 2u; ast_idx < asts_per_kernel; ++ast_idx) {
            _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
                keepalive_operand,
                keepalive_operand,
                num_inputs + ast_idx));
        }
        input_idx = 0u;
    }
    for (; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            keepalive_operand,
            input_operand_offset + input_idx));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "            \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
        "            \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
        keepalive_operand));
    for (input_idx = 0u; input_idx < patch_capacity_instructions; ++input_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "            \"brkpt;\\n\\t\"\n"));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "            \"}\\n\\t\"\n"
        "            : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "%s\"=&f\"(marked%zu)",
            input_idx == 0u ? "" : ", ",
            input_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            ", \"=&f\"(result%zu)",
            ast_idx));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        ", \"=&f\"(keepalive)\n"
        "            : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            buffer,
            buffer_size,
            offset,
            "%s\"f\"(input%zu)",
            input_idx == 0u ? "" : ", ",
            input_idx));
    }
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "            : \"memory\");\n"));
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_gram_stats_recipe_validate(
    const SecantCubinGramStatsRecipe* recipe
) {
    size_t marker_count;
    size_t total_markers;
    size_t shared_columns;
    size_t shared_elements;
    size_t shared_bytes;
    size_t gram_elements;
    size_t cross_elements;
    size_t statistics_elements;

    if (recipe == NULL || recipe->num_kernels == 0u || recipe->asts_per_kernel == 0u ||
        recipe->asts_per_kernel > _SECANT_CUBIN_GRAM_STATS_MAX_FEATURES || recipe->num_inputs == 0u ||
        recipe->num_inputs > SECANT_AST_MAX_INPUTS || recipe->num_targets == 0u || recipe->tile_rows == 0u ||
        recipe->threads_per_block < 32u || recipe->threads_per_block > 512u ||
        recipe->threads_per_block > recipe->tile_rows || recipe->threads_per_block % 32u != 0u ||
        recipe->patch_capacity_instructions == 0u ||
        !_secant_cubin_gram_stats_checked_add(recipe->num_inputs, recipe->asts_per_kernel, &marker_count) ||
        marker_count > _SECANT_CUBIN_MAX_REGISTERS ||
        !_secant_cubin_gram_stats_checked_mul(recipe->num_kernels, marker_count, &total_markers) ||
        total_markers - 1u > 0x7fffffffu - _SECANT_CUBIN_GRAM_STATS_FIRST_MARKER_BITS ||
        !_secant_cubin_gram_stats_checked_add(marker_count, 1u, &shared_columns) ||
        !_secant_cubin_gram_stats_checked_mul(shared_columns, recipe->tile_rows, &shared_elements) ||
        !_secant_cubin_gram_stats_checked_add(shared_elements, 1u, &shared_elements) ||
        !_secant_cubin_gram_stats_checked_mul(shared_elements, sizeof(float), &shared_bytes) ||
        shared_bytes > _SECANT_CUBIN_GRAM_STATS_MAX_SHARED_BYTES ||
        !_secant_cubin_gram_stats_checked_mul(recipe->asts_per_kernel, recipe->asts_per_kernel, &gram_elements) ||
        !_secant_cubin_gram_stats_checked_mul(recipe->asts_per_kernel, recipe->num_targets, &cross_elements) ||
        !_secant_cubin_gram_stats_checked_add(recipe->asts_per_kernel, gram_elements, &statistics_elements) ||
        !_secant_cubin_gram_stats_checked_add(statistics_elements, cross_elements, &statistics_elements)) {
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_gram_stats_source_generate(
    const SecantCubinGramStatsRecipe* recipe,
    char* cuda_source,
    size_t cuda_source_size,
    size_t* cuda_source_size_ret
) {
    size_t kernel_idx;
    size_t offset = 0u;

    if (recipe == NULL || cuda_source_size_ret == NULL) {
        _SECANT_CUBIN_GRAM_STATS_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *cuda_source_size_ret = 0u;
    _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_recipe_validate(recipe));

    for (kernel_idx = 0u; kernel_idx < recipe->num_kernels; ++kernel_idx) {
        size_t input_idx;
        size_t ast_idx;

        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            cuda_source,
            cuda_source_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_cubin_gram_stats_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t input_leading_dimension,\n"
            "    const float* __restrict__ targets,\n"
            "    size_t targets_leading_dimension,\n"
            "    size_t num_rows,\n"
            "    size_t num_asts,\n"
            "    size_t num_targets,\n"
            "    float* __restrict__ statistics\n"
            ") {\n"
            "    __shared__ float tile_storage[%zu];\n"
            "    float* const input_tile = tile_storage + 1u;\n"
            "    float* const feature_tile = input_tile + %zuu;\n"
            "    float* const target_tile = feature_tile + %zuu;\n"
            "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
            "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
            "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n"
            "    const size_t input_tile_elements = (size_t)%zu * tile_num_rows;\n\n"
            "    for (size_t linear = threadIdx.x; linear < input_tile_elements; linear += blockDim.x) {\n"
            "        const size_t column = linear / tile_num_rows;\n"
            "        const size_t tile_row = linear - column * tile_num_rows;\n"
            "        input_tile[column * %zuu + tile_row] =\n"
            "            input[column * input_leading_dimension + tile_begin + tile_row];\n"
            "    }\n"
            "    __syncthreads();\n\n"
            "    #pragma unroll 1\n"
            "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n",
            kernel_idx,
            1u + (recipe->num_inputs + recipe->asts_per_kernel + 1u) * recipe->tile_rows,
            recipe->num_inputs * recipe->tile_rows,
            recipe->asts_per_kernel * recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->num_inputs,
            recipe->tile_rows));
        for (input_idx = 0u; input_idx < recipe->num_inputs; ++input_idx) {
            _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "        const float input%zu = input_tile[(size_t)%zu * %zuu + tile_row];\n",
                input_idx,
                input_idx,
                recipe->tile_rows));
        }
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            cuda_source,
            cuda_source_size,
            &offset,
            "\n"
            "        {\n"));
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_skeleton(
            kernel_idx,
            recipe->asts_per_kernel,
            recipe->num_inputs,
            recipe->patch_capacity_instructions,
            cuda_source,
            cuda_source_size,
            &offset));
        for (ast_idx = 0u; ast_idx < recipe->asts_per_kernel; ++ast_idx) {
            _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "            if (%zuu < num_asts) {\n"
                "                feature_tile[(size_t)%zu * %zuu + tile_row] = result%zu;\n"
                "            }\n",
                ast_idx,
                ast_idx,
                recipe->tile_rows,
                ast_idx));
        }
        _SECANT_CUBIN_GRAM_STATS_CHECK_RET(_secant_cubin_gram_stats_write(
            cuda_source,
            cuda_source_size,
            &offset,
            "        }\n"
            "    }\n"
            "    __syncthreads();\n\n"
            "    for (size_t feature = threadIdx.x; feature < num_asts; feature += blockDim.x) {\n"
            "        float sum = 0.0f;\n"
            "        for (size_t tile_row = 0u; tile_row < tile_num_rows; ++tile_row) {\n"
            "            sum += feature_tile[feature * %zuu + tile_row];\n"
            "        }\n"
            "        atomicAdd(statistics + feature, sum);\n"
            "    }\n\n"
            "    const size_t gram_count = num_asts * num_asts;\n"
            "    for (size_t pair = threadIdx.x; pair < gram_count; pair += blockDim.x) {\n"
            "        const size_t lhs = pair / num_asts;\n"
            "        const size_t rhs = pair - lhs * num_asts;\n"
            "        float sum = 0.0f;\n"
            "        for (size_t tile_row = 0u; tile_row < tile_num_rows; ++tile_row) {\n"
            "            sum += feature_tile[lhs * %zuu + tile_row] * feature_tile[rhs * %zuu + tile_row];\n"
            "        }\n"
            "        atomicAdd(statistics + %zuu + lhs * %zuu + rhs, sum);\n"
            "    }\n\n"
            "    for (size_t target = 0u; target < num_targets; ++target) {\n"
            "        for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
            "            target_tile[tile_row] = targets[target * targets_leading_dimension + tile_begin + tile_row];\n"
            "        }\n"
            "        __syncthreads();\n"
            "        for (size_t feature = threadIdx.x; feature < num_asts; feature += blockDim.x) {\n"
            "            float sum = 0.0f;\n"
            "            for (size_t tile_row = 0u; tile_row < tile_num_rows; ++tile_row) {\n"
            "                sum += feature_tile[feature * %zuu + tile_row] * target_tile[tile_row];\n"
            "            }\n"
            "            atomicAdd(\n"
            "                statistics + %zuu + %zuu + feature * num_targets + target,\n"
            "                sum);\n"
            "        }\n"
            "        __syncthreads();\n"
            "    }\n"
            "%s",
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->asts_per_kernel,
            recipe->asts_per_kernel,
            recipe->tile_rows,
            recipe->asts_per_kernel,
            recipe->asts_per_kernel * recipe->asts_per_kernel,
            kernel_idx + 1u == recipe->num_kernels ? "}\n" : "}\n\n"));
    }
    _SECANT_CUBIN_GRAM_STATS_ERROR_RET(_secant_cubin_gram_stats_finish(
        cuda_source,
        cuda_source_size,
        offset,
        cuda_source_size_ret));
}

#undef _SECANT_CUBIN_GRAM_STATS_CHECK_RET
#undef _SECANT_CUBIN_GRAM_STATS_ERROR_RET
#undef _SECANT_CUBIN_GRAM_STATS_FIRST_MARKER_BITS
#undef _SECANT_CUBIN_GRAM_STATS_MAX_SHARED_BYTES
#undef _SECANT_CUBIN_GRAM_STATS_MAX_FEATURES
