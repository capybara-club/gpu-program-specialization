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

#define _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_INPUT_COLUMNS 32u
#define _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_INPUT_CONSTANTS 32u
#define _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_REGISTERS 256u
#define _SECANT_CUBIN_CONSTANT_OPTIMIZER_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_CUBIN_CONSTANT_OPTIMIZER_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_constant_optimizer_result = (ans); \
    return _secant_cubin_constant_optimizer_result; \
} while (0)
#define _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_constant_optimizer_check_result = (ans); \
    if (_secant_cubin_constant_optimizer_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_ERROR_RET(_secant_cubin_constant_optimizer_check_result); \
    } \
} while (0)

static SecantResult
_secant_cubin_constant_optimizer_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        required = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        required = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (required < 0) {
        return SECANT_ERROR_FORMAT;
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        return SECANT_ERROR_OVERFLOW;
    }
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_constant_optimizer_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            return SECANT_ERROR_INSUFFICIENT_BUFFER;
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_constant_optimizer_reduction(
    size_t first_operand,
    size_t num_operands,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    size_t* result_operand_ret
) {
    size_t left_count;
    size_t left;
    size_t right;

    if (num_operands == 1u) {
        *result_operand_ret = first_operand;
        return SECANT_SUCCESS;
    }
    left_count = num_operands / 2u;
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_reduction(
        first_operand, left_count, buffer, buffer_size, offset, &left));
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_reduction(
        first_operand + left_count, num_operands - left_count, buffer, buffer_size, offset, &right));
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset,
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        left, left, right));
    *result_operand_ret = left;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_constant_optimizer_skeleton(
    size_t kernel_idx,
    size_t num_inputs,
    size_t patch_capacity_instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t num_sources = num_inputs + 1u;
    const size_t marker_count = num_sources + 1u;
    const uint32_t marker = _SECANT_CUBIN_CONSTANT_OPTIMIZER_FIRST_MARKER_BITS +
        (uint32_t)(kernel_idx * marker_count);
    const size_t output_operand = num_sources;
    const size_t keepalive_operand = output_operand + 1u;
    const size_t source_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t source_idx;

    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            buffer, buffer_size, offset,
            "            float marked%zu __attribute__((unused));\n", source_idx));
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset,
        "            float keepalive __attribute__((unused));\n"
        "            asm volatile(\n"
        "                \"{\\n\\t\"\n"
        "                \".reg .u32 keepalive_address;\\n\\t\"\n"
        "                \"brkpt;\\n\\t\"\n"));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            buffer, buffer_size, offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            source_idx, source_operand_offset + source_idx, marker + (uint32_t)source_idx));
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset, "                \"brkpt;\\n\\t\"\n"));
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_reduction(
        0u, num_sources, buffer, buffer_size, offset, &reduction_operand));
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset,
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n"
        "                \"brkpt;\\n\\t\"\n"
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        output_operand, output_operand, marker + (uint32_t)num_sources,
        keepalive_operand, reduction_operand, source_operand_offset));
    for (source_idx = 1u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            buffer, buffer_size, offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand, keepalive_operand, source_operand_offset + source_idx));
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset,
        "                \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
        "                \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
        keepalive_operand));
    for (source_idx = 0u; source_idx < patch_capacity_instructions; ++source_idx) {
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            buffer, buffer_size, offset, "                \"brkpt;\\n\\t\"\n"));
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset, "                \"}\\n\\t\"\n                : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            buffer, buffer_size, offset, "%s\"=&f\"(marked%zu)", source_idx == 0u ? "" : ", ", source_idx));
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset, ", \"+f\"(sse), \"=&f\"(keepalive)\n                : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        if (source_idx < num_inputs) {
            _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
                buffer, buffer_size, offset, "%s\"f\"(input%zu)", source_idx == 0u ? "" : ", ", source_idx));
        } else {
            _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
                buffer, buffer_size, offset, "%s\"f\"(target0)", source_idx == 0u ? "" : ", "));
        }
    }
    return _secant_cubin_constant_optimizer_write(
        buffer, buffer_size, offset, "\n                : \"memory\");\n");
}

SecantResult
_secant_cubin_constant_optimizer_sse_recipe_validate(
    const SecantCubinConstantOptimizerSSERecipe* recipe
) {
    size_t num_inputs;
    size_t marker_count;
    size_t total_markers;
    size_t tile_elements;

    if (recipe == NULL || recipe->num_kernels == 0u || recipe->num_input_columns == 0u ||
        recipe->num_input_columns > _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_INPUT_COLUMNS ||
        recipe->num_input_constants == 0u ||
        recipe->num_input_constants > _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_INPUT_CONSTANTS ||
        (recipe->tile_rows != 128u && recipe->tile_rows != 256u) ||
        recipe->threads_per_block == 0u || recipe->threads_per_block > 1024u ||
        recipe->threads_per_block % 32u != 0u || recipe->threads_per_block > recipe->tile_rows ||
        recipe->patch_capacity_instructions == 0u ||
        !_secant_cubin_checked_add(recipe->num_input_columns, recipe->num_input_constants, &num_inputs) ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cubin_checked_add(num_inputs, 2u, &marker_count) ||
        marker_count > _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_REGISTERS ||
        !_secant_cubin_checked_mul(recipe->num_kernels, marker_count, &total_markers) ||
        total_markers - 1u > 0x7fffffffu - _SECANT_CUBIN_CONSTANT_OPTIMIZER_FIRST_MARKER_BITS ||
        !_secant_cubin_checked_mul(recipe->num_input_columns + 1u, recipe->tile_rows, &tile_elements)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_constant_optimizer_sse_source_generate(
    const SecantCubinConstantOptimizerSSERecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
    const char* reducer_source;
    size_t reducer_source_size;
    size_t num_inputs;
    size_t tile_elements;
    size_t kernel_idx;
    size_t offset = 0u;

    if (required_size_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(
        _secant_cubin_constant_optimizer_sse_recipe_validate(recipe));
    num_inputs = recipe->num_input_columns + recipe->num_input_constants;
    tile_elements = (recipe->num_input_columns + 1u) * recipe->tile_rows;
    reducer_source = secant_cuda_constant_optimizer_reduce_f32_source_get(&reducer_source_size);
    if (reducer_source == NULL || reducer_source_size < 2u ||
        reducer_source[reducer_source_size - 1u] != '\0') {
        return SECANT_ERROR_INVALID_STATE;
    }
    *required_size_ret = 0u;

    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        output, output_size, &offset,
        "static __forceinline__ __device__ uint4 secant_philox_round(uint4 c, uint2 k) {\n"
        "    const unsigned int hi0 = __umulhi(0xd2511f53u, c.x);\n"
        "    const unsigned int hi1 = __umulhi(0xcd9e8d57u, c.z);\n"
        "    const unsigned int lo0 = 0xd2511f53u * c.x;\n"
        "    const unsigned int lo1 = 0xcd9e8d57u * c.z;\n"
        "    return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_philox4x32_10(uint4 c, uint2 k) {\n"
        "    #pragma unroll\n"
        "    for (int round = 0; round < 10; ++round) {\n"
        "        c = secant_philox_round(c, k);\n"
        "        k.x += 0x9e3779b9u;\n"
        "        k.y += 0xbb67ae85u;\n"
        "    }\n"
        "    return c;\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_constant_random4(\n"
        "    unsigned long long setting, unsigned long long ast_index, unsigned long long seed,\n"
        "    unsigned long long generation, unsigned long long iteration, unsigned int group) {\n"
        "    const uint4 counter = make_uint4((unsigned int)setting, (unsigned int)(setting >> 32),\n"
        "        (unsigned int)ast_index, (unsigned int)(ast_index >> 32));\n"
        "    const uint2 key = make_uint2(\n"
        "        (unsigned int)seed ^ (unsigned int)generation * 0x9e3779b9u ^\n"
        "            (unsigned int)(iteration >> 32) * 0x85ebca6bu ^ group * 0x27d4eb2du,\n"
        "        (unsigned int)(seed >> 32) ^ (unsigned int)(generation >> 32) * 0xbb67ae85u ^\n"
        "            (unsigned int)iteration * 0xc2b2ae35u ^ group * 0x165667b1u);\n"
        "    return secant_philox4x32_10(counter, key);\n"
        "}\n\n"
        "static __forceinline__ __device__ float secant_constant_delta(unsigned int word) {\n"
        "    return (float)(word >> 8) * 1.1920928955078125e-7f - 1.0f;\n"
        "}\n\n"));

    for (kernel_idx = 0u; kernel_idx < recipe->num_kernels; ++kernel_idx) {
        size_t input_idx;
        size_t constant_idx;
        size_t group_idx;

        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            output, output_size, &offset,
            "extern \"C\" __global__\n"
            "void secant_cubin_constant_optimizer_sse_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t input_leading_dimension,\n"
            "    const float* __restrict__ current_constants,\n"
            "    size_t num_settings,\n"
            "    const float* __restrict__ target,\n"
            "    size_t num_rows,\n"
            "    unsigned long long ast_index,\n"
            "    unsigned long long seed,\n"
            "    unsigned long long generation,\n"
            "    unsigned long long iteration,\n"
            "    float perturbation_scale,\n"
            "    float* __restrict__ output_sse) {\n"
            "    __shared__ float row_tile[%zu];\n"
            "    float* const input_tile = row_tile;\n"
            "    float* const target_tile = row_tile + %zu;\n"
            "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
            "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
            "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n"
            "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
            "        const size_t row = tile_begin + tile_row;\n",
            kernel_idx, tile_elements, recipe->num_input_columns * recipe->tile_rows,
            recipe->tile_rows, recipe->tile_rows, recipe->tile_rows));
        for (input_idx = 0u; input_idx < recipe->num_input_columns; ++input_idx) {
            _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
                output, output_size, &offset,
                "        input_tile[(size_t)%zu * %zuu + tile_row] = input[(size_t)%zu * input_leading_dimension + row];\n",
                input_idx, recipe->tile_rows, input_idx));
        }
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            output, output_size, &offset,
            "        target_tile[tile_row] = target[row];\n"
            "    }\n\n"
            "    __syncthreads();\n\n"
            "    for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
        for (group_idx = 0u; group_idx < (recipe->num_input_constants + 3u) / 4u; ++group_idx) {
            _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
                output, output_size, &offset,
                "        const uint4 random%zu = setting == 0u ? make_uint4(0u, 0u, 0u, 0u) :\n"
                "            secant_constant_random4((unsigned long long)setting, ast_index, seed, generation, iteration, %zuu);\n",
                group_idx, group_idx));
        }
        for (constant_idx = 0u; constant_idx < recipe->num_input_constants; ++constant_idx) {
            static const char* const lanes[] = {"x", "y", "z", "w"};

            _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
                output, output_size, &offset,
                "        const float input%zu = current_constants[%zu] + (setting == 0u ? 0.0f :\n"
                "            perturbation_scale * secant_constant_delta(random%zu.%s));\n",
                recipe->num_input_columns + constant_idx, constant_idx, constant_idx / 4u, lanes[constant_idx % 4u]));
        }
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            output, output_size, &offset,
            "        float sse = 0.0f;\n\n"
            "        #pragma unroll 1\n"
            "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
        for (input_idx = 0u; input_idx < recipe->num_input_columns; ++input_idx) {
            _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
                output, output_size, &offset,
                "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
                input_idx, input_idx, recipe->tile_rows));
        }
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            output, output_size, &offset,
            "            const float target0 = target_tile[eval_row];\n\n"
            "            {\n"));
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_skeleton(
            kernel_idx, num_inputs, recipe->patch_capacity_instructions, output, output_size, &offset));
        _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
            output, output_size, &offset,
            "            }\n"
            "        }\n\n"
            "        atomicAdd(output_sse + setting, sse);\n"
            "    }\n}%s",
            kernel_idx + 1u == recipe->num_kernels ? "\n" : "\n\n"));
    }
    _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET(_secant_cubin_constant_optimizer_write(
        output, output_size, &offset, "\n%s", reducer_source));
    return _secant_cubin_constant_optimizer_finish(output, output_size, offset, required_size_ret);
}

#undef _SECANT_CUBIN_CONSTANT_OPTIMIZER_CHECK_RET
#undef _SECANT_CUBIN_CONSTANT_OPTIMIZER_ERROR_RET
#undef _SECANT_CUBIN_CONSTANT_OPTIMIZER_FIRST_MARKER_BITS
#undef _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_REGISTERS
#undef _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_INPUT_CONSTANTS
#undef _SECANT_CUBIN_CONSTANT_OPTIMIZER_MAX_INPUT_COLUMNS
