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

#define _SECANT_CUBIN_DYNAMIC_MAX_INPUT_COLUMNS 32u
#define _SECANT_CUBIN_DYNAMIC_MAX_INPUT_CONSTANTS 32u
#define _SECANT_CUBIN_DYNAMIC_MAX_REGISTERS 256u
#define _SECANT_CUBIN_DYNAMIC_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_CUBIN_DYNAMIC_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_dynamic_result = (ans); \
    return _secant_cubin_dynamic_result; \
} while (0)
#define _SECANT_CUBIN_DYNAMIC_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_dynamic_check_result = (ans); \
    if (_secant_cubin_dynamic_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_DYNAMIC_ERROR_RET(_secant_cubin_dynamic_check_result); \
    } \
} while (0)

static int
_secant_cubin_dynamic_checked_add(
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
_secant_cubin_dynamic_checked_mul(
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
_secant_cubin_dynamic_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        _SECANT_CUBIN_DYNAMIC_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        required = vsnprintf(
            buffer + *offset,
            buffer_size - *offset,
            format,
            args);
    } else {
        required = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (required < 0) {
        _SECANT_CUBIN_DYNAMIC_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        _SECANT_CUBIN_DYNAMIC_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_dynamic_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        _SECANT_CUBIN_DYNAMIC_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_CUBIN_DYNAMIC_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_dynamic_reduction(
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
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_reduction(
        first_operand,
        left_count,
        buffer,
        buffer_size,
        offset,
        &left));
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_reduction(
        first_operand + left_count,
        num_operands - left_count,
        buffer,
        buffer_size,
        offset,
        &right));
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        left,
        left,
        right));
    *result_operand_ret = left;
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_dynamic_sse_skeleton(
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
    const uint32_t marker = _SECANT_CUBIN_DYNAMIC_FIRST_MARKER_BITS + (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_sources + num_outputs;
    const size_t source_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t source_idx;
    size_t output_idx;

    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            "            float marked%zu __attribute__((unused));\n",
            source_idx));
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "            float keepalive __attribute__((unused));\n"
        "            asm volatile(\n"
        "                \"{\\n\\t\"\n"
        "                \".reg .u32 keepalive_address;\\n\\t\"\n"
        "                \"brkpt;\\n\\t\"\n"));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            source_idx,
            source_operand_offset + source_idx,
            marker + (uint32_t)source_idx));
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "                \"brkpt;\\n\\t\"\n"));
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_reduction(
        0u,
        num_sources,
        buffer,
        buffer_size,
        offset,
        &reduction_operand));
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        const size_t output_operand = num_sources + output_idx;

        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            output_operand,
            output_operand,
            marker + (uint32_t)(num_sources + output_idx)));
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "                \"brkpt;\\n\\t\"\n"
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        keepalive_operand,
        reduction_operand,
        source_operand_offset));
    for (source_idx = 1u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            keepalive_operand,
            source_operand_offset + source_idx));
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "                \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
        "                \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
        keepalive_operand));
    for (source_idx = 0u;
         source_idx < patch_capacity_instructions;
         ++source_idx) {
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            "                \"brkpt;\\n\\t\"\n"));
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "                \"}\\n\\t\"\n"
        "                : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            "%s\"=&f\"(marked%zu)",
            source_idx == 0u ? "" : ", ",
            source_idx));
    }
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        const size_t ast_idx = output_idx / num_targets;
        const size_t target_idx = output_idx % num_targets;

        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            buffer,
            buffer_size,
            offset,
            ", \"+f\"(sse_%03zu_%03zu)",
            ast_idx,
            target_idx));
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        ", \"=&f\"(keepalive)\n"
        "                : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        if (source_idx < num_inputs) {
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                buffer,
                buffer_size,
                offset,
                "%s\"f\"(input%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx));
        } else {
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                buffer,
                buffer_size,
                offset,
                "%s\"f\"(target%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx - num_inputs));
        }
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "                : \"memory\");\n"));
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_dynamic_constant_sse_recipe_validate(
    const SecantCubinDynamicConstantSSERecipe* recipe
) {
    size_t num_inputs;
    size_t num_pairs;
    size_t num_sources;
    size_t marker_count;
    size_t total_markers;
    size_t row_tile_elements;

    if (recipe == NULL ||
        recipe->num_kernels == 0u ||
        recipe->asts_per_kernel == 0u ||
        recipe->num_input_columns == 0u ||
        recipe->num_input_columns >
            _SECANT_CUBIN_DYNAMIC_MAX_INPUT_COLUMNS ||
        recipe->num_input_constants == 0u ||
        recipe->num_input_constants >
            _SECANT_CUBIN_DYNAMIC_MAX_INPUT_CONSTANTS ||
        recipe->num_targets == 0u ||
        recipe->tile_rows == 0u ||
        recipe->threads_per_block == 0u ||
        recipe->threads_per_block > 1024u ||
        recipe->patch_capacity_instructions == 0u ||
        !_secant_cubin_dynamic_checked_add(
            recipe->num_input_columns,
            recipe->num_input_constants,
            &num_inputs) ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cubin_dynamic_checked_mul(
            recipe->asts_per_kernel,
            recipe->num_targets,
            &num_pairs) ||
        !_secant_cubin_dynamic_checked_add(
            num_inputs,
            recipe->num_targets,
            &num_sources) ||
        !_secant_cubin_dynamic_checked_add(
            num_sources,
            num_pairs,
            &marker_count) ||
        marker_count > _SECANT_CUBIN_DYNAMIC_MAX_REGISTERS ||
        !_secant_cubin_dynamic_checked_mul(
            recipe->num_kernels,
            marker_count,
            &total_markers) ||
        total_markers - 1u >
            0x7fffffffu -
                _SECANT_CUBIN_DYNAMIC_FIRST_MARKER_BITS ||
        !_secant_cubin_dynamic_checked_mul(
            recipe->num_input_columns + recipe->num_targets,
            recipe->tile_rows,
            &row_tile_elements)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_dynamic_constant_sse_source_generate(
    const SecantCubinDynamicConstantSSERecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
    size_t num_inputs;
    size_t row_tile_elements;
    size_t kernel_idx;
    size_t offset = 0u;

    if (required_size_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    _SECANT_CUBIN_DYNAMIC_CHECK_RET(
        _secant_cubin_dynamic_constant_sse_recipe_validate(recipe));
    num_inputs = recipe->num_input_columns + recipe->num_input_constants;
    row_tile_elements = (recipe->num_input_columns + recipe->num_targets) * recipe->tile_rows;
    *required_size_ret = 0u;

    for (kernel_idx = 0u;
         kernel_idx < recipe->num_kernels;
         ++kernel_idx) {
        size_t input_idx;
        size_t constant_idx;
        size_t ast_idx;
        size_t target_idx;

        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_cubin_dynamic_constant_sse_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t input_leading_dimension,\n"
            "    const float* __restrict__ constant_settings,\n"
            "    size_t constants_leading_dimension,\n"
            "    size_t num_settings,\n"
            "    const float* __restrict__ targets,\n"
            "    size_t targets_leading_dimension,\n"
            "    size_t num_rows,\n"
            "    size_t num_asts,\n"
            "    size_t num_targets,\n"
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
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                output,
                output_size,
                &offset,
                "        input_tile[(size_t)%zu * %zuu + tile_row] = input[(size_t)%zu * input_leading_dimension + row];\n",
                input_idx,
                recipe->tile_rows,
                input_idx));
        }
        for (target_idx = 0u;
             target_idx < recipe->num_targets;
             ++target_idx) {
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                output,
                output_size,
                &offset,
                "        target_tile[(size_t)%zu * %zuu + tile_row] = "
                "targets[(size_t)(%zuu < num_targets ? %zuu : 0u) * targets_leading_dimension + row];\n",
                target_idx,
                recipe->tile_rows,
                target_idx,
                target_idx));
        }
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            "    }\n\n"
            "    __syncthreads();\n\n"
            "    for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
        for (constant_idx = 0u;
             constant_idx < recipe->num_input_constants;
             ++constant_idx) {
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                output,
                output_size,
                &offset,
                "        const float input%zu = constant_settings[(size_t)%zu * constants_leading_dimension + setting];\n",
                recipe->num_input_columns + constant_idx,
                constant_idx));
        }
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            "\n"));
        for (ast_idx = 0u;
             ast_idx < recipe->asts_per_kernel;
             ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < recipe->num_targets;
                 ++target_idx) {
                _SECANT_CUBIN_DYNAMIC_CHECK_RET(
                    _secant_cubin_dynamic_write(
                        output,
                        output_size,
                        &offset,
                        "        float sse_%03zu_%03zu = 0.0f;\n",
                        ast_idx,
                        target_idx));
            }
        }
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            "\n"
            "        #pragma unroll 1\n"
            "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
        for (input_idx = 0u;
             input_idx < recipe->num_input_columns;
             ++input_idx) {
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                output,
                output_size,
                &offset,
                "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
                input_idx,
                input_idx,
                recipe->tile_rows));
        }
        for (target_idx = 0u;
             target_idx < recipe->num_targets;
             ++target_idx) {
            _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
                output,
                output_size,
                &offset,
                "            const float target%zu = target_tile[(size_t)%zu * %zuu + eval_row];\n",
                target_idx,
                target_idx,
                recipe->tile_rows));
        }
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            "\n"
            "            {\n"));
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_sse_skeleton(
            kernel_idx,
            recipe->asts_per_kernel,
            num_inputs,
            recipe->num_targets,
            recipe->patch_capacity_instructions,
            output,
            output_size,
            &offset));
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            "            }\n"
            "        }\n\n"));
        for (ast_idx = 0u;
             ast_idx < recipe->asts_per_kernel;
             ++ast_idx) {
            for (target_idx = 0u;
                 target_idx < recipe->num_targets;
                 ++target_idx) {
                _SECANT_CUBIN_DYNAMIC_CHECK_RET(
                    _secant_cubin_dynamic_write(
                        output,
                        output_size,
                        &offset,
                        "        if (%zuu < num_asts && %zuu < num_targets) {\n"
                        "            atomicAdd(output_sse + ((size_t)%zu * num_targets + %zuu) * "
                        "output_leading_dimension + setting, sse_%03zu_%03zu);\n"
                        "        }\n",
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx,
                        ast_idx,
                        target_idx));
            }
        }
        _SECANT_CUBIN_DYNAMIC_CHECK_RET(_secant_cubin_dynamic_write(
            output,
            output_size,
            &offset,
            kernel_idx + 1u == recipe->num_kernels
                ? "    }\n}\n"
                : "    }\n}\n\n"));
    }
    _SECANT_CUBIN_DYNAMIC_ERROR_RET(_secant_cubin_dynamic_finish(
        output,
        output_size,
        offset,
        required_size_ret));
}

#undef _SECANT_CUBIN_DYNAMIC_CHECK_RET
#undef _SECANT_CUBIN_DYNAMIC_ERROR_RET
#undef _SECANT_CUBIN_DYNAMIC_FIRST_MARKER_BITS
#undef _SECANT_CUBIN_DYNAMIC_MAX_REGISTERS
#undef _SECANT_CUBIN_DYNAMIC_MAX_INPUT_CONSTANTS
#undef _SECANT_CUBIN_DYNAMIC_MAX_INPUT_COLUMNS
