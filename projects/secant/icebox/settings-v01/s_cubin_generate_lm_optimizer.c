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

#define S_CUBIN_LM_PARAMETERS SECANT_LM_OPTIMIZER_PARAMETERS
#define S_CUBIN_LM_STATISTICS SECANT_LM_OPTIMIZER_STATISTICS
#define S_CUBIN_LM_FIRST_MARKER_BITS 0x7fc0ffeeu

#define S_CUBIN_LM_ERROR_RET(ans) do { \
    SecantResult lm_error_result_ = (ans); \
    return lm_error_result_; \
} while (0)
#define S_CUBIN_LM_CHECK_RET(ans) do { \
    SecantResult lm_result_ = (ans); \
    if (lm_result_ != SECANT_SUCCESS) { S_CUBIN_LM_ERROR_RET(lm_result_); } \
} while (0)

static SecantResult
s_cubin_lm_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        S_CUBIN_LM_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        required = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        required = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (required < 0) {
        S_CUBIN_LM_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        S_CUBIN_LM_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
s_cubin_lm_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        S_CUBIN_LM_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            S_CUBIN_LM_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
s_cubin_lm_reduction(
    size_t first_operand,
    size_t num_operands,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    size_t* result_operand_ret
) {
    size_t left_count;
    size_t left = 0u;
    size_t right = 0u;

    if (num_operands == 1u) {
        *result_operand_ret = first_operand;
        return SECANT_SUCCESS;
    }
    left_count = num_operands / 2u;
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_reduction(
        first_operand,
        left_count,
        buffer,
        buffer_size,
        offset,
        &left));
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_reduction(
        first_operand + left_count,
        num_operands - left_count,
        buffer,
        buffer_size,
        offset,
        &right));
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
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

static SecantResult
s_cubin_lm_skeleton(
    size_t kernel_idx,
    size_t num_static_input_columns,
    size_t num_parameters,
    size_t patch_capacity_instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t num_inputs = num_static_input_columns + 3u * num_parameters;
    const size_t num_sources = num_inputs + 1u;
    const size_t marker_count = num_sources + S_CUBIN_LM_STATISTICS;
    const uint32_t marker =
        S_CUBIN_LM_FIRST_MARKER_BITS + (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_sources + S_CUBIN_LM_STATISTICS;
    size_t reduction_operand;
    size_t source_idx;
    size_t statistic_idx;

    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer,
        buffer_size,
        offset,
        "            float keepalive __attribute__((unused));\n"
        "            asm volatile(\n"
        "                \"{\\n\\t\"\n"
        "                \".reg .u32 keepalive_address;\\n\\t\"\n"
        "                \"brkpt;\\n\\t\"\n"));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            source_idx,
            source_idx,
            marker + (uint32_t)source_idx));
    }
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer,
        buffer_size,
        offset,
        "                \"brkpt;\\n\\t\"\n"));
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_reduction(
        0u,
        num_sources,
        buffer,
        buffer_size,
        offset,
        &reduction_operand));
    for (statistic_idx = 0u;
         statistic_idx < S_CUBIN_LM_STATISTICS;
         ++statistic_idx) {
        const size_t operand = num_sources + statistic_idx;

        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            operand,
            operand,
            marker + (uint32_t)operand));
    }
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer,
        buffer_size,
        offset,
        "                \"brkpt;\\n\\t\"\n"
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        keepalive_operand,
        reduction_operand,
        0u));
    for (source_idx = 1u; source_idx < num_sources; ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            keepalive_operand,
            source_idx));
    }
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer,
        buffer_size,
        offset,
        "                \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
        "                \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
        keepalive_operand));
    for (source_idx = 0u;
         source_idx < patch_capacity_instructions;
         ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer,
            buffer_size,
            offset,
            "                \"brkpt;\\n\\t\"\n"));
    }
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer,
        buffer_size,
        offset,
        "                \"}\\n\\t\"\n"
        "                : "));
    for (source_idx = 0u;
         source_idx < num_static_input_columns;
         ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer,
            buffer_size,
            offset,
            "%s\"+f\"(static_input%zu)",
            source_idx == 0u ? "" : ", ",
            source_idx));
    }
    for (source_idx = 0u; source_idx < num_parameters; ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer, buffer_size, offset,
            ", \"+f\"(constant%zu)",
            source_idx));
    }
    for (source_idx = 0u; source_idx < num_parameters; ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer, buffer_size, offset,
            ", \"+f\"(mixed%zu)",
            source_idx));
    }
    for (source_idx = 0u; source_idx < num_parameters; ++source_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer, buffer_size, offset,
            ", \"+f\"(mixed_gradient%zu)",
            source_idx));
    }
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer, buffer_size, offset,
        ", \"+f\"(target_value)"));
    for (statistic_idx = 0u;
         statistic_idx < S_CUBIN_LM_STATISTICS;
         ++statistic_idx) {
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            buffer,
            buffer_size,
            offset,
            ", \"+f\"(lm_%02zu)",
            statistic_idx));
    }
    S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
        buffer,
        buffer_size,
        offset,
        ", \"=&f\"(keepalive)\n"
        "                :\n"
        "                : \"memory\");\n"));
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_lm_optimizer_recipe_validate(
    const SecantCubinLMOptimizerRecipe* recipe
) {
    size_t num_inputs;
    size_t num_sources;
    size_t marker_count;
    size_t total_markers;

    if (recipe == NULL || recipe->num_kernels == 0u ||
        recipe->num_input_columns == 0u ||
        recipe->num_input_columns > SECANT_AST_MAX_INPUTS ||
        recipe->num_static_input_columns != recipe->num_input_columns ||
        recipe->num_parameters == 0u ||
        recipe->num_parameters > S_CUBIN_LM_PARAMETERS ||
        recipe->tile_rows == 0u || recipe->threads_per_block == 0u ||
        recipe->threads_per_block > 1024u ||
        recipe->patch_capacity_instructions == 0u ||
        !_secant_cubin_checked_add(
            recipe->num_static_input_columns,
            3u * recipe->num_parameters,
            &num_inputs) ||
        !_secant_cubin_checked_add(num_inputs, 1u, &num_sources) ||
        !_secant_cubin_checked_add(
            num_sources,
            S_CUBIN_LM_STATISTICS,
            &marker_count) ||
        marker_count > _SECANT_CUBIN_MAX_REGISTERS ||
        !_secant_cubin_checked_mul(
            recipe->num_kernels,
            marker_count,
            &total_markers) ||
        total_markers - 1u >
            0x7fffffffu - S_CUBIN_LM_FIRST_MARKER_BITS) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_lm_optimizer_source_generate(
    const SecantCubinLMOptimizerRecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
    size_t offset = 0u;
    size_t kernel_idx;

    if (required_size_ret == NULL) {
        S_CUBIN_LM_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    S_CUBIN_LM_CHECK_RET(_secant_cubin_lm_optimizer_recipe_validate(recipe));
    *required_size_ret = 0u;
    for (kernel_idx = 0u; kernel_idx < recipe->num_kernels; ++kernel_idx) {
        const size_t shared_stride = recipe->num_input_columns | 1u;
        size_t parameter_idx;
        size_t statistic_idx;

        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            output,
            output_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_lm_statistics_%03zu(\n"
            "    const float* __restrict__ input,\n"
            "    size_t num_input_columns,\n"
            "    size_t input_leading_dimension,\n"
            "    const unsigned int* __restrict__ leaf_masks,\n"
            "    const unsigned int* __restrict__ leaf_words,\n"
            "    size_t leaf_words_leading_dimension,\n"
            "    const float* __restrict__ constants,\n"
            "    size_t constants_leading_dimension,\n"
            "    unsigned int num_settings,\n"
            "    unsigned int settings_per_cta,\n"
            "    const float* __restrict__ target,\n"
            "    size_t num_rows,\n"
            "    float* __restrict__ statistics,\n"
            "    size_t statistics_leading_dimension) {\n"
            "    extern __shared__ float row_tile[];\n"
            "    float* const input_tile = row_tile;\n"
            "    float* const target_tile = row_tile + (size_t)%zuu * %zuu;\n"
            "    const size_t shared_stride = %zuu;\n"
            "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
            "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
            "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n"
            "    for (size_t tile_row = threadIdx.x; tile_row < %zuu; tile_row += blockDim.x) {\n"
            "        const size_t row = tile_begin + tile_row;\n"
            "        const bool row_valid = row < num_rows;\n"
            "        #pragma unroll\n"
            "        for (size_t column = 0u; column < %zuu; ++column) {\n"
            "            input_tile[tile_row * shared_stride + column] = row_valid && column < num_input_columns\n"
            "                ? input[column * input_leading_dimension + row] : 0.0f;\n"
            "        }\n"
            "        target_tile[tile_row] = row_valid ? target[row] : 0.0f;\n"
            "    }\n"
            "    __syncthreads();\n"
            "    const unsigned int setting_begin = blockIdx.y * settings_per_cta;\n"
            "    if (setting_begin >= num_settings) { return; }\n"
            "    const unsigned int setting_count = settings_per_cta < num_settings - setting_begin\n"
            "        ? settings_per_cta : num_settings - setting_begin;\n"
            "    for (unsigned int setting_offset = threadIdx.x; setting_offset < setting_count;\n"
            "         setting_offset += blockDim.x) {\n"
            "        const unsigned int setting = setting_begin + setting_offset;\n"
            "        const unsigned int leaf_mask = leaf_masks[setting];\n"
            "        const unsigned int* const words = leaf_words +\n"
            "            (size_t)setting * leaf_words_leading_dimension;\n"
            "        const float* const setting_constants = constants +\n"
            "            (size_t)setting * constants_leading_dimension;\n"
            "        const unsigned int input_tile_address =\n"
            "            (unsigned int)__cvta_generic_to_shared((const void*)input_tile);\n"
            "        const unsigned int shared_stride_bytes =\n"
            "            (unsigned int)(shared_stride * sizeof(float));\n",
            kernel_idx,
            recipe->tile_rows,
            shared_stride,
            shared_stride,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->tile_rows,
            recipe->num_input_columns));
        for (parameter_idx = 0u;
             parameter_idx < recipe->num_parameters;
             ++parameter_idx) {
            S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
                output,
                output_size,
                &offset,
                "        float constant%zu = setting_constants[%zu];\n"
                "        const unsigned int word%zu = words[%zu];\n"
                "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
                "        float mixed_gradient%zu = is_column%zu ? 0.0f : 1.0f;\n"
                "        float mixed%zu = constant%zu;\n"
                "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx,
                parameter_idx));
        }
        for (statistic_idx = 0u;
             statistic_idx < S_CUBIN_LM_STATISTICS;
             ++statistic_idx) {
            S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
                output,
                output_size,
                &offset,
                "        float lm_%02zu = 0.0f;\n",
                statistic_idx));
        }
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            output,
            output_size,
            &offset,
            "        #pragma unroll 1\n"
            "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"
            "            float target_value = target_tile[eval_row];\n"));
        for (parameter_idx = 0u;
             parameter_idx < recipe->num_parameters;
             ++parameter_idx) {
            S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
                output,
                output_size,
                &offset,
                "            asm volatile(\n"
                "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
                "                : \"+f\"(mixed%zu)\n"
                "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
                "                : \"memory\");\n",
                parameter_idx,
                parameter_idx,
                parameter_idx));
        }
        for (parameter_idx = 0u;
             parameter_idx < recipe->num_static_input_columns;
             ++parameter_idx) {
            S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
                output,
                output_size,
                &offset,
                "            float static_input%zu =\n"
                "                input_tile[eval_row * shared_stride + %zuu];\n",
                parameter_idx,
                parameter_idx));
        }
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            output,
            output_size,
            &offset,
            "            {\n"));
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_skeleton(
            kernel_idx,
            recipe->num_static_input_columns,
            recipe->num_parameters,
            recipe->patch_capacity_instructions,
            output,
            output_size,
            &offset));
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            output,
            output_size,
            &offset,
            "            }\n"));
        for (parameter_idx = 0u;
             parameter_idx < recipe->num_parameters;
             ++parameter_idx) {
            S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
                output,
                output_size,
                &offset,
                "            address%zu += shared_stride_bytes;\n",
                parameter_idx));
        }
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            output,
            output_size,
            &offset,
            "        }\n"));
        for (statistic_idx = 0u;
             statistic_idx < S_CUBIN_LM_STATISTICS;
             ++statistic_idx) {
            S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
                output,
                output_size,
                &offset,
                "        atomicAdd(statistics + (size_t)%zuu * statistics_leading_dimension + setting, lm_%02zu);\n",
                statistic_idx,
                statistic_idx));
        }
        S_CUBIN_LM_CHECK_RET(s_cubin_lm_write(
            output,
            output_size,
            &offset,
            kernel_idx + 1u == recipe->num_kernels
                ? "    }\n}\n"
                : "    }\n}\n\n"));
    }
    S_CUBIN_LM_ERROR_RET(s_cubin_lm_finish(
        output,
        output_size,
        offset,
        required_size_ret));
}

#undef S_CUBIN_LM_CHECK_RET
#undef S_CUBIN_LM_ERROR_RET
#undef S_CUBIN_LM_FIRST_MARKER_BITS
#undef S_CUBIN_LM_STATISTICS
#undef S_CUBIN_LM_PARAMETERS
