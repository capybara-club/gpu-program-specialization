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

#define _SECANT_CUBIN_MATERIALIZE_GENERATE_MAX_REGISTERS 256u
#define _SECANT_CUBIN_MATERIALIZE_GENERATE_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_materialize_generate_result = (ans); \
    return _secant_cubin_materialize_generate_result; \
} while (0)
#define _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_materialize_generate_check_result = (ans); \
    if (_secant_cubin_materialize_generate_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(_secant_cubin_materialize_generate_check_result); \
    } \
} while (0)

static int
_secant_cubin_materialize_generate_checked_add(
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
_secant_cubin_materialize_generate_checked_mul(
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
_secant_cubin_materialize_generate_write(
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
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    va_copy(measure_args, args);
    required = vsnprintf(NULL, 0u, format, measure_args);
    va_end(measure_args);
    if (required < 0) {
        va_end(args);
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        va_end(args);
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_ERROR_OVERFLOW);
    }
    if (buffer != NULL) {
        if (*offset > buffer_size || (size_t)required >= buffer_size - *offset) {
            va_end(args);
            _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        if (vsnprintf(
                buffer + *offset,
                buffer_size - *offset,
                format,
                args) != required) {
            va_end(args);
            _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
                SECANT_ERROR_FORMAT);
        }
    }
    va_end(args);
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_materialize_generate_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_materialize_generate_reduction(
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
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_SUCCESS);
    }
    left_operands = num_operands / 2u;
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_reduction(
            first_operand,
            left_operands,
            buffer,
            buffer_size,
            offset,
            &left));
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_reduction(
            first_operand + left_operands,
            num_operands - left_operands,
            buffer,
            buffer_size,
            offset,
            &right));
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
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
_secant_cubin_materialize_generate_skeleton(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t patch_capacity_instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t marker_count = num_inputs + asts_per_kernel;
    const uint32_t marker = _SECANT_CUBIN_MATERIALIZE_GENERATE_FIRST_MARKER_BITS +
        (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_inputs + asts_per_kernel;
    const size_t input_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t input_idx;
    size_t ast_idx;

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "        float marked%zu __attribute__((unused));\n",
                input_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "        float result%zu;\n",
                ast_idx));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            "        float keepalive __attribute__((unused));\n"
            "        asm volatile(\n"
            "            \"{\\n\\t\"\n"
            "            \".reg .u32 keepalive_address;\\n\\t\"\n"
            "            \"brkpt;\\n\\t\"\n"));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
                input_idx,
                input_operand_offset + input_idx,
                marker + (uint32_t)input_idx));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"brkpt;\\n\\t\"\n"));
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_reduction(
            0u,
            num_inputs,
            buffer,
            buffer_size,
            offset,
            &reduction_operand));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
                num_inputs + ast_idx,
                reduction_operand,
                marker + (uint32_t)(num_inputs + ast_idx)));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"brkpt;\\n\\t\"\n"));
    if (asts_per_kernel == 1u) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
                keepalive_operand,
                num_inputs,
                input_operand_offset));
        input_idx = 1u;
    } else {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
                keepalive_operand,
                num_inputs,
                num_inputs + 1u));
        for (ast_idx = 2u; ast_idx < asts_per_kernel; ++ast_idx) {
            _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
                _secant_cubin_materialize_generate_write(
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
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
                keepalive_operand,
                keepalive_operand,
                input_operand_offset + input_idx));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
            "            \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
            keepalive_operand));
    for (input_idx = 0u;
         input_idx < patch_capacity_instructions;
         ++input_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "            \"brkpt;\\n\\t\"\n"));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            "            \"}\\n\\t\"\n"
            "            : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "%s\"=&f\"(marked%zu)",
                input_idx == 0u ? "" : ", ",
                input_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                ", \"=&f\"(result%zu)",
                ast_idx));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            ", \"=&f\"(keepalive)\n"
            "            : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                buffer,
                buffer_size,
                offset,
                "%s\"f\"(input%zu)",
                input_idx == 0u ? "" : ", ",
                input_idx));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_generate_write(
            buffer,
            buffer_size,
            offset,
            "\n"
            "            : \"memory\");\n"));
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_materialize_generate_validate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t patch_capacity_instructions
) {
    size_t marker_count;
    size_t total_markers;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        asts_per_kernel >
            _SECANT_CUBIN_MATERIALIZE_GENERATE_MAX_REGISTERS ||
        num_inputs == 0u ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        patch_capacity_instructions == 0u ||
        !_secant_cubin_materialize_generate_checked_add(
            num_inputs,
            asts_per_kernel,
            &marker_count) ||
        marker_count >
            _SECANT_CUBIN_MATERIALIZE_GENERATE_MAX_REGISTERS ||
        !_secant_cubin_materialize_generate_checked_mul(
            num_kernels,
            marker_count,
            &total_markers) ||
        total_markers - 1u >
            0x7fffffffu -
                _SECANT_CUBIN_MATERIALIZE_GENERATE_FIRST_MARKER_BITS) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_materialize_recipe_validate(
    const SecantCubinMaterializeRecipe* recipe
) {
    if (recipe == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_cubin_materialize_generate_validate(
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        recipe->patch_capacity_instructions);
}

SecantResult
_secant_cubin_materialize_source_generate(
    const SecantCubinMaterializeRecipe* recipe,
    char* cuda_source,
    size_t cuda_source_size,
    size_t* cuda_source_size_ret
) {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t patch_capacity_instructions;
    size_t kernel_idx;
    size_t offset = 0u;

    if (recipe == NULL || cuda_source_size_ret == NULL) {
        _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
            SECANT_ERROR_INVALID_VALUE);
    }
    num_kernels = recipe->num_kernels;
    asts_per_kernel = recipe->asts_per_kernel;
    num_inputs = recipe->num_inputs;
    patch_capacity_instructions = recipe->patch_capacity_instructions;
    *cuda_source_size_ret = 0u;
    _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
        _secant_cubin_materialize_recipe_validate(recipe));

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        size_t input_idx;
        size_t ast_idx;

        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "extern \"C\" __global__\n"
                "void secant_cubin_materialize_%03zu(\n"
                "    const float* __restrict__ input,\n"
                "    size_t input_leading_dimension,\n"
                "    size_t num_rows,\n"
                "    size_t num_asts,\n"
                "    float* __restrict__ output,\n"
                "    size_t output_leading_dimension\n"
                ") {\n"
                "    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    if (row >= num_rows) {\n"
                "        return;\n"
                "    }\n\n",
                kernel_idx));
        for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
            _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
                _secant_cubin_materialize_generate_write(
                    cuda_source,
                    cuda_source_size,
                    &offset,
                    "    const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
                    input_idx,
                    input_idx));
        }
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                "\n"
                "    {\n"));
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_skeleton(
                kernel_idx,
                asts_per_kernel,
                num_inputs,
                patch_capacity_instructions,
                cuda_source,
                cuda_source_size,
                &offset));
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
                _secant_cubin_materialize_generate_write(
                    cuda_source,
                    cuda_source_size,
                    &offset,
                    "        if (%zuu < num_asts) {\n"
                    "            output[(size_t)%zu * output_leading_dimension + row] = result%zu;\n"
                    "        }\n",
                    ast_idx,
                    ast_idx,
                    ast_idx));
        }
        _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET(
            _secant_cubin_materialize_generate_write(
                cuda_source,
                cuda_source_size,
                &offset,
                kernel_idx + 1u == num_kernels
                    ? "    }\n}\n"
                    : "    }\n}\n\n"));
    }
    _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET(
        _secant_cubin_materialize_generate_finish(
            cuda_source,
            cuda_source_size,
            offset,
            cuda_source_size_ret));
}

#undef _SECANT_CUBIN_MATERIALIZE_GENERATE_CHECK_RET
#undef _SECANT_CUBIN_MATERIALIZE_GENERATE_ERROR_RET
#undef _SECANT_CUBIN_MATERIALIZE_GENERATE_FIRST_MARKER_BITS
#undef _SECANT_CUBIN_MATERIALIZE_GENERATE_MAX_REGISTERS
