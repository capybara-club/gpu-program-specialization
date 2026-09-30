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

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

static int
_secant_hsaco_checked_add(size_t lhs, size_t rhs, size_t* result_ret) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

int
_secant_hsaco_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}
SecantResult
_secant_hsaco_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        required = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        required = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (required < 0) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

SecantResult
_secant_hsaco_write_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_emit_reduction(
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
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_reduction(
        first_operand,
        left_count,
        buffer,
        buffer_size,
        offset,
        &left));
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_reduction(
        first_operand + left_count,
        num_operands - left_count,
        buffer,
        buffer_size,
        offset,
        &right));
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        "            \"v_add_f32 %%%zu, %%%zu, %%%zu\\n\\t\"\n",
        left,
        left,
        right));
    *result_operand_ret = left;
    return SECANT_SUCCESS;
}

SecantResult
_secant_hsaco_emit_skeleton(
    _SecantHsacoShape shape,
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
    const size_t num_outputs = shape == _SECANT_HSACO_SHAPE_MATERIALIZE
        ? asts_per_kernel
        : asts_per_kernel * num_targets;
    const size_t marker_count = num_sources + num_outputs;
    const uint32_t marker =
        _SECANT_HSACO_FIRST_MARKER_BITS +
        (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_sources + num_outputs;
    const size_t source_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t source_idx;
    size_t output_idx;

    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "        float marked%zu __attribute__((unused));\n",
            source_idx));
    }
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                buffer,
                buffer_size,
                offset,
                "        float result%zu;\n",
                output_idx));
        }
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        "        float keepalive __attribute__((unused));\n"
        "        asm volatile(\n"
        "            \"s_trap 2\\n\\t\"\n"));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "            \"v_add_f32 %%%zu, 0x%08x, %%%zu\\n\\t\"\n",
            source_idx,
            marker + (uint32_t)source_idx,
            source_operand_offset + source_idx));
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        "            \"s_trap 2\\n\\t\"\n"));
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_reduction(
        0u,
        num_sources,
        buffer,
        buffer_size,
        offset,
        &reduction_operand));
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        const size_t output_operand = num_sources + output_idx;
        const size_t source_operand = shape == _SECANT_HSACO_SHAPE_MATERIALIZE
            ? reduction_operand
            : output_operand;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "            \"v_add_f32 %%%zu, 0x%08x, %%%zu\\n\\t\"\n",
            output_operand,
            marker + (uint32_t)(num_sources + output_idx),
            source_operand));
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        "            \"s_trap 2\\n\\t\"\n"
        "            \"v_add_f32 %%%zu, %%%zu, %%%zu\\n\\t\"\n",
        keepalive_operand,
        reduction_operand,
        source_operand_offset));
    for (source_idx = 1u; source_idx < num_sources; ++source_idx) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "            \"v_add_f32 %%%zu, %%%zu, %%%zu\\n\\t\"\n",
            keepalive_operand,
            keepalive_operand,
            source_operand_offset + source_idx));
    }
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "            \"v_add_f32 %%%zu, %%%zu, %%%zu\\n\\t\"\n",
            keepalive_operand,
            keepalive_operand,
            num_sources + output_idx));
    }
    for (source_idx = 0u;
         source_idx < patch_capacity_instructions;
         ++source_idx) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "            \"s_trap 2\\n\\t\"\n"));
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        "            : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
            buffer,
            buffer_size,
            offset,
            "%s\"=&v\"(marked%zu)",
            source_idx == 0u ? "" : ", ",
            source_idx));
    }
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                buffer,
                buffer_size,
                offset,
                ", \"=&v\"(result%zu)",
                output_idx));
        } else {
            const size_t ast_idx = output_idx / num_targets;
            const size_t target_idx = output_idx % num_targets;

            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                buffer,
                buffer_size,
                offset,
                ", \"+v\"(sse_%03zu_%03zu)",
                ast_idx,
                target_idx));
        }
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        ", \"=&v\"(keepalive)\n"
        "            : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        if (source_idx < num_inputs) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                buffer,
                buffer_size,
                offset,
                "%s\"v\"(input%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx));
        } else {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
                buffer,
                buffer_size,
                offset,
                "%s\"v\"(target%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx - num_inputs));
        }
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "            : \"memory\");\n"));
    return SECANT_SUCCESS;
}

SecantResult
_secant_hsaco_validate_generate_args(
    _SecantHsacoShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    size_t* marker_count_ret
) {
    size_t num_outputs;
    size_t marker_count;
    size_t total_markers;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        asts_per_kernel > _SECANT_HSACO_MAX_REGISTERS ||
        num_inputs == 0u || num_inputs > _SECANT_HSACO_MAX_REGISTERS ||
        patch_capacity_instructions == 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
        num_outputs = asts_per_kernel;
        num_targets = 0u;
    } else if (shape == _SECANT_HSACO_SHAPE_SSE ||
               shape == _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE) {
        if (num_targets == 0u ||
            !_secant_hsaco_checked_mul(
                asts_per_kernel,
                num_targets,
                &num_outputs)) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
        }
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_hsaco_checked_add(num_inputs, num_targets, &marker_count) ||
        !_secant_hsaco_checked_add(marker_count, num_outputs, &marker_count) ||
        marker_count > _SECANT_HSACO_MAX_REGISTERS ||
        !_secant_hsaco_checked_mul(num_kernels, marker_count, &total_markers) ||
        total_markers - 1u >
            0x7fffffffu - _SECANT_HSACO_FIRST_MARKER_BITS) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *marker_count_ret = marker_count;
    return SECANT_SUCCESS;
}
