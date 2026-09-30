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


#define _SECANT_CUBIN_TOGGLE_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_toggle_result = (ans); \
    return _secant_cubin_toggle_result; \
} while (0)
#define _SECANT_CUBIN_TOGGLE_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_toggle_check_result = (ans); \
    if (_secant_cubin_toggle_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_TOGGLE_ERROR_RET(_secant_cubin_toggle_check_result); \
    } \
} while (0)

static SecantResult
_secant_cubin_toggle_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        _SECANT_CUBIN_TOGGLE_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
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
        _SECANT_CUBIN_TOGGLE_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        _SECANT_CUBIN_TOGGLE_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    *offset += (size_t)required;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_toggle_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        _SECANT_CUBIN_TOGGLE_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_CUBIN_TOGGLE_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_toggle_reduction(
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
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_reduction(
        first_operand,
        left_count,
        buffer,
        buffer_size,
        offset,
        &left));
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_reduction(
        first_operand + left_count,
        num_operands - left_count,
        buffer,
        buffer_size,
        offset,
        &right));
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
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
_secant_cubin_toggle_sse_skeleton(
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
    const uint32_t marker = _SECANT_CUBIN_FIRST_MARKER_BITS + (uint32_t)(kernel_idx * marker_count);
    const size_t keepalive_operand = num_sources + num_outputs;
    const size_t source_operand_offset = keepalive_operand + 1u;
    size_t reduction_operand;
    size_t source_idx;
    size_t output_idx;

    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
            buffer,
            buffer_size,
            offset,
            "            float marked%zu __attribute__((unused));\n",
            source_idx));
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        "            float keepalive __attribute__((unused));\n"
        "            asm volatile(\n"
        "                \"{\\n\\t\"\n"
        "                \".reg .u32 keepalive_address;\\n\\t\"\n"
        "                \"brkpt;\\n\\t\"\n"));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            source_idx,
            source_operand_offset + source_idx,
            marker + (uint32_t)source_idx));
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        "                \"brkpt;\\n\\t\"\n"));
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_reduction(
        0u,
        num_sources,
        buffer,
        buffer_size,
        offset,
        &reduction_operand));
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(buffer, buffer_size, offset,
        "                \".reg .b32 toggle_masked;\\n\\t\"\n"
        "                \".reg .pred toggle_predicate;\\n\\t\"\n"
        "                \"and.b32 toggle_masked, %%%zu, 1;\\n\\t\"\n"
        "                \"setp.ne.u32 toggle_predicate, toggle_masked, 0;\\n\\t\"\n"
        "                \"selp.f32 %%0, %%0, %%%zu, toggle_predicate;\\n\\t\"\n",
        source_operand_offset + num_sources, source_operand_offset));
    for (output_idx = 0u; output_idx < num_outputs; ++output_idx) {
        const size_t output_operand = num_sources + output_idx;

        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;\\n\\t\"\n",
            output_operand,
            output_operand,
            marker + (uint32_t)(num_sources + output_idx)));
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        "                \"brkpt;\\n\\t\"\n"
        "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
        keepalive_operand,
        reduction_operand,
        source_operand_offset));
    for (source_idx = 1u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
            buffer,
            buffer_size,
            offset,
            "                \"add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;\\n\\t\"\n",
            keepalive_operand,
            keepalive_operand,
            source_operand_offset + source_idx));
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        "                \"mov.u32 keepalive_address, 0;\\n\\t\"\n"
        "                \"st.volatile.shared.f32 [keepalive_address], %%%zu;\\n\\t\"\n",
        keepalive_operand));
    for (source_idx = 0u;
         source_idx < patch_capacity_instructions;
         ++source_idx) {
        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
            buffer,
            buffer_size,
            offset,
            "                \"brkpt;\\n\\t\"\n"));
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        "                \"}\\n\\t\"\n"
        "                : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
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

        _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
            buffer,
            buffer_size,
            offset,
            ", \"+f\"(sse_%03zu_%03zu)",
            ast_idx,
            target_idx));
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        ", \"=&f\"(keepalive)\n"
        "                : "));
    for (source_idx = 0u; source_idx < num_sources; ++source_idx) {
        if (source_idx < num_inputs) {
            _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
                buffer,
                buffer_size,
                offset,
                "%s\"f\"(input%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx));
        } else {
            _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
                buffer,
                buffer_size,
                offset,
                "%s\"f\"(target%zu)",
                source_idx == 0u ? "" : ", ",
                source_idx - num_inputs));
        }
    }
    _SECANT_CUBIN_TOGGLE_CHECK_RET(_secant_cubin_toggle_write(
        buffer,
        buffer_size,
        offset,
        ", \"r\"(permutation)\n"
        "                : \"memory\");\n"));
    return SECANT_SUCCESS;
}

SecantResult _secant_cubin_toggle_sse_recipe_validate(const SecantCubinToggleSSERecipe *r) {
    size_t sources, markers, all_markers, shared_elements, outputs;
    if (!r || !r->num_kernels || r->num_kernels > 65535 || !r->asts_per_kernel || r->asts_per_kernel > 32 ||
        r->num_inputs > SECANT_AST_MAX_INPUTS || r->num_constants > SECANT_AST_MAX_INPUTS ||
        !r->num_targets || r->num_targets > 32 || !r->tile_rows ||
        !r->threads_per_block || r->threads_per_block > 1024 || r->threads_per_block % 32 ||
        !r->patch_capacity_instructions ||
        !_secant_cubin_checked_add(r->num_inputs, r->num_constants, &sources) || sources > SECANT_AST_MAX_INPUTS ||
        !_secant_cubin_checked_add(sources, r->num_targets, &sources) ||
        !_secant_cubin_checked_mul(r->asts_per_kernel, r->num_targets, &outputs) ||
        !_secant_cubin_checked_add(sources, outputs, &markers) || markers > 192 ||
        !_secant_cubin_checked_mul(markers, r->num_kernels, &all_markers) ||
        all_markers > UINT32_MAX - _SECANT_CUBIN_FIRST_MARKER_BITS ||
        !_secant_cubin_checked_mul(r->num_inputs + r->num_targets, r->tile_rows, &shared_elements) ||
        shared_elements > 12000)
        return SECANT_ERROR_INVALID_VALUE;
    return SECANT_SUCCESS;
}

SecantResult _secant_cubin_toggle_sse_source_generate(const SecantCubinToggleSSERecipe *r,
    char *output, size_t output_size, size_t *required_size_ret) {
    size_t offset = 0, k, c, t, a;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_toggle_sse_recipe_validate(r));
    if (!required_size_ret) return SECANT_ERROR_INVALID_VALUE;
#define WRITE(...) _SECANT_CUBIN_CHECK_RET(_secant_cubin_toggle_write(output, output_size, &offset, __VA_ARGS__))
    WRITE("// Secant 0.3: configuration = bank * 2^toggle_bits + permutation.\n");
    for (k = 0; k < r->num_kernels; ++k) {
        WRITE("extern \"C\" __global__ __launch_bounds__(%zu) void secant_cubin_toggle_sse_%03zu(\n"
              " const float *input, unsigned long long input_ld, const float *targets, unsigned long long target_ld,\n"
              " const float *banks, unsigned long long bank_stride,\n"
              " unsigned long long num_rows, unsigned long long num_configurations, unsigned int toggle_bits,\n"
              " float *output, unsigned long long output_ld, unsigned long long active_asts) {\n"
              " __shared__ float tile[%zu];\n"
              " const unsigned long long row_start = (unsigned long long)blockIdx.x * %zu;\n"
              " for (unsigned int i = threadIdx.x; i < %zu; i += blockDim.x) {\n"
              "  unsigned int col = i / %zu, row = i %% %zu;\n"
              "  unsigned long long absolute_row = row_start + row;\n"
              "  tile[i] = absolute_row >= num_rows ? 0.f : (col < %zu ? input[col * input_ld + absolute_row] : targets[(col - %zu) * target_ld + absolute_row]);\n"
              " }\n __syncthreads();\n"
              " #pragma unroll 1\n"
              " for (unsigned long long configuration = (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x;\n"
              "      configuration < num_configurations; configuration += (unsigned long long)gridDim.y * blockDim.x) {\n"
              "  const unsigned int permutation = (unsigned int)configuration;\n"
              "  const unsigned long long bank = configuration >> toggle_bits;\n",
              r->threads_per_block, k, (r->num_inputs + r->num_targets) * r->tile_rows,
              r->tile_rows, (r->num_inputs + r->num_targets) * r->tile_rows,
              r->tile_rows, r->tile_rows, r->num_inputs, r->num_inputs);
        for (c = 0; c < r->num_constants; ++c)
            WRITE("  const float input%zu = banks[bank * bank_stride + %zu];\n", r->num_inputs + c, c);
        for (a = 0; a < r->asts_per_kernel; ++a) {
            for (t = 0; t < r->num_targets; ++t) WRITE("  float sse_%03zu_%03zu = 0.f;\n", a, t);
        }
        WRITE("  #pragma unroll 1\n  for (unsigned int row = 0; row < %zu && row_start + row < num_rows; ++row) {\n", r->tile_rows);
        for (c = 0; c < r->num_inputs; ++c) WRITE("   float input%zu = tile[%zu + row];\n", c, c * r->tile_rows);
        for (t = 0; t < r->num_targets; ++t) WRITE("   float target%zu = tile[%zu + row];\n", t, (r->num_inputs + t) * r->tile_rows);
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_toggle_sse_skeleton(k, r->asts_per_kernel, r->num_inputs + r->num_constants,
            r->num_targets, r->patch_capacity_instructions, output, output_size, &offset));
        WRITE("  }\n");
        for (a = 0; a < r->asts_per_kernel; ++a)
            for (t = 0; t < r->num_targets; ++t)
                WRITE("  if (%zu < active_asts) atomicAdd(output + %zu * output_ld + configuration, sse_%03zu_%03zu);\n",
                    k * r->asts_per_kernel + a, (k * r->asts_per_kernel + a) * r->num_targets + t, a, t);
        WRITE(" }\n}\n");
    }
#undef WRITE
    return _secant_cubin_toggle_finish(output, output_size, offset, required_size_ret);
}
