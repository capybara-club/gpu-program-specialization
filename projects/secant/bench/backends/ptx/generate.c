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
#include "secant_ptx.h"

#include <stdarg.h>
#include <stdio.h>

#define _SECANT_PTX_ERROR_RET(ans) do { SecantPTXResult secant_ptx_result = (ans); return secant_ptx_result; } while (0)

static SecantPTXResult
_secant_ptx_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int bytes;

    if (offset == NULL || format == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        bytes = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        bytes = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (bytes < 0) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_FORMAT);
    }
    if ((size_t)bytes > SIZE_MAX - *offset) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    *offset += (size_t)bytes;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_write_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    if (offset == SIZE_MAX) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_PTX_SUCCESS;
}

/* PTX Inject CUDA template generators. */
#define S_PTX_GENERATOR_ERROR_RET(ans) do { \
    SecantPTXResult s_ptx_generator_result = (ans); \
    return s_ptx_generator_result; \
} while (0)
#define S_PTX_GENERATOR_CHECK_RET(ans) do { \
    SecantPTXResult s_ptx_generator_check_result = (ans); \
    if (s_ptx_generator_check_result != SECANT_PTX_SUCCESS) { \
        S_PTX_GENERATOR_ERROR_RET(s_ptx_generator_check_result); \
    } \
} while (0)

static SecantPTXResult
_secant_ptx_materialize_emit_site(
    size_t kernel_idx,
    size_t ast_idx,
    size_t num_inputs,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t input_idx;

    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "    {\n"
        "        float result;\n"
        "        asm volatile(\n"
        "            \"{\\n\\t\"\n"
        "            \".reg .f32 %%%%_x0;\\n\\t\"\n"));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            \".reg .f32 %%%%_x%zu;\\n\\t\"\n",
            input_idx + 1u));
    }
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            \"mov.f32 %%%%_x%zu, %%%zu;\\n\\t\"\n",
            input_idx + 1u,
            input_idx + 1u));
    }

    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "            \"// PTX_INJECT_START secant_expr_%03zu_%03zu\\n\\t\"\n"
        "            \"// _x0 o f32 F32 result\\n\\t\"\n",
        kernel_idx,
        ast_idx));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            \"// _x%zu i f32 F32 input%zu\\n\\t\"\n",
            input_idx + 1u,
            input_idx));
    }
    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "            \"// PTX_INJECT_END\\n\\t\"\n"
        "            \"mov.f32 %%0, %%%%_x0;\\n\\t\"\n"
        "            \"}\"\n"
        "            : \"=f\"(result)\n"
        "            : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "%s\"f\"(input%zu)",
            input_idx == 0u ? "" : ", ",
            input_idx));
    }
    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        ");\n"
        "        output[(unsigned long long)%zu * output_leading_dimension + row] = result;\n"
        "    }\n\n",
        ast_idx));

    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_materialize_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t ast_idx;
    size_t input_idx;

    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_static_column_materialize_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    unsigned long long input_leading_dimension,\n"
        "    unsigned long long num_rows,\n"
        "    float* __restrict__ output,\n"
        "    unsigned long long output_leading_dimension\n"
        ") {\n"
        "    const unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (row >= num_rows) {\n"
        "        return;\n"
        "    }\n\n",
        kernel_idx));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "    const float input%zu = input[(unsigned long long)%zu * input_leading_dimension + row];\n",
            input_idx,
            input_idx));
    }
    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_materialize_emit_site(
            kernel_idx,
            ast_idx,
            num_inputs,
            buffer,
            buffer_size,
            offset));
    }

    S_PTX_GENERATOR_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "}\n\n"));
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_materialize_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u ||
        cuda_size_ret == NULL || num_kernels > SIZE_MAX / asts_per_kernel) {
        S_PTX_GENERATOR_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_GENERATOR_CHECK_RET(_secant_ptx_materialize_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            buffer,
            buffer_size,
            &offset));
    }

    S_PTX_GENERATOR_ERROR_RET(_secant_ptx_write_finish(buffer, buffer_size, offset, cuda_size_ret));
}

#undef S_PTX_GENERATOR_CHECK_RET
#undef S_PTX_GENERATOR_ERROR_RET

#define S_PTX_SSE_ERROR_RET(ans) do { SecantPTXResult _secant_ptx_sse_result = (ans); return _secant_ptx_sse_result; } while (0)
#define S_PTX_SSE_CHECK_RET(ans) do { \
    SecantPTXResult _secant_ptx_sse_check_result = (ans); \
    if (_secant_ptx_sse_check_result != SECANT_PTX_SUCCESS) { \
        S_PTX_SSE_ERROR_RET(_secant_ptx_sse_check_result); \
    } \
} while (0)

static int
_secant_ptx_sse_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static SecantPTXResult
_secant_ptx_sse_emit_site(
    size_t kernel_idx,
    size_t ast_idx,
    size_t num_inputs,
    size_t num_targets,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t input_idx;
    size_t target_idx;

    S_PTX_SSE_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "        {\n"
        "            float prediction;\n"
        "            asm volatile(\n"
        "                \"{\\n\\t\"\n"
        "                \".reg .f32 %%%%_x0;\\n\\t\"\n"));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "                \".reg .f32 %%%%_x%zu;\\n\\t\"\n",
            input_idx + 1u));
    }
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "                \"mov.f32 %%%%_x%zu, %%%zu;\\n\\t\"\n",
            input_idx + 1u,
            input_idx + 1u));
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "                \"// PTX_INJECT_START secant_expr_%03zu_%03zu\\n\\t\"\n"
        "                \"// _x0 o f32 F32 result\\n\\t\"\n",
        kernel_idx,
        ast_idx));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "                \"// _x%zu i f32 F32 input%zu\\n\\t\"\n",
            input_idx + 1u,
            input_idx));
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "                \"// PTX_INJECT_END\\n\\t\"\n"
        "                \"mov.f32 %%0, %%%%_x0;\\n\\t\"\n"
        "                \"}\"\n"
        "                : \"=f\"(prediction)\n"
        "                : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "%s\"f\"(input%zu)",
            input_idx == 0u ? "" : ", ",
            input_idx));
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, ");\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            const float error_%03zu = prediction - target%zu;\n"
            "            sse_%03zu_%03zu = error_%03zu * error_%03zu + sse_%03zu_%03zu;\n",
            target_idx,
            target_idx,
            ast_idx,
            target_idx,
            target_idx,
            target_idx,
            ast_idx,
            target_idx));
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "        }\n\n"));
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_sse_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t num_warps = threads_per_block / 32u;
    const size_t num_pairs = asts_per_kernel * num_targets;
    const size_t num_partials = num_pairs * SECANT_PTX_SSE_MAX_WARPS;
    size_t ast_idx;
    size_t target_idx;
    size_t input_idx;

    S_PTX_SSE_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_static_column_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    unsigned long long input_leading_dimension,\n"
        "    const float* __restrict__ targets,\n"
        "    unsigned long long targets_leading_dimension,\n"
        "    unsigned long long num_rows,\n"
        "    float* __restrict__ output_sse,\n"
        "    unsigned long long output_leading_dimension\n"
        ") {\n"
        "    __shared__ float partial_sse[%zu];\n"
        "    const unsigned int lane = threadIdx.x & 31u;\n"
        "    const unsigned int warp = threadIdx.x >> 5u;\n"
        "    const unsigned long long tile_begin = (unsigned long long)blockIdx.x * %zuu;\n"
        "    const unsigned long long remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const unsigned long long tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n",
        kernel_idx,
        num_partials,
        tile_rows,
        tile_rows,
        tile_rows));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "    float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "    for (unsigned long long tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const unsigned long long row = tile_begin + tile_row;\n"));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = input[(unsigned long long)%zu * input_leading_dimension + row];\n",
            input_idx,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const float target%zu = targets[(unsigned long long)%zu * targets_leading_dimension + row];\n",
            target_idx,
            target_idx));
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_sse_emit_site(
            kernel_idx,
            ast_idx,
            num_inputs,
            num_targets,
            buffer,
            buffer_size,
            offset));
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "    }\n\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "    for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                "        sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, sse_%03zu_%03zu, delta);\n"
                "    }\n"
                "    if (lane == 0u) {\n"
                "        partial_sse[(unsigned long long)%zu * %uu + warp] = sse_%03zu_%03zu;\n"
                "    }\n",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx,
                pair_idx,
                SECANT_PTX_SSE_MAX_WARPS,
                ast_idx,
                target_idx));
        }
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "    __syncthreads();\n"
        "    if (warp == 0u) {\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "        float block_sse_%03zu_%03zu = lane < %zuu ? "
                    "partial_sse[(unsigned long long)%zu * %uu + lane] : 0.0f;\n"
                "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                "            block_sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, block_sse_%03zu_%03zu, delta);\n"
                "        }\n"
                "        if (lane == 0u) {\n",
                ast_idx,
                target_idx,
                num_warps,
                pair_idx,
                SECANT_PTX_SSE_MAX_WARPS,
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
            S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "            atomicAdd(output_sse + (unsigned long long)%zu * output_leading_dimension + %zuu, "
                    "block_sse_%03zu_%03zu);\n",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
            S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "        }\n"));
        }
    }
    S_PTX_SSE_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    size_t total_asts;
    size_t num_pairs;
    size_t num_partials;
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u || num_targets == 0u ||
        tile_rows == 0u || threads_per_block < 32u ||
        threads_per_block > tile_rows ||
        threads_per_block > SECANT_PTX_SSE_MAX_WARPS * 32u ||
        threads_per_block % 32u != 0u ||
        cuda_size_ret == NULL) {
        S_PTX_SSE_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    if (!_secant_ptx_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts) ||
        !_secant_ptx_sse_checked_mul(asts_per_kernel, num_targets, &num_pairs) ||
        !_secant_ptx_sse_checked_mul(num_pairs, SECANT_PTX_SSE_MAX_WARPS, &num_partials)) {
        S_PTX_SSE_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    (void)total_asts;
    (void)num_partials;

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_sse_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            num_targets,
            tile_rows,
            threads_per_block,
            buffer,
            buffer_size,
            &offset));
    }
    S_PTX_SSE_ERROR_RET(_secant_ptx_write_finish(buffer, buffer_size, offset, cuda_size_ret));
}

#undef S_PTX_SSE_CHECK_RET
#undef S_PTX_SSE_ERROR_RET

#define S_PTX_DYNAMIC_ERROR_RET(ans) do { \
    SecantPTXResult _secant_ptx_dynamic_result = (ans); \
    return _secant_ptx_dynamic_result; \
} while (0)
#define S_PTX_DYNAMIC_CHECK_RET(ans) do { \
    SecantPTXResult _secant_ptx_dynamic_check_result = (ans); \
    if (_secant_ptx_dynamic_check_result != SECANT_PTX_SUCCESS) { \
        S_PTX_DYNAMIC_ERROR_RET(_secant_ptx_dynamic_check_result); \
    } \
} while (0)

static int
_secant_ptx_dynamic_checked_add(
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

static SecantPTXResult
_secant_ptx_dynamic_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t input_tile_elements = num_input_columns * tile_rows;
    const size_t row_tile_elements = (num_input_columns + num_targets) * tile_rows;
    const size_t num_inputs = num_input_columns + num_input_constants;
    size_t ast_idx;
    size_t constant_idx;
    size_t input_idx;
    size_t target_idx;

    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_dynamic_constant_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    unsigned long long input_leading_dimension,\n"
        "    const float* __restrict__ constant_settings,\n"
        "    unsigned long long constants_leading_dimension,\n"
        "    unsigned long long num_settings,\n"
        "    const float* __restrict__ targets,\n"
        "    unsigned long long targets_leading_dimension,\n"
        "    unsigned long long num_rows,\n"
        "    float* __restrict__ output_sse,\n"
        "    unsigned long long output_leading_dimension\n"
        ") {\n"
        "    __shared__ float row_tile[%zu];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + %zu;\n"
        "    const unsigned long long tile_begin = (unsigned long long)blockIdx.x * %zuu;\n"
        "    const unsigned long long remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const unsigned long long tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n"
        "    for (unsigned long long tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const unsigned long long row = tile_begin + tile_row;\n",
        kernel_idx,
        row_tile_elements,
        input_tile_elements,
        tile_rows,
        tile_rows,
        tile_rows));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        input_tile[(unsigned long long)%zu * %zuu + tile_row] = "
                "input[(unsigned long long)%zu * input_leading_dimension + row];\n",
            input_idx,
            tile_rows,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        target_tile[(unsigned long long)%zu * %zuu + tile_row] = "
                "targets[(unsigned long long)%zu * targets_leading_dimension + row];\n",
            target_idx,
            tile_rows,
            target_idx));
    }
    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    for (unsigned long long setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
    for (constant_idx = 0u;
         constant_idx < num_input_constants;
         ++constant_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = "
                "constant_settings[(unsigned long long)%zu * constants_leading_dimension + setting];\n",
            num_input_columns + constant_idx,
            constant_idx));
    }
    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "        float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "        for (unsigned long long eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            const float input%zu = input_tile[(unsigned long long)%zu * %zuu + eval_row];\n",
            input_idx,
            input_idx,
            tile_rows));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            const float target%zu = target_tile[(unsigned long long)%zu * %zuu + eval_row];\n",
            target_idx,
            target_idx,
            tile_rows));
    }
    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_sse_emit_site(
            kernel_idx,
            ast_idx,
            num_inputs,
            num_targets,
            buffer,
            buffer_size,
            offset));
    }
    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "        }\n\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "        atomicAdd(output_sse + (unsigned long long)%zu * output_leading_dimension + setting, "
                    "sse_%03zu_%03zu);\n",
                pair_idx,
                ast_idx,
                target_idx));
        }
    }
    S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "    }\n"
        "}\n\n"));
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_dynamic_constant_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    size_t num_inputs;
    size_t num_sources;
    size_t row_tile_elements;
    size_t total_asts;
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_input_columns == 0u ||
        num_input_columns >
            SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants >
            SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        num_targets == 0u || tile_rows == 0u ||
        threads_per_block == 0u || threads_per_block > 1024u ||
        cuda_size_ret == NULL ||
        !_secant_ptx_dynamic_checked_add(
            num_input_columns,
            num_input_constants,
            &num_inputs) ||
        !_secant_ptx_dynamic_checked_add(
            num_input_columns,
            num_targets,
            &num_sources) ||
        !_secant_ptx_sse_checked_mul(
            num_kernels,
            asts_per_kernel,
            &total_asts) ||
        !_secant_ptx_sse_checked_mul(
            num_sources,
            tile_rows,
            &row_tile_elements)) {
        S_PTX_DYNAMIC_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    (void)num_inputs;
    (void)total_asts;
    (void)row_tile_elements;

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_dynamic_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_input_constants,
            num_targets,
            tile_rows,
            buffer,
            buffer_size,
            &offset));
    }
    S_PTX_DYNAMIC_ERROR_RET(_secant_ptx_write_finish(
        buffer,
        buffer_size,
        offset,
        cuda_size_ret));
}

#undef S_PTX_DYNAMIC_CHECK_RET
#undef S_PTX_DYNAMIC_ERROR_RET

#define S_PTX_SEARCH_ERROR_RET(ans) do { \
    SecantPTXResult _secant_ptx_search_result = (ans); \
    return _secant_ptx_search_result; \
} while (0)
#define S_PTX_SEARCH_CHECK_RET(ans) do { \
    SecantPTXResult _secant_ptx_search_check_result = (ans); \
    if (_secant_ptx_search_check_result != SECANT_PTX_SUCCESS) { \
        S_PTX_SEARCH_ERROR_RET(_secant_ptx_search_check_result); \
    } \
} while (0)

typedef enum _SecantPTXDynamicLeafStateOwner {
    _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_THREAD = 0,
    _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP = 1
} _SecantPTXDynamicLeafStateOwner;

static SecantPTXResult
_secant_ptx_dynamic_leaf_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    _SecantPTXDynamicLeafStateOwner state_owner,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t shared_stride = num_input_columns | 1u;
    const int warp_owned = state_owner == _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP;
    const size_t num_expression_inputs = num_static_input_columns + num_dynamic_leaves;
    size_t ast_idx;
    size_t input_idx;
    size_t leaf_idx;
    size_t target_idx;

    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_dynamic_leaf_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    unsigned long long num_input_columns,\n"
        "    unsigned long long input_leading_dimension,\n"
        "    const unsigned int* __restrict__ leaf_masks,\n"
        "    const unsigned int* __restrict__ leaf_words,\n"
        "    unsigned long long leaf_words_leading_dimension,\n"
        "    unsigned int num_settings,\n"
        "    unsigned int settings_per_cta,\n"
        "    const float* __restrict__ targets,\n"
        "    unsigned long long targets_leading_dimension,\n"
        "    unsigned long long num_rows,\n"
        "    unsigned long long num_asts,\n"
        "    unsigned long long num_targets,\n"
        "    float* __restrict__ output_sse,\n"
        "    unsigned long long output_leading_dimension\n"
        ") {\n"
        "    extern __shared__ float row_tile[];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + (unsigned long long)%zuu * %zuu;\n"
        "    const unsigned long long tile_begin = (unsigned long long)blockIdx.x * %zuu;\n"
        "    const unsigned long long remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const unsigned long long tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n",
        kernel_idx,
        tile_rows,
        warp_owned ? num_input_columns : shared_stride,
        tile_rows,
        tile_rows,
        tile_rows));
    if (warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "    const unsigned int lane = threadIdx.x & 31u;\n"
            "    const unsigned int warp = threadIdx.x >> 5u;\n"
            "    const unsigned int num_warps = blockDim.x >> 5u;\n\n"));
    } else {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "    const unsigned long long shared_stride = %zuu;\n\n",
            shared_stride));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "    for (unsigned long long tile_row = threadIdx.x; tile_row < %zuu; tile_row += blockDim.x) {\n"
        "        const unsigned long long row = tile_begin + tile_row;\n"
        "        const bool row_valid = row < num_rows;\n"
        "        #pragma unroll\n"
        "        for (unsigned long long column = 0u; column < %zuu; ++column) {\n",
        tile_rows,
        num_input_columns));
    if (warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "            input_tile[column * (unsigned long long)%zuu + tile_row] = "
                "row_valid && column < num_input_columns\n",
            tile_rows));
    } else {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "            input_tile[tile_row * shared_stride + column] = "
                "row_valid && column < num_input_columns\n"));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "                ? input[column * input_leading_dimension + row]\n"
        "                : 0.0f;\n"
        "        }\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        target_tile[(unsigned long long)%zu * %zuu + tile_row] = row_valid\n"
            "            ? targets[(unsigned long long)(%zuu < num_targets ? %zuu : 0u) * "
                "targets_leading_dimension + row]\n"
            "            : 0.0f;\n",
            target_idx,
            tile_rows,
            target_idx,
            target_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    const unsigned int setting_begin = blockIdx.y * settings_per_cta;\n"
        "    const unsigned int setting_count = settings_per_cta < num_settings - setting_begin\n"
        "        ? settings_per_cta : num_settings - setting_begin;\n\n"));
    if (warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "    for (unsigned int setting_offset = warp; setting_offset < setting_count; setting_offset += num_warps) {\n"
            "        const unsigned int setting = setting_begin + setting_offset;\n"));
    } else {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "    for (unsigned int setting_offset = threadIdx.x; setting_offset < setting_count; setting_offset += blockDim.x) {\n"
            "        const unsigned int setting = setting_begin + setting_offset;\n"));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "        const unsigned int leaf_mask = leaf_masks[setting];\n"
        "        const unsigned int* const words = leaf_words + setting * leaf_words_leading_dimension;\n"));
    if (!warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "        const unsigned int input_tile_address =\n"
            "            (unsigned int)__cvta_generic_to_shared((const void*)input_tile);\n"
            "        const unsigned int shared_stride_bytes = (unsigned int)(shared_stride * sizeof(float));\n"));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const unsigned int word%zu = words[%zu];\n"
            "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
            "        float dynamic_input%zu = __uint_as_float(word%zu);\n",
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx));
        if (!warp_owned) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
                leaf_idx, leaf_idx));
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset, "        float sse_%03zu_%03zu = 0.0f;\n", ast_idx, target_idx));
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "        #pragma unroll 1\n"));
    if (warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "        for (unsigned long long eval_row = lane; eval_row < tile_num_rows; eval_row += 32u) {\n"));
    } else {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "        for (unsigned long long eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            const float target%zu = target_tile[(unsigned long long)%zu * %zuu + eval_row];\n",
            target_idx,
            target_idx,
            tile_rows));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        if (warp_owned) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "            const float input%zu = is_column%zu\n"
                "                ? input_tile[(unsigned long long)word%zu * %zuu + eval_row]\n"
                "                : dynamic_input%zu;\n",
                num_static_input_columns + leaf_idx,
                leaf_idx,
                leaf_idx,
                tile_rows,
                leaf_idx));
        } else {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "            asm volatile(\n"
                "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
                "                : \"+f\"(dynamic_input%zu)\n"
                "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
                "                : \"memory\");\n"
                "            const float input%zu = dynamic_input%zu;\n",
                leaf_idx,
                leaf_idx,
                leaf_idx,
                num_static_input_columns + leaf_idx,
                leaf_idx));
        }
    }
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        if (warp_owned) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "            const float input%zu = input_tile[(unsigned long long)%zu * %zuu + eval_row];\n",
                input_idx,
                input_idx,
                tile_rows));
        } else {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "            const float input%zu = input_tile[eval_row * shared_stride + %zuu];\n",
                input_idx,
                input_idx));
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_sse_emit_site(
            kernel_idx,
            ast_idx,
            num_expression_inputs,
            num_targets,
            buffer,
            buffer_size,
            offset));
    }
    if (!warp_owned) {
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "            address%zu += shared_stride_bytes;\n",
                leaf_idx));
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "        }\n\n"));
    if (warp_owned) {
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                    buffer, buffer_size, offset,
                    "        #pragma unroll\n"
                    "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                    "            sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, "
                        "sse_%03zu_%03zu, delta);\n"
                    "        }\n",
                    ast_idx, target_idx, ast_idx, target_idx));
            }
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer,
                buffer_size,
                offset,
                "        if (%s%zuu < num_asts && %zuu < num_targets) {\n"
                "            atomicAdd(output_sse + ((unsigned long long)%zu * num_targets + %zuu) * "
                    "output_leading_dimension + setting, sse_%03zu_%03zu);\n"
                "        }\n",
                warp_owned ? "lane == 0u && " : "",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    _SecantPTXDynamicLeafStateOwner state_owner,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    size_t kernel_idx;
    size_t num_expression_inputs;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        (num_static_input_columns != 0u && num_static_input_columns != num_input_columns) ||
        num_dynamic_leaves == 0u ||
        num_dynamic_leaves > SECANT_PTX_DYNAMIC_LEAF_SSE_MAX_DYNAMIC_LEAVES ||
        num_targets == 0u || tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u || cuda_size_ret == NULL ||
        (state_owner != _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_THREAD &&
         state_owner != _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP) ||
        (state_owner == _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP &&
         threads_per_block % 32u != 0u) ||
        !_secant_ptx_dynamic_checked_add(
            num_static_input_columns, num_dynamic_leaves, &num_expression_inputs) ||
        num_expression_inputs > SECANT_AST_MAX_INPUTS) {
        S_PTX_SEARCH_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_dynamic_leaf_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_static_input_columns,
            num_dynamic_leaves,
            num_targets,
            tile_rows,
            state_owner,
            buffer,
            buffer_size,
            &offset));
    }
    S_PTX_SEARCH_ERROR_RET(_secant_ptx_write_finish(buffer, buffer_size, offset, cuda_size_ret));
}

SecantPTXResult
secant_ptx_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    return _secant_ptx_dynamic_leaf_sse_source_generate(
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_THREAD,
        buffer,
        buffer_size,
        cuda_size_ret);
}

SecantPTXResult
secant_ptx_warp_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    return _secant_ptx_dynamic_leaf_sse_source_generate(
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP,
        buffer,
        buffer_size,
        cuda_size_ret);
}

SecantPTXResult
secant_ptx_dynamic_leaf_lm_statistics_count(
    size_t num_parameters,
    size_t* num_statistics_ret
) {
    if (num_parameters == 0u ||
        num_parameters > SECANT_PTX_DYNAMIC_LEAF_LM_MAX_PARAMETERS ||
        num_statistics_ret == NULL) {
        S_PTX_SEARCH_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *num_statistics_ret = 1u + num_parameters +
        num_parameters * (num_parameters + 1u) / 2u;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_lm_emit_site(
    size_t kernel_idx,
    size_t ast_idx,
    size_t num_inputs,
    size_t num_dynamic_leaves,
    size_t num_targets,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t num_outputs = num_dynamic_leaves + 1u;
    size_t input_idx;
    size_t leaf_idx;
    size_t target_idx;
    size_t lhs;
    size_t rhs;
    size_t stat_idx;

    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "            {\n"
        "                float prediction;\n"));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                float gradient%zu;\n", leaf_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "                asm volatile(\n"
        "                    \"{\\n\\t\"\n"));
    for (input_idx = 0u; input_idx < num_outputs + num_inputs; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                    \".reg .f32 %%%%_x%zu;\\n\\t\"\n", input_idx));
    }
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                    \"mov.f32 %%%%_x%zu, %%%zu;\\n\\t\"\n",
            num_outputs + input_idx,
            num_outputs + input_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "                    \"// PTX_INJECT_START secant_expr_%03zu_%03zu\\n\\t\"\n"
        "                    \"// _x0 o f32 F32 result\\n\\t\"\n",
        kernel_idx, ast_idx));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                    \"// _x%zu o f32 F32 gradient%zu\\n\\t\"\n",
            leaf_idx + 1u, leaf_idx));
    }
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                    \"// _x%zu i f32 F32 input%zu\\n\\t\"\n",
            num_outputs + input_idx, input_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "                    \"// PTX_INJECT_END\\n\\t\"\n"));
    for (leaf_idx = 0u; leaf_idx < num_outputs; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                    \"mov.f32 %%%zu, %%%%_x%zu;\\n\\t\"\n",
            leaf_idx, leaf_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "                    \"}\"\n"
        "                    : \"=f\"(prediction)"));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            ", \"=f\"(gradient%zu)", leaf_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "\n                    : "));
    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "%s\"f\"(input%zu)", input_idx == 0u ? "" : ", ", input_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, ");\n"));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                gradient%zu = is_column%zu ? 0.0f : gradient%zu;\n",
            leaf_idx, leaf_idx, leaf_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "                const float residual%zu = prediction - target%zu;\n"
            "                lm_%03zu_%03zu_000 = fmaf(residual%zu, residual%zu, lm_%03zu_%03zu_000);\n",
            target_idx, target_idx,
            ast_idx, target_idx, target_idx, target_idx, ast_idx, target_idx));
        stat_idx = 1u;
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx, ++stat_idx) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "                lm_%03zu_%03zu_%03zu = fmaf(gradient%zu, residual%zu, lm_%03zu_%03zu_%03zu);\n",
                ast_idx, target_idx, stat_idx, leaf_idx, target_idx,
                ast_idx, target_idx, stat_idx));
        }
        for (lhs = 0u; lhs < num_dynamic_leaves; ++lhs) {
            for (rhs = lhs; rhs < num_dynamic_leaves; ++rhs, ++stat_idx) {
                S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                    buffer, buffer_size, offset,
                    "                lm_%03zu_%03zu_%03zu = fmaf(gradient%zu, gradient%zu, lm_%03zu_%03zu_%03zu);\n",
                    ast_idx, target_idx, stat_idx, lhs, rhs,
                    ast_idx, target_idx, stat_idx));
            }
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "            }\n"));
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_dynamic_leaf_lm_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t num_statistics,
    size_t tile_rows,
    _SecantPTXDynamicLeafStateOwner state_owner,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const int warp_owned = state_owner == _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP;
    const size_t shared_stride = num_input_columns | 1u;
    const size_t num_inputs = num_static_input_columns + num_dynamic_leaves;
    size_t ast_idx;
    size_t target_idx;
    size_t input_idx;
    size_t leaf_idx;
    size_t stat_idx;

    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "extern \"C\" __global__\n"
        "void secant_dynamic_leaf_lm_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    unsigned long long num_input_columns,\n"
        "    unsigned long long input_leading_dimension,\n"
        "    const unsigned int* __restrict__ leaf_masks,\n"
        "    const unsigned int* __restrict__ leaf_words,\n"
        "    unsigned long long leaf_words_leading_dimension,\n"
        "    unsigned int num_settings,\n"
        "    unsigned int settings_per_cta,\n"
        "    const float* __restrict__ targets,\n"
        "    unsigned long long targets_leading_dimension,\n"
        "    unsigned long long num_rows,\n"
        "    unsigned long long num_asts,\n"
        "    unsigned long long num_targets,\n"
        "    float* __restrict__ output_statistics,\n"
        "    unsigned long long output_leading_dimension\n"
        ") {\n"
        "    extern __shared__ float row_tile[];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + (unsigned long long)%zuu * %zuu;\n"
        "    const unsigned long long tile_begin = (unsigned long long)blockIdx.x * %zuu;\n"
        "    const unsigned long long remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const unsigned long long tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n",
        kernel_idx, tile_rows,
        warp_owned ? num_input_columns : shared_stride,
        tile_rows, tile_rows, tile_rows));
    if (warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "    const unsigned int lane = threadIdx.x & 31u;\n"
            "    const unsigned int warp = threadIdx.x >> 5u;\n"
            "    const unsigned int num_warps = blockDim.x >> 5u;\n\n"));
    } else {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "    const unsigned long long shared_stride = %zuu;\n\n", shared_stride));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "    for (unsigned long long tile_row = threadIdx.x; tile_row < %zuu; tile_row += blockDim.x) {\n"
        "        const unsigned long long row = tile_begin + tile_row;\n"
        "        const bool row_valid = row < num_rows;\n"
        "        #pragma unroll\n"
        "        for (unsigned long long column = 0u; column < %zuu; ++column) {\n",
        tile_rows, num_input_columns));
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        warp_owned
            ? "            input_tile[column * (unsigned long long)%zuu + tile_row] = row_valid && column < num_input_columns\n"
            : "            input_tile[tile_row * shared_stride + column] = row_valid && column < num_input_columns\n",
        tile_rows));
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "                ? input[column * input_leading_dimension + row] : 0.0f;\n"
        "        }\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "        target_tile[(unsigned long long)%zu * %zuu + tile_row] = row_valid\n"
            "            ? targets[(unsigned long long)(%zuu < num_targets ? %zuu : 0u) * targets_leading_dimension + row]\n"
            "            : 0.0f;\n",
            target_idx, tile_rows, target_idx, target_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    const unsigned int setting_begin = blockIdx.y * settings_per_cta;\n"
        "    const unsigned int setting_count = settings_per_cta < num_settings - setting_begin\n"
        "        ? settings_per_cta : num_settings - setting_begin;\n\n"));
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        warp_owned
            ? "    for (unsigned int setting_offset = warp; setting_offset < setting_count; setting_offset += num_warps) {\n"
              "        const unsigned int setting = setting_begin + setting_offset;\n"
            : "    for (unsigned int setting_offset = threadIdx.x; setting_offset < setting_count; setting_offset += blockDim.x) {\n"
              "        const unsigned int setting = setting_begin + setting_offset;\n"));
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        "        const unsigned int leaf_mask = leaf_masks[setting];\n"
        "        const unsigned int* const words = leaf_words + setting * leaf_words_leading_dimension;\n"));
    if (!warp_owned) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "        const unsigned int input_tile_address =\n"
            "            (unsigned int)__cvta_generic_to_shared((const void*)input_tile);\n"
            "        const unsigned int shared_stride_bytes = (unsigned int)(shared_stride * sizeof(float));\n"));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "        const unsigned int word%zu = words[%zu];\n"
            "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
            "        float dynamic_input%zu = __uint_as_float(word%zu);\n",
            leaf_idx, leaf_idx, leaf_idx, leaf_idx, leaf_idx, leaf_idx));
        if (!warp_owned) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
                leaf_idx, leaf_idx));
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            for (stat_idx = 0u; stat_idx < num_statistics; ++stat_idx) {
                S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                    buffer, buffer_size, offset,
                    "        float lm_%03zu_%03zu_%03zu = 0.0f;\n",
                    ast_idx, target_idx, stat_idx));
            }
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "\n        #pragma unroll 1\n"));
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset,
        warp_owned
            ? "        for (unsigned long long eval_row = lane; eval_row < tile_num_rows; eval_row += 32u) {\n"
            : "        for (unsigned long long eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            "            const float target%zu = target_tile[(unsigned long long)%zu * %zuu + eval_row];\n",
            target_idx, target_idx, tile_rows));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        if (warp_owned) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "            const float input%zu = is_column%zu\n"
                "                ? input_tile[(unsigned long long)word%zu * %zuu + eval_row]\n"
                "                : dynamic_input%zu;\n",
                num_static_input_columns + leaf_idx, leaf_idx, leaf_idx, tile_rows, leaf_idx));
        } else {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "            asm volatile(\n"
                "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
                "                : \"+f\"(dynamic_input%zu)\n"
                "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
                "                : \"memory\");\n"
                "            const float input%zu = dynamic_input%zu;\n",
                leaf_idx, leaf_idx, leaf_idx,
                num_static_input_columns + leaf_idx, leaf_idx));
        }
    }
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset,
            warp_owned
                ? "            const float input%zu = input_tile[(unsigned long long)%zu * %zuu + eval_row];\n"
                : "            const float input%zu = input_tile[eval_row * shared_stride + %zuu];\n",
            input_idx, input_idx, tile_rows));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_lm_emit_site(
            kernel_idx, ast_idx, num_inputs, num_dynamic_leaves, num_targets,
            buffer, buffer_size, offset));
    }
    if (!warp_owned) {
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                buffer, buffer_size, offset,
                "            address%zu += shared_stride_bytes;\n", leaf_idx));
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "        }\n\n"));
    if (warp_owned) {
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                for (stat_idx = 0u; stat_idx < num_statistics; ++stat_idx) {
                    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                        buffer, buffer_size, offset,
                        "        #pragma unroll\n"
                        "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                        "            lm_%03zu_%03zu_%03zu += __shfl_down_sync(0xffffffffu, lm_%03zu_%03zu_%03zu, delta);\n"
                        "        }\n",
                        ast_idx, target_idx, stat_idx, ast_idx, target_idx, stat_idx));
                }
            }
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            for (stat_idx = 0u; stat_idx < num_statistics; ++stat_idx) {
                S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
                    buffer, buffer_size, offset,
                    "        if (%s%zuu < num_asts && %zuu < num_targets) {\n"
                    "            atomicAdd(output_statistics + ((((unsigned long long)%zu * num_targets + %zuu) * %zuu + %zuu) * output_leading_dimension + setting), lm_%03zu_%03zu_%03zu);\n"
                    "        }\n",
                    warp_owned ? "lane == 0u && " : "",
                    ast_idx, target_idx, ast_idx, target_idx,
                    num_statistics, stat_idx, ast_idx, target_idx, stat_idx));
            }
        }
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_dynamic_leaf_lm_source_generate(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    _SecantPTXDynamicLeafStateOwner state_owner,
    char* buffer, size_t buffer_size, size_t* cuda_size_ret
) {
    size_t num_inputs;
    size_t num_statistics;
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_input_columns == 0u || num_input_columns > SECANT_AST_MAX_INPUTS ||
        (num_static_input_columns != 0u && num_static_input_columns != num_input_columns) ||
        num_dynamic_leaves == 0u ||
        num_dynamic_leaves > SECANT_PTX_DYNAMIC_LEAF_LM_MAX_PARAMETERS ||
        num_targets == 0u || tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u || cuda_size_ret == NULL ||
        (state_owner == _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP && threads_per_block % 32u != 0u) ||
        num_dynamic_leaves > SIZE_MAX - num_static_input_columns) {
        S_PTX_SEARCH_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    num_inputs = num_static_input_columns + num_dynamic_leaves;
    if (num_inputs > SECANT_AST_MAX_INPUTS) {
        S_PTX_SEARCH_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    S_PTX_SEARCH_CHECK_RET(secant_ptx_dynamic_leaf_lm_statistics_count(
        num_dynamic_leaves, &num_statistics));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_dynamic_leaf_lm_emit_kernel(
            kernel_idx, asts_per_kernel, num_input_columns,
            num_static_input_columns, num_dynamic_leaves, num_targets,
            num_statistics, tile_rows, state_owner,
            buffer, buffer_size, &offset));
    }
    S_PTX_SEARCH_ERROR_RET(_secant_ptx_write_finish(
        buffer, buffer_size, offset, cuda_size_ret));
}

SecantPTXResult
secant_ptx_dynamic_leaf_lm_source_generate(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    char* buffer, size_t buffer_size, size_t* cuda_size_ret
) {
    return _secant_ptx_dynamic_leaf_lm_source_generate(
        num_kernels, asts_per_kernel, num_input_columns, num_static_input_columns,
        num_dynamic_leaves, num_targets, tile_rows, threads_per_block,
        _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_THREAD,
        buffer, buffer_size, cuda_size_ret);
}

SecantPTXResult
secant_ptx_warp_dynamic_leaf_lm_source_generate(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    char* buffer, size_t buffer_size, size_t* cuda_size_ret
) {
    return _secant_ptx_dynamic_leaf_lm_source_generate(
        num_kernels, asts_per_kernel, num_input_columns, num_static_input_columns,
        num_dynamic_leaves, num_targets, tile_rows, threads_per_block,
        _SECANT_PTX_DYNAMIC_LEAF_STATE_OWNER_WARP,
        buffer, buffer_size, cuda_size_ret);
}

static SecantPTXResult
_secant_ptx_packed_optimizer_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    static const char* const lanes[] = {"x", "y", "z", "w"};
    const size_t num_inputs = num_input_columns + num_input_constants;
    const size_t tile_elements = (num_input_columns + 1u) * tile_rows;
    size_t ast_idx;
    size_t constant_idx;
    size_t group_idx;
    size_t input_idx;

    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_packed_constant_optimizer_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    unsigned long long input_leading_dimension,\n"
        "    unsigned long long num_settings,\n"
        "    const float* __restrict__ target,\n"
        "    unsigned long long num_rows,\n"
        "    unsigned long long seed,\n"
        "    unsigned long long generation,\n"
        "    unsigned long long iteration,\n"
        "    unsigned long long num_asts,\n"
        "    float* __restrict__ output_sse,\n"
        "    unsigned long long output_leading_dimension\n"
        ") {\n"
        "    __shared__ float row_tile[%zu];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + %zu;\n"
        "    const unsigned long long tile_begin = (unsigned long long)blockIdx.x * %zuu;\n"
        "    const unsigned long long remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const unsigned long long tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n"
        "    for (unsigned long long tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const unsigned long long row = tile_begin + tile_row;\n",
        kernel_idx,
        tile_elements,
        num_input_columns * tile_rows,
        tile_rows,
        tile_rows,
        tile_rows));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        input_tile[(unsigned long long)%zu * %zuu + tile_row] = "
                "input[(unsigned long long)%zu * input_leading_dimension + row];\n",
            input_idx,
            tile_rows,
            input_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "        target_tile[tile_row] = target[row];\n"
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    for (unsigned long long setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
    for (group_idx = 0u; group_idx < (num_input_constants + 3u) / 4u; ++group_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const uint4 random%zu = setting == 0u ? make_uint4(0u, 0u, 0u, 0u) :\n"
            "            secant_packed_eval_random4(setting, seed, generation, iteration, %zuu);\n",
            group_idx,
            group_idx));
    }
    for (constant_idx = 0u; constant_idx < num_input_constants; ++constant_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = setting == 0u ? 0.0f :\n"
            "            secant_packed_eval_delta(random%zu.%s);\n",
            num_input_columns + constant_idx,
            constant_idx / 4u,
            lanes[constant_idx % 4u]));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer, buffer_size, offset, "        float sse_%03zu_000 = 0.0f;\n", ast_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "        #pragma unroll 1\n"
        "        for (unsigned long long eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "            const float input%zu = input_tile[(unsigned long long)%zu * %zuu + eval_row];\n",
            input_idx,
            input_idx,
            tile_rows));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer, buffer_size, offset, "            const float target0 = target_tile[eval_row];\n\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_sse_emit_site(
            kernel_idx,
            ast_idx,
            num_inputs,
            1u,
            buffer,
            buffer_size,
            offset));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "        }\n\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        if (%zuu < num_asts) {\n"
            "            atomicAdd(output_sse + (unsigned long long)%zu * output_leading_dimension + setting, "
                "sse_%03zu_000);\n"
            "        }\n",
            ast_idx,
            ast_idx,
            ast_idx));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_packed_constant_optimizer_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    const char* reducer_source;
    size_t reducer_source_size;
    size_t num_inputs;
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants > SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        tile_rows == 0u || threads_per_block == 0u || threads_per_block > 1024u ||
        cuda_size_ret == NULL ||
        !_secant_ptx_dynamic_checked_add(num_input_columns, num_input_constants, &num_inputs) ||
        num_inputs > SECANT_AST_MAX_INPUTS) {
        S_PTX_SEARCH_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    reducer_source = secant_cuda_packed_constant_optimizer_reduce_f32_source_get(&reducer_source_size);
    if (reducer_source == NULL || reducer_source_size < 2u ||
        reducer_source[reducer_source_size - 1u] != '\0') {
        S_PTX_SEARCH_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
    }
    *cuda_size_ret = 0u;
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(
        buffer,
        buffer_size,
        &offset,
        "static __forceinline__ __device__ uint4 secant_packed_eval_philox_round(uint4 c, uint2 k) {\n"
        "    const unsigned int hi0 = __umulhi(0xd2511f53u, c.x);\n"
        "    const unsigned int hi1 = __umulhi(0xcd9e8d57u, c.z);\n"
        "    const unsigned int lo0 = 0xd2511f53u * c.x;\n"
        "    const unsigned int lo1 = 0xcd9e8d57u * c.z;\n"
        "    return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_packed_eval_philox4x32_10(uint4 c, uint2 k) {\n"
        "    #pragma unroll\n"
        "    for (int round = 0; round < 10; ++round) {\n"
        "        c = secant_packed_eval_philox_round(c, k);\n"
        "        k.x += 0x9e3779b9u;\n"
        "        k.y += 0xbb67ae85u;\n"
        "    }\n"
        "    return c;\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_packed_eval_random4(\n"
        "    unsigned long long setting, unsigned long long seed, unsigned long long generation,\n"
        "    unsigned long long iteration, unsigned int group) {\n"
        "    const uint4 counter = make_uint4((unsigned int)setting, (unsigned int)(setting >> 32), 0u, 0u);\n"
        "    const uint2 key = make_uint2(\n"
        "        (unsigned int)seed ^ (unsigned int)generation * 0x9e3779b9u ^\n"
        "            (unsigned int)(iteration >> 32) * 0x85ebca6bu ^ group * 0x27d4eb2du,\n"
        "        (unsigned int)(seed >> 32) ^ (unsigned int)(generation >> 32) * 0xbb67ae85u ^\n"
        "            (unsigned int)iteration * 0xc2b2ae35u ^ group * 0x165667b1u);\n"
        "    return secant_packed_eval_philox4x32_10(counter, key);\n"
        "}\n\n"
        "static __forceinline__ __device__ float secant_packed_eval_delta(unsigned int word) {\n"
        "    return (float)(word >> 8) * 1.1920928955078125e-7f - 1.0f;\n"
        "}\n\n"));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_SEARCH_CHECK_RET(_secant_ptx_packed_optimizer_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_input_constants,
            tile_rows,
            buffer,
            buffer_size,
            &offset));
    }
    S_PTX_SEARCH_CHECK_RET(_secant_ptx_write(buffer, buffer_size, &offset, "\n%s", reducer_source));
    S_PTX_SEARCH_ERROR_RET(_secant_ptx_write_finish(buffer, buffer_size, offset, cuda_size_ret));
}

#undef S_PTX_SEARCH_CHECK_RET
#undef S_PTX_SEARCH_ERROR_RET

#undef _SECANT_PTX_ERROR_RET
