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
#define S_PTX_GENERATOR_ERROR_RET(ans) do { SecantPTXResult s_ptx_generator_result = (ans); return s_ptx_generator_result; } while (0)
#define S_PTX_GENERATOR_CHECK_RET(ans) do { SecantPTXResult s_ptx_generator_check_result = (ans); if (s_ptx_generator_check_result != SECANT_PTX_SUCCESS) { S_PTX_GENERATOR_ERROR_RET(s_ptx_generator_check_result); } } while (0)

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
#define S_PTX_SSE_CHECK_RET(ans) do { SecantPTXResult _secant_ptx_sse_check_result = (ans); if (_secant_ptx_sse_check_result != SECANT_PTX_SUCCESS) { S_PTX_SSE_ERROR_RET(_secant_ptx_sse_check_result); } } while (0)

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
    SecantSSEReductionMode reduction_mode,
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
                "        float block_sse_%03zu_%03zu = lane < %zuu ? partial_sse[(unsigned long long)%zu * %uu + lane] : 0.0f;\n"
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
            if (reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_ATOMIC) {
                S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                    buffer,
                    buffer_size,
                    offset,
                    "            atomicAdd(output_sse + (unsigned long long)%zu * output_leading_dimension + %zuu, block_sse_%03zu_%03zu);\n",
                    ast_idx,
                    target_idx,
                    ast_idx,
                    target_idx));
            } else {
                const size_t global_pair =
                    kernel_idx * num_pairs + pair_idx;

                S_PTX_SSE_CHECK_RET(_secant_ptx_write(
                    buffer,
                    buffer_size,
                    offset,
                    "            const unsigned long long num_tiles = (num_rows + %zuu - 1u) / %zuu;\n"
                    "            output_sse[(unsigned long long)%zu * num_tiles + blockIdx.x] = block_sse_%03zu_%03zu;\n",
                    tile_rows,
                    tile_rows,
                    global_pair,
                    ast_idx,
                    target_idx));
            }
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
    SecantSSEReductionMode reduction_mode,
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
        (reduction_mode != SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         reduction_mode != SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
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
            reduction_mode,
            buffer,
            buffer_size,
            &offset));
    }
    if (reduction_mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        S_PTX_SSE_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_static_column_sse_reduce(\n"
            "    const float* __restrict__ workspace,\n"
            "    unsigned long long num_tiles,\n"
            "    float* __restrict__ output_sse,\n"
            "    unsigned long long output_leading_dimension\n"
            ") {\n"
            "    const unsigned long long result = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (result < %zuu) {\n"
            "        float sum = 0.0f;\n"
            "        for (unsigned long long tile = 0u; tile < num_tiles; ++tile) {\n"
            "            sum += workspace[result * num_tiles + tile];\n"
            "        }\n"
            "        const unsigned long long ast = result / %zuu;\n"
            "        const unsigned long long target = result %% %zuu;\n"
            "        output_sse[ast * output_leading_dimension + target] = sum;\n"
            "    }\n"
            "}\n\n",
            num_kernels * num_pairs,
            num_targets,
            num_targets));
    }
    S_PTX_SSE_ERROR_RET(_secant_ptx_write_finish(buffer, buffer_size, offset, cuda_size_ret));
}

#undef S_PTX_SSE_CHECK_RET
#undef S_PTX_SSE_ERROR_RET

#define S_PTX_DYNAMIC_ERROR_RET(ans) do { SecantPTXResult _secant_ptx_dynamic_result = (ans); return _secant_ptx_dynamic_result; } while (0)
#define S_PTX_DYNAMIC_CHECK_RET(ans) do { SecantPTXResult _secant_ptx_dynamic_check_result = (ans); if (_secant_ptx_dynamic_check_result != SECANT_PTX_SUCCESS) { S_PTX_DYNAMIC_ERROR_RET(_secant_ptx_dynamic_check_result); } } while (0)

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
    SecantSSEReductionMode reduction_mode,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t input_tile_elements = num_input_columns * tile_rows;
    const size_t row_tile_elements =
        (num_input_columns + num_targets) * tile_rows;
    const size_t num_inputs =
        num_input_columns + num_input_constants;
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
            "        input_tile[(unsigned long long)%zu * %zuu + tile_row] = input[(unsigned long long)%zu * input_leading_dimension + row];\n",
            input_idx,
            tile_rows,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        target_tile[(unsigned long long)%zu * %zuu + tile_row] = targets[(unsigned long long)%zu * targets_leading_dimension + row];\n",
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
            "        const float input%zu = constant_settings[(unsigned long long)%zu * constants_leading_dimension + setting];\n",
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
    if (reduction_mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            offset,
            "        const unsigned long long workspace_num_tiles = "
                "(num_rows + %zuu - 1u) / %zuu;\n",
            tile_rows,
            tile_rows));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            if (reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_ATOMIC) {
                S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
                    buffer,
                    buffer_size,
                    offset,
                    "        atomicAdd(output_sse + (unsigned long long)%zu * output_leading_dimension + setting, sse_%03zu_%03zu);\n",
                    pair_idx,
                    ast_idx,
                    target_idx));
            } else {
                const size_t global_pair =
                    kernel_idx * asts_per_kernel * num_targets +
                    pair_idx;

                S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
                    buffer,
                    buffer_size,
                    offset,
                    "        output_sse[((unsigned long long)%zu * "
                        "num_settings + setting) * workspace_num_tiles + "
                        "blockIdx.x] = sse_%03zu_%03zu;\n",
                    global_pair,
                    ast_idx,
                    target_idx));
            }
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
    SecantSSEReductionMode reduction_mode,
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
        (reduction_mode != SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         reduction_mode != SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
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
            reduction_mode,
            buffer,
            buffer_size,
            &offset));
    }
    if (reduction_mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        S_PTX_DYNAMIC_CHECK_RET(_secant_ptx_write(
            buffer,
            buffer_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_dynamic_constant_sse_reduce(\n"
            "    const float* __restrict__ workspace,\n"
            "    unsigned long long num_tiles,\n"
            "    unsigned long long num_settings,\n"
            "    float* __restrict__ output_sse,\n"
            "    unsigned long long output_leading_dimension\n"
            ") {\n"
            "    const unsigned long long result = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    const unsigned long long num_results = %zuu * num_settings;\n"
            "    if (result < num_results) {\n"
            "        float sum = 0.0f;\n"
            "        for (unsigned long long tile = 0u; tile < num_tiles; ++tile) {\n"
            "            sum += workspace[result * num_tiles + tile];\n"
            "        }\n"
            "        const unsigned long long pair = result / num_settings;\n"
            "        const unsigned long long setting = result %% num_settings;\n"
            "        output_sse[pair * output_leading_dimension + setting] = sum;\n"
            "    }\n"
            "}\n\n",
            num_kernels * asts_per_kernel * num_targets));
    }
    S_PTX_DYNAMIC_ERROR_RET(_secant_ptx_write_finish(
        buffer,
        buffer_size,
        offset,
        cuda_size_ret));
}

#undef S_PTX_DYNAMIC_CHECK_RET
#undef S_PTX_DYNAMIC_ERROR_RET

#undef _SECANT_PTX_ERROR_RET
