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

#include "secant_cuda.h"

#include <stdarg.h>
#include <stdio.h>

#define S_PTX_LM_ERROR_RET(ans) do { \
    SecantPTXResult lm_result_ = (ans); \
    return lm_result_; \
} while (0)
#define S_PTX_LM_CHECK_RET(ans) do { \
    SecantPTXResult lm_check_result_ = (ans); \
    if (lm_check_result_ != SECANT_PTX_SUCCESS) { \
        S_PTX_LM_ERROR_RET(lm_check_result_); \
    } \
} while (0)

enum {
    S_PTX_LM_PARAMETERS = SECANT_PTX_LM_OPTIMIZER_PARAMETERS,
    S_PTX_LM_STATISTICS = SECANT_PTX_LM_OPTIMIZER_STATISTICS
};

static SecantPTXResult
s_ptx_lm_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        required = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        required = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (required < 0) {
        S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    *offset += (size_t)required;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
s_ptx_lm_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
s_ptx_lm_cuda_result_map(SecantCUDAResult result) {
    switch (result) {
        case SECANT_CUDA_SUCCESS:
            return SECANT_PTX_SUCCESS;
        case SECANT_CUDA_ERROR_INVALID_VALUE:
            return SECANT_PTX_ERROR_INVALID_VALUE;
        case SECANT_CUDA_ERROR_OVERFLOW:
            return SECANT_PTX_ERROR_OVERFLOW;
        case SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER:
            return SECANT_PTX_ERROR_INSUFFICIENT_BUFFER;
        case SECANT_CUDA_ERROR_FORMAT:
            return SECANT_PTX_ERROR_FORMAT;
        default:
            return SECANT_PTX_ERROR_INVALID_STATE;
    }
}

static SecantPTXResult
s_ptx_lm_site_emit(
    size_t kernel_idx,
    size_t num_inputs,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t num_outputs = S_PTX_LM_PARAMETERS + 1u;
    size_t idx;

    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "            {\n"
        "                float prediction;\n"));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                float gradient%zu;\n",
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "                asm volatile(\n"
        "                    \"{\\n\\t\"\n"));
    for (idx = 0u; idx < num_outputs + num_inputs; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                    \".reg .f32 %%%%_x%zu;\\n\\t\"\n",
            idx));
    }
    for (idx = 0u; idx < num_inputs; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                    \"mov.f32 %%%%_x%zu, %%%zu;\\n\\t\"\n",
            num_outputs + idx,
            num_outputs + idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "                    \"// PTX_INJECT_START secant_expr_%03zu_000\\n\\t\"\n"
        "                    \"// _x0 o f32 F32 result\\n\\t\"\n",
        kernel_idx));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                    \"// _x%zu o f32 F32 gradient%zu\\n\\t\"\n",
            idx + 1u,
            idx));
    }
    for (idx = 0u; idx < num_inputs; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                    \"// _x%zu i f32 F32 input%zu\\n\\t\"\n",
            num_outputs + idx,
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "                    \"// PTX_INJECT_END\\n\\t\"\n"));
    for (idx = 0u; idx < num_outputs; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                    \"mov.f32 %%%zu, %%%%_x%zu;\\n\\t\"\n",
            idx,
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "                    \"}\"\n"
        "                    : \"=f\"(prediction)"));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            ", \"=f\"(gradient%zu)",
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "\n                    : "));
    for (idx = 0u; idx < num_inputs; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "%s\"f\"(input%zu)",
            idx == 0u ? "" : ", ",
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        ");\n"
        "                const float residual = prediction - target_value;\n"
        "                lm_00 = fmaf(residual, residual, lm_00);\n"));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "                lm_%02zu = fmaf(gradient%zu, residual, lm_%02zu);\n",
            1u + idx,
            idx,
            1u + idx));
    }
    {
        size_t statistic = 1u + S_PTX_LM_PARAMETERS;
        size_t lhs;

        for (lhs = 0u; lhs < S_PTX_LM_PARAMETERS; ++lhs) {
            size_t rhs;

            for (rhs = lhs; rhs < S_PTX_LM_PARAMETERS; ++rhs, ++statistic) {
                S_PTX_LM_CHECK_RET(s_ptx_lm_write(
                    buffer, buffer_size, offset,
                    "                lm_%02zu = fmaf(gradient%zu, gradient%zu, lm_%02zu);\n",
                    statistic,
                    lhs,
                    rhs,
                    statistic));
            }
        }
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "            }\n"));
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
s_ptx_lm_kernel_emit(
    size_t kernel_idx,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t shared_stride = num_input_columns | 1u;
    const size_t site_inputs =
        num_static_input_columns + 3u * S_PTX_LM_PARAMETERS;
    size_t idx;

    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
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
        tile_rows,
        shared_stride,
        shared_stride,
        tile_rows,
        tile_rows,
        tile_rows,
        tile_rows,
        num_input_columns));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "        const float input%zu = setting_constants[%zu];\n"
            "        const unsigned int word%zu = words[%zu];\n"
            "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
            "        const float input%zu = is_column%zu ? 0.0f : 1.0f;\n"
            "        float input%zu = input%zu;\n"
            "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
            num_static_input_columns + idx,
            idx,
            idx,
            idx,
            idx,
            idx,
            num_static_input_columns + 2u * S_PTX_LM_PARAMETERS + idx,
            idx,
            num_static_input_columns + S_PTX_LM_PARAMETERS + idx,
            num_static_input_columns + idx,
            idx,
            idx));
    }
    for (idx = 0u; idx < S_PTX_LM_STATISTICS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "        float lm_%02zu = 0.0f;\n",
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "        #pragma unroll 1\n"
        "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"
        "            const float target_value = target_tile[eval_row];\n"));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "            asm volatile(\n"
            "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
            "                : \"+f\"(input%zu)\n"
            "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
            "                : \"memory\");\n",
            num_static_input_columns + S_PTX_LM_PARAMETERS + idx,
            idx,
            idx));
    }
    for (idx = 0u; idx < num_static_input_columns; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "            const float input%zu = input_tile[eval_row * shared_stride + %zuu];\n",
            idx,
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_site_emit(
        kernel_idx,
        site_inputs,
        buffer,
        buffer_size,
        offset));
    for (idx = 0u; idx < S_PTX_LM_PARAMETERS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "            address%zu += shared_stride_bytes;\n",
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "        }\n"));
    for (idx = 0u; idx < S_PTX_LM_STATISTICS; ++idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_write(
            buffer, buffer_size, offset,
            "        atomicAdd(statistics + (size_t)%zuu * statistics_leading_dimension + setting, lm_%02zu);\n",
            idx,
            idx));
    }
    S_PTX_LM_CHECK_RET(s_ptx_lm_write(
        buffer, buffer_size, offset,
        "    }\n"
        "}\n\n"));
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_lm_optimizer_source_generate(
    size_t num_kernels,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    size_t offset = 0u;
    size_t kernel_idx;
    size_t solver_size = 0u;
    SecantCUDAResult cuda_result;

    if (num_kernels == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        num_static_input_columns != num_input_columns ||
        num_static_input_columns + S_PTX_LM_PARAMETERS > SECANT_AST_MAX_INPUTS ||
        num_static_input_columns + 3u * S_PTX_LM_PARAMETERS > SECANT_AST_MAX_INPUTS ||
        tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u || cuda_size_ret == NULL) {
        S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_PTX_LM_CHECK_RET(s_ptx_lm_kernel_emit(
            kernel_idx,
            num_input_columns,
            num_static_input_columns,
            tile_rows,
            buffer,
            buffer_size,
            &offset));
    }
    cuda_result = secant_cuda_lm_solver_source_generate(
        buffer == NULL ? NULL : buffer + offset,
        buffer == NULL || offset > buffer_size ? 0u : buffer_size - offset,
        &solver_size);
    if (cuda_result != SECANT_CUDA_SUCCESS) {
        S_PTX_LM_ERROR_RET(s_ptx_lm_cuda_result_map(cuda_result));
    }
    if (solver_size == 0u || solver_size - 1u > SIZE_MAX - offset) {
        S_PTX_LM_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    offset += solver_size - 1u;
    S_PTX_LM_CHECK_RET(s_ptx_lm_finish(
        buffer,
        buffer_size,
        offset,
        cuda_size_ret));
    (void)threads_per_block;
    return SECANT_PTX_SUCCESS;
}

#undef S_PTX_LM_CHECK_RET
#undef S_PTX_LM_ERROR_RET
