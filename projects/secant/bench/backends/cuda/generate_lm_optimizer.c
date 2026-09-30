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
#include "secant_cuda.h"

#include <stdarg.h>
#include <stdio.h>

#define S_LM_ERROR_RET(ans) do { \
    SecantCUDAResult lm_error_result_ = (ans); \
    return lm_error_result_; \
} while (0)
#define S_LM_CHECK_RET(ans) do { \
    SecantCUDAResult r_ = (ans); \
    if (r_ != SECANT_CUDA_SUCCESS) { S_LM_ERROR_RET(r_); } \
} while (0)

enum {
    S_LM_PARAMETERS = 8,
    S_LM_STATISTICS = 45
};

static SecantCUDAResult
s_lm_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int required;

    if (offset == NULL || format == NULL) {
        S_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        required = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        required = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (required < 0) {
        S_LM_ERROR_RET(SECANT_CUDA_ERROR_FORMAT);
    }
    if ((size_t)required > SIZE_MAX - *offset) {
        S_LM_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    *offset += (size_t)required;
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
s_lm_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL || offset == SIZE_MAX) {
        S_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            S_LM_ERROR_RET(SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_CUDA_SUCCESS;
}

static size_t
s_lm_hessian_stat(size_t row, size_t column) {
    size_t lhs = row;
    size_t rhs = column;

    if (lhs > rhs) {
        const size_t swap = lhs;
        lhs = rhs;
        rhs = swap;
    }
    return 1u + S_LM_PARAMETERS +
        lhs * (2u * S_LM_PARAMETERS - lhs + 1u) / 2u + (rhs - lhs);
}

static SecantCUDAResult
s_lm_statistics_kernel_emit(
    size_t kernel_idx,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t shared_stride = num_input_columns | 1u;
    size_t input_idx;
    size_t parameter_idx;
    size_t statistic_idx;
    size_t lhs;
    size_t rhs;

    S_LM_CHECK_RET(s_lm_write(
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
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        const float constant%zu = setting_constants[%zu];\n"
            "        const unsigned int word%zu = words[%zu];\n"
            "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
            "        const float mixed_gradient%zu = is_column%zu ? 0.0f : 1.0f;\n"
            "        float mixed%zu = constant%zu;\n"
            "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
            parameter_idx, parameter_idx,
            parameter_idx, parameter_idx,
            parameter_idx, parameter_idx,
            parameter_idx, parameter_idx,
            parameter_idx, parameter_idx,
            parameter_idx, parameter_idx));
    }
    for (statistic_idx = 0u; statistic_idx < S_LM_STATISTICS; ++statistic_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        float lm_%02zu = 0.0f;\n",
            statistic_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "        #pragma unroll 1\n"
        "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"
        "            const float target_value = target_tile[eval_row];\n"));
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "            asm volatile(\n"
            "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
            "                : \"+f\"(mixed%zu)\n"
            "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
            "                : \"memory\");\n",
            parameter_idx, parameter_idx, parameter_idx));
    }
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "            const float input%zu = input_tile[eval_row * shared_stride + %zuu];\n",
            input_idx, input_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "            float prediction;\n"));
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "            float gradient%zu;\n",
            parameter_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "            secant_lm_eval_%03zu(\n",
        kernel_idx));
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "                input%zu,\n",
            input_idx));
    }
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "                constant%zu,\n",
            parameter_idx));
    }
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "                mixed%zu,\n",
            parameter_idx));
    }
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "                mixed_gradient%zu,\n",
            parameter_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "                prediction"));
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            ", gradient%zu",
            parameter_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        ");\n"
        "            const float residual = prediction - target_value;\n"
        "            lm_00 = fmaf(residual, residual, lm_00);\n"));
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "            lm_%02zu = fmaf(gradient%zu, residual, lm_%02zu);\n",
            1u + parameter_idx, parameter_idx, 1u + parameter_idx));
    }
    statistic_idx = 1u + S_LM_PARAMETERS;
    for (lhs = 0u; lhs < S_LM_PARAMETERS; ++lhs) {
        for (rhs = lhs; rhs < S_LM_PARAMETERS; ++rhs, ++statistic_idx) {
            S_LM_CHECK_RET(s_lm_write(
                buffer, buffer_size, offset,
                "            lm_%02zu = fmaf(gradient%zu, gradient%zu, lm_%02zu);\n",
                statistic_idx, lhs, rhs, statistic_idx));
        }
    }
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "            address%zu += shared_stride_bytes;\n",
            parameter_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "        }\n"));
    for (statistic_idx = 0u; statistic_idx < S_LM_STATISTICS; ++statistic_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        atomicAdd(statistics + (size_t)%zuu * statistics_leading_dimension + setting, lm_%02zu);\n",
            statistic_idx, statistic_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "    }\n"
        "}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
s_lm_cholesky_emit(
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t row;
    size_t column;
    size_t k;

    for (row = 0u; row < S_LM_PARAMETERS; ++row) {
        for (column = 0u; column <= row; ++column) {
            const size_t stat = s_lm_hessian_stat(row, column);

            S_LM_CHECK_RET(s_lm_write(
                buffer, buffer_size, offset,
                "        float sum_%zu_%zu = accepted[(size_t)%zuu * statistics_leading_dimension + setting];\n",
                row, column, stat));
            if (row == column) {
                S_LM_CHECK_RET(s_lm_write(
                    buffer, buffer_size, offset,
                    "        sum_%zu_%zu += lambda_try * fmaxf(fabsf(sum_%zu_%zu), diagonal_floor);\n",
                    row, column, row, column));
            }
            for (k = 0u; k < column; ++k) {
                S_LM_CHECK_RET(s_lm_write(
                    buffer, buffer_size, offset,
                    "        sum_%zu_%zu = fmaf(-l_%zu_%zu, l_%zu_%zu, sum_%zu_%zu);\n",
                    row, column, row, k, column, k, row, column));
            }
            if (row == column) {
                S_LM_CHECK_RET(s_lm_write(
                    buffer, buffer_size, offset,
                    "        const bool pivot_ok_%zu = factor_ok && isfinite(sum_%zu_%zu) && sum_%zu_%zu > pivot_floor;\n"
                    "        const float l_%zu_%zu = pivot_ok_%zu ? sqrtf(sum_%zu_%zu) : 1.0f;\n"
                    "        factor_ok = pivot_ok_%zu;\n",
                    row, row, column, row, column,
                    row, column, row, row, column,
                    row));
            } else {
                S_LM_CHECK_RET(s_lm_write(
                    buffer, buffer_size, offset,
                    "        const float l_%zu_%zu = factor_ok ? sum_%zu_%zu / l_%zu_%zu : 0.0f;\n"
                    "        factor_ok = factor_ok && isfinite(l_%zu_%zu);\n",
                    row, column, row, column, column, column,
                    row, column));
            }
        }
    }
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
s_lm_triangular_solve_emit(
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t row;
    size_t k;

    for (row = 0u; row < S_LM_PARAMETERS; ++row) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        float y_%zu = -accepted[(size_t)%zuu * statistics_leading_dimension + setting];\n",
            row, 1u + row));
        for (k = 0u; k < row; ++k) {
            S_LM_CHECK_RET(s_lm_write(
                buffer, buffer_size, offset,
                "        y_%zu = fmaf(-l_%zu_%zu, y_%zu, y_%zu);\n",
                row, row, k, k, row));
        }
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        y_%zu = factor_ok ? y_%zu / l_%zu_%zu : 0.0f;\n"
            "        factor_ok = factor_ok && isfinite(y_%zu);\n",
            row, row, row, row, row));
    }
    for (row = S_LM_PARAMETERS; row != 0u; --row) {
        const size_t i = row - 1u;

        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        float delta_%zu = y_%zu;\n",
            i, i));
        for (k = i + 1u; k < S_LM_PARAMETERS; ++k) {
            S_LM_CHECK_RET(s_lm_write(
                buffer, buffer_size, offset,
                "        delta_%zu = fmaf(-l_%zu_%zu, delta_%zu, delta_%zu);\n",
                i, k, i, k, i));
        }
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        delta_%zu = factor_ok ? delta_%zu / l_%zu_%zu : 0.0f;\n"
            "        factor_ok = factor_ok && isfinite(delta_%zu);\n",
            i, i, i, i, i));
    }
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
s_lm_predicted_reduction_emit(
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t row;
    size_t column;

    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "        float gradient_dot_step = 0.0f;\n"
        "        float step_hessian_step = 0.0f;\n"));
    for (row = 0u; row < S_LM_PARAMETERS; ++row) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "        gradient_dot_step = fmaf(\n"
            "            accepted[(size_t)%zuu * statistics_leading_dimension + setting],\n"
            "            delta_%zu, gradient_dot_step);\n",
            1u + row, row));
        for (column = row; column < S_LM_PARAMETERS; ++column) {
            const size_t stat = s_lm_hessian_stat(row, column);

            S_LM_CHECK_RET(s_lm_write(
                buffer, buffer_size, offset,
                "        step_hessian_step = fmaf(\n"
                "            accepted[(size_t)%zuu * statistics_leading_dimension + setting],\n"
                "            delta_%zu * delta_%zu * %s, step_hessian_step);\n",
                stat,
                row,
                column,
                row == column ? "1.0f" : "2.0f"));
        }
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "        const float attempt_reduction =\n"
        "            -(2.0f * gradient_dot_step + step_hessian_step);\n"
        "        factor_ok = factor_ok && isfinite(attempt_reduction) && attempt_reduction > 0.0f;\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
s_lm_solver_emit(
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t parameter_idx;

    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "extern \"C\" __global__\n"
        "void secant_lm_solve_f32(\n"
        "    const float* __restrict__ evaluated,\n"
        "    float* __restrict__ accepted,\n"
        "    size_t statistics_leading_dimension,\n"
        "    float* __restrict__ current_constants,\n"
        "    float* __restrict__ proposal_constants,\n"
        "    size_t constants_leading_dimension,\n"
        "    float* __restrict__ damping,\n"
        "    float* __restrict__ predicted_reduction,\n"
        "    unsigned int* __restrict__ status,\n"
        "    size_t state_leading_dimension,\n"
        "    size_t num_asts,\n"
        "    size_t num_settings,\n"
        "    unsigned int initialize,\n"
        "    unsigned int produce_proposal,\n"
        "    unsigned int max_damping_attempts,\n"
        "    float damping_up,\n"
        "    float damping_down,\n"
        "    float minimum_damping,\n"
        "    float maximum_damping,\n"
        "    float diagonal_floor,\n"
        "    float pivot_floor) {\n"
        "    const size_t logical = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    const size_t logical_count = num_asts * num_settings;\n"
        "    if (logical >= logical_count || num_settings == 0u) { return; }\n"
        "    const size_t ast = logical / num_settings;\n"
        "    const size_t setting = logical - ast * num_settings;\n"
        "    const size_t state_index = ast * state_leading_dimension + setting;\n"
        "    evaluated += ast * (size_t)%uu * statistics_leading_dimension;\n"
        "    accepted += ast * (size_t)%uu * statistics_leading_dimension;\n"
        "    current_constants += logical * constants_leading_dimension;\n"
        "    proposal_constants += logical * constants_leading_dimension;\n"
        "    float lambda = fminf(fmaxf(damping[state_index], minimum_damping), maximum_damping);\n"
        "    unsigned int state = 0u;\n"
        "    if (initialize != 0u) {\n"
        "        #pragma unroll\n"
        "        for (size_t statistic = 0u; statistic < %uu; ++statistic) {\n"
        "            accepted[statistic * statistics_leading_dimension + setting] =\n"
        "                evaluated[statistic * statistics_leading_dimension + setting];\n"
        "        }\n"
        "    } else {\n"
        "        const float incumbent_sse = accepted[setting];\n"
        "        const float proposal_sse = evaluated[setting];\n"
        "        const float expected = predicted_reduction[state_index];\n"
        "        const float actual = incumbent_sse - proposal_sse;\n"
        "        const bool accept = isfinite(proposal_sse) && isfinite(actual) &&\n"
        "            isfinite(expected) && actual > 0.0f && expected > 0.0f;\n"
        "        if (accept) {\n"
        "            #pragma unroll\n"
        "            for (size_t parameter = 0u; parameter < %uu; ++parameter) {\n"
        "                current_constants[parameter] = proposal_constants[parameter];\n"
        "            }\n"
        "            #pragma unroll\n"
        "            for (size_t statistic = 0u; statistic < %uu; ++statistic) {\n"
        "                accepted[statistic * statistics_leading_dimension + setting] =\n"
        "                    evaluated[statistic * statistics_leading_dimension + setting];\n"
        "            }\n"
        "            lambda = fmaxf(lambda * damping_down, minimum_damping);\n"
        "            state |= 1u;\n"
        "        } else {\n"
        "            lambda = fminf(lambda * damping_up, maximum_damping);\n"
        "        }\n"
        "    }\n"
        "    if (produce_proposal == 0u) {\n"
        "        damping[state_index] = lambda;\n"
        "        if (status != nullptr) { status[state_index] = state; }\n"
        "        return;\n"
        "    }\n"
        "    bool step_ready = false;\n"
        "    float lambda_try = lambda;\n"
        "    unsigned int attempts = 0u;\n"
        "    for (; attempts < max_damping_attempts && !step_ready; ++attempts) {\n"
        "        bool factor_ok = isfinite(lambda_try) && lambda_try >= minimum_damping &&\n"
        "            lambda_try <= maximum_damping;\n",
        S_LM_STATISTICS,
        S_LM_STATISTICS,
        S_LM_STATISTICS,
        S_LM_PARAMETERS,
        S_LM_STATISTICS));
    S_LM_CHECK_RET(s_lm_cholesky_emit(buffer, buffer_size, offset));
    S_LM_CHECK_RET(s_lm_triangular_solve_emit(buffer, buffer_size, offset));
    S_LM_CHECK_RET(s_lm_predicted_reduction_emit(buffer, buffer_size, offset));
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "        if (factor_ok) {\n"
        "            bool constants_ok = true;\n"));
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "            const float proposal_%zu = current_constants[%zu] + delta_%zu;\n"
            "            constants_ok = constants_ok && isfinite(proposal_%zu);\n",
            parameter_idx, parameter_idx, parameter_idx, parameter_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "            if (constants_ok) {\n"));
    for (parameter_idx = 0u; parameter_idx < S_LM_PARAMETERS; ++parameter_idx) {
        S_LM_CHECK_RET(s_lm_write(
            buffer, buffer_size, offset,
            "                proposal_constants[%zu] = proposal_%zu;\n",
            parameter_idx, parameter_idx));
    }
    S_LM_CHECK_RET(s_lm_write(
        buffer, buffer_size, offset,
        "                predicted_reduction[state_index] = attempt_reduction;\n"
        "                step_ready = true;\n"
        "            }\n"
        "        }\n"
        "        if (!step_ready) {\n"
        "            lambda_try = fminf(lambda_try * damping_up, maximum_damping);\n"
        "        }\n"
        "    }\n"
        "    if (!step_ready) {\n"
        "        #pragma unroll\n"
        "        for (size_t parameter = 0u; parameter < %uu; ++parameter) {\n"
        "            proposal_constants[parameter] = current_constants[parameter];\n"
        "        }\n"
        "        predicted_reduction[state_index] = 0.0f;\n"
        "        state |= 4u;\n"
        "    } else {\n"
        "        state |= 2u;\n"
        "    }\n"
        "    damping[state_index] = lambda_try;\n"
        "    if (status != nullptr) { status[state_index] = state | (attempts << 8u); }\n"
        "}\n\n",
        S_LM_PARAMETERS));
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_lm_solver_source_generate(
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
) {
    size_t offset = 0u;

    if (cuda_size_ret == NULL) {
        S_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    S_LM_CHECK_RET(s_lm_solver_emit(buffer, buffer_size, &offset));
    S_LM_CHECK_RET(s_lm_finish(buffer, buffer_size, offset, cuda_size_ret));
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_lm_optimizer_source_generate(
    size_t num_kernels,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    size_t offset = 0u;
    size_t kernel_idx;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        num_static_input_columns != num_input_columns ||
        num_static_input_columns + S_LM_PARAMETERS > SECANT_AST_MAX_INPUTS ||
        tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u || num_routines > SECANT_AST_MAX_ROUTINES ||
        (num_routines != 0u && routines == NULL) || asts == NULL ||
        cuda_size_ret == NULL) {
        S_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        char function_name[64];
        size_t helper_size = 0u;
        uint32_t num_variables = 0u;
        int name_size;

        if (asts[kernel_idx] == NULL) {
            S_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        name_size = snprintf(
            function_name,
            sizeof(function_name),
            "secant_lm_eval_%03zu",
            kernel_idx);
        if (name_size < 0 || (size_t)name_size >= sizeof(function_name)) {
            S_LM_ERROR_RET(SECANT_CUDA_ERROR_FORMAT);
        }
        S_LM_CHECK_RET(secant_cuda_ast_lm_forward_gradient_source_generate(
            function_name,
            num_static_input_columns,
            S_LM_PARAMETERS,
            routines,
            num_routines,
            asts[kernel_idx],
            buffer == NULL ? NULL : buffer + offset,
            buffer == NULL || offset > buffer_size ? 0u : buffer_size - offset,
            &helper_size,
            &num_variables));
        if (helper_size == 0u || helper_size - 1u > SIZE_MAX - offset) {
            S_LM_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
        }
        offset += helper_size - 1u;
        if (num_variables > max_variables) {
            max_variables = num_variables;
        }
    }
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_LM_CHECK_RET(s_lm_statistics_kernel_emit(
            kernel_idx,
            num_input_columns,
            num_static_input_columns,
            tile_rows,
            buffer,
            buffer_size,
            &offset));
    }
    S_LM_CHECK_RET(s_lm_solver_emit(buffer, buffer_size, &offset));
    S_LM_CHECK_RET(s_lm_finish(buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    (void)threads_per_block;
    return SECANT_CUDA_SUCCESS;
}

#undef S_LM_CHECK_RET
#undef S_LM_ERROR_RET
