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
#ifndef IMPLICIT_SINDY_CPU_GRAM_REFERENCE_H
#define IMPLICIT_SINDY_CPU_GRAM_REFERENCE_H

#include "implicit_sindy.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

enum {
    IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS = 0,
    IMPLICIT_SINDY_CPU_GRAM_REFERENCE_ERROR_INVALID_VALUE = 1
};

enum {
    IMPLICIT_SINDY_CPU_GRAM_REFERENCE_FEATURES = 32
};

static float
implicit_sindy_cpu_gram_reference_u32_as_f32(
    uint32_t bits
) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

static uint32_t
implicit_sindy_cpu_gram_reference_i32_as_u32(
    int32_t bits
) {
    union {
        int32_t i;
        uint32_t u;
    } value;
    value.i = bits;
    return value.u;
}

static float
implicit_sindy_cpu_gram_reference_clampf(
    float value,
    float lo,
    float hi
) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float
implicit_sindy_cpu_gram_reference_unary(
    BinaryAstUnaryOp op,
    float x
) {
    const float safe_eps = 1.0e-6f;
    const float log2_e = 1.4426950408889634f;
    const float log10_2 = 0.3010299956639812f;

    switch (op) {
        case BINARY_AST_UNARY_SQUARE_F32:
            return x * x;
        case BINARY_AST_UNARY_CUBE_F32:
            return x * x * x;
        case BINARY_AST_UNARY_NEG_FTZ_F32:
            return -x;
        case BINARY_AST_UNARY_ABS_FTZ_F32:
            return fabsf(x);
        case BINARY_AST_UNARY_RCP_APPROX_FTZ_F32:
            return 1.0f / x;
        case BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32:
            return sqrtf(x);
        case BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32:
            return 1.0f / sqrtf(x);
        case BINARY_AST_UNARY_SIN_APPROX_FTZ_F32:
            return sinf(x);
        case BINARY_AST_UNARY_COS_APPROX_FTZ_F32:
            return cosf(x);
        case BINARY_AST_UNARY_EX2_APPROX_FTZ_F32:
            return exp2f(x);
        case BINARY_AST_UNARY_EXP_APPROX_FTZ_F32:
            return exp2f(x * log2_e);
        case BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32:
            return log2f(x);
        case BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32:
            return log2f(x) * log10_2;
        case BINARY_AST_UNARY_SAFE_RCP_F32:
            return 1.0f / fmaxf(fabsf(x), safe_eps);
        case BINARY_AST_UNARY_SAFE_SQRT_F32:
            return sqrtf(fmaxf(fabsf(x), safe_eps));
        case BINARY_AST_UNARY_SAFE_RSQRT_F32:
            return 1.0f / sqrtf(fmaxf(fabsf(x), safe_eps));
        case BINARY_AST_UNARY_SAFE_EX2_F32:
            return exp2f(implicit_sindy_cpu_gram_reference_clampf(x, -126.0f, 126.0f));
        case BINARY_AST_UNARY_SAFE_EXP_F32:
            return exp2f(implicit_sindy_cpu_gram_reference_clampf(x, -80.0f, 80.0f) * log2_e);
        case BINARY_AST_UNARY_SAFE_LOG2_F32:
            return log2f(fmaxf(fabsf(x), safe_eps));
        case BINARY_AST_UNARY_SAFE_LOG10_F32:
            return log2f(fmaxf(fabsf(x), safe_eps)) * log10_2;
        case BINARY_AST_UNARY_ZERO_F32:
            return 0.0f;
        case BINARY_AST_UNARY_IDENTITY:
        default:
            return x;
    }
}

static float
implicit_sindy_cpu_gram_reference_binary(
    BinaryAstBinaryOp op,
    float lhs,
    float rhs
) {
    const float safe_eps = 1.0e-6f;

    switch (op) {
        case BINARY_AST_BINARY_SUB_FTZ_F32:
            return lhs - rhs;
        case BINARY_AST_BINARY_MUL_FTZ_F32:
            return lhs * rhs;
        case BINARY_AST_BINARY_KEEP_LEFT:
            return lhs;
        case BINARY_AST_BINARY_KEEP_RIGHT:
            return rhs;
        case BINARY_AST_BINARY_DIV_APPROX_FTZ_F32:
            return lhs / rhs;
        case BINARY_AST_BINARY_MIN_FTZ_F32:
            return fminf(lhs, rhs);
        case BINARY_AST_BINARY_MAX_FTZ_F32:
            return fmaxf(lhs, rhs);
        case BINARY_AST_BINARY_SAFE_DIV_F32:
            return lhs / fmaxf(fabsf(rhs), safe_eps);
        case BINARY_AST_BINARY_ADD_FTZ_F32:
        default:
            return lhs + rhs;
    }
}

static void
implicit_sindy_cpu_gram_reference_children(
    int node_idx,
    int* binary_idx,
    int* lhs_idx,
    int* rhs_idx
) {
    if (node_idx < 12) {
        const int local = node_idx - BINARY_AST_NUM_INPUTS;
        *binary_idx = local;
        *lhs_idx = local * 2;
        *rhs_idx = local * 2 + 1;
    } else if (node_idx < 14) {
        const int local = node_idx - 12;
        *binary_idx = 4 + local;
        *lhs_idx = 8 + local * 2;
        *rhs_idx = 8 + local * 2 + 1;
    } else {
        *binary_idx = 6;
        *lhs_idx = 12;
        *rhs_idx = 13;
    }
}

static float
implicit_sindy_cpu_gram_reference_eval_feature(
    const BinaryAST* ast,
    const float* primitive_features,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    int64_t row,
    int32_t leaf_mask,
    const int32_t* leaf_words
) {
    float values[BINARY_AST_NUM_UNARY_OPS];
    uint32_t mask = implicit_sindy_cpu_gram_reference_i32_as_u32(leaf_mask);
    int i;

    for (i = 0; i < BINARY_AST_NUM_INPUTS; ++i) {
        float leaf;
        if (((mask >> i) & 1u) != 0u) {
            const int32_t col = leaf_words[i];
            leaf = (col >= 0 && col < num_primitive_features)
                ? primitive_features[row + (int64_t)col * primitive_feature_stride]
                : 0.0f;
        } else {
            leaf = implicit_sindy_cpu_gram_reference_u32_as_f32(
                implicit_sindy_cpu_gram_reference_i32_as_u32(leaf_words[i])
            );
        }
        values[i] = implicit_sindy_cpu_gram_reference_unary(
            ast->unary[i],
            leaf
        );
    }

    for (i = BINARY_AST_NUM_INPUTS; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
        int binary_idx;
        int lhs_idx;
        int rhs_idx;
        float combined;

        implicit_sindy_cpu_gram_reference_children(
            i,
            &binary_idx,
            &lhs_idx,
            &rhs_idx
        );
        combined = implicit_sindy_cpu_gram_reference_binary(
            ast->binary[binary_idx],
            values[lhs_idx],
            values[rhs_idx]
        );
        values[i] = implicit_sindy_cpu_gram_reference_unary(
            ast->unary[i],
            combined
        );
    }

    return values[BINARY_AST_NUM_UNARY_OPS - 1];
}

static int
implicit_sindy_cpu_gram_reference_stats(
    const BinaryAST* asts,
    size_t num_asts,
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const float* targets,
    int64_t target_rhs_stride,
    int64_t num_target_rhs,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* gram,
    int64_t gram_col_stride,
    float* x_sum,
    float* xty,
    int64_t xty_rhs_stride,
    float* y_sum,
    float* yy
) {
    int64_t row;
    int64_t rhs;
    size_t i;
    size_t j;

    if (asts == NULL ||
        num_asts == 0u ||
        num_asts > IMPLICIT_SINDY_CPU_GRAM_REFERENCE_FEATURES ||
        primitive_features == NULL ||
        row_count <= 0 ||
        primitive_feature_stride <= 0 ||
        num_primitive_features <= 0 ||
        targets == NULL ||
        target_rhs_stride <= 0 ||
        num_target_rhs <= 0 ||
        leaf_masks == NULL ||
        leaf_words == NULL ||
        leaf_words_feature_stride < BINARY_AST_NUM_INPUTS ||
        gram == NULL ||
        gram_col_stride < (int64_t)num_asts ||
        x_sum == NULL ||
        xty == NULL ||
        xty_rhs_stride < (int64_t)num_asts ||
        y_sum == NULL ||
        yy == NULL) {
        return IMPLICIT_SINDY_CPU_GRAM_REFERENCE_ERROR_INVALID_VALUE;
    }

    for (i = 0u; i < num_asts; ++i) {
        x_sum[i] = 0.0f;
        for (j = 0u; j < num_asts; ++j) {
            gram[i + j * (size_t)gram_col_stride] = 0.0f;
        }
    }
    for (rhs = 0; rhs < num_target_rhs; ++rhs) {
        y_sum[rhs] = 0.0f;
        yy[rhs] = 0.0f;
        for (i = 0u; i < num_asts; ++i) {
            xty[(size_t)rhs * (size_t)xty_rhs_stride + i] = 0.0f;
        }
    }

    for (row = 0; row < row_count; ++row) {
        float feature_values[IMPLICIT_SINDY_CPU_GRAM_REFERENCE_FEATURES];

        for (i = 0u; i < num_asts; ++i) {
            feature_values[i] = implicit_sindy_cpu_gram_reference_eval_feature(
                asts + i,
                primitive_features,
                primitive_feature_stride,
                num_primitive_features,
                row,
                leaf_masks[i],
                leaf_words + (int64_t)i * leaf_words_feature_stride
            );
            x_sum[i] += feature_values[i];
        }

        for (i = 0u; i < num_asts; ++i) {
            const float lhs = feature_values[i];
            for (j = 0u; j < num_asts; ++j) {
                gram[i + j * (size_t)gram_col_stride] += lhs * feature_values[j];
            }
        }

        for (rhs = 0; rhs < num_target_rhs; ++rhs) {
            const float y = targets[row + rhs * target_rhs_stride];
            y_sum[rhs] += y;
            yy[rhs] += y * y;
            for (i = 0u; i < num_asts; ++i) {
                xty[(size_t)rhs * (size_t)xty_rhs_stride + i] += feature_values[i] * y;
            }
        }
    }

    return IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS;
}

static int
implicit_sindy_cpu_gram_reference_stats_many_settings(
    const BinaryAST* asts,
    size_t num_asts,
    int64_t num_settings,
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const float* targets,
    int64_t target_rhs_stride,
    int64_t num_target_rhs,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* gram,
    int64_t gram_col_stride,
    float* x_sum,
    float* xty,
    int64_t xty_rhs_stride,
    float* y_sum,
    float* yy
) {
    int64_t setting;

    if (num_settings <= 0) {
        return IMPLICIT_SINDY_CPU_GRAM_REFERENCE_ERROR_INVALID_VALUE;
    }

    for (setting = 0; setting < num_settings; ++setting) {
        const int64_t setting_feature_offset = setting * (int64_t)num_asts;
        const int64_t gram_offset = setting * (int64_t)num_asts * gram_col_stride;
        const int64_t x_sum_offset = setting * (int64_t)num_asts;
        const int64_t rhs_offset = setting * num_target_rhs;
        int result = implicit_sindy_cpu_gram_reference_stats(
            asts,
            num_asts,
            primitive_features,
            row_count,
            primitive_feature_stride,
            num_primitive_features,
            targets,
            target_rhs_stride,
            num_target_rhs,
            leaf_masks + setting_feature_offset,
            leaf_words + setting_feature_offset * leaf_words_feature_stride,
            leaf_words_feature_stride,
            gram + gram_offset,
            gram_col_stride,
            x_sum + x_sum_offset,
            xty + rhs_offset * xty_rhs_stride,
            xty_rhs_stride,
            y_sum + rhs_offset,
            yy + rhs_offset
        );
        if (result != IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS) return result;
    }

    return IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS;
}

static int
implicit_sindy_cpu_gram_reference_columns_many_settings(
    const BinaryAST* asts,
    size_t num_asts,
    int64_t num_settings,
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* output,
    int64_t output_setting_stride,
    int64_t output_feature_stride
) {
    int64_t setting;
    size_t feature;
    int64_t row;

    if (asts == NULL ||
        num_asts == 0u ||
        num_asts > IMPLICIT_SINDY_CPU_GRAM_REFERENCE_FEATURES ||
        num_settings <= 0 ||
        primitive_features == NULL ||
        row_count <= 0 ||
        primitive_feature_stride <= 0 ||
        num_primitive_features <= 0 ||
        leaf_masks == NULL ||
        leaf_words == NULL ||
        leaf_words_feature_stride < BINARY_AST_NUM_INPUTS ||
        output == NULL ||
        output_setting_stride <= 0 ||
        output_feature_stride < row_count) {
        return IMPLICIT_SINDY_CPU_GRAM_REFERENCE_ERROR_INVALID_VALUE;
    }

    for (setting = 0; setting < num_settings; ++setting) {
        const int64_t setting_feature_offset = setting * (int64_t)num_asts;
        for (feature = 0u; feature < num_asts; ++feature) {
            const int64_t feature_offset = setting_feature_offset + (int64_t)feature;
            for (row = 0; row < row_count; ++row) {
                output[
                    setting * output_setting_stride +
                    (int64_t)feature * output_feature_stride +
                    row
                ] = implicit_sindy_cpu_gram_reference_eval_feature(
                    asts + feature,
                    primitive_features,
                    primitive_feature_stride,
                    num_primitive_features,
                    row,
                    leaf_masks[feature_offset],
                    leaf_words + feature_offset * leaf_words_feature_stride
                );
            }
        }
    }

    return IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS;
}

#endif /* IMPLICIT_SINDY_CPU_GRAM_REFERENCE_H */
