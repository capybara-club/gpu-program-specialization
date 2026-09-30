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
#include "implicit_sindy.h"
#include "implicit_sindy_internal_constants.h"

#include <cuda.h>

#include <limits.h>
#include <stdlib.h>

#ifndef IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN_PATH
#error "IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN_PATH must point to the generated solve fatbin"
#endif

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX ifrs_
#include "incbin.h"

INCBIN(implicit_feature_ridge_solve_fatbin, IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN_PATH);

#define IFRS_ERROR_RET(ans)                                     \
    do {                                                        \
        ImplicitFeatureRidgeSolveResult ifrs_result__ = (ans);  \
        return ifrs_result__;                                   \
    } while (0)

#define IFRS_CHECK_RET(ans)                                     \
    do {                                                        \
        ImplicitFeatureRidgeSolveResult ifrs_result__ = (ans);  \
        if (ifrs_result__ != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) { \
            IFRS_ERROR_RET(ifrs_result__);                      \
        }                                                       \
    } while (0)

#define IFRS_CHECK_CUDA_RET(ans)                                \
    do {                                                        \
        CUresult ifrs_cuda_result__ = (ans);                    \
        if (ifrs_cuda_result__ != CUDA_SUCCESS) {               \
            IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA); \
        }                                                       \
    } while (0)

typedef enum {
    IFRS_KERNEL_32_RHS1_POSV_LOWER_COMPAT_FAST = 0,
    IFRS_KERNEL_32_RHS1_POSV = 1,
    IFRS_KERNEL_32_RHS2_POSV = 2,
    IFRS_KERNEL_32_RHS4_POSV = 3,
    IFRS_KERNEL_32_RHS8_POSV = 4,
    IFRS_KERNEL_32_RHS16_POSV = 5,
    IFRS_KERNEL_32_RHS32_POTRF_POTRS = 6,
    IFRS_KERNEL_32_STLSQ_POSV = 7,
    IFRS_KERNEL_COUNT = 8
} IfrsKernel;

typedef struct {
    IfrsKernel kernel;
    const char* symbol;
    int padded_rhs;
    int copy_whole_matrix;
    unsigned int block_x;
    unsigned int shared_bytes;
} IfrsKernelInfo;

struct ImplicitFeatureRidgeSolve {
    CUmodule module;
    CUfunction functions[IFRS_KERNEL_COUNT];
    CUfunction score_mse_function;
};

static const IfrsKernelInfo ifrs_kernel_table[] = {
    {
        IFRS_KERNEL_32_RHS1_POSV_LOWER_COMPAT_FAST,
        "implicit_feature_ridge_solve_32_rhs1_posv_lower_compat_fast",
        1,
        1,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 1 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_RHS1_POSV,
        "implicit_feature_ridge_solve_32_rhs1_posv",
        1,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 1 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_RHS2_POSV,
        "implicit_feature_ridge_solve_32_rhs2_posv",
        2,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 2 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_RHS4_POSV,
        "implicit_feature_ridge_solve_32_rhs4_posv",
        4,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 4 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_RHS8_POSV,
        "implicit_feature_ridge_solve_32_rhs8_posv",
        8,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 8 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_RHS16_POSV,
        "implicit_feature_ridge_solve_32_rhs16_posv",
        16,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 16 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_RHS32_POTRF_POTRS,
        "implicit_feature_ridge_solve_32_rhs32_potrf_potrs",
        32,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 32 + 2 * 32) * sizeof(float))
    },
    {
        IFRS_KERNEL_32_STLSQ_POSV,
        "implicit_feature_ridge_solve_32_stlsq_posv",
        1,
        0,
        32u,
        (unsigned int)((2 * 32 * 32 + 32 * 1 + 2 * 32) * sizeof(float))
    }
};

const char*
implicit_feature_ridge_solve_result_to_string(
    ImplicitFeatureRidgeSolveResult result
) {
    switch (result) {
        case IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS:
            return "IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS";
        case IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE:
            return "IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE";
        case IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_OUT_OF_MEMORY:
            return "IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_OUT_OF_MEMORY";
        case IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA:
            return "IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA";
        case IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_NOT_IMPLEMENTED:
            return "IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_NOT_IMPLEMENTED";
        default:
            return "IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_UNKNOWN";
    }
}

static const void*
ifrs_fatbin(
    size_t* fatbin_bytes_out
) {
    if (fatbin_bytes_out != NULL) {
        *fatbin_bytes_out = (size_t)ifrs_implicit_feature_ridge_solve_fatbin_size;
    }
    return (const void*)ifrs_implicit_feature_ridge_solve_fatbin_data;
}

static ImplicitFeatureRidgeSolveResult
ifrs_choose_padded_rhs(
    int64_t num_rhs,
    int* out_padded_rhs
) {
    static const int supported[] = {1, 2, 4, 8, 16, 32};
    size_t i;

    if (out_padded_rhs == NULL || num_rhs <= 0) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }
    for (i = 0u; i < sizeof(supported) / sizeof(supported[0]); ++i) {
        if (num_rhs <= (int64_t)supported[i]) {
            *out_padded_rhs = supported[i];
            return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
        }
    }
    IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_NOT_IMPLEMENTED);
}

static ImplicitFeatureRidgeSolveResult
ifrs_select_kernel(
    int64_t num_rhs,
    const IfrsKernelInfo** out_info
) {
    int padded_rhs = 0;
    size_t i;

    if (out_info == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    *out_info = NULL;
    IFRS_CHECK_RET(ifrs_choose_padded_rhs(num_rhs, &padded_rhs));

    if (padded_rhs == 1) {
        for (i = 0u; i < sizeof(ifrs_kernel_table) / sizeof(ifrs_kernel_table[0]); ++i) {
            if (ifrs_kernel_table[i].padded_rhs == 1 && ifrs_kernel_table[i].copy_whole_matrix) {
                *out_info = &ifrs_kernel_table[i];
                return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
            }
        }
    }

    for (i = 0u; i < sizeof(ifrs_kernel_table) / sizeof(ifrs_kernel_table[0]); ++i) {
        if (ifrs_kernel_table[i].padded_rhs == padded_rhs) {
            *out_info = &ifrs_kernel_table[i];
            return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
        }
    }
    IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_NOT_IMPLEMENTED);
}

static ImplicitFeatureRidgeSolveResult
ifrs_create_impl(
    ImplicitFeatureRidgeSolve* solve
) {
    size_t fatbin_bytes = 0u;
    const void* fatbin;
    size_t i;

    if (solve == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    fatbin = ifrs_fatbin(&fatbin_bytes);
    if (fatbin == NULL || fatbin_bytes == 0u) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }

    IFRS_CHECK_CUDA_RET(cuModuleLoadData(&solve->module, fatbin));
    for (i = 0u; i < sizeof(ifrs_kernel_table) / sizeof(ifrs_kernel_table[0]); ++i) {
        IFRS_CHECK_CUDA_RET(cuModuleGetFunction(
            &solve->functions[ifrs_kernel_table[i].kernel],
            solve->module,
            ifrs_kernel_table[i].symbol
        ));
    }
    IFRS_CHECK_CUDA_RET(cuModuleGetFunction(
        &solve->score_mse_function,
        solve->module,
        "implicit_feature_ridge_score_mse_kernel"
    ));
    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

ImplicitFeatureRidgeSolveResult
implicit_feature_ridge_solve_create(
    ImplicitFeatureRidgeSolve** out_solve
) {
    ImplicitFeatureRidgeSolve* solve;
    ImplicitFeatureRidgeSolveResult error_ret;

    if (out_solve == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    *out_solve = NULL;

    solve = (ImplicitFeatureRidgeSolve*)calloc(1u, sizeof(*solve));
    if (solve == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_OUT_OF_MEMORY);

    error_ret = ifrs_create_impl(solve);
    if (error_ret != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) {
        if (solve->module != NULL) {
            (void)cuModuleUnload(solve->module);
        }
        free(solve);
        IFRS_ERROR_RET(error_ret);
    }

    *out_solve = solve;
    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

static ImplicitFeatureRidgeSolveResult
ifrs_validate_score_args(
    ImplicitFeatureRidgeSolve* solve,
    int64_t row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    const float* eval_gram,
    int64_t eval_gram_col_stride,
    int64_t eval_gram_setting_stride,
    const float* eval_x_sum,
    int64_t eval_x_sum_setting_stride,
    const float* eval_xty,
    int64_t eval_xty_rhs_stride,
    int64_t eval_xty_setting_stride,
    const float* eval_y_sum,
    int64_t eval_y_sum_rhs_stride,
    int64_t eval_y_sum_setting_stride,
    const float* eval_yy,
    int64_t eval_yy_rhs_stride,
    int64_t eval_yy_setting_stride,
    const float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    const float* x_mean,
    int64_t x_mean_setting_stride,
    const float* x_scale,
    int64_t x_scale_setting_stride,
    const float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride,
    float* mse,
    int64_t mse_rhs_stride,
    int64_t mse_setting_stride,
    int64_t mse_sweep_stride
) {
    if (solve == NULL ||
        row_count <= 0 ||
        num_settings <= 0 ||
        num_rhs <= 0 ||
        num_sweeps <= 0 ||
        num_settings > (int64_t)UINT_MAX ||
        num_sweeps > (int64_t)UINT_MAX ||
        num_rhs > IMPLICIT_SINDY_INTERNAL_FEATURES ||
        eval_gram == NULL ||
        eval_x_sum == NULL ||
        eval_xty == NULL ||
        eval_y_sum == NULL ||
        eval_yy == NULL ||
        beta_standardized == NULL ||
        x_mean == NULL ||
        x_scale == NULL ||
        y_mean == NULL ||
        solve_info == NULL ||
        mse == NULL) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }

    if (eval_gram_col_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        eval_gram_setting_stride <= 0 ||
        eval_x_sum_setting_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        eval_xty_rhs_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        eval_xty_setting_stride <= 0 ||
        eval_y_sum_rhs_stride <= 0 ||
        eval_y_sum_setting_stride <= 0 ||
        eval_yy_rhs_stride <= 0 ||
        eval_yy_setting_stride <= 0 ||
        beta_rhs_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        beta_setting_stride <= 0 ||
        beta_sweep_stride <= 0 ||
        x_mean_setting_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        x_scale_setting_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        y_mean_rhs_stride <= 0 ||
        y_mean_setting_stride <= 0 ||
        solve_info_sweep_stride <= 0 ||
        solve_info_setting_stride <= 0 ||
        mse_rhs_stride <= 0 ||
        mse_setting_stride <= 0 ||
        mse_sweep_stride <= 0) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }

    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

static ImplicitFeatureRidgeSolveResult
ifrs_score_mse_impl(
    ImplicitFeatureRidgeSolve* solve,
    void* stream,
    int64_t row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    const float* eval_gram,
    int64_t eval_gram_col_stride,
    int64_t eval_gram_setting_stride,
    const float* eval_x_sum,
    int64_t eval_x_sum_setting_stride,
    const float* eval_xty,
    int64_t eval_xty_rhs_stride,
    int64_t eval_xty_setting_stride,
    const float* eval_y_sum,
    int64_t eval_y_sum_rhs_stride,
    int64_t eval_y_sum_setting_stride,
    const float* eval_yy,
    int64_t eval_yy_rhs_stride,
    int64_t eval_yy_setting_stride,
    const float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    const float* x_mean,
    int64_t x_mean_setting_stride,
    const float* x_scale,
    int64_t x_scale_setting_stride,
    const float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride,
    float* mse,
    int64_t mse_rhs_stride,
    int64_t mse_setting_stride,
    int64_t mse_sweep_stride
) {
    void* args[34];

    IFRS_CHECK_RET(ifrs_validate_score_args(
        solve,
        row_count,
        num_settings,
        num_rhs,
        num_sweeps,
        eval_gram,
        eval_gram_col_stride,
        eval_gram_setting_stride,
        eval_x_sum,
        eval_x_sum_setting_stride,
        eval_xty,
        eval_xty_rhs_stride,
        eval_xty_setting_stride,
        eval_y_sum,
        eval_y_sum_rhs_stride,
        eval_y_sum_setting_stride,
        eval_yy,
        eval_yy_rhs_stride,
        eval_yy_setting_stride,
        beta_standardized,
        beta_rhs_stride,
        beta_setting_stride,
        beta_sweep_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        y_mean,
        y_mean_rhs_stride,
        y_mean_setting_stride,
        solve_info,
        solve_info_sweep_stride,
        solve_info_setting_stride,
        mse,
        mse_rhs_stride,
        mse_setting_stride,
        mse_sweep_stride
    ));
    if (solve->score_mse_function == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA);

    args[0] = &row_count;
    args[1] = &num_rhs;
    args[2] = (void*)&eval_gram;
    args[3] = &eval_gram_col_stride;
    args[4] = &eval_gram_setting_stride;
    args[5] = (void*)&eval_x_sum;
    args[6] = &eval_x_sum_setting_stride;
    args[7] = (void*)&eval_xty;
    args[8] = &eval_xty_rhs_stride;
    args[9] = &eval_xty_setting_stride;
    args[10] = (void*)&eval_y_sum;
    args[11] = &eval_y_sum_rhs_stride;
    args[12] = &eval_y_sum_setting_stride;
    args[13] = (void*)&eval_yy;
    args[14] = &eval_yy_rhs_stride;
    args[15] = &eval_yy_setting_stride;
    args[16] = (void*)&beta_standardized;
    args[17] = &beta_rhs_stride;
    args[18] = &beta_setting_stride;
    args[19] = &beta_sweep_stride;
    args[20] = (void*)&x_mean;
    args[21] = &x_mean_setting_stride;
    args[22] = (void*)&x_scale;
    args[23] = &x_scale_setting_stride;
    args[24] = (void*)&y_mean;
    args[25] = &y_mean_rhs_stride;
    args[26] = &y_mean_setting_stride;
    args[27] = (void*)&solve_info;
    args[28] = &solve_info_sweep_stride;
    args[29] = &solve_info_setting_stride;
    args[30] = &mse;
    args[31] = &mse_rhs_stride;
    args[32] = &mse_setting_stride;
    args[33] = &mse_sweep_stride;

    IFRS_CHECK_CUDA_RET(cuLaunchKernel(
        solve->score_mse_function,
        (unsigned int)num_settings,
        1u,
        (unsigned int)num_sweeps,
        32u,
        1u,
        1u,
        0u,
        stream,
        args,
        NULL
    ));

    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

ImplicitFeatureRidgeSolveResult
implicit_feature_ridge_score_validation_mse(
    ImplicitFeatureRidgeSolve* solve,
    void* stream,
    int64_t validation_row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    const float* validation_gram,
    int64_t validation_gram_col_stride,
    int64_t validation_gram_setting_stride,
    const float* validation_x_sum,
    int64_t validation_x_sum_setting_stride,
    const float* validation_xty,
    int64_t validation_xty_rhs_stride,
    int64_t validation_xty_setting_stride,
    const float* validation_y_sum,
    int64_t validation_y_sum_rhs_stride,
    int64_t validation_y_sum_setting_stride,
    const float* validation_yy,
    int64_t validation_yy_rhs_stride,
    int64_t validation_yy_setting_stride,
    const float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    const float* x_mean,
    int64_t x_mean_setting_stride,
    const float* x_scale,
    int64_t x_scale_setting_stride,
    const float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride,
    float* mse,
    int64_t mse_rhs_stride,
    int64_t mse_setting_stride,
    int64_t mse_sweep_stride
) {
    IFRS_CHECK_RET(ifrs_score_mse_impl(
        solve,
        stream,
        validation_row_count,
        num_settings,
        num_rhs,
        num_sweeps,
        validation_gram,
        validation_gram_col_stride,
        validation_gram_setting_stride,
        validation_x_sum,
        validation_x_sum_setting_stride,
        validation_xty,
        validation_xty_rhs_stride,
        validation_xty_setting_stride,
        validation_y_sum,
        validation_y_sum_rhs_stride,
        validation_y_sum_setting_stride,
        validation_yy,
        validation_yy_rhs_stride,
        validation_yy_setting_stride,
        beta_standardized,
        beta_rhs_stride,
        beta_setting_stride,
        beta_sweep_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        y_mean,
        y_mean_rhs_stride,
        y_mean_setting_stride,
        solve_info,
        solve_info_sweep_stride,
        solve_info_setting_stride,
        mse,
        mse_rhs_stride,
        mse_setting_stride,
        mse_sweep_stride
    ));
    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

ImplicitFeatureRidgeSolveResult
implicit_feature_ridge_solve_destroy(
    ImplicitFeatureRidgeSolve* solve
) {
    CUresult unload_result;

    if (solve == NULL) return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
    unload_result = CUDA_SUCCESS;
    if (solve->module != NULL) {
        unload_result = cuModuleUnload(solve->module);
    }
    free(solve);
    if (unload_result != CUDA_SUCCESS) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA);
    }
    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

static ImplicitFeatureRidgeSolveResult
ifrs_validate_launch_args(
    ImplicitFeatureRidgeSolve* solve,
    int64_t row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    float scale_epsilon,
    const float* gram,
    int64_t gram_col_stride,
    int64_t gram_setting_stride,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    const float* alphas,
    int64_t alphas_stride,
    float* x_mean,
    int64_t x_mean_setting_stride,
    float* x_scale,
    int64_t x_scale_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride
) {
    if (solve == NULL ||
        row_count <= 0 ||
        num_settings <= 0 ||
        num_rhs <= 0 ||
        num_sweeps <= 0 ||
        scale_epsilon <= 0.0f ||
        num_settings > (int64_t)UINT_MAX ||
        num_sweeps > (int64_t)UINT_MAX ||
        num_rhs > IMPLICIT_SINDY_INTERNAL_FEATURES ||
        gram == NULL ||
        x_sum == NULL ||
        xty == NULL ||
        y_sum == NULL ||
        alphas == NULL ||
        x_mean == NULL ||
        x_scale == NULL ||
        y_mean == NULL ||
        beta_standardized == NULL ||
        solve_info == NULL) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }

    if (gram_col_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        gram_setting_stride <= 0 ||
        x_sum_setting_stride <= 0 ||
        xty_rhs_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        xty_setting_stride <= 0 ||
        y_sum_rhs_stride <= 0 ||
        y_sum_setting_stride <= 0 ||
        alphas_stride <= 0 ||
        x_mean_setting_stride <= 0 ||
        x_scale_setting_stride <= 0 ||
        y_mean_rhs_stride <= 0 ||
        y_mean_setting_stride <= 0 ||
        beta_rhs_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        beta_setting_stride <= 0 ||
        beta_sweep_stride <= 0 ||
        solve_info_sweep_stride <= 0 ||
        solve_info_setting_stride <= 0) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }

    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

ImplicitFeatureRidgeSolveResult
implicit_feature_ridge_solve_posv_sweep(
    ImplicitFeatureRidgeSolve* solve,
    void* stream,
    int64_t row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    float scale_epsilon,
    const float* gram,
    int64_t gram_col_stride,
    int64_t gram_setting_stride,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    const float* alphas,
    int64_t alphas_stride,
    float* x_mean,
    int64_t x_mean_setting_stride,
    float* x_scale,
    int64_t x_scale_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride
) {
    const IfrsKernelInfo* kernel_info = NULL;
    CUfunction function;
    const float* thresholds = NULL;
    int64_t thresholds_stride = 1;
    uint32_t* active_masks = NULL;
    int64_t active_mask_rhs_stride = 1;
    int64_t active_mask_setting_stride = 1;
    int64_t active_mask_sweep_stride = 1;
    int32_t* active_counts = NULL;
    int64_t active_count_rhs_stride = 1;
    int64_t active_count_setting_stride = 1;
    int64_t active_count_sweep_stride = 1;
    int32_t* iteration_counts = NULL;
    int64_t iteration_count_rhs_stride = 1;
    int64_t iteration_count_setting_stride = 1;
    int64_t iteration_count_sweep_stride = 1;
    void* args[45];

    IFRS_CHECK_RET(ifrs_validate_launch_args(
        solve,
        row_count,
        num_settings,
        num_rhs,
        num_sweeps,
        scale_epsilon,
        gram,
        gram_col_stride,
        gram_setting_stride,
        x_sum,
        x_sum_setting_stride,
        xty,
        xty_rhs_stride,
        xty_setting_stride,
        y_sum,
        y_sum_rhs_stride,
        y_sum_setting_stride,
        alphas,
        alphas_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        y_mean,
        y_mean_rhs_stride,
        y_mean_setting_stride,
        beta_standardized,
        beta_rhs_stride,
        beta_setting_stride,
        beta_sweep_stride,
        solve_info,
        solve_info_sweep_stride,
        solve_info_setting_stride
    ));
    IFRS_CHECK_RET(ifrs_select_kernel(num_rhs, &kernel_info));
    if (kernel_info == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_NOT_IMPLEMENTED);
    function = solve->functions[kernel_info->kernel];
    if (function == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA);

    args[0] = &row_count;
    args[1] = &scale_epsilon;
    args[2] = &num_rhs;
    args[3] = &num_sweeps;
    args[4] = (void*)&gram;
    args[5] = &gram_col_stride;
    args[6] = &gram_setting_stride;
    args[7] = (void*)&x_sum;
    args[8] = &x_sum_setting_stride;
    args[9] = (void*)&xty;
    args[10] = &xty_rhs_stride;
    args[11] = &xty_setting_stride;
    args[12] = (void*)&y_sum;
    args[13] = &y_sum_rhs_stride;
    args[14] = &y_sum_setting_stride;
    args[15] = (void*)&alphas;
    args[16] = &alphas_stride;
    args[17] = (void*)&thresholds;
    args[18] = &thresholds_stride;
    args[19] = &x_mean;
    args[20] = &x_mean_setting_stride;
    args[21] = &x_scale;
    args[22] = &x_scale_setting_stride;
    args[23] = &y_mean;
    args[24] = &y_mean_rhs_stride;
    args[25] = &y_mean_setting_stride;
    args[26] = &beta_standardized;
    args[27] = &beta_rhs_stride;
    args[28] = &beta_setting_stride;
    args[29] = &beta_sweep_stride;
    args[30] = &solve_info;
    args[31] = &solve_info_sweep_stride;
    args[32] = &solve_info_setting_stride;
    args[33] = &active_masks;
    args[34] = &active_mask_rhs_stride;
    args[35] = &active_mask_setting_stride;
    args[36] = &active_mask_sweep_stride;
    args[37] = &active_counts;
    args[38] = &active_count_rhs_stride;
    args[39] = &active_count_setting_stride;
    args[40] = &active_count_sweep_stride;
    args[41] = &iteration_counts;
    args[42] = &iteration_count_rhs_stride;
    args[43] = &iteration_count_setting_stride;
    args[44] = &iteration_count_sweep_stride;

    IFRS_CHECK_CUDA_RET(cuLaunchKernel(
        function,
        (unsigned int)num_settings,
        1u,
        1u,
        kernel_info->block_x,
        1u,
        1u,
        kernel_info->shared_bytes,
        stream,
        args,
        NULL
    ));

    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

static ImplicitFeatureRidgeSolveResult
ifrs_validate_stlsq_args(
    ImplicitFeatureRidgeSolve* solve,
    int64_t row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    float scale_epsilon,
    const float* gram,
    int64_t gram_col_stride,
    int64_t gram_setting_stride,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    const float* alphas,
    int64_t alphas_stride,
    const float* thresholds,
    int64_t thresholds_stride,
    float* x_mean,
    int64_t x_mean_setting_stride,
    float* x_scale,
    int64_t x_scale_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    uint32_t* active_masks,
    int64_t active_mask_rhs_stride,
    int64_t active_mask_setting_stride,
    int64_t active_mask_sweep_stride,
    int32_t* active_counts,
    int64_t active_count_rhs_stride,
    int64_t active_count_setting_stride,
    int64_t active_count_sweep_stride,
    int32_t* iteration_counts,
    int64_t iteration_count_rhs_stride,
    int64_t iteration_count_setting_stride,
    int64_t iteration_count_sweep_stride,
    int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride
) {
    IFRS_CHECK_RET(ifrs_validate_launch_args(
        solve,
        row_count,
        num_settings,
        num_rhs,
        num_sweeps,
        scale_epsilon,
        gram,
        gram_col_stride,
        gram_setting_stride,
        x_sum,
        x_sum_setting_stride,
        xty,
        xty_rhs_stride,
        xty_setting_stride,
        y_sum,
        y_sum_rhs_stride,
        y_sum_setting_stride,
        alphas,
        alphas_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        y_mean,
        y_mean_rhs_stride,
        y_mean_setting_stride,
        beta_standardized,
        beta_rhs_stride,
        beta_setting_stride,
        beta_sweep_stride,
        solve_info,
        solve_info_sweep_stride,
        solve_info_setting_stride
    ));

    if (thresholds == NULL ||
        active_masks == NULL ||
        active_counts == NULL ||
        iteration_counts == NULL ||
        thresholds_stride <= 0 ||
        active_mask_rhs_stride <= 0 ||
        active_mask_setting_stride <= 0 ||
        active_mask_sweep_stride <= 0 ||
        active_count_rhs_stride <= 0 ||
        active_count_setting_stride <= 0 ||
        active_count_sweep_stride <= 0 ||
        iteration_count_rhs_stride <= 0 ||
        iteration_count_setting_stride <= 0 ||
        iteration_count_sweep_stride <= 0) {
        IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE);
    }

    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}

ImplicitFeatureRidgeSolveResult
implicit_feature_ridge_solve_stlsq_sweep(
    ImplicitFeatureRidgeSolve* solve,
    void* stream,
    int64_t row_count,
    int64_t num_settings,
    int64_t num_rhs,
    int64_t num_sweeps,
    float scale_epsilon,
    const float* gram,
    int64_t gram_col_stride,
    int64_t gram_setting_stride,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    const float* alphas,
    int64_t alphas_stride,
    const float* thresholds,
    int64_t thresholds_stride,
    float* x_mean,
    int64_t x_mean_setting_stride,
    float* x_scale,
    int64_t x_scale_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    uint32_t* active_masks,
    int64_t active_mask_rhs_stride,
    int64_t active_mask_setting_stride,
    int64_t active_mask_sweep_stride,
    int32_t* active_counts,
    int64_t active_count_rhs_stride,
    int64_t active_count_setting_stride,
    int64_t active_count_sweep_stride,
    int32_t* iteration_counts,
    int64_t iteration_count_rhs_stride,
    int64_t iteration_count_setting_stride,
    int64_t iteration_count_sweep_stride,
    int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride
) {
    const IfrsKernelInfo* kernel_info = &ifrs_kernel_table[IFRS_KERNEL_32_STLSQ_POSV];
    CUfunction function;
    void* args[45];

    IFRS_CHECK_RET(ifrs_validate_stlsq_args(
        solve,
        row_count,
        num_settings,
        num_rhs,
        num_sweeps,
        scale_epsilon,
        gram,
        gram_col_stride,
        gram_setting_stride,
        x_sum,
        x_sum_setting_stride,
        xty,
        xty_rhs_stride,
        xty_setting_stride,
        y_sum,
        y_sum_rhs_stride,
        y_sum_setting_stride,
        alphas,
        alphas_stride,
        thresholds,
        thresholds_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        y_mean,
        y_mean_rhs_stride,
        y_mean_setting_stride,
        beta_standardized,
        beta_rhs_stride,
        beta_setting_stride,
        beta_sweep_stride,
        active_masks,
        active_mask_rhs_stride,
        active_mask_setting_stride,
        active_mask_sweep_stride,
        active_counts,
        active_count_rhs_stride,
        active_count_setting_stride,
        active_count_sweep_stride,
        iteration_counts,
        iteration_count_rhs_stride,
        iteration_count_setting_stride,
        iteration_count_sweep_stride,
        solve_info,
        solve_info_sweep_stride,
        solve_info_setting_stride
    ));

    function = solve->functions[kernel_info->kernel];
    if (function == NULL) IFRS_ERROR_RET(IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA);

    args[0] = &row_count;
    args[1] = &scale_epsilon;
    args[2] = &num_rhs;
    args[3] = &num_sweeps;
    args[4] = (void*)&gram;
    args[5] = &gram_col_stride;
    args[6] = &gram_setting_stride;
    args[7] = (void*)&x_sum;
    args[8] = &x_sum_setting_stride;
    args[9] = (void*)&xty;
    args[10] = &xty_rhs_stride;
    args[11] = &xty_setting_stride;
    args[12] = (void*)&y_sum;
    args[13] = &y_sum_rhs_stride;
    args[14] = &y_sum_setting_stride;
    args[15] = (void*)&alphas;
    args[16] = &alphas_stride;
    args[17] = (void*)&thresholds;
    args[18] = &thresholds_stride;
    args[19] = &x_mean;
    args[20] = &x_mean_setting_stride;
    args[21] = &x_scale;
    args[22] = &x_scale_setting_stride;
    args[23] = &y_mean;
    args[24] = &y_mean_rhs_stride;
    args[25] = &y_mean_setting_stride;
    args[26] = &beta_standardized;
    args[27] = &beta_rhs_stride;
    args[28] = &beta_setting_stride;
    args[29] = &beta_sweep_stride;
    args[30] = &solve_info;
    args[31] = &solve_info_sweep_stride;
    args[32] = &solve_info_setting_stride;
    args[33] = &active_masks;
    args[34] = &active_mask_rhs_stride;
    args[35] = &active_mask_setting_stride;
    args[36] = &active_mask_sweep_stride;
    args[37] = &active_counts;
    args[38] = &active_count_rhs_stride;
    args[39] = &active_count_setting_stride;
    args[40] = &active_count_sweep_stride;
    args[41] = &iteration_counts;
    args[42] = &iteration_count_rhs_stride;
    args[43] = &iteration_count_setting_stride;
    args[44] = &iteration_count_sweep_stride;

    IFRS_CHECK_CUDA_RET(cuLaunchKernel(
        function,
        (unsigned int)num_settings,
        1u,
        1u,
        kernel_info->block_x,
        1u,
        1u,
        kernel_info->shared_bytes,
        stream,
        args,
        NULL
    ));

    return IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;
}
