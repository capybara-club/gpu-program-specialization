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
#ifndef IMPLICIT_SINDY_H
#define IMPLICIT_SINDY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define IMPLICIT_SINDY_PUBLIC_DEC extern "C"
#else
#define IMPLICIT_SINDY_PUBLIC_DEC extern
#endif

typedef enum {
    BINARY_AST_NUM_INPUTS = 8,
    BINARY_AST_NUM_UNARY_OPS = 15,
    BINARY_AST_NUM_BINARY_OPS = 7
} BinaryAstShape;

typedef enum {
    BINARY_AST_UNARY_IDENTITY = 0,
    BINARY_AST_UNARY_SQUARE_F32 = 1,
    BINARY_AST_UNARY_CUBE_F32 = 2,
    BINARY_AST_UNARY_NEG_FTZ_F32 = 3,
    BINARY_AST_UNARY_ABS_FTZ_F32 = 4,
    BINARY_AST_UNARY_RCP_APPROX_FTZ_F32 = 5,
    BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32 = 6,
    BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32 = 7,
    BINARY_AST_UNARY_SIN_APPROX_FTZ_F32 = 8,
    BINARY_AST_UNARY_COS_APPROX_FTZ_F32 = 9,
    BINARY_AST_UNARY_EX2_APPROX_FTZ_F32 = 10,
    BINARY_AST_UNARY_EXP_APPROX_FTZ_F32 = 11,
    BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32 = 12,
    BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32 = 13,
    BINARY_AST_UNARY_SAFE_RCP_F32 = 14,
    BINARY_AST_UNARY_SAFE_SQRT_F32 = 15,
    BINARY_AST_UNARY_SAFE_RSQRT_F32 = 16,
    BINARY_AST_UNARY_SAFE_EX2_F32 = 17,
    BINARY_AST_UNARY_SAFE_EXP_F32 = 18,
    BINARY_AST_UNARY_SAFE_LOG2_F32 = 19,
    BINARY_AST_UNARY_SAFE_LOG10_F32 = 20,
    BINARY_AST_UNARY_ZERO_F32 = 21,
    BINARY_AST_UNARY_NUM_ENUMS = 22
} BinaryAstUnaryOp;

typedef enum {
    BINARY_AST_BINARY_ADD_FTZ_F32 = 0,
    BINARY_AST_BINARY_SUB_FTZ_F32 = 1,
    BINARY_AST_BINARY_MUL_FTZ_F32 = 2,
    BINARY_AST_BINARY_KEEP_LEFT = 3,
    BINARY_AST_BINARY_KEEP_RIGHT = 4,
    BINARY_AST_BINARY_DIV_APPROX_FTZ_F32 = 5,
    BINARY_AST_BINARY_MIN_FTZ_F32 = 6,
    BINARY_AST_BINARY_MAX_FTZ_F32 = 7,
    BINARY_AST_BINARY_SAFE_DIV_F32 = 8,
    BINARY_AST_BINARY_NUM_ENUMS = 9
} BinaryAstBinaryOp;

typedef struct {
    BinaryAstUnaryOp unary[BINARY_AST_NUM_UNARY_OPS];
    BinaryAstBinaryOp binary[BINARY_AST_NUM_BINARY_OPS];
} BinaryAST;

typedef enum {
    IMPLICIT_SINDY_SUCCESS = 0,
    IMPLICIT_SINDY_ERROR_INVALID_VALUE = 1,
    IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY = 2,
    IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED = 3,
    IMPLICIT_SINDY_ERROR_PTX_INJECT = 4,
    IMPLICIT_SINDY_ERROR_STACK_PTX = 5,
    IMPLICIT_SINDY_ERROR_NVPTX = 6,
    IMPLICIT_SINDY_ERROR_THREAD = 7,
    IMPLICIT_SINDY_ERROR_CUDA = 8,
    IMPLICIT_SINDY_ERROR_INTERNAL = 9
} ImplicitSindyResult;

typedef enum {
    IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS = 0,
    IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_INVALID_VALUE = 1,
    IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_OUT_OF_MEMORY = 2,
    IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_CUDA = 3,
    IMPLICIT_FEATURE_RIDGE_SOLVE_ERROR_NOT_IMPLEMENTED = 4
} ImplicitFeatureRidgeSolveResult;

typedef struct ImplicitSindyAstCompiler ImplicitSindyAstCompiler;
typedef struct ImplicitSindyAstColumnCompiler ImplicitSindyAstColumnCompiler;
typedef struct ImplicitSindyAstColumnModule ImplicitSindyAstColumnModule;
typedef struct ImplicitSindyColumnsModule ImplicitSindyColumnsModule;
typedef struct ImplicitSindyGramModule ImplicitSindyGramModule;
typedef struct ImplicitFeatureRidgeSolve ImplicitFeatureRidgeSolve;

IMPLICIT_SINDY_PUBLIC_DEC const char* implicit_sindy_result_to_string(ImplicitSindyResult result);
IMPLICIT_SINDY_PUBLIC_DEC size_t implicit_sindy_ast_cubin_count(size_t num_asts, size_t kernels_per_module);
IMPLICIT_SINDY_PUBLIC_DEC size_t implicit_sindy_ast_column_cubin_count(size_t num_asts, size_t kernels_per_module);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_cubin(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_cubin,
    size_t gram_template_cubin_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_ptx(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_ptx,
    size_t gram_template_ptx_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create_columns(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_ast_compiler_destroy(ImplicitSindyAstCompiler* compiler);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compile_workspace_size(
    const ImplicitSindyAstCompiler* compiler,
    size_t num_asts,
    size_t worker_count,
    size_t scratch_bytes_per_worker,
    size_t* out_bytes
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compile_cubins(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    void** out_cubins,
    size_t* out_cubin_sizes
);

IMPLICIT_SINDY_PUBLIC_DEC void implicit_sindy_free_cubins(void** cubins, size_t* cubin_sizes, size_t num_cubins);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_load_cubin_modules(
    void* const* cubins,
    const size_t* cubin_sizes,
    size_t num_cubins,
    void** out_modules
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_unload_modules(void** modules, size_t num_modules);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_column_compiler_create(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstColumnCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_ast_column_compiler_destroy(ImplicitSindyAstColumnCompiler* compiler);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_column_compile_workspace_size(
    const ImplicitSindyAstColumnCompiler* compiler,
    size_t num_asts,
    size_t worker_count,
    size_t scratch_bytes_per_worker,
    size_t* out_bytes
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_column_compile_cubins(
    const ImplicitSindyAstColumnCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    void** out_cubins,
    size_t* out_cubin_sizes
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_column_module_create(
    void* module,
    size_t kernels_per_module,
    ImplicitSindyAstColumnModule** out_column_module
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_ast_column_module_destroy(ImplicitSindyAstColumnModule* column_module);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_column_launch(
    ImplicitSindyAstColumnModule* column_module,
    size_t kernel_index,
    void* stream,
    int64_t feature_index,
    int64_t num_settings,
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* output,
    int64_t output_setting_stride
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_gram_module_create(
    void* module,
    size_t kernels_per_module,
    ImplicitSindyGramModule** out_gram_module
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_gram_module_destroy(ImplicitSindyGramModule* gram_module);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_gram_launch(
    ImplicitSindyGramModule* gram_module,
    size_t kernel_index,
    void* stream,
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
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_columns_module_create(
    void* module,
    size_t kernels_per_module,
    ImplicitSindyColumnsModule** out_columns_module
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_columns_module_destroy(ImplicitSindyColumnsModule* columns_module);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_columns_launch(
    ImplicitSindyColumnsModule* columns_module,
    size_t kernel_index,
    void* stream,
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
);

IMPLICIT_SINDY_PUBLIC_DEC const char* implicit_feature_ridge_solve_result_to_string(ImplicitFeatureRidgeSolveResult result);
IMPLICIT_SINDY_PUBLIC_DEC ImplicitFeatureRidgeSolveResult implicit_feature_ridge_solve_create(ImplicitFeatureRidgeSolve** out_solve);
IMPLICIT_SINDY_PUBLIC_DEC ImplicitFeatureRidgeSolveResult implicit_feature_ridge_solve_destroy(ImplicitFeatureRidgeSolve* solve);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitFeatureRidgeSolveResult
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
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitFeatureRidgeSolveResult
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
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitFeatureRidgeSolveResult
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
);

#endif /* IMPLICIT_SINDY_H */
