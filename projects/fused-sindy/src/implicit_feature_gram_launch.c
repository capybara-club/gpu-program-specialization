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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IFGL_ERROR_RET(ans)                         \
    do {                                           \
        ImplicitSindyResult ifgl_result__ = (ans); \
        return ifgl_result__;                      \
    } while (0)

#define IFGL_CHECK_RET(ans)                         \
    do {                                           \
        ImplicitSindyResult ifgl_result__ = (ans); \
        if (ifgl_result__ != IMPLICIT_SINDY_SUCCESS) { \
            IFGL_ERROR_RET(ifgl_result__);         \
        }                                          \
    } while (0)

#define IFGL_CHECK_CUDA_RET(ans)                    \
    do {                                           \
        CUresult ifgl_cuda_result__ = (ans);       \
        if (ifgl_cuda_result__ != CUDA_SUCCESS) {  \
            IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_CUDA); \
        }                                          \
    } while (0)

enum {
    IFGL_THREADS = 128,
    IFGL_MAX_KERNELS_PER_MODULE = 64
};

struct ImplicitSindyGramModule {
    CUmodule module;
    size_t kernels_per_module;
    CUfunction gram_functions[];
};

static size_t
ifgl_kernels_per_module_count(
    size_t kernels_per_module
) {
    if (kernels_per_module == 0u || kernels_per_module > IFGL_MAX_KERNELS_PER_MODULE) return 0u;
    return kernels_per_module;
}

static ImplicitSindyResult
ifgl_gram_kernel_symbol(
    char* buffer,
    size_t buffer_size,
    size_t kernels_per_module,
    size_t kernel_index
) {
    int bytes_written;

    if (buffer == NULL || buffer_size == 0u || kernel_index >= kernels_per_module) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    if (kernels_per_module == 1u && kernel_index == 0u) {
        if (strlen("implicit_feature_gram_kernel") + 1u > buffer_size) {
            IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
        }
        strcpy(buffer, "implicit_feature_gram_kernel");
        return IMPLICIT_SINDY_SUCCESS;
    }
    bytes_written = snprintf(
        buffer,
        buffer_size,
        "implicit_feature_gram_kernel_%zu",
        kernel_index
    );
    if (bytes_written < 0 || (size_t)bytes_written >= buffer_size) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
ifgl_create_impl(
    ImplicitSindyGramModule* gram_module,
    void* module,
    size_t kernels_per_module
) {
    size_t kernel_count;
    size_t i;

    if (gram_module == NULL || module == NULL) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    kernel_count = ifgl_kernels_per_module_count(kernels_per_module);
    if (kernel_count == 0u) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    gram_module->module = (CUmodule)module;
    gram_module->kernels_per_module = kernel_count;

    for (i = 0u; i < kernel_count; ++i) {
        char symbol[64];
        IFGL_CHECK_RET(ifgl_gram_kernel_symbol(
            symbol,
            sizeof(symbol),
            kernel_count,
            i
        ));
        IFGL_CHECK_CUDA_RET(cuModuleGetFunction(
            &gram_module->gram_functions[i],
            gram_module->module,
            symbol
        ));
    }

    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_gram_module_create(
    void* module,
    size_t kernels_per_module,
    ImplicitSindyGramModule** out_gram_module
) {
    ImplicitSindyGramModule* gram_module;
    ImplicitSindyResult error_ret;
    size_t kernel_count;
    size_t allocation_size;

    if (out_gram_module == NULL) IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out_gram_module = NULL;

    kernel_count = ifgl_kernels_per_module_count(kernels_per_module);
    if (kernel_count == 0u ||
        kernel_count > (SIZE_MAX - sizeof(*gram_module)) / sizeof(gram_module->gram_functions[0])) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    allocation_size = sizeof(*gram_module) + kernel_count * sizeof(gram_module->gram_functions[0]);
    gram_module = (ImplicitSindyGramModule*)calloc(1u, allocation_size);
    if (gram_module == NULL) IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    error_ret = ifgl_create_impl(
        gram_module,
        module,
        kernels_per_module
    );
    if (error_ret != IMPLICIT_SINDY_SUCCESS) {
        free(gram_module);
        IFGL_ERROR_RET(error_ret);
    }

    *out_gram_module = gram_module;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_gram_module_destroy(
    ImplicitSindyGramModule* gram_module
) {
    free(gram_module);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
ifgl_validate_grid(
    int64_t num_settings
) {
    if (num_settings <= 0 || num_settings > (int64_t)UINT_MAX) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
ifgl_validate_gram_launch(
    ImplicitSindyGramModule* gram_module,
    size_t kernel_index,
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
    if (gram_module == NULL ||
        kernel_index >= gram_module->kernels_per_module ||
        gram_module->gram_functions[kernel_index] == NULL ||
        primitive_features == NULL ||
        row_count <= 0 ||
        primitive_feature_stride <= 0 ||
        num_primitive_features <= 0 ||
        num_primitive_features > IMPLICIT_SINDY_INTERNAL_PRIMITIVE_FEATURES_MAX ||
        targets == NULL ||
        target_rhs_stride <= 0 ||
        num_target_rhs <= 0 ||
        num_target_rhs > IMPLICIT_SINDY_INTERNAL_TARGET_RHS_MAX ||
        leaf_masks == NULL ||
        leaf_words == NULL ||
        leaf_words_feature_stride < IMPLICIT_SINDY_INTERNAL_LEAVES ||
        gram == NULL ||
        gram_col_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        x_sum == NULL ||
        xty == NULL ||
        xty_rhs_stride < IMPLICIT_SINDY_INTERNAL_FEATURES ||
        y_sum == NULL ||
        yy == NULL) {
        IFGL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    IFGL_CHECK_RET(ifgl_validate_grid(num_settings));
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
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
) {
    void* args[17];

    IFGL_CHECK_RET(ifgl_validate_gram_launch(
        gram_module,
        kernel_index,
        num_settings,
        primitive_features,
        row_count,
        primitive_feature_stride,
        num_primitive_features,
        targets,
        target_rhs_stride,
        num_target_rhs,
        leaf_masks,
        leaf_words,
        leaf_words_feature_stride,
        gram,
        gram_col_stride,
        x_sum,
        xty,
        xty_rhs_stride,
        y_sum,
        yy
    ));

    args[0] = (void*)&primitive_features;
    args[1] = &row_count;
    args[2] = &primitive_feature_stride;
    args[3] = &num_primitive_features;
    args[4] = (void*)&targets;
    args[5] = &target_rhs_stride;
    args[6] = &num_target_rhs;
    args[7] = (void*)&leaf_masks;
    args[8] = (void*)&leaf_words;
    args[9] = &leaf_words_feature_stride;
    args[10] = &gram;
    args[11] = &gram_col_stride;
    args[12] = &x_sum;
    args[13] = &xty;
    args[14] = &xty_rhs_stride;
    args[15] = &y_sum;
    args[16] = &yy;

    IFGL_CHECK_CUDA_RET(cuLaunchKernel(
        gram_module->gram_functions[kernel_index],
        (unsigned int)num_settings,
        1u,
        1u,
        IFGL_THREADS,
        1u,
        1u,
        0u,
        (CUstream)stream,
        args,
        NULL
    ));

    return IMPLICIT_SINDY_SUCCESS;
}
