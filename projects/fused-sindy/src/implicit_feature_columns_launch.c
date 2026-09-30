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

#define IFCL_ERROR_RET(ans)                         \
    do {                                           \
        ImplicitSindyResult ifcl_result__ = (ans); \
        return ifcl_result__;                      \
    } while (0)

#define IFCL_CHECK_RET(ans)                         \
    do {                                           \
        ImplicitSindyResult ifcl_result__ = (ans); \
        if (ifcl_result__ != IMPLICIT_SINDY_SUCCESS) { \
            IFCL_ERROR_RET(ifcl_result__);         \
        }                                          \
    } while (0)

#define IFCL_CHECK_CUDA_RET(ans)                    \
    do {                                           \
        CUresult ifcl_cuda_result__ = (ans);       \
        if (ifcl_cuda_result__ != CUDA_SUCCESS) {  \
            IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_CUDA); \
        }                                          \
    } while (0)

enum {
    IFCL_THREADS = 128,
    IFCL_MAX_KERNELS_PER_MODULE = 64
};

struct ImplicitSindyColumnsModule {
    CUmodule module;
    size_t kernels_per_module;
    CUfunction columns_functions[];
};

static size_t
ifcl_kernels_per_module_count(
    size_t kernels_per_module
) {
    if (kernels_per_module == 0u || kernels_per_module > IFCL_MAX_KERNELS_PER_MODULE) return 0u;
    return kernels_per_module;
}

static ImplicitSindyResult
ifcl_columns_kernel_symbol(
    char* buffer,
    size_t buffer_size,
    size_t kernels_per_module,
    size_t kernel_index
) {
    int bytes_written;

    if (buffer == NULL || buffer_size == 0u || kernel_index >= kernels_per_module) {
        IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    if (kernels_per_module == 1u && kernel_index == 0u) {
        if (strlen("implicit_feature_columns_kernel") + 1u > buffer_size) {
            IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
        }
        strcpy(buffer, "implicit_feature_columns_kernel");
        return IMPLICIT_SINDY_SUCCESS;
    }
    bytes_written = snprintf(
        buffer,
        buffer_size,
        "implicit_feature_columns_kernel_%zu",
        kernel_index
    );
    if (bytes_written < 0 || (size_t)bytes_written >= buffer_size) {
        IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
ifcl_create_impl(
    ImplicitSindyColumnsModule* columns_module,
    void* module,
    size_t kernels_per_module
) {
    size_t kernel_count;
    size_t i;

    if (columns_module == NULL || module == NULL) IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);

    kernel_count = ifcl_kernels_per_module_count(kernels_per_module);
    if (kernel_count == 0u) IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);

    columns_module->module = (CUmodule)module;
    columns_module->kernels_per_module = kernel_count;

    for (i = 0u; i < kernel_count; ++i) {
        char symbol[64];
        IFCL_CHECK_RET(ifcl_columns_kernel_symbol(
            symbol,
            sizeof(symbol),
            kernel_count,
            i
        ));
        IFCL_CHECK_CUDA_RET(cuModuleGetFunction(
            &columns_module->columns_functions[i],
            columns_module->module,
            symbol
        ));
    }

    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_columns_module_create(
    void* module,
    size_t kernels_per_module,
    ImplicitSindyColumnsModule** out_columns_module
) {
    ImplicitSindyColumnsModule* columns_module;
    ImplicitSindyResult error_ret;
    size_t kernel_count;
    size_t allocation_size;

    if (out_columns_module == NULL) IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out_columns_module = NULL;

    kernel_count = ifcl_kernels_per_module_count(kernels_per_module);
    if (kernel_count == 0u ||
        kernel_count > (SIZE_MAX - sizeof(*columns_module)) / sizeof(columns_module->columns_functions[0])) {
        IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    allocation_size = sizeof(*columns_module) + kernel_count * sizeof(columns_module->columns_functions[0]);
    columns_module = (ImplicitSindyColumnsModule*)calloc(1u, allocation_size);
    if (columns_module == NULL) IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    error_ret = ifcl_create_impl(
        columns_module,
        module,
        kernels_per_module
    );
    if (error_ret != IMPLICIT_SINDY_SUCCESS) {
        free(columns_module);
        IFCL_ERROR_RET(error_ret);
    }

    *out_columns_module = columns_module;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_columns_module_destroy(
    ImplicitSindyColumnsModule* columns_module
) {
    free(columns_module);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
ifcl_validate_grid(
    int64_t num_settings
) {
    if (num_settings <= 0 || num_settings > (int64_t)UINT_MAX) {
        IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
ifcl_validate_columns_launch(
    ImplicitSindyColumnsModule* columns_module,
    size_t kernel_index,
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
    if (columns_module == NULL ||
        kernel_index >= columns_module->kernels_per_module ||
        columns_module->columns_functions[kernel_index] == NULL ||
        primitive_features == NULL ||
        row_count <= 0 ||
        primitive_feature_stride <= 0 ||
        num_primitive_features <= 0 ||
        num_primitive_features > IMPLICIT_SINDY_INTERNAL_PRIMITIVE_FEATURES_MAX ||
        leaf_masks == NULL ||
        leaf_words == NULL ||
        leaf_words_feature_stride < IMPLICIT_SINDY_INTERNAL_LEAVES ||
        output == NULL ||
        output_setting_stride <= 0 ||
        output_feature_stride < row_count) {
        IFCL_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    IFCL_CHECK_RET(ifcl_validate_grid(num_settings));
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
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
) {
    void* args[10];

    IFCL_CHECK_RET(ifcl_validate_columns_launch(
        columns_module,
        kernel_index,
        num_settings,
        primitive_features,
        row_count,
        primitive_feature_stride,
        num_primitive_features,
        leaf_masks,
        leaf_words,
        leaf_words_feature_stride,
        output,
        output_setting_stride,
        output_feature_stride
    ));

    args[0] = (void*)&primitive_features;
    args[1] = &row_count;
    args[2] = &primitive_feature_stride;
    args[3] = &num_primitive_features;
    args[4] = (void*)&leaf_masks;
    args[5] = (void*)&leaf_words;
    args[6] = &leaf_words_feature_stride;
    args[7] = &output;
    args[8] = &output_setting_stride;
    args[9] = &output_feature_stride;

    IFCL_CHECK_CUDA_RET(cuLaunchKernel(
        columns_module->columns_functions[kernel_index],
        (unsigned int)num_settings,
        1u,
        1u,
        IFCL_THREADS,
        1u,
        1u,
        0u,
        (CUstream)stream,
        args,
        NULL
    ));

    return IMPLICIT_SINDY_SUCCESS;
}
