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
#pragma once

#include <kermac.h>

#include <stdint.h>

// TODO: mse / accuracy

KERMAC_PUBLIC_DEC
KermacResult
kermac_logdet_norm_cpu(
    KermacStackAllocator* hsa,
    KermacTensor factored_matrix,
    KermacTensor logdet_norm,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_norm_H2_cpu(
    KermacStackAllocator* hsa,
    float lambda_reg,
    KermacTensor alpha,
    KermacTensor y,
    KermacTensor norm_H2,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_max_diff_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    float* max_diff_out,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_p_norm_kernel_cpu(
    KermacStackAllocator* hsa,
    float p_power,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_p_norm_kernel_gradient_cpu(
    KermacStackAllocator* hsa,
    float p_power,
    KermacTensor kernel_matrix,
    KermacTensor data_n,
    KermacTensor solution,
    KermacTensor data_m,
    KermacTensor gradient,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_gemm_gold_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_l1_gold_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_l2_gold_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_fmnist_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_fmnist_load(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_fmnist_load_u8(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_mnist_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_mnist_load(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_mnist_load_u8(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_cifar10_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_cifar10_load(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_cifar10_load_u32(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
);

typedef enum {
    KERMAC_RELSPRITES_MODE_BINARY = 0,
    KERMAC_RELSPRITES_MODE_GRAY   = 1,
    KERMAC_RELSPRITES_MODE_COLOR  = 2
} KermacRelSpritesMode;

typedef struct {
    int32_t w;
    int32_t h;
    int32_t n;
    int32_t distractors;
    float noise;
    uint32_t seed;
    KermacRelSpritesMode mode;
} KermacRelSpritesConfig;

#define KERMAC_RELSPRITES_LABEL_COUNT 14

KERMAC_PUBLIC_DEC
KermacResult
kermac_relsprites_dims(
    const KermacRelSpritesConfig* cfg,
    int64_t* num_samples,
    int64_t* num_rows,
    int64_t* num_cols,
    int64_t* num_channels,
    int64_t* num_labels
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_relsprites_generate(
    const KermacRelSpritesConfig* cfg,
    KermacTensor images,
    KermacTensor labels
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_relsprites_generate_u32(
    const KermacRelSpritesConfig* cfg,
    KermacTensor images,
    KermacTensor labels
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_relsprites_generate_u8(
    const KermacRelSpritesConfig* cfg,
    KermacTensor images,
    KermacTensor labels
);
