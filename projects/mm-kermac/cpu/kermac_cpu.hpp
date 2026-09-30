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

#include <kermac.hpp>
#include <kermac_cpu.h>

namespace kermac {

inline
void
logdet_norm_cpu(
    HostStackAllocator& hsa,
    DeviceTensor<float>& factored_matrix,
    DeviceTensor<float>& logdet_norm,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_logdet_norm_cpu(
            hsa.get_ptr(),
            factored_matrix.raw(),
            logdet_norm.raw(),
            stream
        )
    );
}

inline
void
norm_H2_cpu(
    HostStackAllocator& hsa,
    float lambda_reg,
    DeviceTensor<float>& alpha,
    DeviceTensor<float>& y,
    DeviceTensor<float>& norm_H2,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_norm_H2_cpu(
            hsa.get_ptr(),
            lambda_reg,
            alpha.raw(),
            y.raw(),
            norm_H2.raw(),
            stream
        )
    );
}

inline
void
max_diff_cpu(
    HostStackAllocator& hsa,
    DeviceTensor<float>& a,
    DeviceTensor<float>& b,
    float* max_diff_out,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_max_diff_cpu(
            hsa.get_ptr(),
            a.raw(),
            b.raw(),
            max_diff_out,
            stream
        )
    );
}

inline
void
p_norm_kernel_cpu(
    HostStackAllocator& hsa,
    float p_power,
    DeviceTensor<float>& a,
    DeviceTensor<float>& b,
    DeviceTensor<float>& c,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_p_norm_kernel_cpu(
            hsa.get_ptr(),
            p_power,
            a.raw(),
            b.raw(),
            c.raw(),
            stream
        )
    );
}

inline
void
p_norm_kernel_gradient_cpu(
    HostStackAllocator& hsa,
    float p_power,
    DeviceTensor<float>& kernel_matrix,
    DeviceTensor<float>& data_n,
    DeviceTensor<float>& solution,
    DeviceTensor<float>& data_m,
    DeviceTensor<float>& gradient,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_p_norm_kernel_gradient_cpu(
            hsa.get_ptr(),
            p_power,
            kernel_matrix.raw(),
            data_n.raw(),
            solution.raw(),
            data_m.raw(),
            gradient.raw(),
            stream
        )
    );
}

template <MemorySpace SpaceA, MemorySpace SpaceB, MemorySpace SpaceC>
inline
void
gemm_gold_cpu(
    HostStackAllocator& hsa,
    Tensor<float, SpaceA>& a,
    Tensor<float, SpaceB>& b,
    Tensor<float, SpaceC>& c,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_gemm_gold_cpu(
            hsa.get_ptr(),
            a.raw(),
            b.raw(),
            c.raw(),
            stream
        )
    );
}

template <MemorySpace SpaceA, MemorySpace SpaceB, MemorySpace SpaceC>
inline
void
l1_gold_cpu(
    HostStackAllocator& hsa,
    Tensor<float, SpaceA>& a,
    Tensor<float, SpaceB>& b,
    Tensor<float, SpaceC>& c,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_l1_gold_cpu(
            hsa.get_ptr(),
            a.raw(),
            b.raw(),
            c.raw(),
            stream
        )
    );
}

template <MemorySpace SpaceA, MemorySpace SpaceB, MemorySpace SpaceC>
inline
void
l2_gold_cpu(
    HostStackAllocator& hsa,
    Tensor<float, SpaceA>& a,
    Tensor<float, SpaceB>& b,
    Tensor<float, SpaceC>& c,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_l2_gold_cpu(
            hsa.get_ptr(),
            a.raw(),
            b.raw(),
            c.raw(),
            stream
        )
    );
}

inline
void
fmnist_dims(
    int64_t max_num_train,
    int64_t max_num_test,
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_fmnist_dims(
            num_train,
            num_test,
            num_dims,
            num_labels
        )
    );

    *num_train = *num_train > max_num_train ? max_num_train : *num_train;
    *num_test = *num_test > max_num_test ? max_num_test : *num_test;
}

inline
void
fmnist_load(
    HostTensor<float>& x_train,
    HostTensor<float>& y_train,
    HostTensor<float>& x_test,
    HostTensor<float>& y_test
) {

    KERMAC_THROW_IF_ERROR(
        ::kermac_fmnist_load(
            x_train.raw(),
            y_train.raw(),
            x_test.raw(),
            y_test.raw()
        )
    );
}

inline
void
fmnist_load_u8(
    HostTensor<uint8_t>& x_train,
    HostTensor<uint8_t>& y_train,
    HostTensor<uint8_t>& x_test,
    HostTensor<uint8_t>& y_test
) {

    KERMAC_THROW_IF_ERROR(
        ::kermac_fmnist_load_u8(
            x_train.raw(),
            y_train.raw(),
            x_test.raw(),
            y_test.raw()
        )
    );
}

inline
void
mnist_dims(
    int64_t max_num_train,
    int64_t max_num_test,
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_mnist_dims(
            num_train,
            num_test,
            num_dims,
            num_labels
        )
    );

    *num_train = *num_train > max_num_train ? max_num_train : *num_train;
    *num_test = *num_test > max_num_test ? max_num_test : *num_test;
}

inline
void
mnist_load(
    HostTensor<float>& x_train,
    HostTensor<float>& y_train,
    HostTensor<float>& x_test,
    HostTensor<float>& y_test
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_mnist_load(
            x_train.raw(),
            y_train.raw(),
            x_test.raw(),
            y_test.raw()
        )
    );
}

inline
void
mnist_load_u8(
    HostTensor<uint8_t>& x_train,
    HostTensor<uint8_t>& y_train,
    HostTensor<uint8_t>& x_test,
    HostTensor<uint8_t>& y_test
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_mnist_load_u8(
            x_train.raw(),
            y_train.raw(),
            x_test.raw(),
            y_test.raw()
        )
    );
}

inline
void
cifar10_dims(
    int64_t max_num_train,
    int64_t max_num_test,
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_cifar10_dims(
            num_train,
            num_test,
            num_dims,
            num_labels
        )
    );

    *num_train = *num_train > max_num_train ? max_num_train : *num_train;
    *num_test = *num_test > max_num_test ? max_num_test : *num_test;
}

inline
void
cifar10_load(
    HostTensor<float>& x_train,
    HostTensor<float>& y_train,
    HostTensor<float>& x_test,
    HostTensor<float>& y_test
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_cifar10_load(
            x_train.raw(),
            y_train.raw(),
            x_test.raw(),
            y_test.raw()
        )
    );
}

inline
void
cifar10_load_u32(
    HostTensor<uint32_t>& x_train,
    HostTensor<uint8_t>& y_train,
    HostTensor<uint32_t>& x_test,
    HostTensor<uint8_t>& y_test
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_cifar10_load_u32(
            x_train.raw(),
            y_train.raw(),
            x_test.raw(),
            y_test.raw()
        )
    );
}

using RelSpritesMode = KermacRelSpritesMode;
using RelSpritesConfig = KermacRelSpritesConfig;

inline
void
relsprites_dims(
    const RelSpritesConfig& cfg,
    int64_t* num_samples,
    int64_t* num_rows,
    int64_t* num_cols,
    int64_t* num_channels,
    int64_t* num_labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_relsprites_dims(
            &cfg,
            num_samples,
            num_rows,
            num_cols,
            num_channels,
            num_labels
        )
    );
}

inline
void
relsprites_generate(
    const RelSpritesConfig& cfg,
    HostTensor<float>& images,
    HostTensor<float>& labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_relsprites_generate(
            &cfg,
            images.raw(),
            labels.raw()
        )
    );
}

inline
void
relsprites_generate_u32(
    const RelSpritesConfig& cfg,
    HostTensor<uint32_t>& images,
    HostTensor<float>& labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_relsprites_generate_u32(
            &cfg,
            images.raw(),
            labels.raw()
        )
    );
}

inline
void
relsprites_generate_u8(
    const RelSpritesConfig& cfg,
    HostTensor<uint8_t>& images,
    HostTensor<float>& labels
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_relsprites_generate_u8(
            &cfg,
            images.raw(),
            labels.raw()
        )
    );
}

}
