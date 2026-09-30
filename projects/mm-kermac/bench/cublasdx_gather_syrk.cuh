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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <cublasdx.hpp>

namespace kermac_cublasdx_gather_syrk {

inline bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

inline unsigned int get_cuda_device_arch() {
    int device = 0;
    cudaGetDevice(&device);
    int major = 0;
    int minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return static_cast<unsigned int>(major) * 100u + static_cast<unsigned int>(minor) * 10u;
}

template <unsigned int Arch, int Cols, int KTile>
using GatherSyrkBlas = decltype(
    cublasdx::Size<Cols, Cols, KTile>() +
    cublasdx::Precision<float>() +
    cublasdx::Type<cublasdx::type::real>() +
    cublasdx::Function<cublasdx::function::MM>() +
    cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
    cublasdx::Block() +
    cublasdx::SM<Arch>()
);

template <class BLAS, int Cols, int KTile>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void gather_syrk_batched_kernel(
    const float* __restrict__ x,
    int64_t x_ld,
    const int32_t* __restrict__ indices,
    int64_t indices_batch_stride,
    float* __restrict__ out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count
) {
    const int batch = static_cast<int>(blockIdx.x);
    if (batch >= batch_count) {
        return;
    }

    const int tid =
        static_cast<int>(threadIdx.x) +
        static_cast<int>(blockDim.x) * (
            static_cast<int>(threadIdx.y) +
            static_cast<int>(blockDim.y) * static_cast<int>(threadIdx.z)
        );
    const int threads = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);

    extern __shared__ __align__(16) unsigned char smem[];
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<BLAS>(smem);
    auto a_shared = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    __shared__ int32_t batch_indices[Cols];
    const int32_t* batch_indices_src = indices + static_cast<int64_t>(batch) * indices_batch_stride;

    for (int idx = tid; idx < Cols; idx += threads) {
        batch_indices[idx] = batch_indices_src[idx];
    }
    for (int linear = tid; linear < Cols * Cols; linear += threads) {
        const int row = linear % Cols;
        const int col = linear / Cols;
        c_shared(row, col) = 0.0f;
    }
    __syncthreads();

    for (int row0 = 0; row0 < num_rows; row0 += KTile) {
        const int rows_this_tile = min(KTile, num_rows - row0);

        for (int linear = tid; linear < Cols * KTile; linear += threads) {
            const int row = linear % Cols;
            const int k = linear / Cols;
            float value = 0.0f;
            if (k < rows_this_tile) {
                const int32_t feature = batch_indices[row];
                if (feature >= 0) {
                    value = x[static_cast<int64_t>(row0 + k) + static_cast<int64_t>(feature) * x_ld];
                }
            }
            a_shared(row, k) = value;
            b_shared(k, row) = value;
        }
        __syncthreads();

        BLAS().execute(1.0f, a_shared, b_shared, 1.0f, c_shared);
        __syncthreads();
    }

    float* out_batch = out + static_cast<int64_t>(batch) * out_batch_stride;
    for (int linear = tid; linear < Cols * Cols; linear += threads) {
        const int row = linear % Cols;
        const int col = linear / Cols;
        out_batch[static_cast<int64_t>(row) + static_cast<int64_t>(col) * out_ld] = c_shared(row, col);
    }
}

template <unsigned int Arch, int Cols, int KTile>
cudaError_t launch_gather_syrk_batched_kernel(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    using BLAS = GatherSyrkBlas<Arch, Cols, KTile>;
    const unsigned shared_mem_bytes = cublasdx::get_shared_storage_size<BLAS>();
    cudaError_t status = cudaFuncSetAttribute(
        gather_syrk_batched_kernel<BLAS, Cols, KTile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_mem_bytes)
    );
    if (status != cudaSuccess) {
        return status;
    }

    gather_syrk_batched_kernel<BLAS, Cols, KTile>
        <<<static_cast<unsigned>(batch_count), BLAS::block_dim, shared_mem_bytes, stream>>>(
            x,
            x_ld,
            indices,
            indices_batch_stride,
            out,
            out_ld,
            out_batch_stride,
            num_rows,
            batch_count
        );
    return cudaPeekAtLastError();
}

inline void reference_gather_syrk(
    const std::vector<float>& x,
    int64_t x_ld,
    const std::vector<int32_t>& indices,
    int64_t indices_batch_stride,
    std::vector<float>* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    int cols
) {
    out->assign(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(batch_count), 0.0f);
    for (int batch = 0; batch < batch_count; ++batch) {
        const int32_t* batch_indices = indices.data() + static_cast<int64_t>(batch) * indices_batch_stride;
        float* out_batch = out->data() + static_cast<int64_t>(batch) * out_batch_stride;
        for (int col_j = 0; col_j < cols; ++col_j) {
            const int32_t feature_j = batch_indices[col_j];
            for (int col_i = 0; col_i < cols; ++col_i) {
                const int32_t feature_i = batch_indices[col_i];
                double acc = 0.0;
                for (int row = 0; row < num_rows; ++row) {
                    const float a = (feature_i >= 0)
                        ? x[static_cast<int64_t>(row) + static_cast<int64_t>(feature_i) * x_ld]
                        : 0.0f;
                    const float b = (feature_j >= 0)
                        ? x[static_cast<int64_t>(row) + static_cast<int64_t>(feature_j) * x_ld]
                        : 0.0f;
                    acc += static_cast<double>(a) * static_cast<double>(b);
                }
                out_batch[static_cast<int64_t>(col_i) + static_cast<int64_t>(col_j) * out_ld] =
                    static_cast<float>(acc);
            }
        }
    }
}

inline float max_abs_diff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs
) {
    const size_t n = std::min(lhs.size(), rhs.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

}  // namespace kermac_cublasdx_gather_syrk
