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
#ifndef IMPLICIT_FEATURE_COLUMNS_CUH_INCLUDE
#define IMPLICIT_FEATURE_COLUMNS_CUH_INCLUDE

#define IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL 1
#include "implicit_feature_gram.cuh"

namespace implicit_sindy_feature_gram_kernel {

__device__ __forceinline__
void
write_implicit_columns_tile(
    const SharedStorage& storage,
    int row0,
    int row_end,
    int setting_idx,
    float* output,
    int64_t output_setting_stride,
    int64_t output_feature_stride
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int total = kFeatures * kRowsPerTile;

    for (int linear = tid; linear < total; linear += kThreads) {
        const int feature = linear / kRowsPerTile;
        const int row = linear - feature * kRowsPerTile;
        const int global_row = row0 + row;

        if (global_row < row_end) {
            output[
                static_cast<int64_t>(setting_idx) * output_setting_stride +
                static_cast<int64_t>(feature) * output_feature_stride +
                static_cast<int64_t>(global_row)
            ] = storage.implicit_panel[linear];
        }
    }
}

template <int KernelIdx>
__device__ __forceinline__
void
implicit_feature_columns_kernel_device(
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
    __shared__ __align__(16) SharedStorage storage;
    const int setting_idx = static_cast<int>(blockIdx.x);
    const int row_end = static_cast<int>(row_count);

    setup_leaf_refs(
        leaf_masks,
        leaf_words,
        leaf_words_feature_stride,
        num_primitive_features,
        storage,
        setting_idx
    );
    __syncthreads();

    for (int row0 = 0; row0 < row_end; row0 += kRowsPerTile) {
        load_primitive_tile(
            primitive_features,
            primitive_feature_stride,
            num_primitive_features,
            storage,
            row0,
            row_end
        );
        __syncthreads();

        evaluate_implicit_tile<KernelIdx>(
            storage,
            row0,
            row_end
        );
        __syncthreads();

        write_implicit_columns_tile(
            storage,
            row0,
            row_end,
            setting_idx,
            output,
            output_setting_stride,
            output_feature_stride
        );
        __syncthreads();
    }
}

}  // namespace implicit_sindy_feature_gram_kernel

#endif /* IMPLICIT_FEATURE_COLUMNS_CUH_INCLUDE */
