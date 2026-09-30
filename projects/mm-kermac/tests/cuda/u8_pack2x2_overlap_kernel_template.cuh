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

#include <stdint.h>

#include <ptx_inject.h>

template <int IMAGE_N>
__device__ __forceinline__ void
u8_pack2x2_overlap_kernel_body(
    int64_t num_batches,
    const uint8_t* in,
    int64_t batch_stride_in,
    uint32_t* out,
    int64_t batch_stride_out
) {
    static_assert(IMAGE_N >= 2, "IMAGE_N must be at least 2");

    constexpr int kWarpSize = 32;
    constexpr int kImageElems = IMAGE_N * IMAGE_N;
    constexpr int kOutSide = IMAGE_N - 1;
    constexpr int kOutElems = kOutSide * kOutSide;

    __shared__ uint8_t smem_image[kImageElems];

    const unsigned int lane = threadIdx.x & (kWarpSize - 1);
    if (threadIdx.x >= kWarpSize || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }

    for (int64_t batch = (int64_t)blockIdx.x; batch < num_batches; batch += (int64_t)gridDim.x) {
        const int64_t in_base = batch * batch_stride_in;

        for (int idx = (int)lane; idx < kImageElems; idx += kWarpSize) {
            smem_image[idx] = in[in_base + (int64_t)idx];
        }
        __syncwarp();

        const int64_t out_base = batch * batch_stride_out;
        for (int idx = (int)lane; idx < kOutElems; idx += kWarpSize) {
            const int row = idx / kOutSide;
            const int col = idx - row * kOutSide;

            const int top_left = row * IMAGE_N + col;
            const uint32_t p00 = (uint32_t)smem_image[top_left];
            const uint32_t p01 = (uint32_t)smem_image[top_left + 1];
            const uint32_t p10 = (uint32_t)smem_image[top_left + IMAGE_N];
            const uint32_t p11 = (uint32_t)smem_image[top_left + IMAGE_N + 1];

            uint32_t packed = p00 | (p01 << 8) | (p10 << 16) | (p11 << 24);
            uint32_t transformed = 0;
            PTX_INJECT(
                "transform",
                PTX_IN(U32, in_u32, packed),
                PTX_OUT(U32, out_u32, transformed)
            );

            out[out_base + (int64_t)idx] = transformed;
        }
        __syncwarp();
    }
}
