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

#include <astrii.cuh>

namespace astrii {

static const u64 MAX_D = 8192;
static const u64 BLOCK_SIZE = 256;
static const u64 NUM_REG_PER_THREAD = MAX_D / BLOCK_SIZE;
static const i32 NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

template <class T>
__device__ __forceinline__
T
_block_sum(
    i32 lane_id, 
    i32 warp_id, 
    T v
) {
    __shared__ T reduce_smem[NUM_WARPS_PER_BLOCK];

    #pragma unroll
    for (i32 offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }

    if (lane_id == 0) {
        reduce_smem[warp_id] = v;
    }

    __syncthreads();

    v = 0.0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (i32 i = 0; i < NUM_WARPS_PER_BLOCK; i++) {
            v += reduce_smem[i];
        }
    }
    return v;
}

template <class T>
__device__ __forceinline__
T
_norm(
    i32 lane_id, 
    i32 warp_id,
    T (&x)[NUM_REG_PER_THREAD]
) {
    T norm = 0.0;
    #pragma unroll
    for (i32 i = 0; i < NUM_REG_PER_THREAD; i++) {
        T v = x[i];
        norm += v * v;
    }

    norm = _block_sum(lane_id, warp_id, norm);
    return sqrt(norm);
}

// TODO: think about a deeper pipeline for m
template <class T>
__global__
void
kernel_top_eigenvector(
    u64 D, u64 num_iters,
    const T *__restrict__ m, // D,D,B
    u64 stride_m,
    T *__restrict__ v, // D,B
    u64 stride_v
) {
    u64 lane_id = threadIdx.x % WARP_SIZE;
    u64 warp_id = threadIdx.x / WARP_SIZE;

    u64 b = blockIdx.x;
    u64 col = threadIdx.x;
    static const u64 PIPE_M = 2;

    T local_v[NUM_REG_PER_THREAD];
    T local_m[PIPE_M][NUM_REG_PER_THREAD];

    bool toggle_m = false;

    #pragma unroll
    for (u64 i = 0; i < NUM_REG_PER_THREAD; i++) {
        u64 gmem_idx = b * stride_v + i * BLOCK_SIZE + col;
        local_v[i] = v[gmem_idx];
    }

    __syncthreads();
    
    for (i32 iter = 0; iter < num_iters; iter++) {
        // prefetch first row of matrix
        #pragma unroll
        for (u64 i = 0; i < NUM_REG_PER_THREAD; i++) {
            u64 row = 0;
            u64 idx = b * D * stride_m + row * stride_m + i * BLOCK_SIZE + col;
            local_m[toggle_m][i] = m[idx];
        }

        __shared__ T smem_v[MAX_D];
        for (u64 row = 0; row < D; row++) {
            // prefetch next row of matrix
            if (row < D-1) {
                #pragma unroll
                for (u64 i = 0; i < NUM_REG_PER_THREAD; i++) {
                    u64 idx = b * D * stride_m + (row + 1) * stride_m + i * BLOCK_SIZE + col;
                    local_m[!toggle_m][i] = m[idx];
                }
            }

            T dot_product = 0.0;
            #pragma unroll
            for (u64 i = 0; i < NUM_REG_PER_THREAD; i++) {
                dot_product += local_m[toggle_m][i] * local_v[i];
            }
            dot_product = _block_sum(lane_id, warp_id, dot_product);
            if (threadIdx.x == 0) {
                smem_v[row] = dot_product;
            }

            toggle_m = !toggle_m;
        }

        __syncthreads();

        // Load vector in to registers from smem
        #pragma unroll
        for (i32 i = 0; i < NUM_REG_PER_THREAD; i++) {
            u64 idx = i * BLOCK_SIZE + col;
            local_v[i] = smem_v[idx];
        }

        // Normalize vector
        __shared__ T norm_value;
        T norm = _norm(lane_id, warp_id, local_v);
        if (threadIdx.x == 0) {
            norm_value = norm;
        }

        __syncthreads();

        norm = norm_value;
        #pragma unroll
        for (i32 i = 0; i < NUM_REG_PER_THREAD; i++) {
            local_v[i] = local_v[i] / norm;
        }
    }

    #pragma unroll
    for (u64 i = 0; i < NUM_REG_PER_THREAD; i++) {
        u64 idx = b * stride_v + i * BLOCK_SIZE + col;
        v[idx] = local_v[i]; 
    }
}

// TODO: Make this kernel support D less than 8192
// It currently reads out of bounds
template <class T>
void
top_eigenvector_D_8192(
    Philox &philox,
    u64 num_iters,
    DeviceMultiTensor<T> &m, // IN: D,D,B
    DeviceMultiTensor<T> &v, // OUT: D,B
    cudaStream_t stream
) {
    static_assert(std::is_same_v<T, f32>);
    
    ASSERT( m.num_modes == 3 );
    ASSERT( v.num_modes == 2 );
    
    u64 D = v.extent[0];
    u64 B = v.extent[1];
    
    ASSERT( D == MAX_D );

    ASSERT( D == m.extent[0] );
    ASSERT( D == m.extent[1] );
    ASSERT( D == v.extent[0] );

    ASSERT( B == m.extent[2] );
    ASSERT( B == v.extent[1] );

    tensor_rng(philox, RNGType::UNIFORM, v.tensor, c_two<T>, c_negative_one<T>, stream);
    
    kernel_top_eigenvector<<<B, BLOCK_SIZE, 0, stream>>>(
        D, 
        num_iters,
        m.ptr(),
        m.stride[1],
        v.ptr(),
        v.stride[1]
    );
}

template <class T>
void
top_eigenvalue(
    CUTensor &cutensor,
    DeviceStackAllocator &dsa,
    DeviceMultiTensor<T> &m, // D,D,B
    DeviceMultiTensor<T> &v, // D,B
    DeviceMultiTensor<T> &eigenvalue, // B
    cudaStream_t stream
) {
    cutensor_contraction_trinary(
        cutensor,
        dsa, 
        TensorCoreMode::FP32,
        c_one<T>,
        v, "ib",
        m, "ijb",
        v, "jb",
        c_zero<T>,
        eigenvalue, "b",
        eigenvalue, "b",
        stream
    );
}

}
