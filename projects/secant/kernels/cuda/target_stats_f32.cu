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
extern "C" __global__
void secant_cuda_target_stats_f32(
    const float* __restrict__ targets,
    size_t targets_leading_dimension,
    size_t num_rows,
    size_t num_targets,
    float* __restrict__ target_stats,
    size_t target_stats_leading_dimension
) {
    __shared__ float partial_stats[2][16];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    const unsigned int num_warps = blockDim.x >> 5u;
    const size_t target_idx = blockIdx.y;
    float target_sum = 0.0f;
    float target_square_sum = 0.0f;

    if (target_idx >= num_targets || num_warps == 0u || num_warps > 16u) {
        return;
    }

    for (size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         row < num_rows;
         row += (size_t)gridDim.x * blockDim.x) {
        const float target = targets[target_idx * targets_leading_dimension + row];

        target_sum += target;
        target_square_sum += target * target;
    }

    #pragma unroll
    for (unsigned int offset = 16u; offset != 0u; offset >>= 1u) {
        target_sum += __shfl_down_sync(0xffffffffu, target_sum, offset);
        target_square_sum += __shfl_down_sync(0xffffffffu, target_square_sum, offset);
    }
    if (lane == 0u) {
        partial_stats[0][warp] = target_sum;
        partial_stats[1][warp] = target_square_sum;
    }
    __syncthreads();

    if (warp == 0u) {
        target_sum = lane < num_warps ? partial_stats[0][lane] : 0.0f;
        target_square_sum = lane < num_warps ? partial_stats[1][lane] : 0.0f;
        #pragma unroll
        for (unsigned int offset = 16u; offset != 0u; offset >>= 1u) {
            target_sum += __shfl_down_sync(0xffffffffu, target_sum, offset);
            target_square_sum += __shfl_down_sync(0xffffffffu, target_square_sum, offset);
        }
        if (lane == 0u) {
            float* stats = target_stats + target_idx * target_stats_leading_dimension;

            atomicAdd(stats + 0u, target_sum);
            atomicAdd(stats + 1u, target_square_sum);
        }
    }
}
