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
#include <stdio.h>
#include <stdint.h>

#include <philox.cuh>

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32
#define KERMAC_ROW_NORMALIZE_FLAG_CENTER (1u << 0)
#define KERMAC_ROW_NORMALIZE_FLAG_SCALE  (1u << 1)

enum class RNGType {
    UNIFORM,
    NORMAL,
    UINT
};

template<RNGType rng_type, class T>
static
__forceinline__
__device__
auto 
philox4_template(
    curandStatePhilox4_32_10_t* state
) {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint32_t>);
	
	if constexpr (rng_type == RNGType::NORMAL && std::is_same_v<T, float>) {
        return philox_curand_normal4(state);
	} else if constexpr (rng_type == RNGType::UNIFORM && std::is_same_v<T, float>) {
		return philox_curand_uniform4(state);
    } else if constexpr (rng_type == RNGType::UINT && std::is_same_v<T, uint32_t>) {
        return philox_curand4(state);
    }
}

#define _NEAREST_LARGER_MULTIPLE(X,Y) ((((X) - 1) / Y) + 1)

template <RNGType rng_type, class T>
static
__device__
__forceinline__
void
kernel_rng_2D(
    uint64_t launch_id,
    uint64_t seed,
	const int64_t num_rows,
	const int64_t ld_rows,
	const int64_t num_cols,
	const T scale,
	const T shift,
	T*  __restrict__ data
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, uint32_t>);

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t state = philox_init(launch_id, seed, thread_id);

    int64_t warp_id = threadIdx.x / 32;
    int64_t warp_thread_id = threadIdx.x % 32;
    int64_t warps_per_block = blockDim.x / 32;
    int64_t global_warp_idx = warps_per_block * blockIdx.x + warp_id;

    int64_t warp_rows = _NEAREST_LARGER_MULTIPLE(num_rows, 128);

    int64_t global_num_warps = gridDim.x * warps_per_block;

    for (int64_t chunk_idx = global_warp_idx; chunk_idx < warp_rows * num_cols; chunk_idx += global_num_warps) {
        int64_t column = chunk_idx / warp_rows;
        int64_t row_chunk = chunk_idx % warp_rows;

        auto rng = philox4_template<rng_type,T>(&state);
		for (int64_t i = 0; i < 4; i++) {
            int64_t row = row_chunk * 128ull + i * 32 + warp_thread_id;
			if (row < num_rows) {
                T v;
                if constexpr (std::is_same_v<T, uint32_t>) {
                    v = reinterpret_cast<T*>(&rng)[i];
                } else {
                    v = scale * reinterpret_cast<T*>(&rng)[i] + shift;
                }
                data[column * ld_rows + row] = v;
			}
		}
    }
}

extern "C"
__global__
void
kernel_rng_uniform_f32(
    uint64_t launch_id,
    uint64_t seed,
    int64_t num_rows,
	int64_t ld_rows,
	int64_t num_cols,
	float scale,
	float shift,
	float* __restrict__ data
) {
    kernel_rng_2D<RNGType::UNIFORM, float>(
        launch_id,
        seed,
        num_rows,
        ld_rows,
        num_cols,
        scale,
        shift,
        data
    );
}

extern "C"
__global__
void
kernel_rng_normal_f32(
    uint64_t launch_id,
    uint64_t seed,
    int64_t num_rows,
	int64_t ld_rows,
	int64_t num_cols,
	float scale,
	float shift,
	float* __restrict__ data
) {
    kernel_rng_2D<RNGType::NORMAL, float>(
        launch_id,
        seed,
        num_rows,
        ld_rows,
        num_cols,
        scale,
        shift,
        data
    );
}

extern "C"
__global__
void
kernel_rng_u32(
    uint64_t launch_id,
    uint64_t seed,
    int64_t num_rows,
	int64_t ld_rows,
	int64_t num_cols,
	uint32_t* __restrict__ data
) {
    kernel_rng_2D<RNGType::UINT, uint32_t>(
        launch_id,
        seed,
        num_rows,
        ld_rows,
        num_cols,
        0,
        0,
        data
    );
}

template <
    int block_size, 
    bool accuracy, 
    class T
>
static
__forceinline__
__device__
void
template_kernel_mse_accuracy(
    int64_t M, int64_t N,
    const T* __restrict__ a,
    int64_t stride_a,
    int64_t batch_stride_a,
    const T* __restrict__ b,
    int64_t stride_b,
    int64_t batch_stride_b,
    T* __restrict__ mse_sums, 
    int64_t stride_mse_sums,
    int32_t* __restrict__ accuracy_counts,
    int64_t stride_accuracy_counts
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    int64_t lane_id = threadIdx.x % WARP_SIZE;
    int64_t warp_id = threadIdx.x / WARP_SIZE;

    static const int32_t NUM_WARPS_PER_BLOCK = block_size / WARP_SIZE;

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_num = blockIdx.z;

    T current_max_value_a = T(0);
    T current_max_value_b = T(0);

    if (row < M) {
        current_max_value_a = a[batch_num * batch_stride_a + 0 * stride_a + row];
        current_max_value_b = b[batch_num * batch_stride_b + 0 * stride_b + row];
    }

    int32_t current_max_index_a = 0;
    int32_t current_max_index_b = 0;

    T mse_accum = 0;
    if (row < M) {
        for (int64_t col = 0; col < N; col++) {
            int64_t idx_a = batch_num * batch_stride_a + col * stride_a + row;
            int64_t idx_b = batch_num * batch_stride_b + col * stride_b + row;
            T a_v = a[idx_a];
            T b_v = b[idx_b];
            T diff_v = a_v - b_v;
            mse_accum += diff_v * diff_v;

            if constexpr(accuracy) {
                if (a_v > current_max_value_a) {
                    current_max_value_a = a_v;
                    current_max_index_a = col;
                }

                if (b_v > current_max_value_b) {
                    current_max_value_b = b_v;
                    current_max_index_b = col;
                }
            }
        }
    }

    int32_t accuracy_count = 0;
    if constexpr(accuracy) {
        if (row < M) {
            if (N == 1) {
                T a_v = a[batch_num * batch_stride_a + row];
                T b_v = b[batch_num * batch_stride_b + row];
                accuracy_count = (a_v >= (T)0.5) == (b_v >= (T)0.5);
            } else {
                accuracy_count = current_max_index_a == current_max_index_b;
            }
        }
    }

    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2) {
        mse_accum += __shfl_down_sync(FULL_MASK, mse_accum, offset);
        accuracy_count += __shfl_down_sync(FULL_MASK, accuracy_count, offset);
    }

    __shared__ T smem_mse[NUM_WARPS_PER_BLOCK];
    __shared__ T smem_accuracy_counts[NUM_WARPS_PER_BLOCK];
    if (lane_id == 0) {
        smem_mse[warp_id] = mse_accum;
        if constexpr(accuracy) {
            smem_accuracy_counts[warp_id] = accuracy_count;
        }
    }

    __syncthreads();

    mse_accum = 0;
    accuracy_count = 0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int32_t i = 0; i < NUM_WARPS_PER_BLOCK; i++) {
            mse_accum += smem_mse[i];
            if constexpr(accuracy) {
                accuracy_count += smem_accuracy_counts[i];
            }
        }
        mse_sums[batch_num * stride_mse_sums + blockIdx.x] = mse_accum;
        if constexpr(accuracy) {
            accuracy_counts[batch_num * stride_accuracy_counts + blockIdx.x] = accuracy_count;
        }
    }
}

template <
    int block_size, 
    bool accuracy, 
    class T
>
static
__forceinline__
__device__
void
template_kernel_mse_accuracy_final_reduction(
    int64_t M, int64_t N,
    int32_t num_blocks,
    const T* const __restrict__ mse_sums,
    int64_t stride_mse_sums,
    const int32_t* const __restrict__ accuracy_counts,
    int64_t stride_accuracy_counts,
    T* __restrict__ mse,
    int64_t stride_mse,
    T* __restrict__ accuracy_percentage,
    int64_t stride_accuracy_percentage
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    int64_t lane_id = threadIdx.x % WARP_SIZE;
    int64_t warp_id = threadIdx.x / WARP_SIZE;

    int64_t batch_num = blockIdx.z;

    static const int32_t NUM_WARPS_PER_BLOCK = block_size / WARP_SIZE;

    T accum = 0;
    int32_t accuracy_count = 0;
    for (int32_t row = threadIdx.x; row < num_blocks; row += blockDim.x) {
        accum += mse_sums[batch_num * stride_mse_sums + row];
        accuracy_count += accuracy_counts[batch_num * stride_accuracy_counts + row];
    }

    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(FULL_MASK, accum, offset);
        accuracy_count += __shfl_down_sync(FULL_MASK, accuracy_count, offset);
    }

    __shared__ T smem_mse[NUM_WARPS_PER_BLOCK];
    __shared__ T smem_accuracy_counts[NUM_WARPS_PER_BLOCK];
    if (lane_id == 0) {
        smem_mse[warp_id] = accum;
        if constexpr(accuracy) {
            smem_accuracy_counts[warp_id] = accuracy_count;
        }
    }

    __syncthreads();

    accum = 0;
    accuracy_count = 0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int32_t i = 0; i < NUM_WARPS_PER_BLOCK; i++) {
            accum += smem_mse[i];
            accuracy_count += smem_accuracy_counts[i];
        }
        mse[batch_num] = accum / (T)(M * N);
        if constexpr(accuracy) {
            accuracy_percentage[batch_num] = accuracy_count / (T)(M);
        }
    }
}

extern "C"
__global__
void
kernel_mse_accuracy_256_f32(
    int64_t M, int64_t N,
    const float* __restrict__ a,
    int64_t stride_a,
    int64_t batch_stride_a,
    const float* __restrict__ b,
    int64_t stride_b,
    int64_t batch_stride_b,
    float* __restrict__ mse_sums, 
    int64_t stride_mse_sums,
    int32_t* __restrict__ accuracy_counts,
    int64_t stride_accuracy_counts
) {
    template_kernel_mse_accuracy<256, true, float>(
        M, N,
        a, stride_a, batch_stride_a,
        b, stride_b, batch_stride_b,
        mse_sums, stride_mse_sums,
        accuracy_counts, stride_accuracy_counts
    );
}

extern "C"
__global__
void
kernel_mse_256_f32(
    int64_t M, int64_t N,
    const float* __restrict__ a,
    int64_t stride_a,
    int64_t batch_stride_a,
    const float* __restrict__ b,
    int64_t stride_b,
    int64_t batch_stride_b,
    float* __restrict__ mse_sums, 
    int64_t stride_mse_sums,
    int32_t* __restrict__ accuracy_counts,
    int64_t stride_accuracy_counts
) {
    template_kernel_mse_accuracy<256, false, float>(
        M, N,
        a, stride_a, batch_stride_a,
        b, stride_b, batch_stride_b,
        mse_sums, stride_mse_sums,
        accuracy_counts, stride_accuracy_counts
    );
}

extern "C"
__global__
void
kernel_mse_final_reduction_1024_accuracy_f32(
    int64_t M, int64_t N,
    int32_t num_blocks,
    const float* const __restrict__ mse_sums,
    int64_t stride_mse_sums,
    const int32_t* const __restrict__ accuracy_counts,
    int64_t stride_accuracy_counts,
    float* __restrict__ mse,
    int64_t stride_mse,
    float* __restrict__ accuracy_percentage,
    int64_t stride_accuracy_percentage
) {
    template_kernel_mse_accuracy_final_reduction<1024, true, float>(
        M, N,
        num_blocks,
        mse_sums, stride_mse_sums,
        accuracy_counts, stride_accuracy_counts,
        mse, stride_mse, 
        accuracy_percentage,
        stride_accuracy_percentage
    );
}

extern "C"
__global__
void
kernel_mse_final_reduction_1024_f32(
    int64_t M, int64_t N,
    int32_t num_blocks,
    const float* const __restrict__ mse_sums,
    int64_t stride_mse_sums,
    const int32_t* const __restrict__ accuracy_counts,
    int64_t stride_accuracy_counts,
    float* __restrict__ mse,
    int64_t stride_mse,
    float* __restrict__ accuracy_percentage,
    int64_t stride_accuracy_percentage
) {
    template_kernel_mse_accuracy_final_reduction<1024, false, float>(
        M, N,
        num_blocks,
        mse_sums, stride_mse_sums,
        accuracy_counts, stride_accuracy_counts,
        mse, stride_mse, 
        accuracy_percentage,
        stride_accuracy_percentage
    );
}

// lg2.approx.ftz.f32 %f5, %f2;
// mul.ftz.f32 %f6, %f5, 0f3F317218;

static
inline
__device__
void
compute_logdet_norm_batch(
    int64_t num_rows,
    const float* __restrict__ L, 
    int64_t ld_L, 
    int64_t batch_stride_L,
    float* __restrict__ logdet_norm_out
) {
    unsigned int tid     = threadIdx.x;
    unsigned int lane    = tid & 31;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (blockDim.x + 31) >> 5;
   
    int64_t batch = blockIdx.x;

    L += batch * batch_stride_L;

    float local_sum = 0.0f;

    for (int64_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
        float L_ii = L[i + i * ld_L];
        L_ii = fmaxf(L_ii, 1e-30f);
        asm (
            "lg2.approx.ftz.f32 %0, %0;\n\t"
            "mul.ftz.f32 %0, %0, 0f3F317218;"
            : "+f"(L_ii)
        );
        local_sum += L_ii;
    }

    unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    
    __shared__ float warp_sums[32];

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        if (lane < num_warps) {
            block_sum = warp_sums[lane];
        } else {
            block_sum = 0.0f;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }

        if (lane == 0) {
            logdet_norm_out[batch] = (2.0f * block_sum) / (float)num_rows;
        }
    }
}

static
inline
__device__
void
compute_norm_H2_batch(
    float lambda_reg,
    int64_t  num_rows,
    int64_t  num_labels,
    const float* __restrict__ alpha, 
    int64_t batch_stride_alpha,
    int64_t label_stride_alpha,
    const float* __restrict__ y,
    int64_t batch_stride_y,
    int64_t label_stride_y,
    float* __restrict__ norm_H2_out
) {
    unsigned int tid       = threadIdx.x;
    unsigned int lane      = tid & 31;
    unsigned int warp_id   = tid >> 5;
    unsigned int num_warps = (blockDim.x + 31) >> 5;

    int64_t batch = blockIdx.x;

    alpha += batch * batch_stride_alpha;
    y += batch * batch_stride_y;

    float lambda = lambda_reg;

    float local_alpha_dot_y = 0.0f;
    float local_alpha_sq    = 0.0f;

    for (int64_t label = 0; label < num_labels; ++label) {
        const float* alpha_label = alpha + label * label_stride_alpha;
        const float* y_label = y + label * label_stride_y;
        for (int64_t i = tid; i < num_rows; i += blockDim.x) {
            float a  = alpha_label[i];
            float yi = y_label[i];
            local_alpha_dot_y += a * yi;
            local_alpha_sq += a * a;
        }
    }

    unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_alpha_dot_y += __shfl_down_sync(mask, local_alpha_dot_y, offset);
        local_alpha_sq += __shfl_down_sync(mask, local_alpha_sq,    offset);
    }

    __shared__ float2 warp_sums[32];

    if (lane == 0 && warp_id < 32) {
        warp_sums[warp_id] = make_float2(local_alpha_dot_y, local_alpha_sq);
    }
    __syncthreads();

    // Final reduction across warps using warp 0
    float2 block_val = make_float2(0.0f, 0.0f);
    if (warp_id == 0) {
        if (lane < num_warps) {
            block_val = warp_sums[lane];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            block_val.x += __shfl_down_sync(mask, block_val.x, offset);
            block_val.y += __shfl_down_sync(mask, block_val.y, offset);
        }

        if (lane == 0) {
            float alpha_dot_y = block_val.x;
            float alpha_sq    = block_val.y;
            float norm_H2     = alpha_dot_y - lambda * alpha_sq;
            norm_H2_out[batch] = norm_H2;
        }
    }
}

extern "C"
__global__
__launch_bounds__(1024)
void
kernel_logdet_norm_norm_h2(
    float lambda_reg,
    int64_t num_rows,
    int64_t num_labels,
    const float* __restrict__ L, int64_t ld_L, int64_t batch_stride_L,
    const float* __restrict__ alpha, int64_t batch_stride_alpha, int64_t label_stride_alpha,
    const float* __restrict__ y, int64_t batch_stride_y, int64_t label_stride_y,
    float* __restrict__ logdet_norm_out,
    float* __restrict__ norm_H2_out
) {
    compute_logdet_norm_batch(
        num_rows, L, ld_L, batch_stride_L,
        logdet_norm_out
    );

    compute_norm_H2_batch(
        lambda_reg,
        num_rows,
        num_labels,
        alpha, batch_stride_alpha, label_stride_alpha,
        y, batch_stride_y, label_stride_y,
        norm_H2_out
    );
}

#define ROW_STATS_BLOCK_SIZE 256
#define STDEV_EPS 1e-12f

struct WelfordF {
    int   n;
    float mean;
    float M2;
};

__device__
__forceinline__ 
WelfordF 
welford_combine(
    WelfordF a, 
    WelfordF b
) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;

    const float delta = b.mean - a.mean;
    const int   n     = a.n + b.n;
    const float inv_n = 1.0f / (float)n;

    WelfordF out;
    out.n    = n;
    out.mean = a.mean + delta * ((float)b.n * inv_n);
    out.M2   = a.M2 + b.M2 + (delta * delta) * ((float)a.n * (float)b.n * inv_n);
    return out;
}

__device__
__forceinline__ 
WelfordF 
welford_warp_reduce(
    WelfordF v
) {
    const unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        WelfordF o;
        o.n    = __shfl_down_sync(mask, v.n,    off);
        o.mean = __shfl_down_sync(mask, v.mean, off);
        o.M2   = __shfl_down_sync(mask, v.M2,   off);
        v = welford_combine(v, o);
    }
    return v;
}

extern "C"
__global__
void
kernel_row_stats(
    int64_t num_rows,
    int64_t num_cols,
    int64_t num_batches,
    const float* __restrict__ data, // [num_rows, num_cols, num_batches] num_rows is stride 1
    int64_t ld_data,
    int64_t batch_stride_data,
    float* __restrict__ mean_out,   // [num_batches, num_cols] num_batches is stride 1
    int64_t ld_mean,
    float* __restrict__ stdev_out,  // [num_batches, num_cols] num_batches is stride 1
    int64_t ld_stdev
) {
    const int lane_id         = (int)(threadIdx.x & 31);
    const int warp_id_in_block = (int)(threadIdx.x >> 5);
    const int warps_per_block  = ROW_STATS_BLOCK_SIZE >> 5;

    const int64_t task_count        = num_batches * num_cols; // one task per (batch, col)
    const int64_t warps_per_grid    = (int64_t)gridDim.x * (int64_t)warps_per_block;
    int64_t task_idx = (int64_t)blockIdx.x * (int64_t)warps_per_block + (int64_t)warp_id_in_block;

    for (; task_idx < task_count; task_idx += warps_per_grid) {
        const int64_t batch_idx = task_idx / num_cols;
        const int64_t col_idx   = task_idx - batch_idx * num_cols;

        int64_t col_offset = (size_t)batch_idx * (size_t)batch_stride_data + (size_t)col_idx   * (size_t)ld_data;
        const float* __restrict__ col_ptr = data + col_offset;

        WelfordF acc{0, 0.0f, 0.0f};

        for (int64_t row_idx = (int64_t)lane_id; row_idx < num_rows; row_idx += 32) {
            const float v = col_ptr[row_idx];
            acc.n++;
            const float d  = v - acc.mean;
            acc.mean += d / (float)acc.n;
            const float d2 = v - acc.mean;
            acc.M2   += d * d2;
        }

        acc = welford_warp_reduce(acc);

        if (lane_id == 0) {
            float var = 0.0f;
            if (acc.n >= 2) {
                var = acc.M2 / (float)acc.n;
            }
            if (mean_out != NULL) {
                mean_out[(size_t)col_idx * (size_t)ld_mean + (size_t)batch_idx] = acc.mean;
                stdev_out[(size_t)col_idx * (size_t)ld_stdev + (size_t)batch_idx] =
                    sqrtf(var + STDEV_EPS);
            } else {
                // Preserve legacy scale-only behavior: use RMS when no mean output is requested.
                const float rms_sq = var + acc.mean * acc.mean;
                stdev_out[(size_t)col_idx * (size_t)ld_stdev + (size_t)batch_idx] =
                    sqrtf(rms_sq + STDEV_EPS);
            }
        }
    }
}

#ifndef APPLY_ROW_STATS_BLOCK_SIZE
#define APPLY_ROW_STATS_BLOCK_SIZE 256
#endif

#ifndef STDEV_EPS
#define STDEV_EPS 1e-12f
#endif

extern "C"
__global__
void
kernel_apply_row_stats(
    int64_t num_rows,
    int64_t num_cols,
    int64_t num_batches,
    float* __restrict__ data,              // [num_rows, num_cols, num_batches]
    int64_t ld_data,
    int64_t batch_stride_data,
    const float* __restrict__ mean,        // [num_batches, num_cols] (L-major)
    int64_t ld_mean,
    const float* __restrict__ stdev,       // [num_batches, num_cols] (L-major)
    int64_t ld_stdev,
    uint32_t flags
) {
    const bool center_rows = (flags & KERMAC_ROW_NORMALIZE_FLAG_CENTER) != 0;
    const bool scale_rows = (flags & KERMAC_ROW_NORMALIZE_FLAG_SCALE) != 0;
    const int lane_id          = (int)(threadIdx.x & 31);
    const int warp_id_in_block = (int)(threadIdx.x >> 5);
    const int warps_per_block  = (int)(blockDim.x >> 5);

    const int64_t task_count     = num_batches * num_cols; // one task per (batch, col)
    const int64_t warps_per_grid = (int64_t)gridDim.x * (int64_t)warps_per_block;

    int64_t task_idx = (int64_t)blockIdx.x * (int64_t)warps_per_block + (int64_t)warp_id_in_block;

    for (; task_idx < task_count; task_idx += warps_per_grid) {
        const int64_t batch_idx = task_idx / num_cols;
        const int64_t col_idx   = task_idx - batch_idx * num_cols;

        float mean_value = 0.0f;
        float s = 1.0f;
        if (lane_id == 0) {
            if (center_rows && mean != NULL) {
                mean_value = mean[(size_t)col_idx * (size_t)ld_mean + (size_t)batch_idx];
            }
            if (scale_rows && stdev != NULL) {
                s = stdev[(size_t)col_idx * (size_t)ld_stdev + (size_t)batch_idx];
            }
        }
        mean_value = __shfl_sync(0xFFFFFFFFu, mean_value, 0);
        s = __shfl_sync(0xFFFFFFFFu, s, 0);

        s = scale_rows ? fmaxf(s, (float)STDEV_EPS) : 1.0f;
        const float inv_s = scale_rows ? __fdividef(1.0f, s) : 1.0f;

        const int64_t col_offset =
            (int64_t)((size_t)batch_idx * (size_t)batch_stride_data +
                      (size_t)col_idx   * (size_t)ld_data);

        float* __restrict__ col_ptr = data + col_offset;

        for (int64_t row_idx = (int64_t)lane_id; row_idx < num_rows; row_idx += 32) {
            float value = col_ptr[row_idx];
            if (center_rows) {
                value -= mean_value;
            }
            col_ptr[row_idx] = value * inv_s;
        }
    }
}

template <typename T>
__forceinline__
__device__ 
void 
broadcast_2d_to_3d(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t rows, 
    int64_t cols, 
    int64_t batches,
    int64_t ld_src,
    int64_t ld_dst,
    int64_t batch_stride_dst
) {
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    int64_t base_src = r + c * ld_src;
    T v = src[base_src];
    
    int64_t base_dst = r + c * ld_dst;
    for (int64_t b = 0; b < batches; ++b) {
        dst[base_dst + b * batch_stride_dst] = v;
    }
}

extern "C"
__global__
void
kernel_broadcast_2d_to_3d_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t rows,
    int64_t cols,
    int64_t batches,
    int64_t ld_src,
    int64_t ld_dst,
    int64_t batch_stride_dst
) {
    broadcast_2d_to_3d<float>(
        src, dst,
        rows,
        cols,
        batches,
        ld_src,
        ld_dst,
        batch_stride_dst
    );
}
