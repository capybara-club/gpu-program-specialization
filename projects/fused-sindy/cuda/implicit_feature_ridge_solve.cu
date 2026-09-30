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
#include "implicit_sindy_internal_constants.h"

#include <cusolverdx.hpp>

#ifndef __CUDACC_RTC__
#include <math.h>
#include <stdint.h>
#endif

#ifdef __CUDACC_RTC__
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#endif

#ifndef IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM
#error "IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM must be defined"
#endif

namespace implicit_feature_ridge_solve_kernel {

constexpr int kFeatures = IMPLICIT_SINDY_INTERNAL_FEATURES;
constexpr int kThreads = 32;

__device__ __forceinline__
float
f32_abs(
    float value
) {
    return value < 0.0f ? -value : value;
}

__device__ __forceinline__
float
f32_infinity(
    void
) {
    return __int_as_float(0x7f800000);
}

template <int N, int PaddedRhs>
using PosvOp = decltype(
    cusolverdx::Size<N, N, PaddedRhs>() +
    cusolverdx::Precision<float>() +
    cusolverdx::Type<cusolverdx::type::real>() +
    cusolverdx::Function<cusolverdx::function::posv>() +
    cusolverdx::FillMode<cusolverdx::lower>() +
    cusolverdx::Arrangement<cusolverdx::col_major>() +
    cusolverdx::BatchesPerBlock<1>() +
    cusolverdx::SM<IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM>() +
    cusolverdx::Block());

template <int N>
using PotrfOp = decltype(
    cusolverdx::Size<N, N, 1>() +
    cusolverdx::Precision<float>() +
    cusolverdx::Type<cusolverdx::type::real>() +
    cusolverdx::Function<cusolverdx::function::potrf>() +
    cusolverdx::FillMode<cusolverdx::lower>() +
    cusolverdx::Arrangement<cusolverdx::col_major>() +
    cusolverdx::BatchesPerBlock<1>() +
    cusolverdx::SM<IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM>() +
    cusolverdx::Block());

template <int N, int PaddedRhs>
using PotrsOp = decltype(
    cusolverdx::Size<N, N, PaddedRhs>() +
    cusolverdx::Precision<float>() +
    cusolverdx::Type<cusolverdx::type::real>() +
    cusolverdx::Function<cusolverdx::function::potrs>() +
    cusolverdx::FillMode<cusolverdx::lower>() +
    cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>() +
    cusolverdx::BatchesPerBlock<1>() +
    cusolverdx::SM<IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM>() +
    cusolverdx::Block());

template <int N, int PaddedRhs, bool UseDirectPosv>
__device__
void
execute_posv(
    float* a,
    int lda,
    float* b,
    int ldb,
    int* info
) {
    if constexpr (UseDirectPosv) {
        PosvOp<N, PaddedRhs>().execute(a, lda, b, ldb, info);
    } else {
        PotrfOp<N>().execute(a, lda, info);
        __syncthreads();
        if (*info == 0) {
            if constexpr (PaddedRhs <= 16) {
                PotrsOp<N, PaddedRhs>().execute(a, lda, b, ldb);
            } else if constexpr (PaddedRhs == 32) {
                PotrsOp<N, 16>().execute(a, lda, b, ldb);
                __syncthreads();
                PotrsOp<N, 16>().execute(a, lda, b + 16 * ldb, ldb);
            }
        }
    }
}

template <int N>
__device__ __forceinline__
void
load_normalized_matrix(
    float* a_preserved,
    int lda,
    int64_t row_count,
    float scale_epsilon,
    const float* gram,
    int64_t gram_col_stride,
    int64_t gram_setting_stride,
    int64_t setting,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    float* x_mean,
    int64_t x_mean_setting_stride,
    float* x_scale,
    int64_t x_scale_setting_stride,
    float* x_mean_shared,
    float* x_scale_shared,
    int64_t thread_index
) {
    const float inv_count = 1.0f / static_cast<float>(row_count);
    const float scale_floor = scale_epsilon * scale_epsilon;
    const float* gram_base =
        gram + setting * gram_setting_stride;
    const float* x_sum_base =
        x_sum + setting * x_sum_setting_stride;
    float* x_mean_base =
        x_mean + setting * x_mean_setting_stride;
    float* x_scale_base =
        x_scale + setting * x_scale_setting_stride;

    for (int64_t row = thread_index; row < (int64_t)N; row += blockDim.x) {
        const float sum = x_sum_base[row];
        const float raw_diag = gram_base[row * gram_col_stride + row];
        float variance = (raw_diag - sum * sum * inv_count) * inv_count;
        if (variance < scale_floor) {
            variance = scale_floor;
        }

        x_mean_shared[row] = sum * inv_count;
        x_scale_shared[row] = sqrtf(variance);
        x_mean_base[row] = x_mean_shared[row];
        x_scale_base[row] = x_scale_shared[row];
    }
    __syncthreads();

    for (int64_t linear = thread_index; linear < (int64_t)N * (int64_t)N; linear += blockDim.x) {
        const int64_t col = linear / N;
        const int64_t row = linear - col * N;

        if (row > col) {
            const float raw_gram = gram_base[col * gram_col_stride + row];
            const float centered_gram = raw_gram - x_sum_base[row] * x_sum_base[col] * inv_count;
            a_preserved[col * lda + row] =
                centered_gram / (x_scale_shared[row] * x_scale_shared[col]) * inv_count;
        } else if (row == col) {
            a_preserved[col * lda + row] = 1.0f;
        } else {
            a_preserved[col * lda + row] = 0.0f;
        }
    }
}

template <int N>
__device__ __forceinline__
void
copy_matrix_and_apply_alpha(
    float* a,
    int lda,
    const float* a_preserved,
    float alpha,
    int64_t thread_index
) {
    for (int64_t linear = thread_index; linear < (int64_t)N * (int64_t)N; linear += blockDim.x) {
        a[linear] = a_preserved[linear];
    }
    for (int64_t linear = thread_index; linear < N; linear += blockDim.x) {
        a[linear * lda + linear] = 1.0f + alpha;
    }
}

template <int N>
__device__ __forceinline__
void
copy_matrix_and_apply_alpha_mask(
    float* a,
    int lda,
    const float* a_preserved,
    float alpha,
    uint32_t active_mask,
    int64_t thread_index
) {
    for (int64_t linear = thread_index; linear < (int64_t)N * (int64_t)N; linear += blockDim.x) {
        const int64_t col = linear / N;
        const int64_t row = linear - col * N;
        const uint32_t row_active = (active_mask >> row) & 1u;
        const uint32_t col_active = (active_mask >> col) & 1u;

        if (row == col) {
            a[linear] = row_active ? 1.0f + alpha : 1.0f;
        } else if (row_active && col_active) {
            a[linear] = a_preserved[linear];
        } else {
            a[linear] = 0.0f;
        }
    }
}

template <int N, int PaddedRhs>
__device__ __forceinline__
void
load_rhs(
    float* b,
    int ldb,
    int64_t row_count,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const float* x_scale_shared,
    int64_t setting,
    int64_t num_rhs,
    int64_t thread_index
) {
    const float inv_count = 1.0f / static_cast<float>(row_count);
    const float* x_sum_base = x_sum + setting * x_sum_setting_stride;
    const float* xty_base = xty + setting * xty_setting_stride;
    const float* y_sum_base = y_sum + setting * y_sum_setting_stride;
    float* y_mean_base = y_mean + setting * y_mean_setting_stride;

    for (int64_t linear = thread_index; linear < (int64_t)N * (int64_t)PaddedRhs; linear += blockDim.x) {
        const int64_t col = linear / N;
        const int64_t row = linear - col * N;

        if (col < num_rhs) {
            const float mean = y_sum_base[col * y_sum_rhs_stride] * inv_count;
            const float centered_rhs = xty_base[col * xty_rhs_stride + row] - x_sum_base[row] * mean;
            b[col * ldb + row] = centered_rhs / x_scale_shared[row] * inv_count;
            if (row == 0) {
                y_mean_base[col * y_mean_rhs_stride] = mean;
            }
        } else {
            b[col * ldb + row] = 0.0f;
        }
    }
}

template <int N>
__device__ __forceinline__
void
load_rhs_masked(
    float* b,
    int ldb,
    int64_t row_count,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const float* x_scale_shared,
    int64_t setting,
    int64_t rhs,
    uint32_t active_mask,
    int64_t thread_index
) {
    const float inv_count = 1.0f / static_cast<float>(row_count);
    const float* x_sum_base = x_sum + setting * x_sum_setting_stride;
    const float* xty_base = xty + setting * xty_setting_stride;
    const float* y_sum_base = y_sum + setting * y_sum_setting_stride;
    float* y_mean_base = y_mean + setting * y_mean_setting_stride;
    const float mean = y_sum_base[rhs * y_sum_rhs_stride] * inv_count;

    for (int64_t row = thread_index; row < (int64_t)N; row += blockDim.x) {
        const float centered_rhs = xty_base[rhs * xty_rhs_stride + row] - x_sum_base[row] * mean;
        b[row] = ((active_mask >> row) & 1u) != 0u
            ? centered_rhs / x_scale_shared[row] * inv_count
            : 0.0f;
        if (row == 0) {
            y_mean_base[rhs * y_mean_rhs_stride] = mean;
        }
    }
}

template <int N>
__device__ __forceinline__
uint32_t
stlsq_threshold_mask(
    const float* b,
    uint32_t active_mask,
    float threshold,
    int64_t thread_index
) {
    const uint32_t active = (active_mask >> thread_index) & 1u;
    const int keep = active && f32_abs(b[thread_index]) >= threshold;
    return __ballot_sync(0xffffffffu, keep);
}

template <int N>
__device__ __forceinline__
void
zero_rhs(
    float* b,
    int64_t thread_index
) {
    for (int64_t row = thread_index; row < (int64_t)N; row += blockDim.x) {
        b[row] = 0.0f;
    }
}

template <int N, int PaddedRhs, bool UseDirectPosv, bool UseStlsq>
__device__
void
solve_sweep_body(
    int64_t row_count,
    float scale_epsilon,
    int64_t num_rhs,
    int64_t num_sweeps,
    const float* gram,
    int64_t gram_col_stride,
    int64_t gram_setting_stride,
    const float* x_sum,
    int64_t x_sum_setting_stride,
    const float* xty,
    int64_t xty_rhs_stride,
    int64_t xty_setting_stride,
    const float* y_sum,
    int64_t y_sum_rhs_stride,
    int64_t y_sum_setting_stride,
    const float* alphas,
    int64_t alphas_stride,
    const float* thresholds,
    int64_t thresholds_stride,
    float* x_mean,
    int64_t x_mean_setting_stride,
    float* x_scale,
    int64_t x_scale_setting_stride,
    float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride,
    uint32_t* active_masks,
    int64_t active_mask_rhs_stride,
    int64_t active_mask_setting_stride,
    int64_t active_mask_sweep_stride,
    int32_t* active_counts,
    int64_t active_count_rhs_stride,
    int64_t active_count_setting_stride,
    int64_t active_count_sweep_stride,
    int32_t* iteration_counts,
    int64_t iteration_count_rhs_stride,
    int64_t iteration_count_setting_stride,
    int64_t iteration_count_sweep_stride
) {
    constexpr int lda = N;
    constexpr int ldb = N;

    extern __shared__ __align__(sizeof(float)) unsigned char shared_raw[];

    float* a_preserved = reinterpret_cast<float*>(shared_raw);
    float* a = a_preserved + lda * N;
    float* b = a + lda * N;
    float* x_mean_shared = b + ldb * PaddedRhs;
    float* x_scale_shared = x_mean_shared + N;
    __shared__ int info;
    __shared__ uint32_t active_mask_shared;
    __shared__ uint32_t next_active_mask_shared;
    __shared__ int iteration_count_shared;
    __shared__ int solve_info_shared;
    __shared__ int stop_stlsq_shared;

    const int64_t setting = static_cast<int64_t>(blockIdx.x);
    const int64_t thread_index = static_cast<int64_t>(threadIdx.x);

    load_normalized_matrix<N>(
        a_preserved,
        lda,
        row_count,
        scale_epsilon,
        gram,
        gram_col_stride,
        gram_setting_stride,
        setting,
        x_sum,
        x_sum_setting_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        x_mean_shared,
        x_scale_shared,
        thread_index
    );
    __syncthreads();

    for (int64_t sweep = 0; sweep < num_sweeps; ++sweep) {
        const float alpha = alphas[sweep * alphas_stride];

        if constexpr (UseStlsq) {
            const float threshold = thresholds[sweep * thresholds_stride];

            if (thread_index == 0) {
                solve_info_shared = 0;
            }
            __syncthreads();

            for (int64_t rhs = 0; rhs < num_rhs; ++rhs) {
                if (thread_index == 0) {
                    active_mask_shared = 0xffffffffu;
                    iteration_count_shared = 0;
                }
                __syncthreads();

                for (int64_t iteration = 0; iteration < IMPLICIT_SINDY_INTERNAL_STLSQ_MAX_ITERATIONS; ++iteration) {
                    copy_matrix_and_apply_alpha_mask<N>(
                        a,
                        lda,
                        a_preserved,
                        alpha,
                        active_mask_shared,
                        thread_index
                    );
                    load_rhs_masked<N>(
                        b,
                        ldb,
                        row_count,
                        x_sum,
                        x_sum_setting_stride,
                        xty,
                        xty_rhs_stride,
                        xty_setting_stride,
                        y_sum,
                        y_sum_rhs_stride,
                        y_sum_setting_stride,
                        y_mean,
                        y_mean_rhs_stride,
                        y_mean_setting_stride,
                        x_scale_shared,
                        setting,
                        rhs,
                        active_mask_shared,
                        thread_index
                    );

                    if (thread_index == 0) {
                        info = 0;
                    }
                    __syncthreads();

                    execute_posv<N, 1, true>(
                        a,
                        lda,
                        b,
                        ldb,
                        &info
                    );
                    __syncthreads();

                    if (info != 0) {
                        if (thread_index == 0) {
                            solve_info_shared = info;
                            active_mask_shared = 0u;
                            iteration_count_shared = static_cast<int>(iteration + 1);
                        }
                        __syncthreads();
                        break;
                    }

                    {
                        const uint32_t next_active_mask = stlsq_threshold_mask<N>(
                            b,
                            active_mask_shared,
                            threshold,
                            thread_index
                        );

                        if (thread_index == 0) {
                            next_active_mask_shared = next_active_mask;
                        }
                    }
                    __syncthreads();

                    if (thread_index == 0) {
                        iteration_count_shared = static_cast<int>(iteration + 1);
                        if (next_active_mask_shared == active_mask_shared ||
                            next_active_mask_shared == 0u) {
                            active_mask_shared = next_active_mask_shared;
                            stop_stlsq_shared = 1;
                        } else {
                            active_mask_shared = next_active_mask_shared;
                            stop_stlsq_shared = 0;
                        }
                    }
                    __syncthreads();

                    if (stop_stlsq_shared) {
                        break;
                    }
                }

                if (solve_info_shared == 0 && active_mask_shared != 0u) {
                    copy_matrix_and_apply_alpha_mask<N>(
                        a,
                        lda,
                        a_preserved,
                        alpha,
                        active_mask_shared,
                        thread_index
                    );
                    load_rhs_masked<N>(
                        b,
                        ldb,
                        row_count,
                        x_sum,
                        x_sum_setting_stride,
                        xty,
                        xty_rhs_stride,
                        xty_setting_stride,
                        y_sum,
                        y_sum_rhs_stride,
                        y_sum_setting_stride,
                        y_mean,
                        y_mean_rhs_stride,
                        y_mean_setting_stride,
                        x_scale_shared,
                        setting,
                        rhs,
                        active_mask_shared,
                        thread_index
                    );

                    if (thread_index == 0) {
                        info = 0;
                    }
                    __syncthreads();

                    execute_posv<N, 1, true>(
                        a,
                        lda,
                        b,
                        ldb,
                        &info
                    );
                    __syncthreads();

                    if (info != 0 && thread_index == 0) {
                        solve_info_shared = info;
                    }
                } else {
                    zero_rhs<N>(b, thread_index);
                    __syncthreads();
                }
                __syncthreads();

                for (int64_t row = thread_index; row < (int64_t)N; row += blockDim.x) {
                    beta_standardized[
                        sweep * beta_sweep_stride +
                        setting * beta_setting_stride +
                        rhs * beta_rhs_stride +
                        row
                    ] = (solve_info_shared == 0) ? b[row] : 0.0f;
                }

                if (thread_index == 0) {
                    active_masks[
                        sweep * active_mask_sweep_stride +
                        setting * active_mask_setting_stride +
                        rhs * active_mask_rhs_stride
                    ] = active_mask_shared;
                    active_counts[
                        sweep * active_count_sweep_stride +
                        setting * active_count_setting_stride +
                        rhs * active_count_rhs_stride
                    ] = static_cast<int32_t>(__popc(active_mask_shared));
                    iteration_counts[
                        sweep * iteration_count_sweep_stride +
                        setting * iteration_count_setting_stride +
                        rhs * iteration_count_rhs_stride
                    ] = static_cast<int32_t>(iteration_count_shared);
                }
                __syncthreads();
            }

            if (thread_index == 0) {
                solve_info[
                    sweep * solve_info_sweep_stride +
                    setting * solve_info_setting_stride
                ] = solve_info_shared;
            }
        } else {
            copy_matrix_and_apply_alpha<N>(
                a,
                lda,
                a_preserved,
                alpha,
                thread_index
            );
            load_rhs<N, PaddedRhs>(
                b,
                ldb,
                row_count,
                x_sum,
                x_sum_setting_stride,
                xty,
                xty_rhs_stride,
                xty_setting_stride,
                y_sum,
                y_sum_rhs_stride,
                y_sum_setting_stride,
                y_mean,
                y_mean_rhs_stride,
                y_mean_setting_stride,
                x_scale_shared,
                setting,
                num_rhs,
                thread_index
            );

            if (thread_index == 0) {
                info = 0;
            }
            __syncthreads();

            execute_posv<N, PaddedRhs, UseDirectPosv>(
                a,
                lda,
                b,
                ldb,
                &info
            );
            __syncthreads();

            for (int64_t linear = thread_index; linear < (int64_t)N * num_rhs; linear += blockDim.x) {
                const int64_t rhs = linear / N;
                const int64_t row = linear - rhs * N;

                beta_standardized[
                    sweep * beta_sweep_stride +
                    setting * beta_setting_stride +
                    rhs * beta_rhs_stride +
                    row
                ] = (info == 0) ? b[rhs * ldb + row] : 0.0f;
            }

            if (thread_index == 0) {
                solve_info[
                    sweep * solve_info_sweep_stride +
                    setting * solve_info_setting_stride
                ] = info;
            }

            if (sweep + 1 < num_sweeps) {
                __syncthreads();
            }
        }
    }
}

template <int N>
__device__
void
score_mse_body(
    int64_t row_count,
    int64_t num_rhs,
    const float* eval_gram,
    int64_t eval_gram_col_stride,
    int64_t eval_gram_setting_stride,
    const float* eval_x_sum,
    int64_t eval_x_sum_setting_stride,
    const float* eval_xty,
    int64_t eval_xty_rhs_stride,
    int64_t eval_xty_setting_stride,
    const float* eval_y_sum,
    int64_t eval_y_sum_rhs_stride,
    int64_t eval_y_sum_setting_stride,
    const float* eval_yy,
    int64_t eval_yy_rhs_stride,
    int64_t eval_yy_setting_stride,
    const float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    const float* x_mean,
    int64_t x_mean_setting_stride,
    const float* x_scale,
    int64_t x_scale_setting_stride,
    const float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride,
    float* mse,
    int64_t mse_rhs_stride,
    int64_t mse_setting_stride,
    int64_t mse_sweep_stride
) {
    const int64_t setting = static_cast<int64_t>(blockIdx.x);
    const int64_t sweep = static_cast<int64_t>(blockIdx.z);
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    const int64_t eval_rows = row_count;
    const float inv_eval_rows = 1.0f / static_cast<float>(eval_rows);
    const float* eval_gram_base =
        eval_gram +
        setting * eval_gram_setting_stride;
    const float* eval_x_sum_base =
        eval_x_sum +
        setting * eval_x_sum_setting_stride;
    const float* x_mean_base =
        x_mean +
        setting * x_mean_setting_stride;
    const float* x_scale_base =
        x_scale +
        setting * x_scale_setting_stride;

    if (eval_rows <= 0) {
        return;
    }

    if (solve_info[
            sweep * solve_info_sweep_stride +
            setting * solve_info_setting_stride
        ] != 0) {
        for (int64_t rhs = lane; rhs < num_rhs; rhs += warpSize) {
            mse[
                sweep * mse_sweep_stride +
                setting * mse_setting_stride +
                rhs * mse_rhs_stride
            ] = f32_infinity();
        }
        return;
    }

    for (int64_t rhs = lane; rhs < num_rhs; rhs += warpSize) {
        float beta_raw_local[N];
        float intercept = y_mean[
            rhs * y_mean_rhs_stride +
            setting * y_mean_setting_stride
        ];
        float xty_proj = 0.0f;
        float xsum_proj = 0.0f;
        float quad = 0.0f;
        const float y_sum_eval = eval_y_sum[
            rhs * eval_y_sum_rhs_stride +
            setting * eval_y_sum_setting_stride
        ];
        const float yy_eval = eval_yy[
            rhs * eval_yy_rhs_stride +
            setting * eval_yy_setting_stride
        ];

        #pragma unroll
        for (int p = 0; p < N; ++p) {
            const float beta_standardized_value = beta_standardized[
                sweep * beta_sweep_stride +
                setting * beta_setting_stride +
                rhs * beta_rhs_stride +
                p
            ];
            const float beta_raw = beta_standardized_value / x_scale_base[p];

            beta_raw_local[p] = beta_raw;
            intercept -= x_mean_base[p] * beta_raw;
            xty_proj += beta_raw * eval_xty[
                p +
                rhs * eval_xty_rhs_stride +
                setting * eval_xty_setting_stride
            ];
            xsum_proj += beta_raw * eval_x_sum_base[p];
        }

        #pragma unroll
        for (int p = 0; p < N; ++p) {
            float row_dot = 0.0f;

            #pragma unroll
            for (int q = 0; q < N; ++q) {
                const int row = p < q ? p : q;
                const int col = p < q ? q : p;
                const float raw_gram = eval_gram_base[
                    row +
                    static_cast<int64_t>(col) * eval_gram_col_stride
                ];

                row_dot += raw_gram * beta_raw_local[q];
            }

            quad += beta_raw_local[p] * row_dot;
        }

        {
            const float sse =
                yy_eval -
                2.0f * xty_proj +
                quad -
                2.0f * intercept * y_sum_eval +
                2.0f * intercept * xsum_proj +
                static_cast<float>(eval_rows) * intercept * intercept;
            const float mse_value = fmaxf(sse * inv_eval_rows, 0.0f);

            mse[
                sweep * mse_sweep_stride +
                setting * mse_setting_stride +
                rhs * mse_rhs_stride
            ] = mse_value;
        }
    }
}

}  // namespace implicit_feature_ridge_solve_kernel

#define IMPLICIT_FEATURE_RIDGE_SOLVE_KERNEL_ARGS \
    int64_t row_count, \
    float scale_epsilon, \
    int64_t num_rhs, \
    int64_t num_sweeps, \
    const float* gram, \
    int64_t gram_col_stride, \
    int64_t gram_setting_stride, \
    const float* x_sum, \
    int64_t x_sum_setting_stride, \
    const float* xty, \
    int64_t xty_rhs_stride, \
    int64_t xty_setting_stride, \
    const float* y_sum, \
    int64_t y_sum_rhs_stride, \
    int64_t y_sum_setting_stride, \
    const float* alphas, \
    int64_t alphas_stride, \
    const float* thresholds, \
    int64_t thresholds_stride, \
    float* x_mean, \
    int64_t x_mean_setting_stride, \
    float* x_scale, \
    int64_t x_scale_setting_stride, \
    float* y_mean, \
    int64_t y_mean_rhs_stride, \
    int64_t y_mean_setting_stride, \
    float* beta_standardized, \
    int64_t beta_rhs_stride, \
    int64_t beta_setting_stride, \
    int64_t beta_sweep_stride, \
    int32_t* solve_info, \
    int64_t solve_info_sweep_stride, \
    int64_t solve_info_setting_stride, \
    uint32_t* active_masks, \
    int64_t active_mask_rhs_stride, \
    int64_t active_mask_setting_stride, \
    int64_t active_mask_sweep_stride, \
    int32_t* active_counts, \
    int64_t active_count_rhs_stride, \
    int64_t active_count_setting_stride, \
    int64_t active_count_sweep_stride, \
    int32_t* iteration_counts, \
    int64_t iteration_count_rhs_stride, \
    int64_t iteration_count_setting_stride, \
    int64_t iteration_count_sweep_stride

#define IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(NAME, PADDED_RHS, USE_DIRECT_POSV, COPY_WHOLE_MATRIX, USE_STLSQ) \
    extern "C" __global__ __launch_bounds__(implicit_feature_ridge_solve_kernel::kThreads) \
    void \
    NAME( \
        IMPLICIT_FEATURE_RIDGE_SOLVE_KERNEL_ARGS \
    ) { \
        implicit_feature_ridge_solve_kernel::solve_sweep_body< \
            implicit_feature_ridge_solve_kernel::kFeatures, \
            PADDED_RHS, \
            USE_DIRECT_POSV, \
            USE_STLSQ \
        >( \
            row_count, \
            scale_epsilon, \
            num_rhs, \
            num_sweeps, \
            gram, \
            gram_col_stride, \
            gram_setting_stride, \
            x_sum, \
            x_sum_setting_stride, \
            xty, \
            xty_rhs_stride, \
            xty_setting_stride, \
            y_sum, \
            y_sum_rhs_stride, \
            y_sum_setting_stride, \
            alphas, \
            alphas_stride, \
            thresholds, \
            thresholds_stride, \
            x_mean, \
            x_mean_setting_stride, \
            x_scale, \
            x_scale_setting_stride, \
            y_mean, \
            y_mean_rhs_stride, \
            y_mean_setting_stride, \
            beta_standardized, \
            beta_rhs_stride, \
            beta_setting_stride, \
            beta_sweep_stride, \
            solve_info, \
            solve_info_sweep_stride, \
            solve_info_setting_stride, \
            active_masks, \
            active_mask_rhs_stride, \
            active_mask_setting_stride, \
            active_mask_sweep_stride, \
            active_counts, \
            active_count_rhs_stride, \
            active_count_setting_stride, \
            active_count_sweep_stride, \
            iteration_counts, \
            iteration_count_rhs_stride, \
            iteration_count_setting_stride, \
            iteration_count_sweep_stride \
        ); \
    }

IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs1_posv_lower_compat_fast, 1, 1, 1, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs1_posv, 1, 1, 0, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs2_posv, 2, 1, 0, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs4_posv, 4, 1, 0, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs8_posv, 8, 1, 0, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs16_posv, 16, 1, 0, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_rhs32_potrf_potrs, 32, 0, 0, false)
IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL(implicit_feature_ridge_solve_32_stlsq_posv, 1, 1, 0, true)

#undef IMPLICIT_FEATURE_RIDGE_SOLVE_DEFINE_KERNEL
#undef IMPLICIT_FEATURE_RIDGE_SOLVE_KERNEL_ARGS

extern "C" __global__ __launch_bounds__(implicit_feature_ridge_solve_kernel::kThreads)
void
implicit_feature_ridge_score_mse_kernel(
    int64_t row_count,
    int64_t num_rhs,
    const float* eval_gram,
    int64_t eval_gram_col_stride,
    int64_t eval_gram_setting_stride,
    const float* eval_x_sum,
    int64_t eval_x_sum_setting_stride,
    const float* eval_xty,
    int64_t eval_xty_rhs_stride,
    int64_t eval_xty_setting_stride,
    const float* eval_y_sum,
    int64_t eval_y_sum_rhs_stride,
    int64_t eval_y_sum_setting_stride,
    const float* eval_yy,
    int64_t eval_yy_rhs_stride,
    int64_t eval_yy_setting_stride,
    const float* beta_standardized,
    int64_t beta_rhs_stride,
    int64_t beta_setting_stride,
    int64_t beta_sweep_stride,
    const float* x_mean,
    int64_t x_mean_setting_stride,
    const float* x_scale,
    int64_t x_scale_setting_stride,
    const float* y_mean,
    int64_t y_mean_rhs_stride,
    int64_t y_mean_setting_stride,
    const int32_t* solve_info,
    int64_t solve_info_sweep_stride,
    int64_t solve_info_setting_stride,
    float* mse,
    int64_t mse_rhs_stride,
    int64_t mse_setting_stride,
    int64_t mse_sweep_stride
) {
    implicit_feature_ridge_solve_kernel::score_mse_body<
        implicit_feature_ridge_solve_kernel::kFeatures
    >(
        row_count,
        num_rhs,
        eval_gram,
        eval_gram_col_stride,
        eval_gram_setting_stride,
        eval_x_sum,
        eval_x_sum_setting_stride,
        eval_xty,
        eval_xty_rhs_stride,
        eval_xty_setting_stride,
        eval_y_sum,
        eval_y_sum_rhs_stride,
        eval_y_sum_setting_stride,
        eval_yy,
        eval_yy_rhs_stride,
        eval_yy_setting_stride,
        beta_standardized,
        beta_rhs_stride,
        beta_setting_stride,
        beta_sweep_stride,
        x_mean,
        x_mean_setting_stride,
        x_scale,
        x_scale_setting_stride,
        y_mean,
        y_mean_rhs_stride,
        y_mean_setting_stride,
        solve_info,
        solve_info_sweep_stride,
        solve_info_setting_stride,
        mse,
        mse_rhs_stride,
        mse_setting_stride,
        mse_sweep_stride
    );
}
