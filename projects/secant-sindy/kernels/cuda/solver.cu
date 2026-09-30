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
#include <cusolverdx.hpp>

#ifndef SECANT_SINDY_FEATURE_CAPACITY
#error "SECANT_SINDY_FEATURE_CAPACITY must be defined"
#endif
#ifndef SECANT_SINDY_PADDED_TARGETS
#error "SECANT_SINDY_PADDED_TARGETS must be defined"
#endif
#ifndef SECANT_SINDY_STLSQ_MAX_ITERATIONS
#error "SECANT_SINDY_STLSQ_MAX_ITERATIONS must be defined"
#endif
#ifndef SECANT_SINDY_CUSOLVERDX_SM
#error "SECANT_SINDY_CUSOLVERDX_SM must be defined"
#endif

namespace secant_sindy_cuda_solver {

constexpr int kFeatures = SECANT_SINDY_FEATURE_CAPACITY;
constexpr int kPaddedTargets = SECANT_SINDY_PADDED_TARGETS;

template <int N, int PaddedRhs>
using PosvOp = decltype(
    cusolverdx::Size<N, N, PaddedRhs>() +
    cusolverdx::Precision<float>() +
    cusolverdx::Type<cusolverdx::type::real>() +
    cusolverdx::Function<cusolverdx::function::posv>() +
    cusolverdx::FillMode<cusolverdx::lower>() +
    cusolverdx::Arrangement<cusolverdx::col_major>() +
    cusolverdx::BatchesPerBlock<1>() +
    cusolverdx::SM<SECANT_SINDY_CUSOLVERDX_SM>() +
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
    cusolverdx::SM<SECANT_SINDY_CUSOLVERDX_SM>() +
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
    cusolverdx::SM<SECANT_SINDY_CUSOLVERDX_SM>() +
    cusolverdx::Block());

template <int N, int PaddedRhs>
__device__ void solve(float* matrix, float* rhs, int* info) {
    if constexpr (PaddedRhs <= 16) {
        PosvOp<N, PaddedRhs>().execute(matrix, N, rhs, N, info);
    } else {
        PotrfOp<N>().execute(matrix, N, info);
        __syncthreads();
        if (*info == 0) {
            PotrsOp<N, 16>().execute(matrix, N, rhs, N);
            __syncthreads();
            PotrsOp<N, 16>().execute(matrix, N, rhs + 16 * N, N);
        }
    }
}

__device__ __forceinline__ unsigned int active_mask_get(size_t active_features) {
    return active_features == 32u ? 0xffffffffu : ((1u << active_features) - 1u);
}

__device__ void normalized_matrix_load(const float* statistics, size_t active_features, size_t num_rows,
                                       float scale_epsilon, float* normalized, float* means, float* scales) {
    const float* gram = statistics + kFeatures;
    const float inv_rows = 1.0f / (float)num_rows;
    const float variance_floor = scale_epsilon * scale_epsilon;

    for (size_t feature = threadIdx.x; feature < (size_t)kFeatures; feature += blockDim.x) {
        if (feature < active_features) {
            const float sum = statistics[feature];
            float variance = (gram[feature * kFeatures + feature] - sum * sum * inv_rows) * inv_rows;
            if (variance < variance_floor) {
                variance = variance_floor;
            }
            means[feature] = sum * inv_rows;
            scales[feature] = sqrtf(variance);
        } else {
            means[feature] = 0.0f;
            scales[feature] = 1.0f;
        }
    }
    __syncthreads();

    for (size_t linear = threadIdx.x; linear < (size_t)kFeatures * kFeatures; linear += blockDim.x) {
        const size_t col = linear / kFeatures;
        const size_t row = linear - col * kFeatures;
        if (row == col) {
            normalized[col * kFeatures + row] = 1.0f;
        } else if (row > col && row < active_features && col < active_features) {
            const float centered = gram[row * kFeatures + col] -
                statistics[row] * statistics[col] * inv_rows;
            normalized[col * kFeatures + row] = centered * inv_rows / (scales[row] * scales[col]);
        } else {
            normalized[col * kFeatures + row] = 0.0f;
        }
    }
}

__device__ void system_load(const float* normalized, const float* statistics, const float* target_stats,
                            size_t target_stats_leading_dimension, const float* cross, size_t target,
                            size_t num_targets, size_t active_features, size_t num_rows, float alpha,
                            unsigned int active_mask, float* matrix, float* rhs) {
    const float target_sum = target_stats[target * target_stats_leading_dimension];
    const float inv_rows = 1.0f / (float)num_rows;

    for (size_t linear = threadIdx.x; linear < (size_t)kFeatures * kFeatures; linear += blockDim.x) {
        const size_t col = linear / kFeatures;
        const size_t row = linear - col * kFeatures;
        const bool row_active = row < active_features && ((active_mask >> row) & 1u) != 0u;
        const bool col_active = col < active_features && ((active_mask >> col) & 1u) != 0u;
        if (row == col) {
            matrix[col * kFeatures + row] = row_active ? 1.0f + alpha : 1.0f;
        } else {
            matrix[col * kFeatures + row] = row_active && col_active
                ? normalized[col * kFeatures + row]
                : 0.0f;
        }
    }
    for (size_t feature = threadIdx.x; feature < (size_t)kFeatures; feature += blockDim.x) {
        const bool active = feature < active_features && ((active_mask >> feature) & 1u) != 0u;
        const float centered = cross[feature * num_targets + target] - statistics[feature] * target_sum * inv_rows;
        rhs[feature] = active ? centered * inv_rows : 0.0f;
    }
}

__device__ void rhs_scale(float* rhs, const float* scales, size_t num_rhs) {
    for (size_t linear = threadIdx.x; linear < (size_t)kFeatures * num_rhs; linear += blockDim.x) {
        const size_t feature = linear % kFeatures;
        rhs[linear] /= scales[feature];
    }
}

__device__ void raw_solution_score(const float* statistics, const float* target_stats,
                                   size_t target_stats_leading_dimension, size_t num_rows, size_t num_targets,
                                   size_t target, size_t active_features, const float* means, const float* scales,
                                   const float* standardized, int info, float* coefficients, float* intercepts,
                                   float* sse, size_t output_index) {
    const float* gram = statistics + kFeatures;
    const float* cross = gram + kFeatures * kFeatures;
    const float* target_moments = target_stats + target * target_stats_leading_dimension;
    float intercept = target_moments[0] / (float)num_rows;
    float linear_sum = 0.0f;
    float cross_sum = 0.0f;
    float quadratic_sum = 0.0f;
    float raw[kFeatures];

    for (size_t feature = 0u; feature < (size_t)kFeatures; ++feature) {
        raw[feature] = info == 0 && feature < active_features ? standardized[feature] / scales[feature] : 0.0f;
        coefficients[output_index * kFeatures + feature] = raw[feature];
        intercept -= raw[feature] * means[feature];
        linear_sum += raw[feature] * statistics[feature];
        cross_sum += raw[feature] * cross[feature * num_targets + target];
    }
    for (size_t row = 0u; row < active_features; ++row) {
        for (size_t col = 0u; col < active_features; ++col) {
            quadratic_sum += raw[row] * gram[row * kFeatures + col] * raw[col];
        }
    }
    intercepts[output_index] = intercept;
    {
        const float target_sum = target_moments[0];
        float value = target_moments[1] - 2.0f * cross_sum + quadratic_sum -
            2.0f * intercept * target_sum + 2.0f * intercept * linear_sum +
            (float)num_rows * intercept * intercept;
        if (value < 0.0f) {
            value = 0.0f;
        }
        sse[output_index] = value;
    }
}

} /* namespace secant_sindy_cuda_solver */

extern "C" __global__
void secant_sindy_cuda_ridge_solve_f32(
    const float* __restrict__ statistics,
    size_t statistics_leading_dimension,
    const float* __restrict__ target_stats,
    size_t target_stats_leading_dimension,
    size_t num_rows,
    size_t num_asts,
    size_t num_targets,
    size_t num_sweeps,
    const float* __restrict__ alphas,
    float scale_epsilon,
    float* __restrict__ coefficients,
    float* __restrict__ intercepts,
    float* __restrict__ sse,
    int* __restrict__ solve_info
) {
    using namespace secant_sindy_cuda_solver;
    __shared__ float normalized[kFeatures * kFeatures];
    __shared__ float matrix[kFeatures * kFeatures];
    __shared__ float rhs[kFeatures * kPaddedTargets];
    __shared__ float means[kFeatures];
    __shared__ float scales[kFeatures];
    __shared__ int info;
    const size_t cohort = blockIdx.x;
    const size_t num_cohorts = (num_asts + kFeatures - 1u) / kFeatures;
    const size_t feature_begin = cohort * kFeatures;
    const size_t active_features = num_asts - feature_begin < (size_t)kFeatures
        ? num_asts - feature_begin
        : (size_t)kFeatures;
    const float* cohort_stats = statistics + cohort * statistics_leading_dimension;
    const float* cross = cohort_stats + kFeatures + kFeatures * kFeatures;

    if (blockDim.x != 32u || cohort >= num_cohorts || num_targets == 0u || num_sweeps == 0u ||
        num_targets > (size_t)kPaddedTargets || target_stats_leading_dimension < 2u) {
        return;
    }
    normalized_matrix_load(cohort_stats, active_features, num_rows, scale_epsilon, normalized, means, scales);
    __syncthreads();
    for (size_t sweep = 0u; sweep < num_sweeps; ++sweep) {
        for (size_t target = 0u; target < (size_t)kPaddedTargets; ++target) {
            if (target < num_targets) {
                system_load(normalized, cohort_stats, target_stats, target_stats_leading_dimension, cross, target,
                            num_targets, active_features, num_rows, alphas[sweep], active_mask_get(active_features),
                            matrix, rhs + target * kFeatures);
            } else {
                for (size_t feature = threadIdx.x; feature < (size_t)kFeatures; feature += blockDim.x) {
                    rhs[target * kFeatures + feature] = 0.0f;
                }
            }
        }
        __syncthreads();
        rhs_scale(rhs, scales, kPaddedTargets);
        if (threadIdx.x == 0u) {
            info = 0;
        }
        __syncthreads();
        solve<kFeatures, kPaddedTargets>(matrix, rhs, &info);
        __syncthreads();
        if (threadIdx.x < num_targets) {
            const size_t output_index = (sweep * num_cohorts + cohort) * num_targets + threadIdx.x;
            raw_solution_score(cohort_stats, target_stats, target_stats_leading_dimension, num_rows, num_targets,
                               threadIdx.x, active_features, means, scales, rhs + threadIdx.x * kFeatures, info,
                               coefficients, intercepts, sse, output_index);
            solve_info[output_index] = info;
        }
        __syncthreads();
    }
}

extern "C" __global__
void secant_sindy_cuda_stlsq_solve_f32(
    const float* __restrict__ statistics,
    size_t statistics_leading_dimension,
    const float* __restrict__ target_stats,
    size_t target_stats_leading_dimension,
    size_t num_rows,
    size_t num_asts,
    size_t num_targets,
    size_t num_sweeps,
    const float* __restrict__ alphas,
    const float* __restrict__ thresholds,
    size_t max_iterations,
    float scale_epsilon,
    float* __restrict__ coefficients,
    float* __restrict__ intercepts,
    float* __restrict__ sse,
    int* __restrict__ solve_info,
    unsigned int* __restrict__ active_masks,
    int* __restrict__ active_counts,
    int* __restrict__ iteration_counts
) {
    using namespace secant_sindy_cuda_solver;
    __shared__ float normalized[kFeatures * kFeatures];
    __shared__ float matrix[kFeatures * kFeatures];
    __shared__ float rhs[kFeatures];
    __shared__ float means[kFeatures];
    __shared__ float scales[kFeatures];
    __shared__ int info;
    __shared__ unsigned int active_mask;
    __shared__ unsigned int next_mask;
    __shared__ int iterations;
    __shared__ int stop;
    const size_t cohort = blockIdx.x;
    const size_t num_cohorts = (num_asts + kFeatures - 1u) / kFeatures;
    const size_t feature_begin = cohort * kFeatures;
    const size_t active_features = num_asts - feature_begin < (size_t)kFeatures
        ? num_asts - feature_begin
        : (size_t)kFeatures;
    const float* cohort_stats = statistics + cohort * statistics_leading_dimension;
    const float* cross = cohort_stats + kFeatures + kFeatures * kFeatures;

    if (blockDim.x != 32u || cohort >= num_cohorts || num_targets == 0u || num_sweeps == 0u ||
        num_targets > (size_t)kPaddedTargets || target_stats_leading_dimension < 2u) {
        return;
    }
    normalized_matrix_load(cohort_stats, active_features, num_rows, scale_epsilon, normalized, means, scales);
    __syncthreads();

    const size_t iteration_limit = max_iterations < (size_t)SECANT_SINDY_STLSQ_MAX_ITERATIONS
        ? max_iterations
        : (size_t)SECANT_SINDY_STLSQ_MAX_ITERATIONS;
    for (size_t sweep = 0u; sweep < num_sweeps; ++sweep) {
        for (size_t target = 0u; target < num_targets; ++target) {
            if (threadIdx.x == 0u) {
                active_mask = active_mask_get(active_features);
                iterations = 0;
                info = 0;
            }
            __syncthreads();

            for (size_t iteration = 0u; iteration < iteration_limit; ++iteration) {
                system_load(normalized, cohort_stats, target_stats, target_stats_leading_dimension, cross, target,
                            num_targets, active_features, num_rows, alphas[sweep], active_mask, matrix, rhs);
                __syncthreads();
                rhs_scale(rhs, scales, 1u);
                if (threadIdx.x == 0u) {
                    info = 0;
                }
                __syncthreads();
                solve<kFeatures, 1>(matrix, rhs, &info);
                __syncthreads();
                if (info != 0) {
                    if (threadIdx.x == 0u) {
                        active_mask = 0u;
                        iterations = (int)(iteration + 1u);
                    }
                    __syncthreads();
                    break;
                }
                {
                    const bool keep = threadIdx.x < active_features &&
                        ((active_mask >> threadIdx.x) & 1u) != 0u &&
                        fabsf(rhs[threadIdx.x]) >= thresholds[sweep];
                    const unsigned int voted_mask = __ballot_sync(0xffffffffu, keep);
                    if (threadIdx.x == 0u) {
                        next_mask = voted_mask;
                        iterations = (int)(iteration + 1u);
                        stop = next_mask == active_mask || next_mask == 0u;
                        active_mask = next_mask;
                    }
                }
                __syncthreads();
                if (stop != 0) {
                    break;
                }
            }

            if (info == 0 && active_mask != 0u) {
                system_load(normalized, cohort_stats, target_stats, target_stats_leading_dimension, cross, target,
                            num_targets, active_features, num_rows, alphas[sweep], active_mask, matrix, rhs);
                __syncthreads();
                rhs_scale(rhs, scales, 1u);
                if (threadIdx.x == 0u) {
                    info = 0;
                }
                __syncthreads();
                solve<kFeatures, 1>(matrix, rhs, &info);
                __syncthreads();
            } else {
                for (size_t feature = threadIdx.x; feature < (size_t)kFeatures; feature += blockDim.x) {
                    rhs[feature] = 0.0f;
                }
                __syncthreads();
            }

            if (threadIdx.x == 0u) {
                const size_t output_index = (sweep * num_cohorts + cohort) * num_targets + target;
                raw_solution_score(cohort_stats, target_stats, target_stats_leading_dimension, num_rows, num_targets,
                                   target, active_features, means, scales, rhs, info, coefficients, intercepts, sse,
                                   output_index);
                solve_info[output_index] = info;
                active_masks[output_index] = active_mask;
                active_counts[output_index] = __popc(active_mask);
                iteration_counts[output_index] = iterations;
            }
            __syncthreads();
        }
    }
}
