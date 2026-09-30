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
#include "s_internal.h"

#include <math.h>
#include <string.h>

static SecantSindyResult secant_sindy_cpu_header_validate(uint32_t struct_size, uint32_t required_size,
                                                           uint32_t version, uint32_t expected_version,
                                                           uint32_t flags) {
    if (version != expected_version) {
        return SECANT_SINDY_ERROR_UNSUPPORTED_VERSION;
    }
    if (struct_size < required_size || flags != 0u) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    return SECANT_SINDY_SUCCESS;
}

static int secant_sindy_cholesky_solve(size_t capacity, const double* matrix, const double* rhs, double* solution,
                                       int32_t* info_ret) {
    double lower[SECANT_SINDY_MAX_FEATURE_CAPACITY * SECANT_SINDY_MAX_FEATURE_CAPACITY];
    double work[SECANT_SINDY_MAX_FEATURE_CAPACITY];
    size_t row;
    size_t col;

    memset(lower, 0, capacity * capacity * sizeof(*lower));
    for (row = 0u; row < capacity; ++row) {
        for (col = 0u; col <= row; ++col) {
            double value = matrix[row * capacity + col];
            size_t inner;

            for (inner = 0u; inner < col; ++inner) {
                value -= lower[row * capacity + inner] * lower[col * capacity + inner];
            }
            if (row == col) {
                if (!(value > 0.0) || !isfinite(value)) {
                    *info_ret = (int32_t)(row + 1u);
                    memset(solution, 0, capacity * sizeof(*solution));
                    return 0;
                }
                lower[row * capacity + col] = sqrt(value);
            } else {
                lower[row * capacity + col] = value / lower[col * capacity + col];
            }
        }
    }

    for (row = 0u; row < capacity; ++row) {
        double value = rhs[row];
        for (col = 0u; col < row; ++col) {
            value -= lower[row * capacity + col] * work[col];
        }
        work[row] = value / lower[row * capacity + row];
    }
    for (row = capacity; row-- > 0u;) {
        double value = work[row];
        for (col = row + 1u; col < capacity; ++col) {
            value -= lower[col * capacity + row] * solution[col];
        }
        solution[row] = value / lower[row * capacity + row];
    }
    *info_ret = 0;
    return 1;
}

static void secant_sindy_normalized_matrix_build(size_t capacity, size_t active_features, size_t num_rows,
                                                  float scale_epsilon, const float* statistics, double* means,
                                                  double* scales, double* normalized) {
    const double inv_rows = 1.0 / (double)num_rows;
    const double floor_variance = (double)scale_epsilon * (double)scale_epsilon;
    const float* gram = statistics + capacity;
    size_t row;
    size_t col;

    for (row = 0u; row < capacity; ++row) {
        if (row < active_features) {
            const double sum = statistics[row];
            double variance = ((double)gram[row * capacity + row] - sum * sum * inv_rows) * inv_rows;
            if (variance < floor_variance) {
                variance = floor_variance;
            }
            means[row] = sum * inv_rows;
            scales[row] = sqrt(variance);
        } else {
            means[row] = 0.0;
            scales[row] = 1.0;
        }
    }

    for (row = 0u; row < capacity; ++row) {
        for (col = 0u; col < capacity; ++col) {
            if (row == col) {
                normalized[row * capacity + col] = 1.0;
            } else if (row < active_features && col < active_features) {
                const double centered = (double)gram[row * capacity + col] -
                    (double)statistics[row] * (double)statistics[col] * inv_rows;
                normalized[row * capacity + col] = centered * inv_rows / (scales[row] * scales[col]);
            } else {
                normalized[row * capacity + col] = 0.0;
            }
        }
    }
}

static int32_t secant_sindy_popcount_u32(uint32_t value) {
    int32_t count = 0;

    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

static double secant_sindy_raw_solution_write(size_t capacity, size_t active_features, size_t num_rows,
                                              size_t num_targets, size_t target_index, const float* statistics,
                                              const float* target_stats,
                                              const double* means, const double* scales,
                                              const double* standardized, float* coefficients,
                                              float* intercept_ret) {
    const float* gram = statistics + capacity;
    const float* cross = gram + capacity * capacity;
    const double target_sum = target_stats[SECANT_SINDY_TARGET_STAT_SUM_F32];
    double intercept = target_sum / (double)num_rows;
    double linear_sum = 0.0;
    double cross_sum = 0.0;
    double quadratic_sum = 0.0;
    double raw[SECANT_SINDY_MAX_FEATURE_CAPACITY];
    size_t row;
    size_t col;

    for (row = 0u; row < capacity; ++row) {
        raw[row] = row < active_features ? standardized[row] / scales[row] : 0.0;
        coefficients[row] = (float)raw[row];
        intercept -= raw[row] * means[row];
        linear_sum += raw[row] * (double)statistics[row];
        cross_sum += raw[row] * (double)cross[row * num_targets + target_index];
    }
    for (row = 0u; row < active_features; ++row) {
        for (col = 0u; col < active_features; ++col) {
            quadratic_sum += raw[row] * (double)gram[row * capacity + col] * raw[col];
        }
    }
    *intercept_ret = (float)intercept;

    {
        double sse = (double)target_stats[SECANT_SINDY_TARGET_STAT_SUM_SQUARED_F32] - 2.0 * cross_sum +
            quadratic_sum - 2.0 * intercept * target_sum + 2.0 * intercept * linear_sum +
            (double)num_rows * intercept * intercept;
        return sse > 0.0 ? sse : 0.0;
    }
}

static SecantSindyResult secant_sindy_cpu_solve_common_validate(size_t structure_size, size_t required_size,
                                                                uint32_t version, uint32_t flags,
                                                                size_t feature_capacity, size_t num_asts,
                                                                size_t num_rows, size_t num_targets,
                                                                size_t num_sweeps, float scale_epsilon,
                                                                SecantSindyConstHostSpanF32 statistics,
                                                                size_t statistics_leading_dimension,
                                                                SecantSindyConstHostSpanF32 target_stats,
                                                                size_t target_stats_leading_dimension,
                                                                SecantSindyConstHostSpanF32 alphas,
                                                                SecantSindyHostSpanF32 coefficients,
                                                                SecantSindyHostSpanF32 intercepts,
                                                                SecantSindyHostSpanF32 sse,
                                                                SecantSindyHostSpanS32 solve_info,
                                                                size_t* cohorts_ret, size_t* output_vectors_ret) {
    SecantSindyResult result = secant_sindy_cpu_header_validate((uint32_t)structure_size, (uint32_t)required_size,
                                                                version, SECANT_SINDY_CPU_SOLVE_RUN_VERSION_1, flags);
    size_t cohorts;
    size_t stats_per_cohort;
    size_t statistics_extent;
    size_t target_stats_extent;
    size_t output_vectors;
    size_t coefficient_count;

    if (result != SECANT_SINDY_SUCCESS) {
        return result;
    }
    if (feature_capacity == 0u || feature_capacity > SECANT_SINDY_MAX_FEATURE_CAPACITY || num_asts == 0u ||
        num_rows == 0u || num_targets == 0u || num_targets > SECANT_SINDY_MAX_TARGETS || num_sweeps == 0u ||
        !(scale_epsilon > 0.0f) || !isfinite(scale_epsilon) || statistics.data == NULL || target_stats.data == NULL ||
        alphas.data == NULL || coefficients.data == NULL || intercepts.data == NULL || sse.data == NULL ||
        solve_info.data == NULL || target_stats_leading_dimension < SECANT_SINDY_TARGET_STAT_COUNT_F32) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    if (!secant_sindy_checked_add(num_asts, feature_capacity - 1u, &cohorts)) {
        return SECANT_SINDY_ERROR_OVERFLOW;
    }
    cohorts /= feature_capacity;
    if (!secant_sindy_checked_mul(feature_capacity, feature_capacity, &stats_per_cohort) ||
        !secant_sindy_checked_add(stats_per_cohort, feature_capacity, &stats_per_cohort) ||
        !secant_sindy_checked_mul(feature_capacity, num_targets, &statistics_extent) ||
        !secant_sindy_checked_add(stats_per_cohort, statistics_extent, &stats_per_cohort) ||
        !secant_sindy_span_extent(cohorts, statistics_leading_dimension, stats_per_cohort, &statistics_extent) ||
        !secant_sindy_span_extent(num_targets, target_stats_leading_dimension,
                                 SECANT_SINDY_TARGET_STAT_COUNT_F32, &target_stats_extent) ||
        !secant_sindy_checked_mul(num_sweeps, cohorts, &output_vectors) ||
        !secant_sindy_checked_mul(output_vectors, num_targets, &output_vectors) ||
        !secant_sindy_checked_mul(output_vectors, feature_capacity, &coefficient_count)) {
        return SECANT_SINDY_ERROR_OVERFLOW;
    }
    if (statistics_leading_dimension < stats_per_cohort || statistics.num_elements < statistics_extent ||
        target_stats.num_elements < target_stats_extent ||
        alphas.num_elements < num_sweeps || coefficients.num_elements < coefficient_count ||
        intercepts.num_elements < output_vectors || sse.num_elements < output_vectors ||
        solve_info.num_elements < output_vectors) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    *cohorts_ret = cohorts;
    *output_vectors_ret = output_vectors;
    return SECANT_SINDY_SUCCESS;
}

SecantSindyResult secant_sindy_cpu_target_stats_run(const SecantSindyCpuTargetStatsRun* run) {
    SecantSindyResult result;
    size_t target_extent;
    size_t stats_extent;
    size_t target;

    if (run == NULL) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    result = secant_sindy_cpu_header_validate(run->struct_size, (uint32_t)sizeof(*run), run->version,
                                              SECANT_SINDY_CPU_TARGET_STATS_RUN_VERSION_1, run->flags);
    if (result != SECANT_SINDY_SUCCESS) {
        return result;
    }
    if (run->num_rows == 0u || run->num_targets == 0u || run->targets.data == NULL ||
        run->target_stats.data == NULL ||
        (run->num_targets > 1u && run->targets_leading_dimension < run->num_rows) ||
        run->target_stats_leading_dimension < SECANT_SINDY_TARGET_STAT_COUNT_F32 ||
        !secant_sindy_span_extent(run->num_targets, run->targets_leading_dimension, run->num_rows, &target_extent) ||
        !secant_sindy_span_extent(run->num_targets, run->target_stats_leading_dimension,
                                 SECANT_SINDY_TARGET_STAT_COUNT_F32, &stats_extent)) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    if (run->targets.num_elements < target_extent || run->target_stats.num_elements < stats_extent) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    for (target = 0u; target < run->num_targets; ++target) {
        double sum = 0.0;
        double square_sum = 0.0;
        size_t row;
        for (row = 0u; row < run->num_rows; ++row) {
            const double value = run->targets.data[target * run->targets_leading_dimension + row];
            sum += value;
            square_sum += value * value;
        }
        run->target_stats.data[target * run->target_stats_leading_dimension + SECANT_SINDY_TARGET_STAT_SUM_F32] +=
            (float)sum;
        run->target_stats.data[
            target * run->target_stats_leading_dimension + SECANT_SINDY_TARGET_STAT_SUM_SQUARED_F32] +=
            (float)square_sum;
    }
    return SECANT_SINDY_SUCCESS;
}

SecantSindyResult secant_sindy_cpu_ridge_run(const SecantSindyCpuRidgeRun* run) {
    size_t cohorts;
    size_t output_vectors;
    size_t sweep;
    SecantSindyResult result;

    if (run == NULL) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    result = secant_sindy_cpu_solve_common_validate(
        run->struct_size, sizeof(*run), run->version, run->flags, run->feature_capacity, run->num_asts,
        run->num_rows, run->num_targets, run->num_sweeps, run->scale_epsilon, run->statistics,
        run->statistics_leading_dimension, run->target_stats, run->target_stats_leading_dimension, run->alphas,
        run->coefficients, run->intercepts, run->sse, run->solve_info, &cohorts, &output_vectors);
    (void)output_vectors;
    if (result != SECANT_SINDY_SUCCESS) {
        return result;
    }
    for (sweep = 0u; sweep < run->num_sweeps; ++sweep) {
        size_t cohort;
        if (!(run->alphas.data[sweep] >= 0.0f)) {
            return SECANT_SINDY_ERROR_INVALID_VALUE;
        }
        for (cohort = 0u; cohort < cohorts; ++cohort) {
            const size_t remaining = run->num_asts - cohort * run->feature_capacity;
            const size_t active_features = remaining < run->feature_capacity ? remaining : run->feature_capacity;
            const float* statistics = run->statistics.data + cohort * run->statistics_leading_dimension;
            const float* cross_base = statistics + run->feature_capacity + run->feature_capacity * run->feature_capacity;
            double means[SECANT_SINDY_MAX_FEATURE_CAPACITY];
            double scales[SECANT_SINDY_MAX_FEATURE_CAPACITY];
            double normalized[SECANT_SINDY_MAX_FEATURE_CAPACITY * SECANT_SINDY_MAX_FEATURE_CAPACITY];
            size_t target;

            secant_sindy_normalized_matrix_build(run->feature_capacity, active_features, run->num_rows,
                                                  run->scale_epsilon, statistics, means, scales, normalized);
            for (target = 0u; target < run->num_targets; ++target) {
                const size_t output_idx = (sweep * cohorts + cohort) * run->num_targets + target;
                const float* target_stats = run->target_stats.data + target * run->target_stats_leading_dimension;
                double matrix[SECANT_SINDY_MAX_FEATURE_CAPACITY * SECANT_SINDY_MAX_FEATURE_CAPACITY];
                double rhs[SECANT_SINDY_MAX_FEATURE_CAPACITY];
                double solution[SECANT_SINDY_MAX_FEATURE_CAPACITY] = {0};
                int32_t info;
                size_t row;

                for (row = 0u; row < run->feature_capacity; ++row) {
                    size_t col;
                    for (col = 0u; col < run->feature_capacity; ++col) {
                        matrix[row * run->feature_capacity + col] = normalized[row * run->feature_capacity + col];
                    }
                    matrix[row * run->feature_capacity + row] = row < active_features
                        ? 1.0 + (double)run->alphas.data[sweep]
                        : 1.0;
                }
                /* Cross statistics are feature-major, so gather the strided target column. */
                for (row = 0u; row < run->feature_capacity; ++row) {
                    const double centered = (double)cross_base[row * run->num_targets + target] -
                        (double)statistics[row] * (double)target_stats[0] / (double)run->num_rows;
                    rhs[row] = row < active_features
                        ? centered / ((double)run->num_rows * scales[row])
                        : 0.0;
                }
                secant_sindy_cholesky_solve(run->feature_capacity, matrix, rhs, solution, &info);
                run->solve_info.data[output_idx] = info;
                run->sse.data[output_idx] = (float)secant_sindy_raw_solution_write(
                    run->feature_capacity, active_features, run->num_rows, run->num_targets, target, statistics,
                    target_stats, means, scales, info == 0
                        ? solution
                        : (const double[SECANT_SINDY_MAX_FEATURE_CAPACITY]){0},
                    run->coefficients.data + output_idx * run->feature_capacity, run->intercepts.data + output_idx);
            }
        }
    }
    return SECANT_SINDY_SUCCESS;
}

SecantSindyResult secant_sindy_cpu_stlsq_run(const SecantSindyCpuSTLSQRun* run) {
    size_t cohorts;
    size_t output_vectors;
    size_t sweep;
    SecantSindyResult result;

    if (run == NULL) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    result = secant_sindy_cpu_solve_common_validate(
        run->struct_size, sizeof(*run), run->version, run->flags, run->feature_capacity, run->num_asts,
        run->num_rows, run->num_targets, run->num_sweeps, run->scale_epsilon, run->statistics,
        run->statistics_leading_dimension, run->target_stats, run->target_stats_leading_dimension, run->alphas,
        run->coefficients, run->intercepts, run->sse, run->solve_info, &cohorts, &output_vectors);
    if (result != SECANT_SINDY_SUCCESS) {
        return result;
    }
    if (run->max_iterations == 0u || run->thresholds.data == NULL || run->thresholds.num_elements < run->num_sweeps ||
        run->active_masks.data == NULL || run->active_masks.num_elements < output_vectors ||
        run->active_counts.data == NULL || run->active_counts.num_elements < output_vectors ||
        run->iteration_counts.data == NULL || run->iteration_counts.num_elements < output_vectors) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }

    for (sweep = 0u; sweep < run->num_sweeps; ++sweep) {
        size_t cohort;
        if (!(run->alphas.data[sweep] >= 0.0f) || !(run->thresholds.data[sweep] >= 0.0f)) {
            return SECANT_SINDY_ERROR_INVALID_VALUE;
        }
        for (cohort = 0u; cohort < cohorts; ++cohort) {
            const size_t remaining = run->num_asts - cohort * run->feature_capacity;
            const size_t active_features = remaining < run->feature_capacity ? remaining : run->feature_capacity;
            const float* statistics = run->statistics.data + cohort * run->statistics_leading_dimension;
            const float* cross_base = statistics + run->feature_capacity + run->feature_capacity * run->feature_capacity;
            double means[SECANT_SINDY_MAX_FEATURE_CAPACITY];
            double scales[SECANT_SINDY_MAX_FEATURE_CAPACITY];
            double normalized[SECANT_SINDY_MAX_FEATURE_CAPACITY * SECANT_SINDY_MAX_FEATURE_CAPACITY];
            size_t target;

            secant_sindy_normalized_matrix_build(run->feature_capacity, active_features, run->num_rows,
                                                  run->scale_epsilon, statistics, means, scales, normalized);
            for (target = 0u; target < run->num_targets; ++target) {
                const size_t output_idx = (sweep * cohorts + cohort) * run->num_targets + target;
                const float* target_stats = run->target_stats.data + target * run->target_stats_leading_dimension;
                uint32_t active_mask = active_features == 32u ? UINT32_MAX : ((1u << active_features) - 1u);
                double solution[SECANT_SINDY_MAX_FEATURE_CAPACITY] = {0};
                int32_t info = 0;
                size_t iterations = 0u;

                while (active_mask != 0u && iterations < run->max_iterations) {
                    double matrix[SECANT_SINDY_MAX_FEATURE_CAPACITY * SECANT_SINDY_MAX_FEATURE_CAPACITY];
                    double rhs[SECANT_SINDY_MAX_FEATURE_CAPACITY];
                    uint32_t next_mask = 0u;
                    size_t row;

                    for (row = 0u; row < run->feature_capacity; ++row) {
                        const int row_active = ((active_mask >> row) & 1u) != 0u;
                        size_t col;
                        for (col = 0u; col < run->feature_capacity; ++col) {
                            const int col_active = ((active_mask >> col) & 1u) != 0u;
                            matrix[row * run->feature_capacity + col] = row == col
                                ? (row_active ? 1.0 + run->alphas.data[sweep] : 1.0)
                                : (row_active && col_active ? normalized[row * run->feature_capacity + col] : 0.0);
                        }
                        {
                            const double centered = (double)cross_base[row * run->num_targets + target] -
                                (double)statistics[row] * target_stats[0] / (double)run->num_rows;
                            rhs[row] = row_active ? centered / ((double)run->num_rows * scales[row]) : 0.0;
                        }
                    }
                    if (!secant_sindy_cholesky_solve(run->feature_capacity, matrix, rhs, solution, &info)) {
                        active_mask = 0u;
                        ++iterations;
                        break;
                    }
                    for (row = 0u; row < active_features; ++row) {
                        if (((active_mask >> row) & 1u) != 0u && fabs(solution[row]) >= run->thresholds.data[sweep]) {
                            next_mask |= 1u << row;
                        }
                    }
                    ++iterations;
                    if (next_mask == active_mask || next_mask == 0u) {
                        active_mask = next_mask;
                        break;
                    }
                    active_mask = next_mask;
                }

                if (info == 0 && active_mask != 0u) {
                    double matrix[SECANT_SINDY_MAX_FEATURE_CAPACITY * SECANT_SINDY_MAX_FEATURE_CAPACITY];
                    double rhs[SECANT_SINDY_MAX_FEATURE_CAPACITY];
                    size_t row;
                    for (row = 0u; row < run->feature_capacity; ++row) {
                        const int row_active = ((active_mask >> row) & 1u) != 0u;
                        size_t col;
                        for (col = 0u; col < run->feature_capacity; ++col) {
                            const int col_active = ((active_mask >> col) & 1u) != 0u;
                            matrix[row * run->feature_capacity + col] = row == col
                                ? (row_active ? 1.0 + run->alphas.data[sweep] : 1.0)
                                : (row_active && col_active ? normalized[row * run->feature_capacity + col] : 0.0);
                        }
                        {
                            const double centered = (double)cross_base[row * run->num_targets + target] -
                                (double)statistics[row] * target_stats[0] / (double)run->num_rows;
                            rhs[row] = row_active ? centered / ((double)run->num_rows * scales[row]) : 0.0;
                        }
                    }
                    secant_sindy_cholesky_solve(run->feature_capacity, matrix, rhs, solution, &info);
                } else {
                    memset(solution, 0, run->feature_capacity * sizeof(*solution));
                }
                run->solve_info.data[output_idx] = info;
                run->active_masks.data[output_idx] = active_mask;
                run->active_counts.data[output_idx] = secant_sindy_popcount_u32(active_mask);
                run->iteration_counts.data[output_idx] = (int32_t)iterations;
                run->sse.data[output_idx] = (float)secant_sindy_raw_solution_write(
                    run->feature_capacity, active_features, run->num_rows, run->num_targets, target, statistics,
                    target_stats, means, scales, info == 0
                        ? solution
                        : (const double[SECANT_SINDY_MAX_FEATURE_CAPACITY]){0},
                    run->coefficients.data + output_idx * run->feature_capacity, run->intercepts.data + output_idx);
            }
        }
    }
    return SECANT_SINDY_SUCCESS;
}
