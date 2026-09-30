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
#include "secant_sindy.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parity_u32(uint32_t value) {
    value ^= value >> 16u;
    value ^= value >> 8u;
    value ^= value >> 4u;
    value &= 0x0fu;
    return (0x6996u >> value) & 1u;
}

static float walsh_value(size_t row, size_t feature) {
    return parity_u32((uint32_t)row & (uint32_t)(feature + 1u)) != 0 ? -1.0f : 1.0f;
}

static int32_t popcount_u32(uint32_t value) {
    int32_t count = 0;

    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

static void statistics_build(size_t capacity, size_t num_asts, size_t num_rows, size_t num_targets,
                             const float* features, const float* targets, float* statistics,
                             size_t statistics_leading_dimension, float* target_stats) {
    const size_t cohorts = (num_asts + capacity - 1u) / capacity;
    size_t target;
    size_t cohort;

    memset(statistics, 0, cohorts * statistics_leading_dimension * sizeof(*statistics));
    memset(target_stats, 0, num_targets * 2u * sizeof(*target_stats));
    for (target = 0u; target < num_targets; ++target) {
        size_t row;
        for (row = 0u; row < num_rows; ++row) {
            const float value = targets[target * num_rows + row];
            target_stats[target * 2u] += value;
            target_stats[target * 2u + 1u] += value * value;
        }
    }
    for (cohort = 0u; cohort < cohorts; ++cohort) {
        float* cohort_stats = statistics + cohort * statistics_leading_dimension;
        const size_t feature_begin = cohort * capacity;
        const size_t remaining = num_asts - feature_begin;
        const size_t active_features = remaining < capacity ? remaining : capacity;
        size_t lhs;

        for (lhs = 0u; lhs < active_features; ++lhs) {
            const float* lhs_values = features + (feature_begin + lhs) * num_rows;
            size_t row;
            size_t rhs;

            for (row = 0u; row < num_rows; ++row) {
                cohort_stats[lhs] += lhs_values[row];
            }
            for (rhs = 0u; rhs < active_features; ++rhs) {
                const float* rhs_values = features + (feature_begin + rhs) * num_rows;
                for (row = 0u; row < num_rows; ++row) {
                    cohort_stats[capacity + lhs * capacity + rhs] += lhs_values[row] * rhs_values[row];
                }
            }
            for (target = 0u; target < num_targets; ++target) {
                const float* target_values = targets + target * num_rows;
                for (row = 0u; row < num_rows; ++row) {
                    cohort_stats[capacity + capacity * capacity + lhs * num_targets + target] +=
                        lhs_values[row] * target_values[row];
                }
            }
        }
    }
}

static int close_enough(float actual, float expected, float absolute, float relative, const char* label) {
    const float tolerance = absolute + relative * fabsf(expected);

    if (isfinite(actual) && fabsf(actual - expected) <= tolerance) {
        return 1;
    }
    fprintf(stderr, "%s mismatch: actual=%.9g expected=%.9g tolerance=%.9g\n", label, actual, expected, tolerance);
    return 0;
}

static int multi_cohort_test(void) {
    enum {
        CAPACITY = 4,
        ASTS = 9,
        ROWS = 256,
        TARGETS = 2,
        SWEEPS = 3,
        COHORTS = 3,
        STATS_PER_COHORT = CAPACITY + CAPACITY * CAPACITY + CAPACITY * TARGETS,
        OUTPUTS = SWEEPS * COHORTS * TARGETS
    };
    static const float expected_raw[TARGETS][ASTS] = {
        {3.0f, 0.75f, 0.0f, -1.5f, 0.0f, 2.5f, 0.1f, 0.0f, -4.0f},
        {0.0f, 2.0f, -3.5f, 0.0f, 1.25f, 0.0f, 0.75f, 0.0f, 0.0f}
    };
    static const float expected_intercepts[TARGETS] = {2.0f, -4.0f};
    float features[ASTS * ROWS];
    float targets[TARGETS * ROWS];
    float statistics[COHORTS * STATS_PER_COHORT];
    float target_stats[TARGETS * 2u];
    float alphas[SWEEPS] = {0.0f, 0.0f, 0.0f};
    float thresholds[SWEEPS] = {0.0f, 0.5f, 10.0f};
    float coefficients[OUTPUTS * CAPACITY];
    float intercepts[OUTPUTS];
    float sse[OUTPUTS];
    int32_t info[OUTPUTS];
    uint32_t masks[OUTPUTS];
    int32_t counts[OUTPUTS];
    int32_t iterations[OUTPUTS];
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();
    uint32_t masks_expected[SWEEPS][COHORTS][TARGETS];
    size_t feature;
    size_t row;
    size_t target;
    size_t sweep;
    size_t cohort;
    int ok = 1;

    for (feature = 0u; feature < ASTS; ++feature) {
        for (row = 0u; row < ROWS; ++row) {
            features[feature * ROWS + row] = walsh_value(row, feature);
        }
    }
    for (target = 0u; target < TARGETS; ++target) {
        for (row = 0u; row < ROWS; ++row) {
            float value = expected_intercepts[target];
            for (feature = 0u; feature < ASTS; ++feature) {
                value += expected_raw[target][feature] * features[feature * ROWS + row];
            }
            targets[target * ROWS + row] = value;
        }
    }
    statistics_build(CAPACITY, ASTS, ROWS, TARGETS, features, targets, statistics, STATS_PER_COHORT, target_stats);

    for (cohort = 0u; cohort < COHORTS; ++cohort) {
        const size_t active = cohort + 1u == COHORTS ? 1u : CAPACITY;
        const uint32_t full_mask = active == 32u ? UINT32_MAX : ((1u << active) - 1u);
        for (target = 0u; target < TARGETS; ++target) {
            masks_expected[0][cohort][target] = full_mask;
            masks_expected[2][cohort][target] = 0u;
        }
    }
    masks_expected[1][0][0] = 0xbu;
    masks_expected[1][1][0] = 0x2u;
    masks_expected[1][2][0] = 0x1u;
    masks_expected[1][0][1] = 0x6u;
    masks_expected[1][1][1] = 0x5u;
    masks_expected[1][2][1] = 0x0u;

    run.feature_capacity = CAPACITY;
    run.num_asts = ASTS;
    run.num_rows = ROWS;
    run.num_targets = TARGETS;
    run.num_sweeps = SWEEPS;
    run.max_iterations = CAPACITY;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, COHORTS * STATS_PER_COHORT};
    run.statistics_leading_dimension = STATS_PER_COHORT;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, TARGETS * 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){alphas, SWEEPS};
    run.thresholds = (SecantSindyConstHostSpanF32){thresholds, SWEEPS};
    run.coefficients = (SecantSindyHostSpanF32){coefficients, OUTPUTS * CAPACITY};
    run.intercepts = (SecantSindyHostSpanF32){intercepts, OUTPUTS};
    run.sse = (SecantSindyHostSpanF32){sse, OUTPUTS};
    run.solve_info = (SecantSindyHostSpanS32){info, OUTPUTS};
    run.active_masks = (SecantSindyHostSpanU32){masks, OUTPUTS};
    run.active_counts = (SecantSindyHostSpanS32){counts, OUTPUTS};
    run.iteration_counts = (SecantSindyHostSpanS32){iterations, OUTPUTS};
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_SUCCESS) {
        fprintf(stderr, "multi-cohort STLSQ failed\n");
        return 0;
    }

    for (sweep = 0u; sweep < SWEEPS; ++sweep) {
        for (cohort = 0u; cohort < COHORTS; ++cohort) {
            const size_t active = cohort + 1u == COHORTS ? 1u : CAPACITY;
            for (target = 0u; target < TARGETS; ++target) {
                const size_t output = (sweep * COHORTS + cohort) * TARGETS + target;
                const uint32_t expected_mask = masks_expected[sweep][cohort][target];
                const uint32_t initial_mask = active == 32u ? UINT32_MAX : ((1u << active) - 1u);
                const int32_t expected_iterations = sweep == 0u || sweep == 2u || expected_mask == 0u ||
                    expected_mask == initial_mask ? 1 : 2;
                size_t local;

                if (info[output] != 0 || masks[output] != expected_mask ||
                    counts[output] != popcount_u32(expected_mask) ||
                    iterations[output] != expected_iterations || !isfinite(sse[output])) {
                    fprintf(stderr, "multi-cohort metadata mismatch: sweep=%zu cohort=%zu target=%zu\n",
                            sweep, cohort, target);
                    ok = 0;
                }
                ok &= close_enough(intercepts[output], expected_intercepts[target], 2.0e-5f, 2.0e-5f,
                                   "multi-cohort intercept");
                for (local = 0u; local < CAPACITY; ++local) {
                    const size_t global = cohort * CAPACITY + local;
                    const float expected = local < active && ((expected_mask >> local) & 1u) != 0u
                        ? expected_raw[target][global]
                        : 0.0f;
                    ok &= close_enough(coefficients[output * CAPACITY + local], expected, 3.0e-5f, 3.0e-5f,
                                       "multi-cohort coefficient");
                }
            }
        }
    }
    return ok;
}

static int threshold_equality_test(void) {
    float statistics[3] = {0.0f, 64.0f, 32.0f};
    float target_stats[2] = {0.0f, 16.0f};
    float alpha = 0.0f;
    float threshold = 0.5f;
    float coefficient;
    float intercept;
    float sse;
    int32_t info;
    uint32_t mask;
    int32_t count;
    int32_t iterations;
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();

    run.feature_capacity = 1u;
    run.num_asts = 1u;
    run.num_rows = 64u;
    run.num_targets = 1u;
    run.num_sweeps = 1u;
    run.max_iterations = 1u;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, 3u};
    run.statistics_leading_dimension = 3u;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){&alpha, 1u};
    run.thresholds = (SecantSindyConstHostSpanF32){&threshold, 1u};
    run.coefficients = (SecantSindyHostSpanF32){&coefficient, 1u};
    run.intercepts = (SecantSindyHostSpanF32){&intercept, 1u};
    run.sse = (SecantSindyHostSpanF32){&sse, 1u};
    run.solve_info = (SecantSindyHostSpanS32){&info, 1u};
    run.active_masks = (SecantSindyHostSpanU32){&mask, 1u};
    run.active_counts = (SecantSindyHostSpanS32){&count, 1u};
    run.iteration_counts = (SecantSindyHostSpanS32){&iterations, 1u};
    return secant_sindy_cpu_stlsq_run(&run) == SECANT_SINDY_SUCCESS && info == 0 && mask == 1u && count == 1 &&
        iterations == 1 && coefficient == 0.5f && intercept == 0.0f && sse == 0.0f;
}

static int staged_elimination_test(void) {
    enum { CAPACITY = 4, ROWS = 1000, STATS = CAPACITY + CAPACITY * CAPACITY + CAPACITY };
    static const float correlation[CAPACITY][CAPACITY] = {
        {1.0f, 0.8092803f, 0.4066377f, -0.0680922f},
        {0.8092803f, 1.0f, 0.5259593f, -0.1133720f},
        {0.4066377f, 0.5259593f, 1.0f, 0.0224213f},
        {-0.0680922f, -0.1133720f, 0.0224213f, 1.0f}
    };
    static const float rhs[CAPACITY] = {-0.0649781f, 0.0709426f, 0.0121394f, 0.0791524f};
    static const uint32_t expected_masks[3] = {0xbu, 0x3u, 0x3u};
    static const int32_t expected_iterations[3] = {1, 2, 3};
    float statistics[STATS] = {0};
    float target_stats[2] = {0.0f, 1000.0f};
    float alpha[1] = {0.0f};
    float threshold[1] = {0.1f};
    float coefficients[CAPACITY];
    float intercept;
    float sse;
    int32_t info;
    uint32_t mask;
    int32_t count;
    int32_t iterations;
    size_t limit;
    size_t row;
    size_t col;
    int ok = 1;

    for (row = 0u; row < CAPACITY; ++row) {
        for (col = 0u; col < CAPACITY; ++col) {
            statistics[CAPACITY + row * CAPACITY + col] = (float)ROWS * correlation[row][col];
        }
        statistics[CAPACITY + CAPACITY * CAPACITY + row] = (float)ROWS * rhs[row];
    }
    for (limit = 1u; limit <= 3u; ++limit) {
        SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();

        run.feature_capacity = CAPACITY;
        run.num_asts = CAPACITY;
        run.num_rows = ROWS;
        run.num_targets = 1u;
        run.num_sweeps = 1u;
        run.max_iterations = limit;
        run.statistics = (SecantSindyConstHostSpanF32){statistics, STATS};
        run.statistics_leading_dimension = STATS;
        run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
        run.target_stats_leading_dimension = 2u;
        run.alphas = (SecantSindyConstHostSpanF32){alpha, 1u};
        run.thresholds = (SecantSindyConstHostSpanF32){threshold, 1u};
        run.coefficients = (SecantSindyHostSpanF32){coefficients, CAPACITY};
        run.intercepts = (SecantSindyHostSpanF32){&intercept, 1u};
        run.sse = (SecantSindyHostSpanF32){&sse, 1u};
        run.solve_info = (SecantSindyHostSpanS32){&info, 1u};
        run.active_masks = (SecantSindyHostSpanU32){&mask, 1u};
        run.active_counts = (SecantSindyHostSpanS32){&count, 1u};
        run.iteration_counts = (SecantSindyHostSpanS32){&iterations, 1u};
        if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_SUCCESS || info != 0 ||
            mask != expected_masks[limit - 1u] || iterations != expected_iterations[limit - 1u]) {
            fprintf(stderr, "staged elimination mismatch at limit %zu: mask=0x%x iterations=%d info=%d\n",
                    limit, mask, iterations, info);
            ok = 0;
        }
    }
    return ok;
}

static int singular_ridge_test(void) {
    enum { CAPACITY = 2, ROWS = 64, SWEEPS = 2, STATS = CAPACITY + CAPACITY * CAPACITY + CAPACITY };
    float statistics[STATS] = {0.0f, 0.0f, 64.0f, 64.0f, 64.0f, 64.0f, 128.0f, 128.0f};
    float target_stats[2] = {64.0f, 320.0f};
    float alphas[SWEEPS] = {0.0f, 0.1f};
    float thresholds[SWEEPS] = {0.1f, 0.1f};
    float coefficients[SWEEPS * CAPACITY];
    float intercepts[SWEEPS];
    float sse[SWEEPS];
    int32_t info[SWEEPS];
    uint32_t masks[SWEEPS];
    int32_t counts[SWEEPS];
    int32_t iterations[SWEEPS];
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();
    int ok = 1;

    run.feature_capacity = CAPACITY;
    run.num_asts = CAPACITY;
    run.num_rows = ROWS;
    run.num_targets = 1u;
    run.num_sweeps = SWEEPS;
    run.max_iterations = CAPACITY;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, STATS};
    run.statistics_leading_dimension = STATS;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){alphas, SWEEPS};
    run.thresholds = (SecantSindyConstHostSpanF32){thresholds, SWEEPS};
    run.coefficients = (SecantSindyHostSpanF32){coefficients, SWEEPS * CAPACITY};
    run.intercepts = (SecantSindyHostSpanF32){intercepts, SWEEPS};
    run.sse = (SecantSindyHostSpanF32){sse, SWEEPS};
    run.solve_info = (SecantSindyHostSpanS32){info, SWEEPS};
    run.active_masks = (SecantSindyHostSpanU32){masks, SWEEPS};
    run.active_counts = (SecantSindyHostSpanS32){counts, SWEEPS};
    run.iteration_counts = (SecantSindyHostSpanS32){iterations, SWEEPS};
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_SUCCESS) {
        return 0;
    }
    if (info[0] == 0 || masks[0] != 0u || counts[0] != 0 || coefficients[0] != 0.0f || coefficients[1] != 0.0f) {
        fprintf(stderr, "singular unregularized system was not rejected\n");
        ok = 0;
    }
    if (info[1] != 0 || masks[1] != 0x3u || counts[1] != 2 || iterations[1] != 1) {
        fprintf(stderr, "regularized duplicate-feature system failed\n");
        ok = 0;
    }
    ok &= close_enough(coefficients[2], 2.0f / 2.1f, 2.0e-5f, 2.0e-5f, "duplicate coefficient 0");
    ok &= close_enough(coefficients[3], 2.0f / 2.1f, 2.0e-5f, 2.0e-5f, "duplicate coefficient 1");
    ok &= close_enough(intercepts[1], 1.0f, 2.0e-5f, 2.0e-5f, "duplicate intercept");
    return ok;
}

static int mask_boundary_test(void) {
    enum { CAPACITY = 32, ROWS = 256, STATS = CAPACITY + CAPACITY * CAPACITY + CAPACITY };
    float features[CAPACITY * ROWS];
    float targets[ROWS];
    float statistics[STATS];
    float target_stats[2];
    float alpha[1] = {0.0f};
    float threshold[1] = {0.5f};
    float coefficients[CAPACITY];
    float intercept;
    float sse;
    int32_t info;
    uint32_t mask;
    int32_t count;
    int32_t iterations;
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();
    size_t feature;
    size_t row;
    int ok = 1;

    for (feature = 0u; feature < CAPACITY; ++feature) {
        for (row = 0u; row < ROWS; ++row) {
            features[feature * ROWS + row] = walsh_value(row, feature);
        }
    }
    for (row = 0u; row < ROWS; ++row) {
        targets[row] = 3.0f + features[row] + 2.0f * features[31u * ROWS + row];
    }
    statistics_build(CAPACITY, CAPACITY, ROWS, 1u, features, targets, statistics, STATS, target_stats);
    run.feature_capacity = CAPACITY;
    run.num_asts = CAPACITY;
    run.num_rows = ROWS;
    run.num_targets = 1u;
    run.num_sweeps = 1u;
    run.max_iterations = CAPACITY;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, STATS};
    run.statistics_leading_dimension = STATS;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){alpha, 1u};
    run.thresholds = (SecantSindyConstHostSpanF32){threshold, 1u};
    run.coefficients = (SecantSindyHostSpanF32){coefficients, CAPACITY};
    run.intercepts = (SecantSindyHostSpanF32){&intercept, 1u};
    run.sse = (SecantSindyHostSpanF32){&sse, 1u};
    run.solve_info = (SecantSindyHostSpanS32){&info, 1u};
    run.active_masks = (SecantSindyHostSpanU32){&mask, 1u};
    run.active_counts = (SecantSindyHostSpanS32){&count, 1u};
    run.iteration_counts = (SecantSindyHostSpanS32){&iterations, 1u};
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_SUCCESS || info != 0 || mask != 0x80000001u ||
        count != 2 || iterations != 2) {
        fprintf(stderr, "32-feature mask boundary failed: mask=0x%08x count=%d iterations=%d info=%d\n",
                mask, count, iterations, info);
        return 0;
    }
    ok &= close_enough(coefficients[0], 1.0f, 2.0e-5f, 2.0e-5f, "mask bit 0 coefficient");
    ok &= close_enough(coefficients[31], 2.0f, 2.0e-5f, 2.0e-5f, "mask bit 31 coefficient");
    ok &= close_enough(intercept, 3.0f, 2.0e-5f, 2.0e-5f, "mask boundary intercept");
    return ok;
}

static int validation_test(void) {
    float statistics[3] = {0.0f, 1.0f, 0.0f};
    float target_stats[2] = {0.0f, 1.0f};
    float alpha = 0.0f;
    float threshold = 0.0f;
    float coefficient;
    float intercept;
    float sse;
    int32_t info;
    uint32_t mask;
    int32_t count;
    int32_t iterations;
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();

    run.feature_capacity = 1u;
    run.num_asts = 1u;
    run.num_rows = 1u;
    run.num_targets = 1u;
    run.num_sweeps = 1u;
    run.max_iterations = 1u;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, 3u};
    run.statistics_leading_dimension = 3u;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){&alpha, 1u};
    run.thresholds = (SecantSindyConstHostSpanF32){&threshold, 1u};
    run.coefficients = (SecantSindyHostSpanF32){&coefficient, 1u};
    run.intercepts = (SecantSindyHostSpanF32){&intercept, 1u};
    run.sse = (SecantSindyHostSpanF32){&sse, 1u};
    run.solve_info = (SecantSindyHostSpanS32){&info, 1u};
    run.active_masks = (SecantSindyHostSpanU32){&mask, 1u};
    run.active_counts = (SecantSindyHostSpanS32){&count, 1u};
    run.iteration_counts = (SecantSindyHostSpanS32){&iterations, 1u};

    run.max_iterations = 0u;
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_ERROR_INVALID_VALUE) {
        return 0;
    }
    run.max_iterations = 1u;
    alpha = -1.0f;
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_ERROR_INVALID_VALUE) {
        return 0;
    }
    alpha = 0.0f;
    threshold = -1.0f;
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_ERROR_INVALID_VALUE) {
        return 0;
    }
    threshold = NAN;
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_ERROR_INVALID_VALUE) {
        return 0;
    }
    threshold = 0.0f;
    run.active_masks.num_elements = 0u;
    return secant_sindy_cpu_stlsq_run(&run) == SECANT_SINDY_ERROR_INVALID_VALUE;
}

int main(void) {
    if (!multi_cohort_test() || !threshold_equality_test() || !staged_elimination_test() || !singular_ridge_test() ||
        !mask_boundary_test() || !validation_test()) {
        return 1;
    }
    printf("secant-sindy STLSQ edge tests passed\n");
    return 0;
}
