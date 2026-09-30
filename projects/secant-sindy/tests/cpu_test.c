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
#include <stdio.h>
#include <string.h>

enum {
    TEST_CAPACITY = 4,
    TEST_FEATURES = 3,
    TEST_TARGETS = 2,
    TEST_ROWS = 257,
    TEST_STATS = TEST_CAPACITY + TEST_CAPACITY * TEST_CAPACITY + TEST_CAPACITY * TEST_TARGETS
};

static int close_enough(float actual, float expected, float tolerance, const char* label) {
    if (fabsf(actual - expected) <= tolerance) {
        return 1;
    }
    fprintf(stderr, "%s mismatch: actual=%.9g expected=%.9g tolerance=%.9g\n", label, actual, expected, tolerance);
    return 0;
}

int main(void) {
    float features[TEST_FEATURES][TEST_ROWS];
    float targets[TEST_TARGETS][TEST_ROWS];
    float statistics[TEST_STATS] = {0};
    float target_stats[TEST_TARGETS][2] = {{0}};
    float alpha[1] = {0.0f};
    float threshold[1] = {0.5f};
    float coefficients[TEST_TARGETS * TEST_CAPACITY] = {0};
    float intercepts[TEST_TARGETS] = {0};
    float sse[TEST_TARGETS] = {0};
    int32_t info[TEST_TARGETS] = {0};
    uint32_t masks[TEST_TARGETS] = {0};
    int32_t counts[TEST_TARGETS] = {0};
    int32_t iterations[TEST_TARGETS] = {0};
    SecantSindyCpuTargetStatsRun target_run = secant_sindy_cpu_target_stats_run_init();
    SecantSindyCpuRidgeRun ridge = secant_sindy_cpu_ridge_run_init();
    SecantSindyCpuSTLSQRun stlsq = secant_sindy_cpu_stlsq_run_init();
    size_t row;
    size_t lhs;
    size_t rhs;
    int ok = 1;

    for (row = 0u; row < TEST_ROWS; ++row) {
        const float centered = (float)row - 128.0f;
        features[0][row] = centered * (1.0f / 53.0f);
        features[1][row] = sinf((float)row * 0.137f) + (float)(row % 7u) * 0.031f;
        features[2][row] = cosf((float)row * 0.073f) - (float)(row % 11u) * 0.019f;
        targets[0][row] = 2.0f + 3.0f * features[0][row] - 1.5f * features[1][row];
        targets[1][row] = -4.0f + 2.0f * features[1][row] - 3.5f * features[2][row];
    }
    for (lhs = 0u; lhs < TEST_FEATURES; ++lhs) {
        for (row = 0u; row < TEST_ROWS; ++row) {
            statistics[lhs] += features[lhs][row];
        }
        for (rhs = 0u; rhs < TEST_FEATURES; ++rhs) {
            for (row = 0u; row < TEST_ROWS; ++row) {
                statistics[TEST_CAPACITY + lhs * TEST_CAPACITY + rhs] +=
                    features[lhs][row] * features[rhs][row];
            }
        }
        for (rhs = 0u; rhs < TEST_TARGETS; ++rhs) {
            for (row = 0u; row < TEST_ROWS; ++row) {
                statistics[TEST_CAPACITY + TEST_CAPACITY * TEST_CAPACITY + lhs * TEST_TARGETS + rhs] +=
                    features[lhs][row] * targets[rhs][row];
            }
        }
    }

    target_run.targets.data = &targets[0][0];
    target_run.targets.num_elements = TEST_TARGETS * TEST_ROWS;
    target_run.targets_leading_dimension = TEST_ROWS;
    target_run.num_rows = TEST_ROWS;
    target_run.num_targets = TEST_TARGETS;
    target_run.target_stats.data = &target_stats[0][0];
    target_run.target_stats.num_elements = TEST_TARGETS * 2u;
    target_run.target_stats_leading_dimension = 2u;
    if (secant_sindy_cpu_target_stats_run(&target_run) != SECANT_SINDY_SUCCESS) {
        fprintf(stderr, "target stats failed\n");
        return 1;
    }

    ridge.feature_capacity = TEST_CAPACITY;
    ridge.num_asts = TEST_FEATURES;
    ridge.num_rows = TEST_ROWS;
    ridge.num_targets = TEST_TARGETS;
    ridge.num_sweeps = 1u;
    ridge.statistics.data = statistics;
    ridge.statistics.num_elements = TEST_STATS;
    ridge.statistics_leading_dimension = TEST_STATS;
    ridge.target_stats.data = &target_stats[0][0];
    ridge.target_stats.num_elements = TEST_TARGETS * 2u;
    ridge.target_stats_leading_dimension = 2u;
    ridge.alphas.data = alpha;
    ridge.alphas.num_elements = 1u;
    ridge.coefficients.data = coefficients;
    ridge.coefficients.num_elements = TEST_TARGETS * TEST_CAPACITY;
    ridge.intercepts.data = intercepts;
    ridge.intercepts.num_elements = TEST_TARGETS;
    ridge.sse.data = sse;
    ridge.sse.num_elements = TEST_TARGETS;
    ridge.solve_info.data = info;
    ridge.solve_info.num_elements = TEST_TARGETS;
    if (secant_sindy_cpu_ridge_run(&ridge) != SECANT_SINDY_SUCCESS) {
        fprintf(stderr, "ridge failed\n");
        return 1;
    }
    ok &= close_enough(coefficients[0], 3.0f, 2.0e-5f, "ridge y0 x0");
    ok &= close_enough(coefficients[1], -1.5f, 2.0e-5f, "ridge y0 x1");
    ok &= close_enough(coefficients[2], 0.0f, 2.0e-5f, "ridge y0 x2");
    ok &= close_enough(coefficients[3], 0.0f, 0.0f, "ridge y0 inactive");
    ok &= close_enough(intercepts[0], 2.0f, 2.0e-5f, "ridge y0 intercept");
    ok &= close_enough(coefficients[4], 0.0f, 2.0e-5f, "ridge y1 x0");
    ok &= close_enough(coefficients[5], 2.0f, 2.0e-5f, "ridge y1 x1");
    ok &= close_enough(coefficients[6], -3.5f, 2.0e-5f, "ridge y1 x2");
    ok &= close_enough(coefficients[7], 0.0f, 0.0f, "ridge y1 inactive");
    ok &= close_enough(intercepts[1], -4.0f, 2.0e-5f, "ridge y1 intercept");
    ok &= info[0] == 0 && info[1] == 0;
    ok &= sse[0] < 2.0e-3f && sse[1] < 2.0e-3f;

    memset(coefficients, 0, sizeof(coefficients));
    memset(intercepts, 0, sizeof(intercepts));
    memset(sse, 0, sizeof(sse));
    memset(info, 0, sizeof(info));
    stlsq.feature_capacity = TEST_CAPACITY;
    stlsq.num_asts = TEST_FEATURES;
    stlsq.num_rows = TEST_ROWS;
    stlsq.num_targets = TEST_TARGETS;
    stlsq.num_sweeps = 1u;
    stlsq.max_iterations = TEST_CAPACITY;
    stlsq.statistics = ridge.statistics;
    stlsq.statistics_leading_dimension = TEST_STATS;
    stlsq.target_stats = ridge.target_stats;
    stlsq.target_stats_leading_dimension = 2u;
    stlsq.alphas = ridge.alphas;
    stlsq.thresholds.data = threshold;
    stlsq.thresholds.num_elements = 1u;
    stlsq.coefficients = ridge.coefficients;
    stlsq.intercepts = ridge.intercepts;
    stlsq.sse = ridge.sse;
    stlsq.solve_info = ridge.solve_info;
    stlsq.active_masks.data = masks;
    stlsq.active_masks.num_elements = TEST_TARGETS;
    stlsq.active_counts.data = counts;
    stlsq.active_counts.num_elements = TEST_TARGETS;
    stlsq.iteration_counts.data = iterations;
    stlsq.iteration_counts.num_elements = TEST_TARGETS;
    if (secant_sindy_cpu_stlsq_run(&stlsq) != SECANT_SINDY_SUCCESS) {
        fprintf(stderr, "stlsq failed\n");
        return 1;
    }
    ok &= masks[0] == 0x3u && masks[1] == 0x6u;
    ok &= counts[0] == 2 && counts[1] == 2;
    ok &= iterations[0] > 0 && iterations[1] > 0;
    ok &= close_enough(coefficients[0], 3.0f, 2.0e-5f, "stlsq y0 x0");
    ok &= close_enough(coefficients[1], -1.5f, 2.0e-5f, "stlsq y0 x1");
    ok &= close_enough(coefficients[5], 2.0f, 2.0e-5f, "stlsq y1 x1");
    ok &= close_enough(coefficients[6], -3.5f, 2.0e-5f, "stlsq y1 x2");
    ok &= close_enough(intercepts[0], 2.0f, 2.0e-5f, "stlsq y0 intercept");
    ok &= close_enough(intercepts[1], -4.0f, 2.0e-5f, "stlsq y1 intercept");
    ok &= sse[0] < 2.0e-3f && sse[1] < 2.0e-3f;

    if (!ok) {
        return 1;
    }
    printf("secant-sindy CPU ridge/STLSQ test passed\n");
    return 0;
}
