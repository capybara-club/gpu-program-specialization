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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int size_parse(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static uint32_t random_u32(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float random_signed_f32(uint32_t* state) {
    const uint32_t bits = random_u32(state) >> 8u;
    return (float)bits * (2.0f / 16777215.0f) - 1.0f;
}

static void float_array_print(const char* name, const float* values, size_t count) {
    size_t idx;

    printf("%s %zu", name, count);
    for (idx = 0u; idx < count; ++idx) {
        printf(" %.9g", values[idx]);
    }
    putchar('\n');
}

static void s32_array_print(const char* name, const int32_t* values, size_t count) {
    size_t idx;

    printf("%s %zu", name, count);
    for (idx = 0u; idx < count; ++idx) {
        printf(" %d", values[idx]);
    }
    putchar('\n');
}

static void u32_array_print(const char* name, const uint32_t* values, size_t count) {
    size_t idx;

    printf("%s %zu", name, count);
    for (idx = 0u; idx < count; ++idx) {
        printf(" %u", values[idx]);
    }
    putchar('\n');
}

int main(int argc, char** argv) {
    size_t seed_value;
    size_t capacity;
    size_t num_asts;
    size_t num_rows;
    size_t num_targets;
    size_t num_sweeps;
    size_t max_iterations;
    size_t cohorts;
    size_t stats_per_cohort;
    size_t statistics_count;
    size_t output_count;
    size_t coefficient_count;
    uint32_t random_state;
    float* features = NULL;
    float* targets = NULL;
    float* statistics = NULL;
    float* target_stats = NULL;
    float* alphas = NULL;
    float* thresholds = NULL;
    float* coefficients = NULL;
    float* intercepts = NULL;
    float* sse = NULL;
    int32_t* solve_info = NULL;
    uint32_t* active_masks = NULL;
    int32_t* active_counts = NULL;
    int32_t* iteration_counts = NULL;
    SecantSindyCpuSTLSQRun run = secant_sindy_cpu_stlsq_run_init();
    size_t row;
    size_t feature;
    size_t target;
    size_t sweep;
    size_t cohort;
    int status = 1;

    if (argc != 8 || !size_parse(argv[1], &seed_value) || seed_value > UINT32_MAX ||
        !size_parse(argv[2], &capacity) || !size_parse(argv[3], &num_asts) ||
        !size_parse(argv[4], &num_rows) || !size_parse(argv[5], &num_targets) ||
        !size_parse(argv[6], &num_sweeps) || !size_parse(argv[7], &max_iterations) || capacity == 0u ||
        capacity > SECANT_SINDY_MAX_FEATURE_CAPACITY || num_asts == 0u || num_rows == 0u || num_targets == 0u ||
        num_targets > SECANT_SINDY_MAX_TARGETS || num_sweeps == 0u || max_iterations == 0u) {
        fprintf(stderr, "usage: %s seed capacity asts rows targets sweeps max-iterations\n", argv[0]);
        return 2;
    }
    cohorts = (num_asts + capacity - 1u) / capacity;
    stats_per_cohort = capacity + capacity * capacity + capacity * num_targets;
    statistics_count = cohorts * stats_per_cohort;
    output_count = num_sweeps * cohorts * num_targets;
    coefficient_count = output_count * capacity;
    random_state = (uint32_t)seed_value;

    features = (float*)malloc(num_asts * num_rows * sizeof(*features));
    targets = (float*)malloc(num_targets * num_rows * sizeof(*targets));
    statistics = (float*)calloc(statistics_count, sizeof(*statistics));
    target_stats = (float*)calloc(num_targets * 2u, sizeof(*target_stats));
    alphas = (float*)malloc(num_sweeps * sizeof(*alphas));
    thresholds = (float*)malloc(num_sweeps * sizeof(*thresholds));
    coefficients = (float*)malloc(coefficient_count * sizeof(*coefficients));
    intercepts = (float*)malloc(output_count * sizeof(*intercepts));
    sse = (float*)malloc(output_count * sizeof(*sse));
    solve_info = (int32_t*)malloc(output_count * sizeof(*solve_info));
    active_masks = (uint32_t*)malloc(output_count * sizeof(*active_masks));
    active_counts = (int32_t*)malloc(output_count * sizeof(*active_counts));
    iteration_counts = (int32_t*)malloc(output_count * sizeof(*iteration_counts));
    if (features == NULL || targets == NULL || statistics == NULL || target_stats == NULL || alphas == NULL ||
        thresholds == NULL || coefficients == NULL || intercepts == NULL || sse == NULL || solve_info == NULL ||
        active_masks == NULL || active_counts == NULL || iteration_counts == NULL) {
        goto cleanup;
    }

    for (row = 0u; row < num_rows; ++row) {
        for (feature = 0u; feature < num_asts; ++feature) {
            float value = random_signed_f32(&random_state);
            if (feature != 0u) {
                value = 0.8f * value + 0.2f * features[(feature - 1u) * num_rows + row];
            }
            value += 0.125f * (float)((int)(feature % 3u) - 1);
            features[feature * num_rows + row] = value;
        }
    }
    for (target = 0u; target < num_targets; ++target) {
        const float intercept = -0.75f + 0.625f * (float)target;
        for (row = 0u; row < num_rows; ++row) {
            float value = intercept;
            for (feature = 0u; feature < num_asts; ++feature) {
                const int signed_code = (int)((feature * 7u + target * 11u + 3u) % 9u) - 4;
                const float coefficient = (feature + 2u * target) % 5u == 0u ? 0.0f : 0.18f * (float)signed_code;
                value += coefficient * features[feature * num_rows + row];
            }
            targets[target * num_rows + row] = value + 0.01f * random_signed_f32(&random_state);
        }
    }
    for (target = 0u; target < num_targets; ++target) {
        for (row = 0u; row < num_rows; ++row) {
            const float value = targets[target * num_rows + row];
            target_stats[target * 2u] += value;
            target_stats[target * 2u + 1u] += value * value;
        }
    }
    for (cohort = 0u; cohort < cohorts; ++cohort) {
        float* cohort_stats = statistics + cohort * stats_per_cohort;
        const size_t feature_begin = cohort * capacity;
        const size_t remaining = num_asts - feature_begin;
        const size_t active_features = remaining < capacity ? remaining : capacity;
        size_t lhs;

        for (lhs = 0u; lhs < active_features; ++lhs) {
            const float* lhs_values = features + (feature_begin + lhs) * num_rows;
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
                for (row = 0u; row < num_rows; ++row) {
                    cohort_stats[capacity + capacity * capacity + lhs * num_targets + target] +=
                        lhs_values[row] * targets[target * num_rows + row];
                }
            }
        }
    }
    for (sweep = 0u; sweep < num_sweeps; ++sweep) {
        alphas[sweep] = 0.01f + 0.025f * (float)sweep;
        thresholds[sweep] = 0.025f + 0.075f * (float)sweep;
    }

    run.feature_capacity = capacity;
    run.num_asts = num_asts;
    run.num_rows = num_rows;
    run.num_targets = num_targets;
    run.num_sweeps = num_sweeps;
    run.max_iterations = max_iterations;
    run.statistics = (SecantSindyConstHostSpanF32){statistics, statistics_count};
    run.statistics_leading_dimension = stats_per_cohort;
    run.target_stats = (SecantSindyConstHostSpanF32){target_stats, num_targets * 2u};
    run.target_stats_leading_dimension = 2u;
    run.alphas = (SecantSindyConstHostSpanF32){alphas, num_sweeps};
    run.thresholds = (SecantSindyConstHostSpanF32){thresholds, num_sweeps};
    run.coefficients = (SecantSindyHostSpanF32){coefficients, coefficient_count};
    run.intercepts = (SecantSindyHostSpanF32){intercepts, output_count};
    run.sse = (SecantSindyHostSpanF32){sse, output_count};
    run.solve_info = (SecantSindyHostSpanS32){solve_info, output_count};
    run.active_masks = (SecantSindyHostSpanU32){active_masks, output_count};
    run.active_counts = (SecantSindyHostSpanS32){active_counts, output_count};
    run.iteration_counts = (SecantSindyHostSpanS32){iteration_counts, output_count};
    if (secant_sindy_cpu_stlsq_run(&run) != SECANT_SINDY_SUCCESS) {
        goto cleanup;
    }

    printf("meta %zu %zu %zu %zu %zu %zu %zu %.9g\n", capacity, num_asts, num_rows, num_targets, num_sweeps,
           max_iterations, cohorts, run.scale_epsilon);
    float_array_print("alphas", alphas, num_sweeps);
    float_array_print("thresholds", thresholds, num_sweeps);
    float_array_print("statistics", statistics, statistics_count);
    float_array_print("target_stats", target_stats, num_targets * 2u);
    float_array_print("coefficients", coefficients, coefficient_count);
    float_array_print("intercepts", intercepts, output_count);
    float_array_print("sse", sse, output_count);
    s32_array_print("solve_info", solve_info, output_count);
    u32_array_print("active_masks", active_masks, output_count);
    s32_array_print("active_counts", active_counts, output_count);
    s32_array_print("iteration_counts", iteration_counts, output_count);
    status = 0;

cleanup:
    free(iteration_counts);
    free(active_counts);
    free(active_masks);
    free(solve_info);
    free(sse);
    free(intercepts);
    free(coefficients);
    free(thresholds);
    free(alphas);
    free(target_stats);
    free(statistics);
    free(targets);
    free(features);
    return status;
}
