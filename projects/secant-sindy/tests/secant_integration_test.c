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
#include "secant.h"
#include "secant_sindy.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    GRID_X = 64,
    GRID_T = 64,
    ROWS = GRID_X * GRID_T,
    INPUTS = 3,
    FEATURES = 8,
    TARGETS = 1,
    STATS = FEATURES + FEATURES * FEATURES + FEATURES * TARGETS
};

static const SecantAstInstruction ast_one[] = {
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_u[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_ux[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_uxx[] = {
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_u_squared[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_u_ux[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_u_uxx[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction ast_ux_squared[] = {
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const pde_asts[FEATURES] = {
    ast_one,
    ast_u,
    ast_ux,
    ast_uxx,
    ast_u_squared,
    ast_u_ux,
    ast_u_uxx,
    ast_ux_squared
};

static int close_enough(float actual, float expected, float tolerance, const char* label) {
    if (isfinite(actual) && fabsf(actual - expected) <= tolerance * (1.0f + fabsf(expected))) {
        return 1;
    }
    fprintf(stderr, "%s mismatch: actual=%.9g expected=%.9g\n", label, actual, expected);
    return 0;
}

int main(void) {
    const double pi = 3.1415926535897932384626433832795;
    const double advection_speed = 0.7;
    const double diffusivity = 0.08;
    const double amplitudes[4] = {1.0, 0.45, -0.3, 0.2};
    const double phases[4] = {0.1, -0.35, 0.8, -1.1};
    const double wavenumbers[4] = {1.0, 2.0, 3.0, 5.0};
    const float expected_coefficients[FEATURES] = {0.0f, 0.0f, -0.7f, 0.08f, 0.0f, 0.0f, 0.0f, 0.0f};
    float* input = (float*)malloc(INPUTS * ROWS * sizeof(*input));
    float* target = (float*)malloc(ROWS * sizeof(*target));
    float statistics[STATS] = {0};
    float target_stats[2] = {0};
    float alpha = 0.0f;
    float threshold = 0.02f;
    float coefficients[FEATURES] = {0};
    float intercept = 0.0f;
    float sse = 0.0f;
    int32_t solve_info = 0;
    uint32_t active_mask = 0u;
    int32_t active_count = 0;
    int32_t iteration_count = 0;
    SecantCpuGramStatsRun gram_run = secant_cpu_gram_stats_run_init();
    SecantSindyCpuTargetStatsRun target_run = secant_sindy_cpu_target_stats_run_init();
    SecantSindyCpuSTLSQRun solve_run = secant_sindy_cpu_stlsq_run_init();
    size_t time_idx;
    size_t feature;
    int ok = input != NULL && target != NULL;

    for (time_idx = 0u; time_idx < GRID_T && ok; ++time_idx) {
        const double time = 1.25 * (double)time_idx / (double)(GRID_T - 1u);
        size_t space_idx;

        for (space_idx = 0u; space_idx < GRID_X; ++space_idx) {
            const double space = -pi + 2.0 * pi * (double)space_idx / (double)GRID_X;
            const size_t row = time_idx * GRID_X + space_idx;
            double u = 0.0;
            double ux = 0.0;
            double uxx = 0.0;
            double ut = 0.0;
            size_t mode;

            for (mode = 0u; mode < 4u; ++mode) {
                const double k = wavenumbers[mode];
                const double angle = k * (space - advection_speed * time) + phases[mode];
                const double decay = exp(-diffusivity * k * k * time);
                const double sine = sin(angle);
                const double cosine = cos(angle);

                u += amplitudes[mode] * sine * decay;
                ux += amplitudes[mode] * k * cosine * decay;
                uxx -= amplitudes[mode] * k * k * sine * decay;
                ut += amplitudes[mode] * (-advection_speed * k * cosine - diffusivity * k * k * sine) * decay;
            }
            input[row] = (float)u;
            input[ROWS + row] = (float)ux;
            input[2u * ROWS + row] = (float)uxx;
            target[row] = (float)ut;
        }
    }

    gram_run.programs.asts.items = pde_asts;
    gram_run.programs.asts.count = FEATURES;
    gram_run.asts_per_cohort = FEATURES;
    gram_run.num_inputs = INPUTS;
    gram_run.num_targets = TARGETS;
    gram_run.input = (SecantConstHostMatrixF32){input, INPUTS * ROWS, ROWS};
    gram_run.targets = (SecantConstHostMatrixF32){target, ROWS, ROWS};
    gram_run.num_rows = ROWS;
    gram_run.statistics = (SecantHostMatrixF32){statistics, STATS, STATS};
    if (ok && secant_cpu_run_gram_stats(&gram_run) != SECANT_SUCCESS) {
        fprintf(stderr, "Secant Gram-statistics evaluation failed\n");
        ok = 0;
    }

    target_run.targets = (SecantSindyConstHostSpanF32){target, ROWS};
    target_run.targets_leading_dimension = ROWS;
    target_run.num_rows = ROWS;
    target_run.num_targets = TARGETS;
    target_run.target_stats = (SecantSindyHostSpanF32){target_stats, 2u};
    target_run.target_stats_leading_dimension = 2u;
    if (ok && secant_sindy_cpu_target_stats_run(&target_run) != SECANT_SINDY_SUCCESS) {
        fprintf(stderr, "Secant-SINDy target-statistics evaluation failed\n");
        ok = 0;
    }

    solve_run.feature_capacity = FEATURES;
    solve_run.num_asts = FEATURES;
    solve_run.num_rows = ROWS;
    solve_run.num_targets = TARGETS;
    solve_run.num_sweeps = 1u;
    solve_run.max_iterations = FEATURES;
    solve_run.statistics = (SecantSindyConstHostSpanF32){statistics, STATS};
    solve_run.statistics_leading_dimension = STATS;
    solve_run.target_stats = (SecantSindyConstHostSpanF32){target_stats, 2u};
    solve_run.target_stats_leading_dimension = 2u;
    solve_run.alphas = (SecantSindyConstHostSpanF32){&alpha, 1u};
    solve_run.thresholds = (SecantSindyConstHostSpanF32){&threshold, 1u};
    solve_run.coefficients = (SecantSindyHostSpanF32){coefficients, FEATURES};
    solve_run.intercepts = (SecantSindyHostSpanF32){&intercept, 1u};
    solve_run.sse = (SecantSindyHostSpanF32){&sse, 1u};
    solve_run.solve_info = (SecantSindyHostSpanS32){&solve_info, 1u};
    solve_run.active_masks = (SecantSindyHostSpanU32){&active_mask, 1u};
    solve_run.active_counts = (SecantSindyHostSpanS32){&active_count, 1u};
    solve_run.iteration_counts = (SecantSindyHostSpanS32){&iteration_count, 1u};
    if (ok && secant_sindy_cpu_stlsq_run(&solve_run) != SECANT_SINDY_SUCCESS) {
        fprintf(stderr, "Secant-SINDy STLSQ solve failed\n");
        ok = 0;
    }

    if (ok && (solve_info != 0 || active_mask != 0xcu || active_count != 2 || iteration_count < 2)) {
        fprintf(stderr, "PDE active set mismatch: mask=0x%x count=%d iterations=%d info=%d\n",
                active_mask, active_count, iteration_count, solve_info);
        ok = 0;
    }
    for (feature = 0u; feature < FEATURES && ok; ++feature) {
        ok &= close_enough(coefficients[feature], expected_coefficients[feature], 7.5e-4f, "PDE coefficient");
    }
    if (ok) {
        ok &= close_enough(intercept, 0.0f, 7.5e-4f, "PDE intercept");
        if (!isfinite(sse) || sse / (float)ROWS > 1.0e-6f) {
            fprintf(stderr, "PDE MSE too large: %.9g\n", sse / (float)ROWS);
            ok = 0;
        }
    }

    free(target);
    free(input);
    if (!ok) {
        return 1;
    }
    printf("Secant Gram statistics recovered the advection-diffusion PDE through Secant-SINDy STLSQ\n");
    return 0;
}
