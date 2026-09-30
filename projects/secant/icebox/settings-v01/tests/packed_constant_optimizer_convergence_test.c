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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_MAX_CONSTANTS 3u
#define TEST_ROWS 97u
#define TEST_SETTINGS 1024u
#define TEST_ITERATIONS 20u
#define TEST_SEEDS 8u
#define TEST_STATE_WIDTH (1u + 4u * TEST_MAX_CONSTANTS)

typedef float (*TestTargetFunction)(float x);

typedef struct TestCase {
    const char* name;
    const SecantAstInstruction* ast;
    size_t num_constants;
    float initial_constants[TEST_MAX_CONSTANTS];
    float expected_constants[TEST_MAX_CONSTANTS];
    float initial_scale;
    TestTargetFunction target;
} TestCase;

typedef struct TestPolicy {
    const char* name;
    SecantConstantOptimizerUpdateMode update_mode;
    size_t num_elites;
    float momentum;
    float scale_learning_rate;
    float scale_failure_decay;
} TestPolicy;

typedef struct TestResult {
    float constants[TEST_MAX_CONSTANTS];
    float scales[TEST_MAX_CONSTANTS];
    float velocities[TEST_MAX_CONSTANTS];
    float initial_sse;
    float final_sse;
} TestResult;

static const SecantAstInstruction test_affine_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_polynomial_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_dynamic_constant_input_f32(2u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sine_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_sin_f32,
    secant_ast_encode_dynamic_constant_input_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static float
test_affine_target(float x) {
    return 96.0f * x + 0.015625f;
}

static float
test_polynomial_target(float x) {
    return 0.03125f * x * x - 18.0f * x + 240.0f;
}

static float
test_sine_target(float x) {
    return sinf(2.75f * x) - 0.125f;
}

static float
test_initial_prediction(const TestCase* test_case, float x) {
    if (test_case->ast == test_affine_ast) {
        return test_case->initial_constants[0] * x + test_case->initial_constants[1];
    }
    if (test_case->ast == test_polynomial_ast) {
        return test_case->initial_constants[0] * x * x + test_case->initial_constants[1] * x +
            test_case->initial_constants[2];
    }
    return sinf(test_case->initial_constants[0] * x) + test_case->initial_constants[1];
}

static int
test_case_run(const TestCase* test_case, const TestPolicy* policy, uint64_t seed, TestResult* result_ret) {
    const SecantAstInstruction* asts[] = {test_case->ast};
    SecantCpuPackedConstantOptimizerSSERun run = secant_cpu_packed_constant_optimizer_sse_run_init();
    float input[TEST_ROWS];
    float target[TEST_ROWS];
    float constants[TEST_MAX_CONSTANTS] = {0.0f, 0.0f, 0.0f};
    float scales[TEST_MAX_CONSTANTS] = {0.0f, 0.0f, 0.0f};
    float velocities[TEST_MAX_CONSTANTS] = {0.0f, 0.0f, 0.0f};
    float current_sse = INFINITY;
    float sse[TEST_SETTINGS];
    float best[TEST_STATE_WIDTH];
    size_t row;
    size_t constant_idx;

    memset(result_ret, 0, sizeof(*result_ret));
    for (constant_idx = 0u; constant_idx < test_case->num_constants; ++constant_idx) {
        constants[constant_idx] = test_case->initial_constants[constant_idx];
        scales[constant_idx] = test_case->initial_scale;
    }
    for (row = 0u; row < TEST_ROWS; ++row) {
        const float x = -2.0f + 4.0f * (float)row / (float)(TEST_ROWS - 1u);
        const float error = test_initial_prediction(test_case, x) - test_case->target(x);

        input[row] = x;
        target[row] = test_case->target(x);
        result_ret->initial_sse += error * error;
    }

    run.programs.asts.items = asts;
    run.programs.asts.count = 1u;
    run.programs.current_constants.data = constants;
    run.programs.current_constants.num_elements = test_case->num_constants;
    run.programs.current_constants.leading_dimension = test_case->num_constants;
    run.programs.current_constant_scales.data = scales;
    run.programs.current_constant_scales.num_elements = test_case->num_constants;
    run.programs.current_constant_scales.leading_dimension = test_case->num_constants;
    run.num_input_columns = 1u;
    run.num_input_constants = test_case->num_constants;
    run.input.data = input;
    run.input.num_elements = TEST_ROWS;
    run.input.leading_dimension = TEST_ROWS;
    run.target.data = target;
    run.target.num_elements = TEST_ROWS;
    run.target.leading_dimension = TEST_ROWS;
    run.num_rows = TEST_ROWS;
    run.num_settings = TEST_SETTINGS;
    run.num_iterations = TEST_ITERATIONS;
    run.seed = seed;
    run.generation = 23u;
    run.iteration = 0u;
    run.update_mode = policy->update_mode;
    run.num_elites = policy->num_elites;
    run.current_constant_velocities.data = velocities;
    run.current_constant_velocities.num_elements = test_case->num_constants;
    run.current_constant_velocities.leading_dimension = test_case->num_constants;
    run.current_sse.data = &current_sse;
    run.current_sse.num_elements = 1u;
    run.momentum = policy->momentum;
    run.scale_learning_rate = policy->scale_learning_rate;
    run.scale_failure_decay = policy->scale_failure_decay;
    run.minimum_scale = 1.0e-7f;
    run.maximum_scale = 1024.0f;
    run.sse.data = sse;
    run.sse.num_elements = TEST_SETTINGS;
    run.sse.leading_dimension = TEST_SETTINGS;
    run.best.data = best;
    run.best.num_elements = 1u + 4u * test_case->num_constants;
    run.best.leading_dimension = 1u + 4u * test_case->num_constants;

    if (secant_cpu_run_packed_constant_optimizer_sse(&run) != SECANT_SUCCESS || !isfinite(current_sse)) {
        return 0;
    }
    result_ret->final_sse = current_sse;
    for (constant_idx = 0u; constant_idx < test_case->num_constants; ++constant_idx) {
        result_ret->constants[constant_idx] = constants[constant_idx];
        result_ret->scales[constant_idx] = scales[constant_idx];
        result_ret->velocities[constant_idx] = velocities[constant_idx];
    }
    return 1;
}

int
main(void) {
    static const TestCase cases[] = {
        {
            "affine_mixed_scale", test_affine_ast, 2u,
            {0.0f, 0.0f, 0.0f}, {96.0f, 0.015625f, 0.0f}, 8.0f, test_affine_target
        },
        {
            "polynomial_mixed_scale", test_polynomial_ast, 3u,
            {0.0f, 0.0f, 0.0f}, {0.03125f, -18.0f, 240.0f}, 16.0f, test_polynomial_target
        },
        {
            "sine_local", test_sine_ast, 2u,
            {2.0f, 0.5f, 0.0f}, {2.75f, -0.125f, 0.0f}, 0.5f, test_sine_target
        }
    };
    static const TestPolicy policies[] = {
        {"fixed", SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER, 1u, 0.0f, 0.0f, 1.0f},
        {"adaptive_winner", SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER, 1u, 0.0f, 0.35f, 0.5f},
        {"elite_4", SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION, 4u, 0.0f, 0.35f, 0.5f},
        {"elite_8", SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION, 8u, 0.0f, 0.35f, 0.5f},
        {"elite_16", SECANT_CONSTANT_OPTIMIZER_UPDATE_ELITE_DISTRIBUTION, 16u, 0.0f, 0.35f, 0.5f},
        {"winner_momentum_0.8", SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER, 1u, 0.8f, 0.35f, 0.5f}
    };
    size_t case_idx;
    int success = 1;

    printf("case,policy,geomean_rmse,worst_rmse,seed0_c0,seed0_c1,seed0_c2,seed0_scale0,seed0_scale1,"
        "seed0_scale2\n");
    for (case_idx = 0u; case_idx < sizeof(cases) / sizeof(cases[0]); ++case_idx) {
        double geomean_rmse[sizeof(policies) / sizeof(policies[0])];
        size_t policy_idx;

        for (policy_idx = 0u; policy_idx < sizeof(policies) / sizeof(policies[0]); ++policy_idx) {
            TestResult first_result;
            double log_rmse_sum = 0.0;
            float worst_rmse = 0.0f;
            size_t seed_idx;

            for (seed_idx = 0u; seed_idx < TEST_SEEDS; ++seed_idx) {
                TestResult result;
                const uint64_t seed = UINT64_C(0x8d12f3e419ab6705) +
                    UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)seed_idx;
                float rmse;

                if (!test_case_run(&cases[case_idx], &policies[policy_idx], seed, &result)) {
                    fprintf(stderr, "optimizer execution failed for %s/%s seed=%zu\n", cases[case_idx].name,
                        policies[policy_idx].name, seed_idx);
                    return 1;
                }
                if (seed_idx == 0u) {
                    first_result = result;
                }
                rmse = sqrtf(result.final_sse / (float)TEST_ROWS);
                log_rmse_sum += log((double)rmse + 1.0e-30);
                if (rmse > worst_rmse) {
                    worst_rmse = rmse;
                }
                if (!(result.final_sse < result.initial_sse)) {
                    fprintf(stderr, "optimizer did not improve %s/%s seed=%zu\n", cases[case_idx].name,
                        policies[policy_idx].name, seed_idx);
                    success = 0;
                }
            }
            geomean_rmse[policy_idx] = exp(log_rmse_sum / (double)TEST_SEEDS);
            printf("%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                cases[case_idx].name,
                policies[policy_idx].name,
                geomean_rmse[policy_idx], worst_rmse,
                first_result.constants[0], first_result.constants[1], first_result.constants[2],
                first_result.scales[0], first_result.scales[1], first_result.scales[2]);
            if (policy_idx == 0u) {
                size_t constant_idx;

                for (constant_idx = 0u; constant_idx < cases[case_idx].num_constants; ++constant_idx) {
                    if (first_result.scales[constant_idx] != cases[case_idx].initial_scale) {
                        fprintf(stderr, "fixed policy changed a scale for %s\n", cases[case_idx].name);
                        success = 0;
                    }
                }
            }
        }
        if (!(geomean_rmse[1] < 0.1 * geomean_rmse[0])) {
            fprintf(stderr, "adaptive winner did not beat fixed radius for %s\n", cases[case_idx].name);
            success = 0;
        }
        if (!(geomean_rmse[2] < 0.1 * geomean_rmse[0])) {
            fprintf(stderr, "elite distribution did not beat fixed radius for %s\n", cases[case_idx].name);
            success = 0;
        }
    }
    return success ? 0 : 1;
}
