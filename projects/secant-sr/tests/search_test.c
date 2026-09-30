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
#include "secant_sr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double
target_build(float* input, float* target, size_t num_rows) {
    double mean = 0.0;
    double sum_squared_deviation = 0.0;
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        const float x = -1.0f + 2.0f * (float)row / (float)(num_rows - 1u);

        input[row] = x;
        target[row] = x * x * x + x * x + x;
        mean += target[row];
    }
    mean /= (double)num_rows;
    for (row = 0u; row < num_rows; ++row) {
        const double difference = (double)target[row] - mean;

        sum_squared_deviation += difference * difference;
    }
    return sum_squared_deviation;
}

static int
search_test_run(void) {
    static const SecantAstInstructionType unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32
    };
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32
    };
    static const float constants[] = {-1.0f, 0.5f, 1.0f, 2.0f};
    const SecantSRSearchConfig config = {
        4096u,
        96u,
        48u,
        12u,
        24u,
        1u,
        10u,
        4u,
        16u,
        5u,
        1.55,
        0.45,
        0.30,
        0.20,
        0.18,
        0.00005,
        UINT64_C(0x123456789abcdef),
        5u
    };
    const size_t num_rows = 257u;
    SecantSRSearch search = NULL;
    void* storage = NULL;
    float* input = NULL;
    float* target = NULL;
    float* sse = NULL;
    size_t storage_size = 0u;
    double target_ssd;
    SecantSRResult result;
    int status = 1;
    size_t generation;

    result = secant_sr_search_storage_size(
        &config,
        sizeof(unary_ops) / sizeof(unary_ops[0]),
        sizeof(binary_ops) / sizeof(binary_ops[0]),
        0u,
        sizeof(constants) / sizeof(constants[0]),
        &storage_size);
    if (result != SECANT_SR_SUCCESS) {
        fprintf(stderr, "measure failed: %s\n", secant_sr_result_to_string(result));
        return 1;
    }
    storage = malloc(storage_size);
    input = malloc(num_rows * sizeof(*input));
    target = malloc(num_rows * sizeof(*target));
    sse = malloc(config.population_size * sizeof(*sse));
    if (storage == NULL || input == NULL || target == NULL || sse == NULL) {
        fprintf(stderr, "allocation failed\n");
    } else {
        result = secant_sr_search_init(
            &config,
            unary_ops,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            binary_ops,
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            NULL,
            0u,
            constants,
            sizeof(constants) / sizeof(constants[0]),
            storage,
            storage_size,
            &search);
        if (result != SECANT_SR_SUCCESS) {
            fprintf(stderr, "init failed: %s\n", secant_sr_result_to_string(result));
        } else {
            const SecantSRIndividual* initial_individuals = NULL;
            size_t num_initial_individuals = 0u;
            size_t initial_idx;

            if (secant_sr_search_active_max_nodes_get(search) != 5u ||
                secant_sr_search_individuals_get(search, &initial_individuals, &num_initial_individuals) !=
                    SECANT_SR_SUCCESS) {
                fprintf(stderr, "initial active node limit failed\n");
                goto cleanup;
            }
            for (initial_idx = 0u; initial_idx < num_initial_individuals; ++initial_idx) {
                if (initial_individuals[initial_idx].num_nodes > 5u) {
                    fprintf(stderr, "initial AST exceeds active limit\n");
                    goto cleanup;
                }
            }
            if (secant_sr_search_active_max_nodes_set(search, 4u) != SECANT_SR_ERROR_INVALID_VALUE ||
                secant_sr_search_active_max_nodes_set(search, config.max_nodes + 1u) !=
                    SECANT_SR_ERROR_INVALID_VALUE ||
                secant_sr_search_active_max_nodes_set(search, config.max_nodes) != SECANT_SR_SUCCESS) {
                fprintf(stderr, "active node limit raise contract failed\n");
                goto cleanup;
            }
            target_ssd = target_build(input, target, num_rows);
            for (generation = 0u; generation < 40u; ++generation) {
                const SecantSRIndividual* best;

                result = secant_sr_search_cpu_evaluate(
                    search,
                    NULL,
                    0u,
                    input,
                    num_rows,
                    num_rows,
                    target,
                    num_rows,
                    num_rows,
                    target_ssd,
                    sse,
                    config.population_size);
                if (result != SECANT_SR_SUCCESS) {
                    fprintf(stderr, "evaluate failed: %s\n", secant_sr_result_to_string(result));
                    break;
                }
                result = secant_sr_search_best_get(search, &best);
                if (result != SECANT_SR_SUCCESS) {
                    break;
                }
                if (best->fitness.r2 > 0.999999) {
                    status = 0;
                    break;
                }
                result = secant_sr_search_generation_advance(search);
                if (result != SECANT_SR_SUCCESS) {
                    fprintf(stderr, "advance failed: %s\n", secant_sr_result_to_string(result));
                    break;
                }
            }
            if (status != 0) {
                const SecantSRIndividual* best = NULL;

                if (secant_sr_search_best_get(search, &best) == SECANT_SR_SUCCESS) {
                    fprintf(stderr, "search did not converge: generation=%zu r2=%.9g complexity=%u\n",
                        generation, best->fitness.r2, best->complexity);
                }
            }
        }
    }
cleanup:
    free(sse);
    free(target);
    free(input);
    free(storage);
    return status;
}

int
main(void) {
    return search_test_run();
}
