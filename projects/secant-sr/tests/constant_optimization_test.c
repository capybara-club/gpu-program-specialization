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
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double
constant_optimization_target_fill(float* input, float* target, size_t num_rows) {
    double mean = 0.0;
    double target_ssd = 0.0;
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        input[row] = -3.0f + 6.0f * (float)row / (float)(num_rows - 1u);
        target[row] = 2.75f * input[row] - 0.375f;
        mean += target[row];
    }
    mean /= (double)num_rows;
    for (row = 0u; row < num_rows; ++row) {
        const double difference = (double)target[row] - mean;

        target_ssd += difference * difference;
    }
    return target_ssd;
}

static int
constant_optimization_test_run(void) {
    static const SecantAstInstructionType binary_ops[] = {SECANT_AST_INSTRUCTION_TYPE_ADD_F32};
    static const float seed_constants[] = {1.0f};
    static const SecantAstInstruction program[] = {
        secant_ast_encode_static_column_input_f32(0u),
        secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_HALF),
        secant_ast_encode_mul_f32,
        secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_TWO),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    const SecantSRSearchConfig config = {
        1u,
        sizeof(program),
        5u,
        4u,
        8u,
        1u,
        4u,
        1u,
        1u,
        1u,
        1.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        UINT64_C(0x123456789abcdef),
        0u
    };
    const size_t num_rows = 257u;
    SecantSRSearch search = NULL;
    void* storage = NULL;
    float input[257];
    float target[257];
    float sse[1];
    size_t storage_size;
    size_t num_constants = 0u;
    int improved = 0;
    size_t num_f_calls = 0u;
    double target_ssd;
    const SecantSRIndividual* best = NULL;
    SecantSRResult result;
    int status = 1;

    result = secant_sr_search_storage_size(
        &config,
        0u,
        sizeof(binary_ops) / sizeof(binary_ops[0]),
        0u,
        sizeof(seed_constants) / sizeof(seed_constants[0]),
        &storage_size);
    if (result != SECANT_SR_SUCCESS) {
        fprintf(stderr, "storage measure failed: %s\n", secant_sr_result_to_string(result));
        return 1;
    }
    storage = malloc(storage_size);
    if (storage == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    result = secant_sr_search_init(
        &config,
        NULL,
        0u,
        binary_ops,
        sizeof(binary_ops) / sizeof(binary_ops[0]),
        NULL,
        0u,
        seed_constants,
        sizeof(seed_constants) / sizeof(seed_constants[0]),
        storage,
        storage_size,
        &search);
    if (result == SECANT_SR_SUCCESS) {
        SecantSRPopulation* population = &search->populations[search->current_population];

        population->program_used = 0u;
        population->node_used = 0u;
        population->count = 0u;
        result = secant_sr_population_append_program(
            search,
            population,
            program,
            sizeof(program),
            SECANT_SR_ORIGIN_RANDOM,
            SECANT_SR_PARENT_NONE,
            SECANT_SR_PARENT_NONE,
            NULL);
    }
    target_ssd = constant_optimization_target_fill(input, target, num_rows);
    if (result == SECANT_SR_SUCCESS) {
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
            1u);
    }
    if (result == SECANT_SR_SUCCESS) {
        result = secant_sr_search_best_constants_optimize_cpu(
            search,
            NULL,
            0u,
            input,
            num_rows,
            num_rows,
            target,
            num_rows,
            33u,
            num_rows,
            target_ssd,
            16u,
            1u,
            1000u,
            1.0e-3,
            &num_constants,
            &improved,
            &num_f_calls);
    }
    if (result == SECANT_SR_SUCCESS) {
        result = secant_sr_search_best_get(search, &best);
    }
    if (result == SECANT_SR_SUCCESS && num_constants == 2u && improved && num_f_calls > 0u &&
        best != NULL && best->fitness.r2 > 0.999999) {
        status = 0;
    } else {
        fprintf(
            stderr,
            "constant optimization failed: result=%s constants=%zu improved=%d f_calls=%zu r2=%.12g\n",
            secant_sr_result_to_string(result),
            num_constants,
            improved,
            num_f_calls,
            best != NULL ? best->fitness.r2 : -INFINITY);
    }
    free(storage);
    return status;
}

int
main(void) {
    return constant_optimization_test_run();
}
