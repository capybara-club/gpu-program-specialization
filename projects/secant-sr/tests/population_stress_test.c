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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
secant_sr_population_stress_run(void) {
    static const SecantAstInstructionType unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32,
        SECANT_AST_INSTRUCTION_TYPE_ABS_F32,
        SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_COS_F32
    };
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32,
        SECANT_AST_INSTRUCTION_TYPE_MIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_MAX_F32
    };
    static const float constants[] = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
    const SecantSRSearchConfig config = {
        32768u, 128u, 32u, 16u, 32u, 8u, 12u, 4u, 32u, 5u, 1.5,
        0.45, 0.30, 0.20, 0.15, 0.00005, UINT64_C(0xa51deca7), 0u
    };
    SecantSRSearch search = NULL;
    void* storage = NULL;
    float* sse = NULL;
    unsigned char parent_copy[128];
    size_t parent_bytes;
    const SecantAstInstruction* parent_program;
    size_t storage_size = 0u;
    size_t generation;
    size_t origin_counts[5] = {0u, 0u, 0u, 0u, 0u};
    int valid = 1;
    int status = 1;

    if (secant_sr_search_storage_size(
            &config,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            0u,
            sizeof(constants) / sizeof(constants[0]),
            &storage_size) != SECANT_SR_SUCCESS) {
        return 1;
    }
    storage = malloc(storage_size);
    sse = malloc(config.population_size * sizeof(*sse));
    if (storage == NULL || sse == NULL ||
        secant_sr_search_init(
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
            &search) != SECANT_SR_SUCCESS) {
        free(sse);
        free(storage);
        return 1;
    }

    for (generation = 0u; generation < 20u; ++generation) {
        const SecantSRIndividual* individuals;
        size_t num_individuals;
        size_t individual_idx;

        if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS ||
            num_individuals != config.population_size) {
            break;
        }
        for (individual_idx = 0u; individual_idx < num_individuals; ++individual_idx) {
            const SecantSRIndividual* individual = individuals + individual_idx;
            const SecantSRNodeInfo* root = individual->nodes + individual->num_nodes - 1u;

            if (individual->program_bytes > config.max_program_bytes || individual->num_nodes == 0u ||
                individual->num_nodes > config.max_nodes || individual->depth > config.max_depth ||
                individual->complexity > config.max_complexity || root->subtree_offset != 0u ||
                root->subtree_bytes + 1u != individual->program_bytes ||
                root->subtree_nodes != individual->num_nodes || root->subtree_first_node != 0u) {
                fprintf(stderr, "invalid individual generation=%zu index=%zu\n", generation, individual_idx);
                valid = 0;
                break;
            }
            if ((size_t)individual->origin < sizeof(origin_counts) / sizeof(origin_counts[0])) {
                ++origin_counts[individual->origin];
            }
            sse[individual_idx] = (float)(individual->complexity +
                (individual->fingerprint & UINT64_C(0xffff)) * (1.0 / 65536.0));
        }
        if (!valid) {
            break;
        }
        parent_program = individuals[0].program;
        parent_bytes = individuals[0].program_bytes;
        memcpy(parent_copy, parent_program, parent_bytes);
        if (secant_sr_search_scores_set(search, sse, num_individuals, 1024u, 1024.0) != SECANT_SR_SUCCESS ||
            secant_sr_search_generation_advance(search) != SECANT_SR_SUCCESS ||
            memcmp(parent_copy, parent_program, parent_bytes) != 0) {
            fprintf(stderr, "generation transition failed at %zu\n", generation);
            valid = 0;
            break;
        }
    }
    if (valid && generation == 20u && origin_counts[SECANT_SR_ORIGIN_CROSSOVER] != 0u &&
        origin_counts[SECANT_SR_ORIGIN_SUBTREE_MUTATION] != 0u &&
        origin_counts[SECANT_SR_ORIGIN_POINT_MUTATION] != 0u && origin_counts[SECANT_SR_ORIGIN_ELITE] != 0u) {
        status = 0;
    }
    free(sse);
    free(storage);
    return status;
}

int
main(void) {
    return secant_sr_population_stress_run();
}
