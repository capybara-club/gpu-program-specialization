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

#include <stdio.h>
#include <stdlib.h>

static int
quality_diversity_test_run(void) {
    static const SecantAstInstructionType unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_COS_F32
    };
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32
    };
    static const SecantAstInstruction x0[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction x0_plus_x1[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_static_column_input_f32(1),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction sin_x0[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_sin_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction cos_sin_x0[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_sin_f32,
        secant_ast_encode_cos_f32,
        secant_ast_encode_return_f32
    };
    static const SecantAstInstruction* const programs[] = {x0, x0_plus_x1, sin_x0, cos_sin_x0};
    static const size_t program_sizes[] = {sizeof(x0), sizeof(x0_plus_x1), sizeof(sin_x0), sizeof(cos_sin_x0)};
    SecantSRSearchConfig config = {0};
    SecantSRSearch search = NULL;
    SecantSRPopulation* population;
    SecantSRArchiveStats stats;
    const SecantSRIndividual* individuals;
    SecantSRProgramFeatures features[4];
    float sse[4] = {4.0f, 3.0f, 2.0f, 1.0f};
    void* storage = NULL;
    size_t storage_size = 0u;
    size_t idx;
    int status = 1;

    config.population_size = 4u;
    config.max_program_bytes = 32u;
    config.max_nodes = 8u;
    config.max_depth = 8u;
    config.max_complexity = 16u;
    config.num_inputs = 3u;
    config.num_complexity_buckets = 8u;
    config.elites_per_bucket = 1u;
    config.tournament_size = 1u;
    config.complexity_factor = 1.5;
    config.point_mutation_probability = 1.0;
    config.seed = UINT64_C(0x71645f74657374);
    config.num_column_count_buckets = 4u;
    config.num_transcendental_count_buckets = 3u;
    config.archive_parent_probability = 1.0;

    if (secant_sr_search_storage_size(
            &config,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            0u,
            0u,
            &storage_size) != SECANT_SR_SUCCESS) {
        return 1;
    }
    storage = malloc(storage_size);
    if (storage == NULL || secant_sr_search_init(
            &config,
            unary_ops,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            binary_ops,
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            NULL,
            0u,
            NULL,
            0u,
            storage,
            storage_size,
            &search) != SECANT_SR_SUCCESS) {
        free(storage);
        return 1;
    }

    population = &search->populations[search->current_population];
    population->program_used = 0u;
    population->node_used = 0u;
    population->count = 0u;
    for (idx = 0u; idx < sizeof(programs) / sizeof(programs[0]); ++idx) {
        if (secant_sr_population_append_program(
                search,
                population,
                programs[idx],
                program_sizes[idx],
                SECANT_SR_ORIGIN_RANDOM,
                SECANT_SR_PARENT_NONE,
                SECANT_SR_PARENT_NONE,
                NULL) != SECANT_SR_SUCCESS) {
            goto cleanup;
        }
        if (secant_sr_program_features_get(programs[idx], program_sizes[idx], NULL, 0u, features + idx) !=
            SECANT_SR_SUCCESS) {
            goto cleanup;
        }
    }
    individuals = population->individuals;
    if (features[0].num_unique_static_columns != 1u ||
        features[1].num_unique_static_columns != 2u ||
        features[2].num_transcendental_operations != 1u ||
        features[3].num_transcendental_operations != 2u ||
        features[1].num_binary_operations != 1u ||
        features[3].num_unary_operations != 2u) {
        fprintf(stderr, "unexpected structural descriptors\n");
        goto cleanup;
    }
    if (secant_sr_search_scores_set(search, sse, 4u, 16u, 16.0) != SECANT_SR_SUCCESS ||
        secant_sr_search_archive_stats_get(search, &stats) != SECANT_SR_SUCCESS ||
        stats.num_cells != 96u || stats.occupied_cells != 4u || stats.num_elites != 4u) {
        fprintf(stderr, "unexpected QD archive occupancy\n");
        goto cleanup;
    }
    if (secant_sr_search_archive_parent_probability_set(search, -0.1) != SECANT_SR_ERROR_INVALID_VALUE ||
        secant_sr_search_archive_parent_probability_set(search, 0.0) != SECANT_SR_SUCCESS ||
        search->config.archive_parent_probability != 0.0 ||
        secant_sr_search_archive_parent_probability_set(search, 1.0) != SECANT_SR_SUCCESS ||
        search->config.archive_parent_probability != 1.0) {
        fprintf(stderr, "archive-parent probability update failed\n");
        goto cleanup;
    }
    if (secant_sr_search_generation_advance(search) != SECANT_SR_SUCCESS ||
        secant_sr_search_individuals_get(search, &individuals, &idx) != SECANT_SR_SUCCESS || idx != 4u) {
        fprintf(stderr, "archive-parent generation failed\n");
        goto cleanup;
    }
    for (idx = 0u; idx < config.population_size; ++idx) {
        if (individuals[idx].origin != SECANT_SR_ORIGIN_RANDOM && individuals[idx].parent_a != SECANT_SR_PARENT_NONE) {
            fprintf(stderr, "archive parent leaked a population index\n");
            goto cleanup;
        }
    }
    status = 0;

cleanup:
    free(storage);
    return status;
}

int
main(void) {
    return quality_diversity_test_run();
}
