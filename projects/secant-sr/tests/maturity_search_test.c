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
#include <string.h>

static int
program_counts_get(
    const SecantAstInstruction* program,
    size_t* dynamic_ret,
    size_t* concrete_ret
) {
    size_t dynamic = 0u;
    size_t concrete = 0u;
    size_t offset = 0u;

    for (;;) {
        const SecantAstInstructionType type = secant_ast_instruction_type_get(program + offset);
        const size_t instruction_size = secant_ast_instruction_size_get(program + offset);

        if (instruction_size == 0u) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
            if (secant_ast_index_get(program + offset) != dynamic) {
                return 0;
            }
            ++dynamic;
        } else if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 ||
                   type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            ++concrete;
        }
        offset += instruction_size;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *dynamic_ret = dynamic;
            *concrete_ret = concrete;
            return 1;
        }
    }
}

static int
cpu_dynamic_sse_get(
    const SecantAstInstruction* ast,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    const float* input,
    size_t num_rows,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_stride,
    size_t num_settings,
    const float* target,
    float* output
) {
    const SecantAstInstruction* asts[] = {ast};
    SecantCpuDynamicLeafSSERun run = secant_cpu_dynamic_leaf_sse_run_init();

    run.programs.asts.items = asts;
    run.programs.asts.count = 1u;
    run.num_dynamic_leaves = num_dynamic_leaves;
    run.num_input_columns = 2u;
    run.num_static_input_columns = num_static_input_columns;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, 2u * num_rows, num_rows};
    run.leaf_masks = (SecantConstHostSpanU32){leaf_masks, num_settings};
    run.leaf_words = (SecantConstHostMatrixU32){leaf_words, num_settings * leaf_stride, leaf_stride};
    run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
    run.num_rows = num_rows;
    run.num_settings = num_settings;
    run.output = (SecantHostMatrixF32){output, num_settings, num_settings};
    return secant_cpu_run_dynamic_leaf_sse(&run) == SECANT_SUCCESS;
}

static int
cpu_static_sse_get(
    const SecantAstInstruction* ast,
    const float* input,
    size_t num_rows,
    const float* target,
    float* output
) {
    const SecantAstInstruction* asts[] = {ast};
    SecantCpuSSERun run = secant_cpu_sse_run_init();

    run.programs.asts.items = asts;
    run.programs.asts.count = 1u;
    run.num_inputs = 2u;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, 2u * num_rows, num_rows};
    run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
    run.num_rows = num_rows;
    run.output = (SecantHostMatrixF32){output, 1u, 1u};
    return secant_cpu_run_sse(&run) == SECANT_SUCCESS;
}

static int
maturity_search_test_run(void) {
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32
    };
    static const SecantAstInstruction source_program[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
        secant_ast_encode_add_f32,
        secant_ast_encode_static_column_input_f32(1),
        secant_ast_encode_mul_f32,
        secant_ast_encode_return_f32
    };
    SecantSRSearchConfig config = {0};
    SecantSRSearch search = NULL;
    SecantSRPopulation* population;
    uint32_t selected[] = {0u};
    const uint32_t leaf_masks[] = {UINT32_C(5), 0u};
    uint32_t leaf_words[2u * SECANT_AST_MAX_DYNAMIC_LEAVES] = {0};
    uint32_t best_setting[] = {0u};
    SecantAstInstruction projected[64];
    SecantAstInstruction proposal[64];
    const SecantAstInstruction* projected_asts[1];
    const SecantAstInstruction* proposal_asts[1];
    size_t proposal_sizes[1];
    float input[2u * 17u];
    float target[17u];
    float dynamic_sse[2] = {0.0f, 0.0f};
    float static_sse = 0.0f;
    float source_sse[] = {100.0f, 100.0f};
    float promoted_sse[] = {1.0f};
    void* storage = NULL;
    size_t storage_size = 0u;
    size_t required_size;
    size_t max_dynamic;
    size_t dynamic_count;
    size_t concrete_count;
    size_t row;
    int success = 0;

    config.population_size = 2u;
    config.max_program_bytes = 64u;
    config.max_nodes = 16u;
    config.max_depth = 8u;
    config.max_complexity = 32u;
    config.num_inputs = 2u;
    config.num_complexity_buckets = 8u;
    config.elites_per_bucket = 2u;
    config.tournament_size = 1u;
    config.complexity_factor = 1.5;
    config.point_mutation_probability = 1.0;
    config.seed = UINT64_C(0x6d61747572697479);
    if (secant_sr_search_storage_size(&config, 0u, 2u, 0u, 0u, &storage_size) != SECANT_SR_SUCCESS) {
        return 0;
    }
    storage = malloc(storage_size);
    if (storage == NULL || secant_sr_search_init(
            &config, NULL, 0u, binary_ops, 2u, NULL, 0u, NULL, 0u, storage, storage_size, &search) !=
        SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    population = &search->populations[search->current_population];
    population->program_used = 0u;
    population->node_used = 0u;
    population->count = 0u;
    if (secant_sr_population_append_program(
            search,
            population,
            source_program,
            sizeof(source_program),
            SECANT_SR_ORIGIN_RANDOM,
            SECANT_SR_PARENT_NONE,
            SECANT_SR_PARENT_NONE,
            NULL) != SECANT_SR_SUCCESS ||
        secant_sr_population_append_program(
            search,
            population,
            source_program,
            sizeof(source_program),
            SECANT_SR_ORIGIN_ELITE,
            SECANT_SR_PARENT_NONE,
            SECANT_SR_PARENT_NONE,
            NULL) != SECANT_SR_SUCCESS ||
        secant_sr_search_scores_set(search, source_sse, 2u, 17u, 10.0) != SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    for (row = 0u; row < 17u; ++row) {
        input[row] = -1.0f + 0.125f * (float)row;
        input[17u + row] = 0.25f + 0.0625f * (float)row;
        target[row] = input[row] + input[17u + row];
    }
    leaf_words[0] = 0u;
    leaf_words[1] = SECANT_F32_BITS_HALF;
    leaf_words[2] = 1u;
    if (secant_sr_search_dynamic_leaf_selected_programs_write(
            search,
            selected,
            1u,
            SECANT_SR_DYNAMIC_LEAF_PROJECTION_FULL,
            3u,
            UINT64_C(7),
            projected,
            sizeof(projected),
            projected_asts,
            1u,
            &required_size,
            &max_dynamic) != SECANT_SR_SUCCESS ||
        max_dynamic != 3u || !program_counts_get(projected_asts[0], &dynamic_count, &concrete_count) ||
        dynamic_count != 3u || concrete_count != 0u ||
        !cpu_dynamic_sse_get(
            projected_asts[0], 0u, 3u, input, 17u, leaf_masks, leaf_words,
            SECANT_AST_MAX_DYNAMIC_LEAVES, 2u, target, dynamic_sse) ||
        secant_sr_search_dynamic_leaf_selected_proposals_write(
            search,
            selected,
            1u,
            SECANT_SR_DYNAMIC_LEAF_PROJECTION_FULL,
            3u,
            UINT64_C(7),
            leaf_masks,
            2u,
            leaf_words,
            sizeof(leaf_words) / sizeof(leaf_words[0]),
            SECANT_AST_MAX_DYNAMIC_LEAVES,
            2u,
            best_setting,
            proposal,
            sizeof(proposal),
            proposal_asts,
            proposal_sizes,
            1u,
            &required_size) != SECANT_SR_SUCCESS ||
        !cpu_static_sse_get(proposal_asts[0], input, 17u, target, &static_sse) ||
        fabsf(static_sse - dynamic_sse[0]) > 1.0e-4f * (1.0f + fabsf(static_sse)) ||
        secant_sr_search_selected_proposals_apply(
            search,
            selected,
            proposal_asts,
            proposal_sizes,
            promoted_sse,
            NULL,
            1u,
            17u,
            10.0,
            SECANT_SR_ORIGIN_DYNAMIC_LEAF,
            NULL) != SECANT_SR_SUCCESS ||
        search->populations[search->current_population].individuals[0].maturity != SECANT_SR_MATURITY_RESOLVED ||
        search->populations[search->current_population].individuals[0].dynamic_leaf_refinements != 1u) {
        goto cleanup;
    }

    selected[0] = 1u;
    best_setting[0] = 1u;
    promoted_sse[0] = 0.5f;
    dynamic_sse[0] = 0.0f;
    dynamic_sse[1] = 0.0f;
    static_sse = 0.0f;
    if (secant_sr_search_dynamic_leaf_selected_programs_write(
            search,
            selected,
            1u,
            SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED,
            1u,
            UINT64_C(11),
            projected,
            sizeof(projected),
            projected_asts,
            1u,
            &required_size,
            &max_dynamic) != SECANT_SR_SUCCESS ||
        max_dynamic != 1u || !program_counts_get(projected_asts[0], &dynamic_count, &concrete_count) ||
        dynamic_count != 1u || concrete_count != 2u ||
        !cpu_dynamic_sse_get(
            projected_asts[0], 2u, 1u, input, 17u, leaf_masks, leaf_words,
            SECANT_AST_MAX_DYNAMIC_LEAVES, 2u, target, dynamic_sse) ||
        secant_sr_search_dynamic_leaf_selected_proposals_write(
            search,
            selected,
            1u,
            SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED,
            1u,
            UINT64_C(11),
            leaf_masks,
            2u,
            leaf_words,
            sizeof(leaf_words) / sizeof(leaf_words[0]),
            SECANT_AST_MAX_DYNAMIC_LEAVES,
            2u,
            best_setting,
            proposal,
            sizeof(proposal),
            proposal_asts,
            proposal_sizes,
            1u,
            &required_size) != SECANT_SR_SUCCESS ||
        !cpu_static_sse_get(proposal_asts[0], input, 17u, target, &static_sse) ||
        fabsf(static_sse - dynamic_sse[1]) > 1.0e-4f * (1.0f + fabsf(static_sse)) ||
        secant_sr_search_selected_proposals_apply(
            search,
            selected,
            proposal_asts,
            proposal_sizes,
            promoted_sse,
            NULL,
            1u,
            17u,
            10.0,
            SECANT_SR_ORIGIN_DYNAMIC_LEAF,
            NULL) != SECANT_SR_SUCCESS ||
        search->populations[search->current_population].individuals[1].maturity !=
            SECANT_SR_MATURITY_MIXED_REFINED ||
        search->populations[search->current_population].individuals[1].dynamic_leaf_refinements != 1u) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(storage);
    return success;
}

int
main(void) {
    return maturity_search_test_run() ? 0 : 1;
}
