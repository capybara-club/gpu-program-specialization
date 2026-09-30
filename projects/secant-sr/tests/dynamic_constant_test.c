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
#include "secant_instructions.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_POPULATION 32u
#define TEST_SETTINGS 4u

static int
population_limits_validate(SecantSRSearch search) {
    const SecantSRIndividual* individuals;
    size_t num_individuals;
    size_t ast_idx;
    int saw_more_than_15_nodes = 0;
    int saw_more_than_8_leaves = 0;

    if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS ||
        num_individuals != TEST_POPULATION) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_individuals; ++ast_idx) {
        const SecantSRIndividual* individual = individuals + ast_idx;

        if (individual->num_nodes > 30u || individual->num_leaves == 0u || individual->num_leaves > 16u) {
            return 0;
        }
        if (individual->num_nodes > 15u) {
            saw_more_than_15_nodes = 1;
        }
        if (individual->num_leaves > 8u) {
            saw_more_than_8_leaves = 1;
        }
    }
    return saw_more_than_15_nodes && saw_more_than_8_leaves;
}

static int
projected_program_validate(
    const SecantAstInstruction* program,
    size_t* num_dynamic_constants_ret,
    size_t* num_fixed_constants_ret
) {
    size_t expected_idx = 0u;
    size_t num_fixed_constants = 0u;
    size_t offset = 0u;
    int fixed_constants_started = 0;

    for (;;) {
        const SecantAstInstructionType type = secant_ast_instruction_type_get(program + offset);
        const size_t instruction_size = secant_ast_instruction_size_get(program + offset);

        if (instruction_size == 0u) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32) {
            if (fixed_constants_started || secant_ast_index_get(program + offset) != expected_idx) {
                return 0;
            }
            ++expected_idx;
        } else if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            fixed_constants_started = 1;
            ++num_fixed_constants;
        }
        offset += instruction_size;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_dynamic_constants_ret = expected_idx;
            *num_fixed_constants_ret = num_fixed_constants;
            return expected_idx != 0u;
        }
    }
}

static int
projected_mixed_constants_validate(
    const SecantAstInstruction* program,
    size_t expected_constants
) {
    size_t expected_idx = 0u;
    size_t offset = 0u;

    for (;;) {
        const SecantAstInstructionType type = secant_ast_instruction_type_get(program + offset);
        const size_t instruction_size = secant_ast_instruction_size_get(program + offset);

        if (instruction_size == 0u || type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
            if (secant_ast_index_get(program + offset) != expected_idx) {
                return 0;
            }
            ++expected_idx;
        }
        offset += instruction_size;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return expected_idx == expected_constants;
        }
    }
}

static int
materialized_mixed_constants_validate(
    const SecantAstInstruction* source,
    const SecantAstInstruction* proposal,
    uint32_t mask,
    const uint32_t* words
) {
    size_t source_offset = 0u;
    size_t proposal_offset = 0u;
    size_t constant_idx = 0u;

    for (;;) {
        const SecantAstInstruction* source_instruction = source + source_offset;
        const SecantAstInstruction* proposal_instruction = proposal + proposal_offset;
        const SecantAstInstructionType source_type = secant_ast_instruction_type_get(source_instruction);
        const SecantAstInstructionType proposal_type = secant_ast_instruction_type_get(proposal_instruction);
        const size_t source_size = secant_ast_instruction_size_get(source_instruction);
        const size_t proposal_size = secant_ast_instruction_size_get(proposal_instruction);

        if (source_size == 0u || proposal_size == 0u) {
            return 0;
        }
        if (source_type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            const int column = (mask & (UINT32_C(1) << constant_idx)) != 0u;

            if ((column &&
                 (proposal_type != SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 ||
                  secant_ast_index_get(proposal_instruction) != words[constant_idx])) ||
                (!column &&
                 (proposal_type != SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 ||
                  memcmp(proposal_instruction + 1u, words + constant_idx, sizeof(uint32_t)) != 0))) {
                return 0;
            }
            ++constant_idx;
        } else if (source_type != proposal_type || source_size != proposal_size ||
                   memcmp(source_instruction, proposal_instruction, source_size) != 0) {
            return 0;
        }
        source_offset += source_size;
        proposal_offset += proposal_size;
        if (source_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return constant_idx != 0u;
        }
    }
}

static int
materialized_constants_validate(
    const SecantAstInstruction* source,
    const SecantAstInstruction* proposal,
    const float* settings,
    size_t setting
) {
    size_t constant_idx = 0u;
    size_t offset = 0u;

    for (;;) {
        const SecantAstInstruction* source_instruction = source + offset;
        const SecantAstInstruction* proposal_instruction = proposal + offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(proposal_instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(proposal_instruction);

        if (instruction_size == 0u || type != secant_ast_instruction_type_get(source_instruction) ||
            instruction_size != secant_ast_instruction_size_get(source_instruction) ||
            type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            const float expected = constant_idx < 8u
                ? settings[constant_idx * TEST_SETTINGS + setting]
                : secant_ast_constant_f32_get(source_instruction);

            if (secant_ast_constant_f32_get(proposal_instruction) != expected) {
                return 0;
            }
            ++constant_idx;
        }
        offset += instruction_size;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return constant_idx != 0u;
        }
    }
}

static int
optimizer_constants_validate(
    const SecantAstInstruction* source,
    const SecantAstInstruction* proposal,
    const float* constants
) {
    size_t constant_idx = 0u;
    size_t offset = 0u;

    for (;;) {
        const SecantAstInstruction* source_instruction = source + offset;
        const SecantAstInstruction* proposal_instruction = proposal + offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(proposal_instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(proposal_instruction);

        if (instruction_size == 0u || type != secant_ast_instruction_type_get(source_instruction) ||
            instruction_size != secant_ast_instruction_size_get(source_instruction)) {
            return 0;
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            if (secant_ast_constant_f32_get(proposal_instruction) != constants[constant_idx]) {
                return 0;
            }
            ++constant_idx;
        }
        offset += instruction_size;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return constant_idx != 0u;
        }
    }
}

int
main(void) {
    static const SecantAstInstructionType unary_ops[] = {SECANT_AST_INSTRUCTION_TYPE_NEG_F32};
    static const SecantAstInstructionType binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32
    };
    static const float constants[] = {-2.0f, -1.0f, 0.5f, 2.0f};
    SecantSRSearchConfig config = {0};
    SecantSRSearch search = NULL;
    void* search_storage = NULL;
    SecantAstInstruction* program_storage = NULL;
    const SecantAstInstruction** asts = NULL;
    const SecantAstInstruction* const* source_asts = NULL;
    size_t* program_sizes = NULL;
    float* dynamic_sse = NULL;
    float* proposal_sse = NULL;
    float* robustness = NULL;
    float* optimizer_centers = NULL;
    uint32_t selected_indices[TEST_POPULATION];
    uint32_t best_setting_indices[TEST_POPULATION];
    uint32_t mixed_masks[TEST_POPULATION];
    uint32_t mixed_words[TEST_POPULATION * 8u];
    float settings[SECANT_AST_MAX_DYNAMIC_LEAVES * TEST_SETTINGS];
    size_t storage_size = 0u;
    size_t required_size = 0u;
    size_t num_asts = 0u;
    size_t max_constants = 0u;
    size_t num_eligible = 0u;
    size_t ast_idx;
    int status = 1;

    config.population_size = TEST_POPULATION;
    config.max_program_bytes = 192u;
    config.max_nodes = 30u;
    config.max_depth = 16u;
    config.max_complexity = 40u;
    config.num_inputs = 2u;
    config.num_complexity_buckets = 8u;
    config.elites_per_bucket = 2u;
    config.elite_copies_per_generation = 8u;
    config.tournament_size = 4u;
    config.complexity_factor = 1.5;
    config.crossover_probability = 0.45;
    config.subtree_mutation_probability = 0.30;
    config.point_mutation_probability = 0.20;
    config.constant_leaf_probability = 1.0;
    config.parsimony_coefficient = 0.0001;
    config.seed = UINT64_C(0x64656e7365636f6e);
    config.initial_max_nodes = 30u;
    config.initial_max_leaves = 16u;
    config.constant_setting_credit_weight = 0.1;

    if (secant_sr_search_storage_size(
            &config,
            sizeof(unary_ops) / sizeof(unary_ops[0]),
            sizeof(binary_ops) / sizeof(binary_ops[0]),
            0u,
            sizeof(constants) / sizeof(constants[0]),
            &storage_size) != SECANT_SR_SUCCESS) {
        return 1;
    }
    search_storage = malloc(storage_size);
    program_storage = malloc(config.population_size * config.max_program_bytes);
    asts = malloc(config.population_size * sizeof(*asts));
    program_sizes = malloc(config.population_size * sizeof(*program_sizes));
    dynamic_sse = malloc(config.population_size * TEST_SETTINGS * sizeof(*dynamic_sse));
    proposal_sse = malloc(config.population_size * sizeof(*proposal_sse));
    robustness = malloc(config.population_size * sizeof(*robustness));
    optimizer_centers = malloc(config.population_size * 8u * sizeof(*optimizer_centers));
    if (search_storage == NULL || program_storage == NULL || asts == NULL || program_sizes == NULL ||
        dynamic_sse == NULL || proposal_sse == NULL || robustness == NULL || optimizer_centers == NULL ||
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
            search_storage,
            storage_size,
            &search) != SECANT_SR_SUCCESS ||
        secant_sr_search_active_max_leaves_get(search) != 16u ||
        secant_sr_search_active_max_leaves_set(search, 15u) != SECANT_SR_ERROR_INVALID_VALUE ||
        !population_limits_validate(search)) {
        goto cleanup;
    }
    if (secant_sr_search_asts_get(search, &source_asts, &num_asts) != SECANT_SR_SUCCESS ||
        num_asts != TEST_POPULATION) {
        goto cleanup;
    }
    for (ast_idx = 0u; ast_idx < SECANT_AST_MAX_DYNAMIC_LEAVES; ++ast_idx) {
        size_t setting;

        for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
            settings[ast_idx * TEST_SETTINGS + setting] = (float)(10u * ast_idx + setting) + 0.25f;
        }
    }
    if (secant_sr_search_dynamic_constant_indices_sample(
            search,
            8u,
            0.0,
            selected_indices,
            TEST_POPULATION,
            &num_eligible,
            &num_asts) != SECANT_SR_SUCCESS ||
        num_eligible == 0u || num_asts != 0u ||
        secant_sr_search_dynamic_constant_indices_sample(
            search,
            8u,
            1.0,
            selected_indices,
            TEST_POPULATION,
            &num_eligible,
            &num_asts) != SECANT_SR_SUCCESS ||
        num_eligible != num_asts || num_asts == 0u ||
        secant_sr_search_dynamic_constant_programs_selected_write(
            search,
            selected_indices,
            num_asts,
            8u,
            program_storage,
            config.population_size * config.max_program_bytes,
            asts,
            config.population_size,
            &required_size,
            &max_constants) != SECANT_SR_SUCCESS ||
        max_constants > 8u) {
        goto cleanup;
    }
    {
        const SecantSRIndividual* individuals;
        size_t num_individuals;

        if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS ||
            num_individuals != TEST_POPULATION) {
            goto cleanup;
        }
        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            size_t num_dynamic_constants;
            size_t num_fixed_constants;
            size_t setting;
            const size_t best_setting = ast_idx % TEST_SETTINGS;
            const SecantSRIndividual* individual = individuals + selected_indices[ast_idx];

            if (!projected_program_validate(asts[ast_idx], &num_dynamic_constants, &num_fixed_constants) ||
                num_dynamic_constants != individual->num_constants || num_fixed_constants != 0u ||
                (ast_idx != 0u && selected_indices[ast_idx - 1u] >= selected_indices[ast_idx])) {
                goto cleanup;
            }
            best_setting_indices[ast_idx] = (uint32_t)best_setting;
            for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
                const size_t distance = setting > best_setting ? setting - best_setting : best_setting - setting;

                dynamic_sse[ast_idx * TEST_SETTINGS + setting] = 20.0f + 10.0f * (float)distance;
            }
            proposal_sse[ast_idx] = ast_idx == 0u ? 120.0f : 20.0f;
        }
    }
    if (secant_sr_search_constant_optimizer_centers_selected_write(
            search,
            selected_indices,
            num_asts,
            8u,
            optimizer_centers,
            config.population_size * 8u,
            8u) != SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    if (secant_sr_search_dynamic_leaf_selected_programs_write(
            search,
            selected_indices,
            num_asts,
            SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS,
            8u,
            0u,
            program_storage,
            config.population_size * config.max_program_bytes,
            asts,
            config.population_size,
            &required_size,
            &max_constants) != SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    {
        const SecantSRIndividual* individuals;
        size_t num_individuals;

        if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS) {
            goto cleanup;
        }
        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            const SecantSRIndividual* individual = individuals + selected_indices[ast_idx];
            size_t constant_idx;

            if (!projected_mixed_constants_validate(asts[ast_idx], individual->num_constants)) {
                goto cleanup;
            }
            mixed_masks[ast_idx] = ast_idx % 2u != 0u ? UINT32_C(1) : 0u;
            memset(mixed_words + ast_idx * 8u, 0, 8u * sizeof(*mixed_words));
            for (constant_idx = 0u; constant_idx < individual->num_constants; ++constant_idx) {
                if ((mixed_masks[ast_idx] & (UINT32_C(1) << constant_idx)) != 0u) {
                    mixed_words[ast_idx * 8u + constant_idx] = 1u;
                } else {
                    memcpy(
                        mixed_words + ast_idx * 8u + constant_idx,
                        optimizer_centers + ast_idx * 8u + constant_idx,
                        sizeof(uint32_t));
                }
            }
        }
    }
    if (secant_sr_search_dynamic_leaf_selected_bindings_write(
            search,
            selected_indices,
            num_asts,
            SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS,
            8u,
            0u,
            mixed_masks,
            num_asts,
            mixed_words,
            num_asts * 8u,
            8u,
            program_storage,
            config.population_size * config.max_program_bytes,
            asts,
            program_sizes,
            config.population_size,
            &required_size) != SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        if (!materialized_mixed_constants_validate(
                source_asts[selected_indices[ast_idx]],
                asts[ast_idx],
                mixed_masks[ast_idx],
                mixed_words + ast_idx * 8u)) {
            goto cleanup;
        }
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        const SecantSRIndividual* individuals;
        size_t num_individuals;
        size_t constant_idx;

        if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS ||
            selected_indices[ast_idx] >= num_individuals) {
            goto cleanup;
        }
        for (constant_idx = 0u; constant_idx < individuals[selected_indices[ast_idx]].num_constants; ++constant_idx) {
            optimizer_centers[ast_idx * 8u + constant_idx] += 0.125f * (float)(constant_idx + 1u);
        }
    }
    if (secant_sr_search_constant_optimizer_proposals_selected_write(
            search,
            selected_indices,
            num_asts,
            8u,
            optimizer_centers,
            config.population_size * 8u,
            8u,
            program_storage,
            config.population_size * config.max_program_bytes,
            asts,
            program_sizes,
            config.population_size,
            &required_size) != SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        if (!optimizer_constants_validate(
                source_asts[selected_indices[ast_idx]], asts[ast_idx], optimizer_centers + ast_idx * 8u)) {
            goto cleanup;
        }
    }
    if (secant_sr_search_dynamic_constant_proposals_selected_write(
            search,
            selected_indices,
            num_asts,
            8u,
            settings,
            SECANT_AST_MAX_DYNAMIC_LEAVES * TEST_SETTINGS,
            TEST_SETTINGS,
            TEST_SETTINGS,
            best_setting_indices,
            program_storage,
            config.population_size * config.max_program_bytes,
            asts,
            program_sizes,
            config.population_size,
            &required_size) != SECANT_SR_SUCCESS) {
        goto cleanup;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        const size_t best_setting = ast_idx % TEST_SETTINGS;
        double expected_robustness = 0.0;
        size_t setting;

        if (!materialized_constants_validate(
                source_asts[selected_indices[ast_idx]], asts[ast_idx], settings, best_setting)) {
            goto cleanup;
        }
        for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
            expected_robustness += 1.0 - dynamic_sse[ast_idx * TEST_SETTINGS + setting] / 100.0;
        }
        expected_robustness /= TEST_SETTINGS;
        robustness[ast_idx] = (float)expected_robustness;
    }
    {
        float source_sse[TEST_POPULATION];
        const SecantSRIndividual* individuals;
        size_t num_individuals;
        size_t num_promoted = 0u;
        size_t bounded_eligible = 0u;
        size_t bounded_selected = 0u;

        for (ast_idx = 0u; ast_idx < TEST_POPULATION; ++ast_idx) {
            source_sse[ast_idx] = 100.0f;
        }
        if (secant_sr_search_scores_set(search, source_sse, TEST_POPULATION, 100u, 100.0) != SECANT_SR_SUCCESS ||
            secant_sr_search_selected_proposals_apply(
                search,
                selected_indices,
                asts,
                program_sizes,
                proposal_sse,
                robustness,
                num_asts,
                100u,
                100.0,
                SECANT_SR_ORIGIN_DYNAMIC_CONSTANT,
                &num_promoted) != SECANT_SR_SUCCESS ||
            num_promoted != num_asts - 1u ||
            secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS ||
            num_individuals != TEST_POPULATION) {
            goto cleanup;
        }
        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            const size_t population_idx = selected_indices[ast_idx];
            const double expected_r2 = ast_idx == 0u ? 0.0 : 0.8;
            const double expected_score = expected_r2 -
                config.parsimony_coefficient * individuals[population_idx].complexity +
                config.constant_setting_credit_weight * robustness[ast_idx];

            if (fabs(individuals[population_idx].fitness.constant_setting_robustness - robustness[ast_idx]) > 1.0e-6 ||
                fabs(individuals[population_idx].fitness.score - expected_score) > 1.0e-9 ||
                (ast_idx != 0u && individuals[population_idx].origin != SECANT_SR_ORIGIN_DYNAMIC_CONSTANT)) {
                goto cleanup;
            }
        }
        if (secant_sr_search_dynamic_constant_indices_select(
                search,
                8u,
                7u,
                0.25,
                selected_indices,
                TEST_POPULATION,
                &bounded_eligible,
                &bounded_selected) != SECANT_SR_SUCCESS ||
            bounded_eligible != num_eligible || bounded_selected != 7u ||
            secant_sr_search_dynamic_constant_indices_select(
                search,
                8u,
                7u,
                0.25,
                selected_indices,
                6u,
                &bounded_eligible,
                &bounded_selected) != SECANT_SR_ERROR_INSUFFICIENT_BUFFER) {
            goto cleanup;
        }
        for (ast_idx = 0u; ast_idx < 7u; ++ast_idx) {
            const size_t population_idx = selected_indices[ast_idx];

            if (population_idx >= num_individuals || individuals[population_idx].num_constants == 0u ||
                individuals[population_idx].num_constants > 8u ||
                (ast_idx != 0u && selected_indices[ast_idx - 1u] >= selected_indices[ast_idx])) {
                goto cleanup;
            }
        }
    }
    if (secant_sr_search_active_max_leaves_set(search, 30u) != SECANT_SR_SUCCESS ||
        secant_sr_search_active_max_leaves_get(search) != 30u) {
        goto cleanup;
    }
    status = 0;

cleanup:
    free(optimizer_centers);
    free(robustness);
    free(proposal_sse);
    free(dynamic_sse);
    free(program_sizes);
    free(asts);
    free(program_storage);
    free(search_storage);
    return status;
}
