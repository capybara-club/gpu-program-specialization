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

static SecantSRResult
secant_sr_candidate_begin(
    SecantSRSearch search,
    SecantSRPopulation* population,
    size_t maximum_program_bytes,
    SecantAstInstruction** program_ret,
    SecantSRNodeInfo** nodes_ret
) {
    if (population->count >= search->config.population_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_POPULATION_FULL);
    }
    if (maximum_program_bytes > search->config.max_program_bytes ||
        maximum_program_bytes > population->program_capacity - population->program_used ||
        search->config.max_nodes > population->node_capacity - population->node_used) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_ARENA_EXHAUSTED);
    }
    *program_ret = population->programs + population->program_used;
    *nodes_ret = population->nodes + population->node_used;
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_candidate_commit(
    SecantSRSearch search,
    SecantSRPopulation* population,
    size_t expected_program_bytes,
    SecantSROrigin origin,
    uint32_t parent_a,
    uint32_t parent_b,
    SecantSRIndividual** individual_ret
) {
    SecantAstInstruction* program = population->programs + population->program_used;
    SecantSRNodeInfo* nodes = population->nodes + population->node_used;
    SecantSRIndividual* individual = population->individuals + population->count;
    SecantSRProgramFeatures features;
    SecantSRProgramFeatures* features_ptr =
        search->config.num_column_count_buckets > 1u ||
            search->config.num_transcendental_count_buckets > 1u
        ? &features
        : NULL;
    size_t program_bytes;
    size_t num_nodes;
    size_t num_leaves;
    size_t num_constants;
    uint32_t complexity;
    uint16_t depth;
    uint64_t fingerprint;

    _SECANT_SR_CHECK_RET(secant_sr_program_annotate(
        program,
        expected_program_bytes,
        search->routines,
        search->num_routines,
        search->config.max_depth,
        search->config.max_complexity,
        nodes,
        search->active_max_nodes,
        &program_bytes,
        &num_nodes,
        &num_leaves,
        &num_constants,
        &complexity,
        &depth,
        &fingerprint));
    if (program_bytes != expected_program_bytes) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    if (num_leaves > search->active_max_leaves || num_constants > UINT16_MAX) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    if (features_ptr != NULL) {
        _SECANT_SR_CHECK_RET(secant_sr_program_features_get(
            program,
            program_bytes,
            search->routines,
            search->num_routines,
            features_ptr));
    }

    memset(individual, 0, sizeof(*individual));
    individual->program = program;
    individual->nodes = nodes;
    individual->routines = search->routines;
    individual->program_bytes = program_bytes;
    individual->num_nodes = num_nodes;
    individual->num_leaves = num_leaves;
    individual->num_routines = search->num_routines;
    individual->num_constants = (uint16_t)num_constants;
    individual->complexity = complexity;
    individual->depth = depth;
    individual->complexity_bucket = secant_sr_complexity_bucket_get(search, complexity);
    if (features_ptr != NULL) {
        individual->column_count_bucket = secant_sr_column_count_bucket_get(
            search, features.num_unique_static_columns);
        individual->transcendental_count_bucket = secant_sr_transcendental_count_bucket_get(
            search, features.num_transcendental_operations);
    }
    individual->fingerprint = fingerprint;
    individual->birth_generation = population == &search->populations[search->current_population]
        ? search->generation
        : search->generation + 1u;
    individual->parent_a = parent_a;
    individual->parent_b = parent_b;
    individual->origin = origin;
    individual->maturity = origin == SECANT_SR_ORIGIN_RANDOM
        ? SECANT_SR_MATURITY_EXPLORATORY
        : SECANT_SR_MATURITY_RESOLVED;
    individual->fitness.score = -DBL_MAX;

    population->asts[population->count] = program;
    population->program_used += program_bytes;
    population->node_used += num_nodes;
    ++population->count;
    if (individual_ret != NULL) {
        *individual_ret = individual;
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_population_append_program(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantAstInstruction* program,
    size_t program_bytes,
    SecantSROrigin origin,
    uint32_t parent_a,
    uint32_t parent_b,
    SecantSRIndividual** individual_ret
) {
    SecantAstInstruction* destination;
    SecantSRNodeInfo* nodes;

    if (search == NULL || population == NULL || program == NULL || program_bytes == 0u) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(search, population, program_bytes, &destination, &nodes));
    (void)nodes;
    memcpy(destination, program, program_bytes);
    return secant_sr_candidate_commit(
        search,
        population,
        program_bytes,
        origin,
        parent_a,
        parent_b,
        individual_ret);
}

static SecantSRResult
secant_sr_program_byte_write(
    SecantAstInstruction value,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* offset
) {
    if (*offset >= output_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_ARENA_EXHAUSTED);
    }
    output[(*offset)++] = value;
    return SECANT_SR_SUCCESS;
}

typedef struct SecantSROperatorChoice {
    SecantAstInstructionType type;
    uint16_t complexity;
    uint8_t routine_idx;
} SecantSROperatorChoice;

static size_t
secant_sr_operator_count(SecantSRSearch search, uint8_t arity) {
    return arity == 1u
        ? search->num_unary_ops + search->num_unary_routines
        : search->num_binary_ops + search->num_binary_routines;
}

static SecantSROperatorChoice
secant_sr_operator_choose(SecantSRSearch search, uint8_t arity) {
    SecantSROperatorChoice choice;
    const size_t num_builtin_ops = arity == 1u ? search->num_unary_ops : search->num_binary_ops;
    const size_t selection = secant_sr_rng_bounded(search, secant_sr_operator_count(search, arity));

    memset(&choice, 0, sizeof(choice));
    if (selection < num_builtin_ops) {
        choice.type = arity == 1u ? search->unary_ops[selection] : search->binary_ops[selection];
        choice.complexity = secant_sr_instruction_complexity(choice.type);
    } else {
        const size_t routine_selection = selection - num_builtin_ops;
        const uint8_t routine_idx = arity == 1u
            ? search->unary_routine_indices[routine_selection]
            : search->binary_routine_indices[routine_selection];

        choice.type = SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32;
        choice.routine_idx = routine_idx;
        choice.complexity = search->routines[routine_idx].complexity;
    }
    return choice;
}

static SecantSRResult
secant_sr_operator_write(
    SecantSROperatorChoice choice,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* offset
) {
    _SECANT_SR_CHECK_RET(secant_sr_program_byte_write((uint8_t)choice.type, output, output_size, offset));
    if (choice.type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
        _SECANT_SR_CHECK_RET(secant_sr_program_byte_write(choice.routine_idx, output, output_size, offset));
    }
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_leaf_generate(
    SecantSRSearch search,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* offset
) {
    if (search->num_constants != 0u &&
        secant_sr_rng_unit(search) < search->config.constant_leaf_probability) {
        const float value = search->constants[secant_sr_rng_bounded(search, search->num_constants)];
        uint32_t bits;

        if (output_size - *offset < 5u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_ARENA_EXHAUSTED);
        }
        memcpy(&bits, &value, sizeof(bits));
        output[(*offset)++] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32;
        output[(*offset)++] = (uint8_t)(bits >> 0u);
        output[(*offset)++] = (uint8_t)(bits >> 8u);
        output[(*offset)++] = (uint8_t)(bits >> 16u);
        output[(*offset)++] = (uint8_t)(bits >> 24u);
        return SECANT_SR_SUCCESS;
    }
    _SECANT_SR_CHECK_RET(secant_sr_program_byte_write(
        (SecantAstInstruction)SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32,
        output,
        output_size,
        offset));
    return secant_sr_program_byte_write(
        (SecantAstInstruction)secant_sr_rng_bounded(search, search->config.num_inputs), output, output_size, offset);
}

static SecantSRResult
secant_sr_subtree_generate(
    SecantSRSearch search,
    uint32_t target_complexity,
    size_t depth_remaining,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* offset
) {
    int use_binary;
    SecantSROperatorChoice operation;
    const size_t num_unary_operators = secant_sr_operator_count(search, 1u);
    const size_t num_binary_operators = secant_sr_operator_count(search, 2u);

    if (target_complexity <= 1u || depth_remaining <= 1u ||
        (num_unary_operators == 0u && num_binary_operators == 0u)) {
        return secant_sr_leaf_generate(search, output, output_size, offset);
    }

    use_binary = num_binary_operators != 0u &&
        (num_unary_operators == 0u || (target_complexity >= 3u && secant_sr_rng_unit(search) < 0.70));
    if (use_binary) {
        uint32_t child_budget;
        uint32_t left_budget;
        uint32_t right_budget;

        operation = secant_sr_operator_choose(search, 2u);
        child_budget = target_complexity > operation.complexity + 1u
            ? target_complexity - operation.complexity
            : 2u;
        if (child_budget < 2u) {
            child_budget = 2u;
        }
        left_budget = 1u + (uint32_t)secant_sr_rng_bounded(search, child_budget - 1u);
        right_budget = child_budget - left_budget;
        if (right_budget == 0u) {
            right_budget = 1u;
        }
        _SECANT_SR_CHECK_RET(secant_sr_subtree_generate(
            search, left_budget, depth_remaining - 1u, output, output_size, offset));
        _SECANT_SR_CHECK_RET(secant_sr_subtree_generate(
            search, right_budget, depth_remaining - 1u, output, output_size, offset));
    } else {
        uint32_t child_budget;

        operation = secant_sr_operator_choose(search, 1u);
        child_budget = target_complexity > operation.complexity
            ? target_complexity - operation.complexity
            : 1u;
        _SECANT_SR_CHECK_RET(secant_sr_subtree_generate(
            search, child_budget, depth_remaining - 1u, output, output_size, offset));
    }
    return secant_sr_operator_write(operation, output, output_size, offset);
}

SecantSRResult
secant_sr_population_random_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    uint32_t target_complexity,
    SecantSROrigin origin,
    uint32_t parent_a,
    SecantSRIndividual** individual_ret
) {
    SecantAstInstruction* destination;
    SecantSRNodeInfo* nodes;
    size_t program_bytes = 0u;

    _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(
        search, population, search->config.max_program_bytes, &destination, &nodes));
    (void)nodes;
    _SECANT_SR_CHECK_RET(secant_sr_subtree_generate(
        search,
        target_complexity == 0u ? 1u : target_complexity,
        search->config.max_depth,
        destination,
        search->config.max_program_bytes,
        &program_bytes));
    _SECANT_SR_CHECK_RET(secant_sr_program_byte_write(
        (uint8_t)SECANT_AST_INSTRUCTION_TYPE_RETURN_F32,
        destination,
        search->config.max_program_bytes,
        &program_bytes));
    return secant_sr_candidate_commit(
        search,
        population,
        program_bytes,
        origin,
        parent_a,
        SECANT_SR_PARENT_NONE,
        individual_ret);
}

SecantSRResult
secant_sr_population_crossover_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantSRIndividual* parent_a,
    uint32_t parent_a_idx,
    const SecantSRIndividual* parent_b,
    uint32_t parent_b_idx
) {
    const SecantSRNodeInfo* target;
    const SecantSRNodeInfo* donor;
    SecantAstInstruction* destination;
    SecantSRNodeInfo* nodes;
    size_t destination_bytes;
    size_t suffix_offset;
    size_t output_offset = 0u;

    target = parent_a->nodes + secant_sr_rng_bounded(search, parent_a->num_nodes);
    donor = parent_b->nodes + secant_sr_rng_bounded(search, parent_b->num_nodes);
    destination_bytes = parent_a->program_bytes - target->subtree_bytes + donor->subtree_bytes;
    _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(search, population, destination_bytes, &destination, &nodes));
    (void)nodes;

    memcpy(destination, parent_a->program, target->subtree_offset);
    output_offset = target->subtree_offset;
    memcpy(destination + output_offset, parent_b->program + donor->subtree_offset, donor->subtree_bytes);
    output_offset += donor->subtree_bytes;
    suffix_offset = target->subtree_offset + target->subtree_bytes;
    memcpy(destination + output_offset, parent_a->program + suffix_offset, parent_a->program_bytes - suffix_offset);

    return secant_sr_candidate_commit(
        search,
        population,
        destination_bytes,
        SECANT_SR_ORIGIN_CROSSOVER,
        parent_a_idx,
        parent_b_idx,
        NULL);
}

SecantSRResult
secant_sr_population_subtree_mutation_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantSRIndividual* parent,
    uint32_t parent_idx
) {
    const SecantSRNodeInfo* target = parent->nodes + secant_sr_rng_bounded(search, parent->num_nodes);
    SecantAstInstruction* destination;
    SecantSRNodeInfo* nodes;
    size_t suffix_offset = target->subtree_offset + target->subtree_bytes;
    size_t output_offset = target->subtree_offset;
    uint32_t target_complexity;

    _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(
        search, population, search->config.max_program_bytes, &destination, &nodes));
    (void)nodes;
    memcpy(destination, parent->program, target->subtree_offset);
    target_complexity = 1u + (uint32_t)secant_sr_rng_bounded(
        search,
        target->complexity < search->config.max_complexity
            ? target->complexity + 1u
            : search->config.max_complexity);
    _SECANT_SR_CHECK_RET(secant_sr_subtree_generate(
        search,
        target_complexity,
        search->config.max_depth,
        destination,
        search->config.max_program_bytes,
        &output_offset));
    if (parent->program_bytes - suffix_offset > search->config.max_program_bytes - output_offset) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_ARENA_EXHAUSTED);
    }
    memcpy(destination + output_offset, parent->program + suffix_offset, parent->program_bytes - suffix_offset);
    output_offset += parent->program_bytes - suffix_offset;

    return secant_sr_candidate_commit(
        search,
        population,
        output_offset,
        SECANT_SR_ORIGIN_SUBTREE_MUTATION,
        parent_idx,
        SECANT_SR_PARENT_NONE,
        NULL);
}

SecantSRResult
secant_sr_population_point_mutation_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantSRIndividual* parent,
    uint32_t parent_idx
) {
    const SecantSRNodeInfo* target = parent->nodes + secant_sr_rng_bounded(search, parent->num_nodes);
    SecantAstInstruction* destination;
    SecantSRNodeInfo* nodes;
    const SecantAstInstruction* source_instruction = parent->program + target->instruction_offset;
    const SecantAstInstructionType type = secant_ast_instruction_type_get(source_instruction);

    if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
        SecantAstInstruction* instruction;

        _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(
            search, population, parent->program_bytes, &destination, &nodes));
        (void)nodes;
        memcpy(destination, parent->program, parent->program_bytes);
        instruction = destination + target->instruction_offset;
        instruction[1] = (uint8_t)secant_sr_rng_bounded(search, search->config.num_inputs);
    } else if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 && search->num_constants != 0u) {
        const float value = search->constants[secant_sr_rng_bounded(search, search->num_constants)];
        uint32_t bits;
        SecantAstInstruction* instruction;

        _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(
            search, population, parent->program_bytes, &destination, &nodes));
        (void)nodes;
        memcpy(destination, parent->program, parent->program_bytes);
        instruction = destination + target->instruction_offset;
        memcpy(&bits, &value, sizeof(bits));
        instruction[1] = (uint8_t)(bits >> 0u);
        instruction[2] = (uint8_t)(bits >> 8u);
        instruction[3] = (uint8_t)(bits >> 16u);
        instruction[4] = (uint8_t)(bits >> 24u);
    } else if ((target->arity == 1u || target->arity == 2u) &&
               secant_sr_operator_count(search, target->arity) != 0u) {
        const SecantSROperatorChoice choice = secant_sr_operator_choose(search, target->arity);
        const size_t replacement_size = choice.type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32 ? 2u : 1u;
        const size_t destination_bytes = parent->program_bytes - target->instruction_size + replacement_size;
        const size_t suffix_offset = target->instruction_offset + target->instruction_size;
        size_t output_offset = target->instruction_offset;

        _SECANT_SR_CHECK_RET(secant_sr_candidate_begin(
            search, population, destination_bytes, &destination, &nodes));
        (void)nodes;
        memcpy(destination, parent->program, target->instruction_offset);
        _SECANT_SR_CHECK_RET(secant_sr_operator_write(
            choice, destination, destination_bytes, &output_offset));
        memcpy(
            destination + output_offset,
            parent->program + suffix_offset,
            parent->program_bytes - suffix_offset);
        return secant_sr_candidate_commit(
            search,
            population,
            destination_bytes,
            SECANT_SR_ORIGIN_POINT_MUTATION,
            parent_idx,
            SECANT_SR_PARENT_NONE,
            NULL);
    } else {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SEARCH_STALLED);
    }

    return secant_sr_candidate_commit(
        search,
        population,
        parent->program_bytes,
        SECANT_SR_ORIGIN_POINT_MUTATION,
        parent_idx,
        SECANT_SR_PARENT_NONE,
        NULL);
}
