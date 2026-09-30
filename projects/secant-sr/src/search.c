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

static size_t
secant_sr_qd_bucket_count(size_t configured_count) {
    return configured_count != 0u ? configured_count : 1u;
}

static int
secant_sr_config_is_valid(const SecantSRSearchConfig* config) {
    double variation_probability;

    if (config == NULL || config->population_size == 0u || config->population_size > UINT32_MAX ||
        config->max_program_bytes < 2u || config->max_program_bytes > SECANT_AST_MAX_PROGRAM_BYTES ||
        config->max_nodes == 0u || config->max_nodes >= SECANT_AST_MAX_PROGRAM_INSTRUCTIONS ||
        config->initial_max_nodes > config->max_nodes ||
        config->initial_max_leaves > config->max_nodes ||
        config->max_depth == 0u || config->max_depth >= UINT16_MAX ||
        config->max_complexity == 0u || config->max_complexity > UINT16_MAX ||
        config->num_inputs == 0u || config->num_inputs > SECANT_AST_MAX_INPUTS ||
        config->num_complexity_buckets == 0u || config->num_complexity_buckets > UINT16_MAX ||
        config->num_column_count_buckets > UINT16_MAX ||
        config->num_transcendental_count_buckets > UINT16_MAX ||
        config->elites_per_bucket == 0u ||
        config->elite_copies_per_generation > config->population_size ||
        config->tournament_size == 0u || config->complexity_factor <= 1.0 ||
        config->crossover_probability < 0.0 || config->subtree_mutation_probability < 0.0 ||
        config->point_mutation_probability < 0.0 || config->constant_leaf_probability < 0.0 ||
        config->constant_leaf_probability > 1.0 || config->parsimony_coefficient < 0.0 ||
        !isfinite(config->archive_parent_probability) ||
        config->archive_parent_probability < 0.0 || config->archive_parent_probability > 1.0 ||
        !isfinite(config->constant_setting_credit_weight) || config->constant_setting_credit_weight < 0.0) {
        return 0;
    }
    variation_probability = config->crossover_probability + config->subtree_mutation_probability +
        config->point_mutation_probability;
    return variation_probability <= 1.0;
}

static SecantSRResult
secant_sr_routines_validate(const SecantSRRoutine* routines, size_t num_routines) {
    size_t routine_idx;

    if (num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        const SecantSRRoutine* routine = routines + routine_idx;
        size_t instruction_offset = 0u;
        size_t instruction_count;
        size_t derived_arity = 0u;
        int returned = 0;

        if (routine->program == NULL || routine->name == NULL || routine->name[0] == '\0' ||
            routine->complexity == 0u || routine->arity == 0u || routine->arity > 2u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        for (instruction_count = 0u;
             instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
             ++instruction_count) {
            const SecantAstInstruction* instruction = routine->program + instruction_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);

            if (instruction_size == 0u) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
                const size_t arg_idx = secant_ast_index_get(instruction);

                if (arg_idx >= SECANT_AST_MAX_INSTRUCTION_ARGS) {
                    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
                }
                if (derived_arity <= arg_idx) {
                    derived_arity = arg_idx + 1u;
                }
            } else if (type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32 &&
                       secant_ast_index_get(instruction) >= num_routines) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            instruction_offset += instruction_size;
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                returned = 1;
                break;
            }
        }
        if (!returned || derived_arity != routine->arity) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_storage_size(
    const SecantSRSearchConfig* config,
    size_t num_unary_ops,
    size_t num_binary_ops,
    size_t num_routines,
    size_t num_constants,
    size_t* storage_size_ret
) {
    size_t population_programs;
    size_t population_nodes;
    size_t num_archive_cells;
    size_t num_elite_slots;
    size_t elite_programs;
    size_t elite_nodes;
    size_t offset = 0u;
    size_t population_idx;

    if (!secant_sr_config_is_valid(config) || num_routines > SECANT_AST_MAX_ROUTINES ||
        storage_size_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    if (!secant_sr_checked_mul(config->population_size, config->max_program_bytes, &population_programs) ||
        !secant_sr_checked_mul(config->population_size, config->max_nodes, &population_nodes) ||
        !secant_sr_checked_mul(
            config->num_complexity_buckets,
            secant_sr_qd_bucket_count(config->num_column_count_buckets),
            &num_archive_cells) ||
        !secant_sr_checked_mul(
            num_archive_cells,
            secant_sr_qd_bucket_count(config->num_transcendental_count_buckets),
            &num_archive_cells) ||
        !secant_sr_checked_mul(num_archive_cells, config->elites_per_bucket, &num_elite_slots) ||
        !secant_sr_checked_mul(num_elite_slots, config->max_program_bytes, &elite_programs) ||
        !secant_sr_checked_mul(num_elite_slots, config->max_nodes, &elite_nodes)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    if (!secant_sr_layout_add(1u, sizeof(struct SecantSRSearchImpl), &offset) ||
        !secant_sr_layout_add(num_unary_ops, sizeof(SecantAstInstructionType), &offset) ||
        !secant_sr_layout_add(num_binary_ops, sizeof(SecantAstInstructionType), &offset) ||
        !secant_sr_layout_add(num_routines, sizeof(SecantSRRoutine), &offset) ||
        !secant_sr_layout_add(num_routines, sizeof(SecantAstInstruction*), &offset) ||
        !secant_sr_layout_add(num_routines, sizeof(uint8_t), &offset) ||
        !secant_sr_layout_add(num_routines, sizeof(uint8_t), &offset) ||
        !secant_sr_layout_add(num_constants, sizeof(float), &offset)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    for (population_idx = 0u; population_idx < 2u; ++population_idx) {
        if (!secant_sr_layout_add(config->population_size, sizeof(SecantSRIndividual), &offset) ||
            !secant_sr_layout_add(config->population_size, sizeof(SecantAstInstruction*), &offset) ||
            !secant_sr_layout_add(population_programs, sizeof(SecantAstInstruction), &offset) ||
            !secant_sr_layout_add(population_nodes, sizeof(SecantSRNodeInfo), &offset)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
    }
    if (!secant_sr_layout_add(num_elite_slots, sizeof(SecantSREliteSlot), &offset) ||
        !secant_sr_layout_add(num_archive_cells, sizeof(size_t), &offset) ||
        !secant_sr_layout_add(elite_programs, sizeof(SecantAstInstruction), &offset) ||
        !secant_sr_layout_add(elite_nodes, sizeof(SecantSRNodeInfo), &offset)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    *storage_size_ret = offset;
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_operator_lists_validate(
    const SecantAstInstructionType* unary_ops,
    size_t num_unary_ops,
    const SecantAstInstructionType* binary_ops,
    size_t num_binary_ops
) {
    size_t idx;

    if ((num_unary_ops != 0u && unary_ops == NULL) || (num_binary_ops != 0u && binary_ops == NULL)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (idx = 0u; idx < num_unary_ops; ++idx) {
        if (!secant_sr_instruction_is_unary(unary_ops[idx])) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }
    for (idx = 0u; idx < num_binary_ops; ++idx) {
        if (!secant_sr_instruction_is_binary(binary_ops[idx])) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_initial_population_seed(SecantSRSearch search) {
    SecantSRPopulation* population = &search->populations[0];
    size_t individual_idx;

    for (individual_idx = 0u; individual_idx < search->config.population_size; ++individual_idx) {
        uint32_t target_complexity = 1u + (uint32_t)(individual_idx % search->config.max_complexity);
        SecantSRResult result = SECANT_SR_ERROR_SEARCH_STALLED;
        size_t attempt;

        for (attempt = 0u; attempt < 16u; ++attempt) {
            result = secant_sr_population_random_append(
                search,
                population,
                target_complexity,
                SECANT_SR_ORIGIN_RANDOM,
                SECANT_SR_PARENT_NONE,
                NULL);
            if (result == SECANT_SR_SUCCESS) {
                break;
            }
            target_complexity = target_complexity > 1u ? (target_complexity + 1u) / 2u : 1u;
        }
        if (result != SECANT_SR_SUCCESS) {
            return result;
        }
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_init(
    const SecantSRSearchConfig* config,
    const SecantAstInstructionType* unary_ops,
    size_t num_unary_ops,
    const SecantAstInstructionType* binary_ops,
    size_t num_binary_ops,
    const SecantSRRoutine* routines,
    size_t num_routines,
    const float* constants,
    size_t num_constants,
    void* storage,
    size_t storage_size,
    SecantSRSearch* search_ret
) {
    SecantSRSearch search;
    SecantAstInstruction* elite_programs;
    SecantSRNodeInfo* elite_nodes;
    size_t required_size;
    size_t population_programs;
    size_t population_nodes;
    size_t elite_program_count;
    size_t elite_node_count;
    size_t offset = 0u;
    size_t population_idx;
    size_t elite_idx;
    size_t routine_idx;

    if (search_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    *search_ret = NULL;
    _SECANT_SR_CHECK_RET(secant_sr_operator_lists_validate(
        unary_ops, num_unary_ops, binary_ops, num_binary_ops));
    _SECANT_SR_CHECK_RET(secant_sr_routines_validate(routines, num_routines));
    if (num_constants != 0u && constants == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_search_storage_size(
        config, num_unary_ops, num_binary_ops, num_routines, num_constants, &required_size));
    if (storage == NULL || storage_size < required_size ||
        (uintptr_t)storage % sizeof(void*) != 0u) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }

    search = secant_sr_layout_take(storage, 1u, sizeof(*search), &offset);
    memset(search, 0, sizeof(*search));
    search->config = *config;
    search->config.num_column_count_buckets = secant_sr_qd_bucket_count(config->num_column_count_buckets);
    search->config.num_transcendental_count_buckets =
        secant_sr_qd_bucket_count(config->num_transcendental_count_buckets);
    search->num_unary_ops = num_unary_ops;
    search->num_binary_ops = num_binary_ops;
    search->num_routines = num_routines;
    search->num_constants = num_constants;
    search->active_max_nodes = config->initial_max_nodes != 0u ? config->initial_max_nodes : config->max_nodes;
    search->active_max_leaves = config->initial_max_leaves != 0u ? config->initial_max_leaves : config->max_nodes;
    search->unary_ops = secant_sr_layout_take(
        storage, num_unary_ops, sizeof(SecantAstInstructionType), &offset);
    search->binary_ops = secant_sr_layout_take(
        storage, num_binary_ops, sizeof(SecantAstInstructionType), &offset);
    search->routines = secant_sr_layout_take(storage, num_routines, sizeof(SecantSRRoutine), &offset);
    search->routine_programs = secant_sr_layout_take(
        storage, num_routines, sizeof(SecantAstInstruction*), &offset);
    search->unary_routine_indices = secant_sr_layout_take(storage, num_routines, sizeof(uint8_t), &offset);
    search->binary_routine_indices = secant_sr_layout_take(storage, num_routines, sizeof(uint8_t), &offset);
    search->constants = secant_sr_layout_take(storage, num_constants, sizeof(float), &offset);
    if (num_unary_ops != 0u) {
        memcpy(search->unary_ops, unary_ops, num_unary_ops * sizeof(*unary_ops));
    }
    if (num_binary_ops != 0u) {
        memcpy(search->binary_ops, binary_ops, num_binary_ops * sizeof(*binary_ops));
    }
    if (num_routines != 0u) {
        memcpy(search->routines, routines, num_routines * sizeof(*routines));
    }
    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        search->routine_programs[routine_idx] = routines[routine_idx].program;
        if (routines[routine_idx].arity == 1u) {
            search->unary_routine_indices[search->num_unary_routines++] = (uint8_t)routine_idx;
        } else {
            search->binary_routine_indices[search->num_binary_routines++] = (uint8_t)routine_idx;
        }
    }
    if (num_constants != 0u) {
        memcpy(search->constants, constants, num_constants * sizeof(*constants));
    }

    secant_sr_checked_mul(config->population_size, config->max_program_bytes, &population_programs);
    secant_sr_checked_mul(config->population_size, config->max_nodes, &population_nodes);
    for (population_idx = 0u; population_idx < 2u; ++population_idx) {
        SecantSRPopulation* population = search->populations + population_idx;

        population->individuals = secant_sr_layout_take(
            storage, config->population_size, sizeof(SecantSRIndividual), &offset);
        population->asts = secant_sr_layout_take(
            storage, config->population_size, sizeof(SecantAstInstruction*), &offset);
        population->programs = secant_sr_layout_take(
            storage, population_programs, sizeof(SecantAstInstruction), &offset);
        population->nodes = secant_sr_layout_take(
            storage, population_nodes, sizeof(SecantSRNodeInfo), &offset);
        population->program_capacity = population_programs;
        population->node_capacity = population_nodes;
    }

    secant_sr_checked_mul(
        config->num_complexity_buckets,
        search->config.num_column_count_buckets,
        &search->num_archive_cells);
    secant_sr_checked_mul(
        search->num_archive_cells,
        search->config.num_transcendental_count_buckets,
        &search->num_archive_cells);
    secant_sr_checked_mul(search->num_archive_cells, config->elites_per_bucket, &search->num_elite_slots);
    secant_sr_checked_mul(search->num_elite_slots, config->max_program_bytes, &elite_program_count);
    secant_sr_checked_mul(search->num_elite_slots, config->max_nodes, &elite_node_count);
    search->elites = secant_sr_layout_take(storage, search->num_elite_slots, sizeof(SecantSREliteSlot), &offset);
    search->occupied_cells = secant_sr_layout_take(storage, search->num_archive_cells, sizeof(size_t), &offset);
    memset(search->elites, 0, search->num_elite_slots * sizeof(*search->elites));
    elite_programs = secant_sr_layout_take(
        storage, elite_program_count, sizeof(SecantAstInstruction), &offset);
    elite_nodes = secant_sr_layout_take(storage, elite_node_count, sizeof(SecantSRNodeInfo), &offset);
    for (elite_idx = 0u; elite_idx < search->num_elite_slots; ++elite_idx) {
        search->elites[elite_idx].program = elite_programs + elite_idx * config->max_program_bytes;
        search->elites[elite_idx].nodes = elite_nodes + elite_idx * config->max_nodes;
    }
    if (offset != required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }

    search->rng_state = config->seed != 0u ? config->seed : UINT64_C(0x9e3779b97f4a7c15);
    search->optimizer_rng_state = search->rng_state ^ UINT64_C(0xd1b54a32d192ed03);
    if (search->optimizer_rng_state == 0u) {
        search->optimizer_rng_state = UINT64_C(0x94d049bb133111eb);
    }
    search->current_population = 0u;
    search->generation = 0u;
    _SECANT_SR_CHECK_RET(secant_sr_initial_population_seed(search));
    *search_ret = search;
    return SECANT_SR_SUCCESS;
}

uint32_t
secant_sr_search_generation_get(SecantSRSearch search) {
    return search != NULL ? search->generation : 0u;
}

size_t
secant_sr_search_population_size_get(SecantSRSearch search) {
    return search != NULL ? search->populations[search->current_population].count : 0u;
}

SecantSRResult
secant_sr_search_routines_get(
    SecantSRSearch search,
    const SecantAstInstruction* const** routines_ret,
    size_t* num_routines_ret
) {
    if (search == NULL || routines_ret == NULL || num_routines_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    *routines_ret = search->routine_programs;
    *num_routines_ret = search->num_routines;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_active_max_nodes_set(SecantSRSearch search, size_t max_nodes) {
    if (search == NULL || max_nodes < search->active_max_nodes || max_nodes > search->config.max_nodes) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    search->active_max_nodes = max_nodes;
    return SECANT_SR_SUCCESS;
}

size_t
secant_sr_search_active_max_nodes_get(SecantSRSearch search) {
    return search != NULL ? search->active_max_nodes : 0u;
}

SecantSRResult
secant_sr_search_active_max_leaves_set(SecantSRSearch search, size_t max_leaves) {
    if (search == NULL || max_leaves < search->active_max_leaves || max_leaves > search->config.max_nodes) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    search->active_max_leaves = max_leaves;
    return SECANT_SR_SUCCESS;
}

size_t
secant_sr_search_active_max_leaves_get(SecantSRSearch search) {
    return search != NULL ? search->active_max_leaves : 0u;
}

SecantSRResult
secant_sr_search_archive_parent_probability_set(SecantSRSearch search, double probability) {
    if (search == NULL || !isfinite(probability) || probability < 0.0 || probability > 1.0) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    search->config.archive_parent_probability = probability;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_asts_get(
    SecantSRSearch search,
    const SecantAstInstruction* const** asts_ret,
    size_t* num_asts_ret
) {
    const SecantSRPopulation* population;

    if (search == NULL || asts_ret == NULL || num_asts_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    *asts_ret = population->asts;
    *num_asts_ret = population->count;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_individuals_get(
    SecantSRSearch search,
    const SecantSRIndividual** individuals_ret,
    size_t* num_individuals_ret
) {
    const SecantSRPopulation* population;

    if (search == NULL || individuals_ret == NULL || num_individuals_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    *individuals_ret = population->individuals;
    *num_individuals_ret = population->count;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_archive_stats_get(SecantSRSearch search, SecantSRArchiveStats* stats_ret) {
    if (search == NULL || stats_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    stats_ret->num_cells = search->num_archive_cells;
    stats_ret->occupied_cells = search->num_occupied_cells;
    stats_ret->num_elites = search->num_occupied_elites;
    return SECANT_SR_SUCCESS;
}

uint16_t
secant_sr_complexity_bucket_get(SecantSRSearch search, uint32_t complexity) {
    double upper_bound = 1.0;
    size_t bucket_idx;

    for (bucket_idx = 0u; bucket_idx + 1u < search->config.num_complexity_buckets; ++bucket_idx) {
        if ((double)complexity <= upper_bound) {
            return (uint16_t)bucket_idx;
        }
        upper_bound *= search->config.complexity_factor;
    }
    return (uint16_t)(search->config.num_complexity_buckets - 1u);
}

uint16_t
secant_sr_column_count_bucket_get(SecantSRSearch search, uint16_t num_unique_columns) {
    const size_t last_bucket = search->config.num_column_count_buckets - 1u;

    return (uint16_t)(num_unique_columns < last_bucket ? num_unique_columns : last_bucket);
}

uint16_t
secant_sr_transcendental_count_bucket_get(SecantSRSearch search, uint16_t num_operations) {
    const size_t last_bucket = search->config.num_transcendental_count_buckets - 1u;

    return (uint16_t)(num_operations < last_bucket ? num_operations : last_bucket);
}

static size_t
secant_sr_archive_cell_get(SecantSRSearch search, const SecantSRIndividual* individual) {
    return ((size_t)individual->complexity_bucket * search->config.num_column_count_buckets +
        individual->column_count_bucket) * search->config.num_transcendental_count_buckets +
        individual->transcendental_count_bucket;
}

static void
secant_sr_elite_copy(SecantSREliteSlot* slot, const SecantSRIndividual* source) {
    memcpy(slot->program, source->program, source->program_bytes);
    memcpy(slot->nodes, source->nodes, source->num_nodes * sizeof(*source->nodes));
    slot->individual = *source;
    slot->individual.program = slot->program;
    slot->individual.nodes = slot->nodes;
    slot->valid = 1;
}

SecantSRResult
secant_sr_archive_consider(SecantSRSearch search, const SecantSRIndividual* individual) {
    const size_t cell_idx = secant_sr_archive_cell_get(search, individual);
    const size_t first_slot = cell_idx * search->config.elites_per_bucket;
    SecantSREliteSlot* empty = NULL;
    SecantSREliteSlot* worst = NULL;
    size_t slot_idx;

    for (slot_idx = 0u; slot_idx < search->config.elites_per_bucket; ++slot_idx) {
        SecantSREliteSlot* slot = search->elites + first_slot + slot_idx;

        if (slot->valid && slot->individual.fingerprint == individual->fingerprint) {
            if (individual->fitness.score > slot->individual.fitness.score) {
                secant_sr_elite_copy(slot, individual);
            }
            return SECANT_SR_SUCCESS;
        }
        if (!slot->valid) {
            empty = slot;
            break;
        } else if (worst == NULL || slot->individual.fitness.score < worst->individual.fitness.score) {
            worst = slot;
        }
    }
    if (empty != NULL) {
        if (worst == NULL) {
            search->occupied_cells[search->num_occupied_cells++] = cell_idx;
        }
        ++search->num_occupied_elites;
        secant_sr_elite_copy(empty, individual);
    } else if (worst != NULL && individual->fitness.score > worst->individual.fitness.score) {
        secant_sr_elite_copy(worst, individual);
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_scores_set(
    SecantSRSearch search,
    const float* sse,
    size_t num_sse,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    SecantSRPopulation* population;
    size_t individual_idx;

    if (search == NULL || sse == NULL || num_rows == 0u || target_sum_squared_deviation <= 0.0) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    if (num_sse < population->count) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (individual_idx = 0u; individual_idx < population->count; ++individual_idx) {
        SecantSRIndividual* individual = population->individuals + individual_idx;

        secant_sr_individual_score_set(
            search,
            individual,
            sse[individual_idx],
            num_rows,
            target_sum_squared_deviation);
        _SECANT_SR_CHECK_RET(secant_sr_archive_consider(search, individual));
    }
    return SECANT_SR_SUCCESS;
}

void
secant_sr_individual_score_set(
    SecantSRSearch search,
    SecantSRIndividual* individual,
    double sse,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    individual->fitness.sse = sse;
    if (isfinite(sse) && sse >= 0.0) {
        individual->fitness.mse = sse / (double)num_rows;
        individual->fitness.rmse = sqrt(individual->fitness.mse);
        individual->fitness.nmse = sse / target_sum_squared_deviation;
        individual->fitness.r2 = 1.0 - individual->fitness.nmse;
        individual->fitness.score = individual->fitness.r2 -
            search->config.parsimony_coefficient * (double)individual->complexity +
            search->config.constant_setting_credit_weight * individual->fitness.constant_setting_robustness;
    } else {
        individual->fitness.mse = INFINITY;
        individual->fitness.rmse = INFINITY;
        individual->fitness.nmse = INFINITY;
        individual->fitness.r2 = -INFINITY;
        individual->fitness.score = -DBL_MAX;
    }
    individual->scored = 1;
}

static uint32_t
secant_sr_tournament_select(SecantSRSearch search, const SecantSRPopulation* population) {
    uint32_t best_idx = (uint32_t)secant_sr_rng_bounded(search, population->count);
    size_t tournament_idx;

    for (tournament_idx = 1u; tournament_idx < search->config.tournament_size; ++tournament_idx) {
        const uint32_t candidate_idx = (uint32_t)secant_sr_rng_bounded(search, population->count);

        if (population->individuals[candidate_idx].fitness.score > population->individuals[best_idx].fitness.score) {
            best_idx = candidate_idx;
        }
    }
    return best_idx;
}

static const SecantSRIndividual*
secant_sr_parent_select(
    SecantSRSearch search,
    const SecantSRPopulation* population,
    uint32_t* population_idx_ret
) {
    if (search->config.archive_parent_probability != 0.0 && search->num_occupied_cells != 0u &&
        secant_sr_rng_unit(search) < search->config.archive_parent_probability) {
        const size_t occupied_idx = secant_sr_rng_bounded(search, search->num_occupied_cells);
        const size_t cell_idx = search->occupied_cells[occupied_idx];
        const size_t first_slot = cell_idx * search->config.elites_per_bucket;
        const SecantSREliteSlot* best = search->elites + first_slot;
        size_t slot_idx;

        for (slot_idx = 1u; slot_idx < search->config.elites_per_bucket; ++slot_idx) {
            const SecantSREliteSlot* candidate = search->elites + first_slot + slot_idx;

            if (!candidate->valid) {
                break;
            }
            if (candidate->individual.fitness.score > best->individual.fitness.score) {
                best = candidate;
            }
        }

        *population_idx_ret = SECANT_SR_PARENT_NONE;
        return &best->individual;
    }
    *population_idx_ret = secant_sr_tournament_select(search, population);
    return population->individuals + *population_idx_ret;
}

static SecantSREliteSlot*
secant_sr_best_unselected_elite(SecantSRSearch search, uint32_t selection_generation) {
    SecantSREliteSlot* best = NULL;
    size_t slot_idx;

    for (slot_idx = 0u; slot_idx < search->num_elite_slots; ++slot_idx) {
        SecantSREliteSlot* slot = search->elites + slot_idx;

        if (slot->valid && slot->selected_generation != selection_generation &&
            (best == NULL || slot->individual.fitness.score > best->individual.fitness.score)) {
            best = slot;
        }
    }
    return best;
}

static SecantSRResult
secant_sr_elites_copy_to_next(SecantSRSearch search, SecantSRPopulation* next_population) {
    const uint32_t selection_generation = search->generation + 1u;
    size_t elite_idx;

    for (elite_idx = 0u; elite_idx < search->config.elite_copies_per_generation; ++elite_idx) {
        SecantSREliteSlot* elite = secant_sr_best_unselected_elite(search, selection_generation);

        if (elite == NULL) {
            break;
        }
        elite->selected_generation = selection_generation;
        SecantSRIndividual* destination;

        _SECANT_SR_CHECK_RET(secant_sr_population_append_program(
            search,
            next_population,
            elite->individual.program,
            elite->individual.program_bytes,
            SECANT_SR_ORIGIN_ELITE,
            SECANT_SR_PARENT_NONE,
            SECANT_SR_PARENT_NONE,
            &destination));
        destination->fitness.constant_setting_robustness =
            elite->individual.fitness.constant_setting_robustness;
        destination->maturity = elite->individual.maturity;
        destination->dynamic_leaf_refinements = elite->individual.dynamic_leaf_refinements;
        destination->constant_refinements = elite->individual.constant_refinements;
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_generation_advance(SecantSRSearch search) {
    SecantSRPopulation* current;
    SecantSRPopulation* next;
    size_t individual_idx;

    if (search == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    current = &search->populations[search->current_population];
    for (individual_idx = 0u; individual_idx < current->count; ++individual_idx) {
        if (!current->individuals[individual_idx].scored) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_NOT_SCORED);
        }
    }
    next = &search->populations[1u - search->current_population];
    next->program_used = 0u;
    next->node_used = 0u;
    next->count = 0u;
    _SECANT_SR_CHECK_RET(secant_sr_elites_copy_to_next(search, next));

    while (next->count < search->config.population_size) {
        const double choice = secant_sr_rng_unit(search);
        SecantSRResult result = SECANT_SR_ERROR_SEARCH_STALLED;
        size_t attempt;

        for (attempt = 0u; attempt < 16u && result != SECANT_SR_SUCCESS; ++attempt) {
            uint32_t parent_a_idx;
            const SecantSRIndividual* parent_a = secant_sr_parent_select(search, current, &parent_a_idx);

            if (choice < search->config.crossover_probability) {
                uint32_t parent_b_idx;
                const SecantSRIndividual* parent_b = secant_sr_parent_select(search, current, &parent_b_idx);

                result = secant_sr_population_crossover_append(
                    search,
                    next,
                    parent_a,
                    parent_a_idx,
                    parent_b,
                    parent_b_idx);
            } else if (choice < search->config.crossover_probability +
                    search->config.subtree_mutation_probability) {
                result = secant_sr_population_subtree_mutation_append(search, next, parent_a, parent_a_idx);
            } else if (choice < search->config.crossover_probability +
                    search->config.subtree_mutation_probability + search->config.point_mutation_probability) {
                result = secant_sr_population_point_mutation_append(search, next, parent_a, parent_a_idx);
            } else {
                const uint32_t target_complexity = 1u + (uint32_t)secant_sr_rng_bounded(
                    search, search->config.max_complexity);

                result = secant_sr_population_random_append(
                    search,
                    next,
                    target_complexity,
                    SECANT_SR_ORIGIN_RANDOM,
                    parent_a_idx,
                    NULL);
            }
        }
        if (result != SECANT_SR_SUCCESS) {
            result = secant_sr_population_random_append(
                search,
                next,
                1u,
                SECANT_SR_ORIGIN_RANDOM,
                SECANT_SR_PARENT_NONE,
                NULL);
            if (result != SECANT_SR_SUCCESS) {
                return result;
            }
        }
    }

    search->current_population = 1u - search->current_population;
    ++search->generation;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_best_get(
    SecantSRSearch search,
    const SecantSRIndividual** individual_ret
) {
    const SecantSRIndividual* best = NULL;
    size_t slot_idx;

    if (search == NULL || individual_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (slot_idx = 0u; slot_idx < search->num_elite_slots; ++slot_idx) {
        const SecantSREliteSlot* slot = search->elites + slot_idx;

        if (slot->valid && (best == NULL || slot->individual.fitness.score > best->fitness.score)) {
            best = &slot->individual;
        }
    }
    if (best == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_NOT_SCORED);
    }
    *individual_ret = best;
    return SECANT_SR_SUCCESS;
}
