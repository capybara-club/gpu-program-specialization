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
secant_sr_dynamic_constant_transform(
    const SecantAstInstruction* program,
    size_t program_bytes,
    size_t num_constants_to_promote,
    int materialize,
    const float* constant_settings,
    size_t constant_settings_leading_dimension,
    size_t setting,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* required_size_ret,
    size_t* num_constants_ret
) {
    size_t input_offset = 0u;
    size_t output_offset = 0u;
    size_t num_constants = 0u;
    int returned = 0;

    if (program == NULL || program_bytes == 0u || required_size_ret == NULL || num_constants_ret == NULL ||
        (materialize && constant_settings == NULL)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    while (input_offset < program_bytes) {
        const SecantAstInstruction* instruction = program + input_offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);
        size_t encoded_size = instruction_size;

        if (instruction_size == 0u || instruction_size > program_bytes - input_offset || returned) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 && num_constants < num_constants_to_promote) {
            uint32_t bits;

            encoded_size = materialize ? 5u : 2u;
            if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
                if (materialize) {
                    const float value = constant_settings[
                        num_constants * constant_settings_leading_dimension + setting];

                    memcpy(&bits, &value, sizeof(bits));
                    output[output_offset] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32;
                    output[output_offset + 1u] = (uint8_t)(bits >> 0u);
                    output[output_offset + 2u] = (uint8_t)(bits >> 8u);
                    output[output_offset + 3u] = (uint8_t)(bits >> 16u);
                    output[output_offset + 4u] = (uint8_t)(bits >> 24u);
                } else {
                    output[output_offset] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32;
                    output[output_offset + 1u] = (uint8_t)num_constants;
                }
            }
        } else {
            if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 ||
                type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 ||
                type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
                memcpy(output + output_offset, instruction, encoded_size);
            }
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            ++num_constants;
        }
        if (encoded_size > SIZE_MAX - output_offset) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        output_offset += encoded_size;
        input_offset += instruction_size;
        returned = type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
    }
    if (!returned) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    *required_size_ret = output_offset;
    *num_constants_ret = num_constants;
    if (output != NULL && output_size < output_offset) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_dynamic_constant_selection_validate(
    const SecantSRPopulation* population,
    const uint32_t* population_indices,
    size_t num_selected,
    size_t dynamic_constant_capacity
) {
    size_t selected_idx;

    if (population == NULL || population_indices == NULL || num_selected == 0u || dynamic_constant_capacity == 0u ||
        dynamic_constant_capacity > SECANT_AST_MAX_DYNAMIC_LEAVES) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const size_t population_idx = population_indices[selected_idx];

        if (population_idx >= population->count ||
            (selected_idx != 0u && population_indices[selected_idx - 1u] >= population_idx) ||
            population->individuals[population_idx].num_constants == 0u ||
            population->individuals[population_idx].num_constants > dynamic_constant_capacity) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_constant_indices_sample(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    double selection_probability,
    uint32_t* population_indices,
    size_t index_capacity,
    size_t* num_eligible_ret,
    size_t* num_selected_ret
) {
    const SecantSRPopulation* population;
    size_t num_eligible = 0u;
    size_t num_selected = 0u;
    size_t population_idx;

    if (search == NULL || dynamic_constant_capacity == 0u ||
        dynamic_constant_capacity > SECANT_AST_MAX_DYNAMIC_LEAVES || selection_probability < 0.0 ||
        selection_probability > 1.0 || !isfinite(selection_probability) || population_indices == NULL ||
        num_eligible_ret == NULL || num_selected_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    if (population->count > UINT32_MAX) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    for (population_idx = 0u; population_idx < population->count; ++population_idx) {
        const size_t num_constants = population->individuals[population_idx].num_constants;
        int selected;

        if (num_constants == 0u || num_constants > dynamic_constant_capacity) {
            continue;
        }
        ++num_eligible;
        selected = selection_probability >= 1.0 ||
            (selection_probability > 0.0 && secant_sr_optimizer_rng_unit(search) < selection_probability);
        if (!selected) {
            continue;
        }
        if (num_selected == index_capacity) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
        }
        population_indices[num_selected++] = (uint32_t)population_idx;
    }
    *num_eligible_ret = num_eligible;
    *num_selected_ret = num_selected;
    return SECANT_SR_SUCCESS;
}

static int
secant_sr_dynamic_constant_candidate_better(
    const SecantSRIndividual* individuals,
    uint32_t left_idx,
    uint32_t right_idx
) {
    const double left_score = individuals[left_idx].fitness.score;
    const double right_score = individuals[right_idx].fitness.score;

    return left_score > right_score || (left_score == right_score && left_idx < right_idx);
}

static void
secant_sr_dynamic_constant_heap_sift_up(
    const SecantSRIndividual* individuals,
    uint32_t* heap,
    size_t child
) {
    while (child != 0u) {
        const size_t parent = (child - 1u) / 2u;

        if (!secant_sr_dynamic_constant_candidate_better(individuals, heap[parent], heap[child])) {
            break;
        }
        {
            const uint32_t temporary = heap[parent];

            heap[parent] = heap[child];
            heap[child] = temporary;
        }
        child = parent;
    }
}

static void
secant_sr_dynamic_constant_heap_sift_down(
    const SecantSRIndividual* individuals,
    uint32_t* heap,
    size_t count,
    size_t parent
) {
    for (;;) {
        const size_t left = parent * 2u + 1u;
        size_t worse;

        if (left >= count) {
            break;
        }
        worse = left;
        if (left + 1u < count &&
            secant_sr_dynamic_constant_candidate_better(individuals, heap[left], heap[left + 1u])) {
            worse = left + 1u;
        }
        if (!secant_sr_dynamic_constant_candidate_better(individuals, heap[parent], heap[worse])) {
            break;
        }
        {
            const uint32_t temporary = heap[parent];

            heap[parent] = heap[worse];
            heap[worse] = temporary;
        }
        parent = worse;
    }
}

static int
secant_sr_dynamic_constant_index_compare(const void* left, const void* right) {
    const uint32_t left_idx = *(const uint32_t*)left;
    const uint32_t right_idx = *(const uint32_t*)right;

    return left_idx < right_idx ? -1 : left_idx != right_idx;
}

SecantSRResult
secant_sr_search_dynamic_constant_indices_select(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    size_t selection_budget,
    double random_fraction,
    uint32_t* population_indices,
    size_t index_capacity,
    size_t* num_eligible_ret,
    size_t* num_selected_ret
) {
    const SecantSRPopulation* population;
    size_t num_eligible = 0u;
    size_t quality_budget;
    size_t num_selected = 0u;
    size_t population_idx;
    size_t bucket_idx;

    if (search == NULL || dynamic_constant_capacity == 0u ||
        dynamic_constant_capacity > SECANT_AST_MAX_DYNAMIC_LEAVES || selection_budget == 0u ||
        random_fraction < 0.0 || random_fraction > 1.0 || !isfinite(random_fraction) ||
        population_indices == NULL || num_eligible_ret == NULL || num_selected_ret == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    if (population->count > UINT32_MAX) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    for (population_idx = 0u; population_idx < population->count; ++population_idx) {
        const SecantSRIndividual* individual = population->individuals + population_idx;

        if (individual->num_constants == 0u || individual->num_constants > dynamic_constant_capacity) {
            continue;
        }
        if (!individual->scored) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_NOT_SCORED);
        }
        ++num_eligible;
    }
    *num_eligible_ret = num_eligible;
    if (num_eligible == 0u) {
        *num_selected_ret = 0u;
        return SECANT_SR_SUCCESS;
    }
    if (selection_budget > num_eligible) {
        selection_budget = num_eligible;
    }
    if (index_capacity < selection_budget) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    if (selection_budget == num_eligible) {
        for (population_idx = 0u; population_idx < population->count; ++population_idx) {
            const SecantSRIndividual* individual = population->individuals + population_idx;

            if (individual->num_constants != 0u && individual->num_constants <= dynamic_constant_capacity) {
                population_indices[num_selected++] = (uint32_t)population_idx;
            }
        }
        *num_selected_ret = num_selected;
        return SECANT_SR_SUCCESS;
    }

    quality_budget = selection_budget - (size_t)((double)selection_budget * random_fraction);
    for (bucket_idx = 0u; bucket_idx < search->config.num_complexity_buckets; ++bucket_idx) {
        const size_t bucket_quota = quality_budget / search->config.num_complexity_buckets +
            (bucket_idx < quality_budget % search->config.num_complexity_buckets);
        uint32_t* heap = population_indices + num_selected;
        size_t heap_count = 0u;

        for (population_idx = 0u; population_idx < population->count; ++population_idx) {
            const SecantSRIndividual* individual = population->individuals + population_idx;

            if (individual->complexity_bucket != bucket_idx || individual->num_constants == 0u ||
                individual->num_constants > dynamic_constant_capacity || bucket_quota == 0u) {
                continue;
            }
            if (heap_count < bucket_quota) {
                heap[heap_count] = (uint32_t)population_idx;
                secant_sr_dynamic_constant_heap_sift_up(population->individuals, heap, heap_count);
                ++heap_count;
            } else if (secant_sr_dynamic_constant_candidate_better(
                           population->individuals, (uint32_t)population_idx, heap[0])) {
                heap[0] = (uint32_t)population_idx;
                secant_sr_dynamic_constant_heap_sift_down(population->individuals, heap, heap_count, 0u);
            }
        }
        num_selected += heap_count;
    }

    qsort(population_indices, num_selected, sizeof(*population_indices), secant_sr_dynamic_constant_index_compare);
    if (num_selected < selection_budget) {
        const size_t random_budget = selection_budget - num_selected;
        size_t quality_idx = 0u;
        size_t random_seen = 0u;
        size_t random_count = 0u;

        for (population_idx = 0u; population_idx < population->count; ++population_idx) {
            const SecantSRIndividual* individual = population->individuals + population_idx;
            size_t replacement;

            if (individual->num_constants == 0u || individual->num_constants > dynamic_constant_capacity) {
                continue;
            }
            while (quality_idx < num_selected && population_indices[quality_idx] < population_idx) {
                ++quality_idx;
            }
            if (quality_idx < num_selected && population_indices[quality_idx] == population_idx) {
                continue;
            }
            ++random_seen;
            if (random_count < random_budget) {
                population_indices[num_selected + random_count++] = (uint32_t)population_idx;
                continue;
            }
            replacement = (size_t)(secant_sr_optimizer_rng_unit(search) * (double)random_seen);
            if (replacement < random_budget) {
                population_indices[num_selected + replacement] = (uint32_t)population_idx;
            }
        }
        num_selected += random_count;
        qsort(population_indices, num_selected, sizeof(*population_indices), secant_sr_dynamic_constant_index_compare);
    }
    *num_selected_ret = num_selected;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_constant_programs_selected_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    size_t dynamic_constant_capacity,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* max_dynamic_constants_ret
) {
    const SecantSRPopulation* population;
    size_t required_size = 0u;
    size_t max_dynamic_constants = 0u;
    size_t selected_idx;

    if (search == NULL || required_program_storage_ret == NULL || max_dynamic_constants_ret == NULL ||
        ((program_storage == NULL) != (asts == NULL))) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_selection_validate(
        population, population_indices, num_selected, dynamic_constant_capacity));
    if (program_storage != NULL && ast_capacity < num_selected) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* individual = population->individuals + population_indices[selected_idx];
        SecantAstInstruction* destination = program_storage != NULL ? program_storage + required_size : NULL;
        const size_t destination_size = program_storage != NULL && required_size <= program_storage_size
            ? program_storage_size - required_size
            : 0u;
        size_t ast_size;
        size_t num_constants;

        if (program_storage != NULL) {
            asts[selected_idx] = destination;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_transform(
            individual->program,
            individual->program_bytes,
            individual->num_constants,
            0,
            NULL,
            0u,
            0u,
            destination,
            destination_size,
            &ast_size,
            &num_constants));
        if (num_constants != individual->num_constants || !secant_sr_checked_add(required_size, ast_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (num_constants > max_dynamic_constants) {
            max_dynamic_constants = num_constants;
        }
    }
    *required_program_storage_ret = required_size;
    *max_dynamic_constants_ret = max_dynamic_constants;
    if (program_storage != NULL && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_constant_optimizer_centers_selected_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    size_t dynamic_constant_capacity,
    float* current_constants,
    size_t current_constants_num_elements,
    size_t current_constants_leading_dimension
) {
    const SecantSRPopulation* population;
    size_t required_elements;
    size_t selected_idx;

    if (search == NULL || current_constants == NULL ||
        current_constants_leading_dimension < dynamic_constant_capacity ||
        !secant_sr_checked_mul(num_selected, current_constants_leading_dimension, &required_elements) ||
        required_elements > current_constants_num_elements) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_selection_validate(
        population, population_indices, num_selected, dynamic_constant_capacity));
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* individual = population->individuals + population_indices[selected_idx];
        float* destination = current_constants + selected_idx * current_constants_leading_dimension;
        size_t instruction_offset = 0u;
        size_t constant_idx = 0u;

        memset(destination, 0, dynamic_constant_capacity * sizeof(*destination));
        while (instruction_offset < individual->program_bytes) {
            const SecantAstInstruction* instruction = individual->program + instruction_offset;
            const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
            const size_t instruction_size = secant_ast_instruction_size_get(instruction);

            if (instruction_size == 0u || instruction_size > individual->program_bytes - instruction_offset) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            if (type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
                if (constant_idx >= dynamic_constant_capacity) {
                    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
                }
                destination[constant_idx++] = secant_ast_constant_f32_get(instruction);
            }
            instruction_offset += instruction_size;
            if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                break;
            }
        }
        if (constant_idx != individual->num_constants) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_constant_optimizer_proposals_selected_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    size_t dynamic_constant_capacity,
    const float* current_constants,
    size_t current_constants_num_elements,
    size_t current_constants_leading_dimension,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    size_t ast_capacity,
    size_t* required_program_storage_ret
) {
    const SecantSRPopulation* population;
    size_t required_constants;
    size_t required_size = 0u;
    size_t selected_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL;

    if (search == NULL || current_constants == NULL || required_program_storage_ret == NULL ||
        current_constants_leading_dimension < dynamic_constant_capacity ||
        (writing && (program_storage == NULL || asts == NULL || program_sizes == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        !secant_sr_checked_mul(num_selected, current_constants_leading_dimension, &required_constants)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    if (required_constants > current_constants_num_elements || (writing && ast_capacity < num_selected)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_selection_validate(
        population, population_indices, num_selected, dynamic_constant_capacity));
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* source = population->individuals + population_indices[selected_idx];
        SecantAstInstruction* destination = NULL;
        size_t destination_size = 0u;
        size_t materialized_size;
        size_t num_constants;

        if (writing && required_size <= program_storage_size) {
            destination = program_storage + required_size;
            destination_size = program_storage_size - required_size;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_transform(
            source->program,
            source->program_bytes,
            source->num_constants,
            1,
            current_constants + selected_idx * current_constants_leading_dimension,
            1u,
            0u,
            destination,
            destination_size,
            &materialized_size,
            &num_constants));
        if (num_constants != source->num_constants) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (writing) {
            asts[selected_idx] = destination;
            program_sizes[selected_idx] = materialized_size;
        }
        if (!secant_sr_checked_add(required_size, materialized_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
    }
    *required_program_storage_ret = required_size;
    if (writing && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_constant_programs_write(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* num_asts_ret,
    size_t* max_dynamic_constants_ret
) {
    const SecantSRPopulation* population;
    size_t required_size = 0u;
    size_t max_dynamic_constants = 0u;
    size_t ast_idx;

    if (search == NULL || dynamic_constant_capacity == 0u ||
        dynamic_constant_capacity > SECANT_AST_MAX_DYNAMIC_LEAVES ||
        required_program_storage_ret == NULL || num_asts_ret == NULL ||
        max_dynamic_constants_ret == NULL || ((program_storage == NULL) != (asts == NULL))) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    if (program_storage != NULL && ast_capacity < population->count) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (ast_idx = 0u; ast_idx < population->count; ++ast_idx) {
        const SecantSRIndividual* individual = population->individuals + ast_idx;
        const size_t num_constants_to_promote = individual->num_constants <= dynamic_constant_capacity
            ? individual->num_constants
            : 0u;
        SecantAstInstruction* destination = program_storage != NULL ? program_storage + required_size : NULL;
        const size_t destination_size = program_storage != NULL && required_size <= program_storage_size
            ? program_storage_size - required_size
            : 0u;
        size_t ast_size;
        size_t num_constants;

        if (program_storage != NULL) {
            asts[ast_idx] = destination;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_transform(
            individual->program,
            individual->program_bytes,
            num_constants_to_promote,
            0,
            NULL,
            0u,
            0u,
            destination,
            destination_size,
            &ast_size,
            &num_constants));
        if (!secant_sr_checked_add(required_size, ast_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        if (num_constants != individual->num_constants) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (num_constants_to_promote > max_dynamic_constants) {
            max_dynamic_constants = num_constants_to_promote;
        }
    }
    *required_program_storage_ret = required_size;
    *num_asts_ret = population->count;
    *max_dynamic_constants_ret = max_dynamic_constants;
    if (program_storage != NULL && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

static size_t
secant_sr_dynamic_constant_best_setting_get(
    const float* dynamic_sse,
    size_t num_settings,
    double target_sum_squared_deviation,
    int has_constants,
    float* robustness_ret
) {
    size_t best_setting = 0u;
    float best_sse = INFINITY;
    double robustness_sum = 0.0;
    size_t setting;

    for (setting = 0u; setting < num_settings; ++setting) {
        const float candidate_sse = dynamic_sse[setting];

        if (isfinite(candidate_sse) && candidate_sse >= 0.0f) {
            double r2 = 1.0 - (double)candidate_sse / target_sum_squared_deviation;

            if (candidate_sse < best_sse) {
                best_sse = candidate_sse;
                best_setting = setting;
            }
            if (r2 < 0.0) {
                r2 = 0.0;
            } else if (r2 > 1.0) {
                r2 = 1.0;
            }
            robustness_sum += r2;
        }
    }
    *robustness_ret = has_constants ? (float)(robustness_sum / (double)num_settings) : 0.0f;
    return best_setting;
}

SecantSRResult
secant_sr_search_dynamic_constant_proposals_write(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    const float* constant_settings,
    size_t constant_settings_num_elements,
    size_t constant_settings_leading_dimension,
    size_t num_settings,
    const float* dynamic_sse,
    size_t dynamic_sse_num_elements,
    size_t dynamic_sse_leading_dimension,
    double target_sum_squared_deviation,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    float* constant_setting_robustness,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* num_asts_ret
) {
    const SecantSRPopulation* population;
    size_t projected_program_size;
    size_t projected_num_asts;
    size_t max_dynamic_constants;
    size_t required_constants = 0u;
    size_t required_sse;
    size_t required_size = 0u;
    size_t ast_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL ||
        constant_setting_robustness != NULL;

    if (search == NULL || dynamic_constant_capacity == 0u ||
        dynamic_constant_capacity > SECANT_AST_MAX_DYNAMIC_LEAVES ||
        constant_settings == NULL || dynamic_sse == NULL || num_settings == 0u ||
        target_sum_squared_deviation <= 0.0 || required_program_storage_ret == NULL || num_asts_ret == NULL ||
        (writing &&
         (program_storage == NULL || asts == NULL || program_sizes == NULL || constant_setting_robustness == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        constant_settings_leading_dimension < num_settings || dynamic_sse_leading_dimension < num_settings) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_search_dynamic_constant_programs_write(
        search,
        dynamic_constant_capacity,
        NULL,
        0u,
        NULL,
        0u,
        &projected_program_size,
        &projected_num_asts,
        &max_dynamic_constants));
    (void)projected_program_size;
    if (max_dynamic_constants != 0u &&
        (!secant_sr_checked_mul(max_dynamic_constants - 1u, constant_settings_leading_dimension, &required_constants) ||
         !secant_sr_checked_add(required_constants, num_settings, &required_constants))) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    if (required_constants > constant_settings_num_elements ||
        !secant_sr_checked_mul(projected_num_asts - 1u, dynamic_sse_leading_dimension, &required_sse) ||
        !secant_sr_checked_add(required_sse, num_settings, &required_sse) || required_sse > dynamic_sse_num_elements ||
        (writing && ast_capacity < projected_num_asts)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }

    population = &search->populations[search->current_population];
    for (ast_idx = 0u; ast_idx < population->count; ++ast_idx) {
        const SecantSRIndividual* source = population->individuals + ast_idx;
        const size_t num_constants_to_promote = source->num_constants <= dynamic_constant_capacity
            ? source->num_constants
            : 0u;
        SecantAstInstruction* destination = NULL;
        size_t destination_size = 0u;
        size_t projected_size;
        size_t materialized_size;
        size_t num_constants;
        float robustness;
        size_t best_setting;

        _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_transform(
            source->program,
            source->program_bytes,
            num_constants_to_promote,
            0,
            NULL,
            0u,
            0u,
            NULL,
            0u,
            &projected_size,
            &num_constants));
        (void)projected_size;
        best_setting = secant_sr_dynamic_constant_best_setting_get(
            dynamic_sse + ast_idx * dynamic_sse_leading_dimension,
            num_settings,
            target_sum_squared_deviation,
            num_constants_to_promote != 0u,
            &robustness);

        if (writing && required_size <= program_storage_size) {
            destination = program_storage + required_size;
            destination_size = program_storage_size - required_size;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_transform(
            source->program,
            source->program_bytes,
            num_constants_to_promote,
            1,
            constant_settings,
            constant_settings_leading_dimension,
            best_setting,
            destination,
            destination_size,
            &materialized_size,
            &num_constants));
        if (writing) {
            asts[ast_idx] = destination;
            program_sizes[ast_idx] = materialized_size;
            constant_setting_robustness[ast_idx] = robustness;
        }
        if (!secant_sr_checked_add(required_size, materialized_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
    }
    *required_program_storage_ret = required_size;
    *num_asts_ret = population->count;
    if (writing && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_constant_proposals_selected_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    size_t dynamic_constant_capacity,
    const float* constant_settings,
    size_t constant_settings_num_elements,
    size_t constant_settings_leading_dimension,
    size_t num_settings,
    const uint32_t* best_setting_indices,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    size_t ast_capacity,
    size_t* required_program_storage_ret
) {
    const SecantSRPopulation* population;
    size_t required_constants;
    size_t required_size = 0u;
    size_t selected_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL;

    if (search == NULL || dynamic_constant_capacity == 0u ||
        dynamic_constant_capacity > SECANT_AST_MAX_DYNAMIC_LEAVES || constant_settings == NULL ||
        num_settings == 0u || best_setting_indices == NULL ||
        required_program_storage_ret == NULL || constant_settings_leading_dimension < num_settings ||
        (writing && (program_storage == NULL || asts == NULL || program_sizes == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        !secant_sr_checked_mul(dynamic_constant_capacity - 1u, constant_settings_leading_dimension, &required_constants) ||
        !secant_sr_checked_add(required_constants, num_settings, &required_constants)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    if (required_constants > constant_settings_num_elements || (writing && ast_capacity < num_selected)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_selection_validate(
        population, population_indices, num_selected, dynamic_constant_capacity));
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* source = population->individuals + population_indices[selected_idx];
        const size_t best_setting = best_setting_indices[selected_idx];
        SecantAstInstruction* destination = NULL;
        size_t destination_size = 0u;
        size_t materialized_size;
        size_t num_constants;

        if (best_setting >= num_settings) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        if (writing && required_size <= program_storage_size) {
            destination = program_storage + required_size;
            destination_size = program_storage_size - required_size;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_constant_transform(
            source->program,
            source->program_bytes,
            source->num_constants,
            1,
            constant_settings,
            constant_settings_leading_dimension,
            best_setting,
            destination,
            destination_size,
            &materialized_size,
            &num_constants));
        if (num_constants != source->num_constants) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (writing) {
            asts[selected_idx] = destination;
            program_sizes[selected_idx] = materialized_size;
        }
        if (!secant_sr_checked_add(required_size, materialized_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
    }
    *required_program_storage_ret = required_size;
    if (writing && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}
