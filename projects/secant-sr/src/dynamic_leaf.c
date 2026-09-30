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
secant_sr_dynamic_leaf_transform(
    const SecantAstInstruction* program,
    size_t program_bytes,
    int materialize,
    uint32_t leaf_mask,
    const uint32_t* leaf_words,
    size_t num_input_columns,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* required_size_ret,
    size_t* num_leaves_ret
) {
    size_t input_offset = 0u;
    size_t output_offset = 0u;
    size_t num_leaves = 0u;
    int returned = 0;

    if (program == NULL || program_bytes == 0u || required_size_ret == NULL || num_leaves_ret == NULL ||
        (materialize && leaf_words == NULL)) {
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
        if (type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32 ||
            type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            if (num_leaves >= SECANT_AST_MAX_DYNAMIC_LEAVES) {
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
            }
            if (materialize) {
                const uint32_t word = leaf_words[num_leaves];

                encoded_size = ((leaf_mask >> num_leaves) & 1u) != 0u ? 2u : 5u;
                if (encoded_size == 2u && word >= num_input_columns) {
                    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
                }
                if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
                    if (encoded_size == 2u) {
                        output[output_offset] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32;
                        output[output_offset + 1u] = (uint8_t)word;
                    } else {
                        output[output_offset] = (uint8_t)SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32;
                        output[output_offset + 1u] = (uint8_t)(word >> 0u);
                        output[output_offset + 2u] = (uint8_t)(word >> 8u);
                        output[output_offset + 3u] = (uint8_t)(word >> 16u);
                        output[output_offset + 4u] = (uint8_t)(word >> 24u);
                    }
                }
            } else {
                encoded_size = 2u;
                if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
                    output[output_offset] =
                        (uint8_t)SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
                    output[output_offset + 1u] = (uint8_t)num_leaves;
                }
            }
            ++num_leaves;
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
        if (encoded_size > SIZE_MAX - output_offset) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        output_offset += encoded_size;
        input_offset += instruction_size;
        returned = type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
    }
    if (!returned || num_leaves == 0u) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    *required_size_ret = output_offset;
    *num_leaves_ret = num_leaves;
    if (output != NULL && output_size < output_offset) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_leaf_programs_write(
    SecantSRSearch search,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* num_asts_ret,
    size_t* max_dynamic_leaves_ret
) {
    const SecantSRPopulation* population;
    size_t required_size = 0u;
    size_t max_dynamic_leaves = 0u;
    size_t ast_idx;

    if (search == NULL || required_program_storage_ret == NULL || num_asts_ret == NULL ||
        max_dynamic_leaves_ret == NULL || ((program_storage == NULL) != (asts == NULL))) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    if (program_storage != NULL && ast_capacity < population->count) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (ast_idx = 0u; ast_idx < population->count; ++ast_idx) {
        const SecantSRIndividual* individual = population->individuals + ast_idx;
        size_t ast_size;
        size_t num_leaves;
        SecantAstInstruction* destination = program_storage != NULL ? program_storage + required_size : NULL;
        const size_t destination_size = program_storage != NULL && required_size <= program_storage_size
            ? program_storage_size - required_size
            : 0u;

        if (program_storage != NULL) {
            asts[ast_idx] = destination;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_transform(
            individual->program,
            individual->program_bytes,
            0,
            0u,
            NULL,
            search->config.num_inputs,
            destination,
            destination_size,
            &ast_size,
            &num_leaves));
        if (ast_size > SIZE_MAX - required_size) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        required_size += ast_size;
        if (num_leaves > max_dynamic_leaves) {
            max_dynamic_leaves = num_leaves;
        }
    }
    *required_program_storage_ret = required_size;
    *num_asts_ret = population->count;
    *max_dynamic_leaves_ret = max_dynamic_leaves;
    if (program_storage != NULL && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

static uint64_t
secant_sr_dynamic_leaf_mix64(uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static int
secant_sr_dynamic_leaf_source_type_is_eligible(
    SecantAstInstructionType type,
    SecantSRDynamicLeafProjection projection
) {
    return type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 ||
        (projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS &&
         type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32);
}

static uint16_t
secant_sr_refinement_count_increment(uint16_t value) {
    return value == UINT16_MAX ? value : (uint16_t)(value + 1u);
}

static SecantSRMaturity
secant_sr_dynamic_leaf_maturity_advance(SecantSRMaturity maturity) {
    if (maturity == SECANT_SR_MATURITY_EXPLORATORY) {
        return SECANT_SR_MATURITY_RESOLVED;
    }
    return maturity < SECANT_SR_MATURITY_MIXED_REFINED
        ? SECANT_SR_MATURITY_MIXED_REFINED
        : maturity;
}

static SecantSRResult
secant_sr_dynamic_leaf_offsets_select(
    const SecantSRIndividual* source,
    SecantSRDynamicLeafProjection projection,
    size_t max_dynamic_leaves,
    uint64_t projection_seed,
    uint32_t* offsets,
    size_t* num_offsets_ret
) {
    uint64_t scores[SECANT_AST_MAX_DYNAMIC_LEAVES];
    size_t num_eligible = 0u;
    size_t num_selected;
    size_t node_idx;

    if (source == NULL || max_dynamic_leaves == 0u ||
        max_dynamic_leaves > SECANT_AST_MAX_DYNAMIC_LEAVES || offsets == NULL || num_offsets_ret == NULL ||
        (projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_FULL &&
         projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED &&
         projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (node_idx = 0u; node_idx < source->num_nodes; ++node_idx) {
        const SecantSRNodeInfo* node = source->nodes + node_idx;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(
            source->program + node->instruction_offset);

        if (secant_sr_dynamic_leaf_source_type_is_eligible(type, projection)) {
            ++num_eligible;
        }
    }
    if (num_eligible == 0u ||
        (projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED && num_eligible > max_dynamic_leaves)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    num_selected = projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED
        ? num_eligible
        : (num_eligible > 1u
            ? (max_dynamic_leaves < num_eligible - 1u ? max_dynamic_leaves : num_eligible - 1u)
            : 1u);
    num_eligible = 0u;
    for (node_idx = 0u; node_idx < source->num_nodes; ++node_idx) {
        const SecantSRNodeInfo* node = source->nodes + node_idx;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(
            source->program + node->instruction_offset);

        if (secant_sr_dynamic_leaf_source_type_is_eligible(type, projection)) {
            if (projection != SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED) {
                offsets[num_eligible] = node->instruction_offset;
            } else {
                const uint64_t score = secant_sr_dynamic_leaf_mix64(
                    projection_seed ^ source->fingerprint ^
                    ((uint64_t)num_eligible + 1u) * UINT64_C(0x9e3779b97f4a7c15));

                if (num_eligible < num_selected) {
                    offsets[num_eligible] = node->instruction_offset;
                    scores[num_eligible] = score;
                } else {
                    size_t worst_idx = 0u;
                    size_t selected_idx;

                    for (selected_idx = 1u; selected_idx < num_selected; ++selected_idx) {
                        if (scores[selected_idx] > scores[worst_idx]) {
                            worst_idx = selected_idx;
                        }
                    }
                    if (score < scores[worst_idx]) {
                        offsets[worst_idx] = node->instruction_offset;
                        scores[worst_idx] = score;
                    }
                }
            }
            ++num_eligible;
        }
    }
    if (projection == SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED) {
        size_t idx;

        for (idx = 1u; idx < num_selected; ++idx) {
            const uint32_t offset = offsets[idx];
            size_t insertion = idx;

            while (insertion != 0u && offsets[insertion - 1u] > offset) {
                offsets[insertion] = offsets[insertion - 1u];
                --insertion;
            }
            offsets[insertion] = offset;
        }
    }
    *num_offsets_ret = num_selected;
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_dynamic_leaf_selected_transform(
    const SecantSRIndividual* source,
    SecantSRDynamicLeafProjection projection,
    size_t max_dynamic_leaves,
    uint64_t projection_seed,
    int materialize,
    uint32_t leaf_mask,
    const uint32_t* leaf_words,
    size_t num_input_columns,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* required_size_ret,
    size_t* num_dynamic_leaves_ret
) {
    uint32_t selected_offsets[SECANT_AST_MAX_DYNAMIC_LEAVES];
    size_t num_selected;
    size_t selected_idx = 0u;
    size_t input_offset = 0u;
    size_t output_offset = 0u;
    int returned = 0;

    if (source == NULL || required_size_ret == NULL || num_dynamic_leaves_ret == NULL ||
        (materialize && leaf_words == NULL)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_offsets_select(
        source, projection, max_dynamic_leaves, projection_seed, selected_offsets, &num_selected));
    while (input_offset < source->program_bytes) {
        const SecantAstInstruction* instruction = source->program + input_offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);
        const int selected = selected_idx < num_selected && selected_offsets[selected_idx] == input_offset;
        size_t encoded_size = instruction_size;

        if (instruction_size == 0u || instruction_size > source->program_bytes - input_offset || returned) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32 ||
            type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32 ||
            type == SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        if (selected) {
            const size_t dynamic_idx = selected_idx++;

            if (materialize) {
                const uint32_t word = leaf_words[dynamic_idx];
                const int is_column = ((leaf_mask >> dynamic_idx) & 1u) != 0u;

                encoded_size = is_column ? 2u : 5u;
                if (is_column && word >= num_input_columns) {
                    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
                }
                if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
                    output[output_offset] = (uint8_t)(is_column
                        ? SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32
                        : SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32);
                    output[output_offset + 1u] = (uint8_t)(word >> 0u);
                    if (!is_column) {
                        output[output_offset + 2u] = (uint8_t)(word >> 8u);
                        output[output_offset + 3u] = (uint8_t)(word >> 16u);
                        output[output_offset + 4u] = (uint8_t)(word >> 24u);
                    }
                }
            } else {
                encoded_size = 2u;
                if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
                    output[output_offset] =
                        (uint8_t)SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
                    output[output_offset + 1u] = (uint8_t)dynamic_idx;
                }
            }
        } else if (output != NULL && output_offset <= output_size && encoded_size <= output_size - output_offset) {
            memcpy(output + output_offset, instruction, encoded_size);
        }
        if (!secant_sr_checked_add(output_offset, encoded_size, &output_offset)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        input_offset += instruction_size;
        returned = type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
    }
    if (!returned || selected_idx != num_selected) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
    }
    *required_size_ret = output_offset;
    *num_dynamic_leaves_ret = num_selected;
    if (output != NULL && output_size < output_offset) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

static SecantSRResult
secant_sr_dynamic_leaf_population_indices_validate(
    const SecantSRPopulation* population,
    const uint32_t* population_indices,
    size_t num_selected
) {
    size_t idx;

    if (population == NULL || population_indices == NULL || num_selected == 0u) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (idx = 0u; idx < num_selected; ++idx) {
        if (population_indices[idx] >= population->count ||
            (idx != 0u && population_indices[idx - 1u] >= population_indices[idx])) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_leaf_selected_programs_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    SecantSRDynamicLeafProjection projection,
    size_t max_dynamic_leaves,
    uint64_t projection_seed,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* max_dynamic_leaves_ret
) {
    const SecantSRPopulation* population;
    size_t required_size = 0u;
    size_t maximum = 0u;
    size_t selected_idx;

    if (search == NULL || required_program_storage_ret == NULL || max_dynamic_leaves_ret == NULL ||
        ((program_storage == NULL) != (asts == NULL)) ||
        (program_storage == NULL && (program_storage_size != 0u || ast_capacity != 0u))) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_population_indices_validate(
        population, population_indices, num_selected));
    if (program_storage != NULL && ast_capacity < num_selected) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* source = population->individuals + population_indices[selected_idx];
        SecantAstInstruction* destination = program_storage != NULL && required_size <= program_storage_size
            ? program_storage + required_size
            : NULL;
        const size_t destination_size = destination != NULL ? program_storage_size - required_size : 0u;
        size_t program_size;
        size_t num_dynamic;

        if (program_storage != NULL) {
            asts[selected_idx] = destination;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_selected_transform(
            source,
            projection,
            max_dynamic_leaves,
            projection_seed,
            0,
            0u,
            NULL,
            search->config.num_inputs,
            destination,
            destination_size,
            &program_size,
            &num_dynamic));
        if (!secant_sr_checked_add(required_size, program_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
        if (num_dynamic > maximum) {
            maximum = num_dynamic;
        }
    }
    *required_program_storage_ret = required_size;
    *max_dynamic_leaves_ret = maximum;
    if (program_storage != NULL && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_leaf_selected_proposals_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    SecantSRDynamicLeafProjection projection,
    size_t max_dynamic_leaves,
    uint64_t projection_seed,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
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
    size_t required_leaf_words;
    size_t required_size = 0u;
    size_t selected_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL;

    if (search == NULL || leaf_masks == NULL || leaf_words == NULL || num_settings == 0u ||
        best_setting_indices == NULL || required_program_storage_ret == NULL ||
        leaf_masks_num_elements < num_settings || leaf_words_leading_dimension < max_dynamic_leaves ||
        (writing && (program_storage == NULL || asts == NULL || program_sizes == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        !secant_sr_checked_mul(num_settings - 1u, leaf_words_leading_dimension, &required_leaf_words) ||
        !secant_sr_checked_add(required_leaf_words, max_dynamic_leaves, &required_leaf_words) ||
        required_leaf_words > leaf_words_num_elements) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_population_indices_validate(
        population, population_indices, num_selected));
    if (writing && ast_capacity < num_selected) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* source = population->individuals + population_indices[selected_idx];
        const size_t setting = best_setting_indices[selected_idx];
        SecantAstInstruction* destination = writing && required_size <= program_storage_size
            ? program_storage + required_size
            : NULL;
        const size_t destination_size = destination != NULL ? program_storage_size - required_size : 0u;
        size_t program_size;
        size_t num_dynamic;

        if (setting >= num_settings) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_selected_transform(
            source,
            projection,
            max_dynamic_leaves,
            projection_seed,
            1,
            leaf_masks[setting],
            leaf_words + setting * leaf_words_leading_dimension,
            search->config.num_inputs,
            destination,
            destination_size,
            &program_size,
            &num_dynamic));
        (void)num_dynamic;
        if (writing) {
            asts[selected_idx] = destination;
            program_sizes[selected_idx] = program_size;
        }
        if (!secant_sr_checked_add(required_size, program_size, &required_size)) {
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
secant_sr_search_dynamic_leaf_selected_bindings_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    SecantSRDynamicLeafProjection projection,
    size_t max_dynamic_leaves,
    uint64_t projection_seed,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    size_t ast_capacity,
    size_t* required_program_storage_ret
) {
    const SecantSRPopulation* population;
    size_t required_leaf_words;
    size_t required_size = 0u;
    size_t selected_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL;

    if (search == NULL || leaf_masks == NULL || leaf_words == NULL || num_selected == 0u ||
        required_program_storage_ret == NULL || leaf_masks_num_elements < num_selected ||
        leaf_words_leading_dimension < max_dynamic_leaves ||
        (writing && (program_storage == NULL || asts == NULL || program_sizes == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        !secant_sr_checked_mul(num_selected - 1u, leaf_words_leading_dimension, &required_leaf_words) ||
        !secant_sr_checked_add(required_leaf_words, max_dynamic_leaves, &required_leaf_words) ||
        required_leaf_words > leaf_words_num_elements) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_population_indices_validate(
        population, population_indices, num_selected));
    if (writing && ast_capacity < num_selected) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
        const SecantSRIndividual* source = population->individuals + population_indices[selected_idx];
        SecantAstInstruction* destination = writing && required_size <= program_storage_size
            ? program_storage + required_size
            : NULL;
        const size_t destination_size = destination != NULL ? program_storage_size - required_size : 0u;
        size_t program_size;
        size_t num_dynamic;

        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_selected_transform(
            source,
            projection,
            max_dynamic_leaves,
            projection_seed,
            1,
            leaf_masks[selected_idx],
            leaf_words + selected_idx * leaf_words_leading_dimension,
            search->config.num_inputs,
            destination,
            destination_size,
            &program_size,
            &num_dynamic));
        (void)num_dynamic;
        if (writing) {
            asts[selected_idx] = destination;
            program_sizes[selected_idx] = program_size;
        }
        if (!secant_sr_checked_add(required_size, program_size, &required_size)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
        }
    }
    *required_program_storage_ret = required_size;
    if (writing && program_storage_size < required_size) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    return SECANT_SR_SUCCESS;
}

static size_t
secant_sr_dynamic_leaf_best_setting_get(
    const float* dynamic_sse,
    size_t num_settings
) {
    size_t best_setting = 0u;
    float best_sse = INFINITY;
    size_t setting;

    for (setting = 0u; setting < num_settings; ++setting) {
        const float candidate_sse = dynamic_sse[setting];

        if (isfinite(candidate_sse) && candidate_sse >= 0.0f && candidate_sse < best_sse) {
            best_sse = candidate_sse;
            best_setting = setting;
        }
    }
    return best_setting;
}

SecantSRResult
secant_sr_search_dynamic_leaf_proposals_write(
    SecantSRSearch search,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    const float* dynamic_sse,
    size_t dynamic_sse_num_elements,
    size_t dynamic_sse_leading_dimension,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* num_asts_ret
) {
    const SecantSRPopulation* population;
    size_t projected_program_size;
    size_t projected_num_asts;
    size_t max_dynamic_leaves;
    size_t required_leaf_words;
    size_t required_sse;
    size_t required_size = 0u;
    size_t ast_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL;

    if (search == NULL || leaf_masks == NULL || leaf_words == NULL || dynamic_sse == NULL || num_settings == 0u ||
        required_program_storage_ret == NULL || num_asts_ret == NULL ||
        (writing && (program_storage == NULL || asts == NULL || program_sizes == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        dynamic_sse_leading_dimension < num_settings) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_search_dynamic_leaf_programs_write(
        search,
        NULL,
        0u,
        NULL,
        0u,
        &projected_program_size,
        &projected_num_asts,
        &max_dynamic_leaves));
    (void)projected_program_size;
    if (leaf_masks_num_elements < num_settings || leaf_words_leading_dimension < max_dynamic_leaves ||
        !secant_sr_checked_mul(num_settings - 1u, leaf_words_leading_dimension, &required_leaf_words) ||
        !secant_sr_checked_add(required_leaf_words, max_dynamic_leaves, &required_leaf_words) ||
        required_leaf_words > leaf_words_num_elements ||
        !secant_sr_checked_mul(projected_num_asts - 1u, dynamic_sse_leading_dimension, &required_sse) ||
        !secant_sr_checked_add(required_sse, num_settings, &required_sse) ||
        required_sse > dynamic_sse_num_elements ||
        (writing && ast_capacity < projected_num_asts)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }

    population = &search->populations[search->current_population];
    for (ast_idx = 0u; ast_idx < population->count; ++ast_idx) {
        const SecantSRIndividual* source = population->individuals + ast_idx;
        const size_t best_setting = secant_sr_dynamic_leaf_best_setting_get(
            dynamic_sse + ast_idx * dynamic_sse_leading_dimension,
            num_settings);
        SecantAstInstruction* destination = NULL;
        size_t destination_size = 0u;
        size_t materialized_size;
        size_t num_leaves;

        if (writing && required_size <= program_storage_size) {
            destination = program_storage + required_size;
            destination_size = program_storage_size - required_size;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_transform(
            source->program,
            source->program_bytes,
            1,
            leaf_masks[best_setting],
            leaf_words + best_setting * leaf_words_leading_dimension,
            search->config.num_inputs,
            destination,
            destination_size,
            &materialized_size,
            &num_leaves));
        (void)num_leaves;
        if (writing) {
            asts[ast_idx] = destination;
            program_sizes[ast_idx] = materialized_size;
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
secant_sr_search_dynamic_leaf_proposals_from_settings_write(
    SecantSRSearch search,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    const uint32_t* best_setting_indices,
    size_t num_best_settings,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    size_t ast_capacity,
    size_t* required_program_storage_ret
) {
    const SecantSRPopulation* population;
    size_t projected_program_size;
    size_t projected_num_asts;
    size_t max_dynamic_leaves;
    size_t required_leaf_words;
    size_t required_size = 0u;
    size_t ast_idx;
    const int writing = program_storage != NULL || asts != NULL || program_sizes != NULL;

    if (search == NULL || leaf_masks == NULL || leaf_words == NULL || num_settings == 0u ||
        best_setting_indices == NULL || required_program_storage_ret == NULL ||
        (writing && (program_storage == NULL || asts == NULL || program_sizes == NULL)) ||
        (!writing && (program_storage_size != 0u || ast_capacity != 0u)) ||
        leaf_masks_num_elements < num_settings) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_search_dynamic_leaf_programs_write(
        search,
        NULL,
        0u,
        NULL,
        0u,
        &projected_program_size,
        &projected_num_asts,
        &max_dynamic_leaves));
    (void)projected_program_size;
    if (!secant_sr_checked_mul(num_settings - 1u, leaf_words_leading_dimension, &required_leaf_words) ||
        !secant_sr_checked_add(required_leaf_words, max_dynamic_leaves, &required_leaf_words)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_OVERFLOW);
    }
    population = &search->populations[search->current_population];
    if (num_best_settings != projected_num_asts || required_leaf_words > leaf_words_num_elements ||
        leaf_words_leading_dimension < max_dynamic_leaves ||
        (writing && ast_capacity < population->count)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    for (ast_idx = 0u; ast_idx < population->count; ++ast_idx) {
        const SecantSRIndividual* source = population->individuals + ast_idx;
        const size_t best_setting = best_setting_indices[ast_idx];
        SecantAstInstruction* destination = NULL;
        size_t destination_size = 0u;
        size_t materialized_size;
        size_t num_leaves;

        if (best_setting >= num_settings) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        if (writing && required_size <= program_storage_size) {
            destination = program_storage + required_size;
            destination_size = program_storage_size - required_size;
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_transform(
            source->program,
            source->program_bytes,
            1,
            leaf_masks[best_setting],
            leaf_words + best_setting * leaf_words_leading_dimension,
            search->config.num_inputs,
            destination,
            destination_size,
            &materialized_size,
            &num_leaves));
        (void)num_leaves;
        if (writing) {
            asts[ast_idx] = destination;
            program_sizes[ast_idx] = materialized_size;
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
secant_sr_search_selected_proposals_apply(
    SecantSRSearch search,
    const uint32_t* population_indices,
    const SecantAstInstruction* const* proposals,
    const size_t* proposal_program_sizes,
    const float* proposal_sse,
    const float* constant_setting_robustness,
    size_t num_proposals,
    size_t num_rows,
    double target_sum_squared_deviation,
    SecantSROrigin proposal_origin,
    size_t* num_promoted_ret
) {
    SecantSRPopulation* current;
    SecantSRPopulation* next;
    size_t num_promoted = 0u;
    size_t selected_idx = 0u;
    size_t ast_idx;

    if (search == NULL || population_indices == NULL || proposals == NULL || proposal_program_sizes == NULL ||
        proposal_sse == NULL || num_proposals == 0u || num_rows == 0u || target_sum_squared_deviation <= 0.0 ||
        proposal_origin < SECANT_SR_ORIGIN_RANDOM || proposal_origin > SECANT_SR_ORIGIN_DYNAMIC_CONSTANT) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    current = &search->populations[search->current_population];
    for (selected_idx = 0u; selected_idx < num_proposals; ++selected_idx) {
        if (population_indices[selected_idx] >= current->count ||
            (selected_idx != 0u && population_indices[selected_idx - 1u] >= population_indices[selected_idx]) ||
            proposals[selected_idx] == NULL || proposal_program_sizes[selected_idx] == 0u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        if (constant_setting_robustness != NULL &&
            (!isfinite(constant_setting_robustness[selected_idx]) || constant_setting_robustness[selected_idx] < 0.0f ||
             constant_setting_robustness[selected_idx] > 1.0f)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }
    for (ast_idx = 0u; ast_idx < current->count; ++ast_idx) {
        if (!current->individuals[ast_idx].scored) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_NOT_SCORED);
        }
    }

    next = &search->populations[1u - search->current_population];
    next->program_used = 0u;
    next->node_used = 0u;
    next->count = 0u;
    selected_idx = 0u;
    for (ast_idx = 0u; ast_idx < current->count; ++ast_idx) {
        const SecantSRIndividual* source = current->individuals + ast_idx;
        const int selected = selected_idx < num_proposals && population_indices[selected_idx] == ast_idx;
        const int promote = selected && isfinite(proposal_sse[selected_idx]) && proposal_sse[selected_idx] >= 0.0f &&
            (double)proposal_sse[selected_idx] < source->fitness.sse;
        SecantSRIndividual* destination;

        _SECANT_SR_CHECK_RET(secant_sr_population_append_program(
            search,
            next,
            promote ? proposals[selected_idx] : source->program,
            promote ? proposal_program_sizes[selected_idx] : source->program_bytes,
            promote ? proposal_origin : source->origin,
            promote ? (uint32_t)ast_idx : source->parent_a,
            promote ? SECANT_SR_PARENT_NONE : source->parent_b,
            &destination));
        if (promote) {
            destination->birth_generation = search->generation;
            if (proposal_origin == SECANT_SR_ORIGIN_DYNAMIC_LEAF) {
                destination->maturity = secant_sr_dynamic_leaf_maturity_advance(source->maturity);
                destination->dynamic_leaf_refinements = secant_sr_refinement_count_increment(
                    source->dynamic_leaf_refinements);
                destination->constant_refinements = source->constant_refinements;
            } else if (proposal_origin == SECANT_SR_ORIGIN_DYNAMIC_CONSTANT) {
                destination->maturity = SECANT_SR_MATURITY_CONSTANT_REFINED;
                destination->dynamic_leaf_refinements = source->dynamic_leaf_refinements;
                destination->constant_refinements = secant_sr_refinement_count_increment(
                    source->constant_refinements);
            }
            destination->fitness.constant_setting_robustness = constant_setting_robustness != NULL
                ? constant_setting_robustness[selected_idx]
                : 0.0;
            secant_sr_individual_score_set(
                search, destination, proposal_sse[selected_idx], num_rows, target_sum_squared_deviation);
            ++num_promoted;
        } else {
            const SecantAstInstruction* destination_program = destination->program;
            const SecantSRNodeInfo* destination_nodes = destination->nodes;

            *destination = *source;
            destination->program = destination_program;
            destination->nodes = destination_nodes;
            if (selected && constant_setting_robustness != NULL) {
                destination->fitness.constant_setting_robustness = constant_setting_robustness[selected_idx];
                secant_sr_individual_score_set(
                    search, destination, source->fitness.sse, num_rows, target_sum_squared_deviation);
            }
        }
        if (selected) {
            ++selected_idx;
        }
    }
    search->current_population = 1u - search->current_population;
    for (ast_idx = 0u; ast_idx < next->count; ++ast_idx) {
        _SECANT_SR_CHECK_RET(secant_sr_archive_consider(search, next->individuals + ast_idx));
    }
    if (num_promoted_ret != NULL) {
        *num_promoted_ret = num_promoted;
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_proposals_apply(
    SecantSRSearch search,
    const SecantAstInstruction* const* proposals,
    const size_t* proposal_program_sizes,
    const float* proposal_sse,
    const float* constant_setting_robustness,
    size_t num_proposals,
    size_t num_rows,
    double target_sum_squared_deviation,
    SecantSROrigin proposal_origin,
    size_t* num_promoted_ret
) {
    SecantSRPopulation* current;
    SecantSRPopulation* next;
    size_t num_promoted = 0u;
    size_t ast_idx;

    if (search == NULL || proposals == NULL || proposal_program_sizes == NULL || proposal_sse == NULL ||
        num_rows == 0u || target_sum_squared_deviation <= 0.0 ||
        proposal_origin < SECANT_SR_ORIGIN_RANDOM || proposal_origin > SECANT_SR_ORIGIN_DYNAMIC_CONSTANT) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    current = &search->populations[search->current_population];
    if (num_proposals != current->count) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    for (ast_idx = 0u; ast_idx < current->count; ++ast_idx) {
        if (!current->individuals[ast_idx].scored) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_NOT_SCORED);
        }
        if (proposals[ast_idx] == NULL || proposal_program_sizes[ast_idx] == 0u) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        if (constant_setting_robustness != NULL &&
            (!isfinite(constant_setting_robustness[ast_idx]) || constant_setting_robustness[ast_idx] < 0.0f ||
             constant_setting_robustness[ast_idx] > 1.0f)) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
    }

    next = &search->populations[1u - search->current_population];
    next->program_used = 0u;
    next->node_used = 0u;
    next->count = 0u;
    for (ast_idx = 0u; ast_idx < current->count; ++ast_idx) {
        const SecantSRIndividual* source = current->individuals + ast_idx;
        const int promote = isfinite(proposal_sse[ast_idx]) && proposal_sse[ast_idx] >= 0.0f &&
            (double)proposal_sse[ast_idx] < source->fitness.sse;
        SecantSRIndividual* destination;

        _SECANT_SR_CHECK_RET(secant_sr_population_append_program(
            search,
            next,
            promote ? proposals[ast_idx] : source->program,
            promote ? proposal_program_sizes[ast_idx] : source->program_bytes,
            promote ? proposal_origin : source->origin,
            promote ? (uint32_t)ast_idx : source->parent_a,
            promote ? SECANT_SR_PARENT_NONE : source->parent_b,
            &destination));
        if (promote) {
            destination->birth_generation = search->generation;
            if (proposal_origin == SECANT_SR_ORIGIN_DYNAMIC_LEAF) {
                destination->maturity = secant_sr_dynamic_leaf_maturity_advance(source->maturity);
                destination->dynamic_leaf_refinements = secant_sr_refinement_count_increment(
                    source->dynamic_leaf_refinements);
                destination->constant_refinements = source->constant_refinements;
            } else if (proposal_origin == SECANT_SR_ORIGIN_DYNAMIC_CONSTANT) {
                destination->maturity = SECANT_SR_MATURITY_CONSTANT_REFINED;
                destination->dynamic_leaf_refinements = source->dynamic_leaf_refinements;
                destination->constant_refinements = secant_sr_refinement_count_increment(
                    source->constant_refinements);
            }
            destination->fitness.constant_setting_robustness = constant_setting_robustness != NULL
                ? constant_setting_robustness[ast_idx]
                : 0.0;
            secant_sr_individual_score_set(
                search,
                destination,
                proposal_sse[ast_idx],
                num_rows,
                target_sum_squared_deviation);
            ++num_promoted;
        } else {
            const SecantAstInstruction* destination_program = destination->program;
            const SecantSRNodeInfo* destination_nodes = destination->nodes;

            *destination = *source;
            destination->program = destination_program;
            destination->nodes = destination_nodes;
            if (constant_setting_robustness != NULL) {
                destination->fitness.constant_setting_robustness = constant_setting_robustness[ast_idx];
                secant_sr_individual_score_set(
                    search,
                    destination,
                    source->fitness.sse,
                    num_rows,
                    target_sum_squared_deviation);
            }
        }
    }
    search->current_population = 1u - search->current_population;
    for (ast_idx = 0u; ast_idx < next->count; ++ast_idx) {
        _SECANT_SR_CHECK_RET(secant_sr_archive_consider(search, next->individuals + ast_idx));
    }
    if (num_promoted_ret != NULL) {
        *num_promoted_ret = num_promoted;
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_leaf_scores_apply(
    SecantSRSearch search,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    const float* sse,
    size_t sse_num_elements,
    size_t sse_leading_dimension,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    SecantSRPopulation* current;
    SecantSRPopulation* next;
    size_t required_program_storage;
    size_t num_asts;
    size_t max_dynamic_leaves;
    size_t required_leaf_words;
    size_t required_sse;
    size_t ast_idx;

    if (search == NULL || leaf_masks == NULL || leaf_words == NULL || sse == NULL || num_settings == 0u ||
        num_rows == 0u || target_sum_squared_deviation <= 0.0 || sse_leading_dimension < num_settings) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_search_dynamic_leaf_programs_write(
        search,
        NULL,
        0u,
        NULL,
        0u,
        &required_program_storage,
        &num_asts,
        &max_dynamic_leaves));
    (void)required_program_storage;
    if (leaf_masks_num_elements < num_settings || leaf_words_leading_dimension < max_dynamic_leaves ||
        !secant_sr_checked_mul(num_settings - 1u, leaf_words_leading_dimension, &required_leaf_words) ||
        !secant_sr_checked_add(required_leaf_words, max_dynamic_leaves, &required_leaf_words) ||
        required_leaf_words > leaf_words_num_elements ||
        !secant_sr_checked_mul(num_asts - 1u, sse_leading_dimension, &required_sse) ||
        !secant_sr_checked_add(required_sse, num_settings, &required_sse) || required_sse > sse_num_elements) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }

    current = &search->populations[search->current_population];
    next = &search->populations[1u - search->current_population];
    next->program_used = 0u;
    next->node_used = 0u;
    next->count = 0u;
    for (ast_idx = 0u; ast_idx < current->count; ++ast_idx) {
        const SecantSRIndividual* source = current->individuals + ast_idx;
        SecantAstInstruction materialized[SECANT_AST_MAX_PROGRAM_BYTES];
        SecantSRIndividual* destination;
        size_t best_setting = 0u;
        double best_sse = INFINITY;
        size_t materialized_size;
        size_t num_leaves;
        size_t setting;

        for (setting = 0u; setting < num_settings; ++setting) {
            const double candidate_sse = sse[ast_idx * sse_leading_dimension + setting];

            if (isfinite(candidate_sse) && candidate_sse >= 0.0 && candidate_sse < best_sse) {
                best_sse = candidate_sse;
                best_setting = setting;
            }
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_transform(
            source->program,
            source->program_bytes,
            1,
            leaf_masks[best_setting],
            leaf_words + best_setting * leaf_words_leading_dimension,
            search->config.num_inputs,
            materialized,
            sizeof(materialized),
            &materialized_size,
            &num_leaves));
        (void)num_leaves;
        _SECANT_SR_CHECK_RET(secant_sr_population_append_program(
            search,
            next,
            materialized,
            materialized_size,
            SECANT_SR_ORIGIN_DYNAMIC_LEAF,
            (uint32_t)ast_idx,
            SECANT_SR_PARENT_NONE,
            &destination));
        destination->birth_generation = search->generation;
        secant_sr_individual_score_set(
            search,
            destination,
            best_sse,
            num_rows,
            target_sum_squared_deviation);
    }
    search->current_population = 1u - search->current_population;
    for (ast_idx = 0u; ast_idx < next->count; ++ast_idx) {
        _SECANT_SR_CHECK_RET(secant_sr_archive_consider(search, next->individuals + ast_idx));
    }
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_dynamic_leaf_best_scores_apply(
    SecantSRSearch search,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    const uint32_t* best_setting_indices,
    const float* best_sse,
    size_t num_best,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    SecantSRPopulation* current;
    SecantSRPopulation* next;
    size_t required_program_storage;
    size_t num_asts;
    size_t max_dynamic_leaves;
    size_t required_leaf_words;
    size_t ast_idx;

    if (search == NULL || leaf_masks == NULL || leaf_words == NULL || num_settings == 0u ||
        best_setting_indices == NULL || best_sse == NULL || num_rows == 0u || target_sum_squared_deviation <= 0.0) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    _SECANT_SR_CHECK_RET(secant_sr_search_dynamic_leaf_programs_write(
        search,
        NULL,
        0u,
        NULL,
        0u,
        &required_program_storage,
        &num_asts,
        &max_dynamic_leaves));
    (void)required_program_storage;
    if (num_best != num_asts || leaf_masks_num_elements < num_settings ||
        leaf_words_leading_dimension < max_dynamic_leaves ||
        !secant_sr_checked_mul(num_settings - 1u, leaf_words_leading_dimension, &required_leaf_words) ||
        !secant_sr_checked_add(required_leaf_words, max_dynamic_leaves, &required_leaf_words) ||
        required_leaf_words > leaf_words_num_elements) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }

    current = &search->populations[search->current_population];
    next = &search->populations[1u - search->current_population];
    next->program_used = 0u;
    next->node_used = 0u;
    next->count = 0u;
    for (ast_idx = 0u; ast_idx < current->count; ++ast_idx) {
        const SecantSRIndividual* source = current->individuals + ast_idx;
        const size_t best_setting = best_setting_indices[ast_idx];
        SecantAstInstruction materialized[SECANT_AST_MAX_PROGRAM_BYTES];
        SecantSRIndividual* destination;
        size_t materialized_size;
        size_t num_leaves;

        if (best_setting >= num_settings || best_sse[ast_idx] < 0.0f) {
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
        }
        _SECANT_SR_CHECK_RET(secant_sr_dynamic_leaf_transform(
            source->program,
            source->program_bytes,
            1,
            leaf_masks[best_setting],
            leaf_words + best_setting * leaf_words_leading_dimension,
            search->config.num_inputs,
            materialized,
            sizeof(materialized),
            &materialized_size,
            &num_leaves));
        (void)num_leaves;
        _SECANT_SR_CHECK_RET(secant_sr_population_append_program(
            search,
            next,
            materialized,
            materialized_size,
            SECANT_SR_ORIGIN_DYNAMIC_LEAF,
            (uint32_t)ast_idx,
            SECANT_SR_PARENT_NONE,
            &destination));
        destination->birth_generation = search->generation;
        secant_sr_individual_score_set(
            search, destination, best_sse[ast_idx], num_rows, target_sum_squared_deviation);
    }
    search->current_population = 1u - search->current_population;
    for (ast_idx = 0u; ast_idx < next->count; ++ast_idx) {
        _SECANT_SR_CHECK_RET(secant_sr_archive_consider(search, next->individuals + ast_idx));
    }
    return SECANT_SR_SUCCESS;
}
