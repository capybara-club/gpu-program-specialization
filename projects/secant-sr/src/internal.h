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
#ifndef SECANT_SR_INTERNAL_H_INCLUDED
#define SECANT_SR_INTERNAL_H_INCLUDED

#include "secant_sr.h"
#include "secant_instructions.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SECANT_SR_PARENT_NONE UINT32_MAX

#define _SECANT_SR_ERROR_RET(result_) return (result_)
#define _SECANT_SR_CHECK_RET(expression_) \
    do { \
        const SecantSRResult secant_sr_check_result__ = (expression_); \
        if (secant_sr_check_result__ != SECANT_SR_SUCCESS) { \
            return secant_sr_check_result__; \
        } \
    } while (0)

typedef union SecantSRAlignment {
    void* pointer;
    long double long_double_value;
    uint64_t u64;
} SecantSRAlignment;

typedef struct SecantSRPopulation {
    SecantSRIndividual* individuals;
    const SecantAstInstruction** asts;
    SecantAstInstruction* programs;
    SecantSRNodeInfo* nodes;
    size_t program_capacity;
    size_t node_capacity;
    size_t program_used;
    size_t node_used;
    size_t count;
} SecantSRPopulation;

typedef struct SecantSREliteSlot {
    SecantSRIndividual individual;
    SecantAstInstruction* program;
    SecantSRNodeInfo* nodes;
    uint32_t selected_generation;
    int valid;
} SecantSREliteSlot;

struct SecantSRSearchImpl {
    SecantSRSearchConfig config;
    SecantAstInstructionType* unary_ops;
    SecantAstInstructionType* binary_ops;
    SecantSRRoutine* routines;
    const SecantAstInstruction** routine_programs;
    uint8_t* unary_routine_indices;
    uint8_t* binary_routine_indices;
    float* constants;
    size_t num_unary_ops;
    size_t num_binary_ops;
    size_t num_routines;
    size_t num_unary_routines;
    size_t num_binary_routines;
    size_t num_constants;
    size_t active_max_nodes;
    size_t active_max_leaves;
    SecantSRPopulation populations[2];
    SecantSREliteSlot* elites;
    size_t* occupied_cells;
    size_t num_elite_slots;
    size_t num_archive_cells;
    size_t num_occupied_cells;
    size_t num_occupied_elites;
    uint32_t generation;
    uint32_t current_population;
    uint64_t rng_state;
    uint64_t optimizer_rng_state;
};

int secant_sr_checked_add(size_t left, size_t right, size_t* result_ret);
int secant_sr_checked_mul(size_t left, size_t right, size_t* result_ret);
int secant_sr_layout_add(size_t count, size_t element_size, size_t* offset);
void* secant_sr_layout_take(void* storage, size_t count, size_t element_size, size_t* offset);
uint64_t secant_sr_rng_next(SecantSRSearch search);
size_t secant_sr_rng_bounded(SecantSRSearch search, size_t upper_bound);
double secant_sr_rng_unit(SecantSRSearch search);
double secant_sr_optimizer_rng_unit(SecantSRSearch search);

uint8_t secant_sr_instruction_arity(SecantAstInstructionType instruction_type);
uint16_t secant_sr_instruction_complexity(SecantAstInstructionType instruction_type);
int secant_sr_instruction_is_unary(SecantAstInstructionType instruction_type);
int secant_sr_instruction_is_binary(SecantAstInstructionType instruction_type);

SecantSRResult secant_sr_program_annotate(
    const SecantAstInstruction* program,
    size_t program_limit,
    const SecantSRRoutine* routines,
    size_t num_routines,
    size_t max_depth,
    uint32_t max_complexity,
    SecantSRNodeInfo* nodes,
    size_t node_capacity,
    size_t* program_bytes_ret,
    size_t* num_nodes_ret,
    size_t* num_leaves_ret,
    size_t* num_constants_ret,
    uint32_t* complexity_ret,
    uint16_t* depth_ret,
    uint64_t* fingerprint_ret
);

SecantSRResult secant_sr_population_append_program(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantAstInstruction* program,
    size_t program_bytes,
    SecantSROrigin origin,
    uint32_t parent_a,
    uint32_t parent_b,
    SecantSRIndividual** individual_ret
);

SecantSRResult secant_sr_population_random_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    uint32_t target_complexity,
    SecantSROrigin origin,
    uint32_t parent_a,
    SecantSRIndividual** individual_ret
);

SecantSRResult secant_sr_population_crossover_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantSRIndividual* parent_a,
    uint32_t parent_a_idx,
    const SecantSRIndividual* parent_b,
    uint32_t parent_b_idx
);

SecantSRResult secant_sr_population_subtree_mutation_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantSRIndividual* parent,
    uint32_t parent_idx
);

SecantSRResult secant_sr_population_point_mutation_append(
    SecantSRSearch search,
    SecantSRPopulation* population,
    const SecantSRIndividual* parent,
    uint32_t parent_idx
);

uint16_t secant_sr_complexity_bucket_get(SecantSRSearch search, uint32_t complexity);
uint16_t secant_sr_column_count_bucket_get(SecantSRSearch search, uint16_t num_unique_columns);
uint16_t secant_sr_transcendental_count_bucket_get(SecantSRSearch search, uint16_t num_operations);
SecantSRResult secant_sr_archive_consider(SecantSRSearch search, const SecantSRIndividual* individual);
void secant_sr_individual_score_set(
    SecantSRSearch search,
    SecantSRIndividual* individual,
    double sse,
    size_t num_rows,
    double target_sum_squared_deviation
);

#endif /* SECANT_SR_INTERNAL_H_INCLUDED */
