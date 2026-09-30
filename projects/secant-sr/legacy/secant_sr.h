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
#ifndef SECANT_SR_H_INCLUDED
#define SECANT_SR_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_SR_DEC extern "C"
#else
#define SECANT_SR_DEC extern
#endif

typedef enum SecantSRResult {
    SECANT_SR_SUCCESS = 0,
    SECANT_SR_ERROR_INVALID_VALUE = 1,
    SECANT_SR_ERROR_OVERFLOW = 2,
    SECANT_SR_ERROR_INSUFFICIENT_BUFFER = 3,
    SECANT_SR_ERROR_BAD_PROGRAM = 4,
    SECANT_SR_ERROR_ARENA_EXHAUSTED = 5,
    SECANT_SR_ERROR_POPULATION_FULL = 6,
    SECANT_SR_ERROR_NOT_SCORED = 7,
    SECANT_SR_ERROR_SECANT = 8,
    SECANT_SR_ERROR_SEARCH_STALLED = 9,
    SECANT_SR_RESULT_NUM_ENUMS = 10
} SecantSRResult;

typedef enum SecantSROrigin {
    SECANT_SR_ORIGIN_RANDOM = 0,
    SECANT_SR_ORIGIN_CROSSOVER = 1,
    SECANT_SR_ORIGIN_SUBTREE_MUTATION = 2,
    SECANT_SR_ORIGIN_POINT_MUTATION = 3,
    SECANT_SR_ORIGIN_ELITE = 4,
    SECANT_SR_ORIGIN_DYNAMIC_LEAF = 5,
    SECANT_SR_ORIGIN_DYNAMIC_CONSTANT = 6
} SecantSROrigin;

/** Search-local maturity of a concrete population member. */
typedef enum SecantSRMaturity {
    SECANT_SR_MATURITY_EXPLORATORY = 0,
    SECANT_SR_MATURITY_RESOLVED = 1,
    SECANT_SR_MATURITY_MIXED_REFINED = 2,
    SECANT_SR_MATURITY_CONSTANT_REFINED = 3
} SecantSRMaturity;

/** Controls which concrete leaves are temporarily replaced by dynamic leaves. */
typedef enum SecantSRDynamicLeafProjection {
    SECANT_SR_DYNAMIC_LEAF_PROJECTION_FULL = 0,
    SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED = 1,
    SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS = 2
} SecantSRDynamicLeafProjection;

/** Metadata for one expression node in postorder instruction order. */
typedef struct SecantSRNodeInfo {
    uint32_t instruction_offset;
    uint32_t subtree_offset;
    uint32_t subtree_bytes;
    uint16_t subtree_first_node;
    uint16_t subtree_nodes;
    uint16_t depth;
    uint16_t complexity;
    uint8_t instruction_size;
    uint8_t arity;
} SecantSRNodeInfo;

typedef struct SecantSRFitness {
    double sse;
    double mse;
    double rmse;
    double nmse;
    double r2;
    double constant_setting_robustness;
    double score;
} SecantSRFitness;

/** Structural features computed while validating and annotating one program. */
typedef struct SecantSRProgramFeatures {
    uint64_t static_column_mask[2];
    uint64_t operation_mask;
    uint16_t num_unique_static_columns;
    uint16_t num_static_column_occurrences;
    uint16_t num_constants;
    uint16_t num_unary_operations;
    uint16_t num_binary_operations;
    uint16_t num_ternary_operations;
    uint16_t num_transcendental_operations;
} SecantSRProgramFeatures;

/** Immutable search vocabulary entry backed by a shared Secant routine program. */
typedef struct SecantSRRoutine {
    const SecantAstInstruction* program;
    const char* name;
    uint16_t complexity;
    uint8_t arity;
    uint8_t num_transcendental_operations;
} SecantSRRoutine;

/** A population member. Program and node pointers are borrowed from its generation arena. */
typedef struct SecantSRIndividual {
    const SecantAstInstruction* program;
    const SecantSRNodeInfo* nodes;
    const SecantSRRoutine* routines;
    size_t program_bytes;
    size_t num_nodes;
    size_t num_leaves;
    size_t num_routines;
    uint16_t num_constants;
    uint32_t complexity;
    uint16_t depth;
    uint16_t complexity_bucket;
    uint64_t fingerprint;
    uint32_t birth_generation;
    uint32_t parent_a;
    uint32_t parent_b;
    SecantSROrigin origin;
    SecantSRMaturity maturity;
    uint16_t dynamic_leaf_refinements;
    uint16_t constant_refinements;
    SecantSRFitness fitness;
    int scored;
    uint16_t column_count_bucket;
    uint16_t transcendental_count_bucket;
} SecantSRIndividual;

/** Persistent search dimensions and evolutionary policy. */
typedef struct SecantSRSearchConfig {
    size_t population_size;
    size_t max_program_bytes;
    size_t max_nodes;
    size_t max_depth;
    uint32_t max_complexity;
    size_t num_inputs;
    size_t num_complexity_buckets;
    size_t elites_per_bucket;
    size_t elite_copies_per_generation;
    size_t tournament_size;
    double complexity_factor;
    double crossover_probability;
    double subtree_mutation_probability;
    double point_mutation_probability;
    double constant_leaf_probability;
    double parsimony_coefficient;
    uint64_t seed;
    size_t initial_max_nodes;
    /** Initial leaf limit; zero starts at `max_nodes`. */
    size_t initial_max_leaves;
    /** QD buckets for unique static-column count. Zero preserves one legacy bucket. */
    size_t num_column_count_buckets;
    /** QD buckets for transcendental-operation count. Zero preserves one legacy bucket. */
    size_t num_transcendental_count_buckets;
    /** Probability that each variation parent is sampled uniformly from occupied archive cells. */
    double archive_parent_probability;
    /** Selection-score weight for mean clipped R2 over dynamic constant settings. */
    double constant_setting_credit_weight;
} SecantSRSearchConfig;

/** Current occupancy of the persistent quality-diversity archive. */
typedef struct SecantSRArchiveStats {
    size_t num_cells;
    size_t occupied_cells;
    size_t num_elites;
} SecantSRArchiveStats;

typedef struct SecantSRSearchImpl* SecantSRSearch;

SECANT_SR_DEC const char* secant_sr_result_to_string(SecantSRResult result);

/** Computes structural descriptors with one validated linear pass over a program. */
SECANT_SR_DEC SecantSRResult secant_sr_program_features_get(
    const SecantAstInstruction* program,
    size_t program_bytes,
    const SecantSRRoutine* routines,
    size_t num_routines,
    SecantSRProgramFeatures* features_ret
);

/**
 * Copies a program while replacing one leaf with an indexed dynamic leaf.
 *
 * The source program is never modified. This is required when replacing a
 * five-byte fixed constant because the resulting two-byte leaf changes every
 * subsequent instruction offset and invalidates the source annotations.
 */
SECANT_SR_DEC SecantSRResult secant_sr_program_leaf_replace_copy(
    const SecantAstInstruction* program,
    size_t program_bytes,
    size_t leaf_instruction_offset,
    SecantAstInstructionType replacement_type,
    SecantAstIdx replacement_idx,
    SecantAstInstruction* output,
    size_t output_size,
    size_t* required_size_ret
);

/** Measures exact caller-owned storage required by secant_sr_search_init(). */
SECANT_SR_DEC SecantSRResult secant_sr_search_storage_size(
    const SecantSRSearchConfig* config,
    size_t num_unary_ops,
    size_t num_binary_ops,
    size_t num_routines,
    size_t num_constants,
    size_t* storage_size_ret
);

/**
 * Initializes a search in caller-owned storage and seeds its first population.
 * Storage must have the alignment provided by malloc and remain alive for the
 * complete search lifetime.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_init(
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
);

SECANT_SR_DEC uint32_t secant_sr_search_generation_get(SecantSRSearch search);

SECANT_SR_DEC size_t secant_sr_search_population_size_get(SecantSRSearch search);

/** Returns the immutable routine programs shared by every candidate in the search. */
SECANT_SR_DEC SecantSRResult secant_sr_search_routines_get(
    SecantSRSearch search,
    const SecantAstInstruction* const** routines_ret,
    size_t* num_routines_ret
);

/**
 * Raises the active node limit used to create and admit future candidates.
 *
 * Search storage remains sized by `SecantSRSearchConfig.max_nodes`. The active
 * limit starts at `initial_max_nodes`, or `max_nodes` when that field is zero,
 * and may only increase up to the allocation maximum.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_active_max_nodes_set(
    SecantSRSearch search,
    size_t max_nodes
);

/** Returns the current active candidate node limit, or zero for NULL. */
SECANT_SR_DEC size_t secant_sr_search_active_max_nodes_get(SecantSRSearch search);

/**
 * Raises the active leaf limit used to create and admit future candidates.
 *
 * The limit starts at `initial_max_leaves`, or `max_nodes` when that field is
 * zero. It is independent of the active node limit so unary-heavy expressions
 * can use more nodes without exceeding a dynamic kernel's leaf capacity.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_active_max_leaves_set(
    SecantSRSearch search,
    size_t max_leaves
);

/** Returns the current active candidate leaf limit, or zero for NULL. */
SECANT_SR_DEC size_t secant_sr_search_active_max_leaves_get(SecantSRSearch search);

/** Changes archive-parent sampling pressure without rebuilding the archive. */
SECANT_SR_DEC SecantSRResult secant_sr_search_archive_parent_probability_set(
    SecantSRSearch search,
    double probability
);

/** Returns borrowed AST pointers in Secant's expected population order. */
SECANT_SR_DEC SecantSRResult secant_sr_search_asts_get(
    SecantSRSearch search,
    const SecantAstInstruction* const** asts_ret,
    size_t* num_asts_ret
);

/** Returns borrowed individual metadata for the current generation. */
SECANT_SR_DEC SecantSRResult secant_sr_search_individuals_get(
    SecantSRSearch search,
    const SecantSRIndividual** individuals_ret,
    size_t* num_individuals_ret
);

/** Returns quality-diversity archive capacity and current occupancy. */
SECANT_SR_DEC SecantSRResult secant_sr_search_archive_stats_get(
    SecantSRSearch search,
    SecantSRArchiveStats* stats_ret
);

/**
 * Projects the current concrete population into dynamic-leaf structural ASTs.
 *
 * Every static column or fixed f32 constant occurrence becomes a
 * dynamic-column-or-constant leaf. Leaf indices are assigned independently in
 * postorder for each AST, starting at zero. Passing NULL for both output
 * buffers performs an exact measure pass.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_programs_write(
    SecantSRSearch search,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* num_asts_ret,
    size_t* max_dynamic_leaves_ret
);

/**
 * Projects a sorted subset of the current population for dynamic-leaf search.
 *
 * Full projection replaces every static-column and fixed-f32-constant leaf;
 * every selected source must therefore fit `max_dynamic_leaves`. Mixed
 * projection deterministically replaces at most `max_dynamic_leaves` leaves
 * and preserves at least one concrete leaf when the source has more than one.
 * Constants projection replaces every fixed-f32 constant but leaves every
 * static column intact; every selected source's complete constant set must fit
 * `max_dynamic_leaves`.
 * `projection_seed` changes the selected holes without consuming the search's
 * structural RNG stream. Output ASTs remain ordered like `population_indices`.
 * Passing NULL for both output buffers performs an exact measure pass.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_selected_programs_write(
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
);

/** Materializes reduced settings for a selected full or mixed projection. */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_selected_proposals_write(
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
);

/**
 * Materializes one already-reduced binding per selected AST.
 *
 * `leaf_masks` is `[selected AST]` and `leaf_words` is
 * `[selected AST][dynamic leaf]`. Constant words contain raw f32 bits, while
 * column words contain input-column indices. This is the per-AST counterpart
 * to `secant_sr_search_dynamic_leaf_selected_proposals_write()` and is useful
 * when an optimizer returns a different constant vector for every AST.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_selected_bindings_write(
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
);

/**
 * Materializes the best dynamic-leaf binding for every current topology.
 *
 * Dynamic SSE is `[ast][setting]`. This function does not modify or score the
 * search population. Passing NULL for all three output arrays performs an
 * exact measure pass. The returned concrete proposals are suitable for a
 * static SSE run followed by `secant_sr_search_proposals_apply()`.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_proposals_write(
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
);

/** Materializes one already-reduced setting index for every current dynamic-leaf topology. */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_proposals_from_settings_write(
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
);

/**
 * Projects fixed constants in the current population to indexed dynamic constants.
 *
 * Static columns and topology remain unchanged. Constant indices are assigned
 * independently in postorder. An AST is projected only when its complete
 * constant set fits `dynamic_constant_capacity`; over-capacity ASTs remain
 * entirely static. Passing NULL for both output buffers performs an exact
 * measure pass.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_constant_programs_write(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    SecantAstInstruction* program_storage,
    size_t program_storage_size,
    const SecantAstInstruction** asts,
    size_t ast_capacity,
    size_t* required_program_storage_ret,
    size_t* num_asts_ret,
    size_t* max_dynamic_constants_ret
);

/**
 * Independently samples eligible AST indices with a dedicated optimizer RNG stream.
 *
 * Every current-population expression with one through `dynamic_constant_capacity`
 * fixed constants is eligible. Each is selected independently with
 * `selection_probability`; zero selects none and one selects all. Because the
 * selected count is not known in advance, `index_capacity` should normally cover
 * the complete population. Insufficient storage returns
 * `SECANT_SR_ERROR_INSUFFICIENT_BUFFER` after any preceding selected indices have
 * been written.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_constant_indices_sample(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    double selection_probability,
    uint32_t* population_indices,
    size_t index_capacity,
    size_t* num_eligible_ret,
    size_t* num_selected_ret
);

/**
 * Selects a bounded quality-diverse set of ASTs for constant optimization.
 *
 * Every scored current-population expression with one through
 * `dynamic_constant_capacity` fixed constants is eligible. Up to
 * `selection_budget` expressions are returned. The non-random portion is
 * divided evenly among configured complexity buckets and keeps each bucket's
 * highest-scoring expressions. The requested random share and any quality
 * quota left unused by sparse buckets are sampled uniformly without replacement
 * from the other eligible expressions. Returned indices are sorted and unique.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_constant_indices_select(
    SecantSRSearch search,
    size_t dynamic_constant_capacity,
    size_t selection_budget,
    double random_fraction,
    uint32_t* population_indices,
    size_t index_capacity,
    size_t* num_eligible_ret,
    size_t* num_selected_ret
);

/** Projects a sorted, unique selection of eligible ASTs to dense dynamic-constant programs. */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_constant_programs_selected_write(
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
);

/** Writes fixed constant values as `[selected AST][constant]` optimizer centers. */
SECANT_SR_DEC SecantSRResult secant_sr_search_constant_optimizer_centers_selected_write(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    size_t dynamic_constant_capacity,
    float* current_constants,
    size_t current_constants_num_elements,
    size_t current_constants_leading_dimension
);

/** Materializes AST-local optimizer centers back into fixed-constant programs. */
SECANT_SR_DEC SecantSRResult secant_sr_search_constant_optimizer_proposals_selected_write(
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
);

/**
 * Materializes the best constant vector for every current AST.
 *
 * Constant settings are `[constant][setting]` and dynamic SSE is
 * `[ast][setting]`. `constant_setting_robustness` receives the mean clipped
 * training R2 over all finite settings for ASTs with at least one constant.
 * Passing NULL for all four output arrays performs an exact measure pass.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_constant_proposals_write(
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
);

/** Materializes selected ASTs from already-reduced dynamic-constant setting indices. */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_constant_proposals_selected_write(
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
);

/**
 * Rebuilds the current population from statically scored proposals.
 *
 * Every current source must already be scored. A finite proposal replaces its
 * corresponding source only when its SSE is lower. Rejected sources preserve
 * their program, metadata, and score. The operation writes the alternate arena
 * and changes the current population only after every candidate succeeds.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_proposals_apply(
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
);

/** Compares and applies proposals for sorted, unique population indices while copying all other members unchanged. */
SECANT_SR_DEC SecantSRResult secant_sr_search_selected_proposals_apply(
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
);

/**
 * Selects the best runtime leaf setting for every current structure.
 *
 * SSE is `[ast][setting]`. The selected column indices or exact f32 bits are
 * materialized into a new concrete population in the alternate generation
 * arena. The search generation number is unchanged and the resulting
 * population is fully scored and archived. The search configuration's
 * `max_program_bytes` must accommodate five-byte constants selected in place
 * of two-byte dynamic leaves.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_scores_apply(
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
);

/** Applies one already-reduced dynamic-leaf setting and SSE for every current topology. */
SECANT_SR_DEC SecantSRResult secant_sr_search_dynamic_leaf_best_scores_apply(
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
);

/** Applies evaluator-produced SSE values and updates complexity-stratified elites. */
SECANT_SR_DEC SecantSRResult secant_sr_search_scores_set(
    SecantSRSearch search,
    const float* sse,
    size_t num_sse,
    size_t num_rows,
    double target_sum_squared_deviation
);

/** Uses Secant's CPU interpreter as the correctness/reference evaluator. */
SECANT_SR_DEC SecantSRResult secant_sr_search_cpu_evaluate(
    SecantSRSearch search,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t num_rows,
    double target_sum_squared_deviation,
    float* sse_scratch,
    size_t sse_scratch_count
);

/**
 * Refines fixed f32 constants in the best archived expression after search.
 *
 * The best expression is selected before optimization. All of its constants are
 * varied jointly with finite-difference L-BFGS. The current values form the
 * first start and each restart perturbs those original values. The archived
 * expression is changed only when its SSE over all `num_rows` improves. This
 * function never traverses or refines the current population and is intended as
 * a final accuracy audit rather than a generation-stage search operator.
 *
 * This function does not allocate. Its temporary optimizer state has a fixed
 * bound derived from SECANT_AST_MAX_PROGRAM_INSTRUCTIONS. The input and target
 * storage remain caller-owned.
 *
 * @param[in,out] search Initialized search with at least one scored archive elite.
 * @param[in] routines Optional routine table used by the CPU interpreter.
 * @param[in] num_routines Number of entries in `routines`.
 * @param[in] input Column-major input data.
 * @param[in] input_num_elements Number of available float elements in `input`.
 * @param[in] input_leading_dimension Row stride between input columns.
 * @param[in] target Target values for the one fitted output.
 * @param[in] target_num_elements Number of available float elements in `target`.
 * @param[in] optimizer_num_rows Prefix of the input rows used by finite-difference optimization.
 * @param[in] num_rows Full row count used to rescore each selected expression after optimization.
 * @param[in] target_sum_squared_deviation Full target sum of squared deviations used for normalized fitness.
 * @param[in] optimizer_iterations Maximum accepted L-BFGS steps per start.
 * @param[in] optimizer_restarts Number of randomized starts after the original constants.
 * @param[in] optimizer_f_calls_limit Maximum objective calls per selected expression across all starts.
 * @param[in] finite_difference_relative_step Positive relative step used for central finite differences.
 * @param[out] num_constants_ret Optional number of fixed constants in the selected expression.
 * @param[out] improved_ret Optional nonzero result when full-training SSE improved.
 * @param[out] num_f_calls_ret Optional total objective calls for the selected expression.
 */
SECANT_SR_DEC SecantSRResult secant_sr_search_best_constants_optimize_cpu(
    SecantSRSearch search,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t optimizer_num_rows,
    size_t num_rows,
    double target_sum_squared_deviation,
    size_t optimizer_iterations,
    size_t optimizer_restarts,
    size_t optimizer_f_calls_limit,
    double finite_difference_relative_step,
    size_t* num_constants_ret,
    int* improved_ret,
    size_t* num_f_calls_ret
);

/** Creates the next generation by copying elites, crossovers, and mutations into a reset arena. */
SECANT_SR_DEC SecantSRResult secant_sr_search_generation_advance(SecantSRSearch search);

/** Returns the best archived individual across all complexity buckets. */
SECANT_SR_DEC SecantSRResult secant_sr_search_best_get(
    SecantSRSearch search,
    const SecantSRIndividual** individual_ret
);

/** Formats an individual as a readable infix expression using measure-and-write semantics. */
SECANT_SR_DEC SecantSRResult secant_sr_individual_format(
    const SecantSRIndividual* individual,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

#endif /* SECANT_SR_H_INCLUDED */
