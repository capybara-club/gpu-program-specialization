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
#ifndef SECANT_SR_CUBIN_EVALUATOR_H_INCLUDED
#define SECANT_SR_CUBIN_EVALUATOR_H_INCLUDED

#include "cubin_cache.h"
#include "secant.h"
#include "secant_cubin_lm_optimizer_runner.h"

#include <cuda.h>


typedef struct SecantSRCubinTemplateStats {
    size_t cache_hits;
    size_t cache_misses;
    size_t cache_invalidations;
    double cache_lookup_seconds;
    double cache_store_seconds;
    double compile_seconds;
    double estimated_uncached_compile_seconds;
    double source_seconds;
    double inspect_plan_seconds;
    double runner_create_seconds;
    double device_prepare_seconds;
    double prepare_seconds;
} SecantSRCubinTemplateStats;

typedef struct SecantSRCudaSession {
    CUcontext context;
    int compute_capability_major;
    int compute_capability_minor;
} SecantSRCudaSession;

typedef struct SecantSRCubinEvaluator {
    CUcontext context;
    CUdeviceptr input;
    CUdeviceptr target;
    CUdeviceptr constant_settings;
    CUdeviceptr leaf_masks;
    CUdeviceptr leaf_words;
    CUdeviceptr output;
    CUdeviceptr optimizer_best;
    char* source;
    unsigned char* cubin;
    void* plan_storage;
    SecantCubinPlan* plan;
    SecantCubinRunner runner;
    const SecantAstInstruction* const* routines;
    size_t source_size;
    size_t cubin_size;
    size_t plan_storage_size;
    size_t num_routines;
    size_t population_size;
    size_t num_inputs;
    size_t num_static_input_columns;
    size_t num_rows;
    size_t num_dynamic_leaves;
    size_t num_input_constants;
    size_t num_settings;
    size_t leaf_words_leading_dimension;
    int dynamic_leaf;
    int dynamic_constant;
    int constant_optimizer;
    int packed_constant_optimizer;
    int materialize;
    SecantSRCubinTemplateStats template_stats;
    SecantRunnerStats last_stats;
} SecantSRCubinEvaluator;

typedef struct SecantSRCubinStagedEvaluator {
    SecantSRCubinEvaluator static_sse;
    SecantSRCubinEvaluator dynamic_leaf;
    SecantSRCubinEvaluator dynamic_constant;
    SecantSRCubinEvaluator constant_optimizer;
    SecantSRCubinEvaluator materialize;
} SecantSRCubinStagedEvaluator;

typedef struct SecantSRCubinLMEvaluator {
    CUcontext context;
    SecantCubinLMOptimizerRunner runner;
    CUdeviceptr input;
    CUdeviceptr target;
    CUdeviceptr leaf_masks;
    CUdeviceptr leaf_words;
    CUdeviceptr current_constants;
    CUdeviceptr proposal_constants;
    CUdeviceptr evaluated_statistics;
    CUdeviceptr accepted_statistics;
    CUdeviceptr damping;
    CUdeviceptr predicted_reduction;
    CUdeviceptr status;
    const SecantAstInstruction* const* routines;
    size_t num_routines;
    size_t ast_capacity;
    size_t num_input_columns;
    size_t num_rows;
    size_t num_settings;
    SecantRunnerStats last_stats;
} SecantSRCubinLMEvaluator;

int secant_sr_cuda_session_create(SecantSRCudaSession* session);

void secant_sr_cuda_session_destroy(SecantSRCudaSession* session);

int secant_sr_cubin_evaluator_routines_set(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* routines,
    size_t num_routines
);

int secant_sr_cubin_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
);

int secant_sr_cubin_evaluator_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* sse
);

int secant_sr_cubin_dynamic_leaf_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    size_t num_inputs,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
);

int secant_sr_cubin_mixed_dynamic_leaf_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    size_t num_inputs,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
);

int secant_sr_cubin_dynamic_leaf_settings_update(
    SecantSRCubinEvaluator* evaluator,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings
);

int secant_sr_cubin_dynamic_constant_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    const float* constant_settings,
    size_t constant_settings_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
);

int secant_sr_cubin_materialize_evaluator_run_one(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* ast,
    float* output
);

int secant_sr_cubin_dynamic_leaf_evaluator_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* sse
);

int secant_sr_cubin_dynamic_leaf_evaluator_run_device(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts
);

int secant_sr_cubin_dynamic_constant_evaluator_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* sse
);

int secant_sr_cubin_dynamic_constant_evaluator_run_device(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts
);

int secant_sr_cubin_constant_optimizer_evaluator_create(
    size_t num_kernels,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
);

int secant_sr_cubin_constant_optimizer_constants_upload(
    SecantSRCubinEvaluator* evaluator,
    const float* current_constants,
    size_t num_asts
);

int secant_sr_cubin_constant_optimizer_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uint64_t seed,
    uint64_t generation,
    uint64_t iteration,
    uint64_t ast_index_base,
    float perturbation_scale,
    float perturbation_decay,
    size_t num_iterations
);

int secant_sr_cubin_constant_optimizer_constants_download(
    SecantSRCubinEvaluator* evaluator,
    float* current_constants,
    size_t num_asts
);

int secant_sr_cubin_packed_constant_optimizer_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
);

int secant_sr_cubin_packed_constant_optimizer_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* current_constants,
    float* current_constant_scales,
    float* current_constant_velocities,
    float* current_sse,
    uint64_t seed,
    uint64_t generation,
    uint64_t iteration,
    size_t num_iterations,
    float momentum,
    float scale_learning_rate,
    float scale_failure_decay,
    float minimum_scale,
    float maximum_scale
);

int secant_sr_cubin_lm_evaluator_create(
    size_t ast_capacity,
    size_t num_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t num_settings,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinLMEvaluator* evaluator
);

int secant_sr_cubin_lm_evaluator_routines_set(
    SecantSRCubinLMEvaluator* evaluator,
    const SecantAstInstruction* const* routines,
    size_t num_routines
);

int secant_sr_cubin_lm_evaluator_run_mixed(
    SecantSRCubinLMEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    const float* initial_constants,
    size_t settings_per_cta,
    size_t num_iterations,
    float initial_damping,
    float* optimized_constants,
    float* accepted_sse
);

void secant_sr_cubin_lm_evaluator_destroy(SecantSRCubinLMEvaluator* evaluator);

int secant_sr_cubin_staged_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_dynamic_leaves,
    size_t static_tile_rows,
    size_t dynamic_tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_leaf_settings,
    const float* constant_settings,
    size_t num_input_constants,
    size_t constant_settings_leading_dimension,
    size_t num_constant_settings,
    const float* materialize_input,
    size_t materialize_num_rows,
    size_t population_size,
    size_t dynamic_ast_capacity,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinStagedEvaluator* evaluator
);

void secant_sr_cubin_staged_evaluator_destroy(SecantSRCubinStagedEvaluator* evaluator);

void secant_sr_cubin_evaluator_destroy(SecantSRCubinEvaluator* evaluator);

#endif /* SECANT_SR_CUBIN_EVALUATOR_H_INCLUDED */
