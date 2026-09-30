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
#define _POSIX_C_SOURCE 200809L

#include "dataset.h"
#include "leaf_settings.h"
#include "secant_sr.h"
#include "secant_instructions.h"

#ifdef SECANT_SR_HAS_CUBIN
#include "cubin_evaluator.h"
#include "mse_reducer.h"
#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SECANT_SR_APP_LM_PARAMETERS 8u

static const SecantAstInstruction secant_sr_app_square_f32[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction secant_sr_app_cube_f32[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantSRRoutine secant_sr_app_routines[] = {
    {secant_sr_app_square_f32, "square", 1u, 1u, 0u},
    {secant_sr_app_cube_f32, "cube", 1u, 1u, 0u}
};

static const SecantAstInstruction* const secant_sr_app_routine_programs[] = {
    secant_sr_app_square_f32,
    secant_sr_app_cube_f32
};

static double
secant_sr_app_seconds_get(void) {
    struct timespec value;

    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static SecantSRResult
secant_sr_app_best_snapshot_get(
    SecantSRSearch search,
    double* r2_ret,
    int* origin_ret,
    int* maturity_ret
) {
    const SecantSRIndividual* best;
    const SecantSRResult result = secant_sr_search_best_get(search, &best);

    if (result != SECANT_SR_SUCCESS) {
        return result;
    }
    if (r2_ret != NULL) {
        *r2_ret = best->fitness.r2;
    }
    if (origin_ret != NULL) {
        *origin_ret = (int)best->origin;
    }
    if (maturity_ret != NULL) {
        *maturity_ret = (int)best->maturity;
    }
    return SECANT_SR_SUCCESS;
}

typedef struct SecantSRAppCudaState {
#ifdef SECANT_SR_HAS_CUBIN
    SecantSRCudaSession session;
    SecantSRCubinCache cache;
    SecantSRCubinCache* cache_ptr;
#endif
    double context_create_seconds;
    double cache_open_seconds;
} SecantSRAppCudaState;

typedef struct SecantSRAppCheckpoint {
    size_t generation;
    double search_elapsed_seconds;
    double train_r2;
    uint64_t fingerprint;
    uint32_t complexity;
    size_t num_nodes;
    size_t program_bytes;
    double checkpoint_copy_seconds;
} SecantSRAppCheckpoint;

static int
secant_sr_app_cuda_state_create(
    const char* backend,
    const char* cubin_cache_path,
    SecantSRAppCudaState* state
) {
    memset(state, 0, sizeof(*state));
#ifdef SECANT_SR_HAS_CUBIN
    if (strcmp(backend, "cpu") != 0) {
        double begin = secant_sr_app_seconds_get();

        if (!secant_sr_cuda_session_create(&state->session)) {
            return 0;
        }
        state->context_create_seconds = secant_sr_app_seconds_get() - begin;
        if (cubin_cache_path != NULL) {
            begin = secant_sr_app_seconds_get();
            if (!secant_sr_cubin_cache_open(cubin_cache_path, &state->cache)) {
                secant_sr_cuda_session_destroy(&state->session);
                return 0;
            }
            state->cache_open_seconds = secant_sr_app_seconds_get() - begin;
            state->cache_ptr = &state->cache;
        }
    }
    return 1;
#else
    (void)cubin_cache_path;
    return strcmp(backend, "cpu") == 0;
#endif
}

static void
secant_sr_app_cuda_state_destroy(SecantSRAppCudaState* state) {
#ifdef SECANT_SR_HAS_CUBIN
    secant_sr_cubin_cache_close(&state->cache);
    secant_sr_cuda_session_destroy(&state->session);
#endif
    memset(state, 0, sizeof(*state));
}

#ifdef SECANT_SR_HAS_CUBIN
static int
secant_sr_app_evaluator_routines_set(SecantSRCubinEvaluator* evaluator) {
    return secant_sr_cubin_evaluator_routines_set(
        evaluator,
        secant_sr_app_routine_programs,
        sizeof(secant_sr_app_routine_programs) / sizeof(secant_sr_app_routine_programs[0]));
}

static int
secant_sr_app_staged_evaluator_routines_set(SecantSRCubinStagedEvaluator* evaluator) {
    return secant_sr_app_evaluator_routines_set(&evaluator->static_sse) &&
        secant_sr_app_evaluator_routines_set(&evaluator->dynamic_leaf) &&
        secant_sr_app_evaluator_routines_set(&evaluator->dynamic_constant) &&
        secant_sr_app_evaluator_routines_set(&evaluator->constant_optimizer) &&
        secant_sr_app_evaluator_routines_set(&evaluator->materialize);
}

static void
secant_sr_app_template_stats_add(
    SecantSRCubinTemplateStats* total,
    const SecantSRCubinTemplateStats* value
) {
    total->cache_hits += value->cache_hits;
    total->cache_misses += value->cache_misses;
    total->cache_invalidations += value->cache_invalidations;
    total->cache_lookup_seconds += value->cache_lookup_seconds;
    total->cache_store_seconds += value->cache_store_seconds;
    total->compile_seconds += value->compile_seconds;
    total->estimated_uncached_compile_seconds += value->estimated_uncached_compile_seconds;
    total->source_seconds += value->source_seconds;
    total->inspect_plan_seconds += value->inspect_plan_seconds;
    total->runner_create_seconds += value->runner_create_seconds;
    total->device_prepare_seconds += value->device_prepare_seconds;
    total->prepare_seconds += value->prepare_seconds;
}

static void
secant_sr_app_template_stats_print(
    const SecantSRAppCudaState* cuda_state,
    const SecantSRCubinTemplateStats* stats
) {
    printf(
        "template_cache enabled=%d hits=%zu misses=%zu invalidations=%zu open_seconds=%.9f "
        "lookup_seconds=%.9f store_seconds=%.9f actual_compile_seconds=%.9f "
        "estimated_uncached_compile_seconds=%.9f estimated_saved_compile_seconds=%.9f "
        "source_seconds=%.9f inspect_plan_seconds=%.9f runner_create_seconds=%.9f "
        "device_prepare_seconds=%.9f prepare_seconds=%.9f\n",
        cuda_state->cache_ptr != NULL,
        stats->cache_hits,
        stats->cache_misses,
        stats->cache_invalidations,
        cuda_state->cache_open_seconds,
        stats->cache_lookup_seconds,
        stats->cache_store_seconds,
        stats->compile_seconds,
        stats->estimated_uncached_compile_seconds,
        stats->estimated_uncached_compile_seconds - stats->compile_seconds,
        stats->source_seconds,
        stats->inspect_plan_seconds,
        stats->runner_create_seconds,
        stats->device_prepare_seconds,
        stats->prepare_seconds);
}

static SecantSRResult
secant_sr_app_dynamic_leaf_cohort_run(
    SecantSRSearch search,
    const uint32_t* population_indices,
    size_t num_selected,
    SecantSRDynamicLeafProjection projection,
    size_t num_dynamic_leaves,
    uint64_t projection_seed,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_word_elements,
    size_t num_leaf_settings,
    size_t dynamic_ast_capacity,
    size_t num_rows,
    double target_ssd,
    SecantSRCubinEvaluator* dynamic_evaluator,
    SecantSRCubinEvaluator* static_evaluator,
    SecantSRMSEReducer* mse_reducer,
    SecantAstInstruction* program_storage,
    size_t program_storage_capacity,
    const SecantAstInstruction** asts,
    size_t* program_sizes,
    uint32_t* best_setting_indices,
    SecantSRMSEEntry* mse_entries,
    float* proposal_sse,
    int collect_stats,
    size_t* num_promoted_ret,
    double* r2_gain_sum_ret,
    double* r2_gain_max_ret,
    size_t* finite_gain_count_ret,
    size_t* nonfinite_recovered_ret,
    double* compile_seconds,
    double* module_load_seconds,
    double* device_runtime_seconds
) {
    SecantSRResult result;
    size_t required_program_size;
    size_t max_projected_leaves;
    size_t chunk_begin;

    if (num_selected == 0u) {
        *num_promoted_ret = 0u;
        *r2_gain_sum_ret = 0.0;
        *r2_gain_max_ret = 0.0;
        *finite_gain_count_ret = 0u;
        *nonfinite_recovered_ret = 0u;
        return SECANT_SR_SUCCESS;
    }
    result = secant_sr_search_dynamic_leaf_selected_programs_write(
            search,
            population_indices,
            num_selected,
            projection,
            num_dynamic_leaves,
            projection_seed,
            program_storage,
            program_storage_capacity,
            asts,
            num_selected,
            &required_program_size,
            &max_projected_leaves);
    if (result != SECANT_SR_SUCCESS) {
        fprintf(stderr,
            "leaf cohort projection failed mode=%d selected=%zu: %s\n",
            (int)projection,
            num_selected,
            secant_sr_result_to_string(result));
        return result;
    }
    if (required_program_size > program_storage_capacity ||
        max_projected_leaves > dynamic_evaluator->num_dynamic_leaves) {
        return SECANT_SR_ERROR_BAD_PROGRAM;
    }
    for (chunk_begin = 0u; chunk_begin < num_selected; chunk_begin += dynamic_ast_capacity) {
        const size_t chunk_asts = num_selected - chunk_begin < dynamic_ast_capacity
            ? num_selected - chunk_begin
            : dynamic_ast_capacity;
        size_t chunk_idx;

        if (!secant_sr_cubin_dynamic_leaf_evaluator_run_device(
                dynamic_evaluator, asts + chunk_begin, chunk_asts) ||
            !secant_sr_mse_reducer_best_get(
                mse_reducer,
                dynamic_evaluator->output,
                chunk_asts,
                num_leaf_settings,
                num_leaf_settings,
                num_rows,
                target_ssd,
                mse_entries,
                dynamic_ast_capacity)) {
            return SECANT_SR_ERROR_SECANT;
        }
        *compile_seconds += dynamic_evaluator->last_stats.compile_window_seconds;
        *module_load_seconds += dynamic_evaluator->last_stats.module_load_seconds;
        *device_runtime_seconds += dynamic_evaluator->last_stats.runtime_seconds;
        for (chunk_idx = 0u; chunk_idx < chunk_asts; ++chunk_idx) {
            best_setting_indices[chunk_begin + chunk_idx] = mse_entries[chunk_idx].setting_idx;
        }
    }
    result = secant_sr_search_dynamic_leaf_selected_proposals_write(
            search,
            population_indices,
            num_selected,
            projection,
            num_dynamic_leaves,
            projection_seed,
            leaf_masks,
            num_leaf_settings,
            leaf_words,
            leaf_word_elements,
            dynamic_evaluator->leaf_words_leading_dimension,
            num_leaf_settings,
            best_setting_indices,
            program_storage,
            program_storage_capacity,
            asts,
            program_sizes,
            num_selected,
            &required_program_size);
    if (result != SECANT_SR_SUCCESS) {
        fprintf(stderr,
            "leaf cohort materialization failed mode=%d selected=%zu: %s\n",
            (int)projection,
            num_selected,
            secant_sr_result_to_string(result));
        return result;
    }
    if (!secant_sr_cubin_evaluator_run(static_evaluator, asts, num_selected, proposal_sse)) {
        return SECANT_SR_ERROR_SECANT;
    }
    if (collect_stats) {
        const SecantSRIndividual* individuals;
        size_t num_individuals;
        size_t selected_idx;

        *r2_gain_sum_ret = 0.0;
        *r2_gain_max_ret = 0.0;
        *finite_gain_count_ret = 0u;
        *nonfinite_recovered_ret = 0u;
        if (secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS) {
            return SECANT_SR_ERROR_INVALID_VALUE;
        }
        for (selected_idx = 0u; selected_idx < num_selected; ++selected_idx) {
            const size_t population_idx = population_indices[selected_idx];

            if (population_idx >= num_individuals || !isfinite(proposal_sse[selected_idx]) ||
                proposal_sse[selected_idx] < 0.0f ||
                (double)proposal_sse[selected_idx] >= individuals[population_idx].fitness.sse) {
                continue;
            }
            if (!isfinite(individuals[population_idx].fitness.sse)) {
                ++*nonfinite_recovered_ret;
            } else {
                const double gain = (individuals[population_idx].fitness.sse - proposal_sse[selected_idx]) /
                    target_ssd;

                if (isfinite(gain)) {
                    *r2_gain_sum_ret += gain;
                    ++*finite_gain_count_ret;
                    if (gain > *r2_gain_max_ret) {
                        *r2_gain_max_ret = gain;
                    }
                }
            }
        }
    } else {
        *r2_gain_sum_ret = 0.0;
        *r2_gain_max_ret = 0.0;
        *finite_gain_count_ret = 0u;
        *nonfinite_recovered_ret = 0u;
    }
    *compile_seconds += static_evaluator->last_stats.compile_window_seconds;
    *module_load_seconds += static_evaluator->last_stats.module_load_seconds;
    *device_runtime_seconds += static_evaluator->last_stats.runtime_seconds;
    return secant_sr_search_selected_proposals_apply(
        search,
        population_indices,
        asts,
        program_sizes,
        proposal_sse,
        NULL,
        num_selected,
        num_rows,
        target_ssd,
        SECANT_SR_ORIGIN_DYNAMIC_LEAF,
        num_promoted_ret);
}
#endif

static int
secant_sr_app_size_mul(size_t left, size_t right, size_t* result_ret) {
    if (left != 0u && right > SIZE_MAX / left) {
        return 0;
    }
    *result_ret = left * right;
    return 1;
}

static int
secant_sr_app_batch_line_parse(
    char* line,
    char** problem_ret,
    uint64_t* seed_ret,
    char** dataset_binary_path_ret,
    size_t* num_rows_ret,
    size_t* validation_rows_ret
) {
    char* fields[5];
    char* end;
    unsigned long long value;
    size_t field_idx;

    fields[0] = line;
    for (field_idx = 1u; field_idx < 5u; ++field_idx) {
        fields[field_idx] = strchr(fields[field_idx - 1u], '\t');
        if (fields[field_idx] == NULL) {
            return 0;
        }
        *fields[field_idx]++ = '\0';
    }
    end = fields[4] + strlen(fields[4]);
    while (end != fields[4] && (end[-1] == '\n' || end[-1] == '\r')) {
        *--end = '\0';
    }
    if (fields[0][0] == '\0' || strchr(fields[4], '\t') != NULL) {
        return 0;
    }
    value = strtoull(fields[1], &end, 0);
    if (*fields[1] == '\0' || *end != '\0') {
        return 0;
    }
    *seed_ret = (uint64_t)value;
    value = strtoull(fields[3], &end, 10);
    if (*fields[3] == '\0' || *end != '\0' || value > SIZE_MAX) {
        return 0;
    }
    *num_rows_ret = (size_t)value;
    value = strtoull(fields[4], &end, 10);
    if (*fields[4] == '\0' || *end != '\0' || value > SIZE_MAX) {
        return 0;
    }
    *validation_rows_ret = (size_t)value;
    if (*num_rows_ret < 2u || *validation_rows_ret < 2u) {
        return 0;
    }
    *problem_ret = fields[0];
    *dataset_binary_path_ret = fields[2][0] == '\0' ? NULL : fields[2];
    return 1;
}

static double
secant_sr_app_validation_r2(
    const SecantSRDataset* dataset,
    const SecantSRIndividual* individual,
    const float* input,
    const float* target,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    const SecantAstInstruction* asts[] = {individual->program};
    SecantCpuSSERun run = secant_cpu_sse_run_init();
    float sse = 0.0f;

    run.programs.routines.items = secant_sr_app_routine_programs;
    run.programs.routines.count = sizeof(secant_sr_app_routine_programs) /
        sizeof(secant_sr_app_routine_programs[0]);
    run.programs.asts.items = asts;
    run.programs.asts.count = 1u;
    run.num_inputs = dataset->num_inputs;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, dataset->num_inputs * num_rows, num_rows};
    run.targets = (SecantConstHostMatrixF32){target, num_rows, num_rows};
    run.num_rows = num_rows;
    run.output = (SecantHostMatrixF32){&sse, 1u, 1u};
    if (secant_cpu_run_sse(&run) != SECANT_SUCCESS) {
        return -INFINITY;
    }
    return isfinite(sse) ? 1.0 - (double)sse / target_sum_squared_deviation : -INFINITY;
}

static double
secant_sr_app_predictions_r2(
    const float* prediction,
    const float* target,
    size_t num_rows,
    double target_sum_squared_deviation
) {
    double sse = 0.0;
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        const double difference = (double)prediction[row] - (double)target[row];

        sse += difference * difference;
    }
    return isfinite(sse) ? 1.0 - sse / target_sum_squared_deviation : -INFINITY;
}

static uint64_t
secant_sr_app_mix64(uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static int
secant_sr_app_probability_select(double probability, uint64_t seed, uint64_t fingerprint) {
    const uint64_t hash = secant_sr_app_mix64(seed ^ fingerprint);
    const double unit = (double)(hash >> 11u) * (1.0 / 9007199254740992.0);

    return probability >= 1.0 || (probability > 0.0 && unit < probability);
}

static double
secant_sr_app_hash_unit(uint64_t value) {
    return (double)(secant_sr_app_mix64(value) >> 11u) *
        (1.0 / 9007199254740992.0);
}

static int
secant_sr_app_lm_starts_write(
    const float* centers,
    size_t num_asts,
    size_t num_settings,
    size_t starts_per_binding,
    double start_scale,
    uint64_t seed,
    size_t generation,
    const uint32_t* population_indices,
    float* starts,
    size_t starts_num_elements
) {
    const size_t parameters = SECANT_SR_APP_LM_PARAMETERS;
    size_t required_elements;
    size_t ast_idx;

    if (centers == NULL || population_indices == NULL || starts == NULL || num_asts == 0u ||
        num_settings == 0u || starts_per_binding == 0u || num_settings % starts_per_binding != 0u ||
        start_scale < 0.0 || !isfinite(start_scale) ||
        !secant_sr_app_size_mul(num_asts, num_settings, &required_elements) ||
        !secant_sr_app_size_mul(required_elements, parameters, &required_elements) ||
        required_elements > starts_num_elements) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        size_t setting;

        for (setting = 0u; setting < num_settings; ++setting) {
            const size_t start = setting % starts_per_binding;
            size_t parameter;

            for (parameter = 0u; parameter < parameters; ++parameter) {
                const float center = centers[ast_idx * parameters + parameter];
                float value = center;

                if (start != 0u) {
                    const uint64_t key = seed ^ UINT64_C(0x6c6d5f7374617274) ^
                        ((uint64_t)generation * UINT64_C(0x9e3779b97f4a7c15)) ^
                        ((uint64_t)population_indices[ast_idx] * UINT64_C(0xd1b54a32d192ed03)) ^
                        ((uint64_t)setting * UINT64_C(0x94d049bb133111eb)) ^
                        ((uint64_t)parameter * UINT64_C(0xbf58476d1ce4e5b9));
                    const double direction = 2.0 * secant_sr_app_hash_unit(key) - 1.0;
                    const double log_radius = 4.0 * secant_sr_app_hash_unit(
                        key ^ UINT64_C(0xa0761d6478bd642f)) - 2.0;
                    const double candidate = (double)center +
                        start_scale * direction * exp2(log_radius);

                    if (isfinite(candidate) && fabs(candidate) <= (double)FLT_MAX) {
                        value = (float)candidate;
                    }
                }
                starts[(ast_idx * num_settings + setting) * parameters + parameter] = value;
            }
        }
    }
    return 1;
}

static int
secant_sr_app_lm_best_write(
    const float* optimized_constants,
    const float* accepted_sse,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t num_asts,
    size_t num_settings,
    size_t leaf_words_leading_dimension,
    float* best_constants,
    uint32_t* best_leaf_masks,
    uint32_t* best_leaf_words
) {
    const size_t parameters = SECANT_SR_APP_LM_PARAMETERS;
    size_t ast_idx;

    if (optimized_constants == NULL || accepted_sse == NULL || leaf_masks == NULL || leaf_words == NULL ||
        best_constants == NULL || best_leaf_masks == NULL || best_leaf_words == NULL ||
        num_asts == 0u || num_settings == 0u || leaf_words_leading_dimension < parameters) {
        return 0;
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        size_t best_setting = 0u;
        float best_sse = accepted_sse[ast_idx * num_settings];
        size_t setting;
        size_t parameter;

        if (!isfinite(best_sse) || best_sse < 0.0f) {
            best_sse = INFINITY;
        }
        for (setting = 1u; setting < num_settings; ++setting) {
            const float candidate_sse = accepted_sse[ast_idx * num_settings + setting];

            if (isfinite(candidate_sse) && candidate_sse >= 0.0f && candidate_sse < best_sse) {
                best_sse = candidate_sse;
                best_setting = setting;
            }
        }
        best_leaf_masks[ast_idx] = leaf_masks[ast_idx * num_settings + best_setting];
        for (parameter = 0u; parameter < parameters; ++parameter) {
            const size_t setting_row = ast_idx * num_settings + best_setting;
            const float value = optimized_constants[
                (ast_idx * num_settings + best_setting) * parameters + parameter];

            best_constants[ast_idx * parameters + parameter] = value;
            if ((best_leaf_masks[ast_idx] & (UINT32_C(1) << parameter)) != 0u) {
                best_leaf_words[ast_idx * parameters + parameter] = leaf_words[
                    setting_row * leaf_words_leading_dimension + parameter];
            } else {
                memcpy(best_leaf_words + ast_idx * parameters + parameter, &value, sizeof(value));
            }
        }
    }
    return 1;
}

static int
secant_sr_app_constant_optimizer_generation_select(
    const char* phase,
    size_t generation,
    size_t generations,
    size_t interval,
    int dynamic_stage
) {
    const int cadence = generation % interval == 0u;

    if (strcmp(phase, "each") == 0) {
        return cadence;
    }
    if (strcmp(phase, "early") == 0) {
        return dynamic_stage && cadence;
    }
    if (strcmp(phase, "final") == 0) {
        return generation + 1u == generations;
    }
    return (dynamic_stage && cadence) || generation + 1u == generations;
}

static int
secant_sr_app_run(
    const SecantSRDataset* dataset,
    const char* dataset_binary_path,
    const char* backend,
    size_t population_size,
    size_t generations,
    size_t num_rows,
    size_t validation_rows,
    uint64_t seed,
    size_t num_workers,
    size_t num_streams,
    const SecantSRAppCudaState* cuda_state,
    size_t num_leaf_settings,
    SecantSRAppLeafSettingsPolicy leaf_settings_policy,
    size_t num_constant_settings,
    size_t dynamic_batch_asts,
    double constant_optimize_probability,
    size_t constant_optimize_budget,
    size_t constant_optimize_interval,
    double constant_optimize_random_fraction,
    const char* constant_sweep_phase,
    const char* constant_optimizer_mode,
    size_t constant_optimizer_iterations,
    double constant_optimizer_scale,
    double constant_optimizer_decay,
    size_t constant_optimizer_tile_rows,
    size_t lm_batch_asts,
    size_t lm_settings_per_cta,
    size_t lm_starts_per_binding,
    size_t lm_tile_rows,
    size_t lm_threads,
    size_t lm_patch_instructions,
    double lm_initial_damping,
    size_t num_dynamic_leaves,
    size_t mixed_dynamic_leaves,
    double mixed_refine_probability,
    size_t dynamic_generations,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t tile_rows,
    size_t static_tile_rows,
    size_t patch_instructions_per_ast,
    int final_cpu_optimize,
    size_t final_cpu_optimizer_rows,
    size_t final_cpu_optimizer_iterations,
    size_t final_cpu_optimizer_restarts,
    size_t final_cpu_optimizer_f_calls_limit,
    double final_cpu_optimizer_finite_difference_step,
    const char* operator_profile,
    size_t qd_column_buckets,
    size_t qd_transcendental_buckets,
    double archive_parent_probability,
    size_t dynamic_max_nodes,
    double constant_setting_credit_weight,
    double parsimony_coefficient,
    const char* validation_mode,
    const char* stop_metric,
    double stop_r2,
    double time_limit_seconds
) {
    static const SecantAstInstructionType broad_unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32,
        SECANT_AST_INSTRUCTION_TYPE_ABS_F32,
        SECANT_AST_INSTRUCTION_TYPE_SQRT_F32,
        SECANT_AST_INSTRUCTION_TYPE_RCP_F32,
        SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_COS_F32,
        SECANT_AST_INSTRUCTION_TYPE_TANH_F32,
        SECANT_AST_INSTRUCTION_TYPE_EXP_F32,
        SECANT_AST_INSTRUCTION_TYPE_LOG_F32
    };
    static const SecantAstInstructionType broad_binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32,
        SECANT_AST_INSTRUCTION_TYPE_DIV_F32,
        SECANT_AST_INSTRUCTION_TYPE_MIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_MAX_F32
    };
    static const SecantAstInstructionType trig_unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32,
        SECANT_AST_INSTRUCTION_TYPE_ABS_F32,
        SECANT_AST_INSTRUCTION_TYPE_SQRT_F32,
        SECANT_AST_INSTRUCTION_TYPE_RCP_F32,
        SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_COS_F32
    };
    static const SecantAstInstructionType algebraic_unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_NEG_F32,
        SECANT_AST_INSTRUCTION_TYPE_ABS_F32,
        SECANT_AST_INSTRUCTION_TYPE_SQRT_F32,
        SECANT_AST_INSTRUCTION_TYPE_RCP_F32
    };
    static const SecantAstInstructionType scientific_unary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_SQRT_F32,
        SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
        SECANT_AST_INSTRUCTION_TYPE_COS_F32,
        SECANT_AST_INSTRUCTION_TYPE_EXP_F32,
        SECANT_AST_INSTRUCTION_TYPE_LOG_F32
    };
    static const SecantAstInstructionType arithmetic_binary_ops[] = {
        SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
        SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
        SECANT_AST_INSTRUCTION_TYPE_MUL_F32,
        SECANT_AST_INSTRUCTION_TYPE_DIV_F32
    };
    static const float constants[] = {
        -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.25f,
        0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 3.1415927410125732f
    };
    const SecantAstInstructionType* unary_ops = broad_unary_ops;
    const SecantAstInstructionType* binary_ops = broad_binary_ops;
    size_t num_unary_ops = sizeof(broad_unary_ops) / sizeof(broad_unary_ops[0]);
    size_t num_binary_ops = sizeof(broad_binary_ops) / sizeof(broad_binary_ops[0]);
    const int dynamic_leaf_backend = strcmp(backend, "cubin-dynamic-leaf") == 0;
    const int maturity_backend = strcmp(backend, "cubin-maturity") == 0;
    const int maturity_trace = maturity_backend && getenv("SECANT_SR_MATURITY_TRACE") != NULL;
    const int checkpoint_trace = getenv("SECANT_SR_CHECKPOINT_TRACE") != NULL;
    const int legacy_staged_backend = strcmp(backend, "cubin-staged") == 0;
    const int staged_backend = legacy_staged_backend || maturity_backend;
    const int uses_dynamic_leaf = dynamic_leaf_backend || staged_backend;
    const int lm_constant_optimizer = strcmp(constant_optimizer_mode, "lm") == 0;
    const int structured_leaf_settings =
        secant_sr_app_leaf_settings_policy_is_structured(leaf_settings_policy);
    const int rotating_leaf_settings =
        secant_sr_app_leaf_settings_policy_is_rotating(leaf_settings_policy);
    const int validate_each_generation = strcmp(validation_mode, "each") == 0;
    const int stop_on_validation = strcmp(stop_metric, "validation") == 0;
    const SecantSRSearchConfig config = {
        population_size,
        uses_dynamic_leaf ? 192u : 160u,
        dynamic_leaf_backend || maturity_backend ? dynamic_max_nodes : 72u,
        16u,
        40u,
        dataset->num_inputs,
        12u,
        4u,
        population_size < 32u ? population_size : 32u,
        5u,
        1.5,
        0.45,
        0.30,
        0.20,
        0.15,
        parsimony_coefficient,
        seed,
        uses_dynamic_leaf && !maturity_backend && (dynamic_leaf_backend || dynamic_generations != 0u)
            ? dynamic_max_nodes
            : 0u,
        uses_dynamic_leaf && !maturity_backend && (dynamic_leaf_backend || dynamic_generations != 0u)
            ? num_dynamic_leaves
            : 0u,
        qd_column_buckets,
        qd_transcendental_buckets,
        archive_parent_probability,
        constant_setting_credit_weight
    };
    SecantSRSearch search = NULL;
    void* storage = NULL;
    float* input = NULL;
    float* target = NULL;
    float* validation_input = NULL;
    float* validation_target = NULL;
    float* validation_prediction = NULL;
    float* sse = NULL;
    uint32_t* leaf_masks = NULL;
    uint32_t* leaf_words = NULL;
    uint32_t* mixed_leaf_masks = NULL;
    uint32_t* mixed_leaf_words = NULL;
    float* constant_settings = NULL;
    uint32_t* best_setting_indices = NULL;
    uint32_t* selected_population_indices = NULL;
    uint32_t* mixed_population_indices = NULL;
    SecantAstInstruction* dynamic_program_storage = NULL;
    const SecantAstInstruction** dynamic_asts = NULL;
    size_t* proposal_program_sizes = NULL;
    float* constant_scales = NULL;
    float* constant_velocities = NULL;
    float* constant_incumbent_sse = NULL;
    float* lm_initial_constants = NULL;
    float* lm_optimized_constants = NULL;
    float* lm_accepted_sse = NULL;
    uint32_t* lm_leaf_masks = NULL;
    uint32_t* lm_leaf_words = NULL;
    uint32_t* lm_best_leaf_masks = NULL;
    uint32_t* lm_best_leaf_words = NULL;
    SecantSRAppCheckpoint* checkpoints = NULL;
    unsigned char* checkpoint_programs = NULL;
    size_t checkpoint_metadata_bytes = 0u;
    size_t checkpoint_program_bytes = 0u;
    size_t checkpoint_count = 0u;
    size_t storage_size = 0u;
    size_t sse_elements = population_size;
    size_t dynamic_ast_capacity = dynamic_batch_asts < population_size ? dynamic_batch_asts : population_size;
    size_t lm_ast_capacity = lm_batch_asts < dynamic_ast_capacity ? lm_batch_asts : dynamic_ast_capacity;
    size_t constant_ast_capacity = lm_constant_optimizer ? lm_ast_capacity : dynamic_ast_capacity;
    size_t constant_parameter_capacity = lm_constant_optimizer
        ? SECANT_SR_APP_LM_PARAMETERS
        : num_dynamic_leaves;
    size_t dynamic_program_capacity = 0u;
    size_t leaf_word_elements = 0u;
    size_t constant_setting_elements = 0u;
    size_t lm_constant_elements = 0u;
    size_t lm_sse_elements = 0u;
    size_t patch_capacity_instructions = 0u;
    double target_ssd;
    double validation_target_ssd;
    const double app_begin = secant_sr_app_seconds_get();
    double evaluator_destroy_seconds = 0.0;
    SecantSRResult result;
    int status = 1;
    size_t generation;
#ifdef SECANT_SR_HAS_CUBIN
    SecantSRCubinEvaluator cubin_evaluator;
    SecantSRCubinEvaluator mixed_dynamic_evaluator;
    SecantSRCubinStagedEvaluator staged_evaluator;
    SecantSRCubinLMEvaluator lm_evaluator;
    SecantSRCubinTemplateStats template_stats;
    SecantSRMSEReducer mse_reducer;
    SecantSRMSEEntry* mse_entries = NULL;
    int cubin_evaluator_active = 0;
    int mixed_dynamic_evaluator_active = 0;
    int staged_evaluator_active = 0;
    int lm_evaluator_active = 0;
    int mse_reducer_active = 0;

    memset(&cubin_evaluator, 0, sizeof(cubin_evaluator));
    memset(&mixed_dynamic_evaluator, 0, sizeof(mixed_dynamic_evaluator));
    memset(&staged_evaluator, 0, sizeof(staged_evaluator));
    memset(&lm_evaluator, 0, sizeof(lm_evaluator));
    memset(&template_stats, 0, sizeof(template_stats));
    memset(&mse_reducer, 0, sizeof(mse_reducer));
#else
    (void)num_workers;
    (void)num_streams;
    (void)tile_rows;
    (void)num_leaf_settings;
    (void)num_dynamic_leaves;
    (void)cuda_state;
#endif

    if (strcmp(operator_profile, "trig") == 0) {
        unary_ops = trig_unary_ops;
        binary_ops = arithmetic_binary_ops;
        num_unary_ops = sizeof(trig_unary_ops) / sizeof(trig_unary_ops[0]);
        num_binary_ops = sizeof(arithmetic_binary_ops) / sizeof(arithmetic_binary_ops[0]);
    } else if (strcmp(operator_profile, "algebraic") == 0) {
        unary_ops = algebraic_unary_ops;
        binary_ops = arithmetic_binary_ops;
        num_unary_ops = sizeof(algebraic_unary_ops) / sizeof(algebraic_unary_ops[0]);
        num_binary_ops = sizeof(arithmetic_binary_ops) / sizeof(arithmetic_binary_ops[0]);
    } else if (strcmp(operator_profile, "scientific") == 0) {
        unary_ops = scientific_unary_ops;
        binary_ops = arithmetic_binary_ops;
        num_unary_ops = sizeof(scientific_unary_ops) / sizeof(scientific_unary_ops[0]);
        num_binary_ops = sizeof(arithmetic_binary_ops) / sizeof(arithmetic_binary_ops[0]);
    }

    if (!secant_sr_app_size_mul(asts_per_kernel, patch_instructions_per_ast, &patch_capacity_instructions)) {
        fprintf(stderr, "patch capacity size overflow\n");
        return 1;
    }
#ifndef SECANT_SR_HAS_CUBIN
    (void)patch_capacity_instructions;
#endif
    if (uses_dynamic_leaf &&
        (!secant_sr_app_size_mul(population_size, config.max_program_bytes, &dynamic_program_capacity) ||
         !secant_sr_app_size_mul(num_leaf_settings, num_dynamic_leaves, &leaf_word_elements) ||
         !secant_sr_app_size_mul(
             constant_ast_capacity, constant_parameter_capacity, &constant_setting_elements))) {
        fprintf(stderr, "dynamic-leaf allocation size overflow\n");
        return 1;
    }
    if (uses_dynamic_leaf && lm_constant_optimizer &&
        (!secant_sr_app_size_mul(lm_ast_capacity, num_constant_settings, &lm_sse_elements) ||
         !secant_sr_app_size_mul(
             lm_sse_elements, SECANT_SR_APP_LM_PARAMETERS, &lm_constant_elements))) {
        fprintf(stderr, "dynamic-leaf allocation size overflow\n");
        return 1;
    }
    if (checkpoint_trace &&
        (!secant_sr_app_size_mul(generations, sizeof(*checkpoints), &checkpoint_metadata_bytes) ||
         !secant_sr_app_size_mul(generations, config.max_program_bytes, &checkpoint_program_bytes))) {
        fprintf(stderr, "checkpoint allocation size overflow\n");
        return 1;
    }

    result = secant_sr_search_storage_size(
        &config,
        num_unary_ops,
        num_binary_ops,
        sizeof(secant_sr_app_routines) / sizeof(secant_sr_app_routines[0]),
        sizeof(constants) / sizeof(constants[0]),
        &storage_size);
    if (result != SECANT_SR_SUCCESS) {
        fprintf(stderr, "storage measure failed: %s\n", secant_sr_result_to_string(result));
        return 1;
    }
    storage = malloc(storage_size);
    input = malloc(dataset->num_inputs * num_rows * sizeof(*input));
    target = malloc(num_rows * sizeof(*target));
    validation_input = malloc(dataset->num_inputs * validation_rows * sizeof(*validation_input));
    validation_target = malloc(validation_rows * sizeof(*validation_target));
    if (checkpoint_trace) {
        checkpoints = malloc(checkpoint_metadata_bytes);
        checkpoint_programs = malloc(checkpoint_program_bytes);
    }
    if (staged_backend && validate_each_generation) {
        validation_prediction = malloc(validation_rows * sizeof(*validation_prediction));
    }
    sse = malloc(sse_elements * sizeof(*sse));
    if (uses_dynamic_leaf) {
        leaf_masks = malloc(num_leaf_settings * sizeof(*leaf_masks));
        leaf_words = malloc(leaf_word_elements * sizeof(*leaf_words));
        if (structured_leaf_settings && maturity_backend) {
            mixed_leaf_masks = malloc(num_leaf_settings * sizeof(*mixed_leaf_masks));
            mixed_leaf_words = malloc(leaf_word_elements * sizeof(*mixed_leaf_words));
        }
        dynamic_program_storage = malloc(dynamic_program_capacity);
        dynamic_asts = malloc(population_size * sizeof(*dynamic_asts));
        proposal_program_sizes = malloc(population_size * sizeof(*proposal_program_sizes));
        best_setting_indices = malloc(population_size * sizeof(*best_setting_indices));
        selected_population_indices = malloc(population_size * sizeof(*selected_population_indices));
        mixed_population_indices = malloc(population_size * sizeof(*mixed_population_indices));
#ifdef SECANT_SR_HAS_CUBIN
        mse_entries = malloc(dynamic_ast_capacity * sizeof(*mse_entries));
#endif
        if (num_constant_settings != 0u) {
            constant_settings = malloc(constant_setting_elements * sizeof(*constant_settings));
            if (maturity_backend && !lm_constant_optimizer) {
                constant_scales = malloc(constant_setting_elements * sizeof(*constant_scales));
                constant_velocities = malloc(constant_setting_elements * sizeof(*constant_velocities));
                constant_incumbent_sse = malloc(constant_ast_capacity * sizeof(*constant_incumbent_sse));
            }
            if (lm_constant_optimizer) {
                lm_initial_constants = malloc(lm_constant_elements * sizeof(*lm_initial_constants));
                lm_optimized_constants = malloc(lm_constant_elements * sizeof(*lm_optimized_constants));
                lm_accepted_sse = malloc(lm_sse_elements * sizeof(*lm_accepted_sse));
                lm_leaf_masks = malloc(lm_sse_elements * sizeof(*lm_leaf_masks));
                lm_leaf_words = malloc(lm_constant_elements * sizeof(*lm_leaf_words));
                lm_best_leaf_masks = malloc(lm_ast_capacity * sizeof(*lm_best_leaf_masks));
                lm_best_leaf_words = malloc(constant_setting_elements * sizeof(*lm_best_leaf_words));
            }
        }
    }
    if (storage == NULL || input == NULL || target == NULL || validation_input == NULL ||
        validation_target == NULL ||
        (checkpoint_trace && (checkpoints == NULL || checkpoint_programs == NULL)) ||
        (staged_backend && validate_each_generation && validation_prediction == NULL) || sse == NULL ||
        (uses_dynamic_leaf &&
         (leaf_masks == NULL || leaf_words == NULL || dynamic_program_storage == NULL || dynamic_asts == NULL ||
          proposal_program_sizes == NULL || best_setting_indices == NULL || selected_population_indices == NULL ||
          mixed_population_indices == NULL ||
#ifdef SECANT_SR_HAS_CUBIN
          mse_entries == NULL ||
#endif
          (structured_leaf_settings && maturity_backend &&
           (mixed_leaf_masks == NULL || mixed_leaf_words == NULL)) ||
          (num_constant_settings != 0u &&
           (constant_settings == NULL ||
            (maturity_backend && !lm_constant_optimizer &&
             (constant_scales == NULL || constant_velocities == NULL || constant_incumbent_sse == NULL)) ||
            (lm_constant_optimizer &&
             (lm_initial_constants == NULL || lm_optimized_constants == NULL || lm_accepted_sse == NULL ||
              lm_leaf_masks == NULL || lm_leaf_words == NULL || lm_best_leaf_masks == NULL ||
              lm_best_leaf_words == NULL))))))) {
        fprintf(stderr, "allocation failed\n");
    } else {
        if (dataset_binary_path != NULL) {
            if (!secant_sr_dataset_binary_load(
                    dataset_binary_path,
                    dataset->num_inputs,
                    input,
                    target,
                    num_rows,
                    validation_input,
                    validation_target,
                    validation_rows,
                    &target_ssd,
                    &validation_target_ssd)) {
                fprintf(stderr, "dataset binary load failed: %s\n", dataset_binary_path);
                target_ssd = 0.0;
                validation_target_ssd = 0.0;
            }
        } else {
            target_ssd = secant_sr_dataset_fill(dataset, seed ^ UINT64_C(0x747261696e), input, target, num_rows);
            validation_target_ssd = secant_sr_dataset_fill(
                dataset,
                seed ^ UINT64_C(0x76616c6964617465),
                validation_input,
                validation_target,
                validation_rows);
        }
        result = secant_sr_search_init(
            &config,
            unary_ops,
            num_unary_ops,
            binary_ops,
            num_binary_ops,
            secant_sr_app_routines,
            sizeof(secant_sr_app_routines) / sizeof(secant_sr_app_routines[0]),
            constants,
            sizeof(constants) / sizeof(constants[0]),
            storage,
            storage_size,
            &search);
        if (result != SECANT_SR_SUCCESS || target_ssd <= 0.0 || validation_target_ssd <= 0.0) {
            fprintf(stderr, "search init failed: %s\n", secant_sr_result_to_string(result));
        } else {
#ifdef SECANT_SR_HAS_CUBIN
            if (result == SECANT_SR_SUCCESS && uses_dynamic_leaf) {
                if (!structured_leaf_settings) {
                    secant_sr_app_leaf_settings_fill(
                        num_leaf_settings,
                        num_dynamic_leaves,
                        dataset->num_inputs,
                        constants,
                        sizeof(constants) / sizeof(constants[0]),
                        seed ^ UINT64_C(0x6c656166736574),
                        leaf_masks,
                        leaf_words);
                } else if (!secant_sr_app_leaf_settings_virtual_bank_fill(
                               num_leaf_settings,
                               num_dynamic_leaves,
                               num_dynamic_leaves,
                               dataset->num_inputs,
                               constants,
                               sizeof(constants) / sizeof(constants[0]),
                               seed ^ UINT64_C(0x6c656166736574),
                               0u,
                               0u,
                               leaf_masks,
                               leaf_words) ||
                           (maturity_backend &&
                            !secant_sr_app_leaf_settings_virtual_bank_fill(
                                num_leaf_settings,
                                mixed_dynamic_leaves,
                                num_dynamic_leaves,
                                dataset->num_inputs,
                                constants,
                                sizeof(constants) / sizeof(constants[0]),
                                seed ^ UINT64_C(0x6c656166736574),
                                0u,
                                1u,
                                mixed_leaf_masks,
                                mixed_leaf_words))) {
                    fprintf(stderr, "virtual leaf setting bank initialization failed\n");
                    result = SECANT_SR_ERROR_INVALID_VALUE;
                }
                if (result == SECANT_SR_SUCCESS && staged_backend) {
                    staged_evaluator_active = secant_sr_cubin_staged_evaluator_create(
                        num_kernels,
                        asts_per_kernel,
                        dataset->num_inputs,
                        num_dynamic_leaves,
                        static_tile_rows,
                        tile_rows,
                        128u,
                        patch_capacity_instructions,
                        num_workers,
                        num_streams,
                        input,
                        target,
                        leaf_masks,
                        leaf_words,
                        num_dynamic_leaves,
                        num_leaf_settings,
                        NULL,
                        0u,
                        0u,
                        0u,
                        validate_each_generation ? validation_input : NULL,
                        validate_each_generation ? validation_rows : 0u,
                        population_size,
                        dynamic_ast_capacity,
                        num_rows,
                        &cuda_state->session,
                        cuda_state->cache_ptr,
                        &staged_evaluator);
                    if (staged_evaluator_active) {
                        staged_evaluator_active = secant_sr_app_staged_evaluator_routines_set(&staged_evaluator);
                    }
                    if (staged_evaluator_active && maturity_backend) {
                        mixed_dynamic_evaluator_active = secant_sr_cubin_mixed_dynamic_leaf_evaluator_create(
                            num_kernels,
                            asts_per_kernel,
                            num_dynamic_leaves,
                            tile_rows,
                            128u,
                            patch_capacity_instructions,
                            num_workers,
                            num_streams,
                            input,
                            dataset->num_inputs,
                            target,
                            mixed_leaf_masks != NULL ? mixed_leaf_masks : leaf_masks,
                            mixed_leaf_words != NULL ? mixed_leaf_words : leaf_words,
                            num_dynamic_leaves,
                            num_leaf_settings,
                            dynamic_ast_capacity,
                            num_rows,
                            &cuda_state->session,
                            cuda_state->cache_ptr,
                            &mixed_dynamic_evaluator);
                        if (mixed_dynamic_evaluator_active) {
                            mixed_dynamic_evaluator_active =
                                secant_sr_app_evaluator_routines_set(&mixed_dynamic_evaluator);
                        }
                    }
                } else if (result == SECANT_SR_SUCCESS) {
                    cubin_evaluator_active = secant_sr_cubin_dynamic_leaf_evaluator_create(
                        num_kernels,
                        asts_per_kernel,
                        num_dynamic_leaves,
                        tile_rows,
                        128u,
                        patch_capacity_instructions,
                        num_workers,
                        num_streams,
                        input,
                        dataset->num_inputs,
                        target,
                        leaf_masks,
                        leaf_words,
                        num_dynamic_leaves,
                        num_leaf_settings,
                        dynamic_ast_capacity,
                        num_rows,
                        &cuda_state->session,
                        cuda_state->cache_ptr,
                        &cubin_evaluator);
                    if (cubin_evaluator_active) {
                        cubin_evaluator_active = secant_sr_app_evaluator_routines_set(&cubin_evaluator);
                    }
                }
                if ((staged_evaluator_active || cubin_evaluator_active) && uses_dynamic_leaf) {
                    mse_reducer_active = secant_sr_mse_reducer_create(
                        dynamic_ast_capacity,
                        1u,
                        &cuda_state->session,
                        cuda_state->cache_ptr,
                        &mse_reducer);
                    if (!mse_reducer_active) {
                        fprintf(stderr, "MSE reducer creation failed\n");
                    } else {
                        printf(
                            "mse_reducer_cache enabled=%d hits=%zu misses=%zu invalidations=%zu "
                            "lookup_seconds=%.9f store_seconds=%.9f actual_compile_seconds=%.9f "
                            "estimated_uncached_compile_seconds=%.9f\n",
                            cuda_state->cache_ptr != NULL,
                            mse_reducer.cache_hits,
                            mse_reducer.cache_misses,
                            mse_reducer.cache_invalidations,
                            mse_reducer.cache_lookup_seconds,
                            mse_reducer.cache_store_seconds,
                            mse_reducer.compile_seconds,
                            mse_reducer.estimated_uncached_compile_seconds);
                    }
                }
            } else if (result == SECANT_SR_SUCCESS && strcmp(backend, "cubin") == 0) {
                cubin_evaluator_active = secant_sr_cubin_evaluator_create(
                    num_kernels,
                    asts_per_kernel,
                    dataset->num_inputs,
                    tile_rows,
                    128u,
                    patch_capacity_instructions,
                    num_workers,
                    num_streams,
                    input,
                    target,
                    population_size,
                    num_rows,
                    &cuda_state->session,
                    cuda_state->cache_ptr,
                    &cubin_evaluator);
                if (cubin_evaluator_active) {
                    cubin_evaluator_active = secant_sr_app_evaluator_routines_set(&cubin_evaluator);
                }
            }
            if ((strcmp(backend, "cubin") == 0 || dynamic_leaf_backend) &&
                (!cubin_evaluator_active || (dynamic_leaf_backend && !mse_reducer_active))) {
                fprintf(stderr, "CUBIN evaluator creation failed; CUDA_MODULE_LOADING must be EAGER\n");
            }
            if (staged_backend && (!staged_evaluator_active || !mse_reducer_active ||
                                   (maturity_backend && !mixed_dynamic_evaluator_active))) {
                fprintf(stderr, "staged CUBIN evaluator creation failed; CUDA_MODULE_LOADING must be EAGER\n");
            }
            if (cubin_evaluator_active || staged_evaluator_active) {
                const SecantSRCubinTemplateStats* values[5];
                size_t num_values;
                size_t value_idx;

                if (staged_evaluator_active) {
                    values[0] = &staged_evaluator.static_sse.template_stats;
                    values[1] = &staged_evaluator.dynamic_leaf.template_stats;
                    values[2] = &staged_evaluator.dynamic_constant.template_stats;
                    values[3] = &staged_evaluator.materialize.template_stats;
                    num_values = staged_evaluator.dynamic_constant.runner != NULL ? 4u : 3u;
                    if (staged_evaluator.dynamic_constant.runner == NULL) {
                        values[2] = values[3];
                    }
                    if (maturity_backend) {
                        values[num_values++] = &mixed_dynamic_evaluator.template_stats;
                    }
                } else {
                    values[0] = &cubin_evaluator.template_stats;
                    num_values = 1u;
                }
                for (value_idx = 0u; value_idx < num_values; ++value_idx) {
                    secant_sr_app_template_stats_add(&template_stats, values[value_idx]);
                }
                secant_sr_app_template_stats_print(cuda_state, &template_stats);
            }
#else
            if (strcmp(backend, "cubin") == 0 || uses_dynamic_leaf) {
                fprintf(stderr, "this build has no CUBIN evaluator\n");
            }
#endif
            if (strcmp(backend, "cpu") != 0
#ifdef SECANT_SR_HAS_CUBIN
                && !cubin_evaluator_active && !staged_evaluator_active
#endif
            ) {
                result = SECANT_SR_ERROR_INVALID_VALUE;
            }
            for (generation = 0u; generation < generations; ++generation) {
                const SecantSRIndividual* best;
                SecantSRProgramFeatures best_features;
                const double evaluation_begin = secant_sr_app_seconds_get();
                double validation_r2;
                double materialize_validation_r2 = NAN;
                int materialize_validation_diverged = 0;
                double evaluation_seconds;
                double generation_seconds = 0.0;
                double compile_seconds = 0.0;
                double module_load_seconds = 0.0;
                double device_runtime_seconds = 0.0;
                double leaf_setting_update_seconds = 0.0;
                double validation_seconds = 0.0;
                double validation_cpu_seconds = 0.0;
                double validation_gpu_seconds = 0.0;
                double validation_score_seconds = 0.0;
                double validation_gpu_compile_seconds = 0.0;
                double validation_gpu_module_load_seconds = 0.0;
                double validation_gpu_device_runtime_seconds = 0.0;
                double maturity_static_seconds = 0.0;
                double maturity_full_seconds = 0.0;
                double maturity_mixed_seconds = 0.0;
                double maturity_constant_seconds = 0.0;
                double maturity_full_r2_gain_sum = 0.0;
                double maturity_full_r2_gain_max = 0.0;
                double maturity_mixed_r2_gain_sum = 0.0;
                double maturity_mixed_r2_gain_max = 0.0;
                double maturity_constant_r2_gain_sum = 0.0;
                double maturity_constant_r2_gain_max = 0.0;
                double maturity_best_r2_after_static = -INFINITY;
                double maturity_best_r2_after_full = -INFINITY;
                double maturity_best_r2_after_mixed = -INFINITY;
                double maturity_best_r2_after_constant = -INFINITY;
                double stop_value;
                int stop_reached;
                int time_limit_reached;
                size_t maturity_before[4] = {0u, 0u, 0u, 0u};
                size_t maturity_after[4] = {0u, 0u, 0u, 0u};
                size_t maturity_full_selected = 0u;
                size_t maturity_full_promoted = 0u;
                size_t maturity_full_finite_gain_count = 0u;
                size_t maturity_full_nonfinite_recovered = 0u;
                size_t maturity_mixed_eligible = 0u;
                size_t maturity_mixed_selected = 0u;
                size_t maturity_mixed_promoted = 0u;
                size_t maturity_mixed_finite_gain_count = 0u;
                size_t maturity_mixed_nonfinite_recovered = 0u;
                size_t maturity_constant_eligible = 0u;
                size_t maturity_constant_selected = 0u;
                size_t maturity_constant_finite_gain_count = 0u;
                size_t maturity_constant_nonfinite_recovered = 0u;
                size_t lm_column_binding_winners = 0u;
                size_t lm_column_binding_promoted = 0u;
                int maturity_best_origin = -1;
                int maturity_best_maturity = -1;
                size_t dynamic_promoted = 0u;
                size_t constant_promoted = 0u;
                SecantSRArchiveStats archive_stats;
                double evaluator_row_evals = (double)population_size * (double)num_rows;
                const size_t evaluation_max_nodes = secant_sr_search_active_max_nodes_get(search);
                const char* evaluation_stage = dynamic_leaf_backend ||
                        maturity_backend || (legacy_staged_backend && generation < dynamic_generations)
                    ? "dynamic_leaf"
                    : "static_sse";

#ifdef SECANT_SR_HAS_CUBIN
                if (result == SECANT_SR_SUCCESS && generation != 0u && rotating_leaf_settings &&
                    (dynamic_leaf_backend || maturity_backend || generation < dynamic_generations)) {
                    const double update_begin = secant_sr_app_seconds_get();
                    SecantSRCubinEvaluator* full_evaluator = staged_evaluator_active
                        ? &staged_evaluator.dynamic_leaf
                        : &cubin_evaluator;

                    if (structured_leaf_settings) {
                        if (!secant_sr_app_leaf_settings_virtual_bank_fill(
                                num_leaf_settings,
                                num_dynamic_leaves,
                                num_dynamic_leaves,
                                dataset->num_inputs,
                                constants,
                                sizeof(constants) / sizeof(constants[0]),
                                seed ^ UINT64_C(0x6c656166736574),
                                generation,
                                0u,
                                leaf_masks,
                                leaf_words) ||
                            !secant_sr_cubin_dynamic_leaf_settings_update(
                                full_evaluator,
                                leaf_masks,
                                leaf_words,
                                num_dynamic_leaves,
                                num_leaf_settings) ||
                            (maturity_backend &&
                             (!secant_sr_app_leaf_settings_virtual_bank_fill(
                                  num_leaf_settings,
                                  mixed_dynamic_leaves,
                                  num_dynamic_leaves,
                                  dataset->num_inputs,
                                  constants,
                                  sizeof(constants) / sizeof(constants[0]),
                                  seed ^ UINT64_C(0x6c656166736574),
                                  generation,
                                  1u,
                                  mixed_leaf_masks,
                                  mixed_leaf_words) ||
                              !secant_sr_cubin_dynamic_leaf_settings_update(
                                  &mixed_dynamic_evaluator,
                                  mixed_leaf_masks,
                                  mixed_leaf_words,
                                  num_dynamic_leaves,
                                  num_leaf_settings)))) {
                            result = SECANT_SR_ERROR_SECANT;
                        }
                    } else {
                        const uint64_t rotating_seed = (seed ^ UINT64_C(0x6c656166736574)) ^
                            generation * UINT64_C(0x9e3779b97f4a7c15);

                        secant_sr_app_leaf_settings_fill(
                            num_leaf_settings,
                            num_dynamic_leaves,
                            dataset->num_inputs,
                            constants,
                            sizeof(constants) / sizeof(constants[0]),
                            rotating_seed,
                            leaf_masks,
                            leaf_words);
                        if (!secant_sr_cubin_dynamic_leaf_settings_update(
                                full_evaluator,
                                leaf_masks,
                                leaf_words,
                                num_dynamic_leaves,
                                num_leaf_settings) ||
                            (maturity_backend &&
                             !secant_sr_cubin_dynamic_leaf_settings_update(
                                 &mixed_dynamic_evaluator,
                                 leaf_masks,
                                 leaf_words,
                                 num_dynamic_leaves,
                                 num_leaf_settings))) {
                            result = SECANT_SR_ERROR_SECANT;
                        }
                    }
                    if (result != SECANT_SR_SUCCESS) {
                        fprintf(stderr, "leaf setting bank update failed at generation %zu\n", generation);
                        result = SECANT_SR_ERROR_SECANT;
                    }
                    leaf_setting_update_seconds = secant_sr_app_seconds_get() - update_begin;
                }
#endif
                if (result != SECANT_SR_SUCCESS) {
                    /* Preserve the first settings update error. */
                } else if (strcmp(backend, "cpu") == 0) {
                    result = secant_sr_search_cpu_evaluate(
                        search,
                        secant_sr_app_routine_programs,
                        sizeof(secant_sr_app_routine_programs) / sizeof(secant_sr_app_routine_programs[0]),
                        input,
                        dataset->num_inputs * num_rows,
                        num_rows,
                        target,
                        num_rows,
                        num_rows,
                        target_ssd,
                        sse,
                        population_size);
                }
#ifdef SECANT_SR_HAS_CUBIN
                else if (staged_evaluator_active) {
                    const int dynamic_stage = maturity_backend || generation < dynamic_generations;
                    const SecantAstInstruction* const* asts = NULL;
                    size_t num_asts = 0u;

                    result = secant_sr_search_asts_get(search, &asts, &num_asts);
                    if (result == SECANT_SR_SUCCESS && maturity_backend) {
                        const SecantSRIndividual* individuals = NULL;
                        size_t num_individuals = 0u;
                        size_t num_full = 0u;
                        size_t num_mixed_eligible = 0u;
                        size_t num_mixed = 0u;
                        size_t individual_idx;
                        const double static_begin = secant_sr_app_seconds_get();
                        const uint64_t mixed_selection_seed = seed ^ ((uint64_t)generation << 32u) ^
                            UINT64_C(0x6d6978656473656c);

                        if (!secant_sr_cubin_evaluator_run(&staged_evaluator.static_sse, asts, num_asts, sse) ||
                            secant_sr_search_scores_set(search, sse, num_asts, num_rows, target_ssd) !=
                                SECANT_SR_SUCCESS ||
                            secant_sr_search_individuals_get(search, &individuals, &num_individuals) !=
                                SECANT_SR_SUCCESS ||
                            num_individuals != num_asts) {
                            fprintf(stderr, "maturity static baseline evaluation failed\n");
                            result = SECANT_SR_ERROR_SECANT;
                        }
                        maturity_static_seconds = secant_sr_app_seconds_get() - static_begin;
                        compile_seconds += staged_evaluator.static_sse.last_stats.compile_window_seconds;
                        module_load_seconds += staged_evaluator.static_sse.last_stats.module_load_seconds;
                        device_runtime_seconds += staged_evaluator.static_sse.last_stats.runtime_seconds;
                        for (individual_idx = 0u; result == SECANT_SR_SUCCESS && individual_idx < num_asts;
                             ++individual_idx) {
                            const SecantSRIndividual* individual = individuals + individual_idx;

                            if ((size_t)individual->maturity < sizeof(maturity_before) / sizeof(maturity_before[0])) {
                                ++maturity_before[individual->maturity];
                            }
                            if (individual->maturity == SECANT_SR_MATURITY_EXPLORATORY &&
                                individual->num_leaves <= num_dynamic_leaves) {
                                selected_population_indices[num_full++] = (uint32_t)individual_idx;
                            } else {
                                ++num_mixed_eligible;
                                if (secant_sr_app_probability_select(
                                        mixed_refine_probability,
                                        mixed_selection_seed ^ ((uint64_t)individual_idx + 1u) *
                                            UINT64_C(0x9e3779b97f4a7c15),
                                        individual->fingerprint)) {
                                    mixed_population_indices[num_mixed++] = (uint32_t)individual_idx;
                                }
                            }
                        }
                        if (result == SECANT_SR_SUCCESS && maturity_trace) {
                            result = secant_sr_app_best_snapshot_get(
                                search, &maturity_best_r2_after_static, NULL, NULL);
                        }
                        if (result == SECANT_SR_SUCCESS) {
                            size_t promoted = 0u;
                            const double full_begin = secant_sr_app_seconds_get();

                            result = secant_sr_app_dynamic_leaf_cohort_run(
                                search,
                                selected_population_indices,
                                num_full,
                                SECANT_SR_DYNAMIC_LEAF_PROJECTION_FULL,
                                num_dynamic_leaves,
                                seed ^ ((uint64_t)generation << 32u) ^ UINT64_C(0x66756c6c),
                                leaf_masks,
                                leaf_words,
                                leaf_word_elements,
                                num_leaf_settings,
                                dynamic_ast_capacity,
                                num_rows,
                                target_ssd,
                                &staged_evaluator.dynamic_leaf,
                                &staged_evaluator.static_sse,
                                &mse_reducer,
                                dynamic_program_storage,
                                dynamic_program_capacity,
                                dynamic_asts,
                                proposal_program_sizes,
                                best_setting_indices,
                                mse_entries,
                                sse,
                                maturity_trace,
                                &promoted,
                                &maturity_full_r2_gain_sum,
                                &maturity_full_r2_gain_max,
                                &maturity_full_finite_gain_count,
                                &maturity_full_nonfinite_recovered,
                                &compile_seconds,
                                &module_load_seconds,
                                &device_runtime_seconds);
                            maturity_full_seconds = secant_sr_app_seconds_get() - full_begin;
                            maturity_full_selected = num_full;
                            maturity_full_promoted = promoted;
                            dynamic_promoted += promoted;
                            if (result == SECANT_SR_SUCCESS && maturity_trace) {
                                result = secant_sr_app_best_snapshot_get(
                                    search, &maturity_best_r2_after_full, NULL, NULL);
                            }
                        }
                        if (result == SECANT_SR_SUCCESS) {
                            size_t promoted = 0u;
                            const double mixed_begin = secant_sr_app_seconds_get();

                            result = secant_sr_app_dynamic_leaf_cohort_run(
                                search,
                                mixed_population_indices,
                                num_mixed,
                                SECANT_SR_DYNAMIC_LEAF_PROJECTION_MIXED,
                                mixed_dynamic_leaves,
                                seed ^ ((uint64_t)generation << 32u) ^ UINT64_C(0x6d69786564),
                                mixed_leaf_masks != NULL ? mixed_leaf_masks : leaf_masks,
                                mixed_leaf_words != NULL ? mixed_leaf_words : leaf_words,
                                leaf_word_elements,
                                num_leaf_settings,
                                dynamic_ast_capacity,
                                num_rows,
                                target_ssd,
                                &mixed_dynamic_evaluator,
                                &staged_evaluator.static_sse,
                                &mse_reducer,
                                dynamic_program_storage,
                                dynamic_program_capacity,
                                dynamic_asts,
                                proposal_program_sizes,
                                best_setting_indices,
                                mse_entries,
                                sse,
                                maturity_trace,
                                &promoted,
                                &maturity_mixed_r2_gain_sum,
                                &maturity_mixed_r2_gain_max,
                                &maturity_mixed_finite_gain_count,
                                &maturity_mixed_nonfinite_recovered,
                                &compile_seconds,
                                &module_load_seconds,
                                &device_runtime_seconds);
                            maturity_mixed_seconds = secant_sr_app_seconds_get() - mixed_begin;
                            maturity_mixed_eligible = num_mixed_eligible;
                            maturity_mixed_selected = num_mixed;
                            maturity_mixed_promoted = promoted;
                            dynamic_promoted += promoted;
                            if (result == SECANT_SR_SUCCESS && maturity_trace) {
                                result = secant_sr_app_best_snapshot_get(
                                    search, &maturity_best_r2_after_mixed, NULL, NULL);
                            }
                        }
                        evaluator_row_evals = ((double)num_asts +
                            (double)(num_full + num_mixed) * ((double)num_leaf_settings + 1.0)) *
                            (double)num_rows;
                        if (result == SECANT_SR_SUCCESS) {
                            printf(
                                "maturity_leaf_search full=%zu mixed_eligible=%zu mixed=%zu mixed_probability=%.9g "
                                "mixed_holes=%zu promoted=%zu settings=%zu\n",
                                num_full,
                                num_mixed_eligible,
                                num_mixed,
                                mixed_refine_probability,
                                mixed_dynamic_leaves,
                                dynamic_promoted,
                                num_leaf_settings);
                        }
                    } else if (result == SECANT_SR_SUCCESS && dynamic_stage) {
                        size_t required_program_size;
                        size_t max_dynamic_leaves;

                        if (!secant_sr_cubin_evaluator_run(&staged_evaluator.static_sse, asts, num_asts, sse) ||
                            secant_sr_search_scores_set(search, sse, num_asts, num_rows, target_ssd) !=
                                SECANT_SR_SUCCESS) {
                            fprintf(stderr, "staged static baseline evaluation failed\n");
                            result = SECANT_SR_ERROR_SECANT;
                        }
                        compile_seconds += staged_evaluator.static_sse.last_stats.compile_window_seconds;
                        module_load_seconds += staged_evaluator.static_sse.last_stats.module_load_seconds;
                        device_runtime_seconds += staged_evaluator.static_sse.last_stats.runtime_seconds;

                        if (result == SECANT_SR_SUCCESS) {
                            result = secant_sr_search_dynamic_leaf_programs_write(
                                search,
                                dynamic_program_storage,
                                dynamic_program_capacity,
                                dynamic_asts,
                                population_size,
                                &required_program_size,
                                &num_asts,
                                &max_dynamic_leaves);
                        }
                        if (result == SECANT_SR_SUCCESS &&
                            (required_program_size > dynamic_program_capacity ||
                             max_dynamic_leaves > staged_evaluator.dynamic_leaf.num_dynamic_leaves)) {
                            result = SECANT_SR_ERROR_BAD_PROGRAM;
                        }
                        if (result == SECANT_SR_SUCCESS) {
                            size_t chunk_begin;

                            for (chunk_begin = 0u; chunk_begin < num_asts; chunk_begin += dynamic_ast_capacity) {
                                const size_t chunk_asts = num_asts - chunk_begin < dynamic_ast_capacity
                                    ? num_asts - chunk_begin
                                    : dynamic_ast_capacity;
                                size_t chunk_idx;

                                if (!secant_sr_cubin_dynamic_leaf_evaluator_run_device(
                                        &staged_evaluator.dynamic_leaf, dynamic_asts + chunk_begin, chunk_asts) ||
                                    !secant_sr_mse_reducer_best_get(
                                        &mse_reducer,
                                        staged_evaluator.dynamic_leaf.output,
                                        chunk_asts,
                                        num_leaf_settings,
                                        num_leaf_settings,
                                        num_rows,
                                        target_ssd,
                                        mse_entries,
                                        dynamic_ast_capacity)) {
                                    fprintf(stderr, "staged dynamic-leaf reduction failed\n");
                                    result = SECANT_SR_ERROR_SECANT;
                                    break;
                                }
                                compile_seconds += staged_evaluator.dynamic_leaf.last_stats.compile_window_seconds;
                                module_load_seconds += staged_evaluator.dynamic_leaf.last_stats.module_load_seconds;
                                device_runtime_seconds += staged_evaluator.dynamic_leaf.last_stats.runtime_seconds;
                                for (chunk_idx = 0u; chunk_idx < chunk_asts; ++chunk_idx) {
                                    best_setting_indices[chunk_begin + chunk_idx] = mse_entries[chunk_idx].setting_idx;
                                }
                            }
                        }

                        if (result == SECANT_SR_SUCCESS) {
                            result = secant_sr_search_dynamic_leaf_proposals_from_settings_write(
                                search,
                                leaf_masks,
                                num_leaf_settings,
                                leaf_words,
                                leaf_word_elements,
                                num_dynamic_leaves,
                                num_leaf_settings,
                                best_setting_indices,
                                num_asts,
                                dynamic_program_storage,
                                dynamic_program_capacity,
                                dynamic_asts,
                                proposal_program_sizes,
                                population_size,
                                &required_program_size);
                        }
                        if (result == SECANT_SR_SUCCESS &&
                            !secant_sr_cubin_evaluator_run(
                                &staged_evaluator.static_sse, dynamic_asts, num_asts, sse)) {
                            fprintf(stderr, "staged proposal evaluation failed\n");
                            result = SECANT_SR_ERROR_SECANT;
                        }
                        compile_seconds += staged_evaluator.static_sse.last_stats.compile_window_seconds;
                        module_load_seconds += staged_evaluator.static_sse.last_stats.module_load_seconds;
                        device_runtime_seconds += staged_evaluator.static_sse.last_stats.runtime_seconds;

                        if (result == SECANT_SR_SUCCESS) {
                            result = secant_sr_search_proposals_apply(
                                search,
                                dynamic_asts,
                                proposal_program_sizes,
                                sse,
                                NULL,
                                num_asts,
                                num_rows,
                                target_ssd,
                                SECANT_SR_ORIGIN_DYNAMIC_LEAF,
                                &dynamic_promoted);
                        }
                        evaluator_row_evals *= (double)num_leaf_settings + 2.0;
                    } else if (result == SECANT_SR_SUCCESS) {
                        if (!secant_sr_cubin_evaluator_run(&staged_evaluator.static_sse, asts, num_asts, sse) ||
                            secant_sr_search_scores_set(search, sse, num_asts, num_rows, target_ssd) !=
                                SECANT_SR_SUCCESS) {
                            result = SECANT_SR_ERROR_SECANT;
                        }
                        compile_seconds = staged_evaluator.static_sse.last_stats.compile_window_seconds;
                        module_load_seconds = staged_evaluator.static_sse.last_stats.module_load_seconds;
                        device_runtime_seconds = staged_evaluator.static_sse.last_stats.runtime_seconds;
                    }
                    if (result == SECANT_SR_SUCCESS && num_constant_settings != 0u &&
                        (constant_optimize_probability > 0.0 ||
                         (constant_optimize_probability < 0.0 && constant_optimize_budget != 0u)) &&
                        dataset->num_inputs <= 32u && secant_sr_app_constant_optimizer_generation_select(
                            constant_sweep_phase,
                            generation,
                            generations,
                            constant_optimize_interval,
                            dynamic_stage)) {
                        size_t num_eligible = 0u;
                        size_t num_selected = 0u;
                        double constant_optimizer_row_evals = 0.0;
                        const double constant_optimizer_begin = secant_sr_app_seconds_get();

                        result = constant_optimize_probability >= 0.0
                            ? secant_sr_search_dynamic_constant_indices_sample(
                                search,
                                constant_parameter_capacity,
                                constant_optimize_probability,
                                selected_population_indices,
                                population_size,
                                &num_eligible,
                                &num_selected)
                            : secant_sr_search_dynamic_constant_indices_select(
                                search,
                                constant_parameter_capacity,
                                constant_optimize_budget,
                                constant_optimize_random_fraction,
                                selected_population_indices,
                                population_size,
                                &num_eligible,
                                &num_selected);
                        if (result == SECANT_SR_SUCCESS && num_selected != 0u && lm_constant_optimizer &&
                            lm_evaluator.runner == NULL) {
                            const size_t lm_streams = num_streams < lm_ast_capacity
                                ? num_streams
                                : lm_ast_capacity;

                            if (!secant_sr_cubin_lm_evaluator_create(
                                    lm_ast_capacity,
                                    dataset->num_inputs,
                                    lm_tile_rows,
                                    lm_threads,
                                    lm_patch_instructions,
                                    lm_streams,
                                    input,
                                    target,
                                    num_constant_settings,
                                    num_rows,
                                    &cuda_state->session,
                                    &lm_evaluator) ||
                                !secant_sr_cubin_lm_evaluator_routines_set(
                                    &lm_evaluator,
                                    secant_sr_app_routine_programs,
                                    sizeof(secant_sr_app_routine_programs) /
                                        sizeof(secant_sr_app_routine_programs[0]))) {
                                fprintf(stderr, "LM constant-optimizer evaluator creation failed\n");
                                if (lm_evaluator.runner != NULL) {
                                    secant_sr_cubin_lm_evaluator_destroy(&lm_evaluator);
                                }
                                result = SECANT_SR_ERROR_SECANT;
                            } else {
                                lm_evaluator_active = 1;
                            }
                        }
                        if (result == SECANT_SR_SUCCESS && num_selected != 0u && !lm_constant_optimizer &&
                            staged_evaluator.constant_optimizer.runner == NULL) {
                            const int optimizer_created = maturity_backend
                                ? secant_sr_cubin_packed_constant_optimizer_evaluator_create(
                                    num_kernels,
                                    asts_per_kernel,
                                    dataset->num_inputs,
                                    num_dynamic_leaves,
                                    constant_optimizer_tile_rows,
                                    constant_optimizer_tile_rows,
                                    patch_capacity_instructions,
                                    num_workers,
                                    num_streams,
                                    input,
                                    target,
                                    num_constant_settings,
                                    dynamic_ast_capacity,
                                    num_rows,
                                    &cuda_state->session,
                                    cuda_state->cache_ptr,
                                    &staged_evaluator.constant_optimizer)
                                : secant_sr_cubin_constant_optimizer_evaluator_create(
                                    num_kernels,
                                    dataset->num_inputs,
                                    num_dynamic_leaves,
                                    constant_optimizer_tile_rows,
                                    constant_optimizer_tile_rows,
                                    patch_instructions_per_ast,
                                    num_workers,
                                    num_streams,
                                    input,
                                    target,
                                    num_constant_settings,
                                    dynamic_ast_capacity,
                                    num_rows,
                                    &cuda_state->session,
                                    cuda_state->cache_ptr,
                                    &staged_evaluator.constant_optimizer);

                            if (!optimizer_created) {
                                fprintf(stderr, "staged constant-optimizer evaluator creation failed\n");
                                result = SECANT_SR_ERROR_SECANT;
                            } else if (!secant_sr_app_evaluator_routines_set(
                                           &staged_evaluator.constant_optimizer)) {
                                fprintf(stderr, "staged constant-optimizer routine setup failed\n");
                                result = SECANT_SR_ERROR_SECANT;
                            } else {
                                secant_sr_app_template_stats_add(
                                    &template_stats, &staged_evaluator.constant_optimizer.template_stats);
                                secant_sr_app_template_stats_print(cuda_state, &template_stats);
                            }
                        }
                        if (result == SECANT_SR_SUCCESS && num_selected != 0u) {
                            size_t chunk_begin;

                            for (chunk_begin = 0u; chunk_begin < num_selected; chunk_begin += constant_ast_capacity) {
                                const size_t chunk_asts = num_selected - chunk_begin < constant_ast_capacity
                                    ? num_selected - chunk_begin
                                    : constant_ast_capacity;
                                size_t required_program_size = 0u;
                                size_t max_dynamic_constants = 0u;
                                size_t chunk_promoted = 0u;

                                result = lm_constant_optimizer
                                    ? secant_sr_search_dynamic_leaf_selected_programs_write(
                                        search,
                                        selected_population_indices + chunk_begin,
                                        chunk_asts,
                                        SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS,
                                        constant_parameter_capacity,
                                        0u,
                                        dynamic_program_storage,
                                        dynamic_program_capacity,
                                        dynamic_asts,
                                        constant_ast_capacity,
                                        &required_program_size,
                                        &max_dynamic_constants)
                                    : secant_sr_search_dynamic_constant_programs_selected_write(
                                        search,
                                        selected_population_indices + chunk_begin,
                                        chunk_asts,
                                        constant_parameter_capacity,
                                        dynamic_program_storage,
                                        dynamic_program_capacity,
                                        dynamic_asts,
                                        constant_ast_capacity,
                                        &required_program_size,
                                        &max_dynamic_constants);
                                if (result == SECANT_SR_SUCCESS) {
                                    result = secant_sr_search_constant_optimizer_centers_selected_write(
                                        search,
                                        selected_population_indices + chunk_begin,
                                        chunk_asts,
                                        constant_parameter_capacity,
                                        constant_settings,
                                        constant_setting_elements,
                                        constant_parameter_capacity);
                                }
                                if (result == SECANT_SR_SUCCESS && maturity_backend && !lm_constant_optimizer) {
                                    size_t state_idx;

                                    for (state_idx = 0u; state_idx < chunk_asts * constant_parameter_capacity;
                                         ++state_idx) {
                                        constant_scales[state_idx] = (float)constant_optimizer_scale;
                                        constant_velocities[state_idx] = 0.0f;
                                    }
                                    for (state_idx = 0u; state_idx < chunk_asts; ++state_idx) {
                                        constant_incumbent_sse[state_idx] = INFINITY;
                                    }
                                }
                                if (result != SECANT_SR_SUCCESS || required_program_size > dynamic_program_capacity ||
                                    max_dynamic_constants > constant_parameter_capacity ||
                                    (!lm_constant_optimizer && !maturity_backend &&
                                     !secant_sr_cubin_constant_optimizer_constants_upload(
                                         &staged_evaluator.constant_optimizer, constant_settings, chunk_asts))) {
                                    fprintf(stderr, "constant optimizer input preparation failed\n");
                                    result = SECANT_SR_ERROR_SECANT;
                                    break;
                                }
                                if (lm_constant_optimizer) {
                                    const SecantSRIndividual* individuals = NULL;
                                    size_t num_individuals = 0u;
                                    size_t chunk_idx;

                                    if (secant_sr_search_individuals_get(
                                            search, &individuals, &num_individuals) != SECANT_SR_SUCCESS) {
                                        result = SECANT_SR_ERROR_INVALID_VALUE;
                                    }
                                    for (chunk_idx = 0u; result == SECANT_SR_SUCCESS && chunk_idx < chunk_asts;
                                         ++chunk_idx) {
                                        const size_t population_idx =
                                            selected_population_indices[chunk_begin + chunk_idx];
                                        const SecantSRIndividual* individual;

                                        if (population_idx >= num_individuals) {
                                            result = SECANT_SR_ERROR_INVALID_VALUE;
                                            break;
                                        }
                                        individual = individuals + population_idx;
                                        if (!secant_sr_app_lm_binding_settings_fill(
                                                num_constant_settings / lm_starts_per_binding,
                                                lm_starts_per_binding,
                                                individual->num_constants,
                                                SECANT_SR_APP_LM_PARAMETERS,
                                                dataset->num_inputs,
                                                seed ^ UINT64_C(0x6c6d5f62696e64),
                                                generation,
                                                individual->fingerprint,
                                                lm_leaf_masks + chunk_idx * num_constant_settings,
                                                lm_leaf_words + chunk_idx * num_constant_settings *
                                                    SECANT_SR_APP_LM_PARAMETERS)) {
                                            result = SECANT_SR_ERROR_INVALID_VALUE;
                                        }
                                    }
                                    if (result == SECANT_SR_SUCCESS) {
                                        if (!secant_sr_app_lm_starts_write(
                                            constant_settings,
                                            chunk_asts,
                                            num_constant_settings,
                                            lm_starts_per_binding,
                                            constant_optimizer_scale,
                                            seed,
                                            generation,
                                            selected_population_indices + chunk_begin,
                                            lm_initial_constants,
                                            lm_constant_elements) ||
                                         !secant_sr_cubin_lm_evaluator_run_mixed(
                                            &lm_evaluator,
                                            dynamic_asts,
                                            chunk_asts,
                                            lm_leaf_masks,
                                            lm_leaf_words,
                                            lm_initial_constants,
                                            lm_settings_per_cta,
                                            constant_optimizer_iterations,
                                            (float)lm_initial_damping,
                                            lm_optimized_constants,
                                            lm_accepted_sse) ||
                                         !secant_sr_app_lm_best_write(
                                            lm_optimized_constants,
                                            lm_accepted_sse,
                                            lm_leaf_masks,
                                            lm_leaf_words,
                                            chunk_asts,
                                            num_constant_settings,
                                            SECANT_SR_APP_LM_PARAMETERS,
                                            constant_settings,
                                            lm_best_leaf_masks,
                                            lm_best_leaf_words)) {
                                            fprintf(stderr, "LM constant optimizer run failed\n");
                                            result = SECANT_SR_ERROR_SECANT;
                                        } else {
                                            for (chunk_idx = 0u; chunk_idx < chunk_asts; ++chunk_idx) {
                                                if (lm_best_leaf_masks[chunk_idx] != 0u) {
                                                    ++lm_column_binding_winners;
                                                }
                                            }
                                            compile_seconds += lm_evaluator.last_stats.compile_window_seconds;
                                            module_load_seconds += lm_evaluator.last_stats.module_load_seconds;
                                            device_runtime_seconds += lm_evaluator.last_stats.runtime_seconds;
                                        }
                                    }
                                } else if (!(maturity_backend
                                    ? secant_sr_cubin_packed_constant_optimizer_run(
                                        &staged_evaluator.constant_optimizer,
                                        dynamic_asts,
                                        chunk_asts,
                                        constant_settings,
                                        constant_scales,
                                        constant_velocities,
                                        constant_incumbent_sse,
                                        seed,
                                        generation,
                                        chunk_begin,
                                        constant_optimizer_iterations,
                                        0.0f,
                                        0.25f,
                                        (float)constant_optimizer_decay,
                                        1.0e-6f,
                                        1.0e6f)
                                    : secant_sr_cubin_constant_optimizer_run(
                                        &staged_evaluator.constant_optimizer,
                                        dynamic_asts,
                                        chunk_asts,
                                        seed,
                                        generation,
                                        0u,
                                        chunk_begin,
                                        (float)constant_optimizer_scale,
                                        (float)constant_optimizer_decay,
                                        constant_optimizer_iterations))) {
                                    fprintf(stderr, "constant optimizer run failed\n");
                                    result = SECANT_SR_ERROR_SECANT;
                                } else {
                                    compile_seconds +=
                                        staged_evaluator.constant_optimizer.last_stats.compile_window_seconds;
                                    module_load_seconds +=
                                        staged_evaluator.constant_optimizer.last_stats.module_load_seconds;
                                    device_runtime_seconds +=
                                        staged_evaluator.constant_optimizer.last_stats.runtime_seconds;
                                }
                                if (result != SECANT_SR_SUCCESS ||
                                    (!lm_constant_optimizer && !maturity_backend &&
                                     !secant_sr_cubin_constant_optimizer_constants_download(
                                         &staged_evaluator.constant_optimizer, constant_settings, chunk_asts))) {
                                    result = SECANT_SR_ERROR_SECANT;
                                    break;
                                }
                                result = lm_constant_optimizer
                                    ? secant_sr_search_dynamic_leaf_selected_bindings_write(
                                        search,
                                        selected_population_indices + chunk_begin,
                                        chunk_asts,
                                        SECANT_SR_DYNAMIC_LEAF_PROJECTION_CONSTANTS,
                                        constant_parameter_capacity,
                                        0u,
                                        lm_best_leaf_masks,
                                        chunk_asts,
                                        lm_best_leaf_words,
                                        chunk_asts * SECANT_SR_APP_LM_PARAMETERS,
                                        SECANT_SR_APP_LM_PARAMETERS,
                                        dynamic_program_storage,
                                        dynamic_program_capacity,
                                        dynamic_asts,
                                        proposal_program_sizes,
                                        constant_ast_capacity,
                                        &required_program_size)
                                    : secant_sr_search_constant_optimizer_proposals_selected_write(
                                        search,
                                        selected_population_indices + chunk_begin,
                                        chunk_asts,
                                        constant_parameter_capacity,
                                        constant_settings,
                                        constant_setting_elements,
                                        constant_parameter_capacity,
                                        dynamic_program_storage,
                                        dynamic_program_capacity,
                                        dynamic_asts,
                                        proposal_program_sizes,
                                        constant_ast_capacity,
                                        &required_program_size);
                                if (result != SECANT_SR_SUCCESS ||
                                    !secant_sr_cubin_evaluator_run(
                                        &staged_evaluator.static_sse, dynamic_asts, chunk_asts, sse)) {
                                    fprintf(stderr, "constant optimizer proposal evaluation failed\n");
                                    result = SECANT_SR_ERROR_SECANT;
                                    break;
                                }
                                compile_seconds += staged_evaluator.static_sse.last_stats.compile_window_seconds;
                                module_load_seconds += staged_evaluator.static_sse.last_stats.module_load_seconds;
                                device_runtime_seconds += staged_evaluator.static_sse.last_stats.runtime_seconds;
                                if (maturity_trace || lm_constant_optimizer) {
                                    const SecantSRIndividual* individuals;
                                    size_t num_individuals;
                                    size_t chunk_idx;

                                    if (secant_sr_search_individuals_get(
                                            search, &individuals, &num_individuals) != SECANT_SR_SUCCESS) {
                                        result = SECANT_SR_ERROR_INVALID_VALUE;
                                    }
                                    for (chunk_idx = 0u; result == SECANT_SR_SUCCESS && chunk_idx < chunk_asts;
                                         ++chunk_idx) {
                                        const size_t population_idx = selected_population_indices[chunk_begin + chunk_idx];

                                        if (population_idx < num_individuals && isfinite(sse[chunk_idx]) &&
                                            sse[chunk_idx] >= 0.0f &&
                                            (double)sse[chunk_idx] < individuals[population_idx].fitness.sse) {
                                            if (lm_constant_optimizer && lm_best_leaf_masks[chunk_idx] != 0u) {
                                                ++lm_column_binding_promoted;
                                            }
                                            if (maturity_trace) {
                                                if (!isfinite(individuals[population_idx].fitness.sse)) {
                                                    ++maturity_constant_nonfinite_recovered;
                                                } else {
                                                    const double gain =
                                                        (individuals[population_idx].fitness.sse - sse[chunk_idx]) /
                                                        target_ssd;

                                                    if (isfinite(gain)) {
                                                        maturity_constant_r2_gain_sum += gain;
                                                        ++maturity_constant_finite_gain_count;
                                                        if (gain > maturity_constant_r2_gain_max) {
                                                            maturity_constant_r2_gain_max = gain;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if (result != SECANT_SR_SUCCESS) {
                                    break;
                                }
                                result = secant_sr_search_selected_proposals_apply(
                                    search,
                                    selected_population_indices + chunk_begin,
                                    dynamic_asts,
                                    proposal_program_sizes,
                                    sse,
                                    NULL,
                                    chunk_asts,
                                    num_rows,
                                    target_ssd,
                                    SECANT_SR_ORIGIN_DYNAMIC_CONSTANT,
                                    &chunk_promoted);
                                constant_promoted += chunk_promoted;
                                evaluator_row_evals += (double)chunk_asts * (double)num_rows *
                                    (lm_constant_optimizer
                                         ? (double)num_constant_settings *
                                               ((double)constant_optimizer_iterations + 1.0) + 1.0
                                         : (double)num_constant_settings *
                                               (double)constant_optimizer_iterations + 1.0);
                                constant_optimizer_row_evals += (double)chunk_asts * (double)num_rows *
                                    (lm_constant_optimizer
                                         ? (double)num_constant_settings *
                                               ((double)constant_optimizer_iterations + 1.0) + 1.0
                                         : (double)num_constant_settings *
                                               (double)constant_optimizer_iterations + 1.0);
                                if (result != SECANT_SR_SUCCESS) {
                                    break;
                                }
                            }
                        }
                        {
                            const double constant_optimizer_seconds =
                                secant_sr_app_seconds_get() - constant_optimizer_begin;

                            if (maturity_backend) {
                                maturity_constant_eligible = num_eligible;
                                maturity_constant_selected = num_selected;
                                maturity_constant_seconds = constant_optimizer_seconds;
                            }
                            printf(
                                "constant_optimizer shape=%s policy=%s eligible=%zu selected=%zu "
                                "budget=%zu interval=%zu random_fraction=%.9g probability=%.9g "
                                "iterations=%zu settings=%zu bindings=%zu starts_per_binding=%zu "
                                "initial_scale=%.9g decay=%.9g settings_per_cta=%zu "
                                "tile_rows=%zu threads=%zu initial_damping=%.9g "
                                "column_binding_winners=%zu column_binding_promoted=%zu "
                                "promoted=%zu seconds=%.9f row_evals_per_second=%.3f\n",
                                lm_constant_optimizer
                                    ? "lm_thread_owned"
                                    : (maturity_backend ? "packed" : "ast_local"),
                                constant_optimize_probability >= 0.0 ? "probability" : "bounded_diverse",
                                num_eligible,
                                num_selected,
                                constant_optimize_budget,
                                constant_optimize_interval,
                                constant_optimize_random_fraction,
                                constant_optimize_probability,
                                constant_optimizer_iterations,
                                num_constant_settings,
                                lm_constant_optimizer ? num_constant_settings / lm_starts_per_binding : 0u,
                                lm_constant_optimizer ? lm_starts_per_binding : 0u,
                                constant_optimizer_scale,
                                constant_optimizer_decay,
                                lm_constant_optimizer ? lm_settings_per_cta : 0u,
                                lm_constant_optimizer ? lm_tile_rows : constant_optimizer_tile_rows,
                                lm_constant_optimizer ? lm_threads : constant_optimizer_tile_rows,
                                lm_constant_optimizer ? lm_initial_damping : 0.0,
                                lm_column_binding_winners,
                                lm_column_binding_promoted,
                                constant_promoted,
                                constant_optimizer_seconds,
                                constant_optimizer_seconds != 0.0
                                    ? constant_optimizer_row_evals / constant_optimizer_seconds
                                    : 0.0);
                        }
                    }
                }
                else if (cubin_evaluator_active) {
                    const SecantAstInstruction* const* asts = NULL;
                    size_t num_asts = 0u;

                    if (dynamic_leaf_backend) {
                        size_t required_program_size;
                        size_t max_dynamic_leaves;

                        result = secant_sr_search_dynamic_leaf_programs_write(
                            search,
                            dynamic_program_storage,
                            dynamic_program_capacity,
                            dynamic_asts,
                            population_size,
                            &required_program_size,
                            &num_asts,
                            &max_dynamic_leaves);
                        if (result == SECANT_SR_SUCCESS &&
                            (required_program_size > dynamic_program_capacity ||
                             max_dynamic_leaves > cubin_evaluator.num_dynamic_leaves)) {
                            result = SECANT_SR_ERROR_BAD_PROGRAM;
                        }
                        if (result == SECANT_SR_SUCCESS) {
                            size_t chunk_begin;

                            for (chunk_begin = 0u; chunk_begin < num_asts; chunk_begin += dynamic_ast_capacity) {
                                const size_t chunk_asts = num_asts - chunk_begin < dynamic_ast_capacity
                                    ? num_asts - chunk_begin
                                    : dynamic_ast_capacity;
                                size_t chunk_idx;

                                if (!secant_sr_cubin_dynamic_leaf_evaluator_run_device(
                                        &cubin_evaluator, dynamic_asts + chunk_begin, chunk_asts) ||
                                    !secant_sr_mse_reducer_best_get(
                                        &mse_reducer,
                                        cubin_evaluator.output,
                                        chunk_asts,
                                        num_leaf_settings,
                                        num_leaf_settings,
                                        num_rows,
                                        target_ssd,
                                        mse_entries,
                                        dynamic_ast_capacity)) {
                                    fprintf(stderr, "dynamic-leaf CUBIN reduction failed\n");
                                    result = SECANT_SR_ERROR_SECANT;
                                    break;
                                }
                                compile_seconds += cubin_evaluator.last_stats.compile_window_seconds;
                                module_load_seconds += cubin_evaluator.last_stats.module_load_seconds;
                                device_runtime_seconds += cubin_evaluator.last_stats.runtime_seconds;
                                for (chunk_idx = 0u; chunk_idx < chunk_asts; ++chunk_idx) {
                                    best_setting_indices[chunk_begin + chunk_idx] = mse_entries[chunk_idx].setting_idx;
                                    sse[chunk_begin + chunk_idx] = mse_entries[chunk_idx].mse * (float)num_rows;
                                }
                            }
                        }
                        if (result == SECANT_SR_SUCCESS) {
                            result = secant_sr_search_dynamic_leaf_best_scores_apply(
                                search,
                                leaf_masks,
                                num_leaf_settings,
                                leaf_words,
                                leaf_word_elements,
                                num_dynamic_leaves,
                                num_leaf_settings,
                                best_setting_indices,
                                sse,
                                num_asts,
                                num_rows,
                                target_ssd);
                            if (result != SECANT_SR_SUCCESS) {
                                fprintf(stderr, "dynamic-leaf score application failed: %s\n",
                                    secant_sr_result_to_string(result));
                            }
                        }
                    } else {
                        result = secant_sr_search_asts_get(search, &asts, &num_asts);
                        if (result == SECANT_SR_SUCCESS &&
                            (!secant_sr_cubin_evaluator_run(&cubin_evaluator, asts, num_asts, sse) ||
                             secant_sr_search_scores_set(search, sse, num_asts, num_rows, target_ssd) !=
                                 SECANT_SR_SUCCESS)) {
                            result = SECANT_SR_ERROR_SECANT;
                        }
                    }
                    if (!dynamic_leaf_backend) {
                        compile_seconds = cubin_evaluator.last_stats.compile_window_seconds;
                        module_load_seconds = cubin_evaluator.last_stats.module_load_seconds;
                        device_runtime_seconds = cubin_evaluator.last_stats.runtime_seconds;
                    }
                    if (dynamic_leaf_backend) {
                        evaluator_row_evals *= (double)num_leaf_settings;
                    }
                }
#endif
                if (result != SECANT_SR_SUCCESS || secant_sr_search_best_get(search, &best) != SECANT_SR_SUCCESS ||
                    secant_sr_program_features_get(
                        best->program,
                        best->program_bytes,
                        secant_sr_app_routines,
                        sizeof(secant_sr_app_routines) / sizeof(secant_sr_app_routines[0]),
                        &best_features) !=
                        SECANT_SR_SUCCESS) {
                    fprintf(stderr, "generation %zu failed: %s\n", generation, secant_sr_result_to_string(result));
                    break;
                }
                if (secant_sr_search_archive_stats_get(search, &archive_stats) != SECANT_SR_SUCCESS) {
                    fprintf(stderr, "generation %zu archive query failed\n", generation);
                    result = SECANT_SR_ERROR_INVALID_VALUE;
                    break;
                }
                evaluation_seconds = secant_sr_app_seconds_get() - evaluation_begin;
                if (checkpoint_trace) {
                    SecantSRAppCheckpoint* checkpoint;
                    double checkpoint_copy_begin;

                    if (checkpoint_count >= generations || best->program_bytes > config.max_program_bytes) {
                        fprintf(stderr, "generation %zu checkpoint capture failed\n", generation);
                        result = SECANT_SR_ERROR_INSUFFICIENT_BUFFER;
                        break;
                    }
                    checkpoint = checkpoints + checkpoint_count;
                    checkpoint->generation = generation;
                    checkpoint->search_elapsed_seconds = secant_sr_app_seconds_get() - app_begin;
                    checkpoint->train_r2 = best->fitness.r2;
                    checkpoint->fingerprint = best->fingerprint;
                    checkpoint->complexity = best->complexity;
                    checkpoint->num_nodes = best->num_nodes;
                    checkpoint->program_bytes = best->program_bytes;
                    checkpoint_copy_begin = secant_sr_app_seconds_get();
                    memcpy(
                        checkpoint_programs + checkpoint_count * config.max_program_bytes,
                        best->program,
                        best->program_bytes);
                    checkpoint->checkpoint_copy_seconds = secant_sr_app_seconds_get() - checkpoint_copy_begin;
                    ++checkpoint_count;
                }
                if (validate_each_generation) {
                    const double validation_begin = secant_sr_app_seconds_get();

#ifdef SECANT_SR_HAS_CUBIN
                    if (staged_evaluator_active) {
                        const double cpu_begin = secant_sr_app_seconds_get();
                        const double cpu_validation_r2 = secant_sr_app_validation_r2(
                            dataset,
                            best,
                            validation_input,
                            validation_target,
                            validation_rows,
                            validation_target_ssd);
                        const double gpu_begin = secant_sr_app_seconds_get();

                        validation_cpu_seconds = gpu_begin - cpu_begin;

                        if (!secant_sr_cubin_materialize_evaluator_run_one(
                                &staged_evaluator.materialize, best->program, validation_prediction)) {
                            fprintf(stderr, "staged materialize validation failed\n");
                            result = SECANT_SR_ERROR_SECANT;
                            validation_r2 = -INFINITY;
                        } else {
                            const double score_begin = secant_sr_app_seconds_get();
                            const double scale = fmax(1.0, fabs(cpu_validation_r2));

                            validation_gpu_seconds = score_begin - gpu_begin;
                            materialize_validation_r2 = secant_sr_app_predictions_r2(
                                validation_prediction,
                                validation_target,
                                validation_rows,
                                validation_target_ssd);
                            validation_score_seconds = secant_sr_app_seconds_get() - score_begin;
                            validation_r2 = isfinite(cpu_validation_r2) ? cpu_validation_r2 : -INFINITY;
                            if (!isfinite(materialize_validation_r2) || !isfinite(cpu_validation_r2) ||
                                fabs(materialize_validation_r2 - cpu_validation_r2) > 1.0e-3 * scale) {
                                materialize_validation_diverged = 1;
                                fprintf(
                                    stderr,
                                    "staged materialize numerical divergence: gpu_r2=%.17g cpu_r2=%.17g\n",
                                    materialize_validation_r2,
                                    cpu_validation_r2);
                            }
                        }
                        validation_gpu_compile_seconds =
                            staged_evaluator.materialize.last_stats.compile_window_seconds;
                        validation_gpu_module_load_seconds =
                            staged_evaluator.materialize.last_stats.module_load_seconds;
                        validation_gpu_device_runtime_seconds = staged_evaluator.materialize.last_stats.runtime_seconds;
                    } else
#endif
                    {
                        const double cpu_begin = secant_sr_app_seconds_get();

                        validation_r2 = secant_sr_app_validation_r2(
                            dataset,
                            best,
                            validation_input,
                            validation_target,
                            validation_rows,
                            validation_target_ssd);
                        validation_cpu_seconds = secant_sr_app_seconds_get() - cpu_begin;
                    }
                    validation_seconds = secant_sr_app_seconds_get() - validation_begin;
                } else {
                    validation_r2 = NAN;
                }
                if (maturity_trace && result == SECANT_SR_SUCCESS) {
                    const SecantSRIndividual* individuals;
                    size_t num_individuals;
                    size_t individual_idx;

                    if (secant_sr_app_best_snapshot_get(
                            search,
                            &maturity_best_r2_after_constant,
                            &maturity_best_origin,
                            &maturity_best_maturity) != SECANT_SR_SUCCESS ||
                        secant_sr_search_individuals_get(search, &individuals, &num_individuals) != SECANT_SR_SUCCESS) {
                        result = SECANT_SR_ERROR_INVALID_VALUE;
                    }
                    for (individual_idx = 0u; result == SECANT_SR_SUCCESS && individual_idx < num_individuals;
                         ++individual_idx) {
                        const SecantSRMaturity maturity = individuals[individual_idx].maturity;

                        if ((size_t)maturity < sizeof(maturity_after) / sizeof(maturity_after[0])) {
                            ++maturity_after[maturity];
                        }
                    }
                    if (result == SECANT_SR_SUCCESS) {
                        printf(
                            "maturity_stage problem=%s seed=%llu generation=%zu population=%zu "
                            "exploratory_before=%zu resolved_before=%zu mixed_refined_before=%zu "
                            "constant_refined_before=%zu full_selected=%zu full_promoted=%zu "
                            "full_finite_gain_count=%zu full_nonfinite_recovered=%zu "
                            "full_promotion_rate=%.9g full_r2_gain_mean=%.9g full_r2_gain_max=%.9g "
                            "full_seconds=%.9f mixed_eligible=%zu mixed_selected=%zu mixed_promoted=%zu "
                            "mixed_finite_gain_count=%zu mixed_nonfinite_recovered=%zu "
                            "mixed_promotion_rate=%.9g mixed_r2_gain_mean=%.9g mixed_r2_gain_max=%.9g "
                            "mixed_seconds=%.9f constant_eligible=%zu constant_selected=%zu constant_promoted=%zu "
                            "constant_finite_gain_count=%zu constant_nonfinite_recovered=%zu "
                            "constant_promotion_rate=%.9g constant_r2_gain_mean=%.9g constant_r2_gain_max=%.9g "
                            "constant_seconds=%.9f static_seconds=%.9f exploratory_after=%zu resolved_after=%zu "
                            "mixed_refined_after=%zu constant_refined_after=%zu leaf_settings=%zu "
                            "leaf_setting_policy=%s leaf_setting_update_seconds=%.9f "
                            "mixed_holes=%zu mixed_probability=%.9g "
                            "constant_optimizer=%s constant_settings=%zu lm_bindings=%zu "
                            "lm_starts_per_binding=%zu lm_column_binding_winners=%zu "
                            "lm_column_binding_promoted=%zu constant_iterations=%zu "
                            "constant_budget=%zu constant_interval=%zu constant_random_fraction=%.9g "
                            "best_r2_after_static=%.9g best_r2_after_full=%.9g best_r2_after_mixed=%.9g "
                            "best_r2_after_constant=%.9g best_sse_after_static=%.9g "
                            "best_sse_after_full=%.9g best_sse_after_mixed=%.9g "
                            "best_sse_after_constant=%.9g best_origin=%d best_maturity=%d "
                            "train_r2=%.9g validation_r2=%.9g complexity=%u depth=%u evaluation_seconds=%.9f "
                            "elapsed_seconds=%.9f\n",
                            dataset->name,
                            (unsigned long long)seed,
                            generation,
                            num_individuals,
                            maturity_before[SECANT_SR_MATURITY_EXPLORATORY],
                            maturity_before[SECANT_SR_MATURITY_RESOLVED],
                            maturity_before[SECANT_SR_MATURITY_MIXED_REFINED],
                            maturity_before[SECANT_SR_MATURITY_CONSTANT_REFINED],
                            maturity_full_selected,
                            maturity_full_promoted,
                            maturity_full_finite_gain_count,
                            maturity_full_nonfinite_recovered,
                            maturity_full_selected != 0u
                                ? (double)maturity_full_promoted / (double)maturity_full_selected
                                : 0.0,
                            maturity_full_finite_gain_count != 0u
                                ? maturity_full_r2_gain_sum / (double)maturity_full_finite_gain_count
                                : 0.0,
                            maturity_full_r2_gain_max,
                            maturity_full_seconds,
                            maturity_mixed_eligible,
                            maturity_mixed_selected,
                            maturity_mixed_promoted,
                            maturity_mixed_finite_gain_count,
                            maturity_mixed_nonfinite_recovered,
                            maturity_mixed_selected != 0u
                                ? (double)maturity_mixed_promoted / (double)maturity_mixed_selected
                                : 0.0,
                            maturity_mixed_finite_gain_count != 0u
                                ? maturity_mixed_r2_gain_sum / (double)maturity_mixed_finite_gain_count
                                : 0.0,
                            maturity_mixed_r2_gain_max,
                            maturity_mixed_seconds,
                            maturity_constant_eligible,
                            maturity_constant_selected,
                            constant_promoted,
                            maturity_constant_finite_gain_count,
                            maturity_constant_nonfinite_recovered,
                            maturity_constant_selected != 0u
                                ? (double)constant_promoted / (double)maturity_constant_selected
                                : 0.0,
                            maturity_constant_finite_gain_count != 0u
                                ? maturity_constant_r2_gain_sum / (double)maturity_constant_finite_gain_count
                                : 0.0,
                            maturity_constant_r2_gain_max,
                            maturity_constant_seconds,
                            maturity_static_seconds,
                            maturity_after[SECANT_SR_MATURITY_EXPLORATORY],
                            maturity_after[SECANT_SR_MATURITY_RESOLVED],
                            maturity_after[SECANT_SR_MATURITY_MIXED_REFINED],
                            maturity_after[SECANT_SR_MATURITY_CONSTANT_REFINED],
                            num_leaf_settings,
                            secant_sr_app_leaf_settings_policy_name(leaf_settings_policy),
                            leaf_setting_update_seconds,
                            mixed_dynamic_leaves,
                            mixed_refine_probability,
                            constant_optimizer_mode,
                            num_constant_settings,
                            lm_constant_optimizer ? num_constant_settings / lm_starts_per_binding : 0u,
                            lm_constant_optimizer ? lm_starts_per_binding : 0u,
                            lm_column_binding_winners,
                            lm_column_binding_promoted,
                            constant_optimizer_iterations,
                            constant_optimize_budget,
                            constant_optimize_interval,
                            constant_optimize_random_fraction,
                            maturity_best_r2_after_static,
                            maturity_best_r2_after_full,
                            maturity_best_r2_after_mixed,
                            maturity_best_r2_after_constant,
                            target_ssd * (1.0 - maturity_best_r2_after_static),
                            target_ssd * (1.0 - maturity_best_r2_after_full),
                            target_ssd * (1.0 - maturity_best_r2_after_mixed),
                            target_ssd * (1.0 - maturity_best_r2_after_constant),
                            maturity_best_origin,
                            maturity_best_maturity,
                            best->fitness.r2,
                            validation_r2,
                            best->complexity,
                            best->depth,
                            evaluation_seconds,
                            secant_sr_app_seconds_get() - app_begin);
                    }
                }
                stop_value = stop_on_validation ? validation_r2 : best->fitness.r2;
                stop_reached = stop_value > stop_r2;
                time_limit_reached = time_limit_seconds > 0.0 &&
                    secant_sr_app_seconds_get() - app_begin >= time_limit_seconds;
                if (!stop_reached && !time_limit_reached) {
                    const double generation_begin = secant_sr_app_seconds_get();

                    if (legacy_staged_backend && generation + 1u == dynamic_generations) {
                        result = secant_sr_search_active_max_nodes_set(search, config.max_nodes);
                        if (result == SECANT_SR_SUCCESS) {
                            result = secant_sr_search_active_max_leaves_set(search, config.max_nodes);
                        }
                        if (result == SECANT_SR_SUCCESS) {
                            result = secant_sr_search_archive_parent_probability_set(search, 0.0);
                        }
                    }
                    if (result == SECANT_SR_SUCCESS) {
                        result = secant_sr_search_generation_advance(search);
                    }
                    generation_seconds = secant_sr_app_seconds_get() - generation_begin;
                    time_limit_reached = time_limit_seconds > 0.0 &&
                        secant_sr_app_seconds_get() - app_begin >= time_limit_seconds;
                }
                printf(
                    "problem=%s generation=%zu train_r2=%.9g validation_r2=%.9g rmse=%.9g score=%.9g "
                    "complexity=%u nodes=%zu depth=%u bytes=%zu unique_columns=%u column_occurrences=%u constants=%u "
                    "unary_ops=%u binary_ops=%u ternary_ops=%u transcendental_ops=%u "
                    "evaluation_seconds=%.9f row_evals_per_second=%.3f "
                    "materialize_validation_r2=%.9g materialize_validation_diverged=%d "
                    "validation_seconds=%.9f validation_cpu_seconds=%.9f validation_gpu_seconds=%.9f "
                    "validation_score_seconds=%.9f validation_gpu_compile_seconds=%.9f "
                    "validation_gpu_module_load_seconds=%.9f validation_gpu_device_runtime_seconds=%.9f "
                    "generation_seconds=%.9f compile_seconds=%.9f module_load_seconds=%.9f "
                    "device_runtime_seconds=%.9f leaf_setting_policy=%s leaf_setting_update_seconds=%.9f "
                    "evaluation_stage=%s dynamic_promoted=%zu constant_promoted=%zu "
                    "constant_setting_robustness=%.9g active_max_nodes=%zu dynamic_leaves=%zu "
                    "mixed_dynamic_leaves=%zu archive_cells=%zu "
                    "archive_occupied_cells=%zu archive_elites=%zu "
                    "validation_mode=%s stop_metric=%s stop_r2=%.9g stop_value=%.9g stop_reached=%d "
                    "time_limit_seconds=%.9g time_limit_reached=%d elapsed_seconds=%.9f\n",
                    dataset->name,
                    generation,
                    best->fitness.r2,
                    validation_r2,
                    best->fitness.rmse,
                    best->fitness.score,
                    best->complexity,
                    best->num_nodes,
                    best->depth,
                    best->program_bytes,
                    best_features.num_unique_static_columns,
                    best_features.num_static_column_occurrences,
                    best_features.num_constants,
                    best_features.num_unary_operations,
                    best_features.num_binary_operations,
                    best_features.num_ternary_operations,
                    best_features.num_transcendental_operations,
                    evaluation_seconds,
                    evaluator_row_evals / evaluation_seconds,
                    materialize_validation_r2,
                    materialize_validation_diverged,
                    validation_seconds,
                    validation_cpu_seconds,
                    validation_gpu_seconds,
                    validation_score_seconds,
                    validation_gpu_compile_seconds,
                    validation_gpu_module_load_seconds,
                    validation_gpu_device_runtime_seconds,
                    generation_seconds,
                    compile_seconds,
                    module_load_seconds,
                    device_runtime_seconds,
                    secant_sr_app_leaf_settings_policy_name(leaf_settings_policy),
                    leaf_setting_update_seconds,
                    evaluation_stage,
                    dynamic_promoted,
                    constant_promoted,
                    best->fitness.constant_setting_robustness,
                    evaluation_max_nodes,
                    num_dynamic_leaves,
                    mixed_dynamic_leaves,
                    archive_stats.num_cells,
                    archive_stats.occupied_cells,
                    archive_stats.num_elites,
                    validation_mode,
                    stop_metric,
                    stop_r2,
                    stop_value,
                    stop_reached,
                    time_limit_seconds,
                    time_limit_reached,
                    secant_sr_app_seconds_get() - app_begin);
                if (stop_reached || time_limit_reached) {
                    status = 0;
                    break;
                }
                if (result != SECANT_SR_SUCCESS) {
                    fprintf(stderr, "generation advance failed: %s\n", secant_sr_result_to_string(result));
                    break;
                }
            }
            if (generation == generations) {
                status = 0;
            }
            if (status == 0) {
                const SecantSRIndividual* best_before = NULL;
                const SecantSRIndividual* best_after = NULL;
                double train_r2_before = -INFINITY;
                double train_r2_after = -INFINITY;
                double validation_r2_before = -INFINITY;
                double validation_r2_after = -INFINITY;
                double final_optimizer_seconds = 0.0;
                size_t num_constants = 0u;
                size_t num_f_calls = 0u;
                int improved = 0;

                if (secant_sr_search_best_get(search, &best_before) != SECANT_SR_SUCCESS) {
                    status = 1;
                } else {
                    train_r2_before = best_before->fitness.r2;
                }
                if (status == 0 && final_cpu_optimize) {
                    const double optimizer_begin = secant_sr_app_seconds_get();

                    result = secant_sr_search_best_constants_optimize_cpu(
                        search,
                        secant_sr_app_routine_programs,
                        sizeof(secant_sr_app_routine_programs) / sizeof(secant_sr_app_routine_programs[0]),
                        input,
                        dataset->num_inputs * num_rows,
                        num_rows,
                        target,
                        num_rows,
                        final_cpu_optimizer_rows < num_rows ? final_cpu_optimizer_rows : num_rows,
                        num_rows,
                        target_ssd,
                        final_cpu_optimizer_iterations,
                        final_cpu_optimizer_restarts,
                        final_cpu_optimizer_f_calls_limit,
                        final_cpu_optimizer_finite_difference_step,
                        &num_constants,
                        &improved,
                        &num_f_calls);
                    final_optimizer_seconds = secant_sr_app_seconds_get() - optimizer_begin;
                    if (result != SECANT_SR_SUCCESS) {
                        fprintf(stderr, "final CPU constant optimizer failed: %s\n", secant_sr_result_to_string(result));
                        status = 1;
                    }
                }
                if (status == 0 && secant_sr_search_best_get(search, &best_after) == SECANT_SR_SUCCESS) {
                    train_r2_after = best_after->fitness.r2;
                    validation_r2_after = secant_sr_app_validation_r2(
                        dataset,
                        best_after,
                        validation_input,
                        validation_target,
                        validation_rows,
                        validation_target_ssd);
                } else if (status == 0) {
                    status = 1;
                }
                printf(
                    "final_cpu_optimizer enabled=%d constants=%zu improved=%d f_calls=%zu seconds=%.9f "
                    "train_r2_before=%.9g train_r2_after=%.9g validation_r2_before=%.9g "
                    "validation_r2_after=%.9g\n",
                    final_cpu_optimize,
                    num_constants,
                    improved,
                    num_f_calls,
                    final_optimizer_seconds,
                    train_r2_before,
                    train_r2_after,
                    validation_r2_before,
                    validation_r2_after);
            }
            if (status == 0 && checkpoint_trace) {
                size_t checkpoint_idx;

                for (checkpoint_idx = 0u; checkpoint_idx < checkpoint_count; ++checkpoint_idx) {
                    const SecantSRAppCheckpoint* checkpoint = checkpoints + checkpoint_idx;
                    const unsigned char* program =
                        checkpoint_programs + checkpoint_idx * config.max_program_bytes;
                    SecantSRIndividual checkpoint_individual;
                    double validation_begin;
                    double checkpoint_validation_r2;
                    double checkpoint_validation_seconds;
                    size_t byte_idx;

                    memset(&checkpoint_individual, 0, sizeof(checkpoint_individual));
                    checkpoint_individual.program = (const SecantAstInstruction*)program;
                    validation_begin = secant_sr_app_seconds_get();
                    checkpoint_validation_r2 = secant_sr_app_validation_r2(
                        dataset,
                        &checkpoint_individual,
                        validation_input,
                        validation_target,
                        validation_rows,
                        validation_target_ssd);
                    checkpoint_validation_seconds = secant_sr_app_seconds_get() - validation_begin;
                    printf(
                        "checkpoint problem=%s seed=%llu generation=%zu search_elapsed_seconds=%.9f "
                        "train_r2=%.9g validation_r2=%.9g validation_seconds=%.9f complexity=%u nodes=%zu "
                        "fingerprint=%llu program_bytes=%zu checkpoint_copy_seconds=%.9f program_hex=",
                        dataset->name,
                        (unsigned long long)seed,
                        checkpoint->generation,
                        checkpoint->search_elapsed_seconds,
                        checkpoint->train_r2,
                        checkpoint_validation_r2,
                        checkpoint_validation_seconds,
                        checkpoint->complexity,
                        checkpoint->num_nodes,
                        (unsigned long long)checkpoint->fingerprint,
                        checkpoint->program_bytes,
                        checkpoint->checkpoint_copy_seconds);
                    for (byte_idx = 0u; byte_idx < checkpoint->program_bytes; ++byte_idx) {
                        printf("%02x", (unsigned int)program[byte_idx]);
                    }
                    printf("\n");
                }
            }
            {
                const SecantSRIndividual* best = NULL;
                size_t expression_size = 0u;

                if (secant_sr_search_best_get(search, &best) == SECANT_SR_SUCCESS &&
                    secant_sr_individual_format(best, NULL, 0u, &expression_size) == SECANT_SR_SUCCESS) {
                    char* expression = malloc(expression_size);

                    if (expression != NULL) {
                        if (secant_sr_individual_format(best, expression, expression_size, &expression_size) ==
                            SECANT_SR_SUCCESS) {
                            printf("problem=%s target_expression=%s\n", dataset->name, dataset->expression);
                            printf("problem=%s best_expression=%s\n", dataset->name, expression);
                        }
                        free(expression);
                    }
                }
            }
        }
    }
#ifdef SECANT_SR_HAS_CUBIN
    {
        const double evaluator_destroy_begin = secant_sr_app_seconds_get();

        if (staged_evaluator_active) {
            secant_sr_cubin_staged_evaluator_destroy(&staged_evaluator);
        }
        if (lm_evaluator_active) {
            secant_sr_cubin_lm_evaluator_destroy(&lm_evaluator);
        }
        if (cubin_evaluator_active) {
            secant_sr_cubin_evaluator_destroy(&cubin_evaluator);
        }
        if (mixed_dynamic_evaluator_active) {
            secant_sr_cubin_evaluator_destroy(&mixed_dynamic_evaluator);
        }
        if (mse_reducer_active) {
            secant_sr_mse_reducer_destroy(&mse_reducer);
        }
        evaluator_destroy_seconds = secant_sr_app_seconds_get() - evaluator_destroy_begin;
    }
#endif
#ifdef SECANT_SR_HAS_CUBIN
    free(mse_entries);
#endif
    free(lm_best_leaf_words);
    free(lm_best_leaf_masks);
    free(checkpoint_programs);
    free(checkpoints);
    free(lm_leaf_words);
    free(lm_leaf_masks);
    free(lm_accepted_sse);
    free(lm_optimized_constants);
    free(lm_initial_constants);
    free(constant_incumbent_sse);
    free(constant_velocities);
    free(constant_scales);
    free(mixed_population_indices);
    free(selected_population_indices);
    free(best_setting_indices);
    free(constant_settings);
    free(proposal_program_sizes);
    free(dynamic_asts);
    free(dynamic_program_storage);
    free(mixed_leaf_words);
    free(mixed_leaf_masks);
    free(leaf_words);
    free(leaf_masks);
    free(sse);
    free(validation_prediction);
    free(validation_target);
    free(validation_input);
    free(target);
    free(input);
    free(storage);
    printf(
        "request_timing total_seconds=%.9f evaluator_destroy_seconds=%.9f\n",
        secant_sr_app_seconds_get() - app_begin,
        evaluator_destroy_seconds);
    return status;
}

int
main(int argc, char** argv) {
    size_t population_size = 8192u;
    size_t generations = 100u;
    size_t num_rows = 257u;
    size_t validation_rows = 4096u;
    size_t num_workers = 24u;
    size_t num_streams = 8u;
    const char* cubin_cache_path = NULL;
    size_t num_leaf_settings = 4096u;
    SecantSRAppLeafSettingsPolicy leaf_settings_policy = SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY;
    size_t num_constant_settings = 8192u;
    size_t dynamic_batch_asts = 65536u;
    double constant_optimize_probability = -1.0;
    size_t constant_optimize_budget = 4096u;
    size_t constant_optimize_interval = 1u;
    double constant_optimize_random_fraction = 0.25;
    const char* constant_sweep_phase = "each";
    const char* constant_optimizer_mode = "legacy";
    size_t constant_optimizer_iterations = 1u;
    double constant_optimizer_scale = 1.0;
    double constant_optimizer_decay = 0.5;
    size_t constant_optimizer_tile_rows = 128u;
    size_t lm_batch_asts = 16u;
    size_t lm_settings_per_cta = 128u;
    size_t lm_starts_per_binding = 4u;
    size_t lm_tile_rows = 128u;
    size_t lm_threads = 128u;
    size_t lm_patch_instructions = 2048u;
    double lm_initial_damping = 1.0e-3;
    size_t num_dynamic_leaves = 8u;
    size_t mixed_dynamic_leaves = 4u;
    double mixed_refine_probability = 0.25;
    size_t dynamic_max_nodes = 30u;
    size_t dynamic_generations = 5u;
    size_t num_kernels = 64u;
    size_t asts_per_kernel = 32u;
    size_t tile_rows = 0u;
    size_t static_tile_rows = 1024u;
    size_t patch_instructions_per_ast = 64u;
    int final_cpu_optimize = 1;
    size_t final_cpu_optimizer_rows = 257u;
    size_t final_cpu_optimizer_iterations = 8u;
    size_t final_cpu_optimizer_restarts = 2u;
    size_t final_cpu_optimizer_f_calls_limit = 10000u;
    double final_cpu_optimizer_finite_difference_step = 1.0e-3;
    const char* stop_metric = "train";
    const char* validation_mode = "each";
    double stop_r2 = 0.9999999;
    double time_limit_seconds = 0.0;
    uint64_t seed = UINT64_C(0x123456789abcdef);
    const char* backend = "cpu";
    const char* problem_name = "nguyen1";
    const char* dataset_binary_path = NULL;
    const char* batch_manifest_path = NULL;
    const char* operator_profile = "broad";
    size_t qd_column_buckets = 1u;
    size_t qd_transcendental_buckets = 1u;
    double archive_parent_probability = 0.0;
    double constant_setting_credit_weight = 0.0;
    double parsimony_coefficient = 0.00005;
    const SecantSRDataset* dataset = NULL;
    SecantSRDataset external_dataset;
    SecantSRAppCudaState cuda_state;
    int status;
    int arg_idx;

    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--backend") == 0 && arg_idx + 1 < argc) {
            backend = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--problem") == 0 && arg_idx + 1 < argc) {
            problem_name = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--dataset-binary") == 0 && arg_idx + 1 < argc) {
            dataset_binary_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--batch-manifest") == 0 && arg_idx + 1 < argc) {
            batch_manifest_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--list-problems") == 0) {
            secant_sr_datasets_print(stdout);
            return 0;
        } else if (strcmp(argv[arg_idx], "--population") == 0 && arg_idx + 1 < argc) {
            population_size = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--generations") == 0 && arg_idx + 1 < argc) {
            generations = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--rows") == 0 && arg_idx + 1 < argc) {
            num_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--validation-rows") == 0 && arg_idx + 1 < argc) {
            validation_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--seed") == 0 && arg_idx + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++arg_idx], NULL, 0);
        } else if (strcmp(argv[arg_idx], "--workers") == 0 && arg_idx + 1 < argc) {
            num_workers = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--streams") == 0 && arg_idx + 1 < argc) {
            num_streams = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--cubin-cache") == 0 && arg_idx + 1 < argc) {
            cubin_cache_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--leaf-settings") == 0 && arg_idx + 1 < argc) {
            num_leaf_settings = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--leaf-setting-policy") == 0 && arg_idx + 1 < argc) {
            if (!secant_sr_app_leaf_settings_policy_parse(argv[++arg_idx], &leaf_settings_policy)) {
                fprintf(stderr, "invalid leaf setting policy: %s\n", argv[arg_idx]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--constant-settings") == 0 && arg_idx + 1 < argc) {
            num_constant_settings = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--dynamic-batch-asts") == 0 && arg_idx + 1 < argc) {
            dynamic_batch_asts = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--constant-optimize-probability") == 0 && arg_idx + 1 < argc) {
            constant_optimize_probability = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--constant-optimize-budget") == 0 && arg_idx + 1 < argc) {
            constant_optimize_budget = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--constant-optimize-interval") == 0 && arg_idx + 1 < argc) {
            constant_optimize_interval = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--constant-optimize-random-fraction") == 0 && arg_idx + 1 < argc) {
            constant_optimize_random_fraction = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--constant-sweep-phase") == 0 && arg_idx + 1 < argc) {
            constant_sweep_phase = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--constant-optimizer") == 0 && arg_idx + 1 < argc) {
            constant_optimizer_mode = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--constant-optimizer-iterations") == 0 && arg_idx + 1 < argc) {
            constant_optimizer_iterations = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--constant-optimizer-scale") == 0 && arg_idx + 1 < argc) {
            constant_optimizer_scale = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--constant-optimizer-decay") == 0 && arg_idx + 1 < argc) {
            constant_optimizer_decay = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--constant-optimizer-tile-rows") == 0 && arg_idx + 1 < argc) {
            constant_optimizer_tile_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-batch-asts") == 0 && arg_idx + 1 < argc) {
            lm_batch_asts = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-settings-per-cta") == 0 && arg_idx + 1 < argc) {
            lm_settings_per_cta = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-starts-per-binding") == 0 && arg_idx + 1 < argc) {
            lm_starts_per_binding = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-tile-rows") == 0 && arg_idx + 1 < argc) {
            lm_tile_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-threads") == 0 && arg_idx + 1 < argc) {
            lm_threads = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-patch-instructions") == 0 && arg_idx + 1 < argc) {
            lm_patch_instructions = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--lm-initial-damping") == 0 && arg_idx + 1 < argc) {
            lm_initial_damping = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--dynamic-leaves") == 0 && arg_idx + 1 < argc) {
            num_dynamic_leaves = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--mixed-dynamic-leaves") == 0 && arg_idx + 1 < argc) {
            mixed_dynamic_leaves = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--mixed-refine-probability") == 0 && arg_idx + 1 < argc) {
            mixed_refine_probability = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--dynamic-max-nodes") == 0 && arg_idx + 1 < argc) {
            dynamic_max_nodes = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--dynamic-generations") == 0 && arg_idx + 1 < argc) {
            dynamic_generations = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--kernels") == 0 && arg_idx + 1 < argc) {
            num_kernels = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--asts-per-kernel") == 0 && arg_idx + 1 < argc) {
            asts_per_kernel = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--tile-rows") == 0 && arg_idx + 1 < argc) {
            tile_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--static-tile-rows") == 0 && arg_idx + 1 < argc) {
            static_tile_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--patch-instructions-per-ast") == 0 && arg_idx + 1 < argc) {
            patch_instructions_per_ast = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--final-cpu-optimize") == 0 && arg_idx + 1 < argc) {
            final_cpu_optimize = atoi(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "--final-cpu-optimizer-rows") == 0 && arg_idx + 1 < argc) {
            final_cpu_optimizer_rows = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--final-cpu-optimizer-iterations") == 0 && arg_idx + 1 < argc) {
            final_cpu_optimizer_iterations = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--final-cpu-optimizer-restarts") == 0 && arg_idx + 1 < argc) {
            final_cpu_optimizer_restarts = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--final-cpu-optimizer-f-calls-limit") == 0 && arg_idx + 1 < argc) {
            final_cpu_optimizer_f_calls_limit = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--final-cpu-optimizer-finite-difference-step") == 0 &&
                   arg_idx + 1 < argc) {
            final_cpu_optimizer_finite_difference_step = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--operator-profile") == 0 && arg_idx + 1 < argc) {
            operator_profile = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--qd-column-buckets") == 0 && arg_idx + 1 < argc) {
            qd_column_buckets = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--qd-transcendental-buckets") == 0 && arg_idx + 1 < argc) {
            qd_transcendental_buckets = (size_t)strtoull(argv[++arg_idx], NULL, 10);
        } else if (strcmp(argv[arg_idx], "--archive-parent-probability") == 0 && arg_idx + 1 < argc) {
            archive_parent_probability = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--constant-setting-credit-weight") == 0 && arg_idx + 1 < argc) {
            constant_setting_credit_weight = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--parsimony-coefficient") == 0 && arg_idx + 1 < argc) {
            parsimony_coefficient = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--stop-r2") == 0 && arg_idx + 1 < argc) {
            stop_r2 = strtod(argv[++arg_idx], NULL);
        } else if (strcmp(argv[arg_idx], "--stop-metric") == 0 && arg_idx + 1 < argc) {
            stop_metric = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--validation-mode") == 0 && arg_idx + 1 < argc) {
            validation_mode = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--time-limit-seconds") == 0 && arg_idx + 1 < argc) {
            time_limit_seconds = strtod(argv[++arg_idx], NULL);
        } else {
            fprintf(stderr,
                "usage: %s [--backend cpu|cubin|cubin-dynamic-leaf|cubin-staged|cubin-maturity] [--problem NAME] "
                "[--dataset-binary PATH] [--batch-manifest PATH] [--list-problems] "
                "[--population N] "
                "[--generations N] [--rows N] [--validation-rows N] [--seed N] [--workers N] [--streams N] "
                "[--cubin-cache PATH] "
                "[--leaf-settings N] "
                "[--leaf-setting-policy legacy|legacy-rotating|virtual-bank-fixed|virtual-bank] "
                "[--constant-settings N] [--dynamic-batch-asts N] "
                "[--constant-optimize-budget N] [--constant-optimize-interval N] "
                "[--constant-optimize-random-fraction X] [--constant-optimize-probability X] "
                "[--constant-sweep-phase each|early|final|both] "
                "[--constant-optimizer legacy|lm] "
                "[--constant-optimizer-iterations N] [--constant-optimizer-scale X] "
                "[--constant-optimizer-decay X] "
                "[--constant-optimizer-tile-rows 128|256] "
                "[--lm-batch-asts N] [--lm-settings-per-cta N] [--lm-starts-per-binding N] [--lm-tile-rows N] "
                "[--lm-threads N] [--lm-patch-instructions N] [--lm-initial-damping X] "
                "[--dynamic-leaves N] [--mixed-dynamic-leaves N] [--mixed-refine-probability X] "
                "[--dynamic-max-nodes N] "
                "[--dynamic-generations N] "
                "[--kernels N] "
                "[--asts-per-kernel N] [--tile-rows N] [--static-tile-rows N] "
                "[--patch-instructions-per-ast N] [--final-cpu-optimize 0|1] "
                "[--final-cpu-optimizer-iterations N] [--final-cpu-optimizer-rows N] "
                "[--final-cpu-optimizer-restarts N] [--final-cpu-optimizer-f-calls-limit N] "
                "[--qd-column-buckets N] [--qd-transcendental-buckets N] "
                "[--archive-parent-probability X] "
                "[--constant-setting-credit-weight X] [--parsimony-coefficient X] "
                "[--final-cpu-optimizer-finite-difference-step X] "
                "[--operator-profile broad|scientific|trig|algebraic] "
                "[--validation-mode each|final] [--stop-metric train|validation] [--stop-r2 X] "
                "[--time-limit-seconds X]\n",
                argv[0]);
            return 1;
        }
    }
    memset(&external_dataset, 0, sizeof(external_dataset));
    if (batch_manifest_path == NULL) {
        size_t dataset_num_inputs;

        dataset = secant_sr_dataset_find(problem_name);
        if (dataset_binary_path != NULL) {
            if (!secant_sr_dataset_binary_info(
                    dataset_binary_path,
                    &dataset_num_inputs,
                    &num_rows,
                    &validation_rows)) {
                fprintf(stderr, "invalid dataset binary header: %s\n", dataset_binary_path);
                return 1;
            }
            if (dataset == NULL) {
                external_dataset.name = problem_name;
                external_dataset.expression = "external tabular target";
                external_dataset.num_inputs = dataset_num_inputs;
                dataset = &external_dataset;
            } else if (dataset_num_inputs != dataset->num_inputs) {
                fprintf(stderr, "dataset binary input count does not match problem: %s\n", dataset_binary_path);
                return 1;
            }
        } else if (dataset == NULL) {
            fprintf(stderr, "unknown generated problem: %s\n", problem_name);
            secant_sr_datasets_print(stderr);
            return 1;
        }
    }
    if (tile_rows == 0u) {
        tile_rows = strcmp(backend, "cubin-dynamic-leaf") == 0 || strcmp(backend, "cubin-staged") == 0 ||
                strcmp(backend, "cubin-maturity") == 0
            ? 64u
            : 1024u;
    }
    if ((batch_manifest_path != NULL && dataset_binary_path != NULL) ||
        (strcmp(operator_profile, "broad") != 0 && strcmp(operator_profile, "scientific") != 0 &&
         strcmp(operator_profile, "trig") != 0 && strcmp(operator_profile, "algebraic") != 0) ||
        population_size == 0u || generations == 0u || num_rows < 2u || validation_rows < 2u ||
        num_leaf_settings == 0u || dynamic_batch_asts == 0u ||
        num_dynamic_leaves == 0u || mixed_dynamic_leaves == 0u || mixed_dynamic_leaves > num_dynamic_leaves ||
        !isfinite(mixed_refine_probability) || mixed_refine_probability < 0.0 || mixed_refine_probability > 1.0 ||
        num_kernels == 0u || asts_per_kernel == 0u ||
        static_tile_rows == 0u || patch_instructions_per_ast == 0u ||
        (constant_optimize_probability < 0.0 && constant_optimize_probability != -1.0) ||
        constant_optimize_probability > 1.0 ||
        !isfinite(constant_optimize_probability) || constant_optimize_interval == 0u ||
        constant_optimize_random_fraction < 0.0 || constant_optimize_random_fraction > 1.0 ||
        !isfinite(constant_optimize_random_fraction) || (final_cpu_optimize != 0 && final_cpu_optimize != 1) ||
        archive_parent_probability < 0.0 || archive_parent_probability > 1.0 ||
        dynamic_max_nodes == 0u || dynamic_max_nodes > 72u || constant_setting_credit_weight < 0.0 ||
        !isfinite(constant_setting_credit_weight) ||
        (strcmp(constant_sweep_phase, "each") != 0 && strcmp(constant_sweep_phase, "early") != 0 &&
         strcmp(constant_sweep_phase, "final") != 0 && strcmp(constant_sweep_phase, "both") != 0) ||
        (strcmp(constant_optimizer_mode, "legacy") != 0 && strcmp(constant_optimizer_mode, "lm") != 0) ||
        (strcmp(constant_optimizer_mode, "lm") == 0 && strcmp(backend, "cubin-maturity") != 0) ||
        constant_optimizer_iterations == 0u || constant_optimizer_iterations > INT_MAX ||
        constant_optimizer_scale < 0.0 ||
        (strcmp(backend, "cubin-maturity") == 0 && constant_optimizer_scale == 0.0) ||
        !isfinite(constant_optimizer_scale) ||
        constant_optimizer_decay <= 0.0 || constant_optimizer_decay > 1.0 ||
        !isfinite(constant_optimizer_decay) ||
        (constant_optimizer_tile_rows != 128u && constant_optimizer_tile_rows != 256u) ||
        lm_batch_asts == 0u || lm_settings_per_cta == 0u || lm_starts_per_binding == 0u || lm_tile_rows == 0u ||
        lm_threads == 0u || lm_threads > 1024u || lm_patch_instructions == 0u ||
        (strcmp(constant_optimizer_mode, "lm") == 0 && num_constant_settings != 0u &&
         (lm_settings_per_cta > num_constant_settings ||
          num_constant_settings % lm_starts_per_binding != 0u)) ||
        lm_initial_damping <= 0.0 || !isfinite(lm_initial_damping) ||
        parsimony_coefficient < 0.0 || !isfinite(parsimony_coefficient) ||
        (strcmp(validation_mode, "each") != 0 && strcmp(validation_mode, "final") != 0) ||
        (strcmp(stop_metric, "train") != 0 && strcmp(stop_metric, "validation") != 0) ||
        (strcmp(validation_mode, "final") == 0 && strcmp(stop_metric, "validation") == 0) ||
        time_limit_seconds < 0.0 || !isfinite(time_limit_seconds) ||
        final_cpu_optimizer_rows == 0u || final_cpu_optimizer_iterations == 0u ||
        final_cpu_optimizer_f_calls_limit == 0u || final_cpu_optimizer_finite_difference_step <= 0.0 ||
        num_dynamic_leaves > SECANT_AST_MAX_DYNAMIC_LEAVES) {
        fprintf(stderr,
            "population, generations, settings, dynamic leaves, kernels, ASTs per kernel, and patch instructions "
            "per AST must be nonzero; "
            "row counts must be at least two and dynamic leaves must not exceed %u\n",
            SECANT_AST_MAX_DYNAMIC_LEAVES);
        return 1;
    }
    if (!secant_sr_app_cuda_state_create(backend, cubin_cache_path, &cuda_state)) {
        fprintf(stderr, "CUDA session creation failed; CUDA_MODULE_LOADING must be EAGER\n");
        return 1;
    }
    printf(
        "process_setup context_create_seconds=%.9f cache_open_seconds=%.9f\n",
        cuda_state.context_create_seconds,
        cuda_state.cache_open_seconds);
    if (batch_manifest_path != NULL) {
        FILE* manifest = fopen(batch_manifest_path, "rb");
        SecantSRAppCudaState request_cuda_state = cuda_state;
        char* line = NULL;
        size_t line_capacity = 0u;
        size_t request_idx = 0u;

        if (manifest == NULL) {
            fprintf(stderr, "batch manifest open failed: %s\n", batch_manifest_path);
            secant_sr_app_cuda_state_destroy(&cuda_state);
            return 1;
        }
        request_cuda_state.cache_open_seconds = 0.0;
        status = 0;
        while (getline(&line, &line_capacity, manifest) >= 0) {
            char* request_problem;
            char* request_dataset_path;
            uint64_t request_seed;
            size_t request_rows;
            size_t request_validation_rows;
            size_t request_num_inputs = 0u;
            const SecantSRDataset* request_dataset;
            SecantSRDataset request_external_dataset;
            double request_begin;

            if (!secant_sr_app_batch_line_parse(
                    line,
                    &request_problem,
                    &request_seed,
                    &request_dataset_path,
                    &request_rows,
                    &request_validation_rows)) {
                fprintf(stderr, "invalid batch manifest line %zu\n", request_idx + 1u);
                status = 1;
                break;
            }
            memset(&request_external_dataset, 0, sizeof(request_external_dataset));
            request_dataset = secant_sr_dataset_find(request_problem);
            if (request_dataset_path != NULL) {
                if (!secant_sr_dataset_binary_info(
                        request_dataset_path,
                        &request_num_inputs,
                        &request_rows,
                        &request_validation_rows)) {
                    fprintf(stderr, "invalid dataset binary header: %s\n", request_dataset_path);
                    status = 1;
                    break;
                }
                if (request_dataset == NULL) {
                    request_external_dataset.name = request_problem;
                    request_external_dataset.expression = "external tabular target";
                    request_external_dataset.num_inputs = request_num_inputs;
                    request_dataset = &request_external_dataset;
                } else if (request_num_inputs != request_dataset->num_inputs) {
                    fprintf(stderr, "dataset binary input count does not match problem: %s\n", request_dataset_path);
                    status = 1;
                    break;
                }
            } else if (request_dataset == NULL) {
                fprintf(stderr, "unknown generated problem: %s\n", request_problem);
                status = 1;
                break;
            }
            printf("request_begin problem=%s seed=%llu\n", request_problem, (unsigned long long)request_seed);
            request_begin = secant_sr_app_seconds_get();
            status = secant_sr_app_run(
                request_dataset,
                request_dataset_path,
                backend,
                population_size,
                generations,
                request_rows,
                request_validation_rows,
                request_seed,
                num_workers,
                num_streams,
                &request_cuda_state,
                num_leaf_settings,
                leaf_settings_policy,
                num_constant_settings,
                dynamic_batch_asts,
                constant_optimize_probability,
                constant_optimize_budget,
                constant_optimize_interval,
                constant_optimize_random_fraction,
                constant_sweep_phase,
                constant_optimizer_mode,
                constant_optimizer_iterations,
                constant_optimizer_scale,
                constant_optimizer_decay,
                constant_optimizer_tile_rows,
                lm_batch_asts,
                lm_settings_per_cta,
                lm_starts_per_binding,
                lm_tile_rows,
                lm_threads,
                lm_patch_instructions,
                lm_initial_damping,
                num_dynamic_leaves,
                mixed_dynamic_leaves,
                mixed_refine_probability,
                dynamic_generations,
                num_kernels,
                asts_per_kernel,
                tile_rows,
                static_tile_rows,
                patch_instructions_per_ast,
                final_cpu_optimize,
                final_cpu_optimizer_rows,
                final_cpu_optimizer_iterations,
                final_cpu_optimizer_restarts,
                final_cpu_optimizer_f_calls_limit,
                final_cpu_optimizer_finite_difference_step,
                operator_profile,
                qd_column_buckets,
                qd_transcendental_buckets,
                archive_parent_probability,
                dynamic_max_nodes,
                constant_setting_credit_weight,
                parsimony_coefficient,
                validation_mode,
                stop_metric,
                stop_r2,
                time_limit_seconds);
            printf(
                "request_complete problem=%s seed=%llu elapsed_seconds=%.9f status=%d\n",
                request_problem,
                (unsigned long long)request_seed,
                secant_sr_app_seconds_get() - request_begin,
                status);
            fflush(stdout);
            request_idx += 1u;
            if (status != 0) {
                break;
            }
        }
        free(line);
        fclose(manifest);
        secant_sr_app_cuda_state_destroy(&cuda_state);
        return status;
    }
    status = secant_sr_app_run(
        dataset,
        dataset_binary_path,
        backend,
        population_size,
        generations,
        num_rows,
        validation_rows,
        seed,
        num_workers,
        num_streams,
        &cuda_state,
        num_leaf_settings,
        leaf_settings_policy,
        num_constant_settings,
        dynamic_batch_asts,
        constant_optimize_probability,
        constant_optimize_budget,
        constant_optimize_interval,
        constant_optimize_random_fraction,
        constant_sweep_phase,
        constant_optimizer_mode,
        constant_optimizer_iterations,
        constant_optimizer_scale,
        constant_optimizer_decay,
        constant_optimizer_tile_rows,
        lm_batch_asts,
        lm_settings_per_cta,
        lm_starts_per_binding,
        lm_tile_rows,
        lm_threads,
        lm_patch_instructions,
        lm_initial_damping,
        num_dynamic_leaves,
        mixed_dynamic_leaves,
        mixed_refine_probability,
        dynamic_generations,
        num_kernels,
        asts_per_kernel,
        tile_rows,
        static_tile_rows,
        patch_instructions_per_ast,
        final_cpu_optimize,
        final_cpu_optimizer_rows,
        final_cpu_optimizer_iterations,
        final_cpu_optimizer_restarts,
        final_cpu_optimizer_f_calls_limit,
        final_cpu_optimizer_finite_difference_step,
        operator_profile,
        qd_column_buckets,
        qd_transcendental_buckets,
        archive_parent_probability,
        dynamic_max_nodes,
        constant_setting_credit_weight,
        parsimony_coefficient,
        validation_mode,
        stop_metric,
        stop_r2,
        time_limit_seconds);
    secant_sr_app_cuda_state_destroy(&cuda_state);
    return status;
}
