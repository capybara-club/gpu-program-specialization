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
#ifndef SECANT_SYSTEM_ID_H
#define SECANT_SYSTEM_ID_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SSID_API __declspec(dllexport)
#else
#define SSID_API __attribute__((visibility("default")))
#endif

typedef enum ssid_status {
    SSID_OK = 0,
    SSID_INVALID_ARGUMENT = 1,
    SSID_OUT_OF_MEMORY = 2,
    SSID_OUT_OF_RANGE = 3,
    SSID_UNSUPPORTED = 4,
    SSID_QUEUE_FULL = 5,
    SSID_CUDA_ERROR = 6,
    SSID_INTERNAL_ERROR = 7
} ssid_status;

typedef struct ssid_ast_desc {
    uint32_t byte_offset;
    uint32_t byte_count;
    uint16_t site_index;
    uint16_t reserved;
} ssid_ast_desc;

typedef struct ssid_genome_desc {
    uint32_t first_ast;
    uint16_t ast_count;
    uint16_t reserved;
} ssid_genome_desc;

typedef struct ssid_genome_batch {
    const ssid_genome_desc *genomes;
    uint32_t genome_count;
    const ssid_ast_desc *asts;
    uint32_t ast_count;
    const uint8_t *program_bytes;
    size_t program_byte_count;
} ssid_genome_batch;

typedef struct ssid_packed_kernel_desc {
    const char *name;
    uint32_t genome_base;
    uint32_t genome_capacity;
    uint32_t genomes_per_cta;
    uint32_t register_count;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t function_file_offset;
    uint64_t entry_file_offset;
    uint64_t arena_start_file_offset;
    uint32_t arena_instruction_count;
    uint32_t incoming_wait_mask;
    uint64_t continuation_file_offset;
    const uint64_t *dispatch_file_offsets;
    const uint64_t *dispatch_instruction_words;
    uint32_t dispatch_instruction_count;
    const uint64_t *cleanup_file_offsets;
    uint32_t cleanup_count;
    const uint64_t *target_table_file_offsets;
    uint32_t target_table_count;
    const uint64_t *register_count_file_offsets;
    uint32_t register_count_offset_count;
    const uint64_t *register_count_header_file_offsets;
    uint32_t register_count_header_offset_count;
    const uint8_t *input_registers;
    const uint8_t *output_registers;
    const uint8_t *available_registers;
    uint32_t available_register_count;
} ssid_packed_kernel_desc;

typedef struct ssid_template_desc {
    uint32_t abi_version;
    uint32_t architecture;
    const uint8_t *cubin;
    size_t cubin_byte_count;
    const ssid_packed_kernel_desc *kernels;
    uint32_t kernel_count;
    const uint32_t *site_input_offsets;
    const uint32_t *site_input_counts;
    uint32_t site_count;
    uint32_t settings_mode;
} ssid_template_desc;

typedef struct ssid_specialization_stats {
    uint32_t genome_count;
    uint32_t kernel_count;
    uint32_t populated_kernel_count;
    uint32_t maximum_register_count;
    uint64_t sass_instruction_count;
} ssid_specialization_stats;

typedef struct ssid_ad_tape_stats {
    uint32_t node_count;
    uint32_t maximum_stack_depth;
    uint32_t scratch_floats_per_thread;
    uint32_t reserved;
    size_t tape_byte_count;
} ssid_ad_tape_stats;

typedef struct ssid_raw_benchmark_result {
    uint32_t thread_count;
    uint32_t reserved;
    uint64_t module_count;
    uint64_t genome_count;
    uint64_t ast_count;
    uint64_t cubin_bytes_copied;
    double wall_seconds;
    double modules_per_second;
    double genomes_per_second;
    double asts_per_second;
    double cubin_gib_per_second;
} ssid_raw_benchmark_result;

typedef struct ssid_pipeline_config {
    uint32_t specialization_threads;
    uint32_t queue_capacity;
    uint32_t maximum_loaded_modules;
    int32_t device_ordinal;
    uint32_t enable_cuda;
    uint32_t execution_streams;
} ssid_pipeline_config;

#define SSID_OUTPUT_FULL_MSE 0u
#define SSID_OUTPUT_GENOME_WINNERS 1u

typedef struct ssid_fedbatch_launch {
    uint64_t settings_device;
    uint64_t settings_leading_dimension;
    uint64_t bindings_device;
    uint64_t bindings_leading_dimension;
    uint64_t num_settings;
    uint32_t num_genomes;
    uint64_t reference_device;
    uint32_t steps_per_observation;
    uint64_t mse_device;
    uint32_t threads_per_block;
    uint32_t shared_memory_bytes;
    uint32_t output_mode;
    uint32_t reduction_threads;
    uint64_t cta_score_device;
    uint64_t cta_setting_device;
    uint64_t winner_score_device;
    uint64_t winner_setting_device;
} ssid_fedbatch_launch;

/* One thread-owned, dense fed-batch trajectory-LM module launch. */
typedef struct ssid_trajectory_lm_launch {
    uint64_t starts_device;
    uint64_t starts_leading_dimension;
    uint64_t bindings_device;
    uint64_t bindings_leading_dimension;
    uint64_t num_settings;
    uint32_t starts_per_setting;
    uint64_t reference_device;
    uint32_t steps_per_observation;
    uint32_t max_lm_iterations;
    uint32_t max_damping_attempts;
    float initial_damping;
    uint64_t constants_device;
    uint64_t constants_leading_dimension;
    uint64_t mse_device;
    uint64_t iterations_device;
    uint64_t accepted_steps_device;
    uint32_t threads_per_block;
} ssid_trajectory_lm_launch;

typedef struct ssid_ticket_result {
    int32_t status;
    uint32_t worker_index;
    uint32_t launched_kernel_count;
    uint32_t maximum_register_count;
    uint64_t sass_instruction_count;
    double queue_wait_seconds;
    double specialization_seconds;
    double module_load_seconds;
    double function_lookup_seconds;
    double launch_seconds;
    double total_seconds;
} ssid_ticket_result;

typedef struct ssid_template ssid_template;
typedef struct ssid_pipeline ssid_pipeline;
typedef struct ssid_ticket ssid_ticket;
typedef struct ssid_gp ssid_gp;

#define SSID_TEMPLATE_ABI_VERSION 3u
#define SSID_SETTINGS_MATERIALIZED 0u
#define SSID_SETTINGS_HASHED_INCUMBENT 1u

SSID_API const char *ssid_status_string(int status);
SSID_API const char *ssid_last_error(void);

SSID_API int ssid_template_create(const ssid_template_desc *desc, ssid_template **result);
SSID_API void ssid_template_destroy(ssid_template *template_value);
SSID_API size_t ssid_template_cubin_size(const ssid_template *template_value);

SSID_API int ssid_specialize_module(const ssid_template *template_value, const ssid_genome_batch *batch, uint8_t *output_cubin, size_t output_byte_count, ssid_specialization_stats *stats);
SSID_API int ssid_benchmark_specialization_raw(const ssid_template *template_value, const ssid_genome_batch *batch, uint32_t thread_count, uint64_t module_count, ssid_raw_benchmark_result *result);

/* Build the versioned, inspectable postorder reverse-AD tape for one AST. */
SSID_API int ssid_ad_tape_build(const uint8_t *program, size_t program_byte_count, uint32_t input_count, uint8_t *output_tape, size_t output_capacity, size_t *required_byte_count, ssid_ad_tape_stats *stats);

SSID_API int ssid_pipeline_create(const ssid_template *template_value, const ssid_pipeline_config *config, ssid_pipeline **result);
/* Create a persistent CUDA loader for same-sized, already-specialized CUBINs. */
SSID_API int ssid_module_pipeline_create(size_t cubin_byte_count, const ssid_pipeline_config *config, ssid_pipeline **result);
SSID_API int ssid_pipeline_submit(ssid_pipeline *pipeline, const ssid_genome_batch *batch, const ssid_fedbatch_launch *launch, ssid_ticket **result);
/* The CUBIN bytes are borrowed until the returned ticket completes. */
SSID_API int ssid_pipeline_submit_trajectory_lm(ssid_pipeline *pipeline, const uint8_t *cubin, size_t cubin_byte_count, const ssid_trajectory_lm_launch *launch, ssid_ticket **result);
SSID_API int ssid_pipeline_wait_idle(ssid_pipeline *pipeline);
SSID_API void ssid_pipeline_destroy(ssid_pipeline *pipeline);

SSID_API int ssid_ticket_poll(ssid_ticket *ticket, int *complete, ssid_ticket_result *result);
SSID_API int ssid_ticket_wait(ssid_ticket *ticket, ssid_ticket_result *result);
SSID_API const char *ssid_ticket_error(const ssid_ticket *ticket);
SSID_API void ssid_ticket_destroy(ssid_ticket *ticket);

SSID_API int ssid_device_alloc(ssid_pipeline *pipeline, size_t byte_count, uint64_t *device_pointer);
SSID_API int ssid_device_free(ssid_pipeline *pipeline, uint64_t device_pointer);
SSID_API int ssid_device_upload(ssid_pipeline *pipeline, uint64_t destination, const void *source, size_t byte_count);
SSID_API int ssid_device_download(ssid_pipeline *pipeline, void *destination, uint64_t source, size_t byte_count);

/* Minimal system-genome GP. All postorder populations and variation live in C99. */
typedef struct ssid_gp_config {
    uint32_t population_size;
    uint32_t settings_per_genome;
    uint32_t state_count;
    uint32_t constant_count;
    uint32_t max_nodes_per_ast;
    uint32_t max_program_bytes_per_ast;
    uint32_t initial_max_nodes;
    uint32_t max_depth;
    uint32_t tournament_size;
    uint32_t elite_count;
    uint64_t seed;
    float subtree_crossover_probability;
    float whole_site_crossover_probability;
    float subtree_mutation_probability;
    float point_mutation_probability;
    float binding_keep_probability;
    float constant_mutation_scale;
    float parsimony_coefficient;
    float minimum_valid_mse;
} ssid_gp_config;

typedef struct ssid_gp_grammar {
    const uint8_t *unary_operations;
    uint32_t unary_operation_count;
    const uint8_t *binary_operations;
    uint32_t binary_operation_count;
} ssid_gp_grammar;

typedef struct ssid_gp_run_config {
    uint32_t generations;
    uint32_t steps_per_observation;
    uint32_t threads_per_block;
    uint32_t reduction_threads;
    uint32_t shared_memory_bytes;
    uint32_t output_mode;
    uint64_t reference_device;
} ssid_gp_run_config;

typedef struct ssid_gp_run_stats {
    uint32_t generations;
    uint32_t batches_per_generation;
    uint64_t evaluated_genomes;
    uint64_t evaluated_configurations;
    double wall_seconds;
    double winner_materialization_seconds;
    double upload_seconds;
    double pipeline_seconds;
    double download_seconds;
    double evolution_seconds;
    double genomes_per_second;
    double configurations_per_second;
    float best_mse;
    float best_objective;
    uint32_t best_generation;
    uint32_t best_complexity;
} ssid_gp_run_stats;

typedef struct ssid_gp_checkpoint {
    double elapsed_seconds;
    float generation_best_mse;
    float generation_best_objective;
    float best_mse;
    float best_objective;
    uint32_t generation;
    uint32_t best_generation;
    uint32_t best_complexity;
} ssid_gp_checkpoint;

/* Caller-owned, fixed-stride snapshots of the best-so-far genome. */
typedef struct ssid_gp_trace {
    ssid_gp_checkpoint *checkpoints;
    uint8_t *program_bytes;
    uint32_t *ast_byte_counts;
    float *constants;
    uint32_t *bindings;
    size_t program_byte_count;
    size_t ast_byte_count_count;
    size_t constant_count;
    size_t binding_count;
    uint32_t checkpoint_capacity;
    uint32_t checkpoint_count;
    uint32_t checkpoint_stride;
} ssid_gp_trace;

typedef struct ssid_gp_best {
    float mse;
    float objective;
    uint32_t generation;
    uint32_t complexity;
    ssid_genome_batch genome;
    const float *constants;
    const uint32_t *bindings;
} ssid_gp_best;

SSID_API int ssid_gp_create(const ssid_template *template_value, const ssid_gp_config *config, const ssid_gp_grammar *grammar, ssid_gp **result);
SSID_API void ssid_gp_destroy(ssid_gp *gp);
SSID_API uint32_t ssid_gp_generation_get(const ssid_gp *gp);
SSID_API uint32_t ssid_gp_batch_count(const ssid_gp *gp);
SSID_API int ssid_gp_batch_get(const ssid_gp *gp, uint32_t batch_index, ssid_genome_batch *batch, uint32_t *population_offset);
SSID_API int ssid_gp_settings_generate(ssid_gp *gp, float *settings, size_t settings_count, uint32_t *bindings, size_t bindings_count);
SSID_API int ssid_gp_winners_apply(ssid_gp *gp, const float *winner_mse, const uint32_t *winner_settings, const float *settings, size_t settings_count, const uint32_t *bindings, size_t bindings_count);
SSID_API int ssid_gp_generation_advance(ssid_gp *gp);
SSID_API int ssid_gp_best_get(const ssid_gp *gp, ssid_gp_best *best);
/* Borrow one scored member of the current population for an external local search. */
SSID_API int ssid_gp_candidate_get(const ssid_gp *gp, uint32_t genome_index, ssid_gp_best *candidate);
/* Replace only the constants/bindings of the same genome when its score improves. */
SSID_API int ssid_gp_candidate_improve(ssid_gp *gp, uint32_t genome_index, float mse, const float *constants, size_t constant_count, const uint32_t *bindings, size_t binding_count, int *accepted);
SSID_API int ssid_gp_run(ssid_gp *gp, ssid_pipeline *pipeline, const ssid_gp_run_config *config, ssid_gp_run_stats *stats);
SSID_API int ssid_gp_run_traced(ssid_gp *gp, ssid_pipeline *pipeline, const ssid_gp_run_config *config, ssid_gp_run_stats *stats, ssid_gp_trace *trace);

#ifdef __cplusplus
}
#endif

#endif
