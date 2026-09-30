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
#ifndef O_ODEZZA_INTERNAL_H
#define O_ODEZZA_INTERNAL_H

#include "odezza.h"
#include "o_sass.h"

#ifdef ODEZZA_DEBUG
#include <assert.h>
#include <stdio.h>
#define O_RETURN_IF_ERROR(expression)                                                                        \
    do {                                                                                                     \
        OdezzaResult o_result_ = (expression);                                                               \
        if (o_result_ != ODEZZA_SUCCESS) {                                                                   \
            fprintf(stderr, "ODEZZA_ERROR: result %d at %s:%d\n", (int)o_result_, __FILE__, __LINE__);       \
            assert(o_result_ == ODEZZA_SUCCESS);                                                             \
            return o_result_;                                                                                \
        }                                                                                                    \
    } while (0)
#else
#define O_RETURN_IF_ERROR(expression)                                                                        \
    do {                                                                                                     \
        OdezzaResult o_result_ = (expression);                                                               \
        if (o_result_ != ODEZZA_SUCCESS)                                                                     \
            return o_result_;                                                                                \
    } while (0)
#endif

typedef struct OdezzaAstAnalysis {
    size_t instruction_count;
    size_t maximum_stack_depth;
    uint32_t required_toggle_bit_count;
    int contains_toggle;
} OdezzaAstAnalysis;

typedef struct OdezzaScoringPrespecialization {
    unsigned char template_id[32];
    unsigned char cubin_id[32];
    size_t state_count;
    size_t state_capacity;
    uint32_t constant_count;
    uint32_t constant_capacity;
    uint64_t fixed_state_mask[2];
    size_t fixed_rhs_count;
    size_t shared_instruction_count;
    uint32_t high_water_register;
    uint32_t register_count;
} OdezzaScoringPrespecialization;

typedef struct OdezzaScoringSpecializationReport {
    size_t system_count;
    size_t specialized_rhs_count;
    size_t body_instruction_count;
    size_t maximum_body_instruction_count;
    uint32_t high_water_register;
    uint32_t register_count;
} OdezzaScoringSpecializationReport;

typedef struct OdezzaScoringPipelineStats {
    uint64_t submitted_ticket_count;
    uint64_t completed_ticket_count;
    uint64_t cubin_reset_count;
    uint64_t cubin_reset_byte_count;
    uint64_t specialization_failure_count;
    uint64_t cuda_failure_count;
    uint64_t worker_slot_wait_count;
    uint32_t peak_pending_ticket_count;
    uint32_t peak_ready_slot_count;
    uint32_t peak_inflight_module_count;
    double cubin_reset_seconds;
    double specialization_seconds;
    double module_load_seconds;
    double function_lookup_seconds;
    double launch_seconds;
    double completion_wait_seconds;
    double module_unload_seconds;
    double cuda_resource_create_seconds;
} OdezzaScoringPipelineStats;

typedef OSassInstruction OdezzaScoringInstruction;

typedef struct OdezzaScoringCubinInspection {
    size_t cubin_size;
    uint32_t architecture;
    uint32_t register_count;
    size_t state_capacity;
    uint32_t constant_capacity;
    size_t input_count;
    size_t output_count;
    size_t system_capacity;
    uint32_t function_symbol_index;
    size_t function_file_offset;
    size_t function_byte_size;
    size_t scaffold_start_file_offset;
    size_t scaffold_end_file_offset;
    size_t shared_start_file_offset;
    size_t shared_instruction_count;
    size_t arena_start_file_offset;
    size_t arena_instruction_count;
    size_t requested_arena_instruction_count;
    size_t system_patch_capacity;
    int final_fallthrough_branch_elided;
    size_t arena_end_file_offset;
    size_t continuation_file_offset;
    uint32_t incoming_wait_mask;
    uint32_t predicate_register;
    uint8_t permutation_register;
    OdezzaScoringInstruction toggle_test_instruction;
    size_t register_count_file_offset_count;
    const size_t *register_count_file_offsets;
    size_t register_count_header_file_offset_count;
    const size_t *register_count_header_file_offsets;
    size_t dispatch_instruction_count;
    const size_t *dispatch_file_offsets;
    const OdezzaScoringInstruction *dispatch_instructions;
    const uint8_t *input_registers;
    const uint8_t *output_registers;
    const uint8_t *final_output_registers;
    const size_t *output_materialization_file_offsets;
    size_t available_register_count;
    const uint8_t *available_registers;
    size_t cleanup_file_offset_count;
    const size_t *cleanup_file_offsets;
    const size_t *target_table_file_offsets;
    const uint32_t *original_target_values;
    size_t template_id_file_offset;
    unsigned char template_id[32];
} OdezzaScoringCubinInspection;

struct OdezzaScoringTemplate {
    OdezzaScoringTemplateInfo info;
    char *source;
    size_t source_size, cubin_size;
    unsigned char *cubin;
    void *arena;
    const OdezzaScoringCubinInspection *inspection;
    char error[1024];
    int valid;
    double nvrtc_seconds;
};

typedef struct OdezzaNvrtcCompilation OdezzaNvrtcCompilation;

typedef struct OdezzaScoringPipelineTuning {
    uint32_t queue_capacity;
    uint32_t maximum_loaded_modules;
    uint32_t execution_stream_count;
    uint32_t completion_poll_interval_ns;
} OdezzaScoringPipelineTuning;

OdezzaResult odezza_generate_scoring_cuda(size_t, uint32_t, uint32_t, uint32_t, uint32_t, char *, size_t,
                                          size_t *);
OdezzaResult odezza_inspect_scoring_cubin(const void *, size_t, void *, size_t, size_t *,
                                          const OdezzaScoringCubinInspection **);
OdezzaResult odezza_validate_ast_program(OdezzaAstProgram, size_t, uint32_t, uint32_t, int,
                                         OdezzaAstAnalysis *);
OdezzaResult odezza_scoring_specialization_workspace_size(const OdezzaScoringCubinInspection *, size_t *);
OdezzaResult odezza_prespecialize_scoring_cubin(void *, size_t, const OdezzaScoringCubinInspection *, size_t,
                                                uint32_t, const OdezzaScoringRhs *, size_t, void *, size_t,
                                                OdezzaScoringPrespecialization *);
OdezzaResult odezza_specialize_scoring_cubin_systems(void *, size_t, const OdezzaScoringCubinInspection *,
                                                     const OdezzaScoringPrespecialization *, uint32_t,
                                                     const OdezzaScoringSystem *, size_t, void *, size_t,
                                                     OdezzaScoringSpecializationReport *);

OdezzaResult odezza_nvrtc_compilation_create(const char *, const char *, const char *const *, size_t,
                                             OdezzaNvrtcCompilation **);
OdezzaResult odezza_nvrtc_compilation_destroy(OdezzaNvrtcCompilation *);
OdezzaResult odezza_nvrtc_compilation_result(const OdezzaNvrtcCompilation *, OdezzaResult *);
OdezzaResult o_nvrtc_compilation_seconds(const OdezzaNvrtcCompilation *, double *);
OdezzaResult odezza_nvrtc_compilation_native_result(const OdezzaNvrtcCompilation *, int *);
OdezzaResult odezza_nvrtc_compilation_native_result_string(const OdezzaNvrtcCompilation *, const char **);
OdezzaResult odezza_nvrtc_compilation_cubin_size(const OdezzaNvrtcCompilation *, size_t *);
OdezzaResult odezza_nvrtc_compilation_write_cubin(const OdezzaNvrtcCompilation *, void *, size_t);
OdezzaResult odezza_nvrtc_compilation_log_size(const OdezzaNvrtcCompilation *, size_t *);
OdezzaResult odezza_nvrtc_compilation_write_log(const OdezzaNvrtcCompilation *, char *, size_t);

OdezzaResult o_scoring_pipeline_generated_source(const OdezzaScoringPipeline *, const char **, size_t *);
OdezzaResult o_scoring_pipeline_cubin_inspection(const OdezzaScoringPipeline *,
                                                 const OdezzaScoringCubinInspection **);
OdezzaResult o_scoring_pipeline_prespecialized_cubin(const OdezzaScoringPipeline *, const void **, size_t *);
OdezzaResult o_scoring_pipeline_prespecialization(const OdezzaScoringPipeline *,
                                                  OdezzaScoringPrespecialization *);
OdezzaResult o_scoring_pipeline_stats(const OdezzaScoringPipeline *, OdezzaScoringPipelineStats *);
OdezzaResult o_scoring_pipeline_create_with_tuning(const OdezzaScoringPipelineCreateInfo *,
                                                   const OdezzaScoringPipelineTuning *,
                                                   OdezzaScoringPipeline **);

typedef struct OElf {
    const unsigned char *data;
    size_t size;
    uint32_t flags;
    size_t section_table_offset;
    uint16_t section_count;
    uint16_t names_section_index;
    size_t names_offset;
    size_t names_size;
} OElf;

typedef struct OElfSection {
    uint32_t name_offset;
    uint32_t kind;
    uint64_t flags;
    uint64_t address;
    size_t offset;
    size_t size;
    uint32_t link;
    uint32_t info;
    uint64_t alignment;
    size_t entry_size;
} OElfSection;

typedef struct OFunction {
    uint32_t symbol_index;
    size_t file_offset;
    size_t size;
} OFunction;

typedef struct ODataSymbol {
    size_t file_offset;
    size_t size;
} ODataSymbol;

OdezzaResult o_size_add(size_t left, size_t right, size_t *value_ret);
OdezzaResult o_size_multiply(size_t left, size_t right, size_t *value_ret);
OdezzaResult o_check_range(size_t data_size, size_t offset, size_t range_size);
OdezzaResult o_read_u16(const unsigned char *data, size_t size, size_t offset, uint16_t *value_ret);
OdezzaResult o_read_u32(const unsigned char *data, size_t size, size_t offset, uint32_t *value_ret);
OdezzaResult o_read_u64(const unsigned char *data, size_t size, size_t offset, uint64_t *value_ret);
OdezzaResult o_u64_to_size(uint64_t value, size_t *value_ret);
OdezzaResult o_elf_section(const OElf *elf, size_t index, OElfSection *section_ret);
OdezzaResult o_elf_init(const void *data, size_t size, OElf *elf_ret);
OdezzaResult o_elf_string(const OElf *elf, size_t table_offset, size_t table_size, uint32_t string_offset,
                          const unsigned char **string_ret, size_t *length_ret);
OdezzaResult o_section_name_matches(const OElf *elf, const OElfSection *section, const char *name,
                                    int *matches_ret);
OdezzaResult o_section_name_contains(const OElf *elf, const OElfSection *section, const char *needle,
                                     int *contains_ret);
OdezzaResult o_section_name_starts_with(const OElf *elf, const OElfSection *section, const char *prefix,
                                        int *matches_ret);
OdezzaResult o_symbol_name_matches(const OElf *elf, const OElfSection *strings, uint32_t name_offset,
                                   const char *name, int *matches_ret);
OdezzaResult o_find_function(const OElf *elf, const char *name, OFunction *function_ret);
OdezzaResult o_find_data_symbol(const OElf *elf, const char *name, size_t expected_size,
                                ODataSymbol *symbol_ret);
OdezzaResult o_sass_compile_rhs_set(const OdezzaScoringCubinInspection *inspection, size_t state_capacity,
                                    uint32_t constant_capacity, size_t active_state_count,
                                    uint32_t active_constant_count, uint32_t active_toggle_count,
                                    const OdezzaScoringRhs *rhs, size_t rhs_count,
                                    uint32_t incoming_wait_mask, OdezzaScoringInstruction *instructions,
                                    size_t instruction_capacity, size_t *instruction_count_ret,
                                    uint32_t *high_water_register_ret);

/* Shared native LM representation; no service/search types cross this boundary. */
#define O_LM_MAX_STATES 16u
#define O_LM_MAX_INPUTS 24u
#define O_LM_MAX_SITES 256u
#define O_LM_MAX_NODES 8192u
#define O_LM_MAX_OUTPUTS 24u

typedef struct OLmNode {
    uint32_t value, child[4];
    uint16_t depth;
    uint8_t opcode, arity;
} OLmNode;
typedef struct OLmExpressions {
    OLmNode nodes[O_LM_MAX_NODES];
    uint32_t count;
    uint32_t roots[O_LM_MAX_STATES * (O_LM_MAX_INPUTS + 1u)];
    OdezzaResult result;
} OLmExpressions;
typedef struct OLmSite {
    OSassInstruction predicate_save, predicate_restore;
    uint32_t preserves_predicate;
    size_t scaffold, entry, patch, continuation;
    uint32_t input_count, output_count, output_indices[O_LM_MAX_OUTPUTS];
    size_t input_offsets[O_LM_MAX_INPUTS];
    uint8_t inputs[O_LM_MAX_INPUTS], sources[O_LM_MAX_INPUTS], waits[O_LM_MAX_INPUTS];
    uint8_t outputs[O_LM_MAX_OUTPUTS], available[255];
    size_t available_count;
    uint32_t predicate;
    uint8_t permutation;
    OSassInstruction toggle;
} OLmSite;
typedef struct OLmInspection {
    OdezzaLmShape shape;
    size_t cubin_size;
    uint32_t architecture, registers, site_count;
    size_t register_offsets[32], header_offsets[32];
    size_t register_offset_count, header_offset_count;
    unsigned char cubin_id[32];
    OLmSite sites[O_LM_MAX_SITES];
} OLmInspection;

OdezzaResult o_lm_validate_shape(const OdezzaLmShape *);
OdezzaResult o_lm_site_layout(const OdezzaLmShape *, OLmInspection *);
OdezzaResult o_lm_generate_cuda(const OdezzaLmShape *, char *, size_t, size_t *);
OdezzaResult o_lm_inspect(const void *, size_t, const OdezzaLmShape *, OLmInspection *);
OdezzaResult o_lm_expressions(const OdezzaAstProgram *, const OdezzaLmShape *, uint32_t, OLmExpressions *);
OdezzaResult o_lm_specialize(const void *, size_t, const OLmInspection *, const OdezzaAstProgram *, uint32_t,
                             OLmExpressions *, OSassInstruction *, void *, uint32_t *);
OdezzaResult o_sass_compile_group(const OdezzaScoringCubinInspection *, const OLmExpressions *,
                                  const uint32_t *, size_t, uint32_t, OSassInstruction *, size_t, size_t *,
                                  uint32_t *);

#endif
