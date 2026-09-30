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
#include "o_odezza_internal.h"

#include "o_sha256.h"
#include "o_specialize_scoring_cubin.h"

#include <stdint.h>
#include <string.h>

#define O_INSTRUCTION_BYTES O_SASS_INSTRUCTION_BYTES
#define O_REGISTER_RZ O_SASS_RZ
#define O_REGISTER_PAD 2u
#define O_BARRIER_SLOTS O_SASS_BARRIER_SLOTS
#define O_WAIT_BARRIER_MASK O_SASS_WAIT_MASK
#define O_MAX_AST_STACK_DEPTH 128u
#define O_OPCODE_LOP3_IMM 0x7812u
#define O_OPCODE_BRA 0x7947u

#define O_NEG_ONE_BITS 0xbf800000u
#define O_SIN_COS_SCALE_BITS 0x3e22f983u
#define O_LN_TWO_BITS 0x3f317218u
#define O_LOG2_E_BITS 0x3fb8aa3bu

#define O_TRY(expression) \
    do { \
        OdezzaResult o_result_ = (expression); \
        if (o_result_ != ODEZZA_SUCCESS) return o_result_; \
    } while (0)

static void o_mask_set(uint64_t mask[2], size_t index) { mask[index / 64u] |= UINT64_C(1) << (index % 64u); }

static int o_mask_get(const uint64_t mask[2], size_t index) { return (int)((mask[index / 64u] >> (index % 64u)) & 1u); }

static OdezzaResult o_cubin_id(const void *cubin, size_t cubin_size, unsigned char digest[32]) {
    OSha256 sha;
    O_TRY(o_sha256_init(&sha));
    O_TRY(o_sha256_update(&sha, cubin, cubin_size));
    return o_sha256_final(&sha, digest);
}

static int o_patch_span_valid(size_t size, size_t offset, size_t count, size_t width) {
    return offset <= size && count <= (size - offset) / width;
}

/* Preflight every destination before the commit phase. This keeps malformed
 * patch metadata from causing an out-of-bounds write or a partial commit.
 * Array pointers and their declared lengths remain the internal caller's contract. */
static OdezzaResult o_validate_patch_destinations(const OdezzaScoringCubinInspection *inspection, size_t cubin_size) {
    size_t index;
    if (!o_sass_architecture_supported(inspection->architecture)) return ODEZZA_ERROR_UNSUPPORTED;
    if (inspection->system_capacity > 65535u || inspection->state_capacity == 0u || inspection->state_capacity > 128u || inspection->constant_capacity > 255u ||
        inspection->output_count != inspection->state_capacity || inspection->input_count != inspection->state_capacity + inspection->constant_capacity ||
        inspection->input_count > 255u || inspection->register_count > 255u || inspection->available_register_count > 255u ||
        inspection->incoming_wait_mask > O_SASS_WAIT_MASK || inspection->predicate_register >= 7u ||
        (inspection->available_register_count != 0u && inspection->available_registers == NULL) ||
        (inspection->cleanup_file_offset_count != 0u && inspection->cleanup_file_offsets == NULL) ||
        (inspection->register_count_file_offset_count != 0u && inspection->register_count_file_offsets == NULL) ||
        (inspection->register_count_header_file_offset_count != 0u && inspection->register_count_header_file_offsets == NULL))
        return ODEZZA_ERROR_FORMAT;
    if (!o_patch_span_valid(cubin_size, inspection->shared_start_file_offset, inspection->shared_instruction_count, O_INSTRUCTION_BYTES) ||
        !o_patch_span_valid(cubin_size, inspection->arena_start_file_offset, inspection->arena_instruction_count, O_INSTRUCTION_BYTES) ||
        !o_patch_span_valid(cubin_size, inspection->continuation_file_offset, 1u, O_INSTRUCTION_BYTES) ||
        inspection->arena_end_file_offset > cubin_size || inspection->arena_end_file_offset < inspection->arena_start_file_offset)
        return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < inspection->output_count; ++index) {
        if (inspection->output_registers[index] >= O_SASS_RZ || inspection->final_output_registers[index] >= O_SASS_RZ ||
            !o_patch_span_valid(cubin_size, inspection->output_materialization_file_offsets[index], 1u, O_INSTRUCTION_BYTES))
            return ODEZZA_ERROR_FORMAT;
    }
    for (index = 0u; index < inspection->cleanup_file_offset_count; ++index)
        if (!o_patch_span_valid(cubin_size, inspection->cleanup_file_offsets[index], 1u, O_INSTRUCTION_BYTES)) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < inspection->dispatch_instruction_count; ++index)
        if (!o_patch_span_valid(cubin_size, inspection->dispatch_file_offsets[index], 1u, O_INSTRUCTION_BYTES)) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; inspection->system_capacity > 1u && index < inspection->system_capacity; ++index)
        if (!o_patch_span_valid(cubin_size, inspection->target_table_file_offsets[index], 1u, sizeof(uint32_t))) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < inspection->register_count_file_offset_count; ++index)
        if (!o_patch_span_valid(cubin_size, inspection->register_count_file_offsets[index], 1u, sizeof(uint32_t))) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < inspection->register_count_header_file_offset_count; ++index)
        if (!o_patch_span_valid(cubin_size, inspection->register_count_header_file_offsets[index], 1u, 1u)) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < inspection->available_register_count; ++index) {
        size_t other;
        for (other = 0u; other < index; ++other)
            if (inspection->available_registers[index] == inspection->available_registers[other]) return ODEZZA_ERROR_FORMAT;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_validate_inspection(
    const void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection
) {
    if (cubin == NULL || inspection == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (inspection->cubin_size != cubin_size || inspection->template_id_file_offset > cubin_size || 32u > cubin_size - inspection->template_id_file_offset)
        return ODEZZA_ERROR_FORMAT;
    if (memcmp((const unsigned char *)cubin + inspection->template_id_file_offset, inspection->template_id, 32u) != 0) return ODEZZA_ERROR_FORMAT;
    if (inspection->input_registers == NULL || inspection->output_registers == NULL || inspection->final_output_registers == NULL ||
        inspection->output_materialization_file_offsets == NULL || inspection->system_capacity == 0u)
        return ODEZZA_ERROR_FORMAT;
    if (inspection->permutation_register == O_REGISTER_RZ || (inspection->toggle_test_instruction.word0 & 0xffffu) != O_OPCODE_LOP3_IMM)
        return ODEZZA_ERROR_FORMAT;
    if (inspection->system_capacity == 1u) {
        if (inspection->dispatch_instruction_count != 0u || inspection->dispatch_file_offsets != NULL || inspection->dispatch_instructions != NULL ||
            inspection->target_table_file_offsets != NULL || inspection->original_target_values != NULL)
            return ODEZZA_ERROR_FORMAT;
    } else if (inspection->dispatch_instruction_count == 0u || inspection->dispatch_file_offsets == NULL || inspection->dispatch_instructions == NULL ||
               inspection->target_table_file_offsets == NULL || inspection->original_target_values == NULL) {
        return ODEZZA_ERROR_FORMAT;
    }
    return o_validate_patch_destinations(inspection, cubin_size);
}

static OdezzaResult o_validate_layout(
    const OdezzaScoringCubinInspection *inspection,
    size_t state_count,
    uint32_t constant_count,
    const OdezzaScoringRhs *fixed_rhs,
    size_t fixed_rhs_count,
    uint64_t fixed_mask[2]
) {
    size_t index;
    if (inspection == NULL || state_count == 0u || state_count > inspection->state_capacity || constant_count > inspection->constant_capacity ||
        fixed_rhs_count > state_count || (fixed_rhs_count != 0u && fixed_rhs == NULL))
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (inspection->output_count != inspection->state_capacity || inspection->input_count != inspection->state_capacity + inspection->constant_capacity)
        return ODEZZA_ERROR_STATE_LAYOUT;
    memset(fixed_mask, 0, 2u * sizeof(uint64_t));
    for (index = 0u; index < fixed_rhs_count; ++index) {
        OdezzaAstAnalysis analysis;
        size_t output = fixed_rhs[index].state_index;
        if (output >= state_count || o_mask_get(fixed_mask, output)) return ODEZZA_ERROR_STATE_LAYOUT;
        O_TRY(odezza_validate_ast_program(fixed_rhs[index].program, state_count, constant_count, 0u, 0, &analysis));
        if (analysis.contains_toggle) return ODEZZA_ERROR_AST;
        o_mask_set(fixed_mask, output);
    }
    return ODEZZA_SUCCESS;
}

static void o_write_u32(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8u);
    bytes[2] = (unsigned char)(value >> 16u);
    bytes[3] = (unsigned char)(value >> 24u);
}

static OdezzaResult o_write_encoded_instruction(unsigned char *cubin, size_t cubin_size, size_t offset, OSassEncoding encoding) {
    OSassInstruction instruction;
    O_TRY(o_sass_instruction(encoding, &instruction));
    return o_sass_store(cubin, cubin_size, offset, instruction);
}

static OdezzaResult o_final_register_count(uint32_t original, uint32_t high_water, uint32_t *count_ret) {
    if (count_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *count_ret = original;
    if (high_water > original) {
        if (high_water > O_REGISTER_RZ - O_REGISTER_PAD) return ODEZZA_ERROR_REGISTER_PRESSURE;
        *count_ret = high_water + O_REGISTER_PAD;
    }
    return ODEZZA_SUCCESS;
}

static void o_patch_register_count(
    unsigned char *cubin,
    const OdezzaScoringCubinInspection *inspection,
    uint32_t register_count
) {
    size_t index;
    for (index = 0u; index < inspection->register_count_file_offset_count; ++index) {
        o_write_u32(cubin + inspection->register_count_file_offsets[index], register_count);
    }
    for (index = 0u; index < inspection->register_count_header_file_offset_count; ++index) {
        cubin[inspection->register_count_header_file_offsets[index]] = (unsigned char)register_count;
    }
}

OdezzaResult odezza_scoring_specialization_workspace_size(
    const OdezzaScoringCubinInspection *inspection,
    size_t *workspace_size_ret
) {
    size_t count;
    if (inspection == NULL || workspace_size_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    count =
        inspection->shared_instruction_count > inspection->arena_instruction_count ? inspection->shared_instruction_count : inspection->arena_instruction_count;
    if (count > SIZE_MAX / O_INSTRUCTION_BYTES) return ODEZZA_ERROR_OVERFLOW;
    *workspace_size_ret = count * O_INSTRUCTION_BYTES;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_prespecialize_scoring_cubin(
    void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    size_t state_count,
    uint32_t constant_count,
    const OdezzaScoringRhs *fixed_rhs,
    size_t fixed_rhs_count,
    void *workspace,
    size_t workspace_size,
    OdezzaScoringPrespecialization *prespecialization_ret
) {
    static const uint8_t zero_program_bytes[] = {
        ODEZZA_AST_LITERAL_F32,
        0u,
        0u,
        0u,
        0u,
        ODEZZA_AST_RETURN_F32
    };
    OdezzaScoringInstruction *generated = (OdezzaScoringInstruction *)workspace;
    OdezzaScoringRhs compiled_rhs[256];
    OdezzaScoringPrespecialization result;
    size_t required_workspace;
    size_t instruction_count;
    uint32_t high_water;
    uint32_t register_count;
    size_t branch_offset;
    size_t dispatch_start;
    size_t remaining;
    OdezzaScoringInstruction branch;
    size_t index;
    size_t compiled_rhs_count = 0u;
    unsigned char *bytes = (unsigned char *)cubin;

    if (prespecialization_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_validate_inspection(cubin, cubin_size, inspection));
    O_TRY(odezza_scoring_specialization_workspace_size(inspection, &required_workspace));
    if (workspace == NULL || ((uintptr_t)workspace & 7u) != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (workspace_size < required_workspace) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    memset(&result, 0, sizeof(result));
    O_TRY(o_validate_layout(inspection, state_count, constant_count, fixed_rhs, fixed_rhs_count, result.fixed_state_mask));
    for (index = 0u; index < fixed_rhs_count; ++index) compiled_rhs[compiled_rhs_count++] = fixed_rhs[index];
    for (index = state_count; index < inspection->state_capacity; ++index) {
        compiled_rhs[compiled_rhs_count].state_index = (uint8_t)index;
        compiled_rhs[compiled_rhs_count].program.bytes = zero_program_bytes;
        compiled_rhs[compiled_rhs_count].program.byte_count = sizeof(zero_program_bytes);
        ++compiled_rhs_count;
    }
    O_TRY(o_sass_compile_rhs_set(
        inspection,
        inspection->state_capacity,
        inspection->constant_capacity,
        state_count,
        constant_count,
        0u,
        compiled_rhs,
        compiled_rhs_count,
        inspection->incoming_wait_mask,
        generated,
        inspection->shared_instruction_count,
        &instruction_count,
        &high_water
    ));
    dispatch_start = inspection->system_capacity == 1u ? inspection->arena_start_file_offset : inspection->dispatch_file_offsets[0];
    if (instruction_count > SIZE_MAX / O_INSTRUCTION_BYTES) return ODEZZA_ERROR_OVERFLOW;
    branch_offset = inspection->shared_start_file_offset + instruction_count * O_INSTRUCTION_BYTES;
    if (branch_offset >= dispatch_start || (dispatch_start - branch_offset) % O_INSTRUCTION_BYTES != 0u) return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
    remaining = (dispatch_start - branch_offset) / O_INSTRUCTION_BYTES;
    O_TRY(o_sass_instruction(o_sass_branch(remaining, inspection->architecture), &branch));
    if (instruction_count == 0u) branch.word1 |= (uint64_t)(inspection->incoming_wait_mask & O_WAIT_BARRIER_MASK) << 52u;
    if (instruction_count == inspection->shared_instruction_count) return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
    generated[instruction_count++] = branch;
    O_TRY(o_final_register_count(inspection->register_count, high_water, &register_count));

    for (index = 0u; index < inspection->shared_instruction_count; ++index) {
        O_TRY(o_write_encoded_instruction(bytes, cubin_size, inspection->shared_start_file_offset + index * O_INSTRUCTION_BYTES, o_sass_nop(0u)));
    }
    for (index = 0u; index < inspection->arena_instruction_count; ++index) {
        O_TRY(o_write_encoded_instruction(bytes, cubin_size, inspection->arena_start_file_offset + index * O_INSTRUCTION_BYTES, o_sass_nop(0u)));
    }
    for (index = 0u; index < inspection->cleanup_file_offset_count; ++index) {
        O_TRY(o_write_encoded_instruction(bytes, cubin_size, inspection->cleanup_file_offsets[index], o_sass_nop(0u)));
    }
    for (index = 0u; index < inspection->output_count; ++index) {
        O_TRY(o_write_encoded_instruction(bytes, cubin_size, inspection->output_materialization_file_offsets[index], o_sass_mov(inspection->final_output_registers[index], inspection->output_registers[index], 0u)));
    }
    for (index = 0u; index < inspection->dispatch_instruction_count; ++index) {
        O_TRY(o_sass_store(bytes, cubin_size, inspection->dispatch_file_offsets[index], inspection->dispatch_instructions[index]));
    }
    for (index = 0u; index < instruction_count; ++index) {
        O_TRY(o_sass_store(bytes, cubin_size, inspection->shared_start_file_offset + index * O_INSTRUCTION_BYTES, generated[index]));
    }
    if (inspection->system_capacity > 1u) {
        for (index = 0u; index < inspection->system_capacity; ++index) {
            o_write_u32(bytes + inspection->target_table_file_offsets[index], inspection->original_target_values[index]);
        }
    }
    o_patch_register_count(bytes, inspection, register_count);

    memcpy(result.template_id, inspection->template_id, sizeof(result.template_id));
    result.state_count = state_count;
    result.state_capacity = inspection->state_capacity;
    result.constant_count = constant_count;
    result.constant_capacity = inspection->constant_capacity;
    result.fixed_rhs_count = fixed_rhs_count;
    result.shared_instruction_count = instruction_count;
    result.high_water_register = high_water;
    result.register_count = register_count;
    O_TRY(o_cubin_id(cubin, cubin_size, result.cubin_id));
    *prespecialization_ret = result;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_validate_candidate_system(
    const OdezzaScoringPrespecialization *prespecialization,
    uint32_t active_toggle_count,
    const OdezzaScoringSystem *system
) {
    uint64_t seen[2] = {0u, 0u};
    size_t required = prespecialization->state_count - prespecialization->fixed_rhs_count;
    size_t index;
    if (system == NULL || system->rhs_count != required || (required != 0u && system->rhs == NULL)) return ODEZZA_ERROR_STATE_LAYOUT;
    for (index = 0u; index < system->rhs_count; ++index) {
        size_t state_index = system->rhs[index].state_index;
        if (state_index >= prespecialization->state_count || o_mask_get(prespecialization->fixed_state_mask, state_index) || o_mask_get(seen, state_index))
            return ODEZZA_ERROR_STATE_LAYOUT;
        O_TRY(odezza_validate_ast_program(system->rhs[index].program, prespecialization->state_count, prespecialization->constant_count,
                                          active_toggle_count, 1, NULL));
        o_mask_set(seen, state_index);
    }
    for (index = 0u; index < prespecialization->state_count; ++index) {
        if (!o_mask_get(prespecialization->fixed_state_mask, index) && !o_mask_get(seen, index)) return ODEZZA_ERROR_STATE_LAYOUT;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_specialize_scoring_cubin_systems_impl(
    void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    uint32_t active_toggle_count,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    void *workspace,
    size_t workspace_size,
    OdezzaScoringSpecializationReport *report_ret,
    int fresh_prespecialized_copy
) {
    OdezzaScoringInstruction *generated = (OdezzaScoringInstruction *)workspace;
    OdezzaScoringSpecializationReport report;
    unsigned char current_cubin_id[32];
    size_t required_workspace;
    size_t total_instruction_count = 0u;
    uint32_t high_water;
    uint32_t register_count;
    size_t system_index;
    unsigned char *bytes = (unsigned char *)cubin;

    if (inspection == NULL || prespecialization == NULL || systems == NULL || system_count == 0u || system_count > inspection->system_capacity ||
        system_count > 65535u || active_toggle_count > ODEZZA_MAX_TOGGLE_BITS)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (!fresh_prespecialized_copy) {
        O_TRY(o_validate_inspection(cubin, cubin_size, inspection));
    } else if (cubin == NULL || inspection->cubin_size != cubin_size) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (memcmp(prespecialization->template_id, inspection->template_id, 32u) != 0) return ODEZZA_ERROR_FORMAT;
    if (prespecialization->state_capacity != inspection->state_capacity ||
        prespecialization->constant_capacity != inspection->constant_capacity || inspection->output_count != inspection->state_capacity ||
        inspection->input_count != inspection->state_capacity + inspection->constant_capacity)
        return ODEZZA_ERROR_STATE_LAYOUT;
    if (!fresh_prespecialized_copy) {
        O_TRY(o_cubin_id(cubin, cubin_size, current_cubin_id));
        if (memcmp(current_cubin_id, prespecialization->cubin_id, 32u) != 0) {
            return ODEZZA_ERROR_FORMAT;
        }
    }
    O_TRY(odezza_scoring_specialization_workspace_size(inspection, &required_workspace));
    if (workspace == NULL || ((uintptr_t)workspace & 7u) != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (workspace_size < required_workspace) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    memset(&report, 0, sizeof(report));
    high_water = prespecialization->high_water_register;

    for (system_index = 0u; system_index < system_count; ++system_index) {
        size_t body_instruction_count;
        uint32_t body_high_water;
        size_t body_start;
        size_t branch_offset;
        size_t remaining;
        OdezzaScoringInstruction branch;
        O_TRY(o_validate_candidate_system(prespecialization, active_toggle_count, &systems[system_index]));
        O_TRY(o_sass_compile_rhs_set(
            inspection,
            inspection->state_capacity,
            inspection->constant_capacity,
            prespecialization->state_count,
            prespecialization->constant_count,
            active_toggle_count,
            systems[system_index].rhs,
            systems[system_index].rhs_count,
            0u,
            generated + total_instruction_count,
            inspection->arena_instruction_count - total_instruction_count,
            &body_instruction_count,
            &body_high_water
        ));
        if (body_instruction_count + 1u > inspection->system_patch_capacity) return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
        body_start = inspection->arena_start_file_offset + total_instruction_count * O_INSTRUCTION_BYTES;
        branch_offset = body_start + body_instruction_count * O_INSTRUCTION_BYTES;
        if (branch_offset >= inspection->arena_end_file_offset || inspection->continuation_file_offset <= branch_offset ||
            (inspection->continuation_file_offset - branch_offset) % O_INSTRUCTION_BYTES != 0u)
            return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
        remaining = (inspection->continuation_file_offset - branch_offset) / O_INSTRUCTION_BYTES;
        O_TRY(o_sass_instruction(o_sass_branch(remaining, inspection->architecture), &branch));
        if (total_instruction_count + body_instruction_count >= inspection->arena_instruction_count) return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
        generated[total_instruction_count + body_instruction_count] = branch;
        ++body_instruction_count;
        if (body_start < inspection->function_file_offset || body_start - inspection->function_file_offset > UINT32_MAX) return ODEZZA_ERROR_OVERFLOW;
        total_instruction_count += body_instruction_count;
        report.specialized_rhs_count += systems[system_index].rhs_count;
        if (report.maximum_body_instruction_count < body_instruction_count) report.maximum_body_instruction_count = body_instruction_count;
        if (high_water < body_high_water) high_water = body_high_water;
    }
    if (!fresh_prespecialized_copy) {
        size_t branch_count = 0u;
        size_t instruction_index;
        for (instruction_index = 0u; instruction_index < total_instruction_count; ++instruction_index) {
            if ((generated[instruction_index].word0 & 0xffffu) == O_OPCODE_BRA) ++branch_count;
        }
        if (branch_count != system_count) return ODEZZA_ERROR_FORMAT;
    }
    O_TRY(o_final_register_count(inspection->register_count, high_water, &register_count));

    if (!fresh_prespecialized_copy) {
        for (system_index = 0u; system_index < inspection->arena_instruction_count; ++system_index) {
            O_TRY(o_write_encoded_instruction(bytes, cubin_size, inspection->arena_start_file_offset + system_index * O_INSTRUCTION_BYTES, o_sass_nop(0u)));
        }
    }
    for (system_index = 0u; system_index < total_instruction_count; ++system_index) {
        O_TRY(o_sass_store(bytes, cubin_size, inspection->arena_start_file_offset + system_index * O_INSTRUCTION_BYTES, generated[system_index]));
    }
    if (inspection->system_capacity > 1u) {
        size_t body_cursor = 0u;
        uint32_t fallback = (uint32_t)(inspection->arena_start_file_offset - inspection->function_file_offset);
        uint32_t target = fallback;
        for (system_index = 0u; system_index < inspection->system_capacity; ++system_index) {
            if (system_index < system_count) {
                size_t body_start = inspection->arena_start_file_offset + body_cursor * O_INSTRUCTION_BYTES;
                target = (uint32_t)(body_start - inspection->function_file_offset);
                while (body_cursor < total_instruction_count && (generated[body_cursor].word0 & 0xffffu) != O_OPCODE_BRA)
                    ++body_cursor;
                ++body_cursor;
            } else {
                target = fallback;
            }
            o_write_u32(bytes + inspection->target_table_file_offsets[system_index], target);
            if (system_index == 0u) fallback = target;
        }
    }
    o_patch_register_count(bytes, inspection, register_count);

    report.system_count = system_count;
    report.body_instruction_count = total_instruction_count;
    report.high_water_register = high_water;
    report.register_count = register_count;
    if (report_ret != NULL) *report_ret = report;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_specialize_scoring_cubin_systems(
    void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    uint32_t active_toggle_count,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    void *workspace,
    size_t workspace_size,
    OdezzaScoringSpecializationReport *report_ret
) {
    return o_specialize_scoring_cubin_systems_impl(
        cubin,
        cubin_size,
        inspection,
        prespecialization,
        active_toggle_count,
        systems,
        system_count,
        workspace,
        workspace_size,
        report_ret,
        0
    );
}

OdezzaResult o_specialize_scoring_cubin_systems_fresh(
    void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    uint32_t active_toggle_count,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    void *workspace,
    size_t workspace_size,
    OdezzaScoringSpecializationReport *report_ret
) {
    return o_specialize_scoring_cubin_systems_impl(
        cubin,
        cubin_size,
        inspection,
        prespecialization,
        active_toggle_count,
        systems,
        system_count,
        workspace,
        workspace_size,
        report_ret,
        1
    );
}
