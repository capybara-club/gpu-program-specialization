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

#include <stdint.h>
#include <string.h>

#define O_ELF_SECTION_BYTES 64u
#define O_ELF_SYMBOL_BYTES 24u
#define O_INSTRUCTION_BYTES 16u
#define O_MAX_MARKER_VALUES 255u
#define O_ARENA_ALIGNMENT 8u

#define O_BPT_WORD0 UINT64_C(0x000000040000795c)
#define O_OPCODE_FADD_REG 0x7221u
#define O_OPCODE_FADD_IMM 0x7421u
#define O_OPCODE_FSEL 0x7208u
#define O_OPCODE_LOP3_IMM 0x7812u
#define O_OPCODE_STS_RZ 0x7388u
#define O_OPCODE_STS_UR 0x7988u
#define O_OPCODE_UMOV 0x7c82u
#define O_NVINFO_FORMAT_U32 0x04u
#define O_NVINFO_ATTR_REGCOUNT 0x2fu
#define O_NVINFO_ATTR_SIZE 8u
#define O_WAIT_BARRIER_MASK 0x3fu
#define O_LOP3_TOGGLE_WORD0_LOW 0x00ff7812u
#define O_LOP3_TOGGLE_WORD1_BASE UINT64_C(0x0780c0ff)

#define O_FIRST_MARKER_BITS 0x7fc0a11eu
#define O_SHARED_BEGIN_MARKER_BITS 0x7fc1a11eu
#define O_SHARED_END_MARKER_BITS 0x7fc2a11eu
#define O_OUTPUT_LIVE_MARKER_BITS 0x7fc3a11eu

#define O_TRY(expression) \
    do { \
        OdezzaResult o_result_ = (expression); \
        if (o_result_ != ODEZZA_SUCCESS) return o_result_; \
    } while (0)

typedef struct OMarkerRecord {
    size_t offset;
    uint8_t destination;
    uint8_t source;
    uint8_t wait_mask;
    int present;
} OMarkerRecord;

typedef struct OScaffoldState {
    uint32_t predicate_register;
    uint8_t permutation_register;
    OdezzaScoringInstruction toggle_test_instruction;
    size_t toggle_test_count;
    size_t predicate_select_count;
} OScaffoldState;

typedef struct OInspectionAnalysis {
    OElf elf;
    OFunction function;
    ODataSymbol template_id_symbol;
    ODataSymbol state_capacity_symbol;
    ODataSymbol constant_capacity_symbol;
    ODataSymbol system_capacity_symbol;
    OElfSection target_section;
    int has_target_section;
    uint32_t architecture;
    uint32_t register_count;
    size_t state_capacity;
    uint32_t constant_capacity;
    size_t register_count_offset_count;
    size_t register_count_header_offset_count;
    size_t input_count;
    size_t output_count;
    size_t system_capacity;
    size_t scaffold_start;
    size_t scaffold_end;
    size_t shared_start;
    size_t shared_instruction_count;
    size_t shared_end;
    size_t arena_start;
    size_t arena_instruction_count;
    size_t requested_arena_instruction_count;
    size_t system_patch_capacity;
    int final_fallthrough_branch_elided;
    size_t arena_end;
    uint32_t incoming_wait_mask;
    uint32_t predicate_register;
    uint8_t permutation_register;
    OdezzaScoringInstruction toggle_test_instruction;
    size_t dispatch_instruction_count;
    size_t cleanup_offset_count;
    uint8_t input_registers[O_MAX_MARKER_VALUES];
    uint8_t output_registers[O_MAX_MARKER_VALUES];
    uint8_t final_output_registers[O_MAX_MARKER_VALUES];
    size_t output_materialization_offsets[O_MAX_MARKER_VALUES];
    uint8_t available_registers[O_MAX_MARKER_VALUES];
    size_t available_register_count;
    unsigned char template_id[32];
} OInspectionAnalysis;

typedef struct OArena {
    unsigned char *data;
    size_t size;
    size_t offset;
} OArena;

typedef struct OInspectionStorage {
    OdezzaScoringCubinInspection *inspection;
    size_t *register_count_offsets;
    size_t *register_count_header_offsets;
    size_t *dispatch_offsets;
    OdezzaScoringInstruction *dispatch_instructions;
    uint8_t *input_registers;
    uint8_t *output_registers;
    uint8_t *final_output_registers;
    size_t *output_materialization_offsets;
    uint8_t *available_registers;
    size_t *cleanup_offsets;
    size_t *target_table_offsets;
    uint32_t *original_target_values;
} OInspectionStorage;

static OdezzaResult o_find_target_section(const OElf *elf, OElfSection *section_ret, int *found_ret) {
    size_t index;
    int found = 0;
    if (elf == NULL || section_ret == NULL || found_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *found_ret = 0;
    for (index = 0u; index < elf->section_count; ++index) {
        OElfSection section;
        int matches;
        O_TRY(o_elf_section(elf, index, &section));
        O_TRY(o_section_name_matches(elf, &section, ".nv.constant2.odezza_scoring", &matches));
        if (!matches) continue;
        if (found) return ODEZZA_ERROR_FORMAT;
        *section_ret = section;
        found = 1;
    }
    if (!found) return ODEZZA_SUCCESS;
    O_TRY(o_check_range(elf->size, section_ret->offset, section_ret->size));
    *found_ret = 1;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_architecture(const OElf *elf, uint32_t *architecture_ret) {
    uint32_t low;
    uint32_t high;
    if (elf == NULL || architecture_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    low = elf->flags & 0xffu;
    high = (elf->flags >> 8u) & 0xffu;
    *architecture_ret = low >= 30u ? low : (high >= 30u ? high : 0u);
    if (*architecture_ret != 89u && *architecture_ret != 90u && *architecture_ret != 120u) return ODEZZA_ERROR_UNSUPPORTED;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_analyze_register_counts(
    const OElf *elf,
    uint32_t symbol_index,
    uint32_t *register_count_ret,
    size_t *file_offset_count_ret,
    size_t *header_offset_count_ret
) {
    uint32_t register_count = 0u;
    size_t file_count = 0u;
    size_t header_count = 0u;
    int found = 0;
    size_t section_index;
    if (elf == NULL || register_count_ret == NULL || file_offset_count_ret == NULL || header_offset_count_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (section_index = 0u; section_index < elf->section_count; ++section_index) {
        OElfSection section;
        int contains_info;
        int text_section;
        O_TRY(o_elf_section(elf, section_index, &section));
        O_TRY(o_section_name_contains(elf, &section, ".nv.info", &contains_info));
        if (contains_info) {
            size_t relative;
            O_TRY(o_check_range(elf->size, section.offset, section.size));
            for (relative = 0u; relative + 12u <= section.size; relative += 4u) {
                uint16_t attribute_size;
                uint32_t attribute_symbol;
                uint32_t value;
                size_t offset = section.offset + relative;
                if (elf->data[offset] != O_NVINFO_FORMAT_U32 || elf->data[offset + 1u] != O_NVINFO_ATTR_REGCOUNT) continue;
                O_TRY(o_read_u16(elf->data, elf->size, offset + 2u, &attribute_size));
                O_TRY(o_read_u32(elf->data, elf->size, offset + 4u, &attribute_symbol));
                if (attribute_size != O_NVINFO_ATTR_SIZE || attribute_symbol != symbol_index) continue;
                O_TRY(o_read_u32(elf->data, elf->size, offset + 8u, &value));
                if (found && value != register_count) return ODEZZA_ERROR_FORMAT;
                register_count = value;
                found = 1;
                ++file_count;
            }
        }
        O_TRY(o_section_name_starts_with(elf, &section, ".text.", &text_section));
        if (text_section && (section.info & 0x00ffffffu) == symbol_index) {
            uint32_t value = section.info >> 24u;
            if (value != 0u) {
                if (found && value != register_count) return ODEZZA_ERROR_FORMAT;
                register_count = value;
                found = 1;
                ++header_count;
            }
        }
    }
    if (!found) return ODEZZA_ERROR_FORMAT;
    *register_count_ret = register_count;
    *file_offset_count_ret = file_count;
    *header_offset_count_ret = header_count;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_instruction_words(const OElf *elf, size_t offset, uint64_t *word0_ret, uint64_t *word1_ret) {
    if (elf == NULL || word0_ret == NULL || word1_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_read_u64(elf->data, elf->size, offset, word0_ret));
    return o_read_u64(elf->data, elf->size, offset + 8u, word1_ret);
}

static OdezzaResult o_is_zero_umov(uint64_t word0, uint32_t architecture, int *matches_ret) {
    if (matches_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *matches_ret = (word0 & 0xffffu) == O_OPCODE_UMOV && (architecture < 90u || ((word0 >> 32u) & 0xffu) == (architecture == 90u ? 0x3fu : 0xffu));
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_marker_immediate(uint32_t immediate, int *matches_ret) {
    uint32_t marker_delta = immediate - O_FIRST_MARKER_BITS;
    uint32_t live_delta = immediate - O_OUTPUT_LIVE_MARKER_BITS;
    if (matches_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *matches_ret = marker_delta < O_MAX_MARKER_VALUES || live_delta < O_MAX_MARKER_VALUES || immediate == O_SHARED_BEGIN_MARKER_BITS ||
                   immediate == O_SHARED_END_MARKER_BITS;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_append_available(uint8_t value, uint8_t values[O_MAX_MARKER_VALUES], size_t *count) {
    size_t index;
    if (values == NULL || count == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (index = 0u; index < *count; ++index)
        if (values[index] == value) return ODEZZA_SUCCESS;
    if (*count == O_MAX_MARKER_VALUES) return ODEZZA_ERROR_OVERFLOW;
    values[(*count)++] = value;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_classify_scaffold_instruction(
    uint64_t word0,
    uint64_t word1,
    uint32_t architecture,
    OScaffoldState *state,
    int *cleanup_ret,
    int *available_ret,
    uint8_t *available_register_ret
) {
    uint32_t opcode;
    int marker;
    int zero_umov;
    if (state == NULL || cleanup_ret == NULL || available_ret == NULL || available_register_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *cleanup_ret = 0;
    *available_ret = 0;
    *available_register_ret = 0u;
    opcode = (uint32_t)(word0 & 0xffffu);
    if (word0 == O_BPT_WORD0) {
        *cleanup_ret = 1;
        return ODEZZA_SUCCESS;
    }
    if (opcode == O_OPCODE_FADD_IMM) {
        O_TRY(o_marker_immediate((uint32_t)(word0 >> 32u), &marker));
        if (marker) *cleanup_ret = 1;
        return ODEZZA_SUCCESS;
    }
    if (opcode == O_OPCODE_FADD_REG) {
        *cleanup_ret = 1;
        *available_ret = 1;
        *available_register_ret = (uint8_t)(word0 >> 16u);
        return ODEZZA_SUCCESS;
    }
    if (opcode == O_OPCODE_LOP3_IMM && ((uint32_t)word0 & 0x00ffffffu) == O_LOP3_TOGGLE_WORD0_LOW && (uint32_t)(word0 >> 32u) == 1u) {
        uint32_t predicate = (uint32_t)((word1 >> 17u) & 0x7u);
        uint64_t predicate_mask = UINT64_C(0x7) << 17u;
        if (((word1 & ((UINT64_C(1) << 40u) - 1u)) & ~predicate_mask) != O_LOP3_TOGGLE_WORD1_BASE) return ODEZZA_ERROR_FORMAT;
        if (state->toggle_test_count != 0u) return ODEZZA_ERROR_FORMAT;
        state->predicate_register = predicate;
        state->permutation_register = (uint8_t)(word0 >> 24u);
        state->toggle_test_instruction.word0 = word0;
        state->toggle_test_instruction.word1 = word1;
        ++state->toggle_test_count;
        *cleanup_ret = 1;
        return ODEZZA_SUCCESS;
    }
    if (opcode == O_OPCODE_FSEL) {
        uint32_t predicate = (uint32_t)((word1 >> 23u) & 0x7u);
        uint64_t predicate_mask = UINT64_C(0x7) << 23u;
        if ((word1 & ((UINT64_C(1) << 40u) - 1u)) & ~predicate_mask) return ODEZZA_ERROR_FORMAT;
        if (state->toggle_test_count == 0u || state->predicate_register != predicate) return ODEZZA_ERROR_FORMAT;
        ++state->predicate_select_count;
        *cleanup_ret = 1;
        *available_ret = 1;
        *available_register_ret = (uint8_t)(word0 >> 16u);
        return ODEZZA_SUCCESS;
    }
    if ((opcode & 0x0fffu) == (O_OPCODE_FADD_REG & 0x0fffu) && (opcode >> 12u) < 7u) {
        uint32_t predicate = opcode >> 12u;
        if ((word1 & ((UINT64_C(1) << 40u) - 1u)) != UINT64_C(0x00010000) || state->toggle_test_count == 0u ||
            state->predicate_register != predicate)
            return ODEZZA_ERROR_FORMAT;
        ++state->predicate_select_count;
        *cleanup_ret = 1;
        *available_ret = 1;
        *available_register_ret = (uint8_t)(word0 >> 16u);
        return ODEZZA_SUCCESS;
    }
    O_TRY(o_is_zero_umov(word0, architecture, &zero_umov));
    if (opcode == O_OPCODE_STS_RZ || opcode == O_OPCODE_STS_UR || zero_umov) *cleanup_ret = 1;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_register_reserved(const OInspectionAnalysis *analysis, uint8_t value, int *reserved_ret) {
    size_t index;
    if (analysis == NULL || reserved_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *reserved_ret = value == 255u;
    for (index = 0u; index < analysis->input_count; ++index)
        if (analysis->input_registers[index] == value) *reserved_ret = 1;
    if (analysis->permutation_register == value) *reserved_ret = 1;
    for (index = 0u; index < analysis->output_count; ++index) {
        if (analysis->output_registers[index] == value || analysis->final_output_registers[index] == value) *reserved_ret = 1;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_analyze_scoring_cubin(const void *cubin, size_t cubin_size, OInspectionAnalysis *analysis) {
    OMarkerRecord markers[O_MAX_MARKER_VALUES];
    OMarkerRecord live[O_MAX_MARKER_VALUES];
    OMarkerRecord shared_begin;
    OMarkerRecord shared_end;
    size_t marker_count = 0u;
    size_t live_count = 0u;
    size_t function_end;
    size_t offset;
    size_t bpt_count = 0u;
    size_t first_bpt = 0u;
    size_t last_bpt = 0u;
    uint32_t minimum_target = UINT32_MAX;
    uint32_t encoded_system_capacity;
    uint32_t encoded_capacity;
    size_t index;
    OScaffoldState scaffold_state;

    if (cubin == NULL || analysis == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(analysis, 0, sizeof(*analysis));
    memset(markers, 0, sizeof(markers));
    memset(live, 0, sizeof(live));
    memset(&shared_begin, 0, sizeof(shared_begin));
    memset(&shared_end, 0, sizeof(shared_end));
    memset(&scaffold_state, 0, sizeof(scaffold_state));
    O_TRY(o_elf_init(cubin, cubin_size, &analysis->elf));
    O_TRY(o_architecture(&analysis->elf, &analysis->architecture));
    O_TRY(o_find_function(&analysis->elf, "odezza_scoring", &analysis->function));
    O_TRY(o_find_data_symbol(&analysis->elf, "odezza_template_id", 32u, &analysis->template_id_symbol));
    O_TRY(o_find_data_symbol(&analysis->elf, "odezza_state_capacity", 4u, &analysis->state_capacity_symbol));
    O_TRY(o_find_data_symbol(&analysis->elf, "odezza_constant_capacity", 4u, &analysis->constant_capacity_symbol));
    O_TRY(o_find_data_symbol(&analysis->elf, "odezza_system_capacity", 4u, &analysis->system_capacity_symbol));
    memcpy(analysis->template_id, analysis->elf.data + analysis->template_id_symbol.file_offset, 32u);
    O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, analysis->system_capacity_symbol.file_offset, &encoded_system_capacity));
    analysis->system_capacity = encoded_system_capacity;
    if (analysis->system_capacity == 0u || analysis->system_capacity > 65535u) return ODEZZA_ERROR_FORMAT;
    O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, analysis->state_capacity_symbol.file_offset, &encoded_capacity));
    analysis->state_capacity = encoded_capacity;
    O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, analysis->constant_capacity_symbol.file_offset, &analysis->constant_capacity));
    if (analysis->state_capacity == 0u || analysis->state_capacity > 128u) return ODEZZA_ERROR_FORMAT;
    O_TRY(o_analyze_register_counts(&analysis->elf, analysis->function.symbol_index, &analysis->register_count, &analysis->register_count_offset_count,
                                    &analysis->register_count_header_offset_count));
    O_TRY(o_find_target_section(&analysis->elf, &analysis->target_section, &analysis->has_target_section));
    if (analysis->system_capacity > 1u) {
        if (!analysis->has_target_section || analysis->target_section.size != analysis->system_capacity * 4u) return ODEZZA_ERROR_FORMAT;
        for (index = 0u; index < analysis->system_capacity; ++index) {
            uint32_t target;
            size_t other;
            O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, analysis->target_section.offset + index * 4u, &target));
            if (target % O_INSTRUCTION_BYTES != 0u || target >= analysis->function.size) return ODEZZA_ERROR_FORMAT;
            if (target < minimum_target) minimum_target = target;
            for (other = 0u; other < index; ++other) {
                uint32_t previous;
                O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, analysis->target_section.offset + other * 4u, &previous));
                if (previous == target) return ODEZZA_ERROR_FORMAT;
            }
        }
        O_TRY(o_size_add(analysis->function.file_offset, minimum_target, &analysis->arena_start));
    } else if (analysis->has_target_section && analysis->target_section.size != 0u) {
        return ODEZZA_ERROR_FORMAT;
    }
    O_TRY(o_size_add(analysis->function.file_offset, analysis->function.size, &function_end));

    for (offset = analysis->function.file_offset; offset < function_end; offset += O_INSTRUCTION_BYTES) {
        uint64_t word0;
        uint64_t word1;
        uint32_t immediate;
        O_TRY(o_instruction_words(&analysis->elf, offset, &word0, &word1));
        if (word0 == O_BPT_WORD0) {
            if (bpt_count == 0u) first_bpt = offset;
            last_bpt = offset;
            ++bpt_count;
        }
        if ((word0 & 0xffffu) != O_OPCODE_FADD_IMM) continue;
        immediate = (uint32_t)(word0 >> 32u);
        if (immediate - O_FIRST_MARKER_BITS < O_MAX_MARKER_VALUES) {
            size_t marker_index = immediate - O_FIRST_MARKER_BITS;
            if (markers[marker_index].present) return ODEZZA_ERROR_FORMAT;
            markers[marker_index].offset = offset;
            markers[marker_index].destination = (uint8_t)(word0 >> 16u);
            markers[marker_index].source = (uint8_t)(word0 >> 24u);
            markers[marker_index].wait_mask = (uint8_t)((word1 >> 52u) & O_WAIT_BARRIER_MASK);
            markers[marker_index].present = 1;
            if (marker_count <= marker_index) marker_count = marker_index + 1u;
        }
        if (immediate - O_OUTPUT_LIVE_MARKER_BITS < O_MAX_MARKER_VALUES) {
            size_t live_index = immediate - O_OUTPUT_LIVE_MARKER_BITS;
            if (live[live_index].present) return ODEZZA_ERROR_FORMAT;
            live[live_index].offset = offset;
            live[live_index].destination = (uint8_t)(word0 >> 16u);
            live[live_index].source = (uint8_t)(word0 >> 24u);
            live[live_index].wait_mask = (uint8_t)((word1 >> 52u) & O_WAIT_BARRIER_MASK);
            live[live_index].present = 1;
            if (live_count <= live_index) live_count = live_index + 1u;
        }
        if (immediate == O_SHARED_BEGIN_MARKER_BITS || immediate == O_SHARED_END_MARKER_BITS) {
            OMarkerRecord *record = immediate == O_SHARED_BEGIN_MARKER_BITS ? &shared_begin : &shared_end;
            if (record->present) return ODEZZA_ERROR_FORMAT;
            record->offset = offset;
            record->destination = (uint8_t)(word0 >> 16u);
            record->source = (uint8_t)(word0 >> 24u);
            record->wait_mask = (uint8_t)((word1 >> 52u) & O_WAIT_BARRIER_MASK);
            record->present = 1;
        }
    }
    if (bpt_count == 0u || marker_count == 0u || live_count == 0u || live_count >= marker_count || !shared_begin.present || !shared_end.present)
        return ODEZZA_ERROR_FORMAT;
    if (analysis->system_capacity == 1u) {
        for (offset = shared_end.offset + O_INSTRUCTION_BYTES; offset < function_end; offset += O_INSTRUCTION_BYTES) {
            uint64_t word0;
            uint64_t word1;
            O_TRY(o_instruction_words(&analysis->elf, offset, &word0, &word1));
            if (word0 == O_BPT_WORD0) {
                analysis->arena_start = offset;
                break;
            }
        }
        if (analysis->arena_start == 0u) return ODEZZA_ERROR_FORMAT;
    }
    for (index = 0u; index < marker_count; ++index)
        if (!markers[index].present) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < live_count; ++index)
        if (!live[index].present) return ODEZZA_ERROR_FORMAT;
    analysis->input_count = marker_count - live_count;
    analysis->output_count = live_count;
    if (analysis->output_count != analysis->state_capacity || analysis->input_count != analysis->state_capacity + analysis->constant_capacity)
        return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < analysis->input_count; ++index) {
        if (markers[index].offset >= shared_begin.offset) return ODEZZA_ERROR_FORMAT;
        analysis->input_registers[index] = markers[index].source;
        analysis->incoming_wait_mask |= markers[index].wait_mask;
        O_TRY(o_append_available(markers[index].destination, analysis->available_registers, &analysis->available_register_count));
    }
    analysis->arena_end = SIZE_MAX;
    for (index = 0u; index < analysis->output_count; ++index) {
        OMarkerRecord final_record = markers[analysis->input_count + index];
        size_t other;
        if (final_record.offset <= analysis->arena_start || live[index].destination != final_record.source) return ODEZZA_ERROR_FORMAT;
        analysis->output_registers[index] = live[index].destination;
        analysis->final_output_registers[index] = final_record.destination;
        analysis->output_materialization_offsets[index] = final_record.offset;
        if (final_record.offset < analysis->arena_end) analysis->arena_end = final_record.offset;
        for (other = 0u; other < index; ++other) {
            if (analysis->output_registers[other] == analysis->output_registers[index] ||
                analysis->final_output_registers[other] == analysis->final_output_registers[index])
                return ODEZZA_ERROR_FORMAT;
        }
    }
    for (index = 0u; index < analysis->input_count; ++index) {
        size_t other;
        for (other = 0u; other < index; ++other)
            if (analysis->input_registers[other] == analysis->input_registers[index]) return ODEZZA_ERROR_FORMAT;
        for (other = 0u; other < analysis->output_count; ++other) {
            if (analysis->input_registers[index] == analysis->output_registers[other] ||
                analysis->input_registers[index] == analysis->final_output_registers[other])
                return ODEZZA_ERROR_FORMAT;
        }
    }
    if (analysis->arena_end <= analysis->arena_start || (analysis->arena_end - analysis->arena_start) % O_INSTRUCTION_BYTES != 0u) return ODEZZA_ERROR_FORMAT;
    analysis->arena_instruction_count = (analysis->arena_end - analysis->arena_start) / O_INSTRUCTION_BYTES;
    if (analysis->arena_instruction_count % analysis->system_capacity == 0u) {
        analysis->requested_arena_instruction_count = analysis->arena_instruction_count;
        analysis->final_fallthrough_branch_elided = 0;
    } else if ((analysis->arena_instruction_count + 1u) % analysis->system_capacity == 0u) {
        analysis->requested_arena_instruction_count = analysis->arena_instruction_count + 1u;
        analysis->final_fallthrough_branch_elided = 1;
    } else {
        return ODEZZA_ERROR_FORMAT;
    }
    analysis->system_patch_capacity = analysis->requested_arena_instruction_count / analysis->system_capacity;
    if (analysis->requested_arena_instruction_count < 2u * analysis->system_capacity) return ODEZZA_ERROR_FORMAT;

    analysis->shared_start = 0u;
    analysis->shared_instruction_count = 0u;
    for (offset = shared_begin.offset + O_INSTRUCTION_BYTES; offset < shared_end.offset; offset += O_INSTRUCTION_BYTES) {
        uint64_t word0;
        uint64_t word1;
        O_TRY(o_instruction_words(&analysis->elf, offset, &word0, &word1));
        if (word0 != O_BPT_WORD0) continue;
        if (analysis->shared_instruction_count == 0u)
            analysis->shared_start = offset;
        else if (offset != analysis->shared_start + analysis->shared_instruction_count * O_INSTRUCTION_BYTES)
            return ODEZZA_ERROR_FORMAT;
        ++analysis->shared_instruction_count;
    }
    if (analysis->shared_instruction_count == 0u) return ODEZZA_ERROR_FORMAT;
    O_TRY(o_size_add(analysis->shared_start, analysis->shared_instruction_count * O_INSTRUCTION_BYTES, &analysis->shared_end));
    if (analysis->shared_end > shared_end.offset || shared_end.offset >= analysis->arena_start) return ODEZZA_ERROR_FORMAT;
    analysis->scaffold_start = first_bpt;
    O_TRY(o_size_add(last_bpt, O_INSTRUCTION_BYTES, &analysis->scaffold_end));
    if (analysis->scaffold_end <= analysis->arena_end || analysis->scaffold_end > function_end) return ODEZZA_ERROR_FORMAT;

    for (offset = analysis->scaffold_start; offset < analysis->scaffold_end; offset += O_INSTRUCTION_BYTES) {
        uint64_t word0;
        uint64_t word1;
        int cleanup;
        int available;
        uint8_t available_register;
        if ((offset >= analysis->shared_start && offset < analysis->shared_end) || (offset >= analysis->arena_start && offset < analysis->arena_end)) continue;
        O_TRY(o_instruction_words(&analysis->elf, offset, &word0, &word1));
        O_TRY(o_classify_scaffold_instruction(word0, word1, analysis->architecture, &scaffold_state, &cleanup, &available, &available_register));
        if (cleanup) ++analysis->cleanup_offset_count;
        if (available) O_TRY(o_append_available(available_register, analysis->available_registers, &analysis->available_register_count));
        if (offset >= analysis->shared_end && offset < analysis->arena_start && !cleanup) ++analysis->dispatch_instruction_count;
    }
    if (scaffold_state.toggle_test_count != 1u || scaffold_state.predicate_select_count != 1u || scaffold_state.predicate_register >= 7u)
        return ODEZZA_ERROR_FORMAT;
    analysis->predicate_register = scaffold_state.predicate_register;
    analysis->permutation_register = scaffold_state.permutation_register;
    analysis->toggle_test_instruction = scaffold_state.toggle_test_instruction;
    analysis->incoming_wait_mask |= (uint32_t)(scaffold_state.toggle_test_instruction.word1 >> 52u) & O_WAIT_BARRIER_MASK;
    if (analysis->permutation_register == 255u) return ODEZZA_ERROR_FORMAT;
    for (index = 0u; index < analysis->input_count; ++index) {
        if (analysis->input_registers[index] == analysis->permutation_register) return ODEZZA_ERROR_FORMAT;
    }
    for (index = 0u; index < analysis->output_count; ++index) {
        if (analysis->output_registers[index] == analysis->permutation_register ||
            analysis->final_output_registers[index] == analysis->permutation_register)
            return ODEZZA_ERROR_FORMAT;
    }
    if ((analysis->system_capacity == 1u && analysis->dispatch_instruction_count != 0u) ||
        (analysis->system_capacity > 1u && (analysis->dispatch_instruction_count < 3u || analysis->dispatch_instruction_count > 10u)))
        return ODEZZA_ERROR_FORMAT;
    {
        size_t write_index = 0u;
        for (index = 0u; index < analysis->available_register_count; ++index) {
            int reserved;
            O_TRY(o_register_reserved(analysis, analysis->available_registers[index], &reserved));
            if (!reserved) analysis->available_registers[write_index++] = analysis->available_registers[index];
        }
        analysis->available_register_count = write_index;
    }
    if (analysis->available_register_count == 0u) return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_arena_take(OArena *arena, size_t byte_count, void **memory_ret) {
    size_t aligned;
    if (arena == NULL || memory_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (arena->offset > SIZE_MAX - (O_ARENA_ALIGNMENT - 1u)) return ODEZZA_ERROR_OVERFLOW;
    aligned = (arena->offset + O_ARENA_ALIGNMENT - 1u) & ~(O_ARENA_ALIGNMENT - 1u);
    if (byte_count > SIZE_MAX - aligned) return ODEZZA_ERROR_OVERFLOW;
    if (arena->data != NULL && (aligned > arena->size || byte_count > arena->size - aligned)) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    *memory_ret = arena->data == NULL || byte_count == 0u ? NULL : arena->data + aligned;
    arena->offset = aligned + byte_count;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_layout_inspection(
    const OInspectionAnalysis *analysis,
    void *arena_memory,
    size_t arena_size,
    OInspectionStorage *storage_ret,
    size_t *required_ret
) {
    OArena arena;
    size_t bytes;
    if (analysis == NULL || storage_ret == NULL || required_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(storage_ret, 0, sizeof(*storage_ret));
    arena.data = (unsigned char *)arena_memory;
    arena.size = arena_size;
    arena.offset = 0u;
    O_TRY(o_arena_take(&arena, sizeof(*storage_ret->inspection), (void **)&storage_ret->inspection));
    O_TRY(o_size_multiply(analysis->register_count_offset_count, sizeof(size_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->register_count_offsets));
    O_TRY(o_size_multiply(analysis->register_count_header_offset_count, sizeof(size_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->register_count_header_offsets));
    O_TRY(o_size_multiply(analysis->dispatch_instruction_count, sizeof(size_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->dispatch_offsets));
    O_TRY(o_size_multiply(analysis->dispatch_instruction_count, sizeof(OdezzaScoringInstruction), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->dispatch_instructions));
    O_TRY(o_arena_take(&arena, analysis->input_count, (void **)&storage_ret->input_registers));
    O_TRY(o_arena_take(&arena, analysis->output_count, (void **)&storage_ret->output_registers));
    O_TRY(o_arena_take(&arena, analysis->output_count, (void **)&storage_ret->final_output_registers));
    O_TRY(o_size_multiply(analysis->output_count, sizeof(size_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->output_materialization_offsets));
    O_TRY(o_arena_take(&arena, analysis->available_register_count, (void **)&storage_ret->available_registers));
    O_TRY(o_size_multiply(analysis->cleanup_offset_count, sizeof(size_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->cleanup_offsets));
    O_TRY(o_size_multiply(analysis->system_capacity > 1u ? analysis->system_capacity : 0u, sizeof(size_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->target_table_offsets));
    O_TRY(o_size_multiply(analysis->system_capacity > 1u ? analysis->system_capacity : 0u, sizeof(uint32_t), &bytes));
    O_TRY(o_arena_take(&arena, bytes, (void **)&storage_ret->original_target_values));
    *required_ret = arena.offset;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_fill_register_count_offsets(const OInspectionAnalysis *analysis, OInspectionStorage *storage) {
    size_t file_index = 0u;
    size_t header_index = 0u;
    size_t section_index;
    if (analysis == NULL || storage == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (section_index = 0u; section_index < analysis->elf.section_count; ++section_index) {
        OElfSection section;
        int contains_info;
        int text_section;
        O_TRY(o_elf_section(&analysis->elf, section_index, &section));
        O_TRY(o_section_name_contains(&analysis->elf, &section, ".nv.info", &contains_info));
        if (contains_info) {
            size_t relative;
            for (relative = 0u; relative + 12u <= section.size; relative += 4u) {
                size_t offset = section.offset + relative;
                uint16_t attribute_size;
                uint32_t attribute_symbol;
                if (analysis->elf.data[offset] != O_NVINFO_FORMAT_U32 || analysis->elf.data[offset + 1u] != O_NVINFO_ATTR_REGCOUNT) continue;
                O_TRY(o_read_u16(analysis->elf.data, analysis->elf.size, offset + 2u, &attribute_size));
                O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, offset + 4u, &attribute_symbol));
                if (attribute_size == O_NVINFO_ATTR_SIZE && attribute_symbol == analysis->function.symbol_index)
                    storage->register_count_offsets[file_index++] = offset + 8u;
            }
        }
        O_TRY(o_section_name_starts_with(&analysis->elf, &section, ".text.", &text_section));
        if (text_section && (section.info & 0x00ffffffu) == analysis->function.symbol_index && (section.info >> 24u) != 0u) {
            storage->register_count_header_offsets[header_index++] = analysis->elf.section_table_offset + O_ELF_SECTION_BYTES * section_index + 47u;
        }
    }
    if (file_index != analysis->register_count_offset_count || header_index != analysis->register_count_header_offset_count) return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_fill_scaffold_arrays(const OInspectionAnalysis *analysis, OInspectionStorage *storage) {
    OScaffoldState state;
    size_t cleanup_index = 0u;
    size_t dispatch_index = 0u;
    size_t offset;
    if (analysis == NULL || storage == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(&state, 0, sizeof(state));
    for (offset = analysis->scaffold_start; offset < analysis->scaffold_end; offset += O_INSTRUCTION_BYTES) {
        uint64_t word0;
        uint64_t word1;
        int cleanup;
        int available;
        uint8_t available_register;
        if ((offset >= analysis->shared_start && offset < analysis->shared_end) || (offset >= analysis->arena_start && offset < analysis->arena_end)) continue;
        O_TRY(o_instruction_words(&analysis->elf, offset, &word0, &word1));
        O_TRY(o_classify_scaffold_instruction(word0, word1, analysis->architecture, &state, &cleanup, &available, &available_register));
        if (cleanup) storage->cleanup_offsets[cleanup_index++] = offset;
        if (offset >= analysis->shared_end && offset < analysis->arena_start && !cleanup) {
            storage->dispatch_offsets[dispatch_index] = offset;
            storage->dispatch_instructions[dispatch_index].word0 = word0;
            storage->dispatch_instructions[dispatch_index].word1 = word1;
            ++dispatch_index;
        }
    }
    if (cleanup_index != analysis->cleanup_offset_count || dispatch_index != analysis->dispatch_instruction_count) return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_materialize_inspection(const OInspectionAnalysis *analysis, OInspectionStorage *storage) {
    OdezzaScoringCubinInspection *inspection;
    size_t index;
    if (analysis == NULL || storage == NULL || storage->inspection == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    inspection = storage->inspection;
    memset(inspection, 0, sizeof(*inspection));
    O_TRY(o_fill_register_count_offsets(analysis, storage));
    O_TRY(o_fill_scaffold_arrays(analysis, storage));
    memcpy(storage->input_registers, analysis->input_registers, analysis->input_count);
    memcpy(storage->output_registers, analysis->output_registers, analysis->output_count);
    memcpy(storage->final_output_registers, analysis->final_output_registers, analysis->output_count);
    memcpy(
        storage->output_materialization_offsets,
        analysis->output_materialization_offsets,
        analysis->output_count * sizeof(size_t)
    );
    memcpy(storage->available_registers, analysis->available_registers, analysis->available_register_count);
    if (analysis->system_capacity > 1u) {
        for (index = 0u; index < analysis->system_capacity; ++index) {
            storage->target_table_offsets[index] = analysis->target_section.offset + 4u * index;
            O_TRY(o_read_u32(analysis->elf.data, analysis->elf.size, storage->target_table_offsets[index], &storage->original_target_values[index]));
        }
    }
    inspection->cubin_size = analysis->elf.size;
    inspection->architecture = analysis->architecture;
    inspection->register_count = analysis->register_count;
    inspection->state_capacity = analysis->state_capacity;
    inspection->constant_capacity = analysis->constant_capacity;
    inspection->input_count = analysis->input_count;
    inspection->output_count = analysis->output_count;
    inspection->system_capacity = analysis->system_capacity;
    inspection->function_symbol_index = analysis->function.symbol_index;
    inspection->function_file_offset = analysis->function.file_offset;
    inspection->function_byte_size = analysis->function.size;
    inspection->scaffold_start_file_offset = analysis->scaffold_start;
    inspection->scaffold_end_file_offset = analysis->scaffold_end;
    inspection->shared_start_file_offset = analysis->shared_start;
    inspection->shared_instruction_count = analysis->shared_instruction_count;
    inspection->arena_start_file_offset = analysis->arena_start;
    inspection->arena_instruction_count = analysis->arena_instruction_count;
    inspection->requested_arena_instruction_count = analysis->requested_arena_instruction_count;
    inspection->system_patch_capacity = analysis->system_patch_capacity;
    inspection->final_fallthrough_branch_elided = analysis->final_fallthrough_branch_elided;
    inspection->arena_end_file_offset = analysis->arena_end;
    inspection->continuation_file_offset = analysis->arena_end;
    inspection->incoming_wait_mask = analysis->incoming_wait_mask;
    inspection->predicate_register = analysis->predicate_register;
    inspection->permutation_register = analysis->permutation_register;
    inspection->toggle_test_instruction = analysis->toggle_test_instruction;
    inspection->register_count_file_offset_count = analysis->register_count_offset_count;
    inspection->register_count_file_offsets = storage->register_count_offsets;
    inspection->register_count_header_file_offset_count = analysis->register_count_header_offset_count;
    inspection->register_count_header_file_offsets = storage->register_count_header_offsets;
    inspection->dispatch_instruction_count = analysis->dispatch_instruction_count;
    inspection->dispatch_file_offsets = storage->dispatch_offsets;
    inspection->dispatch_instructions = storage->dispatch_instructions;
    inspection->input_registers = storage->input_registers;
    inspection->output_registers = storage->output_registers;
    inspection->final_output_registers = storage->final_output_registers;
    inspection->output_materialization_file_offsets = storage->output_materialization_offsets;
    inspection->available_register_count = analysis->available_register_count;
    inspection->available_registers = storage->available_registers;
    inspection->cleanup_file_offset_count = analysis->cleanup_offset_count;
    inspection->cleanup_file_offsets = storage->cleanup_offsets;
    inspection->target_table_file_offsets = storage->target_table_offsets;
    inspection->original_target_values = storage->original_target_values;
    inspection->template_id_file_offset = analysis->template_id_symbol.file_offset;
    memcpy(inspection->template_id, analysis->template_id, sizeof(inspection->template_id));
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_inspect_scoring_cubin(
    const void *cubin,
    size_t cubin_size,
    void *arena,
    size_t arena_size,
    size_t *arena_size_ret,
    const OdezzaScoringCubinInspection **inspection_ret
) {
    OInspectionAnalysis analysis;
    OInspectionStorage measured;
    OInspectionStorage storage;
    size_t required;
    size_t materialized_size;

    if (arena_size_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *arena_size_ret = 0u;
    if (inspection_ret != NULL) *inspection_ret = NULL;
    O_TRY(o_analyze_scoring_cubin(cubin, cubin_size, &analysis));
    O_TRY(o_layout_inspection(&analysis, NULL, 0u, &measured, &required));
    *arena_size_ret = required;
    if (arena == NULL) return ODEZZA_SUCCESS;
    if (inspection_ret == NULL || ((uintptr_t)arena & (O_ARENA_ALIGNMENT - 1u)) != 0u) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (arena_size < required) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    O_TRY(o_layout_inspection(&analysis, arena, arena_size, &storage, &materialized_size));
    if (materialized_size != required) return ODEZZA_ERROR_FORMAT;
    O_TRY(o_materialize_inspection(&analysis, &storage));
    *inspection_ret = storage.inspection;
    return ODEZZA_SUCCESS;
}
