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
#ifndef O_SCORING_FIXTURE_H
#define O_SCORING_FIXTURE_H

#include "o_odezza_internal.h"
#include <string.h>

#define CUBIN_BYTES 8192u
#define WORKSPACE_BYTES 4096u

typedef union AlignedBytes {
    uint64_t alignment;
    unsigned char bytes[WORKSPACE_BYTES];
} AlignedBytes;

static void initialize_inspection(unsigned char cubin[CUBIN_BYTES], OdezzaScoringCubinInspection *inspection) {
    static const size_t dispatch_offsets[] = {800u};
    static const OdezzaScoringInstruction dispatch_instructions[] = {{UINT64_C(0x1234), UINT64_C(0x5678)}};
    static const uint8_t input_registers[] = {0u, 1u, 2u, 3u, 4u, 5u};
    static const uint8_t output_registers[] = {8u, 9u};
    static const uint8_t final_output_registers[] = {10u, 11u};
    static const size_t output_materialization_offsets[] = {3072u, 3088u};
    static const size_t cleanup_offsets[] = {3120u};
    static const size_t target_table_offsets[] = {3500u, 3504u, 3508u, 3512u};
    static const uint32_t original_targets[] = {1024u, 1536u, 2048u, 2560u};
    size_t index;

    memset(cubin, 0xa5, CUBIN_BYTES);
    memset(inspection, 0, sizeof(*inspection));
    inspection->cubin_size = CUBIN_BYTES;
    inspection->architecture = 120u;
    inspection->register_count = 32u;
    inspection->state_capacity = 2u;
    inspection->constant_capacity = 4u;
    inspection->input_count = 6u;
    inspection->output_count = 2u;
    inspection->system_capacity = 4u;
    inspection->function_file_offset = 0u;
    inspection->function_byte_size = 3400u;
    inspection->shared_start_file_offset = 256u;
    inspection->shared_instruction_count = 32u;
    inspection->arena_start_file_offset = 1024u;
    inspection->arena_instruction_count = 128u;
    inspection->requested_arena_instruction_count = 128u;
    inspection->system_patch_capacity = 32u;
    inspection->arena_end_file_offset = 3072u;
    inspection->continuation_file_offset = 3072u;
    inspection->predicate_register = 1u;
    inspection->permutation_register = 6u;
    inspection->toggle_test_instruction.word0 = UINT64_C(0x0000000106ff7812);
    inspection->toggle_test_instruction.word1 = UINT64_C(0x000fc0000782c0ff);
    inspection->dispatch_instruction_count = 1u;
    inspection->dispatch_file_offsets = dispatch_offsets;
    inspection->dispatch_instructions = dispatch_instructions;
    inspection->input_registers = input_registers;
    inspection->output_registers = output_registers;
    inspection->final_output_registers = final_output_registers;
    inspection->output_materialization_file_offsets = output_materialization_offsets;
    inspection->available_register_count = 0u;
    inspection->available_registers = NULL;
    inspection->cleanup_file_offset_count = 1u;
    inspection->cleanup_file_offsets = cleanup_offsets;
    inspection->target_table_file_offsets = target_table_offsets;
    inspection->original_target_values = original_targets;
    inspection->template_id_file_offset = 4000u;
    for (index = 0u; index < 32u; ++index)
        inspection->template_id[index] = (unsigned char)(index * 7u + 3u);
    memcpy(cubin + inspection->template_id_file_offset, inspection->template_id, 32u);
}

#endif
