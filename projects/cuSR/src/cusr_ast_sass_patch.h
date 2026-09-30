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
#ifndef CUSR_AST_SASS_PATCH_H_INCLUDED
#define CUSR_AST_SASS_PATCH_H_INCLUDED

#include "cusr_ast_sass.h"
#include "cusr_sass_inspect.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#define CUSR_AST_SASS_PATCH_PUBLIC_DEF extern "C"
#else
#define CUSR_AST_SASS_PATCH_PUBLIC_DEF
#endif

typedef enum CusrAstSassPatchResult {
    CUSR_AST_SASS_PATCH_SUCCESS = 0,
    CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE = 1,
    CUSR_AST_SASS_PATCH_ERROR_OVERFLOW = 2,
    CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE = 3,
    CUSR_AST_SASS_PATCH_ERROR_AST_SASS = 4,
    CUSR_AST_SASS_PATCH_ERROR_PROGRAM_COUNT = 5,
    CUSR_AST_SASS_PATCH_ERROR_ARCH_MISMATCH = 6
} CusrAstSassPatchResult;

typedef enum CusrAstSassPatchEpilogue {
    CUSR_AST_SASS_PATCH_EPILOGUE_SSE = 0,
    CUSR_AST_SASS_PATCH_EPILOGUE_VALUE = 1
} CusrAstSassPatchEpilogue;

typedef struct CusrAstSassPatchStats {
    size_t sites_patched;
    size_t asts_patched;
    size_t sass_instructions_written;
    size_t sass_bytes_written;
    uint32_t max_original_register_count;
    uint32_t max_expanded_register_count;
    uint32_t max_patched_register_count;
} CusrAstSassPatchStats;

CUSR_AST_SASS_PATCH_PUBLIC_DEF
const char*
cusr_ast_sass_patch_result_to_string(CusrAstSassPatchResult result);

/*
 * Patching is in-place and intentionally destructive. A successful call
 * rewrites every inspected site and resets each kernel's register count for
 * the supplied programs, so the same cubin allocation can be patched again.
 * On error, the cubin may be partially modified and no rollback is performed.
 * Every kernel receives the same positive number of programs. num_programs
 * must therefore be divisible by inspect->num_kernels, and that quotient must
 * not exceed inspect->ast_capacity. The epilogue either updates one SSE
 * accumulator per AST or writes each AST value to its output register. The
 * patcher emits one branch over the unused site tail and writes only that used
 * prefix. The supplied capability must match inspect->sass_arch.
 */
CUSR_AST_SASS_PATCH_PUBLIC_DEF
CusrAstSassPatchResult
cusr_ast_sass_patch_cubin(
    const CusrSassInspectHandle* inspect,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrAstSassPatchEpilogue epilogue,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* const* programs,
    size_t num_programs,
    unsigned char* cubin,
    size_t cubin_size,
    CusrAstSassPatchStats* stats_ret
);

CUSR_AST_SASS_PATCH_PUBLIC_DEF
CusrAstSassPatchResult
cusr_ast_sass_patch_cubin_max_register_count(
    const CusrSassInspectHandle* inspect,
    const unsigned char* cubin,
    size_t cubin_size,
    uint32_t* max_register_count_ret
);

#endif /* CUSR_AST_SASS_PATCH_H_INCLUDED */

#ifdef CUSR_AST_SASS_PATCH_IMPLEMENTATION
#ifndef CUSR_AST_SASS_PATCH_IMPLEMENTATION_ONCE
#define CUSR_AST_SASS_PATCH_IMPLEMENTATION_ONCE

#ifndef CUSR_AST_SASS_IMPLEMENTATION_ONCE
#error "CUSR_AST_SASS_PATCH_IMPLEMENTATION requires CUSR_AST_SASS_IMPLEMENTATION"
#endif

#define CUSR_AST_SASS_PATCH_OPCODE_BRA 0x7947u
#define CUSR_AST_SASS_PATCH_REST_BRA 0x0003800000ull
#define CUSR_AST_SASS_PATCH_BRANCH_ENCODING_SM8X 1u
#define CUSR_AST_SASS_PATCH_BRANCH_ENCODING_SM90_PLUS 2u

#define _CUSR_AST_SASS_PATCH_ERROR_RET(ans) \
    do { \
        CusrAstSassPatchResult cusr_ast_sass_patch_result = (ans); \
        return cusr_ast_sass_patch_result; \
    } while (0)

#define _CUSR_AST_SASS_PATCH_CHECK_RET(ans) \
    do { \
        CusrAstSassPatchResult cusr_ast_sass_patch_check_ret = (ans); \
        if (cusr_ast_sass_patch_check_ret != CUSR_AST_SASS_PATCH_SUCCESS) { \
            _CUSR_AST_SASS_PATCH_ERROR_RET(cusr_ast_sass_patch_check_ret); \
        } \
    } while (0)

#define _CUSR_AST_SASS_PATCH_AST_CHECK_RET(ans) \
    do { \
        CusrAstSassResult cusr_ast_sass_patch_ast_result = (ans); \
        if (cusr_ast_sass_patch_ast_result != CUSR_AST_SASS_SUCCESS) { \
            _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_AST_SASS); \
        } \
    } while (0)

#ifndef CUSR_AST_SASS_PATCH_REGISTER_PAD
#define CUSR_AST_SASS_PATCH_REGISTER_PAD 2u
#endif

static uint32_t
cusr_ast_sass_patch_read_u32(const unsigned char* bytes)
{
    return
        ((uint32_t)bytes[0]) |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static void
cusr_ast_sass_patch_write_u32(unsigned char* bytes, uint32_t value)
{
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
}

static void
cusr_ast_sass_patch_write_u64(unsigned char* bytes, uint64_t value)
{
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    bytes[4] = (unsigned char)((value >> 32) & 0xffu);
    bytes[5] = (unsigned char)((value >> 40) & 0xffu);
    bytes[6] = (unsigned char)((value >> 48) & 0xffu);
    bytes[7] = (unsigned char)((value >> 56) & 0xffu);
}

static int
cusr_ast_sass_patch_range_ok(size_t size, size_t offset, size_t bytes)
{
    return offset <= size && bytes <= size - offset;
}

static int
cusr_ast_sass_patch_checked_mul(size_t a, size_t b, size_t* out)
{
    if (a != 0u && b > ((size_t)-1) / a) {
        return 0;
    }

    *out = a * b;
    return 1;
}

static void
cusr_ast_sass_patch_stats_max_u32(uint32_t* value, uint32_t candidate)
{
    if (candidate > *value) {
        *value = candidate;
    }
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_validate_capability(uint32_t capability_major, uint32_t capability_minor)
{
    if (capability_minor > 9u) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    if (capability_major == 8u ||
        capability_major == 9u ||
        capability_major == 10u ||
        capability_major == 11u ||
        capability_major == 12u) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
    }

    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
}

static uint32_t
cusr_ast_sass_patch_branch_encoding(uint32_t capability_major)
{
    return capability_major == 8u
        ? CUSR_AST_SASS_PATCH_BRANCH_ENCODING_SM8X
        : CUSR_AST_SASS_PATCH_BRANCH_ENCODING_SM90_PLUS;
}

static CusrSassInstruction
cusr_ast_sass_patch_bra_forward(size_t target_delta_instructions, uint32_t branch_encoding)
{
    CusrSassInstruction instruction;
    uint64_t branch_target_field;

    if (branch_encoding == CUSR_AST_SASS_PATCH_BRANCH_ENCODING_SM8X) {
        branch_target_field = (uint64_t)(16u * target_delta_instructions - 16u);
        instruction.word0 =
            (branch_target_field << 32) |
            (uint64_t)CUSR_AST_SASS_PATCH_OPCODE_BRA;
    } else {
        branch_target_field = (uint64_t)(4u * target_delta_instructions - 4u);
        instruction.word0 =
            ((branch_target_field & 0xffull) << 16) |
            ((branch_target_field & ~0xffull) << 26) |
            (uint64_t)CUSR_AST_SASS_PATCH_OPCODE_BRA;
    }

    instruction.word1 = cusr_ast_sass_alu_control(0u, CUSR_AST_SASS_PATCH_REST_BRA);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_patch_fadd_neg_lhs(uint8_t dst, uint8_t lhs, uint8_t rhs, uint32_t wait_mask)
{
    CusrSassInstruction instruction = cusr_ast_sass_fadd_reg(dst, lhs, rhs, wait_mask);
    instruction.word1 |= 0x100ull;
    return instruction;
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_kernel_original_regcount(
    const CusrSassInspectHandle* inspect,
    const CusrSassInspectKernel* kernel,
    uint32_t* regcount_ret)
{
    uint32_t regcount = 0u;
    size_t i;

    if (kernel->num_regcount_records == 0u) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    for (i = 0u; i < kernel->num_regcount_records; ++i) {
        const CusrSassInspectRegcountRecord* record =
            inspect->regcount_records + kernel->first_regcount_record + i;

        if (record->value > regcount) {
            regcount = record->value;
        }
    }

    *regcount_ret = regcount;
    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_kernel_current_regcount(
    const unsigned char* cubin,
    size_t cubin_size,
    const CusrSassInspectHandle* inspect,
    const CusrSassInspectKernel* kernel,
    uint32_t* regcount_ret)
{
    uint32_t regcount = 0u;
    size_t i;

    if (kernel->num_regcount_records == 0u) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    for (i = 0u; i < kernel->num_regcount_records; ++i) {
        const CusrSassInspectRegcountRecord* record =
            inspect->regcount_records + kernel->first_regcount_record + i;
        uint32_t value;

        if (!cusr_ast_sass_patch_range_ok(cubin_size, record->value_file_offset, 4u)) {
            _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE);
        }

        value = cusr_ast_sass_patch_read_u32(cubin + record->value_file_offset);
        if (value > regcount) {
            regcount = value;
        }
    }

    *regcount_ret = regcount;
    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_kernel_regcount(
    unsigned char* cubin,
    size_t cubin_size,
    const CusrSassInspectHandle* inspect,
    const CusrSassInspectKernel* kernel,
    uint32_t regcount)
{
    size_t i;

    for (i = 0u; i < kernel->num_regcount_records; ++i) {
        const CusrSassInspectRegcountRecord* record =
            inspect->regcount_records + kernel->first_regcount_record + i;

        if (!cusr_ast_sass_patch_range_ok(cubin_size, record->value_file_offset, 4u)) {
            _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE);
        }

        cusr_ast_sass_patch_write_u32(cubin + record->value_file_offset, regcount);
    }

    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_write_site(
    unsigned char* cubin,
    size_t cubin_size,
    const CusrSassInspectSite* site,
    const CusrSassInstruction* sass,
    size_t sass_count)
{
    size_t i;
    size_t bytes;

    if (!cusr_ast_sass_patch_checked_mul(sass_count, CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES, &bytes)) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_OVERFLOW);
    }

    if (!cusr_ast_sass_patch_range_ok(
            cubin_size,
            site->start_file_offset,
            bytes)) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE);
    }

    for (i = 0u; i < sass_count; ++i) {
        unsigned char* dst = cubin + site->start_file_offset + i * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
        cusr_ast_sass_patch_write_u64(dst, sass[i].word0);
        cusr_ast_sass_patch_write_u64(dst + 8u, sass[i].word1);
    }

    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

static inline CusrAstSassPatchResult
cusr_ast_sass_patch_emit_sse_update(
    CusrAstSassAssembler* assembler,
    CusrAstOperand prediction,
    uint8_t target_register,
    uint8_t sse_register)
{
    CusrAstOperand error;
    uint32_t wait_mask;

    _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &prediction));

    if (prediction.owned && !cusr_ast_sass_register_is_reserved(assembler, prediction.reg)) {
        error = prediction;
    } else {
        _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &error));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &prediction, 1u);
    _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_patch_fadd_neg_lhs(error.reg, target_register, prediction.reg, wait_mask)
    ));

    if (prediction.owned && prediction.reg != error.reg) {
        cusr_ast_sass_release_register(assembler, prediction);
    }

    error.barrier_active = 0u;
    error.barrier_slot = 0u;
    error.barrier_token = 0u;
    _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_ffma_reg(sse_register, error.reg, error.reg, sse_register, 0u)
    ));
    cusr_ast_sass_release_register(assembler, error);

    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

static inline CusrAstSassPatchResult
cusr_ast_sass_patch_emit_value(
    CusrAstSassAssembler* assembler,
    CusrAstOperand value,
    uint8_t output_register)
{
    uint32_t wait_mask;

    if (value.is_immediate) {
        _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_emit(
            assembler,
            cusr_ast_sass_fadd_imm(output_register, CUSR_AST_SASS_REGISTER_RZ, value.immediate_bits, 0u)
        ));
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &value, 1u);
    _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_mov(output_register, value.reg, wait_mask)
    ));

    if (value.owned && value.reg != output_register) {
        cusr_ast_sass_release_register(assembler, value);
    }

    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_generate_site(
    const CusrSassInspectSite* site,
    size_t site_instructions,
    uint32_t original_register_count,
    uint32_t capability_major,
    CusrAstSassPatchEpilogue epilogue,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* const* programs,
    size_t num_programs,
    CusrSassInstruction* sass,
    size_t sass_capacity,
    size_t* sass_count_ret,
    uint32_t* expanded_register_count_ret)
{
    CusrAstSassAssembler assembler;
    const uint32_t branch_encoding = cusr_ast_sass_patch_branch_encoding(capability_major);
    size_t input_idx;
    size_t program_idx;

    if (site == NULL ||
        programs == NULL ||
        num_programs == 0u ||
        num_programs > site->num_output_regs ||
        sass == NULL ||
        site_instructions == 0u ||
        site_instructions > sass_capacity ||
        sass_count_ret == NULL ||
        expanded_register_count_ret == NULL ||
        original_register_count >= CUSR_AST_SASS_REGISTER_RZ ||
        (epilogue != CUSR_AST_SASS_PATCH_EPILOGUE_SSE &&
         epilogue != CUSR_AST_SASS_PATCH_EPILOGUE_VALUE) ||
        site->target_reg == CUSR_AST_SASS_REGISTER_RZ) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    memset(&assembler, 0, sizeof(assembler));
    assembler.sass = sass;
    assembler.capacity = site_instructions;
    assembler.next_register = original_register_count;
    assembler.high_water_register = original_register_count;
    assembler.result_register = CUSR_AST_SASS_REGISTER_RZ;
    assembler.next_barrier_token = 1u;
    assembler.pending_wait_mask = site->incoming_wait_mask;

    for (input_idx = 0u; input_idx < CUSR_SASS_INSPECT_INPUTS; ++input_idx) {
        cusr_ast_sass_reserve_register(&assembler, site->input_regs[input_idx]);
    }
    cusr_ast_sass_reserve_register(&assembler, site->target_reg);

    for (program_idx = 0u; program_idx < site->num_output_regs; ++program_idx) {
        if (site->output_regs[program_idx] == CUSR_AST_SASS_REGISTER_RZ) {
            _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
        }
        cusr_ast_sass_reserve_register(&assembler, site->output_regs[program_idx]);
    }

    cusr_ast_sass_prepare_free_registers(
        &assembler,
        site->available_regs,
        site->num_available_regs
    );

    for (program_idx = 0u; program_idx < num_programs; ++program_idx) {
        CusrAstOperand prediction;

        if (programs[program_idx] == NULL || assembler.stack_size != 0u) {
            _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_AST_SASS);
        }

        _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_compile_frame(
            &assembler,
            site->input_regs,
            CUSR_SASS_INSPECT_INPUTS,
            routines,
            num_routines,
            programs[program_idx],
            NULL,
            0u,
            0u
        ));
        _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_pop(&assembler, &prediction));
        if (epilogue == CUSR_AST_SASS_PATCH_EPILOGUE_SSE) {
            _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_emit_sse_update(
                &assembler,
                prediction,
                site->target_reg,
                site->output_regs[program_idx]
            ));
        } else {
            _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_emit_value(
                &assembler,
                prediction,
                site->output_regs[program_idx]
            ));
        }
    }

    if (assembler.stack_size != 0u) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_AST_SASS);
    }

    if (cusr_ast_sass_active_barrier_mask(&assembler) != 0u) {
        _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_emit(
            &assembler,
            cusr_ast_sass_nop(cusr_ast_sass_active_barrier_mask(&assembler))
        ));
    }

    if (assembler.count < site_instructions) {
        const size_t remaining = site_instructions - assembler.count;
        const CusrSassInstruction tail = remaining == 1u
            ? cusr_ast_sass_nop(0u)
            : cusr_ast_sass_patch_bra_forward(remaining, branch_encoding);
        _CUSR_AST_SASS_PATCH_AST_CHECK_RET(cusr_ast_sass_emit(&assembler, tail));
    }

    *sass_count_ret = assembler.count;
    *expanded_register_count_ret = assembler.high_water_register;
    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

CUSR_AST_SASS_PATCH_PUBLIC_DEF
const char*
cusr_ast_sass_patch_result_to_string(CusrAstSassPatchResult result)
{
    switch (result) {
        case CUSR_AST_SASS_PATCH_SUCCESS:
            return "CUSR_AST_SASS_PATCH_SUCCESS";
        case CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE:
            return "CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE";
        case CUSR_AST_SASS_PATCH_ERROR_OVERFLOW:
            return "CUSR_AST_SASS_PATCH_ERROR_OVERFLOW";
        case CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE:
            return "CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE";
        case CUSR_AST_SASS_PATCH_ERROR_AST_SASS:
            return "CUSR_AST_SASS_PATCH_ERROR_AST_SASS";
        case CUSR_AST_SASS_PATCH_ERROR_PROGRAM_COUNT:
            return "CUSR_AST_SASS_PATCH_ERROR_PROGRAM_COUNT";
        case CUSR_AST_SASS_PATCH_ERROR_ARCH_MISMATCH:
            return "CUSR_AST_SASS_PATCH_ERROR_ARCH_MISMATCH";
    }

    return "CUSR_AST_SASS_PATCH_ERROR_UNKNOWN";
}

static CusrAstSassPatchResult
cusr_ast_sass_patch_cubin_impl(
    const CusrSassInspectHandle* inspect,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrAstSassPatchEpilogue epilogue,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* const* programs,
    size_t num_programs,
    unsigned char* cubin,
    size_t cubin_size,
    CusrAstSassPatchStats* stats_ret)
{
    CusrSassInstruction sass[CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS];
    size_t programs_per_kernel;
    size_t kernel_idx;

    if (stats_ret != NULL) {
        memset(stats_ret, 0, sizeof(*stats_ret));
    }

    if (cubin == NULL ||
        inspect == NULL ||
        inspect->kernels == NULL ||
        inspect->sites == NULL ||
        inspect->regcount_records == NULL ||
        inspect->num_kernels == 0u ||
        inspect->ast_capacity == 0u ||
        inspect->ast_capacity > CUSR_SASS_INSPECT_MAX_OUTPUTS) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    if (programs == NULL) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    if (epilogue != CUSR_AST_SASS_PATCH_EPILOGUE_SSE &&
        epilogue != CUSR_AST_SASS_PATCH_EPILOGUE_VALUE) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_validate_capability(
        capability_major,
        capability_minor
    ));

    if (inspect->sass_arch != capability_major * 10u + capability_minor) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_ARCH_MISMATCH);
    }

    if (num_programs == 0u || num_programs % inspect->num_kernels != 0u) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_PROGRAM_COUNT);
    }
    programs_per_kernel = num_programs / inspect->num_kernels;
    if (programs_per_kernel == 0u || programs_per_kernel > inspect->ast_capacity) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_PROGRAM_COUNT);
    }

    for (kernel_idx = 0u; kernel_idx < inspect->num_kernels; ++kernel_idx) {
        const CusrSassInspectKernel* kernel = inspect->kernels + kernel_idx;
        uint32_t original_regcount = 0u;
        uint32_t patched_register_count;
        size_t site_offset;

        _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_kernel_original_regcount(
            inspect,
            kernel,
            &original_regcount
        ));

        if (stats_ret != NULL) {
            cusr_ast_sass_patch_stats_max_u32(&stats_ret->max_original_register_count, original_regcount);
        }

        patched_register_count = original_regcount;

        for (site_offset = 0u; site_offset < kernel->num_sites; ++site_offset) {
            const CusrSassInspectSite* site = inspect->sites + kernel->first_site + site_offset;
            const CusrAstInstruction* const* kernel_programs;
            size_t kernel_program_offset;
            size_t sass_count = 0u;
            uint32_t expanded_register_count = original_regcount;
            uint32_t required_register_count;
            size_t site_bytes;
            size_t site_instructions;

            if (site->end_file_offset <= site->start_file_offset) {
                _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE);
            }

            site_bytes = site->end_file_offset - site->start_file_offset;
            if ((site_bytes % CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES) != 0u) {
                _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE);
            }

            site_instructions = site_bytes / CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
            if (site_instructions == 0u ||
                site_instructions > CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS) {
                _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_CUBIN_RANGE);
            }

            if (site->num_output_regs != inspect->ast_capacity ||
                !cusr_ast_sass_patch_checked_mul(kernel_idx, programs_per_kernel, &kernel_program_offset)) {
                _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
            }
            kernel_programs = programs + kernel_program_offset;

            _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_generate_site(
                site,
                site_instructions,
                original_regcount,
                capability_major,
                epilogue,
                routines,
                num_routines,
                kernel_programs,
                programs_per_kernel,
                sass,
                CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS,
                &sass_count,
                &expanded_register_count
            ));

            if (sass_count == 0u || sass_count > site_instructions) {
                _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_AST_SASS);
            }

            required_register_count = expanded_register_count;
            if (expanded_register_count > original_regcount) {
                if (CUSR_AST_SASS_PATCH_REGISTER_PAD >
                    CUSR_AST_SASS_REGISTER_RZ - required_register_count) {
                    required_register_count = CUSR_AST_SASS_REGISTER_RZ;
                } else {
                    required_register_count += CUSR_AST_SASS_PATCH_REGISTER_PAD;
                }
            }

            if (required_register_count > patched_register_count) {
                patched_register_count = required_register_count;
            }

            _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_write_site(
                cubin,
                cubin_size,
                site,
                sass,
                sass_count
            ));

            if (stats_ret != NULL) {
                stats_ret->sites_patched += 1u;
                stats_ret->asts_patched += programs_per_kernel;
                stats_ret->sass_instructions_written += sass_count;
                stats_ret->sass_bytes_written += sass_count * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
                cusr_ast_sass_patch_stats_max_u32(&stats_ret->max_expanded_register_count, expanded_register_count);
            }
        }

        _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_kernel_regcount(
            cubin,
            cubin_size,
            inspect,
            kernel,
            patched_register_count
        ));

        if (stats_ret != NULL) {
            cusr_ast_sass_patch_stats_max_u32(&stats_ret->max_patched_register_count, patched_register_count);
        }
    }

    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

CUSR_AST_SASS_PATCH_PUBLIC_DEF
CusrAstSassPatchResult
cusr_ast_sass_patch_cubin_max_register_count(
    const CusrSassInspectHandle* inspect,
    const unsigned char* cubin,
    size_t cubin_size,
    uint32_t* max_register_count_ret)
{
    size_t kernel_idx;
    uint32_t max_register_count = 0u;

    if (inspect == NULL ||
        cubin == NULL ||
        max_register_count_ret == NULL ||
        inspect->kernels == NULL ||
        inspect->regcount_records == NULL) {
        _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_ERROR_INVALID_VALUE);
    }

    for (kernel_idx = 0u; kernel_idx < inspect->num_kernels; ++kernel_idx) {
        const CusrSassInspectKernel* kernel = inspect->kernels + kernel_idx;
        uint32_t register_count = 0u;

        _CUSR_AST_SASS_PATCH_CHECK_RET(cusr_ast_sass_patch_kernel_current_regcount(
            cubin,
            cubin_size,
            inspect,
            kernel,
            &register_count
        ));

        cusr_ast_sass_patch_stats_max_u32(&max_register_count, register_count);
    }

    *max_register_count_ret = max_register_count;
    _CUSR_AST_SASS_PATCH_ERROR_RET(CUSR_AST_SASS_PATCH_SUCCESS);
}

CUSR_AST_SASS_PATCH_PUBLIC_DEF
CusrAstSassPatchResult
cusr_ast_sass_patch_cubin(
    const CusrSassInspectHandle* inspect,
    uint32_t capability_major,
    uint32_t capability_minor,
    CusrAstSassPatchEpilogue epilogue,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* const* programs,
    size_t num_programs,
    unsigned char* cubin,
    size_t cubin_size,
    CusrAstSassPatchStats* stats_ret)
{
    return cusr_ast_sass_patch_cubin_impl(
        inspect,
        capability_major,
        capability_minor,
        epilogue,
        routines,
        num_routines,
        programs,
        num_programs,
        cubin,
        cubin_size,
        stats_ret
    );
}

#endif /* CUSR_AST_SASS_PATCH_IMPLEMENTATION_ONCE */
#endif /* CUSR_AST_SASS_PATCH_IMPLEMENTATION */
