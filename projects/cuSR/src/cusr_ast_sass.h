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
#ifndef CUSR_AST_SASS_H_INCLUDED
#define CUSR_AST_SASS_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "cusr_ast.h"

#ifdef __cplusplus
#define CUSR_AST_SASS_PUBLIC_DEF extern "C"
#else
#define CUSR_AST_SASS_PUBLIC_DEF
#endif

#ifndef CUSR_AST_SASS_STACK_DEPTH
#define CUSR_AST_SASS_STACK_DEPTH 128u
#endif

#ifndef CUSR_AST_SASS_MAX_FREE_REGISTERS
#define CUSR_AST_SASS_MAX_FREE_REGISTERS 256u
#endif

#define CUSR_AST_SASS_NUM_BARRIER_SLOTS 6u
#define CUSR_AST_SASS_REGISTER_RZ 255u

typedef enum CusrAstSassResult {
    CUSR_AST_SASS_SUCCESS = 0,
    CUSR_AST_SASS_ERROR_INVALID_VALUE = 1,
    CUSR_AST_SASS_ERROR_STACK_OVERFLOW = 2,
    CUSR_AST_SASS_ERROR_STACK_UNDERFLOW = 3,
    CUSR_AST_SASS_ERROR_BAD_PROGRAM = 4,
    CUSR_AST_SASS_ERROR_UNSUPPORTED_OP = 5,
    CUSR_AST_SASS_ERROR_INSUFFICIENT_INSTRUCTIONS = 6,
    CUSR_AST_SASS_ERROR_REGISTER_OVERFLOW = 7,
    CUSR_AST_SASS_ERROR_BARRIER_OVERFLOW = 8,
    CUSR_AST_SASS_ERROR_TOO_MANY_ARGS = 9,
    CUSR_AST_SASS_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS = 10,
    CUSR_AST_SASS_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS = 11,
    CUSR_AST_SASS_ERROR_ROUTINE_DEPTH_EXCEEDED = 12
} CusrAstSassResult;

typedef struct CusrSassInstruction {
    uint64_t word0;
    uint64_t word1;
} CusrSassInstruction;

CUSR_AST_SASS_PUBLIC_DEF
const char*
cusr_ast_sass_result_to_string(CusrAstSassResult result);

/* Emits one AST only. sass_capacity limits output; no branch or padding is emitted. */
CUSR_AST_SASS_PUBLIC_DEF
CusrAstSassResult
cusr_ast_sass_generate(
    const uint8_t* input_registers,
    size_t num_input_registers,
    const uint8_t* available_registers,
    size_t num_available_registers,
    uint8_t result_register,
    uint32_t incoming_wait_mask,
    uint32_t first_new_register,
    uint32_t capability_major,
    uint32_t capability_minor,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* instructions,
    CusrSassInstruction* sass,
    size_t sass_capacity,
    size_t* sass_count_ret,
    uint32_t* expanded_register_count_ret
);

CUSR_AST_SASS_PUBLIC_DEF
void
cusr_ast_sass_print(
    FILE* stream,
    const CusrSassInstruction* sass,
    size_t num_instructions
);

#endif /* CUSR_AST_SASS_H_INCLUDED */

#ifdef CUSR_AST_SASS_IMPLEMENTATION
#ifndef CUSR_AST_SASS_IMPLEMENTATION_ONCE
#define CUSR_AST_SASS_IMPLEMENTATION_ONCE

#include <string.h>

#define CUSR_AST_SASS_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]

CUSR_AST_SASS_STATIC_ASSERT(sizeof(CusrAstInstruction) == 8, cusr_ast_sass_instruction_must_be_eight_bytes);

#ifndef CUSR_AST_SASS_ROUTINE_DEPTH
#define CUSR_AST_SASS_ROUTINE_DEPTH 8u
#endif

#define CUSR_AST_SASS_OPCODE_MOV 0x7202u
#define CUSR_AST_SASS_OPCODE_FMNMX_REG 0x7209u
#define CUSR_AST_SASS_OPCODE_FMUL_REG 0x7220u
#define CUSR_AST_SASS_OPCODE_FADD_REG 0x7221u
#define CUSR_AST_SASS_OPCODE_FFMA_REG 0x7223u
#define CUSR_AST_SASS_OPCODE_FADD_IMM 0x7421u
#define CUSR_AST_SASS_OPCODE_FMUL_IMM 0x7820u
#define CUSR_AST_SASS_OPCODE_MUFU 0x7308u
#define CUSR_AST_SASS_OPCODE_NOP 0x7918u
#define CUSR_AST_SASS_OPCODE_BRA 0x7947u

#define CUSR_AST_SASS_CONTROL_ALU_HI 0xfcu
#define CUSR_AST_SASS_CONTROL_ALU_STALL 0xcu
#define CUSR_AST_SASS_CONTROL_BARRIER_BASE 0xe2u
#define CUSR_AST_SASS_CONTROL_MUFU_STALL 0x6u

/* Low control/source fields matched against ptxas SASS output for these f32 forms. */
#define CUSR_AST_SASS_REST_FADD_IMM 0x0000010000ull
#define CUSR_AST_SASS_REST_FADD_ABS 0x0000010200ull
#define CUSR_AST_SASS_REST_FMUL 0x0000410000ull
#define CUSR_AST_SASS_REST_FMUL_RZ 0x000040c000ull
#define CUSR_AST_SASS_REST_FMNMX_MIN 0x0003810000ull
#define CUSR_AST_SASS_REST_FMNMX_MAX 0x0007810000ull
#define CUSR_AST_SASS_REST_FFMA_BASE 0x0000010000ull
#define CUSR_AST_SASS_REST_MOV 0x0000000f00ull
#define CUSR_AST_SASS_REST_MUFU_COS 0x0000000000ull
#define CUSR_AST_SASS_REST_MUFU_SIN 0x0000000400ull
#define CUSR_AST_SASS_REST_MUFU_EX2 0x0000000800ull
#define CUSR_AST_SASS_REST_MUFU_LG2 0x0000000c00ull
#define CUSR_AST_SASS_REST_MUFU_RCP 0x0000001000ull
#define CUSR_AST_SASS_REST_MUFU_RSQ 0x0000001400ull
#define CUSR_AST_SASS_REST_MUFU_SQRT 0x0000002000ull
#define CUSR_AST_SASS_REST_MUFU_TANH 0x0000002400ull

#define CUSR_AST_SASS_NOP_WORD0 0x0000000000007918ull
#define CUSR_AST_SASS_NOP_WORD1 0x000fc00000000000ull

#define CUSR_AST_SASS_NEG_ONE_BITS 0xbf800000u
#define CUSR_AST_SASS_ZERO_BITS 0x00000000u
#define CUSR_AST_SASS_SIN_COS_SCALE_BITS 0x3e22f983u

#define _CUSR_AST_SASS_ERROR_RET(ans) \
    do { \
        CusrAstSassResult cusr_ast_sass_result = (ans); \
        return cusr_ast_sass_result; \
    } while (0)

#define _CUSR_AST_SASS_CHECK_RET(ans) \
    do { \
        CusrAstSassResult cusr_ast_sass_check_ret = (ans); \
        if (cusr_ast_sass_check_ret != CUSR_AST_SASS_SUCCESS) { \
            _CUSR_AST_SASS_ERROR_RET(cusr_ast_sass_check_ret); \
        } \
    } while (0)

typedef struct CusrAstOperand {
    uint8_t is_immediate;
    uint8_t owned;
    uint8_t reg;
    uint32_t immediate_bits;
    uint8_t barrier_active;
    uint8_t barrier_slot;
    uint32_t barrier_token;
} CusrAstOperand;

typedef struct CusrAstSassAssembler {
    CusrSassInstruction* sass;
    size_t capacity;
    size_t count;
    CusrAstOperand stack[CUSR_AST_SASS_STACK_DEPTH];
    size_t stack_size;
    uint8_t free_registers[CUSR_AST_SASS_MAX_FREE_REGISTERS];
    size_t num_free_registers;
    uint32_t next_register;
    uint32_t high_water_register;
    uint8_t result_register;
    uint8_t reserved_registers[256u];
    uint8_t barrier_slot_active[CUSR_AST_SASS_NUM_BARRIER_SLOTS];
    uint32_t barrier_slot_token[CUSR_AST_SASS_NUM_BARRIER_SLOTS];
    uint32_t next_barrier_token;
    uint32_t pending_wait_mask;
} CusrAstSassAssembler;

static uint64_t
cusr_ast_sass_control(uint32_t wait_mask, uint32_t control_hi, uint32_t stall, uint64_t rest)
{
    return
        (((uint64_t)wait_mask & 0xfffull) << 52) |
        (((uint64_t)control_hi & 0xffull) << 44) |
        (((uint64_t)stall & 0xfull) << 40) |
        rest;
}

static uint64_t
cusr_ast_sass_alu_control(uint32_t wait_mask, uint64_t rest)
{
    return cusr_ast_sass_control(
        wait_mask,
        CUSR_AST_SASS_CONTROL_ALU_HI,
        CUSR_AST_SASS_CONTROL_ALU_STALL,
        rest
    );
}

static uint64_t
cusr_ast_sass_barrier_control(uint32_t wait_mask, uint32_t barrier_slot, uint64_t rest)
{
    return cusr_ast_sass_control(
        wait_mask,
        CUSR_AST_SASS_CONTROL_BARRIER_BASE + 4u * barrier_slot,
        CUSR_AST_SASS_CONTROL_MUFU_STALL,
        rest
    );
}

static CusrSassInstruction
cusr_ast_sass_nop(uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 = CUSR_AST_SASS_NOP_WORD0;
    instruction.word1 = (((uint64_t)wait_mask & 0xfffull) << 52) | CUSR_AST_SASS_NOP_WORD1;
    return instruction;
}

static CusrAstSassResult
cusr_ast_sass_validate_capability(uint32_t capability_major, uint32_t capability_minor)
{
    if (capability_minor > 9u) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_INVALID_VALUE);
    }

    if (capability_major == 8u ||
        capability_major == 9u ||
        capability_major == 10u ||
        capability_major == 11u ||
        capability_major == 12u) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_SUCCESS);
    }

    _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_INVALID_VALUE);
}

static CusrSassInstruction
cusr_ast_sass_mov(uint8_t dst, uint8_t src, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)src << 32) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_MOV;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_MOV);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fadd_reg(uint8_t dst, uint8_t lhs, uint8_t rhs, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FADD_REG;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, 0u);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fadd_imm(uint8_t dst, uint8_t src, uint32_t imm_bits, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)imm_bits << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FADD_IMM;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_FADD_IMM);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fmul_reg(uint8_t dst, uint8_t lhs, uint8_t rhs, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FMUL_REG;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_FMUL);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fmul_imm(uint8_t dst, uint8_t src, uint32_t imm_bits, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)imm_bits << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FMUL_IMM;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_FMUL);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fmul_imm_rz(uint8_t dst, uint8_t src, uint32_t imm_bits, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)imm_bits << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FMUL_IMM;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_FMUL_RZ);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fadd_abs(uint8_t dst, uint8_t src, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)0x800000ffu << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FADD_REG;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_FADD_ABS);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_fmnmx_reg(uint8_t dst, uint8_t lhs, uint8_t rhs, uint64_t rest, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FMNMX_REG;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, rest);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_ffma_reg(uint8_t dst, uint8_t lhs, uint8_t rhs, uint8_t addend, uint32_t wait_mask)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_FFMA_REG;
    instruction.word1 = cusr_ast_sass_alu_control(wait_mask, CUSR_AST_SASS_REST_FFMA_BASE | (uint64_t)addend);
    return instruction;
}

static CusrSassInstruction
cusr_ast_sass_mufu(uint8_t dst, uint8_t src, uint32_t wait_mask, uint32_t barrier_slot, uint64_t rest)
{
    CusrSassInstruction instruction;
    instruction.word0 =
        ((uint64_t)src << 32) |
        ((uint64_t)dst << 16) |
        CUSR_AST_SASS_OPCODE_MUFU;
    instruction.word1 = cusr_ast_sass_barrier_control(wait_mask, barrier_slot, rest);
    return instruction;
}

static CusrAstSassResult
cusr_ast_sass_emit(CusrAstSassAssembler* assembler, CusrSassInstruction instruction)
{
    if (assembler->count >= assembler->capacity) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_INSUFFICIENT_INSTRUCTIONS);
    }

    if (assembler->pending_wait_mask != 0u) {
        instruction.word1 |= ((uint64_t)assembler->pending_wait_mask & 0xfffull) << 52;
        assembler->pending_wait_mask = 0u;
    }

    assembler->sass[assembler->count++] = instruction;
    return CUSR_AST_SASS_SUCCESS;
}

static uint32_t
cusr_ast_sass_wait_mask_for(CusrAstSassAssembler* assembler, const CusrAstOperand* operands, size_t num_operands)
{
    uint32_t wait_mask = 0u;
    size_t i;

    for (i = 0u; i < num_operands; ++i) {
        const CusrAstOperand* operand = operands + i;

        if (operand->barrier_active &&
            operand->barrier_slot < CUSR_AST_SASS_NUM_BARRIER_SLOTS &&
            assembler->barrier_slot_active[operand->barrier_slot] &&
            assembler->barrier_slot_token[operand->barrier_slot] == operand->barrier_token) {
            wait_mask |= 1u << operand->barrier_slot;
            assembler->barrier_slot_active[operand->barrier_slot] = 0u;
            assembler->barrier_slot_token[operand->barrier_slot] = 0u;
        }
    }

    return wait_mask;
}

static uint32_t
cusr_ast_sass_active_barrier_mask(const CusrAstSassAssembler* assembler)
{
    uint32_t mask = 0u;
    size_t i;

    for (i = 0u; i < CUSR_AST_SASS_NUM_BARRIER_SLOTS; ++i) {
        if (assembler->barrier_slot_active[i]) {
            mask |= 1u << i;
        }
    }

    return mask;
}

static CusrAstSassResult
cusr_ast_sass_allocate_barrier(CusrAstSassAssembler* assembler, uint8_t* barrier_slot_ret, uint32_t* barrier_token_ret)
{
    size_t i;

    for (i = 0u; i < CUSR_AST_SASS_NUM_BARRIER_SLOTS; ++i) {
        if (!assembler->barrier_slot_active[i]) {
            assembler->barrier_slot_active[i] = 1u;
            assembler->barrier_slot_token[i] = assembler->next_barrier_token++;
            *barrier_slot_ret = (uint8_t)i;
            *barrier_token_ret = assembler->barrier_slot_token[i];
            return CUSR_AST_SASS_SUCCESS;
        }
    }

    {
        const uint32_t wait_mask = cusr_ast_sass_active_barrier_mask(assembler);
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(assembler, cusr_ast_sass_nop(wait_mask)));
        memset(assembler->barrier_slot_active, 0, sizeof(assembler->barrier_slot_active));
        memset(assembler->barrier_slot_token, 0, sizeof(assembler->barrier_slot_token));
    }

    assembler->barrier_slot_active[0] = 1u;
    assembler->barrier_slot_token[0] = assembler->next_barrier_token++;
    *barrier_slot_ret = 0u;
    *barrier_token_ret = assembler->barrier_slot_token[0];
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_push(CusrAstSassAssembler* assembler, CusrAstOperand operand)
{
    if (assembler->stack_size >= CUSR_AST_SASS_STACK_DEPTH) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_STACK_OVERFLOW);
    }

    assembler->stack[assembler->stack_size++] = operand;
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_pop(CusrAstSassAssembler* assembler, CusrAstOperand* operand_ret)
{
    if (assembler->stack_size == 0u) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_STACK_UNDERFLOW);
    }

    assembler->stack_size -= 1u;
    *operand_ret = assembler->stack[assembler->stack_size];
    return CUSR_AST_SASS_SUCCESS;
}

static void
cusr_ast_sass_reserve_register(CusrAstSassAssembler* assembler, uint8_t reg)
{
    assembler->reserved_registers[reg] = 1u;
}

static int
cusr_ast_sass_register_is_reserved(const CusrAstSassAssembler* assembler, uint8_t reg)
{
    return reg == CUSR_AST_SASS_REGISTER_RZ || assembler->reserved_registers[reg] != 0u;
}

static void
cusr_ast_sass_prepare_free_registers(
    CusrAstSassAssembler* assembler,
    const uint8_t* available_registers,
    size_t num_available_registers)
{
    size_t i;

    for (i = 0u; i < num_available_registers && assembler->num_free_registers < CUSR_AST_SASS_MAX_FREE_REGISTERS; ++i) {
        uint8_t reg = available_registers[i];
        size_t j;
        int seen = 0;

        if (cusr_ast_sass_register_is_reserved(assembler, reg)) {
            continue;
        }

        for (j = 0u; j < assembler->num_free_registers; ++j) {
            if (assembler->free_registers[j] == reg) {
                seen = 1;
                break;
            }
        }

        if (!seen) {
            assembler->free_registers[assembler->num_free_registers++] = reg;
        }
    }
}

static CusrAstSassResult
cusr_ast_sass_allocate_register(CusrAstSassAssembler* assembler, CusrAstOperand* operand_ret)
{
    CusrAstOperand operand;
    uint32_t reg;

    memset(&operand, 0, sizeof(operand));
    operand.owned = 1u;

    if (assembler->num_free_registers != 0u) {
        assembler->num_free_registers -= 1u;
        reg = assembler->free_registers[assembler->num_free_registers];
    } else {
        reg = assembler->next_register;
        if (reg >= CUSR_AST_SASS_REGISTER_RZ) {
            _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_REGISTER_OVERFLOW);
        }
        assembler->next_register += 1u;
        if (assembler->next_register > assembler->high_water_register) {
            assembler->high_water_register = assembler->next_register;
        }
    }

    operand.reg = (uint8_t)reg;
    *operand_ret = operand;
    return CUSR_AST_SASS_SUCCESS;
}

static void
cusr_ast_sass_release_register(CusrAstSassAssembler* assembler, CusrAstOperand operand)
{
    if (operand.owned &&
        !operand.is_immediate &&
        !cusr_ast_sass_register_is_reserved(assembler, operand.reg) &&
        assembler->num_free_registers < CUSR_AST_SASS_MAX_FREE_REGISTERS) {
        assembler->free_registers[assembler->num_free_registers++] = operand.reg;
    }
}

static void
cusr_ast_sass_reclaim_routine_args(
    CusrAstSassAssembler* assembler,
    const CusrAstOperand* args,
    size_t num_args,
    CusrAstOperand* result)
{
    size_t i;

    for (i = 0u; i < num_args; ++i) {
        size_t j;
        int duplicate = 0;

        if (!args[i].owned || args[i].is_immediate) {
            continue;
        }

        for (j = 0u; j < i; ++j) {
            if (args[j].owned &&
                !args[j].is_immediate &&
                args[j].reg == args[i].reg) {
                duplicate = 1;
                break;
            }
        }

        if (duplicate) {
            continue;
        }

        if (!result->is_immediate && result->reg == args[i].reg) {
            result->owned = 1u;
        } else {
            assembler->pending_wait_mask |= cusr_ast_sass_wait_mask_for(assembler, args + i, 1u);
            cusr_ast_sass_release_register(assembler, args[i]);
        }
    }
}

static CusrAstSassResult
cusr_ast_sass_ensure_register(CusrAstSassAssembler* assembler, CusrAstOperand* operand)
{
    CusrAstOperand dst;

    if (!operand->is_immediate) {
        return CUSR_AST_SASS_SUCCESS;
    }

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &dst));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fadd_imm(dst.reg, CUSR_AST_SASS_REGISTER_RZ, operand->immediate_bits, 0u)
    ));

    *operand = dst;
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstOperand
cusr_ast_sass_choose_output(CusrAstSassAssembler* assembler, const CusrAstOperand* lhs, const CusrAstOperand* rhs)
{
    (void)assembler;
    if (lhs->owned && !lhs->is_immediate) {
        return *lhs;
    }
    if (rhs->owned && !rhs->is_immediate) {
        return *rhs;
    }

    return *lhs;
}

static CusrAstSassResult
cusr_ast_sass_choose_or_allocate_output(
    CusrAstSassAssembler* assembler,
    const CusrAstOperand* lhs,
    const CusrAstOperand* rhs,
    CusrAstOperand* dst_ret)
{
    CusrAstOperand dst = cusr_ast_sass_choose_output(assembler, lhs, rhs);

    if (dst.owned && !dst.is_immediate) {
        *dst_ret = dst;
        return CUSR_AST_SASS_SUCCESS;
    }

    return cusr_ast_sass_allocate_register(assembler, dst_ret);
}

static void
cusr_ast_sass_release_consumed(
    CusrAstSassAssembler* assembler,
    CusrAstOperand lhs,
    CusrAstOperand rhs,
    CusrAstOperand keep)
{
    if (lhs.owned && !lhs.is_immediate && lhs.reg != keep.reg) {
        cusr_ast_sass_release_register(assembler, lhs);
    }
    if (rhs.owned && !rhs.is_immediate && rhs.reg != keep.reg) {
        cusr_ast_sass_release_register(assembler, rhs);
    }
}

static void
cusr_ast_sass_release_consumed3(
    CusrAstSassAssembler* assembler,
    CusrAstOperand a,
    CusrAstOperand b,
    CusrAstOperand c,
    CusrAstOperand keep)
{
    if (a.owned && !a.is_immediate && a.reg != keep.reg) {
        cusr_ast_sass_release_register(assembler, a);
    }
    if (b.owned && !b.is_immediate && b.reg != keep.reg) {
        cusr_ast_sass_release_register(assembler, b);
    }
    if (c.owned && !c.is_immediate && c.reg != keep.reg) {
        cusr_ast_sass_release_register(assembler, c);
    }
}

static CusrAstSassResult
cusr_ast_sass_emit_binary_alu(CusrAstSassAssembler* assembler, CusrAstOp op)
{
    CusrAstOperand rhs;
    CusrAstOperand lhs;
    CusrAstOperand dst;
    CusrAstOperand wait_operands[2];
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &lhs));

    if (op == CUSR_AST_OP_ADD || op == CUSR_AST_OP_MUL) {
        if (lhs.is_immediate && !rhs.is_immediate) {
            CusrAstOperand swap = lhs;
            lhs = rhs;
            rhs = swap;
        } else if (lhs.is_immediate && rhs.is_immediate) {
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &lhs));
        }

        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_choose_or_allocate_output(assembler, &lhs, &rhs, &dst));
        wait_operands[0] = lhs;
        wait_operands[1] = rhs;
        wait_mask = cusr_ast_sass_wait_mask_for(assembler, wait_operands, 2u);

        if (op == CUSR_AST_OP_ADD) {
            if (rhs.is_immediate) {
                _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
                    assembler,
                    cusr_ast_sass_fadd_imm(dst.reg, lhs.reg, rhs.immediate_bits, wait_mask)
                ));
            } else {
                _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
                    assembler,
                    cusr_ast_sass_fadd_reg(dst.reg, lhs.reg, rhs.reg, wait_mask)
                ));
            }
        } else {
            if (rhs.is_immediate) {
                _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
                    assembler,
                    cusr_ast_sass_fmul_imm(dst.reg, lhs.reg, rhs.immediate_bits, wait_mask)
                ));
            } else {
                _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
                    assembler,
                    cusr_ast_sass_fmul_reg(dst.reg, lhs.reg, rhs.reg, wait_mask)
                ));
            }
        }

        dst.barrier_active = 0u;
        cusr_ast_sass_release_consumed(assembler, lhs, rhs, dst);
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
        return CUSR_AST_SASS_SUCCESS;
    }

    _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_UNSUPPORTED_OP);
}

static CusrAstSassResult
cusr_ast_sass_emit_minmax(CusrAstSassAssembler* assembler, uint64_t rest)
{
    CusrAstOperand rhs;
    CusrAstOperand lhs;
    CusrAstOperand dst;
    CusrAstOperand wait_operands[2];
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &lhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &lhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_choose_or_allocate_output(assembler, &lhs, &rhs, &dst));

    wait_operands[0] = lhs;
    wait_operands[1] = rhs;
    wait_mask = cusr_ast_sass_wait_mask_for(assembler, wait_operands, 2u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fmnmx_reg(dst.reg, lhs.reg, rhs.reg, rest, wait_mask)
    ));

    dst.barrier_active = 0u;
    cusr_ast_sass_release_consumed(assembler, lhs, rhs, dst);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_fma(CusrAstSassAssembler* assembler)
{
    CusrAstOperand addend;
    CusrAstOperand rhs;
    CusrAstOperand lhs;
    CusrAstOperand dst;
    CusrAstOperand wait_operands[3];
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &addend));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &lhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &lhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &addend));

    if (lhs.owned && !lhs.is_immediate) {
        dst = lhs;
    } else if (rhs.owned && !rhs.is_immediate) {
        dst = rhs;
    } else if (addend.owned && !addend.is_immediate) {
        dst = addend;
    } else {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &dst));
    }

    wait_operands[0] = lhs;
    wait_operands[1] = rhs;
    wait_operands[2] = addend;
    wait_mask = cusr_ast_sass_wait_mask_for(assembler, wait_operands, 3u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_ffma_reg(dst.reg, lhs.reg, rhs.reg, addend.reg, wait_mask)
    ));

    dst.barrier_active = 0u;
    cusr_ast_sass_release_consumed3(assembler, lhs, rhs, addend, dst);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_neg(CusrAstSassAssembler* assembler)
{
    CusrAstOperand value;
    CusrAstOperand dst;
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &value));

    if (value.is_immediate) {
        value.immediate_bits ^= 0x80000000u;
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, value));
        return CUSR_AST_SASS_SUCCESS;
    }

    dst = value;
    if (!dst.owned) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &dst));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &value, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fmul_imm(dst.reg, value.reg, CUSR_AST_SASS_NEG_ONE_BITS, wait_mask)
    ));

    dst.barrier_active = 0u;
    if (value.owned && value.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, value);
    }
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_abs(CusrAstSassAssembler* assembler)
{
    CusrAstOperand value;
    CusrAstOperand dst;
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &value));

    if (value.is_immediate) {
        value.immediate_bits &= 0x7fffffffu;
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, value));
        return CUSR_AST_SASS_SUCCESS;
    }

    dst = value;
    if (!dst.owned) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &dst));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &value, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fadd_abs(dst.reg, value.reg, wait_mask)
    ));

    dst.barrier_active = 0u;
    if (value.owned && value.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, value);
    }
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_sub(CusrAstSassAssembler* assembler)
{
    CusrAstOperand rhs;
    CusrAstOperand lhs;
    CusrAstOperand neg_rhs;
    CusrAstOperand dst;
    CusrAstOperand wait_operands[2];
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &lhs));

    if (rhs.is_immediate) {
        rhs.immediate_bits ^= 0x80000000u;
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, lhs));
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, rhs));
        return cusr_ast_sass_emit_binary_alu(assembler, CUSR_AST_OP_ADD);
    }

    neg_rhs = rhs;
    if (!neg_rhs.owned) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &neg_rhs));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &rhs, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fmul_imm(neg_rhs.reg, rhs.reg, CUSR_AST_SASS_NEG_ONE_BITS, wait_mask)
    ));
    neg_rhs.barrier_active = 0u;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_choose_or_allocate_output(assembler, &lhs, &neg_rhs, &dst));
    wait_operands[0] = lhs;
    wait_operands[1] = neg_rhs;
    wait_mask = cusr_ast_sass_wait_mask_for(assembler, wait_operands, 2u);

    if (lhs.is_immediate) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &lhs));
    }

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fadd_reg(dst.reg, lhs.reg, neg_rhs.reg, wait_mask)
    ));
    dst.barrier_active = 0u;

    if (rhs.owned && rhs.reg != neg_rhs.reg && rhs.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, rhs);
    }
    if (lhs.owned && !lhs.is_immediate && lhs.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, lhs);
    }
    if (neg_rhs.owned && neg_rhs.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, neg_rhs);
    }

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_mufu_unary(CusrAstSassAssembler* assembler, uint64_t rest)
{
    CusrAstOperand value;
    CusrAstOperand dst;
    uint8_t barrier_slot = 0u;
    uint32_t barrier_token = 0u;
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &value));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &value));

    dst = value;
    if (!dst.owned) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &dst));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &value, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_barrier(assembler, &barrier_slot, &barrier_token));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_mufu(dst.reg, value.reg, wait_mask, barrier_slot, rest)
    ));

    dst.barrier_active = 1u;
    dst.barrier_slot = barrier_slot;
    dst.barrier_token = barrier_token;

    if (value.owned && value.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, value);
    }
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_sincos(CusrAstSassAssembler* assembler, uint64_t rest)
{
    CusrAstOperand value;
    CusrAstOperand dst;
    uint8_t barrier_slot = 0u;
    uint32_t barrier_token = 0u;
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &value));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &value));

    dst = value;
    if (!dst.owned) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &dst));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &value, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fmul_imm_rz(dst.reg, value.reg, CUSR_AST_SASS_SIN_COS_SCALE_BITS, wait_mask)
    ));

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_barrier(assembler, &barrier_slot, &barrier_token));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_mufu(dst.reg, dst.reg, 0u, barrier_slot, rest)
    ));

    dst.barrier_active = 1u;
    dst.barrier_slot = barrier_slot;
    dst.barrier_token = barrier_token;

    if (value.owned && value.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, value);
    }
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_emit_div(CusrAstSassAssembler* assembler)
{
    CusrAstOperand rhs;
    CusrAstOperand lhs;
    CusrAstOperand recip;
    CusrAstOperand dst;
    CusrAstOperand wait_operands[2];
    uint8_t barrier_slot = 0u;
    uint32_t barrier_token = 0u;
    uint32_t wait_mask;

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &rhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &lhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &lhs));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_ensure_register(assembler, &rhs));

    recip = rhs;
    if (!recip.owned) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_register(assembler, &recip));
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &rhs, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_allocate_barrier(assembler, &barrier_slot, &barrier_token));
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_mufu(recip.reg, rhs.reg, wait_mask, barrier_slot, CUSR_AST_SASS_REST_MUFU_RCP)
    ));
    recip.barrier_active = 1u;
    recip.barrier_slot = barrier_slot;
    recip.barrier_token = barrier_token;

    dst = lhs.owned ? lhs : recip;
    wait_operands[0] = lhs;
    wait_operands[1] = recip;
    wait_mask = cusr_ast_sass_wait_mask_for(assembler, wait_operands, 2u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_fmul_reg(dst.reg, lhs.reg, recip.reg, wait_mask)
    ));
    dst.barrier_active = 0u;

    if (lhs.owned && lhs.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, lhs);
    }
    if (rhs.owned && rhs.reg != recip.reg && rhs.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, rhs);
    }
    if (recip.owned && recip.reg != dst.reg) {
        cusr_ast_sass_release_register(assembler, recip);
    }

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, dst));
    return CUSR_AST_SASS_SUCCESS;
}

static uint16_t
cusr_ast_sass_op_num_args(CusrAstOp op)
{
    switch (op) {
        case CUSR_AST_OP_ADD:
        case CUSR_AST_OP_SUB:
        case CUSR_AST_OP_MUL:
        case CUSR_AST_OP_DIV:
        case CUSR_AST_OP_MIN:
        case CUSR_AST_OP_MAX:
            return 2u;
        case CUSR_AST_OP_FMA:
            return 3u;
        case CUSR_AST_OP_NEG:
        case CUSR_AST_OP_SQRT:
        case CUSR_AST_OP_RCP:
        case CUSR_AST_OP_ABS:
        case CUSR_AST_OP_SIN:
        case CUSR_AST_OP_COS:
        case CUSR_AST_OP_EX2:
        case CUSR_AST_OP_LG2:
        case CUSR_AST_OP_RSQRT:
        case CUSR_AST_OP_TANH:
            return 1u;
        case CUSR_AST_OP_NONE:
        case CUSR_AST_OP_NUM_ENUMS:
        default:
            return 0u;
    }
}

static CusrAstSassResult
cusr_ast_sass_emit_op(CusrAstSassAssembler* assembler, CusrAstOp op)
{
    switch (op) {
        case CUSR_AST_OP_ADD:
        case CUSR_AST_OP_MUL:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_binary_alu(assembler, op));
            break;
        case CUSR_AST_OP_MIN:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_minmax(assembler, CUSR_AST_SASS_REST_FMNMX_MIN));
            break;
        case CUSR_AST_OP_MAX:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_minmax(assembler, CUSR_AST_SASS_REST_FMNMX_MAX));
            break;
        case CUSR_AST_OP_FMA:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_fma(assembler));
            break;
        case CUSR_AST_OP_SUB:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_sub(assembler));
            break;
        case CUSR_AST_OP_DIV:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_div(assembler));
            break;
        case CUSR_AST_OP_NEG:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_neg(assembler));
            break;
        case CUSR_AST_OP_ABS:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_abs(assembler));
            break;
        case CUSR_AST_OP_SQRT:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_mufu_unary(assembler, CUSR_AST_SASS_REST_MUFU_SQRT));
            break;
        case CUSR_AST_OP_RCP:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_mufu_unary(assembler, CUSR_AST_SASS_REST_MUFU_RCP));
            break;
        case CUSR_AST_OP_SIN:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_sincos(assembler, CUSR_AST_SASS_REST_MUFU_SIN));
            break;
        case CUSR_AST_OP_COS:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_sincos(assembler, CUSR_AST_SASS_REST_MUFU_COS));
            break;
        case CUSR_AST_OP_EX2:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_mufu_unary(assembler, CUSR_AST_SASS_REST_MUFU_EX2));
            break;
        case CUSR_AST_OP_LG2:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_mufu_unary(assembler, CUSR_AST_SASS_REST_MUFU_LG2));
            break;
        case CUSR_AST_OP_RSQRT:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_mufu_unary(assembler, CUSR_AST_SASS_REST_MUFU_RSQ));
            break;
        case CUSR_AST_OP_TANH:
            _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_mufu_unary(assembler, CUSR_AST_SASS_REST_MUFU_TANH));
            break;
        case CUSR_AST_OP_NONE:
        case CUSR_AST_OP_NUM_ENUMS:
        default:
            _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_UNSUPPORTED_OP);
    }

    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_finish(CusrAstSassAssembler* assembler)
{
    CusrAstOperand root;
    uint32_t wait_mask;

    if (assembler->stack_size != 1u) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_BAD_PROGRAM);
    }

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_pop(assembler, &root));

    if (root.is_immediate) {
        _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
            assembler,
            cusr_ast_sass_fadd_imm(assembler->result_register, CUSR_AST_SASS_REGISTER_RZ, root.immediate_bits, 0u)
        ));
        return CUSR_AST_SASS_SUCCESS;
    }

    wait_mask = cusr_ast_sass_wait_mask_for(assembler, &root, 1u);
    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit(
        assembler,
        cusr_ast_sass_mov(assembler->result_register, root.reg, wait_mask)
    ));

    if (root.owned && root.reg != assembler->result_register) {
        cusr_ast_sass_release_register(assembler, root);
    }

    return CUSR_AST_SASS_SUCCESS;
}

static CusrAstSassResult
cusr_ast_sass_compile_frame(
    CusrAstSassAssembler* assembler,
    const uint8_t* input_registers,
    size_t num_input_registers,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* instructions,
    const CusrAstOperand* routine_args,
    size_t num_routine_args,
    size_t frame_depth)
{
    const size_t base_stack_size = assembler->stack_size;
    size_t instruction_idx;

    for (instruction_idx = 0u; instruction_idx < CUSR_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
        const CusrAstInstruction instruction = instructions[instruction_idx];

        switch ((CusrAstInstructionType)instruction.instruction_type) {
            case CUSR_AST_INSTRUCTION_TYPE_INPUT:
                if (instruction.payload.idx >= num_input_registers) {
                    _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_BAD_PROGRAM);
                }
                {
                    CusrAstOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    operand.reg = input_registers[instruction.payload.idx];
                    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, operand));
                }
                break;

            case CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS:
                {
                    CusrAstOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    operand.is_immediate = 1u;
                    operand.immediate_bits = instruction.payload.bits;
                    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, operand));
                }
                break;

            case CUSR_AST_INSTRUCTION_TYPE_ROUTINE_ARG:
                if (routine_args == NULL || instruction.payload.idx >= num_routine_args) {
                    _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                }
                _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_push(assembler, routine_args[instruction.payload.idx]));
                break;

            case CUSR_AST_INSTRUCTION_TYPE_OP:
                {
                    const CusrAstOp op = (CusrAstOp)instruction.payload.op;
                    const uint16_t expected_args = cusr_ast_sass_op_num_args(op);

                    if (instruction.aux > CUSR_AST_MAX_INSTRUCTION_ARGS) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_TOO_MANY_ARGS);
                    }

                    if (expected_args == 0u) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_UNSUPPORTED_OP);
                    }

                    if (instruction.aux != expected_args) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_INVALID_VALUE);
                    }

                    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_emit_op(assembler, op));
                }
                break;

            case CUSR_AST_INSTRUCTION_TYPE_ROUTINE:
                {
                    const CusrAstIdx routine_idx = instruction.payload.idx;
                    const size_t num_args = instruction.aux;
                    const size_t next_frame_depth = frame_depth + 1u;
                    CusrAstOperand caller_args[CUSR_AST_MAX_INSTRUCTION_ARGS];
                    CusrAstOperand next_routine_args[CUSR_AST_MAX_INSTRUCTION_ARGS];
                    size_t arg_start_idx;
                    size_t arg_idx;

                    if (instruction.aux > CUSR_AST_MAX_INSTRUCTION_ARGS) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_TOO_MANY_ARGS);
                    }

                    if (routines == NULL ||
                        routine_idx >= num_routines ||
                        routines[routine_idx] == NULL) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                    }

                    if (next_frame_depth > CUSR_AST_SASS_ROUTINE_DEPTH) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_ROUTINE_DEPTH_EXCEEDED);
                    }

                    if (assembler->stack_size < num_args) {
                        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_STACK_UNDERFLOW);
                    }

                    arg_start_idx = assembler->stack_size - num_args;
                    for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
                        caller_args[arg_idx] = assembler->stack[arg_start_idx + arg_idx];
                        next_routine_args[arg_idx] = caller_args[arg_idx];
                        next_routine_args[arg_idx].owned = 0u;
                    }

                    assembler->stack_size = arg_start_idx;

                    _CUSR_AST_SASS_CHECK_RET(
                        cusr_ast_sass_compile_frame(
                            assembler,
                            input_registers,
                            num_input_registers,
                            routines,
                            num_routines,
                            routines[routine_idx],
                            next_routine_args,
                            num_args,
                            next_frame_depth
                        )
                    );

                    cusr_ast_sass_reclaim_routine_args(
                        assembler,
                        caller_args,
                        num_args,
                        assembler->stack + arg_start_idx
                    );
                }
                break;

            case CUSR_AST_INSTRUCTION_TYPE_RETURN:
                if (assembler->stack_size != base_stack_size + 1u) {
                    _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_INVALID_VALUE);
                }

                return CUSR_AST_SASS_SUCCESS;

            case CUSR_AST_INSTRUCTION_TYPE_NONE:
            default:
                _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_BAD_PROGRAM);
        }
    }

    _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_BAD_PROGRAM);
}

CUSR_AST_SASS_PUBLIC_DEF
const char*
cusr_ast_sass_result_to_string(CusrAstSassResult result)
{
    switch (result) {
        case CUSR_AST_SASS_SUCCESS:
            return "CUSR_AST_SASS_SUCCESS";
        case CUSR_AST_SASS_ERROR_INVALID_VALUE:
            return "CUSR_AST_SASS_ERROR_INVALID_VALUE";
        case CUSR_AST_SASS_ERROR_STACK_OVERFLOW:
            return "CUSR_AST_SASS_ERROR_STACK_OVERFLOW";
        case CUSR_AST_SASS_ERROR_STACK_UNDERFLOW:
            return "CUSR_AST_SASS_ERROR_STACK_UNDERFLOW";
        case CUSR_AST_SASS_ERROR_BAD_PROGRAM:
            return "CUSR_AST_SASS_ERROR_BAD_PROGRAM";
        case CUSR_AST_SASS_ERROR_UNSUPPORTED_OP:
            return "CUSR_AST_SASS_ERROR_UNSUPPORTED_OP";
        case CUSR_AST_SASS_ERROR_INSUFFICIENT_INSTRUCTIONS:
            return "CUSR_AST_SASS_ERROR_INSUFFICIENT_INSTRUCTIONS";
        case CUSR_AST_SASS_ERROR_REGISTER_OVERFLOW:
            return "CUSR_AST_SASS_ERROR_REGISTER_OVERFLOW";
        case CUSR_AST_SASS_ERROR_BARRIER_OVERFLOW:
            return "CUSR_AST_SASS_ERROR_BARRIER_OVERFLOW";
        case CUSR_AST_SASS_ERROR_TOO_MANY_ARGS:
            return "CUSR_AST_SASS_ERROR_TOO_MANY_ARGS";
        case CUSR_AST_SASS_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS:
            return "CUSR_AST_SASS_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS";
        case CUSR_AST_SASS_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS:
            return "CUSR_AST_SASS_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS";
        case CUSR_AST_SASS_ERROR_ROUTINE_DEPTH_EXCEEDED:
            return "CUSR_AST_SASS_ERROR_ROUTINE_DEPTH_EXCEEDED";
    }

    return "CUSR_AST_SASS_ERROR_UNKNOWN";
}

CUSR_AST_SASS_PUBLIC_DEF
CusrAstSassResult
cusr_ast_sass_generate(
    const uint8_t* input_registers,
    size_t num_input_registers,
    const uint8_t* available_registers,
    size_t num_available_registers,
    uint8_t result_register,
    uint32_t incoming_wait_mask,
    uint32_t first_new_register,
    uint32_t capability_major,
    uint32_t capability_minor,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* instructions,
    CusrSassInstruction* sass,
    size_t sass_capacity,
    size_t* sass_count_ret,
    uint32_t* expanded_register_count_ret)
{
    CusrAstSassAssembler assembler;

    if (sass_count_ret != NULL) {
        *sass_count_ret = 0u;
    }
    if (expanded_register_count_ret != NULL) {
        *expanded_register_count_ret = first_new_register;
    }

    if (instructions == NULL ||
        input_registers == NULL ||
        num_input_registers == 0u ||
        sass == NULL ||
        sass_capacity == 0u ||
        first_new_register >= CUSR_AST_SASS_REGISTER_RZ ||
        result_register == CUSR_AST_SASS_REGISTER_RZ) {
        _CUSR_AST_SASS_ERROR_RET(CUSR_AST_SASS_ERROR_INVALID_VALUE);
    }

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_validate_capability(capability_major, capability_minor));

    memset(&assembler, 0, sizeof(assembler));
    assembler.sass = sass;
    assembler.capacity = sass_capacity;
    assembler.next_register = first_new_register;
    assembler.high_water_register = first_new_register;
    assembler.result_register = result_register;
    assembler.next_barrier_token = 1u;
    assembler.pending_wait_mask = incoming_wait_mask;

    cusr_ast_sass_reserve_register(&assembler, result_register);
    {
        size_t input_idx;
        for (input_idx = 0u; input_idx < num_input_registers; ++input_idx) {
            cusr_ast_sass_reserve_register(&assembler, input_registers[input_idx]);
        }
    }

    cusr_ast_sass_prepare_free_registers(
        &assembler,
        available_registers,
        num_available_registers
    );

    _CUSR_AST_SASS_CHECK_RET(
        cusr_ast_sass_compile_frame(
            &assembler,
            input_registers,
            num_input_registers,
            routines,
            num_routines,
            instructions,
            NULL,
            0u,
            0u
        )
    );

    _CUSR_AST_SASS_CHECK_RET(cusr_ast_sass_finish(&assembler));

    if (sass_count_ret != NULL) {
        *sass_count_ret = assembler.count;
    }
    if (expanded_register_count_ret != NULL) {
        *expanded_register_count_ret = assembler.high_water_register;
    }

    return CUSR_AST_SASS_SUCCESS;
}

static const char*
cusr_ast_sass_mufu_name(uint64_t word1)
{
    switch (word1 & 0xffffull) {
        case CUSR_AST_SASS_REST_MUFU_COS:
            return "MUFU.COS";
        case CUSR_AST_SASS_REST_MUFU_SIN:
            return "MUFU.SIN";
        case CUSR_AST_SASS_REST_MUFU_EX2:
            return "MUFU.EX2";
        case CUSR_AST_SASS_REST_MUFU_LG2:
            return "MUFU.LG2";
        case CUSR_AST_SASS_REST_MUFU_RCP:
            return "MUFU.RCP";
        case CUSR_AST_SASS_REST_MUFU_RSQ:
            return "MUFU.RSQ";
        case CUSR_AST_SASS_REST_MUFU_SQRT:
            return "MUFU.SQRT";
        case CUSR_AST_SASS_REST_MUFU_TANH:
            return "MUFU.TANH";
    }

    return "MUFU.?";
}

CUSR_AST_SASS_PUBLIC_DEF
void
cusr_ast_sass_print(FILE* stream, const CusrSassInstruction* sass, size_t num_instructions)
{
    size_t i;

    if (stream == NULL || sass == NULL) {
        return;
    }

    for (i = 0u; i < num_instructions; ++i) {
        const uint64_t word0 = sass[i].word0;
        const uint64_t word1 = sass[i].word1;
        const uint32_t opcode = (uint32_t)(word0 & 0xffffu);
        const uint32_t wait = (uint32_t)((word1 >> 52) & 0xfffu);
        const uint32_t ctrl = (uint32_t)((word1 >> 44) & 0xffu);
        const uint32_t stall = (uint32_t)((word1 >> 40) & 0xfu);
        const uint32_t dst = (uint32_t)((word0 >> 16) & 0xffu);
        const uint32_t src0 = (uint32_t)((word0 >> 24) & 0xffu);
        const uint32_t src1 = (uint32_t)((word0 >> 32) & 0xffu);
        const uint32_t imm = (uint32_t)((word0 >> 32) & 0xffffffffu);
        const uint64_t rest = word1 & 0xffffffffffull;

        fprintf(stream, "%04zu  word0=0x%016llx word1=0x%016llx wait=0x%03x ctrl=%02x stall=%x  ",
            i,
            (unsigned long long)word0,
            (unsigned long long)word1,
            wait,
            ctrl,
            stall);

        if (word0 == CUSR_AST_SASS_NOP_WORD0) {
            fprintf(stream, "NOP\n");
        } else if (opcode == CUSR_AST_SASS_OPCODE_MOV) {
            fprintf(stream, "MOV R%u, R%u\n", dst, src1);
        } else if (opcode == CUSR_AST_SASS_OPCODE_FMNMX_REG) {
            fprintf(
                stream,
                "FMNMX.FTZ R%u, R%u, R%u, %s\n",
                dst,
                src0,
                src1,
                rest == CUSR_AST_SASS_REST_FMNMX_MIN ? "PT" : "!PT"
            );
        } else if (opcode == CUSR_AST_SASS_OPCODE_FFMA_REG) {
            fprintf(stream, "FFMA.FTZ R%u, R%u, R%u, R%u\n", dst, src0, src1, (unsigned)(word1 & 0xffu));
        } else if (opcode == CUSR_AST_SASS_OPCODE_FADD_REG &&
                   ((uint32_t)(word0 >> 32)) == 0x800000ffu &&
                   rest == CUSR_AST_SASS_REST_FADD_ABS) {
            fprintf(stream, "FADD.FTZ R%u, |R%u|, -RZ\n", dst, src0);
        } else if (opcode == CUSR_AST_SASS_OPCODE_FADD_REG) {
            fprintf(stream, "FADD.FTZ R%u, R%u, R%u\n", dst, src0, src1);
        } else if (opcode == CUSR_AST_SASS_OPCODE_FMUL_REG) {
            fprintf(stream, "FMUL.FTZ R%u, R%u, R%u\n", dst, src0, src1);
        } else if (opcode == CUSR_AST_SASS_OPCODE_FADD_IMM) {
            fprintf(stream, "FADD.FTZ R%u, R%u, 0x%08x\n", dst, src0, imm);
        } else if (opcode == CUSR_AST_SASS_OPCODE_FMUL_IMM) {
            fprintf(stream, "%s R%u, R%u, 0x%08x\n", rest == CUSR_AST_SASS_REST_FMUL_RZ ? "FMUL.RZ" : "FMUL.FTZ", dst, src0, imm);
        } else if (opcode == CUSR_AST_SASS_OPCODE_MUFU) {
            fprintf(stream, "%s R%u, R%u\n", cusr_ast_sass_mufu_name(word1), dst, src1);
        } else if (opcode == CUSR_AST_SASS_OPCODE_BRA) {
            const uint64_t target_field =
                ((word0 >> 16) & 0xffull) |
                ((word0 >> 26) & ~0xffull);
            fprintf(stream, "BRA +%llu\n", (unsigned long long)((target_field + 4u) / 4u));
        } else {
            fprintf(stream, "UNKNOWN opcode=0x%04x\n", opcode);
        }
    }
}

#endif /* CUSR_AST_SASS_IMPLEMENTATION_ONCE */
#endif /* CUSR_AST_SASS_IMPLEMENTATION */
