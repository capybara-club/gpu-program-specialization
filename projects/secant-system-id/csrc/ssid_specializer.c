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
#include "ssid_internal.h"

#include <stdlib.h>
#include <string.h>

#define SSID_INSTRUCTION_BYTES 16u
#define SSID_REGISTER_RZ 255u
#define SSID_REGISTER_PAD 2u
#define SSID_MAX_STACK 128u
#define SSID_MAX_PACKED_GENOMES 128u
#define SSID_BARRIER_SLOTS 6u
#define SSID_WAIT_BARRIER_MASK ((1u << SSID_BARRIER_SLOTS) - 1u)

#define SSID_NONE 0x80u
#define SSID_CONSTANT_BITS_F32 0x81u
#define SSID_RETURN_F32 0x83u
#define SSID_ADD_F32 0x85u
#define SSID_SUB_F32 0x86u
#define SSID_MUL_F32 0x87u
#define SSID_DIV_F32 0x88u
#define SSID_NEG_F32 0x89u
#define SSID_SQRT_F32 0x8au
#define SSID_RCP_F32 0x8bu
#define SSID_ABS_F32 0x8cu
#define SSID_MIN_F32 0x8du
#define SSID_MAX_F32 0x8eu
#define SSID_FMA_F32 0x8fu
#define SSID_SIN_F32 0x90u
#define SSID_COS_F32 0x91u
#define SSID_EX2_F32 0x92u
#define SSID_LG2_F32 0x93u
#define SSID_RSQRT_F32 0x94u
#define SSID_TANH_F32 0x95u
#define SSID_STATIC_COLUMN_INPUT_F32 0xb6u
#define SSID_EXP_F32 0xbau
#define SSID_LOG_F32 0xbbu

#define SSID_NOP_WORD0 UINT64_C(0x0000000000007918)
#define SSID_NOP_WORD1 UINT64_C(0x000fc00000000000)
#define SSID_OPCODE_MOV UINT64_C(0x7202)
#define SSID_OPCODE_FMNMX_REG UINT64_C(0x7209)
#define SSID_OPCODE_FMUL_REG UINT64_C(0x7220)
#define SSID_OPCODE_FADD_REG UINT64_C(0x7221)
#define SSID_OPCODE_FFMA_REG UINT64_C(0x7223)
#define SSID_OPCODE_FADD_IMM UINT64_C(0x7421)
#define SSID_OPCODE_FMUL_IMM UINT64_C(0x7820)
#define SSID_OPCODE_MUFU UINT64_C(0x7308)
#define SSID_OPCODE_BRA UINT64_C(0x7947)

#define SSID_CONTROL_ALU_HI UINT64_C(0xfc)
#define SSID_CONTROL_ALU_STALL UINT64_C(0xc)
#define SSID_CONTROL_BARRIER_BASE UINT64_C(0xe2)
#define SSID_CONTROL_MUFU_STALL UINT64_C(0x6)
#define SSID_REST_FADD_IMM UINT64_C(0x0000010000)
#define SSID_REST_FADD_ABS UINT64_C(0x0000010200)
#define SSID_REST_FMUL UINT64_C(0x0000410000)
#define SSID_REST_FMUL_RZ UINT64_C(0x000040c000)
#define SSID_REST_FMNMX_MIN UINT64_C(0x0003810000)
#define SSID_REST_FMNMX_MAX UINT64_C(0x0007810000)
#define SSID_REST_FFMA UINT64_C(0x0000010000)
#define SSID_REST_MOV UINT64_C(0x0000000f00)
#define SSID_REST_MUFU_COS UINT64_C(0x0000000000)
#define SSID_REST_MUFU_SIN UINT64_C(0x0000000400)
#define SSID_REST_MUFU_EX2 UINT64_C(0x0000000800)
#define SSID_REST_MUFU_LG2 UINT64_C(0x0000000c00)
#define SSID_REST_MUFU_RCP UINT64_C(0x0000001000)
#define SSID_REST_MUFU_RSQ UINT64_C(0x0000001400)
#define SSID_REST_MUFU_SQRT UINT64_C(0x0000002000)
#define SSID_REST_MUFU_TANH UINT64_C(0x0000002400)
#define SSID_REST_BRA UINT64_C(0x0003800000)

#define SSID_NEG_ONE_BITS UINT32_C(0xbf800000)
#define SSID_SIN_COS_SCALE_BITS UINT32_C(0x3e22f983)
#define SSID_LN_TWO_BITS UINT32_C(0x3f317218)
#define SSID_LOG2_E_BITS UINT32_C(0x3fb8aa3b)

typedef struct ssid_instruction {
    uint64_t word0;
    uint64_t word1;
} ssid_instruction;

typedef struct ssid_operand {
    uint32_t immediate_bits;
    uint64_t barrier_token;
    uint8_t reg;
    int8_t barrier_slot;
    uint8_t immediate;
    uint8_t owned;
} ssid_operand;

typedef struct ssid_assembler {
    uint8_t *output;
    uint32_t capacity;
    uint32_t count;
    ssid_operand stack[SSID_MAX_STACK];
    uint32_t stack_count;
    uint8_t free_registers[SSID_REGISTER_RZ];
    uint32_t free_count;
    uint32_t next_register;
    uint32_t high_water_register;
    uint8_t result_register;
    uint32_t pending_wait_mask;
    uint64_t active_barriers[SSID_BARRIER_SLOTS];
    uint64_t next_barrier_token;
} ssid_assembler;

static void ssid_store_u32(uint8_t *destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
    destination[2] = (uint8_t)(value >> 16);
    destination[3] = (uint8_t)(value >> 24);
}

static uint32_t ssid_load_u32(const uint8_t *source) {
    return (uint32_t)source[0] | ((uint32_t)source[1] << 8) | ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static void ssid_store_u64(uint8_t *destination, uint64_t value) {
    uint32_t index;
    for (index = 0; index < 8; ++index) destination[index] = (uint8_t)(value >> (8u * index));
}

static void ssid_store_instruction(uint8_t *destination, ssid_instruction instruction) {
    ssid_store_u64(destination, instruction.word0);
    ssid_store_u64(destination + 8, instruction.word1);
}

static uint64_t ssid_control(uint32_t wait_mask, uint64_t high, uint64_t stall, uint64_t rest) {
    return ((uint64_t)(wait_mask & SSID_WAIT_BARRIER_MASK) << 52) | ((high & UINT64_C(0xff)) << 44) | ((stall & UINT64_C(0xf)) << 40) | rest;
}

static uint64_t ssid_alu_control(uint32_t wait_mask, uint64_t rest) {
    return ssid_control(wait_mask, SSID_CONTROL_ALU_HI, SSID_CONTROL_ALU_STALL, rest);
}

static uint64_t ssid_barrier_control(uint32_t wait_mask, uint32_t slot, uint64_t rest) {
    return ssid_control(wait_mask, SSID_CONTROL_BARRIER_BASE + 4u * slot, SSID_CONTROL_MUFU_STALL, rest);
}

static ssid_instruction ssid_nop(uint32_t wait_mask) {
    ssid_instruction result = {SSID_NOP_WORD0, ((uint64_t)(wait_mask & SSID_WAIT_BARRIER_MASK) << 52) | SSID_NOP_WORD1};
    return result;
}

static ssid_instruction ssid_mov(uint32_t destination, uint32_t source, uint32_t wait_mask) {
    ssid_instruction result = {((uint64_t)source << 32) | ((uint64_t)destination << 16) | SSID_OPCODE_MOV, ssid_alu_control(wait_mask, SSID_REST_MOV)};
    return result;
}

static ssid_instruction ssid_fadd_reg(uint32_t destination, uint32_t lhs, uint32_t rhs, uint32_t wait_mask) {
    ssid_instruction result = {((uint64_t)rhs << 32) | ((uint64_t)lhs << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FADD_REG, ssid_alu_control(wait_mask, 0)};
    return result;
}

static ssid_instruction ssid_fadd_imm(uint32_t destination, uint32_t source, uint32_t bits, uint32_t wait_mask) {
    ssid_instruction result = {((uint64_t)bits << 32) | ((uint64_t)source << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FADD_IMM, ssid_alu_control(wait_mask, SSID_REST_FADD_IMM)};
    return result;
}

static ssid_instruction ssid_fmul_reg(uint32_t destination, uint32_t lhs, uint32_t rhs, uint32_t wait_mask) {
    ssid_instruction result = {((uint64_t)rhs << 32) | ((uint64_t)lhs << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FMUL_REG, ssid_alu_control(wait_mask, SSID_REST_FMUL)};
    return result;
}

static ssid_instruction ssid_fmul_imm(uint32_t destination, uint32_t source, uint32_t bits, uint32_t wait_mask, uint64_t rest) {
    ssid_instruction result = {((uint64_t)bits << 32) | ((uint64_t)source << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FMUL_IMM, ssid_alu_control(wait_mask, rest)};
    return result;
}

static ssid_instruction ssid_abs(uint32_t destination, uint32_t source, uint32_t wait_mask) {
    ssid_instruction result = {(UINT64_C(0x800000ff) << 32) | ((uint64_t)source << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FADD_REG, ssid_alu_control(wait_mask, SSID_REST_FADD_ABS)};
    return result;
}

static ssid_instruction ssid_fmnmx(uint32_t destination, uint32_t lhs, uint32_t rhs, uint64_t rest, uint32_t wait_mask) {
    ssid_instruction result = {((uint64_t)rhs << 32) | ((uint64_t)lhs << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FMNMX_REG, ssid_alu_control(wait_mask, rest)};
    return result;
}

static ssid_instruction ssid_ffma(uint32_t destination, uint32_t lhs, uint32_t rhs, uint32_t addend, uint32_t wait_mask) {
    ssid_instruction result = {((uint64_t)rhs << 32) | ((uint64_t)lhs << 24) | ((uint64_t)destination << 16) | SSID_OPCODE_FFMA_REG, ssid_alu_control(wait_mask, SSID_REST_FFMA | addend)};
    return result;
}

static ssid_instruction ssid_mufu(uint32_t destination, uint32_t source, uint32_t wait_mask, uint32_t slot, uint64_t rest) {
    ssid_instruction result = {((uint64_t)source << 32) | ((uint64_t)destination << 16) | SSID_OPCODE_MUFU, ssid_barrier_control(wait_mask, slot, rest)};
    return result;
}

static ssid_instruction ssid_branch(uint32_t remaining, uint32_t architecture) {
    uint64_t word0;
    if (architecture / 10u == 8u) {
        uint64_t target = 16u * (uint64_t)remaining - 16u;
        word0 = (target << 32) | SSID_OPCODE_BRA;
    } else {
        uint64_t target = 4u * (uint64_t)remaining - 4u;
        word0 = ((target & UINT64_C(0xff)) << 16) | ((target & ~UINT64_C(0xff)) << 26) | SSID_OPCODE_BRA;
    }
    {
        ssid_instruction result = {word0, ssid_alu_control(0, SSID_REST_BRA)};
        return result;
    }
}

static int ssid_emit(ssid_assembler *assembler, ssid_instruction instruction) {
    if (assembler->count >= assembler->capacity) {
        ssid_set_error("specialized expressions exhausted the compact SASS arena");
        return SSID_OUT_OF_RANGE;
    }
    if (assembler->pending_wait_mask != 0) {
        instruction.word1 |= (uint64_t)(assembler->pending_wait_mask & SSID_WAIT_BARRIER_MASK) << 52;
        assembler->pending_wait_mask = 0;
    }
    ssid_store_instruction(assembler->output + (size_t)assembler->count * SSID_INSTRUCTION_BYTES, instruction);
    assembler->count += 1;
    return SSID_OK;
}

static ssid_operand ssid_register_operand(uint32_t reg, int owned) {
    ssid_operand result;
    memset(&result, 0, sizeof(result));
    result.reg = (uint8_t)reg;
    result.owned = owned ? 1u : 0u;
    result.barrier_slot = -1;
    return result;
}

static ssid_operand ssid_immediate_operand(uint32_t bits) {
    ssid_operand result;
    memset(&result, 0, sizeof(result));
    result.immediate = 1;
    result.immediate_bits = bits;
    result.barrier_slot = -1;
    return result;
}

static int ssid_allocate(ssid_assembler *assembler, ssid_operand *result) {
    if (assembler->free_count != 0) {
        *result = ssid_register_operand(assembler->free_registers[--assembler->free_count], 1);
        return SSID_OK;
    }
    if (assembler->next_register >= SSID_REGISTER_RZ) {
        ssid_set_error("specialized expression exhausted the register file");
        return SSID_OUT_OF_RANGE;
    }
    *result = ssid_register_operand(assembler->next_register, 1);
    assembler->next_register += 1;
    if (assembler->next_register > assembler->high_water_register) assembler->high_water_register = assembler->next_register;
    return SSID_OK;
}

static uint32_t ssid_wait_mask(ssid_assembler *assembler, const ssid_operand *operands, uint32_t count) {
    uint32_t mask = 0;
    uint32_t index;
    for (index = 0; index < count; ++index) {
        int slot = operands[index].barrier_slot;
        if (slot >= 0 && slot < (int)SSID_BARRIER_SLOTS && assembler->active_barriers[slot] == operands[index].barrier_token) {
            mask |= 1u << slot;
            assembler->active_barriers[slot] = 0;
        }
    }
    return mask;
}

static void ssid_release(ssid_assembler *assembler, ssid_operand operand, const ssid_operand *keep) {
    uint32_t index;
    if (!operand.owned || operand.immediate) return;
    if (keep != NULL && operand.reg == keep->reg) return;
    assembler->pending_wait_mask |= ssid_wait_mask(assembler, &operand, 1);
    for (index = 0; index < assembler->free_count; ++index) if (assembler->free_registers[index] == operand.reg) return;
    assembler->free_registers[assembler->free_count++] = operand.reg;
}

static int ssid_ensure_register(ssid_assembler *assembler, ssid_operand input, ssid_operand *result) {
    int status;
    if (!input.immediate) {
        *result = input;
        return SSID_OK;
    }
    status = ssid_allocate(assembler, result);
    if (status != SSID_OK) return status;
    return ssid_emit(assembler, ssid_fadd_imm(result->reg, SSID_REGISTER_RZ, input.immediate_bits, 0));
}

static int ssid_stack_push(ssid_assembler *assembler, ssid_operand operand) {
    if (assembler->stack_count >= SSID_MAX_STACK) {
        ssid_set_error("postorder AST exceeds the C99 stack limit");
        return SSID_OUT_OF_RANGE;
    }
    assembler->stack[assembler->stack_count++] = operand;
    return SSID_OK;
}

static int ssid_stack_pop(ssid_assembler *assembler, ssid_operand *operand) {
    if (assembler->stack_count == 0) {
        ssid_set_error("postorder AST stack underflow");
        return SSID_INVALID_ARGUMENT;
    }
    *operand = assembler->stack[--assembler->stack_count];
    return SSID_OK;
}

static ssid_operand ssid_choose_output(ssid_assembler *assembler, const ssid_operand *operands, uint32_t count, int *status) {
    uint32_t index;
    ssid_operand result;
    for (index = 0; index < count; ++index) {
        if (operands[index].owned && !operands[index].immediate) {
            *status = SSID_OK;
            return operands[index];
        }
    }
    *status = ssid_allocate(assembler, &result);
    return result;
}

static int ssid_allocate_barrier(ssid_assembler *assembler, uint32_t *slot, uint64_t *token) {
    uint32_t index;
    for (index = 0; index < SSID_BARRIER_SLOTS; ++index) {
        if (assembler->active_barriers[index] == 0) {
            *slot = index;
            *token = assembler->next_barrier_token++;
            assembler->active_barriers[index] = *token;
            return SSID_OK;
        }
    }
    {
        uint32_t mask = 0;
        for (index = 0; index < SSID_BARRIER_SLOTS; ++index) mask |= 1u << index;
        if (ssid_emit(assembler, ssid_nop(mask)) != SSID_OK) return SSID_OUT_OF_RANGE;
        memset(assembler->active_barriers, 0, sizeof(assembler->active_barriers));
    }
    *slot = 0;
    *token = assembler->next_barrier_token++;
    assembler->active_barriers[0] = *token;
    return SSID_OK;
}

static int ssid_add_or_multiply(ssid_assembler *assembler, int multiply) {
    ssid_operand lhs, rhs, target, operands[2];
    uint32_t wait;
    int status;
    if ((status = ssid_stack_pop(assembler, &rhs)) != SSID_OK || (status = ssid_stack_pop(assembler, &lhs)) != SSID_OK) return status;
    if (lhs.immediate && !rhs.immediate) {
        ssid_operand swap = lhs;
        lhs = rhs;
        rhs = swap;
    } else if (lhs.immediate && rhs.immediate) {
        if ((status = ssid_ensure_register(assembler, lhs, &lhs)) != SSID_OK) return status;
    }
    operands[0] = lhs;
    operands[1] = rhs;
    target = ssid_choose_output(assembler, operands, 2, &status);
    if (status != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, operands, 2);
    if (multiply) status = ssid_emit(assembler, rhs.immediate ? ssid_fmul_imm(target.reg, lhs.reg, rhs.immediate_bits, wait, SSID_REST_FMUL) : ssid_fmul_reg(target.reg, lhs.reg, rhs.reg, wait));
    else status = ssid_emit(assembler, rhs.immediate ? ssid_fadd_imm(target.reg, lhs.reg, rhs.immediate_bits, wait) : ssid_fadd_reg(target.reg, lhs.reg, rhs.reg, wait));
    if (status != SSID_OK) return status;
    ssid_release(assembler, lhs, &target);
    ssid_release(assembler, rhs, &target);
    target.barrier_slot = -1;
    return ssid_stack_push(assembler, target);
}

static int ssid_subtract(ssid_assembler *assembler) {
    ssid_operand lhs, rhs, negated, target, operands[2];
    uint32_t wait;
    int status;
    if ((status = ssid_stack_pop(assembler, &rhs)) != SSID_OK || (status = ssid_stack_pop(assembler, &lhs)) != SSID_OK) return status;
    if (rhs.immediate) {
        rhs.immediate_bits ^= UINT32_C(0x80000000);
        if ((status = ssid_stack_push(assembler, lhs)) != SSID_OK || (status = ssid_stack_push(assembler, rhs)) != SSID_OK) return status;
        return ssid_add_or_multiply(assembler, 0);
    }
    if (rhs.owned) negated = rhs;
    else if ((status = ssid_allocate(assembler, &negated)) != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, &rhs, 1);
    if ((status = ssid_emit(assembler, ssid_fmul_imm(negated.reg, rhs.reg, SSID_NEG_ONE_BITS, wait, SSID_REST_FMUL))) != SSID_OK) return status;
    negated.barrier_slot = -1;
    if ((status = ssid_ensure_register(assembler, lhs, &lhs)) != SSID_OK) return status;
    operands[0] = lhs;
    operands[1] = negated;
    target = ssid_choose_output(assembler, operands, 2, &status);
    if (status != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, operands, 2);
    if ((status = ssid_emit(assembler, ssid_fadd_reg(target.reg, lhs.reg, negated.reg, wait))) != SSID_OK) return status;
    ssid_release(assembler, rhs, rhs.reg == negated.reg ? &target : NULL);
    ssid_release(assembler, lhs, &target);
    ssid_release(assembler, negated, &target);
    target.barrier_slot = -1;
    return ssid_stack_push(assembler, target);
}

static int ssid_divide(ssid_assembler *assembler) {
    ssid_operand lhs, rhs, reciprocal, target, operands[2];
    uint32_t wait, slot;
    uint64_t token;
    int status;
    if ((status = ssid_stack_pop(assembler, &rhs)) != SSID_OK || (status = ssid_ensure_register(assembler, rhs, &rhs)) != SSID_OK || (status = ssid_stack_pop(assembler, &lhs)) != SSID_OK || (status = ssid_ensure_register(assembler, lhs, &lhs)) != SSID_OK) return status;
    if (rhs.owned) reciprocal = rhs;
    else if ((status = ssid_allocate(assembler, &reciprocal)) != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, &rhs, 1);
    if ((status = ssid_allocate_barrier(assembler, &slot, &token)) != SSID_OK || (status = ssid_emit(assembler, ssid_mufu(reciprocal.reg, rhs.reg, wait, slot, SSID_REST_MUFU_RCP))) != SSID_OK) return status;
    reciprocal.barrier_slot = (int8_t)slot;
    reciprocal.barrier_token = token;
    target = lhs.owned ? lhs : reciprocal;
    operands[0] = lhs;
    operands[1] = reciprocal;
    wait = ssid_wait_mask(assembler, operands, 2);
    if ((status = ssid_emit(assembler, ssid_fmul_reg(target.reg, lhs.reg, reciprocal.reg, wait))) != SSID_OK) return status;
    ssid_release(assembler, lhs, &target);
    if (rhs.reg != reciprocal.reg) ssid_release(assembler, rhs, &target);
    ssid_release(assembler, reciprocal, &target);
    target.barrier_slot = -1;
    return ssid_stack_push(assembler, target);
}

static int ssid_negate(ssid_assembler *assembler) {
    ssid_operand value, target;
    uint32_t wait;
    int status;
    if ((status = ssid_stack_pop(assembler, &value)) != SSID_OK) return status;
    if (value.immediate) {
        value.immediate_bits ^= UINT32_C(0x80000000);
        return ssid_stack_push(assembler, value);
    }
    if (value.owned) target = value;
    else if ((status = ssid_allocate(assembler, &target)) != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, &value, 1);
    if ((status = ssid_emit(assembler, ssid_fmul_imm(target.reg, value.reg, SSID_NEG_ONE_BITS, wait, SSID_REST_FMUL))) != SSID_OK) return status;
    ssid_release(assembler, value, &target);
    target.barrier_slot = -1;
    return ssid_stack_push(assembler, target);
}

static int ssid_unary(ssid_assembler *assembler, uint8_t opcode) {
    ssid_operand value, target;
    uint32_t wait, slot;
    uint64_t token, rest = 0;
    int status;
    if ((status = ssid_stack_pop(assembler, &value)) != SSID_OK || (status = ssid_ensure_register(assembler, value, &value)) != SSID_OK) return status;
    if (value.owned) target = value;
    else if ((status = ssid_allocate(assembler, &target)) != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, &value, 1);
    if (opcode == SSID_ABS_F32) {
        status = ssid_emit(assembler, ssid_abs(target.reg, value.reg, wait));
        target.barrier_slot = -1;
    } else {
        switch (opcode) {
            case SSID_SQRT_F32: rest = SSID_REST_MUFU_SQRT; break;
            case SSID_RCP_F32: rest = SSID_REST_MUFU_RCP; break;
            case SSID_SIN_F32: rest = SSID_REST_MUFU_SIN; break;
            case SSID_COS_F32: rest = SSID_REST_MUFU_COS; break;
            case SSID_EX2_F32: rest = SSID_REST_MUFU_EX2; break;
            case SSID_LG2_F32: rest = SSID_REST_MUFU_LG2; break;
            case SSID_RSQRT_F32: rest = SSID_REST_MUFU_RSQ; break;
            case SSID_TANH_F32: rest = SSID_REST_MUFU_TANH; break;
            default: return SSID_UNSUPPORTED;
        }
        if (opcode == SSID_SIN_F32 || opcode == SSID_COS_F32) {
            if ((status = ssid_emit(assembler, ssid_fmul_imm(target.reg, value.reg, SSID_SIN_COS_SCALE_BITS, wait, SSID_REST_FMUL_RZ))) != SSID_OK) return status;
            wait = 0;
        }
        if ((status = ssid_allocate_barrier(assembler, &slot, &token)) != SSID_OK || (status = ssid_emit(assembler, ssid_mufu(target.reg, (opcode == SSID_SIN_F32 || opcode == SSID_COS_F32) ? target.reg : value.reg, wait, slot, rest))) != SSID_OK) return status;
        target.barrier_slot = (int8_t)slot;
        target.barrier_token = token;
    }
    if (status != SSID_OK) return status;
    ssid_release(assembler, value, &target);
    return ssid_stack_push(assembler, target);
}

static int ssid_minmax(ssid_assembler *assembler, int maximum) {
    ssid_operand lhs, rhs, target, operands[2];
    uint32_t wait;
    int status;
    if ((status = ssid_stack_pop(assembler, &rhs)) != SSID_OK || (status = ssid_ensure_register(assembler, rhs, &rhs)) != SSID_OK || (status = ssid_stack_pop(assembler, &lhs)) != SSID_OK || (status = ssid_ensure_register(assembler, lhs, &lhs)) != SSID_OK) return status;
    operands[0] = lhs;
    operands[1] = rhs;
    target = ssid_choose_output(assembler, operands, 2, &status);
    if (status != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, operands, 2);
    if ((status = ssid_emit(assembler, ssid_fmnmx(target.reg, lhs.reg, rhs.reg, maximum ? SSID_REST_FMNMX_MAX : SSID_REST_FMNMX_MIN, wait))) != SSID_OK) return status;
    ssid_release(assembler, lhs, &target);
    ssid_release(assembler, rhs, &target);
    target.barrier_slot = -1;
    return ssid_stack_push(assembler, target);
}

static int ssid_fma(ssid_assembler *assembler) {
    ssid_operand lhs, rhs, addend, target, operands[3];
    uint32_t wait;
    int status;
    if ((status = ssid_stack_pop(assembler, &addend)) != SSID_OK || (status = ssid_ensure_register(assembler, addend, &addend)) != SSID_OK || (status = ssid_stack_pop(assembler, &rhs)) != SSID_OK || (status = ssid_ensure_register(assembler, rhs, &rhs)) != SSID_OK || (status = ssid_stack_pop(assembler, &lhs)) != SSID_OK || (status = ssid_ensure_register(assembler, lhs, &lhs)) != SSID_OK) return status;
    operands[0] = lhs;
    operands[1] = rhs;
    operands[2] = addend;
    target = ssid_choose_output(assembler, operands, 3, &status);
    if (status != SSID_OK) return status;
    wait = ssid_wait_mask(assembler, operands, 3);
    if ((status = ssid_emit(assembler, ssid_ffma(target.reg, lhs.reg, rhs.reg, addend.reg, wait))) != SSID_OK) return status;
    ssid_release(assembler, lhs, &target);
    ssid_release(assembler, rhs, &target);
    ssid_release(assembler, addend, &target);
    target.barrier_slot = -1;
    return ssid_stack_push(assembler, target);
}

static void ssid_assembler_initialize(ssid_assembler *assembler, uint8_t *output, uint32_t capacity, const ssid_kernel_plan *kernel, uint32_t site_index) {
    uint8_t reserved[256] = {0};
    uint32_t index;
    memset(assembler, 0, sizeof(*assembler));
    assembler->output = output;
    assembler->capacity = capacity;
    assembler->next_register = kernel->register_count;
    assembler->high_water_register = kernel->register_count;
    assembler->result_register = kernel->output_registers[site_index];
    assembler->pending_wait_mask = 0;
    assembler->next_barrier_token = 1;
    assembler->stack_count = 0;
    for (index = 0; index < kernel->input_count; ++index) reserved[kernel->input_registers[index]] = 1;
    for (index = 0; index < kernel->output_count; ++index) reserved[kernel->output_registers[index]] = 1;
    reserved[SSID_REGISTER_RZ] = 1;
    for (index = 0; index < kernel->available_register_count; ++index) {
        uint8_t reg = kernel->available_registers[index];
        if (!reserved[reg]) assembler->free_registers[assembler->free_count++] = reg;
    }
}

static int ssid_compile_program(ssid_assembler *assembler, const uint8_t *program, uint32_t byte_count, const ssid_kernel_plan *kernel, uint32_t allowed_offset, uint32_t allowed_count) {
    uint32_t cursor = 0;
    int returned = 0;
    int status = SSID_OK;
    while (cursor < byte_count) {
        uint8_t opcode = program[cursor++];
        if (returned) {
            ssid_set_error("postorder instructions follow RETURN_F32");
            return SSID_INVALID_ARGUMENT;
        }
        if (opcode == SSID_STATIC_COLUMN_INPUT_F32) {
            uint32_t input;
            if (cursor >= byte_count) {
                ssid_set_error("truncated static input instruction");
                return SSID_INVALID_ARGUMENT;
            }
            input = program[cursor++];
            if (input >= kernel->input_count || input < allowed_offset || input - allowed_offset >= allowed_count) {
                ssid_set_error("AST input %u is outside its missing-site leaf range", input);
                return SSID_INVALID_ARGUMENT;
            }
            status = ssid_stack_push(assembler, ssid_register_operand(kernel->input_registers[input], 0));
        } else if (opcode == SSID_CONSTANT_BITS_F32) {
            uint32_t bits;
            if (byte_count - cursor < 4) {
                ssid_set_error("truncated FP32 constant instruction");
                return SSID_INVALID_ARGUMENT;
            }
            bits = ssid_load_u32(program + cursor);
            cursor += 4;
            status = ssid_stack_push(assembler, ssid_immediate_operand(bits));
        } else if (opcode == SSID_ADD_F32) status = ssid_add_or_multiply(assembler, 0);
        else if (opcode == SSID_MUL_F32) status = ssid_add_or_multiply(assembler, 1);
        else if (opcode == SSID_SUB_F32) status = ssid_subtract(assembler);
        else if (opcode == SSID_DIV_F32) status = ssid_divide(assembler);
        else if (opcode == SSID_NEG_F32) status = ssid_negate(assembler);
        else if (opcode == SSID_ABS_F32 || opcode == SSID_SQRT_F32 || opcode == SSID_RCP_F32 || opcode == SSID_SIN_F32 || opcode == SSID_COS_F32 || opcode == SSID_EX2_F32 || opcode == SSID_LG2_F32 || opcode == SSID_RSQRT_F32 || opcode == SSID_TANH_F32) status = ssid_unary(assembler, opcode);
        else if (opcode == SSID_MIN_F32) status = ssid_minmax(assembler, 0);
        else if (opcode == SSID_MAX_F32) status = ssid_minmax(assembler, 1);
        else if (opcode == SSID_FMA_F32) status = ssid_fma(assembler);
        else if (opcode == SSID_EXP_F32) {
            status = ssid_stack_push(assembler, ssid_immediate_operand(SSID_LOG2_E_BITS));
            if (status == SSID_OK) status = ssid_add_or_multiply(assembler, 1);
            if (status == SSID_OK) status = ssid_unary(assembler, SSID_EX2_F32);
        } else if (opcode == SSID_LOG_F32) {
            status = ssid_unary(assembler, SSID_LG2_F32);
            if (status == SSID_OK) status = ssid_stack_push(assembler, ssid_immediate_operand(SSID_LN_TWO_BITS));
            if (status == SSID_OK) status = ssid_add_or_multiply(assembler, 1);
        } else if (opcode == SSID_RETURN_F32) {
            returned = 1;
            if (cursor != byte_count || assembler->stack_count != 1) {
                ssid_set_error("RETURN_F32 must terminate a program with one stack value");
                return SSID_INVALID_ARGUMENT;
            }
        } else {
            ssid_set_error("C99 SASS writer does not support opcode 0x%02x", opcode);
            return SSID_UNSUPPORTED;
        }
        if (status != SSID_OK) return status;
    }
    if (!returned || assembler->stack_count != 1) {
        ssid_set_error("postorder program did not produce one result");
        return SSID_INVALID_ARGUMENT;
    }
    {
        ssid_operand root;
        uint32_t wait;
        uint32_t slot;
        if ((status = ssid_stack_pop(assembler, &root)) != SSID_OK) return status;
        if (root.immediate) status = ssid_emit(assembler, ssid_fadd_imm(assembler->result_register, SSID_REGISTER_RZ, root.immediate_bits, 0));
        else {
            wait = ssid_wait_mask(assembler, &root, 1);
            status = ssid_emit(assembler, ssid_mov(assembler->result_register, root.reg, wait));
        }
        if (status != SSID_OK) return status;
        wait = 0;
        for (slot = 0; slot < SSID_BARRIER_SLOTS; ++slot) if (assembler->active_barriers[slot] != 0) wait |= 1u << slot;
        if (wait != 0) {
            status = ssid_emit(assembler, ssid_nop(wait));
            memset(assembler->active_barriers, 0, sizeof(assembler->active_barriers));
        }
    }
    return status;
}

static const ssid_ast_desc *ssid_ast_for_site(const ssid_genome_batch *batch, const ssid_genome_desc *genome, uint32_t site_index) {
    uint32_t index;
    for (index = 0; index < genome->ast_count; ++index) {
        const ssid_ast_desc *candidate = &batch->asts[genome->first_ast + index];
        if (candidate->site_index == site_index) return candidate;
    }
    return NULL;
}

static int ssid_specialize_kernel(const ssid_template *template_value, const ssid_kernel_plan *kernel, const ssid_genome_batch *batch, uint8_t *output, ssid_specialization_stats *stats) {
    uint32_t selected_count;
    uint32_t packed_count = 0;
    uint32_t body_offsets_count = 0;
    uint32_t body_offsets[SSID_MAX_PACKED_GENOMES];
    uint32_t expanded_register_count = kernel->register_count;
    uint32_t index;
    ssid_instruction nop = ssid_nop(0);
    if (kernel->genome_base >= batch->genome_count) return SSID_OK;
    selected_count = batch->genome_count - kernel->genome_base;
    if (selected_count > kernel->genome_capacity) selected_count = kernel->genome_capacity;
    if (selected_count > SSID_MAX_PACKED_GENOMES) {
        ssid_set_error("packed specialization exceeds fixed body-offset scratch capacity");
        return SSID_OUT_OF_RANGE;
    }
    for (index = 0; kernel->entry_file_offset + (uint64_t)index * SSID_INSTRUCTION_BYTES < kernel->arena_start_file_offset; ++index) ssid_store_instruction(output + kernel->entry_file_offset + (uint64_t)index * SSID_INSTRUCTION_BYTES, nop);
    for (index = 0; index < kernel->arena_instruction_count; ++index) ssid_store_instruction(output + kernel->arena_start_file_offset + (uint64_t)index * SSID_INSTRUCTION_BYTES, nop);
    for (index = 0; index < kernel->cleanup_count; ++index) ssid_store_instruction(output + kernel->cleanup_file_offsets[index], nop);
    {
        uint64_t dispatch_start = kernel->dispatch_file_offsets[0];
        uint64_t distance = dispatch_start - kernel->entry_file_offset;
        if (dispatch_start <= kernel->entry_file_offset || distance % SSID_INSTRUCTION_BYTES != 0) {
            ssid_set_error("packed dispatch fast-path branch is misaligned");
            return SSID_INVALID_ARGUMENT;
        }
        ssid_instruction entry_branch = ssid_branch((uint32_t)(distance / SSID_INSTRUCTION_BYTES), template_value->architecture);
        entry_branch.word1 |= (uint64_t)(kernel->incoming_wait_mask & SSID_WAIT_BARRIER_MASK) << 52;
        ssid_store_instruction(output + kernel->entry_file_offset, entry_branch);
    }
    for (index = 0; index < kernel->dispatch_instruction_count; ++index) {
        ssid_instruction instruction = {kernel->dispatch_instruction_words[index * 2], kernel->dispatch_instruction_words[index * 2 + 1]};
        ssid_store_instruction(output + kernel->dispatch_file_offsets[index], instruction);
    }
    for (index = 0; index < selected_count; ++index) {
        const ssid_genome_desc *genome = &batch->genomes[kernel->genome_base + index];
        uint32_t site_index;
        uint32_t body_start = packed_count;
        for (site_index = 0; site_index < template_value->site_count; ++site_index) {
            const ssid_ast_desc *ast = ssid_ast_for_site(batch, genome, site_index);
            ssid_assembler assembler;
            int status;
            if (ast == NULL) {
                ssid_set_error("genome is missing AST site %u", site_index);
                return SSID_INVALID_ARGUMENT;
            }
            ssid_assembler_initialize(&assembler, output + kernel->arena_start_file_offset + (uint64_t)packed_count * SSID_INSTRUCTION_BYTES, kernel->arena_instruction_count - packed_count, kernel, site_index);
            status = ssid_compile_program(&assembler, batch->program_bytes + ast->byte_offset, ast->byte_count, kernel, template_value->site_input_offsets[site_index], template_value->site_input_counts[site_index]);
            if (status != SSID_OK) return status;
            packed_count += assembler.count;
            stats->sass_instruction_count += assembler.count;
            if (assembler.high_water_register > expanded_register_count) expanded_register_count = assembler.high_water_register;
        }
        if (packed_count >= kernel->arena_instruction_count) {
            ssid_set_error("packed genome bodies leave no room for their continuation branch");
            return SSID_OUT_OF_RANGE;
        }
        {
            uint64_t branch_offset = kernel->arena_start_file_offset + (uint64_t)packed_count * SSID_INSTRUCTION_BYTES;
            uint64_t distance;
            if (kernel->continuation_file_offset <= branch_offset || (kernel->continuation_file_offset - branch_offset) % SSID_INSTRUCTION_BYTES != 0) {
                ssid_set_error("packed genome continuation branch is invalid");
                return SSID_INVALID_ARGUMENT;
            }
            distance = kernel->continuation_file_offset - branch_offset;
            ssid_store_instruction(output + branch_offset, ssid_branch((uint32_t)(distance / SSID_INSTRUCTION_BYTES), template_value->architecture));
        }
        body_offsets[body_offsets_count++] = (uint32_t)(kernel->arena_start_file_offset + (uint64_t)body_start * SSID_INSTRUCTION_BYTES - kernel->function_file_offset);
        packed_count += 1;
        stats->sass_instruction_count += 1;
    }
    for (index = 0; index < kernel->target_table_count; ++index) {
        uint32_t target = index < body_offsets_count ? body_offsets[index] : body_offsets[0];
        ssid_store_u32(output + kernel->target_table_file_offsets[index], target);
    }
    if (expanded_register_count > kernel->register_count) {
        uint32_t final_count;
        if (expanded_register_count > SSID_REGISTER_RZ - SSID_REGISTER_PAD) {
            ssid_set_error("specialized expressions leave no register padding");
            return SSID_OUT_OF_RANGE;
        }
        final_count = expanded_register_count + SSID_REGISTER_PAD;
        for (index = 0; index < kernel->register_count_offset_count; ++index) ssid_store_u32(output + kernel->register_count_file_offsets[index], final_count);
        for (index = 0; index < kernel->register_count_header_offset_count; ++index) output[kernel->register_count_header_file_offsets[index]] = (uint8_t)final_count;
        expanded_register_count = final_count;
    }
    if (expanded_register_count > stats->maximum_register_count) stats->maximum_register_count = expanded_register_count;
    stats->populated_kernel_count += 1;
    return SSID_OK;
}

int ssid_specialize_module(const ssid_template *template_value, const ssid_genome_batch *batch, uint8_t *output_cubin, size_t output_byte_count, ssid_specialization_stats *stats) {
    ssid_specialization_stats local_stats;
    uint32_t kernel_index;
    int status;
    if (template_value == NULL || output_cubin == NULL || output_byte_count != template_value->cubin_byte_count) {
        ssid_set_error("specialization output must be one exact-size CUBIN buffer");
        return SSID_INVALID_ARGUMENT;
    }
    if ((status = ssid_validate_batch(template_value, batch)) != SSID_OK) return status;
    memset(&local_stats, 0, sizeof(local_stats));
    local_stats.genome_count = batch->genome_count;
    local_stats.kernel_count = template_value->kernel_count;
    memcpy(output_cubin, template_value->cubin, template_value->cubin_byte_count);
    for (kernel_index = 0; kernel_index < template_value->kernel_count; ++kernel_index) {
        status = ssid_specialize_kernel(template_value, &template_value->kernels[kernel_index], batch, output_cubin, &local_stats);
        if (status != SSID_OK) return status;
    }
    if (stats != NULL) *stats = local_stats;
    return SSID_OK;
}
