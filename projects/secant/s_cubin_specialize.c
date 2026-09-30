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
#include "s_cubin_internal.h"

#include "s_ast_internal.h"

#include <math.h>
#include <string.h>

typedef struct _SecantCubinOperand {
    uint8_t is_immediate;
    uint8_t owned;
    uint8_t reg;
    uint32_t immediate_bits;
    uint8_t barrier_active;
    uint8_t barrier_slot;
    uint32_t barrier_token;
} _SecantCubinOperand;

typedef struct _SecantCubinAssembler {
    const _SecantCubinSite* toggle_site;
    unsigned char* code;
    size_t capacity;
    size_t count;
    _SecantCubinOperand stack[_SECANT_CUBIN_STACK_DEPTH];
    size_t stack_size;
    uint8_t free_registers[_SECANT_CUBIN_MAX_REGISTERS];
    size_t num_free_registers;
    uint32_t next_register;
    uint32_t high_water_register;
    uint8_t result_register;
    uint8_t reserved_registers[_SECANT_CUBIN_MAX_REGISTERS];
    uint8_t barrier_slot_active[_SECANT_CUBIN_BARRIER_SLOTS];
    uint32_t barrier_slot_token[_SECANT_CUBIN_BARRIER_SLOTS];
    uint32_t next_barrier_token;
    uint32_t pending_wait_mask;
} _SecantCubinAssembler;

static uint64_t
_secant_cubin_control(uint32_t wait_mask, uint32_t control_hi, uint32_t stall, uint64_t rest) {
    return
        (((uint64_t)wait_mask & 0xfffull) << 52) |
        (((uint64_t)control_hi & 0xffull) << 44) |
        (((uint64_t)stall & 0xfull) << 40) |
        rest;
}

static uint64_t
_secant_cubin_alu_control(uint32_t wait_mask, uint64_t rest) {
    return _secant_cubin_control(
        wait_mask,
        _SECANT_CUBIN_CONTROL_ALU_HI,
        _SECANT_CUBIN_CONTROL_ALU_STALL,
        rest);
}

static uint64_t
_secant_cubin_barrier_control(
    uint32_t wait_mask,
    uint32_t barrier_slot,
    uint64_t rest
) {
    return _secant_cubin_control(
        wait_mask,
        _SECANT_CUBIN_CONTROL_BARRIER_BASE + 4u * barrier_slot,
        _SECANT_CUBIN_CONTROL_MUFU_STALL,
        rest);
}

static _SecantCubinSassInstruction
_secant_cubin_sass_nop(uint32_t wait_mask) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 = _SECANT_CUBIN_NOP_WORD0;
    instruction.word1 = (((uint64_t)wait_mask & 0xfffull) << 52) | _SECANT_CUBIN_NOP_WORD1;
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_mov(uint8_t dst, uint8_t src, uint32_t wait_mask) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)src << 32) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_MOV;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, _SECANT_CUBIN_REST_MOV);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fadd_reg(
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FADD_REG;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, 0u);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fadd_neg_lhs(
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction = _secant_cubin_sass_fadd_reg(dst, lhs, rhs, wait_mask);

    instruction.word1 |= 0x100ull;
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fadd_imm(
    uint8_t dst,
    uint8_t src,
    uint32_t immediate_bits,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)immediate_bits << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FADD_IMM;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, _SECANT_CUBIN_REST_FADD_IMM);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fmul_reg(
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FMUL_REG;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, _SECANT_CUBIN_REST_FMUL);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fmul_imm(
    uint8_t dst,
    uint8_t src,
    uint32_t immediate_bits,
    uint32_t wait_mask,
    uint64_t rest
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)immediate_bits << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FMUL_IMM;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, rest);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_abs(uint8_t dst, uint8_t src, uint32_t wait_mask) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)0x800000ffu << 32) |
        ((uint64_t)src << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FADD_REG;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, _SECANT_CUBIN_REST_FADD_ABS);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fmnmx(
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs,
    uint64_t rest,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FMNMX_REG;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, rest);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_fsel(
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs,
    uint8_t predicate,
    int invert_predicate,
    int negate_lhs,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction;
    uint64_t rest = (uint64_t)predicate << 23;

    if (invert_predicate) {
        rest |= _SECANT_CUBIN_REST_FSEL_INVERT_PREDICATE;
    }
    if (negate_lhs) {
        rest |= _SECANT_CUBIN_REST_FSEL_NEGATE_LHS;
    }
    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FSEL;
    instruction.word1 = _secant_cubin_alu_control(wait_mask, rest);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_ffma(
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs,
    uint8_t addend,
    uint32_t wait_mask
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)rhs << 32) |
        ((uint64_t)lhs << 24) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_FFMA_REG;
    instruction.word1 = _secant_cubin_alu_control(
        wait_mask,
        _SECANT_CUBIN_REST_FFMA | (uint64_t)addend);
    return instruction;
}

static _SecantCubinSassInstruction
_secant_cubin_sass_mufu(
    uint8_t dst,
    uint8_t src,
    uint32_t wait_mask,
    uint32_t barrier_slot,
    uint64_t rest
) {
    _SecantCubinSassInstruction instruction;

    instruction.word0 =
        ((uint64_t)src << 32) |
        ((uint64_t)dst << 16) |
        _SECANT_CUBIN_OPCODE_MUFU;
    instruction.word1 = _secant_cubin_barrier_control(
        wait_mask,
        barrier_slot,
        rest);
    return instruction;
}

static SecantResult
_secant_cubin_assembler_emit(
    _SecantCubinAssembler* assembler,
    _SecantCubinSassInstruction instruction
) {
    if (assembler->count >= assembler->capacity) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INSUFFICIENT_PATCH_SPACE);
    }
    if (assembler->pending_wait_mask != 0u) {
        instruction.word1 |= ((uint64_t)assembler->pending_wait_mask & 0xfffull) << 52;
        assembler->pending_wait_mask = 0u;
    }
    _secant_cubin_write_instruction(
        assembler->code +
            assembler->count * _SECANT_CUBIN_INSTRUCTION_BYTES,
        instruction);
    assembler->count += 1u;
    return SECANT_SUCCESS;
}

static uint32_t
_secant_cubin_active_barrier_mask(const _SecantCubinAssembler* assembler) {
    uint32_t mask = 0u;
    size_t slot;

    for (slot = 0u; slot < _SECANT_CUBIN_BARRIER_SLOTS; ++slot) {
        if (assembler->barrier_slot_active[slot]) {
            mask |= 1u << slot;
        }
    }
    return mask;
}

static uint32_t
_secant_cubin_wait_mask_for(
    _SecantCubinAssembler* assembler,
    const _SecantCubinOperand* operands,
    size_t num_operands
) {
    uint32_t wait_mask = 0u;
    size_t operand_idx;

    for (operand_idx = 0u; operand_idx < num_operands; ++operand_idx) {
        const _SecantCubinOperand* operand = operands + operand_idx;

        if (operand->barrier_active &&
            operand->barrier_slot < _SECANT_CUBIN_BARRIER_SLOTS &&
            assembler->barrier_slot_active[operand->barrier_slot] &&
            assembler->barrier_slot_token[operand->barrier_slot] == operand->barrier_token) {
            wait_mask |= 1u << operand->barrier_slot;
            assembler->barrier_slot_active[operand->barrier_slot] = 0u;
            assembler->barrier_slot_token[operand->barrier_slot] = 0u;
        }
    }
    return wait_mask;
}

static SecantResult
_secant_cubin_allocate_barrier(
    _SecantCubinAssembler* assembler,
    uint8_t* barrier_slot_ret,
    uint32_t* barrier_token_ret
) {
    size_t slot;

    for (slot = 0u; slot < _SECANT_CUBIN_BARRIER_SLOTS; ++slot) {
        if (!assembler->barrier_slot_active[slot]) {
            assembler->barrier_slot_active[slot] = 1u;
            assembler->barrier_slot_token[slot] = assembler->next_barrier_token++;
            *barrier_slot_ret = (uint8_t)slot;
            *barrier_token_ret = assembler->barrier_slot_token[slot];
            return SECANT_SUCCESS;
        }
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_nop(_secant_cubin_active_barrier_mask(assembler))));
    memset(assembler->barrier_slot_active, 0, sizeof(assembler->barrier_slot_active));
    memset(assembler->barrier_slot_token, 0, sizeof(assembler->barrier_slot_token));
    assembler->barrier_slot_active[0] = 1u;
    assembler->barrier_slot_token[0] = assembler->next_barrier_token++;
    *barrier_slot_ret = 0u;
    *barrier_token_ret = assembler->barrier_slot_token[0];
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_push(_SecantCubinAssembler* assembler, _SecantCubinOperand operand) {
    if (assembler->stack_size >= _SECANT_CUBIN_STACK_DEPTH) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_STACK_OVERFLOW);
    }
    assembler->stack[assembler->stack_size++] = operand;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_pop(
    _SecantCubinAssembler* assembler,
    _SecantCubinOperand* operand_ret
) {
    if (assembler->stack_size == 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_STACK_UNDERFLOW);
    }
    assembler->stack_size -= 1u;
    *operand_ret = assembler->stack[assembler->stack_size];
    return SECANT_SUCCESS;
}

static void
_secant_cubin_reserve_register(_SecantCubinAssembler* assembler, uint8_t reg) {
    assembler->reserved_registers[reg] = 1u;
}

static int
_secant_cubin_register_is_reserved(
    const _SecantCubinAssembler* assembler,
    uint8_t reg
) {
    return
        reg == _SECANT_CUBIN_REGISTER_RZ ||
        assembler->reserved_registers[reg] != 0u;
}

static void
_secant_cubin_prepare_free_registers(
    _SecantCubinAssembler* assembler,
    const uint8_t* available_registers,
    size_t num_available_registers
) {
    size_t register_idx;

    for (register_idx = 0u;
         register_idx < num_available_registers &&
             assembler->num_free_registers < _SECANT_CUBIN_MAX_REGISTERS;
         ++register_idx) {
        const uint8_t reg = available_registers[register_idx];
        size_t existing_idx;
        int seen = 0;

        if (_secant_cubin_register_is_reserved(assembler, reg)) {
            continue;
        }
        for (existing_idx = 0u;
             existing_idx < assembler->num_free_registers;
             ++existing_idx) {
            if (assembler->free_registers[existing_idx] == reg) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            assembler->free_registers[assembler->num_free_registers++] = reg;
        }
    }
}

static SecantResult
_secant_cubin_allocate_register(
    _SecantCubinAssembler* assembler,
    _SecantCubinOperand* operand_ret
) {
    _SecantCubinOperand operand;
    uint32_t reg;

    memset(&operand, 0, sizeof(operand));
    operand.owned = 1u;
    if (assembler->num_free_registers != 0u) {
        assembler->num_free_registers -= 1u;
        reg = assembler->free_registers[assembler->num_free_registers];
    } else {
        reg = assembler->next_register;
        if (reg >= _SECANT_CUBIN_REGISTER_RZ) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_REGISTER_OVERFLOW);
        }
        assembler->next_register += 1u;
        if (assembler->next_register > assembler->high_water_register) {
            assembler->high_water_register = assembler->next_register;
        }
    }
    operand.reg = (uint8_t)reg;
    *operand_ret = operand;
    return SECANT_SUCCESS;
}

static void
_secant_cubin_release_register(
    _SecantCubinAssembler* assembler,
    _SecantCubinOperand operand
) {
    if (operand.owned && !operand.is_immediate &&
        !_secant_cubin_register_is_reserved(assembler, operand.reg) &&
        assembler->num_free_registers < _SECANT_CUBIN_MAX_REGISTERS) {
        assembler->free_registers[assembler->num_free_registers++] = operand.reg;
    }
}

static void
_secant_cubin_reclaim_routine_args(
    _SecantCubinAssembler* assembler,
    const _SecantCubinOperand* args,
    size_t num_args,
    _SecantCubinOperand* result
) {
    size_t arg_idx;

    for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
        size_t prior_idx;
        int duplicate = 0;

        if (!args[arg_idx].owned || args[arg_idx].is_immediate) {
            continue;
        }
        for (prior_idx = 0u; prior_idx < arg_idx; ++prior_idx) {
            if (args[prior_idx].owned && !args[prior_idx].is_immediate &&
                args[prior_idx].reg == args[arg_idx].reg) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        if (!result->is_immediate && result->reg == args[arg_idx].reg) {
            result->owned = 1u;
        } else {
            assembler->pending_wait_mask |= _secant_cubin_wait_mask_for(assembler, args + arg_idx, 1u);
            _secant_cubin_release_register(assembler, args[arg_idx]);
        }
    }
}

static SecantResult
_secant_cubin_ensure_register(
    _SecantCubinAssembler* assembler,
    _SecantCubinOperand* operand
) {
    _SecantCubinOperand dst;

    if (!operand->is_immediate) {
        return SECANT_SUCCESS;
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &dst));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fadd_imm(
            dst.reg,
            _SECANT_CUBIN_REGISTER_RZ,
            operand->immediate_bits,
            0u)));
    *operand = dst;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_choose_output(
    _SecantCubinAssembler* assembler,
    const _SecantCubinOperand* lhs,
    const _SecantCubinOperand* rhs,
    _SecantCubinOperand* dst_ret
) {
    if (lhs->owned && !lhs->is_immediate) {
        *dst_ret = *lhs;
        return SECANT_SUCCESS;
    }
    if (rhs->owned && !rhs->is_immediate) {
        *dst_ret = *rhs;
        return SECANT_SUCCESS;
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_allocate_register(assembler, dst_ret));
}

static void
_secant_cubin_release_consumed(
    _SecantCubinAssembler* assembler,
    _SecantCubinOperand lhs,
    _SecantCubinOperand rhs,
    _SecantCubinOperand keep
) {
    if (lhs.owned && !lhs.is_immediate && lhs.reg != keep.reg) {
        _secant_cubin_release_register(assembler, lhs);
    }
    if (rhs.owned && !rhs.is_immediate && rhs.reg != keep.reg) {
        _secant_cubin_release_register(assembler, rhs);
    }
}

static void
_secant_cubin_release_consumed3(
    _SecantCubinAssembler* assembler,
    _SecantCubinOperand a,
    _SecantCubinOperand b,
    _SecantCubinOperand c,
    _SecantCubinOperand keep
) {
    if (a.owned && !a.is_immediate && a.reg != keep.reg) {
        _secant_cubin_release_register(assembler, a);
    }
    if (b.owned && !b.is_immediate && b.reg != keep.reg) {
        _secant_cubin_release_register(assembler, b);
    }
    if (c.owned && !c.is_immediate && c.reg != keep.reg) {
        _secant_cubin_release_register(assembler, c);
    }
}

static SecantResult
_secant_cubin_emit_add_or_mul(
    _SecantCubinAssembler* assembler,
    int is_mul
) {
    _SecantCubinOperand rhs;
    _SecantCubinOperand lhs;
    _SecantCubinOperand dst;
    _SecantCubinOperand waits[2];
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &lhs));
    if (lhs.is_immediate && !rhs.is_immediate) {
        _SecantCubinOperand swap = lhs;
        lhs = rhs;
        rhs = swap;
    } else if (lhs.is_immediate && rhs.is_immediate) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &lhs));
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_choose_output(
        assembler,
        &lhs,
        &rhs,
        &dst));
    waits[0] = lhs;
    waits[1] = rhs;
    wait_mask = _secant_cubin_wait_mask_for(assembler, waits, 2u);
    if (is_mul) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            assembler,
            rhs.is_immediate
                ? _secant_cubin_sass_fmul_imm(
                    dst.reg,
                    lhs.reg,
                    rhs.immediate_bits,
                    wait_mask,
                    _SECANT_CUBIN_REST_FMUL)
                : _secant_cubin_sass_fmul_reg(
                    dst.reg,
                    lhs.reg,
                    rhs.reg,
                    wait_mask)));
    } else {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            assembler,
            rhs.is_immediate
                ? _secant_cubin_sass_fadd_imm(
                    dst.reg,
                    lhs.reg,
                    rhs.immediate_bits,
                    wait_mask)
                : _secant_cubin_sass_fadd_reg(
                    dst.reg,
                    lhs.reg,
                    rhs.reg,
                    wait_mask)));
    }
    dst.barrier_active = 0u;
    _secant_cubin_release_consumed(assembler, lhs, rhs, dst);
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_minmax(_SecantCubinAssembler* assembler, uint64_t rest) {
    _SecantCubinOperand rhs;
    _SecantCubinOperand lhs;
    _SecantCubinOperand dst;
    _SecantCubinOperand waits[2];
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_choose_output(
        assembler,
        &lhs,
        &rhs,
        &dst));
    waits[0] = lhs;
    waits[1] = rhs;
    wait_mask = _secant_cubin_wait_mask_for(assembler, waits, 2u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fmnmx(
            dst.reg,
            lhs.reg,
            rhs.reg,
            rest,
            wait_mask)));
    dst.barrier_active = 0u;
    _secant_cubin_release_consumed(assembler, lhs, rhs, dst);
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_fma(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand addend;
    _SecantCubinOperand rhs;
    _SecantCubinOperand lhs;
    _SecantCubinOperand dst;
    _SecantCubinOperand waits[3];
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &addend));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &addend));
    if (lhs.owned) {
        dst = lhs;
    } else if (rhs.owned) {
        dst = rhs;
    } else if (addend.owned) {
        dst = addend;
    } else {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &dst));
    }
    waits[0] = lhs;
    waits[1] = rhs;
    waits[2] = addend;
    wait_mask = _secant_cubin_wait_mask_for(assembler, waits, 3u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_ffma(
            dst.reg,
            lhs.reg,
            rhs.reg,
            addend.reg,
            wait_mask)));
    dst.barrier_active = 0u;
    _secant_cubin_release_consumed3(assembler, lhs, rhs, addend, dst);
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_neg(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand value;
    _SecantCubinOperand dst;
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &value));
    if (value.is_immediate) {
        value.immediate_bits ^= 0x80000000u;
        _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, value));
    }
    dst = value;
    if (!dst.owned) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &dst));
    }
    wait_mask = _secant_cubin_wait_mask_for(assembler, &value, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fmul_imm(
            dst.reg,
            value.reg,
            _SECANT_CUBIN_NEG_ONE_BITS,
            wait_mask,
            _SECANT_CUBIN_REST_FMUL)));
    dst.barrier_active = 0u;
    if (value.owned && value.reg != dst.reg) {
        _secant_cubin_release_register(assembler, value);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_abs(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand value;
    _SecantCubinOperand dst;
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &value));
    if (value.is_immediate) {
        value.immediate_bits &= 0x7fffffffu;
        _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, value));
    }
    dst = value;
    if (!dst.owned) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &dst));
    }
    wait_mask = _secant_cubin_wait_mask_for(assembler, &value, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_abs(dst.reg, value.reg, wait_mask)));
    dst.barrier_active = 0u;
    if (value.owned && value.reg != dst.reg) {
        _secant_cubin_release_register(assembler, value);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_sub(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand rhs;
    _SecantCubinOperand lhs;
    _SecantCubinOperand neg_rhs;
    _SecantCubinOperand dst;
    _SecantCubinOperand waits[2];
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &lhs));
    if (rhs.is_immediate) {
        rhs.immediate_bits ^= 0x80000000u;
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(assembler, lhs));
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(assembler, rhs));
        _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_add_or_mul(assembler, 0));
    }
    neg_rhs = rhs;
    if (!neg_rhs.owned) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &neg_rhs));
    }
    wait_mask = _secant_cubin_wait_mask_for(assembler, &rhs, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fmul_imm(
            neg_rhs.reg,
            rhs.reg,
            _SECANT_CUBIN_NEG_ONE_BITS,
            wait_mask,
            _SECANT_CUBIN_REST_FMUL)));
    neg_rhs.barrier_active = 0u;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_choose_output(
        assembler,
        &lhs,
        &neg_rhs,
        &dst));
    waits[0] = lhs;
    waits[1] = neg_rhs;
    wait_mask = _secant_cubin_wait_mask_for(assembler, waits, 2u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fadd_reg(
            dst.reg,
            lhs.reg,
            neg_rhs.reg,
            wait_mask)));
    dst.barrier_active = 0u;
    if (rhs.owned && rhs.reg != neg_rhs.reg && rhs.reg != dst.reg) {
        _secant_cubin_release_register(assembler, rhs);
    }
    if (lhs.owned && lhs.reg != dst.reg) {
        _secant_cubin_release_register(assembler, lhs);
    }
    if (neg_rhs.owned && neg_rhs.reg != dst.reg) {
        _secant_cubin_release_register(assembler, neg_rhs);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_mufu(_SecantCubinAssembler* assembler, uint64_t rest) {
    _SecantCubinOperand value;
    _SecantCubinOperand dst;
    uint8_t barrier_slot;
    uint32_t barrier_token;
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &value));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &value));
    dst = value;
    if (!dst.owned) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &dst));
    }
    wait_mask = _secant_cubin_wait_mask_for(assembler, &value, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_barrier(
        assembler,
        &barrier_slot,
        &barrier_token));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_mufu(
            dst.reg,
            value.reg,
            wait_mask,
            barrier_slot,
            rest)));
    dst.barrier_active = 1u;
    dst.barrier_slot = barrier_slot;
    dst.barrier_token = barrier_token;
    if (value.owned && value.reg != dst.reg) {
        _secant_cubin_release_register(assembler, value);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_sincos(_SecantCubinAssembler* assembler, uint64_t rest) {
    _SecantCubinOperand value;
    _SecantCubinOperand dst;
    uint8_t barrier_slot;
    uint32_t barrier_token;
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &value));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &value));
    dst = value;
    if (!dst.owned) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &dst));
    }
    wait_mask = _secant_cubin_wait_mask_for(assembler, &value, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fmul_imm(
            dst.reg,
            value.reg,
            _SECANT_CUBIN_SIN_COS_SCALE_BITS,
            wait_mask,
            _SECANT_CUBIN_REST_FMUL_RZ)));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_barrier(
        assembler,
        &barrier_slot,
        &barrier_token));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_mufu(
            dst.reg,
            dst.reg,
            0u,
            barrier_slot,
            rest)));
    dst.barrier_active = 1u;
    dst.barrier_slot = barrier_slot;
    dst.barrier_token = barrier_token;
    if (value.owned && value.reg != dst.reg) {
        _secant_cubin_release_register(assembler, value);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_exp(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand scale;

    memset(&scale, 0, sizeof(scale));
    scale.is_immediate = 1u;
    scale.immediate_bits = _SECANT_CUBIN_LOG2_E_BITS;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(assembler, scale));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_emit_add_or_mul(assembler, 1));
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
        assembler,
        _SECANT_CUBIN_REST_MUFU_EX2));
}

static SecantResult
_secant_cubin_emit_log(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand scale;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_emit_mufu(
        assembler,
        _SECANT_CUBIN_REST_MUFU_LG2));
    memset(&scale, 0, sizeof(scale));
    scale.is_immediate = 1u;
    scale.immediate_bits = _SECANT_CUBIN_LN_TWO_BITS;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(assembler, scale));
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_add_or_mul(assembler, 1));
}

static SecantResult
_secant_cubin_emit_div(_SecantCubinAssembler* assembler) {
    _SecantCubinOperand rhs;
    _SecantCubinOperand lhs;
    _SecantCubinOperand reciprocal;
    _SecantCubinOperand dst;
    _SecantCubinOperand waits[2];
    uint8_t barrier_slot;
    uint32_t barrier_token;
    uint32_t wait_mask;

    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &rhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &lhs));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(assembler, &rhs));
    reciprocal = rhs;
    if (!reciprocal.owned) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(
            assembler,
            &reciprocal));
    }
    wait_mask = _secant_cubin_wait_mask_for(assembler, &rhs, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_barrier(
        assembler,
        &barrier_slot,
        &barrier_token));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_mufu(
            reciprocal.reg,
            rhs.reg,
            wait_mask,
            barrier_slot,
            _SECANT_CUBIN_REST_MUFU_RCP)));
    reciprocal.barrier_active = 1u;
    reciprocal.barrier_slot = barrier_slot;
    reciprocal.barrier_token = barrier_token;
    dst = lhs.owned ? lhs : reciprocal;
    waits[0] = lhs;
    waits[1] = reciprocal;
    wait_mask = _secant_cubin_wait_mask_for(assembler, waits, 2u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
        assembler,
        _secant_cubin_sass_fmul_reg(
            dst.reg,
            lhs.reg,
            reciprocal.reg,
            wait_mask)));
    dst.barrier_active = 0u;
    if (lhs.owned && lhs.reg != dst.reg) {
        _secant_cubin_release_register(assembler, lhs);
    }
    if (rhs.owned && rhs.reg != reciprocal.reg && rhs.reg != dst.reg) {
        _secant_cubin_release_register(assembler, rhs);
    }
    if (reciprocal.owned && reciprocal.reg != dst.reg) {
        _secant_cubin_release_register(assembler, reciprocal);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_push(assembler, dst));
}

static SecantResult
_secant_cubin_emit_op(
    _SecantCubinAssembler* assembler,
    SecantAstInstructionType instruction_type
) {
    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_add_or_mul(assembler, 0));
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_add_or_mul(assembler, 1));
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_sub(assembler));
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_div(assembler));
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_neg(assembler));
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_abs(assembler));
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_minmax(
                assembler,
                _SECANT_CUBIN_REST_FMNMX_MIN));
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_minmax(
                assembler,
                _SECANT_CUBIN_REST_FMNMX_MAX));
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_fma(assembler));
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
                assembler,
                _SECANT_CUBIN_REST_MUFU_SQRT));
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
                assembler,
                _SECANT_CUBIN_REST_MUFU_RCP));
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_sincos(
                assembler,
                _SECANT_CUBIN_REST_MUFU_SIN));
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_sincos(
                assembler,
                _SECANT_CUBIN_REST_MUFU_COS));
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
                assembler,
                _SECANT_CUBIN_REST_MUFU_EX2));
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
                assembler,
                _SECANT_CUBIN_REST_MUFU_LG2));
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
                assembler,
                _SECANT_CUBIN_REST_MUFU_RSQ));
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_mufu(
                assembler,
                _SECANT_CUBIN_REST_MUFU_TANH));
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_exp(assembler));
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
            _SECANT_CUBIN_ERROR_RET(_secant_cubin_emit_log(assembler));
        default:
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_UNSUPPORTED_OP);
    }
}

static SecantResult
_secant_cubin_routine_num_args(
    const SecantAstInstruction* instructions,
    size_t* num_args_ret
) {
    size_t instruction_count;
    size_t instruction_offset = 0u;
    size_t num_args = 0u;

    if (instructions == NULL || num_args_ret == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            const SecantAstIdx arg_idx = secant_ast_index_get(instruction);

            if (arg_idx >= SECANT_AST_MAX_INSTRUCTION_ARGS) {
                _SECANT_CUBIN_ERROR_RET(
                    SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
            }
            if ((size_t)arg_idx + 1u > num_args) {
                num_args = (size_t)arg_idx + 1u;
            }
        } else if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_SUCCESS;
        }
        instruction_offset += instruction_size;
    }
    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
}

static SecantResult
_secant_cubin_routine_num_args_build(
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    uint8_t* routine_num_args
) {
    size_t routine_idx;

    if (num_routines != 0u && (routines == NULL || routine_num_args == NULL)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t num_args;

        if (routines[routine_idx] == NULL) {
            _SECANT_CUBIN_ERROR_RET(
                SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
        }
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_routine_num_args(
            routines[routine_idx],
            &num_args));
        routine_num_args[routine_idx] = (uint8_t)num_args;
    }
    return SECANT_SUCCESS;
}

static SecantResult _secant_cubin_toggle_test(_SecantCubinAssembler *a, unsigned bit) {
    _SecantCubinSassInstruction instruction;
    const _SecantCubinSite *site = a->toggle_site;
    if (!site || bit >= 32) return SECANT_ERROR_BAD_PROGRAM;
    instruction.word0 = (site->toggle_test.word0 & UINT64_C(0xffffff)) |
        ((uint64_t)site->permutation_reg << 24) | ((uint64_t)(UINT32_C(1) << bit) << 32);
    instruction.word1 = _secant_cubin_alu_control(0, site->toggle_test.word1 & ((UINT64_C(1) << 40) - 1));
    return _secant_cubin_assembler_emit(a, instruction);
}
static SecantResult _secant_cubin_select(_SecantCubinAssembler *a,
    _SecantCubinOperand left, _SecantCubinOperand right, unsigned bit, _SecantCubinOperand *out) {
    _SecantCubinOperand operands[2];
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(a, &left));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(a, &right));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_choose_output(a, &left, &right, out));
    operands[0] = left; operands[1] = right;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_toggle_test(a, bit));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(a,
        _secant_cubin_sass_fsel(out->reg, right.reg, left.reg, a->toggle_site->predicate_reg,
            0, 0, _secant_cubin_wait_mask_for(a, operands, 2))));
    _secant_cubin_release_consumed(a, left, right, *out);
    return SECANT_SUCCESS;
}
static SecantResult _secant_cubin_emit_toggle(_SecantCubinAssembler *a, const uint8_t *bits, size_t count) {
    _SecantCubinOperand leaves[4], low, high, result;
    size_t i;
    if (!a->toggle_site || bits[0] >= 32 || (count == 4 && (bits[1] >= 32 || bits[0] == bits[1])))
        return SECANT_ERROR_BAD_PROGRAM;
    for (i = count; i; --i) _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(a, leaves + i - 1));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_select(a, leaves[0], leaves[1], bits[0], &low));
    if (count == 2) return _secant_cubin_push(a, low);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_select(a, leaves[2], leaves[3], bits[0], &high));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_select(a, low, high, bits[1], &result));
    return _secant_cubin_push(a, result);
}

static SecantResult
_secant_cubin_compile_frame(
    _SecantCubinAssembler* assembler,
    const uint8_t* input_registers,
    size_t num_input_registers,
    _SecantCubinShape shape,
    size_t num_static_column_registers,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const uint8_t* routine_num_args,
    const SecantAstInstruction* instructions,
    const _SecantCubinOperand* routine_args,
    size_t num_routine_args,
    size_t frame_depth
) {
    const size_t base_stack_size = assembler->stack_size;
    size_t instruction_count;
    size_t instruction_offset = 0u;
    size_t recent_direct = 0u;

    if (instructions == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
    }
    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }

        switch (instruction_type) {
            case SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32:
            case SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32:
            case SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32: {
                _SecantCubinOperand operand;
                size_t input_idx = secant_ast_index_get(instruction);

                if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32) {
                    if (input_idx >= num_static_column_registers) return SECANT_ERROR_BAD_PROGRAM;
                } else {
                    if (shape != _SECANT_CUBIN_SHAPE_TOGGLE_SSE || input_idx >= num_input_registers - num_static_column_registers)
                        return SECANT_ERROR_BAD_PROGRAM;
                    input_idx += num_static_column_registers;
                }
                memset(&operand, 0, sizeof(operand));
                operand.reg = input_registers[input_idx];
                if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32) {
                    _SecantCubinOperand candidate;
                    const float scale = secant_ast_affine_bank_scale_get(instruction);
                    const float offset = secant_ast_affine_bank_offset_get(instruction);
                    if (!isfinite(scale) || !isfinite(offset)) return SECANT_ERROR_BAD_PROGRAM;
                    _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(assembler, &candidate));
                    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(assembler,
                        _secant_cubin_sass_fmul_imm(candidate.reg, operand.reg,
                            secant_ast_affine_bank_scale_bits_get(instruction), 0u, _SECANT_CUBIN_REST_FMUL)));
                    _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(assembler,
                        _secant_cubin_sass_fadd_imm(candidate.reg, candidate.reg,
                            secant_ast_affine_bank_offset_bits_get(instruction), 0u)));
                    operand = candidate;
                }
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(assembler, operand));
                break;
            }
            case SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32:
            case SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32: {
                size_t count = instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ? 2 : 4;
                if (recent_direct < count) return SECANT_ERROR_BAD_PROGRAM;
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_emit_toggle(assembler, instruction + 1, count));
                break;
            }
            case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32: {
                _SecantCubinOperand operand;

                memset(&operand, 0, sizeof(operand));
                operand.is_immediate = 1u;
                operand.immediate_bits = secant_ast_constant_f32_bits_get(instruction);
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(assembler, operand));
                break;
            }
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
                if (routine_args == NULL || secant_ast_index_get(instruction) >= num_routine_args) {
                    _SECANT_CUBIN_ERROR_RET(
                        SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                }
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_push(
                    assembler,
                    routine_args[secant_ast_index_get(instruction)]));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32: {
                const size_t routine_idx = secant_ast_index_get(instruction);
                const size_t next_frame_depth = frame_depth + 1u;
                _SecantCubinOperand caller_args[SECANT_AST_MAX_INSTRUCTION_ARGS];
                _SecantCubinOperand next_args[SECANT_AST_MAX_INSTRUCTION_ARGS];
                size_t num_args;
                size_t arg_start;
                size_t arg_idx;

                if (routines == NULL || routine_idx >= num_routines ||
                    routines[routine_idx] == NULL) {
                    _SECANT_CUBIN_ERROR_RET(
                        SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                }
                if (next_frame_depth > _SECANT_CUBIN_ROUTINE_DEPTH) {
                    _SECANT_CUBIN_ERROR_RET(
                        SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED);
                }
                num_args = routine_num_args[routine_idx];
                if (assembler->stack_size < num_args) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_STACK_UNDERFLOW);
                }
                arg_start = assembler->stack_size - num_args;
                for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
                    caller_args[arg_idx] = assembler->stack[arg_start + arg_idx];
                    next_args[arg_idx] = caller_args[arg_idx];
                    next_args[arg_idx].owned = 0u;
                }
                assembler->stack_size = arg_start;
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_compile_frame(
                    assembler,
                    input_registers,
                    num_input_registers,
                    shape,
                    num_static_column_registers,
                    routines,
                    num_routines,
                    routine_num_args,
                    routines[routine_idx],
                    next_args,
                    num_args,
                    next_frame_depth
                ));
                _secant_cubin_reclaim_routine_args(
                    assembler,
                    caller_args,
                    num_args,
                    assembler->stack + arg_start);
                break;
            }
            case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32:
                if (assembler->stack_size != base_stack_size + 1u) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
                }
                return SECANT_SUCCESS;
            default:
                if (!secant_internal_ast_instruction_is_f32(instruction_type) ||
                    secant_internal_ast_instruction_num_args(instruction_type) == 0u) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_UNSUPPORTED_OP);
                }
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_emit_op(
                    assembler,
                    instruction_type));
                break;
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) ++recent_direct;
        else recent_direct = 0;
        instruction_offset += instruction_size;
    }
    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
}

static SecantResult
_secant_cubin_generate_ast_sass(
    const uint8_t* input_registers,
    size_t num_input_registers,
    _SecantCubinShape shape,
    size_t num_static_column_registers,
    const uint8_t* available_registers,
    size_t num_available_registers,
    uint8_t result_register,
    uint32_t incoming_wait_mask,
    uint32_t first_new_register,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const uint8_t* routine_num_args,
    const SecantAstInstruction* instructions,
    unsigned char* code,
    size_t sass_capacity,
    size_t* sass_count_ret,
    uint32_t* expanded_register_count_ret
) {
    _SecantCubinAssembler assembler;
    _SecantCubinOperand root;
    uint32_t wait_mask;
    size_t input_idx;

    if (sass_count_ret == NULL || expanded_register_count_ret == NULL ||
        instructions == NULL || input_registers == NULL ||
        num_input_registers == 0u || code == NULL || sass_capacity == 0u ||
        first_new_register > _SECANT_CUBIN_REGISTER_RZ ||
        result_register == _SECANT_CUBIN_REGISTER_RZ) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *sass_count_ret = 0u;
    *expanded_register_count_ret = first_new_register;
    memset(&assembler, 0, sizeof(assembler));
    assembler.code = code;
    assembler.capacity = sass_capacity;
    assembler.next_register = first_new_register;
    assembler.high_water_register = first_new_register;
    assembler.result_register = result_register;
    assembler.next_barrier_token = 1u;
    assembler.pending_wait_mask = incoming_wait_mask;
    for (input_idx = 0u; input_idx < num_input_registers; ++input_idx) {
        _secant_cubin_reserve_register(&assembler, input_registers[input_idx]);
    }
    _secant_cubin_prepare_free_registers(
        &assembler,
        available_registers,
        num_available_registers);
    _secant_cubin_prepare_free_registers(&assembler, &result_register, 1u);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_compile_frame(
        &assembler,
        input_registers,
        num_input_registers,
        shape,
        num_static_column_registers,
        routines,
        num_routines,
        routine_num_args,
        instructions,
        NULL,
        0u,
        0u
    ));
    if (assembler.stack_size != 1u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(&assembler, &root));
    if (root.is_immediate) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_fadd_imm(
                result_register,
                _SECANT_CUBIN_REGISTER_RZ,
                root.immediate_bits,
                0u)));
    } else {
        wait_mask = _secant_cubin_wait_mask_for(&assembler, &root, 1u);
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_mov(result_register, root.reg, wait_mask)));
    }
    if (_secant_cubin_active_barrier_mask(&assembler) != 0u) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_nop(
                _secant_cubin_active_barrier_mask(&assembler))));
    }
    *sass_count_ret = assembler.count;
    *expanded_register_count_ret = assembler.high_water_register;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_generate_sse_sass(
    const uint8_t* input_registers,
    size_t num_input_registers,
    _SecantCubinShape shape,
    size_t num_static_column_registers,
    const uint8_t* target_registers,
    size_t num_target_registers,
    const uint8_t* available_registers,
    size_t num_available_registers,
    const uint8_t* sse_registers,
    size_t num_sse_registers,
    uint32_t incoming_wait_mask,
    uint32_t first_new_register,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const uint8_t* routine_num_args,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const _SecantCubinSite* toggle_site,
    unsigned char* code,
    size_t sass_capacity,
    size_t* sass_count_ret,
    uint32_t* expanded_register_count_ret
) {
    _SecantCubinAssembler assembler;
    size_t expected_sse_registers;
    size_t input_idx;
    size_t target_idx;
    size_t ast_idx;

    if (sass_count_ret == NULL || expanded_register_count_ret == NULL ||
        input_registers == NULL ||
        target_registers == NULL || num_target_registers == 0u ||
        sse_registers == NULL || asts == NULL || num_asts == 0u ||
        code == NULL || sass_capacity == 0u ||
        first_new_register > _SECANT_CUBIN_REGISTER_RZ ||
        !_secant_cubin_checked_mul(
            num_asts,
            num_target_registers,
            &expected_sse_registers) ||
        num_sse_registers != expected_sse_registers) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *sass_count_ret = 0u;
    *expanded_register_count_ret = first_new_register;
    memset(&assembler, 0, sizeof(assembler));
    assembler.toggle_site = toggle_site;
    if (toggle_site) _secant_cubin_reserve_register(&assembler, toggle_site->permutation_reg);
    assembler.code = code;
    assembler.capacity = sass_capacity;
    assembler.next_register = first_new_register;
    assembler.high_water_register = first_new_register;
    assembler.result_register = _SECANT_CUBIN_REGISTER_RZ;
    assembler.next_barrier_token = 1u;
    assembler.pending_wait_mask = incoming_wait_mask;
    for (input_idx = 0u; input_idx < num_input_registers; ++input_idx) {
        _secant_cubin_reserve_register(&assembler, input_registers[input_idx]);
    }
    for (target_idx = 0u; target_idx < num_target_registers; ++target_idx) {
        _secant_cubin_reserve_register(&assembler, target_registers[target_idx]);
    }
    for (ast_idx = 0u; ast_idx < num_sse_registers; ++ast_idx) {
        if (sse_registers[ast_idx] == _SECANT_CUBIN_REGISTER_RZ) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
        }
        _secant_cubin_reserve_register(&assembler, sse_registers[ast_idx]);
    }
    _secant_cubin_prepare_free_registers(
        &assembler,
        available_registers,
        num_available_registers);
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        _SecantCubinOperand prediction;

        if (asts[ast_idx] == NULL || assembler.stack_size != 0u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_compile_frame(
            &assembler,
            input_registers,
            num_input_registers,
            shape,
            num_static_column_registers,
            routines,
            num_routines,
            routine_num_args,
            asts[ast_idx],
            NULL,
            0u,
            0u
        ));
        if (assembler.stack_size != 1u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(&assembler, &prediction));
        for (target_idx = 0u; target_idx < num_target_registers; ++target_idx) {
            _SecantCubinOperand error;
            uint32_t wait_mask = 0u;
            const int reuse_prediction =
                target_idx + 1u == num_target_registers &&
                prediction.owned &&
                !prediction.is_immediate &&
                !_secant_cubin_register_is_reserved(
                    &assembler,
                    prediction.reg);

            if (reuse_prediction) {
                error = prediction;
            } else {
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_allocate_register(
                    &assembler,
                    &error));
            }
            if (prediction.is_immediate) {
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
                    &assembler,
                    _secant_cubin_sass_fadd_imm(
                        error.reg,
                        target_registers[target_idx],
                        prediction.immediate_bits ^ 0x80000000u,
                        0u)));
            } else {
                wait_mask = _secant_cubin_wait_mask_for(
                    &assembler,
                    &prediction,
                    1u);
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
                    &assembler,
                    _secant_cubin_sass_fadd_neg_lhs(
                        error.reg,
                        target_registers[target_idx],
                        prediction.reg,
                        wait_mask)));
            }
            error.barrier_active = 0u;
            error.barrier_slot = 0u;
            error.barrier_token = 0u;
            _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
                &assembler,
                _secant_cubin_sass_ffma(
                    sse_registers[
                        ast_idx * num_target_registers + target_idx],
                    error.reg,
                    error.reg,
                    sse_registers[
                        ast_idx * num_target_registers + target_idx],
                    0u)));
            _secant_cubin_release_register(&assembler, error);
            if (reuse_prediction) {
                prediction.owned = 0u;
            }
        }
        if (prediction.owned && !prediction.is_immediate) {
            assembler.pending_wait_mask |= _secant_cubin_wait_mask_for(
                &assembler,
                &prediction,
                1u);
            _secant_cubin_release_register(&assembler, prediction);
        }
    }
    if (assembler.stack_size != 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
    }
    if (_secant_cubin_active_barrier_mask(&assembler) != 0u) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_nop(
                _secant_cubin_active_barrier_mask(&assembler))));
    }
    *sass_count_ret = assembler.count;
    *expanded_register_count_ret = assembler.high_water_register;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_generate_affine_stats_sass(
    const uint8_t* input_registers,
    size_t num_input_registers,
    const uint8_t* target_registers,
    size_t num_target_registers,
    const uint8_t* available_registers,
    size_t num_available_registers,
    const uint8_t* accumulator_registers,
    size_t num_accumulator_registers,
    size_t asts_capacity,
    uint32_t incoming_wait_mask,
    uint32_t first_new_register,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const uint8_t* routine_num_args,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    unsigned char* code,
    size_t sass_capacity,
    size_t* sass_count_ret,
    uint32_t* expanded_register_count_ret
) {
    _SecantCubinAssembler assembler;
    size_t num_prediction_registers;
    size_t num_cross_registers;
    size_t expected_accumulator_registers;
    size_t cross_register_offset;
    size_t accumulator_idx;
    size_t input_idx;
    size_t target_idx;
    size_t ast_idx;

    if (sass_count_ret == NULL || expanded_register_count_ret == NULL ||
        input_registers == NULL ||
        target_registers == NULL || num_target_registers == 0u ||
        accumulator_registers == NULL || asts == NULL || num_asts == 0u ||
        num_asts > asts_capacity || code == NULL || sass_capacity == 0u ||
        first_new_register > _SECANT_CUBIN_REGISTER_RZ ||
        !_secant_cubin_checked_mul(asts_capacity, 2u, &num_prediction_registers) ||
        !_secant_cubin_checked_mul(asts_capacity, num_target_registers, &num_cross_registers) ||
        !_secant_cubin_checked_add(
            num_prediction_registers,
            num_cross_registers,
            &expected_accumulator_registers) ||
        num_accumulator_registers != expected_accumulator_registers) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    cross_register_offset = num_prediction_registers;
    *sass_count_ret = 0u;
    *expanded_register_count_ret = first_new_register;
    memset(&assembler, 0, sizeof(assembler));
    assembler.code = code;
    assembler.capacity = sass_capacity;
    assembler.next_register = first_new_register;
    assembler.high_water_register = first_new_register;
    assembler.result_register = _SECANT_CUBIN_REGISTER_RZ;
    assembler.next_barrier_token = 1u;
    assembler.pending_wait_mask = incoming_wait_mask;
    for (input_idx = 0u; input_idx < num_input_registers; ++input_idx) {
        _secant_cubin_reserve_register(&assembler, input_registers[input_idx]);
    }
    for (target_idx = 0u; target_idx < num_target_registers; ++target_idx) {
        _secant_cubin_reserve_register(&assembler, target_registers[target_idx]);
    }
    for (accumulator_idx = 0u; accumulator_idx < num_accumulator_registers; ++accumulator_idx) {
        if (accumulator_registers[accumulator_idx] == _SECANT_CUBIN_REGISTER_RZ) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
        }
        _secant_cubin_reserve_register(&assembler, accumulator_registers[accumulator_idx]);
    }
    _secant_cubin_prepare_free_registers(&assembler, available_registers, num_available_registers);
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        _SecantCubinOperand prediction;
        const uint8_t sum = accumulator_registers[2u * ast_idx];
        const uint8_t square_sum = accumulator_registers[2u * ast_idx + 1u];
        uint32_t wait_mask;

        if (asts[ast_idx] == NULL || assembler.stack_size != 0u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_compile_frame(
            &assembler,
            input_registers,
            num_input_registers,
            _SECANT_CUBIN_SHAPE_AFFINE_STATS,
            num_input_registers,
            routines,
            num_routines,
            routine_num_args,
            asts[ast_idx],
            NULL,
            0u,
            0u
        ));
        if (assembler.stack_size != 1u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_pop(&assembler, &prediction));
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_ensure_register(&assembler, &prediction));
        wait_mask = _secant_cubin_wait_mask_for(&assembler, &prediction, 1u);
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_fadd_reg(sum, sum, prediction.reg, wait_mask)));
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_ffma(square_sum, prediction.reg, prediction.reg, square_sum, 0u)));
        for (target_idx = 0u; target_idx < num_target_registers; ++target_idx) {
            const uint8_t cross_sum = accumulator_registers[
                cross_register_offset + ast_idx * num_target_registers + target_idx];

            _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
                &assembler,
                _secant_cubin_sass_ffma(
                    cross_sum,
                    prediction.reg,
                    target_registers[target_idx],
                    cross_sum,
                    0u)));
        }
        _secant_cubin_release_register(&assembler, prediction);
    }
    if (assembler.stack_size != 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
    }
    if (_secant_cubin_active_barrier_mask(&assembler) != 0u) {
        _SECANT_CUBIN_CHECK_RET(_secant_cubin_assembler_emit(
            &assembler,
            _secant_cubin_sass_nop(_secant_cubin_active_barrier_mask(&assembler))));
    }
    *sass_count_ret = assembler.count;
    *expanded_register_count_ret = assembler.high_water_register;
    return SECANT_SUCCESS;
}

static _SecantCubinSassInstruction
_secant_cubin_branch(size_t target_delta_instructions, uint32_t capability_major) {
    _SecantCubinSassInstruction instruction;
    uint64_t target;

    if (capability_major == 8u) {
        target = (uint64_t)(16u * target_delta_instructions - 16u);
        instruction.word0 = (target << 32) | _SECANT_CUBIN_OPCODE_BRA;
    } else {
        target = (uint64_t)(4u * target_delta_instructions - 4u);
        instruction.word0 =
            ((target & 0xffull) << 16) |
            ((target & ~0xffull) << 26) |
            _SECANT_CUBIN_OPCODE_BRA;
    }
    instruction.word1 = _secant_cubin_alu_control(0u, _SECANT_CUBIN_REST_BRA);
    return instruction;
}
static SecantResult
_secant_cubin_patch(
    const SecantCubinPlan* plan,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    size_t cubin_size,
    unsigned char* cubin
) {
    uint8_t routine_num_args[SECANT_AST_MAX_ROUTINES];
    size_t expected_asts;
    size_t kernel_idx;

    if (plan == NULL || asts == NULL || cubin == NULL ||
        num_routines > SECANT_AST_MAX_ROUTINES ||
        (num_routines != 0u && routines == NULL) ||
        !_secant_cubin_checked_mul(
            plan->num_kernels,
            plan->asts_per_kernel,
            &expected_asts) ||
        num_asts == 0u || num_asts > expected_asts) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (cubin_size != plan->cubin_size) {
        _SECANT_CUBIN_ERROR_RET(
            cubin_size < plan->cubin_size
                ? SECANT_ERROR_INSUFFICIENT_BUFFER
                : SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_routine_num_args_build(
        routines,
        num_routines,
        routine_num_args));
    for (kernel_idx = 0u; kernel_idx < plan->num_kernels; ++kernel_idx) {
        const _SecantCubinKernel* kernel = plan->kernels + kernel_idx;
        const _SecantCubinSite* site = &kernel->site;
        unsigned char* site_code;
        size_t sass_count = 0u;
        const size_t num_static_column_registers = plan->num_input_registers - plan->num_constant_registers;
        uint32_t expanded_register_count = kernel->register_count;
        const size_t first_ast = kernel_idx * plan->asts_per_kernel;
        size_t kernel_num_asts = 0u;
        size_t ast_idx;

        if (first_ast < num_asts) {
            kernel_num_asts = num_asts - first_ast;
            if (kernel_num_asts > plan->asts_per_kernel) {
                kernel_num_asts = plan->asts_per_kernel;
            }
        }
        if (site->num_instructions >
                SIZE_MAX / _SECANT_CUBIN_INSTRUCTION_BYTES ||
            !_secant_cubin_range_ok(
                cubin_size,
                site->start_file_offset,
                site->num_instructions *
                    _SECANT_CUBIN_INSTRUCTION_BYTES) ||
            !_secant_cubin_range_ok(
                cubin_size,
                site->load_fence_file_offset,
                _SECANT_CUBIN_INSTRUCTION_BYTES)) {
            _SECANT_CUBIN_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_PATCH_SPACE);
        }
        site_code = cubin + site->start_file_offset;
        if (kernel_num_asts != 0u &&
            (plan->shape == _SECANT_CUBIN_SHAPE_SSE || plan->shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE)) {
            size_t num_sse_registers;

            if (!_secant_cubin_checked_mul(kernel_num_asts, site->num_target_regs, &num_sse_registers)) {
                _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
            }
            _SECANT_CUBIN_CHECK_RET(_secant_cubin_generate_sse_sass(
                site->input_regs,
                site->num_input_regs,
                plan->shape,
                num_static_column_registers,
                site->target_regs,
                site->num_target_regs,
                site->available_regs,
                site->num_available_regs,
                site->output_regs,
                num_sse_registers,
                site->incoming_wait_mask,
                kernel->register_count,
                routines,
                num_routines,
                routine_num_args,
                asts + first_ast,
                kernel_num_asts,
                plan->shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE ? site : NULL,
                site_code,
                site->num_instructions,
                &sass_count,
                &expanded_register_count
            ));
        } else if (kernel_num_asts != 0u && plan->shape == _SECANT_CUBIN_SHAPE_AFFINE_STATS) {
            _SECANT_CUBIN_CHECK_RET(_secant_cubin_generate_affine_stats_sass(
                site->input_regs,
                site->num_input_regs,
                site->target_regs,
                site->num_target_regs,
                site->available_regs,
                site->num_available_regs,
                site->output_regs,
                site->num_output_regs,
                plan->asts_per_kernel,
                site->incoming_wait_mask,
                kernel->register_count,
                routines,
                num_routines,
                routine_num_args,
                asts + first_ast,
                kernel_num_asts,
                site_code,
                site->num_instructions,
                &sass_count,
                &expanded_register_count));
        } else if (plan->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
                   plan->shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
            for (ast_idx = 0u;
                 ast_idx < kernel_num_asts;
                 ++ast_idx) {
                size_t ast_sass_count = 0u;
                uint32_t ast_register_count = kernel->register_count;
                const size_t program_idx = first_ast + ast_idx;

                if (asts[program_idx] == NULL) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
                }
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_generate_ast_sass(
                    site->input_regs,
                    site->num_input_regs,
                    plan->shape,
                    num_static_column_registers,
                    site->available_regs,
                    site->num_available_regs,
                    site->output_regs[ast_idx],
                    ast_idx == 0u ? site->incoming_wait_mask : 0u,
                    kernel->register_count,
                    routines,
                    num_routines,
                    routine_num_args,
                    asts[program_idx],
                    site_code +
                        sass_count * _SECANT_CUBIN_INSTRUCTION_BYTES,
                    site->num_instructions - sass_count,
                    &ast_sass_count,
                    &ast_register_count));
                sass_count += ast_sass_count;
                if (ast_register_count > expanded_register_count) {
                    expanded_register_count = ast_register_count;
                }
            }
        }
        if (sass_count < site->num_instructions) {
            const size_t remaining = site->num_instructions - sass_count;

            _secant_cubin_write_instruction(
                site_code +
                    sass_count * _SECANT_CUBIN_INSTRUCTION_BYTES,
                remaining == 1u
                    ? _secant_cubin_sass_nop(0u)
                    : _secant_cubin_branch(
                        remaining,
                        plan->compute_capability_major));
            sass_count += 1u;
        }
        _secant_cubin_write_instruction(
            cubin + site->load_fence_file_offset,
            _secant_cubin_sass_nop(0u));
        if (expanded_register_count > kernel->register_count) {
            if (expanded_register_count > _SECANT_CUBIN_REGISTER_RZ - _SECANT_CUBIN_REGISTER_PAD) {
                _SECANT_CUBIN_ERROR_RET(
                    SECANT_ERROR_REGISTER_OVERFLOW);
            }
            expanded_register_count += _SECANT_CUBIN_REGISTER_PAD;
        }
        for (ast_idx = 0u;
             ast_idx < kernel->num_register_counts;
             ++ast_idx) {
            const size_t register_count_file_offset = plan->register_count_file_offsets[
                kernel->first_register_count + ast_idx];

            if (!_secant_cubin_range_ok(
                    cubin_size,
                    register_count_file_offset,
                    sizeof(uint32_t))) {
                _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
            }
            _secant_cubin_write_u32(
                cubin + register_count_file_offset,
                expanded_register_count);
        }
    }
    return SECANT_SUCCESS;
}

SecantResult
secant_cubin_specialize_into(
    const SecantCubinPlan* plan,
    const SecantAstProgramSet* programs,
    void* cubin,
    size_t cubin_size
) {
    SecantResult result;

    if (plan == NULL || programs == NULL || cubin == NULL ||
        programs->asts.items == NULL || programs->asts.count == 0u ||
        (programs->routines.count != 0u && programs->routines.items == NULL)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    result = _secant_cubin_patch(
    plan,
    programs->routines.items,
    programs->routines.count,
    programs->asts.items,
    programs->asts.count,
    cubin_size,
    (unsigned char*)cubin
);
    if (result != SECANT_SUCCESS) {
        _SECANT_CUBIN_ERROR_RET(result);
    }
    return SECANT_SUCCESS;
}


#undef _SECANT_CUBIN_CHECK_RET
#undef _SECANT_CUBIN_ERROR_RET
