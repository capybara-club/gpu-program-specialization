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

#define O_TRY(expression)                                                                                    \
    do {                                                                                                     \
        OdezzaResult o_result_ = (expression);                                                               \
        if (o_result_ != ODEZZA_SUCCESS)                                                                     \
            return o_result_;                                                                                \
    } while (0)

typedef struct OOperand {
    uint32_t immediate_bits;
    uint32_t barrier_token;
    uint8_t reg;
    int8_t barrier_slot;
    unsigned char immediate;
    unsigned char owned;
} OOperand;

typedef struct OAssembler {
    const OdezzaScoringCubinInspection *inspection;
    size_t state_capacity;
    uint32_t constant_capacity;
    size_t active_state_count;
    uint32_t active_constant_count;
    uint32_t active_toggle_count;
    OSassWriter writer;
    OOperand stack[O_MAX_AST_STACK_DEPTH];
    size_t stack_count;
    uint8_t free_registers[255];
    size_t free_register_count;
    uint32_t next_register;
    uint32_t high_water_register;
    uint8_t result_register;
    uint32_t active_barrier_tokens[O_BARRIER_SLOTS];
    uint32_t next_barrier_token;
} OAssembler;

static OdezzaResult o_emit(OAssembler *assembler, OSassEncoding instruction) {
    if (assembler == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    return o_sass_writer_emit(&assembler->writer, instruction);
}

static OdezzaResult o_allocate(OAssembler *assembler, OOperand *operand_ret) {
    uint32_t value;
    if (assembler == NULL || operand_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(operand_ret, 0, sizeof(*operand_ret));
    operand_ret->barrier_slot = -1;
    operand_ret->owned = 1u;
    if (assembler->free_register_count != 0u) {
        value = assembler->free_registers[--assembler->free_register_count];
    } else {
        if (assembler->next_register >= O_REGISTER_RZ)
            return ODEZZA_ERROR_REGISTER_PRESSURE;
        value = assembler->next_register++;
    }
    operand_ret->reg = (uint8_t)value;
    if (assembler->high_water_register < value + 1u)
        assembler->high_water_register = value + 1u;
    return ODEZZA_SUCCESS;
}

static uint32_t o_wait_mask_for(OAssembler *assembler, const OOperand *operands, size_t operand_count) {
    uint32_t mask = 0u;
    size_t index;
    for (index = 0u; index < operand_count; ++index) {
        int slot = operands[index].barrier_slot;
        if (slot >= 0 && slot < (int)O_BARRIER_SLOTS &&
            assembler->active_barrier_tokens[slot] == operands[index].barrier_token) {
            mask |= 1u << (uint32_t)slot;
            assembler->active_barrier_tokens[slot] = 0u;
        }
    }
    return mask;
}

static void o_release(OAssembler *assembler, OOperand operand, const OOperand *keep) {
    size_t index;
    if (!operand.owned || operand.immediate || (keep != NULL && operand.reg == keep->reg))
        return;
    assembler->writer.pending_wait_mask |= o_wait_mask_for(assembler, &operand, 1u);
    for (index = 0u; index < assembler->free_register_count; ++index) {
        if (assembler->free_registers[index] == operand.reg)
            return;
    }
    assembler->free_registers[assembler->free_register_count++] = operand.reg;
}

static OdezzaResult o_allocate_barrier(OAssembler *assembler, uint32_t *slot_ret, uint32_t *token_ret) {
    uint32_t slot;
    if (assembler == NULL || slot_ret == NULL || token_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (slot = 0u; slot < O_BARRIER_SLOTS; ++slot) {
        if (assembler->active_barrier_tokens[slot] == 0u)
            break;
    }
    if (slot == O_BARRIER_SLOTS) {
        uint32_t mask = 0u;
        for (slot = 0u; slot < O_BARRIER_SLOTS; ++slot)
            mask |= 1u << slot;
        O_TRY(o_emit(assembler, o_sass_nop(mask)));
        memset(assembler->active_barrier_tokens, 0, sizeof(assembler->active_barrier_tokens));
        slot = 0u;
    }
    *token_ret = assembler->next_barrier_token++;
    assembler->active_barrier_tokens[slot] = *token_ret;
    *slot_ret = slot;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_ensure_register(OAssembler *assembler, OOperand operand, OOperand *operand_ret) {
    if (assembler == NULL || operand_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (!operand.immediate) {
        *operand_ret = operand;
        return ODEZZA_SUCCESS;
    }
    O_TRY(o_allocate(assembler, operand_ret));
    return o_emit(assembler,
                  o_sass_fadd_immediate(operand_ret->reg, O_REGISTER_RZ, operand.immediate_bits, 0u));
}

static OdezzaResult o_choose_output(OAssembler *assembler, const OOperand *operands, size_t operand_count,
                                    OOperand *operand_ret) {
    size_t index;
    if (assembler == NULL || operands == NULL || operand_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (index = 0u; index < operand_count; ++index) {
        if (operands[index].owned && !operands[index].immediate) {
            *operand_ret = operands[index];
            return ODEZZA_SUCCESS;
        }
    }
    return o_allocate(assembler, operand_ret);
}

static OdezzaResult o_push(OAssembler *assembler, OOperand operand) {
    if (assembler == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (assembler->stack_count == O_MAX_AST_STACK_DEPTH)
        return ODEZZA_ERROR_AST;
    assembler->stack[assembler->stack_count++] = operand;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_pop(OAssembler *assembler, OOperand *operand_ret) {
    if (assembler == NULL || operand_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (assembler->stack_count == 0u)
        return ODEZZA_ERROR_AST;
    *operand_ret = assembler->stack[--assembler->stack_count];
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_source_operand(OAssembler *assembler, int constant, uint8_t index,
                                     OOperand *operand_ret) {
    size_t input_index;
    if (assembler == NULL || operand_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    input_index = constant ? assembler->state_capacity + index : index;
    if (input_index >= assembler->inspection->input_count)
        return ODEZZA_ERROR_AST;
    memset(operand_ret, 0, sizeof(*operand_ret));
    operand_ret->barrier_slot = -1;
    operand_ret->reg = assembler->inspection->input_registers[input_index];
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_select_level(OAssembler *assembler, uint8_t bit_index, OOperand *inputs,
                                   size_t input_count, OOperand *outputs) {
    size_t index;
    if (assembler == NULL || inputs == NULL || outputs == NULL || input_count < 2u ||
        (input_count & 1u) != 0u)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (bit_index >= assembler->active_toggle_count)
        return ODEZZA_ERROR_AST;
    O_TRY(o_emit(assembler, o_sass_toggle_test(assembler->inspection->toggle_test_instruction,
                                               assembler->inspection->permutation_register,
                                               assembler->inspection->predicate_register, bit_index)));
    for (index = 0u; index < input_count; index += 2u) {
        OOperand pair[2];
        O_TRY(o_ensure_register(assembler, inputs[index], &pair[0]));
        O_TRY(o_ensure_register(assembler, inputs[index + 1u], &pair[1]));
        O_TRY(o_choose_output(assembler, pair, 2u, &outputs[index / 2u]));
        O_TRY(o_emit(assembler, o_sass_fsel(outputs[index / 2u].reg, pair[1].reg, pair[0].reg,
                                            assembler->inspection->predicate_register)));
        o_release(assembler, pair[0], &outputs[index / 2u]);
        o_release(assembler, pair[1], &outputs[index / 2u]);
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_add_or_multiply(OAssembler *assembler, int multiply) {
    OOperand lhs;
    OOperand rhs;
    OOperand operands[2];
    OOperand target;
    uint32_t wait;
    O_TRY(o_pop(assembler, &rhs));
    O_TRY(o_pop(assembler, &lhs));
    if (lhs.immediate && !rhs.immediate) {
        OOperand temporary = lhs;
        lhs = rhs;
        rhs = temporary;
    } else if (lhs.immediate && rhs.immediate) {
        O_TRY(o_ensure_register(assembler, lhs, &lhs));
    }
    operands[0] = lhs;
    operands[1] = rhs;
    O_TRY(o_choose_output(assembler, operands, 2u, &target));
    wait = o_wait_mask_for(assembler, operands, 2u);
    if (multiply) {
        O_TRY(o_emit(assembler, rhs.immediate ? o_sass_fmul_immediate(target.reg, lhs.reg, rhs.immediate_bits,
                                                                      wait, O_SASS_MULTIPLY_NORMAL)
                                              : o_sass_fmul_register(target.reg, lhs.reg, rhs.reg, wait)));
    } else {
        O_TRY(o_emit(assembler, rhs.immediate
                                    ? o_sass_fadd_immediate(target.reg, lhs.reg, rhs.immediate_bits, wait)
                                    : o_sass_fadd_register(target.reg, lhs.reg, rhs.reg, wait)));
    }
    o_release(assembler, lhs, &target);
    o_release(assembler, rhs, &target);
    target.barrier_slot = -1;
    return o_push(assembler, target);
}

static OdezzaResult o_subtract(OAssembler *assembler) {
    OOperand lhs;
    OOperand rhs;
    OOperand negated;
    OOperand target;
    OOperand operands[2];
    O_TRY(o_pop(assembler, &rhs));
    O_TRY(o_pop(assembler, &lhs));
    if (rhs.immediate) {
        rhs.immediate_bits ^= 0x80000000u;
        O_TRY(o_push(assembler, lhs));
        O_TRY(o_push(assembler, rhs));
        return o_add_or_multiply(assembler, 0);
    }
    if (rhs.owned) {
        negated = rhs;
    } else {
        O_TRY(o_allocate(assembler, &negated));
    }
    O_TRY(o_emit(assembler,
                 o_sass_fmul_immediate(negated.reg, rhs.reg, O_NEG_ONE_BITS,
                                       o_wait_mask_for(assembler, &rhs, 1u), O_SASS_MULTIPLY_NORMAL)));
    O_TRY(o_ensure_register(assembler, lhs, &lhs));
    operands[0] = lhs;
    operands[1] = negated;
    O_TRY(o_choose_output(assembler, operands, 2u, &target));
    O_TRY(o_emit(assembler, o_sass_fadd_register(target.reg, lhs.reg, negated.reg,
                                                 o_wait_mask_for(assembler, operands, 2u))));
    if (rhs.reg != negated.reg)
        o_release(assembler, rhs, NULL);
    o_release(assembler, lhs, &target);
    o_release(assembler, negated, &target);
    return o_push(assembler, target);
}

static OdezzaResult o_divide(OAssembler *assembler) {
    OOperand lhs;
    OOperand rhs;
    OOperand reciprocal;
    OOperand target;
    OOperand operands[2];
    uint32_t slot;
    uint32_t token;
    O_TRY(o_pop(assembler, &rhs));
    O_TRY(o_pop(assembler, &lhs));
    O_TRY(o_ensure_register(assembler, rhs, &rhs));
    O_TRY(o_ensure_register(assembler, lhs, &lhs));
    if (rhs.owned)
        reciprocal = rhs;
    else
        O_TRY(o_allocate(assembler, &reciprocal));
    O_TRY(o_allocate_barrier(assembler, &slot, &token));
    O_TRY(o_emit(assembler, o_sass_mufu(reciprocal.reg, rhs.reg, o_wait_mask_for(assembler, &rhs, 1u), slot,
                                        O_SASS_MUFU_RCP)));
    reciprocal.barrier_slot = (int8_t)slot;
    reciprocal.barrier_token = token;
    target = lhs.owned ? lhs : reciprocal;
    operands[0] = lhs;
    operands[1] = reciprocal;
    O_TRY(o_emit(assembler, o_sass_fmul_register(target.reg, lhs.reg, reciprocal.reg,
                                                 o_wait_mask_for(assembler, operands, 2u))));
    o_release(assembler, lhs, &target);
    if (rhs.reg != reciprocal.reg)
        o_release(assembler, rhs, &target);
    o_release(assembler, reciprocal, &target);
    target.barrier_slot = -1;
    return o_push(assembler, target);
}

static OdezzaResult o_unary(OAssembler *assembler, OdezzaAstOpcode opcode) {
    OOperand value;
    OOperand target;
    uint32_t wait;
    OSassMufu rest = O_SASS_MUFU_COS;
    uint32_t slot;
    uint32_t token;
    int sin_cos = 0;
    O_TRY(o_pop(assembler, &value));
    if (opcode == ODEZZA_AST_NEG_F32 && value.immediate) {
        value.immediate_bits ^= 0x80000000u;
        return o_push(assembler, value);
    }
    O_TRY(o_ensure_register(assembler, value, &value));
    if (value.owned)
        target = value;
    else
        O_TRY(o_allocate(assembler, &target));
    wait = o_wait_mask_for(assembler, &value, 1u);
    if (opcode == ODEZZA_AST_NEG_F32) {
        O_TRY(o_emit(assembler, o_sass_fmul_immediate(target.reg, value.reg, O_NEG_ONE_BITS, wait,
                                                      O_SASS_MULTIPLY_NORMAL)));
    } else if (opcode == ODEZZA_AST_ABS_F32) {
        O_TRY(o_emit(assembler, o_sass_absolute(target.reg, value.reg, wait)));
    } else {
        switch (opcode) {
        case ODEZZA_AST_SQRT_F32:
            rest = O_SASS_MUFU_SQRT;
            break;
        case ODEZZA_AST_RCP_F32:
            rest = O_SASS_MUFU_RCP;
            break;
        case ODEZZA_AST_SIN_F32:
            rest = O_SASS_MUFU_SIN;
            sin_cos = 1;
            break;
        case ODEZZA_AST_COS_F32:
            rest = O_SASS_MUFU_COS;
            sin_cos = 1;
            break;
        case ODEZZA_AST_EX2_F32:
            rest = O_SASS_MUFU_EX2;
            break;
        case ODEZZA_AST_LG2_F32:
            rest = O_SASS_MUFU_LG2;
            break;
        case ODEZZA_AST_RSQRT_F32:
            rest = O_SASS_MUFU_RSQ;
            break;
        case ODEZZA_AST_TANH_F32:
            rest = O_SASS_MUFU_TANH;
            break;
        default:
            return ODEZZA_ERROR_AST;
        }
        if (sin_cos) {
            O_TRY(o_emit(assembler, o_sass_fmul_immediate(target.reg, value.reg, O_SIN_COS_SCALE_BITS, wait,
                                                          O_SASS_MULTIPLY_RZ)));
            wait = 0u;
        }
        O_TRY(o_allocate_barrier(assembler, &slot, &token));
        O_TRY(o_emit(assembler, o_sass_mufu(target.reg, sin_cos ? target.reg : value.reg, wait, slot, rest)));
        target.barrier_slot = (int8_t)slot;
        target.barrier_token = token;
    }
    o_release(assembler, value, &target);
    return o_push(assembler, target);
}

static OdezzaResult o_minmax(OAssembler *assembler, int maximum) {
    OOperand lhs;
    OOperand rhs;
    OOperand operands[2];
    OOperand target;
    O_TRY(o_pop(assembler, &rhs));
    O_TRY(o_pop(assembler, &lhs));
    O_TRY(o_ensure_register(assembler, rhs, &rhs));
    O_TRY(o_ensure_register(assembler, lhs, &lhs));
    operands[0] = lhs;
    operands[1] = rhs;
    O_TRY(o_choose_output(assembler, operands, 2u, &target));
    O_TRY(o_emit(assembler, o_sass_fmnmx(target.reg, lhs.reg, rhs.reg, maximum,
                                         o_wait_mask_for(assembler, operands, 2u))));
    o_release(assembler, lhs, &target);
    o_release(assembler, rhs, &target);
    return o_push(assembler, target);
}

static OdezzaResult o_fused_multiply_add(OAssembler *assembler) {
    OOperand lhs;
    OOperand rhs;
    OOperand addend;
    OOperand operands[3];
    OOperand target;
    O_TRY(o_pop(assembler, &addend));
    O_TRY(o_pop(assembler, &rhs));
    O_TRY(o_pop(assembler, &lhs));
    O_TRY(o_ensure_register(assembler, addend, &addend));
    O_TRY(o_ensure_register(assembler, rhs, &rhs));
    O_TRY(o_ensure_register(assembler, lhs, &lhs));
    operands[0] = lhs;
    operands[1] = rhs;
    operands[2] = addend;
    O_TRY(o_choose_output(assembler, operands, 3u, &target));
    O_TRY(o_emit(assembler, o_sass_ffma(target.reg, lhs.reg, rhs.reg, addend.reg,
                                        o_wait_mask_for(assembler, operands, 3u))));
    o_release(assembler, lhs, &target);
    o_release(assembler, rhs, &target);
    o_release(assembler, addend, &target);
    return o_push(assembler, target);
}

static uint32_t o_ast_read_u32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) | ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static OdezzaResult o_compile_toggle(OAssembler *assembler, const uint8_t *bit_indices, size_t bit_count) {
    OOperand choices[4];
    OOperand reduced[2];
    size_t choice_count = (size_t)1u << bit_count;
    size_t level;
    size_t index;
    if (assembler == NULL || bit_indices == NULL || bit_count == 0u || bit_count > 2u)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (index = choice_count; index != 0u; --index) {
        O_TRY(o_pop(assembler, &choices[index - 1u]));
    }
    for (level = 0u; level < bit_count; ++level) {
        O_TRY(o_select_level(assembler, bit_indices[level], choices, choice_count, reduced));
        choice_count /= 2u;
        memcpy(choices, reduced, choice_count * sizeof(choices[0]));
    }
    return o_push(assembler, choices[0]);
}

static OdezzaResult o_compile_program(OAssembler *assembler, OdezzaAstProgram program) {
    size_t offset = 0u;
    int returned = 0;
    /* Every caller validates the program and its payload bounds before assembly. */
    while (offset < program.byte_count) {
        OdezzaAstOpcode opcode = (OdezzaAstOpcode)program.bytes[offset++];
        OOperand value;
        if (opcode == ODEZZA_AST_STATE_F32 || opcode == ODEZZA_AST_CONSTANT_F32) {
            O_TRY(o_source_operand(assembler, opcode == ODEZZA_AST_CONSTANT_F32, program.bytes[offset++],
                                   &value));
            O_TRY(o_push(assembler, value));
        } else if (opcode == ODEZZA_AST_LITERAL_F32) {
            memset(&value, 0, sizeof(value));
            value.barrier_slot = -1;
            value.immediate = 1u;
            value.immediate_bits = o_ast_read_u32(program.bytes + offset);
            offset += 4u;
            O_TRY(o_push(assembler, value));
        } else if (opcode == ODEZZA_AST_TOGGLE2_F32) {
            O_TRY(o_compile_toggle(assembler, program.bytes + offset, 1u));
            offset += 1u;
        } else if (opcode == ODEZZA_AST_TOGGLE4_F32) {
            O_TRY(o_compile_toggle(assembler, program.bytes + offset, 2u));
            offset += 2u;
        } else if (opcode == ODEZZA_AST_ADD_F32) {
            O_TRY(o_add_or_multiply(assembler, 0));
        } else if (opcode == ODEZZA_AST_MUL_F32) {
            O_TRY(o_add_or_multiply(assembler, 1));
        } else if (opcode == ODEZZA_AST_SUB_F32) {
            O_TRY(o_subtract(assembler));
        } else if (opcode == ODEZZA_AST_DIV_F32) {
            O_TRY(o_divide(assembler));
        } else if (opcode == ODEZZA_AST_NEG_F32 || opcode == ODEZZA_AST_ABS_F32 ||
                   opcode == ODEZZA_AST_SQRT_F32 || opcode == ODEZZA_AST_RCP_F32 ||
                   opcode == ODEZZA_AST_SIN_F32 || opcode == ODEZZA_AST_COS_F32 ||
                   opcode == ODEZZA_AST_EX2_F32 || opcode == ODEZZA_AST_LG2_F32 ||
                   opcode == ODEZZA_AST_RSQRT_F32 || opcode == ODEZZA_AST_TANH_F32) {
            O_TRY(o_unary(assembler, opcode));
        } else if (opcode == ODEZZA_AST_MIN_F32 || opcode == ODEZZA_AST_MAX_F32) {
            O_TRY(o_minmax(assembler, opcode == ODEZZA_AST_MAX_F32));
        } else if (opcode == ODEZZA_AST_FMA_F32) {
            O_TRY(o_fused_multiply_add(assembler));
        } else if (opcode == ODEZZA_AST_EXP_F32) {
            memset(&value, 0, sizeof(value));
            value.barrier_slot = -1;
            value.immediate = 1u;
            value.immediate_bits = O_LOG2_E_BITS;
            O_TRY(o_push(assembler, value));
            O_TRY(o_add_or_multiply(assembler, 1));
            O_TRY(o_unary(assembler, ODEZZA_AST_EX2_F32));
        } else if (opcode == ODEZZA_AST_LOG_F32) {
            O_TRY(o_unary(assembler, ODEZZA_AST_LG2_F32));
            memset(&value, 0, sizeof(value));
            value.barrier_slot = -1;
            value.immediate = 1u;
            value.immediate_bits = O_LN_TWO_BITS;
            O_TRY(o_push(assembler, value));
            O_TRY(o_add_or_multiply(assembler, 1));
        } else if (opcode == ODEZZA_AST_RETURN_F32) {
            returned = 1;
            break;
        } else {
            return ODEZZA_ERROR_AST;
        }
    }
    if (!returned || assembler->stack_count != 1u)
        return ODEZZA_ERROR_AST;
    {
        OOperand root;
        uint32_t active_mask = 0u;
        uint32_t slot;
        O_TRY(o_pop(assembler, &root));
        if (root.immediate)
            O_TRY(o_emit(assembler, o_sass_fadd_immediate(assembler->result_register, O_REGISTER_RZ,
                                                          root.immediate_bits, 0u)));
        else
            O_TRY(o_emit(assembler, o_sass_mov(assembler->result_register, root.reg,
                                               o_wait_mask_for(assembler, &root, 1u))));
        for (slot = 0u; slot < O_BARRIER_SLOTS; ++slot) {
            if (assembler->active_barrier_tokens[slot] != 0u)
                active_mask |= 1u << slot;
        }
        if (active_mask != 0u) {
            O_TRY(o_emit(assembler, o_sass_nop(active_mask)));
            memset(assembler->active_barrier_tokens, 0, sizeof(assembler->active_barrier_tokens));
        }
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_assembler_init(OAssembler *assembler, const OdezzaScoringCubinInspection *inspection,
                                     size_t state_capacity, uint32_t constant_capacity,
                                     size_t active_state_count, uint32_t active_constant_count,
                                     uint32_t active_toggle_count, uint8_t result_register,
                                     uint32_t incoming_wait_mask, OdezzaScoringInstruction *instructions,
                                     size_t instruction_capacity) {
    size_t index;
    if (assembler == NULL || inspection == NULL || instructions == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (!o_sass_architecture_supported(inspection->architecture))
        return ODEZZA_ERROR_UNSUPPORTED;
    if (inspection->predicate_register >= 7u)
        return ODEZZA_ERROR_FORMAT;
    memset(assembler, 0, sizeof(*assembler));
    assembler->inspection = inspection;
    assembler->state_capacity = state_capacity;
    assembler->constant_capacity = constant_capacity;
    assembler->active_state_count = active_state_count;
    assembler->active_constant_count = active_constant_count;
    assembler->active_toggle_count = active_toggle_count;
    O_TRY(o_sass_writer_init(&assembler->writer, instructions, instruction_capacity, incoming_wait_mask));
    assembler->next_register = inspection->register_count;
    assembler->high_water_register = inspection->register_count;
    assembler->result_register = result_register;
    assembler->next_barrier_token = 1u;
    for (index = 0u; index < inspection->available_register_count; ++index) {
        uint8_t value = inspection->available_registers[index];
        size_t other;
        int reserved = value == O_REGISTER_RZ || value == inspection->permutation_register;
        for (other = 0u; other < inspection->input_count; ++other) {
            int active = other < active_state_count ||
                         (other >= state_capacity && other < state_capacity + active_constant_count);
            if (active && inspection->input_registers[other] == value)
                reserved = 1;
        }
        for (other = 0u; other < inspection->output_count; ++other)
            if (inspection->output_registers[other] == value ||
                inspection->final_output_registers[other] == value)
                reserved = 1;
        if (!reserved)
            assembler->free_registers[assembler->free_register_count++] = value;
    }
    for (index = 0u; index < inspection->input_count; ++index) {
        int active = index < active_state_count ||
                     (index >= state_capacity && index < state_capacity + active_constant_count);
        uint8_t value = inspection->input_registers[index];
        size_t other;
        int duplicate = 0;
        if (active || value == O_REGISTER_RZ || value == inspection->permutation_register)
            continue;
        for (other = 0u; other < assembler->free_register_count; ++other) {
            if (assembler->free_registers[other] == value)
                duplicate = 1;
        }
        if (!duplicate)
            assembler->free_registers[assembler->free_register_count++] = value;
    }
    return ODEZZA_SUCCESS;
}

OdezzaResult o_sass_compile_rhs_set(const OdezzaScoringCubinInspection *inspection, size_t state_capacity,
                                    uint32_t constant_capacity, size_t active_state_count,
                                    uint32_t active_constant_count, uint32_t active_toggle_count,
                                    const OdezzaScoringRhs *rhs, size_t rhs_count,
                                    uint32_t incoming_wait_mask, OdezzaScoringInstruction *instructions,
                                    size_t instruction_capacity, size_t *instruction_count_ret,
                                    uint32_t *high_water_register_ret) {
    size_t index;
    size_t count = 0u;
    uint32_t high_water = inspection->register_count;
    for (index = 0u; index < rhs_count; ++index) {
        OAssembler assembler;
        OdezzaResult result;
        O_TRY(o_assembler_init(
            &assembler, inspection, state_capacity, constant_capacity, active_state_count,
            active_constant_count, active_toggle_count, inspection->output_registers[rhs[index].state_index],
            index == 0u ? incoming_wait_mask : 0u, instructions + count, instruction_capacity - count));
        result = o_compile_program(&assembler, rhs[index].program);
        if (result != ODEZZA_SUCCESS)
            return result;
        count += assembler.writer.count;
        if (high_water < assembler.high_water_register)
            high_water = assembler.high_water_register;
    }
    *instruction_count_ret = count;
    *high_water_register_ret = high_water;
    return ODEZZA_SUCCESS;
}

/* Shared DAG lowering for analytic LM output groups. Register/barrier encoding
 * is identical to scoring; remaining-use counts preserve common subexpressions. */
typedef struct OGroup {
    OOperand values[O_LM_MAX_NODES];
    uint32_t remaining[O_LM_MAX_NODES];
    unsigned char reachable[O_LM_MAX_NODES];
} OGroup;
static void o_group_visit(OGroup *g, const OLmExpressions *c, uint32_t x) {
    uint32_t i;
    if (g->reachable[x])
        return;
    g->reachable[x] = 1;
    for (i = 0; i < c->nodes[x].arity; ++i) {
        uint32_t a = c->nodes[x].child[i];
        ++g->remaining[a];
        o_group_visit(g, c, a);
    }
}
static OdezzaResult o_group_operation(OAssembler *a, const OLmNode *n) {
    OOperand v;
    uint8_t bits[2] = {(uint8_t)n->value, (uint8_t)(n->value >> 8)};
    switch (n->opcode) {
    case ODEZZA_AST_ADD_F32:
        return o_add_or_multiply(a, 0);
    case ODEZZA_AST_MUL_F32:
        return o_add_or_multiply(a, 1);
    case ODEZZA_AST_SUB_F32:
        return o_subtract(a);
    case ODEZZA_AST_DIV_F32:
        return o_divide(a);
    case ODEZZA_AST_FMA_F32:
        return o_fused_multiply_add(a);
    case ODEZZA_AST_TOGGLE2_F32:
        return o_compile_toggle(a, bits, 1);
    case ODEZZA_AST_TOGGLE4_F32:
        return o_compile_toggle(a, bits, 2);
    case ODEZZA_AST_EXP_F32:
    case ODEZZA_AST_LOG_F32:
        if (n->opcode == ODEZZA_AST_LOG_F32)
            O_TRY(o_unary(a, ODEZZA_AST_LG2_F32));
        memset(&v, 0, sizeof(v));
        v.barrier_slot = -1;
        v.immediate = 1;
        v.immediate_bits = n->opcode == ODEZZA_AST_EXP_F32 ? O_LOG2_E_BITS : O_LN_TWO_BITS;
        O_TRY(o_push(a, v));
        O_TRY(o_add_or_multiply(a, 1));
        return n->opcode == ODEZZA_AST_EXP_F32 ? o_unary(a, ODEZZA_AST_EX2_F32) : ODEZZA_SUCCESS;
    default:
        return o_unary(a, (OdezzaAstOpcode)n->opcode);
    }
}
OdezzaResult o_sass_compile_group(const OdezzaScoringCubinInspection *p, const OLmExpressions *c,
                                  const uint32_t *roots, size_t count, uint32_t bits, OSassInstruction *out,
                                  size_t capacity, size_t *used, uint32_t *high) {
    OGroup g;
    OAssembler a;
    uint32_t i, j;
    size_t k;
    memset(&g, 0, sizeof(g));
    O_TRY(o_assembler_init(&a, p, p->state_capacity, p->constant_capacity, p->state_capacity,
                           p->constant_capacity, bits, 255, 0, out, capacity));
    for (k = 0; k < count; ++k) {
        if (roots[k] >= c->count)
            return ODEZZA_ERROR_AST;
        ++g.remaining[roots[k]];
        o_group_visit(&g, c, roots[k]);
    }
    for (i = 0; i < c->count; ++i) {
        const OLmNode *n = &c->nodes[i];
        OOperand v;
        if (!g.reachable[i])
            continue;
        if (!n->arity) {
            memset(&v, 0, sizeof(v));
            v.barrier_slot = -1;
            if (n->opcode == ODEZZA_AST_LITERAL_F32) {
                v.immediate = 1;
                v.immediate_bits = n->value;
            } else
                O_TRY(o_source_operand(&a, n->opcode == ODEZZA_AST_CONSTANT_F32, (uint8_t)n->value, &v));
        } else {
            for (j = 0; j < n->arity; ++j) {
                uint32_t child = n->child[j];
                OOperand value = g.values[child];
                if (child >= i || !g.remaining[child])
                    return ODEZZA_ERROR_AST;
                --g.remaining[child];
                if (g.remaining[child])
                    value.owned = 0;
                O_TRY(o_push(&a, value));
            }
            O_TRY(o_group_operation(&a, n));
            O_TRY(o_pop(&a, &v));
            /* Non-owned output may alias a live child (for example identity
             * simplification); an extra root reference must retain that register.
             * Current arithmetic operators always allocate or transfer ownership. */
        }
        g.values[i] = v;
    }
    for (k = 0; k < count; ++k) {
        uint32_t root = roots[k];
        OOperand v = g.values[root];
        if (v.immediate)
            O_TRY(o_emit(&a, o_sass_fadd_immediate(p->output_registers[k], 255, v.immediate_bits, 0)));
        else
            O_TRY(o_emit(&a, o_sass_mov(p->output_registers[k], v.reg, o_wait_mask_for(&a, &v, 1))));
        if (--g.remaining[root] == 0)
            o_release(&a, v, NULL);
    }
    {
        uint32_t mask = a.writer.pending_wait_mask;
        for (i = 0; i < O_BARRIER_SLOTS; ++i)
            if (a.active_barrier_tokens[i])
                mask |= 1u << i;
        if (mask)
            O_TRY(o_emit(&a, o_sass_nop(mask)));
    }
    *used = a.writer.count;
    *high = a.high_water_register;
    return ODEZZA_SUCCESS;
}
