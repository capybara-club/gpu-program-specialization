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
#include "internal.h"

typedef struct _SecantHsacoOperand {
    uint8_t reg;
    uint8_t owned;
} _SecantHsacoOperand;

typedef struct _SecantHsacoAssembler {
    unsigned char* code;
    size_t capacity;
    size_t count;
    uint8_t free_registers[_SECANT_HSACO_MAX_REGISTERS];
    size_t num_free_registers;
    uint32_t next_register;
    uint32_t high_water_register;
    _SecantHsacoOperand stack[_SECANT_HSACO_STACK_DEPTH];
    size_t stack_size;
    const SecantAstInstruction* const* routines;
    size_t num_routines;
} _SecantHsacoAssembler;

static SecantResult
_secant_hsaco_routine_num_args(
    const SecantAstInstruction* instructions,
    size_t* num_args_ret
) {
    size_t instruction_idx;
    size_t num_args = 0u;

    if (instructions == NULL || num_args_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction =
            instructions + instruction_idx;
        const SecantAstInstructionType instruction_type =
            (SecantAstInstructionType)instruction->instruction_type;

        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            if (instruction->payload.idx >=
                SECANT_AST_MAX_INSTRUCTION_ARGS) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
            }
            if ((size_t)instruction->payload.idx + 1u > num_args) {
                num_args = (size_t)instruction->payload.idx + 1u;
            }
        } else if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_SUCCESS;
        }
    }
    _SECANT_HSACO_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
}

static SecantResult
_secant_hsaco_emit_u32(
    _SecantHsacoAssembler* assembler,
    uint32_t word
) {
    if (assembler->count > assembler->capacity ||
        sizeof(word) > assembler->capacity - assembler->count) {
        _SECANT_HSACO_ERROR_RET(
            SECANT_ERROR_INSUFFICIENT_PATCH_SPACE);
    }
    _secant_hsaco_write_u32(
        assembler->code + assembler->count,
        word);
    assembler->count += sizeof(word);
    return SECANT_SUCCESS;
}

static uint32_t
_secant_hsaco_vop2(
    uint32_t opcode,
    uint8_t dst,
    uint8_t src0,
    uint8_t src1
) {
    return opcode |
        ((uint32_t)dst << 17u) |
        ((uint32_t)src1 << 9u) |
        (_SECANT_HSACO_VGPR_SOURCE + src0);
}

static uint32_t
_secant_hsaco_vop1(
    uint32_t opcode,
    uint8_t dst,
    uint8_t src
) {
    return _SECANT_HSACO_VOP1 |
        ((uint32_t)dst << 17u) |
        (opcode << 9u) |
        (_SECANT_HSACO_VGPR_SOURCE + src);
}

static SecantResult
_secant_hsaco_emit_vop2(
    _SecantHsacoAssembler* assembler,
    uint32_t opcode,
    uint8_t dst,
    uint8_t lhs,
    uint8_t rhs
) {
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_u32(
        assembler,
        _secant_hsaco_vop2(opcode, dst, lhs, rhs)));
}

static SecantResult
_secant_hsaco_emit_vop2_literal(
    _SecantHsacoAssembler* assembler,
    uint32_t opcode,
    uint8_t dst,
    uint32_t literal,
    uint8_t rhs
) {
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_u32(
        assembler,
        opcode |
            ((uint32_t)dst << 17u) |
            ((uint32_t)rhs << 9u) |
            _SECANT_HSACO_LITERAL_SOURCE));
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_u32(
        assembler,
        literal));
}

static SecantResult
_secant_hsaco_emit_vop3(
    _SecantHsacoAssembler* assembler,
    uint32_t opcode,
    uint8_t dst,
    uint8_t src0,
    uint8_t src1,
    uint8_t src2
) {
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_u32(
        assembler,
        opcode | dst));
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_u32(
        assembler,
        (_SECANT_HSACO_VGPR_SOURCE + src0) |
            ((_SECANT_HSACO_VGPR_SOURCE + src1) << 9u) |
            ((_SECANT_HSACO_VGPR_SOURCE + src2) << 18u)));
}

static SecantResult
_secant_hsaco_emit_vop1(
    _SecantHsacoAssembler* assembler,
    uint32_t opcode,
    uint8_t dst,
    uint8_t src
) {
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_u32(
        assembler,
        _secant_hsaco_vop1(opcode, dst, src)));
}

static SecantResult
_secant_hsaco_emit_mov_literal(
    _SecantHsacoAssembler* assembler,
    uint8_t dst,
    uint32_t bits
) {
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_u32(
        assembler,
        _SECANT_HSACO_VOP1 |
            ((uint32_t)dst << 17u) |
            (_SECANT_HSACO_VOP1_MOV << 9u) |
            _SECANT_HSACO_LITERAL_SOURCE));
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_u32(assembler, bits));
}

static SecantResult
_secant_hsaco_allocate_register(
    _SecantHsacoAssembler* assembler,
    uint8_t* register_ret
) {
    uint32_t reg;

    if (assembler->num_free_registers != 0u) {
        *register_ret =
            assembler->free_registers[--assembler->num_free_registers];
        return SECANT_SUCCESS;
    }
    reg = assembler->next_register;
    if (reg >= _SECANT_HSACO_MAX_REGISTERS) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_REGISTER_OVERFLOW);
    }
    assembler->next_register += 1u;
    if (assembler->next_register > assembler->high_water_register) {
        assembler->high_water_register = assembler->next_register;
    }
    *register_ret = (uint8_t)reg;
    return SECANT_SUCCESS;
}

static void
_secant_hsaco_release_operand(
    _SecantHsacoAssembler* assembler,
    _SecantHsacoOperand operand,
    uint8_t retained_reg
) {
    if (operand.owned && operand.reg != retained_reg &&
        assembler->num_free_registers <
            _SECANT_HSACO_MAX_REGISTERS) {
        assembler->free_registers[
            assembler->num_free_registers++] = operand.reg;
    }
}

static SecantResult
_secant_hsaco_push(
    _SecantHsacoAssembler* assembler,
    _SecantHsacoOperand operand
) {
    if (assembler->stack_size >= _SECANT_HSACO_STACK_DEPTH) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_STACK_OVERFLOW);
    }
    assembler->stack[assembler->stack_size++] = operand;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_pop(
    _SecantHsacoAssembler* assembler,
    _SecantHsacoOperand* operand_ret
) {
    if (assembler->stack_size == 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_STACK_UNDERFLOW);
    }
    *operand_ret = assembler->stack[--assembler->stack_size];
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_operation_destination(
    _SecantHsacoAssembler* assembler,
    const _SecantHsacoOperand* operands,
    size_t num_operands,
    uint8_t* destination_ret
) {
    size_t operand_idx;

    for (operand_idx = 0u; operand_idx < num_operands; ++operand_idx) {
        if (operands[operand_idx].owned) {
            *destination_ret = operands[operand_idx].reg;
            return SECANT_SUCCESS;
        }
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_allocate_register(
        assembler,
        destination_ret));
}

static SecantResult
_secant_hsaco_emit_operation(
    _SecantHsacoAssembler* assembler,
    SecantAstInstructionType instruction_type,
    const _SecantHsacoOperand* operands,
    uint8_t dst
) {
    union {
        float f;
        uint32_t bits;
    } constant;
    uint8_t temporary;

    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2(
                assembler,
                _SECANT_HSACO_V_ADD_F32,
                dst,
                operands[0].reg,
                operands[1].reg));
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2(
                assembler,
                _SECANT_HSACO_V_SUB_F32,
                dst,
                operands[0].reg,
                operands[1].reg));
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2(
                assembler,
                _SECANT_HSACO_V_MUL_F32,
                dst,
                operands[0].reg,
                operands[1].reg));
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2(
                assembler,
                _SECANT_HSACO_V_MIN_F32,
                dst,
                operands[0].reg,
                operands[1].reg));
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2(
                assembler,
                _SECANT_HSACO_V_MAX_F32,
                dst,
                operands[0].reg,
                operands[1].reg));
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_XOR_B32,
                dst,
                0x80000000u,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_AND_B32,
                dst,
                0x7fffffffu,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_SQRT,
                dst,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_RCP,
                dst,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_RSQ,
                dst,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_EXP,
                dst,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_LOG,
                dst,
                operands[0].reg));
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
            constant.f = 0.15915494309189535f;
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_MUL_F32,
                dst,
                constant.bits,
                operands[0].reg));
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop1(
                assembler,
                instruction_type == SECANT_AST_INSTRUCTION_TYPE_SIN_F32
                    ? _SECANT_HSACO_VOP1_SIN
                    : _SECANT_HSACO_VOP1_COS,
                dst,
                dst));
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_allocate_register(
                assembler,
                &temporary));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_RCP,
                temporary,
                operands[1].reg));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop2(
                assembler,
                _SECANT_HSACO_V_MUL_F32,
                dst,
                operands[0].reg,
                temporary));
            assembler->free_registers[
                assembler->num_free_registers++] = temporary;
            return SECANT_SUCCESS;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop3(
                assembler,
                _SECANT_HSACO_V_FMA_F32,
                dst,
                operands[0].reg,
                operands[1].reg,
                operands[2].reg));
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
            constant.f = -2.8853900817779268f;
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_MUL_F32,
                dst,
                constant.bits,
                operands[0].reg));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_EXP,
                dst,
                dst));
            constant.f = 1.0f;
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_ADD_F32,
                dst,
                constant.bits,
                dst));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop1(
                assembler,
                _SECANT_HSACO_VOP1_RCP,
                dst,
                dst));
            constant.f = 2.0f;
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_MUL_F32,
                dst,
                constant.bits,
                dst));
            constant.f = -1.0f;
            _SECANT_HSACO_ERROR_RET(_secant_hsaco_emit_vop2_literal(
                assembler,
                _SECANT_HSACO_V_ADD_F32,
                dst,
                constant.bits,
                dst));
        default:
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_UNSUPPORTED_OP);
    }
}

static SecantResult
_secant_hsaco_compile_frame(
    _SecantHsacoAssembler* assembler,
    const uint8_t* input_registers,
    size_t num_input_registers,
    const SecantAstInstruction* instructions,
    const _SecantHsacoOperand* routine_args,
    size_t num_routine_args,
    size_t routine_depth,
    _SecantHsacoOperand* result_ret
) {
    const size_t frame_stack_base = assembler->stack_size;
    size_t instruction_idx;

    if (instructions == NULL || result_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (routine_depth > _SECANT_HSACO_ROUTINE_DEPTH) {
        _SECANT_HSACO_ERROR_RET(
            SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED);
    }
    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction =
            instructions + instruction_idx;
        const SecantAstInstructionType instruction_type =
            (SecantAstInstructionType)instruction->instruction_type;

        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32) {
            if (instruction->payload.idx >= num_input_registers) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_BAD_PROGRAM);
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_push(
                assembler,
                (_SecantHsacoOperand){
                    input_registers[instruction->payload.idx],
                    0u
                }));
        } else if (
            instruction_type ==
                SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            uint8_t reg;

            _SECANT_HSACO_CHECK_RET(_secant_hsaco_allocate_register(
                assembler,
                &reg));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_mov_literal(
                assembler,
                reg,
                secant_ast_constant_f32_bits_get(instruction)));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_push(
                assembler,
                (_SecantHsacoOperand){ reg, 1u }));
        } else if (
            instruction_type ==
                SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            if (routine_args == NULL ||
                instruction->payload.idx >= num_routine_args) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_push(
                assembler,
                (_SecantHsacoOperand){
                    routine_args[instruction->payload.idx].reg,
                    0u
                }));
        } else if (
            instruction_type ==
                SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
            _SecantHsacoOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
            _SecantHsacoOperand routine_result;
            size_t arg_idx;
            size_t expected_args;

            if (instruction->payload.idx >= assembler->num_routines ||
                assembler->routines == NULL ||
                assembler->routines[instruction->payload.idx] == NULL) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
            }
            if (instruction->aux > SECANT_AST_MAX_INSTRUCTION_ARGS) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_TOO_MANY_ARGS);
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_routine_num_args(
                assembler->routines[instruction->payload.idx],
                &expected_args));
            if ((size_t)instruction->aux != expected_args) {
                _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
            }
            for (arg_idx = instruction->aux;
                 arg_idx != 0u;
                 --arg_idx) {
                _SECANT_HSACO_CHECK_RET(_secant_hsaco_pop(
                    assembler,
                    args + arg_idx - 1u));
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_compile_frame(
                assembler,
                input_registers,
                num_input_registers,
                assembler->routines[instruction->payload.idx],
                args,
                instruction->aux,
                routine_depth + 1u,
                &routine_result));
            for (arg_idx = 0u;
                 arg_idx < instruction->aux;
                 ++arg_idx) {
                _secant_hsaco_release_operand(
                    assembler,
                    args[arg_idx],
                    routine_result.reg);
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_push(
                assembler,
                routine_result));
        } else if (
            instruction_type ==
                SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            if (assembler->stack_size != frame_stack_base + 1u) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_BAD_PROGRAM);
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_pop(
                assembler,
                result_ret));
            return SECANT_SUCCESS;
        } else {
            _SecantHsacoOperand operands[
                SECANT_AST_MAX_INSTRUCTION_ARGS];
            const size_t num_operands =
                instruction_type < SECANT_AST_INSTRUCTION_TYPE_NUM_ENUMS &&
                secant_ast_instruction_value_type[instruction_type] ==
                    SECANT_AST_VALUE_TYPE_F32
                    ? secant_ast_instruction_num_args[instruction_type]
                    : 0u;
            uint8_t destination;
            size_t operand_idx;

            if (num_operands == 0u ||
                num_operands > SECANT_AST_MAX_INSTRUCTION_ARGS) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_UNSUPPORTED_OP);
            }
            for (operand_idx = num_operands;
                 operand_idx != 0u;
                 --operand_idx) {
                _SECANT_HSACO_CHECK_RET(_secant_hsaco_pop(
                    assembler,
                    operands + operand_idx - 1u));
            }
            _SECANT_HSACO_CHECK_RET(
                _secant_hsaco_operation_destination(
                    assembler,
                    operands,
                    num_operands,
                    &destination));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_operation(
                assembler,
                instruction_type,
                operands,
                destination));
            for (operand_idx = 0u;
                 operand_idx < num_operands;
                 ++operand_idx) {
                _secant_hsaco_release_operand(
                    assembler,
                    operands[operand_idx],
                    destination);
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_push(
                assembler,
                (_SecantHsacoOperand){ destination, 1u }));
        }
    }
    _SECANT_HSACO_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
}

static void
_secant_hsaco_prepare_assembler(
    _SecantHsacoAssembler* assembler,
    unsigned char* code,
    size_t capacity,
    const uint8_t* available_registers,
    size_t num_available_registers,
    uint32_t first_new_register,
    const SecantAstInstruction* const* routines,
    size_t num_routines
) {
    size_t register_idx;

    memset(assembler, 0, sizeof(*assembler));
    assembler->code = code;
    assembler->capacity = capacity;
    assembler->next_register = first_new_register;
    assembler->high_water_register = first_new_register;
    assembler->routines = routines;
    assembler->num_routines = num_routines;
    for (register_idx = 0u;
         register_idx < num_available_registers;
         ++register_idx) {
        assembler->free_registers[
            assembler->num_free_registers++] =
            available_registers[register_idx];
    }
}

static SecantResult
_secant_hsaco_generate_materialize_ast(
    const _SecantHsacoSite* site,
    uint32_t first_new_register,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    uint8_t result_register,
    unsigned char* code,
    size_t capacity,
    size_t* code_size_ret,
    uint32_t* expanded_register_count_ret
) {
    _SecantHsacoAssembler assembler;
    _SecantHsacoOperand result;

    if (site == NULL || ast == NULL || code == NULL ||
        code_size_ret == NULL ||
        expanded_register_count_ret == NULL ||
        (num_routines != 0u && routines == NULL)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _secant_hsaco_prepare_assembler(
        &assembler,
        code,
        capacity,
        site->available_regs,
        site->num_available_regs,
        first_new_register,
        routines,
        num_routines);
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_compile_frame(
        &assembler,
        site->input_regs,
        site->num_input_regs,
        ast,
        NULL,
        0u,
        0u,
        &result));
    if (result.reg != result_register) {
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop1(
            &assembler,
            _SECANT_HSACO_VOP1_MOV,
            result_register,
            result.reg));
    }
    *code_size_ret = assembler.count;
    *expanded_register_count_ret = assembler.high_water_register;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_generate_sse(
    const _SecantHsacoSite* site,
    uint32_t first_new_register,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    unsigned char* code,
    size_t capacity,
    size_t* code_size_ret,
    uint32_t* expanded_register_count_ret
) {
    _SecantHsacoAssembler assembler;
    size_t expected_outputs;
    size_t ast_idx;

    if (site == NULL || asts == NULL || num_asts == 0u ||
        code == NULL || code_size_ret == NULL ||
        expanded_register_count_ret == NULL ||
        (num_routines != 0u && routines == NULL) ||
        !_secant_hsaco_checked_mul(
            num_asts,
            site->num_target_regs,
            &expected_outputs) ||
        expected_outputs != site->num_output_regs) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _secant_hsaco_prepare_assembler(
        &assembler,
        code,
        capacity,
        site->available_regs,
        site->num_available_regs,
        first_new_register,
        routines,
        num_routines);
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        _SecantHsacoOperand prediction;
        size_t target_idx;

        if (asts[ast_idx] == NULL || assembler.stack_size != 0u) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_compile_frame(
            &assembler,
            site->input_regs,
            site->num_input_regs,
            asts[ast_idx],
            NULL,
            0u,
            0u,
            &prediction));
        for (target_idx = 0u;
             target_idx < site->num_target_regs;
             ++target_idx) {
            const int reuse_prediction =
                target_idx + 1u == site->num_target_regs &&
                prediction.owned;
            uint8_t error_reg;
            const uint8_t sse_reg = site->output_regs[
                ast_idx * site->num_target_regs + target_idx];

            if (reuse_prediction) {
                error_reg = prediction.reg;
            } else {
                _SECANT_HSACO_CHECK_RET(
                    _secant_hsaco_allocate_register(
                        &assembler,
                        &error_reg));
            }
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop2(
                &assembler,
                _SECANT_HSACO_V_SUB_F32,
                error_reg,
                prediction.reg,
                site->target_regs[target_idx]));
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_emit_vop3(
                &assembler,
                _SECANT_HSACO_V_FMA_F32,
                sse_reg,
                error_reg,
                error_reg,
                sse_reg));
            assembler.free_registers[
                assembler.num_free_registers++] = error_reg;
            if (reuse_prediction) {
                prediction.owned = 0u;
            }
        }
        if (prediction.owned) {
            assembler.free_registers[
                assembler.num_free_registers++] = prediction.reg;
        }
    }
    if (assembler.stack_size != 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
    }
    *code_size_ret = assembler.count;
    *expanded_register_count_ret = assembler.high_water_register;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_write_metadata_vgpr_count(
    const _SecantHsacoKernel* kernel,
    uint32_t register_count,
    size_t hsaco_size,
    unsigned char* hsaco
) {
    size_t encoded_size;

    if (kernel == NULL || hsaco == NULL || register_count == 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    switch (kernel->metadata_vgpr_encoding) {
        case _SECANT_HSACO_MSGPACK_FIXINT:
            if (register_count > 0x7fu) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_REGISTER_OVERFLOW);
            }
            encoded_size = 1u;
            break;
        case _SECANT_HSACO_MSGPACK_UINT8:
            if (register_count > UINT8_MAX) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_REGISTER_OVERFLOW);
            }
            encoded_size = 2u;
            break;
        case _SECANT_HSACO_MSGPACK_UINT16:
            if (register_count > UINT16_MAX) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_REGISTER_OVERFLOW);
            }
            encoded_size = 3u;
            break;
        case _SECANT_HSACO_MSGPACK_UINT32:
            encoded_size = 5u;
            break;
        case _SECANT_HSACO_MSGPACK_UINT64:
            encoded_size = 9u;
            break;
        default:
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_STATE);
    }
    if (!_secant_hsaco_range_ok(
            hsaco_size,
            kernel->metadata_vgpr_file_offset,
            encoded_size)) {
        _SECANT_HSACO_ERROR_RET(
            SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    switch (kernel->metadata_vgpr_encoding) {
        case _SECANT_HSACO_MSGPACK_FIXINT:
            hsaco[kernel->metadata_vgpr_file_offset] =
                (unsigned char)register_count;
            break;
        case _SECANT_HSACO_MSGPACK_UINT8:
            hsaco[kernel->metadata_vgpr_file_offset + 1u] =
                (unsigned char)register_count;
            break;
        case _SECANT_HSACO_MSGPACK_UINT16:
            _secant_hsaco_write_be_u16(
                hsaco + kernel->metadata_vgpr_file_offset + 1u,
                (uint16_t)register_count);
            break;
        case _SECANT_HSACO_MSGPACK_UINT32:
            _secant_hsaco_write_be_u32(
                hsaco + kernel->metadata_vgpr_file_offset + 1u,
                register_count);
            break;
        case _SECANT_HSACO_MSGPACK_UINT64:
            _secant_hsaco_write_be_u64(
                hsaco + kernel->metadata_vgpr_file_offset + 1u,
                register_count);
            break;
        default:
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_STATE);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_patch(
    const _SecantHsacoTemplate* template_data,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    size_t hsaco_size,
    unsigned char* hsaco
) {
    size_t expected_asts;
    size_t kernel_idx;

    if (template_data == NULL || asts == NULL || hsaco == NULL ||
        (num_routines != 0u && routines == NULL) ||
        !_secant_hsaco_checked_mul(
            template_data->num_kernels,
            template_data->asts_per_kernel,
            &expected_asts) ||
        num_asts != expected_asts) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (hsaco_size != template_data->hsaco_size) {
        _SECANT_HSACO_ERROR_RET(
            hsaco_size < template_data->hsaco_size
                ? SECANT_ERROR_INSUFFICIENT_BUFFER
                : SECANT_ERROR_INVALID_VALUE);
    }
    for (kernel_idx = 0u;
         kernel_idx < template_data->num_kernels;
         ++kernel_idx) {
        const _SecantHsacoKernel* kernel =
            template_data->kernels + kernel_idx;
        const _SecantHsacoSite* site = &kernel->site;
        unsigned char* site_code;
        size_t code_size = 0u;
        uint32_t expanded_register_count = kernel->register_count;
        size_t ast_idx;

        if (!_secant_hsaco_range_ok(
                hsaco_size,
                site->start_file_offset,
                site->patch_size) ||
            !_secant_hsaco_range_ok(
                hsaco_size,
                site->load_fence_file_offset,
                sizeof(uint32_t)) ||
            !_secant_hsaco_range_ok(
                hsaco_size,
                kernel->compute_pgm_rsrc1_file_offset,
                sizeof(uint32_t)) ||
            !_secant_hsaco_range_ok(
                hsaco_size,
                kernel->num_vgpr_symbol_value_file_offset,
                sizeof(uint64_t))) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_PATCH_SPACE);
        }
        site_code = hsaco + site->start_file_offset;
        if (template_data->shape == _SECANT_HSACO_SHAPE_SSE ||
            template_data->shape ==
                _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_generate_sse(
                site,
                kernel->register_count,
                routines,
                num_routines,
                asts + kernel_idx * template_data->asts_per_kernel,
                template_data->asts_per_kernel,
                site_code,
                site->patch_size,
                &code_size,
                &expanded_register_count));
        } else {
            for (ast_idx = 0u;
                 ast_idx < template_data->asts_per_kernel;
                 ++ast_idx) {
                const size_t program_idx =
                    kernel_idx * template_data->asts_per_kernel +
                    ast_idx;
                size_t ast_code_size = 0u;
                uint32_t ast_register_count =
                    kernel->register_count;

                if (asts[program_idx] == NULL) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_INVALID_VALUE);
                }
                _SECANT_HSACO_CHECK_RET(
                    _secant_hsaco_generate_materialize_ast(
                        site,
                        kernel->register_count,
                        routines,
                        num_routines,
                        asts[program_idx],
                        site->output_regs[ast_idx],
                        site_code + code_size,
                        site->patch_size - code_size,
                        &ast_code_size,
                        &ast_register_count));
                code_size += ast_code_size;
                if (ast_register_count >
                    expanded_register_count) {
                    expanded_register_count =
                        ast_register_count;
                }
            }
        }
        if (code_size < site->patch_size) {
            const size_t remaining = site->patch_size - code_size;

            if (remaining == sizeof(uint32_t)) {
                _secant_hsaco_write_u32(
                    site_code + code_size,
                    _SECANT_HSACO_NOP_WORD);
            } else if (
                remaining >= 2u * sizeof(uint32_t) &&
                remaining / sizeof(uint32_t) - 1u <= 0x7fffu) {
                _secant_hsaco_write_u32(
                    site_code + code_size,
                    _SECANT_HSACO_BRANCH_WORD |
                        (uint32_t)(
                            remaining / sizeof(uint32_t) - 1u));
            } else {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_INSUFFICIENT_PATCH_SPACE);
            }
            code_size += sizeof(uint32_t);
        }
        if (expanded_register_count == 0u ||
            expanded_register_count >
                _SECANT_HSACO_MAX_REGISTERS) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_REGISTER_OVERFLOW);
        }
        if (code_size > site->patch_size) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_INSUFFICIENT_PATCH_SPACE);
        }
        _SECANT_HSACO_CHECK_RET(
            _secant_hsaco_write_metadata_vgpr_count(
                kernel,
                expanded_register_count,
                hsaco_size,
                hsaco));
        _secant_hsaco_write_u32(
            hsaco + site->load_fence_file_offset,
            _SECANT_HSACO_NOP_WORD);
        {
            const uint32_t encoded_vgprs =
                (expanded_register_count + 7u) / 8u - 1u;
            const uint32_t resource =
                (kernel->compute_pgm_rsrc1 &
                 ~_SECANT_HSACO_COMPUTE_PGM_RSRC1_VGPRS_MASK) |
                encoded_vgprs;

            _secant_hsaco_write_u32(
                hsaco + kernel->compute_pgm_rsrc1_file_offset,
                resource);
            _secant_hsaco_write_u64(
                hsaco +
                    kernel->num_vgpr_symbol_value_file_offset,
                expanded_register_count);
        }
    }
    return SECANT_SUCCESS;
}
SecantResult
secant_hsaco_specialize_into(
    const SecantHsacoPlan* plan,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    void* hsaco,
    size_t hsaco_size
) {
    if (plan == NULL || hsaco == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_patch(
        &plan->template_data,
        routines,
        num_routines,
        asts,
        num_asts,
        hsaco_size,
        (unsigned char*)hsaco));
}

#undef _SECANT_HSACO_CHECK_RET
#undef _SECANT_HSACO_ERROR_RET
