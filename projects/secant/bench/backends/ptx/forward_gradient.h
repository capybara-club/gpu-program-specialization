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
#ifndef SECANT_PTX_FORWARD_GRADIENT_H_INCLUDED
#define SECANT_PTX_FORWARD_GRADIENT_H_INCLUDED

/* Internal analytic forward-mode compiler for an already-lowered AST PTX tree. */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define _SECANT_PTX_FG_MAX_PARAMETERS 8u
#define _SECANT_PTX_FG_MAX_REGISTERS 16384u
#define _SECANT_PTX_FG_MAX_PREDICATES 4096u
#define _SECANT_PTX_FG_ROUTINE_DEPTH 8u

typedef struct _SecantPTXFGValue {
    AstPtxInstruction primal;
    AstPtxInstruction derivative[_SECANT_PTX_FG_MAX_PARAMETERS];
} _SecantPTXFGValue;

typedef struct _SecantPTXFGAssembler {
    _SecantPTXFGValue stack[SECANT_AST_MAX_STACK_DEPTH];
    size_t stack_size;
    size_t num_parameters;
    int mixed_parameter_inputs;
    uint32_t next_register;
    uint32_t next_predicate;
    char* buffer;
    size_t buffer_size;
    size_t offset;
} _SecantPTXFGAssembler;

static SecantPTXResult
_secant_ptx_fg_write(_SecantPTXFGAssembler* assembler, const char* format, ...) {
    va_list args;
    int bytes;

    if (assembler == NULL || format == NULL) {
        return SECANT_PTX_ERROR_INVALID_VALUE;
    }
    va_start(args, format);
    if (assembler->buffer != NULL && assembler->offset < assembler->buffer_size) {
        bytes = vsnprintf(
            assembler->buffer + assembler->offset,
            assembler->buffer_size - assembler->offset,
            format,
            args);
    } else {
        bytes = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (bytes < 0) {
        return SECANT_PTX_ERROR_FORMAT;
    }
    if ((size_t)bytes > SIZE_MAX - assembler->offset) {
        return SECANT_PTX_ERROR_OVERFLOW;
    }
    assembler->offset += (size_t)bytes;
    if (assembler->buffer != NULL && assembler->offset >= assembler->buffer_size) {
        return SECANT_PTX_ERROR_INSUFFICIENT_BUFFER;
    }
    return SECANT_PTX_SUCCESS;
}

static AstPtxInstruction
_secant_ptx_fg_constant(float value) {
    return ast_ptx_encode_constant(value);
}

static SecantPTXResult
_secant_ptx_fg_value_write(
    _SecantPTXFGAssembler* assembler,
    AstPtxInstruction value
) {
    switch ((AstPtxInstructionType)value.instruction_type) {
        case AST_PTX_INSTRUCTION_TYPE_CONSTANT: {
            uint32_t bits;
            memcpy(&bits, &value.payload.f, sizeof(bits));
            return _secant_ptx_fg_write(assembler, "0f%08x", (unsigned)bits);
        }
        case AST_PTX_INSTRUCTION_TYPE_REGISTER:
            return _secant_ptx_fg_write(assembler, "%%ad%u", (unsigned)value.payload.idx);
        case AST_PTX_INSTRUCTION_TYPE_INPUT:
        default:
            return SECANT_PTX_ERROR_INVALID_STATE;
    }
}

/* INPUT payloads cannot hold a pointer portably; resolve them before emission. */
static SecantPTXResult
_secant_ptx_fg_operand_write(
    _SecantPTXFGAssembler* assembler,
    AstPtxInstruction value,
    const char* const* input_register_names,
    size_t num_inputs
) {
    if ((AstPtxInstructionType)value.instruction_type == AST_PTX_INSTRUCTION_TYPE_INPUT) {
        if (value.payload.idx >= num_inputs || input_register_names[value.payload.idx] == NULL) {
            return SECANT_PTX_ERROR_INVALID_VALUE;
        }
        return _secant_ptx_fg_write(
            assembler, "%%%s", input_register_names[value.payload.idx]);
    }
    return _secant_ptx_fg_value_write(assembler, value);
}

static SecantPTXResult
_secant_ptx_fg_emit_operation(
    _SecantPTXFGAssembler* assembler,
    const char* operation,
    const AstPtxInstruction* args,
    size_t num_args,
    const char* const* input_register_names,
    size_t num_inputs,
    AstPtxInstruction* result_ret
) {
    size_t arg_idx;
    AstPtxInstruction result;

    if (assembler->next_register >= _SECANT_PTX_FG_MAX_REGISTERS) {
        return SECANT_PTX_ERROR_STACK_OVERFLOW;
    }
    result = ast_ptx_encode_register(assembler->next_register++);
    if (_secant_ptx_fg_write(
            assembler, "\t%s %%ad%u", operation, (unsigned)result.payload.idx) !=
        SECANT_PTX_SUCCESS) {
        return SECANT_PTX_ERROR_INSUFFICIENT_BUFFER;
    }
    for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
        SecantPTXResult result_code = _secant_ptx_fg_write(assembler, ", ");
        if (result_code != SECANT_PTX_SUCCESS) {
            return result_code;
        }
        result_code = _secant_ptx_fg_operand_write(
            assembler, args[arg_idx], input_register_names, num_inputs);
        if (result_code != SECANT_PTX_SUCCESS) {
            return result_code;
        }
    }
    if (_secant_ptx_fg_write(assembler, ";\n") != SECANT_PTX_SUCCESS) {
        return SECANT_PTX_ERROR_INSUFFICIENT_BUFFER;
    }
    *result_ret = result;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_fg_emit_predicate(
    _SecantPTXFGAssembler* assembler,
    const char* comparison,
    AstPtxInstruction lhs,
    AstPtxInstruction rhs,
    const char* const* input_register_names,
    size_t num_inputs,
    uint32_t* predicate_ret
) {
    SecantPTXResult result;
    const uint32_t predicate = assembler->next_predicate++;

    if (predicate >= _SECANT_PTX_FG_MAX_PREDICATES) {
        return SECANT_PTX_ERROR_STACK_OVERFLOW;
    }
    result = _secant_ptx_fg_write(
        assembler, "\tsetp.%s.f32 %%adp%u, ", comparison, (unsigned)predicate);
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_operand_write(assembler, lhs, input_register_names, num_inputs);
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_write(assembler, ", ");
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_operand_write(assembler, rhs, input_register_names, num_inputs);
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_write(assembler, ";\n");
    if (result != SECANT_PTX_SUCCESS) return result;
    *predicate_ret = predicate;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_fg_emit_select(
    _SecantPTXFGAssembler* assembler,
    AstPtxInstruction when_true,
    AstPtxInstruction when_false,
    uint32_t predicate,
    const char* const* input_register_names,
    size_t num_inputs,
    AstPtxInstruction* result_ret
) {
    SecantPTXResult result;
    AstPtxInstruction output;

    if (assembler->next_register >= _SECANT_PTX_FG_MAX_REGISTERS) {
        return SECANT_PTX_ERROR_STACK_OVERFLOW;
    }
    output = ast_ptx_encode_register(assembler->next_register++);
    result = _secant_ptx_fg_write(
        assembler, "\tselp.f32 %%ad%u, ", (unsigned)output.payload.idx);
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_operand_write(assembler, when_true, input_register_names, num_inputs);
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_write(assembler, ", ");
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_operand_write(assembler, when_false, input_register_names, num_inputs);
    if (result != SECANT_PTX_SUCCESS) return result;
    result = _secant_ptx_fg_write(assembler, ", %%adp%u;\n", (unsigned)predicate);
    if (result != SECANT_PTX_SUCCESS) return result;
    *result_ret = output;
    return SECANT_PTX_SUCCESS;
}

#define S_PTX_FG_CHECK(call) do { \
    SecantPTXResult _fg_result = (call); \
    if (_fg_result != SECANT_PTX_SUCCESS) return _fg_result; \
} while (0)

static SecantPTXResult
_secant_ptx_fg_derivative_emit(
    _SecantPTXFGAssembler* assembler,
    AstPtxPtxInstruction operation,
    const _SecantPTXFGValue* args,
    AstPtxInstruction primal,
    size_t derivative_idx,
    const char* const* input_register_names,
    size_t num_inputs,
    AstPtxInstruction* derivative_ret
) {
    AstPtxInstruction operands[3];
    AstPtxInstruction temp0;
    AstPtxInstruction temp1;
    AstPtxInstruction temp2;
    uint32_t predicate;

    switch (operation) {
        case AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32:
        case AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32:
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = args[1].derivative[derivative_idx];
            return _secant_ptx_fg_emit_operation(
                assembler,
                operation == AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32
                    ? "add.ftz.f32" : "sub.ftz.f32",
                operands, 2u, input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32:
            operands[0] = args[0].primal;
            operands[1] = args[1].derivative[derivative_idx];
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = args[1].primal;
            operands[2] = temp0;
            return _secant_ptx_fg_emit_operation(
                assembler, "fma.rn.ftz.f32", operands, 3u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32:
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = args[1].primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = args[0].primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "neg.ftz.f32", operands, 1u,
                input_register_names, num_inputs, &temp1));
            operands[0] = temp1;
            operands[1] = args[1].derivative[derivative_idx];
            operands[2] = temp0;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "fma.rn.ftz.f32", operands, 3u,
                input_register_names, num_inputs, &temp1));
            operands[0] = args[1].primal;
            operands[1] = args[1].primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp2));
            operands[0] = temp1;
            operands[1] = temp2;
            return _secant_ptx_fg_emit_operation(
                assembler, "div.approx.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32:
            operands[0] = args[0].derivative[derivative_idx];
            return _secant_ptx_fg_emit_operation(
                assembler, "neg.ftz.f32", operands, 1u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32:
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "div.approx.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = _secant_ptx_fg_constant(0.5f);
            operands[1] = temp0;
            return _secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32:
            operands[0] = primal;
            operands[1] = primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = temp0;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp1));
            operands[0] = temp1;
            return _secant_ptx_fg_emit_operation(
                assembler, "neg.ftz.f32", operands, 1u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32:
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_predicate(
                assembler, "lt", args[0].primal, _secant_ptx_fg_constant(0.0f),
                input_register_names, num_inputs, &predicate));
            operands[0] = args[0].derivative[derivative_idx];
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "neg.ftz.f32", operands, 1u,
                input_register_names, num_inputs, &temp0));
            return _secant_ptx_fg_emit_select(
                assembler, temp0, args[0].derivative[derivative_idx], predicate,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32:
        case AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32:
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_predicate(
                assembler,
                operation == AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32 ? "le" : "ge",
                args[0].primal, args[1].primal,
                input_register_names, num_inputs, &predicate));
            return _secant_ptx_fg_emit_select(
                assembler,
                args[0].derivative[derivative_idx],
                args[1].derivative[derivative_idx],
                predicate, input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32:
            operands[0] = args[0].primal;
            operands[1] = args[1].derivative[derivative_idx];
            operands[2] = args[2].derivative[derivative_idx];
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "fma.rn.ftz.f32", operands, 3u,
                input_register_names, num_inputs, &temp0));
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = args[1].primal;
            operands[2] = temp0;
            return _secant_ptx_fg_emit_operation(
                assembler, "fma.rn.ftz.f32", operands, 3u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32:
        case AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32:
            operands[0] = args[0].primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler,
                operation == AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32
                    ? "cos.approx.ftz.f32" : "sin.approx.ftz.f32",
                operands, 1u, input_register_names, num_inputs, &temp0));
            if (operation == AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32) {
                operands[0] = temp0;
                S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                    assembler, "neg.ftz.f32", operands, 1u,
                    input_register_names, num_inputs, &temp0));
            }
            operands[0] = temp0;
            operands[1] = args[0].derivative[derivative_idx];
            return _secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32:
            operands[0] = _secant_ptx_fg_constant(0.6931471805599453f);
            operands[1] = primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = temp0;
            operands[1] = args[0].derivative[derivative_idx];
            return _secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32:
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = args[0].primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "div.approx.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = _secant_ptx_fg_constant(1.4426950408889634f);
            operands[1] = temp0;
            return _secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32:
            operands[0] = primal;
            operands[1] = primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = primal;
            operands[1] = temp0;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp1));
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = temp1;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp2));
            operands[0] = _secant_ptx_fg_constant(-0.5f);
            operands[1] = temp2;
            return _secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        case AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32:
            operands[0] = primal;
            operands[1] = primal;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp0));
            operands[0] = _secant_ptx_fg_constant(1.0f);
            operands[1] = temp0;
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler, "sub.ftz.f32", operands, 2u,
                input_register_names, num_inputs, &temp1));
            operands[0] = args[0].derivative[derivative_idx];
            operands[1] = temp1;
            return _secant_ptx_fg_emit_operation(
                assembler, "mul.ftz.f32", operands, 2u,
                input_register_names, num_inputs, derivative_ret);
        default:
            return SECANT_PTX_ERROR_UNSUPPORTED_OP;
    }
}

static SecantPTXResult
_secant_ptx_fg_compile_frame(
    _SecantPTXFGAssembler* assembler,
    const char* const* input_register_names,
    size_t num_inputs,
    const AstPtxInstruction* const* routines,
    size_t num_routines,
    const AstPtxInstruction* instructions,
    const _SecantPTXFGValue* routine_args,
    size_t num_routine_args,
    size_t frame_depth
) {
    const size_t base_stack_size = assembler->stack_size;
    size_t instruction_idx;

    for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
        const AstPtxInstruction instruction = instructions[instruction_idx];
        const AstPtxInstructionType type = (AstPtxInstructionType)instruction.instruction_type;

        if (type == AST_PTX_INSTRUCTION_TYPE_INPUT ||
            type == AST_PTX_INSTRUCTION_TYPE_CONSTANT) {
            _SecantPTXFGValue value;
            size_t derivative_idx;

            if (assembler->stack_size >= SECANT_AST_MAX_STACK_DEPTH) {
                return SECANT_PTX_ERROR_STACK_OVERFLOW;
            }
            memset(&value, 0, sizeof(value));
            value.primal = instruction;
            for (derivative_idx = 0u; derivative_idx < assembler->num_parameters; ++derivative_idx) {
                if (assembler->mixed_parameter_inputs &&
                    type == AST_PTX_INSTRUCTION_TYPE_INPUT) {
                    const size_t static_inputs =
                        num_inputs - 3u * assembler->num_parameters;
                    const size_t raw_input = static_inputs + derivative_idx;
                    const size_t mixed_input =
                        static_inputs + assembler->num_parameters + derivative_idx;
                    const size_t mixed_gradient_input =
                        static_inputs + 2u * assembler->num_parameters + derivative_idx;

                    value.derivative[derivative_idx] =
                        instruction.payload.idx == raw_input
                            ? _secant_ptx_fg_constant(1.0f)
                            : instruction.payload.idx == mixed_input
                                ? ast_ptx_encode_input(mixed_gradient_input)
                                : _secant_ptx_fg_constant(0.0f);
                } else {
                    const int active = type == AST_PTX_INSTRUCTION_TYPE_INPUT &&
                        instruction.payload.idx + assembler->num_parameters >= num_inputs &&
                        instruction.payload.idx ==
                            num_inputs - assembler->num_parameters + derivative_idx;
                    value.derivative[derivative_idx] =
                        _secant_ptx_fg_constant(active ? 1.0f : 0.0f);
                }
            }
            assembler->stack[assembler->stack_size++] = value;
        } else if (type == AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG) {
            if (routine_args == NULL || instruction.payload.idx >= num_routine_args ||
                assembler->stack_size >= SECANT_AST_MAX_STACK_DEPTH) {
                return SECANT_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS;
            }
            assembler->stack[assembler->stack_size++] = routine_args[instruction.payload.idx];
        } else if (type == AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION) {
            _SecantPTXFGValue args[3];
            _SecantPTXFGValue value;
            AstPtxInstruction primal_args[3];
            const size_t num_args = instruction.aux;
            size_t arg_idx;
            size_t derivative_idx;

            if (num_args == 0u || num_args > 3u || assembler->stack_size < num_args ||
                instruction.payload.idx >= AST_PTX_PTX_INSTRUCTION_NUM_ENUMS) {
                return SECANT_PTX_ERROR_BAD_PROGRAM;
            }
            assembler->stack_size -= num_args;
            for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
                args[arg_idx] = assembler->stack[assembler->stack_size + arg_idx];
                primal_args[arg_idx] = args[arg_idx].primal;
            }
            S_PTX_FG_CHECK(_secant_ptx_fg_emit_operation(
                assembler,
                ast_ptx_ptx_instruction_names[instruction.payload.idx],
                primal_args,
                num_args,
                input_register_names,
                num_inputs,
                &value.primal));
            for (derivative_idx = 0u; derivative_idx < assembler->num_parameters; ++derivative_idx) {
                S_PTX_FG_CHECK(_secant_ptx_fg_derivative_emit(
                    assembler,
                    (AstPtxPtxInstruction)instruction.payload.idx,
                    args,
                    value.primal,
                    derivative_idx,
                    input_register_names,
                    num_inputs,
                    value.derivative + derivative_idx));
            }
            assembler->stack[assembler->stack_size++] = value;
        } else if (type == AST_PTX_INSTRUCTION_TYPE_ROUTINE) {
            _SecantPTXFGValue args[SECANT_AST_MAX_INSTRUCTION_ARGS];
            const size_t num_args = instruction.aux;

            if (num_args > SECANT_AST_MAX_INSTRUCTION_ARGS ||
                assembler->stack_size < num_args || routines == NULL ||
                instruction.payload.idx >= num_routines ||
                routines[instruction.payload.idx] == NULL) {
                return SECANT_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS;
            }
            if (frame_depth >= _SECANT_PTX_FG_ROUTINE_DEPTH) {
                return SECANT_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED;
            }
            assembler->stack_size -= num_args;
            memcpy(args, assembler->stack + assembler->stack_size, num_args * sizeof(*args));
            S_PTX_FG_CHECK(_secant_ptx_fg_compile_frame(
                assembler, input_register_names, num_inputs,
                routines, num_routines, routines[instruction.payload.idx],
                args, num_args, frame_depth + 1u));
        } else if (type == AST_PTX_INSTRUCTION_TYPE_RETURN) {
            if (assembler->stack_size != base_stack_size + 1u) {
                return SECANT_PTX_ERROR_BAD_PROGRAM;
            }
            return SECANT_PTX_SUCCESS;
        } else {
            return SECANT_PTX_ERROR_BAD_PROGRAM;
        }
    }
    return SECANT_PTX_ERROR_BAD_PROGRAM;
}

static SecantPTXResult
_secant_ptx_forward_gradient_compile(
    const char* const* output_register_names,
    size_t num_parameters,
    const char* const* input_register_names,
    size_t num_inputs,
    const AstPtxInstruction* const* routines,
    size_t num_routines,
    const AstPtxInstruction* instructions,
    int mixed_parameter_inputs,
    char* buffer,
    size_t buffer_size,
    size_t* bytes_written_ret
) {
    _SecantPTXFGAssembler assembler;
    _SecantPTXFGValue value;
    size_t output_idx;

    if (output_register_names == NULL || input_register_names == NULL ||
        num_parameters == 0u || num_parameters > _SECANT_PTX_FG_MAX_PARAMETERS ||
        num_inputs < num_parameters ||
        (mixed_parameter_inputs && num_inputs < 3u * num_parameters) ||
        instructions == NULL || bytes_written_ret == NULL) {
        return SECANT_PTX_ERROR_INVALID_VALUE;
    }
    memset(&assembler, 0, sizeof(assembler));
    assembler.num_parameters = num_parameters;
    assembler.mixed_parameter_inputs = mixed_parameter_inputs;
    assembler.buffer = buffer;
    assembler.buffer_size = buffer_size;
    S_PTX_FG_CHECK(_secant_ptx_fg_write(
        &assembler,
        "\t{\n\t.reg .f32 %%ad<%u>;\n\t.reg .pred %%adp<%u>;\n",
        (unsigned)_SECANT_PTX_FG_MAX_REGISTERS,
        (unsigned)_SECANT_PTX_FG_MAX_PREDICATES));
    S_PTX_FG_CHECK(_secant_ptx_fg_compile_frame(
        &assembler, input_register_names, num_inputs,
        routines, num_routines, instructions, NULL, 0u, 0u));
    value = assembler.stack[0];
    for (output_idx = 0u; output_idx < num_parameters + 1u; ++output_idx) {
        const AstPtxInstruction output = output_idx == 0u
            ? value.primal : value.derivative[output_idx - 1u];
        S_PTX_FG_CHECK(_secant_ptx_fg_write(
            &assembler, "\tmov.f32 %%%s, ", output_register_names[output_idx]));
        S_PTX_FG_CHECK(_secant_ptx_fg_operand_write(
            &assembler, output, input_register_names, num_inputs));
        S_PTX_FG_CHECK(_secant_ptx_fg_write(&assembler, ";\n"));
    }
    S_PTX_FG_CHECK(_secant_ptx_fg_write(&assembler, "\t}"));
    *bytes_written_ret = assembler.offset;
    if (buffer != NULL) {
        if (assembler.offset >= buffer_size) {
            return SECANT_PTX_ERROR_INSUFFICIENT_BUFFER;
        }
        buffer[assembler.offset] = '\0';
    }
    return SECANT_PTX_SUCCESS;
}

#undef S_PTX_FG_CHECK

#endif
