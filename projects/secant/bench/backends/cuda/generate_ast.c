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
#include "secant_cuda.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define SECANT_CUDA_AST_ROUTINE_DEPTH 8u

#define S_CUDA_AST_ERROR_RET(ans) do { \
    SecantCUDAResult secant_cuda_ast_result = (ans); \
    return secant_cuda_ast_result; \
} while (0)

#define S_CUDA_AST_CHECK_RET(ans) do { \
    SecantCUDAResult secant_cuda_ast_check_result = (ans); \
    if (secant_cuda_ast_check_result != SECANT_CUDA_SUCCESS) { \
        S_CUDA_AST_ERROR_RET(secant_cuda_ast_check_result); \
    } \
} while (0)

typedef enum SCudaAstOperandKind {
    S_CUDA_AST_OPERAND_INPUT = 0,
    S_CUDA_AST_OPERAND_DYNAMIC_CONSTANT = 1,
    S_CUDA_AST_OPERAND_IMMEDIATE = 2,
    S_CUDA_AST_OPERAND_VARIABLE = 3,
    S_CUDA_AST_OPERAND_MIXED_LEAF = 4
} SCudaAstOperandKind;

typedef struct SCudaAstOperand {
    SCudaAstOperandKind kind;
    uint32_t value;
} SCudaAstOperand;

typedef struct SCudaAstAssembler {
    SCudaAstOperand stack[SECANT_AST_MAX_STACK_DEPTH];
    size_t stack_size;
    uint32_t next_variable;
} SCudaAstAssembler;

static SecantCUDAResult
_secant_cuda_ast_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int bytes;

    if (offset == NULL || format == NULL) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        bytes = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        bytes = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (bytes < 0) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_FORMAT);
    }
    if ((size_t)bytes > SIZE_MAX - *offset) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    *offset += (size_t)bytes;
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_write_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (offset == SIZE_MAX) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_CUDA_SUCCESS;
}

static int
_secant_cuda_ast_identifier_is_valid(const char* name) {
    const unsigned char* cursor = (const unsigned char*)name;

    if (cursor == NULL ||
        !((*cursor >= 'a' && *cursor <= 'z') ||
          (*cursor >= 'A' && *cursor <= 'Z') || *cursor == '_')) {
        return 0;
    }
    for (++cursor; *cursor != '\0'; ++cursor) {
        if (!((*cursor >= 'a' && *cursor <= 'z') ||
              (*cursor >= 'A' && *cursor <= 'Z') ||
              (*cursor >= '0' && *cursor <= '9') || *cursor == '_')) {
            return 0;
        }
    }
    return 1;
}

static uint8_t
_secant_cuda_ast_num_args(SecantAstInstructionType type) {
    switch (type) {
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
            return 1u;
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            return 2u;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            return 3u;
        default:
            return 0u;
    }
}

static SecantCUDAResult
_secant_cuda_ast_routine_num_args(
    const SecantAstInstruction* instructions,
    size_t* num_args_ret
) {
    size_t instruction_offset = 0u;
    size_t num_args = 0u;
    size_t instruction_idx;

    if (instructions == NULL || num_args_ret == NULL) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            const size_t arg_idx = secant_ast_index_get(instruction);

            if (arg_idx >= SECANT_AST_MAX_INSTRUCTION_ARGS) {
                S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_TOO_MANY_ARGS);
            }
            if (arg_idx + 1u > num_args) {
                num_args = arg_idx + 1u;
            }
        }
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_CUDA_SUCCESS;
        }
        instruction_offset += instruction_size;
    }
    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
}

static SecantCUDAResult
_secant_cuda_ast_push(
    SCudaAstAssembler* assembler,
    SCudaAstOperand operand
) {
    if (assembler == NULL || assembler->stack_size >= SECANT_AST_MAX_STACK_DEPTH) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_STACK_OVERFLOW);
    }
    assembler->stack[assembler->stack_size++] = operand;
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_pop(
    SCudaAstAssembler* assembler,
    SCudaAstOperand* operand_ret
) {
    if (assembler == NULL || operand_ret == NULL) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (assembler->stack_size == 0u) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_STACK_UNDERFLOW);
    }
    assembler->stack_size -= 1u;
    *operand_ret = assembler->stack[assembler->stack_size];
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_operand_emit(
    SCudaAstOperand operand,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (operand.kind) {
        case S_CUDA_AST_OPERAND_INPUT:
            return _secant_cuda_ast_write(
                buffer, buffer_size, offset, "input%u", (unsigned)operand.value);
        case S_CUDA_AST_OPERAND_DYNAMIC_CONSTANT:
            return _secant_cuda_ast_write(
                buffer, buffer_size, offset, "constant%u", (unsigned)operand.value);
        case S_CUDA_AST_OPERAND_MIXED_LEAF:
            return _secant_cuda_ast_write(
                buffer, buffer_size, offset, "mixed%u", (unsigned)operand.value);
        case S_CUDA_AST_OPERAND_IMMEDIATE:
            return _secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "__uint_as_float(0x%08xu)",
                (unsigned)operand.value);
        case S_CUDA_AST_OPERAND_VARIABLE:
            return _secant_cuda_ast_write(
                buffer, buffer_size, offset, "x%u", (unsigned)operand.value);
        default:
            S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
    }
}

static SecantCUDAResult
_secant_cuda_ast_derivative_operand_emit(
    SCudaAstOperand operand,
    size_t derivative_idx,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    if (operand.kind == S_CUDA_AST_OPERAND_DYNAMIC_CONSTANT) {
        return _secant_cuda_ast_write(
            buffer,
            buffer_size,
            offset,
            operand.value == derivative_idx ? "1.0f" : "0.0f");
    }
    if (operand.kind == S_CUDA_AST_OPERAND_MIXED_LEAF) {
        return _secant_cuda_ast_write(
            buffer,
            buffer_size,
            offset,
            operand.value == derivative_idx ? "mixed_gradient%u" : "0.0f",
            (unsigned)operand.value);
    }
    if (operand.kind == S_CUDA_AST_OPERAND_VARIABLE) {
        return _secant_cuda_ast_write(
            buffer,
            buffer_size,
            offset,
            "dx%u_%zu",
            (unsigned)operand.value,
            derivative_idx);
    }
    return _secant_cuda_ast_write(buffer, buffer_size, offset, "0.0f");
}

static SecantCUDAResult
_secant_cuda_ast_primal_expression_emit(
    SecantAstInstructionType type,
    const SCudaAstOperand* args,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (type) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_ADD_F32
                    ? " + "
                    : (type == SECANT_AST_INSTRUCTION_TYPE_SUB_F32 ? " - " : " * ")));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer, buffer_size, offset, "secant_cuda_ast_div_f32("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "-("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
            {
                const char* function_name = NULL;

                switch (type) {
                    case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: function_name = "secant_cuda_ast_sqrt_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: function_name = "secant_cuda_ast_rcp_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: function_name = "secant_cuda_ast_sin_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_COS_F32: function_name = "secant_cuda_ast_cos_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: function_name = "secant_cuda_ast_ex2_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: function_name = "secant_cuda_ast_lg2_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: function_name = "secant_cuda_ast_rsqrt_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: function_name = "secant_cuda_ast_tanh_f32"; break;
                    default: break;
                }
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                    buffer, buffer_size, offset, "%s(", function_name));
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            }
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "fabsf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_MIN_F32 ? "fminf(" : "fmaxf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "fmaf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[2], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_EXP_F32 ? "expf(" : "logf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        default:
            S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_OP);
    }
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_forward_expression_emit(
    SecantAstInstructionType type,
    const SCudaAstOperand* args,
    SCudaAstOperand result,
    size_t derivative_idx,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (type) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_ADD_F32 ? " + " : " - "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[1], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "fmaf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[1], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer, buffer_size, offset, "secant_cuda_ast_div_f32(fmaf(-("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "), "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[1], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "), "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "-("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer, buffer_size, offset, "0.5f * secant_cuda_ast_div_f32("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "-("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " < 0.0f ? -("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ") : "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_MIN_F32 ? " <= " : " >= "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " ? "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " : "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[1], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "fmaf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[1], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", fmaf("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[1], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[2], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "))"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_SIN_F32
                    ? "secant_cuda_ast_cos_f32("
                    : "-(secant_cuda_ast_sin_f32("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                type == SECANT_AST_INSTRUCTION_TYPE_SIN_F32 ? ") * " : ") * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            if (type == SECANT_AST_INSTRUCTION_TYPE_COS_F32) {
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            }
            break;
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "__uint_as_float(0x3f317218u) * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "__uint_as_float(0x3fb8aa3bu) * secant_cuda_ast_div_f32("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, "-0.5f * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * (1.0f - "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, " * "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer, buffer_size, offset, "secant_cuda_ast_div_f32("));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                args[0], derivative_idx, buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ", "));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(args[0], buffer, buffer_size, offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ")"));
            break;
        default:
            S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_OP);
    }
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_operation_emit(
    SCudaAstAssembler* assembler,
    SecantAstInstructionType type,
    size_t num_dynamic_constants,
    int emit_gradient,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    SCudaAstOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
    SCudaAstOperand result;
    const size_t num_args = _secant_cuda_ast_num_args(type);
    size_t derivative_idx;
    size_t arg_idx;

    if (num_args == 0u) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_OP);
    }
    for (arg_idx = num_args; arg_idx != 0u; --arg_idx) {
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_pop(assembler, args + arg_idx - 1u));
    }
    if (assembler->next_variable == UINT32_MAX) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    result.kind = S_CUDA_AST_OPERAND_VARIABLE;
    result.value = assembler->next_variable++;

    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
        buffer, buffer_size, offset, "    const float x%u = ", (unsigned)result.value));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_primal_expression_emit(
        type, args, buffer, buffer_size, offset));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ";\n"));

    if (emit_gradient) {
        for (derivative_idx = 0u;
             derivative_idx < num_dynamic_constants;
             ++derivative_idx) {
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "    const float dx%u_%zu = ",
                (unsigned)result.value,
                derivative_idx));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_forward_expression_emit(
                type,
                args,
                result,
                derivative_idx,
                buffer,
                buffer_size,
                offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, offset, ";\n"));
        }
    }
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(assembler, result));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_frame_compile(
    SCudaAstAssembler* assembler,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* instructions,
    const SCudaAstOperand* routine_args,
    size_t num_routine_args,
    size_t routine_depth,
    int emit_gradient,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    SCudaAstOperand* result_ret
) {
    const size_t base_stack_size = assembler->stack_size;
    size_t instruction_offset = 0u;
    size_t instruction_idx;

    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);
        SCudaAstOperand operand;

        if (instruction_size == 0u) {
            S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        memset(&operand, 0, sizeof(operand));
        switch (type) {
            case SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32:
                if (secant_ast_index_get(instruction) >= num_input_columns) {
                    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
                }
                operand.kind = S_CUDA_AST_OPERAND_INPUT;
                operand.value = secant_ast_index_get(instruction);
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(assembler, operand));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32:
                if (emit_gradient == 2) {
                    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_OP);
                }
                if (secant_ast_index_get(instruction) >= num_dynamic_constants) {
                    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
                }
                operand.kind = S_CUDA_AST_OPERAND_DYNAMIC_CONSTANT;
                operand.value = secant_ast_index_get(instruction);
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(assembler, operand));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
                operand.kind = S_CUDA_AST_OPERAND_IMMEDIATE;
                operand.value = secant_ast_constant_f32_bits_get(instruction);
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(assembler, operand));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
                if (routine_args == NULL ||
                    secant_ast_index_get(instruction) >= num_routine_args) {
                    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                }
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(
                    assembler,
                    routine_args[secant_ast_index_get(instruction)]));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
            case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
            case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
            case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
            case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
            case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
            case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
            case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
            case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
            case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
            case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
            case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
            case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operation_emit(
                    assembler,
                    type,
                    num_dynamic_constants,
                    emit_gradient,
                    buffer,
                    buffer_size,
                    offset));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
                {
                    SCudaAstOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
                    const size_t routine_idx = secant_ast_index_get(instruction);
                    size_t num_args;
                    size_t arg_idx;

                    if (routines == NULL || routine_idx >= num_routines ||
                        routines[routine_idx] == NULL) {
                        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                    }
                    if (routine_depth >= SECANT_CUDA_AST_ROUTINE_DEPTH) {
                        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED);
                    }
                    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_routine_num_args(
                        routines[routine_idx], &num_args));
                    if (num_args > SECANT_AST_MAX_INSTRUCTION_ARGS) {
                        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_TOO_MANY_ARGS);
                    }
                    for (arg_idx = num_args; arg_idx != 0u; --arg_idx) {
                        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_pop(
                            assembler, args + arg_idx - 1u));
                    }
                    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_frame_compile(
                        assembler,
                        num_input_columns,
                        num_dynamic_constants,
                        routines,
                        num_routines,
                        routines[routine_idx],
                        args,
                        num_args,
                        routine_depth + 1u,
                        emit_gradient,
                        buffer,
                        buffer_size,
                        offset,
                        &operand));
                    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(assembler, operand));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32:
                if (assembler->stack_size != base_stack_size + 1u) {
                    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
                }
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_pop(assembler, result_ret));
                return SECANT_CUDA_SUCCESS;
            case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32:
                if ((emit_gradient != 2 && emit_gradient != 3) ||
                    secant_ast_index_get(instruction) >= num_dynamic_constants) {
                    S_CUDA_AST_ERROR_RET((emit_gradient == 2 || emit_gradient == 3)
                        ? SECANT_CUDA_ERROR_INVALID_VALUE
                        : SECANT_CUDA_ERROR_UNSUPPORTED_OP);
                }
                operand.kind = emit_gradient == 3
                    ? S_CUDA_AST_OPERAND_MIXED_LEAF
                    : S_CUDA_AST_OPERAND_DYNAMIC_CONSTANT;
                operand.value = secant_ast_index_get(instruction);
                S_CUDA_AST_CHECK_RET(_secant_cuda_ast_push(assembler, operand));
                break;
            case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_COLUMN_INPUT_F32:
                S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_OP);
            default:
                S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        instruction_offset += instruction_size;
    }
    S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
}

static SecantCUDAResult
_secant_cuda_ast_prelude_emit(
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    return _secant_cuda_ast_write(
        buffer,
        buffer_size,
        offset,
        "#ifndef SECANT_CUDA_AST_HELPERS_DEFINED\n"
        "#define SECANT_CUDA_AST_HELPERS_DEFINED 1\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_div_f32(float lhs, float rhs) {\n"
        "    float result;\n"
        "    asm volatile(\"div.approx.ftz.f32 %%0, %%1, %%2;\" : \"=f\"(result) : \"f\"(lhs), \"f\"(rhs));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_sqrt_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"sqrt.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_rcp_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"rcp.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_sin_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"sin.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_cos_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"cos.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_ex2_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"ex2.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_lg2_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"lg2.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_rsqrt_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"rsqrt.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ast_tanh_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"tanh.approx.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n"
        "#endif\n\n");
}

static SecantCUDAResult
_secant_cuda_ast_parameters_emit(
    size_t num_input_columns,
    size_t num_dynamic_constants,
    int emit_gradient,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t parameter_idx = 0u;
    size_t input_idx;
    const size_t num_parameters = num_input_columns + num_dynamic_constants +
        (emit_gradient == 3 ? 2u * num_dynamic_constants : 0u) +
        (emit_gradient ? num_dynamic_constants + 1u : 0u);

    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx, ++parameter_idx) {
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
            buffer,
            buffer_size,
            offset,
            "    const float input%zu%s\n",
            input_idx,
            parameter_idx + 1u == num_parameters ? "" : ","));
    }
    for (input_idx = 0u; input_idx < num_dynamic_constants; ++input_idx, ++parameter_idx) {
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
            buffer,
            buffer_size,
            offset,
            "    const float constant%zu%s\n",
            input_idx,
            parameter_idx + 1u == num_parameters ? "" : ","));
    }
    if (emit_gradient == 3) {
        for (input_idx = 0u; input_idx < num_dynamic_constants; ++input_idx, ++parameter_idx) {
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "    const float mixed%zu%s\n",
                input_idx,
                parameter_idx + 1u == num_parameters ? "" : ","));
        }
        for (input_idx = 0u; input_idx < num_dynamic_constants; ++input_idx, ++parameter_idx) {
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "    const float mixed_gradient%zu%s\n",
                input_idx,
                parameter_idx + 1u == num_parameters ? "" : ","));
        }
    }
    if (emit_gradient) {
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
            buffer,
            buffer_size,
            offset,
            "    float& value_ret%s\n",
            num_dynamic_constants == 0u ? "" : ","));
        for (input_idx = 0u; input_idx < num_dynamic_constants; ++input_idx) {
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                offset,
                "    float& gradient%zu_ret%s\n",
                input_idx,
                input_idx + 1u == num_dynamic_constants ? "" : ","));
        }
    }
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_ast_source_generate(
    const char* function_name,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    int emit_gradient,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    SCudaAstAssembler assembler;
    SCudaAstOperand result;
    size_t offset = 0u;
    size_t derivative_idx;

    if (!_secant_cuda_ast_identifier_is_valid(function_name) ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        num_dynamic_constants > SECANT_AST_MAX_INPUTS ||
        num_input_columns + num_dynamic_constants > SECANT_AST_MAX_INPUTS ||
        num_routines > SECANT_AST_MAX_ROUTINES || ast == NULL ||
        cuda_size_ret == NULL || (num_routines != 0u && routines == NULL) ||
        (emit_gradient && num_dynamic_constants == 0u)) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    if (num_variables_ret != NULL) {
        *num_variables_ret = 0u;
    }
    memset(&assembler, 0, sizeof(assembler));

    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_prelude_emit(buffer, buffer_size, &offset));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
        buffer,
        buffer_size,
        &offset,
        emit_gradient
            ? "static __device__ __forceinline__ void %s(\n"
            : "static __device__ __forceinline__ float %s(\n",
        function_name));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_parameters_emit(
        num_input_columns,
        num_dynamic_constants,
        emit_gradient,
        buffer,
        buffer_size,
        &offset));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, ")\n{\n"));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_frame_compile(
        &assembler,
        num_input_columns,
        num_dynamic_constants,
        routines,
        num_routines,
        ast,
        NULL,
        0u,
        0u,
        emit_gradient,
        buffer,
        buffer_size,
        &offset,
        &result));
    if (assembler.stack_size != 0u) {
        S_CUDA_AST_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
    }

    if (emit_gradient) {
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, "\n    value_ret = "));
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, &offset));
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, ";\n"));
        for (derivative_idx = 0u;
             derivative_idx < num_dynamic_constants;
             ++derivative_idx) {
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(
                buffer,
                buffer_size,
                &offset,
                "    gradient%zu_ret = ",
                derivative_idx));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_derivative_operand_emit(
                result,
                derivative_idx,
                buffer,
                buffer_size,
                &offset));
            S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, ";\n"));
        }
    } else {
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, "\n    return "));
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_operand_emit(result, buffer, buffer_size, &offset));
        S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, ";\n"));
    }
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write(buffer, buffer_size, &offset, "}\n"));
    S_CUDA_AST_CHECK_RET(_secant_cuda_ast_write_finish(
        buffer, buffer_size, offset, cuda_size_ret));
    if (num_variables_ret != NULL) {
        *num_variables_ret = assembler.next_variable;
    }
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_ast_primal_source_generate(
    const char* function_name,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    return _secant_cuda_ast_source_generate(
        function_name,
        num_input_columns,
        num_dynamic_constants,
        routines,
        num_routines,
        ast,
        0,
        buffer,
        buffer_size,
        cuda_size_ret,
        num_variables_ret);
}

SecantCUDAResult
secant_cuda_ast_forward_gradient_source_generate(
    const char* function_name,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    return _secant_cuda_ast_source_generate(
        function_name,
        num_input_columns,
        num_dynamic_constants,
        routines,
        num_routines,
        ast,
        1,
        buffer,
        buffer_size,
        cuda_size_ret,
        num_variables_ret);
}

SecantCUDAResult
secant_cuda_ast_dynamic_leaf_forward_gradient_source_generate(
    const char* function_name,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    return _secant_cuda_ast_source_generate(
        function_name,
        num_static_input_columns,
        num_dynamic_leaves,
        routines,
        num_routines,
        ast,
        2,
        buffer,
        buffer_size,
        cuda_size_ret,
        num_variables_ret);
}

SecantCUDAResult
secant_cuda_ast_lm_forward_gradient_source_generate(
    const char* function_name,
    size_t num_static_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    return _secant_cuda_ast_source_generate(
        function_name,
        num_static_input_columns,
        num_dynamic_constants,
        routines,
        num_routines,
        ast,
        3,
        buffer,
        buffer_size,
        cuda_size_ret,
        num_variables_ret);
}

#undef S_CUDA_AST_CHECK_RET
#undef S_CUDA_AST_ERROR_RET
