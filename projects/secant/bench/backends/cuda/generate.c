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

#ifndef SECANT_CUDA_ROUTINE_DEPTH
#define SECANT_CUDA_ROUTINE_DEPTH 8u
#endif

#define _SECANT_CUDA_ERROR_RET(ans) do { SecantCUDAResult secant_cuda_result = (ans); return secant_cuda_result; } while (0)

static SecantCUDAResult
_secant_cuda_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int bytes;

    if (offset == NULL || format == NULL) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        bytes = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        bytes = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (bytes < 0) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_FORMAT);
    }
    if ((size_t)bytes > SIZE_MAX - *offset) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    *offset += (size_t)bytes;
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_write_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (offset == SIZE_MAX) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_CUDA_SUCCESS;
}

/* CUDA expression renderer and kernel generators. */
#define S_CUDA_ERROR_RET(ans) \
    do { \
        SecantCUDAResult _secant_cuda_result = (ans); \
        return _secant_cuda_result; \
    } while (0)

#define S_CUDA_CHECK_RET(ans) \
    do { \
        SecantCUDAResult _secant_cuda_check_result = (ans); \
        if (_secant_cuda_check_result != SECANT_CUDA_SUCCESS) { \
            S_CUDA_ERROR_RET(_secant_cuda_check_result); \
        } \
    } while (0)

static uint16_t
_secant_cuda_instruction_num_args(
    SecantAstInstructionType instruction_type
) {
    switch (instruction_type) {
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

typedef enum SCudaOperandKind {
    S_CUDA_OPERAND_INPUT = 0,
    S_CUDA_OPERAND_IMMEDIATE = 1,
    S_CUDA_OPERAND_VARIABLE = 2,
    S_CUDA_OPERAND_ROUTINE_ARG = 3
} SCudaOperandKind;

typedef struct SCudaOperand {
    SCudaOperandKind kind;
    uint32_t value;
    uint8_t owned;
} SCudaOperand;

typedef struct SCudaAssembler {
    SCudaOperand stack[SECANT_AST_MAX_STACK_DEPTH];
    size_t stack_size;
    uint32_t free_variables[SECANT_AST_MAX_STACK_DEPTH];
    size_t num_free_variables;
    uint32_t next_variable;
    uint32_t high_water_variable;
} SCudaAssembler;

static int
_secant_cuda_identifier_is_valid(const char* name) {
    const unsigned char* cursor = (const unsigned char*)name;

    if (cursor == NULL || !((*cursor >= 'a' && *cursor <= 'z') || (*cursor >= 'A' && *cursor <= 'Z') || *cursor == '_')) {
        return 0;
    }

    for (++cursor; *cursor != '\0'; ++cursor) {
        if (!((*cursor >= 'a' && *cursor <= 'z') ||
              (*cursor >= 'A' && *cursor <= 'Z') ||
              (*cursor >= '0' && *cursor <= '9') ||
              *cursor == '_')) {
            return 0;
        }
    }

    return 1;
}

static SecantCUDAResult
_secant_cuda_emit_routine_name(
    const char* const* routine_names,
    size_t routine_idx,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    if (routine_names == NULL) {
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "secant_cuda_routine_%03zu", routine_idx));
    } else {
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "secant_cuda_routine_%s", routine_names[routine_idx]));
    }

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_routine_num_args(const SecantAstInstruction* instructions, size_t* num_args_ret) {
    size_t instruction_count;
    size_t instruction_offset = 0u;
    size_t num_args = 0u;

    if (instructions == NULL || num_args_ret == NULL) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }

    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            const size_t arg_idx = secant_ast_index_get(instruction);

            if (arg_idx >= SECANT_AST_MAX_INSTRUCTION_ARGS) {
                S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_TOO_MANY_ARGS);
            }
            if (arg_idx + 1u > num_args) {
                num_args = arg_idx + 1u;
            }
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_CUDA_SUCCESS;
        }
        instruction_offset += instruction_size;
    }

    S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
}

static SecantCUDAResult
_secant_cuda_emit_prelude(char* buffer, size_t buffer_size, size_t* offset) {
    S_CUDA_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_div_f32(float lhs, float rhs) {\n"
        "    float result;\n"
        "    asm volatile(\"div.approx.ftz.f32 %%0, %%1, %%2;\" : \"=f\"(result) : \"f\"(lhs), \"f\"(rhs));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_sqrt_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"sqrt.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_rcp_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"rcp.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_sin_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"sin.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_cos_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"cos.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_ex2_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"ex2.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_lg2_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"lg2.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_rsqrt_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"rsqrt.approx.ftz.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_cuda_tanh_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"tanh.approx.f32 %%0, %%1;\" : \"=f\"(result) : \"f\"(value));\n"
        "    return result;\n"
        "}\n\n"));

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_push(SCudaAssembler* assembler, SCudaOperand operand) {
    if (assembler->stack_size >= SECANT_AST_MAX_STACK_DEPTH) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_STACK_OVERFLOW);
    }

    assembler->stack[assembler->stack_size++] = operand;
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_pop(SCudaAssembler* assembler, SCudaOperand* operand_ret) {
    if (assembler->stack_size == 0u) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_STACK_UNDERFLOW);
    }

    assembler->stack_size -= 1u;
    *operand_ret = assembler->stack[assembler->stack_size];
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_allocate_variable(SCudaAssembler* assembler, SCudaOperand* operand_ret, int* declaration_ret) {
    SCudaOperand operand;

    memset(&operand, 0, sizeof(operand));
    operand.kind = S_CUDA_OPERAND_VARIABLE;
    operand.owned = 1u;

    if (assembler->num_free_variables != 0u) {
        assembler->num_free_variables -= 1u;
        operand.value = assembler->free_variables[assembler->num_free_variables];
        *declaration_ret = 0;
    } else {
        if (assembler->next_variable > SECANT_AST_MAX_STACK_DEPTH) {
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_STACK_OVERFLOW);
        }
        operand.value = assembler->next_variable++;
        if (operand.value > assembler->high_water_variable) {
            assembler->high_water_variable = operand.value;
        }
        *declaration_ret = 1;
    }

    *operand_ret = operand;
    return SECANT_CUDA_SUCCESS;
}

static void
_secant_cuda_release_variable(SCudaAssembler* assembler, SCudaOperand operand) {
    if (operand.kind == S_CUDA_OPERAND_VARIABLE && operand.owned &&
        assembler->num_free_variables < SECANT_AST_MAX_STACK_DEPTH) {
        assembler->free_variables[assembler->num_free_variables++] = operand.value;
    }
}

static SecantCUDAResult
_secant_cuda_emit_operand(
    SCudaOperand operand,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (operand.kind) {
        case S_CUDA_OPERAND_INPUT:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "input%u", (unsigned)operand.value));
            break;
        case S_CUDA_OPERAND_IMMEDIATE:
            S_CUDA_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "__uint_as_float(0x%08xu)",
                (unsigned)operand.value));
            break;
        case S_CUDA_OPERAND_VARIABLE:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "x%u", (unsigned)operand.value));
            break;
        case S_CUDA_OPERAND_ROUTINE_ARG:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "arg%u", (unsigned)operand.value));
            break;
        default:
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
    }

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_emit_assignment_prefix(
    SCudaOperand dst,
    int declaration,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    if (declaration) {
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "%sfloat x%u = ", indent, (unsigned)dst.value));
    } else {
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "%sx%u = ", indent, (unsigned)dst.value));
    }

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_emit_operation_expression(
    SecantAstInstructionType op,
    const SCudaOperand* args,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (op) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "("));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                op == SECANT_AST_INSTRUCTION_TYPE_ADD_F32
                    ? " + "
                    : (op == SECANT_AST_INSTRUCTION_TYPE_SUB_F32 ? " - " : " * ")));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[1], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "secant_cuda_div_f32("));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ", "));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[1], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "-("));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "fabsf("));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            S_CUDA_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                op == SECANT_AST_INSTRUCTION_TYPE_MIN_F32 ? "fminf(" : "fmaxf("));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ", "));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[1], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "fmaf("));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ", "));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[1], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ", "));
            S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[2], buffer, buffer_size, offset));
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
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

                switch (op) {
                    case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: function_name = "secant_cuda_sqrt_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: function_name = "secant_cuda_rcp_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: function_name = "secant_cuda_sin_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_COS_F32: function_name = "secant_cuda_cos_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: function_name = "secant_cuda_ex2_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: function_name = "secant_cuda_lg2_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: function_name = "secant_cuda_rsqrt_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: function_name = "secant_cuda_tanh_f32"; break;
                    default: break;
                }

                S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "%s(", function_name));
                S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[0], buffer, buffer_size, offset));
                S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
            }
            break;
        default:
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_OP);
    }

    return SECANT_CUDA_SUCCESS;
}

static int
_secant_cuda_operand_matches(SCudaOperand lhs, SCudaOperand rhs) {
    return lhs.kind == S_CUDA_OPERAND_VARIABLE && rhs.kind == S_CUDA_OPERAND_VARIABLE && lhs.value == rhs.value;
}

static void
_secant_cuda_release_consumed(SCudaAssembler* assembler, const SCudaOperand* args, size_t num_args, SCudaOperand keep) {
    size_t i;

    for (i = 0u; i < num_args; ++i) {
        size_t j;
        int duplicate = 0;

        if (!args[i].owned || args[i].kind != S_CUDA_OPERAND_VARIABLE || _secant_cuda_operand_matches(args[i], keep)) {
            continue;
        }

        for (j = 0u; j < i; ++j) {
            if (args[j].owned && _secant_cuda_operand_matches(args[j], args[i])) {
                duplicate = 1;
                break;
            }
        }

        if (!duplicate) {
            _secant_cuda_release_variable(assembler, args[i]);
        }
    }
}

static SecantCUDAResult
_secant_cuda_emit_op(
    SCudaAssembler* assembler,
    SecantAstInstructionType op,
    size_t num_args,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    SCudaOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
    SCudaOperand dst;
    int declaration = 0;
    size_t i;

    for (i = num_args; i != 0u; --i) {
        S_CUDA_CHECK_RET(_secant_cuda_pop(assembler, args + i - 1u));
    }

    memset(&dst, 0, sizeof(dst));
    for (i = 0u; i < num_args; ++i) {
        if (args[i].kind == S_CUDA_OPERAND_VARIABLE && args[i].owned) {
            dst = args[i];
            break;
        }
    }

    if (i == num_args) {
        S_CUDA_CHECK_RET(_secant_cuda_allocate_variable(assembler, &dst, &declaration));
    }

    S_CUDA_CHECK_RET(_secant_cuda_emit_assignment_prefix(dst, declaration, indent, buffer, buffer_size, offset));
    S_CUDA_CHECK_RET(_secant_cuda_emit_operation_expression(op, args, buffer, buffer_size, offset));
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ";\n"));

    dst.owned = 1u;
    _secant_cuda_release_consumed(assembler, args, num_args, dst);
    S_CUDA_CHECK_RET(_secant_cuda_push(assembler, dst));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_emit_routine_call(
    SCudaAssembler* assembler,
    size_t routine_idx,
    size_t num_args,
    const char* const* routine_names,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    SCudaOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
    SCudaOperand dst;
    int declaration = 0;
    size_t i;

    for (i = num_args; i != 0u; --i) {
        S_CUDA_CHECK_RET(_secant_cuda_pop(assembler, args + i - 1u));
    }

    memset(&dst, 0, sizeof(dst));
    for (i = 0u; i < num_args; ++i) {
        if (args[i].kind == S_CUDA_OPERAND_VARIABLE && args[i].owned) {
            dst = args[i];
            break;
        }
    }

    if (i == num_args) {
        S_CUDA_CHECK_RET(_secant_cuda_allocate_variable(assembler, &dst, &declaration));
    }

    S_CUDA_CHECK_RET(_secant_cuda_emit_assignment_prefix(dst, declaration, indent, buffer, buffer_size, offset));
    S_CUDA_CHECK_RET(_secant_cuda_emit_routine_name(routine_names, routine_idx, buffer, buffer_size, offset));
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "("));
    for (i = 0u; i < num_args; ++i) {
        if (i != 0u) {
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ", "));
        }
        S_CUDA_CHECK_RET(_secant_cuda_emit_operand(args[i], buffer, buffer_size, offset));
    }
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ");\n"));

    dst.owned = 1u;
    _secant_cuda_release_consumed(assembler, args, num_args, dst);
    S_CUDA_CHECK_RET(_secant_cuda_push(assembler, dst));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_compile_frame(
    SCudaAssembler* assembler,
    size_t num_static_inputs,
    size_t num_dynamic_inputs,
    SecantAstInstructionType dynamic_input_type,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    size_t num_routine_args,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const size_t base_stack_size = assembler->stack_size;
    size_t instruction_count;
    size_t instruction_offset = 0u;

    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        switch (instruction_type) {
            case SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32:
                {
                    SCudaOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    if (secant_ast_index_get(instruction) >= num_static_inputs) {
                        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
                    }
                    operand.kind = S_CUDA_OPERAND_INPUT;
                    operand.value = secant_ast_index_get(instruction);
                    S_CUDA_CHECK_RET(_secant_cuda_push(assembler, operand));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32:
            case SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32:
                {
                    SCudaOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    if (instruction_type != dynamic_input_type ||
                        secant_ast_index_get(instruction) >= num_dynamic_inputs) {
                        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
                    }
                    operand.kind = S_CUDA_OPERAND_INPUT;
                    operand.value = num_static_inputs + secant_ast_index_get(instruction);
                    S_CUDA_CHECK_RET(_secant_cuda_push(assembler, operand));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
                {
                    SCudaOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    operand.kind = S_CUDA_OPERAND_IMMEDIATE;
                    operand.value = secant_ast_constant_f32_bits_get(instruction);
                    S_CUDA_CHECK_RET(_secant_cuda_push(assembler, operand));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
                {
                    SCudaOperand operand;

                    if (secant_ast_index_get(instruction) >= num_routine_args) {
                        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                    }
                    memset(&operand, 0, sizeof(operand));
                    operand.kind = S_CUDA_OPERAND_ROUTINE_ARG;
                    operand.value = secant_ast_index_get(instruction);
                    S_CUDA_CHECK_RET(_secant_cuda_push(assembler, operand));
                }
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
                {
                    const uint16_t num_args = _secant_cuda_instruction_num_args(instruction_type);

                    S_CUDA_CHECK_RET(_secant_cuda_emit_op(
                        assembler,
                        instruction_type,
                        num_args,
                        indent,
                        buffer,
                        buffer_size,
                        offset));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
                {
                    const size_t routine_idx = secant_ast_index_get(instruction);
                    size_t num_args;

                    if (routines == NULL || routine_idx >= num_routines ||
                        routines[routine_idx] == NULL) {
                        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                    }
                    S_CUDA_CHECK_RET(_secant_cuda_routine_num_args(
                        routines[routine_idx],
                        &num_args));
                    if (assembler->stack_size < num_args) {
                        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_STACK_UNDERFLOW);
                    }
                    S_CUDA_CHECK_RET(_secant_cuda_emit_routine_call(
                        assembler,
                        routine_idx,
                        num_args,
                        routine_names,
                        indent,
                        buffer,
                        buffer_size,
                        offset));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32:
                if (assembler->stack_size != base_stack_size + 1u) {
                    S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
                }
                return SECANT_CUDA_SUCCESS;
            default:
                S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        instruction_offset += instruction_size;
    }

    S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
}

static SecantCUDAResult
_secant_cuda_emit_body(
    size_t num_static_inputs,
    size_t num_dynamic_inputs,
    SecantAstInstructionType dynamic_input_type,
    size_t num_routine_args,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    const char* indent,
    const char* result_prefix,
    const char* result_suffix,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* num_variables_ret
) {
    SCudaAssembler assembler;
    SCudaOperand result;

    if (num_static_inputs > SECANT_AST_MAX_INPUTS ||
        num_dynamic_inputs > SECANT_AST_MAX_INPUTS - num_static_inputs ||
        num_routines > SECANT_AST_MAX_ROUTINES ||
        num_routine_args > SECANT_AST_MAX_INSTRUCTION_ARGS || instructions == NULL ||
        indent == NULL || result_prefix == NULL || result_suffix == NULL || offset == NULL) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }

    memset(&assembler, 0, sizeof(assembler));
    assembler.next_variable = 1u;

    S_CUDA_CHECK_RET(_secant_cuda_compile_frame(
        &assembler,
        num_static_inputs,
        num_dynamic_inputs,
        dynamic_input_type,
        routines,
        num_routines,
        routine_names,
        instructions,
        num_routine_args,
        indent,
        buffer,
        buffer_size,
        offset));

    if (assembler.stack_size != 1u) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
    }
    S_CUDA_CHECK_RET(_secant_cuda_pop(&assembler, &result));
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "%s%s", indent, result_prefix));
    S_CUDA_CHECK_RET(_secant_cuda_emit_operand(result, buffer, buffer_size, offset));
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "%s", result_suffix));

    if (num_variables_ret != NULL) {
        *num_variables_ret = assembler.high_water_variable;
    }

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_emit_expression(
    size_t num_static_inputs,
    size_t num_dynamic_inputs,
    SecantAstInstructionType dynamic_input_type,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    const char* indent,
    const char* result_prefix,
    const char* result_suffix,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* num_variables_ret
) {
    if (num_static_inputs > SECANT_AST_MAX_INPUTS ||
        num_dynamic_inputs > SECANT_AST_MAX_INPUTS - num_static_inputs ||
        num_static_inputs + num_dynamic_inputs == 0u) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }

    S_CUDA_ERROR_RET(_secant_cuda_emit_body(
        num_static_inputs,
        num_dynamic_inputs,
        dynamic_input_type,
        0u,
        routines,
        num_routines,
        routine_names,
        instructions,
        indent,
        result_prefix,
        result_suffix,
        buffer,
        buffer_size,
        offset,
        num_variables_ret));
}

static SecantCUDAResult
_secant_cuda_emit_routine_parameters(size_t num_args, char* buffer, size_t buffer_size, size_t* offset) {
    size_t arg_idx;

    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "("));
    for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
        if (arg_idx != 0u) {
            S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ", "));
        }
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "const float arg%zu", arg_idx));
    }
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_validate_routine_graph(
    size_t routine_idx,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    size_t* path,
    size_t depth
) {
    const SecantAstInstruction* instructions = routines[routine_idx];
    size_t instruction_count;
    size_t instruction_offset = 0u;

    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
            const size_t callee_idx = secant_ast_index_get(instruction);
            size_t path_idx;

            if (callee_idx >= num_routines || routines[callee_idx] == NULL) {
                S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
            }
            if (depth >= SECANT_CUDA_ROUTINE_DEPTH) {
                S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED);
            }
            for (path_idx = 0u; path_idx < depth; ++path_idx) {
                if (path[path_idx] == callee_idx) {
                    S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED);
                }
            }

            path[depth] = callee_idx;
            S_CUDA_CHECK_RET(_secant_cuda_validate_routine_graph(callee_idx, routines, num_routines, path, depth + 1u));
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return SECANT_CUDA_SUCCESS;
        }
        instruction_offset += instruction_size;
    }

    S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_BAD_PROGRAM);
}

static SecantCUDAResult
_secant_cuda_emit_routines(
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t routine_idx;
    size_t path[SECANT_CUDA_ROUTINE_DEPTH];

    if (offset == NULL || (num_routines != 0u && routines == NULL)) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t previous_idx;

        if (routines[routine_idx] == NULL ||
            (routine_names != NULL && !_secant_cuda_identifier_is_valid(routine_names[routine_idx]))) {
            S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        if (routine_names != NULL) {
            for (previous_idx = 0u; previous_idx < routine_idx; ++previous_idx) {
                if (strcmp(routine_names[previous_idx], routine_names[routine_idx]) == 0) {
                    S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
                }
            }
        }

        path[0] = routine_idx;
        S_CUDA_CHECK_RET(_secant_cuda_validate_routine_graph(routine_idx, routines, num_routines, path, 1u));
    }

    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t num_args;

        S_CUDA_CHECK_RET(_secant_cuda_routine_num_args(routines[routine_idx], &num_args));
        S_CUDA_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "static __device__ __forceinline__ __attribute__((unused)) float "));
        S_CUDA_CHECK_RET(_secant_cuda_emit_routine_name(routine_names, routine_idx, buffer, buffer_size, offset));
        S_CUDA_CHECK_RET(_secant_cuda_emit_routine_parameters(num_args, buffer, buffer_size, offset));
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ";\n"));
    }
    if (num_routines != 0u) {
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "\n"));
    }

    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t num_args;

        S_CUDA_CHECK_RET(_secant_cuda_routine_num_args(routines[routine_idx], &num_args));
        S_CUDA_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "static __device__ __forceinline__ __attribute__((unused)) float "));
        S_CUDA_CHECK_RET(_secant_cuda_emit_routine_name(routine_names, routine_idx, buffer, buffer_size, offset));
        S_CUDA_CHECK_RET(_secant_cuda_emit_routine_parameters(num_args, buffer, buffer_size, offset));
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "\n{\n"));
        S_CUDA_CHECK_RET(_secant_cuda_emit_body(
            0u,
            0u,
            SECANT_AST_INSTRUCTION_TYPE_NONE,
            num_args,
            routines,
            num_routines,
            routine_names,
            routines[routine_idx],
            "    ",
            "return ",
            ";\n",
            buffer,
            buffer_size,
            offset,
            NULL));
        S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "}\n\n"));
    }

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_emit_function(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* num_variables_ret
) {
    size_t input_idx;

    if (offset == NULL || !_secant_cuda_identifier_is_valid(function_name) || num_inputs == 0u ||
        num_inputs > SECANT_AST_MAX_INPUTS || instructions == NULL) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }

    S_CUDA_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "static __device__ __forceinline__ float %s(\n",
        function_name));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_CUDA_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "    const float input%zu%s\n",
            input_idx,
            input_idx + 1u == num_inputs ? "" : ","));
    }
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ")\n{\n"));

    S_CUDA_CHECK_RET(_secant_cuda_emit_expression(
        num_inputs,
        0u,
        SECANT_AST_INSTRUCTION_TYPE_NONE,
        routines,
        num_routines,
        routine_names,
        instructions,
        "    ",
        "return ",
        ";\n",
        buffer,
        buffer_size,
        offset,
        num_variables_ret));
    S_CUDA_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "}\n\n"));

    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_function_source_generate_impl(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    size_t offset = 0u;

    S_CUDA_CHECK_RET(_secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_CUDA_CHECK_RET(_secant_cuda_emit_routines(routines, num_routines, routine_names, buffer, buffer_size, &offset));
    S_CUDA_CHECK_RET(_secant_cuda_emit_function(
        function_name,
        num_inputs,
        routines,
        num_routines,
        routine_names,
        instructions,
        buffer,
        buffer_size,
        &offset,
        num_variables_ret));
    S_CUDA_CHECK_RET(_secant_cuda_write_finish(buffer, buffer_size, offset, cuda_size_ret));
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_function_source_generate(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
) {
    if (cuda_size_ret == NULL) {
        S_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }

    *cuda_size_ret = 0u;
    if (num_variables_ret != NULL) {
        *num_variables_ret = 0u;
    }

    S_CUDA_ERROR_RET(_secant_cuda_function_source_generate_impl(
        function_name,
        num_inputs,
        routines,
        num_routines,
        routine_names,
        instructions,
        buffer,
        buffer_size,
        cuda_size_ret,
        num_variables_ret));
}

#undef S_CUDA_CHECK_RET
#undef S_CUDA_ERROR_RET

#define S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(ans) \
    do { \
        SecantCUDAResult _secant_cuda_materialize_result = (ans); \
        return _secant_cuda_materialize_result; \
    } while (0)

#define S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(ans) \
    do { \
        SecantCUDAResult _secant_cuda_materialize_check_result = (ans); \
        if (_secant_cuda_materialize_check_result != SECANT_CUDA_SUCCESS) { \
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(_secant_cuda_materialize_check_result); \
        } \
    } while (0)

static int
_secant_cuda_materialize_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }

    *result_ret = lhs * rhs;
    return 1;
}

static SecantCUDAResult
_secant_cuda_materialize_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    size_t ast_idx;
    size_t input_idx;

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_static_column_materialize_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t input_leading_dimension,\n"
        "    size_t num_rows,\n"
        "    float* __restrict__ output,\n"
        "    size_t output_leading_dimension\n"
        ") {\n"
        "    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (row >= num_rows) {\n"
        "        return;\n"
        "    }\n\n",
        kernel_idx));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "    const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            input_idx));
    }
    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        char result_prefix[128];
        uint32_t num_variables = 0u;
        const int result_prefix_bytes = snprintf(
            result_prefix,
            sizeof(result_prefix),
            "output[(size_t)%zu * output_leading_dimension + row] = ",
            ast_idx);

        if (result_prefix_bytes < 0) {
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_CUDA_ERROR_FORMAT);
        }
        if ((size_t)result_prefix_bytes >= sizeof(result_prefix)) {
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
        }
        if (asts[ast_idx] == NULL) {
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }

        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    {\n"));
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_emit_expression(
            num_inputs,
            0u,
            SECANT_AST_INSTRUCTION_TYPE_NONE,
            routines,
            num_routines,
            routine_names,
            asts[ast_idx],
            "        ",
            result_prefix,
            ";\n",
            buffer,
            buffer_size,
            offset,
            &num_variables));
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    }\n\n"));

        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_materialize_impl(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    uint32_t max_variables = 0u;
    size_t total_asts;
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        num_routines > SECANT_AST_MAX_ROUTINES ||
        asts == NULL || cuda_size_ret == NULL ||
        (num_routines != 0u && routines == NULL)) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    if (!_secant_cuda_materialize_checked_mul(num_kernels, asts_per_kernel, &total_asts)) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }

    (void)total_asts;
    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_emit_routines(
        routines,
        num_routines,
        routine_names,
        buffer,
        buffer_size,
        &offset));

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_materialize_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            routines,
            num_routines,
            routine_names,
            asts + kernel_idx * asts_per_kernel,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_cuda_write_finish(buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }

    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_materialize_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    if (cuda_size_ret == NULL) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }

    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(_secant_cuda_materialize_impl(
        num_kernels,
        asts_per_kernel,
        num_inputs,
        routines,
        num_routines,
        routine_names,
        asts,
        buffer,
        buffer_size,
        cuda_size_ret,
        max_variables_ret));
}

#undef S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET
#undef S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET

#define S_CUDA_SSE_ERROR_RET(ans) do { \
    SecantCUDAResult _secant_cuda_sse_result = (ans); \
    return _secant_cuda_sse_result; \
} while (0)
#define S_CUDA_SSE_CHECK_RET(ans) do { \
    SecantCUDAResult _secant_cuda_sse_check_result = (ans); \
    if (_secant_cuda_sse_check_result != SECANT_CUDA_SUCCESS) { \
        S_CUDA_SSE_ERROR_RET(_secant_cuda_sse_check_result); \
    } \
} while (0)

static int
_secant_cuda_sse_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static SecantCUDAResult
_secant_cuda_sse_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    const size_t num_warps = threads_per_block / 32u;
    const size_t num_pairs = asts_per_kernel * num_targets;
    const size_t num_partials = num_pairs * SECANT_CUDA_SSE_MAX_WARPS;
    size_t ast_idx;
    size_t target_idx;
    size_t input_idx;

    S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_static_column_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t input_leading_dimension,\n"
        "    const float* __restrict__ targets,\n"
        "    size_t targets_leading_dimension,\n"
        "    size_t num_rows,\n"
        "    float* __restrict__ output_sse,\n"
        "    size_t output_leading_dimension\n"
        ") {\n"
        "    __shared__ float partial_sse[%zu];\n"
        "    const unsigned int lane = threadIdx.x & 31u;\n"
        "    const unsigned int warp = threadIdx.x >> 5u;\n"
        "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
        "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n",
        kernel_idx,
        num_partials,
        tile_rows,
        tile_rows,
        tile_rows));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "    float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const size_t row = tile_begin + tile_row;\n"));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        const float target%zu = targets[(size_t)%zu * targets_leading_dimension + row];\n",
            target_idx,
            target_idx));
    }
    S_CUDA_SSE_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_CUDA_SSE_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        S_CUDA_SSE_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "        {\n            float prediction;\n"));
        S_CUDA_SSE_CHECK_RET(_secant_cuda_emit_expression(
            num_inputs,
            0u,
            SECANT_AST_INSTRUCTION_TYPE_NONE,
            routines,
            num_routines,
            routine_names,
            asts[ast_idx],
            "            ",
            "prediction = ",
            ";\n",
            buffer,
            buffer_size,
            offset,
            &num_variables));
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "            const float error_%03zu = prediction - target%zu;\n"
                "            sse_%03zu_%03zu = error_%03zu * error_%03zu + sse_%03zu_%03zu;\n",
                target_idx,
                target_idx,
                ast_idx,
                target_idx,
                target_idx,
                target_idx,
                ast_idx,
                target_idx));
        }
        S_CUDA_SSE_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "        }\n\n"));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    S_CUDA_SSE_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    }\n\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "    for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                "        sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, sse_%03zu_%03zu, delta);\n"
                "    }\n"
                "    if (lane == 0u) {\n"
                "        partial_sse[(size_t)%zu * %uu + warp] = sse_%03zu_%03zu;\n"
                "    }\n",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx,
                pair_idx,
                SECANT_CUDA_SSE_MAX_WARPS,
                ast_idx,
                target_idx));
        }
    }

    S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "    __syncthreads();\n"
        "    if (warp == 0u) {\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "        float block_sse_%03zu_%03zu = lane < %zuu ? partial_sse[(size_t)%zu * %uu + lane] : 0.0f;\n"
                "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                "            block_sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, block_sse_%03zu_%03zu, delta);\n"
                "        }\n"
                "        if (lane == 0u) {\n",
                ast_idx,
                target_idx,
                num_warps,
                pair_idx,
                SECANT_CUDA_SSE_MAX_WARPS,
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
            S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "            atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + %zuu, "
                "block_sse_%03zu_%03zu);\n",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
            S_CUDA_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "        }\n"));
        }
    }
    S_CUDA_SSE_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    }\n}\n\n"));

    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    size_t total_asts;
    size_t num_pairs;
    size_t num_partials;
    size_t kernel_idx;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u || num_targets == 0u ||
        tile_rows == 0u || threads_per_block < 32u ||
        threads_per_block > tile_rows ||
        threads_per_block > SECANT_CUDA_SSE_MAX_WARPS * 32u ||
        threads_per_block % 32u != 0u ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        num_routines > SECANT_AST_MAX_ROUTINES || asts == NULL ||
        cuda_size_ret == NULL || (num_routines != 0u && routines == NULL)) {
        S_CUDA_SSE_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    if (!_secant_cuda_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts) ||
        !_secant_cuda_sse_checked_mul(asts_per_kernel, num_targets, &num_pairs) ||
        !_secant_cuda_sse_checked_mul(num_pairs, SECANT_CUDA_SSE_MAX_WARPS, &num_partials)) {
        S_CUDA_SSE_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    (void)total_asts;
    (void)num_partials;

    S_CUDA_SSE_CHECK_RET(_secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_CUDA_SSE_CHECK_RET(_secant_cuda_emit_routines(routines, num_routines, routine_names, buffer, buffer_size, &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_CUDA_SSE_CHECK_RET(_secant_cuda_sse_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            num_targets,
            tile_rows,
            threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts + kernel_idx * asts_per_kernel,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }
    S_CUDA_SSE_CHECK_RET(_secant_cuda_write_finish(buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_CUDA_SUCCESS;
}

#undef S_CUDA_SSE_CHECK_RET
#undef S_CUDA_SSE_ERROR_RET

#define S_CUDA_DYNAMIC_CONSTANT_SSE_ERROR_RET(ans) \
    do { \
        SecantCUDAResult _secant_cuda_dynamic_constant_sse_result = (ans); \
        return _secant_cuda_dynamic_constant_sse_result; \
    } while (0)

#define S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(ans) \
    do { \
        SecantCUDAResult _secant_cuda_dynamic_constant_sse_check_result = \
            (ans); \
        if (_secant_cuda_dynamic_constant_sse_check_result != \
            SECANT_CUDA_SUCCESS) { \
            S_CUDA_DYNAMIC_CONSTANT_SSE_ERROR_RET( \
                _secant_cuda_dynamic_constant_sse_check_result); \
        } \
    } while (0)

static int
_secant_cuda_dynamic_constant_sse_checked_add(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static SecantCUDAResult
_secant_cuda_dynamic_constant_sse_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    const size_t input_tile_elements = num_input_columns * tile_rows;
    const size_t target_tile_elements = num_targets * tile_rows;
    const size_t row_tile_elements = input_tile_elements + target_tile_elements;
    size_t ast_idx;
    size_t constant_idx;
    size_t input_idx;
    size_t target_idx;

    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_dynamic_constant_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t input_leading_dimension,\n"
        "    const float* __restrict__ constant_settings,\n"
        "    size_t constants_leading_dimension,\n"
        "    size_t num_settings,\n"
        "    const float* __restrict__ targets,\n"
        "    size_t targets_leading_dimension,\n"
        "    size_t num_rows,\n"
        "    float* __restrict__ output_sse,\n"
        "    size_t output_leading_dimension\n"
        ")\n"
        "{\n"
        "    __shared__ float row_tile[%zu];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + %zu;\n"
        "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
        "    const size_t remaining_rows = "
            "tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const size_t tile_num_rows = "
            "remaining_rows < %zuu ? remaining_rows : %zuu;\n\n"
        "    for (size_t tile_row = threadIdx.x; "
            "tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const size_t row = tile_begin + tile_row;\n",
        kernel_idx,
        row_tile_elements,
        input_tile_elements,
        tile_rows,
        tile_rows,
        tile_rows));

    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        input_tile[(size_t)%zu * %zuu + tile_row] = "
                "input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            tile_rows,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        target_tile[(size_t)%zu * %zuu + tile_row] = "
                "targets[(size_t)%zu * targets_leading_dimension + row];\n",
            target_idx,
            tile_rows,
            target_idx));
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    for (size_t setting = threadIdx.x; "
            "setting < num_settings; setting += blockDim.x) {\n"));

    for (constant_idx = 0u;
         constant_idx < num_input_constants;
         ++constant_idx) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = "
                "constant_settings[(size_t)%zu * "
                "constants_leading_dimension + setting];\n",
            num_input_columns + constant_idx,
            constant_idx));
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "        float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "        for (size_t eval_row = 0u; "
            "eval_row < tile_num_rows; ++eval_row) {\n"));

    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            const float input%zu = "
                "input_tile[(size_t)%zu * %zuu + eval_row];\n",
            input_idx,
            input_idx,
            tile_rows));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            const float target%zu = "
                "target_tile[(size_t)%zu * %zuu + eval_row];\n",
            target_idx,
            target_idx,
            tile_rows));
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_CUDA_DYNAMIC_CONSTANT_SSE_ERROR_RET(
                SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            {\n"
            "                float prediction;\n"));
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_emit_expression(
            num_input_columns,
            num_input_constants,
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
            routines,
            num_routines,
            routine_names,
            asts[ast_idx],
            "                ",
            "prediction = ",
            ";\n",
            buffer,
            buffer_size,
            offset,
            &num_variables));
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "                const float error_%03zu = "
                    "prediction - target%zu;\n"
                "                sse_%03zu_%03zu = "
                    "error_%03zu * error_%03zu + sse_%03zu_%03zu;\n",
                target_idx,
                target_idx,
                ast_idx,
                target_idx,
                target_idx,
                target_idx,
                ast_idx,
                target_idx));
        }
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            }\n\n"));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "        }\n\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(
                _secant_cuda_write(
                    buffer,
                    buffer_size,
                    offset,
                    "        atomicAdd(\n"
                    "            output_sse + (size_t)%zu * "
                        "output_leading_dimension + setting,\n"
                    "            sse_%03zu_%03zu);\n",
                    pair_idx,
                    ast_idx,
                    target_idx));
        }
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "    }\n"
        "}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_dynamic_constant_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    size_t num_inputs;
    size_t num_pairs;
    size_t num_sources;
    size_t row_tile_elements;
    size_t total_asts;
    size_t kernel_idx;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_input_columns == 0u ||
        num_input_columns >
            SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants >
            SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        num_targets == 0u || tile_rows == 0u ||
        threads_per_block == 0u || threads_per_block > 1024u ||
        num_routines > SECANT_AST_MAX_ROUTINES ||
        asts == NULL || cuda_size_ret == NULL ||
        (num_routines != 0u && routines == NULL)) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_ERROR_RET(
            SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    if (!_secant_cuda_dynamic_constant_sse_checked_add(
            num_input_columns,
            num_input_constants,
            &num_inputs) ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cuda_dynamic_constant_sse_checked_add(
            num_input_columns,
            num_targets,
            &num_sources) ||
        !_secant_cuda_sse_checked_mul(
            num_kernels,
            asts_per_kernel,
            &total_asts) ||
        !_secant_cuda_sse_checked_mul(
            asts_per_kernel,
            num_targets,
            &num_pairs) ||
        !_secant_cuda_sse_checked_mul(
            num_sources,
            tile_rows,
            &row_tile_elements)) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    (void)num_inputs;
    (void)total_asts;
    (void)num_pairs;
    (void)row_tile_elements;

    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(
        _secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_emit_routines(
        routines,
        num_routines,
        routine_names,
        buffer,
        buffer_size,
        &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(
            _secant_cuda_dynamic_constant_sse_emit_kernel(
                kernel_idx,
                asts_per_kernel,
                num_input_columns,
                num_input_constants,
                num_targets,
                tile_rows,
                routines,
                num_routines,
                routine_names,
                asts + kernel_idx * asts_per_kernel,
                buffer,
                buffer_size,
                &offset,
                &max_variables));
    }
    S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET(_secant_cuda_write_finish(
        buffer,
        buffer_size,
        offset,
        cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_CUDA_SUCCESS;
}

#undef S_CUDA_DYNAMIC_CONSTANT_SSE_CHECK_RET
#undef S_CUDA_DYNAMIC_CONSTANT_SSE_ERROR_RET

#define S_CUDA_DYNAMIC_LEAF_ERROR_RET(ans) do { \
    SecantCUDAResult _secant_cuda_dynamic_leaf_result = (ans); \
    return _secant_cuda_dynamic_leaf_result; \
} while (0)
#define S_CUDA_DYNAMIC_LEAF_CHECK_RET(ans) do { \
    SecantCUDAResult _secant_cuda_dynamic_leaf_check_result = (ans); \
    if (_secant_cuda_dynamic_leaf_check_result != SECANT_CUDA_SUCCESS) { \
        S_CUDA_DYNAMIC_LEAF_ERROR_RET(_secant_cuda_dynamic_leaf_check_result); \
    } \
} while (0)

/*
 * Keep state ownership separate from the statistic payload. The warp-owned
 * traversal below is intended to be reused by the LM statistics generator;
 * SSE is only the first one-value accumulator plugged into it.
 */
typedef enum _SecantCUDADynamicLeafStateOwner {
    _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_THREAD = 0,
    _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP = 1
} _SecantCUDADynamicLeafStateOwner;

static SecantCUDAResult
_secant_cuda_dynamic_leaf_sse_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    _SecantCUDADynamicLeafStateOwner state_owner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    const size_t shared_stride = num_input_columns | 1u;
    const int warp_owned = state_owner == _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP;
    size_t ast_idx;
    size_t input_idx;
    size_t leaf_idx;
    size_t target_idx;

    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "extern \"C\" __global__\n"
        "void secant_dynamic_leaf_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t num_input_columns,\n"
        "    size_t input_leading_dimension,\n"
        "    const unsigned int* __restrict__ leaf_masks,\n"
        "    const unsigned int* __restrict__ leaf_words,\n"
        "    size_t leaf_words_leading_dimension,\n"
        "    unsigned int num_settings,\n"
        "    unsigned int settings_per_cta,\n"
        "    const float* __restrict__ targets,\n"
        "    size_t targets_leading_dimension,\n"
        "    size_t num_rows,\n"
        "    size_t num_asts,\n"
        "    size_t num_targets,\n"
        "    float* __restrict__ output_sse,\n"
        "    size_t output_leading_dimension\n"
        ") {\n"
        "    extern __shared__ float row_tile[];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + (size_t)%zuu * %zuu;\n"
        "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
        "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n",
        kernel_idx,
        tile_rows,
        warp_owned ? num_input_columns : shared_stride,
        tile_rows,
        tile_rows,
        tile_rows));
    if (warp_owned) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "    const unsigned int lane = threadIdx.x & 31u;\n"
            "    const unsigned int warp = threadIdx.x >> 5u;\n"
            "    const unsigned int num_warps = blockDim.x >> 5u;\n\n"));
    } else {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "    const size_t shared_stride = %zuu;\n\n",
            shared_stride));
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "    for (size_t tile_row = threadIdx.x; tile_row < %zuu; tile_row += blockDim.x) {\n"
        "        const size_t row = tile_begin + tile_row;\n"
        "        const bool row_valid = row < num_rows;\n"
        "        #pragma unroll\n"
        "        for (size_t column = 0u; column < %zuu; ++column) {\n",
        tile_rows,
        num_input_columns));
    if (warp_owned) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "            input_tile[column * (size_t)%zuu + tile_row] = "
                "row_valid && column < num_input_columns\n",
            tile_rows));
    } else {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "            input_tile[tile_row * shared_stride + column] = "
                "row_valid && column < num_input_columns\n"));
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "                ? input[column * input_leading_dimension + row]\n"
        "                : 0.0f;\n"
        "        }\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        target_tile[(size_t)%zu * %zuu + tile_row] = row_valid\n"
            "            ? targets[(size_t)(%zuu < num_targets ? %zuu : 0u) * targets_leading_dimension + row]\n"
            "            : 0.0f;\n",
            target_idx,
            tile_rows,
            target_idx,
            target_idx));
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    const unsigned int setting_begin = blockIdx.y * settings_per_cta;\n"
        "    const unsigned int setting_count = settings_per_cta < num_settings - setting_begin\n"
        "        ? settings_per_cta : num_settings - setting_begin;\n\n"));
    if (warp_owned) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "    for (unsigned int setting_offset = warp; setting_offset < setting_count; setting_offset += num_warps) {\n"
            "        const unsigned int setting = setting_begin + setting_offset;\n"));
    } else {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "    for (unsigned int setting_offset = threadIdx.x; setting_offset < setting_count; setting_offset += blockDim.x) {\n"
            "        const unsigned int setting = setting_begin + setting_offset;\n"));
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "        const unsigned int leaf_mask = leaf_masks[setting];\n"
        "        const unsigned int* const words = leaf_words + setting * leaf_words_leading_dimension;\n"));
    if (!warp_owned) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        const unsigned int input_tile_address =\n"
            "            (unsigned int)__cvta_generic_to_shared((const void*)input_tile);\n"
            "        const unsigned int shared_stride_bytes = "
                "(unsigned int)(shared_stride * sizeof(float));\n"));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        const unsigned int word%zu = words[%zu];\n"
            "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
            "        float dynamic_input%zu = __uint_as_float(word%zu);\n",
            leaf_idx, leaf_idx, leaf_idx, leaf_idx, leaf_idx, leaf_idx));
        if (!warp_owned) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
                leaf_idx, leaf_idx));
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "        float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "\n"
        "        #pragma unroll 1\n"));
    if (warp_owned) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        for (size_t eval_row = lane; eval_row < tile_num_rows; eval_row += 32u) {\n"));
    } else {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            const float target%zu = target_tile[(size_t)%zu * %zuu + eval_row];\n",
            target_idx,
            target_idx,
            tile_rows));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        if (warp_owned) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            const float input%zu = is_column%zu\n"
                "                ? input_tile[(size_t)word%zu * %zuu + eval_row]\n"
                "                : dynamic_input%zu;\n",
                num_static_input_columns + leaf_idx,
                leaf_idx,
                leaf_idx,
                tile_rows,
                leaf_idx));
        } else {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            asm volatile(\n"
                "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; "
                    "@p ld.shared.f32 %%0, [%%1]; }\"\n"
                "                : \"+f\"(dynamic_input%zu)\n"
                "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
                "                : \"memory\");\n"
                "            const float input%zu = dynamic_input%zu;\n",
                leaf_idx,
                leaf_idx,
                leaf_idx,
                num_static_input_columns + leaf_idx,
                leaf_idx));
        }
    }
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        if (warp_owned) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
                input_idx,
                input_idx,
                tile_rows));
        } else {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            const float input%zu = input_tile[eval_row * shared_stride + %zuu];\n",
                input_idx,
                input_idx));
        }
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_CUDA_DYNAMIC_LEAF_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            {\n"
            "                float prediction;\n"));
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_emit_expression(
            num_static_input_columns,
            num_dynamic_leaves,
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
            routines,
            num_routines,
            routine_names,
            asts[ast_idx],
            "                ",
            "prediction = ",
            ";\n",
            buffer,
            buffer_size,
            offset,
            &num_variables));
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "                const float error_%03zu = prediction - target%zu;\n"
                "                sse_%03zu_%03zu = error_%03zu * error_%03zu + sse_%03zu_%03zu;\n",
                target_idx,
                target_idx,
                ast_idx,
                target_idx,
                target_idx,
                target_idx,
                ast_idx,
                target_idx));
        }
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "            }\n\n"));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    if (!warp_owned) {
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            address%zu += shared_stride_bytes;\n",
                leaf_idx));
        }
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "        }\n\n"));
    if (warp_owned) {
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                    buffer, buffer_size, offset,
                    "        #pragma unroll\n"
                    "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                    "            sse_%03zu_%03zu += __shfl_down_sync(0xffffffffu, "
                        "sse_%03zu_%03zu, delta);\n"
                    "        }\n",
                    ast_idx, target_idx, ast_idx, target_idx));
            }
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "        if (%s%zuu < num_asts && %zuu < num_targets) {\n"
                "            atomicAdd(output_sse + ((size_t)%zu * num_targets + %zuu) * "
                    "output_leading_dimension + setting, sse_%03zu_%03zu);\n"
                "        }\n",
                warp_owned ? "lane == 0u && " : "",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
        }
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    _SecantCUDADynamicLeafStateOwner state_owner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    size_t kernel_idx;
    size_t num_expression_inputs;
    size_t total_asts;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        (num_static_input_columns != 0u && num_static_input_columns != num_input_columns) ||
        num_dynamic_leaves == 0u ||
        num_dynamic_leaves > SECANT_CUDA_DYNAMIC_LEAF_SSE_MAX_DYNAMIC_LEAVES ||
        num_targets == 0u || tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u ||
        (state_owner != _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_THREAD &&
         state_owner != _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP) ||
        (state_owner == _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP &&
         threads_per_block % 32u != 0u) ||
        asts == NULL || cuda_size_ret == NULL ||
        num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL) ||
        !_secant_cuda_dynamic_constant_sse_checked_add(
            num_static_input_columns, num_dynamic_leaves, &num_expression_inputs) ||
        num_expression_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cuda_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts)) {
        S_CUDA_DYNAMIC_LEAF_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    (void)total_asts;
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_emit_routines(
        routines, num_routines, routine_names, buffer, buffer_size, &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_dynamic_leaf_sse_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_static_input_columns,
            num_dynamic_leaves,
            num_targets,
            tile_rows,
            state_owner,
            routines,
            num_routines,
            routine_names,
            asts + kernel_idx * asts_per_kernel,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }
    S_CUDA_DYNAMIC_LEAF_CHECK_RET(_secant_cuda_write_finish(buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    return _secant_cuda_dynamic_leaf_sse_source_generate(
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_THREAD,
        routines,
        num_routines,
        routine_names,
        asts,
        buffer,
        buffer_size,
        cuda_size_ret,
        max_variables_ret);
}

SecantCUDAResult
secant_cuda_warp_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    return _secant_cuda_dynamic_leaf_sse_source_generate(
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP,
        routines,
        num_routines,
        routine_names,
        asts,
        buffer,
        buffer_size,
        cuda_size_ret,
        max_variables_ret);
}

#undef S_CUDA_DYNAMIC_LEAF_CHECK_RET
#undef S_CUDA_DYNAMIC_LEAF_ERROR_RET

#define S_CUDA_LM_ERROR_RET(ans) do { SecantCUDAResult _r = (ans); return _r; } while (0)
#define S_CUDA_LM_CHECK_RET(ans) do { \
    SecantCUDAResult _r = (ans); \
    if (_r != SECANT_CUDA_SUCCESS) { S_CUDA_LM_ERROR_RET(_r); } \
} while (0)

SecantCUDAResult
secant_cuda_dynamic_leaf_lm_statistics_count(
    size_t num_parameters,
    size_t* num_statistics_ret
) {
    size_t triangle;

    if (num_parameters == 0u ||
        num_parameters > SECANT_CUDA_DYNAMIC_LEAF_LM_MAX_PARAMETERS ||
        num_statistics_ret == NULL ||
        num_parameters == SIZE_MAX) {
        S_CUDA_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    triangle = num_parameters * (num_parameters + 1u) / 2u;
    *num_statistics_ret = 1u + num_parameters + triangle;
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_dynamic_leaf_lm_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t num_statistics,
    size_t tile_rows,
    _SecantCUDADynamicLeafStateOwner state_owner,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    const int warp_owned = state_owner == _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP;
    const size_t shared_stride = num_input_columns | 1u;
    size_t ast_idx;
    size_t target_idx;
    size_t input_idx;
    size_t leaf_idx;
    size_t stat_idx;

    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "extern \"C\" __global__\n"
        "void secant_dynamic_leaf_lm_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t num_input_columns,\n"
        "    size_t input_leading_dimension,\n"
        "    const unsigned int* __restrict__ leaf_masks,\n"
        "    const unsigned int* __restrict__ leaf_words,\n"
        "    size_t leaf_words_leading_dimension,\n"
        "    unsigned int num_settings,\n"
        "    unsigned int settings_per_cta,\n"
        "    const float* __restrict__ targets,\n"
        "    size_t targets_leading_dimension,\n"
        "    size_t num_rows,\n"
        "    size_t num_asts,\n"
        "    size_t num_targets,\n"
        "    float* __restrict__ output_statistics,\n"
        "    size_t output_leading_dimension\n"
        ") {\n"
        "    extern __shared__ float row_tile[];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + (size_t)%zuu * %zuu;\n"
        "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
        "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n",
        kernel_idx,
        tile_rows,
        warp_owned ? num_input_columns : shared_stride,
        tile_rows,
        tile_rows,
        tile_rows));
    if (warp_owned) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "    const unsigned int lane = threadIdx.x & 31u;\n"
            "    const unsigned int warp = threadIdx.x >> 5u;\n"
            "    const unsigned int num_warps = blockDim.x >> 5u;\n\n"));
    } else {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "    const size_t shared_stride = %zuu;\n\n",
            shared_stride));
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "    for (size_t tile_row = threadIdx.x; tile_row < %zuu; tile_row += blockDim.x) {\n"
        "        const size_t row = tile_begin + tile_row;\n"
        "        const bool row_valid = row < num_rows;\n"
        "        #pragma unroll\n"
        "        for (size_t column = 0u; column < %zuu; ++column) {\n",
        tile_rows, num_input_columns));
    if (warp_owned) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "            input_tile[column * (size_t)%zuu + tile_row] = row_valid && column < num_input_columns\n",
            tile_rows));
    } else {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "            input_tile[tile_row * shared_stride + column] = row_valid && column < num_input_columns\n"));
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "                ? input[column * input_leading_dimension + row] : 0.0f;\n"
        "        }\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        target_tile[(size_t)%zu * %zuu + tile_row] = row_valid\n"
            "            ? targets[(size_t)(%zuu < num_targets ? %zuu : 0u) * targets_leading_dimension + row]\n"
            "            : 0.0f;\n",
            target_idx, tile_rows, target_idx, target_idx));
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    const unsigned int setting_begin = blockIdx.y * settings_per_cta;\n"
        "    const unsigned int setting_count = settings_per_cta < num_settings - setting_begin\n"
        "        ? settings_per_cta : num_settings - setting_begin;\n\n"));
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        warp_owned
            ? "    for (unsigned int setting_offset = warp; setting_offset < setting_count; setting_offset += num_warps) {\n"
              "        const unsigned int setting = setting_begin + setting_offset;\n"
            : "    for (unsigned int setting_offset = threadIdx.x; setting_offset < setting_count; setting_offset += blockDim.x) {\n"
              "        const unsigned int setting = setting_begin + setting_offset;\n"));
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "        const unsigned int leaf_mask = leaf_masks[setting];\n"
        "        const unsigned int* const words = leaf_words + setting * leaf_words_leading_dimension;\n"));
    if (!warp_owned) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        const unsigned int input_tile_address =\n"
            "            (unsigned int)__cvta_generic_to_shared((const void*)input_tile);\n"
            "        const unsigned int shared_stride_bytes = (unsigned int)(shared_stride * sizeof(float));\n"));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "        const unsigned int word%zu = words[%zu];\n"
            "        const unsigned int is_column%zu = (leaf_mask >> %zu) & 1u;\n"
            "        float dynamic_input%zu = __uint_as_float(word%zu);\n",
            leaf_idx, leaf_idx, leaf_idx, leaf_idx, leaf_idx, leaf_idx));
        if (!warp_owned) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "        unsigned int address%zu = input_tile_address + word%zu * sizeof(float);\n",
                leaf_idx, leaf_idx));
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            for (stat_idx = 0u; stat_idx < num_statistics; ++stat_idx) {
                S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                    buffer, buffer_size, offset,
                    "        float lm_%03zu_%03zu_%03zu = 0.0f;\n",
                    ast_idx, target_idx, stat_idx));
            }
        }
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        "\n        #pragma unroll 1\n"));
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset,
        warp_owned
            ? "        for (size_t eval_row = lane; eval_row < tile_num_rows; eval_row += 32u) {\n"
            : "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "            const float target%zu = target_tile[(size_t)%zu * %zuu + eval_row];\n",
            target_idx, target_idx, tile_rows));
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        if (warp_owned) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            const float input%zu = is_column%zu\n"
                "                ? input_tile[(size_t)word%zu * %zuu + eval_row]\n"
                "                : dynamic_input%zu;\n",
                num_static_input_columns + leaf_idx, leaf_idx, leaf_idx, tile_rows, leaf_idx));
        } else {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            asm volatile(\n"
                "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
                "                : \"+f\"(dynamic_input%zu)\n"
                "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
                "                : \"memory\");\n"
                "            const float input%zu = dynamic_input%zu;\n",
                leaf_idx, leaf_idx, leaf_idx,
                num_static_input_columns + leaf_idx, leaf_idx));
        }
    }
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        if (warp_owned) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
                input_idx, input_idx, tile_rows));
        } else {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            const float input%zu = input_tile[eval_row * shared_stride + %zuu];\n",
                input_idx, input_idx));
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        size_t row_stat;
        size_t lhs;
        size_t rhs;

        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "            {\n"
            "                float prediction;\n"));
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "                float gradient%zu;\n", leaf_idx));
        }
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "                secant_dynamic_leaf_lm_eval_%03zu_%03zu(\n",
            kernel_idx, ast_idx));
        for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "                    input%zu,\n", input_idx));
        }
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "                    input%zu,\n", num_static_input_columns + leaf_idx));
        }
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset,
            "                    prediction"));
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                ", gradient%zu", leaf_idx));
        }
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, ");\n"));
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "                gradient%zu = is_column%zu ? 0.0f : gradient%zu;\n",
                leaf_idx, leaf_idx, leaf_idx));
        }
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "                const float residual%zu = prediction - target%zu;\n"
                "                lm_%03zu_%03zu_000 = fmaf(residual%zu, residual%zu, lm_%03zu_%03zu_000);\n",
                target_idx, target_idx,
                ast_idx, target_idx, target_idx, target_idx, ast_idx, target_idx));
            row_stat = 1u;
            for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx, ++row_stat) {
                S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                    buffer, buffer_size, offset,
                    "                lm_%03zu_%03zu_%03zu = fmaf(gradient%zu, residual%zu, lm_%03zu_%03zu_%03zu);\n",
                    ast_idx, target_idx, row_stat, leaf_idx, target_idx,
                    ast_idx, target_idx, row_stat));
            }
            for (lhs = 0u; lhs < num_dynamic_leaves; ++lhs) {
                for (rhs = lhs; rhs < num_dynamic_leaves; ++rhs, ++row_stat) {
                    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                        buffer, buffer_size, offset,
                        "                lm_%03zu_%03zu_%03zu = fmaf(gradient%zu, gradient%zu, lm_%03zu_%03zu_%03zu);\n",
                        ast_idx, target_idx, row_stat, lhs, rhs,
                        ast_idx, target_idx, row_stat));
                }
            }
        }
        S_CUDA_LM_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "            }\n"));
    }
    if (!warp_owned) {
        for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
            S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                buffer, buffer_size, offset,
                "            address%zu += shared_stride_bytes;\n", leaf_idx));
        }
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "        }\n\n"));
    if (warp_owned) {
        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                for (stat_idx = 0u; stat_idx < num_statistics; ++stat_idx) {
                    S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                        buffer, buffer_size, offset,
                        "        #pragma unroll\n"
                        "        for (unsigned int delta = 16u; delta != 0u; delta >>= 1u) {\n"
                        "            lm_%03zu_%03zu_%03zu += __shfl_down_sync(0xffffffffu, lm_%03zu_%03zu_%03zu, delta);\n"
                        "        }\n",
                        ast_idx, target_idx, stat_idx, ast_idx, target_idx, stat_idx));
                }
            }
        }
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            for (stat_idx = 0u; stat_idx < num_statistics; ++stat_idx) {
                S_CUDA_LM_CHECK_RET(_secant_cuda_write(
                    buffer, buffer_size, offset,
                    "        if (%s%zuu < num_asts && %zuu < num_targets) {\n"
                    "            atomicAdd(output_statistics + ((((size_t)%zu * num_targets + %zuu) * %zuu + %zuu) * output_leading_dimension + setting), lm_%03zu_%03zu_%03zu);\n"
                    "        }\n",
                    warp_owned ? "lane == 0u && " : "",
                    ast_idx, target_idx,
                    ast_idx, target_idx, num_statistics, stat_idx,
                    ast_idx, target_idx, stat_idx));
            }
        }
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_dynamic_leaf_lm_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    _SecantCUDADynamicLeafStateOwner state_owner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    size_t total_asts;
    size_t num_expression_inputs;
    size_t num_statistics;
    size_t offset = 0u;
    size_t ast_idx;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_input_columns == 0u || num_input_columns > SECANT_AST_MAX_INPUTS ||
        (num_static_input_columns != 0u && num_static_input_columns != num_input_columns) ||
        num_dynamic_leaves == 0u ||
        num_dynamic_leaves > SECANT_CUDA_DYNAMIC_LEAF_LM_MAX_PARAMETERS ||
        num_targets == 0u || tile_rows == 0u || threads_per_block == 0u ||
        threads_per_block > 1024u ||
        (state_owner == _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP && threads_per_block % 32u != 0u) ||
        (state_owner != _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_THREAD &&
         state_owner != _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP) ||
        asts == NULL || cuda_size_ret == NULL ||
        num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL) ||
        !_secant_cuda_dynamic_constant_sse_checked_add(
            num_static_input_columns, num_dynamic_leaves, &num_expression_inputs) ||
        num_expression_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cuda_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts)) {
        S_CUDA_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    S_CUDA_LM_CHECK_RET(secant_cuda_dynamic_leaf_lm_statistics_count(
        num_dynamic_leaves, &num_statistics));
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    for (ast_idx = 0u; ast_idx < total_asts; ++ast_idx) {
        char function_name[64];
        size_t helper_size = 0u;
        uint32_t num_variables = 0u;
        int name_size = snprintf(
            function_name,
            sizeof(function_name),
            "secant_dynamic_leaf_lm_eval_%03zu_%03zu",
            ast_idx / asts_per_kernel,
            ast_idx % asts_per_kernel);

        if (name_size < 0 || (size_t)name_size >= sizeof(function_name) || asts[ast_idx] == NULL) {
            S_CUDA_LM_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        S_CUDA_LM_CHECK_RET(secant_cuda_ast_dynamic_leaf_forward_gradient_source_generate(
            function_name,
            num_static_input_columns,
            num_dynamic_leaves,
            routines,
            num_routines,
            asts[ast_idx],
            buffer == NULL ? NULL : buffer + offset,
            buffer == NULL || offset > buffer_size ? 0u : buffer_size - offset,
            &helper_size,
            &num_variables));
        if (helper_size == 0u || helper_size - 1u > SIZE_MAX - offset) {
            S_CUDA_LM_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
        }
        offset += helper_size - 1u;
        if (num_variables > max_variables) {
            max_variables = num_variables;
        }
    }
    for (ast_idx = 0u; ast_idx < num_kernels; ++ast_idx) {
        S_CUDA_LM_CHECK_RET(_secant_cuda_dynamic_leaf_lm_emit_kernel(
            ast_idx,
            asts_per_kernel,
            num_input_columns,
            num_static_input_columns,
            num_dynamic_leaves,
            num_targets,
            num_statistics,
            tile_rows,
            state_owner,
            buffer,
            buffer_size,
            &offset));
    }
    S_CUDA_LM_CHECK_RET(_secant_cuda_write_finish(buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_dynamic_leaf_lm_source_generate(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    const SecantAstInstruction* const* routines, size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer, size_t buffer_size, size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    return _secant_cuda_dynamic_leaf_lm_source_generate(
        num_kernels, asts_per_kernel, num_input_columns, num_static_input_columns,
        num_dynamic_leaves, num_targets, tile_rows, threads_per_block,
        _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_THREAD,
        routines, num_routines, asts, buffer, buffer_size,
        cuda_size_ret, max_variables_ret);
}

SecantCUDAResult
secant_cuda_warp_dynamic_leaf_lm_source_generate(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    const SecantAstInstruction* const* routines, size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer, size_t buffer_size, size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    return _secant_cuda_dynamic_leaf_lm_source_generate(
        num_kernels, asts_per_kernel, num_input_columns, num_static_input_columns,
        num_dynamic_leaves, num_targets, tile_rows, threads_per_block,
        _SECANT_CUDA_DYNAMIC_LEAF_STATE_OWNER_WARP,
        routines, num_routines, asts, buffer, buffer_size,
        cuda_size_ret, max_variables_ret);
}

#undef S_CUDA_LM_CHECK_RET
#undef S_CUDA_LM_ERROR_RET

#define S_CUDA_PHILOX_SELECT_ERROR_RET(ans) do { \
    SecantCUDAResult _secant_cuda_philox_select_result = (ans); \
    return _secant_cuda_philox_select_result; \
} while (0)
#define S_CUDA_PHILOX_SELECT_CHECK_RET(ans) do { \
    SecantCUDAResult _secant_cuda_philox_select_check_result = (ans); \
    if (_secant_cuda_philox_select_check_result != SECANT_CUDA_SUCCESS) { \
        S_CUDA_PHILOX_SELECT_ERROR_RET(_secant_cuda_philox_select_check_result); \
    } \
} while (0)

static SecantCUDAResult
_secant_cuda_philox_dynamic_leaf_helpers_emit(
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "static __forceinline__ __device__ uint4 secant_leaf_philox_round(uint4 c, uint2 k) {\n"
        "    const unsigned int hi0 = __umulhi(0xd2511f53u, c.x);\n"
        "    const unsigned int hi1 = __umulhi(0xcd9e8d57u, c.z);\n"
        "    const unsigned int lo0 = 0xd2511f53u * c.x;\n"
        "    const unsigned int lo1 = 0xcd9e8d57u * c.z;\n"
        "    return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_leaf_philox4x32_10(uint4 c, uint2 k) {\n"
        "    #pragma unroll\n"
        "    for (int round = 0; round < 10; ++round) {\n"
        "        c = secant_leaf_philox_round(c, k);\n"
        "        k.x += 0x9e3779b9u;\n"
        "        k.y += 0xbb67ae85u;\n"
        "    }\n"
        "    return c;\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_leaf_random4(\n"
        "    unsigned long long setting, unsigned long long seed,\n"
        "    unsigned long long epoch, unsigned int leaf) {\n"
        "    const uint4 counter = make_uint4(\n"
        "        (unsigned int)setting, (unsigned int)(setting >> 32), leaf, (unsigned int)epoch);\n"
        "    const uint2 key = make_uint2(\n"
        "        (unsigned int)seed ^ (unsigned int)(epoch >> 32) * 0x9e3779b9u,\n"
        "        (unsigned int)(seed >> 32) ^ (unsigned int)epoch * 0xbb67ae85u);\n"
        "    return secant_leaf_philox4x32_10(counter, key);\n"
        "}\n\n"
        "static __forceinline__ __device__ float secant_leaf_constant(\n"
        "    unsigned int word, float radius) {\n"
        "    return radius * ((float)(word >> 8) * 1.1920928955078125e-7f - 1.0f);\n"
        "}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_philox_dynamic_leaf_select_kernel_emit(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    const size_t shared_stride = num_input_columns | 1u;
    size_t ast_idx;
    size_t input_idx;
    size_t leaf_idx;

    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_philox_dynamic_leaf_select_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t num_input_columns,\n"
        "    size_t input_leading_dimension,\n"
        "    const float* __restrict__ target,\n"
        "    size_t num_rows,\n"
        "    size_t num_asts,\n"
        "    unsigned long long seed,\n"
        "    unsigned long long epoch,\n"
        "    unsigned long long setting_offset,\n"
        "    unsigned int column_threshold,\n"
        "    float constant_radius,\n"
        "    size_t num_setting_passes,\n"
        "    unsigned int* __restrict__ output_best_indices,\n"
        "    float* __restrict__ output_best_sse\n"
        ") {\n"
        "    __shared__ float row_tile[%zu];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + (size_t)%zuu * %zuu;\n"
        "    if (blockDim.x != %zuu || num_rows == 0u || num_rows > %zuu ||\n"
        "        num_input_columns == 0u || num_input_columns > %zuu ||\n"
        "        num_asts == 0u || num_asts > %zuu || num_setting_passes == 0u ||\n"
        "        num_setting_passes > 0xffffffffu / %zuu || output_best_indices == nullptr) {\n"
        "        return;\n"
        "    }\n"
        "    for (size_t row = threadIdx.x; row < num_rows; row += blockDim.x) {\n"
        "        #pragma unroll\n"
        "        for (size_t column = 0u; column < %zuu; ++column) {\n"
        "            input_tile[row * %zuu + column] = column < num_input_columns\n"
        "                ? input[column * input_leading_dimension + row]\n"
        "                : 0.0f;\n"
        "        }\n"
        "        target_tile[row] = target[row];\n"
        "    }\n"
        "    __syncthreads();\n\n"
        "    float best_sse = __int_as_float(0x7f800000);\n"
        "    unsigned int best_index = 0xffffffffu;\n"
        "    for (size_t setting_pass = 0u; setting_pass < num_setting_passes; ++setting_pass) {\n"
        "        const unsigned long long setting = setting_offset +\n"
        "            (((unsigned long long)blockIdx.x * num_setting_passes + setting_pass) * blockDim.x +\n"
        "                threadIdx.x);\n"
        "        const unsigned int input_tile_address =\n"
        "            (unsigned int)__cvta_generic_to_shared((const void*)input_tile);\n"
        "        const unsigned int shared_stride_bytes = %zuu;\n",
        kernel_idx,
        tile_rows * shared_stride + tile_rows,
        tile_rows,
        shared_stride,
        threads_per_block,
        tile_rows,
        num_input_columns,
        asts_per_kernel,
        asts_per_kernel,
        num_input_columns,
        shared_stride,
        shared_stride * sizeof(float)));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        const uint4 random%zu = secant_leaf_random4(setting, seed, epoch, %zuu);\n"
            "        const unsigned int is_column%zu = random%zu.x < column_threshold;\n"
            "        const unsigned int column%zu = __umulhi(random%zu.y, (unsigned int)num_input_columns);\n"
            "        float dynamic_input%zu = secant_leaf_constant(random%zu.z, constant_radius);\n"
            "        unsigned int address%zu = input_tile_address + column%zu * sizeof(float);\n",
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx,
            leaf_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        float sse_%03zu = 0.0f;\n",
            ast_idx));
    }
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "        #pragma unroll 1\n"
        "        for (size_t eval_row = 0u; eval_row < num_rows; ++eval_row) {\n"
        "            const float target_value = target_tile[eval_row];\n"));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            asm volatile(\n"
            "                \"{ .reg .pred p; setp.ne.u32 p, %%2, 0; @p ld.shared.f32 %%0, [%%1]; }\"\n"
            "                : \"+f\"(dynamic_input%zu)\n"
            "                : \"r\"(address%zu), \"r\"(is_column%zu)\n"
            "                : \"memory\");\n"
            "            const float input%zu = dynamic_input%zu;\n",
            leaf_idx,
            leaf_idx,
            leaf_idx,
            num_static_input_columns + leaf_idx,
            leaf_idx));
    }
    for (input_idx = 0u; input_idx < num_static_input_columns; ++input_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            const float input%zu = input_tile[eval_row * %zuu + %zuu];\n",
            input_idx,
            shared_stride,
            input_idx));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_CUDA_PHILOX_SELECT_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            {\n"
            "                float prediction;\n"));
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_emit_expression(
            num_static_input_columns,
            num_dynamic_leaves,
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
            routines,
            num_routines,
            routine_names,
            asts[ast_idx],
            "                ",
            "prediction = ",
            ";\n",
            buffer,
            buffer_size,
            offset,
            &num_variables));
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "                const float error = prediction - target_value;\n"
            "                sse_%03zu = error * error + sse_%03zu;\n"
            "            }\n",
            ast_idx,
            ast_idx));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            address%zu += shared_stride_bytes;\n",
            leaf_idx));
    }
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "        }\n"
        "        const unsigned int pass_base = (unsigned int)setting_pass * %zuu;\n",
        asts_per_kernel));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        if (%zuu < num_asts) {\n"
            "            const unsigned int candidate_index = pass_base + %zuu;\n"
            "            const unsigned int candidate_bits = __float_as_uint(sse_%03zu);\n"
            "            if ((candidate_bits & 0x80000000u) == 0u &&\n"
            "                (candidate_bits & 0x7f800000u) != 0x7f800000u &&\n"
            "                (sse_%03zu < best_sse ||\n"
            "                 (sse_%03zu == best_sse && candidate_index < best_index))) {\n"
            "                best_sse = sse_%03zu;\n"
            "                best_index = candidate_index;\n"
            "            }\n"
            "        }\n",
            ast_idx,
            ast_idx,
            ast_idx,
            ast_idx,
            ast_idx,
            ast_idx));
    }
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "    }\n"
        "    const size_t output_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    output_best_indices[output_index] = best_index;\n"
        "    if (output_best_sse != nullptr) {\n"
        "        output_best_sse[output_index] = best_sse;\n"
        "    }\n"
        "}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_philox_dynamic_leaf_reconstruct_emit(
    size_t asts_per_kernel,
    size_t num_dynamic_leaves,
    size_t threads_per_block,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t leaf_idx;

    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_philox_dynamic_leaf_reconstruct(\n"
        "    const unsigned int* __restrict__ best_indices,\n"
        "    size_t num_outputs,\n"
        "    size_t outputs_per_kernel,\n"
        "    size_t num_input_columns,\n"
        "    unsigned long long seed,\n"
        "    unsigned long long epoch,\n"
        "    unsigned long long setting_offset,\n"
        "    unsigned int column_threshold,\n"
        "    float constant_radius,\n"
        "    size_t num_setting_passes,\n"
        "    unsigned int* __restrict__ output_ast_indices,\n"
        "    unsigned int* __restrict__ output_leaf_masks,\n"
        "    unsigned int* __restrict__ output_leaf_words,\n"
        "    size_t output_leaf_words_leading_dimension\n"
        ") {\n"
        "    const size_t output_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (output_idx >= num_outputs || outputs_per_kernel == 0u ||\n"
        "        num_input_columns == 0u || num_setting_passes == 0u) {\n"
        "        return;\n"
        "    }\n"
        "    const unsigned int best_index = best_indices[output_idx];\n"
        "    const unsigned int setting_pass = best_index / %zuu;\n"
        "    const size_t local_output = output_idx %% outputs_per_kernel;\n"
        "    const size_t setting_block = local_output / %zuu;\n"
        "    const size_t setting_thread = local_output %% %zuu;\n"
        "    if (best_index == 0xffffffffu || setting_pass >= num_setting_passes) {\n"
        "        output_ast_indices[output_idx] = 0xffffffffu;\n"
        "        output_leaf_masks[output_idx] = 0u;\n"
        "        return;\n"
        "    }\n"
        "    const unsigned long long setting = setting_offset +\n"
        "        (((unsigned long long)setting_block * num_setting_passes + setting_pass) * %zuu +\n"
        "            setting_thread);\n"
        "    unsigned int leaf_mask = 0u;\n"
        "    output_ast_indices[output_idx] = best_index %% %zuu;\n",
        asts_per_kernel,
        threads_per_block,
        threads_per_block,
        threads_per_block,
        asts_per_kernel));
    for (leaf_idx = 0u; leaf_idx < num_dynamic_leaves; ++leaf_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "    {\n"
            "        const uint4 random = secant_leaf_random4(setting, seed, epoch, %zuu);\n"
            "        const unsigned int is_column = random.x < column_threshold;\n"
            "        if (is_column != 0u) {\n"
            "            leaf_mask |= 1u << %zuu;\n"
            "        }\n"
            "        output_leaf_words[output_idx * output_leaf_words_leading_dimension + %zuu] =\n"
            "            is_column != 0u\n"
            "                ? __umulhi(random.y, (unsigned int)num_input_columns)\n"
            "                : __float_as_uint(secant_leaf_constant(random.z, constant_radius));\n"
            "    }\n",
            leaf_idx,
            leaf_idx,
            leaf_idx));
    }
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "    output_leaf_masks[output_idx] = leaf_mask;\n"
        "}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_philox_dynamic_leaf_select_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    size_t kernel_idx;
    size_t num_expression_inputs;
    size_t total_asts;
    size_t shared_elements;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || asts_per_kernel > UINT32_MAX ||
        num_input_columns == 0u || num_input_columns > SECANT_AST_MAX_INPUTS ||
        (num_static_input_columns != 0u && num_static_input_columns != num_input_columns) ||
        num_dynamic_leaves == 0u || num_dynamic_leaves > 32u ||
        tile_rows == 0u || threads_per_block == 0u || threads_per_block > 1024u ||
        num_routines > SECANT_AST_MAX_ROUTINES || asts == NULL || cuda_size_ret == NULL ||
        (num_routines != 0u && routines == NULL) ||
        !_secant_cuda_dynamic_constant_sse_checked_add(
            num_static_input_columns, num_dynamic_leaves, &num_expression_inputs) ||
        num_expression_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cuda_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts) ||
        !_secant_cuda_sse_checked_mul(tile_rows, num_input_columns | 1u, &shared_elements) ||
        !_secant_cuda_dynamic_constant_sse_checked_add(shared_elements, tile_rows, &shared_elements)) {
        S_CUDA_PHILOX_SELECT_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    (void)total_asts;
    (void)shared_elements;
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_philox_dynamic_leaf_helpers_emit(buffer, buffer_size, &offset));
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_emit_routines(
        routines, num_routines, routine_names, buffer, buffer_size, &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_philox_dynamic_leaf_select_kernel_emit(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_static_input_columns,
            num_dynamic_leaves,
            tile_rows,
            threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts + kernel_idx * asts_per_kernel,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_philox_dynamic_leaf_reconstruct_emit(
        asts_per_kernel,
        num_dynamic_leaves,
        threads_per_block,
        buffer,
        buffer_size,
        &offset));
    S_CUDA_PHILOX_SELECT_CHECK_RET(_secant_cuda_write_finish(buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_CUDA_SUCCESS;
}

#undef S_CUDA_PHILOX_SELECT_CHECK_RET
#undef S_CUDA_PHILOX_SELECT_ERROR_RET

#define S_CUDA_PACKED_OPTIMIZER_ERROR_RET(ans) do { \
    SecantCUDAResult _secant_cuda_packed_optimizer_result = (ans); \
    return _secant_cuda_packed_optimizer_result; \
} while (0)
#define S_CUDA_PACKED_OPTIMIZER_CHECK_RET(ans) do { \
    SecantCUDAResult _secant_cuda_packed_optimizer_check_result = (ans); \
    if (_secant_cuda_packed_optimizer_check_result != SECANT_CUDA_SUCCESS) { \
        S_CUDA_PACKED_OPTIMIZER_ERROR_RET(_secant_cuda_packed_optimizer_check_result); \
    } \
} while (0)

static SecantCUDAResult
_secant_cuda_packed_optimizer_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const float* current_constants,
    const float* current_constant_scales,
    size_t current_constants_leading_dimension,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    static const char* const lanes[] = {"x", "y", "z", "w"};
    const size_t tile_elements = (num_input_columns + 1u) * tile_rows;
    size_t ast_idx;
    size_t constant_idx;
    size_t group_idx;
    size_t input_idx;

    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "extern \"C\" __global__\n"
        "void secant_packed_constant_optimizer_sse_%03zu(\n"
        "    const float* __restrict__ input,\n"
        "    size_t input_leading_dimension,\n"
        "    size_t num_settings,\n"
        "    const float* __restrict__ target,\n"
        "    size_t num_rows,\n"
        "    unsigned long long seed,\n"
        "    unsigned long long generation,\n"
        "    unsigned long long iteration,\n"
        "    size_t num_asts,\n"
        "    float* __restrict__ output_sse,\n"
        "    size_t output_leading_dimension\n"
        ") {\n"
        "    __shared__ float row_tile[%zu];\n"
        "    float* const input_tile = row_tile;\n"
        "    float* const target_tile = row_tile + %zu;\n"
        "    const size_t tile_begin = (size_t)blockIdx.x * %zuu;\n"
        "    const size_t remaining_rows = tile_begin < num_rows ? num_rows - tile_begin : 0u;\n"
        "    const size_t tile_num_rows = remaining_rows < %zuu ? remaining_rows : %zuu;\n\n"
        "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const size_t row = tile_begin + tile_row;\n",
        kernel_idx,
        tile_elements,
        num_input_columns * tile_rows,
        tile_rows,
        tile_rows,
        tile_rows));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        input_tile[(size_t)%zu * %zuu + tile_row] = "
                "input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            tile_rows,
            input_idx));
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "        target_tile[tile_row] = target[row];\n"
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
    for (group_idx = 0u; group_idx < (num_input_constants + 3u) / 4u; ++group_idx) {
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        const uint4 random%zu = setting == 0u ? make_uint4(0u, 0u, 0u, 0u) :\n"
            "            secant_packed_eval_random4((unsigned long long)setting, seed, generation, iteration, %zuu);\n",
            group_idx,
            group_idx));
    }
    for (constant_idx = 0u; constant_idx < num_input_constants; ++constant_idx) {
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        const float jitter%zu = setting == 0u ? 0.0f :\n"
            "            secant_packed_eval_delta(random%zu.%s);\n",
            constant_idx,
            constant_idx / 4u,
            lanes[constant_idx % 4u]));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer, buffer_size, offset, "        float sse_%03zu_000 = 0.0f;\n", ast_idx));
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "        #pragma unroll 1\n"
        "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
            input_idx,
            input_idx,
            tile_rows));
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, offset, "            const float target0 = target_tile[eval_row];\n\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_CUDA_PACKED_OPTIMIZER_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "            {\n"
            "                float prediction;\n"));
        for (constant_idx = 0u; constant_idx < num_input_constants; ++constant_idx) {
            const size_t state_idx = ast_idx * current_constants_leading_dimension + constant_idx;
            uint32_t center_bits;
            uint32_t scale_bits;

            memcpy(&center_bits, current_constants + state_idx, sizeof(center_bits));
            memcpy(&scale_bits, current_constant_scales + state_idx, sizeof(scale_bits));
            S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
                buffer,
                buffer_size,
                offset,
                "                const float input%zu = secant_cuda_scale_center_f32(\n"
                "                    jitter%zu, __uint_as_float(0x%08xu), __uint_as_float(0x%08xu));\n",
                num_input_columns + constant_idx,
                constant_idx,
                scale_bits,
                center_bits));
        }
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_emit_expression(
            num_input_columns,
            num_input_constants,
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
            routines,
            num_routines,
            routine_names,
            asts[ast_idx],
            "                ",
            "prediction = ",
            ";\n",
            buffer,
            buffer_size,
            offset,
            &num_variables));
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "                const float error = prediction - target0;\n"
            "                sse_%03zu_000 = error * error + sse_%03zu_000;\n"
            "            }\n\n",
            ast_idx,
            ast_idx));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "        }\n\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
            buffer,
            buffer_size,
            offset,
            "        if (%zuu < num_asts) {\n"
            "            atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + setting, "
                "sse_%03zu_000);\n"
            "        }\n",
            ast_idx,
            ast_idx,
            ast_idx));
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(buffer, buffer_size, offset, "    }\n}\n\n"));
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_packed_constant_optimizer_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const float* current_constants,
    const float* current_constant_scales,
    size_t current_constants_leading_dimension,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
) {
    const char* reducer_source;
    size_t reducer_source_size;
    size_t kernel_idx;
    size_t num_inputs;
    size_t total_asts;
    size_t required_state;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants > SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        tile_rows == 0u || threads_per_block == 0u || threads_per_block > 1024u ||
        asts == NULL || current_constants == NULL || current_constant_scales == NULL ||
        current_constants_leading_dimension < num_input_constants || cuda_size_ret == NULL ||
        num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL) ||
        !_secant_cuda_dynamic_constant_sse_checked_add(
            num_input_columns, num_input_constants, &num_inputs) ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        !_secant_cuda_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts) ||
        !_secant_cuda_sse_checked_mul(total_asts, current_constants_leading_dimension, &required_state)) {
        S_CUDA_PACKED_OPTIMIZER_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    (void)required_state;
    reducer_source = secant_cuda_packed_constant_optimizer_reduce_f32_source_get(&reducer_source_size);
    if (reducer_source == NULL || reducer_source_size < 2u ||
        reducer_source[reducer_source_size - 1u] != '\0') {
        S_CUDA_PACKED_OPTIMIZER_ERROR_RET(SECANT_CUDA_ERROR_INVALID_STATE);
    }
    *cuda_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_emit_prelude(buffer, buffer_size, &offset));
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
        buffer,
        buffer_size,
        &offset,
        "static __device__ __forceinline__ float secant_cuda_scale_center_f32(\n"
        "    float jitter, float scale, float center) {\n"
        "    float result;\n"
        "    asm volatile(\n"
        "        \"mul.ftz.f32 %%0, %%1, %%2;\\n\\tadd.ftz.f32 %%0, %%0, %%3;\"\n"
        "        : \"=&f\"(result) : \"f\"(jitter), \"f\"(scale), \"f\"(center));\n"
        "    return result;\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_packed_eval_philox_round(uint4 c, uint2 k) {\n"
        "    const unsigned int hi0 = __umulhi(0xd2511f53u, c.x);\n"
        "    const unsigned int hi1 = __umulhi(0xcd9e8d57u, c.z);\n"
        "    const unsigned int lo0 = 0xd2511f53u * c.x;\n"
        "    const unsigned int lo1 = 0xcd9e8d57u * c.z;\n"
        "    return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_packed_eval_philox4x32_10(uint4 c, uint2 k) {\n"
        "    #pragma unroll\n"
        "    for (int round = 0; round < 10; ++round) {\n"
        "        c = secant_packed_eval_philox_round(c, k);\n"
        "        k.x += 0x9e3779b9u;\n"
        "        k.y += 0xbb67ae85u;\n"
        "    }\n"
        "    return c;\n"
        "}\n\n"
        "static __forceinline__ __device__ uint4 secant_packed_eval_random4(\n"
        "    unsigned long long setting, unsigned long long seed, unsigned long long generation,\n"
        "    unsigned long long iteration, unsigned int group) {\n"
        "    const uint4 counter = make_uint4((unsigned int)setting, (unsigned int)(setting >> 32), 0u, 0u);\n"
        "    const uint2 key = make_uint2(\n"
        "        (unsigned int)seed ^ (unsigned int)generation * 0x9e3779b9u ^\n"
        "            (unsigned int)(iteration >> 32) * 0x85ebca6bu ^ group * 0x27d4eb2du,\n"
        "        (unsigned int)(seed >> 32) ^ (unsigned int)(generation >> 32) * 0xbb67ae85u ^\n"
        "            (unsigned int)iteration * 0xc2b2ae35u ^ group * 0x165667b1u);\n"
        "    return secant_packed_eval_philox4x32_10(counter, key);\n"
        "}\n\n"
        "static __forceinline__ __device__ float secant_packed_eval_delta(unsigned int word) {\n"
        "    return (float)(word >> 8) * 1.1920928955078125e-7f - 1.0f;\n"
        "}\n\n"));
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_emit_routines(
        routines, num_routines, routine_names, buffer, buffer_size, &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        const size_t first_ast = kernel_idx * asts_per_kernel;

        S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_packed_optimizer_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_input_constants,
            tile_rows,
            routines,
            num_routines,
            routine_names,
            asts + first_ast,
            current_constants + first_ast * current_constants_leading_dimension,
            current_constant_scales + first_ast * current_constants_leading_dimension,
            current_constants_leading_dimension,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write(
        buffer, buffer_size, &offset, "\n%s", reducer_source));
    S_CUDA_PACKED_OPTIMIZER_CHECK_RET(_secant_cuda_write_finish(
        buffer, buffer_size, offset, cuda_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_CUDA_SUCCESS;
}

#undef S_CUDA_PACKED_OPTIMIZER_CHECK_RET
#undef S_CUDA_PACKED_OPTIMIZER_ERROR_RET

#undef _SECANT_CUDA_ERROR_RET
