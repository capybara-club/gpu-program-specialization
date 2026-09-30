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
#include "secant_hip.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifndef SECANT_HIP_ROUTINE_DEPTH
#define SECANT_HIP_ROUTINE_DEPTH 8u
#endif

#define _SECANT_HIP_ERROR_RET(ans) do { SecantHIPResult secant_hip_result = (ans); return secant_hip_result; } while (0)

static SecantHIPResult
_secant_hip_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
) {
    va_list args;
    int bytes;

    if (offset == NULL || format == NULL) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    va_start(args, format);
    if (buffer != NULL && *offset < buffer_size) {
        bytes = vsnprintf(buffer + *offset, buffer_size - *offset, format, args);
    } else {
        bytes = vsnprintf(NULL, 0u, format, args);
    }
    va_end(args);
    if (bytes < 0) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_FORMAT);
    }
    if ((size_t)bytes > SIZE_MAX - *offset) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }
    *offset += (size_t)bytes;
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_write_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
) {
    if (size_ret == NULL) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    if (offset == SIZE_MAX) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }
    *size_ret = offset + 1u;
    if (buffer != NULL) {
        if (buffer_size <= offset) {
            _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INSUFFICIENT_BUFFER);
        }
        buffer[offset] = '\0';
    }
    return SECANT_HIP_SUCCESS;
}

/* HIP expression renderer and kernel generators. */
#define S_HIP_ERROR_RET(ans) \
    do { \
        SecantHIPResult _secant_hip_result = (ans); \
        return _secant_hip_result; \
    } while (0)

#define S_HIP_CHECK_RET(ans) \
    do { \
        SecantHIPResult _secant_hip_check_result = (ans); \
        if (_secant_hip_check_result != SECANT_HIP_SUCCESS) { \
            S_HIP_ERROR_RET(_secant_hip_check_result); \
        } \
    } while (0)

typedef char SSecantAstInstructionMustBeEightBytes[(sizeof(SecantAstInstruction) == 8u) ? 1 : -1];

typedef enum SHipOperandKind {
    S_HIP_OPERAND_INPUT = 0,
    S_HIP_OPERAND_IMMEDIATE = 1,
    S_HIP_OPERAND_VARIABLE = 2,
    S_HIP_OPERAND_ROUTINE_ARG = 3
} SHipOperandKind;

typedef struct SHipOperand {
    SHipOperandKind kind;
    uint32_t value;
    uint8_t owned;
} SHipOperand;

typedef struct SHipAssembler {
    SHipOperand stack[SECANT_AST_MAX_STACK_DEPTH];
    size_t stack_size;
    uint32_t free_variables[SECANT_AST_MAX_STACK_DEPTH];
    size_t num_free_variables;
    uint32_t next_variable;
    uint32_t high_water_variable;
} SHipAssembler;

static int
_secant_hip_identifier_is_valid(const char* name) {
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

static SecantHIPResult
_secant_hip_emit_routine_name(
    const char* const* routine_names,
    size_t routine_idx,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    if (routine_names == NULL) {
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "secant_hip_routine_%03zu", routine_idx));
    } else {
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "secant_hip_routine_%s", routine_names[routine_idx]));
    }

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_routine_num_args(const SecantAstInstruction* instructions, size_t* num_args_ret) {
    size_t instruction_idx;
    size_t num_args = 0u;

    if (instructions == NULL || num_args_ret == NULL) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }

    for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
        const SecantAstInstruction instruction = instructions[instruction_idx];

        if ((SecantAstInstructionType)instruction.instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            const size_t arg_idx = instruction.payload.idx;

            if (arg_idx >= SECANT_AST_MAX_INSTRUCTION_ARGS) {
                S_HIP_ERROR_RET(SECANT_HIP_ERROR_TOO_MANY_ARGS);
            }
            if (arg_idx + 1u > num_args) {
                num_args = arg_idx + 1u;
            }
        }
        if ((SecantAstInstructionType)instruction.instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_HIP_SUCCESS;
        }
    }

    S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
}

static SecantHIPResult
_secant_hip_emit_prelude(char* buffer, size_t buffer_size, size_t* offset) {
    S_HIP_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "#include <hip/hip_runtime.h>\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_rcp_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"v_rcp_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_div_f32(float lhs, float rhs) {\n"
        "    return lhs * secant_hip_rcp_f32(rhs);\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_sqrt_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"v_sqrt_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_sin_f32(float value) {\n"
        "    float result;\n"
        "    value *= 0.15915494309189535f;\n"
        "    asm volatile(\"v_sin_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_cos_f32(float value) {\n"
        "    float result;\n"
        "    value *= 0.15915494309189535f;\n"
        "    asm volatile(\"v_cos_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_ex2_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"v_exp_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_lg2_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"v_log_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_rsqrt_f32(float value) {\n"
        "    float result;\n"
        "    asm volatile(\"v_rsq_f32 %0, %1\" : \"=v\"(result) : \"v\"(value));\n"
        "    return result;\n"
        "}\n\n"
        "static __device__ __forceinline__ __attribute__((unused)) float secant_hip_tanh_f32(float value) {\n"
        "    value = secant_hip_ex2_f32(value * -2.8853900817779268f);\n"
        "    value = secant_hip_rcp_f32(value + 1.0f);\n"
        "    return value * 2.0f - 1.0f;\n"
        "}\n\n"));

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_push(SHipAssembler* assembler, SHipOperand operand) {
    if (assembler->stack_size >= SECANT_AST_MAX_STACK_DEPTH) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_STACK_OVERFLOW);
    }

    assembler->stack[assembler->stack_size++] = operand;
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_pop(SHipAssembler* assembler, SHipOperand* operand_ret) {
    if (assembler->stack_size == 0u) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_STACK_UNDERFLOW);
    }

    assembler->stack_size -= 1u;
    *operand_ret = assembler->stack[assembler->stack_size];
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_allocate_variable(SHipAssembler* assembler, SHipOperand* operand_ret, int* declaration_ret) {
    SHipOperand operand;

    memset(&operand, 0, sizeof(operand));
    operand.kind = S_HIP_OPERAND_VARIABLE;
    operand.owned = 1u;

    if (assembler->num_free_variables != 0u) {
        assembler->num_free_variables -= 1u;
        operand.value = assembler->free_variables[assembler->num_free_variables];
        *declaration_ret = 0;
    } else {
        if (assembler->next_variable > SECANT_AST_MAX_STACK_DEPTH) {
            S_HIP_ERROR_RET(SECANT_HIP_ERROR_STACK_OVERFLOW);
        }
        operand.value = assembler->next_variable++;
        if (operand.value > assembler->high_water_variable) {
            assembler->high_water_variable = operand.value;
        }
        *declaration_ret = 1;
    }

    *operand_ret = operand;
    return SECANT_HIP_SUCCESS;
}

static void
_secant_hip_release_variable(SHipAssembler* assembler, SHipOperand operand) {
    if (operand.kind == S_HIP_OPERAND_VARIABLE && operand.owned && assembler->num_free_variables < SECANT_AST_MAX_STACK_DEPTH) {
        assembler->free_variables[assembler->num_free_variables++] = operand.value;
    }
}

static SecantHIPResult
_secant_hip_emit_operand(
    SHipOperand operand,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (operand.kind) {
        case S_HIP_OPERAND_INPUT:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "input%u", (unsigned)operand.value));
            break;
        case S_HIP_OPERAND_IMMEDIATE:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "__uint_as_float(0x%08xu)", (unsigned)operand.value));
            break;
        case S_HIP_OPERAND_VARIABLE:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "x%u", (unsigned)operand.value));
            break;
        case S_HIP_OPERAND_ROUTINE_ARG:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "arg%u", (unsigned)operand.value));
            break;
        default:
            S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
    }

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_emit_assignment_prefix(
    SHipOperand dst,
    int declaration,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    if (declaration) {
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "%sfloat x%u = ", indent, (unsigned)dst.value));
    } else {
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "%sx%u = ", indent, (unsigned)dst.value));
    }

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_emit_operation_expression(
    SecantAstInstructionType op,
    const SHipOperand* args,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    switch (op) {
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "("));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(
                buffer,
                buffer_size,
                offset,
                op == SECANT_AST_INSTRUCTION_TYPE_ADD_F32 ? " + " : (op == SECANT_AST_INSTRUCTION_TYPE_SUB_F32 ? " - " : " * ")));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[1], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "secant_hip_div_f32("));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ", "));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[1], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "-("));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "fabsf("));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, op == SECANT_AST_INSTRUCTION_TYPE_MIN_F32 ? "fminf(" : "fmaxf("));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ", "));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[1], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "fmaf("));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ", "));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[1], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ", "));
            S_HIP_CHECK_RET(_secant_hip_emit_operand(args[2], buffer, buffer_size, offset));
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
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
                    case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: function_name = "secant_hip_sqrt_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: function_name = "secant_hip_rcp_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: function_name = "secant_hip_sin_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_COS_F32: function_name = "secant_hip_cos_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: function_name = "secant_hip_ex2_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: function_name = "secant_hip_lg2_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: function_name = "secant_hip_rsqrt_f32"; break;
                    case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: function_name = "secant_hip_tanh_f32"; break;
                    default: break;
                }

                S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "%s(", function_name));
                S_HIP_CHECK_RET(_secant_hip_emit_operand(args[0], buffer, buffer_size, offset));
                S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
            }
            break;
        case SECANT_AST_INSTRUCTION_TYPE_NONE:
        case SECANT_AST_INSTRUCTION_TYPE_NUM_ENUMS:
        default:
            S_HIP_ERROR_RET(SECANT_HIP_ERROR_UNSUPPORTED_OP);
    }

    return SECANT_HIP_SUCCESS;
}

static int
_secant_hip_operand_matches(SHipOperand lhs, SHipOperand rhs) {
    return lhs.kind == S_HIP_OPERAND_VARIABLE && rhs.kind == S_HIP_OPERAND_VARIABLE && lhs.value == rhs.value;
}

static void
_secant_hip_release_consumed(SHipAssembler* assembler, const SHipOperand* args, size_t num_args, SHipOperand keep) {
    size_t i;

    for (i = 0u; i < num_args; ++i) {
        size_t j;
        int duplicate = 0;

        if (!args[i].owned || args[i].kind != S_HIP_OPERAND_VARIABLE || _secant_hip_operand_matches(args[i], keep)) {
            continue;
        }

        for (j = 0u; j < i; ++j) {
            if (args[j].owned && _secant_hip_operand_matches(args[j], args[i])) {
                duplicate = 1;
                break;
            }
        }

        if (!duplicate) {
            _secant_hip_release_variable(assembler, args[i]);
        }
    }
}

static SecantHIPResult
_secant_hip_emit_op(
    SHipAssembler* assembler,
    SecantAstInstructionType op,
    size_t num_args,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    SHipOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
    SHipOperand dst;
    int declaration = 0;
    size_t i;

    for (i = num_args; i != 0u; --i) {
        S_HIP_CHECK_RET(_secant_hip_pop(assembler, args + i - 1u));
    }

    memset(&dst, 0, sizeof(dst));
    for (i = 0u; i < num_args; ++i) {
        if (args[i].kind == S_HIP_OPERAND_VARIABLE && args[i].owned) {
            dst = args[i];
            break;
        }
    }

    if (i == num_args) {
        S_HIP_CHECK_RET(_secant_hip_allocate_variable(assembler, &dst, &declaration));
    }

    S_HIP_CHECK_RET(_secant_hip_emit_assignment_prefix(dst, declaration, indent, buffer, buffer_size, offset));
    S_HIP_CHECK_RET(_secant_hip_emit_operation_expression(op, args, buffer, buffer_size, offset));
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ";\n"));

    dst.owned = 1u;
    _secant_hip_release_consumed(assembler, args, num_args, dst);
    S_HIP_CHECK_RET(_secant_hip_push(assembler, dst));
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_emit_routine_call(
    SHipAssembler* assembler,
    size_t routine_idx,
    size_t num_args,
    const char* const* routine_names,
    const char* indent,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    SHipOperand args[SECANT_AST_MAX_INSTRUCTION_ARGS];
    SHipOperand dst;
    int declaration = 0;
    size_t i;

    for (i = num_args; i != 0u; --i) {
        S_HIP_CHECK_RET(_secant_hip_pop(assembler, args + i - 1u));
    }

    memset(&dst, 0, sizeof(dst));
    for (i = 0u; i < num_args; ++i) {
        if (args[i].kind == S_HIP_OPERAND_VARIABLE && args[i].owned) {
            dst = args[i];
            break;
        }
    }

    if (i == num_args) {
        S_HIP_CHECK_RET(_secant_hip_allocate_variable(assembler, &dst, &declaration));
    }

    S_HIP_CHECK_RET(_secant_hip_emit_assignment_prefix(dst, declaration, indent, buffer, buffer_size, offset));
    S_HIP_CHECK_RET(_secant_hip_emit_routine_name(routine_names, routine_idx, buffer, buffer_size, offset));
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "("));
    for (i = 0u; i < num_args; ++i) {
        if (i != 0u) {
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ", "));
        }
        S_HIP_CHECK_RET(_secant_hip_emit_operand(args[i], buffer, buffer_size, offset));
    }
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ");\n"));

    dst.owned = 1u;
    _secant_hip_release_consumed(assembler, args, num_args, dst);
    S_HIP_CHECK_RET(_secant_hip_push(assembler, dst));
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_compile_frame(
    SHipAssembler* assembler,
    size_t num_inputs,
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
    size_t instruction_idx;

    for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
        const SecantAstInstruction instruction = instructions[instruction_idx];

        switch ((SecantAstInstructionType)instruction.instruction_type) {
            case SECANT_AST_INSTRUCTION_TYPE_STATIC_COLUMN_INPUT_F32:
                {
                    SHipOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    if (instruction.payload.idx >= num_inputs) {
                        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
                    }
                    operand.kind = S_HIP_OPERAND_INPUT;
                    operand.value = instruction.payload.idx;
                    S_HIP_CHECK_RET(_secant_hip_push(assembler, operand));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
                {
                    SHipOperand operand;
                    memset(&operand, 0, sizeof(operand));
                    operand.kind = S_HIP_OPERAND_IMMEDIATE;
                    operand.value =
                        secant_ast_constant_f32_bits_get(&instruction);
                    S_HIP_CHECK_RET(_secant_hip_push(assembler, operand));
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
                {
                    SHipOperand operand;

                    if (instruction.payload.idx >= num_routine_args) {
                        S_HIP_ERROR_RET(SECANT_HIP_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                    }
                    memset(&operand, 0, sizeof(operand));
                    operand.kind = S_HIP_OPERAND_ROUTINE_ARG;
                    operand.value = instruction.payload.idx;
                    S_HIP_CHECK_RET(_secant_hip_push(assembler, operand));
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
                    const SecantAstInstructionType instruction_type =
                        (SecantAstInstructionType)instruction.instruction_type;
                    const uint16_t num_args =
                        secant_ast_instruction_num_args[instruction_type];

                    S_HIP_CHECK_RET(_secant_hip_emit_op(
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
                    const size_t num_args = instruction.aux;
                    size_t expected_args;

                    if (num_args > SECANT_AST_MAX_INSTRUCTION_ARGS) {
                        S_HIP_ERROR_RET(SECANT_HIP_ERROR_TOO_MANY_ARGS);
                    }
                    if (assembler->stack_size < num_args) {
                        S_HIP_ERROR_RET(SECANT_HIP_ERROR_STACK_UNDERFLOW);
                    }
                    if (routines == NULL || instruction.payload.idx >= num_routines || routines[instruction.payload.idx] == NULL) {
                        S_HIP_ERROR_RET(SECANT_HIP_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                    }
                    S_HIP_CHECK_RET(_secant_hip_routine_num_args(routines[instruction.payload.idx], &expected_args));
                    if (num_args != expected_args) {
                        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
                    }
                    S_HIP_CHECK_RET(_secant_hip_emit_routine_call(
                        assembler,
                        instruction.payload.idx,
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
                    S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
                }
                return SECANT_HIP_SUCCESS;
            case SECANT_AST_INSTRUCTION_TYPE_NONE:
            case SECANT_AST_INSTRUCTION_TYPE_NUM_ENUMS:
            default:
                S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
        }
    }

    S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
}

static SecantHIPResult
_secant_hip_emit_body(
    size_t num_inputs,
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
    SHipAssembler assembler;
    SHipOperand result;

    if (num_inputs > UINT32_MAX || num_routine_args > SECANT_AST_MAX_INSTRUCTION_ARGS || instructions == NULL ||
        indent == NULL || result_prefix == NULL || result_suffix == NULL || offset == NULL) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }

    memset(&assembler, 0, sizeof(assembler));
    assembler.next_variable = 1u;

    S_HIP_CHECK_RET(_secant_hip_compile_frame(
        &assembler,
        num_inputs,
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
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
    }
    S_HIP_CHECK_RET(_secant_hip_pop(&assembler, &result));
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "%s%s", indent, result_prefix));
    S_HIP_CHECK_RET(_secant_hip_emit_operand(result, buffer, buffer_size, offset));
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "%s", result_suffix));

    if (num_variables_ret != NULL) {
        *num_variables_ret = assembler.high_water_variable;
    }

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_emit_expression(
    size_t num_inputs,
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
    if (num_inputs == 0u) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }

    S_HIP_ERROR_RET(_secant_hip_emit_body(
        num_inputs,
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

static SecantHIPResult
_secant_hip_emit_routine_parameters(size_t num_args, char* buffer, size_t buffer_size, size_t* offset) {
    size_t arg_idx;

    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "("));
    for (arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
        if (arg_idx != 0u) {
            S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ", "));
        }
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "const float arg%zu", arg_idx));
    }
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")"));
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_validate_routine_graph(
    size_t routine_idx,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    size_t* path,
    size_t depth
) {
    const SecantAstInstruction* instructions = routines[routine_idx];
    size_t instruction_idx;

    for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
        const SecantAstInstruction instruction = instructions[instruction_idx];

        if ((SecantAstInstructionType)instruction.instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
            const size_t callee_idx = instruction.payload.idx;
            size_t callee_args;
            size_t path_idx;

            if (callee_idx >= num_routines || routines[callee_idx] == NULL) {
                S_HIP_ERROR_RET(SECANT_HIP_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
            }
            S_HIP_CHECK_RET(_secant_hip_routine_num_args(routines[callee_idx], &callee_args));
            if (instruction.aux != callee_args) {
                S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
            }
            if (depth >= SECANT_HIP_ROUTINE_DEPTH) {
                S_HIP_ERROR_RET(SECANT_HIP_ERROR_ROUTINE_DEPTH_EXCEEDED);
            }
            for (path_idx = 0u; path_idx < depth; ++path_idx) {
                if (path[path_idx] == callee_idx) {
                    S_HIP_ERROR_RET(SECANT_HIP_ERROR_ROUTINE_DEPTH_EXCEEDED);
                }
            }

            path[depth] = callee_idx;
            S_HIP_CHECK_RET(_secant_hip_validate_routine_graph(callee_idx, routines, num_routines, path, depth + 1u));
        }
        if ((SecantAstInstructionType)instruction.instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return SECANT_HIP_SUCCESS;
        }
    }

    S_HIP_ERROR_RET(SECANT_HIP_ERROR_BAD_PROGRAM);
}

static SecantHIPResult
_secant_hip_emit_routines(
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    char* buffer,
    size_t buffer_size,
    size_t* offset
) {
    size_t routine_idx;
    size_t path[SECANT_HIP_ROUTINE_DEPTH];

    if (offset == NULL || (num_routines != 0u && routines == NULL)) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t previous_idx;

        if (routines[routine_idx] == NULL ||
            (routine_names != NULL && !_secant_hip_identifier_is_valid(routine_names[routine_idx]))) {
            S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
        }
        if (routine_names != NULL) {
            for (previous_idx = 0u; previous_idx < routine_idx; ++previous_idx) {
                if (strcmp(routine_names[previous_idx], routine_names[routine_idx]) == 0) {
                    S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
                }
            }
        }

        path[0] = routine_idx;
        S_HIP_CHECK_RET(_secant_hip_validate_routine_graph(routine_idx, routines, num_routines, path, 1u));
    }

    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t num_args;

        S_HIP_CHECK_RET(_secant_hip_routine_num_args(routines[routine_idx], &num_args));
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "static __device__ __forceinline__ __attribute__((unused)) float "));
        S_HIP_CHECK_RET(_secant_hip_emit_routine_name(routine_names, routine_idx, buffer, buffer_size, offset));
        S_HIP_CHECK_RET(_secant_hip_emit_routine_parameters(num_args, buffer, buffer_size, offset));
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ";\n"));
    }
    if (num_routines != 0u) {
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "\n"));
    }

    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t num_args;

        S_HIP_CHECK_RET(_secant_hip_routine_num_args(routines[routine_idx], &num_args));
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "static __device__ __forceinline__ __attribute__((unused)) float "));
        S_HIP_CHECK_RET(_secant_hip_emit_routine_name(routine_names, routine_idx, buffer, buffer_size, offset));
        S_HIP_CHECK_RET(_secant_hip_emit_routine_parameters(num_args, buffer, buffer_size, offset));
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "\n{\n"));
        S_HIP_CHECK_RET(_secant_hip_emit_body(
            0u,
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
        S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "}\n\n"));
    }

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_emit_function(
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

    if (offset == NULL || !_secant_hip_identifier_is_valid(function_name) || num_inputs == 0u ||
        num_inputs > UINT32_MAX || instructions == NULL) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }

    S_HIP_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "static __device__ __forceinline__ float %s(\n",
        function_name));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_HIP_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "    const float input%zu%s\n",
            input_idx,
            input_idx + 1u == num_inputs ? "" : ","));
    }
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, ")\n{\n"));

    S_HIP_CHECK_RET(_secant_hip_emit_expression(
        num_inputs,
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
    S_HIP_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "}\n\n"));

    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_function_source_generate_impl(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* num_variables_ret
) {
    size_t offset = 0u;

    S_HIP_CHECK_RET(_secant_hip_emit_prelude(buffer, buffer_size, &offset));
    S_HIP_CHECK_RET(_secant_hip_emit_routines(routines, num_routines, routine_names, buffer, buffer_size, &offset));
    S_HIP_CHECK_RET(_secant_hip_emit_function(
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
    S_HIP_CHECK_RET(_secant_hip_write_finish(buffer, buffer_size, offset, hip_size_ret));
    return SECANT_HIP_SUCCESS;
}

SecantHIPResult
secant_hip_function_source_generate(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* num_variables_ret
) {
    if (hip_size_ret == NULL) {
        S_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }

    *hip_size_ret = 0u;
    if (num_variables_ret != NULL) {
        *num_variables_ret = 0u;
    }

    S_HIP_ERROR_RET(_secant_hip_function_source_generate_impl(
        function_name,
        num_inputs,
        routines,
        num_routines,
        routine_names,
        instructions,
        buffer,
        buffer_size,
        hip_size_ret,
        num_variables_ret));
}

#undef S_HIP_CHECK_RET
#undef S_HIP_ERROR_RET

#define S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(ans) \
    do { \
        SecantHIPResult _secant_hip_materialize_result = (ans); \
        return _secant_hip_materialize_result; \
    } while (0)

#define S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(ans) \
    do { \
        SecantHIPResult _secant_hip_materialize_check_result = (ans); \
        if (_secant_hip_materialize_check_result != SECANT_HIP_SUCCESS) { \
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(_secant_hip_materialize_check_result); \
        } \
    } while (0)

static int
_secant_hip_materialize_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }

    *result_ret = lhs * rhs;
    return 1;
}

static SecantHIPResult
_secant_hip_materialize_emit_kernel(
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

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write(
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
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "    const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            input_idx));
    }
    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        char result_prefix[128];
        uint32_t num_variables = 0u;
        const int result_prefix_bytes = snprintf(
            result_prefix,
            sizeof(result_prefix),
            "output[(size_t)%zu * output_leading_dimension + row] = ",
            ast_idx);

        if (result_prefix_bytes < 0) {
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_HIP_ERROR_FORMAT);
        }
        if ((size_t)result_prefix_bytes >= sizeof(result_prefix)) {
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
        }
        if (asts[ast_idx] == NULL) {
            S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
        }

        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "    {\n"));
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_emit_expression(
            num_inputs,
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
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "    }\n\n"));

        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "}\n\n"));
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_materialize_impl(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
) {
    uint32_t max_variables = 0u;
    size_t total_asts;
    size_t kernel_idx;
    size_t offset = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u || num_inputs > UINT32_MAX ||
        asts == NULL || hip_size_ret == NULL ||
        (num_routines != 0u && routines == NULL)) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    if (!_secant_hip_materialize_checked_mul(num_kernels, asts_per_kernel, &total_asts)) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }

    (void)total_asts;
    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_emit_prelude(buffer, buffer_size, &offset));
    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_emit_routines(
        routines,
        num_routines,
        routine_names,
        buffer,
        buffer_size,
        &offset));

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_materialize_emit_kernel(
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

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET(_secant_hip_write_finish(buffer, buffer_size, offset, hip_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }

    return SECANT_HIP_SUCCESS;
}

SecantHIPResult
secant_hip_materialize_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
) {
    if (hip_size_ret == NULL) {
        S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }

    *hip_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }

    S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET(_secant_hip_materialize_impl(
        num_kernels,
        asts_per_kernel,
        num_inputs,
        routines,
        num_routines,
        routine_names,
        asts,
        buffer,
        buffer_size,
        hip_size_ret,
        max_variables_ret));
}

#undef S_STATIC_COLUMN_MATERIALIZE_GENERATOR_CHECK_RET
#undef S_STATIC_COLUMN_MATERIALIZE_GENERATOR_ERROR_RET

#define S_HIP_SSE_ERROR_RET(ans) do { SecantHIPResult _secant_hip_sse_result = (ans); return _secant_hip_sse_result; } while (0)
#define S_HIP_SSE_CHECK_RET(ans) do { SecantHIPResult _secant_hip_sse_check_result = (ans); if (_secant_hip_sse_check_result != SECANT_HIP_SUCCESS) { S_HIP_SSE_ERROR_RET(_secant_hip_sse_check_result); } } while (0)

static int
_secant_hip_sse_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static SecantHIPResult
_secant_hip_sse_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    uint32_t* max_variables
) {
    const size_t num_pairs = asts_per_kernel * num_targets;
    const size_t num_partials = num_pairs * SECANT_HIP_SSE_MAX_WAVES;
    size_t ast_idx;
    size_t target_idx;
    size_t input_idx;

    (void)threads_per_block;
    S_HIP_SSE_CHECK_RET(_secant_hip_write(
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
        "    const unsigned int lane = threadIdx.x %% warpSize;\n"
        "    const unsigned int wave = threadIdx.x / warpSize;\n"
        "    const unsigned int num_waves = blockDim.x / warpSize;\n"
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
            S_HIP_SSE_CHECK_RET(_secant_hip_write(
                buffer,
                buffer_size,
                offset,
                "    float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_HIP_SSE_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "    for (size_t tile_row = threadIdx.x; tile_row < tile_num_rows; tile_row += blockDim.x) {\n"
        "        const size_t row = tile_begin + tile_row;\n"));

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        S_HIP_SSE_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_HIP_SSE_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "        const float target%zu = targets[(size_t)%zu * targets_leading_dimension + row];\n",
            target_idx,
            target_idx));
    }
    S_HIP_SSE_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_HIP_SSE_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
        }
        S_HIP_SSE_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "        {\n            float prediction;\n"));
        S_HIP_SSE_CHECK_RET(_secant_hip_emit_expression(
            num_inputs,
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
            S_HIP_SSE_CHECK_RET(_secant_hip_write(
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
        S_HIP_SSE_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "        }\n\n"));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    S_HIP_SSE_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "    }\n\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_HIP_SSE_CHECK_RET(_secant_hip_write(
                buffer,
                buffer_size,
                offset,
                "    for (unsigned int delta = warpSize / 2u; delta != 0u; delta >>= 1u) {\n"
                "        sse_%03zu_%03zu += __shfl_down(sse_%03zu_%03zu, delta, warpSize);\n"
                "    }\n"
                "    if (lane == 0u) {\n"
                "        partial_sse[(size_t)%zu * %uu + wave] = sse_%03zu_%03zu;\n"
                "    }\n",
                ast_idx,
                target_idx,
                ast_idx,
                target_idx,
                pair_idx,
                SECANT_HIP_SSE_MAX_WAVES,
                ast_idx,
                target_idx));
        }
    }

    S_HIP_SSE_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "    __syncthreads();\n"
        "    if (wave == 0u) {\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            S_HIP_SSE_CHECK_RET(_secant_hip_write(
                buffer,
                buffer_size,
                offset,
                "        float block_sse_%03zu_%03zu = lane < num_waves ? partial_sse[(size_t)%zu * %uu + lane] : 0.0f;\n"
                "        for (unsigned int delta = warpSize / 2u; delta != 0u; delta >>= 1u) {\n"
                "            block_sse_%03zu_%03zu += __shfl_down(block_sse_%03zu_%03zu, delta, warpSize);\n"
                "        }\n"
                "        if (lane == 0u) {\n",
                ast_idx,
                target_idx,
                pair_idx,
                SECANT_HIP_SSE_MAX_WAVES,
                ast_idx,
                target_idx,
                ast_idx,
                target_idx));
            if (reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_ATOMIC) {
                S_HIP_SSE_CHECK_RET(_secant_hip_write(
                    buffer,
                    buffer_size,
                    offset,
                    "            atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + %zuu, block_sse_%03zu_%03zu);\n",
                    ast_idx,
                    target_idx,
                    ast_idx,
                    target_idx));
            } else {
                const size_t global_pair =
                    kernel_idx * num_pairs + pair_idx;

                S_HIP_SSE_CHECK_RET(_secant_hip_write(
                    buffer,
                    buffer_size,
                    offset,
                    "            const size_t num_tiles = (num_rows + %zuu - 1u) / %zuu;\n"
                    "            output_sse[(size_t)%zu * num_tiles + blockIdx.x] = block_sse_%03zu_%03zu;\n",
                    tile_rows,
                    tile_rows,
                    global_pair,
                    ast_idx,
                    target_idx));
            }
            S_HIP_SSE_CHECK_RET(_secant_hip_write(
                buffer,
                buffer_size,
                offset,
                "        }\n"));
        }
    }
    S_HIP_SSE_CHECK_RET(_secant_hip_write(buffer, buffer_size, offset, "    }\n}\n\n"));

    return SECANT_HIP_SUCCESS;
}

SecantHIPResult
secant_hip_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
) {
    size_t total_asts;
    size_t num_pairs;
    size_t num_partials;
    size_t kernel_idx;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u || num_targets == 0u ||
        tile_rows == 0u || threads_per_block < 64u ||
        threads_per_block > tile_rows ||
        threads_per_block > SECANT_HIP_SSE_MAX_WAVES * 32u ||
        threads_per_block % 64u != 0u ||
        (reduction_mode != SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         reduction_mode != SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
        num_inputs > UINT32_MAX || asts == NULL ||
        hip_size_ret == NULL || (num_routines != 0u && routines == NULL)) {
        S_HIP_SSE_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    *hip_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    if (!_secant_hip_sse_checked_mul(num_kernels, asts_per_kernel, &total_asts) ||
        !_secant_hip_sse_checked_mul(asts_per_kernel, num_targets, &num_pairs) ||
        !_secant_hip_sse_checked_mul(num_pairs, SECANT_HIP_SSE_MAX_WAVES, &num_partials)) {
        S_HIP_SSE_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }
    (void)total_asts;
    (void)num_partials;

    S_HIP_SSE_CHECK_RET(_secant_hip_emit_prelude(buffer, buffer_size, &offset));
    S_HIP_SSE_CHECK_RET(_secant_hip_emit_routines(routines, num_routines, routine_names, buffer, buffer_size, &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_HIP_SSE_CHECK_RET(_secant_hip_sse_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_inputs,
            num_targets,
            tile_rows,
            threads_per_block,
            reduction_mode,
            routines,
            num_routines,
            routine_names,
            asts + kernel_idx * asts_per_kernel,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }
    if (reduction_mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        S_HIP_SSE_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_static_column_sse_reduce(\n"
            "    const float* __restrict__ workspace,\n"
            "    size_t num_tiles,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    const size_t result = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (result < %zuu) {\n"
            "        float sum = 0.0f;\n"
            "        for (size_t tile = 0u; tile < num_tiles; ++tile) {\n"
            "            sum += workspace[result * num_tiles + tile];\n"
            "        }\n"
            "        const size_t ast = result / %zuu;\n"
            "        const size_t target = result %% %zuu;\n"
            "        output_sse[ast * output_leading_dimension + target] = sum;\n"
            "    }\n"
            "}\n\n",
            num_kernels * num_pairs,
            num_targets,
            num_targets));
    }
    S_HIP_SSE_CHECK_RET(_secant_hip_write_finish(buffer, buffer_size, offset, hip_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_HIP_SUCCESS;
}

#undef S_HIP_SSE_CHECK_RET
#undef S_HIP_SSE_ERROR_RET

#define S_HIP_DYNAMIC_ERROR_RET(ans) do { SecantHIPResult _secant_hip_dynamic_result = (ans); return _secant_hip_dynamic_result; } while (0)
#define S_HIP_DYNAMIC_CHECK_RET(ans) do { SecantHIPResult _secant_hip_dynamic_check_result = (ans); if (_secant_hip_dynamic_check_result != SECANT_HIP_SUCCESS) { S_HIP_DYNAMIC_ERROR_RET(_secant_hip_dynamic_check_result); } } while (0)

static int
_secant_hip_dynamic_checked_add(
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

static SecantHIPResult
_secant_hip_dynamic_emit_kernel(
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    SecantSSEReductionMode reduction_mode,
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
    const size_t row_tile_elements =
        (num_input_columns + num_targets) * tile_rows;
    size_t ast_idx;
    size_t constant_idx;
    size_t input_idx;
    size_t target_idx;

    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
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
        row_tile_elements,
        input_tile_elements,
        tile_rows,
        tile_rows,
        tile_rows));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "        input_tile[(size_t)%zu * %zuu + tile_row] = input[(size_t)%zu * input_leading_dimension + row];\n",
            input_idx,
            tile_rows,
            input_idx));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "        target_tile[(size_t)%zu * %zuu + tile_row] = targets[(size_t)%zu * targets_leading_dimension + row];\n",
            target_idx,
            tile_rows,
            target_idx));
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "    }\n\n"
        "    __syncthreads();\n\n"
        "    for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {\n"));
    for (constant_idx = 0u;
         constant_idx < num_input_constants;
         ++constant_idx) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "        const float input%zu = constant_settings[(size_t)%zu * constants_leading_dimension + setting];\n",
            num_input_columns + constant_idx,
            constant_idx));
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
                buffer,
                buffer_size,
                offset,
                "        float sse_%03zu_%03zu = 0.0f;\n",
                ast_idx,
                target_idx));
        }
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "\n"
        "        for (size_t eval_row = 0u; eval_row < tile_num_rows; ++eval_row) {\n"));
    for (input_idx = 0u; input_idx < num_input_columns; ++input_idx) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "            const float input%zu = input_tile[(size_t)%zu * %zuu + eval_row];\n",
            input_idx,
            input_idx,
            tile_rows));
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "            const float target%zu = target_tile[(size_t)%zu * %zuu + eval_row];\n",
            target_idx,
            target_idx,
            tile_rows));
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "\n"));
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        uint32_t num_variables = 0u;

        if (asts[ast_idx] == NULL) {
            S_HIP_DYNAMIC_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
        }
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "            {\n"
            "                float prediction;\n"));
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_emit_expression(
            num_input_columns + num_input_constants,
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
            S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
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
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "            }\n\n"));
        if (num_variables > *max_variables) {
            *max_variables = num_variables;
        }
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "        }\n\n"));
    if (reduction_mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            offset,
            "        const size_t workspace_num_tiles = "
                "(num_rows + %zuu - 1u) / %zuu;\n",
            tile_rows,
            tile_rows));
    }
    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
            const size_t pair_idx = ast_idx * num_targets + target_idx;

            if (reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_ATOMIC) {
                S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
                    buffer,
                    buffer_size,
                    offset,
                    "        atomicAdd(output_sse + (size_t)%zu * output_leading_dimension + setting, sse_%03zu_%03zu);\n",
                    pair_idx,
                    ast_idx,
                    target_idx));
            } else {
                const size_t global_pair =
                    kernel_idx * asts_per_kernel * num_targets +
                    pair_idx;

                S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
                    buffer,
                    buffer_size,
                    offset,
                    "        output_sse[((size_t)%zu * num_settings + "
                        "setting) * workspace_num_tiles + blockIdx.x] = "
                        "sse_%03zu_%03zu;\n",
                    global_pair,
                    ast_idx,
                    target_idx));
            }
        }
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
        buffer,
        buffer_size,
        offset,
        "    }\n"
        "}\n\n"));
    return SECANT_HIP_SUCCESS;
}

SecantHIPResult
secant_hip_dynamic_constant_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
) {
    size_t num_inputs;
    size_t num_sources;
    size_t row_tile_elements;
    size_t total_asts;
    size_t kernel_idx;
    size_t offset = 0u;
    uint32_t max_variables = 0u;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_input_columns == 0u ||
        num_input_columns >
            SECANT_HIP_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants >
            SECANT_HIP_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        num_targets == 0u || tile_rows == 0u ||
        threads_per_block == 0u || threads_per_block > 1024u ||
        (reduction_mode != SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         reduction_mode != SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
        asts == NULL || hip_size_ret == NULL ||
        (num_routines != 0u && routines == NULL) ||
        !_secant_hip_dynamic_checked_add(
            num_input_columns,
            num_input_constants,
            &num_inputs) ||
        !_secant_hip_dynamic_checked_add(
            num_input_columns,
            num_targets,
            &num_sources) ||
        !_secant_hip_sse_checked_mul(
            num_kernels,
            asts_per_kernel,
            &total_asts) ||
        !_secant_hip_sse_checked_mul(
            num_sources,
            tile_rows,
            &row_tile_elements)) {
        S_HIP_DYNAMIC_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    *hip_size_ret = 0u;
    if (max_variables_ret != NULL) {
        *max_variables_ret = 0u;
    }
    (void)num_inputs;
    (void)total_asts;
    (void)row_tile_elements;

    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_emit_prelude(
        buffer,
        buffer_size,
        &offset));
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_emit_routines(
        routines,
        num_routines,
        routine_names,
        buffer,
        buffer_size,
        &offset));
    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_dynamic_emit_kernel(
            kernel_idx,
            asts_per_kernel,
            num_input_columns,
            num_input_constants,
            num_targets,
            tile_rows,
            reduction_mode,
            routines,
            num_routines,
            routine_names,
            asts + kernel_idx * asts_per_kernel,
            buffer,
            buffer_size,
            &offset,
            &max_variables));
    }
    if (reduction_mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write(
            buffer,
            buffer_size,
            &offset,
            "extern \"C\" __global__\n"
            "void secant_dynamic_constant_sse_reduce(\n"
            "    const float* __restrict__ workspace,\n"
            "    size_t num_tiles,\n"
            "    size_t num_settings,\n"
            "    float* __restrict__ output_sse,\n"
            "    size_t output_leading_dimension\n"
            ") {\n"
            "    const size_t result = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    const size_t num_results = %zuu * num_settings;\n"
            "    if (result < num_results) {\n"
            "        float sum = 0.0f;\n"
            "        for (size_t tile = 0u; tile < num_tiles; ++tile) {\n"
            "            sum += workspace[result * num_tiles + tile];\n"
            "        }\n"
            "        const size_t pair = result / num_settings;\n"
            "        const size_t setting = result %% num_settings;\n"
            "        output_sse[pair * output_leading_dimension + setting] = sum;\n"
            "    }\n"
            "}\n\n",
            num_kernels * asts_per_kernel * num_targets));
    }
    S_HIP_DYNAMIC_CHECK_RET(_secant_hip_write_finish(
        buffer,
        buffer_size,
        offset,
        hip_size_ret));
    if (max_variables_ret != NULL) {
        *max_variables_ret = max_variables;
    }
    return SECANT_HIP_SUCCESS;
}

#undef S_HIP_DYNAMIC_CHECK_RET
#undef S_HIP_DYNAMIC_ERROR_RET

#undef _SECANT_HIP_ERROR_RET
