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
#ifndef CUSR_AST_SASS_CPU_H_INCLUDED
#define CUSR_AST_SASS_CPU_H_INCLUDED

#include "cusr_ast.h"
#include "cusr_settings.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef CUSR_AST_SASS_CPU_STACK_DEPTH
#define CUSR_AST_SASS_CPU_STACK_DEPTH 128u
#endif

#ifndef CUSR_AST_SASS_CPU_MAX_ROUTINE_ARGS
#define CUSR_AST_SASS_CPU_MAX_ROUTINE_ARGS 4u
#endif

#ifndef CUSR_AST_SASS_CPU_ROUTINE_DEPTH
#define CUSR_AST_SASS_CPU_ROUTINE_DEPTH 8u
#endif

typedef enum CusrAstSassCpuResult {
    CUSR_AST_SASS_CPU_SUCCESS = 0,
    CUSR_AST_SASS_CPU_ERROR_INVALID_VALUE = 1,
    CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM = 2
} CusrAstSassCpuResult;

#if defined(__GNUC__) || defined(__clang__)
#define CUSR_AST_SASS_CPU_UNUSED __attribute__((unused))
#else
#define CUSR_AST_SASS_CPU_UNUSED
#endif

static CusrAstSassCpuResult
cusr_ast_sass_cpu_eval_frame_f32(
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* instructions,
    const float* inputs,
    size_t num_inputs,
    const float* routine_args,
    size_t num_routine_args,
    size_t frame_depth,
    float* value_ret)
{
    float stack[CUSR_AST_SASS_CPU_STACK_DEPTH];
    size_t stack_size = 0u;
    size_t instruction_idx;

    if (instructions == NULL || value_ret == NULL) {
        return CUSR_AST_SASS_CPU_ERROR_INVALID_VALUE;
    }

    for (instruction_idx = 0u; instruction_idx < CUSR_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
        const CusrAstInstruction instruction = instructions[instruction_idx];

        switch ((CusrAstInstructionType)instruction.instruction_type) {
            case CUSR_AST_INSTRUCTION_TYPE_INPUT:
                if (inputs == NULL || instruction.payload.idx >= num_inputs || stack_size >= CUSR_AST_SASS_CPU_STACK_DEPTH) {
                    return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                }
                stack[stack_size++] = inputs[instruction.payload.idx];
                break;

            case CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS:
                if (stack_size >= CUSR_AST_SASS_CPU_STACK_DEPTH) {
                    return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                }
                memcpy(stack + stack_size, &instruction.payload.bits, sizeof(float));
                stack_size += 1u;
                break;

            case CUSR_AST_INSTRUCTION_TYPE_ROUTINE_ARG:
                if (routine_args == NULL || instruction.payload.idx >= num_routine_args || stack_size >= CUSR_AST_SASS_CPU_STACK_DEPTH) {
                    return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                }
                stack[stack_size++] = routine_args[instruction.payload.idx];
                break;

            case CUSR_AST_INSTRUCTION_TYPE_OP:
                switch ((CusrAstOp)instruction.payload.op) {
                    case CUSR_AST_OP_ADD:
                    case CUSR_AST_OP_SUB:
                    case CUSR_AST_OP_MUL:
                    case CUSR_AST_OP_DIV:
                    case CUSR_AST_OP_MIN:
                    case CUSR_AST_OP_MAX:
                        {
                            float lhs;
                            float rhs;

                            if (stack_size < 2u) {
                                return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                            }

                            rhs = stack[--stack_size];
                            lhs = stack[--stack_size];

                            if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_ADD) {
                                stack[stack_size++] = lhs + rhs;
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_SUB) {
                                stack[stack_size++] = lhs - rhs;
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_MUL) {
                                stack[stack_size++] = lhs * rhs;
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_DIV) {
                                stack[stack_size++] = lhs / rhs;
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_MIN) {
                                stack[stack_size++] = fminf(lhs, rhs);
                            } else {
                                stack[stack_size++] = fmaxf(lhs, rhs);
                            }
                        }
                        break;

                    case CUSR_AST_OP_FMA:
                        {
                            float a;
                            float b;
                            float c;

                            if (stack_size < 3u) {
                                return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                            }

                            c = stack[--stack_size];
                            b = stack[--stack_size];
                            a = stack[--stack_size];
                            stack[stack_size++] = fmaf(a, b, c);
                        }
                        break;

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
                        {
                            float value;

                            if (stack_size < 1u) {
                                return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                            }

                            value = stack[--stack_size];

                            if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_NEG) {
                                stack[stack_size++] = -value;
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_SQRT) {
                                stack[stack_size++] = sqrtf(value);
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_RCP) {
                                stack[stack_size++] = 1.0f / value;
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_ABS) {
                                stack[stack_size++] = fabsf(value);
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_SIN) {
                                stack[stack_size++] = sinf(value);
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_COS) {
                                stack[stack_size++] = cosf(value);
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_EX2) {
                                stack[stack_size++] = exp2f(value);
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_LG2) {
                                stack[stack_size++] = log2f(value);
                            } else if ((CusrAstOp)instruction.payload.op == CUSR_AST_OP_TANH) {
                                stack[stack_size++] = tanhf(value);
                            } else {
                                stack[stack_size++] = 1.0f / sqrtf(value);
                            }
                        }
                        break;

                    case CUSR_AST_OP_NONE:
                    case CUSR_AST_OP_NUM_ENUMS:
                    default:
                        return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                }
                break;

            case CUSR_AST_INSTRUCTION_TYPE_ROUTINE:
                {
                    float args[CUSR_AST_SASS_CPU_MAX_ROUTINE_ARGS];
                    float value = 0.0f;
                    CusrAstSassCpuResult result;
                    size_t arg_start;
                    size_t arg_idx;

                    if (instruction.aux > CUSR_AST_SASS_CPU_MAX_ROUTINE_ARGS ||
                        instruction.payload.idx >= num_routines ||
                        routines == NULL ||
                        routines[instruction.payload.idx] == NULL ||
                        frame_depth + 1u > CUSR_AST_SASS_CPU_ROUTINE_DEPTH ||
                        stack_size < instruction.aux) {
                        return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                    }

                    arg_start = stack_size - instruction.aux;
                    for (arg_idx = 0u; arg_idx < instruction.aux; ++arg_idx) {
                        args[arg_idx] = stack[arg_start + arg_idx];
                    }
                    stack_size = arg_start;

                    result = cusr_ast_sass_cpu_eval_frame_f32(
                        routines,
                        num_routines,
                        routines[instruction.payload.idx],
                        inputs,
                        num_inputs,
                        args,
                        instruction.aux,
                        frame_depth + 1u,
                        &value
                    );
                    if (result != CUSR_AST_SASS_CPU_SUCCESS || stack_size >= CUSR_AST_SASS_CPU_STACK_DEPTH) {
                        return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                    }

                    stack[stack_size++] = value;
                }
                break;

            case CUSR_AST_INSTRUCTION_TYPE_RETURN:
                if (stack_size != 1u) {
                    return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
                }
                *value_ret = stack[0];
                return CUSR_AST_SASS_CPU_SUCCESS;

            case CUSR_AST_INSTRUCTION_TYPE_NONE:
            default:
                return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
        }
    }

    return CUSR_AST_SASS_CPU_ERROR_BAD_PROGRAM;
}

static CUSR_AST_SASS_CPU_UNUSED CusrAstSassCpuResult
cusr_ast_sass_cpu_eval_program_f32(
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* program,
    const float* inputs,
    size_t num_inputs,
    float* value_ret)
{
    return cusr_ast_sass_cpu_eval_frame_f32(
        routines,
        num_routines,
        program,
        inputs,
        num_inputs,
        NULL,
        0u,
        0u,
        value_ret
    );
}

static CUSR_AST_SASS_CPU_UNUSED CusrAstSassCpuResult
cusr_ast_sass_cpu_eval_programs_f32(
    const float* x,
    uint32_t num_rows,
    uint32_t leading_dim,
    const CusrSettingF32* settings,
    uint32_t num_settings,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* programs,
    size_t num_programs,
    size_t program_stride,
    float* output)
{
    size_t program_idx;

    if (x == NULL ||
        settings == NULL ||
        programs == NULL ||
        output == NULL ||
        num_rows == 0u ||
        leading_dim < num_rows ||
        program_stride == 0u) {
        return CUSR_AST_SASS_CPU_ERROR_INVALID_VALUE;
    }

    for (program_idx = 0u; program_idx < num_programs; ++program_idx) {
        uint32_t setting_idx;
        const CusrAstInstruction* program = programs + program_idx * program_stride;

        for (setting_idx = 0u; setting_idx < num_settings; ++setting_idx) {
            uint32_t row;

            for (row = 0u; row < num_rows; ++row) {
                float inputs[CUSR_NUM_INPUTS];
                float value = 0.0f;
                uint32_t input_idx;
                CusrAstSassCpuResult result;

                for (input_idx = 0u; input_idx < CUSR_NUM_INPUTS; ++input_idx) {
                    inputs[input_idx] = cusr_setting_load_f32(x, row, leading_dim, settings + setting_idx, input_idx);
                }

                result = cusr_ast_sass_cpu_eval_program_f32(
                    routines,
                    num_routines,
                    program,
                    inputs,
                    CUSR_NUM_INPUTS,
                    &value
                );
                if (result != CUSR_AST_SASS_CPU_SUCCESS) {
                    return result;
                }

                output[((size_t)program_idx * (size_t)num_settings + (size_t)setting_idx) * (size_t)num_rows + (size_t)row] = value;
            }
        }
    }

    return CUSR_AST_SASS_CPU_SUCCESS;
}

static CUSR_AST_SASS_CPU_UNUSED CusrAstSassCpuResult
cusr_ast_sass_cpu_eval_program_ptrs_f32(
    const float* x,
    uint32_t num_rows,
    uint32_t leading_dim,
    const CusrSettingF32* settings,
    uint32_t num_settings,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const CusrAstInstruction* const* programs,
    size_t num_programs,
    float* output)
{
    size_t program_idx;

    if (x == NULL ||
        settings == NULL ||
        programs == NULL ||
        output == NULL ||
        num_rows == 0u ||
        leading_dim < num_rows) {
        return CUSR_AST_SASS_CPU_ERROR_INVALID_VALUE;
    }

    for (program_idx = 0u; program_idx < num_programs; ++program_idx) {
        uint32_t setting_idx;
        const CusrAstInstruction* program = programs[program_idx];

        if (program == NULL) {
            return CUSR_AST_SASS_CPU_ERROR_INVALID_VALUE;
        }

        for (setting_idx = 0u; setting_idx < num_settings; ++setting_idx) {
            uint32_t row;

            for (row = 0u; row < num_rows; ++row) {
                float inputs[CUSR_NUM_INPUTS];
                float value = 0.0f;
                uint32_t input_idx;
                CusrAstSassCpuResult result;

                for (input_idx = 0u; input_idx < CUSR_NUM_INPUTS; ++input_idx) {
                    inputs[input_idx] = cusr_setting_load_f32(x, row, leading_dim, settings + setting_idx, input_idx);
                }

                result = cusr_ast_sass_cpu_eval_program_f32(
                    routines,
                    num_routines,
                    program,
                    inputs,
                    CUSR_NUM_INPUTS,
                    &value
                );
                if (result != CUSR_AST_SASS_CPU_SUCCESS) {
                    return result;
                }

                output[((size_t)program_idx * (size_t)num_settings + (size_t)setting_idx) * (size_t)num_rows + (size_t)row] = value;
            }
        }
    }

    return CUSR_AST_SASS_CPU_SUCCESS;
}

static CUSR_AST_SASS_CPU_UNUSED CusrAstSassCpuResult
cusr_ast_sass_cpu_eval_ast_f32(
    const float* x,
    uint32_t num_rows,
    uint32_t leading_dim,
    const CusrSettingF32* settings,
    uint32_t num_settings,
    const CusrAstInstruction* ast,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    float* output)
{
    return cusr_ast_sass_cpu_eval_programs_f32(
        x,
        num_rows,
        leading_dim,
        settings,
        num_settings,
        routines,
        num_routines,
        ast,
        1u,
        CUSR_AST_MAX_PROGRAM_INSTRUCTIONS,
        output
    );
}

#undef CUSR_AST_SASS_CPU_UNUSED

#endif /* CUSR_AST_SASS_CPU_H_INCLUDED */
