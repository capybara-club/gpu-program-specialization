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
#include "secant.h"

#include "s_toggle_internal.h"
#include <math.h>
#include <stdint.h>

#ifndef SECANT_CPU_ROUTINE_DEPTH
#define SECANT_CPU_ROUTINE_DEPTH 8u
#endif

#define _SECANT_CPU_GRAM_STATS_MAX_FEATURES 32u

#define _SECANT_CPU_ERROR_RET(ans) do { SecantResult secant_cpu_result = (ans); return secant_cpu_result; } while (0)
#define _SECANT_CPU_CHECK_RET(ans) do { \
    SecantResult secant_cpu_check_result = (ans); \
    if (secant_cpu_check_result != SECANT_SUCCESS) { _SECANT_CPU_ERROR_RET(secant_cpu_check_result); } \
} while (0)

static int
_secant_cpu_checked_add(
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

static int
_secant_cpu_checked_mul(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
_secant_cpu_span_required(
    size_t outer_count,
    size_t leading_dimension,
    size_t inner_count,
    size_t* required_ret
) {
    size_t offset;

    return outer_count != 0u && inner_count != 0u &&
        _secant_cpu_checked_mul(
            outer_count - 1u,
            leading_dimension,
            &offset) &&
        _secant_cpu_checked_add(
            offset,
            inner_count,
            required_ret);
}

static SecantResult
_secant_cpu_ranges_validate(
    const void* const* addresses,
    const size_t* num_elements,
    size_t count
) {
    uintptr_t starts[8];
    uintptr_t ends[8];
    size_t idx;

    if (addresses == NULL || num_elements == NULL || count > 8u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    for (idx = 0u; idx < count; ++idx) {
        size_t num_bytes;

        if (addresses[idx] == NULL ||
            !_secant_cpu_checked_mul(num_elements[idx], sizeof(uint32_t), &num_bytes)) {
            return SECANT_ERROR_OVERFLOW;
        }
        starts[idx] = (uintptr_t)addresses[idx];
        if (num_bytes > UINTPTR_MAX - starts[idx]) {
            return SECANT_ERROR_OVERFLOW;
        }
        ends[idx] = starts[idx] + num_bytes;
    }
    for (idx = 0u; idx < count; ++idx) {
        size_t other_idx;

        for (other_idx = idx + 1u; other_idx < count; ++other_idx) {
            if (starts[idx] < ends[other_idx] && starts[other_idx] < ends[idx]) {
                return SECANT_ERROR_INVALID_VALUE;
            }
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_routine_num_args(const SecantAstInstruction* instructions, size_t* num_args_ret) {
    size_t instruction_count;
    size_t instruction_offset = 0u;
    size_t num_args = 0u;

    if (instructions == NULL || num_args_ret == NULL) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            _SECANT_CPU_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            const SecantAstIdx arg_idx = secant_ast_index_get(instruction);

            if (arg_idx >= SECANT_AST_MAX_INSTRUCTION_ARGS) {
                _SECANT_CPU_ERROR_RET(SECANT_ERROR_TOO_MANY_ARGS);
            }
            if ((size_t)arg_idx + 1u > num_args) {
                num_args = (size_t)arg_idx + 1u;
            }
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_SUCCESS;
        }
        instruction_offset += instruction_size;
    }
    _SECANT_CPU_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
}

static SecantResult
_secant_cpu_routine_num_args_build(
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    uint8_t* routine_num_args
) {
    size_t routine_idx;

    if (num_routines != 0u && (routines == NULL || routine_num_args == NULL)) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    for (routine_idx = 0u; routine_idx < num_routines; ++routine_idx) {
        size_t num_args;

        if (routines[routine_idx] == NULL) {
            _SECANT_CPU_ERROR_RET(
                SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
        }
        _SECANT_CPU_CHECK_RET(_secant_cpu_routine_num_args(
            routines[routine_idx],
            &num_args));
        routine_num_args[routine_idx] = (uint8_t)num_args;
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_eval_frame(
    size_t num_input_columns,
    size_t num_input_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const uint8_t* routine_num_args,
    const SecantAstInstruction* instructions,
    const float* input,
    size_t input_leading_dimension,
    const float* constants,
    uint32_t toggle_bits,
    uint32_t permutation,
    size_t row,
    const float* routine_args,
    size_t num_routine_args,
    size_t frame_depth,
    float* result_ret
) {
    float stack[SECANT_AST_MAX_STACK_DEPTH];
    size_t stack_size = 0u;
    size_t recent_direct = 0u;
    size_t instruction_count;
    size_t instruction_offset = 0u;

    if (instructions == NULL || result_ret == NULL ||
        num_input_constants > SIZE_MAX - num_input_columns ||
        toggle_bits > 32u ||
        (num_routines != 0u && routines == NULL)) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }

    for (instruction_count = 0u;
         instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_count) {
        const SecantAstInstruction* instruction = instructions + instruction_offset;
        const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);
        const size_t instruction_size = secant_ast_instruction_size_get(instruction);

        if (instruction_size == 0u) {
            _SECANT_CPU_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        if (!secant_internal_ast_instruction_is_f32(instruction_type)) {
            _SECANT_CPU_ERROR_RET(SECANT_ERROR_UNSUPPORTED_OP);
        }
        switch (instruction_type) {
            case SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32:
                {
                    const size_t input_idx = secant_ast_index_get(instruction);

                    if (input_idx >= num_input_columns || input == NULL) {
                        _SECANT_CPU_ERROR_RET(
                            SECANT_ERROR_INVALID_VALUE);
                    }
                    if (stack_size == SECANT_AST_MAX_STACK_DEPTH) {
                        _SECANT_CPU_ERROR_RET(
                            SECANT_ERROR_STACK_OVERFLOW);
                    }
                    stack[stack_size++] = input[
                        input_idx * input_leading_dimension + row];
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32:
            case SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32: {
                size_t index = secant_ast_index_get(instruction);
                if (index >= num_input_constants || constants == NULL) return SECANT_ERROR_BAD_PROGRAM;
                if (stack_size == SECANT_AST_MAX_STACK_DEPTH) return SECANT_ERROR_STACK_OVERFLOW;
                if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32) {
                    const float scale = secant_ast_affine_bank_scale_get(instruction);
                    const float offset = secant_ast_affine_bank_offset_get(instruction);
                    volatile float product;
                    if (!isfinite(scale) || !isfinite(offset)) return SECANT_ERROR_BAD_PROGRAM;
                    product = constants[index] * scale;
                    stack[stack_size++] = product + offset;
                } else stack[stack_size++] = constants[index];
                break;
            }
            case SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32:
            case SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32: {
                size_t count = instruction_type == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ? 2u : 4u;
                unsigned bit0 = instruction[1];
                unsigned choice;
                if (bit0 >= toggle_bits || recent_direct < count || stack_size < count)
                    return SECANT_ERROR_BAD_PROGRAM;
                choice = (permutation >> bit0) & 1u;
                if (count == 4u) {
                    unsigned bit1 = instruction[2];
                    if (bit1 >= toggle_bits || bit1 == bit0) return SECANT_ERROR_BAD_PROGRAM;
                    choice |= ((permutation >> bit1) & 1u) << 1;
                }
                stack[stack_size - count] = stack[stack_size - count + choice];
                stack_size -= count - 1u;
                break;
            }
            case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
                if (stack_size == SECANT_AST_MAX_STACK_DEPTH) {
                    _SECANT_CPU_ERROR_RET(SECANT_ERROR_STACK_OVERFLOW);
                }
                stack[stack_size++] = secant_ast_constant_f32_get(instruction);
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
                if (routine_args == NULL || secant_ast_index_get(instruction) >= num_routine_args) {
                    _SECANT_CPU_ERROR_RET(SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                }
                if (stack_size == SECANT_AST_MAX_STACK_DEPTH) {
                    _SECANT_CPU_ERROR_RET(SECANT_ERROR_STACK_OVERFLOW);
                }
                stack[stack_size++] = routine_args[secant_ast_index_get(instruction)];
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
                {
                    const uint8_t num_args = secant_internal_ast_instruction_num_args(instruction_type);
                    const size_t arg_start = stack_size >= num_args ? stack_size - num_args : 0u;
                    float value;

                    if (stack_size < num_args) {
                        _SECANT_CPU_ERROR_RET(SECANT_ERROR_STACK_UNDERFLOW);
                    }
                    switch (instruction_type) {
                        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32: value = stack[arg_start] + stack[arg_start + 1u]; break;
                        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32: value = stack[arg_start] - stack[arg_start + 1u]; break;
                        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32: value = stack[arg_start] * stack[arg_start + 1u]; break;
                        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32: value = stack[arg_start] / stack[arg_start + 1u]; break;
                        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32: value = -stack[arg_start]; break;
                        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: value = sqrtf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: value = 1.0f / stack[arg_start]; break;
                        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32: value = fabsf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32: value = fminf(stack[arg_start], stack[arg_start + 1u]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32: value = fmaxf(stack[arg_start], stack[arg_start + 1u]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32:
                            value = fmaf(stack[arg_start], stack[arg_start + 1u], stack[arg_start + 2u]);
                            break;
                        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: value = sinf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_COS_F32: value = cosf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: value = exp2f(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: value = log2f(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: value = 1.0f / sqrtf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: value = tanhf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_EXP_F32: value = expf(stack[arg_start]); break;
                        case SECANT_AST_INSTRUCTION_TYPE_LOG_F32: value = logf(stack[arg_start]); break;
                        default: _SECANT_CPU_ERROR_RET(SECANT_ERROR_UNSUPPORTED_OP);
                    }
                    stack_size = arg_start;
                    stack[stack_size++] = value;
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
                {
                    const size_t routine_idx = secant_ast_index_get(instruction);
                    size_t arg_start;
                    size_t expected_args;
                    float value;

                    if (routine_idx >= num_routines || routines[routine_idx] == NULL) {
                        _SECANT_CPU_ERROR_RET(SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                    }
                    if (frame_depth >= SECANT_CPU_ROUTINE_DEPTH) {
                        _SECANT_CPU_ERROR_RET(SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED);
                    }
                    expected_args = routine_num_args[routine_idx];
                    if (stack_size < expected_args) {
                        _SECANT_CPU_ERROR_RET(SECANT_ERROR_STACK_UNDERFLOW);
                    }
                    arg_start = stack_size - expected_args;
                    _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(
                        num_input_columns,
                        num_input_constants,
                        routines,
                        num_routines,
                        routine_num_args,
                        routines[routine_idx],
                        input,
                        input_leading_dimension,
                        constants,
                        toggle_bits,
                        permutation,
                        row,
                        stack + arg_start,
                        expected_args,
                        frame_depth + 1u,
                        &value
                    ));
                    stack_size = arg_start;
                    stack[stack_size++] = value;
                }
                break;
            case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32:
                if (stack_size != 1u) {
                    _SECANT_CPU_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
                }
                *result_ret = stack[0];
                return SECANT_SUCCESS;
            default:
                _SECANT_CPU_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) ++recent_direct;
        else recent_direct = 0u;
        instruction_offset += instruction_size;
    }
    _SECANT_CPU_ERROR_RET(SECANT_ERROR_BAD_PROGRAM);
}

static SecantResult
_secant_cpu_validate_materialize_buffers(
    size_t num_inputs,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_rows,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    const void* range_addresses[2];
    size_t range_elements[2];
    size_t required_input_elements;
    size_t required_output_elements;

    if (num_inputs == 0u || num_inputs > SECANT_AST_MAX_INPUTS ||
        num_asts == 0u || input == NULL || output == NULL || num_rows == 0u ||
        input_leading_dimension < num_rows || output_leading_dimension < num_rows) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (num_inputs - 1u > (SIZE_MAX - num_rows) / input_leading_dimension ||
        num_asts - 1u > (SIZE_MAX - num_rows) / output_leading_dimension) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    required_input_elements = (num_inputs - 1u) * input_leading_dimension + num_rows;
    required_output_elements = (num_asts - 1u) * output_leading_dimension + num_rows;
    if (input_num_elements < required_input_elements || output_num_elements < required_output_elements) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    range_addresses[0] = input;
    range_addresses[1] = output;
    range_elements[0] = required_input_elements;
    range_elements[1] = required_output_elements;
    _SECANT_CPU_CHECK_RET(_secant_cpu_ranges_validate(range_addresses, range_elements, 2u));
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_materialize_run(
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_rows,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    uint8_t routine_num_args[SECANT_AST_MAX_ROUTINES];
    size_t ast_idx;

    if (asts == NULL || num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL)) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CPU_CHECK_RET(_secant_cpu_validate_materialize_buffers(
        num_inputs,
        num_asts,
        input,
        input_num_elements,
        input_leading_dimension,
        num_rows,
        output,
        output_num_elements,
        output_leading_dimension));
    _SECANT_CPU_CHECK_RET(_secant_cpu_routine_num_args_build(
        routines,
        num_routines,
        routine_num_args));

    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        size_t row;

        if (asts[ast_idx] == NULL) {
            _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
        }
        for (row = 0u; row < num_rows; ++row) {
            float value;

            _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(
                num_inputs,
                0u,
                routines,
                num_routines,
                routine_num_args,
                asts[ast_idx],
                input,
                input_leading_dimension,
                NULL,
                0u,
                0u,
                row,
                NULL,
                0u,
                0u,
                &value
            ));
            output[ast_idx * output_leading_dimension + row] = value;
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_sse_run(
    size_t num_inputs,
    size_t num_targets,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    float* output_sse,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    const void* range_addresses[3];
    size_t range_elements[3];
    uint8_t routine_num_args[SECANT_AST_MAX_ROUTINES];
    size_t required_input_elements;
    size_t required_target_elements;
    size_t required_output_elements;
    size_t row;

    if (num_inputs == 0u || num_inputs > SECANT_AST_MAX_INPUTS ||
        num_targets == 0u || num_asts == 0u || asts == NULL ||
        num_routines > SECANT_AST_MAX_ROUTINES ||
        (num_routines != 0u && routines == NULL) || input == NULL || targets == NULL ||
        output_sse == NULL || num_rows == 0u || input_leading_dimension < num_rows ||
        targets_leading_dimension < num_rows || output_leading_dimension < num_targets) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (num_inputs - 1u > (SIZE_MAX - num_rows) / input_leading_dimension ||
        num_targets - 1u > (SIZE_MAX - num_rows) / targets_leading_dimension ||
        num_asts - 1u > (SIZE_MAX - num_targets) / output_leading_dimension) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    required_input_elements = (num_inputs - 1u) * input_leading_dimension + num_rows;
    required_target_elements = (num_targets - 1u) * targets_leading_dimension + num_rows;
    required_output_elements = (num_asts - 1u) * output_leading_dimension + num_targets;
    if (input_num_elements < required_input_elements || targets_num_elements < required_target_elements ||
        output_num_elements < required_output_elements) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    range_addresses[0] = input;
    range_addresses[1] = targets;
    range_addresses[2] = output_sse;
    range_elements[0] = required_input_elements;
    range_elements[1] = required_target_elements;
    range_elements[2] = required_output_elements;
    _SECANT_CPU_CHECK_RET(_secant_cpu_ranges_validate(range_addresses, range_elements, 3u));
    _SECANT_CPU_CHECK_RET(_secant_cpu_routine_num_args_build(
        routines,
        num_routines,
        routine_num_args));

    for (row = 0u; row < num_rows; ++row) {
        size_t ast_idx;

        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            float value;
            size_t target_idx;

            if (asts[ast_idx] == NULL) {
                _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
            }
            _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(
                num_inputs,
                0u,
                routines,
                num_routines,
                routine_num_args,
                asts[ast_idx],
                input,
                input_leading_dimension,
                NULL,
                0u,
                0u,
                row,
                NULL,
                0u,
                0u,
                &value
            ));
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                const float error = value - targets[target_idx * targets_leading_dimension + row];

                output_sse[ast_idx * output_leading_dimension + target_idx] += error * error;
            }
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_affine_stats_run(
    size_t num_inputs,
    size_t num_targets,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    float* ast_stats,
    size_t ast_stats_num_elements,
    size_t ast_stats_leading_dimension
) {
    const void* range_addresses[3];
    size_t range_elements[3];
    uint8_t routine_num_args[SECANT_AST_MAX_ROUTINES];
    size_t ast_stats_per_ast;
    size_t required_input_elements;
    size_t required_target_elements;
    size_t required_ast_stats_elements;
    const size_t range_count = 3u;
    size_t row;

    if (num_inputs == 0u || num_inputs > SECANT_AST_MAX_INPUTS || num_targets == 0u || num_asts == 0u ||
        asts == NULL || num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL) ||
        input == NULL || targets == NULL || ast_stats == NULL || num_rows == 0u ||
        input_leading_dimension < num_rows || targets_leading_dimension < num_rows ||
        !_secant_cpu_checked_add(
            SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32,
            num_targets,
            &ast_stats_per_ast) ||
        ast_stats_leading_dimension < ast_stats_per_ast) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_cpu_span_required(num_inputs, input_leading_dimension, num_rows, &required_input_elements) ||
        !_secant_cpu_span_required(num_targets, targets_leading_dimension, num_rows, &required_target_elements) ||
        !_secant_cpu_span_required(
            num_asts,
            ast_stats_leading_dimension,
            ast_stats_per_ast,
            &required_ast_stats_elements)) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    if (input_num_elements < required_input_elements || targets_num_elements < required_target_elements ||
        ast_stats_num_elements < required_ast_stats_elements) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    range_addresses[0] = input;
    range_addresses[1] = targets;
    range_addresses[2] = ast_stats;
    range_elements[0] = required_input_elements;
    range_elements[1] = required_target_elements;
    range_elements[2] = required_ast_stats_elements;
    _SECANT_CPU_CHECK_RET(_secant_cpu_ranges_validate(range_addresses, range_elements, range_count));
    _SECANT_CPU_CHECK_RET(_secant_cpu_routine_num_args_build(routines, num_routines, routine_num_args));

    for (row = 0u; row < num_rows; ++row) {
        size_t target_idx;
        size_t ast_idx;

        for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
            float prediction;
            float* stats = ast_stats + ast_idx * ast_stats_leading_dimension;

            if (asts[ast_idx] == NULL) {
                _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
            }
            _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(
                num_inputs,
                0u,
                routines,
                num_routines,
                routine_num_args,
                asts[ast_idx],
                input,
                input_leading_dimension,
                NULL,
                0u,
                0u,
                row,
                NULL,
                0u,
                0u,
                &prediction
            ));
            stats[SECANT_AFFINE_AST_STAT_SUM_PREDICTION_F32] += prediction;
            stats[SECANT_AFFINE_AST_STAT_SUM_PREDICTION_SQUARED_F32] += prediction * prediction;
            for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                const float target = targets[target_idx * targets_leading_dimension + row];

                stats[SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 + target_idx] += prediction * target;
            }
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_gram_stats_run(
    size_t asts_per_cohort,
    size_t num_inputs,
    size_t num_targets,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    float* statistics,
    size_t statistics_num_elements,
    size_t statistics_leading_dimension
) {
    const void* range_addresses[3];
    size_t range_elements[3];
    uint8_t routine_num_args[SECANT_AST_MAX_ROUTINES];
    float features[_SECANT_CPU_GRAM_STATS_MAX_FEATURES];
    size_t gram_count;
    size_t cross_count;
    size_t statistics_count;
    size_t num_cohorts;
    size_t required_input_elements;
    size_t required_target_elements;
    size_t required_statistics_elements;
    size_t cohort_idx;

    if (asts_per_cohort == 0u || asts_per_cohort > _SECANT_CPU_GRAM_STATS_MAX_FEATURES ||
        num_inputs == 0u || num_inputs > SECANT_AST_MAX_INPUTS || num_targets == 0u || num_asts == 0u ||
        asts == NULL || num_routines > SECANT_AST_MAX_ROUTINES || (num_routines != 0u && routines == NULL) ||
        input == NULL || targets == NULL || statistics == NULL || num_rows == 0u ||
        input_leading_dimension < num_rows || targets_leading_dimension < num_rows ||
        !_secant_cpu_checked_mul(asts_per_cohort, asts_per_cohort, &gram_count) ||
        !_secant_cpu_checked_mul(asts_per_cohort, num_targets, &cross_count) ||
        !_secant_cpu_checked_add(asts_per_cohort, gram_count, &statistics_count) ||
        !_secant_cpu_checked_add(statistics_count, cross_count, &statistics_count) ||
        statistics_leading_dimension < statistics_count ||
        num_asts > SIZE_MAX - (asts_per_cohort - 1u)) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    num_cohorts = (num_asts + asts_per_cohort - 1u) / asts_per_cohort;
    if (!_secant_cpu_span_required(num_inputs, input_leading_dimension, num_rows, &required_input_elements) ||
        !_secant_cpu_span_required(num_targets, targets_leading_dimension, num_rows, &required_target_elements) ||
        !_secant_cpu_span_required(
            num_cohorts,
            statistics_leading_dimension,
            statistics_count,
            &required_statistics_elements)) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    if (input_num_elements < required_input_elements || targets_num_elements < required_target_elements ||
        statistics_num_elements < required_statistics_elements) {
        _SECANT_CPU_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    range_addresses[0] = input;
    range_addresses[1] = targets;
    range_addresses[2] = statistics;
    range_elements[0] = required_input_elements;
    range_elements[1] = required_target_elements;
    range_elements[2] = required_statistics_elements;
    _SECANT_CPU_CHECK_RET(_secant_cpu_ranges_validate(range_addresses, range_elements, 3u));
    _SECANT_CPU_CHECK_RET(_secant_cpu_routine_num_args_build(routines, num_routines, routine_num_args));

    for (cohort_idx = 0u; cohort_idx < num_cohorts; ++cohort_idx) {
        const size_t first_ast = cohort_idx * asts_per_cohort;
        size_t cohort_asts = num_asts - first_ast;
        float* cohort_statistics = statistics + cohort_idx * statistics_leading_dimension;
        size_t row;

        if (cohort_asts > asts_per_cohort) {
            cohort_asts = asts_per_cohort;
        }
        for (row = 0u; row < num_rows; ++row) {
            size_t feature_idx;

            for (feature_idx = 0u; feature_idx < cohort_asts; ++feature_idx) {
                const SecantAstInstruction* ast = asts[first_ast + feature_idx];

                if (ast == NULL) {
                    _SECANT_CPU_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
                }
                _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(
                    num_inputs,
                    0u,
                    routines,
                    num_routines,
                    routine_num_args,
                    ast,
                    input,
                    input_leading_dimension,
                    NULL,
                    0u,
                    0u,
                    row,
                    NULL,
                    0u,
                    0u,
                    features + feature_idx
                ));
                cohort_statistics[feature_idx] += features[feature_idx];
            }
            for (feature_idx = 0u; feature_idx < cohort_asts; ++feature_idx) {
                size_t other_idx;
                size_t target_idx;

                for (other_idx = 0u; other_idx < cohort_asts; ++other_idx) {
                    cohort_statistics[
                        asts_per_cohort + feature_idx * asts_per_cohort + other_idx] +=
                        features[feature_idx] * features[other_idx];
                }
                for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
                    cohort_statistics[
                        asts_per_cohort + gram_count + feature_idx * num_targets + target_idx] +=
                        features[feature_idx] * targets[target_idx * targets_leading_dimension + row];
                }
            }
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_toggle_sse_run(const SecantCpuToggleSSERun *r) {
    size_t configs, bank_span, input_span, target_span, output_span, pairs;
    size_t a, b, p, row, t, range_count = 0;
    const void *addresses[4];
    size_t elements[4];
    uint8_t arities[SECANT_AST_MAX_ROUTINES];
    SecantResult result;
    if (!r || !r->programs.asts.items || !r->num_rows || !r->num_targets ||
        r->num_inputs > SECANT_AST_MAX_INPUTS ||
        r->programs.routines.count > SECANT_AST_MAX_ROUTINES)
        return SECANT_ERROR_INVALID_VALUE;
    result = s_toggle_layout(r->programs.asts.count, r->num_banks, r->toggle_bits,
        r->num_constants, r->constants.bank_stride, &configs, &bank_span);
    if (result != SECANT_SUCCESS) return result;
    if ((r->num_inputs && r->input.leading_dimension < r->num_rows) || r->targets.leading_dimension < r->num_rows ||
        r->output.leading_dimension < configs) return SECANT_ERROR_INVALID_VALUE;
    if (!s_span_size(r->num_inputs, r->input.leading_dimension, r->num_rows, &input_span) ||
        !s_span_size(r->num_targets, r->targets.leading_dimension, r->num_rows, &target_span) ||
        !s_size_mul(r->programs.asts.count, r->num_targets, &pairs) ||
        !s_span_size(pairs, r->output.leading_dimension, configs, &output_span))
        return SECANT_ERROR_OVERFLOW;
    if (r->input.num_elements < input_span || r->targets.num_elements < target_span ||
        r->constants.num_elements < bank_span || r->output.num_elements < output_span)
        return SECANT_ERROR_INSUFFICIENT_BUFFER;
#define ADD_RANGE(ptr, span) do { if (span) { if (!(ptr)) return SECANT_ERROR_INVALID_VALUE; \
    addresses[range_count] = (ptr); elements[range_count++] = (span); } } while (0)
    ADD_RANGE(r->input.data, input_span);
    ADD_RANGE(r->constants.data, bank_span);
    ADD_RANGE(r->targets.data, target_span);
    ADD_RANGE(r->output.data, output_span);
#undef ADD_RANGE
    _SECANT_CPU_CHECK_RET(_secant_cpu_ranges_validate(addresses, elements, range_count));
    _SECANT_CPU_CHECK_RET(_secant_cpu_routine_num_args_build(r->programs.routines.items,
        r->programs.routines.count, arities));
    /* Validate every program before clearing any output. */
    for (a = 0; a < r->programs.asts.count; ++a) {
        float unused;
        const float *c = r->num_constants ? r->constants.data : NULL;
        _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(r->num_inputs, r->num_constants,
            r->programs.routines.items, r->programs.routines.count, arities, r->programs.asts.items[a],
            r->input.data, r->input.leading_dimension, c, r->toggle_bits, 0, 0, NULL, 0, 0, &unused));
    }
    for (a = 0; a < r->programs.asts.count; ++a) {
        for (t = 0; t < r->num_targets; ++t)
            memset(r->output.data + (a * r->num_targets + t) * r->output.leading_dimension,
                0, configs * sizeof(float));
        for (b = 0; b < r->num_banks; ++b) {
            const float *c = r->num_constants ? r->constants.data + b * r->constants.bank_stride : NULL;
            for (p = 0; p < (size_t)(UINT64_C(1) << r->toggle_bits); ++p) {
                size_t configuration = (b << r->toggle_bits) + p;
                for (row = 0; row < r->num_rows; ++row) {
                    float value;
                    _SECANT_CPU_CHECK_RET(_secant_cpu_eval_frame(r->num_inputs, r->num_constants,
                        r->programs.routines.items, r->programs.routines.count, arities, r->programs.asts.items[a],
                        r->input.data, r->input.leading_dimension, c, r->toggle_bits, (uint32_t)p, row,
                        NULL, 0, 0, &value));
                    for (t = 0; t < r->num_targets; ++t) {
                        float error = value - r->targets.data[t * r->targets.leading_dimension + row];
                        r->output.data[(a * r->num_targets + t) * r->output.leading_dimension + configuration] += error * error;
                    }
                }
            }
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cpu_run_header_validate(
    const SecantCpuRunHeader* run,
    size_t required_size
) {
    if (run == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if ((size_t)run->struct_size < sizeof(*run)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if (run->version != SECANT_CPU_RUN_VERSION_3) {
        return SECANT_ERROR_UNSUPPORTED_VERSION;
    }
    if ((size_t)run->struct_size < required_size || run->flags != 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return SECANT_SUCCESS;
}

SecantResult
secant_cpu_run(
    const SecantCpuRunHeader* run
) {
    SecantResult result;

    if (run == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    result = _secant_cpu_run_header_validate(run, sizeof(*run));
    if (result != SECANT_SUCCESS) {
        return result;
    }
    switch (run->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32: {
            const SecantCpuMaterializeRun* materialize =
                (const SecantCpuMaterializeRun*)(const void*)run;

            result = _secant_cpu_run_header_validate(run, sizeof(*materialize));
            if (result != SECANT_SUCCESS) {
                return result;
            }
            return _secant_cpu_materialize_run(
                materialize->num_inputs,
                materialize->programs.routines.items,
                materialize->programs.routines.count,
                materialize->programs.asts.items,
                materialize->programs.asts.count,
                materialize->input.data,
                materialize->input.num_elements,
                materialize->input.leading_dimension,
                materialize->num_rows,
                materialize->output.data,
                materialize->output.num_elements,
                materialize->output.leading_dimension);
        }
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32: {
            const SecantCpuSSERun* sse = (const SecantCpuSSERun*)(const void*)run;

            result = _secant_cpu_run_header_validate(run, sizeof(*sse));
            if (result != SECANT_SUCCESS) {
                return result;
            }
            return _secant_cpu_sse_run(
                sse->num_inputs,
                sse->num_targets,
                sse->programs.routines.items,
                sse->programs.routines.count,
                sse->programs.asts.items,
                sse->programs.asts.count,
                sse->input.data,
                sse->input.num_elements,
                sse->input.leading_dimension,
                sse->targets.data,
                sse->targets.num_elements,
                sse->targets.leading_dimension,
                sse->num_rows,
                sse->output.data,
                sse->output.num_elements,
                sse->output.leading_dimension);
        }
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32: {
            const SecantCpuAffineStatsRun* affine = (const SecantCpuAffineStatsRun*)(const void*)run;

            result = _secant_cpu_run_header_validate(run, sizeof(*affine));
            if (result != SECANT_SUCCESS) {
                return result;
            }
            return _secant_cpu_affine_stats_run(
                affine->num_inputs,
                affine->num_targets,
                affine->programs.routines.items,
                affine->programs.routines.count,
                affine->programs.asts.items,
                affine->programs.asts.count,
                affine->input.data,
                affine->input.num_elements,
                affine->input.leading_dimension,
                affine->targets.data,
                affine->targets.num_elements,
                affine->targets.leading_dimension,
                affine->num_rows,
                affine->ast_stats.data,
                affine->ast_stats.num_elements,
                affine->ast_stats.leading_dimension);
        }
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32: {
            const SecantCpuGramStatsRun* gram = (const SecantCpuGramStatsRun*)(const void*)run;

            result = _secant_cpu_run_header_validate(run, sizeof(*gram));
            if (result != SECANT_SUCCESS) {
                return result;
            }
            return _secant_cpu_gram_stats_run(
                gram->asts_per_cohort,
                gram->num_inputs,
                gram->num_targets,
                gram->programs.routines.items,
                gram->programs.routines.count,
                gram->programs.asts.items,
                gram->programs.asts.count,
                gram->input.data,
                gram->input.num_elements,
                gram->input.leading_dimension,
                gram->targets.data,
                gram->targets.num_elements,
                gram->targets.leading_dimension,
                gram->num_rows,
                gram->statistics.data,
                gram->statistics.num_elements,
                gram->statistics.leading_dimension);
        }
        case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32:
            result = _secant_cpu_run_header_validate(run, sizeof(SecantCpuToggleSSERun));
            return result == SECANT_SUCCESS ? _secant_cpu_toggle_sse_run((const SecantCpuToggleSSERun*)run) : result;
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
}

#undef _SECANT_CPU_CHECK_RET
#undef _SECANT_CPU_ERROR_RET
#undef _SECANT_CPU_GRAM_STATS_MAX_FEATURES
