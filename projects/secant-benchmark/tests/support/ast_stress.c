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
#include "ast_stress.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define SECANT_TEST_AST_STRESS_ROUTINE_EPSILON 1.0f
#define SECANT_TEST_AST_STRESS_ROUTINE_OFFSET 1.0f

typedef enum SecantTestAstStressRoutine {
    SECANT_TEST_AST_STRESS_ROUTINE_SAFE_DIV = 0,
    SECANT_TEST_AST_STRESS_ROUTINE_SAFE_SQRT = 1,
    SECANT_TEST_AST_STRESS_ROUTINE_SAFE_RSQRT = 2,
    SECANT_TEST_AST_STRESS_ROUTINE_SAFE_LG2 = 3,
    SECANT_TEST_AST_STRESS_ROUTINE_BOUNDED_EX2 = 4,
    SECANT_TEST_AST_STRESS_ROUTINE_SAFE_RCP = 5
} SecantTestAstStressRoutine;

static const SecantAstInstruction
secant_test_ast_stress_routine_safe_div[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32(SECANT_TEST_AST_STRESS_ROUTINE_EPSILON),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction
secant_test_ast_stress_routine_safe_sqrt[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32(SECANT_TEST_AST_STRESS_ROUTINE_EPSILON),
    secant_ast_encode_add_f32,
    secant_ast_encode_sqrt_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction
secant_test_ast_stress_routine_safe_rsqrt[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32(SECANT_TEST_AST_STRESS_ROUTINE_OFFSET),
    secant_ast_encode_add_f32,
    secant_ast_encode_rsqrt_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction
secant_test_ast_stress_routine_safe_lg2[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_abs_f32,
    secant_ast_encode_constant_f32(SECANT_TEST_AST_STRESS_ROUTINE_OFFSET),
    secant_ast_encode_add_f32,
    secant_ast_encode_lg2_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction
secant_test_ast_stress_routine_bounded_ex2[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_constant_f32(0.25f),
    secant_ast_encode_mul_f32,
    secant_ast_encode_ex2_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction
secant_test_ast_stress_routine_safe_rcp[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32(SECANT_TEST_AST_STRESS_ROUTINE_EPSILON),
    secant_ast_encode_add_f32,
    secant_ast_encode_div_f32,
    secant_ast_encode_return_f32
};

const SecantAstInstruction* const
secant_test_ast_stress_routines[
    SECANT_TEST_AST_STRESS_ROUTINE_COUNT] = {
        secant_test_ast_stress_routine_safe_div,
        secant_test_ast_stress_routine_safe_sqrt,
        secant_test_ast_stress_routine_safe_rsqrt,
        secant_test_ast_stress_routine_safe_lg2,
        secant_test_ast_stress_routine_bounded_ex2,
        secant_test_ast_stress_routine_safe_rcp
    };

static uint64_t
secant_test_ast_stress_random_u64(uint64_t* state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));

    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint32_t
secant_test_ast_stress_random_u32(uint64_t* state) {
    return (uint32_t)(secant_test_ast_stress_random_u64(state) >> 32);
}

static const char*
secant_test_ast_stress_instruction_name(
    SecantAstInstructionType instruction_type
) {
    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_INPUT_F32: return "input";
        case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32: return "constant";
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32: return "routine";
        case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32: return "return";
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32: return "routine_arg";
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32: return "add";
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32: return "sub";
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32: return "mul";
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32: return "div";
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32: return "neg";
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: return "sqrt";
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: return "rcp";
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32: return "abs";
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32: return "min";
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32: return "max";
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32: return "fma";
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: return "sin";
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32: return "cos";
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: return "ex2";
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: return "lg2";
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: return "rsqrt";
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: return "tanh";
        default: return "invalid";
    }
}

static void
secant_test_ast_stress_program_dump(
    const SecantAstInstruction* program,
    size_t program_size
) {
    size_t instruction_idx;

    for (instruction_idx = 0u;
         instruction_idx < program_size;
         ++instruction_idx) {
        const SecantAstInstruction instruction = program[instruction_idx];
        const SecantAstInstructionType instruction_type =
            (SecantAstInstructionType)instruction.instruction_type;

        fprintf(
            stderr,
            "  %03zu %-12s",
            instruction_idx,
            secant_test_ast_stress_instruction_name(instruction_type));
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_INPUT_F32 ||
            instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            fprintf(stderr, " %u", instruction.payload.idx);
        } else if (instruction_type ==
            SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            fprintf(
                stderr,
                " %.9g (0x%08x)",
                (double)secant_ast_constant_f32_get(&instruction),
                secant_ast_constant_f32_bits_get(&instruction));
        } else if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
            fprintf(
                stderr,
                " idx=%u args=%u",
                instruction.payload.idx,
                instruction.aux);
        }
        fputc('\n', stderr);
    }
}

static int
secant_test_ast_stress_emit_instruction(
    SecantAstInstruction* program,
    size_t* program_size,
    SecantAstInstruction instruction,
    uint32_t* instruction_mask
) {
    if (*program_size >= SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY) {
        return 0;
    }
    program[(*program_size)++] = instruction;
    if (instruction.instruction_type < 32u) {
        *instruction_mask |= UINT32_C(1) << instruction.instruction_type;
    }
    return 1;
}

static int
secant_test_ast_stress_emit_unary(
    SecantAstInstruction* program,
    size_t* program_size,
    int allow_mufu,
    uint64_t* random_state,
    uint32_t* instruction_mask
) {
    SecantAstInstruction instruction;

    if (!allow_mufu) {
        instruction =
            (secant_test_ast_stress_random_u32(random_state) & 1u) != 0u
                ? secant_ast_encode_neg_f32
                : secant_ast_encode_abs_f32;
        return secant_test_ast_stress_emit_instruction(
            program,
            program_size,
            instruction,
            instruction_mask);
    }
    switch (secant_test_ast_stress_random_u32(random_state) % 10u) {
        case 0u:
            instruction = secant_ast_encode_neg_f32;
            break;
        case 1u:
            instruction = secant_ast_encode_abs_f32;
            break;
        case 2u:
            instruction = secant_ast_encode_sin_f32;
            break;
        case 3u:
            instruction = secant_ast_encode_cos_f32;
            break;
        case 4u:
            instruction = secant_ast_encode_tanh_f32;
            break;
        case 5u:
            instruction = secant_ast_encode_routine_f32(
                SECANT_TEST_AST_STRESS_ROUTINE_SAFE_SQRT,
                1u);
            break;
        case 6u:
            instruction = secant_ast_encode_routine_f32(
                SECANT_TEST_AST_STRESS_ROUTINE_SAFE_RSQRT,
                1u);
            break;
        case 7u:
            instruction = secant_ast_encode_routine_f32(
                SECANT_TEST_AST_STRESS_ROUTINE_SAFE_LG2,
                1u);
            break;
        case 8u:
            instruction = secant_ast_encode_routine_f32(
                SECANT_TEST_AST_STRESS_ROUTINE_BOUNDED_EX2,
                1u);
            break;
        default:
            instruction = secant_ast_encode_routine_f32(
                SECANT_TEST_AST_STRESS_ROUTINE_SAFE_RCP,
                1u);
            break;
    }
    return secant_test_ast_stress_emit_instruction(
        program,
        program_size,
        instruction,
        instruction_mask);
}

static int
secant_test_ast_stress_emit_unary_chain(
    SecantAstInstruction* program,
    size_t* program_size,
    size_t max_unary_depth,
    int allow_mufu,
    uint64_t* random_state,
    uint32_t* instruction_mask
) {
    const size_t unary_depth = max_unary_depth == 0u
        ? 0u
        : secant_test_ast_stress_random_u32(random_state) %
            (max_unary_depth + 1u);
    size_t unary_idx;

    for (unary_idx = 0u; unary_idx < unary_depth; ++unary_idx) {
        if (!secant_test_ast_stress_emit_unary(
                program,
                program_size,
                allow_mufu,
                random_state,
                instruction_mask)) {
            return 0;
        }
    }
    return 1;
}

static int
secant_test_ast_stress_emit_leaf(
    SecantAstInstruction* program,
    size_t* program_size,
    size_t num_inputs,
    size_t max_unary_depth,
    int allow_mufu,
    uint64_t* random_state,
    uint32_t* instruction_mask
) {
    static const float constants[] = {
        -1.0f,
        -0.5f,
        -0.25f,
        0.125f,
        0.25f,
        0.5f,
        1.0f
    };
    SecantAstInstruction instruction;

    if ((secant_test_ast_stress_random_u32(random_state) & 3u) == 0u) {
        instruction = secant_ast_encode_constant_f32(
            constants[
                secant_test_ast_stress_random_u32(random_state) %
                (sizeof(constants) / sizeof(constants[0]))]);
    } else {
        instruction = secant_ast_encode_input_f32(
            secant_test_ast_stress_random_u32(random_state) %
                (uint32_t)num_inputs);
    }
    return secant_test_ast_stress_emit_instruction(
            program,
            program_size,
            instruction,
            instruction_mask) &&
        secant_test_ast_stress_emit_unary_chain(
            program,
            program_size,
            max_unary_depth,
            allow_mufu,
            random_state,
            instruction_mask);
}

static int
secant_test_ast_stress_emit_expression(
    SecantAstInstruction* program,
    size_t* program_size,
    size_t num_inputs,
    size_t num_leaves,
    size_t max_unary_depth,
    int allow_mufu,
    uint64_t* random_state,
    uint32_t* instruction_mask
) {
    size_t first_leaves;
    size_t second_leaves;
    size_t third_leaves;
    int use_fma;
    SecantAstInstruction instruction;

    if (num_leaves == 1u) {
        return secant_test_ast_stress_emit_leaf(
            program,
            program_size,
            num_inputs,
            max_unary_depth,
            allow_mufu,
            random_state,
            instruction_mask);
    }
    use_fma = num_leaves >= 3u &&
        secant_test_ast_stress_random_u32(random_state) % 5u == 0u;
    if (use_fma) {
        first_leaves =
            1u + secant_test_ast_stress_random_u32(random_state) %
                (num_leaves - 2u);
        second_leaves =
            1u + secant_test_ast_stress_random_u32(random_state) %
                (num_leaves - first_leaves - 1u);
        third_leaves = num_leaves - first_leaves - second_leaves;
        if (!secant_test_ast_stress_emit_expression(
                program,
                program_size,
                num_inputs,
                first_leaves,
                max_unary_depth,
                allow_mufu,
                random_state,
                instruction_mask) ||
            !secant_test_ast_stress_emit_expression(
                program,
                program_size,
                num_inputs,
                second_leaves,
                max_unary_depth,
                allow_mufu,
                random_state,
                instruction_mask) ||
            !secant_test_ast_stress_emit_expression(
                program,
                program_size,
                num_inputs,
                third_leaves,
                max_unary_depth,
                allow_mufu,
                random_state,
                instruction_mask) ||
            !secant_test_ast_stress_emit_instruction(
                program,
                program_size,
                secant_ast_encode_fma_f32,
                instruction_mask)) {
            return 0;
        }
    } else {
        first_leaves =
            1u + secant_test_ast_stress_random_u32(random_state) %
                (num_leaves - 1u);
        second_leaves = num_leaves - first_leaves;
        if (!secant_test_ast_stress_emit_expression(
                program,
                program_size,
                num_inputs,
                first_leaves,
                max_unary_depth,
                allow_mufu,
                random_state,
                instruction_mask) ||
            !secant_test_ast_stress_emit_expression(
                program,
                program_size,
                num_inputs,
                second_leaves,
                max_unary_depth,
                allow_mufu,
                random_state,
                instruction_mask)) {
            return 0;
        }
        switch (secant_test_ast_stress_random_u32(random_state) %
            (allow_mufu ? 6u : 5u)) {
            case 0u:
                instruction = secant_ast_encode_add_f32;
                break;
            case 1u:
                instruction = secant_ast_encode_sub_f32;
                break;
            case 2u:
                instruction = secant_ast_encode_mul_f32;
                break;
            case 3u:
                instruction = secant_ast_encode_min_f32;
                break;
            case 4u:
                instruction = secant_ast_encode_max_f32;
                break;
            default:
                instruction = secant_ast_encode_routine_f32(
                    SECANT_TEST_AST_STRESS_ROUTINE_SAFE_DIV,
                    2u);
                break;
        }
        if (!secant_test_ast_stress_emit_instruction(
                program,
                program_size,
                instruction,
                instruction_mask)) {
            return 0;
        }
    }
    return secant_test_ast_stress_emit_unary_chain(
        program,
        program_size,
        max_unary_depth,
        allow_mufu,
        random_state,
        instruction_mask);
}

int
secant_test_ast_stress_program_generate(
    SecantAstInstruction* program,
    size_t* program_size_ret,
    size_t num_inputs,
    size_t max_leaves,
    size_t max_unary_depth,
    int allow_mufu,
    uint64_t seed,
    size_t iteration,
    size_t ast_idx,
    uint32_t* instruction_mask_ret
) {
    uint64_t random_state =
        seed ^
        ((uint64_t)iteration + 1u) * UINT64_C(0xd1b54a32d192ed03) ^
        ((uint64_t)ast_idx + 1u) * UINT64_C(0x94d049bb133111eb);
    const size_t num_leaves =
        1u + secant_test_ast_stress_random_u32(&random_state) % max_leaves;
    size_t program_size = 0u;

    if (program == NULL || program_size_ret == NULL ||
        num_inputs == 0u || num_inputs > UINT32_MAX ||
        max_leaves == 0u || instruction_mask_ret == NULL) {
        return 0;
    }
    *instruction_mask_ret = 0u;
    memset(
        program,
        0,
        SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY * sizeof(*program));
    if (!secant_test_ast_stress_emit_expression(
            program,
            &program_size,
            num_inputs,
            num_leaves,
            max_unary_depth,
            allow_mufu,
            &random_state,
            instruction_mask_ret) ||
        (allow_mufu &&
         !secant_test_ast_stress_emit_instruction(
             program,
             &program_size,
             (secant_test_ast_stress_random_u32(&random_state) & 1u) != 0u
                 ? secant_ast_encode_sin_f32
                 : secant_ast_encode_cos_f32,
             instruction_mask_ret)) ||
        !secant_test_ast_stress_emit_instruction(
            program,
            &program_size,
            secant_ast_encode_return_f32,
            instruction_mask_ret)) {
        return 0;
    }
    *program_size_ret = program_size;
    return 1;
}

float
secant_test_ast_stress_data_value(
    size_t column,
    size_t row,
    uint64_t seed
) {
    uint64_t random_state =
        seed ^
        ((uint64_t)column + 1u) * UINT64_C(0x9e3779b97f4a7c15) ^
        ((uint64_t)row + 1u) * UINT64_C(0xd1b54a32d192ed03);
    const uint32_t bits =
        secant_test_ast_stress_random_u32(&random_state);

    return ((float)(bits & 0xffffu) / 32767.5f - 1.0f) * 0.75f;
}

void
secant_test_ast_stress_data_fill(
    size_t num_inputs,
    size_t num_targets,
    size_t rows,
    uint64_t seed,
    float* input,
    float* targets
) {
    size_t input_idx;
    size_t target_idx;

    for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
        size_t row;

        for (row = 0u; row < rows; ++row) {
            input[input_idx * rows + row] =
                secant_test_ast_stress_data_value(input_idx, row, seed);
        }
    }
    for (target_idx = 0u; target_idx < num_targets; ++target_idx) {
        size_t row;

        for (row = 0u; row < rows; ++row) {
            const float lhs = secant_test_ast_stress_data_value(
                target_idx,
                row,
                seed);
            const float rhs = secant_test_ast_stress_data_value(
                target_idx + 1u,
                row,
                seed);

            targets[target_idx * rows + row] =
                (0.25f + 0.125f * (float)target_idx) * lhs + rhs;
        }
    }
}

SecantResult
secant_test_ast_stress_cpu_evaluate(
    size_t num_inputs,
    size_t num_targets,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    const float* targets,
    size_t rows,
    float* materialize_output,
    float* sse_output
) {
    SecantResult result;

    result = secant_cpu_run_static_column_materialize(
        num_inputs,
        secant_test_ast_stress_routines,
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
        asts,
        num_asts,
        input,
        num_inputs * rows,
        rows,
        rows,
        materialize_output,
        num_asts * rows,
        rows);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    memset(
        sse_output,
        0,
        num_asts * num_targets * sizeof(*sse_output));
    return secant_cpu_run_static_column_sse(
        num_inputs,
        num_targets,
        secant_test_ast_stress_routines,
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT,
        asts,
        num_asts,
        input,
        num_inputs * rows,
        rows,
        targets,
        num_targets * rows,
        rows,
        rows,
        sse_output,
        num_asts * num_targets,
        num_targets);
}

static int
secant_test_ast_stress_program_uses_mufu(uint32_t instruction_mask) {
    const uint32_t mufu_mask =
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_DIV_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_SQRT_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_RCP_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_SIN_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_COS_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_EX2_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_LG2_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32) |
        (UINT32_C(1) << SECANT_AST_INSTRUCTION_TYPE_TANH_F32);

    return (instruction_mask & mufu_mask) != 0u;
}

size_t
secant_test_ast_stress_program_mufu_count(
    const SecantAstInstruction* program,
    size_t program_size
) {
    size_t instruction_idx;
    size_t count = 0u;

    for (instruction_idx = 0u;
         instruction_idx < program_size;
         ++instruction_idx) {
        const SecantAstInstruction instruction = program[instruction_idx];

        switch ((SecantAstInstructionType)instruction.instruction_type) {
            case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
                count += instruction.payload.idx ==
                    SECANT_TEST_AST_STRESS_ROUTINE_BOUNDED_EX2
                        ? 2u
                        : 1u;
                break;
            case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
            case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
            case SECANT_AST_INSTRUCTION_TYPE_RCP_F32:
            case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
            case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
            case SECANT_AST_INSTRUCTION_TYPE_EX2_F32:
            case SECANT_AST_INSTRUCTION_TYPE_LG2_F32:
            case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32:
            case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
                ++count;
                break;
            default:
                break;
        }
    }
    return count;
}

static int
secant_test_ast_stress_close(
    float expected,
    float actual,
    float absolute_tolerance,
    float relative_tolerance,
    float* absolute_error_ret,
    float* relative_error_ret
) {
    const float absolute_error = fabsf(expected - actual);
    const float scale = fmaxf(fabsf(expected), fabsf(actual));
    const float relative_error = scale == 0.0f
        ? 0.0f
        : absolute_error / scale;

    *absolute_error_ret = absolute_error;
    *relative_error_ret = relative_error;
    return isfinite(expected) && isfinite(actual) &&
        absolute_error <= absolute_tolerance + relative_tolerance * scale;
}

int
secant_test_ast_stress_compare(
    const char* backend,
    const char* shape,
    const float* expected,
    const float* actual,
    size_t count,
    size_t value_stride,
    size_t iteration,
    uint64_t seed,
    const SecantAstInstruction* const* asts,
    const size_t* program_sizes,
    const uint32_t* program_masks,
    const size_t* program_mufu_counts,
    float base_absolute_tolerance,
    float base_relative_tolerance,
    float per_mufu_absolute_tolerance,
    float per_mufu_relative_tolerance,
    float* max_absolute_errors_ret,
    float* max_relative_errors_ret
) {
    size_t value_idx;

    for (value_idx = 0u; value_idx < count; ++value_idx) {
        const size_t ast_idx = value_idx / value_stride;
        const size_t mufu_count = program_mufu_counts[ast_idx];
        const size_t class_idx = mufu_count == 0u ? 0u : 1u;
        const float absolute_tolerance =
            base_absolute_tolerance +
            (float)mufu_count * per_mufu_absolute_tolerance;
        const float relative_tolerance =
            base_relative_tolerance +
            (float)mufu_count * per_mufu_relative_tolerance;
        float absolute_error;
        float relative_error;

        if (!secant_test_ast_stress_close(
                expected[value_idx],
                actual[value_idx],
                absolute_tolerance,
                relative_tolerance,
                &absolute_error,
                &relative_error)) {
            const size_t inner_idx = value_idx % value_stride;

            fprintf(
                stderr,
                "%s %s mismatch seed=%" PRIu64 " iteration=%zu "
                "ast=%zu value=%zu class=%s mufu_ops=%zu "
                "expected=%.9g actual=%.9g "
                "abs_error=%.9g rel_error=%.9g\n",
                backend,
                shape,
                seed,
                iteration,
                ast_idx,
                inner_idx,
                secant_test_ast_stress_program_uses_mufu(
                    program_masks[ast_idx])
                        ? "mufu"
                        : "alu",
                mufu_count,
                (double)expected[value_idx],
                (double)actual[value_idx],
                (double)absolute_error,
                (double)relative_error);
            secant_test_ast_stress_program_dump(
                asts[ast_idx],
                program_sizes[ast_idx]);
            return 0;
        }
        if (absolute_error > max_absolute_errors_ret[class_idx]) {
            max_absolute_errors_ret[class_idx] = absolute_error;
        }
        if (relative_error > max_relative_errors_ret[class_idx]) {
            max_relative_errors_ret[class_idx] = relative_error;
        }
    }
    return 1;
}
