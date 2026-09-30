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
#include "s_ast_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_ROWS 17u
#define TEST_INPUTS 3u
#define TEST_TARGETS 2u
#define TEST_ASTS 3u

static const SecantAstInstruction test_safe_div[] = {
    secant_ast_encode_routine_arg_f32(0u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_routine_arg_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_constant_f32_bits(
        SECANT_F32_BITS_ONE_MILLIONTH),
    secant_ast_encode_add_f32,
    secant_ast_encode_rcp_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_add[] = {
    secant_ast_encode_column_f32(0u),
    secant_ast_encode_column_f32(1u),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_sin[] = {
    secant_ast_encode_column_f32(2u),
    secant_ast_encode_sin_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction test_div[] = {
    secant_ast_encode_column_f32(0u),
    secant_ast_encode_column_f32(1u),
    secant_ast_encode_routine_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction* const test_routines[] = { test_safe_div };
static const SecantAstInstruction* const test_asts[] = { test_add, test_sin, test_div };
static const SecantAstInstruction test_s32[] = {
    secant_ast_encode_constant_s32(1),
    secant_ast_encode_return_s32
};
static const SecantAstInstruction* const test_s32_asts[] = { test_s32 };

static int
test_close(float actual, float expected) {
    return fabsf(actual - expected) <= 1.0e-5f * (1.0f + fabsf(expected));
}

int
main(void) {
    SecantAstInstruction deep_program[
        (SECANT_AST_MAX_STACK_DEPTH + 1u) * 5u + 1u];
    const SecantAstInstruction* deep_asts[] = { deep_program };
    float input[TEST_INPUTS * TEST_ROWS];
    float targets[TEST_TARGETS * TEST_ROWS];
    float materialized[TEST_ASTS * TEST_ROWS];
    float sse[TEST_ASTS * TEST_TARGETS];
    float expected_sse[TEST_ASTS * TEST_TARGETS];
    SecantCpuMaterializeRun materialize_run = secant_cpu_materialize_run_init();
    SecantCpuSSERun sse_run = secant_cpu_sse_run_init();
    SecantResult result;
    size_t row;
    size_t ast_idx;
    size_t deep_program_size = 0u;

    for (row = 0u; row < TEST_ROWS; ++row) {
        input[row] = 0.5f + 0.1f * (float)row;
        input[TEST_ROWS + row] = 1.25f + 0.03f * (float)row;
        input[2u * TEST_ROWS + row] = -0.75f + 0.07f * (float)row;
        targets[row] = 0.2f * (float)row;
        targets[TEST_ROWS + row] = 1.0f - 0.04f * (float)row;
    }
    memset(materialized, 0, sizeof(materialized));
    materialize_run.programs.routines.items = test_routines;
    materialize_run.programs.routines.count = 1u;
    materialize_run.programs.asts.items = test_asts;
    materialize_run.programs.asts.count = TEST_ASTS;
    materialize_run.num_inputs = TEST_INPUTS;
    materialize_run.input.data = input;
    materialize_run.input.num_elements = TEST_INPUTS * TEST_ROWS;
    materialize_run.input.leading_dimension = TEST_ROWS;
    materialize_run.num_rows = TEST_ROWS;
    materialize_run.output.data = materialized;
    materialize_run.output.num_elements = TEST_ASTS * TEST_ROWS;
    materialize_run.output.leading_dimension = TEST_ROWS;
    result = secant_cpu_run(&materialize_run.header);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr, "materialize failed: %s\n", secant_result_to_string(result));
        return 1;
    }
    for (row = 0u; row < TEST_ROWS; ++row) {
        const float input0 = input[row];
        const float input1 = input[TEST_ROWS + row];
        const float input2 = input[2u * TEST_ROWS + row];
        const float expected[TEST_ASTS] = {
            input0 + input1,
            sinf(input2),
            input0 * input1 / (input1 * input1 + 1.0e-6f)
        };

        for (ast_idx = 0u; ast_idx < TEST_ASTS; ++ast_idx) {
            if (!test_close(materialized[ast_idx * TEST_ROWS + row], expected[ast_idx])) {
                fprintf(stderr, "materialize mismatch at ast=%zu row=%zu\n", ast_idx, row);
                return 1;
            }
        }
    }
    materialize_run.output.data = input;
    materialize_run.output.num_elements = TEST_INPUTS * TEST_ROWS;
    if (secant_cpu_run_materialize(&materialize_run) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "CPU materialize accepted overlapping input and output\n");
        return 1;
    }
    materialize_run.output.data = materialized;
    materialize_run.output.num_elements = TEST_ASTS * TEST_ROWS;
    {
        struct ExtendedCpuRun {
            SecantCpuMaterializeRun run;
            uint64_t trailing;
        } extended_run;

        memset(&extended_run, 0, sizeof(extended_run));
        extended_run.run = materialize_run;
        extended_run.run.header.struct_size = (uint32_t)sizeof(extended_run);
        extended_run.trailing = UINT64_C(0x1122334455667788);
        if (secant_cpu_run_materialize(&extended_run.run) != SECANT_SUCCESS ||
            extended_run.trailing != UINT64_C(0x1122334455667788)) {
            fprintf(stderr, "CPU run rejected or modified a compatible trailing extension\n");
            return 1;
        }
    }

    for (ast_idx = 0u; ast_idx < TEST_ASTS * TEST_TARGETS; ++ast_idx) {
        sse[ast_idx] = 0.25f;
        expected_sse[ast_idx] = 0.25f;
    }
    for (ast_idx = 0u; ast_idx < TEST_ASTS; ++ast_idx) {
        size_t target_idx;

        for (target_idx = 0u; target_idx < TEST_TARGETS; ++target_idx) {
            for (row = 0u; row < TEST_ROWS; ++row) {
                const float error =
                    materialized[ast_idx * TEST_ROWS + row] -
                    targets[target_idx * TEST_ROWS + row];

                expected_sse[ast_idx * TEST_TARGETS + target_idx] += error * error;
            }
        }
    }
    sse_run.programs.routines.items = test_routines;
    sse_run.programs.routines.count = 1u;
    sse_run.programs.asts.items = test_asts;
    sse_run.programs.asts.count = TEST_ASTS;
    sse_run.num_inputs = TEST_INPUTS;
    sse_run.num_targets = TEST_TARGETS;
    sse_run.input.data = input;
    sse_run.input.num_elements = TEST_INPUTS * TEST_ROWS;
    sse_run.input.leading_dimension = TEST_ROWS;
    sse_run.targets.data = targets;
    sse_run.targets.num_elements = TEST_TARGETS * TEST_ROWS;
    sse_run.targets.leading_dimension = TEST_ROWS;
    sse_run.num_rows = TEST_ROWS;
    sse_run.output.data = sse;
    sse_run.output.num_elements = TEST_ASTS * TEST_TARGETS;
    sse_run.output.leading_dimension = TEST_TARGETS;
    result = secant_cpu_run(&sse_run.header);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr, "SSE failed: %s\n", secant_result_to_string(result));
        return 1;
    }
    for (ast_idx = 0u; ast_idx < TEST_ASTS * TEST_TARGETS; ++ast_idx) {
        if (!test_close(sse[ast_idx], expected_sse[ast_idx])) {
            fprintf(stderr, "SSE mismatch at %zu: %.9g != %.9g\n",
                ast_idx, sse[ast_idx], expected_sse[ast_idx]);
            return 1;
        }
    }
    for (ast_idx = 0u;
         ast_idx < SECANT_AST_MAX_STACK_DEPTH + 1u;
         ++ast_idx) {
        if (secant_internal_ast_constant_f32_write(
                1.0f,
                deep_program,
                sizeof(deep_program),
                &deep_program_size) != SECANT_SUCCESS) {
            return 1;
        }
    }
    if (secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_RETURN_F32,
            0u,
            0u,
            deep_program,
            sizeof(deep_program),
            &deep_program_size) != SECANT_SUCCESS) {
        return 1;
    }
    materialize_run.programs.routines.items = NULL;
    materialize_run.programs.routines.count = 0u;
    materialize_run.programs.asts.items = deep_asts;
    materialize_run.programs.asts.count = 1u;
    materialize_run.output.num_elements = TEST_ROWS;
    result = secant_cpu_run(&materialize_run.header);
    if (result != SECANT_ERROR_STACK_OVERFLOW) {
        fprintf(
            stderr,
            "deep AST returned %s instead of stack overflow\n",
            secant_result_to_string(result));
        return 1;
    }
    materialize_run.programs.asts.items = test_s32_asts;
    result = secant_cpu_run(&materialize_run.header);
    if (result != SECANT_ERROR_UNSUPPORTED_OP) {
        fprintf(
            stderr,
            "s32 AST returned %s instead of unsupported op\n",
            secant_result_to_string(result));
        return 1;
    }
    materialize_run.programs.asts.items = test_asts;
    materialize_run.programs.asts.count = TEST_ASTS;
    materialize_run.output.num_elements = TEST_ASTS * TEST_ROWS;
    materialize_run.header.version = 0u;
    if (secant_cpu_run(&materialize_run.header) != SECANT_ERROR_UNSUPPORTED_VERSION) {
        fprintf(stderr, "CPU run accepted an unsupported descriptor version\n");
        return 1;
    }
    materialize_run.header.version = SECANT_CPU_RUN_VERSION_3;
    materialize_run.header.struct_size = (uint32_t)(sizeof(materialize_run.header) - 1u);
    if (secant_cpu_run(&materialize_run.header) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "CPU run accepted a truncated descriptor header\n");
        return 1;
    }
    materialize_run.header.struct_size = (uint32_t)sizeof(materialize_run);
    materialize_run.header.flags = 1u;
    if (secant_cpu_run(&materialize_run.header) != SECANT_ERROR_INVALID_VALUE) {
        fprintf(stderr, "CPU run accepted unsupported descriptor flags\n");
        return 1;
    }
    materialize_run.header.flags = 0u;
    materialize_run.header.shape = UINT32_MAX;
    if (secant_cpu_run(&materialize_run.header) != SECANT_ERROR_UNSUPPORTED_SHAPE) {
        fprintf(stderr, "CPU run accepted an unknown kernel shape\n");
        return 1;
    }
    return 0;
}
