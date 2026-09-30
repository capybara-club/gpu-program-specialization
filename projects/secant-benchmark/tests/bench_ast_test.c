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
#include "bench_ast.h"
#include "secant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SECANT_PORTABLE_ALU_INCLUDE_AST
#define SECANT_PORTABLE_ALU_INCLUDE_NATIVE
#include "../bench/corpus/portable_alu_v1.h"

#define SECANT_PORTABLE_ALU_TEST_CASES 8u

static float
portable_expected(size_t ast_idx, const float* x) {
    switch (ast_idx) {
        case 0u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_0(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 1u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_1(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 2u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_2(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 3u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_3(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 4u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_4(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 5u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_5(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        case 6u:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_6(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        default:
            return SECANT_PORTABLE_ALU_SCALAR_EXPR_7(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    }
}

static int
test_portable_corpus(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    float input[SECANT_PORTABLE_ALU_NUM_INPUTS];
    float output[SECANT_PORTABLE_ALU_TEST_CASES];
    size_t ast_idx;

    secant_bench_ast_fill(
        1u,
        1u,
        SECANT_PORTABLE_ALU_TEST_CASES,
        SECANT_PORTABLE_ALU_NUM_INPUTS,
        1234567u,
        SECANT_BENCH_AST_MODE_ALU,
        programs,
        asts);
    if (strcmp(
            secant_bench_ast_corpus_name(SECANT_BENCH_AST_MODE_ALU),
            SECANT_PORTABLE_ALU_CORPUS_NAME) != 0 ||
        strcmp(
            secant_bench_ast_corpus_definition_hash(
                SECANT_BENCH_AST_MODE_ALU),
            SECANT_PORTABLE_ALU_CORPUS_HASH) != 0) {
        return 1;
    }
    for (ast_idx = 0u;
         ast_idx < SECANT_PORTABLE_ALU_TEST_CASES;
         ++ast_idx) {
        if (memcmp(
                asts[ast_idx],
                secant_portable_alu_asts[ast_idx],
                SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS *
                    sizeof(SecantAstInstruction)) != 0) {
            return 1;
        }
        input[ast_idx] = 0.25f + 0.125f * (float)ast_idx;
    }
    if (secant_cpu_run_static_column_materialize(
            SECANT_PORTABLE_ALU_NUM_INPUTS,
            NULL,
            0u,
            asts,
            SECANT_PORTABLE_ALU_TEST_CASES,
            input,
            SECANT_PORTABLE_ALU_NUM_INPUTS,
            1u,
            1u,
            output,
            SECANT_PORTABLE_ALU_TEST_CASES,
            1u) != SECANT_SUCCESS) {
        return 1;
    }
    for (ast_idx = 0u;
         ast_idx < SECANT_PORTABLE_ALU_TEST_CASES;
         ++ast_idx) {
        if (fabsf(output[ast_idx] - portable_expected(ast_idx, input)) >
            1.0e-6f) {
            return 1;
        }
    }
    return 0;
}

static int
test_run(
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
) {
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    size_t num_routines = 0u;
    size_t instruction_idx;
    float input[8];
    float output[24];
    size_t value_idx;
    int saw_rcp = 0;
    int saw_div = 0;

    secant_bench_ast_fill(
        2u,
        3u,
        4u,
        8u,
        7u,
        SECANT_BENCH_AST_MODE_MUFU,
        programs,
        asts);
    secant_bench_ast_get_routines(
        SECANT_BENCH_AST_MODE_MUFU,
        &routines,
        &num_routines,
        &routine_names);
    if (routines == NULL ||
        routine_names == NULL ||
        num_routines != SECANT_BENCH_ROUTINE_COUNT) {
        return 1;
    }

    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstructionType type =
            (SecantAstInstructionType)
                routines[SECANT_BENCH_ROUTINE_SAFE_DIV][instruction_idx]
                    .instruction_type;

        saw_rcp |= type == SECANT_AST_INSTRUCTION_TYPE_RCP_F32;
        saw_div |= type == SECANT_AST_INSTRUCTION_TYPE_DIV_F32;
        if (type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            break;
        }
    }
    if (!saw_rcp || saw_div ||
        secant_bench_ast_corpus_hash(programs, 24u) == 0u) {
        return 1;
    }
    for (value_idx = 0u; value_idx < 8u; ++value_idx) {
        input[value_idx] = 0.25f + 0.125f * (float)value_idx;
    }
    if (secant_cpu_run_static_column_materialize(
            8u,
            routines,
            num_routines,
            asts,
            24u,
            input,
            8u,
            1u,
            1u,
            output,
            24u,
            1u) != SECANT_SUCCESS) {
        return 1;
    }
    for (value_idx = 0u; value_idx < 24u; ++value_idx) {
        if (!isfinite(output[value_idx])) {
            return 1;
        }
    }
    return 0;
}

int
main(void) {
    const SecantAstInstruction** asts = NULL;
    SecantAstInstruction* programs = NULL;
    size_t program_bytes = 0u;
    size_t pointer_bytes = 0u;
    int result = 1;

    if (secant_bench_ast_storage_sizes(
            2u,
            3u,
            4u,
            &program_bytes,
            &pointer_bytes)) {
        programs = (SecantAstInstruction*)malloc(program_bytes);
        asts = (const SecantAstInstruction**)malloc(pointer_bytes);
        if (programs != NULL && asts != NULL) {
            result = test_portable_corpus(programs, asts);
            if (result == 0) {
                result = test_run(programs, asts);
            }
        }
    }
    free(asts);
    free(programs);
    return result;
}
