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
#ifndef SECANT_BENCH_AST_H_INCLUDED
#define SECANT_BENCH_AST_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>

#define SECANT_BENCH_AST_PROGRAM_STRIDE 64u
#define SECANT_BENCH_AST_NUM_LEAVES 8u
#define SECANT_BENCH_AST_ROUTINE_EPSILON_BITS 0x3c23d70au

typedef enum SecantBenchAstMode {
    SECANT_BENCH_AST_MODE_SIMPLE = 0,
    SECANT_BENCH_AST_MODE_ALU = 1,
    SECANT_BENCH_AST_MODE_MUFU = 2
} SecantBenchAstMode;

typedef enum SecantBenchRoutine {
    SECANT_BENCH_ROUTINE_SAFE_DIV = 0,
    SECANT_BENCH_ROUTINE_SAFE_SQRT = 1,
    SECANT_BENCH_ROUTINE_SAFE_RSQRT = 2,
    SECANT_BENCH_ROUTINE_COUNT = 3
} SecantBenchRoutine;

uint32_t secant_bench_ast_hash32(uint32_t value);

const char* secant_bench_ast_corpus_name(SecantBenchAstMode mode);

const char* secant_bench_ast_corpus_definition_hash(SecantBenchAstMode mode);

int secant_bench_ast_storage_sizes(
    size_t num_modules,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t* program_bytes_ret,
    size_t* pointer_bytes_ret
);

void secant_bench_ast_fill(
    size_t num_modules,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    uint32_t seed,
    SecantBenchAstMode mode,
    SecantAstInstruction* programs,
    const SecantAstInstruction** asts
);

void secant_bench_ast_get_routines(
    SecantBenchAstMode mode,
    const SecantAstInstruction* const** routines_ret,
    size_t* num_routines_ret,
    const char* const** routine_names_ret
);

uint64_t secant_bench_ast_corpus_hash(
    const SecantAstInstruction* programs,
    size_t num_asts
);

#endif /* SECANT_BENCH_AST_H_INCLUDED */
