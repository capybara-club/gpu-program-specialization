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
#ifndef SECANT_TEST_AST_STRESS_H_INCLUDED
#define SECANT_TEST_AST_STRESS_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>

#define SECANT_TEST_AST_STRESS_PROGRAM_CAPACITY \
    (128u * SECANT_AST_MAX_INSTRUCTION_BYTES)
#define SECANT_TEST_AST_STRESS_ROUTINE_COUNT 6u

extern const SecantAstInstruction* const
    secant_test_ast_stress_routines[
        SECANT_TEST_AST_STRESS_ROUTINE_COUNT];

int secant_test_ast_stress_program_generate(
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
);

float secant_test_ast_stress_data_value(
    size_t column,
    size_t row,
    uint64_t seed
);

void secant_test_ast_stress_data_fill(
    size_t num_inputs,
    size_t num_targets,
    size_t rows,
    uint64_t seed,
    float* input,
    float* targets
);

SecantResult secant_test_ast_stress_cpu_evaluate(
    size_t num_inputs,
    size_t num_targets,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    const float* targets,
    size_t rows,
    float* materialize_output,
    float* sse_output
);

size_t secant_test_ast_stress_program_mufu_count(
    const SecantAstInstruction* program,
    size_t program_size
);

int secant_test_ast_stress_compare(
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
);

#endif
