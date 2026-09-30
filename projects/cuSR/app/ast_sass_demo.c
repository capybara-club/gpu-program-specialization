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
#define CUSR_AST_SASS_IMPLEMENTATION
#include <cusr_ast_sass.h>
#include <cusr_ast_routines.h>

#include <stdio.h>

#define CUSR_AST_SASS_DEMO_MAX_SASS 64u

static const CusrAstInstruction cusr_ast_sass_demo_routine[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_mul,
    cusr_ast_encode_routine_arg(2u),
    cusr_ast_encode_add,
    cusr_ast_encode_sqrt,
    cusr_ast_encode_return
};

static const CusrAstInstruction* const cusr_ast_sass_demo_routines[] = {
    cusr_ast_sass_demo_routine
};

static const CusrAstInstruction cusr_ast_sass_demo_program[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_input(1u),
    cusr_ast_encode_input(2u),
    cusr_ast_encode_routine(0u, 3u),
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_ast_sass_two_sqrt_program[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_sqrt,
    cusr_ast_encode_input(1u),
    cusr_ast_encode_sqrt,
    cusr_ast_encode_add,
    cusr_ast_encode_return
};

static const CusrAstInstruction cusr_ast_sass_routine_library_program[] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_safe_sqrt,
    cusr_ast_encode_input(1u),
    cusr_ast_encode_exp,
    cusr_ast_encode_add,
    cusr_ast_encode_input(2u),
    cusr_ast_encode_log10,
    cusr_ast_encode_add,
    cusr_ast_encode_input(3u),
    cusr_ast_encode_input(4u),
    cusr_ast_encode_safe_div,
    cusr_ast_encode_add,
    cusr_ast_encode_input(5u),
    cusr_ast_encode_input(6u),
    cusr_ast_encode_pow,
    cusr_ast_encode_add,
    cusr_ast_encode_return
};

static int
cusr_ast_sass_demo_run_program(
    const char* label,
    const CusrAstInstruction* program,
    const CusrAstInstruction* const* routines,
    size_t num_routines,
    const uint8_t* available_registers,
    size_t num_available_registers,
    uint32_t first_new_register,
    CusrAstSassResult expected_result)
{
    const uint8_t input_registers[8] = { 6u, 12u, 15u, 14u, 13u, 16u, 17u, 18u };
    CusrSassInstruction sass[CUSR_AST_SASS_DEMO_MAX_SASS];
    size_t sass_count = 0u;
    uint32_t expanded_register_count = first_new_register;
    CusrAstSassResult result;

    result = cusr_ast_sass_generate(
        input_registers,
        8u,
        available_registers,
        num_available_registers,
        10u,
        0x01cu,
        first_new_register,
        12u,
        0u,
        routines,
        num_routines,
        program,
        sass,
        CUSR_AST_SASS_DEMO_MAX_SASS,
        &sass_count,
        &expanded_register_count
    );

    printf("%s result=%s", label, cusr_ast_sass_result_to_string(result));

    if (result != expected_result) {
        fprintf(
            stderr,
            "\nexpected %s\n",
            cusr_ast_sass_result_to_string(expected_result)
        );
        return 1;
    }

    if (result != CUSR_AST_SASS_SUCCESS) {
        printf("\n");
        return 0;
    }

    printf(" sass_count=%zu expanded_register_count=%u\n", sass_count, expanded_register_count);
    cusr_ast_sass_print(stdout, sass, sass_count);
    return 0;
}

int
main(void)
{
    const uint8_t available_registers[4] = { 10u, 19u, 11u, 20u };

    if (cusr_ast_sass_demo_run_program(
        "site_regs",
        cusr_ast_sass_demo_program,
        cusr_ast_sass_demo_routines,
        1u,
        available_registers,
        4u,
        30u,
        CUSR_AST_SASS_SUCCESS
    ) != 0) {
        return 1;
    }

    printf("\n");
    if (cusr_ast_sass_demo_run_program(
        "expanded_regs",
        cusr_ast_sass_demo_program,
        cusr_ast_sass_demo_routines,
        1u,
        NULL,
        0u,
        30u,
        CUSR_AST_SASS_SUCCESS
    ) != 0) {
        return 1;
    }

    printf("\n");
    if (cusr_ast_sass_demo_run_program(
        "last_register_success",
        cusr_ast_sass_demo_program,
        cusr_ast_sass_demo_routines,
        1u,
        NULL,
        0u,
        254u,
        CUSR_AST_SASS_SUCCESS
    ) != 0) {
        return 1;
    }

    printf("\n");
    if (cusr_ast_sass_demo_run_program(
        "routine_library",
        cusr_ast_sass_routine_library_program,
        cusr_ast_routines,
        CUSR_AST_ROUTINES_COUNT,
        available_registers,
        4u,
        30u,
        CUSR_AST_SASS_SUCCESS
    ) != 0) {
        return 1;
    }

    printf("\n");
    if (cusr_ast_sass_demo_run_program(
        "register_overflow",
        cusr_ast_sass_two_sqrt_program,
        NULL,
        0u,
        NULL,
        0u,
        254u,
        CUSR_AST_SASS_ERROR_REGISTER_OVERFLOW
    ) != 0) {
        return 1;
    }

    return 0;
}
