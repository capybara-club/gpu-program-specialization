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
/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#define AST_PTX_IMPLEMENTATION
#include <ast_ptx_interpreter.h>

#include <check_result_helper.h>

#include <cmath>
#include <cstdio>
#include <cstring>

enum {
    ROUTINE_MUL = 0,
    NUM_ROUTINES = 1
};

static const AstPtxInstruction mul_routine[] = {
    ast_ptx_encode_routine_arg(0u),
    ast_ptx_encode_routine_arg(1u),
    ast_ptx_encode_ptx_instruction_mul_ftz_f32,
    ast_ptx_encode_return
};

static const AstPtxInstruction* const routines[NUM_ROUTINES] = {
    mul_routine
};

static const AstPtxInstruction program[] = {
    ast_ptx_encode_input(0u),
    ast_ptx_encode_input(1u),
    ast_ptx_encode_routine(ROUTINE_MUL, 2u),
    ast_ptx_encode_constant(1.0f),
    ast_ptx_encode_ptx_instruction_add_ftz_f32,
    ast_ptx_encode_return
};

static
void
astPtxCheck(AstPtxResult result) {
    if (result != AST_PTX_SUCCESS) {
        std::fprintf(stderr, "astPtxCheck: %d\n", static_cast<int>(result));
        ASSERT(0);
    }
}

int
main() {
    static const char expected_ptx[] =
        "\t{\n"
        "\t.reg .f32 %ast<2>;\n"
        "\tmul.ftz.f32 %ast0, %x, %y;\n"
        "\tadd.ftz.f32 %ast1, %ast0, 0f3f800000;\n"
        "\tmov.f32 %z, %ast1;\n"
        "\t}";

    const char* const input_register_names[] = {
        "x",
        "y"
    };

    size_t measured_bytes = 0u;
    astPtxCheck(
        ast_ptx_compile(
            "z",
            input_register_names,
            2u,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            program,
            NULL,
            0u,
            &measured_bytes
        )
    );

    ASSERT(measured_bytes == std::strlen(expected_ptx));

    char buffer[256];
    size_t rendered_bytes = 0u;
    astPtxCheck(
        ast_ptx_compile(
            "z",
            input_register_names,
            2u,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            program,
            buffer,
            sizeof(buffer),
            &rendered_bytes
        )
    );

    ASSERT(rendered_bytes == measured_bytes);
    ASSERT(std::strcmp(buffer, expected_ptx) == 0);

    static const float input_values[] = {
        5.0f,
        3.0f
    };

    float interpreted = 0.0f;
    astPtxCheck(
        ast_ptx_interpret(
            program,
            routines,
            NUM_ROUTINES,
            input_values,
            2u,
            &interpreted
        )
    );

    ASSERT(std::fabs(interpreted - 16.0f) <= 1.0e-5f);

    return 0;
}
