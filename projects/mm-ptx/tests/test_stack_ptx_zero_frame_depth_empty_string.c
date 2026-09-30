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

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <check_result_helper.h>

#include <stdlib.h>

int
main(void) {
    static const char* stack_literal_prefixes[] = {
        "u32"
    };

    static const StackPtxArgTypeInfo arg_type_info[] = {
        { .stack_idx = 0, .num_vec_elems = 0 }
    };

    static const StackPtxStackInfo stack_info = {
        .ptx_instruction_strings = NULL,
        .ptx_instruction_descriptors = NULL,
        .num_ptx_instructions = 0,
        .special_register_strings = NULL,
        .special_register_descriptors = NULL,
        .num_special_registers = 0,
        .stack_literal_prefixes = stack_literal_prefixes,
        .num_stacks = 1,
        .arg_type_info = arg_type_info,
        .num_arg_types = 1
    };

    static const StackPtxCompilerInfo compiler_info = {
        .max_ast_size = 1,
        .max_ast_to_visit_stack_depth = 1,
        .stack_size = 1,
        .max_frame_depth = 0,
        .store_size = 0
    };

    size_t workspace_size = 0;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_info,
            &workspace_size
        )
    );

    void* workspace = malloc(workspace_size);
    ASSERT(workspace != NULL);

    char buffer[1] = { 'x' };
    size_t bytes_written = 123;

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_info,
            NULL,
            NULL,
            0,
            NULL,
            0,
            NULL,
            0,
            0,
            workspace,
            workspace_size,
            buffer,
            sizeof(buffer),
            &bytes_written
        )
    );

    ASSERT(bytes_written == 0);
    ASSERT(buffer[0] == '\0');

    free(workspace);

    return 0;
}
