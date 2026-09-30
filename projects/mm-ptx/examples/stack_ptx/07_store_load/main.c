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

#include <stdio.h>
#include <stdlib.h>

// Import Stack PTX for this compilation using STACK_PTX_IMPLEMENTATION
#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.h>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>

typedef enum {
    REGISTER_INPUT_0,
    REGISTER_INPUT_1,
    REGISTER_OUTPUT_0,
    REGISTER_OUTPUT_1,
    REGISTER_OUTPUT_2,
    REGISTER_NUM_ENUMS
} Register;

static const StackPtxRegister registers[] = {
    [REGISTER_INPUT_0] =    { .name = "input_register_0",   .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_INPUT_1] =    { .name = "input_register_1",   .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_0] =   { .name = "output_register_0",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_1] =   { .name = "output_register_1",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_2] =   { .name = "output_register_2",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
};
static const size_t num_registers = STACK_PTX_ARRAY_NUM_ELEMS(registers);

// We need to add the names of the instructions so when an instruction is performed
// the compiler knows what to call it in ptx. Here the names come from the import
// "stack_ptx_generated_descriptions.h"

// Similarly we need to specify what to call an input register when the compiler
// processes input #0 and input #1.

// We describe a series of instructions. We push some U32 values on to the stack and
// perform some ptx operations on them. We push input_0 and input_1 on the stack as well 
// so we should see these registers show up as operands in the output ptx.
// stack_ptx_encode_return is mandatory at the end of the instructions list as a null terminator.

typedef enum {
    VAR_DIFF,
    VAR_ABS_DIFF,
    VAR_NUM_ENUMS
} Var;

static const StackPtxInstruction instructions[] = {
    stack_ptx_encode_input(REGISTER_INPUT_0),
    stack_ptx_encode_input(REGISTER_INPUT_1),
    stack_ptx_encode_ptx_instruction_sub_ftz_f32,
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, VAR_DIFF),
    stack_ptx_encode_load(VAR_DIFF),
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, VAR_ABS_DIFF),
    stack_ptx_encode_load(VAR_ABS_DIFF),
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_load(VAR_ABS_DIFF),
    stack_ptx_encode_ptx_instruction_cos_approx_ftz_f32,
    stack_ptx_encode_load(VAR_DIFF),
    stack_ptx_encode_ptx_instruction_sin_approx_ftz_f32,
    stack_ptx_encode_return
};

// We're going to request one U32 register value from what is on the U32 stack.
// The value on the U32 stack is an AST that will be evalutated to generate 
// the PTX code to give the result.

static const size_t requests[] = {
    REGISTER_OUTPUT_0, REGISTER_OUTPUT_1, REGISTER_OUTPUT_2
};
static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

// We will allow the stack machine to run "execution_limit" number of operations before halting
static const size_t execution_limit = 100;

int
main() {
    // We first need to query the workspace necessary for the `stack_ptx_compile` function.
    // We set the max AST size to use during the run and 
    // the AST visit depth. Every operation is encoded in to the AST during the execution of the program
    // before compilation. AST visit depth is the max stack depth as we visit the AST during compilation.
    // We use the stackPtxCheck macro to error out on a non STACK_PTX_SUCCESS result.
    size_t stack_ptx_workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

    // We allocate the memory for the workspace.
    void* stack_ptx_workspace = malloc(stack_ptx_workspace_size);
    
    char* buffer = NULL;
    size_t required = 0;
    size_t capacity = 0;
    
    // We first "measure" the size of the output buffer by passing NULL as the output buffer.
    // This will give us the amount of bytes we require to write the output ptx.
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            requests, num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            NULL,
            capacity,
            &required
        )
    );

    // Allocate the required buffer size, factoring in the string null terminator
    capacity = required + 1;
    buffer = (char*)malloc(capacity);

    // Now compile to ptx with the actual buffer.
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            REGISTER_NUM_ENUMS,
            NULL,
            0,
            requests,
            num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            buffer,
            capacity,
            &required
        )
    );

    // We don't need the workspace anymore.
    free(stack_ptx_workspace);

    // Print the ptx buffer
    printf("%s\n", buffer);

    // We don't need the buffer anymore.
    free(buffer);

    return 0;
}
