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

// Describe the routines as an index, this will be useful later.
enum RoutineIdx {
    ROUTINE_0,
    ROUTINE_1,
    ROUTINE_2,
    NUM_ROUTINES
};

// Describe the routines, use designated initializers to position
// them properly in the array for indexing.
static const StackPtxInstruction* routines[] = {
    [ROUTINE_0] = (StackPtxInstruction[]){
        stack_ptx_encode_constant_u32(1023),
        stack_ptx_encode_return
    },
    [ROUTINE_1] = (StackPtxInstruction[]){
        stack_ptx_encode_constant_u32(1024),
        stack_ptx_encode_return
    },
    // ROUTINE_2 will call ROUTINE_1
    [ROUTINE_2] = (StackPtxInstruction[]){
        stack_ptx_encode_constant_u32(1025),
        stack_ptx_encode_routine(ROUTINE_1),
        stack_ptx_encode_return
    }
};

// Set up the register names idx.
typedef enum {
    REGISTER_INPUT_0,
    REGISTER_INPUT_1,
    REGISTER_OUTPUT,
    REGISTER_NUM_ENUMS
} Register;

static const StackPtxRegister registers[] = {
    { .name = "input_register_0", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    { .name = "input_register_1", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    { .name = "output_register",  .stack_idx = STACK_PTX_STACK_TYPE_U32 },
};

// Set up the instructions we will run
static const StackPtxInstruction instructions[] = {
    stack_ptx_encode_constant_u32(1),
    stack_ptx_encode_constant_u32(2),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_constant_u32(3),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_input(REGISTER_INPUT_0),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_input(REGISTER_INPUT_1),
    stack_ptx_encode_routine(ROUTINE_2),
    stack_ptx_encode_ptx_instruction_mul_lo_u32,
    stack_ptx_encode_return
};

// We're going to request one U32 register value from what is on the U32 stack.
// The value on the U32 stack is an AST that will be evalutated to generate 
// the PTX code to give the result.
static const size_t request = REGISTER_OUTPUT;

// Set execution limit for running instructions
static const int execution_limit = 100;

int
main() {
    size_t stack_ptx_workspace_size;
    stackPtxCheck( 
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

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
            registers,
            REGISTER_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            &request,
            1,
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
            routines,
            NUM_ROUTINES,
            &request,
            1,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            buffer,
            capacity,
            &required
        )
    );

    free(stack_ptx_workspace);

    // Print the ptx buffer
    printf("%s\n", buffer);

    free(buffer);

    return 0;
}
