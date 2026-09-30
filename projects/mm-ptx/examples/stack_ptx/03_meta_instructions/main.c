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
    REGISTER_OUTPUT_3,
    REGISTER_OUTPUT_4,
    REGISTER_NUM_ENUMS
} Register;

static const StackPtxRegister registers[] = {
    [REGISTER_INPUT_0] =   { .name = "input_register_0", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_INPUT_1] =   { .name = "input_register_1", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT_0] =  { .name = "output_register_0", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT_1] =  { .name = "output_register_1", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT_2] =  { .name = "output_register_2", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT_3] =  { .name = "output_register_3", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT_4] =  { .name = "output_register_4", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
};
static const size_t num_registers = REGISTER_NUM_ENUMS;

// We describe two inputs that are both of type U32.
// The second operand in `stack_ptx_encode_input` is the index that let's the compiler know
// where to find the register name from the `register_names` array.
// Using an enum like `RegisterName` helps keep the indices consistent.
static const StackPtxInstruction input_0 = stack_ptx_encode_input(REGISTER_INPUT_0);
__attribute__((unused))
static const StackPtxInstruction input_1 = stack_ptx_encode_input(REGISTER_INPUT_1);

// We're going to request one U32 register value from what is on the U32 stack.
// The value on the U32 stack is an AST that will be evalutated to generate 
// the PTX code to give the result.
static const size_t request = REGISTER_OUTPUT_0;

// We will allow the stack machine to run "execution_limit" number of operations before halting
static const int execution_limit = 100;

static
void
run_instructions(
    const StackPtxInstruction* instructions,
    const size_t* requests,
    int32_t num_requests,
    void* workspace,
    size_t workspace_size
) {
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
            num_registers,
            NULL, 
            0, 
            requests, 
            num_requests,
            execution_limit,
            workspace,
            workspace_size,
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
            num_registers,
            NULL, 
            0, 
            requests,
            num_requests,
            execution_limit,
            workspace,
            workspace_size,
            buffer,
            capacity,
            &required
        )
    );

    // Print the ptx buffer
    printf("%s\n", buffer);

    free(buffer);
}

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

    // We can start by just adding 1 to the input_0
    static const StackPtxInstruction simple_instructions[] = {
        stack_ptx_encode_constant_u32(1),
        input_0,
        stack_ptx_encode_ptx_instruction_add_u32,
        stack_ptx_encode_return
    };

    printf("simple_instructions:\n");
    run_instructions(simple_instructions, &request, 1, stack_ptx_workspace, stack_ptx_workspace_size);

    // We can start by just adding 1 to the input_0
    // We add the swap meta instruction to the list. This will operate on the
    // stack and flip the order.
    static const StackPtxInstruction flipped_order_instructions[] = {
        stack_ptx_encode_constant_u32(1),
        input_0,
        stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_ptx_instruction_add_u32,
        stack_ptx_encode_return
    };
    
    printf("flipped_order_instructions:\n");
    run_instructions(flipped_order_instructions, &request, 1, stack_ptx_workspace, stack_ptx_workspace_size);

    // We can add 1 to input_0 and then duplicate that result and multiply
    // both together. This will be (1 + input_0) * (1 + input_0). We should
    // only see the (1 + input_0) result computed once because it's value
    // is already stored in the AST. You'll see the register represented twice
    // in the mul operation.
    static const StackPtxInstruction add_dup_instructions[] = {
        stack_ptx_encode_constant_u32(1),
        input_0,
        stack_ptx_encode_ptx_instruction_add_u32,
        stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_ptx_instruction_mul_lo_u32,
        stack_ptx_encode_return
    };
    
    printf("add_dup_instructions:\n");
    run_instructions(add_dup_instructions, &request, 1, stack_ptx_workspace, stack_ptx_workspace_size);

    // We'll now create an array of many requests.
    static const size_t many_requests[] = {
        REGISTER_OUTPUT_0,
        REGISTER_OUTPUT_1,
        REGISTER_OUTPUT_2,
        REGISTER_OUTPUT_3,
        REGISTER_OUTPUT_4,
    };

    // We'll first just add all these constants to the U32 stack and
    // request 5 different outputs.
    static const StackPtxInstruction forward_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_return
    };

    printf("forward_instructions:\n");
    run_instructions(forward_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    // REVERSE
    // Now we'll reverse the stack with a meta instruction
    static const StackPtxInstruction reverse_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_reverse(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_return
    };

    printf("reverse_instructions:\n");
    run_instructions(reverse_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    // YANK_DUP
    // We can use the special stack_ptx_encode_meta_constant to 
    // put an integer in a special stack to use as an operand for 
    // a following meta instruction. In this case we'll push 2 on the stack and use Yank Dup
    // to pull the value 2 deep in the stack and duplicate it to the top of the stack.
    // This should bring the value 3 to the top such that the order is:
    // 3,5,4,3,2
    static const StackPtxInstruction yank_dup_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_constant(2),
        stack_ptx_encode_meta_yank_dup(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_return
    };

    printf("yank_dup_instructions:\n");
    run_instructions(yank_dup_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    // SWAP_WITH
    // We'll now try swap_with that uses the meta_constant to swap the top of the stack with.
    static const StackPtxInstruction swap_with_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_constant(2),
        stack_ptx_encode_meta_swap_with(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_return
    };

    printf("swap_with_instructions:\n");
    run_instructions(swap_with_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    // REPLACE
    // Replace
    static const StackPtxInstruction replace_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_constant(2),
        stack_ptx_encode_meta_replace(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_return
    };

    printf("replace_instructions:\n");
    run_instructions(replace_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    // DROP
    // Drop the top 2 values from the U32 stack.
    static const StackPtxInstruction drop_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_constant(2),
        stack_ptx_encode_meta_drop(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_return
    };

    printf("drop_instructions:\n");
    run_instructions(drop_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    // CLEAR
    // Clear all values from the U32 stack, then push input_0.
    // If clear is working, output_0 should resolve to input_0 (not to 5).
    static const StackPtxInstruction clear_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_clear(STACK_PTX_STACK_TYPE_U32),
        input_0,
        stack_ptx_encode_return
    };

    printf("clear_instructions:\n");
    run_instructions(clear_instructions, &request, 1, stack_ptx_workspace, stack_ptx_workspace_size);

    // ROTATE
    // Drop the top 2 values from the U32 stack.
    static const StackPtxInstruction rotate_instructions[] = {
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_constant_u32(1),
        stack_ptx_encode_constant_u32(2),
        stack_ptx_encode_constant_u32(3),
        stack_ptx_encode_constant_u32(4),
        stack_ptx_encode_constant_u32(5),
        stack_ptx_encode_meta_constant(2),
        stack_ptx_encode_meta_rotate(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_return
    };

    printf("rotate_instructions:\n");
    run_instructions(rotate_instructions, many_requests, STACK_PTX_ARRAY_NUM_ELEMS(many_requests), stack_ptx_workspace, stack_ptx_workspace_size);

    free(stack_ptx_workspace);

    return 0;
}
