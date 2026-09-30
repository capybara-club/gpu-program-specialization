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
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.hpp>

#include <check_result_helper.h>

using namespace stack_ptx;

static const StackPtxCompilerInfo compiler_info = {
    100,
	20,
	4,
	128,
	16
};

enum class Register {
    IN_0,
    IN_1,
    OUT_0,
    NUM_ENUMS
};

static const StackPtxRegister registers[] = {
    {"in_0", static_cast<size_t>(StackType::U32)},
    {"in_1", static_cast<size_t>(StackType::U32)},
    {"out_0", static_cast<size_t>(StackType::U32)}
};
static const size_t num_registers = static_cast<size_t>(Register::NUM_ENUMS);

static const StackPtxInstruction instructions[] = {
    encode_constant_u32(10),
    encode_constant_u32(20),
    encode_meta_swap(StackType::U32),
    encode_ptx_instruction_add_u32,
    encode_input(static_cast<uint16_t>(Register::IN_0)),
    encode_ptx_instruction_add_u32,
    encode_return
};

static const size_t requests[] = {
    static_cast<size_t>(Register::OUT_0)
};
static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

static const size_t execution_limit = 100; 

int
main() {
    size_t workspace_bytes;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx::stack_info,
            &workspace_bytes
        )
    );

    void* workspace = malloc(workspace_bytes);

    size_t required;
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx::stack_info,
            instructions,
            registers,          num_registers,
            NULL,               0,
            requests,           num_requests,
            execution_limit,
            workspace,          workspace_bytes,
            NULL,               0,
            &required
        )
    );

    size_t capacity = required + 1;
    char* buffer = (char*)malloc(capacity);

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx::stack_info,
            instructions,
            registers,          num_registers,
            NULL,               0,
            requests,           num_requests,
            execution_limit,
            workspace,          workspace_bytes,
            buffer,             capacity,
            &required
        )
    );

    printf("%s\n", buffer);

    free(buffer);
    free(workspace);
}