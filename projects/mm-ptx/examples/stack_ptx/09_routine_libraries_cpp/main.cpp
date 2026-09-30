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

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.hpp>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>

#include "routine_library_u32.hpp"
#include "routine_library_f32.hpp"

// We need the routine library enum entries before defining the routines.
enum RoutineIdx {
    ROUTINE_LIBRARY_CPP_U32,
    ROUTINE_LIBRARY_CPP_F32,
    ROUTINE_0,
    ROUTINE_1,
    ROUTINE_2,
    NUM_ROUTINES
};

#include "routine_library_u32_impl.hpp"
#include "routine_library_f32_impl.hpp"

static const StackPtxInstruction routine_0[] = {
    stack_ptx::encode_constant_u32(1023),
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_1[] = {
    stack_ptx::encode_constant_u32(1024),
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_2[] = {
    stack_ptx::encode_constant_u32(1025),
    stack_ptx::encode_routine(ROUTINE_1),
    stack_ptx::encode_return
};

static const StackPtxInstruction* routines[] = {
    ROUTINE_LIBRARY_CPP_U32_INITIALIZERS,
    ROUTINE_LIBRARY_CPP_F32_INITIALIZERS,
    routine_0,
    routine_1,
    routine_2
};

enum class Register {
    Input0,
    Input1,
    Output,
    NumEnums
};

static const StackPtxRegister registers[] = {
    {"input_register_0", static_cast<size_t>(stack_ptx::StackType::U32)},
    {"input_register_1", static_cast<size_t>(stack_ptx::StackType::U32)},
    {"output_register", static_cast<size_t>(stack_ptx::StackType::U32)}
};

static const size_t num_registers = static_cast<size_t>(Register::NumEnums);

static const StackPtxInstruction instructions[] = {
    stack_ptx::encode_constant_u32(1),
    stack_ptx::encode_constant_u32(2),
    stack_ptx::encode_ptx_instruction_add_u32,
    stack_ptx::encode_constant_u32(3),
    stack_ptx::encode_ptx_instruction_add_u32,
    stack_ptx::encode_input(static_cast<uint16_t>(Register::Input0)),
    stack_ptx::encode_ptx_instruction_add_u32,
    stack_ptx::encode_input(static_cast<uint16_t>(Register::Input1)),
    stack_ptx::encode_routine(ROUTINE_2),
    stack_ptx::encode_routine(ROUTINE_LIBRARY_CPP_U32_FUNC_0),
    stack_ptx::encode_ptx_instruction_mul_lo_u32,
    stack_ptx::encode_return
};

static const size_t requests[] = { static_cast<size_t>(Register::Output) };
static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

static const int execution_limit = 100;

int
main() {
    size_t stack_ptx_workspace_size = 0;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx::stack_info,
            &stack_ptx_workspace_size
        )
    );

    void* stack_ptx_workspace = malloc(stack_ptx_workspace_size);
    ASSERT(stack_ptx_workspace != NULL);

    size_t required = 0;
    size_t capacity = 0;

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx::stack_info,
            instructions,
            registers,
            num_registers,
            routines,
            NUM_ROUTINES,
            requests,
            num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            NULL,
            capacity,
            &required
        )
    );

    capacity = required + 1;
    char* buffer = (char*)malloc(capacity);
    ASSERT(buffer != NULL);

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx::stack_info,
            instructions,
            registers,
            num_registers,
            routines,
            NUM_ROUTINES,
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

    printf("%s\n", buffer);

    free(buffer);
    free(stack_ptx_workspace);

    return 0;
}
