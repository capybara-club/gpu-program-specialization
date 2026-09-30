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
#include <stack_ptx.h>
#include <ptx_inject.h>

#define STACK_PTX_INJECT_SERIALIZE_DEBUG
#define STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#include <stack_ptx_inject_serialize.h>

#include <stack_ptx_default_info.h>
#include <stack_ptx_example_descriptions.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <check_result_helper.h>

int
main() {
    uint8_t* serialized_compiler_info_buffer = NULL;
    size_t serialized_compiler_info_buffer_size = 0;
    size_t serialized_compiler_info_buffer_used = 0;
    uint8_t* deserialized_compiler_info_buffer = NULL;
    size_t deserialized_compiler_info_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_serialize(
            &compiler_info,
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_size
        )
    );

    serialized_compiler_info_buffer = (uint8_t*)malloc(serialized_compiler_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_serialize(
            &compiler_info,
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_size
        )
    );

    StackPtxCompilerInfo* deserialized_compiler_info = NULL;

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_deserialize(
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_used,
            deserialized_compiler_info_buffer,
            deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info
        )
    );

    deserialized_compiler_info_buffer = (uint8_t*)malloc(deserialized_compiler_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_deserialize(
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_used,
            deserialized_compiler_info_buffer,
            deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info
        )
    );

    ASSERT(
        stack_ptx_compiler_info_equal(
            &compiler_info,
            deserialized_compiler_info
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_print(&compiler_info)
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_print(deserialized_compiler_info)
    );

    uint8_t* serialized_stack_info_buffer = NULL;
    size_t serialized_stack_info_buffer_size = 0;
    size_t serialized_stack_info_buffer_used = 0;
    uint8_t* deserialized_stack_info_buffer = NULL;
    size_t deserialized_stack_info_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_serialize(
            &stack_ptx_stack_info,
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_size
        )
    );

    serialized_stack_info_buffer = (uint8_t*)malloc(serialized_stack_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_serialize(
            &stack_ptx_stack_info,
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_size
        )
    );

    StackPtxStackInfo* deserialized_stack_info = NULL;

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_deserialize(
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_used,
            deserialized_stack_info_buffer,
            deserialized_stack_info_buffer_size,
            &deserialized_stack_info_buffer_size,
            &deserialized_stack_info
        )
    );

    deserialized_stack_info_buffer = (uint8_t*)malloc(deserialized_stack_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_deserialize(
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_used,
            deserialized_stack_info_buffer,
            deserialized_stack_info_buffer_size,
            &deserialized_stack_info_buffer_size,
            &deserialized_stack_info
        )
    );

    ASSERT(
        stack_ptx_stack_info_equal(
            &stack_ptx_stack_info,
            deserialized_stack_info
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_print(&stack_ptx_stack_info)
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_print(deserialized_stack_info)
    );

    uint8_t* serialized_registers_buffer = NULL;
    size_t serialized_registers_buffer_size = 0;
    size_t serialized_registers_buffer_used = 0;
    uint8_t* deserialized_registers_buffer = NULL;
    size_t deserialized_registers_buffer_size = 0;

    StackPtxRegister registers[] = {
        { .name = "dog", .stack_idx = 1 },
        { .name = "cat", .stack_idx = 2 },
    };
    size_t num_registers = STACK_PTX_ARRAY_NUM_ELEMS(registers);

    StackPtxRegister* deserialized_registers = NULL;
    size_t deserialized_num_registers;

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_serialize(
            registers,
            num_registers,
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_size
        )
    );

    serialized_registers_buffer = malloc(serialized_registers_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_serialize(
            registers,
            num_registers,
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_deserialize(
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_used,
            deserialized_registers_buffer,
            deserialized_registers_buffer_size,
            &deserialized_registers_buffer_size,
            &deserialized_registers,
            &deserialized_num_registers
        )
    );

    deserialized_registers_buffer = (uint8_t*)malloc(deserialized_registers_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_deserialize(
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_used,
            deserialized_registers_buffer,
            deserialized_registers_buffer_size,
            &deserialized_registers_buffer_size,
            &deserialized_registers,
            &deserialized_num_registers
        )
    );

    ASSERT(
        stack_ptx_registers_equal(
            registers,
            num_registers,
            deserialized_registers,
            deserialized_num_registers
        )
    );

    uint8_t* serialized_requests_buffer = NULL;
    size_t serialized_requests_buffer_size = 0;
    size_t serialized_requests_buffer_used = 0;
    uint8_t* deserialized_requests_buffer = NULL;
    size_t deserialized_requests_buffer_size = 0;
    
    size_t requests_0[] = {
        10, 13
    };
    size_t num_requests_0 = STACK_PTX_ARRAY_NUM_ELEMS(requests_0);

    size_t requests_1[] = {
        22, 19, 18
    };
    size_t num_requests_1 = STACK_PTX_ARRAY_NUM_ELEMS(requests_1);

    const size_t* request_stubs[] = {
        requests_0,
        requests_1
    };
    size_t num_request_stubs = STACK_PTX_ARRAY_NUM_ELEMS(request_stubs);
    size_t request_stubs_sizes[] = {
        num_requests_0,
        num_requests_1
    };

    size_t** deserialized_request_stubs = NULL;
    size_t* deserialized_request_stubs_sizes = NULL;
    size_t deserialized_num_request_stubs;

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_serialize(
            request_stubs,
            request_stubs_sizes,
            num_request_stubs,
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_size
        )
    );

    serialized_requests_buffer = malloc(serialized_requests_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_serialize(
            request_stubs,
            request_stubs_sizes,
            num_request_stubs,
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_deserialize(
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_used,
            deserialized_requests_buffer,
            deserialized_requests_buffer_size,
            &deserialized_requests_buffer_size,
            &deserialized_request_stubs,
            &deserialized_request_stubs_sizes,
            &deserialized_num_request_stubs
        )
    );

    deserialized_requests_buffer = (uint8_t*)malloc(deserialized_requests_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_deserialize(
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_used,
            deserialized_requests_buffer,
            deserialized_requests_buffer_size,
            &deserialized_requests_buffer_size,
            &deserialized_request_stubs,
            &deserialized_request_stubs_sizes,
            &deserialized_num_request_stubs
        )
    );

    ASSERT(
        stack_ptx_requests_equal(
            request_stubs,
            request_stubs_sizes,
            num_request_stubs,
            (const size_t**)deserialized_request_stubs,
            deserialized_request_stubs_sizes,
            deserialized_num_request_stubs
        )
    );

    const StackPtxInstruction instruction_stub_0[] = {
        stack_ptx_encode_constant_u32(1.0),
        stack_ptx_encode_constant_u32(2.0),
        stack_ptx_encode_ptx_instruction_add_u32,
        stack_ptx_encode_return
    };

    const StackPtxInstruction instruction_stub_1[] = {
        stack_ptx_encode_constant_f32(1.0),
        stack_ptx_encode_constant_f32(2.0),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    const StackPtxInstruction instruction_stub_2[] = {
        stack_ptx_encode_constant_f32(10.0f),
        stack_ptx_encode_return
    };

    const StackPtxInstruction* instruction_stubs[] = {
        instruction_stub_0,
        instruction_stub_1,
        instruction_stub_2
    };

    size_t num_instruction_stubs = STACK_PTX_ARRAY_NUM_ELEMS(instruction_stubs);

    uint8_t* serialized_instructions_buffer = NULL;
    size_t serialized_instructions_buffer_size = 0;
    size_t serialized_instructions_buffer_used = 0;
    uint8_t* deserialized_instructions_buffer = NULL;
    size_t deserialized_instructions_buffer_size = 0;

    StackPtxInstruction** deserialized_instruction_stubs = NULL;
    size_t deserialized_num_instruction_stubs;

    stackPtxInjectSerializeCheck(
        stack_ptx_instructions_serialize(
            instruction_stubs,
            num_instruction_stubs,
            serialized_instructions_buffer,
            serialized_instructions_buffer_size,
            &serialized_instructions_buffer_size
        )
    );

    serialized_instructions_buffer = malloc(serialized_instructions_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_instructions_serialize(
            instruction_stubs,
            num_instruction_stubs,
            serialized_instructions_buffer,
            serialized_instructions_buffer_size,
            &serialized_instructions_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_instructions_deserialize(
            serialized_instructions_buffer,
            serialized_instructions_buffer_size,
            &serialized_instructions_buffer_used,
            deserialized_instructions_buffer,
            deserialized_instructions_buffer_size,
            &deserialized_instructions_buffer_size,
            &deserialized_instruction_stubs,
            &deserialized_num_instruction_stubs
        )
    );

    deserialized_instructions_buffer = (uint8_t*)malloc(deserialized_instructions_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_instructions_deserialize(
            serialized_instructions_buffer,
            serialized_instructions_buffer_size,
            &serialized_instructions_buffer_used,
            deserialized_instructions_buffer,
            deserialized_instructions_buffer_size,
            &deserialized_instructions_buffer_size,
            &deserialized_instruction_stubs,
            &deserialized_num_instruction_stubs
        )
    );

    ASSERT(
        stack_ptx_instructions_equal(
            instruction_stubs,
            num_instruction_stubs,
            (const StackPtxInstruction**)deserialized_instruction_stubs,
            deserialized_num_instruction_stubs
        )
    );

    const char* ptx = "something\n";
    uint8_t* serialized_ptx_buffer = NULL;
    size_t serialized_ptx_buffer_size = 0;
    size_t serialized_ptx_buffer_used = 0;
    uint8_t* deserialized_ptx_buffer = NULL;
    size_t deserialized_ptx_buffer_size = 0;

    char* deserialized_ptx = NULL;

    stackPtxInjectSerializeCheck(
        ptx_inject_ptx_serialize(
            ptx,
            serialized_ptx_buffer,
            serialized_ptx_buffer_size,
            &serialized_ptx_buffer_size
        )
    );

    serialized_ptx_buffer = malloc(serialized_ptx_buffer_size);

    stackPtxInjectSerializeCheck(
        ptx_inject_ptx_serialize(
            ptx,
            serialized_ptx_buffer,
            serialized_ptx_buffer_size,
            &serialized_ptx_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        ptx_inject_ptx_deserialize(
            serialized_ptx_buffer,
            serialized_ptx_buffer_size,
            &serialized_ptx_buffer_used,
            deserialized_ptx_buffer,
            deserialized_ptx_buffer_size,
            &deserialized_ptx_buffer_size,
            &deserialized_ptx
        )
    );

    deserialized_ptx_buffer = (uint8_t*)malloc(deserialized_ptx_buffer_size);

    stackPtxInjectSerializeCheck(
        ptx_inject_ptx_deserialize(
            serialized_ptx_buffer,
            serialized_ptx_buffer_size,
            &serialized_ptx_buffer_used,
            deserialized_ptx_buffer,
            deserialized_ptx_buffer_size,
            &deserialized_ptx_buffer_size,
            &deserialized_ptx
        )
    );

    ASSERT(
        ptx_inject_ptx_equal(
            ptx,
            deserialized_ptx
        )
    );

    StackPtxExtraInfo extra = { .device_capability_major = 8, .device_capability_minor = 9, .execution_limit = 100 };
    uint8_t* serialized_extra_buffer = NULL;
    size_t serialized_extra_buffer_size = 0;
    size_t serialized_extra_buffer_used = 0;
    uint8_t* deserialized_extra_buffer = NULL;
    size_t deserialized_extra_buffer_size = 0;

    StackPtxExtraInfo* deserialized_extra = NULL;

    stackPtxInjectSerializeCheck(
        stack_ptx_extra_serialize(
            &extra,
            serialized_extra_buffer,
            serialized_extra_buffer_size,
            &serialized_extra_buffer_size
        )
    );

    serialized_extra_buffer = malloc(serialized_extra_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_extra_serialize(
            &extra,
            serialized_extra_buffer,
            serialized_extra_buffer_size,
            &serialized_extra_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_extra_deserialize(
            serialized_extra_buffer,
            serialized_extra_buffer_size,
            &serialized_extra_buffer_used,
            deserialized_extra_buffer,
            deserialized_extra_buffer_size,
            &deserialized_extra_buffer_size,
            &deserialized_extra
        )
    );

    deserialized_extra_buffer = (uint8_t*)malloc(deserialized_extra_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_extra_deserialize(
            serialized_extra_buffer,
            serialized_extra_buffer_size,
            &serialized_extra_buffer_used,
            deserialized_extra_buffer,
            deserialized_extra_buffer_size,
            &deserialized_extra_buffer_size,
            &deserialized_extra
        )
    );

    ASSERT(
        stack_ptx_extra_equal(
            &extra,
            deserialized_extra
        )
    );

}