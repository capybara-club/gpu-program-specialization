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

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#define STACK_PTX_INJECT_SERIALIZE_DEBUG
#define STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#include <stack_ptx_inject_serialize.h>

#include <stack_ptx_example_descriptions.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <check_result_helper.h>
#include <cuda.h>
#include <cuda_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

INCTXT(annotated_ptx, PTX_KERNEL);

static const StackPtxCompilerInfo stack_ptx_compiler_info = {
	.max_ast_size = 100,
	.max_ast_to_visit_stack_depth = 20,
	.max_frame_depth = 4,
	.stack_size = 128,
	.store_size = 16
};

int
main() {
    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    enum Register {
        REGISTER_X,
        REGISTER_Y,
        REGISTER_Z,
        REGISTER_NUM_ENUMS
    };

    StackPtxRegister registers[] = {
        [REGISTER_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
        [REGISTER_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
        [REGISTER_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    };
    static const size_t num_registers = REGISTER_NUM_ENUMS;

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "v_x", NULL, &registers[REGISTER_X].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "v_y", NULL, &registers[REGISTER_Y].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "v_z", NULL, &registers[REGISTER_Z].name, NULL, NULL, NULL) );

    static const size_t requests[] = {
        REGISTER_Z
    };

    static const size_t* request_stubs[]      = { requests };
    static const size_t  request_stub_sizes[] = { STACK_PTX_ARRAY_NUM_ELEMS(requests) };
    static const size_t  num_request_stubs    = { STACK_PTX_ARRAY_NUM_ELEMS(request_stubs) };

    static const StackPtxInstruction add_inputs[] = {
        stack_ptx_encode_input(REGISTER_X),
        stack_ptx_encode_input(REGISTER_Y),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction* instruction_stubs[] = { add_inputs };
    static const size_t num_instruction_stubs = { STACK_PTX_ARRAY_NUM_ELEMS(instruction_stubs) };

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    StackPtxExtraInfo extra = {
        .device_capability_major = device_compute_capability_major,
        .device_capability_minor = device_compute_capability_minor,
        .execution_limit = 100
    };

    StackPtxInjectCompilerStateSerialize compiler_state_serialize = {
        .annotated_ptx = g_annotated_ptx_data,
        .compiler_info = &stack_ptx_compiler_info,
        .stack_info = &stack_ptx_stack_info,
        .extra = &extra,
        .registers = registers,
        .num_registers = num_registers,
        .request_stubs = request_stubs,
        .request_stub_sizes = request_stub_sizes,
        .num_request_stubs = num_request_stubs
    };

    uint8_t* serialized_buffer = NULL;
    size_t serialized_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_serialize(
            &compiler_state_serialize,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        )
    );

    serialized_buffer = malloc(serialized_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_serialize(
            &compiler_state_serialize,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        )
    );

    StackPtxInjectCompilerStateDeserialize* compiler_state_deserialize;

    size_t serialized_buffer_used;
    void* deserialized_buffer = NULL;
    size_t deserialized_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_deserialize(
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_used,
            deserialized_buffer,
            deserialized_buffer_size,
            &deserialized_buffer_size,
            &compiler_state_deserialize
        )
    );

    deserialized_buffer = malloc(deserialized_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_deserialize(
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_used,
            deserialized_buffer,
            deserialized_buffer_size,
            &deserialized_buffer_size,
            &compiler_state_deserialize
        )
    );

    ASSERT(
        stack_ptx_inject_compiler_state_equal(
            &compiler_state_serialize,
            compiler_state_deserialize
        )
    );

    free(serialized_buffer);
    free(deserialized_buffer);
}