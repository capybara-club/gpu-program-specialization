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

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.h>
#include <stack_ptx_default_info.h>
#include <routine_library_philox.h>

#include <check_result_helper.h>
#include <ptx_inject_helper.h>
#include <nvptx_helper.h>
#include <cuda_helper.h>
#include <cuda.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

INCTXT(annotated_ptx, PTX_KERNEL);

static const int execution_limit = 100000;

enum {
    k_iterations = 2,
    k_values_per_call = 4,
    k_total_values = k_iterations * k_values_per_call
};

static const float k_uniform_tol = 1e-6f;
static const float k_normal_tol = 1e-3f;

static
bool
has_cuda_device(void) {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

enum Register {
    ROUTINE_LIBRARY_PHILOX_REGISTER_DECL,
    REGISTER_U32_X,
    REGISTER_U32_Y,
    REGISTER_U32_Z,
    REGISTER_U32_W,
    REGISTER_UNIFORM_X,
    REGISTER_UNIFORM_Y,
    REGISTER_UNIFORM_Z,
    REGISTER_UNIFORM_W,
    REGISTER_NORMAL_X,
    REGISTER_NORMAL_Y,
    REGISTER_NORMAL_Z,
    REGISTER_NORMAL_W,
    REGISTER_NUM_ENUMS
};

static const StackPtxRegister register_template[] = {
    ROUTINE_LIBRARY_PHILOX_REGISTER_IMPL,
    [REGISTER_U32_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_U32_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_U32_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_U32_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_UNIFORM_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_UNIFORM_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_UNIFORM_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_UNIFORM_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_NORMAL_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_NORMAL_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_NORMAL_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    [REGISTER_NORMAL_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
};

static const size_t num_registers = REGISTER_NUM_ENUMS;

enum RoutineIdx {
    ROUTINE_LIBRARY_PHILOX,
    NUM_ROUTINES
};

#include <routine_library_philox_impl.h>

static const StackPtxInstruction* routines[] = {
    ROUTINE_LIBRARY_PHILOX_INITIALIZERS
};

static const size_t requests_u32[] = {
    ROUTINE_LIBRARY_PHILOX_REQUEST,
    REGISTER_U32_W,
    REGISTER_U32_Z,
    REGISTER_U32_Y,
    REGISTER_U32_X,
};

static const size_t requests_uniform[] = {
    ROUTINE_LIBRARY_PHILOX_REQUEST,
    REGISTER_UNIFORM_X,
    REGISTER_UNIFORM_Y,
    REGISTER_UNIFORM_Z,
    REGISTER_UNIFORM_W
};

static const size_t requests_normal[] = {
    ROUTINE_LIBRARY_PHILOX_REQUEST,
    REGISTER_NORMAL_X,
    REGISTER_NORMAL_Y,
    REGISTER_NORMAL_Z,
    REGISTER_NORMAL_W
};

static const StackPtxInstruction philox_u32_inputs[] = {
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_PUSH_STATE),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_CURAND),
    stack_ptx_encode_return
};

static const StackPtxInstruction philox_uniform_inputs[] = {
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_PUSH_STATE),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_CURAND_UNIFORM),
    stack_ptx_encode_return
};

static const StackPtxInstruction philox_normal_inputs[] = {
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_PUSH_STATE),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_CURAND_NORMAL),
    stack_ptx_encode_return
};

static
void
prepare_registers(
    StackPtxRegister* registers,
    PtxInjectHandle ptx_inject,
    size_t inject_idx,
    const char* const names[4],
    const size_t register_ids[4]
) {
    memcpy(registers, register_template, sizeof(register_template));

    ptxInjectCheck( ptx_inject_philox_populate_registers(ptx_inject, inject_idx, registers) );

    for (size_t i = 0; i < 4; ++i) {
        ptxInjectCheck(
            ptx_inject_variable_info_by_name(
                ptx_inject,
                inject_idx,
                names[i],
                NULL,
                &registers[register_ids[i]].name,
                NULL,
                NULL,
                NULL
            )
        );
    }
}

static
char*
compile_stack_ptx_stub(
    const StackPtxInstruction* instructions,
    const StackPtxRegister* registers,
    size_t num_regs,
    const size_t* requests,
    size_t num_requests,
    void* workspace,
    size_t workspace_size
) {
    size_t required = 0;
    size_t capacity = 0;

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            num_regs,
            routines,
            NUM_ROUTINES,
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

    capacity = required + 1;
    char* stub_buffer = (char *)malloc(capacity);
    ASSERT(stub_buffer != NULL);

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            num_regs,
            routines,
            NUM_ROUTINES,
            requests,
            num_requests,
            execution_limit,
            workspace,
            workspace_size,
            stub_buffer,
            capacity,
            &required
        )
    );

    return stub_buffer;
}

int
main() {
    if (!has_cuda_device()) {
        fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    size_t inject_idx_u32 = 0;
    size_t inject_idx_uniform = 0;
    size_t inject_idx_normal = 0;

    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "philox_u32", &inject_idx_u32, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "philox_uniform", &inject_idx_uniform, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "philox_normal", &inject_idx_normal, NULL, NULL) );

    size_t num_injects = 0;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );
    ASSERT(num_injects == 3);

    size_t stack_ptx_workspace_size = 0;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

    void* stack_ptx_workspace = malloc(stack_ptx_workspace_size);
    ASSERT(stack_ptx_workspace != NULL);

    StackPtxRegister registers[REGISTER_NUM_ENUMS];

    const char* const u32_names[] = {"u32_x", "u32_y", "u32_z", "u32_w"};
    const size_t u32_regs[] = {REGISTER_U32_X, REGISTER_U32_Y, REGISTER_U32_Z, REGISTER_U32_W};
    prepare_registers(registers, ptx_inject, inject_idx_u32, u32_names, u32_regs);
    char* stub_u32 = compile_stack_ptx_stub(
        philox_u32_inputs,
        registers,
        num_registers,
        requests_u32,
        STACK_PTX_ARRAY_NUM_ELEMS(requests_u32),
        stack_ptx_workspace,
        stack_ptx_workspace_size
    );

    const char* const uniform_names[] = {"uniform_x", "uniform_y", "uniform_z", "uniform_w"};
    const size_t uniform_regs[] = {REGISTER_UNIFORM_X, REGISTER_UNIFORM_Y, REGISTER_UNIFORM_Z, REGISTER_UNIFORM_W};
    prepare_registers(registers, ptx_inject, inject_idx_uniform, uniform_names, uniform_regs);
    char* stub_uniform = compile_stack_ptx_stub(
        philox_uniform_inputs,
        registers,
        num_registers,
        requests_uniform,
        STACK_PTX_ARRAY_NUM_ELEMS(requests_uniform),
        stack_ptx_workspace,
        stack_ptx_workspace_size
    );

    const char* const normal_names[] = {"normal_x", "normal_y", "normal_z", "normal_w"};
    const size_t normal_regs[] = {REGISTER_NORMAL_X, REGISTER_NORMAL_Y, REGISTER_NORMAL_Z, REGISTER_NORMAL_W};
    prepare_registers(registers, ptx_inject, inject_idx_normal, normal_names, normal_regs);
    char* stub_normal = compile_stack_ptx_stub(
        philox_normal_inputs,
        registers,
        num_registers,
        requests_normal,
        STACK_PTX_ARRAY_NUM_ELEMS(requests_normal),
        stack_ptx_workspace,
        stack_ptx_workspace_size
    );

    const char** stubs = (const char **)calloc(num_injects, sizeof(*stubs));
    ASSERT(stubs != NULL);
    stubs[inject_idx_u32] = stub_u32;
    stubs[inject_idx_uniform] = stub_uniform;
    stubs[inject_idx_normal] = stub_normal;

    size_t num_bytes_written = 0;
    char* rendered_ptx = render_injected_ptx(ptx_inject, stubs, num_injects, &num_bytes_written);

    free(stub_u32);
    free(stub_uniform);
    free(stub_normal);
    free(stubs);

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    cuCheck( cuDeviceGet(&cu_device, 0) );

    int device_compute_capability_major = 0;
    int device_compute_capability_minor = 0;
    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    void* sass = nvptx_compile(
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written,
        NULL,
        false
    );
    free(rendered_ptx);

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    CUmodule cu_module;
    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    free(sass);

    CUfunction cu_function;
    cuCheck( cuModuleGetFunction(&cu_function, cu_module, "kernel") );

    const size_t u32_bytes = k_total_values * sizeof(uint32_t);
    const size_t f32_bytes = k_total_values * sizeof(float);

    CUdeviceptr d_out_u32;
    CUdeviceptr d_ref_u32;
    CUdeviceptr d_out_uniform;
    CUdeviceptr d_ref_uniform;
    CUdeviceptr d_out_normal;
    CUdeviceptr d_ref_normal;

    cuCheck( cuMemAlloc(&d_out_u32, u32_bytes) );
    cuCheck( cuMemAlloc(&d_ref_u32, u32_bytes) );
    cuCheck( cuMemAlloc(&d_out_uniform, f32_bytes) );
    cuCheck( cuMemAlloc(&d_ref_uniform, f32_bytes) );
    cuCheck( cuMemAlloc(&d_out_normal, f32_bytes) );
    cuCheck( cuMemAlloc(&d_ref_normal, f32_bytes) );

    void* args[] = {
        (void*)&d_out_u32,
        (void*)&d_ref_u32,
        (void*)&d_out_uniform,
        (void*)&d_ref_uniform,
        (void*)&d_out_normal,
        (void*)&d_ref_normal
    };

    cuCheck(
        cuLaunchKernel(
            cu_function,
            1, 1, 1,
            1, 1, 1,
            0, 0,
            args,
            NULL
        )
    );

    cuCheck( cuCtxSynchronize() );

    uint32_t h_out_u32[k_total_values];
    uint32_t h_ref_u32[k_total_values];
    float h_out_uniform[k_total_values];
    float h_ref_uniform[k_total_values];
    float h_out_normal[k_total_values];
    float h_ref_normal[k_total_values];

    cuCheck( cuMemcpyDtoH(h_out_u32, d_out_u32, u32_bytes) );
    cuCheck( cuMemcpyDtoH(h_ref_u32, d_ref_u32, u32_bytes) );
    cuCheck( cuMemcpyDtoH(h_out_uniform, d_out_uniform, f32_bytes) );
    cuCheck( cuMemcpyDtoH(h_ref_uniform, d_ref_uniform, f32_bytes) );
    cuCheck( cuMemcpyDtoH(h_out_normal, d_out_normal, f32_bytes) );
    cuCheck( cuMemcpyDtoH(h_ref_normal, d_ref_normal, f32_bytes) );

    for (size_t i = 0; i < k_total_values; ++i) {
        ASSERT(h_out_u32[i] == h_ref_u32[i]);
        ASSERT(fabsf(h_out_uniform[i] - h_ref_uniform[i]) <= k_uniform_tol);
        ASSERT(fabsf(h_out_normal[i] - h_ref_normal[i]) <= k_normal_tol);
    }

    cuCheck( cuMemFree(d_out_u32) );
    cuCheck( cuMemFree(d_ref_u32) );
    cuCheck( cuMemFree(d_out_uniform) );
    cuCheck( cuMemFree(d_ref_uniform) );
    cuCheck( cuMemFree(d_out_normal) );
    cuCheck( cuMemFree(d_ref_normal) );

    cuCheck( cuModuleUnload(cu_module) );
    cuCheck( cuCtxDestroy(cu_context) );

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    return 0;
}
