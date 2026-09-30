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

#include <check_result_helper.h>
#include <cuda.h>
#include <ptx_inject_helper.h>
#include <nvptx_helper.h>
#include <cuda_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

static const int execution_limit = 100;

static
float
run_stack_ptx_instructions(
    int device_compute_capability_major,
    int device_compute_capability_minor,
    CUdeviceptr d_out,
    PtxInjectHandle ptx_inject,
    const StackPtxRegister* registers,
    size_t num_registers,
    const size_t* requests,
    size_t num_requests,
    const StackPtxInstruction* instructions,
    void* workspace,
    size_t workspace_size
) {
    size_t required = 0;
    size_t capacity = 0;
    
    // Get the size of the buffer that will store the generated ptx
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            num_registers,
            NULL, 0,
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

    // Account for null terminator
    capacity = required + 1;
    char* stub_buffer = (char *)malloc(capacity);

    // Run again to store the generated ptx to the buffer
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            num_registers,
            NULL, 0,
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

    printf("buffer:\n%s\n", stub_buffer);

    // If there are more than one injects in the ptx, then we need to know which order to send the
    // stubs to 'ptx_inject_render_ptx. In this case because there is only one, 'func' will
    // always be at index 0.

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    const char* ptx_stubs[1];
    ptx_stubs[inject_func_idx] = stub_buffer;

    // We'll use the local helper this time
    size_t num_bytes_written;
    char* rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 1, &num_bytes_written);

    // We should now see the add instruction inside the ptx.
    printf(
        "---------------------------------------------\n"
        "%s"
        "---------------------------------------------\n",
        rendered_ptx
    );
    
    // We can now compile this ptx to sass
    void* sass = nvptx_compile(device_compute_capability_major, device_compute_capability_minor, rendered_ptx, num_bytes_written, NULL, false);

    // Free rendered_ptx buffer
    free(rendered_ptx);

    // Now let's run this kernel!

    CUmodule cu_module;
    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    // We can free the sass
    free(sass);

    CUfunction cu_function;
    // Because we added 'extern "C"' "kernel" works as a name unmangled
    cuCheck( cuModuleGetFunction(&cu_function, cu_module, "kernel") );

    void* args[] = {
        (void*)&d_out
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
    cuCheck( cuModuleUnload(cu_module) );

    float h_out;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );

    return h_out;
}

int
main() {
    printf("\nAnnotated PTX\n"
        "---------------------------------------------\n"
        "%.*s"
        "---------------------------------------------\n\n",
         g_annotated_ptx_size, g_annotated_ptx_data
    );

    // The cmake plumbing already used the ptxinject cli tool compiled inside the 
    // project to process kernel.cu. The cuda was then compiled by nvcc as part of
    // the cmake process as well. INCBIN added the ptx to this file as g_annotated_ptx_data.

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    enum Register {
        REGISTER_X_Y,
        REGISTER_Z,
        REGISTER_W,
        REGISTER_NUM_ENUMS
    };

    StackPtxRegister registers[] = {
        [REGISTER_X_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F16X2},
        [REGISTER_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F16},
        [REGISTER_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    };
    static const size_t num_registers = REGISTER_NUM_ENUMS;

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "x_y", NULL, &registers[REGISTER_X_Y].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "z", NULL, &registers[REGISTER_Z].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "w", NULL, &registers[REGISTER_W].name, NULL, NULL, NULL) );

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

    static const size_t requests[] = {
        REGISTER_W
    };
    static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

    static const StackPtxInstruction add_inputs[] = {
        stack_ptx_encode_input(REGISTER_Z),
        stack_ptx_encode_input(REGISTER_X_Y),
        stack_ptx_encode_ptx_instruction_f16x2_to_v2_f16,
        stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F16),
        stack_ptx_encode_ptx_instruction_f16_to_f32,
        stack_ptx_encode_ptx_instruction_f16_to_f32,
        stack_ptx_encode_ptx_instruction_f16_to_f32,
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        // stack_ptx_encode_input(REGISTER_Y),
        // stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    CUdeviceptr d_out;
    cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

    printf("Inject (add_inputs) PTX\n");

    float add_result = 
        run_stack_ptx_instructions(
            device_compute_capability_major,
            device_compute_capability_minor,
            d_out,
            ptx_inject,
            registers, num_registers,
            requests,
            num_requests,
            add_inputs,
            stack_ptx_workspace,
            stack_ptx_workspace_size
        );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuCtxDestroy(cu_context) );

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    printf("Inject (add_inputs) result (should be 12):\t%f\n", add_result);
}
