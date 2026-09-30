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

#include <ptx_inject_helper.h>

#include <cuda.h>
#include <cuda_helper.h>
#include <nvptx_helper.h>

#include <check_result_helper.h>
#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

#define STUB_BUFFER_SIZE 1000000ull

int
main() {
    printf("Annotated PTX:\n"
        "---------------------------------------------\n"
        "%.*s"
        "---------------------------------------------\n\n",
         g_annotated_ptx_size, g_annotated_ptx_data
    );

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    print_ptx_inject_info(ptx_inject);

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    const char* register_name_x;
    const char* register_name_y;
    const char* register_name_z;
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "v_x", NULL, &register_name_x, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "v_y", NULL, &register_name_y, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "v_z", NULL, &register_name_z, NULL, NULL, NULL) );

     // We will inject a simple add instruction that just does z = y + x;
    char *stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    snprintf(stub_buffer, STUB_BUFFER_SIZE, 
        "\tadd.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
        register_name_x,
        register_name_y,
        register_name_z
    );

    // If there are more than one injects in the ptx, then we need to know which order to send the
    // stubs to 'ptx_inject_render_ptx. In this case because there is only one, 'func' will
    // always be at index 0.

    const char* ptx_stubs[1];
    ptx_stubs[inject_func_idx] = stub_buffer;

    size_t num_bytes_written;
    char* rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 1, &num_bytes_written);

    // We should now see the add instruction inside the ptx.
    printf(
        "Rendered ptx:\n"
        "---------------------------------------------\n"
        "%s\n"
        "---------------------------------------------\n\n",
        rendered_ptx
    );

    // We can now destroy the PtxInjectHandle
    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    cuCheck( cuInit(0) );
    CUdevice device;
    
    cuCheck( cuDeviceGet(&device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    // We can now compile this ptx to sass
    void* sass = nvptx_compile(device_compute_capability_major, device_compute_capability_minor, rendered_ptx, num_bytes_written, NULL, false);

    // Free rendered_ptx buffer
    free(rendered_ptx);

    // Now let's run this kernel!
    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, device) );

    CUdeviceptr d_out;
    cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

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

    float h_out;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );

    // For this given example, the result should be 8
    printf("Result (should be 8): %f\n", h_out);

    cuCheck( cuModuleUnload(cu_module) );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuCtxDestroy(cu_context) );
}
