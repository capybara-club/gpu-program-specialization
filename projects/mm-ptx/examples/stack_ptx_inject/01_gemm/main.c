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

#include <cuda_helper.h>
#include <nvptx_helper.h>
#include <ptx_inject_helper.h>
#include <mma_helper.h>
#include <time.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

#define STUB_BUFFER_SIZE 1000000ull

#define CLOCK_MULTIPLIER 1000000.0

static const int execution_limit = 100;

enum InjectSite {
    INJECT_MMA,
    INJECT_EPILOGUE,
    INJECT_NUM_SITES
};

typedef struct {
    const char* name;  // "multiply", "accumulate", "epilogue"
    size_t idx;        // filled at runtime via ptx_inject_inject_info_by_name
} InjectSiteInfo;

static InjectSiteInfo g_inject_sites[INJECT_NUM_SITES] = {
    [INJECT_MMA]   =    { "mma",   0 },
    [INJECT_EPILOGUE] = { "epilogue",   0 },
};

enum Register {
    REGISTER_MMA_V_A_IN,
    REGISTER_MMA_V_B_IN,
    REGISTER_MMA_V_C_MOD,
    REGISTER_EPILOGUE_V_C_IN,
    REGISTER_EPILOGUE_V_C_OUT,
    REGISTER_NUM_ENUMS
};

static StackPtxRegister registers[] = {
    [REGISTER_MMA_V_A_IN] =         { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_MMA_V_B_IN] =         { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_MMA_V_C_MOD] =        { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_EPILOGUE_V_C_IN] =    { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_EPILOGUE_V_C_OUT] =   { NULL, STACK_PTX_STACK_TYPE_F32 },
};
static const size_t num_registers = REGISTER_NUM_ENUMS;

typedef struct {
    enum InjectSite site;
    const char* var_name;   // name as seen in PTX_INJECT annotations
    enum Register reg;      // which entry in registers[] to fill
} RegisterBinding;

static const RegisterBinding g_register_bindings[] = {
    { INJECT_MMA,   "v_a",  REGISTER_MMA_V_A_IN},
    { INJECT_MMA,   "v_b",  REGISTER_MMA_V_B_IN},
    { INJECT_MMA,   "v_c",  REGISTER_MMA_V_C_MOD},

    { INJECT_EPILOGUE, "v_c_in", REGISTER_EPILOGUE_V_C_IN },
    { INJECT_EPILOGUE, "v_c_out",REGISTER_EPILOGUE_V_C_OUT},

};
static const size_t g_num_register_bindings = sizeof(g_register_bindings) / sizeof(g_register_bindings[0]);

static
void
run_custom_mma(
    int M, int N, int K,
    CUdeviceptr d_a, int lda,
    CUdeviceptr d_b, int ldb,
    CUdeviceptr d_c, int ldc,
    int device_compute_capability_major,
    int device_compute_capability_minor,
    const char* rendered_ptx,
    size_t rendered_ptx_num_bytes
) {
    CUmodule cu_module;
    CUfunction cu_function;

    double time_start = clock();
    void* sass = 
        nvptx_compile(
            device_compute_capability_major, 
            device_compute_capability_minor, 
            rendered_ptx, 
            rendered_ptx_num_bytes,
            NULL,
            false
        );
    double time_end = clock();
    printf("Compile to sass:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    free(sass);

    cuCheck( cuModuleGetFunction(&cu_function, cu_module, "gemm_nt") );

    static const unsigned int block_dim = 256;

    void* args[] = {
        (void*)&M,
        (void*)&N,
        (void*)&K,
        (void*)&d_a,
        (void*)&lda,
        (void*)&d_b,
        (void*)&ldb,
        (void*)&d_c,
        (void*)&ldc
    };

    cuCheck( 
        cuLaunchKernel(
            cu_function,
            1, 1, 1,
            block_dim, 1, 1,
            0, 0, 
            args,
            NULL
        )
    );

    cuCheck( cuCtxSynchronize() );
    cuCheck( cuModuleUnload(cu_module) );
}

int
main() {
    double time_start, time_end;
    // The cmake plumbing already used the ptxinject cli tool compiled inside the 
    // project to process kernel.cu. The cuda was then compiled by nvcc as part of
    // the cmake process as well. INCBIN added the ptx to this file as g_annotated_ptx_data.

    time_start = clock();
    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );
    time_end = clock();

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

    printf("Parse/Process Inject PTX: %6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    printf("\n");
    // Print the stats of the injects that we're found
    print_ptx_inject_info(ptx_inject);
    printf("\n");

    size_t num_injects_found;
    ptxInjectCheck(ptx_inject_num_injects(ptx_inject, &num_injects_found));
    ASSERT(num_injects_found == INJECT_NUM_SITES);

    for (size_t s = 0; s < INJECT_NUM_SITES; ++s) {
        ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, g_inject_sites[s].name, &g_inject_sites[s].idx, NULL, NULL) );
    }

    char* mma_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    char* epilogue_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);

    const char* ptx_stubs[2];

    // We now need to ask which index in the stub buffer each inject needs to be at
    size_t mma_func_idx   = g_inject_sites[INJECT_MMA].idx;
    size_t epilogue_func_idx   = g_inject_sites[INJECT_EPILOGUE].idx;
    
    // Set up the buffers to go in to the proper indicies as specified by the inject_info function calls.
    ptx_stubs[mma_func_idx] = mma_stub_buffer;
    ptx_stubs[epilogue_func_idx] = epilogue_stub_buffer;

    for (size_t i = 0; i < g_num_register_bindings; ++i) {
        const RegisterBinding* b = &g_register_bindings[i];
        size_t inject_idx = g_inject_sites[b->site].idx;

        ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_idx, b->var_name, NULL, &registers[b->reg].name, NULL, NULL, NULL) );
    }

    // We need the multiply instructions to be assigned to the "diff" register name
    const size_t mma_requests[] = { REGISTER_MMA_V_C_MOD };
    static const size_t num_mma_requests = STACK_PTX_ARRAY_NUM_ELEMS(mma_requests);

    const size_t epilogue_requests[] = { REGISTER_EPILOGUE_V_C_OUT };
    static const size_t num_epilogue_requests = STACK_PTX_ARRAY_NUM_ELEMS(epilogue_requests);

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    CUcontext cu_context;
    
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    char* rendered_ptx;
    size_t num_bytes_written;

    static const int M = 128;
    static const int N = 128;
    static const int K = 8;
    static const int lda = 128;
    static const int ldb = 128;
    static const int ldc = 128;

    static const int matrix_print_limit = 5;

    float* h_a = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* h_b = (float*)malloc((size_t)N * (size_t)K * sizeof(float));
    float* h_c = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* h_c_gold = (float*)malloc((size_t)M * (size_t)N * sizeof(float));

    for (int j = 0; j < M*K; ++j) h_a[j] = 2*(rand() / (double)(RAND_MAX)) - 1;
    for (int j = 0; j < N*K; ++j) h_b[j] = 2*(rand() / (double)(RAND_MAX)) - 1;

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_c;

    cuCheck( cuMemAlloc(&d_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_b, (size_t)N * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_c, (size_t)M * (size_t)N * sizeof(float)) );

    cuCheck( cuMemcpyHtoD(d_a, h_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemcpyHtoD(d_b, h_b, (size_t)N * (size_t)K * sizeof(float)) );

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            (StackPtxInstruction[]) {
                stack_ptx_encode_input(REGISTER_EPILOGUE_V_C_IN),
                stack_ptx_encode_return
            },
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            epilogue_requests,
            num_epilogue_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            epilogue_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    // Multiply as gemm or `dot product`
    // diff = v_a * v_b
    static const StackPtxInstruction mma_gemm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MMA_V_C_MOD),
        stack_ptx_encode_input(REGISTER_MMA_V_B_IN),
        stack_ptx_encode_input(REGISTER_MMA_V_A_IN),
        stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
        stack_ptx_encode_return
    };

    // Multiply as L1 Norm
    // diff = abs(v_a - v_b)
    static const StackPtxInstruction mma_l1_norm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MMA_V_A_IN),
        stack_ptx_encode_input(REGISTER_MMA_V_B_IN),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_ptx_instruction_abs_ftz_f32,
        stack_ptx_encode_input(REGISTER_MMA_V_C_MOD),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    // Multiply as L2 Norm
    // diff = (v_a - v_b) * (v_a - v_b)
    static const StackPtxInstruction mma_l2_norm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MMA_V_A_IN),
        stack_ptx_encode_input(REGISTER_MMA_V_B_IN),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32), // Duplicate the ast value at the top of the f32 stack
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_input(REGISTER_MMA_V_C_MOD),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };
    
    // Set up the multiply operator to be just v_a * v_b for a mma gemm
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            mma_gemm_instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            mma_requests,
            num_mma_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            mma_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 2, &num_bytes_written);
    time_end = clock();
    printf("MMA\n");
    printf("---------------------------------------------\n");
    printf("Render PTX:\t\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    
    // Run the kernel
    time_start = clock();
    run_custom_mma(
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc, 
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    time_end = clock();
    printf("Compile, Load, Run:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    free(rendered_ptx);
    
    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    // Compute a cpu version of the gemm
    gemm_gold(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    printf("\n");
    printf("MMA (Inject Kernel)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");
    printf("MMA (CPU Gold)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c_gold, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");

    // We can now easily run this also as an L1 norm kernel.
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            mma_l1_norm_instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            mma_requests,
            num_mma_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            mma_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 2, &num_bytes_written);
    time_end = clock();
    printf("L1\n");
    printf("---------------------------------------------\n");
    printf("Render PTX:\t\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    // Run the kernel
    time_start = clock();
    run_custom_mma(
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc, 
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    time_end = clock();
    printf("Compile, Load, Run:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    // Compute a cpu version of the L1 norm
    l1_gold(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    printf("\n");
    printf("L1 (Inject Kernel)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");
    printf("L1 (CPU Gold)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c_gold, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");

    // Lastly we will run this as an L2 norm kernel
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            mma_l2_norm_instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            mma_requests,
            num_mma_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            mma_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 2, &num_bytes_written);
    time_end = clock();
    printf("L2\n");
    printf("---------------------------------------------\n");
    printf("Render PTX:\t\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    
    time_start = clock();
    run_custom_mma(
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc, 
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    time_end = clock();
    printf("Compile, Load, Run:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    l2_gold(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    printf("\n");
    printf("L2 (Inject Kernel)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");
    printf("L2 (CPU Gold)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c_gold, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");

    cuCheck( cuMemFree(d_c) );
    cuCheck( cuMemFree(d_b) );
    cuCheck( cuMemFree(d_a) );

    free(h_c_gold);
    free(h_c);
    free(h_b);
    free(h_a);

    cuCheck( cuCtxDestroy(cu_context) );

    free(epilogue_stub_buffer);
    free(mma_stub_buffer);

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );
    
    return 0;
}
