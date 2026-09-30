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

#define AST_PTX_IMPLEMENTATION
#include <ast_ptx_interpreter.h>

#include <check_result_helper.h>
#include <cuda_helper.h>
#include <nvptx_helper.h>
#include <ptx_inject_helper.h>

#include <cuda.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

INCTXT(annotated_ptx, PTX_KERNEL);

enum {
    ROUTINE_MUL = 0,
    NUM_ROUTINES = 1
};

static const AstPtxInstruction mul_routine[] = {
    ast_ptx_encode_routine_arg(0u),
    ast_ptx_encode_routine_arg(1u),
    ast_ptx_encode_ptx_instruction_mul_ftz_f32,
    ast_ptx_encode_return
};

static const AstPtxInstruction* const routines[NUM_ROUTINES] = {
    mul_routine
};

static const AstPtxInstruction program[] = {
    ast_ptx_encode_input(0u),
    ast_ptx_encode_input(1u),
    ast_ptx_encode_routine(ROUTINE_MUL, 2u),
    ast_ptx_encode_constant(1.0f),
    ast_ptx_encode_ptx_instruction_add_ftz_f32,
    ast_ptx_encode_return
};

static
void
astPtxCheck(AstPtxResult result) {
    if (result != AST_PTX_SUCCESS) {
        fprintf(stderr, "astPtxCheck: %d\n", (int)result);
        ASSERT(0);
    }
}

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

static
char*
compile_ast_stub(
    const char* output_register_name,
    const char* input_x_register_name,
    const char* input_y_register_name
) {
    const char* const input_register_names[] = {
        input_x_register_name,
        input_y_register_name
    };

    size_t required = 0u;
    astPtxCheck(
        ast_ptx_compile(
            output_register_name,
            input_register_names,
            2u,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            program,
            NULL,
            0u,
            &required
        )
    );

    char* stub = (char*)malloc(required + 1u);
    ASSERT(stub != NULL);

    astPtxCheck(
        ast_ptx_compile(
            output_register_name,
            input_register_names,
            2u,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            program,
            stub,
            required + 1u,
            &required
        )
    );

    return stub;
}

static
float
run_stub_on_gpu(
    int device_compute_capability_major,
    int device_compute_capability_minor,
    CUdeviceptr d_out,
    PtxInjectHandle ptx_inject,
    const char* stub
) {
    size_t inject_func_idx = 0u;
    ptxInjectCheck(
        ptx_inject_inject_info_by_name(
            ptx_inject,
            "func",
            &inject_func_idx,
            NULL,
            NULL
        )
    );

    const char* ptx_stubs[1] = { NULL };
    ptx_stubs[inject_func_idx] = stub;

    size_t rendered_ptx_bytes = 0u;
    char* rendered_ptx =
        render_injected_ptx(
            ptx_inject,
            ptx_stubs,
            1u,
            &rendered_ptx_bytes
        );

    void* sass =
        nvptx_compile(
            device_compute_capability_major,
            device_compute_capability_minor,
            rendered_ptx,
            rendered_ptx_bytes,
            NULL,
            false
        );

    free(rendered_ptx);

    CUmodule cu_module;
    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    free(sass);

    CUfunction cu_function;
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

    float h_out = 0.0f;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );

    cuCheck( cuModuleUnload(cu_module) );
    return h_out;
}

int
main(void) {
    static const float input_values[] = {
        5.0f,
        3.0f
    };

    float interpreted = 0.0f;
    astPtxCheck(
        ast_ptx_interpret(
            program,
            routines,
            NUM_ROUTINES,
            input_values,
            2u,
            &interpreted
        )
    );

    ASSERT(fabsf(interpreted - 16.0f) <= 1.0e-5f);

    if (!has_cuda_device()) {
        fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    size_t inject_func_idx = 0u;
    ptxInjectCheck(
        ptx_inject_inject_info_by_name(
            ptx_inject,
            "func",
            &inject_func_idx,
            NULL,
            NULL
        )
    );

    const char* register_name_x = NULL;
    const char* register_name_y = NULL;
    const char* register_name_z = NULL;

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_func_idx,
            "v_x",
            NULL,
            &register_name_x,
            NULL,
            NULL,
            NULL
        )
    );

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_func_idx,
            "v_y",
            NULL,
            &register_name_y,
            NULL,
            NULL,
            NULL
        )
    );

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_func_idx,
            "v_z",
            NULL,
            &register_name_z,
            NULL,
            NULL,
            NULL
        )
    );

    char* ast_stub =
        compile_ast_stub(
            register_name_z,
            register_name_x,
            register_name_y
        );

    cuCheck( cuInit(0) );

    CUdevice cu_device;
    cuCheck( cuDeviceGet(&cu_device, 0) );

    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    get_device_capability(
        cu_device,
        &compute_capability_major,
        &compute_capability_minor
    );

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    CUdeviceptr d_out;
    cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

    const float ast_gpu_result =
        run_stub_on_gpu(
            compute_capability_major,
            compute_capability_minor,
            d_out,
            ptx_inject,
            ast_stub
        );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuCtxDestroy(cu_context) );

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    free(ast_stub);

    ASSERT(fabsf(ast_gpu_result - interpreted) <= 1.0e-5f);
    ASSERT(fabsf(ast_gpu_result - 16.0f) <= 1.0e-5f);

    return 0;
}
