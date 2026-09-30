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
#include <kermac.h>
#include <k_internal.h>

#include <nvPTXCompiler.h>

#include <kermac_stack_ptx_descriptions.h>

#include <ptx_inject.h>

#include <stack_ptx.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

INCTXT(elementwise_annotated_ptx, PTX_ELEMENTWISE);

typedef enum {
    KERMAC_ELEMENTWISE_REGISTER_ROW,
    KERMAC_ELEMENTWISE_REGISTER_COL,
    KERMAC_ELEMENTWISE_REGISTER_V,
    KERMAC_ELEMENTWISE_REGISTER_NUM_ENUMS
} KermacElementwiseRegister;

typedef enum {
    INJECT_SITE_FUNC,
    INJECT_SITE_NUM_ENUMS
} InjectSite;

typedef struct {
    const char* name;
    size_t idx;
} InjectSiteInfo;

static InjectSiteInfo g_inject_sites[INJECT_SITE_NUM_ENUMS] = {
    [INJECT_SITE_FUNC]   =       { "func",        0 },
};

typedef struct {
    InjectSite site;
    const char* var_name;
    KermacElementwiseRegister reg;
} RegisterBinding;

static const RegisterBinding g_register_bindings[] = {
    { INJECT_SITE_FUNC,      "row",   KERMAC_ELEMENTWISE_REGISTER_ROW },
    { INJECT_SITE_FUNC,      "col",   KERMAC_ELEMENTWISE_REGISTER_COL },
    { INJECT_SITE_FUNC,      "v",   KERMAC_ELEMENTWISE_REGISTER_V },
};
static const size_t g_num_register_bindings = sizeof(g_register_bindings) / sizeof(g_register_bindings[0]);

static
inline
KermacResult 
_kermac_elementwise_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacElementwise* elementwise,
    KermacMatrixPackedType packed_type,
    const StackPtxInstruction* instructions
) {
    if (!elementwise) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    elementwise->module = (CUmodule)0;
    elementwise->function = (CUfunction)0;
    elementwise->packed_type = packed_type;

    StackPtxRegister registers[] = {
        [KERMAC_ELEMENTWISE_REGISTER_ROW] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
        [KERMAC_ELEMENTWISE_REGISTER_COL] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
        [KERMAC_ELEMENTWISE_REGISTER_V] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    };
    static const size_t num_registers = KERMAC_ELEMENTWISE_REGISTER_NUM_ENUMS;

    KermacResult result = KERMAC_SUCCESS;
    PtxInjectHandle ptx_inject = NULL;

    PtxInjectResult ptx_status = ptx_inject_create(&ptx_inject, g_elementwise_annotated_ptx_data);
    if (ptx_status != PTX_INJECT_SUCCESS) {
        result = KERMAC_ERROR_PTX_INJECT;
        goto cleanup;
    }

    size_t num_injects_found;
    ptx_status = ptx_inject_num_injects(ptx_inject, &num_injects_found);
    if (ptx_status != PTX_INJECT_SUCCESS) {
        result = KERMAC_ERROR_PTX_INJECT;
        goto cleanup;
    }
    if (num_injects_found != INJECT_SITE_NUM_ENUMS) {
        result = KERMAC_ERROR_INTERNAL;
        goto cleanup;
    }

    for (size_t s = 0; s < INJECT_SITE_NUM_ENUMS; ++s) {
        ptx_status = ptx_inject_inject_info_by_name(
            ptx_inject,
            g_inject_sites[s].name,
            &g_inject_sites[s].idx,
            NULL,
            NULL
        );
        if (ptx_status != PTX_INJECT_SUCCESS) {
            result = KERMAC_ERROR_PTX_INJECT;
            goto cleanup;
        }
    }

    for (size_t i = 0; i < g_num_register_bindings; ++i) {
        const RegisterBinding* b = &g_register_bindings[i];
        size_t inject_idx = g_inject_sites[b->site].idx;

        ptx_status = ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_idx,
            b->var_name,
            NULL,
            &registers[b->reg].name,
            NULL,
            NULL,
            NULL
        );
        if (ptx_status != PTX_INJECT_SUCCESS) {
            result = KERMAC_ERROR_PTX_INJECT;
            goto cleanup;
        }
    }

    static const size_t func_requests[] = { KERMAC_ELEMENTWISE_REGISTER_V };
    static const size_t num_func_requests = sizeof(func_requests) / sizeof(*func_requests);
    static const int execution_limit = 100;

    const StackPtxInstruction* instruction_stubs[] = { instructions };
    const size_t* request_stubs[] = { func_requests };
    const size_t request_stub_sizes[] = { num_func_requests };
    size_t num_stubs = 1;

    result =
        kermac_stack_ptx_inject_compile(
            handle,
            hsa, 
            ptx_inject,
            &compiler_info, &stack_ptx_stack_info, execution_limit,
            registers, num_registers,
            instruction_stubs,
            request_stubs, request_stub_sizes,
            num_stubs,
            &elementwise->module
        );
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    CUresult cuda_result = cuModuleGetFunction(&elementwise->function, elementwise->module, "kernel");
    if (cuda_result != CUDA_SUCCESS) {
        result = KERMAC_ERROR_CUDA;
        goto cleanup;
    }
cleanup:
    if (ptx_inject != NULL) {
        (void)ptx_inject_destroy(ptx_inject);
    }
    if (result != KERMAC_SUCCESS && elementwise->module != (CUmodule)0) {
        (void)cuModuleUnload(elementwise->module);
        elementwise->module = (CUmodule)0;
        elementwise->function = (CUfunction)0;
        _KERMAC_ERROR( result );
    }
    return result;
}

// This is 0f3FB8AA3B for ex2 in ptx:
// mul.ftz.f32 	%f2, %f1, 0f3FB8AA3B;
// ex2.approx.ftz.f32 	%f3, %f2;
static const float EXP_TAB_SCALE_F32_EXPLICIT = 0x1.715476p+0;

KERMAC_PUBLIC_DEF
KermacResult 
kermac_elementwise_laplace_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacElementwise* elementwise,
    float bandwidth
) {
    const StackPtxInstruction instructions[] = {
        stack_ptx_encode_input(KERMAC_ELEMENTWISE_REGISTER_V),
        stack_ptx_encode_constant_f32(-1.0f / bandwidth),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_constant_f32(EXP_TAB_SCALE_F32_EXPLICIT),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
        stack_ptx_encode_return
    };

    return _kermac_elementwise_create(
        handle,
        hsa,
        elementwise,
        KERMAC_MATRIX_PACKED_TYPE_FULL,
        instructions
    );
}

KERMAC_PUBLIC_DEF
KermacResult 
kermac_elementwise_laplace_symm_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacElementwise* elementwise,
    KermacMatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
) {
    const StackPtxInstruction instructions[] = {
        stack_ptx_encode_input(KERMAC_ELEMENTWISE_REGISTER_V),

        // Working on raw distance
        stack_ptx_encode_constant_f32(epsilon),
        stack_ptx_encode_ptx_instruction_setp_lt_ftz_f32,
        stack_ptx_encode_constant_f32(0.0f),
        stack_ptx_encode_input(KERMAC_ELEMENTWISE_REGISTER_V),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_constant_f32(-1.0f / bandwidth),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,

        stack_ptx_encode_constant_f32(EXP_TAB_SCALE_F32_EXPLICIT),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,

        stack_ptx_encode_input(KERMAC_ELEMENTWISE_REGISTER_ROW),
        stack_ptx_encode_input(KERMAC_ELEMENTWISE_REGISTER_COL),
        stack_ptx_encode_ptx_instruction_setp_eq_u32,
        stack_ptx_encode_constant_f32(1.0 + regularizer),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_return
    };

    return _kermac_elementwise_create(
        handle, hsa,
        elementwise,
        packed_type,
        instructions
    );
}

KERMAC_PUBLIC_DEF 
KermacResult 
kermac_elementwise_destroy(
    KermacElementwise elementwise
) {
    _KERMAC_CUDA_CHECK_RET( cuModuleUnload(elementwise.module) );
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_elementwise_run(
    KermacElementwise elementwise,
    KermacTensor a,
    KermacTensor c,
    CUstream stream
) {
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (c.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (c.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (a.num_modes != 2 && a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (c.num_modes != 2 && c.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool c_is_dry = c.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && c_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !c_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    int64_t M = c.extent[0];
    int64_t N = c.extent[1];
    int64_t L = c.num_modes == 2 ? 1 : c.extent[2];
    int64_t L_a = a.num_modes == 2 ? 1 : a.extent[2];

    if (M != a.extent[0] || N != a.extent[1] || L != L_a) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    if (!is_dry) {
        void* a_ptr;
        void* c_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(c.memory, &c_ptr) );

        int triangle; // 0 -> FULL, 1 -> LOWER, 2 -> UPPER
        switch (elementwise.packed_type) {
            case KERMAC_MATRIX_PACKED_TYPE_FULL: {
                triangle = 0;
            } break;
            case KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE: {
                triangle = 1;
            } break;
            case KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE: {
                triangle = 2;
            } break;
            default:
                _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }

        void *args[] = {
            (void*)&M,
            (void*)&N,
            (void*)&a_ptr, (void*)&a.stride[1], (void*)&a.stride[2],
            (void*)&c_ptr, (void*)&c.stride[1], (void*)&c.stride[2],
            (void*)&triangle
        };

        static const unsigned int BLOCK_SIZE_X = 32;
        static const unsigned int BLOCK_SIZE_Y = 8;

        unsigned int num_blocks_M = _KERMAC_NEAREST_LARGER_MULTIPLE(M, BLOCK_SIZE_X);
        unsigned int num_blocks_N = _KERMAC_NEAREST_LARGER_MULTIPLE(N, BLOCK_SIZE_Y);

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                elementwise.function,
                num_blocks_M, num_blocks_N, L,
                BLOCK_SIZE_X, BLOCK_SIZE_Y, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}
