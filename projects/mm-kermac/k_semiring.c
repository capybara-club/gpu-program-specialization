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

INCTXT(semiring_nt_ptx, KERMAC_SEMIRING_NT_PTX);
INCTXT(semiring_nt_lower_ptx, KERMAC_SEMIRING_NT_LOWER_PTX);
INCTXT(semiring_nt_upper_ptx, KERMAC_SEMIRING_NT_UPPER_PTX);

typedef enum {
    KERMAC_SEMIRING_REGISTER_MMA_A,
    KERMAC_SEMIRING_REGISTER_MMA_B,
    KERMAC_SEMIRING_REGISTER_MMA_HYPER_0,
    KERMAC_SEMIRING_REGISTER_MMA_HYPER_1,
    KERMAC_SEMIRING_REGISTER_MMA_HYPER_2,
    KERMAC_SEMIRING_REGISTER_MMA_HYPER_3,
    KERMAC_SEMIRING_REGISTER_MMA_C,
    KERMAC_SEMIRING_REGISTER_EPILOGUE_IS_DIAG,
    KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_0,
    KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_1,
    KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_2,
    KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_3,
    KERMAC_SEMIRING_REGISTER_EPILOGUE_E,
    KERMAC_SEMIRING_REGISTER_NUM_ENUMS
} KermacSemiringRegister;

typedef enum {
    INJECT_SITE_MMA,
    INJECT_SITE_EPILOGUE,
    INJECT_SITE_NUM_ENUMS
} InjectSite;

typedef struct {
    const char* name;
    size_t idx;
} InjectSiteInfo;

static InjectSiteInfo g_inject_sites[INJECT_SITE_NUM_ENUMS] = {
    [INJECT_SITE_MMA]   =       { "mma",        0 },
    [INJECT_SITE_EPILOGUE]   =  { "epilogue",   0 },
};

typedef struct {
    InjectSite site;
    const char* var_name;
    KermacSemiringRegister reg;
} RegisterBinding;

static const RegisterBinding g_register_bindings[] = {
    // multiply
    { INJECT_SITE_MMA,      "a",   KERMAC_SEMIRING_REGISTER_MMA_A },
    { INJECT_SITE_MMA,      "b",   KERMAC_SEMIRING_REGISTER_MMA_B },
    { INJECT_SITE_MMA,      "h_0",   KERMAC_SEMIRING_REGISTER_MMA_HYPER_0 },
    { INJECT_SITE_MMA,      "h_1",   KERMAC_SEMIRING_REGISTER_MMA_HYPER_1 },
    { INJECT_SITE_MMA,      "h_2",   KERMAC_SEMIRING_REGISTER_MMA_HYPER_2 },
    { INJECT_SITE_MMA,      "h_3",   KERMAC_SEMIRING_REGISTER_MMA_HYPER_3 },
    { INJECT_SITE_MMA,      "c",   KERMAC_SEMIRING_REGISTER_MMA_C },

    // epilogue
    { INJECT_SITE_EPILOGUE, "is_diag", KERMAC_SEMIRING_REGISTER_EPILOGUE_IS_DIAG },
    { INJECT_SITE_EPILOGUE, "h_0", KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_0 },
    { INJECT_SITE_EPILOGUE, "h_1", KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_1 },
    { INJECT_SITE_EPILOGUE, "h_2", KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_2 },
    { INJECT_SITE_EPILOGUE, "h_3", KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_3 },
    { INJECT_SITE_EPILOGUE, "e",    KERMAC_SEMIRING_REGISTER_EPILOGUE_E },
};
static const size_t g_num_register_bindings = sizeof(g_register_bindings) / sizeof(g_register_bindings[0]);

static
inline
KermacResult 
_kermac_semiring_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    const StackPtxInstruction* mma_instructions,
    const StackPtxInstruction* epilogue_instructions
) {
    if (!semiring) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    semiring->module = (CUmodule)0;
    semiring->function = (CUfunction)0;
    semiring->packed_type = packed_type;

    StackPtxRegister registers[] = {
        [KERMAC_SEMIRING_REGISTER_MMA_A]                        = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_MMA_B]                        = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_MMA_HYPER_0]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_MMA_HYPER_1]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_MMA_HYPER_2]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_MMA_HYPER_3]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_MMA_C]                        = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_EPILOGUE_IS_DIAG]             = { NULL, STACK_PTX_STACK_TYPE_U32 },
        [KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_0]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_1]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_2]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_EPILOGUE_HYPER_3]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_REGISTER_EPILOGUE_E]                   = { NULL, STACK_PTX_STACK_TYPE_F32 },
    };
    static const size_t num_registers = KERMAC_SEMIRING_REGISTER_NUM_ENUMS;

    const char* annotated_ptx = NULL;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        annotated_ptx = g_semiring_nt_lower_ptx_data;
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        annotated_ptx = g_semiring_nt_upper_ptx_data;
    } else {
        annotated_ptx = g_semiring_nt_ptx_data;
    }
    KermacResult result = KERMAC_SUCCESS;
    PtxInjectHandle ptx_inject = NULL;

    PtxInjectResult ptx_status = ptx_inject_create(&ptx_inject, annotated_ptx);
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

    static const size_t mma_requests[] = { KERMAC_SEMIRING_REGISTER_MMA_C };
    static const size_t num_mma_requests = sizeof(mma_requests) / sizeof(*mma_requests);

    static const size_t epilogue_requests[] = { KERMAC_SEMIRING_REGISTER_MMA_C };
    static const size_t num_epilogue_requests = sizeof(epilogue_requests) / sizeof(*epilogue_requests);
    
    static const int execution_limit = 100;
    
    const StackPtxInstruction* instruction_stubs[] = { mma_instructions, epilogue_instructions };
    const size_t* request_stubs[] = { mma_requests, epilogue_requests };
    const size_t request_stub_sizes[] = { num_mma_requests, num_epilogue_requests };
    size_t num_stubs = 2;

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
            &semiring->module
        );
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    CUresult cuda_result = cuModuleGetFunction(&semiring->function, semiring->module, "cute_semiring");
    if (cuda_result != CUDA_SUCCESS) {
        result = KERMAC_ERROR_CUDA;
        goto cleanup;
    }

cleanup:
    if (ptx_inject != NULL) {
        (void)ptx_inject_destroy(ptx_inject);
    }
    if (result != KERMAC_SUCCESS && semiring->module != (CUmodule)0) {
        (void)cuModuleUnload(semiring->module);
        semiring->module = (CUmodule)0;
        semiring->function = (CUfunction)0;
    }
    return result;
}

KERMAC_PUBLIC_DEF 
KermacResult 
kermac_semiring_destroy(
    KermacSemiring semiring
) {
    _KERMAC_CUDA_CHECK_RET( cuModuleUnload(semiring.module) );
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEC
KermacResult
kermac_semiring_run(
    KermacSemiring semiring,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
) {
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (c.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (c.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (a.num_modes != 2 && a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (b.num_modes != 2 && b.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (c.num_modes != 2 && c.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool c_is_dry = c.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry && c_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !c_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    int64_t M = c.extent[0];
    int64_t N = c.extent[1];
    int64_t L = c.num_modes == 2 ? 1 : c.extent[2];
    int64_t L_a = a.num_modes == 2 ? 1 : a.extent[2];
    int64_t L_b = b.num_modes == 2 ? 1 : b.extent[2];

    int64_t K = a.extent[1];

    if (M != a.extent[0] || K != a.extent[1] || L != L_a) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (N != b.extent[0] || K != b.extent[1] || L != L_b) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    if (semiring.packed_type != KERMAC_MATRIX_PACKED_TYPE_FULL) {
        if (M != N) {
            _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
        }
    }
    
    if (!is_dry) {
        void* a_ptr;
        void* b_ptr;
        void* c_ptr;

        void* h0_ptr = NULL;
        void* h1_ptr = NULL;
        void* h2_ptr = NULL;
        void* h3_ptr = NULL;

        int64_t batch_stride_h0 = 0;
        int64_t batch_stride_h1 = 0;
        int64_t batch_stride_h2 = 0;
        int64_t batch_stride_h3 = 0;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b.memory, &b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(c.memory, &c_ptr) );

        void *args[] = {
            (void*)&M,
            (void*)&N,
            (void*)&K,
            (void*)&L,
            (void*)&a_ptr, (void*)&a.stride[1], (void*)&a.stride[2],
            (void*)&b_ptr, (void*)&b.stride[1], (void*)&b.stride[2],
            (void*)&c_ptr, (void*)&c.stride[1], (void*)&c.stride[2],
            (void*)&h0_ptr, (void*)&batch_stride_h0,
            (void*)&h1_ptr, (void*)&batch_stride_h1,
            (void*)&h2_ptr, (void*)&batch_stride_h2,
            (void*)&h3_ptr, (void*)&batch_stride_h3,
        };

        int num_blocks_M;
        int num_blocks_N;

        if (semiring.packed_type == KERMAC_MATRIX_PACKED_TYPE_FULL) {
            num_blocks_M = _KERMAC_NEAREST_LARGER_MULTIPLE(M, 128);
            num_blocks_N = _KERMAC_NEAREST_LARGER_MULTIPLE(N, 128);
        } else {
            const int64_t num_blocks_side = _KERMAC_NEAREST_LARGER_MULTIPLE(M, 128);
            const int64_t num_blocks_syrk = num_blocks_side * (num_blocks_side + 1ll) / 2ll;

            num_blocks_M = (int)num_blocks_syrk;
            num_blocks_N = 1;
        }

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                semiring.function,
                num_blocks_M, num_blocks_N, L,
                256, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF KermacResult kermac_semiring_gemm_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring
) {
    static const StackPtxInstruction mma_mma_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_MMA_C),
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_MMA_A),
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_MMA_B),
        stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction epilogue_mma_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_EPILOGUE_E),
        stack_ptx_encode_return
    };

    return 
        _kermac_semiring_create(
            handle,
            hsa,
            semiring,
            KERMAC_MATRIX_PACKED_TYPE_FULL,
            mma_mma_instructions,
            epilogue_mma_instructions
        );
}

static const StackPtxInstruction mma_l2_norm_instructions[] = {
    stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_MMA_C),
    stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_MMA_A),
    stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_MMA_B),
    stack_ptx_encode_ptx_instruction_sub_ftz_f32,
    stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_return
};

KERMAC_PUBLIC_DEF
KermacResult 
kermac_semiring_norm_l2_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring
) {
    static const StackPtxInstruction epilogue_norm_l2_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_EPILOGUE_E),
        stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
        stack_ptx_encode_return
    };

    return 
        _kermac_semiring_create(
            handle,
            hsa,
            semiring,
            KERMAC_MATRIX_PACKED_TYPE_FULL,
            mma_l2_norm_instructions,
            epilogue_norm_l2_instructions
        );
}

KERMAC_PUBLIC_DEF
KermacResult 
kermac_semiring_norm_l2_symm_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type
) {
    static const StackPtxInstruction epilogue_norm_l2_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_EPILOGUE_E),
        stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
        stack_ptx_encode_return
    };

    return 
        _kermac_semiring_create(
            handle,
            hsa,
            semiring,
            packed_type,
            mma_l2_norm_instructions,
            epilogue_norm_l2_instructions
        );
}

KERMAC_PUBLIC_DEF 
KermacResult 
kermac_semiring_laplace_l2_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    float bandwidth
) {
    const StackPtxInstruction epilogue_l2_norm_laplace_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_EPILOGUE_E),

        // Finish distance function with epilogue
        stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,

        // exp(dist * -gamma)
        stack_ptx_encode_constant_f32(-1.0f / bandwidth),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_constant_f32(M_LOG2_E_F32_EXPLICIT),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
        stack_ptx_encode_return
    };

    return 
        _kermac_semiring_create(
            handle,
            hsa,
            semiring,
            KERMAC_MATRIX_PACKED_TYPE_FULL,
            mma_l2_norm_instructions,
            epilogue_l2_norm_laplace_instructions
        );
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_semiring_laplace_l2_symm_create(
    KermacHandle handle, 
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
) {
    const StackPtxInstruction epilogue_l2_norm_laplace_regularize_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_EPILOGUE_E),
        stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 0),
        stack_ptx_encode_load(0),
        // If x < epsilon set to 0.0
        stack_ptx_encode_constant_f32(epsilon),
        stack_ptx_encode_ptx_instruction_setp_lt_ftz_f32,
        stack_ptx_encode_constant_f32(0.0f),
        stack_ptx_encode_load(0),
        stack_ptx_encode_ptx_instruction_selp_f32,

        // Finish distance function with epilogue

        // dist * -gamma
        stack_ptx_encode_constant_f32(-1.0f / bandwidth),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,

        // exp(dist * -gamma)
        stack_ptx_encode_constant_f32(M_LOG2_E_F32_EXPLICIT),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,

        // If diagonal, set to 1.0 + regularizer
        stack_ptx_encode_input(KERMAC_SEMIRING_REGISTER_EPILOGUE_IS_DIAG),
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_ptx_instruction_setp_ne_u32,
        stack_ptx_encode_constant_f32(1.0 + regularizer),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_return
    };

    return 
        _kermac_semiring_create(
            handle,
            hsa,
            semiring,
            packed_type,
            mma_l2_norm_instructions,
            epilogue_l2_norm_laplace_regularize_instructions
        );
}
