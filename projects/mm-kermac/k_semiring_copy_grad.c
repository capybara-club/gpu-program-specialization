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

INCTXT(semiring_nt_lower_copy_grad_ptx, KERMAC_SEMIRING_NT_LOWER_COPY_GRAD_PTX);
INCTXT(semiring_nt_upper_copy_grad_ptx, KERMAC_SEMIRING_NT_UPPER_COPY_GRAD_PTX);

typedef enum {
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_A,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_B,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_0,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_1,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_2,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_3,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_C,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_IS_DIAG,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_0,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_1,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_2,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_3,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_E,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_G,
    KERMAC_SEMIRING_COPY_GRAD_REGISTER_NUM_ENUMS
} KermacSemiringCopyGradRegister;

typedef struct {
    const char* name;
    size_t idx;
} InjectSiteInfo;

typedef enum {
    INJECT_SITE_MMA,
    INJECT_SITE_EPILOGUE,
    INJECT_SITE_NUM_ENUMS
} InjectSite;

typedef struct {
    size_t site;
    const char* var_name;
    size_t reg;
} RegisterBinding;

static InjectSiteInfo g_inject_sites_copy_grad[INJECT_SITE_NUM_ENUMS] = {
    [INJECT_SITE_MMA]      = { "mma",      0 },
    [INJECT_SITE_EPILOGUE] = { "epilogue", 0 },
};

static const RegisterBinding g_register_bindings_copy_grad[] = {
    // multiply
    { INJECT_SITE_MMA, "a",   KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_A },
    { INJECT_SITE_MMA, "b",   KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_B },
    { INJECT_SITE_MMA, "h_0", KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_0 },
    { INJECT_SITE_MMA, "h_1", KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_1 },
    { INJECT_SITE_MMA, "h_2", KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_2 },
    { INJECT_SITE_MMA, "h_3", KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_3 },
    { INJECT_SITE_MMA, "c",   KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_C },

    // epilogue
    { INJECT_SITE_EPILOGUE, "is_diag", KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_IS_DIAG },
    { INJECT_SITE_EPILOGUE, "h_0", KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_0 },
    { INJECT_SITE_EPILOGUE, "h_1", KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_1 },
    { INJECT_SITE_EPILOGUE, "h_2", KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_2 },
    { INJECT_SITE_EPILOGUE, "h_3", KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_3 },
    { INJECT_SITE_EPILOGUE, "e",    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_E },
    { INJECT_SITE_EPILOGUE, "g",    KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_G },
};
static const size_t g_num_register_bindings_copy_grad =
    sizeof(g_register_bindings_copy_grad) / sizeof(g_register_bindings_copy_grad[0]);

static
inline
KermacResult
_kermac_semiring_copy_grad_create_impl(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    const char* annotated_ptx,
    InjectSiteInfo* inject_sites,
    size_t num_inject_sites,
    const RegisterBinding* register_bindings,
    size_t num_register_bindings,
    StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction** instruction_stubs,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs
) {
    if (!semiring) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    semiring->module = (CUmodule)0;
    semiring->function = (CUfunction)0;
    semiring->packed_type = packed_type;

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
    if (num_injects_found != num_inject_sites) {
        result = KERMAC_ERROR_INTERNAL;
        goto cleanup;
    }

    for (size_t s = 0; s < num_inject_sites; ++s) {
        ptx_status = ptx_inject_inject_info_by_name(
            ptx_inject,
            inject_sites[s].name,
            &inject_sites[s].idx,
            NULL,
            NULL
        );
        if (ptx_status != PTX_INJECT_SUCCESS) {
            result = KERMAC_ERROR_PTX_INJECT;
            goto cleanup;
        }
    }

    for (size_t i = 0; i < num_register_bindings; ++i) {
        const RegisterBinding* b = &register_bindings[i];
        size_t inject_idx = inject_sites[b->site].idx;

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

    static const int execution_limit = 100;

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

static
inline
KermacResult
_kermac_semiring_copy_grad_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    const StackPtxInstruction* mma_instructions,
    const StackPtxInstruction* epilogue_instructions
) {
    StackPtxRegister registers[] = {
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_A]                        = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_B]                        = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_0]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_1]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_2]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_HYPER_3]                  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_C]                        = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_IS_DIAG]             = { NULL, STACK_PTX_STACK_TYPE_U32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_0]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_1]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_2]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_HYPER_3]             = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_E]                   = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_G]                   = { NULL, STACK_PTX_STACK_TYPE_F32 },
    };
    static const size_t num_registers = KERMAC_SEMIRING_COPY_GRAD_REGISTER_NUM_ENUMS;

    const char* annotated_ptx = NULL;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        annotated_ptx = g_semiring_nt_lower_copy_grad_ptx_data;
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        annotated_ptx = g_semiring_nt_upper_copy_grad_ptx_data;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    static const size_t mma_requests[] = { KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_C };
    static const size_t num_mma_requests = sizeof(mma_requests) / sizeof(*mma_requests);

    static const size_t epilogue_requests[] = {
        KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_E,
        KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_G
    };
    static const size_t num_epilogue_requests = sizeof(epilogue_requests) / sizeof(*epilogue_requests);

    const StackPtxInstruction* instruction_stubs[] = { mma_instructions, epilogue_instructions };
    const size_t* request_stubs[] = { mma_requests, epilogue_requests };
    const size_t request_stub_sizes[] = { num_mma_requests, num_epilogue_requests };
    size_t num_stubs = 2;

    return _kermac_semiring_copy_grad_create_impl(
        handle,
        hsa,
        semiring,
        packed_type,
        annotated_ptx,
        g_inject_sites_copy_grad,
        INJECT_SITE_NUM_ENUMS,
        g_register_bindings_copy_grad,
        g_num_register_bindings_copy_grad,
        registers,
        num_registers,
        instruction_stubs,
        request_stubs,
        request_stub_sizes,
        num_stubs
    );
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_semiring_laplace_l2_symm_copy_grad_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
) {
    const StackPtxInstruction mma_l2_norm_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_C),
        stack_ptx_encode_input(KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_A),
        stack_ptx_encode_input(KERMAC_SEMIRING_COPY_GRAD_REGISTER_MMA_B),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    const StackPtxInstruction epilogue_l2_norm_laplace_copy_grad_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_E),
        stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 0),

        stack_ptx_encode_load(0),
        stack_ptx_encode_constant_f32(epsilon),
        stack_ptx_encode_ptx_instruction_setp_lt_ftz_f32,
        stack_ptx_encode_constant_f32(0.0f),
        stack_ptx_encode_load(0),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_constant_f32(-1.0f / bandwidth),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_constant_f32(M_LOG2_E_F32_EXPLICIT),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,

        stack_ptx_encode_input(KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_IS_DIAG),
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_ptx_instruction_setp_ne_u32,
        stack_ptx_encode_constant_f32(1.0 + regularizer),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 1),

        stack_ptx_encode_load(0),
        stack_ptx_encode_load(1),
        stack_ptx_encode_ptx_instruction_div_approx_ftz_f32,

        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 2),

        stack_ptx_encode_load(0),
        stack_ptx_encode_constant_f32(epsilon),
        stack_ptx_encode_ptx_instruction_setp_lt_ftz_f32,
        stack_ptx_encode_constant_f32(0.0f),
        stack_ptx_encode_load(2),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_input(KERMAC_SEMIRING_COPY_GRAD_REGISTER_EPILOGUE_IS_DIAG),
        stack_ptx_encode_constant_u32(0),
        stack_ptx_encode_ptx_instruction_setp_ne_u32,
        stack_ptx_encode_constant_f32(0.0f),
        stack_ptx_encode_ptx_instruction_selp_f32,

        stack_ptx_encode_load(1),
        stack_ptx_encode_return
    };

    return
        _kermac_semiring_copy_grad_create(
            handle,
            hsa,
            semiring,
            packed_type,
            mma_l2_norm_instructions,
            epilogue_l2_norm_laplace_copy_grad_instructions
        );
}
