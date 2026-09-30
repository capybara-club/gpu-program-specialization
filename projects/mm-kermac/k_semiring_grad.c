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

INCTXT(semiring_grad_ptx, KERMAC_SEMIRING_GRAD_PTX);

typedef enum {
    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_A,
    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_B,
    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_D,
    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_DIFF,
    KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_C,
    KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_DIFF,
    KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_E,
    KERMAC_SEMIRING_GRAD_REGISTER_EPILOGUE_E,
    KERMAC_SEMIRING_GRAD_REGISTER_NUM_ENUMS
} KermacSemiringGradRegister;

typedef enum {
    INJECT_SITE_MULTIPLY,
    INJECT_SITE_ACCUMULATE,
    INJECT_SITE_EPILOGUE,
    INJECT_SITE_NUM_ENUMS
} InjectSite;

typedef struct {
    const char* name;
    size_t idx;
} InjectSiteInfo;

static InjectSiteInfo g_inject_sites[INJECT_SITE_NUM_ENUMS] = {
    [INJECT_SITE_MULTIPLY] =   { "multiply",   0 },
    [INJECT_SITE_ACCUMULATE] = { "accumulate", 0 },
    [INJECT_SITE_EPILOGUE] =   { "epilogue",   0 },
};

typedef struct {
    InjectSite site;
    const char* var_name;
    KermacSemiringGradRegister reg;
} RegisterBinding;

static const RegisterBinding g_register_bindings[] = {
    { INJECT_SITE_MULTIPLY,   "a",    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_A },
    { INJECT_SITE_MULTIPLY,   "b",    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_B },
    { INJECT_SITE_MULTIPLY,   "d",    KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_D },
    { INJECT_SITE_MULTIPLY,   "diff", KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_DIFF },

    { INJECT_SITE_ACCUMULATE, "c",    KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_C },
    { INJECT_SITE_ACCUMULATE, "diff", KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_DIFF },
    { INJECT_SITE_ACCUMULATE, "e",    KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_E },

    { INJECT_SITE_EPILOGUE,   "e",    KERMAC_SEMIRING_GRAD_REGISTER_EPILOGUE_E },
};
static const size_t g_num_register_bindings = sizeof(g_register_bindings) / sizeof(g_register_bindings[0]);

static
inline
KermacResult
_kermac_semiring_gradient_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiringGradient* semiring,
    const StackPtxInstruction* multiply_instructions,
    const StackPtxInstruction* accumulate_instructions,
    const StackPtxInstruction* epilogue_instructions
) {
    if (!semiring) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    semiring->module = (CUmodule)0;
    semiring->function = (CUfunction)0;

    StackPtxRegister registers[] = {
        [KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_A]    = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_B]    = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_D]    = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_DIFF] = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_C]  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_DIFF] = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_E]  = { NULL, STACK_PTX_STACK_TYPE_F32 },
        [KERMAC_SEMIRING_GRAD_REGISTER_EPILOGUE_E]    = { NULL, STACK_PTX_STACK_TYPE_F32 },
    };
    static const size_t num_registers = KERMAC_SEMIRING_GRAD_REGISTER_NUM_ENUMS;

    KermacResult result = KERMAC_SUCCESS;
    PtxInjectHandle ptx_inject = NULL;

    PtxInjectResult ptx_status = ptx_inject_create(&ptx_inject, g_semiring_grad_ptx_data);
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

    static const size_t multiply_requests[] = { KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_DIFF };
    static const size_t num_multiply_requests = sizeof(multiply_requests) / sizeof(*multiply_requests);

    static const size_t accumulate_requests[] = { KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_E };
    static const size_t num_accumulate_requests = sizeof(accumulate_requests) / sizeof(*accumulate_requests);

    static const size_t epilogue_requests[] = { KERMAC_SEMIRING_GRAD_REGISTER_EPILOGUE_E };
    static const size_t num_epilogue_requests = sizeof(epilogue_requests) / sizeof(*epilogue_requests);

    static const int execution_limit = 100;

    const StackPtxInstruction* instruction_stubs[] = {
        multiply_instructions,
        accumulate_instructions,
        epilogue_instructions
    };
    const size_t* request_stubs[] = {
        multiply_requests,
        accumulate_requests,
        epilogue_requests
    };
    const size_t request_stub_sizes[] = {
        num_multiply_requests,
        num_accumulate_requests,
        num_epilogue_requests
    };
    size_t num_stubs = 3;

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

    CUresult cuda_result = cuModuleGetFunction(
        &semiring->function,
        semiring->module,
        "cute_semiring_gradient"
    );
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
kermac_semiring_gradient_norm_l2_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiringGradient* semiring
) {
    // For L2 gradients: diff = (d - b) * a, accum = c * diff + e.
    static const StackPtxInstruction multiply_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_B),
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_D),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_MULTIPLY_A),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction accumulate_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_C),
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_DIFF),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_ACCUMULATE_E),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction epilogue_instructions[] = {
        stack_ptx_encode_input(KERMAC_SEMIRING_GRAD_REGISTER_EPILOGUE_E),
        stack_ptx_encode_return
    };

    return _kermac_semiring_gradient_create(
        handle,
        hsa,
        semiring,
        multiply_instructions,
        accumulate_instructions,
        epilogue_instructions
    );
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_semiring_gradient_destroy(
    KermacSemiringGradient semiring
) {
    _KERMAC_CUDA_CHECK_RET( cuModuleUnload(semiring.module) );
    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_merge_batch_size(
    int64_t* L,
    int64_t candidate
) {
    if (candidate <= 1) {
        return KERMAC_SUCCESS;
    }
    if (*L == 1) {
        *L = candidate;
        return KERMAC_SUCCESS;
    }
    if (*L != candidate) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_semiring_gradient_run(
    KermacSemiringGradient semiring,
    KermacTensor kernel_matrix,
    KermacTensor data_n,
    KermacTensor solution,
    KermacTensor data_m,
    KermacTensor gradient,
    float alpha,
    float beta,
    CUstream stream
) {
    if (kernel_matrix.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (data_n.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (solution.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (data_m.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (gradient.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (kernel_matrix.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (data_n.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (solution.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (data_m.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (gradient.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (kernel_matrix.stride[0] != 1 ||
        data_n.stride[0] != 1 ||
        solution.stride[0] != 1 ||
        data_m.stride[0] != 1 ||
        gradient.stride[0] != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (kernel_matrix.num_modes != 2 && kernel_matrix.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (data_n.num_modes != 2 && data_n.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (solution.num_modes != 2 && solution.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (data_m.num_modes != 2 && data_m.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (gradient.num_modes != 3 && gradient.num_modes != 4) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t M = data_m.extent[0];
    int64_t D = data_m.extent[1];
    int64_t N = data_n.extent[0];
    int64_t C = solution.extent[1];

    if (kernel_matrix.extent[0] != M || kernel_matrix.extent[1] != N) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (data_n.extent[1] != D) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (solution.extent[0] != N) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (gradient.extent[0] != M || gradient.extent[1] != D || gradient.extent[2] != C) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    int64_t L = 1;
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, kernel_matrix.num_modes == 3 ? kernel_matrix.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, data_n.num_modes == 3 ? data_n.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, solution.num_modes == 3 ? solution.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, data_m.num_modes == 3 ? data_m.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, gradient.num_modes == 4 ? gradient.extent[3] : 1) );

    if (gradient.num_modes == 3 && L != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (gradient.num_modes == 4 && gradient.extent[3] != L) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    bool a_is_dry = kernel_matrix.memory.stack_allocator->is_dry;
    bool b_is_dry = data_n.memory.stack_allocator->is_dry;
    bool c_is_dry = solution.memory.stack_allocator->is_dry;
    bool d_is_dry = data_m.memory.stack_allocator->is_dry;
    bool e_is_dry = gradient.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry && c_is_dry && d_is_dry && e_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !c_is_dry && !d_is_dry && !e_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* a_ptr = NULL;
        void* b_ptr = NULL;
        void* c_ptr = NULL;
        void* d_ptr = NULL;
        void* e_ptr = NULL;

        _KERMAC_CHECK_RET( kermac_memory_pointer(kernel_matrix.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(data_n.memory, &b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(solution.memory, &c_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(data_m.memory, &d_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(gradient.memory, &e_ptr) );

        int64_t m = M;
        int64_t n = D;
        int64_t o = C;
        int64_t k = N;
        int64_t l = L;

        int64_t num_blocks_M = _KERMAC_NEAREST_LARGER_MULTIPLE(m, 32);
        int64_t num_blocks_N = _KERMAC_NEAREST_LARGER_MULTIPLE(n, 32);
        int64_t num_blocks_O = _KERMAC_NEAREST_LARGER_MULTIPLE(o, 32);

        uint64_t ldA = (uint64_t)kernel_matrix.stride[1];
        uint64_t ldB = (uint64_t)data_n.stride[1];
        uint64_t ldC = (uint64_t)solution.stride[1];
        uint64_t ldD = (uint64_t)data_m.stride[1];
        uint64_t ldE_N = (uint64_t)gradient.stride[1];
        uint64_t ldE_O = (uint64_t)gradient.stride[2];

        uint64_t batch_stride_a = kernel_matrix.num_modes < 3 ? 0 : (uint64_t)kernel_matrix.stride[2];
        uint64_t batch_stride_b = data_n.num_modes < 3 ? 0 : (uint64_t)data_n.stride[2];
        uint64_t batch_stride_c = solution.num_modes < 3 ? 0 : (uint64_t)solution.stride[2];
        uint64_t batch_stride_d = data_m.num_modes < 3 ? 0 : (uint64_t)data_m.stride[2];
        uint64_t batch_stride_e = gradient.num_modes < 4 ? 0 : (uint64_t)gradient.stride[3];

        void* args[] = {
            (void*)&m, (void*)&n, (void*)&o, (void*)&k, (void*)&l,
            (void*)&num_blocks_M,
            (void*)&alpha, (void*)&beta,
            (void*)&a_ptr, (void*)&ldA, (void*)&batch_stride_a,
            (void*)&b_ptr, (void*)&ldB, (void*)&batch_stride_b,
            (void*)&c_ptr, (void*)&ldC, (void*)&batch_stride_c,
            (void*)&d_ptr, (void*)&ldD, (void*)&batch_stride_d,
            (void*)&e_ptr, (void*)&ldE_N, (void*)&ldE_O, (void*)&batch_stride_e
        };

        _KERMAC_CUDA_CHECK_RET(
            cuLaunchKernel(
                semiring.function,
                (unsigned int)(num_blocks_M * l),
                (unsigned int)num_blocks_N,
                (unsigned int)num_blocks_O,
                256, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}
