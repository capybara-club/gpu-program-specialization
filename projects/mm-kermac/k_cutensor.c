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
#include "kermac.h"
#include "k_internal.h"

#include <cutensor.h>

#include <string.h>

static 
inline 
cutensorHandle_t
_kermac_cutensor_get(
    KermacHandle h
) {
    return (cutensorHandle_t)h->cutensor;
}

static 
inline 
void
_kermac_cutensor_set(KermacHandle h, cutensorHandle_t ct) {
    h->cutensor = (void*)ct;
}

KermacResult 
_kermac_cutensor_create(
    KermacHandle h
) {
    cutensorHandle_t ct = NULL;
    _KERMAC_CUTENSOR_CHECK_RET( cutensorCreate(&ct) );
    _kermac_cutensor_set(h, ct);
    return KERMAC_SUCCESS;
}

KermacResult 
_kermac_cutensor_destroy(
    KermacHandle h
) {
    if (!h || !h->cutensor){
        return KERMAC_SUCCESS;
    }

    _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroy(_kermac_cutensor_get(h)) );
    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_cutensor_create_tensor_descriptor(
    KermacHandle handle,
    cutensorTensorDescriptor_t* desc,
    KermacTensor tensor
) {
    _KERMAC_CUTENSOR_CHECK_RET( 
        cutensorCreateTensorDescriptor(
            handle->cutensor,
            desc,
            tensor.num_modes,
            tensor.extent,
            tensor.stride,
            CUDA_R_32F,
            _kermac_memory_space_alignments[KERMAC_MEMORY_SPACE_DEVICE]
        )
    );
    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_permute(
    KermacHandle handle,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    CUstream stream,
    cutensorTensorDescriptor_t* desc_a,
    cutensorTensorDescriptor_t* desc_b,
    cutensorOperationDescriptor_t* desc_op,
    cutensorPlanPreference_t* plan_pref,
    cutensorPlan_t* plan
) {
if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (strlen(modes_a) != a.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_b) != b.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );

    int32_t i_modes_a[4] = {0};
    int32_t i_modes_b[4] = {0};

    for (int32_t i = 0; i < a.num_modes; i++) i_modes_a[i] = (int32_t)modes_a[i];
    for (int32_t i = 0; i < b.num_modes; i++) i_modes_b[i] = (int32_t)modes_b[i];

    cutensorComputeDescriptor_t compute_descriptor = CUTENSOR_COMPUTE_DESC_32F;
    
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_a, a) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_b, b) );

    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorCreatePermutation(
            handle->cutensor, desc_op,
            *desc_a, i_modes_a, CUTENSOR_OP_IDENTITY,
            *desc_b, i_modes_b,
            compute_descriptor
        )
    );

    cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;
    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorCreatePlanPreference(
            handle->cutensor, plan_pref,
            cutensor_algo,
            CUTENSOR_JIT_MODE_NONE
        )
    );

    _KERMAC_CUTENSOR_CHECK_RET( 
        cutensorCreatePlan(
            handle->cutensor, plan,
            *desc_op,
            *plan_pref,
            0
        )
    );

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* a_ptr;
        void* b_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b.memory, &b_ptr) );

        _KERMAC_CUTENSOR_CHECK_RET(
            cutensorPermute(
                handle->cutensor, *plan,
                &alpha, a_ptr, b_ptr,
                stream
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_permute(
    KermacHandle handle,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    CUstream stream
) {
    cutensorTensorDescriptor_t desc_a = {0};
    cutensorTensorDescriptor_t desc_b = {0};
    cutensorOperationDescriptor_t desc_op = {0};
    cutensorPlanPreference_t plan_pref = {0};
    cutensorPlan_t plan = {0};

    KermacResult status = _kermac_permute(
        handle,
        alpha, a, modes_a, b, modes_b,
        stream, 
        &desc_a, &desc_b,
        &desc_op, &plan_pref, &plan
    );

    if (plan != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyPlan(plan) );
    }
    if (plan_pref != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyPlanPreference(plan_pref) );
    }
    if (desc_op != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyOperationDescriptor(desc_op) );
    }
    if (desc_b != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_b) );
    }
    if (desc_a != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_a) );
    }

    return status;
}

static
inline
KermacResult
_kermac_contraction(
    KermacHandle handle,
    KermacStackAllocator* dsa,
    KermacTensorCoreMode tcm,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    float beta,
    KermacTensor c, const char* modes_c,
    KermacTensor d, const char* modes_d,
    CUstream stream,
    cutensorTensorDescriptor_t* desc_a,
    cutensorTensorDescriptor_t* desc_b,
    cutensorTensorDescriptor_t* desc_c,
    cutensorTensorDescriptor_t* desc_d,
    cutensorOperationDescriptor_t* desc_op,
    cutensorPlanPreference_t* plan_pref,
    cutensorPlan_t* plan,
    KermacMemory* workspace
) {
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (c.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (d.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (c.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (d.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (strlen(modes_a) != a.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_b) != b.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_c) != c.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_d) != d.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );

    int32_t i_modes_a[4] = {0};
    int32_t i_modes_b[4] = {0};
    int32_t i_modes_c[4] = {0};
    int32_t i_modes_d[4] = {0};

    for (int32_t i = 0; i < a.num_modes; i++) i_modes_a[i] = (int32_t)modes_a[i];
    for (int32_t i = 0; i < b.num_modes; i++) i_modes_b[i] = (int32_t)modes_b[i];
    for (int32_t i = 0; i < c.num_modes; i++) i_modes_c[i] = (int32_t)modes_c[i];
    for (int32_t i = 0; i < d.num_modes; i++) i_modes_d[i] = (int32_t)modes_d[i];

    cutensorComputeDescriptor_t compute_descriptor;
    switch(tcm) {
        case KERMAC_TENSOR_CORE_MODE_F32: {
            compute_descriptor = CUTENSOR_COMPUTE_DESC_32F;
        } break;
        case KERMAC_TENSOR_CORE_MODE_TF32: {
            compute_descriptor = CUTENSOR_COMPUTE_DESC_TF32;
        } break;
        default: {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        } break;
    }
    
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_a, a) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_b, b) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_c, c) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_d, d) );

    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorCreateContraction(
            handle->cutensor, desc_op,
            *desc_a, i_modes_a, CUTENSOR_OP_IDENTITY,
            *desc_b, i_modes_b, CUTENSOR_OP_IDENTITY,
            *desc_c, i_modes_c, CUTENSOR_OP_IDENTITY,
            *desc_d, i_modes_d,
            compute_descriptor
        )
    );

    cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;
    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorCreatePlanPreference(
            handle->cutensor, plan_pref,
            cutensor_algo,
            CUTENSOR_JIT_MODE_NONE
        )
    );

    size_t workspace_size_estimate = 0;
    _KERMAC_CUTENSOR_CHECK_RET( 
        cutensorEstimateWorkspaceSize(
            handle->cutensor, *desc_op, *plan_pref,
            CUTENSOR_WORKSPACE_DEFAULT,
            &workspace_size_estimate
        )
    );

    _KERMAC_CUTENSOR_CHECK_RET( 
        cutensorCreatePlan(
            handle->cutensor, plan,
            *desc_op,
            *plan_pref,
            workspace_size_estimate
        )
    );

    size_t actual_workspace_size = 0;
    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorPlanGetAttribute(
            handle->cutensor, *plan,
            CUTENSOR_PLAN_REQUIRED_WORKSPACE,
            &actual_workspace_size,
            sizeof(size_t)
        )
    );

    _KERMAC_CHECK_RET(
        _kermac_memory_create(
            workspace,
            dsa,
            actual_workspace_size,
            _kermac_memory_space_alignments[KERMAC_MEMORY_SPACE_DEVICE]
        )
    );

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool c_is_dry = c.memory.stack_allocator->is_dry;
    bool d_is_dry = d.memory.stack_allocator->is_dry;
    bool dsa_is_dry = dsa->is_dry;
    bool is_dry;
    if (a_is_dry && b_is_dry && c_is_dry && d_is_dry && dsa_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !c_is_dry && !d_is_dry && !dsa_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* a_ptr;
        void* b_ptr;
        void* c_ptr;
        void* d_ptr;
        void* workspace_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b.memory, &b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(c.memory, &c_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(d.memory, &d_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(*workspace, &workspace_ptr) );

        _KERMAC_CUTENSOR_CHECK_RET(
            cutensorContract(
                handle->cutensor, *plan,
                &alpha, a_ptr, b_ptr,
                &beta, c_ptr, d_ptr,
                workspace_ptr, actual_workspace_size,
                stream
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_contraction(
    KermacHandle handle,
    KermacStackAllocator* dsa,
    KermacTensorCoreMode tcm,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    float beta,
    KermacTensor c, const char* modes_c,
    KermacTensor d, const char* modes_d,
    CUstream stream
) {
    cutensorTensorDescriptor_t desc_a = {0};
    cutensorTensorDescriptor_t desc_b = {0};
    cutensorTensorDescriptor_t desc_c = {0};
    cutensorTensorDescriptor_t desc_d = {0};
    cutensorOperationDescriptor_t desc_op = {0};
    cutensorPlanPreference_t plan_pref = {0};
    cutensorPlan_t plan = {0};
    KermacMemory workspace = {0};

    KermacResult status = _kermac_contraction(
        handle, dsa, tcm,
        alpha, a, modes_a, b, modes_b,
        beta, c, modes_c, d, modes_d, 
        stream, 
        &desc_a, &desc_b, &desc_c, &desc_d,
        &desc_op, &plan_pref, &plan, 
        &workspace
    );

    if (workspace.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( _kermac_memory_destroy(workspace) );
    }
    if (plan != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyPlan(plan) );
    }
    if (plan_pref != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyPlanPreference(plan_pref) );
    }
    if (desc_op != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyOperationDescriptor(desc_op) );
    }
    if (desc_d != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_d) );
    }
    if (desc_c != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_c) );
    }
    if (desc_b != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_b) );
    }
    if (desc_a != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_a) );
    }

    return status;
}

static
inline
KermacResult
_kermac_contraction_trinary(
    KermacHandle handle,
    KermacStackAllocator* dsa,
    KermacTensorCoreMode tcm,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    KermacTensor c, const char* modes_c,
    float beta,
    KermacTensor d, const char* modes_d,
    KermacTensor e, const char* modes_e,
    CUstream stream,
    cutensorTensorDescriptor_t* desc_a,
    cutensorTensorDescriptor_t* desc_b,
    cutensorTensorDescriptor_t* desc_c,
    cutensorTensorDescriptor_t* desc_d,
    cutensorTensorDescriptor_t* desc_e,
    cutensorOperationDescriptor_t* desc_op,
    cutensorPlanPreference_t* plan_pref,
    cutensorPlan_t* plan,
    KermacMemory* workspace
) {
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (c.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (d.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (e.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (c.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (d.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (e.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (strlen(modes_a) != a.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_b) != b.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_c) != c.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_d) != d.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    if (strlen(modes_e) != e.num_modes) _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );

    int32_t i_modes_a[4] = {0};
    int32_t i_modes_b[4] = {0};
    int32_t i_modes_c[4] = {0};
    int32_t i_modes_d[4] = {0};
    int32_t i_modes_e[4] = {0};

    for (int32_t i = 0; i < a.num_modes; i++) i_modes_a[i] = (int32_t)modes_a[i];
    for (int32_t i = 0; i < b.num_modes; i++) i_modes_b[i] = (int32_t)modes_b[i];
    for (int32_t i = 0; i < c.num_modes; i++) i_modes_c[i] = (int32_t)modes_c[i];
    for (int32_t i = 0; i < d.num_modes; i++) i_modes_d[i] = (int32_t)modes_d[i];
    for (int32_t i = 0; i < e.num_modes; i++) i_modes_e[i] = (int32_t)modes_e[i];

    cutensorComputeDescriptor_t compute_descriptor;
    switch(tcm) {
        case KERMAC_TENSOR_CORE_MODE_F32: {
            compute_descriptor = CUTENSOR_COMPUTE_DESC_32F;
        } break;
        case KERMAC_TENSOR_CORE_MODE_TF32: {
            compute_descriptor = CUTENSOR_COMPUTE_DESC_TF32;
        } break;
        default: {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        } break;
    }
    
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_a, a) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_b, b) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_c, c) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_d, d) );
    _KERMAC_CHECK_RET( _kermac_cutensor_create_tensor_descriptor(handle, desc_e, e) );

    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorCreateContractionTrinary(
            handle->cutensor, desc_op,
            *desc_a, i_modes_a, CUTENSOR_OP_IDENTITY,
            *desc_b, i_modes_b, CUTENSOR_OP_IDENTITY,
            *desc_c, i_modes_c, CUTENSOR_OP_IDENTITY,
            *desc_d, i_modes_d, CUTENSOR_OP_IDENTITY,
            *desc_e, i_modes_e,
            compute_descriptor
        )
    );

    cutensorAlgo_t cutensor_algo = CUTENSOR_ALGO_DEFAULT;
    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorCreatePlanPreference(
            handle->cutensor, plan_pref,
            cutensor_algo,
            CUTENSOR_JIT_MODE_NONE
        )
    );

    size_t workspace_size_estimate = 0;
    _KERMAC_CUTENSOR_CHECK_RET( 
        cutensorEstimateWorkspaceSize(
            handle->cutensor, *desc_op, *plan_pref,
            CUTENSOR_WORKSPACE_DEFAULT,
            &workspace_size_estimate
        )
    );

    _KERMAC_CUTENSOR_CHECK_RET( 
        cutensorCreatePlan(
            handle->cutensor, plan,
            *desc_op,
            *plan_pref,
            workspace_size_estimate
        )
    );

    size_t actual_workspace_size = 0;
    _KERMAC_CUTENSOR_CHECK_RET(
        cutensorPlanGetAttribute(
            handle->cutensor, *plan,
            CUTENSOR_PLAN_REQUIRED_WORKSPACE,
            &actual_workspace_size,
            sizeof(size_t)
        )
    );

    _KERMAC_CHECK_RET(
        _kermac_memory_create(
            workspace,
            dsa,
            actual_workspace_size,
            _kermac_memory_space_alignments[KERMAC_MEMORY_SPACE_DEVICE]
        )
    );

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool c_is_dry = c.memory.stack_allocator->is_dry;
    bool d_is_dry = d.memory.stack_allocator->is_dry;
    bool e_is_dry = e.memory.stack_allocator->is_dry;
    bool dsa_is_dry = dsa->is_dry;
    bool is_dry;
    if (a_is_dry && b_is_dry && c_is_dry && d_is_dry && e_is_dry && dsa_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !c_is_dry && !d_is_dry && !e_is_dry && !dsa_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* a_ptr;
        void* b_ptr;
        void* c_ptr;
        void* d_ptr;
        void* e_ptr;
        void* workspace_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b.memory, &b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(c.memory, &c_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(d.memory, &d_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(e.memory, &e_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(*workspace, &workspace_ptr) );

        _KERMAC_CUTENSOR_CHECK_RET(
            cutensorContractTrinary(
                handle->cutensor, *plan,
                &alpha, a_ptr, b_ptr, c_ptr,
                &beta, d_ptr, e_ptr,
                workspace_ptr, actual_workspace_size,
                stream
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_contraction_trinary(
    KermacHandle handle,
    KermacStackAllocator* dsa,
    KermacTensorCoreMode tcm,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    KermacTensor c, const char* modes_c,
    float beta,
    KermacTensor d, const char* modes_d,
    KermacTensor e, const char* modes_e,
    CUstream stream
) {
    cutensorTensorDescriptor_t desc_a = {0};
    cutensorTensorDescriptor_t desc_b = {0};
    cutensorTensorDescriptor_t desc_c = {0};
    cutensorTensorDescriptor_t desc_d = {0};
    cutensorTensorDescriptor_t desc_e = {0};
    cutensorOperationDescriptor_t desc_op = {0};
    cutensorPlanPreference_t plan_pref = {0};
    cutensorPlan_t plan = {0};
    KermacMemory workspace = {0};

    KermacResult status = _kermac_contraction_trinary(
        handle, dsa, tcm,
        alpha, a, modes_a, b, modes_b, c, modes_c, 
        beta, d, modes_d, e, modes_e, 
        stream, 
        &desc_a, &desc_b, &desc_c, &desc_d, &desc_e,
        &desc_op, &plan_pref, &plan, 
        &workspace
    );

    if (workspace.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( _kermac_memory_destroy(workspace) );
    }
    if (plan != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyPlan(plan) );
    }
    if (plan_pref != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyPlanPreference(plan_pref) );
    }
    if (desc_op != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyOperationDescriptor(desc_op) );
    }
    if (desc_e != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_e) );
    }
    if (desc_d != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_d) );
    }
    if (desc_c != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_c) );
    }
    if (desc_b != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_b) );
    }
    if (desc_a != NULL) {
        _KERMAC_CUTENSOR_CHECK_RET( cutensorDestroyTensorDescriptor(desc_a) );
    }

    return status;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_cutensor_gradient_norm_l2(
    KermacHandle handle,
    KermacStackAllocator* dsa,
    KermacTensorCoreMode tcm,
    float bandwidth,
    KermacTensor kernel_matrix,
    KermacTensor data_n,
    KermacTensor solution,
    KermacTensor data_m,
    KermacTensor gradient,
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
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
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
    if (bandwidth <= 0.0f) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
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

    bool is_batched = (kernel_matrix.num_modes == 3) ||
                      (data_n.num_modes == 3) ||
                      (solution.num_modes == 3) ||
                      (data_m.num_modes == 3) ||
                      (gradient.num_modes == 4);

    int64_t batch_count = 1;
    if (is_batched) {
        if (kernel_matrix.num_modes != 3 ||
            data_n.num_modes != 3 ||
            solution.num_modes != 3 ||
            data_m.num_modes != 3 ||
            gradient.num_modes != 4) {
            _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
        }
        batch_count = kernel_matrix.extent[2];
        if (data_n.extent[2] != batch_count ||
            solution.extent[2] != batch_count ||
            data_m.extent[2] != batch_count) {
            _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
        }
        if (gradient.extent[3] != batch_count) {
            _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
        }
    } else {
        if (kernel_matrix.num_modes != 2 ||
            data_n.num_modes != 2 ||
            solution.num_modes != 2 ||
            data_m.num_modes != 2 ||
            gradient.num_modes != 3) {
            _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
        }
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
    if (is_batched && batch_count <= 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    const char* modes_solution = is_batched ? "ncl" : "nc";
    const char* modes_kernel = is_batched ? "mnl" : "mn";
    const char* modes_data_n = is_batched ? "ndl" : "nd";
    const char* modes_data_m = is_batched ? "mdl" : "md";
    const char* modes_grad = is_batched ? "mdcl" : "mdc";

    KermacTensor term_x = (KermacTensor){0};
    KermacResult status = kermac_tensor_create(
        &term_x,
        KERMAC_DATA_TYPE_FLOAT,
        gradient.extent,
        dsa
    );
    if (status != KERMAC_SUCCESS) {
        return status;
    }

    status = kermac_contraction_trinary(
        handle, dsa, tcm,
        1.0f,
        solution, modes_solution,
        kernel_matrix, modes_kernel,
        data_n, modes_data_n,
        0.0f,
        term_x, modes_grad,
        term_x, modes_grad,
        stream
    );
    if (status != KERMAC_SUCCESS) {
        (void)kermac_tensor_destroy(term_x);
        return status;
    }

    const float scale = 1.0f / bandwidth;
    status = kermac_contraction_trinary(
        handle, dsa, tcm,
        scale,
        solution, modes_solution,
        kernel_matrix, modes_kernel,
        data_m, modes_data_m,
        -scale,
        term_x, modes_grad,
        gradient, modes_grad,
        stream
    );

    {
        KermacResult destroy_status = kermac_tensor_destroy(term_x);
        if (status == KERMAC_SUCCESS) {
            status = destroy_status;
        }
    }

    return status;
}
