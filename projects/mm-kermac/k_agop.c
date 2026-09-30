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

static
inline
KermacResult
_kermac_agop_backend_select(
    KermacAgopBackend backend,
    bool* use_cutensor,
    bool* use_semiring,
    KermacTensorCoreMode* tcm
) {
    *use_cutensor = false;
    *use_semiring = false;
    *tcm = KERMAC_TENSOR_CORE_MODE_F32;

    switch (backend) {
        case KERMAC_AGOP_BACKEND_CUTENSOR_F32:
            *use_cutensor = true;
            *tcm = KERMAC_TENSOR_CORE_MODE_F32;
            return KERMAC_SUCCESS;
        case KERMAC_AGOP_BACKEND_CUTENSOR_TF32:
            *use_cutensor = true;
            *tcm = KERMAC_TENSOR_CORE_MODE_TF32;
            return KERMAC_SUCCESS;
        case KERMAC_AGOP_BACKEND_FUSED:
            // FUSED currently maps to the semiring gradient path.
            *use_semiring = true;
            *tcm = KERMAC_TENSOR_CORE_MODE_F32;
            return KERMAC_SUCCESS;
        default:
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_agop_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacAgop* agop
) {
    if (handle == NULL || hsa == NULL || agop == NULL) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    agop->semiring_gradient = (KermacSemiringGradient){0};
    agop->has_semiring_gradient = false;

    _KERMAC_CHECK_RET(
        kermac_semiring_gradient_norm_l2_create(handle, hsa, &agop->semiring_gradient)
    );
    agop->has_semiring_gradient = true;

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_agop_destroy(
    KermacAgop agop
) {
    if (agop.has_semiring_gradient) {
        _KERMAC_CHECK_RET( kermac_semiring_gradient_destroy(agop.semiring_gradient) );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_agop_run(
    KermacHandle handle,
    KermacStackAllocator* dsa,
    KermacAgop agop,
    KermacAgopBackend backend,
    KermacAgopOutput output,
    float bandwidth,
    KermacTensor kernel_matrix,
    KermacTensor data_n,
    KermacTensor solution,
    KermacTensor data_m,
    KermacTensor feature_matrix,
    CUstream stream
) {
    if (handle == NULL || dsa == NULL) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (bandwidth <= 0.0f) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (kernel_matrix.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE ||
        data_n.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE ||
        solution.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE ||
        data_m.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE ||
        feature_matrix.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (kernel_matrix.data_type != KERMAC_DATA_TYPE_FLOAT ||
        data_n.data_type != KERMAC_DATA_TYPE_FLOAT ||
        solution.data_type != KERMAC_DATA_TYPE_FLOAT ||
        data_m.data_type != KERMAC_DATA_TYPE_FLOAT ||
        feature_matrix.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (kernel_matrix.stride[0] != 1 ||
        data_n.stride[0] != 1 ||
        solution.stride[0] != 1 ||
        data_m.stride[0] != 1 ||
        feature_matrix.stride[0] != 1) {
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

    bool is_batched = (kernel_matrix.num_modes == 3) ||
                      (data_n.num_modes == 3) ||
                      (solution.num_modes == 3) ||
                      (data_m.num_modes == 3);

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

    int64_t L = 1;
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, kernel_matrix.num_modes == 3 ? kernel_matrix.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, data_n.num_modes == 3 ? data_n.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, solution.num_modes == 3 ? solution.extent[2] : 1) );
    _KERMAC_CHECK_RET( _kermac_merge_batch_size(&L, data_m.num_modes == 3 ? data_m.extent[2] : 1) );

    if (!is_batched && L != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    if (output == KERMAC_AGOP_OUTPUT_FULL) {
        if (feature_matrix.num_modes == 2) {
            if (is_batched || feature_matrix.extent[0] != D || feature_matrix.extent[1] != D) {
                _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
            }
        } else if (feature_matrix.num_modes == 3) {
            if (!is_batched ||
                feature_matrix.extent[0] != D ||
                feature_matrix.extent[1] != D ||
                feature_matrix.extent[2] != L) {
                _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
            }
        } else {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else if (output == KERMAC_AGOP_OUTPUT_DIAG) {
        if (feature_matrix.num_modes == 1) {
            if (is_batched || feature_matrix.extent[0] != D) {
                _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
            }
        } else if (feature_matrix.num_modes == 2) {
            if (!is_batched ||
                feature_matrix.extent[0] != D ||
                feature_matrix.extent[1] != L) {
                _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
            }
        } else {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool use_cutensor = false;
    bool use_semiring = false;
    KermacTensorCoreMode tcm = KERMAC_TENSOR_CORE_MODE_F32;
    _KERMAC_CHECK_RET(
        _kermac_agop_backend_select(backend, &use_cutensor, &use_semiring, &tcm)
    );

    if (use_semiring && !agop.has_semiring_gradient) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool a_is_dry = kernel_matrix.memory.stack_allocator->is_dry;
    bool b_is_dry = data_n.memory.stack_allocator->is_dry;
    bool c_is_dry = solution.memory.stack_allocator->is_dry;
    bool d_is_dry = data_m.memory.stack_allocator->is_dry;
    bool e_is_dry = feature_matrix.memory.stack_allocator->is_dry;
    bool dsa_is_dry = dsa->is_dry;

    bool all_dry = a_is_dry && b_is_dry && c_is_dry && d_is_dry && e_is_dry && dsa_is_dry;
    bool all_live = !a_is_dry && !b_is_dry && !c_is_dry && !d_is_dry && !e_is_dry && !dsa_is_dry;
    if (!all_dry && !all_live) {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    int64_t grad_extents[4] = { M, D, C, 0 };
    if (is_batched) {
        grad_extents[3] = L;
    }

    KermacTensor grad = {0};
    _KERMAC_CHECK_RET(
        kermac_tensor_create(&grad, KERMAC_DATA_TYPE_FLOAT, grad_extents, dsa)
    );

    KermacResult status = KERMAC_SUCCESS;
    if (use_cutensor) {
        status = kermac_cutensor_gradient_norm_l2(
            handle,
            dsa,
            tcm,
            bandwidth,
            kernel_matrix,
            data_n,
            solution,
            data_m,
            grad,
            stream
        );
    } else {
        const float grad_scale = 1.0f / bandwidth;
        status = kermac_semiring_gradient_run(
            agop.semiring_gradient,
            kernel_matrix,
            data_n,
            solution,
            data_m,
            grad,
            grad_scale,
            0.0f,
            stream
        );
    }

    if (status == KERMAC_SUCCESS) {
        const float agop_scale = 1.0f / (float)M;
        const char* modes_a = is_batched ? "mdcl" : "mdc";
        const char* modes_b = NULL;
        const char* modes_out = NULL;

        if (output == KERMAC_AGOP_OUTPUT_FULL) {
            modes_b = is_batched ? "mecl" : "mec";
            modes_out = (feature_matrix.num_modes == 3) ? "del" : "de";
        } else {
            modes_b = modes_a;
            modes_out = (feature_matrix.num_modes == 2) ? "dl" : "d";
        }

        status = kermac_contraction(
            handle,
            dsa,
            tcm,
            agop_scale,
            grad, modes_a,
            grad, modes_b,
            0.0f,
            feature_matrix, modes_out,
            feature_matrix, modes_out,
            stream
        );
    }

    {
        KermacResult destroy_status = kermac_tensor_destroy(grad);
        if (status == KERMAC_SUCCESS) {
            status = destroy_status;
        }
    }

    return status;
}
