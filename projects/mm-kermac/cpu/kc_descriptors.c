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
#include <kermac_cpu.h>
#include <k_internal.h>

#include <math.h>

static
inline
void
compute_logdet_norm_batch(
    int64_t num_rows, 
    int64_t batch_count,
    const float *L, 
    int64_t ldL, 
    int64_t batch_stride_L,
    float *logdet_norm_out
) {
    for (int b = 0; b < batch_count; ++b) {
        const float *Lb = L + b * batch_stride_L;

        double sum_log = 0.0;

        for (int64_t i = 0; i < num_rows; ++i) {
            float Lii = Lb[i + i * ldL];
            sum_log += logf(Lii);
        }

        double logdet_A = 2.0 * sum_log;
        logdet_norm_out[b] = (float)(logdet_A / (double)num_rows);
    }
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_logdet_norm_cpu(
    KermacStackAllocator* hsa,
    KermacTensor factored_matrix,
    KermacTensor logdet_norm,
    CUstream stream
) {
    if (factored_matrix.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (logdet_norm.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (factored_matrix.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (logdet_norm.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (factored_matrix.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (logdet_norm.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    int64_t num_rows = factored_matrix.extent[0];
    int64_t num_batches = factored_matrix.extent[2];

    if (
        factored_matrix.extent[0] != num_rows || 
        factored_matrix.extent[1] != num_rows || 
        factored_matrix.extent[2] != num_batches
    ) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (logdet_norm.extent[0] != num_batches) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }

    bool hsa_is_dry = hsa->is_dry;
    bool factored_matrix_is_dry = factored_matrix.memory.stack_allocator->is_dry;
    bool logdet_norm_is_dry = logdet_norm.memory.stack_allocator->is_dry;

    bool is_dry;
    if (hsa_is_dry && factored_matrix_is_dry && logdet_norm_is_dry) {
        is_dry = true;
    } else if (!hsa_is_dry && !factored_matrix_is_dry && !logdet_norm_is_dry) {
        is_dry = false;
    } else if (!hsa_is_dry && factored_matrix_is_dry && logdet_norm_is_dry) {
        is_dry = true;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    KermacTensor h_factored_matrix = {0};
    KermacTensor h_logdet_norm = {0};

    _KERMAC_CHECK_RET( kermac_tensor_create(&h_factored_matrix, KERMAC_DATA_TYPE_FLOAT, factored_matrix.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_logdet_norm, KERMAC_DATA_TYPE_FLOAT, logdet_norm.extent, hsa) );

    _KERMAC_CHECK_RET( kermac_tensor_copy(factored_matrix, h_factored_matrix, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(logdet_norm, h_logdet_norm, stream) );

    if (!is_dry) {
        float* h_factored_matrix_ptr;
        float* h_logdet_norm_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(h_factored_matrix.memory, (void**)&h_factored_matrix_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_logdet_norm.memory, (void**)&h_logdet_norm_ptr) );

        compute_logdet_norm_batch(
            num_rows, num_batches, 
            h_factored_matrix_ptr, h_factored_matrix.stride[1], h_factored_matrix.stride[2],
            h_logdet_norm_ptr
        );
    }

    _KERMAC_CHECK_RET( kermac_tensor_copy(h_factored_matrix, factored_matrix, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(h_logdet_norm, logdet_norm, stream) );

    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_logdet_norm) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_factored_matrix) );

    return KERMAC_SUCCESS;
}

static
inline
void
compute_norm_H2_batch(
    int64_t n,
    int64_t batch_count,
    int64_t num_labels,
    const float *alpha,
    int64_t stride_alpha,
    int64_t label_stride_alpha,
    const float *y,
    int64_t stride_y,
    int64_t label_stride_y,
    const float lambda_reg,
    float *norm_H2_out
) {
    for (int64_t b = 0; b < batch_count; ++b) {
        const float *alpha_b = alpha + b * stride_alpha;
        const float *y_b     = y     + b * stride_y;
        float lambda_b       = lambda_reg;

        double alpha_dot_y = 0.0;
        double alpha_sq    = 0.0;

        for (int64_t label = 0; label < num_labels; ++label) {
            const float *alpha_l = alpha_b + label * label_stride_alpha;
            const float *y_l     = y_b + label * label_stride_y;
            for (int64_t i = 0; i < n; ++i) {
                float a = alpha_l[i];
                float yi = y_l[i];
                alpha_dot_y += (double)a * (double)yi;
                alpha_sq    += (double)a * (double)a;
            }
        }

        double norm_H2 = alpha_dot_y - (double)lambda_b * alpha_sq;
        norm_H2_out[b] = (float)norm_H2;
    }
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_norm_H2_cpu(
    KermacStackAllocator* hsa,
    float lambda_reg,
    KermacTensor alpha,
    KermacTensor y,
    KermacTensor norm_H2,
    CUstream stream
) {
    if (alpha.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (y.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (norm_H2.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (alpha.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (y.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (norm_H2.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (alpha.num_modes != 2 && alpha.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (y.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (norm_H2.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = alpha.extent[0];
    int64_t num_batches = 0;
    int64_t num_labels = 1;
    int64_t alpha_batch_stride = 0;
    int64_t alpha_label_stride = 0;
    int64_t y_batch_stride = 0;
    int64_t y_label_stride = 0;

    if (alpha.num_modes == 2) {
        num_batches = alpha.extent[1];
        if (alpha.extent[0] != num_rows || alpha.extent[1] != num_batches) {
            _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
        }
        alpha_batch_stride = alpha.stride[1];
        alpha_label_stride = 0;
        num_labels = 1;
    } else {
        num_labels = alpha.extent[1];
        num_batches = alpha.extent[2];
        if (alpha.extent[0] != num_rows || alpha.extent[2] != num_batches) {
            _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
        }
        alpha_batch_stride = alpha.stride[2];
        alpha_label_stride = alpha.stride[1];
    }
    if (y.extent[0] != num_rows) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (y.extent[1] == num_batches && num_labels == 1) {
        y_batch_stride = y.stride[1];
        y_label_stride = 0;
    } else if (y.extent[1] == num_labels) {
        y_batch_stride = 0;
        y_label_stride = y.stride[1];
    } else if (y.extent[1] == 1) {
        y_batch_stride = 0;
        y_label_stride = 0;
    } else {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (norm_H2.extent[0] != num_batches) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }

    bool hsa_is_dry = hsa->is_dry;
    bool alpha_is_dry = alpha.memory.stack_allocator->is_dry;
    bool y_is_dry = y.memory.stack_allocator->is_dry;
    bool norm_H2_is_dry = norm_H2.memory.stack_allocator->is_dry;

    bool is_dry;
    if (hsa_is_dry && alpha_is_dry && y_is_dry && norm_H2_is_dry) {
        is_dry = true;
    } else if (!hsa_is_dry && !alpha_is_dry && !y_is_dry && !norm_H2_is_dry) {
        is_dry = false;
    } else if (!hsa_is_dry && alpha_is_dry && y_is_dry && norm_H2_is_dry) {
        is_dry = true;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    KermacTensor h_alpha = {0};
    KermacTensor h_y = {0};
    KermacTensor h_norm_H2 = {0};

    _KERMAC_CHECK_RET( kermac_tensor_create(&h_alpha, KERMAC_DATA_TYPE_FLOAT, alpha.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_y, KERMAC_DATA_TYPE_FLOAT, y.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_norm_H2, KERMAC_DATA_TYPE_FLOAT, norm_H2.extent, hsa) );

    _KERMAC_CHECK_RET( kermac_tensor_copy(alpha, h_alpha, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(y, h_y, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(norm_H2, h_norm_H2, stream) );

    if (!is_dry) {
        float* h_alpha_ptr;
        float* h_y_ptr;
        float* h_norm_H2_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(h_alpha.memory, (void**)&h_alpha_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_y.memory, (void**)&h_y_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_norm_H2.memory, (void**)&h_norm_H2_ptr) );

        compute_norm_H2_batch(
            num_rows, num_batches, num_labels,
            h_alpha_ptr, alpha_batch_stride, alpha_label_stride,
            h_y_ptr, y_batch_stride, y_label_stride,
            lambda_reg,
            h_norm_H2_ptr
        );
    }

    _KERMAC_CHECK_RET( kermac_tensor_copy(h_alpha, alpha, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(h_y, y, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(h_norm_H2, norm_H2, stream) );

    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_norm_H2) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_y) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_alpha) );

    return KERMAC_SUCCESS;
}
