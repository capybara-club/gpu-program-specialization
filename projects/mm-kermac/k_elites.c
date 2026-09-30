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

KERMAC_PUBLIC_DEF
KermacResult
kermac_logdet_norm_norm_h2(
    KermacHandle kermac,
    float lambda_reg,
    KermacTensor factored_matrix,   // N,N,L
    KermacTensor alpha,             // N,L or N,1,L
    KermacTensor y,                 // N,L or N,1 (broadcast)
    KermacTensor logdet_norm,       // L
    KermacTensor norm_H2,           // L
    CUstream stream
) {
    if (factored_matrix.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (alpha.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (y.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (logdet_norm.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (norm_H2.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (factored_matrix.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (alpha.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (y.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (logdet_norm.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (norm_H2.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (factored_matrix.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (alpha.num_modes != 2 && alpha.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (y.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (logdet_norm.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (norm_H2.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = factored_matrix.extent[0];
    int64_t num_batches = factored_matrix.extent[2];
    int64_t num_labels = 1;
    int64_t alpha_batch_stride = 0;
    int64_t alpha_label_stride = 0;
    int64_t y_batch_stride = 0;
    int64_t y_label_stride = 0;

    if (
        factored_matrix.extent[0] != num_rows || 
        factored_matrix.extent[1] != num_rows || 
        factored_matrix.extent[2] != num_batches
    ) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (alpha.num_modes == 2) {
        if (alpha.extent[0] != num_rows || alpha.extent[1] != num_batches) {
            _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
        }
        alpha_batch_stride = alpha.stride[1];
        alpha_label_stride = 0;
        num_labels = 1;
    } else {
        num_labels = alpha.extent[1];
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
    if (logdet_norm.extent[0] != num_batches) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (norm_H2.extent[0] != num_batches) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }

    bool factored_matrix_is_dry = factored_matrix.memory.stack_allocator->is_dry;
    bool alpha_is_dry = alpha.memory.stack_allocator->is_dry;
    bool y_is_dry = y.memory.stack_allocator->is_dry;
    bool logdet_norm_is_dry = logdet_norm.memory.stack_allocator->is_dry;
    bool norm_H2_is_dry = norm_H2.memory.stack_allocator->is_dry;

    bool is_dry;
    if (factored_matrix_is_dry && alpha_is_dry && y_is_dry && logdet_norm_is_dry && norm_H2_is_dry) {
        is_dry = true;
    } else if (!factored_matrix_is_dry && !alpha_is_dry && !y_is_dry && !logdet_norm_is_dry && !norm_H2_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    CUfunction function = kermac->functions[_KERMAC_KERNEL_FUNCTION_LOGDET_NORM_NORM_H2_F32];

    int block_size = 32 * _KERMAC_NEAREST_LARGER_MULTIPLE(num_rows, 32);
    block_size = _KERMAC_MIN(block_size, 1024);

    const int num_blocks = num_batches;
    if (!is_dry) {
        void* factored_matrix_ptr;
        void* alpha_ptr;
        void* y_ptr;
        void* logdet_norm_ptr;
        void* norm_H2_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(factored_matrix.memory, &factored_matrix_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(alpha.memory, &alpha_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(y.memory, &y_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(logdet_norm.memory, &logdet_norm_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(norm_H2.memory, &norm_H2_ptr) );

        void* args[] = {
            (void*)&lambda_reg,
            (void*)&num_rows,
            (void*)&num_labels,
            (void*)&factored_matrix_ptr,
            (void*)&factored_matrix.stride[1],
            (void*)&factored_matrix.stride[2],
            (void*)&alpha_ptr,
            (void*)&alpha_batch_stride,
            (void*)&alpha_label_stride,
            (void*)&y_ptr,
            (void*)&y_batch_stride,
            (void*)&y_label_stride,
            (void*)&logdet_norm_ptr,
            (void*)&norm_H2_ptr
        };

        _KERMAC_CUDA_CHECK_RET(
            cuLaunchKernel(
                function,
                num_blocks, 1, 1,
                block_size, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

#ifndef ROW_STATS_BLOCK_SIZE
#define ROW_STATS_BLOCK_SIZE 256
#endif

static
KermacResult
kermac_row_stats_impl(
    KermacHandle kermac,
    KermacTensor a,
    const KermacTensor* mean,
    KermacTensor stdev,
    CUstream stream
) {
    const bool has_mean = mean != NULL;

    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (has_mean && mean->memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (stdev.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (has_mean && mean->data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (stdev.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (has_mean && mean->num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (stdev.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    const int64_t num_rows    = a.extent[0];
    const int64_t num_cols    = a.extent[1];
    const int64_t num_batches = a.extent[2];

    if (has_mean &&
        (mean->extent[0] != num_batches || mean->extent[1] != num_cols)) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (stdev.extent[0] != num_batches || stdev.extent[1] != num_cols) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }

    const bool a_is_dry     = a.memory.stack_allocator->is_dry;
    const bool mean_is_dry  = has_mean ? mean->memory.stack_allocator->is_dry : a_is_dry;
    const bool stdev_is_dry = stdev.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && mean_is_dry && stdev_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !mean_is_dry && !stdev_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    CUfunction function = kermac->functions[_KERMAC_KERNEL_FUNCTION_ROW_STATS_F32];

    const int block_size      = ROW_STATS_BLOCK_SIZE;
    const int warps_per_block = block_size >> 5;         // / 32

    const int64_t task_count = num_batches * num_cols;   // one task per (batch, col)
    if (task_count == 0) {
        return KERMAC_SUCCESS;
    }

    int64_t blocks_needed = (task_count + (int64_t)warps_per_block - 1) / (int64_t)warps_per_block;

    const int64_t max_blocks = 65535;
    const int num_blocks = (int)_KERMAC_MIN(blocks_needed, max_blocks);

    if (!is_dry) {
        void* a_ptr;
        void* mean_ptr = NULL;
        void* stdev_ptr;
        int64_t mean_ld = 0;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        if (has_mean) {
            _KERMAC_CHECK_RET( kermac_memory_pointer(mean->memory, &mean_ptr) );
            mean_ld = mean->stride[1];
        }
        _KERMAC_CHECK_RET( kermac_memory_pointer(stdev.memory, &stdev_ptr) );

        void* args[] = {
            (void*)&num_rows,
            (void*)&num_cols,
            (void*)&num_batches,
            (void*)&a_ptr,
            (void*)&a.stride[1],
            (void*)&a.stride[2],
            (void*)&mean_ptr,
            (void*)&mean_ld,
            (void*)&stdev_ptr,
            (void*)&stdev.stride[1]
        };

        _KERMAC_CUDA_CHECK_RET(
            cuLaunchKernel(
                function,
                (unsigned int)num_blocks, 1, 1,
                (unsigned int)block_size, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEC
KermacResult
kermac_row_stats(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor mean,
    KermacTensor stdev,
    CUstream stream
) {
    return kermac_row_stats_impl(kermac, a, &mean, stdev, stream);
}

KERMAC_PUBLIC_DEC
KermacResult
kermac_stdev_rows(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor stdev,
    CUstream stream
) {
    return kermac_row_stats_impl(kermac, a, NULL, stdev, stream);
}

#ifndef APPLY_ROW_STATS_BLOCK_SIZE
#define APPLY_ROW_STATS_BLOCK_SIZE 256
#endif

static
KermacResult
kermac_apply_row_stats_impl(
    KermacHandle kermac,
    KermacTensor a,
    const KermacTensor* mean,
    KermacTensor stdev,
    uint32_t flags,
    CUstream stream
) {
    const bool has_center = (flags & KERMAC_ROW_NORMALIZE_FLAG_CENTER) != 0;
    const bool has_scale = (flags & KERMAC_ROW_NORMALIZE_FLAG_SCALE) != 0;

    if (!has_center && !has_scale) {
        return KERMAC_SUCCESS;
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (has_center) {
        if (mean == NULL) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
        if (mean->memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
            _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
        }
    }
    if (stdev.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (has_center && mean->data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (stdev.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (has_center && mean->num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (stdev.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    const int64_t num_rows    = a.extent[0];
    const int64_t num_cols    = a.extent[1];
    const int64_t num_batches = a.extent[2];

    if (has_center &&
        (mean->extent[0] != num_batches || mean->extent[1] != num_cols)) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }
    if (stdev.extent[0] != num_batches || stdev.extent[1] != num_cols) {
        _KERMAC_CHECK_RET( KERMAC_ERROR_INVALID_VALUE );
    }

    const bool a_is_dry     = a.memory.stack_allocator->is_dry;
    const bool mean_is_dry  = has_center ? mean->memory.stack_allocator->is_dry : a_is_dry;
    const bool stdev_is_dry = stdev.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && mean_is_dry && stdev_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !mean_is_dry && !stdev_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    CUfunction function = kermac->functions[_KERMAC_KERNEL_FUNCTION_APPLY_ROW_STATS_F32];

    const int block_size      = APPLY_ROW_STATS_BLOCK_SIZE;
    const int warps_per_block = block_size >> 5;

    const int64_t task_count = num_batches * num_cols;
    if (task_count == 0) {
        return KERMAC_SUCCESS;
    }

    int64_t blocks_needed = (task_count + (int64_t)warps_per_block - 1) / (int64_t)warps_per_block;
    const int64_t max_blocks = 65535;
    const int num_blocks = (int)_KERMAC_MIN(blocks_needed, max_blocks);

    if (!is_dry) {
        void* a_ptr;
        void* mean_ptr = NULL;
        void* stdev_ptr;
        int64_t mean_ld = 0;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        if (has_center) {
            _KERMAC_CHECK_RET( kermac_memory_pointer(mean->memory, &mean_ptr) );
            mean_ld = mean->stride[1];
        }
        _KERMAC_CHECK_RET( kermac_memory_pointer(stdev.memory, &stdev_ptr) );

        void* args[] = {
            (void*)&num_rows,
            (void*)&num_cols,
            (void*)&num_batches,
            (void*)&a_ptr,
            (void*)&a.stride[1],
            (void*)&a.stride[2],
            (void*)&mean_ptr,
            (void*)&mean_ld,
            (void*)&stdev_ptr,
            (void*)&stdev.stride[1],
            (void*)&flags
        };

        _KERMAC_CUDA_CHECK_RET(
            cuLaunchKernel(
                function,
                (unsigned int)num_blocks, 1, 1,
                (unsigned int)block_size, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_apply_row_stats(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor mean,
    KermacTensor stdev,
    uint32_t flags,
    CUstream stream
) {
    return kermac_apply_row_stats_impl(kermac, a, &mean, stdev, flags, stream);
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_apply_stdev_rows(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor stdev,
    CUstream stream
) {
    return kermac_apply_row_stats_impl(
        kermac,
        a,
        NULL,
        stdev,
        KERMAC_ROW_NORMALIZE_FLAG_SCALE,
        stream
    );
}
