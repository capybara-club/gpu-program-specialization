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
_kermac_mse_accuracy(
    KermacHandle kermac,
    KermacStackAllocator* dsa,
    KermacTensor a,         // M,N,B
    KermacTensor b,         // M,N,B
    KermacTensor mse,       // B
    KermacTensor* accuracy, // B
    CUstream stream,
    KermacTensor* temp_mse_sum_tensor_ptr,
    KermacTensor* temp_accuracy_counts_tensor_ptr
) {
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (mse.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (mse.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (accuracy != NULL) {
        if (accuracy->data_type != KERMAC_DATA_TYPE_FLOAT) {
            _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
        }
    }

    if (mse.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (accuracy != NULL) {
        if (accuracy->num_modes != 1) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    if (a.num_modes < 1 || b.num_modes < 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t a_batch = a.num_modes < 3 ? 1 : a.extent[a.num_modes-1];
    int64_t b_batch = b.num_modes < 3 ? 1 : b.extent[b.num_modes-1];
    int64_t mse_batch = mse.extent[mse.num_modes-1];
    int64_t accuracy_batch = accuracy == NULL ? mse_batch : accuracy->extent[accuracy->num_modes-1];

    int64_t a_num_rows = a.extent[0];
    int64_t b_num_rows = b.extent[0];

    if (a_num_rows != b_num_rows) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t a_effective_columns = _kermac_calculate_effective_cols(a) / a_batch;
    int64_t b_effective_columns = _kermac_calculate_effective_cols(b) / b_batch;
    if (a_effective_columns != b_effective_columns) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (a_num_rows <= 0 || a_effective_columns <= 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (b_batch != 1 && b_batch != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (mse_batch != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (accuracy != NULL && accuracy_batch != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool mse_is_dry = mse.memory.stack_allocator->is_dry;
    bool accuracy_is_dry = accuracy == NULL ? mse_is_dry : accuracy->memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry && mse_is_dry && accuracy_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !mse_is_dry && !accuracy_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    CUfunction function_mse;
    CUfunction function_mse_final;
    if (accuracy == NULL) {
        function_mse = kermac->functions[_KERMAC_KERNEL_FUNCTION_MSE_256_F32];
        function_mse_final = kermac->functions[_KERMAC_KERNEL_FUNCTION_MSE_FINAL_REDUCTION_1024_F32];
    } else {
        function_mse = kermac->functions[_KERMAC_KERNEL_FUNCTION_MSE_ACCURACY_256_F32];
        function_mse_final = kermac->functions[_KERMAC_KERNEL_FUNCTION_MSE_FINAL_REDUCTION_1024_ACCURACY_F32];
    }

    static const int block_size = 256;
    static const int final_block_size = 1024;

    const int num_blocks = _KERMAC_NEAREST_LARGER_MULTIPLE(a_num_rows, block_size);

    _KERMAC_CHECK_RET( 
        kermac_tensor_create(
            temp_mse_sum_tensor_ptr, 
            KERMAC_DATA_TYPE_FLOAT, 
            (KermacExtent){num_blocks, a_batch}, 
            dsa
        )
    );

    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            temp_accuracy_counts_tensor_ptr, 
            KERMAC_DATA_TYPE_INT, 
            (KermacExtent){num_blocks, a_batch}, 
            dsa
        )
    );

    if (!is_dry) {
        int64_t a_ld = a.stride[1];
        int64_t batch_stride_a = a_batch == 1 ? 0 : a.stride[a.num_modes-1];
        int64_t b_ld = b.stride[1];
        int64_t batch_stride_b = b_batch == 1 ? 0 : b.stride[b.num_modes-1];
        int64_t mse_ld = mse.stride[1];
        int64_t accuracy_ld = accuracy == NULL ? 0 : accuracy->stride[1];
        int64_t temp_mse_sum_ld = temp_mse_sum_tensor_ptr->stride[1];
        int64_t temp_accuracy_counts_ld = temp_accuracy_counts_tensor_ptr->stride[1];

        void* a_ptr;
        void* b_ptr;
        void* mse_ptr;
        void* accuracy_ptr;
        void* temp_mse_sum_ptr;
        void* temp_accuracy_counts_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b.memory, &b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(mse.memory, &mse_ptr) );
        if (accuracy != NULL) {
            _KERMAC_CHECK_RET( kermac_memory_pointer(accuracy->memory, &accuracy_ptr) );
        } else {
            accuracy_ptr = NULL;
        }
        _KERMAC_CHECK_RET( kermac_memory_pointer(temp_mse_sum_tensor_ptr->memory, &temp_mse_sum_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(temp_accuracy_counts_tensor_ptr->memory, &temp_accuracy_counts_ptr) );

        void* args_mse[] = {
            (void*)&a_num_rows,
            (void*)&a_effective_columns,
            (void*)&a_ptr,
            (void*)&a_ld,
            (void*)&batch_stride_a,
            (void*)&b_ptr,
            (void*)&b_ld,
            (void*)&batch_stride_b,
            (void*)&temp_mse_sum_ptr,
            (void*)&temp_mse_sum_ld,
            (void*)&temp_accuracy_counts_ptr,
            (void*)&temp_accuracy_counts_ld
        };

        void* args_mse_final[] = {
            (void*)&a_num_rows,
            (void*)&a_effective_columns,
            (void*)&num_blocks,
            (void*)&temp_mse_sum_ptr,
            (void*)&temp_mse_sum_ld,
            (void*)&temp_accuracy_counts_ptr,
            (void*)&temp_accuracy_counts_ld,
            (void*)&mse_ptr,
            (void*)&mse_ld,
            (void*)&accuracy_ptr,
            (void*)&accuracy_ld
        };

        _KERMAC_CUDA_CHECK_RET(
            cuLaunchKernel(
                function_mse,
                num_blocks, 1, a_batch,
                block_size, 1, 1,
                0, stream,
                args_mse, 0
            )
        );

        _KERMAC_CUDA_CHECK_RET(
            cuLaunchKernel(
                function_mse_final,
                1, 1, a_batch,
                final_block_size, 1, 1,
                0, stream,
                args_mse_final, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_mse(
    KermacHandle kermac,
    KermacStackAllocator* dsa,
    KermacTensor a,         // *,B
    KermacTensor b,         // *,B
    KermacTensor mse,       // B
    CUstream stream
) {
    // KermacTensor accuracy = {0};
    KermacTensor temp_mse_sum_tensor_ptr = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor temp_accuracy_counts_tensor_ptr = KERMAC_ZERO_INIT(KermacTensor);

    KermacResult result =
        _kermac_mse_accuracy(
            kermac,
            dsa, a, b, mse, NULL, stream,
            &temp_mse_sum_tensor_ptr, &temp_accuracy_counts_tensor_ptr
        );
    
    if (temp_accuracy_counts_tensor_ptr.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(temp_accuracy_counts_tensor_ptr) );
    }
    if (temp_mse_sum_tensor_ptr.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(temp_mse_sum_tensor_ptr) );
    }

    return result;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_mse_accuracy(
    KermacHandle kermac,
    KermacStackAllocator* dsa,
    KermacTensor a,         // *,B
    KermacTensor b,         // *,B
    KermacTensor mse,       // B
    KermacTensor accuracy,  // B
    CUstream stream
) {
    KermacTensor temp_mse_sum_tensor_ptr = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor temp_accuracy_counts_tensor_ptr = KERMAC_ZERO_INIT(KermacTensor);

    KermacResult result =
        _kermac_mse_accuracy(
            kermac,
            dsa, a, b, mse, &accuracy, stream,
            &temp_mse_sum_tensor_ptr, &temp_accuracy_counts_tensor_ptr
        );
    
    if (temp_accuracy_counts_tensor_ptr.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(temp_accuracy_counts_tensor_ptr) );
    }
    if (temp_mse_sum_tensor_ptr.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(temp_mse_sum_tensor_ptr) );
    }

    return result;
}
