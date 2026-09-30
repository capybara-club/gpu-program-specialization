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

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

INCBIN(unsigned char, legacy_kernels_cubin, KERMAC_LEGACY_KERNELS_CUBIN);

static const char * _kermac_legacy_kernel_function_names[] = {
    [_KERMAC_LEGACY_KERNEL_FUNCTION_COPY_DIAGONALS_FULL_F32] =                 "kernel_copy_diagonals_full_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_COPY_TRIANGLES_LOWER_F32] =                "kernel_copy_triangles_lower_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_COPY_TRIANGLES_UPPER_F32] =                "kernel_copy_triangles_upper_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_FULL_SYMM_F32]=                    "kernel_laplace_full_symm_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_FULL_F32] =                        "kernel_laplace_full_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_LOWER_SYMM_COPY_GRAD_F32] =        "kernel_laplace_lower_symm_copy_grad_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_UPPER_SYMM_COPY_GRAD_F32] =        "kernel_laplace_upper_symm_copy_grad_f32",
    [_KERMAC_LEGACY_KERNEL_FUNCTION_POINTER_ARRAY_F32] =                       "kernel_pointer_array_f32"
};
_KERMAC_STATIC_ASSERT(
	_KERMAC_ARRAY_NUM_ELEMS(_kermac_legacy_kernel_function_names) == _KERMAC_LEGACY_KERNEL_FUNCTION_NUM_ENUMS, 
	_kermac_legacy_kernel_function_names
);

KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_create(
    KermacHandle kermac,
    KermacLegacyHandle* legacy
) {
    _KERMAC_CUDA_CHECK_RET( cuModuleLoadDataEx(&legacy->module, g_legacy_kernels_cubin_data, 0, 0, 0) );

    for (int i = 0; i < _KERMAC_LEGACY_KERNEL_FUNCTION_NUM_ENUMS; i++) {
        _KermacLegacyKernelFunction function = (_KermacLegacyKernelFunction)i;
        _KERMAC_CUDA_CHECK_RET(
            cuModuleGetFunction(
                &legacy->functions[function],
                legacy->module,
                _kermac_legacy_kernel_function_names[function]
            )
        );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_destroy(
    KermacLegacyHandle legacy
) {
    _KERMAC_CUDA_CHECK_RET( cuModuleUnload(legacy.module) );
    return KERMAC_SUCCESS;
}


KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_copy_diagonal(
    KermacLegacyHandle legacy,
    KermacTensor symmetric_tensor,
    KermacTensor diagonal_tensor,
    CUstream stream
) {
    size_t num_rows = symmetric_tensor.extent[0];
    size_t ldA = symmetric_tensor.stride[1];
    size_t batch_stride_A = symmetric_tensor.stride[2];
    size_t ldV = diagonal_tensor.stride[1];
    size_t symmetric_tensor_batch = symmetric_tensor.num_modes == 2 ? 1 : symmetric_tensor.extent[2];
    size_t diagonal_tensor_batch = diagonal_tensor.num_modes == 1 ? 1 : diagonal_tensor.extent[1];

    if (symmetric_tensor.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (diagonal_tensor.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (symmetric_tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (diagonal_tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (symmetric_tensor.num_modes != 2 && symmetric_tensor.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (diagonal_tensor.num_modes != 1 && diagonal_tensor.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_rows != symmetric_tensor.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_rows != diagonal_tensor.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (symmetric_tensor_batch != diagonal_tensor_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool symmetric_tensor_is_dry = symmetric_tensor.memory.stack_allocator->is_dry;
    bool diagonal_tensor_is_dry = diagonal_tensor.memory.stack_allocator->is_dry;
    bool is_dry;


    if (symmetric_tensor_is_dry && diagonal_tensor_is_dry) {
        is_dry = true;
    } else if (!symmetric_tensor_is_dry && !diagonal_tensor_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* symmetric_tensor_ptr;
        void* diagonal_tensor_ptr;
        
        _KERMAC_CHECK_RET( kermac_memory_pointer(symmetric_tensor.memory, &symmetric_tensor_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(diagonal_tensor.memory, &diagonal_tensor_ptr) );

        void* args[] = {
            (void*)&num_rows,
            (void*)&symmetric_tensor_ptr,
            (void*)&ldA,
            (void*)&batch_stride_A,
            (void*)&diagonal_tensor_ptr,
            (void*)&ldV
        };

        static const int block_size = 512;
        const int num_blocks = _KERMAC_NEAREST_LARGER_MULTIPLE(num_rows, block_size);
        const int batch = symmetric_tensor_batch;

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_COPY_DIAGONALS_FULL_F32],
                num_blocks, 1, batch,
                block_size, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

#define _KERMAC_KERNEL_TILE_DIM 16

KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_copy_triangle(
    KermacLegacyHandle legacy,
    KermacMatrixPackedType packed_type,
    KermacTensor symmetric_tensor,
    float diagonal_value,
    CUstream stream
) {
    size_t num_rows = symmetric_tensor.extent[0];
    size_t ldA = symmetric_tensor.stride[1];
    size_t batch_stride_A = symmetric_tensor.stride[2];
    size_t batch = symmetric_tensor.num_modes == 2 ? 1 : symmetric_tensor.extent[2];

    if (symmetric_tensor.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (symmetric_tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (num_rows != symmetric_tensor.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (symmetric_tensor.num_modes != 2 && symmetric_tensor.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    CUfunction function;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_COPY_TRIANGLES_LOWER_F32];
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_COPY_TRIANGLES_UPPER_F32];
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool symmetric_tensor_is_dry = symmetric_tensor.memory.stack_allocator->is_dry;
    if (!symmetric_tensor_is_dry) {
        void* symmetric_tensor_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(symmetric_tensor.memory, &symmetric_tensor_ptr) );

        const int num_blocks = _KERMAC_NEAREST_LARGER_MULTIPLE(num_rows, _KERMAC_KERNEL_TILE_DIM);

        void *args[] = {
            (void*)&num_rows,
            (void*)&diagonal_value,
            (void*)&symmetric_tensor_ptr,
            (void*)&ldA,
            (void*)&batch_stride_A
        };

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                function,
                num_blocks, num_blocks, batch,
                _KERMAC_KERNEL_TILE_DIM, _KERMAC_KERNEL_TILE_DIM, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_laplace(
    KermacLegacyHandle legacy,
    KermacTensor tensor,
    KermacTensor norm_x,
    KermacTensor norm_y,
    float bandwidth,
    float epsilon,
    CUstream stream
) { 
    if (tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (norm_x.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (norm_x.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (tensor.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (norm_x.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (norm_x.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    size_t tensor_num_rows = tensor.extent[0];
    size_t tensor_num_cols = tensor.extent[1];
    size_t tensor_batch = tensor.num_modes == 2 ? 1 : tensor.extent[2];

    size_t norm_x_num_rows = norm_x.extent[0];
    size_t norm_x_batch = norm_x.num_modes == 1 ? 1 : norm_x.extent[1];

    size_t norm_y_num_rows = norm_y.extent[0];
    size_t norm_y_batch = norm_y.num_modes == 1 ? 1 : norm_y.extent[1];

    if (tensor_num_rows != norm_x_num_rows) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (tensor_num_cols != norm_y_num_rows) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (tensor_batch != norm_x_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (tensor_batch != norm_y_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    CUfunction function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_FULL_F32];

    bool tensor_is_dry = tensor.memory.stack_allocator->is_dry;
    bool norm_x_is_dry = norm_x.memory.stack_allocator->is_dry;
    bool norm_y_is_dry = norm_y.memory.stack_allocator->is_dry;

    bool is_dry;
    if (tensor_is_dry && norm_x_is_dry && norm_y_is_dry) {
        is_dry = true;
    } else if (!tensor_is_dry && !norm_x_is_dry && !norm_y_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* tensor_ptr;
        void* norm_x_ptr;
        void* norm_y_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(tensor.memory, &tensor_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(norm_x.memory, &norm_x_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(norm_y.memory, &norm_y_ptr) );

        size_t tensor_batch_stride = tensor.stride[2];
        size_t tensor_ld = tensor.stride[1];
        size_t offset = 0;
        float regularization = 0.0f;

        size_t norm_x_ld = norm_x.stride[1];
        size_t norm_y_ld = norm_y.stride[1];

        void *args[] = {
            (void*)&tensor_num_rows,
            (void*)&tensor_num_cols,
            (void*)&tensor_ptr,
            (void*)&tensor_ld,
            (void*)&tensor_batch_stride,
            (void*)&norm_x_ptr,
            (void*)&norm_x_ld,
            (void*)&norm_y_ptr,
            (void*)&norm_y_ld,
            (void*)&bandwidth,
            (void*)&regularization,
            (void*)&epsilon,
            (void*)&offset,
            (void*)&offset
        };

        const int num_blocks_M = _KERMAC_NEAREST_LARGER_MULTIPLE(tensor_num_rows, _KERMAC_KERNEL_TILE_DIM);
        const int num_blocks_N = _KERMAC_NEAREST_LARGER_MULTIPLE(tensor_num_cols, _KERMAC_KERNEL_TILE_DIM);

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                function,
                num_blocks_M, num_blocks_N, tensor_batch,
                _KERMAC_KERNEL_TILE_DIM, _KERMAC_KERNEL_TILE_DIM, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_legacy_laplace_symmetric(
    KermacLegacyHandle legacy,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator *dsa,
    KermacTensor tensor,
    float bandwidth,
    float regularization,
    float epsilon,
    CUstream stream,
    KermacTensor* norm
) {
    int64_t num_rows = tensor.extent[0];
    int64_t ldA = tensor.stride[1];
    int64_t batch_stride_A = tensor.stride[2];
    int64_t batch = tensor.num_modes == 2 ? 1 : tensor.extent[2];

    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (num_rows != tensor.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (tensor.num_modes != 2 && tensor.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    CUfunction function;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_LOWER_SYMM_COPY_GRAD_F32];
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_UPPER_SYMM_COPY_GRAD_F32];
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_FULL) {
        function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_FULL_SYMM_F32];
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            norm,
            KERMAC_DATA_TYPE_FLOAT,
            (KermacExtent){num_rows, batch},
            dsa
        )
    );

    _KERMAC_CHECK_RET(
        kermac_legacy_copy_diagonal(
            legacy,
            tensor,
            *norm,
            stream
        )
    );

    size_t ld_norm = norm->stride[1];
    size_t offset = 0;

    bool dsa_is_dry = dsa->is_dry;
    bool tensor_is_dry = tensor.memory.stack_allocator->is_dry;
    bool norm_is_dry = norm->memory.stack_allocator->is_dry;
    bool is_dry;

    if (dsa_is_dry && tensor_is_dry && norm_is_dry) {
        is_dry = true;
    } else if (!dsa_is_dry && !tensor_is_dry && !norm_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* tensor_ptr;
        void* norm_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(tensor.memory, &tensor_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(norm->memory, &norm_ptr) );

        void *args[] = {
            (void*)&num_rows,
            (void*)&num_rows,
            (void*)&tensor_ptr,
            (void*)&ldA,
            (void*)&batch_stride_A,
            (void*)&norm_ptr,
            (void*)&ld_norm,
            (void*)&norm_ptr,
            (void*)&ld_norm,
            (void*)&bandwidth,
            (void*)&regularization,
            (void*)&epsilon,
            (void*)&offset,
            (void*)&offset
        };

        const int num_blocks_M = _KERMAC_NEAREST_LARGER_MULTIPLE(num_rows, _KERMAC_KERNEL_TILE_DIM);
        const int num_blocks_N = _KERMAC_NEAREST_LARGER_MULTIPLE(num_rows, _KERMAC_KERNEL_TILE_DIM);

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                function,
                num_blocks_M, num_blocks_N, batch,
                _KERMAC_KERNEL_TILE_DIM, _KERMAC_KERNEL_TILE_DIM, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_laplace_symmetric(
    KermacLegacyHandle legacy,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator *dsa,
    KermacTensor tensor,
    float bandwidth,
    float regularization,
    float epsilon,
    CUstream stream
) {
    KermacTensor norm = KERMAC_ZERO_INIT(KermacTensor);
    KermacResult result = _kermac_legacy_laplace_symmetric(
        legacy, packed_type, dsa, tensor,
        bandwidth, regularization, epsilon, stream,
        &norm
    );
    if (norm.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(norm) );
    }
    return result;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_solve_compute_array(
    KermacLegacyHandle legacy,
    KermacTensor a,
    KermacTensor a_array,
    CUstream stream
) {
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a_array.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a_array.num_modes != 0 && a_array.num_modes != 1 && a_array.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a_array.data_type != KERMAC_DATA_TYPE_POINTER) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    bool has_labels = a_array.num_modes == 2;
    if (!has_labels) {
        if (a.num_modes != 2 && a.num_modes != 3) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else {
        if (a.num_modes != 2 && a.num_modes != 3 && a.num_modes != 4) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    int64_t a_batch = 1;
    int64_t a_labels = 1;
    int64_t a_batch_stride = 0;
    int64_t a_label_stride = 0;

    if (!has_labels) {
        a_batch = a.num_modes == 2 ? 1 : a.extent[2];
        a_batch_stride = a.stride[2];
        int64_t a_array_batch = a_array.num_modes == 0 ? 1 : a_array.extent[0];
        if (a_batch != a_array_batch) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else {
        bool labels_in_columns = false;
        if (a.num_modes == 2) {
            labels_in_columns = true;
        } else if (a.num_modes == 3) {
            labels_in_columns = (a.extent[1] != 1);
        }

        if (labels_in_columns) {
            a_labels = a.extent[1];
            if (a.num_modes == 3) {
                a_batch = a.extent[2];
                a_batch_stride = a.stride[2];
            } else {
                a_batch = 1;
                a_batch_stride = 0;
            }
            a_label_stride = a.stride[1];
        } else {
            if (a.num_modes == 3) {
                a_batch = 1;
                a_labels = a.extent[2];
                a_batch_stride = a.stride[2];
                a_label_stride = a.stride[2];
            } else {
                a_batch = a.extent[2];
                a_labels = a.extent[3];
                a_batch_stride = a.stride[2];
                a_label_stride = a.stride[3];
            }
        }
        if (a_array.extent[0] != a_batch || a_array.extent[1] != a_labels) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool a_array_is_dry = a_array.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && a_array_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !a_array_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    CUfunction function = legacy.functions[_KERMAC_LEGACY_KERNEL_FUNCTION_POINTER_ARRAY_F32];

    if (!is_dry) {
        float* a_ptr;
        float** a_array_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, (void**)&a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(a_array.memory, (void**)&a_array_ptr) );

        unsigned int threads = 256;
        unsigned int blocks  = (a_batch + threads - 1) / threads;
        int64_t a_array_label_stride = has_labels ? a_array.stride[1] : 0;

        for (int64_t label = 0; label < a_labels; ++label) {
            float* a_ptr_label = a_ptr;
            float** a_array_ptr_label = a_array_ptr;
            if (has_labels) {
                a_ptr_label = a_ptr + label * a_label_stride;
                a_array_ptr_label = a_array_ptr + label * a_array_label_stride;
            }
            void* args[] = {
                (void*)&a_ptr_label,
                (void*)&a_array_ptr_label,
                (void*)&a_batch_stride,
                (void*)&a_batch
            };

            _KERMAC_CUDA_CHECK_RET( 
                cuLaunchKernel(
                    function,
                    blocks, 1, 1,
                    threads, 1, 1,
                    0, stream,
                    args, 0
                )
            );
        }
    }

    return KERMAC_SUCCESS;
}

static
KermacResult
_kermac_legacy_potrs_batched_trsm(
    KermacHandle kermac,
    KermacMatrixPackedType packed_type,
    KermacTensor a,
    KermacTensor b,
    KermacTensor a_array,
    KermacTensor b_array,
    KermacTensor solve_info,
    CUstream stream
)
{
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a_array.data_type != KERMAC_DATA_TYPE_POINTER) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b_array.data_type != KERMAC_DATA_TYPE_POINTER) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (solve_info.data_type != KERMAC_DATA_TYPE_INT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a_array.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b_array.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (solve_info.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (a.num_modes != 2 && a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (b.num_modes != 2 && b.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (solve_info.num_modes != 1 || solve_info.extent[0] != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = a.extent[0];
    int64_t a_batch  = (a.num_modes == 2) ? 1 : a.extent[2];
    int64_t ldA      = a.stride[1];
    int64_t ldB      = b.stride[1];
    int64_t num_rhs  = b.extent[1];
    int64_t b_batch  = (b.num_modes == 2) ? 1 : b.extent[2];

    if (b.num_modes != a.num_modes) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (num_rows != a.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_rows != b.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a_batch != b_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t a_array_batch = (a_array.num_modes == 0) ? 1 : a_array.extent[0];
    if (a_array.num_modes != 0 && a_array.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a_array_batch != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (b_array.num_modes != 0 && b_array.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    {
        int64_t b_array_batch = (b_array.num_modes == 0) ? 1 : b_array.extent[0];
        if (b_array_batch != b_batch) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    if (num_rows > INT_MAX || num_rhs > INT_MAX || ldA > INT_MAX || ldB > INT_MAX || a_batch > INT_MAX) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    cublasFillMode_t uplo;
    cublasOperation_t first_op;
    cublasOperation_t second_op;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_LOWER;
        first_op = CUBLAS_OP_N;
        second_op = CUBLAS_OP_T;
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_UPPER;
        first_op = CUBLAS_OP_T;
        second_op = CUBLAS_OP_N;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool a_array_is_dry = a_array.memory.stack_allocator->is_dry;
    bool b_array_is_dry = b_array.memory.stack_allocator->is_dry;
    bool solve_info_is_dry = solve_info.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry && a_array_is_dry && b_array_is_dry && solve_info_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !a_array_is_dry && !b_array_is_dry && !solve_info_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    cublasHandle_t cublas = kermac->cublas;
    cudaStream_t previous_cublas_stream = NULL;
    _KERMAC_CUBLAS_CHECK_RET( cublasGetStream(cublas, &previous_cublas_stream) );
    _KERMAC_CUBLAS_CHECK_RET( cublasSetStream(cublas, stream) );

    if (!is_dry) {
        float** d_Aarray = NULL;
        float** d_Barray = NULL;
        int* d_solve_info = NULL;
        _KERMAC_CHECK_RET( kermac_memory_pointer(a_array.memory, (void**)&d_Aarray) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b_array.memory, (void**)&d_Barray) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(solve_info.memory, (void**)&d_solve_info) );

        _KERMAC_CUDA_CHECK_RET( cuMemsetD32Async((CUdeviceptr)d_solve_info, 0u, 1u, stream) );

        const float alpha = 1.0f;
        int n = (int)num_rows;
        int nrhs = (int)num_rhs;
        int lda_i = (int)ldA;
        int ldb_i = (int)ldB;
        int batch = (int)a_batch;

        _KERMAC_CUBLAS_CHECK_RET(
            cublasStrsmBatched(
                cublas,
                CUBLAS_SIDE_LEFT,
                uplo,
                first_op,
                CUBLAS_DIAG_NON_UNIT,
                n,
                nrhs,
                &alpha,
                (const float* const*)d_Aarray,
                lda_i,
                d_Barray,
                ldb_i,
                batch
            )
        );

        _KERMAC_CUBLAS_CHECK_RET(
            cublasStrsmBatched(
                cublas,
                CUBLAS_SIDE_LEFT,
                uplo,
                second_op,
                CUBLAS_DIAG_NON_UNIT,
                n,
                nrhs,
                &alpha,
                (const float* const*)d_Aarray,
                lda_i,
                d_Barray,
                ldb_i,
                batch
            )
        );
    }

    _KERMAC_CUBLAS_CHECK_RET( cublasSetStream(cublas, previous_cublas_stream) );
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_legacy_solve(
    KermacHandle kermac,
    KermacMatrixPackedType packed_type,
    KermacTensor a,
    KermacTensor b,
    KermacTensor a_array,   // device array of pointers to A_i
    KermacTensor b_array,   // device array of pointers to B_i
    KermacTensor factor_info, // int[batch]
    KermacTensor solve_info,  // int[1]  (scalar info from potrsBatched)
    CUstream stream
)
{
    // ---- Type checks -------------------------------------------------------
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a_array.data_type != KERMAC_DATA_TYPE_POINTER) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b_array.data_type != KERMAC_DATA_TYPE_POINTER) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (factor_info.data_type != KERMAC_DATA_TYPE_INT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (solve_info.data_type != KERMAC_DATA_TYPE_INT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    // ---- Memory space checks ----------------------------------------------
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a_array.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b_array.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (factor_info.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (solve_info.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    // ---- Mode / extent checks ---------------------------------------------
    if (a.num_modes != 2 && a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (b.num_modes != 2 && b.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (factor_info.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (solve_info.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = a.extent[0];
    int64_t a_batch  = (a.num_modes == 2) ? 1 : a.extent[2];
    int64_t ldA      = a.stride[1];
    int64_t ldB      = b.stride[1];
    int64_t num_rhs  = b.extent[1];
    int64_t b_batch  = (b.num_modes == 2) ? 1 : b.extent[2];

    if (b.num_modes != a.num_modes) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (num_rows != a.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE ); // A must be square
    }
    if (num_rows != b.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE ); // A rows == B rows
    }
    if (a_batch != b_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    // Pointer arrays can be 0D (single pointer) or 1D (array of pointers)
    int64_t a_array_batch = (a_array.num_modes == 0) ? 1 : a_array.extent[0];
    if (a_array.num_modes != 0 && a_array.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (a_array_batch != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (b_array.num_modes != 0 && b_array.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    {
        int64_t b_array_batch = (b_array.num_modes == 0) ? 1 : b_array.extent[0];
        if (b_array_batch != b_batch) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    if (factor_info.extent[0] != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (solve_info.extent[0] != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    // Basic int-range sanity (optional; mirrors style of other checks)
    if (num_rows > INT_MAX || num_rhs > INT_MAX ||
        ldA      > INT_MAX || ldB    > INT_MAX ||
        a_batch  > INT_MAX) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    // ---- UPLO mapping ------------------------------------------------------
    cublasFillMode_t uplo;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_LOWER;
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_UPPER;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    // ---- Dry-run consistency -----------------------------------------------
    bool a_is_dry           = a.memory.stack_allocator->is_dry;
    bool b_is_dry           = b.memory.stack_allocator->is_dry;
    bool a_array_is_dry     = a_array.memory.stack_allocator->is_dry;
    bool b_array_is_dry     = b_array.memory.stack_allocator->is_dry;
    bool factor_info_is_dry = factor_info.memory.stack_allocator->is_dry;
    bool solve_info_is_dry  = solve_info.memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry && a_array_is_dry && b_array_is_dry &&
        factor_info_is_dry && solve_info_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !a_array_is_dry && !b_array_is_dry &&
               !factor_info_is_dry && !solve_info_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    CUstream previous_cusolver_stream;
    _KERMAC_CUSOLVER_CHECK_RET(
        cusolverDnGetStream(kermac->cusolver, &previous_cusolver_stream)
    );
    _KERMAC_CUSOLVER_CHECK_RET(
        cusolverDnSetStream(kermac->cusolver, stream)
    );

    if (!is_dry) {
        float **d_Aarray = NULL;
        float **d_Barray = NULL;
        void   *d_factor_info = NULL;
        void   *d_solve_info = NULL;
        _KERMAC_CHECK_RET( kermac_memory_pointer(a_array.memory, (void**)&d_Aarray) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(b_array.memory, (void**)&d_Barray) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(factor_info.memory, &d_factor_info) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(solve_info.memory, &d_solve_info) );

        int n      = (int)num_rows;
        int nrhs   = (int)num_rhs;
        int lda_i  = (int)ldA;
        int ldb_i  = (int)ldB;
        int batch  = (int)a_batch;

        _KERMAC_CUSOLVER_CHECK_RET(
            cusolverDnSpotrfBatched(
                kermac->cusolver,
                uplo,
                n,
                d_Aarray,
                lda_i,
                d_factor_info,
                batch
            )
        );

        if (nrhs == 1) {
            _KERMAC_CUSOLVER_CHECK_RET(
                cusolverDnSpotrsBatched(
                    kermac->cusolver,
                    uplo,
                    n,
                    nrhs,
                    d_Aarray,
                    lda_i,
                    d_Barray,
                    ldb_i,
                    d_solve_info,
                    batch
                )
            );
        }
    }

    _KERMAC_CUSOLVER_CHECK_RET(
        cusolverDnSetStream(kermac->cusolver, previous_cusolver_stream)
    );

    if (num_rhs > 1) {
        _KERMAC_CHECK_RET(
            _kermac_legacy_potrs_batched_trsm(
                kermac,
                packed_type,
                a,
                b,
                a_array,
                b_array,
                solve_info,
                stream
            )
        );
    }

    return KERMAC_SUCCESS;
}
