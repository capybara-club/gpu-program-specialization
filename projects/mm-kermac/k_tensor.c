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

#include <string.h>

KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_create(
    KermacTensor* tensor,
    KermacDataType data_type,
    const int64_t* extents,
    KermacStackAllocator* stack_allocator
) {
    size_t alignment_bytes = _kermac_memory_space_alignments[stack_allocator->memory_space];
    size_t data_type_size = _kermac_data_type_sizes[data_type];
    uint32_t num_modes = 0;
    for (int i = 0; i < 4; i++) {
        int64_t extent = extents[i];
        tensor->extent[i] = extent;
        if (extent == 0) {
            break;
        }
        if (extent < 0) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
        num_modes++;
    }

    size_t data_type_alignment = _KERMAC_TENSOR_LD_ALIGNMENT_BYTES / data_type_size;
    size_t ld_rows = data_type_alignment * _KERMAC_NEAREST_LARGER_MULTIPLE(extents[0], data_type_alignment);

    tensor->num_modes = num_modes;
    tensor->extent[0] = 0 < num_modes ? extents[0] : 0;
    tensor->extent[1] = 1 < num_modes ? extents[1] : 0;
    tensor->extent[2] = 2 < num_modes ? extents[2] : 0;
    tensor->extent[3] = 3 < num_modes ? extents[3] : 0;
    tensor->stride[0] = 1;
    tensor->stride[1] = ld_rows;
    tensor->stride[2] = ld_rows * tensor->extent[1];
    tensor->stride[3] = ld_rows * tensor->extent[1] * tensor->extent[2];

    // Note: The following stride and num_bytes calculations do not check for
    // integer overflow. Tensor dimensions would need to be unrealistically
    // large (exceeding SIZE_MAX) to overflow. This follows the convention
    // of libraries like cuBLAS which similarly do not guard against overflow.
    size_t num_bytes;
    if (num_modes == 0) {
        num_bytes = data_type_size;
    } else if (num_modes == 1) {
        num_bytes = data_type_size * ld_rows;
    } else if (num_modes == 2) {
        num_bytes = data_type_size * ld_rows * tensor->extent[1];
    } else if (num_modes == 3) {
        num_bytes = data_type_size * ld_rows * tensor->extent[1] * tensor->extent[2];
    } else if (num_modes == 4) {
        num_bytes = data_type_size * ld_rows * tensor->extent[1] * tensor->extent[2] * tensor->extent[3];
    } else {
        // Should not happen due to "< 4" in above for loop
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    _KERMAC_CHECK_RET(
        _kermac_memory_create(
            &tensor->memory,
            stack_allocator,
            num_bytes,
            alignment_bytes
        )
    );

    tensor->data_type = data_type;

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_destroy(
    KermacTensor tensor
) {
    _KERMAC_CHECK_RET( _kermac_memory_destroy(tensor.memory) );
    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_tensor_set(
    KermacTensor host_tensor_src,
    KermacTensor device_tensor_dst,
    CUstream stream
) {
    if (host_tensor_src.data_type != device_tensor_dst.data_type) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (host_tensor_src.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (device_tensor_dst.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows_src = host_tensor_src.num_modes == 0 ? 1 : host_tensor_src.extent[0];
    int64_t num_cols_src = _kermac_calculate_effective_cols(host_tensor_src);

    int64_t num_rows_dst = device_tensor_dst.num_modes == 0 ? 1 : device_tensor_dst.extent[0];
    int64_t num_cols_dst = _kermac_calculate_effective_cols(device_tensor_dst);

    if (num_rows_src != num_rows_dst || num_cols_src != num_cols_dst) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    size_t elem_size = _kermac_data_type_sizes[host_tensor_src.data_type];

    int64_t ld_tensor_src = host_tensor_src.num_modes == 0 ? 1 : host_tensor_src.stride[1];
    int64_t ld_tensor_dst = device_tensor_dst.num_modes == 0 ? 1 : device_tensor_dst.stride[1];

    bool host_tensor_is_dry = host_tensor_src.memory.stack_allocator->is_dry;
    bool device_tensor_is_dry = device_tensor_dst.memory.stack_allocator->is_dry;
    bool is_dry;
    if (host_tensor_is_dry && device_tensor_is_dry) {
        is_dry = true;
    } else if (!host_tensor_is_dry && !device_tensor_is_dry) {
        is_dry = false;
    } else if (!host_tensor_is_dry && device_tensor_is_dry) {
        is_dry = true;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* host_tensor_ptr;
        void* device_tensor_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(host_tensor_src.memory, &host_tensor_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(device_tensor_dst.memory, &device_tensor_ptr) );

        cublasStatus_t status = 
            cublasSetMatrixAsync(
                num_rows_src, num_cols_src,
                elem_size,
                host_tensor_ptr, ld_tensor_src,
                device_tensor_ptr, ld_tensor_dst,
                stream
            );
        if (status != CUBLAS_STATUS_SUCCESS) {
            _KERMAC_ERROR( KERMAC_ERROR_CUBLAS );
        }
    }

    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_tensor_device_to_device(
    KermacTensor device_tensor_src,
    KermacTensor device_tensor_dst,
    CUstream stream
) {
    if (device_tensor_src.data_type != device_tensor_dst.data_type) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (device_tensor_src.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (device_tensor_dst.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows_src = device_tensor_src.num_modes == 0 ? 1 : device_tensor_src.extent[0];
    int64_t num_cols_src = _kermac_calculate_effective_cols(device_tensor_src);

    int64_t num_rows_dst = device_tensor_dst.num_modes == 0 ? 1 : device_tensor_dst.extent[0];
    int64_t num_cols_dst = _kermac_calculate_effective_cols(device_tensor_dst);

    if (num_rows_src != num_rows_dst || num_cols_src != num_cols_dst) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    size_t elem_size = _kermac_data_type_sizes[device_tensor_src.data_type];

    int64_t ld_tensor_src = device_tensor_src.num_modes == 0 ? 1 : device_tensor_src.stride[1];
    int64_t ld_tensor_dst = device_tensor_dst.num_modes == 0 ? 1 : device_tensor_dst.stride[1];

    bool device_tensor_src_is_dry = device_tensor_src.memory.stack_allocator->is_dry;
    bool device_tensor_dst_is_dry = device_tensor_dst.memory.stack_allocator->is_dry;
    bool is_dry;
    if (device_tensor_src_is_dry && device_tensor_dst_is_dry) {
        is_dry = true;
    } else if (!device_tensor_src_is_dry && !device_tensor_dst_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* device_tensor_src_ptr;
        void* device_tensor_dst_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(device_tensor_src.memory, &device_tensor_src_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(device_tensor_dst.memory, &device_tensor_dst_ptr) );

        CUDA_MEMCPY2D cuda_memcpy_2d = {0};

        cuda_memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cuda_memcpy_2d.srcDevice = (CUdeviceptr)device_tensor_src_ptr;
        cuda_memcpy_2d.srcPitch = ld_tensor_src * elem_size;

        cuda_memcpy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cuda_memcpy_2d.dstDevice = (CUdeviceptr)device_tensor_dst_ptr;
        cuda_memcpy_2d.dstPitch = ld_tensor_dst * elem_size;

        cuda_memcpy_2d.WidthInBytes = num_rows_dst * elem_size;
        cuda_memcpy_2d.Height = num_cols_dst;
        _KERMAC_CUDA_CHECK_RET( cuMemcpy2DAsync(&cuda_memcpy_2d, stream) );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_copy(
    KermacTensor src,
    KermacTensor dst,
    CUstream stream
) {
    int64_t num_effective_modes;
    bool broadcast = false;
    if (src.num_modes == dst.num_modes) {
        if (src.extent[src.num_modes-1] == 1 && dst.extent[dst.num_modes-1] != 1) {
            num_effective_modes = src.num_modes - 1;
            broadcast = true;
        } else if (src.extent[src.num_modes-1] == dst.extent[dst.num_modes-1]) {
            num_effective_modes = src.num_modes;
            broadcast = false;
        } else {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else if (src.num_modes + 1 == dst.num_modes) {
        num_effective_modes = src.num_modes;
        broadcast = true;
    } else if (src.num_modes == dst.num_modes + 1) {
        num_effective_modes = dst.num_modes;
        broadcast = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    for (uint32_t mode = 0; mode < num_effective_modes; mode++) {
        if (src.extent[mode] != dst.extent[mode]) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    if (broadcast) {
        int64_t L = dst.extent[dst.num_modes-1];
        for (int64_t l = 0; l < L; l++) {
            KermacTensor dst_view;
            _KERMAC_CHECK_RET( kermac_tensor_view_create(dst, &dst_view, dst.num_modes-1, l) );
            KermacResult res = kermac_tensor_copy(src, dst_view, stream);
            KermacResult destroy_res = kermac_tensor_view_destroy(dst_view);
            _KERMAC_CHECK_RET( res );
            _KERMAC_CHECK_RET( destroy_res );
        }
    } else {
        bool src_is_host = src.memory.stack_allocator->memory_space == KERMAC_MEMORY_SPACE_HOST;
        bool dst_is_host = dst.memory.stack_allocator->memory_space == KERMAC_MEMORY_SPACE_HOST;

        if (src_is_host && !dst_is_host) {
            return _kermac_tensor_set(src, dst, stream);
        } else if (!src_is_host && dst_is_host) {
            return _kermac_tensor_get(src, dst, stream);
        } else if (!src_is_host && !dst_is_host) {
            return _kermac_tensor_device_to_device(src, dst, stream);
        } else {
            return KERMAC_ERROR_INVALID_VALUE;
        }
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult 
kermac_tensor_broadcast_copy(
    KermacHandle handle,
    KermacTensor src, 
    KermacTensor dst, 
    CUstream stream
) {
    if (src.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (dst.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (src.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (dst.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (src.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (dst.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = src.extent[0];
    int64_t num_cols = src.extent[1];
    int64_t num_batches = dst.extent[2];

    if (num_rows != dst.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_cols != dst.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool src_is_dry = src.memory.stack_allocator->is_dry;
    bool dst_is_dry = dst.memory.stack_allocator->is_dry;

    bool is_dry;
    if (src_is_dry && dst_is_dry) {
        is_dry = true;
    } else if (!src_is_dry && !dst_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if (!is_dry) {
        void* src_ptr;
        void* dst_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(src.memory, &src_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(dst.memory, &dst_ptr) );

        CUfunction function = handle->functions[_KERMAC_KERNEL_BROADCAST_2D_TO_3D_F32];

        void* args[] = {
            (void*)&src_ptr,
            (void*)&dst_ptr,
            (void*)&num_rows,
            (void*)&num_cols,
            (void*)&num_batches,
            (void*)&src.stride[1],
            (void*)&dst.stride[1],
            (void*)&dst.stride[2]
        };

        unsigned int block_x = 32;
        unsigned int block_y = 8;

        unsigned int grid_x = (num_rows + block_x - 1) / block_x;
        unsigned int grid_y = (num_cols + block_y - 1) / block_y;

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                function,
                grid_x, grid_y, 1,
                block_x, block_y, 1,
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
_kermac_tensor_host_print(
    int edge_items,
    KermacTensor tensor
) {
    if (
        tensor.data_type != KERMAC_DATA_TYPE_POINTER &&
        tensor.data_type != KERMAC_DATA_TYPE_FLOAT && 
        tensor.data_type != KERMAC_DATA_TYPE_DOUBLE &&
        tensor.data_type != KERMAC_DATA_TYPE_INT &&
        tensor.data_type != KERMAC_DATA_TYPE_UINT && 
        tensor.data_type != KERMAC_DATA_TYPE_BYTE
    ) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    bool tensor_is_dry = tensor.memory.stack_allocator->is_dry;
    if (!tensor_is_dry) {
        void* tensor_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(tensor.memory, &tensor_ptr) );

        int64_t limit = (int64_t)edge_items;
        for (int64_t d3 = 0; d3 < tensor.extent[3] || (tensor.num_modes < 4 && d3 < 1); d3++) {
            for (int64_t d2 = 0; d2 < tensor.extent[2] || (tensor.num_modes < 3 && d2 < 1); d2++) {
                for (int64_t d0 = 0; d0 < tensor.extent[0] || (tensor.num_modes < 1 && d0 < 1); d0++) {
                    for (int64_t d1 = 0; d1 < tensor.extent[1] || (tensor.num_modes < 2 && d1 < 1); d1++) {
                        if (tensor.data_type == KERMAC_DATA_TYPE_POINTER) {
                            void** ptr = (void**)tensor_ptr;
                            void* v = ptr[d3 * tensor.stride[3] + d2 * tensor.stride[2] + d1 * tensor.stride[1] + d0];
                            printf("%p,", v);
                        } else if (tensor.data_type == KERMAC_DATA_TYPE_FLOAT) {
                            float* ptr = (float*)tensor_ptr;
                            float v = ptr[d3 * tensor.stride[3] + d2 * tensor.stride[2] + d1 * tensor.stride[1] + d0];
                            printf("%03.3f,", v);
                        } else if (tensor.data_type == KERMAC_DATA_TYPE_DOUBLE) {
                            double* ptr = (double*)tensor_ptr;
                            double v = ptr[d3 * tensor.stride[3] + d2 * tensor.stride[2] + d1 * tensor.stride[1] + d0];
                            printf("%03.3f,", v);
                        } else if (tensor.data_type == KERMAC_DATA_TYPE_INT) {
                            int* ptr = (int*)tensor_ptr;
                            int v = ptr[d3 * tensor.stride[3] + d2 * tensor.stride[2] + d1 * tensor.stride[1] + d0];
                            printf("%d,", v);
                        } else if (tensor.data_type == KERMAC_DATA_TYPE_UINT) {
                            uint32_t* ptr = (uint32_t*)tensor_ptr;
                            uint32_t v = ptr[d3 * tensor.stride[3] + d2 * tensor.stride[2] + d1 * tensor.stride[1] + d0];
                            printf("%u,", v);
                        } else if (tensor.data_type == KERMAC_DATA_TYPE_BYTE) {
                            uint8_t* ptr = (uint8_t*)tensor_ptr;
                            uint8_t v = ptr[d3 * tensor.stride[3] + d2 * tensor.stride[2] + d1 * tensor.stride[1] + d0];
                            printf("%3d,", (int)v);
                        }
                        if (limit != 0 && d1 == limit - 1 && tensor.extent[1] > limit * 2) {
                            printf("...,");
                            d1 = tensor.extent[1] - limit - 1;
                        }
                    }
                    printf("\n");
                    if (limit != 0 && d0 == limit - 1 && tensor.extent[0] > limit * 2) {
                        printf("...\n");
                        d0 = tensor.extent[0] - limit - 1;
                    }
                }
                if (d2 + 1 < tensor.extent[2]) {
                    printf("---------------\n");
                }
            }
            if (d3 + 1 < tensor.extent[3]) {
                printf("+++++++++++++++++\n");
            }
        }
        printf("\n");
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_print(
    KermacTensor tensor,
    KermacStackAllocator* hsa,
    int edge_items,
    CUstream stream
) {
    if (tensor.memory.stack_allocator->memory_space == KERMAC_MEMORY_SPACE_DEVICE) {
        if (hsa == NULL) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }

        if (hsa->memory_space != KERMAC_MEMORY_SPACE_HOST) {
            _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
        }

        KermacTensor h_tensor = KERMAC_ZERO_INIT(KermacTensor);
        _KERMAC_CHECK_RET( 
            kermac_tensor_create(
                &h_tensor,
                tensor.data_type, 
                tensor.extent,
                hsa
            )
        );
        _KERMAC_CHECK_RET( _kermac_tensor_get(tensor, h_tensor, stream) );
        _KERMAC_CUDA_CHECK_RET( cuStreamSynchronize(stream) );
        KermacResult result = KERMAC_SUCCESS;
        if (!tensor.memory.stack_allocator->is_dry) {
            result = _kermac_tensor_host_print(edge_items, h_tensor);
        }
        _KERMAC_CHECK_RET( kermac_tensor_destroy(h_tensor) );
        _KERMAC_CHECK_RET( result );
    } else if (tensor.memory.stack_allocator->memory_space == KERMAC_MEMORY_SPACE_HOST) {
        if (hsa != NULL) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
        _KERMAC_CHECK_RET( _kermac_tensor_host_print(edge_items, tensor) );
    }

    return KERMAC_SUCCESS;
}

// Thread-safety: This function is NOT thread-safe. Concurrent calls to
// kermac_tensor_view_create on the same tensor or tensors sharing the same
// stack_allocator will result in undefined behavior due to unsynchronized
// modification of current_stack_counter. Callers must ensure external
// synchronization if concurrent view creation is required.
KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_view_create(
    KermacTensor tensor,
    KermacTensor *view,
    uint32_t mode,
    size_t slice
) {
    // Make sure we're not creating a non-contiguous tensor
    if (mode == tensor.num_modes) {
        // Tensor doesnt have a batch in this mode
        if (slice != 0) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else { 
        if (mode + 1 != tensor.num_modes) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }

        if (slice >= tensor.extent[mode]) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    size_t data_type_size = _kermac_data_type_sizes[tensor.data_type];
    memcpy(view, &tensor, sizeof(KermacTensor));
    view->memory.stack_count = tensor.memory.stack_allocator->current_stack_counter++;
    view->memory.is_view = true;
    view->memory.offset += view->stride[mode] * slice * data_type_size;
    view->memory.num_bytes = view->stride[mode] * data_type_size;
    view->extent[mode] = 0;
    view->num_modes = tensor.num_modes-1;

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_view_destroy(
    KermacTensor view
) {
    if (!view.memory.is_view) {
         _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    return kermac_tensor_destroy(view);
}
