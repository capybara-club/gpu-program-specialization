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
#pragma once

#include <stack_ptx.h>
#include <ptx_inject.h>
#include <nvPTXCompiler.h>

#define _KERMAC_NEAREST_LARGER_MULTIPLE(X,Y) ((((X) - 1) / Y) + 1)
#define _KERMAC_MAX(a, b) ((a) > (b) ? (a) : (b))
#define _KERMAC_MIN(a, b) ((a) < (b) ? (a) : (b))
#define _KERMAC_ARRAY_NUM_ELEMS(array) sizeof((array)) / sizeof(*(array))
#define _KERMAC_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]

#ifdef KERMAC_DEBUG
    #include <assert.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif

#ifdef KERMAC_DEBUG
    #define _KERMAC_ERROR(ans)                                                                      \
        do {                                                                                        \
            KermacResult _result = (ans);                                                           \
            const char* error_name = kermac_result_to_string(_result);                              \
            fprintf(stderr, "KERMAC_ERROR: %s \n  %s %d\n", error_name, __FILE__, __LINE__);        \
            assert(0);                                                                              \
            exit(1);                                                                                \
        } while(0);
#else
    #define _KERMAC_ERROR(ans)                              \
        do {                                                \
            KermacResult _result = (ans);                   \
            return _result;                                 \
        } while(0);
#endif // KERMAC_DEBUG

#ifdef KERMAC_DEBUG
    #define _KERMAC_CHECK_RET(ans)                                                                  \
        do {                                                                                        \
            KermacResult _result = (ans);                                                           \
            if (_result != KERMAC_SUCCESS) {                                                        \
                const char* error_name = kermac_result_to_string(_result);                          \
                fprintf(stderr, "KERMAC_CHECK: %s \n  %s %d\n", error_name, __FILE__, __LINE__);    \
                assert(0);                                                                          \
                exit(1);                                                                            \
                return _result;                                                                     \
            }                                                                                       \
        } while(0);
#else
    #define _KERMAC_CHECK_RET(ans)                          \
        do {                                                \
            KermacResult _result = (ans);                   \
            if (_result != KERMAC_SUCCESS) return _result;  \
        } while(0);
#endif // KERMAC_DEBUG

#ifdef KERMAC_DEBUG
    #define _KERMAC_CUDA_CHECK_RET(ans)                                                             \
        do {                                                                                        \
            CUresult _result = (ans);                                                               \
            if (_result != CUDA_SUCCESS) {                                                          \
                const char* _error_name;                                                            \
                CUresult _name_result = cuGetErrorName(_result, &_error_name);                      \
                if (_name_result != CUDA_SUCCESS) {                                                 \
                    fprintf(stderr, "KERMAC_CUDA_CHECK: failed to get error name: %d %d %s %d\n",   \
                        _result, _name_result, __FILE__, __LINE__                                   \
                    );                                                                              \
                    _KERMAC_CHECK_RET( KERMAC_ERROR_CUDA );                                         \
                }                                                                                   \
                const char* _error_string;                                                          \
                CUresult _string_result = cuGetErrorString(_result, &_error_string);                \
                if (_string_result != CUDA_SUCCESS) {                                               \
                    fprintf(stderr, "KERMAC_CUDA_CHECK: failed to get error string: %d %d %s %d\n", \
                        _result, _string_result, __FILE__, __LINE__                                 \
                    );                                                                              \
                    _KERMAC_CHECK_RET( KERMAC_ERROR_CUDA );                                         \
                }                                                                                   \
                fprintf(stderr, "KERMAC_CUDA_CHECK: %s \n  %s \n  %s %d\n",                         \
                    _error_name, _error_string, __FILE__, __LINE__                                  \
                );                                                                                  \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUDA );                                             \
            }                                                                                       \
        } while(0)
#else 
    #define _KERMAC_CUDA_CHECK_RET(ans)                 \
        do {                                            \
            CUresult _status = (ans);                   \
            if (_status != CUDA_SUCCESS) {              \
                return KERMAC_ERROR_CUDA;               \
            }                                           \
        } while(0)
#endif // KERMAC_DEBUG

#ifdef KERMAC_DEBUG
    #define _KERMAC_CUBLAS_CHECK_RET(ans)                                   \
        do {                                                                \
            cublasStatus_t _status = (ans);                                 \
            if (_status != CUBLAS_STATUS_SUCCESS) {                         \
                fprintf(stderr, "KERMAC_CUBLAS_CHECK: %d %s %d\n",          \
                    _status, __FILE__, __LINE__                             \
                );                                                          \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUBLAS );                   \
            }                                                               \
        } while(0)
#else
    #define _KERMAC_CUBLAS_CHECK_RET(ans)                 \
        do {                                              \
            cublasStatus_t _status = (ans);               \
            if (_status != CUBLAS_STATUS_SUCCESS) {       \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUBLAS ); \
            }                                             \
        } while(0)
#endif

#ifdef KERMAC_DEBUG
    #define _KERMAC_CUSOLVER_CHECK_RET(ans)                                 \
        do {                                                                \
            cusolverStatus_t _status = (ans);                               \
            if (_status != CUSOLVER_STATUS_SUCCESS) {                       \
                fprintf(stderr, "KERMAC_CUSOLVER_CHECK: %d %s %d\n",        \
                    _status, __FILE__, __LINE__                             \
                );                                                          \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUSOLVER );                 \
            }                                                               \
        } while(0)
#else 
    #define _KERMAC_CUSOLVER_CHECK_RET(ans)                 \
        do {                                                \
            cusolverStatus_t _status = (ans);               \
            if (_status != CUSOLVER_STATUS_SUCCESS) {       \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUSOLVER ); \
            }                                               \
        } while(0)
#endif

#ifdef KERMAC_DEBUG
    #define _KERMAC_CUTENSOR_CHECK_RET(ans)                                 \
        do {                                                                \
            cutensorStatus_t _status = (ans);                               \
            if (_status != CUTENSOR_STATUS_SUCCESS) {                       \
                const char*  _error_name = cutensorGetErrorString(_status); \
                fprintf(stderr, "KERMAC_CUTENSOR_CHECK: %s \n  %s %d\n",    \
                    _error_name, __FILE__, __LINE__                         \
                );                                                          \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUTENSOR );                 \
            }                                                               \
        } while(0)
#else 
    #define _KERMAC_CUTENSOR_CHECK_RET(ans)                 \
        do {                                                \
            cutensorStatus_t _status = (ans);               \
            if (_status != CUTENSOR_STATUS_SUCCESS) {       \
                _KERMAC_CHECK_RET( KERMAC_ERROR_CUTENSOR ); \
            }                                               \
        } while(0)
#endif // KERMAC_DEBUG

#ifdef KERMAC_DEBUG
    #define _KERMAC_STACK_PTX_CHECK_RET(ans)                                    \
        do {                                                                    \
            StackPtxResult _status = (ans);                                     \
            if (_status != STACK_PTX_SUCCESS) {                                 \
                const char*  _error_name = stack_ptx_result_to_string(_status); \
                fprintf(stderr, "KERMAC_STACK_PTX_CHECK: %s \n  %s %d\n",       \
                    _error_name, __FILE__, __LINE__                             \
                );                                                              \
                _KERMAC_CHECK_RET( KERMAC_ERROR_STACK_PTX );                    \
            }                                                                   \
        } while(0)
#else 
    #define _KERMAC_STACK_PTX_CHECK_RET(ans)                    \
        do {                                                    \
            StackPtxResult _status = (ans);                     \
            if (_status != STACK_PTX_SUCCESS) {                 \
                _KERMAC_CHECK_RET( KERMAC_ERROR_STACK_PTX );    \
            }                                                   \
        } while(0)
#endif // KERMAC_DEBUG

#ifdef KERMAC_DEBUG
    #define _KERMAC_PTX_INJECT_CHECK_RET(ans)                                       \
        do {                                                                        \
            PtxInjectResult _status = (ans);                                        \
            if (_status != PTX_INJECT_SUCCESS) {                                    \
                const char*  _error_name = ptx_inject_result_to_string(_status);    \
                fprintf(stderr, "KERMAC_PTX_INJECT_CHECK: %s \n  %s %d\n",          \
                    _error_name, __FILE__, __LINE__                                 \
                );                                                                  \
                _KERMAC_CHECK_RET( KERMAC_ERROR_PTX_INJECT );                       \
            }                                                                       \
        } while(0)
#else 
    #define _KERMAC_PTX_INJECT_CHECK_RET(ans)                   \
        do {                                                    \
            PtxInjectResult _status = (ans);                    \
            if (_status != PTX_INJECT_SUCCESS) {                \
                _KERMAC_CHECK_RET( KERMAC_ERROR_PTX_INJECT );   \
            }                                                   \
        } while(0)
#endif // KERMAC_DEBUG

#ifdef KERMAC_DEBUG
    #define _KERMAC_NVPTX_CHECK_RET(ans)                                        \
        do {                                                                    \
            static const char* nvPtxCompileResult_strings[] = {                 \
                "NVPTXCOMPILE_SUCCESS",                                         \
                "NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE",                   \
                "NVPTXCOMPILE_ERROR_INVALID_INPUT",                             \
                "NVPTXCOMPILE_ERROR_COMPILATION_FAILURE",                       \
                "NVPTXCOMPILE_ERROR_INTERNAL",                                  \
                "NVPTXCOMPILE_ERROR_OUT_OF_MEMORY",                             \
                "NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE",            \
                "NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION",                   \
                "NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC",                  \
                "NVPTXCOMPILE_ERROR_CANCELLED"                                  \
            };                                                                  \
            nvPTXCompileResult _status = (ans);                                 \
            if (_status != NVPTXCOMPILE_SUCCESS) {                              \
                const char* _error_name;                                        \
                if (_status >= 0 && _status < sizeof(nvPtxCompileResult_strings)/sizeof(nvPtxCompileResult_strings[0])) { \
                    _error_name = nvPtxCompileResult_strings[_status];          \
                } else {                                                        \
                    _error_name = "NVPTXCOMPILE_ERROR_UNKNOWN";                 \
                }                                                               \
                fprintf(stderr, "KERMAC_NVPTX_CHECK: %s \n  %s %d\n",           \
                    _error_name, __FILE__, __LINE__                             \
                );                                                              \
                _KERMAC_CHECK_RET( KERMAC_ERROR_NVPTX );                        \
            }                                                                   \
        } while(0)
#else 
    #define _KERMAC_NVPTX_CHECK_RET(ans)                  \
        do {                                              \
            nvPTXCompileResult _status = (ans);           \
            if (_status != NVPTXCOMPILE_SUCCESS) {        \
                _KERMAC_CHECK_RET( KERMAC_ERROR_NVPTX );  \
            }                                             \
        } while(0)
#endif // KERMAC_DEBUG

#include <cublas_v2.h>
#include <cusolverDn.h>

#define _KERMAC_HOST_ALLOC_ALIGNMENT 64ull
#define _KERMAC_MAX_STREAMS 128
#define _KERMAC_TENSOR_LD_ALIGNMENT_BYTES 128ull

// Ensure alignment values are powers of 2 (required by posix_memalign)
_KERMAC_STATIC_ASSERT(
    (_KERMAC_HOST_ALLOC_ALIGNMENT & (_KERMAC_HOST_ALLOC_ALIGNMENT - 1)) == 0 && _KERMAC_HOST_ALLOC_ALIGNMENT > 0,
    host_alloc_alignment_must_be_power_of_two
);
_KERMAC_STATIC_ASSERT(
    (_KERMAC_TENSOR_LD_ALIGNMENT_BYTES & (_KERMAC_TENSOR_LD_ALIGNMENT_BYTES - 1)) == 0 && _KERMAC_TENSOR_LD_ALIGNMENT_BYTES > 0,
    tensor_ld_alignment_must_be_power_of_two
);

#ifdef __cplusplus
    #define KERMAC_ZERO_INIT(type) type{}
#else
    #define KERMAC_ZERO_INIT(type) (type){0}
#endif

typedef enum {
    _KERMAC_KERNEL_FUNCTION_PHILOX_F32_UNIFORM,
    _KERMAC_KERNEL_FUNCTION_PHILOX_F32_NORMAL,
    _KERMAC_KERNEL_FUNCTION_PHILOX_U32,
    _KERMAC_KERNEL_FUNCTION_MSE_ACCURACY_256_F32,
    _KERMAC_KERNEL_FUNCTION_MSE_256_F32,
    _KERMAC_KERNEL_FUNCTION_MSE_FINAL_REDUCTION_1024_ACCURACY_F32,
    _KERMAC_KERNEL_FUNCTION_MSE_FINAL_REDUCTION_1024_F32,
    _KERMAC_KERNEL_FUNCTION_LOGDET_NORM_NORM_H2_F32,
    _KERMAC_KERNEL_FUNCTION_ROW_STATS_F32,
    _KERMAC_KERNEL_FUNCTION_APPLY_ROW_STATS_F32,
    _KERMAC_KERNEL_BROADCAST_2D_TO_3D_F32,
    _KERMAC_KERNEL_FUNCTION_NUM_ENUMS
} _KermacKernelFunction;

typedef enum {
    _KERMAC_DEVICE_ATTRIBUTES_TEXTURE_ALIGNMENT,
    _KERMAC_DEVICE_ATTRIBUTES_MULTIPROCESSOR_COUNT,
    _KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MAJOR,
    _KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MINOR,
    _KERMAC_DEVICE_ATTRIBUTES_MAX_THREADS_PER_MULTIPROCESSOR,
    _KERMAC_DEVICE_ATTRIBUTES_MAX_THREADS_PER_BLOCK,
    _KERMAC_DEVICE_ATTRIBUTES_NUM_ENUMS
} _KermacDeviceAttribute;

struct KermacHandleImpl {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction functions[_KERMAC_KERNEL_FUNCTION_NUM_ENUMS];
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    void* cutensor;
    int device_attributes[_KERMAC_DEVICE_ATTRIBUTES_NUM_ENUMS];
    CUevent primary_event_start;
    CUevent primary_event_end;
    CUevent events[_KERMAC_MAX_STREAMS];
    uint64_t philox_launch_id;
    uint64_t philox_seed;
};

static const char * _kermac_kernel_function_names[] = {
    [_KERMAC_KERNEL_FUNCTION_PHILOX_F32_UNIFORM] =                      "kernel_rng_uniform_f32",
    [_KERMAC_KERNEL_FUNCTION_PHILOX_F32_NORMAL] =                       "kernel_rng_normal_f32",
    [_KERMAC_KERNEL_FUNCTION_PHILOX_U32] =                              "kernel_rng_u32",
    [_KERMAC_KERNEL_FUNCTION_MSE_ACCURACY_256_F32] =                    "kernel_mse_accuracy_256_f32",
    [_KERMAC_KERNEL_FUNCTION_MSE_256_F32] =                             "kernel_mse_256_f32",
    [_KERMAC_KERNEL_FUNCTION_MSE_FINAL_REDUCTION_1024_ACCURACY_F32] =   "kernel_mse_final_reduction_1024_accuracy_f32",
    [_KERMAC_KERNEL_FUNCTION_MSE_FINAL_REDUCTION_1024_F32] =            "kernel_mse_final_reduction_1024_f32",
    [_KERMAC_KERNEL_FUNCTION_LOGDET_NORM_NORM_H2_F32] =                 "kernel_logdet_norm_norm_h2",
    [_KERMAC_KERNEL_FUNCTION_ROW_STATS_F32] =                           "kernel_row_stats",
    [_KERMAC_KERNEL_FUNCTION_APPLY_ROW_STATS_F32] =                     "kernel_apply_row_stats",
    [_KERMAC_KERNEL_BROADCAST_2D_TO_3D_F32] =                           "kernel_broadcast_2d_to_3d_f32"
};
_KERMAC_STATIC_ASSERT(
	_KERMAC_ARRAY_NUM_ELEMS(_kermac_kernel_function_names) == _KERMAC_KERNEL_FUNCTION_NUM_ENUMS, 
	_kermac_kernel_function_names
);

static const CUdevice_attribute _kermac_device_attribute_to_cudevice_attribute[] = {
    [_KERMAC_DEVICE_ATTRIBUTES_TEXTURE_ALIGNMENT] =                 CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
    [_KERMAC_DEVICE_ATTRIBUTES_MULTIPROCESSOR_COUNT] =              CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    [_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MAJOR] =          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    [_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MINOR] =          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    [_KERMAC_DEVICE_ATTRIBUTES_MAX_THREADS_PER_MULTIPROCESSOR] =    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    [_KERMAC_DEVICE_ATTRIBUTES_MAX_THREADS_PER_BLOCK] =             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
};
_KERMAC_STATIC_ASSERT(
	_KERMAC_ARRAY_NUM_ELEMS(_kermac_device_attribute_to_cudevice_attribute) == _KERMAC_DEVICE_ATTRIBUTES_NUM_ENUMS, 
	_kermac_device_attribute_to_cudevice_attribute
);

static const size_t _kermac_data_type_sizes[] = {
    [KERMAC_DATA_TYPE_POINTER] = sizeof(void*),
    [KERMAC_DATA_TYPE_FLOAT] = sizeof(float),
    [KERMAC_DATA_TYPE_DOUBLE] = sizeof(double),
    [KERMAC_DATA_TYPE_INT] = sizeof(int32_t),
    [KERMAC_DATA_TYPE_UINT] = sizeof(uint32_t),
    [KERMAC_DATA_TYPE_BYTE] = sizeof(uint8_t)
};
_KERMAC_STATIC_ASSERT(
	_KERMAC_ARRAY_NUM_ELEMS(_kermac_data_type_sizes) == KERMAC_DATA_TYPE_NUM_ENUMS, 
	_kermac_data_type_sizes
);

static const size_t _kermac_memory_space_alignments[] = {
    [KERMAC_MEMORY_SPACE_HOST] = 64ull,
    [KERMAC_MEMORY_SPACE_DEVICE] = 512ull,
};
_KERMAC_STATIC_ASSERT(
	_KERMAC_ARRAY_NUM_ELEMS(_kermac_memory_space_alignments) == KERMAC_MEMORY_SPACE_NUM_ENUMS, 
	_kermac_memory_space_alignments
);

static const StackPtxCompilerInfo compiler_info = {
	.max_ast_size = 100,
	.max_ast_to_visit_stack_depth = 20,
	.stack_size = 128,
	.max_frame_depth = 4,
	.store_size = 16
};

// This is 0f3F317218 for ln(2) in ptx:
// lg2.approx.ftz.f32 %f5, %f2;
// mul.ftz.f32 %f6, %f5, 0f3F317218; 
static const float M_LN2_F32_EXPLICIT = 0x1.62e43p-1;

// This is 0f3FB8AA3B for ex2 in ptx:
// mul.ftz.f32 	%f2, %f1, 0f3FB8AA3B;
// ex2.approx.ftz.f32 	%f3, %f2;
static const float M_LOG2_E_F32_EXPLICIT = 0x1.715476p+0;

static
inline
KermacResult
_kermac_memory_create(
    KermacMemory* memory,
    KermacStackAllocator* stack_allocator,
    size_t num_bytes,
    size_t alignment_bytes
) {
    size_t alignment_multiple = _KERMAC_NEAREST_LARGER_MULTIPLE(stack_allocator->current_offset, alignment_bytes);
    size_t aligned_offset = alignment_bytes * alignment_multiple;

    if (stack_allocator->memory != NULL) {
        if (aligned_offset + num_bytes > stack_allocator->allocated_bytes) {
            _KERMAC_ERROR( KERMAC_ERROR_OUT_OF_MEMORY );
        }
    }

    stack_allocator->current_offset = aligned_offset + num_bytes;
    stack_allocator->largest_total_offset = 
        _KERMAC_MAX(
            stack_allocator->largest_total_offset, 
            stack_allocator->current_offset
        );

    *memory = (KermacMemory) {
        .stack_allocator = stack_allocator,
        .offset = aligned_offset,
        .stack_count = stack_allocator->current_stack_counter++,
        .num_bytes = num_bytes,
        .is_view = false
    };
    
    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_memory_destroy(
    KermacMemory memory
) {
    if (memory.stack_count != memory.stack_allocator->current_stack_counter-1) {
        _KERMAC_ERROR( KERMAC_ERROR_STACK_OUT_OF_ORDER );
    }

    if (!memory.is_view && memory.offset > memory.stack_allocator->current_offset) {
        _KERMAC_ERROR( KERMAC_ERROR_STACK_OUT_OF_ORDER );
    }

    memory.stack_allocator->current_stack_counter--;
    if (!memory.is_view) {
        memory.stack_allocator->current_offset = memory.offset;
    }

    return KERMAC_SUCCESS;
}

static
inline
int64_t
_kermac_calculate_effective_cols(
    KermacTensor tensor
) {
    int64_t effective_cols = 1;
    for (int i = 1; i < tensor.num_modes; i++) {
        effective_cols *= tensor.extent[i];
    }
    return effective_cols;
}

static
inline
KermacResult
_kermac_tensor_get(
    KermacTensor device_tensor_src,
    KermacTensor host_tensor_dst,
    CUstream stream
) {
    if (device_tensor_src.data_type != host_tensor_dst.data_type) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (device_tensor_src.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (host_tensor_dst.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows_src = device_tensor_src.num_modes == 0 ? 1 : device_tensor_src.extent[0];
    int64_t num_cols_src = _kermac_calculate_effective_cols(device_tensor_src);

    int64_t num_rows_dst = host_tensor_dst.num_modes == 0 ? 1 : host_tensor_dst.extent[0];
    int64_t num_cols_dst = _kermac_calculate_effective_cols(host_tensor_dst);

    if (num_rows_src != num_rows_dst || num_cols_src != num_cols_dst) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    size_t elem_size = _kermac_data_type_sizes[device_tensor_src.data_type];

    int64_t ld_tensor_src = device_tensor_src.num_modes == 0 ? 1 : device_tensor_src.stride[1];
    int64_t ld_tensor_dst = host_tensor_dst.num_modes == 0 ? 1 : host_tensor_dst.stride[1];

    bool device_tensor_is_dry = device_tensor_src.memory.stack_allocator->is_dry;
    bool host_tensor_is_dry = host_tensor_dst.memory.stack_allocator->is_dry;
    bool is_dry;
    if (device_tensor_is_dry && host_tensor_is_dry) {
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
        _KERMAC_CHECK_RET( kermac_memory_pointer(device_tensor_src.memory, &device_tensor_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(host_tensor_dst.memory, &host_tensor_ptr) );

        cublasStatus_t status = 
            cublasGetMatrixAsync(
                num_rows_src, 
                num_cols_src,
                elem_size,
                device_tensor_ptr, ld_tensor_src,
                host_tensor_ptr, ld_tensor_dst,
                stream
            );
        if (status != CUBLAS_STATUS_SUCCESS) {
            _KERMAC_ERROR( KERMAC_ERROR_CUBLAS );
        }
    }

    return KERMAC_SUCCESS;
}
