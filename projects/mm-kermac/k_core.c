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

INCBIN(unsigned char, kernels_cubin, KERMAC_KERNELS_CUBIN);

static const char* const _kermac_result_strings[] = {
	[KERMAC_SUCCESS] = 						    "KERMAC_SUCCESS",
	[KERMAC_ERROR_OUT_OF_MEMORY] = 			    "KERMAC_ERROR_OUT_OF_MEMORY",
    [KERMAC_ERROR_STACK_OUT_OF_ORDER] =         "KERMAC_ERROR_STACK_OUT_OF_ORDER",
    [KERMAC_ERROR_INTERNAL] =                   "KERMAC_ERROR_INTERNAL",
    [KERMAC_ERROR_INCONSISTENT_ALLOCATION] =    "KERMAC_ERROR_INCONSISTENT_ALLOCATION",
    [KERMAC_ERROR_INSUFFICIENT_BUFFER] =        "KERMAC_ERROR_INSUFFICIENT_BUFFER",
    [KERMAC_ERROR_TOO_MANY_STREAMS] =           "KERMAC_ERROR_TOO_MANY_STREAMS",
    [KERMAC_ERROR_INVALID_VALUE] =              "KERMAC_ERROR_INVALID_VALUE",
    [KERMAC_ERROR_WRONG_DATA_TYPE] =            "KERMAC_ERROR_WRONG_DATA_TYPE",
    [KERMAC_ERROR_SIZE_MISMATCH] =              "KERMAC_ERROR_SIZE_MISMATCH",
    [KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE] =     "KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE",
    [KERMAC_ERROR_CUBLAS] =                     "KERMAC_ERROR_CUBLAS",
    [KERMAC_ERROR_CUSOLVER] =                   "KERMAC_ERROR_CUSOLVER",
    [KERMAC_ERROR_CUTENSOR] =                   "KERMAC_ERROR_CUTENSOR",
    [KERMAC_ERROR_CUDA] =                       "KERMAC_ERROR_CUDA",
    [KERMAC_ERROR_STACK_PTX] =                  "KERMAC_ERROR_STACK_PTX",
    [KERMAC_ERROR_PTX_INJECT] =                 "KERMAC_ERROR_PTX_INJECT",
    [KERMAC_ERROR_NVPTX] =                      "KERMAC_ERROR_NVPTX",
    [KERMAC_ERROR_THIRDPARTY] =                 "KERMAC_ERROR_THIRDPARTY",
    [KERMAC_ERROR_NVRTC] =                      "KERMAC_ERROR_NVRTC"
};
_KERMAC_STATIC_ASSERT(
	_KERMAC_ARRAY_NUM_ELEMS(_kermac_result_strings) == KERMAC_RESULT_NUM_ENUMS, 
	_kermac_result_strings
);

KERMAC_PUBLIC_DEF
const char*
kermac_result_to_string(
    KermacResult result
) {
    if (result < 0 || result >= KERMAC_RESULT_NUM_ENUMS) {
        return "KERMAC_ERROR_UNKNOWN_ERROR_CODE";
    }
    return _kermac_result_strings[result];
}

KermacResult _kermac_cutensor_create(KermacHandle h);
KermacResult _kermac_cutensor_destroy(KermacHandle h);

__attribute__((unused))
static
CUresult
_kermac_context_create(CUcontext* context, CUdevice device) {
    #ifdef CUDA_VERSION_13
        return cuCtxCreate(context, NULL, 0, device);
    #else
        return cuCtxCreate(context, 0, device);
    #endif
}


KERMAC_PUBLIC_DEF
KermacResult
kermac_create(
    KermacHandle* handle
) {
    if (!handle) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    KermacHandle h = (KermacHandle)calloc(1, sizeof(*h));
    if (!h) {
        _KERMAC_ERROR( KERMAC_ERROR_OUT_OF_MEMORY );
    }

    _KERMAC_CUDA_CHECK_RET( cuInit(0) );
    _KERMAC_CUDA_CHECK_RET( cuDeviceGet(&h->device, 0) );

    for (int i = 0; i < _KERMAC_DEVICE_ATTRIBUTES_NUM_ENUMS; i++) {
        _KermacDeviceAttribute device_attribute = (_KermacDeviceAttribute)i;
        CUdevice_attribute cudevice_attribute = _kermac_device_attribute_to_cudevice_attribute[device_attribute];
        _KERMAC_CUDA_CHECK_RET( 
            cuDeviceGetAttribute(
                &h->device_attributes[device_attribute], 
                cudevice_attribute, h->device
            )
        );
    }

    _KERMAC_CUDA_CHECK_RET( _kermac_context_create(&h->context, h->device) );
    _KERMAC_CUDA_CHECK_RET( cuModuleLoadDataEx(&h->module, g_kernels_cubin_data, 0, 0, 0) );

    for (int i = 0; i < _KERMAC_KERNEL_FUNCTION_NUM_ENUMS; i++) {
        _KermacKernelFunction function = (_KermacKernelFunction)i;
        _KERMAC_CUDA_CHECK_RET(
            cuModuleGetFunction(
                &h->functions[function],
                h->module,
                _kermac_kernel_function_names[function]
            )
        );
    }
    _KERMAC_CUBLAS_CHECK_RET( cublasCreate(&h->cublas) );
    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnCreate(&h->cusolver) );

    _KERMAC_CHECK_RET( _kermac_cutensor_create(h) );

    _KERMAC_CUDA_CHECK_RET( cuEventCreate(&h->primary_event_start, CU_EVENT_DISABLE_TIMING) );
    _KERMAC_CUDA_CHECK_RET( cuEventCreate(&h->primary_event_end, CU_EVENT_DISABLE_TIMING) );
    _KERMAC_CUDA_CHECK_RET( cuEventRecord(h->primary_event_end, 0) );
    for (int i = 0; i < _KERMAC_MAX_STREAMS; i++) {
        _KERMAC_CUDA_CHECK_RET(
            cuEventCreate(&h->events[i], CU_EVENT_DISABLE_TIMING)
        );
    }
    h->philox_launch_id = 0;
    h->philox_seed = 1234;

    *handle = h;
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_destroy(
    KermacHandle handle
) {
    if (!handle) {
        return KERMAC_SUCCESS;  // nothing to do
    }

    for (int i = 0; i < _KERMAC_MAX_STREAMS; i++) {
        _KERMAC_CUDA_CHECK_RET(
            cuEventDestroy(handle->events[i])
        );
    }
    _KERMAC_CUDA_CHECK_RET( cuEventDestroy(handle->primary_event_end) );
    _KERMAC_CUDA_CHECK_RET( cuEventDestroy(handle->primary_event_start) );
    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnDestroy(handle->cusolver) );
    _KERMAC_CUBLAS_CHECK_RET( cublasDestroy(handle->cublas) );
    _KERMAC_CHECK_RET( _kermac_cutensor_destroy(handle) );
    _KERMAC_CUDA_CHECK_RET( cuModuleUnload(handle->module) );
    _KERMAC_CUDA_CHECK_RET( cuCtxDestroy(handle->context) );

    free(handle);
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_make_current(
    KermacHandle handle
) {
    if (!handle) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    _KERMAC_CUDA_CHECK_RET( cuCtxSetCurrent(handle->context) );
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_device_capability(
    KermacHandle handle,
    int* major_out,
    int* minor_out
) {
    if (!handle) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );  // nothing to do
    }
    if (!major_out || !minor_out) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    *major_out = handle->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MAJOR];
    *minor_out = handle->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MINOR];

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF 
KermacResult 
kermac_host_alloc(
    void** ptr,
    size_t num_bytes
) {
    *ptr = NULL;
    if (posix_memalign(ptr, _KERMAC_HOST_ALLOC_ALIGNMENT, num_bytes) != 0) {
        _KERMAC_ERROR( KERMAC_ERROR_OUT_OF_MEMORY );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_host_alloc_pinned(
    void** ptr,
    size_t num_bytes
) {
    *ptr = NULL;
    if (num_bytes == 0) {
        return KERMAC_SUCCESS;
    }
    _KERMAC_CUDA_CHECK_RET( cuMemHostAlloc(ptr, num_bytes, CU_MEMHOSTALLOC_PORTABLE) );
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult 
kermac_host_free(
    void* ptr
) {
    if (ptr) {
        free(ptr);
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_host_free_pinned(
    void* ptr
) {
    if (ptr) {
        _KERMAC_CUDA_CHECK_RET( cuMemFreeHost(ptr) );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF 
KermacResult 
kermac_device_alloc(
    KermacHandle kermac,
    void** ptr,
    size_t num_bytes
) {
    CUdeviceptr device_memory_ptr = {0};
    _KERMAC_CUDA_CHECK_RET( cuMemAlloc(&device_memory_ptr, num_bytes) );
    *ptr = (void*)device_memory_ptr;
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF 
KermacResult 
kermac_device_free(
    KermacHandle kermac, 
    void* ptr
) {
    if (ptr) {
        CUdeviceptr device_memory_ptr = (CUdeviceptr)ptr;
        _KERMAC_CUDA_CHECK_RET( cuMemFree(device_memory_ptr) );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_stack_allocator_create(
    KermacStackAllocator* stack_allocator,
    KermacMemorySpace memory_space,
    void* memory,
    size_t num_bytes
) {
    *stack_allocator = (KermacStackAllocator) {
        .current_offset = 0,
        .largest_total_offset = 0,
        .current_stack_counter = 0,
        .memory = memory,
        .allocated_bytes = num_bytes,
        .memory_space = memory_space,
        .is_dry = memory == NULL
    };
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_stack_allocator_destroy(
    KermacStackAllocator stack_allocator
) {
    if (stack_allocator.current_offset != 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_memory_pointer(
    KermacMemory memory,
    void** ptr
) {
    if (memory.stack_allocator->is_dry) {
        _KERMAC_ERROR( KERMAC_ERROR_INTERNAL );
    }
    *ptr = (uint8_t*)memory.stack_allocator->memory + memory.offset;
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_rng_f32(
    KermacHandle kermac,
    KermacTensor device_tensor,
    KermacRNGType rng_type,
    float scale, float shift,
    CUstream stream
) {
    if (device_tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (device_tensor.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = device_tensor.num_modes == 0 ? 1 : device_tensor.extent[0];
    int64_t ld_rows = device_tensor.num_modes == 0 ? 1 : device_tensor.stride[1];
    int64_t num_cols = _kermac_calculate_effective_cols(device_tensor);

    CUfunction rng_function;
    if (rng_type == KERMAC_RNG_TYPE_UNIFORM) {
        rng_function = kermac->functions[_KERMAC_KERNEL_FUNCTION_PHILOX_F32_UNIFORM];
    } else if (rng_type == KERMAC_RNG_TYPE_NORMAL) {
        rng_function = kermac->functions[_KERMAC_KERNEL_FUNCTION_PHILOX_F32_NORMAL];
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool device_tensor_is_dry = device_tensor.memory.stack_allocator->is_dry;
    bool is_dry;

    if (device_tensor_is_dry) {
        is_dry = true;
    } else if (!device_tensor_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if(!is_dry) {
        
        void* device_tensor_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(device_tensor.memory, &device_tensor_ptr) );

        void *args[] = {
            (void*)&kermac->philox_launch_id,
            (void*)&kermac->philox_seed,
            (void*)&num_rows,
            (void*)&ld_rows,
            (void*)&num_cols,
            (void*)&scale,
            (void*)&shift,
            (void*)&device_tensor_ptr
        };
        kermac->philox_launch_id++;

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                rng_function,
                kermac->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_MULTIPROCESSOR_COUNT], 1, 1,
                256, 1, 1,
                0, stream,
                args, 0
            )
        );
        
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_tensor_rng_u32(
    KermacHandle kermac,
    KermacTensor device_tensor,
    CUstream stream
) {
    if (device_tensor.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (device_tensor.data_type != KERMAC_DATA_TYPE_UINT) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = device_tensor.num_modes == 0 ? 1 : device_tensor.extent[0];
    int64_t ld_rows = device_tensor.num_modes == 0 ? 1 : device_tensor.stride[1];
    int64_t num_cols = _kermac_calculate_effective_cols(device_tensor);

    CUfunction rng_function = kermac->functions[_KERMAC_KERNEL_FUNCTION_PHILOX_U32];

    bool device_tensor_is_dry = device_tensor.memory.stack_allocator->is_dry;
    bool is_dry;

    if (device_tensor_is_dry) {
        is_dry = true;
    } else if (!device_tensor_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    if(!is_dry) {
        void* device_tensor_ptr;
        _KERMAC_CHECK_RET( kermac_memory_pointer(device_tensor.memory, &device_tensor_ptr) );

        void *args[] = {
            (void*)&kermac->philox_launch_id,
            (void*)&kermac->philox_seed,
            (void*)&num_rows,
            (void*)&ld_rows,
            (void*)&num_cols,
            (void*)&device_tensor_ptr
        };
        kermac->philox_launch_id++;

        _KERMAC_CUDA_CHECK_RET( 
            cuLaunchKernel(
                rng_function,
                kermac->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_MULTIPROCESSOR_COUNT], 1, 1,
                256, 1, 1,
                0, stream,
                args, 0
            )
        );
    }

    return KERMAC_SUCCESS;
}
