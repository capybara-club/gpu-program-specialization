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

#include <cuda.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <stack_ptx.h>
#include <ptx_inject.h>

#ifdef __cplusplus
#define KERMAC_PUBLIC_DEC extern "C"
#define KERMAC_PUBLIC_DEF extern "C"
#else
#define KERMAC_PUBLIC_DEC extern
#define KERMAC_PUBLIC_DEF
#endif

#define KermacExtent int64_t[4]

typedef enum {
    KERMAC_SUCCESS = 0,
    KERMAC_ERROR_OUT_OF_MEMORY,
    KERMAC_ERROR_STACK_OUT_OF_ORDER,
    KERMAC_ERROR_INTERNAL,
    KERMAC_ERROR_INCONSISTENT_ALLOCATION,
    KERMAC_ERROR_INSUFFICIENT_BUFFER,
    KERMAC_ERROR_TOO_MANY_STREAMS,
    KERMAC_ERROR_INVALID_VALUE,
    KERMAC_ERROR_WRONG_DATA_TYPE,
    KERMAC_ERROR_SIZE_MISMATCH,
    KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE,
    KERMAC_ERROR_CUBLAS,
    KERMAC_ERROR_CUSOLVER,
    KERMAC_ERROR_CUTENSOR,
    KERMAC_ERROR_CUDA,
    KERMAC_ERROR_STACK_PTX,
    KERMAC_ERROR_PTX_INJECT,
    KERMAC_ERROR_NVPTX,
    KERMAC_ERROR_THIRDPARTY,
    KERMAC_ERROR_NVRTC,
    KERMAC_RESULT_NUM_ENUMS
} KermacResult;

typedef enum {
    KERMAC_MEMORY_SPACE_HOST,
    KERMAC_MEMORY_SPACE_DEVICE,
    KERMAC_MEMORY_SPACE_NUM_ENUMS
} KermacMemorySpace;

typedef enum {
    KERMAC_RNG_TYPE_UNIFORM,
    KERMAC_RNG_TYPE_NORMAL
} KermacRNGType;

typedef enum {
    KERMAC_TENSOR_CORE_MODE_F32,
    KERMAC_TENSOR_CORE_MODE_TF32,
    KERMAC_TENSOR_CORE_MODE_NUM_ENUMS
} KermacTensorCoreMode;

typedef enum {
    KERMAC_AGOP_BACKEND_CUTENSOR_F32,
    KERMAC_AGOP_BACKEND_CUTENSOR_TF32,
    KERMAC_AGOP_BACKEND_FUSED,
    KERMAC_AGOP_BACKEND_NUM_ENUMS
} KermacAgopBackend;

typedef enum {
    KERMAC_AGOP_OUTPUT_FULL,
    KERMAC_AGOP_OUTPUT_DIAG,
    KERMAC_AGOP_OUTPUT_NUM_ENUMS
} KermacAgopOutput;

typedef enum {
    KERMAC_MATRIX_PACKED_TYPE_FULL,
    KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE,
    KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE
} KermacMatrixPackedType;

typedef struct {
    size_t current_offset;
    size_t largest_total_offset;
    int current_stack_counter;
    void* memory;
    size_t allocated_bytes;
    KermacMemorySpace memory_space;
    bool is_dry;
} KermacStackAllocator;

typedef struct {
    KermacStackAllocator* stack_allocator;
    size_t offset;
    int stack_count;
    size_t num_bytes;
    bool is_view;
} KermacMemory;

typedef enum {
    KERMAC_DATA_TYPE_POINTER,
    KERMAC_DATA_TYPE_FLOAT,
    KERMAC_DATA_TYPE_DOUBLE,
    KERMAC_DATA_TYPE_INT,
    KERMAC_DATA_TYPE_UINT,
    KERMAC_DATA_TYPE_BYTE,
    KERMAC_DATA_TYPE_NUM_ENUMS
} KermacDataType;

typedef struct {
    KermacDataType data_type;
    int64_t extent[4];
    int64_t stride[4];
    uint32_t num_modes;
    KermacMemory memory;
} KermacTensor;

typedef struct {
    CUmodule module;
    CUfunction function;
    KermacMatrixPackedType packed_type;
} KermacElementwise;

typedef struct {
    CUmodule module;
    CUfunction function;
    KermacMatrixPackedType packed_type;
} KermacSemiring;

typedef struct {
    CUmodule module;
    CUfunction function;
} KermacSemiringGradient;

typedef struct {
    KermacSemiringGradient semiring_gradient;
    bool has_semiring_gradient;
} KermacAgop;

typedef struct KermacHandleImpl* KermacHandle;

KERMAC_PUBLIC_DEC const char* kermac_result_to_string(KermacResult result);
KERMAC_PUBLIC_DEC KermacResult kermac_create(KermacHandle* handle);
KERMAC_PUBLIC_DEC KermacResult kermac_destroy(KermacHandle handle);
KERMAC_PUBLIC_DEC KermacResult kermac_make_current(KermacHandle handle);
KERMAC_PUBLIC_DEC KermacResult kermac_device_capability(KermacHandle handle, int* major_out, int* minor_out);

KERMAC_PUBLIC_DEC KermacResult kermac_host_alloc(void** ptr, size_t num_bytes);
KERMAC_PUBLIC_DEC KermacResult kermac_host_free(void* ptr);
KERMAC_PUBLIC_DEC KermacResult kermac_host_alloc_pinned(void** ptr, size_t num_bytes);
KERMAC_PUBLIC_DEC KermacResult kermac_host_free_pinned(void* ptr);

KERMAC_PUBLIC_DEC KermacResult kermac_device_alloc(KermacHandle kermac, void** ptr, size_t num_bytes);
KERMAC_PUBLIC_DEC KermacResult kermac_device_free(KermacHandle kermac, void* ptr);

KERMAC_PUBLIC_DEC
KermacResult
kermac_stack_allocator_create(
    KermacStackAllocator* stack_allocator,
    KermacMemorySpace memory_space,
    void* memory,
    size_t num_bytes
);

KERMAC_PUBLIC_DEC KermacResult kermac_stack_allocator_destroy(KermacStackAllocator stack_allocator);
KERMAC_PUBLIC_DEC KermacResult kermac_memory_pointer(KermacMemory memory, void** ptr);

KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_create(
    KermacTensor* tensor,
    KermacDataType data_type,
    const int64_t* extents,
    KermacStackAllocator* stack_allocator
);

KERMAC_PUBLIC_DEC KermacResult kermac_tensor_destroy(KermacTensor tensor);
KERMAC_PUBLIC_DEC KermacResult kermac_tensor_copy(KermacTensor src, KermacTensor dst, CUstream stream);

KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_broadcast_copy(
    KermacHandle handle, 
    KermacTensor src, 
    KermacTensor dst, 
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_print(
    KermacTensor tensor,
    KermacStackAllocator* hsa,
    int edge_items,
    CUstream stream
);

// Thread-safety: NOT thread-safe. Concurrent calls on the same tensor or
// tensors sharing the same stack_allocator require external synchronization.
KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_view_create(
    KermacTensor tensor,
    KermacTensor *view,
    uint32_t mode,
    size_t slice
);

KERMAC_PUBLIC_DEC KermacResult kermac_tensor_view_destroy(KermacTensor view);

KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_rng_f32(
    KermacHandle kermac,
    KermacTensor device_tensor,
    KermacRNGType rng_type,
    float scale, float shift,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_tensor_rng_u32(
    KermacHandle kermac,
    KermacTensor device_tensor,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_solve(
    KermacHandle handle,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator* dsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor factor_info,
    KermacTensor solve_info,
    CUstream primary_stream,
    CUstream* secondary_streams,
    size_t num_secondary_streams
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_syev_batched(
    KermacHandle handle,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator* dsa,
    KermacTensor a,
    KermacTensor w,
    KermacTensor info,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_permute(
    KermacHandle handle,
    float alpha,
    KermacTensor a, const char* modes_a,
    KermacTensor b, const char* modes_b,
    CUstream stream
);

KERMAC_PUBLIC_DEC
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
);

KERMAC_PUBLIC_DEC
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
);

KERMAC_PUBLIC_DEC
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
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_mse(
    KermacHandle kermac,
    KermacStackAllocator* dsa,
    KermacTensor a,         // *,B
    KermacTensor b,         // *,B
    KermacTensor mse,       // B
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_mse_accuracy(
    KermacHandle kermac,
    KermacStackAllocator* dsa,
    KermacTensor a,         // *,B
    KermacTensor b,         // *,B
    KermacTensor mse,       // B
    KermacTensor accuracy,  // B
    CUstream stream
);

KERMAC_PUBLIC_DEC KermacResult kermac_elementwise_laplace_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacElementwise* elementwise,
    float bandwidth
);

KERMAC_PUBLIC_DEC KermacResult kermac_elementwise_laplace_symm_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacElementwise* elementwise,
    KermacMatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
);

KERMAC_PUBLIC_DEC KermacResult kermac_elementwise_destroy(
    KermacElementwise elementwise
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_elementwise_run(
    KermacElementwise elementwise,
    KermacTensor a,
    KermacTensor c,
    CUstream stream
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_gemm_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_norm_l2_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_norm_l2_symm_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_laplace_l2_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    float bandwidth
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_laplace_l2_symm_create(
    KermacHandle handle, 
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_laplace_l2_symm_copy_grad_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiring* semiring,
    KermacMatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_destroy(KermacSemiring semiring);

KERMAC_PUBLIC_DEC 
KermacResult
kermac_semiring_run(
    KermacSemiring semiring,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_gradient_norm_l2_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacSemiringGradient* semiring
);

KERMAC_PUBLIC_DEC KermacResult kermac_semiring_gradient_destroy(KermacSemiringGradient semiring);

KERMAC_PUBLIC_DEC
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
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_agop_create(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    KermacAgop* agop
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_agop_destroy(
    KermacAgop agop
);

KERMAC_PUBLIC_DEC
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
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_logdet_norm_norm_h2(
    KermacHandle kermac,
    float lambda_reg,
    KermacTensor factored_matrix,
    KermacTensor alpha,
    KermacTensor y,
    KermacTensor logdet_norm,
    KermacTensor norm_H2,
    CUstream stream
);

typedef enum {
    KERMAC_ROW_NORMALIZE_FLAG_NONE = 0,
    KERMAC_ROW_NORMALIZE_FLAG_CENTER = 1u << 0,
    KERMAC_ROW_NORMALIZE_FLAG_SCALE = 1u << 1
} KermacRowNormalizeFlags;

KERMAC_PUBLIC_DEC
KermacResult
kermac_row_stats(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor mean,
    KermacTensor stdev,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_apply_row_stats(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor mean,
    KermacTensor stdev,
    uint32_t flags,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_stdev_rows(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor stdev,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_apply_stdev_rows(
    KermacHandle kermac,
    KermacTensor a,
    KermacTensor stdev,
    CUstream stream
);

typedef enum {
    _KERMAC_LEGACY_KERNEL_FUNCTION_COPY_DIAGONALS_FULL_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_COPY_TRIANGLES_LOWER_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_COPY_TRIANGLES_UPPER_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_FULL_SYMM_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_FULL_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_LOWER_SYMM_COPY_GRAD_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_LAPLACE_UPPER_SYMM_COPY_GRAD_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_POINTER_ARRAY_F32,
    _KERMAC_LEGACY_KERNEL_FUNCTION_NUM_ENUMS
} _KermacLegacyKernelFunction;

typedef struct {
    KermacHandle kermac;
    CUmodule module;
    CUfunction functions[_KERMAC_LEGACY_KERNEL_FUNCTION_NUM_ENUMS];
} KermacLegacyHandle;

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_create(
    KermacHandle kermac,
    KermacLegacyHandle* legacy
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_destroy(
    KermacLegacyHandle legacy
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_copy_diagonal(
    KermacLegacyHandle legacy,
    KermacTensor symmetric_tensor,
    KermacTensor diagonal,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_copy_triangle(
    KermacLegacyHandle legacy,
    KermacMatrixPackedType packed_type,
    KermacTensor symmetric_tensor,
    float diagonal_value,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_laplace(
    KermacLegacyHandle legacy,
    KermacTensor tensor,
    KermacTensor norm_x,
    KermacTensor norm_y,
    float bandwidth,
    float epsilon,
    CUstream stream
);

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
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_solve_compute_array(
    KermacLegacyHandle legacy,
    KermacTensor a,
    KermacTensor a_array,
    CUstream stream
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_legacy_solve(
    KermacHandle kermac,
    KermacMatrixPackedType packed_type,
    KermacTensor a,
    KermacTensor b,
    KermacTensor a_array,
    KermacTensor b_array,
    KermacTensor factor_info,
    KermacTensor solve_info,
    CUstream stream
);

typedef struct KermacCompilerHandleImpl* KermacCompilerHandle;
typedef struct KermacNvrtcCompilerHandleImpl* KermacNvrtcCompilerHandle;

KERMAC_PUBLIC_DEC
KermacResult
kermac_compiler_create(
    KermacCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    bool verbose
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_compiler_cubin(
    KermacCompilerHandle compiler,
    void* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_compiler_destroy(
    KermacCompilerHandle compiler
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_nvrtc_compiler_create(
    KermacNvrtcCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    const char* program_name,
    const char* program_source,
    const char* const* headers,
    const char* const* include_names,
    size_t num_headers,
    const char* const* options,
    size_t num_options,
    bool verbose
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_nvrtc_compiler_ptx(
    KermacNvrtcCompilerHandle compiler,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_nvrtc_compiler_destroy(
    KermacNvrtcCompilerHandle compiler
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_stack_ptx_inject_compile(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
);

KERMAC_PUBLIC_DEC
KermacResult
kermac_stack_ptx_inject_compile_with_routines(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const StackPtxInstruction* const* routines,
    size_t num_routines,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
);
