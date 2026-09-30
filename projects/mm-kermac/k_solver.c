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
_kermac_solve(
    KermacHandle handle,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator* dsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor factor_info,
    KermacTensor solve_info,
    CUstream primary_stream,
    CUstream* secondary_streams,
    size_t num_secondary_streams,
    KermacTensor* workspace,
    KermacTensor* a_view,
    KermacTensor* b_view,
    KermacTensor* factor_info_view,
    KermacTensor* solve_info_view,
    KermacTensor* workspace_view
) {
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (factor_info.data_type != KERMAC_DATA_TYPE_INT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (solve_info.data_type != KERMAC_DATA_TYPE_INT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (factor_info.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (solve_info.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (num_secondary_streams > _KERMAC_MAX_STREAMS) {
        _KERMAC_ERROR( KERMAC_ERROR_TOO_MANY_STREAMS );
    }

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
    int64_t a_batch = a.num_modes == 2 ? 1 : a.extent[2];
    int64_t b_batch = b.num_modes == 2 ? 1 : b.extent[2];
    int64_t ldA = a.stride[1];
    int64_t ldB = b.stride[1];
    int64_t num_rhs = b.extent[1];

    if (num_rows != a.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_rows != b.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a_batch != b_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a_batch != factor_info.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a_batch != solve_info.extent[0]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    cublasFillMode_t uplo;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_LOWER;
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_UPPER;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    cudaDataType data_type = CUDA_R_32F;
    cusolverDnParams_t params = NULL;
    size_t workspace_device_size;
    size_t workspace_host_size;
    _KERMAC_CUSOLVER_CHECK_RET(
        cusolverDnXpotrf_bufferSize(
            handle->cusolver,
            params,
            uplo,
            num_rows,
            data_type,
            NULL,
            ldA,
            data_type,
            &workspace_device_size,
            &workspace_host_size
        )
    );
    if (workspace_host_size != 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            workspace, KERMAC_DATA_TYPE_BYTE,
            (KermacExtent){(int64_t)workspace_device_size, a_batch},
            dsa
        )
    );

    _KERMAC_CUDA_CHECK_RET(
        cuStreamWaitEvent(primary_stream, handle->primary_event_end, CU_EVENT_WAIT_DEFAULT)
    );
    _KERMAC_CUDA_CHECK_RET( cuEventRecord(handle->primary_event_start, primary_stream) );

    CUstream previous_cusolver_stream;
    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnGetStream(handle->cusolver, &previous_cusolver_stream) );

    for (size_t i = 0; i < num_secondary_streams; i++) {
        _KERMAC_CUDA_CHECK_RET( 
            cuStreamWaitEvent(secondary_streams[i], handle->primary_event_start, CU_EVENT_WAIT_DEFAULT)
        );
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool factor_info_is_dry = factor_info.memory.stack_allocator->is_dry;
    bool solve_info_is_dry = solve_info.memory.stack_allocator->is_dry;
    bool workspace_is_dry = workspace->memory.stack_allocator->is_dry;

    bool is_dry;
    if (a_is_dry && b_is_dry && factor_info_is_dry && solve_info_is_dry && workspace_is_dry) {
        is_dry = true;
    } else if (!a_is_dry && !b_is_dry && !factor_info_is_dry && !solve_info_is_dry && !workspace_is_dry) {
        is_dry = false;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }
    
    for (size_t i = 0; i < a_batch; i++) {
        size_t stream_idx = i % num_secondary_streams;
        CUstream secondary_stream = secondary_streams[stream_idx];
        _KERMAC_CUSOLVER_CHECK_RET( cusolverDnSetStream(handle->cusolver, secondary_stream) );

        _KERMAC_CHECK_RET( kermac_tensor_view_create(a, a_view, 2, i) );
        _KERMAC_CHECK_RET( kermac_tensor_view_create(b, b_view, 2, i) );
        _KERMAC_CHECK_RET( kermac_tensor_view_create(*workspace, workspace_view, 1, i) );
        _KERMAC_CHECK_RET( kermac_tensor_view_create(factor_info, factor_info_view, 0, i) );
        _KERMAC_CHECK_RET( kermac_tensor_view_create(solve_info, solve_info_view, 0, i) );

        if (!is_dry) {
            void* a_view_ptr;
            void* b_view_ptr;
            void* workspace_view_ptr;
            int* factor_info_view_ptr;
            int* solve_info_view_ptr;

            _KERMAC_CHECK_RET( kermac_memory_pointer(a_view->memory, &a_view_ptr) );
            _KERMAC_CHECK_RET( kermac_memory_pointer(b_view->memory, &b_view_ptr) );
            _KERMAC_CHECK_RET( kermac_memory_pointer(workspace_view->memory, &workspace_view_ptr) );
            _KERMAC_CHECK_RET( kermac_memory_pointer(factor_info_view->memory, (void**)&factor_info_view_ptr) );
            _KERMAC_CHECK_RET( kermac_memory_pointer(solve_info_view->memory, (void**)&solve_info_view_ptr) );

            _KERMAC_CUSOLVER_CHECK_RET(
                cusolverDnXpotrf(
                    handle->cusolver,
                    params,
                    uplo,
                    num_rows,
                    data_type,
                    a_view_ptr,
                    ldA,
                    data_type,
                    workspace_view_ptr,
                    workspace_device_size,
                    NULL,
                    workspace_host_size,
                    factor_info_view_ptr
                )
            );

            _KERMAC_CUSOLVER_CHECK_RET(
                cusolverDnXpotrs(
                    handle->cusolver,
                    params,
                    uplo,
                    num_rows,
                    num_rhs,
                    data_type,
                    a_view_ptr,
                    ldA,
                    data_type,
                    b_view_ptr,
                    ldB,
                    solve_info_view_ptr
                )
            );
        }

        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(*solve_info_view) );
        *solve_info_view = KERMAC_ZERO_INIT(KermacTensor);
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(*factor_info_view) );
        *factor_info_view = KERMAC_ZERO_INIT(KermacTensor);
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(*workspace_view) );
        *workspace_view = KERMAC_ZERO_INIT(KermacTensor);
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(*b_view) );
        *b_view = KERMAC_ZERO_INIT(KermacTensor);
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(*a_view) );
        *a_view = KERMAC_ZERO_INIT(KermacTensor);
    }

    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnSetStream(handle->cusolver, previous_cusolver_stream) );

    for (size_t i = 0; i < num_secondary_streams; i++) {
        _KERMAC_CUDA_CHECK_RET( cuEventRecord(handle->events[i], secondary_streams[i]) );
        _KERMAC_CUDA_CHECK_RET( cuStreamWaitEvent(primary_stream, handle->events[i], CU_EVENT_WAIT_DEFAULT) );
    }

    _KERMAC_CUDA_CHECK_RET( cuEventRecord(handle->primary_event_end, primary_stream) );

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_solve(
    KermacHandle handle,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator *dsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor factor_info,
    KermacTensor solve_info,
    CUstream primary_stream,
    CUstream *secondary_streams,
    size_t num_secondary_streams
) {
    KermacTensor workspace = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor a_view = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor b_view = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor factor_info_view = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor solve_info_view = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor workspace_view = KERMAC_ZERO_INIT(KermacTensor);

    KermacResult status = 
        _kermac_solve(
            handle,
            packed_type,
            dsa, a, b, 
            factor_info,
            solve_info,
            primary_stream,
            secondary_streams,
            num_secondary_streams,
            &workspace,
            &a_view, &b_view,
            &factor_info_view, &solve_info_view,
            &workspace_view
        );
    if (workspace.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(workspace) );
    }
    if (solve_info_view.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(solve_info_view) );
    }
    if (factor_info_view.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(factor_info_view) );
    }
    if (workspace_view.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(workspace_view) );
    }
    if (b_view.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(b_view) );
    }
    if (a_view.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_view_destroy(a_view) );
    }
    return status;
}

static
inline
KermacResult
_kermac_syev_batched(
    KermacHandle handle,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator* dsa,
    KermacTensor a,
    KermacTensor w,
    KermacTensor info,
    CUstream stream,
    KermacTensor* workspace_device,
    KermacTensor* workspace_w_contiguous,
    void** workspace_host
) {
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (w.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (info.data_type != KERMAC_DATA_TYPE_INT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (dsa->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (w.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (info.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (a.num_modes != 2 && a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (w.num_modes != 1 && w.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (info.num_modes != 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t num_rows = a.extent[0];
    int64_t a_batch = a.num_modes == 2 ? 1 : a.extent[2];
    int64_t ldA = a.stride[1];
    bool w_is_2d = w.num_modes == 2;

    if (num_rows < 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (a_batch < 1) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_rows != a.extent[1]) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (ldA < num_rows) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (info.extent[0] != a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (num_rows > INT64_MAX / a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    int64_t w_values_count = num_rows * a_batch;
    if (!w_is_2d) {
        if (w.extent[0] != w_values_count) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    } else {
        if (w.extent[0] != num_rows || w.extent[1] != a_batch) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }
    if (num_rows > 0 && ldA > INT64_MAX / num_rows) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    int64_t n_times_lda = num_rows * ldA;
    if (a_batch > 0 && n_times_lda > INT64_MAX / a_batch) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (n_times_lda * a_batch > INT32_MAX) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    cublasFillMode_t uplo;
    if (packed_type == KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_LOWER;
    } else if (packed_type == KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
        uplo = CUBLAS_FILL_MODE_UPPER;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    cusolverDnParams_t params = NULL;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cudaDataType data_type = CUDA_R_32F;
    size_t workspace_device_size = 0;
    size_t workspace_host_size = 0;
    _KERMAC_CUSOLVER_CHECK_RET(
        cusolverDnXsyevBatched_bufferSize(
            handle->cusolver,
            params,
            jobz,
            uplo,
            num_rows,
            data_type,
            NULL,
            ldA,
            data_type,
            NULL,
            data_type,
            &workspace_device_size,
            &workspace_host_size,
            a_batch
        )
    );

    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            workspace_device,
            KERMAC_DATA_TYPE_BYTE,
            (KermacExtent){(int64_t)workspace_device_size},
            dsa
        )
    );

    if (w_is_2d) {
        if (w_values_count > INT64_MAX / (int64_t)sizeof(float)) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
        _KERMAC_CHECK_RET(
            kermac_tensor_create(
                workspace_w_contiguous,
                KERMAC_DATA_TYPE_BYTE,
                (KermacExtent){w_values_count * (int64_t)sizeof(float)},
                dsa
            )
        );
    }

    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool w_is_dry = w.memory.stack_allocator->is_dry;
    bool info_is_dry = info.memory.stack_allocator->is_dry;
    bool workspace_device_is_dry = workspace_device->memory.stack_allocator->is_dry;
    bool is_dry;
    if (w_is_2d) {
        bool workspace_w_contiguous_is_dry = workspace_w_contiguous->memory.stack_allocator->is_dry;
        if (
            a_is_dry &&
            w_is_dry &&
            info_is_dry &&
            workspace_device_is_dry &&
            workspace_w_contiguous_is_dry
        ) {
            is_dry = true;
        } else if (
            !a_is_dry &&
            !w_is_dry &&
            !info_is_dry &&
            !workspace_device_is_dry &&
            !workspace_w_contiguous_is_dry
        ) {
            is_dry = false;
        } else {
            _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
        }
    } else {
        if (a_is_dry && w_is_dry && info_is_dry && workspace_device_is_dry) {
            is_dry = true;
        } else if (!a_is_dry && !w_is_dry && !info_is_dry && !workspace_device_is_dry) {
            is_dry = false;
        } else {
            _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
        }
    }

    if (!is_dry && workspace_host_size > 0) {
        _KERMAC_CHECK_RET( kermac_host_alloc(workspace_host, workspace_host_size) );
    }

    _KERMAC_CUDA_CHECK_RET(
        cuStreamWaitEvent(stream, handle->primary_event_end, CU_EVENT_WAIT_DEFAULT)
    );
    _KERMAC_CUDA_CHECK_RET( cuEventRecord(handle->primary_event_start, stream) );

    CUstream previous_cusolver_stream;
    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnGetStream(handle->cusolver, &previous_cusolver_stream) );
    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnSetStream(handle->cusolver, stream) );

    if (!is_dry) {
        void* a_ptr;
        void* w_ptr;
        void* workspace_device_ptr;
        void* workspace_w_contiguous_ptr = NULL;
        int* info_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(a.memory, &a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(w.memory, &w_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(workspace_device->memory, &workspace_device_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(info.memory, (void**)&info_ptr) );

        void* w_compute_ptr = w_ptr;
        if (w_is_2d) {
            _KERMAC_CHECK_RET(
                kermac_memory_pointer(workspace_w_contiguous->memory, &workspace_w_contiguous_ptr)
            );
            w_compute_ptr = workspace_w_contiguous_ptr;
        }

        _KERMAC_CUSOLVER_CHECK_RET(
            cusolverDnXsyevBatched(
                handle->cusolver,
                params,
                jobz,
                uplo,
                num_rows,
                data_type,
                a_ptr,
                ldA,
                data_type,
                w_compute_ptr,
                data_type,
                workspace_device_ptr,
                workspace_device_size,
                *workspace_host,
                workspace_host_size,
                info_ptr,
                a_batch
            )
        );

        if (w_is_2d) {
            CUDA_MEMCPY2D eigen_copy = {0};

            eigen_copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            eigen_copy.srcDevice = (CUdeviceptr)workspace_w_contiguous_ptr;
            eigen_copy.srcPitch = num_rows * sizeof(float);

            eigen_copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            eigen_copy.dstDevice = (CUdeviceptr)w_ptr;
            eigen_copy.dstPitch = w.stride[1] * sizeof(float);

            eigen_copy.WidthInBytes = num_rows * sizeof(float);
            eigen_copy.Height = a_batch;
            _KERMAC_CUDA_CHECK_RET( cuMemcpy2DAsync(&eigen_copy, stream) );
        }
    }

    _KERMAC_CUSOLVER_CHECK_RET( cusolverDnSetStream(handle->cusolver, previous_cusolver_stream) );
    _KERMAC_CUDA_CHECK_RET( cuEventRecord(handle->primary_event_end, stream) );

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_syev_batched(
    KermacHandle handle,
    KermacMatrixPackedType packed_type,
    KermacStackAllocator* dsa,
    KermacTensor a,
    KermacTensor w,
    KermacTensor info,
    CUstream stream
) {
    KermacTensor workspace_device = KERMAC_ZERO_INIT(KermacTensor);
    KermacTensor workspace_w_contiguous = KERMAC_ZERO_INIT(KermacTensor);
    void* workspace_host = NULL;

    KermacResult status =
        _kermac_syev_batched(
            handle,
            packed_type,
            dsa,
            a,
            w,
            info,
            stream,
            &workspace_device,
            &workspace_w_contiguous,
            &workspace_host
        );

    if (workspace_host != NULL) {
        _KERMAC_CHECK_RET( kermac_host_free(workspace_host) );
    }
    if (workspace_w_contiguous.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(workspace_w_contiguous) );
    }
    if (workspace_device.memory.stack_allocator != NULL) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(workspace_device) );
    }

    return status;
}
