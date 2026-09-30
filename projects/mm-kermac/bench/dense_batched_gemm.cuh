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

#include <cstdint>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace kermac_dense_batched_gemm {

inline const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

__global__ void assemble_dense_panels_kernel(
    const float* __restrict__ x,
    int64_t x_ld,
    const int32_t* __restrict__ indices,
    int64_t indices_batch_stride,
    float* __restrict__ panels,
    int64_t panel_ld,
    int64_t panel_batch_stride,
    int32_t num_rows,
    int32_t cols,
    int32_t batch_count
) {
    const int64_t total = static_cast<int64_t>(batch_count) * static_cast<int64_t>(cols) * static_cast<int64_t>(num_rows);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    int64_t linear = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);

    while (linear < total) {
        const int32_t row = static_cast<int32_t>(linear % num_rows);
        const int64_t tmp = linear / num_rows;
        const int32_t col = static_cast<int32_t>(tmp % cols);
        const int32_t batch = static_cast<int32_t>(tmp / cols);
        const int32_t feature = indices[static_cast<int64_t>(batch) * indices_batch_stride + col];
        panels[static_cast<int64_t>(batch) * panel_batch_stride + row + static_cast<int64_t>(col) * panel_ld] =
            feature >= 0 ? x[row + static_cast<int64_t>(feature) * x_ld] : 0.0f;
        linear += stride;
    }
}

inline cudaError_t launch_assemble_dense_panels(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* panels,
    int64_t panel_ld,
    int64_t panel_batch_stride,
    int32_t num_rows,
    int32_t cols,
    int32_t batch_count,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    const int64_t total = static_cast<int64_t>(batch_count) * static_cast<int64_t>(cols) * static_cast<int64_t>(num_rows);
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    const int grid = blocks < 65535 ? blocks : 65535;
    assemble_dense_panels_kernel<<<grid, kThreads, 0, stream>>>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        panels,
        panel_ld,
        panel_batch_stride,
        num_rows,
        cols,
        batch_count
    );
    return cudaPeekAtLastError();
}

inline cublasStatus_t launch_dense_batched_gemm(
    cublasHandle_t handle,
    cublasComputeType_t compute_type,
    cublasGemmAlgo_t algo,
    const float* panels,
    int64_t panel_ld,
    int64_t panel_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t cols,
    int32_t batch_count
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        cols,
        cols,
        num_rows,
        &alpha,
        panels,
        CUDA_R_32F,
        static_cast<int>(panel_ld),
        static_cast<long long>(panel_batch_stride),
        panels,
        CUDA_R_32F,
        static_cast<int>(panel_ld),
        static_cast<long long>(panel_batch_stride),
        &beta,
        out,
        CUDA_R_32F,
        static_cast<int>(out_ld),
        static_cast<long long>(out_batch_stride),
        batch_count,
        compute_type,
        algo
    );
}

struct DenseBatchedGemmLtPlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic = {};
    size_t workspace_size = 0;
};

inline cublasStatus_t create_dense_batched_gemm_lt_plan(
    DenseBatchedGemmLtPlan* plan,
    cublasComputeType_t compute_type,
    int64_t panel_ld,
    int64_t panel_batch_stride,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t cols,
    int32_t batch_count,
    size_t workspace_size
) {
    cublasStatus_t status = cublasLtCreate(&plan->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = cublasLtMatmulDescCreate(&plan->operation_desc, compute_type, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(
        plan->operation_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transa,
        sizeof(transa)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatmulDescSetAttribute(
        plan->operation_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = cublasLtMatrixLayoutCreate(&plan->a_desc, CUDA_R_32F, num_rows, cols, panel_ld);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        plan->a_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        plan->a_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &panel_batch_stride,
        sizeof(panel_batch_stride)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = cublasLtMatrixLayoutCreate(&plan->b_desc, CUDA_R_32F, num_rows, cols, panel_ld);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        plan->b_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        plan->b_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &panel_batch_stride,
        sizeof(panel_batch_stride)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_32F, cols, cols, out_ld);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        plan->c_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        plan->c_desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &out_batch_stride,
        sizeof(out_batch_stride)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = cublasLtMatmulPreferenceCreate(&plan->preference);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatmulPreferenceSetAttribute(
        plan->preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    int returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        plan->handle,
        plan->operation_desc,
        plan->a_desc,
        plan->b_desc,
        plan->c_desc,
        plan->c_desc,
        plan->preference,
        1,
        &plan->heuristic,
        &returned_results
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (returned_results == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    plan->workspace_size = workspace_size;
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t launch_dense_batched_gemm_lt(
    const DenseBatchedGemmLtPlan& plan,
    const float* panels,
    float* out,
    void* workspace,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasLtMatmul(
        plan.handle,
        plan.operation_desc,
        &alpha,
        panels,
        plan.a_desc,
        panels,
        plan.b_desc,
        &beta,
        out,
        plan.c_desc,
        out,
        plan.c_desc,
        &plan.heuristic.algo,
        workspace,
        plan.workspace_size,
        stream
    );
    return status;
}

inline cublasStatus_t destroy_dense_batched_gemm_lt_plan(DenseBatchedGemmLtPlan* plan) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    if (plan->preference) {
        status = cublasLtMatmulPreferenceDestroy(plan->preference);
        plan->preference = nullptr;
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    if (plan->c_desc) {
        status = cublasLtMatrixLayoutDestroy(plan->c_desc);
        plan->c_desc = nullptr;
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    if (plan->b_desc) {
        status = cublasLtMatrixLayoutDestroy(plan->b_desc);
        plan->b_desc = nullptr;
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    if (plan->a_desc) {
        status = cublasLtMatrixLayoutDestroy(plan->a_desc);
        plan->a_desc = nullptr;
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    if (plan->operation_desc) {
        status = cublasLtMatmulDescDestroy(plan->operation_desc);
        plan->operation_desc = nullptr;
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    if (plan->handle) {
        status = cublasLtDestroy(plan->handle);
        plan->handle = nullptr;
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    return status;
}

}  // namespace kermac_dense_batched_gemm
