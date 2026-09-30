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
// fmnist_legacy_solver.cpp

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include <cuda.h>

#include <kermac.hpp>
#include <kermac_cpu.hpp>
#include <fmnist.h>
#include <check_result_helper.h>

using namespace kermac;

constexpr std::size_t NUM_SECONDARY_STREAMS = 128;

static const MatrixPackedType train_packed_type = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
static const MatrixPackedType solve_packed_type = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;

static bool has_cuda_device() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

struct KernelOption {
    enum Mode {
        Fused, // Uses only semiring
        Split,
        Legacy
    } mode;

    Semiring* semiring;
    Elementwise* elementwise;
    float legacy_bandwidth;
    float legacy_regularizer;
    float legacy_epsilon;
};

void
func_legacy(
    Kermac&               handle,
    LegacyHandle&         legacy,
    HostStackAllocator&   hsa,
    DeviceStackAllocator& dsa,
    KernelOption&         train_op,
    KernelOption&         test_op,
    CUstream              stream
) {
    bool is_dry = dsa.get().is_dry;

    const int64_t max_num_train = 10000;
    const int64_t max_num_test  = 10000;
    const int64_t num_batches   = 10;
    const float mse_threshold = 0.03f;
    const float acc_threshold = 0.85f;

    TensorCoreMode tcm = KERMAC_TENSOR_CORE_MODE_F32;

    int64_t num_train, num_test, num_dims, num_labels;

    fmnist_dims(
        max_num_train,
        max_num_test,
        &num_train,
        &num_test,
        &num_dims,
        &num_labels
    );

    HostTensor<float> x_train(hsa, num_train, num_dims);
    HostTensor<float> y_train(hsa, num_train, num_labels);
    HostTensor<float> x_test (hsa, num_test, num_dims);
    HostTensor<float> y_test (hsa, num_test, num_labels);

    fmnist_load(x_train, y_train, x_test, y_test);

    DeviceTensor<float> d_solution(dsa, num_train, num_labels, num_batches);
    DeviceTensor<float> d_x_train(dsa, num_train, num_dims,   num_batches);
    DeviceTensor<float> d_y_train(dsa, num_train, num_labels);
    DeviceTensor<float> d_x_test (dsa, num_test,  num_dims,   num_batches);
    DeviceTensor<float> d_y_test (dsa, num_test,  num_labels);

    d_x_train.copy_from(x_train, stream);
    d_y_train.copy_from(y_train, stream);
    d_x_test.copy_from(x_test, stream);
    d_y_test.copy_from(y_test, stream);

    tensor_broadcast_copy(handle, d_y_train, d_solution, stream);

    {
        DeviceTensor<float> d_kernel_matrix(dsa, num_train, num_train, num_batches);

        train_op.semiring->run(d_x_train, d_x_train, d_kernel_matrix, stream);

        DeviceTensor<float*> d_kernel_matrix_array(dsa, num_batches);
        DeviceTensor<float*> d_solution_array(dsa, num_batches);
        solve_compute_array(legacy, d_kernel_matrix, d_kernel_matrix_array, stream);
        solve_compute_array(legacy, d_solution, d_solution_array, stream);

        DeviceTensor<int> d_factor_info(dsa, num_batches);
        DeviceTensor<int> d_solve_info(dsa, 1);

        solve(
            handle,
            solve_packed_type,
            d_kernel_matrix,
            d_solution,
            d_kernel_matrix_array,
            d_solution_array,
            d_factor_info,
            d_solve_info,
            stream
        );

        HostTensor<int> h_factor_info(hsa, d_factor_info.extent());
        HostTensor<int> h_solve_info(hsa, d_solve_info.extent());

        h_factor_info.copy_from(d_factor_info, stream);
        h_solve_info.copy_from(d_solve_info, stream);

        if (!is_dry) {
            int* h_factor_info_ptr = h_factor_info.ptr();
            int* h_solve_info_ptr = h_solve_info.ptr();
            for (int64_t i = 0; i < num_batches; i++) {
                ASSERT( h_factor_info_ptr[i] == 0 );
            }
            ASSERT( h_solve_info_ptr[0] == 0 );
        }
    }

    {
        DeviceTensor<float> d_kernel_matrix(dsa, num_train, num_test, num_batches);
        DeviceTensor<float> d_preds(dsa, num_test, num_labels, num_batches);

        test_op.semiring->run(d_x_train, d_x_test, d_kernel_matrix, stream);

        contraction(
            handle,
            dsa,
            tcm,
            1.0f,
            d_kernel_matrix, "mnl",
            d_solution, "mcl",
            0.0f,
            d_preds,         "ncl",
            d_preds,         "ncl",
            stream
        );

        DeviceTensor<float> mse_tensor(dsa, num_batches);
        DeviceTensor<float> acc_tensor(dsa, num_batches);

        HostTensor<float> h_mse_tensor(hsa, mse_tensor.extent());
        HostTensor<float> h_acc_tensor(hsa, acc_tensor.extent());

        mse_accuracy(
            handle,
            dsa,
            d_preds,
            d_y_test,
            mse_tensor,
            acc_tensor,
            stream
        );

        if (!is_dry) {
            h_mse_tensor.copy_from(mse_tensor, stream);
            h_acc_tensor.copy_from(acc_tensor, stream);

            float* h_mse_tensor_ptr = h_mse_tensor.ptr();
            float* h_acc_tensor_ptr = h_acc_tensor.ptr();

            for (int64_t i = 0; i < num_batches; i++) {
                ASSERT( std::isfinite(h_mse_tensor_ptr[i]) );
                ASSERT( std::isfinite(h_acc_tensor_ptr[i]) );
                ASSERT( h_mse_tensor_ptr[i] >= 0.0f );
                ASSERT( h_acc_tensor_ptr[i] >= 0.0f && h_acc_tensor_ptr[i] <= 1.0f );
                ASSERT( h_mse_tensor_ptr[i] <= mse_threshold );
                ASSERT( h_acc_tensor_ptr[i] >= acc_threshold );
            }
        }

        mse(
            handle,
            dsa,
            d_preds,
            d_y_test,
            mse_tensor,
            stream
        );

        if (!is_dry) {
            h_mse_tensor.copy_from(mse_tensor, stream);

            float* h_mse_tensor_ptr = h_mse_tensor.ptr();

            for (int64_t i = 0; i < num_batches; i++) {
                ASSERT( std::isfinite(h_mse_tensor_ptr[i]) );
                ASSERT( h_mse_tensor_ptr[i] >= 0.0f );
                ASSERT( h_mse_tensor_ptr[i] <= mse_threshold );
            }
        }
    }
}

int
main() {
    if (!has_cuda_device()) {
        std::fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }
    try {
        CUstream stream;

        Kermac handle(NUM_SECONDARY_STREAMS);
        LegacyHandle legacy(handle);

        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        static const float bandwidth   = 10.0f;
        static const float regularizer = 1e-3f;
        static const float epsilon     = 1e-5f;

        size_t host_bytes = 1ull << 30;
        void* host_memory_ptr = host_alloc(host_bytes);
        HostStackAllocator hsa(host_memory_ptr, host_bytes);

        Semiring train_laplace_l2_symm_semiring =
            Semiring::laplace_l2_symm(
                handle,
                hsa,
                train_packed_type,
                bandwidth,
                regularizer,
                epsilon
            );

        Semiring test_laplace_l2_semiring = Semiring::laplace_l2(handle, hsa, bandwidth);

        KernelOption train_fused_op;
        train_fused_op.mode = KernelOption::Fused;
        train_fused_op.semiring = &train_laplace_l2_symm_semiring;
        train_fused_op.elementwise = nullptr;

        KernelOption test_fused_op;
        test_fused_op.mode = KernelOption::Fused;
        test_fused_op.semiring = &test_laplace_l2_semiring;
        test_fused_op.elementwise = nullptr;

        std::size_t device_bytes = 0;
        {
            DeviceStackAllocator dsa_dry(nullptr, 0);
            func_legacy(handle, legacy, hsa, dsa_dry, train_fused_op, test_fused_op, stream);
            device_bytes = static_cast<std::size_t>(dsa_dry.get().largest_total_offset);
        }

        CUdeviceptr device_memory_ptr = (CUdeviceptr)device_alloc(handle, device_bytes);
        {
            DeviceStackAllocator dsa_real(
                reinterpret_cast<void*>(device_memory_ptr),
                device_bytes
            );
            func_legacy(handle, legacy, hsa, dsa_real, train_fused_op, test_fused_op, stream);
        }

        device_free(handle, (void*)device_memory_ptr);
        host_free(host_memory_ptr);

        cuCheck(cuStreamDestroy(stream));
        return 0;
    }
    catch (const kermac::Error& e) {
        std::fprintf(stderr, "Kermac error: %s\n", e.what());
        return 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
}
