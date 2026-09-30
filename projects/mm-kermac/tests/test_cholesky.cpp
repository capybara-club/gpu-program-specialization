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

#include <iostream>

#include <kermac.hpp>
#include <kermac_cpu.hpp>
#include <check_result_helper.h>

#include <math.h>

static const size_t NUM_BYTES = 1ull << 28;

static const int64_t N = 100;
static const int64_t K = 10;
static const int64_t L = 10;

static const float lambda_reg = 1e-3;

static const double a_tol = 1e-5;
static const double r_tol = 1e-4;

using namespace kermac;

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

int 
main() {
    if (!has_cuda_device()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }
    try {
        Kermac kermac(2);
        KermacLegacyHandle legacy;
        CUstream stream;
        cuCheck( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );

        void* host_mem = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);
        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            Semiring laplace_l2_symm = 
                Semiring::laplace_l2_symm(
                    kermac, 
                    hsa, 
                    MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE, 
                    10.0f,
                    1e-3, 
                    1e-5
                );

            Semiring laplace_l2_full = 
                Semiring::laplace_l2_symm(
                    kermac, 
                    hsa, 
                    MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_FULL, 
                    10.0f,
                    1e-3, 
                    1e-5
                );
            
            DeviceTensor<float> d_a(dsa, N, K, L);
            DeviceTensor<float> d_matrix(dsa, N, N, L);
            DeviceTensor<float> d_labels(dsa, N, 1);
            DeviceTensor<float> d_sol(dsa, N, 1, L);
            DeviceTensor<int> d_factor_info(dsa, L);
            DeviceTensor<int> d_solve_info(dsa, L);

            rng(kermac, d_a, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
            rng(kermac, d_labels, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);

            tensor_broadcast_copy(kermac, d_labels, d_sol, stream);

            laplace_l2_symm.run(d_a, d_a, d_matrix, stream);

            solve(
                kermac,
                MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
                dsa,
                d_matrix,
                d_sol,
                d_factor_info,
                d_solve_info,
                stream
            );

            {
                HostTensor<int> h_factor_info(hsa, d_factor_info.extent());
                h_factor_info.copy_from(d_factor_info, stream);
                int* h_factor_info_ptr = h_factor_info.ptr();
                for (size_t i = 0; i < L; i++) {
                    ASSERT( h_factor_info_ptr[i] == 0 );
                }

                HostTensor<int> h_solve_info(hsa, d_solve_info.extent());
                h_solve_info.copy_from(d_solve_info, stream);
                int* h_solve_info_ptr = h_solve_info.ptr();
                for (size_t i = 0; i < L; i++) {
                    ASSERT( h_solve_info_ptr[i] == 0 );
                }
            }

            laplace_l2_full.run(d_a, d_a, d_matrix, stream);

            DeviceTensor<float> d_preds(dsa, N, 1, L);
            contraction(
                kermac, dsa, TensorCoreMode::KERMAC_TENSOR_CORE_MODE_F32,
                1.0f, 
                d_matrix, "mnl",
                d_sol, "mcl",
                0.0f,
                d_preds, "ncl",
                d_preds, "ncl",
                stream
            );

            DeviceTensor<float> d_labels_broadcast(dsa, N, 1, L);
            tensor_broadcast_copy(kermac, d_labels, d_labels_broadcast, stream);

            float max_diff;
            max_diff_cpu(
                hsa,
                d_preds,
                d_labels_broadcast,
                &max_diff,
                stream
            );

            ASSERT( max_diff < 1e-4 );
        }
        cuCheck( cuStreamSynchronize(stream) );
        host_free(host_mem);
        device_free(kermac, device_mem);
        cuCheck( cuStreamDestroy(stream) );
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
