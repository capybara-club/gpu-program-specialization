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

#include <cusolverDn.h>

#include <math.h>

static const size_t NUM_BYTES = 1ull << 28;

static const int64_t N = 1000;
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
        LegacyHandle legacy(kermac);
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
            DeviceTensor<int> d_solve_info(dsa, 1);

            rng(kermac, d_a, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
            rng(kermac, d_labels, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);

            tensor_broadcast_copy(kermac, d_labels, d_sol, stream);

            laplace_l2_symm.run(d_a, d_a, d_matrix, stream);

            DeviceTensor<float*> d_matrix_array(dsa, L);
            DeviceTensor<float*> d_sol_array(dsa, L);

            solve_compute_array(legacy, d_matrix, d_matrix_array, stream);
            solve_compute_array(legacy, d_sol, d_sol_array, stream);

            solve(
                kermac,
                MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
                d_matrix,
                d_sol,
                d_matrix_array,
                d_sol_array,
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
                for (size_t i = 0; i < 1; i++) {
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

            ASSERT( max_diff < 1e-3 );
        {
            static const int64_t kKnownN = 3;
            static const int64_t kKnownBatch = 2;
            static const int64_t kKnownLabels = 3;

            DeviceTensor<float> d_a_known(dsa, kKnownN, kKnownN, kKnownBatch);
            DeviceTensor<float> d_b_known(dsa, kKnownN, kKnownLabels, kKnownBatch);
            DeviceTensor<float*> d_a_array_known(dsa, kKnownBatch);
            DeviceTensor<float*> d_b_array_known(dsa, kKnownBatch);
            DeviceTensor<int> d_factor_info_known(dsa, kKnownBatch);
            DeviceTensor<int> d_solve_info_known(dsa, 1);

            HostTensor<float> h_a_known(hsa, d_a_known.extent());
            HostTensor<float> h_b_known(hsa, d_b_known.extent());
            HostTensor<float> h_x_expected(hsa, d_b_known.extent());

            const float kA0[kKnownN][kKnownN] = {
                {4.0f, 2.0f, 0.0f},
                {2.0f, 5.0f, 2.0f},
                {0.0f, 2.0f, 5.0f}
            };
            const float kA1[kKnownN][kKnownN] = {
                {9.0f, 3.0f, 3.0f},
                {3.0f, 5.0f, 3.0f},
                {3.0f, 3.0f, 3.0f}
            };
            const float kX[kKnownBatch][kKnownLabels][kKnownN] = {
                {
                    {1.0f, 2.0f, 3.0f},
                    {0.0f, -1.0f, 2.0f},
                    {-2.0f, 1.0f, 0.0f}
                },
                {
                    {1.0f, 0.0f, 1.0f},
                    {2.0f, -1.0f, 1.0f},
                    {0.5f, -2.0f, 3.0f}
                }
            };

            float b_values[kKnownBatch][kKnownLabels][kKnownN] = {};
            for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                const float (*A)[kKnownN] = batch == 0 ? kA0 : kA1;
                for (int64_t label = 0; label < kKnownLabels; ++label) {
                    for (int64_t row = 0; row < kKnownN; ++row) {
                        float sum = 0.0f;
                        for (int64_t col = 0; col < kKnownN; ++col) {
                            sum += A[row][col] * kX[batch][label][col];
                        }
                        b_values[batch][label][row] = sum;
                    }
                }
            }

            {
                const auto a_raw = h_a_known.raw();
                float* a_ptr = h_a_known.ptr();
                for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                    const float (*A)[kKnownN] = batch == 0 ? kA0 : kA1;
                    for (int64_t col = 0; col < kKnownN; ++col) {
                        for (int64_t row = 0; row < kKnownN; ++row) {
                            int64_t idx = batch * a_raw.stride[2] + col * a_raw.stride[1] + row;
                            a_ptr[idx] = A[row][col];
                        }
                    }
                }
            }

            {
                const auto b_raw = h_b_known.raw();
                float* b_ptr = h_b_known.ptr();
                float* x_ptr = h_x_expected.ptr();
                for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                    for (int64_t label = 0; label < kKnownLabels; ++label) {
                        for (int64_t row = 0; row < kKnownN; ++row) {
                            int64_t idx = batch * b_raw.stride[2]
                                + label * b_raw.stride[1]
                                + row;
                            b_ptr[idx] = b_values[batch][label][row];
                            x_ptr[idx] = kX[batch][label][row];
                        }
                    }
                }
            }

            d_a_known.copy_from(h_a_known, stream);
            d_b_known.copy_from(h_b_known, stream);

            solve_compute_array(legacy, d_a_known, d_a_array_known, stream);
            solve_compute_array(legacy, d_b_known, d_b_array_known, stream);

            solve(
                kermac,
                MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
                d_a_known,
                d_b_known,
                d_a_array_known,
                d_b_array_known,
                d_factor_info_known,
                d_solve_info_known,
                stream
            );

            HostTensor<float> h_solution(hsa, d_b_known.extent());
            HostTensor<int> h_factor_info_known(hsa, d_factor_info_known.extent());
            HostTensor<int> h_solve_info_known(hsa, d_solve_info_known.extent());
            h_solution.copy_from(d_b_known, stream);
            h_factor_info_known.copy_from(d_factor_info_known, stream);
            h_solve_info_known.copy_from(d_solve_info_known, stream);
            cuCheck( cuStreamSynchronize(stream) );

            {
                int* h_factor_info_ptr = h_factor_info_known.ptr();
                for (int64_t i = 0; i < kKnownBatch; i++) {
                    ASSERT( h_factor_info_ptr[i] == 0 );
                }

                int* h_solve_info_ptr = h_solve_info_known.ptr();
                ASSERT( h_solve_info_ptr[0] == 0 );
            }

            double max_diff = 0.0;
            {
                const auto sol_raw = h_solution.raw();
                const float* sol_ptr = h_solution.ptr();
                const float* exp_ptr = h_x_expected.ptr();
                for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                    for (int64_t label = 0; label < kKnownLabels; ++label) {
                        for (int64_t row = 0; row < kKnownN; ++row) {
                            int64_t idx = batch * sol_raw.stride[2]
                                + label * sol_raw.stride[1]
                                + row;
                            double diff = fabs(sol_ptr[idx] - exp_ptr[idx]);
                            if (diff > max_diff) {
                                max_diff = diff;
                            }
                        }
                    }
                }
            }
            ASSERT( max_diff < 1e-4 );
        }
        {
            static const int64_t kKnownN = 3;
            static const int64_t kKnownBatch = 2;
            static const int64_t kKnownRhs = 3;

            DeviceTensor<float> d_a_known(dsa, kKnownN, kKnownN, kKnownBatch);
            DeviceTensor<float> d_b_known(dsa, kKnownN, kKnownRhs, kKnownBatch);
            DeviceTensor<float*> d_a_array_known(dsa, kKnownBatch);
            DeviceTensor<float*> d_b_array_known(dsa, kKnownBatch);
            DeviceTensor<int> d_solve_info_known(dsa, 1);

            HostTensor<float> h_a_known(hsa, d_a_known.extent());
            HostTensor<float> h_b_known(hsa, d_b_known.extent());
            HostTensor<float> h_x_expected(hsa, d_b_known.extent());

            const float kA0[kKnownN][kKnownN] = {
                {4.0f, 2.0f, 0.0f},
                {2.0f, 5.0f, 2.0f},
                {0.0f, 2.0f, 5.0f}
            };
            const float kA1[kKnownN][kKnownN] = {
                {9.0f, 3.0f, 3.0f},
                {3.0f, 5.0f, 3.0f},
                {3.0f, 3.0f, 3.0f}
            };
            const float kX[kKnownBatch][kKnownRhs][kKnownN] = {
                {
                    {1.0f, 2.0f, 3.0f},
                    {0.0f, -1.0f, 2.0f},
                    {-2.0f, 1.0f, 0.0f}
                },
                {
                    {1.0f, 0.0f, 1.0f},
                    {2.0f, -1.0f, 1.0f},
                    {0.5f, -2.0f, 3.0f}
                }
            };

            float b_values[kKnownBatch][kKnownRhs][kKnownN] = {};
            for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                const float (*A)[kKnownN] = batch == 0 ? kA0 : kA1;
                for (int64_t rhs = 0; rhs < kKnownRhs; ++rhs) {
                    for (int64_t row = 0; row < kKnownN; ++row) {
                        float sum = 0.0f;
                        for (int64_t col = 0; col < kKnownN; ++col) {
                            sum += A[row][col] * kX[batch][rhs][col];
                        }
                        b_values[batch][rhs][row] = sum;
                    }
                }
            }

            {
                const auto a_raw = h_a_known.raw();
                float* a_ptr = h_a_known.ptr();
                for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                    const float (*A)[kKnownN] = batch == 0 ? kA0 : kA1;
                    for (int64_t col = 0; col < kKnownN; ++col) {
                        for (int64_t row = 0; row < kKnownN; ++row) {
                            int64_t idx = batch * a_raw.stride[2] + col * a_raw.stride[1] + row;
                            a_ptr[idx] = A[row][col];
                        }
                    }
                }
            }

            {
                const auto b_raw = h_b_known.raw();
                float* b_ptr = h_b_known.ptr();
                float* x_ptr = h_x_expected.ptr();
                for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                    for (int64_t rhs = 0; rhs < kKnownRhs; ++rhs) {
                        for (int64_t row = 0; row < kKnownN; ++row) {
                            int64_t idx = batch * b_raw.stride[2]
                                + rhs * b_raw.stride[1]
                                + row;
                            b_ptr[idx] = b_values[batch][rhs][row];
                            x_ptr[idx] = kX[batch][rhs][row];
                        }
                    }
                }
            }

            d_a_known.copy_from(h_a_known, stream);
            d_b_known.copy_from(h_b_known, stream);

            solve_compute_array(legacy, d_a_known, d_a_array_known, stream);
            solve_compute_array(legacy, d_b_known, d_b_array_known, stream);

            DeviceTensor<int> d_factor_info_known(dsa, kKnownBatch);

            solve(
                kermac,
                MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
                d_a_known,
                d_b_known,
                d_a_array_known,
                d_b_array_known,
                d_factor_info_known,
                d_solve_info_known,
                stream
            );

            HostTensor<float> h_solution(hsa, d_b_known.extent());
            HostTensor<int> h_factor_info_known(hsa, d_factor_info_known.extent());
            HostTensor<int> h_solve_info_known(hsa, d_solve_info_known.extent());
            h_solution.copy_from(d_b_known, stream);
            h_factor_info_known.copy_from(d_factor_info_known, stream);
            h_solve_info_known.copy_from(d_solve_info_known, stream);
            cuCheck( cuStreamSynchronize(stream) );

            for (int64_t i = 0; i < kKnownBatch; ++i) {
                ASSERT( h_factor_info_known.ptr()[i] == 0 );
            }
            ASSERT( h_solve_info_known.ptr()[0] == 0 );

            double max_diff = 0.0;
            {
                const auto sol_raw = h_solution.raw();
                const float* sol_ptr = h_solution.ptr();
                const float* exp_ptr = h_x_expected.ptr();
                for (int64_t batch = 0; batch < kKnownBatch; ++batch) {
                    for (int64_t rhs = 0; rhs < kKnownRhs; ++rhs) {
                        for (int64_t row = 0; row < kKnownN; ++row) {
                            int64_t idx = batch * sol_raw.stride[2]
                                + rhs * sol_raw.stride[1]
                                + row;
                            double diff = fabs(sol_ptr[idx] - exp_ptr[idx]);
                            if (diff > max_diff) {
                                max_diff = diff;
                            }
                        }
                    }
                }
            }
            ASSERT( max_diff < 1e-4 );
        }
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
