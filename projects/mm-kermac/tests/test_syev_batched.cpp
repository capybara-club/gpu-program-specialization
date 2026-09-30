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
#include <cmath>
#include <cstdint>
#include <iostream>

#include <kermac.hpp>
#include <check_result_helper.h>

using namespace kermac;

static const size_t NUM_BYTES = 1ull << 24;
static const int64_t N = 3;
static const int64_t BATCH = 3;

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

static int64_t index_matrix(const KermacTensor& t, int64_t row, int64_t col, int64_t batch) {
    return batch * t.stride[2] + col * t.stride[1] + row;
}

static int64_t index_w(const KermacTensor& t, int64_t eig_idx, int64_t batch) {
    return batch * t.stride[1] + eig_idx;
}

int main() {
    if (!has_cuda_device()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }

    const float matrices[BATCH][N][N] = {
        {
            {4.0f, 1.0f, 0.0f},
            {1.0f, 3.0f, 0.0f},
            {0.0f, 0.0f, 2.0f}
        },
        {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 5.0f, 0.0f},
            {0.0f, 0.0f, 9.0f}
        },
        {
            {2.0f, 1.0f, 0.0f},
            {1.0f, 2.0f, 1.0f},
            {0.0f, 1.0f, 2.0f}
        }
    };

    const double expected_eigs[BATCH][N] = {
        {2.0, 2.381966011250105, 4.618033988749895},
        {1.0, 5.0, 9.0},
        {0.585786437626905, 2.0, 3.414213562373095}
    };

    const double eig_tol = 2e-4;
    const double ortho_tol = 3e-4;
    const double residual_tol = 5e-4;

    try {
        Kermac kermac(2);
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);
        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            DeviceTensor<float> d_a(dsa, N, N, BATCH);
            DeviceTensor<float> d_w(dsa, N, BATCH);
            DeviceTensor<int32_t> d_info(dsa, BATCH);

            HostTensor<float> h_a(hsa, d_a.extent());
            float* h_a_ptr = h_a.ptr();
            const KermacTensor h_a_raw = h_a.raw();

            for (int64_t b = 0; b < BATCH; ++b) {
                for (int64_t col = 0; col < N; ++col) {
                    for (int64_t row = 0; row < N; ++row) {
                        h_a_ptr[index_matrix(h_a_raw, row, col, b)] = matrices[b][row][col];
                    }
                }
            }

            d_a.copy_from(h_a, stream);

            syev_batched(
                kermac,
                MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
                dsa,
                d_a,
                d_w,
                d_info,
                stream
            );

            HostTensor<float> h_vectors(hsa, d_a.extent());
            HostTensor<float> h_w(hsa, d_w.extent());
            HostTensor<int32_t> h_info(hsa, d_info.extent());

            h_vectors.copy_from(d_a, stream);
            h_w.copy_from(d_w, stream);
            h_info.copy_from(d_info, stream);
            cuCheck(cuStreamSynchronize(stream));

            const int32_t* h_info_ptr = h_info.ptr();
            for (int64_t b = 0; b < BATCH; ++b) {
                ASSERT(h_info_ptr[b] == 0);
            }

            const float* h_w_ptr = h_w.ptr();
            const KermacTensor h_w_raw = h_w.raw();
            for (int64_t b = 0; b < BATCH; ++b) {
                for (int64_t i = 0; i < N; ++i) {
                    double got = (double)h_w_ptr[index_w(h_w_raw, i, b)];
                    double expected = expected_eigs[b][i];
                    ASSERT(std::fabs(got - expected) < eig_tol);
                }
            }

            const float* h_vectors_ptr = h_vectors.ptr();
            const KermacTensor h_vectors_raw = h_vectors.raw();

            for (int64_t b = 0; b < BATCH; ++b) {
                for (int64_t j = 0; j < N; ++j) {
                    for (int64_t k = 0; k < N; ++k) {
                        double dot = 0.0;
                        for (int64_t r = 0; r < N; ++r) {
                            double vj = (double)h_vectors_ptr[index_matrix(h_vectors_raw, r, j, b)];
                            double vk = (double)h_vectors_ptr[index_matrix(h_vectors_raw, r, k, b)];
                            dot += vj * vk;
                        }
                        if (j == k) {
                            ASSERT(std::fabs(dot - 1.0) < ortho_tol);
                        } else {
                            ASSERT(std::fabs(dot) < ortho_tol);
                        }
                    }
                }

                for (int64_t j = 0; j < N; ++j) {
                    const double lambda = (double)h_w_ptr[index_w(h_w_raw, j, b)];
                    double residual_sq = 0.0;
                    for (int64_t row = 0; row < N; ++row) {
                        double av = 0.0;
                        for (int64_t col = 0; col < N; ++col) {
                            double a_rc = (double)matrices[b][row][col];
                            double v_c = (double)h_vectors_ptr[index_matrix(h_vectors_raw, col, j, b)];
                            av += a_rc * v_c;
                        }
                        double v_row = (double)h_vectors_ptr[index_matrix(h_vectors_raw, row, j, b)];
                        double err = av - lambda * v_row;
                        residual_sq += err * err;
                    }
                    ASSERT(std::sqrt(residual_sq) < residual_tol);
                }
            }
        }

        host_free(host_mem);
        device_free(kermac, device_mem);
        cuCheck(cuStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
