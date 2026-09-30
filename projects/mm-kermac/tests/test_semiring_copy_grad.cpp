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
#include <kermac_cpu.hpp>
#include <check_result_helper.h>

using namespace kermac;

static const size_t NUM_BYTES = 1ull << 26;

static constexpr float kTolLaplaceGrad = 5e-2f;

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

static int64_t batch_count(const KermacTensor& t) {
    return t.num_modes < 3 ? 1 : t.extent[2];
}

static int64_t batch_stride(const KermacTensor& t) {
    return t.num_modes < 3 ? 0 : t.stride[2];
}

static void fill_tensor_pattern(
    HostTensor<float>& t,
    float row_scale,
    float col_scale,
    float batch_scale,
    float bias
) {
    const KermacTensor raw = t.raw();
    const int64_t rows = raw.extent[0];
    const int64_t cols = raw.extent[1];
    const int64_t batches = batch_count(raw);
    const int64_t ld = raw.stride[1];
    const int64_t bs = batch_stride(raw);
    float* ptr = t.ptr();

    for (int64_t b = 0; b < batches; ++b) {
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t r = 0; r < rows; ++r) {
                const int64_t idx = b * bs + c * ld + r;
                ptr[idx] = bias
                    + row_scale * (float)(r + 1)
                    + col_scale * (float)(c + 1)
                    + batch_scale * (float)(b + 1);
            }
        }
    }
}

static void compute_expected_laplace_symm_copy_grad(
    const HostTensor<float>& base,
    HostTensor<float>& expected,
    MatrixPackedType packed_type,
    float bandwidth,
    float regularizer,
    float epsilon
) {
    const KermacTensor b = base.raw();
    const KermacTensor e = expected.raw();

    const int64_t rows = b.extent[0];
    const int64_t cols = b.extent[1];
    const int64_t batches = batch_count(b);

    const int64_t b_ld = b.stride[1];
    const int64_t e_ld = e.stride[1];
    const int64_t b_bs = batch_stride(b);
    const int64_t e_bs = batch_stride(e);

    const float* b_ptr = base.ptr();
    float* e_ptr = expected.ptr();

    for (int64_t b_idx = 0; b_idx < batches; ++b_idx) {
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t r = 0; r < rows; ++r) {
                const int64_t base_idx = b_idx * b_bs + c * b_ld + r;
                const int64_t out_idx = b_idx * e_bs + c * e_ld + r;
                float dist = std::sqrt(b_ptr[base_idx]);
                const bool near_zero = dist < epsilon;
                const bool is_diag = r == c;
                if (near_zero) {
                    dist = 0.0f;
                }
                float val = std::exp(-dist / bandwidth);
                if (is_diag) {
                    val = 1.0f + regularizer;
                }
                float grad = 0.0f;
                if (!is_diag && !near_zero) {
                    grad = val / dist;
                }

                bool use_value = true;
                if (packed_type == MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE) {
                    use_value = r >= c;
                } else if (packed_type == MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE) {
                    use_value = c >= r;
                }

                e_ptr[out_idx] = use_value ? val : grad;
            }
        }
    }
}

static void check_close(
    const char* label,
    const HostTensor<float>& got,
    const HostTensor<float>& expected,
    float tol
) {
    const KermacTensor g = got.raw();
    const KermacTensor e = expected.raw();

    ASSERT(g.num_modes == e.num_modes);
    ASSERT(g.extent[0] == e.extent[0]);
    ASSERT(g.extent[1] == e.extent[1]);
    if (g.num_modes > 2 || e.num_modes > 2) {
        ASSERT(g.extent[2] == e.extent[2]);
    }

    const int64_t rows = g.extent[0];
    const int64_t cols = g.extent[1];
    const int64_t batches = batch_count(g);

    const int64_t g_ld = g.stride[1];
    const int64_t e_ld = e.stride[1];
    const int64_t g_bs = batch_stride(g);
    const int64_t e_bs = batch_stride(e);

    const float* g_ptr = got.ptr();
    const float* e_ptr = expected.ptr();

    float max_diff = 0.0f;
    int64_t max_r = 0;
    int64_t max_c = 0;
    int64_t max_b = 0;
    float max_g = 0.0f;
    float max_e = 0.0f;

    for (int64_t b_idx = 0; b_idx < batches; ++b_idx) {
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t r = 0; r < rows; ++r) {
                const int64_t g_idx = b_idx * g_bs + c * g_ld + r;
                const int64_t e_idx = b_idx * e_bs + c * e_ld + r;
                const float diff = std::fabs(g_ptr[g_idx] - e_ptr[e_idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_r = r;
                    max_c = c;
                    max_b = b_idx;
                    max_g = g_ptr[g_idx];
                    max_e = e_ptr[e_idx];
                }
            }
        }
    }

    if (max_diff > tol) {
        std::cerr << "FAIL: " << label
                  << " max_diff=" << max_diff
                  << " at (row=" << max_r
                  << ", col=" << max_c
                  << ", batch=" << max_b
                  << "), got=" << max_g
                  << ", expected=" << max_e << "\n";
        ASSERT(false);
    }
}

static void run_semiring_copy_grad_tests(
    Kermac& kermac,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    CUstream stream
) {
    const int64_t M = 63;
    const int64_t K = 29;
    const int64_t L = 3;

    HostTensor<float> h_a(hsa, M, K, L);
    DeviceTensor<float> d_a(dsa, M, K, L);

    fill_tensor_pattern(h_a, 0.0005f, 0.0007f, 0.0003f, 0.02f);
    d_a.copy_from(h_a, stream);

    DeviceTensor<float> d_c(dsa, M, M, L);
    DeviceTensor<float> d_gold(dsa, M, M, L);

    const float bandwidth = 1.0f;
    const float regularizer = 1e-3f;
    const float epsilon = 0.05f;

    const MatrixPackedType packed_types[] = {
        MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
        MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE
    };

    for (MatrixPackedType packed_type : packed_types) {
        Semiring semiring = Semiring::laplace_l2_symm_copy_grad(
            kermac,
            hsa,
            packed_type,
            bandwidth,
            regularizer,
            epsilon
        );
        semiring.run(d_a, d_a, d_c, stream);
        l2_gold_cpu(hsa, d_a, d_a, d_gold, stream);

        HostTensor<float> h_c(hsa, d_c.extent());
        HostTensor<float> h_base(hsa, d_gold.extent());
        HostTensor<float> h_expected(hsa, d_gold.extent());
        h_c.copy_from(d_c, stream);
        h_base.copy_from(d_gold, stream);
        cuCheck(cuStreamSynchronize(stream));

        compute_expected_laplace_symm_copy_grad(
            h_base,
            h_expected,
            packed_type,
            bandwidth,
            regularizer,
            epsilon
        );

        const char* label =
            packed_type == MatrixPackedType::KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE
                ? "semiring laplace_l2_symm_copy_grad lower"
                : "semiring laplace_l2_symm_copy_grad upper";
        check_close(label, h_c, h_expected, kTolLaplaceGrad);
    }
}

int main() {
    if (!has_cuda_device()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }

    try {
        Kermac kermac(2);
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);

        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            run_semiring_copy_grad_tests(kermac, hsa, dsa, stream);
        }

        cuCheck(cuStreamSynchronize(stream));
        host_free(host_mem);
        device_free(kermac, device_mem);
        cuCheck(cuStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
