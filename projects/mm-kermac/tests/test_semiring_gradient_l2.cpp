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

static const size_t NUM_BYTES = 1ull << 26;
static constexpr float kTol = 1e-3f;

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

static int64_t batch_count_2d3d(const KermacTensor& t) {
    return t.num_modes < 3 ? 1 : t.extent[2];
}

static int64_t batch_stride_2d3d(const KermacTensor& t) {
    return t.num_modes < 3 ? 0 : t.stride[2];
}

static int64_t batch_count_4d(const KermacTensor& t) {
    return t.num_modes < 4 ? 1 : t.extent[3];
}

static int64_t batch_stride_4d(const KermacTensor& t) {
    return t.num_modes < 4 ? 0 : t.stride[3];
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
    const int64_t batches = batch_count_2d3d(raw);
    const int64_t ld = raw.stride[1];
    const int64_t bs = batch_stride_2d3d(raw);
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
    ASSERT(g.extent[2] == e.extent[2]);
    if (g.num_modes == 4) {
        ASSERT(g.extent[3] == e.extent[3]);
    }

    const int64_t M = g.extent[0];
    const int64_t D = g.extent[1];
    const int64_t C = g.extent[2];
    const int64_t L = batch_count_4d(g);

    const int64_t g_ld_d = g.stride[1];
    const int64_t g_ld_c = g.stride[2];
    const int64_t g_bs = batch_stride_4d(g);

    const int64_t e_ld_d = e.stride[1];
    const int64_t e_ld_c = e.stride[2];
    const int64_t e_bs = batch_stride_4d(e);

    const float* g_ptr = got.ptr();
    const float* e_ptr = expected.ptr();

    float max_diff = 0.0f;
    int64_t max_m = 0;
    int64_t max_d = 0;
    int64_t max_c = 0;
    int64_t max_l = 0;
    float max_g = 0.0f;
    float max_e = 0.0f;

    for (int64_t l = 0; l < L; ++l) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t d = 0; d < D; ++d) {
                for (int64_t m = 0; m < M; ++m) {
                    const int64_t g_idx = l * g_bs + c * g_ld_c + d * g_ld_d + m;
                    const int64_t e_idx = l * e_bs + c * e_ld_c + d * e_ld_d + m;
                    const float diff = std::fabs(g_ptr[g_idx] - e_ptr[e_idx]);
                    if (diff > max_diff) {
                        max_diff = diff;
                        max_m = m;
                        max_d = d;
                        max_c = c;
                        max_l = l;
                        max_g = g_ptr[g_idx];
                        max_e = e_ptr[e_idx];
                    }
                }
            }
        }
    }

    if (max_diff > tol) {
        std::cerr << "FAIL: " << label
                  << " max_diff=" << max_diff
                  << " at (m=" << max_m
                  << ", d=" << max_d
                  << ", c=" << max_c
                  << ", l=" << max_l
                  << "), got=" << max_g
                  << ", expected=" << max_e << "\n";
        ASSERT(false);
    }
}

static void run_semiring_grad_l2_test(
    Kermac& kermac,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    CUstream stream
) {
    const int64_t M = 33;
    const int64_t N = 29;
    const int64_t D = 17;
    const int64_t C = 5;
    const int64_t L = 3;

    HostTensor<float> h_kernel(hsa, M, N, L);
    HostTensor<float> h_data_n(hsa, N, D, L);
    HostTensor<float> h_solution(hsa, N, C, L);
    HostTensor<float> h_data_m(hsa, M, D, L);

    fill_tensor_pattern(h_kernel, 0.001f, -0.002f, 0.0003f, 0.1f);
    fill_tensor_pattern(h_data_n, 0.0007f, 0.0012f, 0.0002f, -0.05f);
    fill_tensor_pattern(h_solution, -0.0009f, 0.0004f, 0.0001f, 0.03f);
    fill_tensor_pattern(h_data_m, 0.0011f, -0.0006f, 0.0005f, 0.02f);

    DeviceTensor<float> d_kernel(dsa, h_kernel.extent());
    DeviceTensor<float> d_data_n(dsa, h_data_n.extent());
    DeviceTensor<float> d_solution(dsa, h_solution.extent());
    DeviceTensor<float> d_data_m(dsa, h_data_m.extent());

    d_kernel.copy_from(h_kernel, stream);
    d_data_n.copy_from(h_data_n, stream);
    d_solution.copy_from(h_solution, stream);
    d_data_m.copy_from(h_data_m, stream);

    DeviceTensor<float> d_grad(dsa, M, D, C, L);
    DeviceTensor<float> d_expected(dsa, M, D, C, L);

    {
        SemiringGradient grad = SemiringGradient::norm_l2(kermac, hsa);
        grad.run(d_kernel, d_data_n, d_solution, d_data_m, d_grad, 1.0f, 0.0f, stream);
    }

    cutensor_gradient_norm_l2(
        kermac, dsa, TensorCoreMode::KERMAC_TENSOR_CORE_MODE_F32,
        1.0f,
        d_kernel,
        d_data_n,
        d_solution,
        d_data_m,
        d_expected,
        stream
    );

    HostTensor<float> h_grad(hsa, d_grad.extent());
    HostTensor<float> h_expected(hsa, d_expected.extent());
    h_grad.copy_from(d_grad, stream);
    h_expected.copy_from(d_expected, stream);
    cuCheck(cuStreamSynchronize(stream));

    check_close("semiring_grad_l2", h_grad, h_expected, kTol);
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

            run_semiring_grad_l2_test(kermac, hsa, dsa, stream);
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
