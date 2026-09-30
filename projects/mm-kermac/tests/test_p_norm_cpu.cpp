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
    const int64_t batches = raw.extent[2];
    const int64_t ld = raw.stride[1];
    const int64_t bs = raw.stride[2];
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

            const int64_t M = 57;
            const int64_t N = 41;
            const int64_t K = 23;
            const int64_t L = 3;

            HostTensor<float> h_a(hsa, M, K, L);
            HostTensor<float> h_b(hsa, N, K, L);
            fill_tensor_pattern(h_a, 0.001f, 0.002f, 0.0007f, 0.05f);
            fill_tensor_pattern(h_b, 0.0015f, -0.001f, 0.0004f, 0.02f);

            DeviceTensor<float> d_a(dsa, M, K, L);
            DeviceTensor<float> d_b(dsa, N, K, L);
            d_a.copy_from(h_a, stream);
            d_b.copy_from(h_b, stream);

            DeviceTensor<float> d_pnorm(dsa, M, N, L);
            DeviceTensor<float> d_gold(dsa, M, N, L);

            p_norm_kernel_cpu(hsa, 2.0f, d_a, d_b, d_pnorm, stream);
            l2_gold_cpu(hsa, d_a, d_b, d_gold, stream);

            HostTensor<float> h_pnorm(hsa, d_pnorm.extent());
            HostTensor<float> h_gold(hsa, d_gold.extent());
            h_pnorm.copy_from(d_pnorm, stream);
            h_gold.copy_from(d_gold, stream);
            cuCheck(cuStreamSynchronize(stream));

            const KermacTensor p_raw = h_pnorm.raw();
            const KermacTensor g_raw = h_gold.raw();
            const float* p_ptr = h_pnorm.ptr();
            const float* g_ptr = h_gold.ptr();

            float max_diff = 0.0f;
            int64_t max_r = 0;
            int64_t max_c = 0;
            int64_t max_b = 0;

            for (int64_t b = 0; b < L; ++b) {
                for (int64_t c = 0; c < N; ++c) {
                    for (int64_t r = 0; r < M; ++r) {
                        const int64_t p_idx = b * p_raw.stride[2] + c * p_raw.stride[1] + r;
                        const int64_t g_idx = b * g_raw.stride[2] + c * g_raw.stride[1] + r;
                        const float expected = std::sqrt(g_ptr[g_idx]);
                        const float diff = std::fabs(p_ptr[p_idx] - expected);
                        if (diff > max_diff) {
                            max_diff = diff;
                            max_r = r;
                            max_c = c;
                            max_b = b;
                        }
                    }
                }
            }

            if (max_diff > kTol) {
                std::cerr << "FAIL: max_diff=" << max_diff
                          << " at (row=" << max_r
                          << ", col=" << max_c
                          << ", batch=" << max_b << ")\n";
                return -1;
            }
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
