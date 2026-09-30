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
#include <cmath>

#include <kermac.hpp>
#include <kermac_cpu.hpp>
#include <check_result_helper.h>

using namespace kermac;

static const size_t NUM_BYTES = 1ull << 28;

static const int64_t NUM_ROWS    = 1000;
static const int64_t NUM_COLS    = 37;
static const int64_t NUM_BATCHES = 11;

static bool has_cuda_device() {
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return false;
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
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem   = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);

        {
            HostStackAllocator   hsa(host_mem,   NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            // Host tensors
            HostTensor<float> h_a(hsa, NUM_ROWS, NUM_COLS, NUM_BATCHES);
            HostTensor<float> h_mean(hsa, NUM_BATCHES, NUM_COLS);
            HostTensor<float> h_stdev(hsa, NUM_BATCHES, NUM_COLS);
            HostTensor<float> h_expected(hsa, NUM_ROWS, NUM_COLS, NUM_BATCHES);

            // Device tensors
            DeviceTensor<float> d_a(dsa, NUM_ROWS, NUM_COLS, NUM_BATCHES);
            DeviceTensor<float> d_mean(dsa, NUM_BATCHES, NUM_COLS);
            DeviceTensor<float> d_stdev(dsa, NUM_BATCHES, NUM_COLS);
            DeviceTensor<float> d_expected(dsa, NUM_ROWS, NUM_COLS, NUM_BATCHES);

            float* a_ptr   = h_a.ptr();
            float* m_ptr   = h_mean.ptr();
            float* s_ptr   = h_stdev.ptr();
            float* ex_ptr  = h_expected.ptr();

            const int64_t ld_a  = h_a.raw().stride[1];
            const int64_t bs_a  = h_a.raw().stride[2];

            const int64_t ld_m  = h_mean.raw().stride[1];
            const int64_t ld_s  = h_stdev.raw().stride[1];

            const int64_t ld_ex = h_expected.raw().stride[1];
            const int64_t bs_ex = h_expected.raw().stride[2];

            // Fill stdev and a, and build exact expected = a / stdev
            for (int64_t b = 0; b < NUM_BATCHES; ++b) {
                for (int64_t c = 0; c < NUM_COLS; ++c) {
                    // Distinct, positive scale per (batch,col)
                    const float mean = (float)(0.20 + 0.02 * (double)b - 0.001 * (double)c);
                    const float s = (float)(0.5 + 0.01 * (double)b + 0.001 * (double)c);
                    m_ptr[(size_t)c * (size_t)ld_m + (size_t)b] = mean;
                    s_ptr[(size_t)c * (size_t)ld_s + (size_t)b] = s;

                    float* col_a  = a_ptr  + (size_t)b * (size_t)bs_a  + (size_t)c * (size_t)ld_a;
                    float* col_ex = ex_ptr + (size_t)b * (size_t)bs_ex + (size_t)c * (size_t)ld_ex;

                    for (int64_t r = 0; r < NUM_ROWS; ++r) {
                        // Deterministic, nontrivial values that vary with r/c/b (catches indexing bugs)
                        const float v = (float)(
                            0.25 * (double)(r + 1) +
                            0.10 * (double)b +
                            0.01 * (double)c
                        );
                        col_a[r]  = v;
                        col_ex[r] = (v - mean) / s;
                    }
                }
            }

            // Copy host -> device
            d_a.copy_from(h_a, stream);
            d_mean.copy_from(h_mean, stream);
            d_stdev.copy_from(h_stdev, stream);
            d_expected.copy_from(h_expected, stream);

            apply_row_stats(
                kermac,
                d_a,
                d_mean,
                d_stdev,
                KERMAC_ROW_NORMALIZE_FLAG_CENTER | KERMAC_ROW_NORMALIZE_FLAG_SCALE,
                stream
            );

            // Compare
            float max_diff = 0.0f;
            max_diff_cpu(hsa, d_a, d_expected, &max_diff, stream);

            if (max_diff > 1e-4f) {
                std::cerr << "FAIL: max_diff = " << max_diff << "\n";
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
