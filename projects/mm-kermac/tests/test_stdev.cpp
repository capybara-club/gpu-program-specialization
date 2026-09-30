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

// Pick an even row count so the +/- pattern has exact mean 0.
static const int64_t NUM_ROWS    = 1020;
static const int64_t NUM_COLS    = 37;
static const int64_t NUM_BATCHES = 11;

// Must match kernel's epsilon (STDEV_EPS)
static constexpr double STDEV_EPS_D = 1e-12;

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
            HostTensor<float> h_expected_mean(hsa, NUM_BATCHES, NUM_COLS);
            HostTensor<float> h_expected_stdev(hsa, NUM_BATCHES, NUM_COLS);

            // Device tensors
            DeviceTensor<float> d_a(dsa, NUM_ROWS, NUM_COLS, NUM_BATCHES);
            DeviceTensor<float> d_mean(dsa, NUM_BATCHES, NUM_COLS);
            DeviceTensor<float> d_stdev(dsa, NUM_BATCHES, NUM_COLS);
            DeviceTensor<float> d_expected_mean(dsa, NUM_BATCHES, NUM_COLS);
            DeviceTensor<float> d_expected_stdev(dsa, NUM_BATCHES, NUM_COLS);

            // Populate h_a with an alternating +/-s pattern per (batch, col),
            // and h_expected with analytically-known stdev = sqrt(s^2 + eps).
            float* a_ptr = h_a.ptr();
            float* mean_ptr = h_expected_mean.ptr();
            float* stdev_ptr = h_expected_stdev.ptr();

            const int64_t ld_a  = h_a.raw().stride[1];   // == NUM_ROWS for default layout
            const int64_t bs_a  = h_a.raw().stride[2];   // == NUM_ROWS*NUM_COLS
            const int64_t ld_mean  = h_expected_mean.raw().stride[1];
            const int64_t ld_stdev = h_expected_stdev.raw().stride[1];

            for (int64_t b = 0; b < NUM_BATCHES; ++b) {
                for (int64_t c = 0; c < NUM_COLS; ++c) {
                    const double s = 0.1 + 0.01 * (double)b + 0.001 * (double)c;
                    const float  sf = (float)s;

                    float* col_ptr = a_ptr + (size_t)b * (size_t)bs_a + (size_t)c * (size_t)ld_a;
                    for (int64_t r = 0; r < NUM_ROWS; ++r) {
                        col_ptr[r] = (r & 1) ? sf : -sf;
                    }

                    mean_ptr[(size_t)c * (size_t)ld_mean + (size_t)b] = 0.0f;
                    stdev_ptr[(size_t)c * (size_t)ld_stdev + (size_t)b] =
                        (float)std::sqrt(s * s + STDEV_EPS_D);
                }
            }

            // Copy host -> device
            d_a.copy_from(h_a, stream);
            d_expected_mean.copy_from(h_expected_mean, stream);
            d_expected_stdev.copy_from(h_expected_stdev, stream);

            // Run kernel via C++ wrapper
            row_stats(kermac, d_a, d_mean, d_stdev, stream);

            // Compare on CPU using the existing helper
            float max_diff = 0.0f;
            max_diff_cpu(hsa, d_mean, d_expected_mean, &max_diff, stream);
            if (max_diff > 1e-4f) {
                std::cerr << "FAIL mean: max_diff = " << max_diff << "\n";
                return -1;
            }

            max_diff = 0.0f;
            max_diff_cpu(hsa, d_stdev, d_expected_stdev, &max_diff, stream);
            if (max_diff > 1e-4f) {
                std::cerr << "FAIL stdev: max_diff = " << max_diff << "\n";
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
