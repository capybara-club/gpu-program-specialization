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

static const int64_t N = 1000;
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

static int run_case(
    const char* label,
    Kermac& kermac,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    CUstream stream,
    bool alpha_has_label_mode,
    bool y_broadcast,
    int64_t label_count
) {
    auto run_with_tensors = [&](DeviceTensor<float>& d_alpha, DeviceTensor<float>& d_y) -> int {
        DeviceTensor<float> d_feature_matrix(dsa, N, N, L);
        DeviceTensor<float> d_logdet_norm(dsa, L);
        DeviceTensor<float> d_norm_H2(dsa, L);
        DeviceTensor<float> d_logdet_norm_cpu(dsa, L);
        DeviceTensor<float> d_norm_H2_cpu(dsa, L);

        rng(kermac, d_feature_matrix, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
        rng(kermac, d_alpha, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
        rng(kermac, d_y, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);

        logdet_norm_norm_h2(
            kermac,
            lambda_reg,
            d_feature_matrix, d_alpha, d_y,
            d_logdet_norm, d_norm_H2,
            stream
        );

        logdet_norm_cpu(
            hsa,
            d_feature_matrix,
            d_logdet_norm_cpu,
            stream
        );

        norm_H2_cpu(
            hsa, lambda_reg,
            d_alpha, d_y, d_norm_H2_cpu,
            stream
        );

        float logdet_norm_max_diff;
        float norm_H2_max_diff;
        max_diff_cpu(
            hsa,
            d_logdet_norm,
            d_logdet_norm_cpu,
            &logdet_norm_max_diff,
            stream
        );

        max_diff_cpu(
            hsa,
            d_norm_H2,
            d_norm_H2_cpu,
            &norm_H2_max_diff,
            stream
        );

        if (logdet_norm_max_diff > 1e-4f || norm_H2_max_diff > 1e-4f) {
            std::cerr << "Mismatch in " << label
                      << " (logdet_norm=" << logdet_norm_max_diff
                      << ", norm_H2=" << norm_H2_max_diff << ")\n";
            return -1;
        }
        return 0;
    };

    if (label_count > 1) {
        DeviceTensor<float> d_alpha(dsa, N, label_count, L);
        if (y_broadcast) {
            DeviceTensor<float> d_y(dsa, N, 1);
            return run_with_tensors(d_alpha, d_y);
        }
        DeviceTensor<float> d_y(dsa, N, label_count);
        return run_with_tensors(d_alpha, d_y);
    }

    if (alpha_has_label_mode) {
        DeviceTensor<float> d_alpha(dsa, N, 1, L);
        if (y_broadcast) {
            DeviceTensor<float> d_y(dsa, N, 1);
            return run_with_tensors(d_alpha, d_y);
        }
        DeviceTensor<float> d_y(dsa, N, L);
        return run_with_tensors(d_alpha, d_y);
    }

    DeviceTensor<float> d_alpha(dsa, N, L);
    if (y_broadcast) {
        DeviceTensor<float> d_y(dsa, N, 1);
        return run_with_tensors(d_alpha, d_y);
    }
    DeviceTensor<float> d_y(dsa, N, L);
    return run_with_tensors(d_alpha, d_y);
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
        cuCheck( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );

        void* host_mem = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);
        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            if (run_case("alpha[N,L] y[N,L]", kermac, hsa, dsa, stream, false, false, 1) != 0) return -1;
            if (run_case("alpha[N,L] y[N,1]", kermac, hsa, dsa, stream, false, true, 1) != 0) return -1;
            if (run_case("alpha[N,1,L] y[N,L]", kermac, hsa, dsa, stream, true, false, 1) != 0) return -1;
            if (run_case("alpha[N,1,L] y[N,1]", kermac, hsa, dsa, stream, true, true, 1) != 0) return -1;
            if (run_case("alpha[N,K,L] y[N,K]", kermac, hsa, dsa, stream, false, false, 3) != 0) return -1;
            if (run_case("alpha[N,K,L] y[N,1]", kermac, hsa, dsa, stream, false, true, 3) != 0) return -1;
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
