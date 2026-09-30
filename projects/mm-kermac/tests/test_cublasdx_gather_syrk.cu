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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "cublasdx_gather_syrk.cuh"

using namespace kermac_cublasdx_gather_syrk;

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(expr)                                                          \
    do {                                                                                   \
        const cudaError_t _status = (expr);                                                \
        if (_status != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,    \
                         cudaGetErrorString(_status));                                     \
            std::exit(static_cast<int>(_status));                                          \
        }                                                                                  \
    } while (0)
#endif

struct TestCase {
    int cols;
    int k_tile;
    int num_rows;
    int num_features;
    int batch_count;
    bool share_indices;
    bool duplicate_indices;
    uint64_t seed;
};

static void fill_feature_bank(
    std::vector<float>* x,
    int64_t x_ld,
    int num_rows,
    int num_features,
    uint64_t seed
) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    x->assign(static_cast<size_t>(x_ld) * static_cast<size_t>(num_features), 0.0f);
    for (int feature = 0; feature < num_features; ++feature) {
        for (int row = 0; row < num_rows; ++row) {
            (*x)[static_cast<size_t>(row) + static_cast<size_t>(feature) * static_cast<size_t>(x_ld)] =
                dist(rng);
        }
    }
}

static void fill_indices(
    std::vector<int32_t>* indices,
    int cols,
    int batch_count,
    int num_features,
    bool share_indices,
    bool duplicate_indices,
    uint64_t seed
) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int32_t> dist(0, num_features - 1);
    const int actual_batches = share_indices ? 1 : batch_count;
    indices->assign(static_cast<size_t>(cols * actual_batches), 0);

    for (int batch = 0; batch < actual_batches; ++batch) {
        for (int col = 0; col < cols; ++col) {
            (*indices)[static_cast<size_t>(batch * cols + col)] = dist(rng);
        }
        if (!duplicate_indices) {
            auto begin = indices->begin() + static_cast<std::ptrdiff_t>(batch * cols);
            auto end = begin + cols;
            std::sort(begin, end);
            end = std::unique(begin, end);
            while (end != begin + cols) {
                *end = dist(rng);
                ++end;
                std::sort(begin, begin + cols);
                end = std::unique(begin, begin + cols);
            }
        } else {
            for (int col = 1; col < cols; col += 7) {
                (*indices)[static_cast<size_t>(batch * cols + col)] =
                    (*indices)[static_cast<size_t>(batch * cols)];
            }
        }
    }
}

template <unsigned int Arch, int Cols, int KTile>
static float run_case_impl(const TestCase& tc) {
    const int64_t x_ld = static_cast<int64_t>(tc.num_rows) + 3;
    const int64_t indices_batch_stride = tc.share_indices ? 0 : Cols;
    const int64_t out_ld = Cols;
    const int64_t out_batch_stride = static_cast<int64_t>(Cols) * static_cast<int64_t>(Cols);

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    std::vector<float> h_ref;
    fill_feature_bank(&h_x, x_ld, tc.num_rows, tc.num_features, tc.seed);
    fill_indices(
        &h_indices,
        Cols,
        tc.batch_count,
        tc.num_features,
        tc.share_indices,
        tc.duplicate_indices,
        tc.seed + 1
    );

    float* d_x = nullptr;
    int32_t* d_indices = nullptr;
    float* d_out = nullptr;
    const size_t x_bytes = h_x.size() * sizeof(float);
    const size_t indices_bytes = h_indices.size() * sizeof(int32_t);
    const size_t out_bytes =
        static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count) * sizeof(float);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_indices, indices_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_indices, h_indices.data(), indices_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemset(d_out, 0, out_bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const cudaError_t status = launch_gather_syrk_batched_kernel<Arch, Cols, KTile>(
        d_x,
        x_ld,
        d_indices,
        indices_batch_stride,
        d_out,
        out_ld,
        out_batch_stride,
        tc.num_rows,
        tc.batch_count,
        stream
    );
    CUDA_CHECK_AND_EXIT(status);
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    reference_gather_syrk(
        h_x,
        x_ld,
        h_indices,
        indices_batch_stride,
        &h_ref,
        out_ld,
        out_batch_stride,
        tc.num_rows,
        tc.batch_count,
        Cols
    );

    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return max_abs_diff(h_out, h_ref);
}

template <unsigned int Arch, int Cols>
static float dispatch_k_tile(const TestCase& tc) {
    switch (tc.k_tile) {
        case 16: return run_case_impl<Arch, Cols, 16>(tc);
        case 32: return run_case_impl<Arch, Cols, 32>(tc);
        case 64: return run_case_impl<Arch, Cols, 64>(tc);
        default:
            throw std::runtime_error("unsupported k_tile");
    }
}

template <unsigned int Arch>
static float dispatch_cols(const TestCase& tc) {
    switch (tc.cols) {
        case 32: return dispatch_k_tile<Arch, 32>(tc);
        case 104: return dispatch_k_tile<Arch, 104>(tc);
        default:
            throw std::runtime_error("unsupported cols");
    }
}

static float dispatch_arch(const TestCase& tc) {
    switch (get_cuda_device_arch()) {
#ifdef KERMAC_CUBLASDX_ENABLE_SM_80
        case 800: return dispatch_cols<800>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_86
        case 860: return dispatch_cols<860>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_87
        case 870: return dispatch_cols<870>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_89
        case 890: return dispatch_cols<890>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_90
        case 900: return dispatch_cols<900>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_100
        case 1000: return dispatch_cols<1000>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_101
        case 1010: return dispatch_cols<1010>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_103
        case 1030: return dispatch_cols<1030>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_110
        case 1100: return dispatch_cols<1100>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_120
        case 1200: return dispatch_cols<1200>(tc);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_121
        case 1210: return dispatch_cols<1210>(tc);
#endif
        default:
            throw std::runtime_error("current SM is not enabled for the cublasDx test target");
    }
}

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }

    try {
        const std::vector<TestCase> cases = {
            {32, 16, 13, 257, 3, false, false, 1234},
            {32, 32, 97, 513, 4, true, false, 2345},
            {32, 64, 65, 331, 2, false, true, 3456},
            {104, 32, 33, 521, 2, false, false, 4567},
        };

        for (const TestCase& tc : cases) {
            const float diff = dispatch_arch(tc);
            if (!(diff <= 5e-3f)) {
                std::fprintf(
                    stderr,
                    "FAIL cols=%d k_tile=%d rows=%d batch=%d diff=%.8f\n",
                    tc.cols,
                    tc.k_tile,
                    tc.num_rows,
                    tc.batch_count,
                    diff
                );
                return 1;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Test failed with exception: %s\n", e.what());
        return 1;
    }
}
