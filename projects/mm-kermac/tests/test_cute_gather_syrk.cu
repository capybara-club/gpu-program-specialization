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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cute_gather_syrk.cuh"
#include "dense_batched_gemm.cuh"

using namespace kermac_cute_gather_syrk;
using namespace kermac_dense_batched_gemm;

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

#ifndef CUBLAS_CHECK_AND_EXIT
#define CUBLAS_CHECK_AND_EXIT(expr)                                                       \
    do {                                                                                  \
        const cublasStatus_t _status = (expr);                                            \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                           \
            std::fprintf(stderr, "%s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,   \
                         cublas_status_to_string(_status));                               \
            std::exit(static_cast<int>(_status));                                         \
        }                                                                                 \
    } while (0)
#endif

struct TestCase {
    int num_rows;
    int num_features;
    int batch_count;
    bool share_indices;
    bool duplicate_indices;
    uint64_t seed;
};

struct KernelConfig {
    const char* name;
    int tile_m;
    int tile_n;
    int stages;
    bool cpasync;
};

struct Fp32KernelConfig {
    const char* name;
    int k_tile;
    int stages;
    int threads;
    bool cpasync;
    bool syrk_pruned;
    bool cute_backend;
};

struct DenseKernelConfig {
    const char* name;
    cublasComputeType_t compute_type;
    cublasGemmAlgo_t algo;
    bool use_tf32_reference;
};

struct DenseLtKernelConfig {
    const char* name;
    cublasComputeType_t compute_type;
    size_t workspace_bytes;
    bool use_tf32_reference;
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
            (*x)[static_cast<size_t>(row) + static_cast<size_t>(feature) * static_cast<size_t>(x_ld)] = dist(rng);
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

template <int TileM, int TileN, int Stages>
static cudaError_t launch_config(
    bool cpasync,
    const float* d_x,
    int64_t x_ld,
    const int32_t* d_indices,
    int64_t indices_batch_stride,
    float* d_out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    if (cpasync) {
        return launch_gather_syrk_batched_kernel_cpasync<TileM, TileN, 32, Stages>(
            d_x,
            x_ld,
            d_indices,
            indices_batch_stride,
            d_out,
            out_ld,
            out_batch_stride,
            num_rows,
            batch_count,
            stream
        );
    }
    return launch_gather_syrk_batched_kernel<TileM, TileN, 32, Stages>(
        d_x,
        x_ld,
        d_indices,
        indices_batch_stride,
        d_out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

static cudaError_t launch_kernel_config(
    const KernelConfig& kernel_cfg,
    const float* d_x,
    int64_t x_ld,
    const int32_t* d_indices,
    int64_t indices_batch_stride,
    float* d_out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    if (kernel_cfg.tile_m == 32 && kernel_cfg.tile_n == 32) {
        switch (kernel_cfg.stages) {
            case 1: return launch_config<32, 32, 1>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
            case 2: return launch_config<32, 32, 2>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
            case 3: return launch_config<32, 32, 3>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
            case 4: return launch_config<32, 32, 4>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
            default: return cudaErrorInvalidValue;
        }
    }

    return cudaErrorInvalidValue;
}

template <int KTile, int Stages, int Threads>
static cudaError_t launch_fp32_config(
    bool cpasync,
    bool syrk_pruned,
    const float* d_x,
    int64_t x_ld,
    const int32_t* d_indices,
    int64_t indices_batch_stride,
    float* d_out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    if (syrk_pruned) {
        if (cpasync) {
            return launch_gather_syrk_batched_kernel_fp32_syrk_cpasync<KTile, Stages, Threads>(
                d_x,
                x_ld,
                d_indices,
                indices_batch_stride,
                d_out,
                out_ld,
                out_batch_stride,
                num_rows,
                batch_count,
                stream
            );
        }
        return launch_gather_syrk_batched_kernel_fp32_syrk<KTile, Stages, Threads>(
            d_x,
            x_ld,
            d_indices,
            indices_batch_stride,
            d_out,
            out_ld,
            out_batch_stride,
            num_rows,
            batch_count,
            stream
        );
    }

    if (cpasync) {
        return launch_gather_syrk_batched_kernel_fp32_simt_cpasync<KTile, Stages, Threads>(
            d_x,
            x_ld,
            d_indices,
            indices_batch_stride,
            d_out,
            out_ld,
            out_batch_stride,
            num_rows,
            batch_count,
            stream
        );
    }
    return launch_gather_syrk_batched_kernel_fp32_simt<KTile, Stages, Threads>(
        d_x,
        x_ld,
        d_indices,
        indices_batch_stride,
        d_out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <int KTile, int Stages>
static cudaError_t launch_fp32_cute_config(
    bool cpasync,
    const float* d_x,
    int64_t x_ld,
    const int32_t* d_indices,
    int64_t indices_batch_stride,
    float* d_out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    if (cpasync) {
        return launch_gather_syrk_batched_kernel_fp32_cute_cpasync<KTile, Stages>(
            d_x,
            x_ld,
            d_indices,
            indices_batch_stride,
            d_out,
            out_ld,
            out_batch_stride,
            num_rows,
            batch_count,
            stream
        );
    }
    return launch_gather_syrk_batched_kernel_fp32_cute<KTile, Stages>(
        d_x,
        x_ld,
        d_indices,
        indices_batch_stride,
        d_out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

static cudaError_t launch_fp32_kernel_config(
    const Fp32KernelConfig& kernel_cfg,
    const float* d_x,
    int64_t x_ld,
    const int32_t* d_indices,
    int64_t indices_batch_stride,
    float* d_out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    if (kernel_cfg.cute_backend) {
        if (kernel_cfg.syrk_pruned) {
            return cudaErrorInvalidValue;
        }
        switch (kernel_cfg.k_tile) {
            case 32:
                switch (kernel_cfg.stages) {
                    case 1: return launch_fp32_cute_config<32, 1>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 2: return launch_fp32_cute_config<32, 2>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 3: return launch_fp32_cute_config<32, 3>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 4: return launch_fp32_cute_config<32, 4>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    default: return cudaErrorInvalidValue;
                }
            case 64:
                switch (kernel_cfg.stages) {
                    case 1: return launch_fp32_cute_config<64, 1>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 2: return launch_fp32_cute_config<64, 2>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 3: return launch_fp32_cute_config<64, 3>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 4: return launch_fp32_cute_config<64, 4>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    default: return cudaErrorInvalidValue;
                }
            case 128:
                switch (kernel_cfg.stages) {
                    case 1: return launch_fp32_cute_config<128, 1>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 2: return launch_fp32_cute_config<128, 2>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 3: return launch_fp32_cute_config<128, 3>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    case 4: return launch_fp32_cute_config<128, 4>(kernel_cfg.cpasync, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                    default: return cudaErrorInvalidValue;
                }
            default:
                return cudaErrorInvalidValue;
        }
    }

    switch (kernel_cfg.threads) {
        case 128:
            switch (kernel_cfg.k_tile) {
                case 64:
                    switch (kernel_cfg.stages) {
                        case 1: return launch_fp32_config<64, 1, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 2: return launch_fp32_config<64, 2, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 3: return launch_fp32_config<64, 3, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 4: return launch_fp32_config<64, 4, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        default: return cudaErrorInvalidValue;
                    }
                case 128:
                    switch (kernel_cfg.stages) {
                        case 1: return launch_fp32_config<128, 1, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 2: return launch_fp32_config<128, 2, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 3: return launch_fp32_config<128, 3, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 4: return launch_fp32_config<128, 4, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        default: return cudaErrorInvalidValue;
                    }
                default: return cudaErrorInvalidValue;
            }
        case 256:
            switch (kernel_cfg.k_tile) {
                case 64:
                    switch (kernel_cfg.stages) {
                        case 1: return launch_fp32_config<64, 1, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 2: return launch_fp32_config<64, 2, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 3: return launch_fp32_config<64, 3, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 4: return launch_fp32_config<64, 4, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        default: return cudaErrorInvalidValue;
                    }
                case 128:
                    switch (kernel_cfg.stages) {
                        case 1: return launch_fp32_config<128, 1, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 2: return launch_fp32_config<128, 2, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 3: return launch_fp32_config<128, 3, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 4: return launch_fp32_config<128, 4, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        default: return cudaErrorInvalidValue;
                    }
                default: return cudaErrorInvalidValue;
            }
        default:
            return cudaErrorInvalidValue;
    }
}

static float run_case(const TestCase& tc, const KernelConfig& kernel_cfg) {
    constexpr int kCols = 32;
    const int64_t x_ld = static_cast<int64_t>(tc.num_rows) + 3;
    const int64_t indices_batch_stride = tc.share_indices ? 0 : kCols;
    const int64_t out_ld = kCols;
    const int64_t out_batch_stride = static_cast<int64_t>(kCols) * static_cast<int64_t>(kCols);

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    std::vector<float> h_ref;
    fill_feature_bank(&h_x, x_ld, tc.num_rows, tc.num_features, tc.seed);
    fill_indices(
        &h_indices,
        kCols,
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
    CUDA_CHECK_AND_EXIT(launch_kernel_config(
        kernel_cfg,
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
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    reference_gather_syrk_tf32(
        h_x,
        x_ld,
        h_indices,
        indices_batch_stride,
        &h_ref,
        out_ld,
        out_batch_stride,
        tc.num_rows,
        tc.batch_count,
        kCols
    );

    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return max_abs_diff(h_out, h_ref);
}

static float run_fp32_case(const TestCase& tc, const Fp32KernelConfig& kernel_cfg) {
    constexpr int kCols = 32;
    const int64_t x_ld = static_cast<int64_t>(tc.num_rows) + 3;
    const int64_t indices_batch_stride = tc.share_indices ? 0 : kCols;
    const int64_t out_ld = kCols;
    const int64_t out_batch_stride = static_cast<int64_t>(kCols) * static_cast<int64_t>(kCols);

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    std::vector<float> h_ref;
    fill_feature_bank(&h_x, x_ld, tc.num_rows, tc.num_features, tc.seed);
    fill_indices(
        &h_indices,
        kCols,
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
    CUDA_CHECK_AND_EXIT(launch_fp32_kernel_config(
        kernel_cfg,
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
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    reference_gather_syrk_fp32(
        h_x,
        x_ld,
        h_indices,
        indices_batch_stride,
        &h_ref,
        out_ld,
        out_batch_stride,
        tc.num_rows,
        tc.batch_count,
        kCols
    );

    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return max_abs_diff(h_out, h_ref);
}

static float run_dense_case(const TestCase& tc, const DenseKernelConfig& kernel_cfg) {
    constexpr int kCols = 32;
    const int64_t x_ld = static_cast<int64_t>(tc.num_rows) + 3;
    const int64_t indices_batch_stride = tc.share_indices ? 0 : kCols;
    const int64_t panel_ld = tc.num_rows;
    const int64_t panel_batch_stride = static_cast<int64_t>(tc.num_rows) * static_cast<int64_t>(kCols);
    const int64_t out_ld = kCols;
    const int64_t out_batch_stride = static_cast<int64_t>(kCols) * static_cast<int64_t>(kCols);

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    std::vector<float> h_ref;
    fill_feature_bank(&h_x, x_ld, tc.num_rows, tc.num_features, tc.seed);
    fill_indices(
        &h_indices,
        kCols,
        tc.batch_count,
        tc.num_features,
        tc.share_indices,
        tc.duplicate_indices,
        tc.seed + 1
    );

    float* d_x = nullptr;
    int32_t* d_indices = nullptr;
    float* d_panels = nullptr;
    float* d_out = nullptr;
    const size_t x_bytes = h_x.size() * sizeof(float);
    const size_t indices_bytes = h_indices.size() * sizeof(int32_t);
    const size_t panel_bytes =
        static_cast<size_t>(panel_batch_stride) * static_cast<size_t>(tc.batch_count) * sizeof(float);
    const size_t out_bytes =
        static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count) * sizeof(float);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_indices, indices_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_panels, panel_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_indices, h_indices.data(), indices_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemset(d_out, 0, out_bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK_AND_EXIT(launch_assemble_dense_panels(
        d_x,
        x_ld,
        d_indices,
        indices_batch_stride,
        d_panels,
        panel_ld,
        panel_batch_stride,
        tc.num_rows,
        kCols,
        tc.batch_count,
        stream
    ));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, stream));
    CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm(
        handle,
        kernel_cfg.compute_type,
        kernel_cfg.algo,
        d_panels,
        panel_ld,
        panel_batch_stride,
        d_out,
        out_ld,
        out_batch_stride,
        tc.num_rows,
        kCols,
        tc.batch_count
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    if (kernel_cfg.use_tf32_reference) {
        reference_gather_syrk_tf32(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref,
            out_ld,
            out_batch_stride,
            tc.num_rows,
            tc.batch_count,
            kCols
        );
    } else {
        reference_gather_syrk_fp32(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref,
            out_ld,
            out_batch_stride,
            tc.num_rows,
            tc.batch_count,
            kCols
        );
    }

    CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_panels));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return max_abs_diff(h_out, h_ref);
}

static float run_dense_lt_case(const TestCase& tc, const DenseLtKernelConfig& kernel_cfg) {
    constexpr int kCols = 32;
    const int64_t x_ld = static_cast<int64_t>(tc.num_rows) + 3;
    const int64_t indices_batch_stride = tc.share_indices ? 0 : kCols;
    const int64_t panel_ld = tc.num_rows;
    const int64_t panel_batch_stride = static_cast<int64_t>(tc.num_rows) * static_cast<int64_t>(kCols);
    const int64_t out_ld = kCols;
    const int64_t out_batch_stride = static_cast<int64_t>(kCols) * static_cast<int64_t>(kCols);

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    std::vector<float> h_ref;
    fill_feature_bank(&h_x, x_ld, tc.num_rows, tc.num_features, tc.seed);
    fill_indices(
        &h_indices,
        kCols,
        tc.batch_count,
        tc.num_features,
        tc.share_indices,
        tc.duplicate_indices,
        tc.seed + 1
    );

    float* d_x = nullptr;
    int32_t* d_indices = nullptr;
    float* d_panels = nullptr;
    float* d_out = nullptr;
    void* d_workspace = nullptr;
    const size_t x_bytes = h_x.size() * sizeof(float);
    const size_t indices_bytes = h_indices.size() * sizeof(int32_t);
    const size_t panel_bytes =
        static_cast<size_t>(panel_batch_stride) * static_cast<size_t>(tc.batch_count) * sizeof(float);
    const size_t out_bytes =
        static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count) * sizeof(float);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_indices, indices_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_panels, panel_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_out, out_bytes));
    if (kernel_cfg.workspace_bytes > 0) {
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_workspace, kernel_cfg.workspace_bytes));
    }
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_indices, h_indices.data(), indices_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemset(d_out, 0, out_bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK_AND_EXIT(launch_assemble_dense_panels(
        d_x,
        x_ld,
        d_indices,
        indices_batch_stride,
        d_panels,
        panel_ld,
        panel_batch_stride,
        tc.num_rows,
        kCols,
        tc.batch_count,
        stream
    ));

    DenseBatchedGemmLtPlan plan;
    CUBLAS_CHECK_AND_EXIT(create_dense_batched_gemm_lt_plan(
        &plan,
        kernel_cfg.compute_type,
        panel_ld,
        panel_batch_stride,
        out_ld,
        out_batch_stride,
        tc.num_rows,
        kCols,
        tc.batch_count,
        kernel_cfg.workspace_bytes
    ));
    CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm_lt(
        plan,
        d_panels,
        d_out,
        d_workspace,
        stream
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(tc.batch_count));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    if (kernel_cfg.use_tf32_reference) {
        reference_gather_syrk_tf32(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref,
            out_ld,
            out_batch_stride,
            tc.num_rows,
            tc.batch_count,
            kCols
        );
    } else {
        reference_gather_syrk_fp32(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref,
            out_ld,
            out_batch_stride,
            tc.num_rows,
            tc.batch_count,
            kCols
        );
    }

    CUBLAS_CHECK_AND_EXIT(destroy_dense_batched_gemm_lt_plan(&plan));
    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_panels));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    if (d_workspace) {
        CUDA_CHECK_AND_EXIT(cudaFree(d_workspace));
    }
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return max_abs_diff(h_out, h_ref);
}

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }
    if (get_cuda_device_arch() < 800u) {
        std::printf("SKIP: CuTe TF32 kernel requires SM80+\n");
        return 77;
    }

    try {
        const std::vector<TestCase> cases = {
            {31, 257, 3, false, false, 1234},
            {32, 257, 3, false, false, 2234},
            {33, 257, 3, false, false, 3234},
            {65, 331, 2, false, true, 3456},
            {97, 513, 4, true, false, 2345},
            {129, 257, 3, false, false, 4234},
        };
        const std::vector<KernelConfig> kernels = {
            {"scalar_cta32x32_p1", 32, 32, 1, false},
            {"scalar_cta32x32_p2", 32, 32, 2, false},
            {"scalar_cta32x32_p3", 32, 32, 3, false},
            {"scalar_cta32x32_p4", 32, 32, 4, false},
            {"cpasync_cta32x32_p1", 32, 32, 1, true},
            {"cpasync_cta32x32_p2", 32, 32, 2, true},
            {"cpasync_cta32x32_p3", 32, 32, 3, true},
            {"cpasync_cta32x32_p4", 32, 32, 4, true},
        };
        const std::vector<Fp32KernelConfig> fp32_kernels = {
            {"fp32_cute_scalar_k32_p1", 32, 1, 0, false, false, true},
            {"fp32_cute_scalar_k32_p2", 32, 2, 0, false, false, true},
            {"fp32_cute_cpasync_k32_p2", 32, 2, 0, true, false, true},
            {"fp32_cute_cpasync_k32_p3", 32, 3, 0, true, false, true},
            {"fp32_cute_cpasync_k64_p2", 64, 2, 0, true, false, true},
            {"fp32_full_scalar_k64_t128_p1", 64, 1, 128, false, false, false},
            {"fp32_full_scalar_k64_t128_p2", 64, 2, 128, false, false, false},
            {"fp32_full_cpasync_k64_t128_p2", 64, 2, 128, true, false, false},
            {"fp32_full_cpasync_k128_t128_p2", 128, 2, 128, true, false, false},
            {"fp32_full_cpasync_k64_t256_p2", 64, 2, 256, true, false, false},
            {"fp32_syrk_scalar_k64_t128_p1", 64, 1, 128, false, true, false},
            {"fp32_syrk_scalar_k64_t128_p2", 64, 2, 128, false, true, false},
            {"fp32_syrk_cpasync_k64_t128_p1", 64, 1, 128, true, true, false},
            {"fp32_syrk_cpasync_k64_t128_p2", 64, 2, 128, true, true, false},
            {"fp32_syrk_cpasync_k64_t128_p3", 64, 3, 128, true, true, false},
            {"fp32_syrk_cpasync_k128_t128_p2", 128, 2, 128, true, true, false},
            {"fp32_syrk_cpasync_k64_t256_p2", 64, 2, 256, true, true, false},
        };
        const std::vector<DenseKernelConfig> dense_kernels = {
            {"cublas_dense_batched_gemm_tf32", CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP, true},
            {"cublas_dense_batched_gemm_fp32_gold", CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT, false},
        };
        const std::vector<DenseLtKernelConfig> dense_lt_kernels = {
            {"cublaslt_dense_batched_gemm_tf32_ws32m", CUBLAS_COMPUTE_32F_FAST_TF32, 32u * 1024u * 1024u, true},
            {"cublaslt_dense_batched_gemm_fp32_gold_ws32m", CUBLAS_COMPUTE_32F, 32u * 1024u * 1024u, false},
        };

        for (const TestCase& tc : cases) {
            for (const KernelConfig& kernel_cfg : kernels) {
                const float diff = run_case(tc, kernel_cfg);
                if (!(diff <= 1e-2f)) {
                    std::fprintf(
                        stderr,
                        "FAIL kernel=%s rows=%d features=%d batch=%d diff=%.8f\n",
                        kernel_cfg.name,
                        tc.num_rows,
                        tc.num_features,
                        tc.batch_count,
                        diff
                    );
                    return 1;
                }
            }
            for (const Fp32KernelConfig& kernel_cfg : fp32_kernels) {
                const float diff = run_fp32_case(tc, kernel_cfg);
                const float tol = std::max(1e-4f, 2.0e-6f * static_cast<float>(tc.num_rows));
                if (!(diff <= tol)) {
                    std::fprintf(
                        stderr,
                        "FAIL fp32_kernel=%s rows=%d features=%d batch=%d diff=%.8f\n",
                        kernel_cfg.name,
                        tc.num_rows,
                        tc.num_features,
                        tc.batch_count,
                        diff
                    );
                    return 1;
                }
            }
            for (const DenseKernelConfig& dense_cfg : dense_kernels) {
                const float diff = run_dense_case(tc, dense_cfg);
                const float tol = dense_cfg.use_tf32_reference
                    ? 1e-2f
                    : std::max(1e-4f, 2.0e-6f * static_cast<float>(tc.num_rows));
                if (!(diff <= tol)) {
                    std::fprintf(
                        stderr,
                        "FAIL dense=%s rows=%d features=%d batch=%d diff=%.8f\n",
                        dense_cfg.name,
                        tc.num_rows,
                        tc.num_features,
                        tc.batch_count,
                        diff
                    );
                    return 1;
                }
            }
            for (const DenseLtKernelConfig& dense_cfg : dense_lt_kernels) {
                const float diff = run_dense_lt_case(tc, dense_cfg);
                const float tol = dense_cfg.use_tf32_reference
                    ? 1e-2f
                    : std::max(1e-4f, 2.0e-6f * static_cast<float>(tc.num_rows));
                if (!(diff <= tol)) {
                    std::fprintf(
                        stderr,
                        "FAIL dense_lt=%s rows=%d features=%d batch=%d diff=%.8f\n",
                        dense_cfg.name,
                        tc.num_rows,
                        tc.num_features,
                        tc.batch_count,
                        diff
                    );
                    return 1;
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Test failed with exception: %s\n", e.what());
        return 1;
    }
}
