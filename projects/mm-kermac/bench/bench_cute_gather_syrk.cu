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
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
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

struct BenchConfig {
    int64_t num_rows = 8192;
    int64_t num_features = 8192;
    int64_t batch_count = 4096;
    int64_t cols = 32;
    int64_t ld_pad = 0;
    int warmup = 3;
    int iters = 10;
    uint64_t seed = 1234;
    bool check = false;
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

static void print_usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --num-rows <rows>       Number of rows in x (default 8192)\n"
        "  --num-features <cols>   Number of feature columns in x (default 8192)\n"
        "  --batch-count <count>   Number of gathered batches (default 4096)\n"
        "  --cols <cols>           Gathered columns per batch (must be 32; default 32)\n"
        "  --ld-pad <rows>         Extra leading-dimension padding on x (default 0)\n"
        "  --warmup <iters>        Warmup iterations (default 3)\n"
        "  --iters <iters>         Timed iterations (default 10)\n"
        "  --seed <val>            RNG seed (default 1234)\n"
        "  --check <0|1>           Run correctness checks and print error versus cuBLASLt FP32 gold (default 0)\n"
        "  --help                  Show this message\n",
        exe
    );
}

static bool parse_int64(const char* arg, int64_t* out) {
    char* end = nullptr;
    long long v = std::strtoll(arg, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<int64_t>(v);
    return true;
}

static bool parse_int(const char* arg, int* out) {
    char* end = nullptr;
    long v = std::strtol(arg, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

static bool parse_u64(const char* arg, uint64_t* out) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(arg, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<uint64_t>(v);
    return true;
}

static bool parse_bool01(const char* arg, bool* out) {
    if (std::strcmp(arg, "0") == 0) {
        *out = false;
        return true;
    }
    if (std::strcmp(arg, "1") == 0) {
        *out = true;
        return true;
    }
    return false;
}

static bool parse_args(int argc, char** argv, BenchConfig* cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for argument: %s\n", argv[i]);
            return false;
        }
        const char* key = argv[i];
        const char* val = argv[i + 1];
        bool ok = true;
        if (std::strcmp(key, "--num-rows") == 0) {
            ok = parse_int64(val, &cfg->num_rows);
        } else if (std::strcmp(key, "--num-features") == 0) {
            ok = parse_int64(val, &cfg->num_features);
        } else if (std::strcmp(key, "--batch-count") == 0) {
            ok = parse_int64(val, &cfg->batch_count);
        } else if (std::strcmp(key, "--cols") == 0) {
            ok = parse_int64(val, &cfg->cols);
        } else if (std::strcmp(key, "--ld-pad") == 0) {
            ok = parse_int64(val, &cfg->ld_pad);
        } else if (std::strcmp(key, "--warmup") == 0) {
            ok = parse_int(val, &cfg->warmup);
        } else if (std::strcmp(key, "--iters") == 0) {
            ok = parse_int(val, &cfg->iters);
        } else if (std::strcmp(key, "--seed") == 0) {
            ok = parse_u64(val, &cfg->seed);
        } else if (std::strcmp(key, "--check") == 0) {
            ok = parse_bool01(val, &cfg->check);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", key);
            return false;
        }
        if (!ok) {
            std::fprintf(stderr, "Invalid value for %s: %s\n", key, val);
            return false;
        }
        ++i;
    }
    return true;
}

template <typename SetupFn, typename TimedFn>
static float benchmark_ms(cudaStream_t stream, int warmup, int iters, SetupFn&& setup, TimedFn&& timed) {
    if (warmup < 0) warmup = 0;
    if (iters < 1) iters = 1;

    for (int i = 0; i < warmup; ++i) {
        setup();
        timed();
    }
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<cudaEvent_t> starts(static_cast<size_t>(iters));
    std::vector<cudaEvent_t> stops(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&starts[static_cast<size_t>(i)]));
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&stops[static_cast<size_t>(i)]));
    }

    for (int i = 0; i < iters; ++i) {
        setup();
        CUDA_CHECK_AND_EXIT(cudaEventRecord(starts[static_cast<size_t>(i)], stream));
        timed();
        CUDA_CHECK_AND_EXIT(cudaEventRecord(stops[static_cast<size_t>(i)], stream));
    }

    CUDA_CHECK_AND_EXIT(cudaEventSynchronize(stops.back()));

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        float ms = 0.0f;
        CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&ms, starts[static_cast<size_t>(i)], stops[static_cast<size_t>(i)]));
        total_ms += ms;
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(starts[static_cast<size_t>(i)]));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(stops[static_cast<size_t>(i)]));
    }

    return total_ms / static_cast<float>(iters);
}

static void fill_feature_bank(
    std::vector<float>* x,
    int64_t x_ld,
    int64_t num_rows,
    int64_t num_features,
    uint64_t seed
) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    x->assign(static_cast<size_t>(x_ld) * static_cast<size_t>(num_features), 0.0f);
    for (int64_t feature = 0; feature < num_features; ++feature) {
        for (int64_t row = 0; row < num_rows; ++row) {
            (*x)[static_cast<size_t>(row) + static_cast<size_t>(feature) * static_cast<size_t>(x_ld)] = dist(rng);
        }
    }
}

static void fill_unique_indices(
    std::vector<int32_t>* indices,
    int64_t cols,
    int64_t batch_count,
    int64_t num_features,
    uint64_t seed
) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(num_features - 1));
    indices->assign(static_cast<size_t>(cols * batch_count), 0);

    for (int64_t batch = 0; batch < batch_count; ++batch) {
        std::unordered_set<int32_t> used;
        while (static_cast<int64_t>(used.size()) < cols) {
            used.insert(dist(rng));
        }
        size_t offset = static_cast<size_t>(batch * cols);
        size_t idx = 0;
        for (int32_t feature : used) {
            (*indices)[offset + idx] = feature;
            ++idx;
        }
        std::sort(indices->begin() + static_cast<std::ptrdiff_t>(offset),
                  indices->begin() + static_cast<std::ptrdiff_t>(offset + cols));
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
                case 32:
                    switch (kernel_cfg.stages) {
                        case 1: return launch_fp32_config<32, 1, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 2: return launch_fp32_config<32, 2, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 3: return launch_fp32_config<32, 3, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 4: return launch_fp32_config<32, 4, 128>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        default: return cudaErrorInvalidValue;
                    }
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
                case 32:
                    switch (kernel_cfg.stages) {
                        case 1: return launch_fp32_config<32, 1, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 2: return launch_fp32_config<32, 2, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 3: return launch_fp32_config<32, 3, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        case 4: return launch_fp32_config<32, 4, 256>(kernel_cfg.cpasync, kernel_cfg.syrk_pruned, d_x, x_ld, d_indices, indices_batch_stride, d_out, out_ld, out_batch_stride, num_rows, batch_count, stream);
                        default: return cudaErrorInvalidValue;
                    }
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

static double relative_l2_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    const size_t n = std::min(lhs.size(), rhs.size());
    long double diff_sq = 0.0;
    long double ref_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const long double d = static_cast<long double>(lhs[i]) - static_cast<long double>(rhs[i]);
        const long double r = static_cast<long double>(rhs[i]);
        diff_sq += d * d;
        ref_sq += r * r;
    }
    if (ref_sq == 0.0) {
        return diff_sq == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return std::sqrt(static_cast<double>(diff_sq / ref_sq));
}

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 1;
    }
    if (!has_cuda_device()) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    if (cfg.cols != 32) {
        std::fprintf(stderr, "This CuTe benchmark currently supports --cols 32 only\n");
        return 1;
    }

    const int64_t x_ld = cfg.num_rows + cfg.ld_pad;
    const int64_t indices_batch_stride = cfg.cols;
    const int64_t panel_ld = cfg.num_rows;
    const int64_t panel_batch_stride = cfg.num_rows * cfg.cols;
    const int64_t out_ld = cfg.cols;
    const int64_t out_batch_stride = cfg.cols * cfg.cols;

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    fill_feature_bank(&h_x, x_ld, cfg.num_rows, cfg.num_features, cfg.seed);
    fill_unique_indices(&h_indices, cfg.cols, cfg.batch_count, cfg.num_features, cfg.seed + 1);

    float* d_x = nullptr;
    int32_t* d_indices = nullptr;
    float* d_panels = nullptr;
    float* d_out = nullptr;
    const size_t x_bytes = h_x.size() * sizeof(float);
    const size_t indices_bytes = h_indices.size() * sizeof(int32_t);
    const size_t panel_bytes =
        static_cast<size_t>(panel_batch_stride) * static_cast<size_t>(cfg.batch_count) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count) * sizeof(float);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_indices, indices_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_panels, panel_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_indices, h_indices.data(), indices_bytes, cudaMemcpyHostToDevice));

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
        static_cast<int32_t>(cfg.num_rows),
        static_cast<int32_t>(cfg.cols),
        static_cast<int32_t>(cfg.batch_count),
        stream
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    cublasHandle_t cublas_handle = nullptr;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&cublas_handle));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(cublas_handle, stream));

    const std::vector<KernelConfig> kernels = {
        {"cute_tf32_scalar_cta32x32_p1", 32, 32, 1, false},
        {"cute_tf32_scalar_cta32x32_p2", 32, 32, 2, false},
        {"cute_tf32_scalar_cta32x32_p3", 32, 32, 3, false},
        {"cute_tf32_scalar_cta32x32_p4", 32, 32, 4, false},
        {"cute_tf32_cpasync_cta32x32_p1", 32, 32, 1, true},
        {"cute_tf32_cpasync_cta32x32_p2", 32, 32, 2, true},
        {"cute_tf32_cpasync_cta32x32_p3", 32, 32, 3, true},
        {"cute_tf32_cpasync_cta32x32_p4", 32, 32, 4, true},
    };
    const std::vector<Fp32KernelConfig> fp32_kernels = {
        {"cute_fp32_cute_scalar_k32_p1", 32, 1, 0, false, false, true},
        {"cute_fp32_cute_scalar_k32_p2", 32, 2, 0, false, false, true},
        {"cute_fp32_cute_scalar_k32_p3", 32, 3, 0, false, false, true},
        {"cute_fp32_cute_scalar_k32_p4", 32, 4, 0, false, false, true},
        {"cute_fp32_cute_cpasync_k32_p1", 32, 1, 0, true, false, true},
        {"cute_fp32_cute_cpasync_k32_p2", 32, 2, 0, true, false, true},
        {"cute_fp32_cute_cpasync_k32_p3", 32, 3, 0, true, false, true},
        {"cute_fp32_cute_cpasync_k32_p4", 32, 4, 0, true, false, true},
        {"cute_fp32_cute_scalar_k64_p2", 64, 2, 0, false, false, true},
        {"cute_fp32_cute_cpasync_k64_p2", 64, 2, 0, true, false, true},
        {"cute_fp32_cute_cpasync_k128_p2", 128, 2, 0, true, false, true},
        {"cute_fp32_full_scalar_k64_t128_p1", 64, 1, 128, false, false, false},
        {"cute_fp32_full_scalar_k64_t128_p2", 64, 2, 128, false, false, false},
        {"cute_fp32_full_cpasync_k64_t128_p2", 64, 2, 128, true, false, false},
        {"cute_fp32_full_cpasync_k128_t128_p2", 128, 2, 128, true, false, false},
        {"cute_fp32_full_cpasync_k64_t256_p2", 64, 2, 256, true, false, false},
        {"cute_fp32_syrk_scalar_k64_t128_p1", 64, 1, 128, false, true, false},
        {"cute_fp32_syrk_scalar_k64_t128_p2", 64, 2, 128, false, true, false},
        {"cute_fp32_syrk_cpasync_k64_t128_p1", 64, 1, 128, true, true, false},
        {"cute_fp32_syrk_cpasync_k64_t128_p2", 64, 2, 128, true, true, false},
        {"cute_fp32_syrk_cpasync_k64_t128_p3", 64, 3, 128, true, true, false},
        {"cute_fp32_syrk_cpasync_k128_t128_p2", 128, 2, 128, true, true, false},
        {"cute_fp32_syrk_cpasync_k64_t256_p2", 64, 2, 256, true, true, false},
    };
    const std::vector<DenseKernelConfig> dense_kernels = {
        {"cublas_dense_batched_gemm_tf32", CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP, true},
        {"cublas_dense_batched_gemm_fp32_gold", CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT, false},
    };
    const std::vector<DenseLtKernelConfig> dense_lt_kernels = {
        {"cublaslt_dense_batched_gemm_tf32_ws0", CUBLAS_COMPUTE_32F_FAST_TF32, 0, true},
        {"cublaslt_dense_batched_gemm_tf32_ws32m", CUBLAS_COMPUTE_32F_FAST_TF32, 32u * 1024u * 1024u, true},
        {"cublaslt_dense_batched_gemm_tf32_ws128m", CUBLAS_COMPUTE_32F_FAST_TF32, 128u * 1024u * 1024u, true},
        {"cublaslt_dense_batched_gemm_fp32_gold_ws32m", CUBLAS_COMPUTE_32F, 32u * 1024u * 1024u, false},
    };

    size_t max_lt_workspace_bytes = 0;
    for (const DenseLtKernelConfig& cfg_lt : dense_lt_kernels) {
        max_lt_workspace_bytes = std::max(max_lt_workspace_bytes, cfg_lt.workspace_bytes);
    }
    void* d_lt_workspace = nullptr;
    if (max_lt_workspace_bytes > 0) {
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_lt_workspace, max_lt_workspace_bytes));
    }
    std::vector<DenseBatchedGemmLtPlan> dense_lt_plans(dense_lt_kernels.size());
    for (size_t i = 0; i < dense_lt_kernels.size(); ++i) {
        CUBLAS_CHECK_AND_EXIT(create_dense_batched_gemm_lt_plan(
            &dense_lt_plans[i],
            dense_lt_kernels[i].compute_type,
            panel_ld,
            panel_batch_stride,
            out_ld,
            out_batch_stride,
            static_cast<int32_t>(cfg.num_rows),
            static_cast<int32_t>(cfg.cols),
            static_cast<int32_t>(cfg.batch_count),
            dense_lt_kernels[i].workspace_bytes
        ));
    }

    std::vector<float> h_ref_tf32;
    std::vector<float> h_ref_fp32;
    std::vector<float> h_ref_fp32_gold;
    if (cfg.check) {
        reference_gather_syrk_tf32(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref_tf32,
            out_ld,
            out_batch_stride,
            static_cast<int32_t>(cfg.num_rows),
            static_cast<int32_t>(cfg.batch_count),
            static_cast<int>(cfg.cols)
        );
        reference_gather_syrk_fp32(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref_fp32,
            out_ld,
            out_batch_stride,
            static_cast<int32_t>(cfg.num_rows),
            static_cast<int32_t>(cfg.batch_count),
            static_cast<int>(cfg.cols)
        );

        ssize_t fp32_gold_lt_index = -1;
        for (size_t i = 0; i < dense_lt_kernels.size(); ++i) {
            if (dense_lt_kernels[i].compute_type == CUBLAS_COMPUTE_32F) {
                fp32_gold_lt_index = static_cast<ssize_t>(i);
                break;
            }
        }
        if (fp32_gold_lt_index >= 0) {
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm_lt(
                dense_lt_plans[static_cast<size_t>(fp32_gold_lt_index)],
                d_panels,
                d_out,
                d_lt_workspace,
                stream
            ));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
            h_ref_fp32_gold.resize(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count));
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_ref_fp32_gold.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
        } else {
            h_ref_fp32_gold = h_ref_fp32;
        }
    }

    if (cfg.check) {
        std::printf("| kernel | ms | GF/s | batches/s | max_abs_vs_fp32_gold | rel_l2_vs_fp32_gold |\n");
        std::printf("| --- | ---: | ---: | ---: | ---: | ---: |\n");
    } else {
        std::printf("| kernel | ms | GF/s | batches/s |\n");
        std::printf("| --- | ---: | ---: | ---: |\n");
    }

    for (const KernelConfig& kernel_cfg : kernels) {
        float max_abs_vs_fp32 = 0.0f;
        double rel_l2_vs_fp32 = 0.0;
        if (cfg.check) {
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            CUDA_CHECK_AND_EXIT(launch_kernel_config(
                kernel_cfg,
                d_x,
                x_ld,
                d_indices,
                indices_batch_stride,
                d_out,
                out_ld,
                out_batch_stride,
                static_cast<int32_t>(cfg.num_rows),
                static_cast<int32_t>(cfg.batch_count),
                stream
            ));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count));
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
            const float diff = max_abs_diff(h_out, h_ref_tf32);
            const float tol = std::max(1e-2f, 2.5e-5f * static_cast<float>(cfg.num_rows));
            if (!(diff <= tol)) {
                std::fprintf(stderr, "Correctness check failed for %s: max_abs_diff=%g\n", kernel_cfg.name, diff);
                return 1;
            }
            max_abs_vs_fp32 = max_abs_diff(h_out, h_ref_fp32_gold);
            rel_l2_vs_fp32 = relative_l2_diff(h_out, h_ref_fp32_gold);
        }

        const float ms = benchmark_ms(
            stream,
            cfg.warmup,
            cfg.iters,
            [&]() {
                CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            },
            [&]() {
                CUDA_CHECK_AND_EXIT(launch_kernel_config(
                    kernel_cfg,
                    d_x,
                    x_ld,
                    d_indices,
                    indices_batch_stride,
                    d_out,
                    out_ld,
                    out_batch_stride,
                    static_cast<int32_t>(cfg.num_rows),
                    static_cast<int32_t>(cfg.batch_count),
                    stream
                ));
            }
        );

        const double flops = 2.0 * static_cast<double>(cfg.batch_count) * static_cast<double>(cfg.cols) *
                             static_cast<double>(cfg.cols) * static_cast<double>(cfg.num_rows);
        const double seconds = static_cast<double>(ms) * 1.0e-3;
        const double gflops = flops / seconds / 1.0e9;
        const double batches_per_sec = static_cast<double>(cfg.batch_count) / seconds;

        if (cfg.check) {
            std::printf(
                "| %s | %.2f ms | %'.0f GF/s | %'.0f /s | %.6g | %.6g |\n",
                kernel_cfg.name,
                ms,
                gflops,
                batches_per_sec,
                max_abs_vs_fp32,
                rel_l2_vs_fp32
            );
        } else {
            std::printf("| %s | %.2f ms | %'.0f GF/s | %'.0f /s |\n", kernel_cfg.name, ms, gflops, batches_per_sec);
        }
    }

    for (const Fp32KernelConfig& kernel_cfg : fp32_kernels) {
        float max_abs_vs_fp32 = 0.0f;
        double rel_l2_vs_fp32 = 0.0;
        if (cfg.check) {
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            CUDA_CHECK_AND_EXIT(launch_fp32_kernel_config(
                kernel_cfg,
                d_x,
                x_ld,
                d_indices,
                indices_batch_stride,
                d_out,
                out_ld,
                out_batch_stride,
                static_cast<int32_t>(cfg.num_rows),
                static_cast<int32_t>(cfg.batch_count),
                stream
            ));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count));
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
            const float diff = max_abs_diff(h_out, h_ref_fp32);
            const float tol = std::max(1e-4f, 2.0e-6f * static_cast<float>(cfg.num_rows));
            if (!(diff <= tol)) {
                std::fprintf(stderr, "Correctness check failed for %s: max_abs_diff=%g\n", kernel_cfg.name, diff);
                return 1;
            }
            max_abs_vs_fp32 = max_abs_diff(h_out, h_ref_fp32_gold);
            rel_l2_vs_fp32 = relative_l2_diff(h_out, h_ref_fp32_gold);
        }

        const float ms = benchmark_ms(
            stream,
            cfg.warmup,
            cfg.iters,
            [&]() {
                CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            },
            [&]() {
                CUDA_CHECK_AND_EXIT(launch_fp32_kernel_config(
                    kernel_cfg,
                    d_x,
                    x_ld,
                    d_indices,
                    indices_batch_stride,
                    d_out,
                    out_ld,
                    out_batch_stride,
                    static_cast<int32_t>(cfg.num_rows),
                    static_cast<int32_t>(cfg.batch_count),
                    stream
                ));
            }
        );

        const double flops = kernel_cfg.syrk_pruned
            ? static_cast<double>(cfg.batch_count) * static_cast<double>(cfg.cols) *
                static_cast<double>(cfg.cols + 1) * static_cast<double>(cfg.num_rows)
            : 2.0 * static_cast<double>(cfg.batch_count) * static_cast<double>(cfg.cols) *
                static_cast<double>(cfg.cols) * static_cast<double>(cfg.num_rows);
        const double seconds = static_cast<double>(ms) * 1.0e-3;
        const double gflops = flops / seconds / 1.0e9;
        const double batches_per_sec = static_cast<double>(cfg.batch_count) / seconds;

        if (cfg.check) {
            std::printf(
                "| %s | %.2f ms | %'.0f GF/s | %'.0f /s | %.6g | %.6g |\n",
                kernel_cfg.name,
                ms,
                gflops,
                batches_per_sec,
                max_abs_vs_fp32,
                rel_l2_vs_fp32
            );
        } else {
            std::printf("| %s | %.2f ms | %'.0f GF/s | %'.0f /s |\n", kernel_cfg.name, ms, gflops, batches_per_sec);
        }
    }

    for (const DenseKernelConfig& dense_cfg : dense_kernels) {
        float max_abs_vs_fp32 = 0.0f;
        double rel_l2_vs_fp32 = 0.0;
        if (cfg.check) {
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm(
                cublas_handle,
                dense_cfg.compute_type,
                dense_cfg.algo,
                d_panels,
                panel_ld,
                panel_batch_stride,
                d_out,
                out_ld,
                out_batch_stride,
                static_cast<int32_t>(cfg.num_rows),
                static_cast<int32_t>(cfg.cols),
                static_cast<int32_t>(cfg.batch_count)
            ));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count));
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
            const std::vector<float>& h_ref = dense_cfg.use_tf32_reference ? h_ref_tf32 : h_ref_fp32;
            const float diff = max_abs_diff(h_out, h_ref);
            const float tol = dense_cfg.use_tf32_reference
                ? std::max(1e-2f, 2.5e-5f * static_cast<float>(cfg.num_rows))
                : std::max(1e-4f, 2.0e-6f * static_cast<float>(cfg.num_rows));
            if (!(diff <= tol)) {
                std::fprintf(stderr, "Correctness check failed for %s: max_abs_diff=%g\n", dense_cfg.name, diff);
                return 1;
            }
            max_abs_vs_fp32 = max_abs_diff(h_out, h_ref_fp32_gold);
            rel_l2_vs_fp32 = relative_l2_diff(h_out, h_ref_fp32_gold);
        }

        const float ms = benchmark_ms(
            stream,
            cfg.warmup,
            cfg.iters,
            [&]() {
                CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            },
            [&]() {
                CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm(
                    cublas_handle,
                    dense_cfg.compute_type,
                    dense_cfg.algo,
                    d_panels,
                    panel_ld,
                    panel_batch_stride,
                    d_out,
                    out_ld,
                    out_batch_stride,
                    static_cast<int32_t>(cfg.num_rows),
                    static_cast<int32_t>(cfg.cols),
                    static_cast<int32_t>(cfg.batch_count)
                ));
            }
        );

        const double flops = 2.0 * static_cast<double>(cfg.batch_count) * static_cast<double>(cfg.cols) *
                             static_cast<double>(cfg.cols) * static_cast<double>(cfg.num_rows);
        const double seconds = static_cast<double>(ms) * 1.0e-3;
        const double gflops = flops / seconds / 1.0e9;
        const double batches_per_sec = static_cast<double>(cfg.batch_count) / seconds;

        if (cfg.check) {
            std::printf(
                "| %s | %.2f ms | %'.0f GF/s | %'.0f /s | %.6g | %.6g |\n",
                dense_cfg.name,
                ms,
                gflops,
                batches_per_sec,
                max_abs_vs_fp32,
                rel_l2_vs_fp32
            );
        } else {
            std::printf("| %s | %.2f ms | %'.0f GF/s | %'.0f /s |\n", dense_cfg.name, ms, gflops, batches_per_sec);
        }
    }

    for (size_t i = 0; i < dense_lt_kernels.size(); ++i) {
        const DenseLtKernelConfig& dense_cfg = dense_lt_kernels[i];
        const DenseBatchedGemmLtPlan& plan = dense_lt_plans[i];
        float max_abs_vs_fp32 = 0.0f;
        double rel_l2_vs_fp32 = 0.0;
        if (cfg.check) {
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm_lt(
                plan,
                d_panels,
                d_out,
                d_lt_workspace,
                stream
            ));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count));
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
            const std::vector<float>& h_ref = dense_cfg.use_tf32_reference ? h_ref_tf32 : h_ref_fp32;
            const float diff = max_abs_diff(h_out, h_ref);
            const float tol = dense_cfg.use_tf32_reference
                ? std::max(1e-2f, 2.5e-5f * static_cast<float>(cfg.num_rows))
                : std::max(1e-4f, 2.0e-6f * static_cast<float>(cfg.num_rows));
            if (!(diff <= tol)) {
                std::fprintf(stderr, "Correctness check failed for %s: max_abs_diff=%g\n", dense_cfg.name, diff);
                return 1;
            }
            max_abs_vs_fp32 = max_abs_diff(h_out, h_ref_fp32_gold);
            rel_l2_vs_fp32 = relative_l2_diff(h_out, h_ref_fp32_gold);
        }

        const float ms = benchmark_ms(
            stream,
            cfg.warmup,
            cfg.iters,
            [&]() {
                CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
            },
            [&]() {
                CUBLAS_CHECK_AND_EXIT(launch_dense_batched_gemm_lt(
                    plan,
                    d_panels,
                    d_out,
                    d_lt_workspace,
                    stream
                ));
            }
        );

        const double flops = 2.0 * static_cast<double>(cfg.batch_count) * static_cast<double>(cfg.cols) *
                             static_cast<double>(cfg.cols) * static_cast<double>(cfg.num_rows);
        const double seconds = static_cast<double>(ms) * 1.0e-3;
        const double gflops = flops / seconds / 1.0e9;
        const double batches_per_sec = static_cast<double>(cfg.batch_count) / seconds;

        if (cfg.check) {
            std::printf(
                "| %s | %.2f ms | %'.0f GF/s | %'.0f /s | %.6g | %.6g |\n",
                dense_cfg.name,
                ms,
                gflops,
                batches_per_sec,
                max_abs_vs_fp32,
                rel_l2_vs_fp32
            );
        } else {
            std::printf("| %s | %.2f ms | %'.0f GF/s | %'.0f /s |\n", dense_cfg.name, ms, gflops, batches_per_sec);
        }
    }

    CUBLAS_CHECK_AND_EXIT(cublasDestroy(cublas_handle));
    for (DenseBatchedGemmLtPlan& plan : dense_lt_plans) {
        CUBLAS_CHECK_AND_EXIT(destroy_dense_batched_gemm_lt_plan(&plan));
    }
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_panels));
    if (d_lt_workspace) {
        CUDA_CHECK_AND_EXIT(cudaFree(d_lt_workspace));
    }
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    return 0;
}
