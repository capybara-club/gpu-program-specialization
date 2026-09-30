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
#include <unordered_set>
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

struct BenchConfig {
    int64_t num_rows = 4096;
    int64_t num_features = 1 << 20;
    int64_t batch_count = 1024;
    int64_t cols = 32;
    int64_t k_tile = 32;
    int64_t ld_pad = 0;
    int warmup = 5;
    int iters = 20;
    uint64_t seed = 1234;
    bool check = false;
};

struct SizeDisplay {
    double value;
    const char* unit;
};

static void print_usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --num-rows <rows>       Number of rows in x (default 4096)\n"
        "  --num-features <cols>   Number of feature columns in x (default 1048576)\n"
        "  --batch-count <count>   Number of gathered cohorts / batches (default 1024)\n"
        "  --cols <cols>           Gathered columns per batch (supported: 32,104; default 32)\n"
        "  --k-tile <rows>         Rows staged per GEMM chunk (supported: 8,16,32,64,128; default 32)\n"
        "  --ld-pad <rows>         Extra leading-dimension padding on x (default 0)\n"
        "  --warmup <iters>        Warmup iterations (default 5)\n"
        "  --iters <iters>         Timed iterations (default 20)\n"
        "  --seed <val>            RNG seed (default 1234)\n"
        "  --check <0|1>           Compare one run against CPU reference (default 0)\n"
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
        } else if (std::strcmp(key, "--k-tile") == 0) {
            ok = parse_int64(val, &cfg->k_tile);
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

static SizeDisplay format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    double value = static_cast<double>(bytes);
    size_t idx = 0;
    while (value >= 1024.0 && idx + 1 < (sizeof(units) / sizeof(units[0]))) {
        value /= 1024.0;
        ++idx;
    }
    return {value, units[idx]};
}

static bool checked_bytes(size_t* out, int64_t a, int64_t b, size_t elem_size) {
    if (a <= 0 || b <= 0) {
        return false;
    }
    __uint128_t total =
        static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b) * static_cast<__uint128_t>(elem_size);
    if (total > static_cast<__uint128_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *out = static_cast<size_t>(total);
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
        CUDA_CHECK_AND_EXIT(
            cudaEventElapsedTime(
                &ms,
                starts[static_cast<size_t>(i)],
                stops[static_cast<size_t>(i)]
            )
        );
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
            (*x)[static_cast<size_t>(row) + static_cast<size_t>(feature) * static_cast<size_t>(x_ld)] =
                dist(rng);
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
        for (int32_t value : used) {
            (*indices)[offset + idx] = value;
            ++idx;
        }
        std::sort(indices->begin() + static_cast<std::ptrdiff_t>(offset),
                  indices->begin() + static_cast<std::ptrdiff_t>(offset + cols));
    }
}

static void print_result(
    float ms,
    int64_t num_rows,
    int64_t cols,
    int64_t batch_count,
    size_t feature_bank_bytes,
    size_t index_bytes
) {
    struct TimeDisplay {
        double value;
        const char* unit;
    };
    auto format_time_ms = [](double ms_value) -> TimeDisplay {
        if (ms_value >= 1.0) return {ms_value, "ms"};
        if (ms_value >= 1e-3) return {ms_value * 1e3, "us"};
        return {ms_value * 1e6, "ns"};
    };

    const double seconds = static_cast<double>(ms) * 1e-3;
    const double tile_reads = static_cast<double>(num_rows) * static_cast<double>(cols);
    const double flops_per_batch = 2.0 * tile_reads * static_cast<double>(cols);
    const double effective_gflops =
        flops_per_batch * static_cast<double>(batch_count) / seconds / 1e9;
    const double batches_per_second = static_cast<double>(batch_count) / seconds;

    const SizeDisplay x_size = format_bytes(feature_bank_bytes);
    const SizeDisplay i_size = format_bytes(index_bytes);
    const TimeDisplay t = format_time_ms(ms);

    std::printf("feature_bank: %.3f %s\n", x_size.value, x_size.unit);
    std::printf("indices:      %.3f %s\n", i_size.value, i_size.unit);
    std::printf("time:         %.3f %s\n", t.value, t.unit);
    std::printf("batches/s:    %.3f\n", batches_per_second);
    std::printf("effective:    %.3f GFLOP/s\n", effective_gflops);
}

template <unsigned int Arch, int Cols, int KTile>
static int run_benchmark_impl(const BenchConfig& cfg) {
    const int64_t x_ld = cfg.num_rows + cfg.ld_pad;
    const int64_t indices_batch_stride = Cols;
    const int64_t out_ld = Cols;
    const int64_t out_batch_stride = static_cast<int64_t>(Cols) * static_cast<int64_t>(Cols);

    std::vector<float> h_x;
    std::vector<int32_t> h_indices;
    fill_feature_bank(&h_x, x_ld, cfg.num_rows, cfg.num_features, cfg.seed);
    fill_unique_indices(&h_indices, Cols, cfg.batch_count, cfg.num_features, cfg.seed + 1);

    const size_t x_bytes = h_x.size() * sizeof(float);
    const size_t indices_bytes = h_indices.size() * sizeof(int32_t);
    const size_t out_bytes = static_cast<size_t>(out_batch_stride) *
        static_cast<size_t>(cfg.batch_count) * sizeof(float);

    float* d_x = nullptr;
    int32_t* d_indices = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_indices, indices_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_indices, h_indices.data(), indices_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto setup = [&]() {
        CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_out, 0, out_bytes, stream));
    };
    auto timed = [&]() {
        const cudaError_t status = launch_gather_syrk_batched_kernel<Arch, Cols, KTile>(
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
        );
        CUDA_CHECK_AND_EXIT(status);
    };

    const float ms = benchmark_ms(stream, cfg.warmup, cfg.iters, setup, timed);
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    print_result(ms, cfg.num_rows, Cols, cfg.batch_count, x_bytes, indices_bytes);

    if (cfg.check) {
        std::vector<float> h_out(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(cfg.batch_count));
        std::vector<float> h_ref;
        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
        reference_gather_syrk(
            h_x,
            x_ld,
            h_indices,
            indices_batch_stride,
            &h_ref,
            out_ld,
            out_batch_stride,
            static_cast<int32_t>(cfg.num_rows),
            static_cast<int32_t>(cfg.batch_count),
            Cols
        );
        const float diff = max_abs_diff(h_out, h_ref);
        std::printf("max_abs_diff: %.8f\n", diff);
        if (!(diff <= 5e-3f)) {
            std::fprintf(stderr, "Reference check failed: max_abs_diff=%.8f\n", diff);
            CUDA_CHECK_AND_EXIT(cudaFree(d_x));
            CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
            CUDA_CHECK_AND_EXIT(cudaFree(d_out));
            CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
            return 1;
        }
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_indices));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return 0;
}

template <unsigned int Arch, int Cols>
static int dispatch_k_tile(const BenchConfig& cfg) {
    switch (cfg.k_tile) {
        case 8: return run_benchmark_impl<Arch, Cols, 8>(cfg);
        case 16: return run_benchmark_impl<Arch, Cols, 16>(cfg);
        case 32: return run_benchmark_impl<Arch, Cols, 32>(cfg);
        case 64: return run_benchmark_impl<Arch, Cols, 64>(cfg);
        case 128: return run_benchmark_impl<Arch, Cols, 128>(cfg);
        default:
            std::fprintf(stderr, "Unsupported --k-tile %lld\n", static_cast<long long>(cfg.k_tile));
            return 1;
    }
}

template <unsigned int Arch>
static int dispatch_cols(const BenchConfig& cfg) {
    switch (cfg.cols) {
        case 32: return dispatch_k_tile<Arch, 32>(cfg);
        case 104: return dispatch_k_tile<Arch, 104>(cfg);
        default:
            std::fprintf(stderr, "Unsupported --cols %lld\n", static_cast<long long>(cfg.cols));
            return 1;
    }
}

static int dispatch_arch(const BenchConfig& cfg) {
    switch (get_cuda_device_arch()) {
#ifdef KERMAC_CUBLASDX_ENABLE_SM_80
        case 800: return dispatch_cols<800>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_86
        case 860: return dispatch_cols<860>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_87
        case 870: return dispatch_cols<870>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_89
        case 890: return dispatch_cols<890>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_90
        case 900: return dispatch_cols<900>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_100
        case 1000: return dispatch_cols<1000>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_101
        case 1010: return dispatch_cols<1010>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_103
        case 1030: return dispatch_cols<1030>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_110
        case 1100: return dispatch_cols<1100>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_120
        case 1200: return dispatch_cols<1200>(cfg);
#endif
#ifdef KERMAC_CUBLASDX_ENABLE_SM_121
        case 1210: return dispatch_cols<1210>(cfg);
#endif
        default:
            std::fprintf(
                stderr,
                "Benchmark not configured for SM %u. Reconfigure with "
                "-DKERMAC_CUSOLVERDX_CUDA_ARCHITECTURES=<sm-list>.\n",
                get_cuda_device_arch()
            );
            return 1;
    }
}

int main(int argc, char** argv) {
    if (!has_cuda_device()) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }

    BenchConfig cfg;
    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 1;
    }
    if (cfg.num_rows <= 0 || cfg.num_features <= 0 || cfg.batch_count <= 0) {
        std::fprintf(stderr, "num-rows, num-features, and batch-count must be > 0\n");
        return 1;
    }
    if (cfg.ld_pad < 0) {
        std::fprintf(stderr, "ld-pad must be >= 0\n");
        return 1;
    }
    if (cfg.num_features < cfg.cols) {
        std::fprintf(stderr, "num-features must be >= cols\n");
        return 1;
    }
    if (cfg.iters < 1 || cfg.warmup < 0) {
        std::fprintf(stderr, "iters must be >= 1 and warmup must be >= 0\n");
        return 1;
    }

    size_t dummy = 0;
    if (!checked_bytes(&dummy, cfg.num_rows + cfg.ld_pad, cfg.num_features, sizeof(float)) ||
        !checked_bytes(&dummy, cfg.cols, cfg.batch_count, sizeof(int32_t)) ||
        !checked_bytes(&dummy, cfg.cols * cfg.cols, cfg.batch_count, sizeof(float))) {
        std::fprintf(stderr, "Requested allocation is too large\n");
        return 1;
    }

    try {
        return dispatch_arch(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Benchmark failed with exception: %s\n", e.what());
        return 1;
    }
}
