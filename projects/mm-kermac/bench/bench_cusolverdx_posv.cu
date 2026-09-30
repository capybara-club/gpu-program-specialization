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
#include <vector>

#include <cuda_runtime.h>

#include <cusolverdx.hpp>

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
    int64_t N = 32;
    int64_t C = 1;
    int64_t L = 1024;
    int warmup = 5;
    int iters = 20;
    float regularizer = 1e-3f;
    float offdiag_scale = 5e-2f;
    uint64_t seed = 1234;
};

struct SizeDisplay {
    double value;
    const char* unit;
};

static bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

static void print_usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --N <rows>             Matrix size (A is NxN) (supported: 16,32,33,64,65,128; default 32)\n"
        "  --C <rhs>              RHS columns / labels (supported: 1,4,8,16,32; default 1)\n"
        "  --L <batch>            Batch size (default 1024)\n"
        "  --warmup <iters>       Warmup iterations (default 5)\n"
        "  --iters <iters>        Timed iterations (default 20)\n"
        "  --regularizer <val>    Extra diagonal regularizer (default 1e-3)\n"
        "  --offdiag-scale <val>  Off-diagonal random scale (default 5e-2)\n"
        "  --seed <val>           RNG seed for matrix/RHS generation (default 1234)\n"
        "  --help                 Show this message\n",
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

static bool parse_float(const char* arg, float* out) {
    char* end = nullptr;
    float v = std::strtof(arg, &end);
    if (!end || *end != '\0') {
        return false;
    }
    *out = v;
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

static bool parse_args(int argc, char** argv, BenchConfig* cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
        const char* key = argv[i];
        const char* val = argv[i + 1];
        bool ok = true;
        if (std::strcmp(key, "--N") == 0) {
            ok = parse_int64(val, &cfg->N);
        } else if (std::strcmp(key, "--C") == 0) {
            ok = parse_int64(val, &cfg->C);
        } else if (std::strcmp(key, "--L") == 0) {
            ok = parse_int64(val, &cfg->L);
        } else if (std::strcmp(key, "--warmup") == 0) {
            ok = parse_int(val, &cfg->warmup);
        } else if (std::strcmp(key, "--iters") == 0) {
            ok = parse_int(val, &cfg->iters);
        } else if (std::strcmp(key, "--regularizer") == 0) {
            ok = parse_float(val, &cfg->regularizer);
        } else if (std::strcmp(key, "--offdiag-scale") == 0) {
            ok = parse_float(val, &cfg->offdiag_scale);
        } else if (std::strcmp(key, "--seed") == 0) {
            ok = parse_u64(val, &cfg->seed);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", key);
            print_usage(argv[0]);
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

static bool checked_tensor_bytes(
    size_t* out,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    size_t elem_size
) {
    if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0) {
        return false;
    }
    __uint128_t total = static_cast<__uint128_t>(dim0)
        * static_cast<__uint128_t>(dim1)
        * static_cast<__uint128_t>(dim2)
        * static_cast<__uint128_t>(elem_size);
    if (total > static_cast<__uint128_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *out = static_cast<size_t>(total);
    return true;
}

template <typename SetupFn, typename TimedFn>
static float benchmark_ms_excluding_setup(
    cudaStream_t stream,
    int warmup,
    int iters,
    SetupFn&& setup,
    TimedFn&& timed
) {
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
    }

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(starts[static_cast<size_t>(i)]));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(stops[static_cast<size_t>(i)]));
    }

    return total_ms / static_cast<float>(iters);
}

static void print_result(
    const char* label,
    float ms,
    int64_t n,
    int64_t rhs,
    int64_t batches
) {
    struct TimeDisplay {
        double value;
        const char* unit;
    };
    auto format_time_ms = [](double ms_value) -> TimeDisplay {
        if (ms_value >= 1.0) {
            return {ms_value, "ms"};
        }
        if (ms_value >= 1e-3) {
            return {ms_value * 1e3, "us"};
        }
        return {ms_value * 1e6, "ns"};
    };

    const double ms_d = static_cast<double>(ms);
    const double seconds = ms_d * 1e-3;
    const double ms_per_batch = ms_d / static_cast<double>(batches);
    const double batches_per_s = static_cast<double>(batches) / seconds;

    const double flops_per_batch =
        (static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n)) / 3.0 +
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(rhs);
    const double gflops_s =
        (flops_per_batch * static_cast<double>(batches)) / seconds / 1e9;

    const TimeDisplay total_time = format_time_ms(ms_d);
    const TimeDisplay batch_time = format_time_ms(ms_per_batch);

    std::printf(
        "%-32s %10.3f %2s  %10.3f %2s/batch  %10.3f batches/s  %10.3f GFLOP/s\n",
        label,
        total_time.value,
        total_time.unit,
        batch_time.value,
        batch_time.unit,
        batches_per_s,
        gflops_s
    );
}

static double compute_max_relative_residual(
    const std::vector<float>& a,
    const std::vector<float>& b_orig,
    const std::vector<float>& x,
    int64_t n,
    int64_t rhs,
    int64_t batches
) {
    const size_t a_batch_stride = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t b_batch_stride = static_cast<size_t>(n) * static_cast<size_t>(rhs);
    double max_rel = 0.0;

    for (int64_t batch = 0; batch < batches; ++batch) {
        const size_t a_batch_offset = static_cast<size_t>(batch) * a_batch_stride;
        const size_t b_batch_offset = static_cast<size_t>(batch) * b_batch_stride;
        for (int64_t col = 0; col < rhs; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                double ax = 0.0;
                double denom = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    const double a_val =
                        static_cast<double>(
                            a[a_batch_offset +
                              static_cast<size_t>(k) * static_cast<size_t>(n) +
                              static_cast<size_t>(row)]
                        );
                    const double x_val =
                        static_cast<double>(
                            x[b_batch_offset +
                              static_cast<size_t>(col) * static_cast<size_t>(n) +
                              static_cast<size_t>(k)]
                        );
                    ax += a_val * x_val;
                    denom += std::fabs(a_val) * std::fabs(x_val);
                }
                const double b_val =
                    static_cast<double>(
                        b_orig[b_batch_offset +
                               static_cast<size_t>(col) * static_cast<size_t>(n) +
                               static_cast<size_t>(row)]
                    );
                const double resid = std::fabs(ax - b_val);
                const double rel = resid / (denom + std::fabs(b_val) + 1e-12);
                max_rel = std::max(max_rel, rel);
            }
        }
    }

    return max_rel;
}

static void fill_spd_matrix(
    std::vector<float>& a,
    int64_t n,
    int64_t batches,
    float regularizer,
    float offdiag_scale,
    uint64_t seed
) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> offdiag_dist(-offdiag_scale, offdiag_scale);
    std::vector<float> row_abs_sum(static_cast<size_t>(n));
    const size_t batch_stride = static_cast<size_t>(n) * static_cast<size_t>(n);

    for (int64_t batch = 0; batch < batches; ++batch) {
        std::fill(row_abs_sum.begin(), row_abs_sum.end(), 0.0f);
        const size_t batch_offset = static_cast<size_t>(batch) * batch_stride;
        for (int64_t col = 0; col < n; ++col) {
            for (int64_t row = 0; row < col; ++row) {
                const float value = offdiag_dist(rng);
                a[batch_offset + static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)] = value;
                a[batch_offset + static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = value;
                row_abs_sum[static_cast<size_t>(row)] += std::fabs(value);
                row_abs_sum[static_cast<size_t>(col)] += std::fabs(value);
            }
        }
        for (int64_t diag = 0; diag < n; ++diag) {
            a[batch_offset + static_cast<size_t>(diag) * static_cast<size_t>(n) + static_cast<size_t>(diag)] =
                row_abs_sum[static_cast<size_t>(diag)] + 1.0f + regularizer;
        }
    }
}

static void fill_rhs(
    std::vector<float>& b,
    int64_t n,
    int64_t rhs,
    int64_t batches,
    uint64_t seed
) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> value_dist(0.0f, 1.0f);
    const size_t batch_stride = static_cast<size_t>(n) * static_cast<size_t>(rhs);

    for (int64_t batch = 0; batch < batches; ++batch) {
        const size_t batch_offset = static_cast<size_t>(batch) * batch_stride;
        for (int64_t col = 0; col < rhs; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                b[batch_offset + static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(row)] =
                    value_dist(rng);
            }
        }
    }
}

static unsigned int get_cuda_device_arch() {
    int device = 0;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
    int major = 0;
    int minor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)
    );
    CUDA_CHECK_AND_EXIT(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)
    );
    return static_cast<unsigned>(major) * 100u + static_cast<unsigned>(minor) * 10u;
}

template<class POSV, unsigned int BatchesPerBlock, class DataType = typename POSV::a_data_type>
__global__ __launch_bounds__(POSV::max_threads_per_block)
void posv_strided_kernel(
    DataType* A,
    unsigned int lda_gmem,
    DataType* B,
    unsigned int ldb_gmem,
    typename POSV::status_type* info,
    unsigned int batches
) {
    constexpr unsigned int m = POSV::m_size;
    constexpr unsigned int nrhs = POSV::k_size;
    constexpr unsigned int lda_smem = POSV::lda;
    constexpr unsigned int ldb_smem = POSV::ldb;
    constexpr size_t one_batch_size_a_smem =
        static_cast<size_t>(lda_smem) * static_cast<size_t>(m);
    constexpr size_t one_batch_size_b_smem =
        static_cast<size_t>(ldb_smem) * static_cast<size_t>(nrhs);

    const size_t one_batch_size_a_gmem =
        static_cast<size_t>(lda_gmem) * static_cast<size_t>(m);
    const size_t one_batch_size_b_gmem =
        static_cast<size_t>(ldb_gmem) * static_cast<size_t>(nrhs);

    extern __shared__ __align__(sizeof(DataType)) unsigned char shared_mem[];
    DataType* As = reinterpret_cast<DataType*>(shared_mem);
    DataType* Bs = As + one_batch_size_a_smem * static_cast<size_t>(BatchesPerBlock);

    const unsigned int batch0 = blockIdx.x * BatchesPerBlock;
    if (batch0 >= batches) {
        return;
    }

    for (unsigned int local = 0; local < BatchesPerBlock; ++local) {
        DataType* As_local = As + static_cast<size_t>(local) * one_batch_size_a_smem;
        DataType* Bs_local = Bs + static_cast<size_t>(local) * one_batch_size_b_smem;
        const unsigned int batch = batch0 + local;

        for (size_t idx = threadIdx.x; idx < one_batch_size_a_smem; idx += blockDim.x) {
            As_local[idx] = DataType(0);
        }
        for (size_t idx = threadIdx.x; idx < one_batch_size_b_smem; idx += blockDim.x) {
            Bs_local[idx] = DataType(0);
        }
        __syncthreads();

        if (batch < batches) {
            const DataType* Ag = A + static_cast<size_t>(batch) * one_batch_size_a_gmem;
            const DataType* Bg = B + static_cast<size_t>(batch) * one_batch_size_b_gmem;

            for (size_t idx = threadIdx.x; idx < static_cast<size_t>(m) * static_cast<size_t>(m); idx += blockDim.x) {
                const unsigned int row = static_cast<unsigned int>(idx % m);
                const unsigned int col = static_cast<unsigned int>(idx / m);
                As_local[static_cast<size_t>(row) + static_cast<size_t>(col) * lda_smem] =
                    Ag[static_cast<size_t>(row) + static_cast<size_t>(col) * lda_gmem];
            }
            for (size_t idx = threadIdx.x; idx < static_cast<size_t>(m) * static_cast<size_t>(nrhs); idx += blockDim.x) {
                const unsigned int row = static_cast<unsigned int>(idx % m);
                const unsigned int col = static_cast<unsigned int>(idx / m);
                Bs_local[static_cast<size_t>(row) + static_cast<size_t>(col) * ldb_smem] =
                    Bg[static_cast<size_t>(row) + static_cast<size_t>(col) * ldb_gmem];
            }
        } else {
            for (size_t idx = threadIdx.x; idx < m; idx += blockDim.x) {
                As_local[idx + idx * lda_smem] = DataType(1);
            }
        }
        __syncthreads();
    }

    POSV().execute(As, lda_smem, Bs, &info[batch0]);
    __syncthreads();

    for (unsigned int local = 0; local < BatchesPerBlock; ++local) {
        const unsigned int batch = batch0 + local;
        if (batch >= batches) {
            continue;
        }
        const DataType* As_local = As + static_cast<size_t>(local) * one_batch_size_a_smem;
        const DataType* Bs_local = Bs + static_cast<size_t>(local) * one_batch_size_b_smem;
        DataType* Ag = A + static_cast<size_t>(batch) * one_batch_size_a_gmem;
        DataType* Bg = B + static_cast<size_t>(batch) * one_batch_size_b_gmem;

        for (size_t idx = threadIdx.x; idx < static_cast<size_t>(m) * static_cast<size_t>(m); idx += blockDim.x) {
            const unsigned int row = static_cast<unsigned int>(idx % m);
            const unsigned int col = static_cast<unsigned int>(idx / m);
            Ag[static_cast<size_t>(row) + static_cast<size_t>(col) * lda_gmem] =
                As_local[static_cast<size_t>(row) + static_cast<size_t>(col) * lda_smem];
        }
        for (size_t idx = threadIdx.x; idx < static_cast<size_t>(m) * static_cast<size_t>(nrhs); idx += blockDim.x) {
            const unsigned int row = static_cast<unsigned int>(idx % m);
            const unsigned int col = static_cast<unsigned int>(idx / m);
            Bg[static_cast<size_t>(row) + static_cast<size_t>(col) * ldb_gmem] =
                Bs_local[static_cast<size_t>(row) + static_cast<size_t>(col) * ldb_smem];
        }
        __syncthreads();
    }
}

template <int Arch, int N, int NRHS>
static int run_benchmark_impl(const BenchConfig& cfg) {
    using namespace cusolverdx;

    using POSV = decltype(
        Size<N, N, NRHS>() +
        Precision<float>() +
        Type<type::real>() +
        Function<function::posv>() +
        FillMode<lower>() +
        Arrangement<col_major>() +
        SM<Arch>() +
        Block()
    );

    constexpr unsigned int bpb = POSV::batches_per_block;
    constexpr dim3 block_dim = POSV::block_dim;
    constexpr size_t shared_mem_bytes = POSV::shared_memory_size;
    constexpr unsigned int lda = N;
    constexpr unsigned int ldb = N;

    const int64_t padded_batches =
        ((cfg.L + static_cast<int64_t>(bpb) - 1) / static_cast<int64_t>(bpb)) * static_cast<int64_t>(bpb);

    size_t matrix_bytes = 0;
    size_t rhs_bytes = 0;
    if (!checked_tensor_bytes(&matrix_bytes, N, N, cfg.L, sizeof(float)) ||
        !checked_tensor_bytes(&rhs_bytes, N, NRHS, cfg.L, sizeof(float))) {
        std::fprintf(stderr, "Input sizes overflow size_t\n");
        return 1;
    }
    const size_t info_bytes = static_cast<size_t>(padded_batches) * sizeof(int);
    const size_t device_bytes = matrix_bytes * 2 + rhs_bytes * 2 + info_bytes;
    const size_t host_bytes = matrix_bytes + rhs_bytes + info_bytes;

    std::vector<float> h_a_orig(static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(cfg.L));
    std::vector<float> h_b_orig(static_cast<size_t>(N) * static_cast<size_t>(NRHS) * static_cast<size_t>(cfg.L));
    std::vector<float> h_x(static_cast<size_t>(N) * static_cast<size_t>(NRHS) * static_cast<size_t>(cfg.L));
    std::vector<int> h_info(static_cast<size_t>(padded_batches), 0);

    fill_spd_matrix(h_a_orig, N, cfg.L, cfg.regularizer, cfg.offdiag_scale, cfg.seed);
    fill_rhs(h_b_orig, N, NRHS, cfg.L, cfg.seed + 1);

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float* d_a_orig = nullptr;
    float* d_a = nullptr;
    float* d_b_orig = nullptr;
    float* d_b = nullptr;
    int* d_info = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_a_orig), matrix_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_a), matrix_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_b_orig), rhs_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_b), rhs_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), info_bytes));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
        d_a_orig,
        h_a_orig.data(),
        matrix_bytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
        d_b_orig,
        h_b_orig.data(),
        rhs_bytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    using Kernel = void (*)(
        float*,
        unsigned int,
        float*,
        unsigned int,
        typename POSV::status_type*,
        unsigned int
    );
    const Kernel kernel = posv_strided_kernel<POSV, POSV::batches_per_block>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_mem_bytes)
    ));

    std::printf("cuSolverDx POSV Solve-Only Benchmark\n");
    std::printf(
        "N=%d C=%d L=%ld arch=%d bpb=%u block_dim=%u warmup=%d iters=%d\n",
        N,
        NRHS,
        (long)cfg.L,
        Arch,
        bpb,
        block_dim.x,
        cfg.warmup,
        cfg.iters
    );
    std::printf(
        "regularizer=%.6g offdiag_scale=%.6g seed=%llu shared_mem=%zu\n",
        cfg.regularizer,
        cfg.offdiag_scale,
        static_cast<unsigned long long>(cfg.seed),
        shared_mem_bytes
    );
    SizeDisplay device_size = format_bytes(device_bytes);
    SizeDisplay host_size = format_bytes(host_bytes);
    std::printf(
        "Allocating device_bytes=%zu (%.3f %s)\n",
        device_bytes,
        device_size.value,
        device_size.unit
    );
    std::printf(
        "Host bytes=%zu (%.3f %s)\n\n",
        host_bytes,
        host_size.value,
        host_size.unit
    );
    std::fflush(stdout);

    const float ms = benchmark_ms_excluding_setup(
        stream,
        cfg.warmup,
        cfg.iters,
        [&]() {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                d_a,
                d_a_orig,
                matrix_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                d_b,
                d_b_orig,
                rhs_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ));
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_info, 0, info_bytes, stream));
        },
        [&]() {
            const unsigned int blocks =
                static_cast<unsigned int>((cfg.L + static_cast<int64_t>(bpb) - 1) / static_cast<int64_t>(bpb));
            kernel<<<blocks, block_dim, shared_mem_bytes, stream>>>(
                d_a,
                lda,
                d_b,
                ldb,
                d_info,
                static_cast<unsigned int>(cfg.L)
            );
            CUDA_CHECK_AND_EXIT(cudaGetLastError());
        }
    );

    print_result("cusolverdx_posv", ms, N, NRHS, cfg.L);

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
        h_info.data(),
        d_info,
        info_bytes,
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    for (int64_t i = 0; i < cfg.L; ++i) {
        if (h_info[static_cast<size_t>(i)] != 0) {
            std::fprintf(
                stderr,
                "cusolverdx_posv failed: info[%ld]=%d\n",
                (long)i,
                h_info[static_cast<size_t>(i)]
            );
            CUDA_CHECK_AND_EXIT(cudaFree(d_info));
            CUDA_CHECK_AND_EXIT(cudaFree(d_b));
            CUDA_CHECK_AND_EXIT(cudaFree(d_b_orig));
            CUDA_CHECK_AND_EXIT(cudaFree(d_a));
            CUDA_CHECK_AND_EXIT(cudaFree(d_a_orig));
            CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
            return 1;
        }
    }

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
        h_x.data(),
        d_b,
        rhs_bytes,
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    const double max_rel_residual =
        compute_max_relative_residual(h_a_orig, h_b_orig, h_x, N, NRHS, cfg.L);
    std::printf("max_relative_residual=%.3e\n", max_rel_residual);
    if (!(max_rel_residual <= 1e-4)) {
        std::fprintf(
            stderr,
            "cusolverdx_posv failed residual check: max_relative_residual=%.6e\n",
            max_rel_residual
        );
        CUDA_CHECK_AND_EXIT(cudaFree(d_info));
        CUDA_CHECK_AND_EXIT(cudaFree(d_b));
        CUDA_CHECK_AND_EXIT(cudaFree(d_b_orig));
        CUDA_CHECK_AND_EXIT(cudaFree(d_a));
        CUDA_CHECK_AND_EXIT(cudaFree(d_a_orig));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaFree(d_b));
    CUDA_CHECK_AND_EXIT(cudaFree(d_b_orig));
    CUDA_CHECK_AND_EXIT(cudaFree(d_a));
    CUDA_CHECK_AND_EXIT(cudaFree(d_a_orig));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return 0;
}

template <int Arch, int N>
static int dispatch_rhs(const BenchConfig& cfg) {
    switch (cfg.C) {
        case 1: return run_benchmark_impl<Arch, N, 1>(cfg);
        case 4: return run_benchmark_impl<Arch, N, 4>(cfg);
        case 8: return run_benchmark_impl<Arch, N, 8>(cfg);
        case 16: return run_benchmark_impl<Arch, N, 16>(cfg);
        case 32: return run_benchmark_impl<Arch, N, 32>(cfg);
        default:
            std::fprintf(
                stderr,
                "Unsupported C=%ld. Supported values are 1, 4, 8, 16, 32.\n",
                (long)cfg.C
            );
            return 1;
    }
}

template <int Arch>
static int dispatch_n(const BenchConfig& cfg) {
    switch (cfg.N) {
        case 16: return dispatch_rhs<Arch, 16>(cfg);
        case 32: return dispatch_rhs<Arch, 32>(cfg);
        case 33: return dispatch_rhs<Arch, 33>(cfg);
        case 64: return dispatch_rhs<Arch, 64>(cfg);
        case 65: return dispatch_rhs<Arch, 65>(cfg);
        case 128: return dispatch_rhs<Arch, 128>(cfg);
        default:
            std::fprintf(
                stderr,
                "Unsupported N=%ld. Supported values are 16, 32, 33, 64, 65, 128.\n",
                (long)cfg.N
            );
            return 1;
    }
}

static int dispatch_arch(const BenchConfig& cfg) {
    const unsigned int arch = get_cuda_device_arch();
    switch (arch) {
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_80
        case 800: return dispatch_n<800>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_86
        case 860: return dispatch_n<860>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_87
        case 870: return dispatch_n<870>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_89
        case 890: return dispatch_n<890>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_90
        case 900: return dispatch_n<900>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_100
        case 1000: return dispatch_n<1000>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_101
        case 1010: return dispatch_n<1010>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_103
        case 1030: return dispatch_n<1030>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_110
        case 1100: return dispatch_n<1100>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_120
        case 1200: return dispatch_n<1200>(cfg);
#endif
#ifdef KERMAC_CUSOLVERDX_ENABLE_SM_121
        case 1210: return dispatch_n<1210>(cfg);
#endif
        default:
            std::fprintf(
                stderr,
                "Benchmark not configured for SM %u. Reconfigure with "
                "-DKERMAC_CUSOLVERDX_CUDA_ARCHITECTURES=<sm-list>.\n",
                arch
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
        return 1;
    }

    if (cfg.N <= 0 || cfg.C <= 0 || cfg.L <= 0) {
        std::fprintf(stderr, "Invalid dimensions: N,C,L must be > 0\n");
        return 1;
    }
    if (cfg.iters < 1) {
        std::fprintf(stderr, "iters must be >= 1\n");
        return 1;
    }
    if (cfg.warmup < 0) {
        std::fprintf(stderr, "warmup must be >= 0\n");
        return 1;
    }
    if (cfg.regularizer <= 0.0f) {
        std::fprintf(stderr, "regularizer must be > 0\n");
        return 1;
    }
    if (cfg.offdiag_scale < 0.0f) {
        std::fprintf(stderr, "offdiag-scale must be >= 0\n");
        return 1;
    }

    try {
        return dispatch_arch(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Benchmark failed with exception: %s\n", e.what());
        return 1;
    }
}
