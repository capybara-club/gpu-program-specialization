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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <kermac.hpp>
#include <check_result_helper.h>

using namespace kermac;

enum class SolverMode {
    New,
    Legacy,
    Both,
};

struct BenchConfig {
    int64_t N = 2048;
    int64_t K = 128;
    int64_t C = 32;
    int64_t L = 4;
    int warmup = 5;
    int iters = 20;
    int streams = 2;
    float bandwidth = 10.0f;
    float regularizer = 1e-3f;
    float epsilon = 1e-5f;
    size_t host_bytes = 1ull << 28;
    MatrixPackedType packed = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
    SolverMode mode = SolverMode::Both;
};

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

static const char* packed_to_string(MatrixPackedType packed) {
    switch (packed) {
        case KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE:
            return "lower";
        case KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE:
            return "upper";
        default:
            return "unknown";
    }
}

static const char* mode_to_string(SolverMode mode) {
    switch (mode) {
        case SolverMode::New:
            return "new";
        case SolverMode::Legacy:
            return "legacy";
        case SolverMode::Both:
            return "both";
        default:
            return "unknown";
    }
}

static void print_usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --N <rows>             Matrix size (A is NxN) (default 2048)\n"
        "  --K <dims>             Feature dim for kernel gen (default 128)\n"
        "  --C <rhs>              RHS columns / labels (default 32)\n"
        "  --L <batch>            Batch size (default 4)\n"
        "  --packed <lower|upper> Triangle used by solver (default lower)\n"
        "  --mode <new|legacy|both> Which solver(s) to time (default both)\n"
        "  --streams <n>          Secondary streams for new solver (default 2)\n"
        "  --warmup <iters>       Warmup iterations (default 5)\n"
        "  --iters <iters>        Timed iterations (default 20)\n"
        "  --bandwidth <val>      Laplace bandwidth (default 10.0)\n"
        "  --regularizer <val>    Diagonal regularizer (default 1e-3)\n"
        "  --epsilon <val>        Laplace epsilon (default 1e-5)\n"
        "  --host-bytes <val>     Host allocator bytes (default 1<<28)\n"
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

static bool parse_size(const char* arg, size_t* out) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(arg, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<size_t>(v);
    return true;
}

static bool parse_packed(const char* arg, MatrixPackedType* out) {
    if (std::strcmp(arg, "lower") == 0) {
        *out = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
        return true;
    }
    if (std::strcmp(arg, "upper") == 0) {
        *out = KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE;
        return true;
    }
    return false;
}

static bool parse_mode(const char* arg, SolverMode* out) {
    if (std::strcmp(arg, "new") == 0) {
        *out = SolverMode::New;
        return true;
    }
    if (std::strcmp(arg, "legacy") == 0) {
        *out = SolverMode::Legacy;
        return true;
    }
    if (std::strcmp(arg, "both") == 0) {
        *out = SolverMode::Both;
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
            print_usage(argv[0]);
            return false;
        }
        const char* key = argv[i];
        const char* val = argv[i + 1];
        bool ok = true;
        if (std::strcmp(key, "--N") == 0) {
            ok = parse_int64(val, &cfg->N);
        } else if (std::strcmp(key, "--K") == 0) {
            ok = parse_int64(val, &cfg->K);
        } else if (std::strcmp(key, "--C") == 0) {
            ok = parse_int64(val, &cfg->C);
        } else if (std::strcmp(key, "--L") == 0) {
            ok = parse_int64(val, &cfg->L);
        } else if (std::strcmp(key, "--packed") == 0) {
            ok = parse_packed(val, &cfg->packed);
        } else if (std::strcmp(key, "--mode") == 0) {
            ok = parse_mode(val, &cfg->mode);
        } else if (std::strcmp(key, "--streams") == 0) {
            ok = parse_int(val, &cfg->streams);
        } else if (std::strcmp(key, "--warmup") == 0) {
            ok = parse_int(val, &cfg->warmup);
        } else if (std::strcmp(key, "--iters") == 0) {
            ok = parse_int(val, &cfg->iters);
        } else if (std::strcmp(key, "--bandwidth") == 0) {
            ok = parse_float(val, &cfg->bandwidth);
        } else if (std::strcmp(key, "--regularizer") == 0) {
            ok = parse_float(val, &cfg->regularizer);
        } else if (std::strcmp(key, "--epsilon") == 0) {
            ok = parse_float(val, &cfg->epsilon);
        } else if (std::strcmp(key, "--host-bytes") == 0) {
            ok = parse_size(val, &cfg->host_bytes);
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

struct SizeDisplay {
    double value;
    const char* unit;
};

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

template <typename SetupFn, typename TimedFn>
static float benchmark_ms_excluding_setup(
    CUstream stream,
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
    cuCheck(cuStreamSynchronize(stream));

    std::vector<CUevent> starts(static_cast<size_t>(iters));
    std::vector<CUevent> stops(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cuCheck(cuEventCreate(&starts[static_cast<size_t>(i)], CU_EVENT_DEFAULT));
        cuCheck(cuEventCreate(&stops[static_cast<size_t>(i)], CU_EVENT_DEFAULT));
    }

    for (int i = 0; i < iters; ++i) {
        setup();
        cuCheck(cuEventRecord(starts[static_cast<size_t>(i)], stream));
        timed();
        cuCheck(cuEventRecord(stops[static_cast<size_t>(i)], stream));
    }

    cuCheck(cuEventSynchronize(stops.back()));

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        float ms = 0.0f;
        cuCheck(cuEventElapsedTime(&ms, starts[static_cast<size_t>(i)], stops[static_cast<size_t>(i)]));
        total_ms += ms;
    }

    for (int i = 0; i < iters; ++i) {
        cuCheck(cuEventDestroy(starts[static_cast<size_t>(i)]));
        cuCheck(cuEventDestroy(stops[static_cast<size_t>(i)]));
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

    // Rough flops: potrf ~ 1/3 n^3, potrs ~ 2 n^2 rhs.
    const double flops_per_batch =
        (static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n)) / 3.0 +
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(rhs);
    const double gflops_s = (flops_per_batch * static_cast<double>(batches)) / seconds / 1e9;

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

int main(int argc, char** argv) {
    if (!has_cuda_device()) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }

    BenchConfig cfg;
    if (!parse_args(argc, argv, &cfg)) {
        return 1;
    }

    if (cfg.N <= 0 || cfg.K <= 0 || cfg.C <= 0 || cfg.L <= 0) {
        std::fprintf(stderr, "Invalid dimensions: N,K,C,L must be > 0\n");
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
    if (cfg.host_bytes == 0) {
        std::fprintf(stderr, "host-bytes must be > 0\n");
        return 1;
    }
    if (cfg.streams < 0) {
        std::fprintf(stderr, "streams must be >= 0\n");
        return 1;
    }
    if ((cfg.mode == SolverMode::New || cfg.mode == SolverMode::Both) && cfg.streams < 1) {
        std::fprintf(stderr, "streams must be >= 1 when benchmarking the new solver\n");
        return 1;
    }

    try {
        Kermac kermac(static_cast<size_t>(cfg.streams));
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem = host_alloc(cfg.host_bytes);
        HostStackAllocator hsa(host_mem, cfg.host_bytes);

        Semiring kernel_symm = Semiring::laplace_l2_symm(
            kermac, hsa, cfg.packed, cfg.bandwidth, cfg.regularizer, cfg.epsilon
        );

        LegacyHandle legacy(kermac);

        size_t device_bytes = 0;
        {
            DeviceStackAllocator dsa_dry(nullptr, 0);
            DeviceTensor<float> d_x(dsa_dry, cfg.N, cfg.K, cfg.L);
            DeviceTensor<float> d_a_orig(dsa_dry, cfg.N, cfg.N, cfg.L);
            DeviceTensor<float> d_a(dsa_dry, cfg.N, cfg.N, cfg.L);

            if (cfg.mode == SolverMode::New || cfg.mode == SolverMode::Both) {
                DeviceTensor<float> d_b_orig(dsa_dry, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_b(dsa_dry, cfg.N, cfg.C, cfg.L);
                DeviceTensor<int32_t> d_factor_info(dsa_dry, cfg.L);
                DeviceTensor<int32_t> d_solve_info(dsa_dry, cfg.L);

                solve(kermac, cfg.packed, dsa_dry, d_a, d_b, d_factor_info, d_solve_info, stream);
            }

            if (cfg.mode == SolverMode::Legacy || cfg.mode == SolverMode::Both) {
                DeviceTensor<float> d_b_orig_legacy(dsa_dry, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_b_legacy(dsa_dry, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float*> d_a_array(dsa_dry, cfg.L);
                DeviceTensor<float*> d_b_array(dsa_dry, cfg.L);
                DeviceTensor<int> d_factor_info_legacy(dsa_dry, cfg.L);
                DeviceTensor<int> d_solve_info_legacy(dsa_dry, 1);

                solve_compute_array(legacy, d_a, d_a_array, stream);
                solve_compute_array(legacy, d_b_legacy, d_b_array, stream);
                solve(
                    kermac,
                    cfg.packed,
                    d_a,
                    d_b_legacy,
                    d_a_array,
                    d_b_array,
                    d_factor_info_legacy,
                    d_solve_info_legacy,
                    stream
                );
            }

            device_bytes = dsa_dry.get().largest_total_offset;
        }

        std::printf("Solve Benchmark\n");
        std::printf(
            "N=%ld K=%ld C=%ld L=%ld packed=%s mode=%s streams=%d warmup=%d iters=%d\n",
            (long)cfg.N,
            (long)cfg.K,
            (long)cfg.C,
            (long)cfg.L,
            packed_to_string(cfg.packed),
            mode_to_string(cfg.mode),
            cfg.streams,
            cfg.warmup,
            cfg.iters
        );
        std::printf(
            "bandwidth=%.6g regularizer=%.6g epsilon=%.6g\n",
            cfg.bandwidth,
            cfg.regularizer,
            cfg.epsilon
        );
        SizeDisplay device_size = format_bytes(device_bytes);
        SizeDisplay host_size = format_bytes(cfg.host_bytes);
        std::printf(
            "Allocating device_bytes=%zu (%.3f %s)\n",
            device_bytes,
            device_size.value,
            device_size.unit
        );
        std::printf(
            "Host bytes=%zu (%.3f %s)\n\n",
            cfg.host_bytes,
            host_size.value,
            host_size.unit
        );
        std::fflush(stdout);
        void* device_mem = device_alloc(kermac, device_bytes);

        {
            DeviceStackAllocator dsa(device_mem, device_bytes);

            DeviceTensor<float> d_x(dsa, cfg.N, cfg.K, cfg.L);
            DeviceTensor<float> d_a_orig(dsa, cfg.N, cfg.N, cfg.L);
            DeviceTensor<float> d_a(dsa, cfg.N, cfg.N, cfg.L);

            rng(kermac, d_x, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);

            kernel_symm.run(d_x, d_x, d_a_orig, stream);

            cuCheck(cuStreamSynchronize(stream));

            if (cfg.mode == SolverMode::New || cfg.mode == SolverMode::Both) {
                DeviceTensor<float> d_b_orig(dsa, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_b(dsa, cfg.N, cfg.C, cfg.L);
                DeviceTensor<int32_t> d_factor_info(dsa, cfg.L);
                DeviceTensor<int32_t> d_solve_info(dsa, cfg.L);

                rng(kermac, d_b_orig, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
                cuCheck(cuStreamSynchronize(stream));

                float ms = benchmark_ms_excluding_setup(
                    stream,
                    cfg.warmup,
                    cfg.iters,
                    [&]() {
                        d_a.copy_from(d_a_orig, stream);
                        d_b.copy_from(d_b_orig, stream);
                    },
                    [&]() {
                        solve(kermac, cfg.packed, dsa, d_a, d_b, d_factor_info, d_solve_info, stream);
                    }
                );

                print_result("solve_xpotrf_xpotrs", ms, cfg.N, cfg.C, cfg.L);

                HostTensor<int32_t> h_factor_info(hsa, d_factor_info.extent());
                HostTensor<int32_t> h_solve_info(hsa, d_solve_info.extent());
                h_factor_info.copy_from(d_factor_info, stream);
                h_solve_info.copy_from(d_solve_info, stream);
                cuCheck(cuStreamSynchronize(stream));

                const int32_t* factor_ptr = h_factor_info.ptr();
                const int32_t* solve_ptr = h_solve_info.ptr();
                for (int64_t i = 0; i < cfg.L; ++i) {
                    if (factor_ptr[static_cast<size_t>(i)] != 0 || solve_ptr[static_cast<size_t>(i)] != 0) {
                        std::fprintf(
                            stderr,
                            "solve_xpotrf_xpotrs failed: factor_info[%ld]=%d solve_info[%ld]=%d\n",
                            (long)i,
                            factor_ptr[static_cast<size_t>(i)],
                            (long)i,
                            solve_ptr[static_cast<size_t>(i)]
                        );
                        return 1;
                    }
                }
            }

            if (cfg.mode == SolverMode::Legacy || cfg.mode == SolverMode::Both) {
                DeviceTensor<float> d_b_orig_legacy(dsa, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_b_legacy(dsa, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float*> d_a_array(dsa, cfg.L);
                DeviceTensor<float*> d_b_array(dsa, cfg.L);
                DeviceTensor<int> d_factor_info_legacy(dsa, cfg.L);
                DeviceTensor<int> d_solve_info_legacy(dsa, 1);

                rng(kermac, d_b_orig_legacy, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
                solve_compute_array(legacy, d_a, d_a_array, stream);
                solve_compute_array(legacy, d_b_legacy, d_b_array, stream);
                cuCheck(cuStreamSynchronize(stream));

                float ms = benchmark_ms_excluding_setup(
                    stream,
                    cfg.warmup,
                    cfg.iters,
                    [&]() {
                        d_a.copy_from(d_a_orig, stream);
                        d_b_legacy.copy_from(d_b_orig_legacy, stream);
                    },
                    [&]() {
                        solve(
                            kermac,
                            cfg.packed,
                            d_a,
                            d_b_legacy,
                            d_a_array,
                            d_b_array,
                            d_factor_info_legacy,
                            d_solve_info_legacy,
                            stream
                        );
                    }
                );

                print_result("legacy_potrfBatched_solve", ms, cfg.N, cfg.C, cfg.L);

                HostTensor<int> h_factor_info_legacy(hsa, d_factor_info_legacy.extent());
                HostTensor<int> h_solve_info_legacy(hsa, d_solve_info_legacy.extent());
                h_factor_info_legacy.copy_from(d_factor_info_legacy, stream);
                h_solve_info_legacy.copy_from(d_solve_info_legacy, stream);
                cuCheck(cuStreamSynchronize(stream));

                const int* factor_ptr = h_factor_info_legacy.ptr();
                for (int64_t i = 0; i < cfg.L; ++i) {
                    if (factor_ptr[static_cast<size_t>(i)] != 0) {
                        std::fprintf(
                            stderr,
                            "legacy solver failed: factor_info[%ld]=%d\n",
                            (long)i,
                            factor_ptr[static_cast<size_t>(i)]
                        );
                        return 1;
                    }
                }
                if (h_solve_info_legacy.ptr()[0] != 0) {
                    std::fprintf(
                        stderr,
                        "legacy solver failed: solve_info[0]=%d\n",
                        h_solve_info_legacy.ptr()[0]
                    );
                    return 1;
                }
            }
        }

        device_free(kermac, device_mem);
        host_free(host_mem);
        cuCheck(cuStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Benchmark failed with exception: %s\n", e.what());
        return 1;
    }
}
