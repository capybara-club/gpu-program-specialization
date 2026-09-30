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

#include <kermac.hpp>
#include <check_result_helper.h>

using namespace kermac;

struct BenchConfig {
    int64_t M = 4096;
    int64_t N = 3072;
    int64_t K = 128;
    int64_t C = 32;
    int64_t L = 4;
    int warmup = 5;
    int iters = 20;
    float bandwidth = 10.0f;
    float regularizer = 1e-3f;
    float epsilon = 1e-5f;
    size_t host_bytes = 1ull << 28;
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

static void print_usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --M <rows>           Symmetric size / non-symmetric rows (default 4096)\n"
        "  --N <cols>           Non-symmetric cols (default 3072)\n"
        "  --K <dims>           Feature dimension (default 128)\n"
        "  --C <channels>       Gradient channels (default 32)\n"
        "  --L <batch>          Batch size (default 4)\n"
        "  --warmup <iters>     Warmup iterations (default 5)\n"
        "  --iters <iters>      Timed iterations (default 20)\n"
        "  --bandwidth <val>    Laplace bandwidth (default 10.0)\n"
        "  --regularizer <val>  Symmetric regularizer (default 1e-3)\n"
        "  --epsilon <val>      Laplace epsilon (default 1e-5)\n"
        "  --host-bytes <val>   Host allocator bytes (default 1<<28)\n"
        "  --help               Show this message\n",
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
        if (std::strcmp(key, "--M") == 0) {
            ok = parse_int64(val, &cfg->M);
        } else if (std::strcmp(key, "--N") == 0) {
            ok = parse_int64(val, &cfg->N);
        } else if (std::strcmp(key, "--K") == 0) {
            ok = parse_int64(val, &cfg->K);
        } else if (std::strcmp(key, "--C") == 0) {
            ok = parse_int64(val, &cfg->C);
        } else if (std::strcmp(key, "--L") == 0) {
            ok = parse_int64(val, &cfg->L);
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
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    double value = static_cast<double>(bytes);
    size_t idx = 0;
    while (value >= 1024.0 && idx + 1 < (sizeof(units) / sizeof(units[0]))) {
        value /= 1024.0;
        ++idx;
    }
    return {value, units[idx]};
}

template <typename Fn>
static float benchmark_ms(
    CUstream stream,
    int warmup,
    int iters,
    Fn&& fn
) {
    if (warmup < 0) warmup = 0;
    if (iters < 1) iters = 1;

    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cuCheck(cuStreamSynchronize(stream));

    CUevent start;
    CUevent stop;
    cuCheck(cuEventCreate(&start, CU_EVENT_DEFAULT));
    cuCheck(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    cuCheck(cuEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    cuCheck(cuEventRecord(stop, stream));
    cuCheck(cuEventSynchronize(stop));

    float ms = 0.0f;
    cuCheck(cuEventElapsedTime(&ms, start, stop));
    cuCheck(cuEventDestroy(start));
    cuCheck(cuEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

static void print_result(
    const char* label,
    float ms,
    int64_t rows,
    int64_t cols,
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

    const double elems = static_cast<double>(rows) * static_cast<double>(cols) * static_cast<double>(batches);
    const double elems_per_s = elems / (ms * 1e-3);
    const double ms_per_batch = ms / static_cast<double>(batches);
    const double batches_per_s = static_cast<double>(batches) / (ms * 1e-3);
    const TimeDisplay total_time = format_time_ms(ms);
    const TimeDisplay batch_time = format_time_ms(ms_per_batch);
    std::printf(
        "%-40s %10.3f %2s  %10.3f %2s/batch  %10.3f batches/s  %10.3f Gels/s\n",
        label,
        total_time.value,
        total_time.unit,
        batch_time.value,
        batch_time.unit,
        batches_per_s,
        elems_per_s / 1e9
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
    if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0 || cfg.C <= 0 || cfg.L <= 0) {
        std::fprintf(stderr, "Invalid dimensions: M,N,K,C,L must be > 0\n");
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

    try {
        Kermac kermac(2);
        LegacyHandle legacy(kermac);
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem = host_alloc(cfg.host_bytes);
        HostStackAllocator hsa(host_mem, cfg.host_bytes);

        TensorCoreMode tcm = KERMAC_TENSOR_CORE_MODE_F32;

        Semiring semiring_nonsymm = Semiring::laplace_l2(kermac, hsa, cfg.bandwidth);
        Semiring semiring_symm_full = Semiring::laplace_l2_symm(
            kermac, hsa, KERMAC_MATRIX_PACKED_TYPE_FULL,
            cfg.bandwidth, cfg.regularizer, cfg.epsilon
        );
        Semiring semiring_symm_lower = Semiring::laplace_l2_symm(
            kermac, hsa, KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE,
            cfg.bandwidth, cfg.regularizer, cfg.epsilon
        );
        Semiring semiring_symm_upper = Semiring::laplace_l2_symm(
            kermac, hsa, KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE,
            cfg.bandwidth, cfg.regularizer, cfg.epsilon
        );
        SemiringGradient semiring_grad_l2 = SemiringGradient::norm_l2(kermac, hsa);
        Agop agop = Agop::create(kermac, hsa);

        size_t device_bytes = 0;
        {
            DeviceStackAllocator dsa_dry(nullptr, 0);

            DeviceTensor<float> d_a(dsa_dry, cfg.M, cfg.K, cfg.L);
            DeviceTensor<float> d_b(dsa_dry, cfg.N, cfg.K, cfg.L);
            {
                DeviceTensor<float> d_kernel(dsa_dry, cfg.M, cfg.N, cfg.L);
                DeviceTensor<float> d_norm_a(dsa_dry, cfg.M, cfg.L);
                DeviceTensor<float> d_norm_b(dsa_dry, cfg.N, cfg.L);

                contraction(
                    kermac, dsa_dry, tcm,
                    1.0f,
                    d_a, "mkl",
                    d_b, "nkl",
                    0.0f,
                    d_kernel, "mnl",
                    d_kernel, "mnl",
                    stream
                );

                contraction(
                    kermac, dsa_dry, tcm,
                    1.0f,
                    d_a, "mkl",
                    d_a, "mkl",
                    0.0f,
                    d_norm_a, "ml",
                    d_norm_a, "ml",
                    stream
                );

                contraction(
                    kermac, dsa_dry, tcm,
                    1.0f,
                    d_b, "mkl",
                    d_b, "mkl",
                    0.0f,
                    d_norm_b, "ml",
                    d_norm_b, "ml",
                    stream
                );

                laplace(
                    legacy,
                    d_kernel,
                    d_norm_a,
                    d_norm_b,
                    cfg.bandwidth,
                    cfg.epsilon,
                    stream
                );
            }

            {
                DeviceTensor<float> d_kernel_symm(dsa_dry, cfg.M, cfg.M, cfg.L);

                contraction(
                    kermac, dsa_dry, tcm,
                    1.0f,
                    d_a, "mkl",
                    d_a, "nkl",
                    0.0f,
                    d_kernel_symm, "mnl",
                    d_kernel_symm, "mnl",
                    stream
                );

                laplace_symmetric(
                    legacy,
                    KERMAC_MATRIX_PACKED_TYPE_FULL,
                    dsa_dry,
                    d_kernel_symm,
                    cfg.bandwidth,
                    cfg.regularizer,
                    cfg.epsilon,
                    stream
                );
            }

            {
                DeviceTensor<float> d_kernel_grad(dsa_dry, cfg.M, cfg.N, cfg.L);
                DeviceTensor<float> d_solution(dsa_dry, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_grad(dsa_dry, cfg.M, cfg.K, cfg.C, cfg.L);

                cutensor_gradient_norm_l2(
                    kermac, dsa_dry, tcm,
                    cfg.bandwidth,
                    d_kernel_grad,
                    d_b,
                    d_solution,
                    d_a,
                    d_grad,
                    stream
                );

                semiring_grad_l2.run(
                    d_kernel_grad,
                    d_b,
                    d_solution,
                    d_a,
                    d_grad,
                    1.0f,
                    0.0f,
                    stream
                );
            }

            {
                DeviceTensor<float> d_kernel_grad(dsa_dry, cfg.M, cfg.N, cfg.L);
                DeviceTensor<float> d_solution(dsa_dry, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_agop_full(dsa_dry, cfg.K, cfg.K, cfg.L);
                DeviceTensor<float> d_agop_diag(dsa_dry, cfg.K, cfg.L);

                agop.run(
                    kermac, dsa_dry,
                    AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32,
                    AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
                    cfg.bandwidth,
                    d_kernel_grad,
                    d_b,
                    d_solution,
                    d_a,
                    d_agop_full,
                    stream
                );

                agop.run(
                    kermac, dsa_dry,
                    AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32,
                    AgopOutput::KERMAC_AGOP_OUTPUT_DIAG,
                    cfg.bandwidth,
                    d_kernel_grad,
                    d_b,
                    d_solution,
                    d_a,
                    d_agop_diag,
                    stream
                );
            }

            device_bytes = dsa_dry.get().largest_total_offset;
        }

        std::printf("Kernel Matrix Benchmark\n");
        std::printf("M=%ld N=%ld K=%ld C=%ld L=%ld warmup=%d iters=%d\n",
            (long)cfg.M, (long)cfg.N, (long)cfg.K, (long)cfg.C, (long)cfg.L, cfg.warmup, cfg.iters);
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

            DeviceTensor<float> d_a(dsa, cfg.M, cfg.K, cfg.L);
            DeviceTensor<float> d_b(dsa, cfg.N, cfg.K, cfg.L);

            rng(kermac, d_a, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
            rng(kermac, d_b, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
            cuCheck(cuStreamSynchronize(stream));

            std::printf("\nNon-symmetric kernel matrix (M x N)\n");
            std::printf("---------------------------------------------\n");

            {
                DeviceTensor<float> d_kernel(dsa, cfg.M, cfg.N, cfg.L);

                float ms = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    semiring_nonsymm.run(d_a, d_b, d_kernel, stream);
                });
                print_result("semiring_laplace_l2", ms, cfg.M, cfg.N, cfg.L);
            }

            {
                DeviceTensor<float> d_kernel(dsa, cfg.M, cfg.N, cfg.L);
                DeviceTensor<float> d_norm_a(dsa, cfg.M, cfg.L);
                DeviceTensor<float> d_norm_b(dsa, cfg.N, cfg.L);

                float ms = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    contraction(
                        kermac, dsa, tcm,
                        1.0f,
                        d_a, "mkl",
                        d_b, "nkl",
                        0.0f,
                        d_kernel, "mnl",
                        d_kernel, "mnl",
                        stream
                    );

                    contraction(
                        kermac, dsa, tcm,
                        1.0f,
                        d_a, "mkl",
                        d_a, "mkl",
                        0.0f,
                        d_norm_a, "ml",
                        d_norm_a, "ml",
                        stream
                    );

                    contraction(
                        kermac, dsa, tcm,
                        1.0f,
                        d_b, "mkl",
                        d_b, "mkl",
                        0.0f,
                        d_norm_b, "ml",
                        d_norm_b, "ml",
                        stream
                    );

                    laplace(
                        legacy,
                        d_kernel,
                        d_norm_a,
                        d_norm_b,
                        cfg.bandwidth,
                        cfg.epsilon,
                        stream
                    );
                });
                print_result("legacy_dot+laplace", ms, cfg.M, cfg.N, cfg.L);
            }

            std::printf("\nL2 gradient (M x D x C)\n");
            std::printf("---------------------------------------------\n");

            {
                DeviceTensor<float> d_kernel_grad(dsa, cfg.M, cfg.N, cfg.L);
                DeviceTensor<float> d_solution(dsa, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_grad(dsa, cfg.M, cfg.K, cfg.C, cfg.L);

                rng(kermac, d_kernel_grad, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
                rng(kermac, d_solution, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
                cuCheck(cuStreamSynchronize(stream));

                float ms = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    semiring_grad_l2.run(d_kernel_grad, d_b, d_solution, d_a, d_grad, 1.0f, 0.0f, stream);
                });
                print_result("semiring_grad_l2", ms, cfg.M * cfg.K, cfg.C, cfg.L);

                float ms_cutensor = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    cutensor_gradient_norm_l2(
                        kermac, dsa, tcm,
                        cfg.bandwidth,
                        d_kernel_grad,
                        d_b,
                        d_solution,
                        d_a,
                        d_grad,
                        stream
                    );
                });
                print_result("cutensor_grad_l2", ms_cutensor, cfg.M * cfg.K, cfg.C, cfg.L);
            }

            {
                DeviceTensor<float> d_kernel_grad(dsa, cfg.M, cfg.N, cfg.L);
                DeviceTensor<float> d_solution(dsa, cfg.N, cfg.C, cfg.L);
                DeviceTensor<float> d_agop_full(dsa, cfg.K, cfg.K, cfg.L);
                DeviceTensor<float> d_agop_diag(dsa, cfg.K, cfg.L);

                rng(kermac, d_kernel_grad, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
                rng(kermac, d_solution, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);
                cuCheck(cuStreamSynchronize(stream));

                std::printf("\nAGOP (D x D)\n");
                std::printf("---------------------------------------------\n");

                float ms_full_cutensor = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    agop.run(
                        kermac, dsa,
                        AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32,
                        AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
                        cfg.bandwidth,
                        d_kernel_grad,
                        d_b,
                        d_solution,
                        d_a,
                        d_agop_full,
                        stream
                    );
                });
                print_result("agop_full_cutensor_f32", ms_full_cutensor, cfg.K, cfg.K, cfg.L);

                float ms_full_fused = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    agop.run(
                        kermac, dsa,
                        AgopBackend::KERMAC_AGOP_BACKEND_FUSED,
                        AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
                        cfg.bandwidth,
                        d_kernel_grad,
                        d_b,
                        d_solution,
                        d_a,
                        d_agop_full,
                        stream
                    );
                });
                print_result("agop_full_fused", ms_full_fused, cfg.K, cfg.K, cfg.L);

                std::printf("\nAGOP (D)\n");
                std::printf("---------------------------------------------\n");

                float ms_diag_cutensor = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    agop.run(
                        kermac, dsa,
                        AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32,
                        AgopOutput::KERMAC_AGOP_OUTPUT_DIAG,
                        cfg.bandwidth,
                        d_kernel_grad,
                        d_b,
                        d_solution,
                        d_a,
                        d_agop_diag,
                        stream
                    );
                });
                print_result("agop_diag_cutensor_f32", ms_diag_cutensor, cfg.K, 1, cfg.L);

                float ms_diag_fused = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    agop.run(
                        kermac, dsa,
                        AgopBackend::KERMAC_AGOP_BACKEND_FUSED,
                        AgopOutput::KERMAC_AGOP_OUTPUT_DIAG,
                        cfg.bandwidth,
                        d_kernel_grad,
                        d_b,
                        d_solution,
                        d_a,
                        d_agop_diag,
                        stream
                    );
                });
                print_result("agop_diag_fused", ms_diag_fused, cfg.K, 1, cfg.L);
            }

            std::printf("\nSymmetric kernel matrix (M x M)\n");
            std::printf("---------------------------------------------\n");

            auto run_symm = [&](const char* suffix, Semiring& semiring, MatrixPackedType packed_type) {
                DeviceTensor<float> d_kernel(dsa, cfg.M, cfg.M, cfg.L);
                char semiring_label[64];
                std::snprintf(semiring_label, sizeof(semiring_label), "semiring_laplace_symm_%s", suffix);

                float ms = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    semiring.run(d_a, d_a, d_kernel, stream);
                });
                print_result(semiring_label, ms, cfg.M, cfg.M, cfg.L);

                float ms_legacy = benchmark_ms(stream, cfg.warmup, cfg.iters, [&]() {
                    contraction(
                        kermac, dsa, tcm,
                        1.0f,
                        d_a, "mkl",
                        d_a, "nkl",
                        0.0f,
                        d_kernel, "mnl",
                        d_kernel, "mnl",
                        stream
                    );

                    laplace_symmetric(
                        legacy,
                        packed_type,
                        dsa,
                        d_kernel,
                        cfg.bandwidth,
                        cfg.regularizer,
                        cfg.epsilon,
                        stream
                    );
                });
                char legacy_label[64];
                std::snprintf(legacy_label, sizeof(legacy_label), "legacy_laplace_symm_%s", suffix);
                print_result(legacy_label, ms_legacy, cfg.M, cfg.M, cfg.L);
            };

            run_symm("full", semiring_symm_full, KERMAC_MATRIX_PACKED_TYPE_FULL);
            run_symm("lower", semiring_symm_lower, KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE);
            run_symm("upper", semiring_symm_upper, KERMAC_MATRIX_PACKED_TYPE_UPPER_TRIANGLE);
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
