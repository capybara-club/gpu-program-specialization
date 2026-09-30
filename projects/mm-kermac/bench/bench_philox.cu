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
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <limits>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                 \
                cudaGetErrorString(status), __FILE__, __LINE__              \
            );                                                               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

enum class Mode {
    UNIFORM_F32,
    NORMAL_F32,
    U32
};

struct BenchConfig {
    int device = 0;
    int64_t rows = 4096;
    int64_t cols = 4096;
    uint64_t elements = 0;
    uint64_t rng_iters = 1024;
    bool no_store = false;
    int warmup = 5;
    int iters = 20;
    int threads = 256;
    int blocks = 0;
    int blocks_per_sm = 1;
    uint64_t seed = 1234ull;
    double scale = 1.0;
    double shift = 0.0;
    Mode mode = Mode::UNIFORM_F32;
};

static void print_usage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --rows <rows>            Output rows (default 4096)\n"
        "  --cols <cols>            Output cols (default 4096)\n"
        "  --elements <count>       Total elements (overrides rows/cols)\n"
        "  --rng-iters <count>      RNG iterations per thread for --no-store (default 1024)\n"
        "  --no-store               Generate RNG values without writing outputs (ignores rows/cols/elements)\n"
        "  --mode <name>            uniform_f32 | normal_f32 | u32\n"
        "  --scale <val>            Scale applied to RNG output (default 1.0)\n"
        "  --shift <val>            Shift applied to RNG output (default 0.0)\n"
        "  --warmup <iters>         Warmup iterations (default 5)\n"
        "  --iters <iters>          Timed iterations (default 20)\n"
        "  --threads <count>        Threads per block (default 256)\n"
        "  --blocks <count>         Blocks in grid (overrides blocks-per-sm)\n"
        "  --blocks-per-sm <count>  Blocks per SM (default 1)\n"
        "  --seed <val>             RNG seed (default 1234)\n"
        "  --device <id>            CUDA device index (default 0)\n"
        "  --help                   Show this message\n",
        exe
    );
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

static bool parse_int64(const char* arg, int64_t* out) {
    char* end = nullptr;
    long long v = std::strtoll(arg, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<int64_t>(v);
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

static bool parse_double(const char* arg, double* out) {
    char* end = nullptr;
    double v = std::strtod(arg, &end);
    if (!end || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

static bool parse_mode(const char* arg, Mode* out) {
    if (std::strcmp(arg, "uniform_f32") == 0 || std::strcmp(arg, "uniform") == 0) {
        *out = Mode::UNIFORM_F32;
        return true;
    }
    if (std::strcmp(arg, "normal_f32") == 0 || std::strcmp(arg, "normal") == 0) {
        *out = Mode::NORMAL_F32;
        return true;
    }
    if (std::strcmp(arg, "u32") == 0) {
        *out = Mode::U32;
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
        if (std::strcmp(argv[i], "--no-store") == 0) {
            cfg->no_store = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
        const char* key = argv[i];
        const char* val = argv[i + 1];
        bool ok = true;
        if (std::strcmp(key, "--rows") == 0) {
            ok = parse_int64(val, &cfg->rows);
        } else if (std::strcmp(key, "--cols") == 0) {
            ok = parse_int64(val, &cfg->cols);
        } else if (std::strcmp(key, "--elements") == 0) {
            ok = parse_u64(val, &cfg->elements);
        } else if (std::strcmp(key, "--rng-iters") == 0) {
            ok = parse_u64(val, &cfg->rng_iters);
        } else if (std::strcmp(key, "--mode") == 0) {
            ok = parse_mode(val, &cfg->mode);
        } else if (std::strcmp(key, "--scale") == 0) {
            ok = parse_double(val, &cfg->scale);
        } else if (std::strcmp(key, "--shift") == 0) {
            ok = parse_double(val, &cfg->shift);
        } else if (std::strcmp(key, "--warmup") == 0) {
            ok = parse_int(val, &cfg->warmup);
        } else if (std::strcmp(key, "--iters") == 0) {
            ok = parse_int(val, &cfg->iters);
        } else if (std::strcmp(key, "--threads") == 0) {
            ok = parse_int(val, &cfg->threads);
        } else if (std::strcmp(key, "--blocks") == 0) {
            ok = parse_int(val, &cfg->blocks);
        } else if (std::strcmp(key, "--blocks-per-sm") == 0) {
            ok = parse_int(val, &cfg->blocks_per_sm);
        } else if (std::strcmp(key, "--seed") == 0) {
            ok = parse_u64(val, &cfg->seed);
        } else if (std::strcmp(key, "--device") == 0) {
            ok = parse_int(val, &cfg->device);
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

__global__
void philox_init(curandStatePhilox4_32_10_t* states, uint64_t seed) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(static_cast<unsigned long long>(seed), static_cast<unsigned long long>(thread_id), 0, &states[thread_id]);
}

__global__
void philox_fill_uniform_f32(
    float* out,
    uint64_t total_elements,
    curandStatePhilox4_32_10_t* states,
    float scale,
    float shift
) {
    uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total_threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t local_state = states[thread_id];
    uint64_t idx = thread_id * 4;
    uint64_t stride = total_threads * 4;
    while (idx < total_elements) {
        float4 v = curand_uniform4(&local_state);
        if (idx < total_elements) out[idx] = v.x * scale + shift;
        if (idx + 1 < total_elements) out[idx + 1] = v.y * scale + shift;
        if (idx + 2 < total_elements) out[idx + 2] = v.z * scale + shift;
        if (idx + 3 < total_elements) out[idx + 3] = v.w * scale + shift;
        idx += stride;
    }
    states[thread_id] = local_state;
}

__global__
void philox_fill_normal_f32(
    float* out,
    uint64_t total_elements,
    curandStatePhilox4_32_10_t* states,
    float scale,
    float shift
) {
    uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total_threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t local_state = states[thread_id];
    uint64_t idx = thread_id * 4;
    uint64_t stride = total_threads * 4;
    while (idx < total_elements) {
        float4 v = curand_normal4(&local_state);
        if (idx < total_elements) out[idx] = v.x * scale + shift;
        if (idx + 1 < total_elements) out[idx + 1] = v.y * scale + shift;
        if (idx + 2 < total_elements) out[idx + 2] = v.z * scale + shift;
        if (idx + 3 < total_elements) out[idx + 3] = v.w * scale + shift;
        idx += stride;
    }
    states[thread_id] = local_state;
}

__global__
void philox_fill_u32(
    uint32_t* out,
    uint64_t total_elements,
    curandStatePhilox4_32_10_t* states
) {
    uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total_threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t local_state = states[thread_id];
    uint64_t idx = thread_id * 4;
    uint64_t stride = total_threads * 4;
    while (idx < total_elements) {
        uint4 v = curand4(&local_state);
        if (idx < total_elements) out[idx] = v.x;
        if (idx + 1 < total_elements) out[idx + 1] = v.y;
        if (idx + 2 < total_elements) out[idx + 2] = v.z;
        if (idx + 3 < total_elements) out[idx + 3] = v.w;
        idx += stride;
    }
    states[thread_id] = local_state;
}

// Keeps RNG outputs live in no-store kernels without writing them.
__device__ __forceinline__ void consume_u32(uint32_t v) {
    asm volatile("" : "+r"(v));
}

__global__
void philox_compute_uniform_f32(
    curandStatePhilox4_32_10_t* states,
    uint64_t iters
) {
    uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t local_state = states[thread_id];
    uint32_t acc = 0;
    for (uint64_t i = 0; i < iters; ++i) {
        float4 v = curand_uniform4(&local_state);
        acc ^= __float_as_uint(v.x);
        acc ^= __float_as_uint(v.y);
        acc ^= __float_as_uint(v.z);
        acc ^= __float_as_uint(v.w);
    }
    consume_u32(acc);
    states[thread_id] = local_state;
}

__global__
void philox_compute_normal_f32(
    curandStatePhilox4_32_10_t* states,
    uint64_t iters
) {
    uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t local_state = states[thread_id];
    uint32_t acc = 0;
    for (uint64_t i = 0; i < iters; ++i) {
        float4 v = curand_normal4(&local_state);
        acc ^= __float_as_uint(v.x);
        acc ^= __float_as_uint(v.y);
        acc ^= __float_as_uint(v.z);
        acc ^= __float_as_uint(v.w);
    }
    consume_u32(acc);
    states[thread_id] = local_state;
}

__global__
void philox_compute_u32(
    curandStatePhilox4_32_10_t* states,
    uint64_t iters
) {
    uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t local_state = states[thread_id];
    uint32_t acc = 0;
    for (uint64_t i = 0; i < iters; ++i) {
        uint4 v = curand4(&local_state);
        acc ^= v.x;
        acc ^= v.y;
        acc ^= v.z;
        acc ^= v.w;
    }
    consume_u32(acc);
    states[thread_id] = local_state;
}

template <typename LaunchFn>
static float benchmark_ms(int warmup, int iters, LaunchFn&& launch) {
    if (warmup < 0) warmup = 0;
    if (iters < 1) iters = 1;

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

static const char* mode_label(Mode mode) {
    switch (mode) {
        case Mode::UNIFORM_F32: return "uniform_f32";
        case Mode::NORMAL_F32: return "normal_f32";
        case Mode::U32: return "u32";
    }
    return "unknown";
}

struct SizeDisplay {
    double value;
    const char* unit;
};

static SizeDisplay format_bytes_decimal(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    double value = static_cast<double>(bytes);
    size_t idx = 0;
    while (value >= 1000.0 && idx + 1 < (sizeof(units) / sizeof(units[0]))) {
        value /= 1000.0;
        ++idx;
    }
    return {value, units[idx]};
}

static void print_kv(const char* key, const char* fmt, ...) {
    std::printf("  %-18s ", key);
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
}

static bool has_cuda_device() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return false;
    }
    return count > 0;
}

template <typename T, typename LaunchFn>
static void run_bench_store(
    const BenchConfig& cfg,
    const cudaDeviceProp& prop,
    uint64_t total_elements,
    const char* label,
    LaunchFn&& launch_fn
) {
    int blocks = cfg.blocks;
    if (blocks <= 0) {
        if (cfg.blocks_per_sm <= 0) {
            std::fprintf(stderr, "blocks-per-sm must be > 0\n");
            std::exit(1);
        }
        if (prop.multiProcessorCount > 0 && cfg.blocks_per_sm > std::numeric_limits<int>::max() / prop.multiProcessorCount) {
            std::fprintf(stderr, "blocks-per-sm too large for this device\n");
            std::exit(1);
        }
        blocks = prop.multiProcessorCount * cfg.blocks_per_sm;
    }

    if (blocks <= 0 || blocks > prop.maxGridSize[0]) {
        std::fprintf(stderr, "Invalid block count: %d\n", blocks);
        std::exit(1);
    }
    if (cfg.threads <= 0 || cfg.threads > prop.maxThreadsPerBlock) {
        std::fprintf(stderr, "Invalid thread count: %d\n", cfg.threads);
        std::exit(1);
    }

    uint64_t total_threads = static_cast<uint64_t>(blocks) * static_cast<uint64_t>(cfg.threads);
    if (total_threads == 0) {
        std::fprintf(stderr, "Total threads must be > 0\n");
        std::exit(1);
    }
    if (total_threads > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(curandStatePhilox4_32_10_t))) {
        std::fprintf(stderr, "State allocation too large\n");
        std::exit(1);
    }
    if (total_elements > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        std::fprintf(stderr, "Output allocation too large\n");
        std::exit(1);
    }

    curandStatePhilox4_32_10_t* d_states = nullptr;
    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, total_threads * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(T)));

    philox_init<<<blocks, cfg.threads>>>(d_states, cfg.seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_fn(d_out, total_elements, d_states, blocks, cfg.threads);
    };

    float ms = benchmark_ms(cfg.warmup, cfg.iters, launch);

    uint64_t bytes_u64 = total_elements * static_cast<uint64_t>(sizeof(T));
    SizeDisplay size = format_bytes_decimal(bytes_u64);
    double bytes = static_cast<double>(bytes_u64);
    double gelem_s = static_cast<double>(total_elements) / (ms * 1e-3) / 1e9;
    double gib_s = bytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);

    std::printf("philox_bench\n");
    print_kv("device", "\"%s\"", prop.name);
    print_kv("device_id", "%d", cfg.device);
    print_kv("cc", "%d.%d", prop.major, prop.minor);
    print_kv("sms", "%d", prop.multiProcessorCount);
    print_kv("mode", "%s", label);
    print_kv("nostore", "%d", 0);
    print_kv("rows", "%lld", static_cast<long long>(cfg.rows));
    print_kv("cols", "%lld", static_cast<long long>(cfg.cols));
    print_kv("elems", "%llu", static_cast<unsigned long long>(total_elements));
    print_kv("size_bytes", "%llu", static_cast<unsigned long long>(bytes_u64));
    print_kv("size", "%.3f %s", size.value, size.unit);
    print_kv("threads", "%d", cfg.threads);
    print_kv("blocks", "%d", blocks);
    print_kv("blocks_per_sm", "%d", cfg.blocks_per_sm);
    print_kv("warmup", "%d", cfg.warmup);
    print_kv("iters", "%d", cfg.iters);
    print_kv("seed", "%llu", static_cast<unsigned long long>(cfg.seed));
    print_kv("scale", "%.6g", cfg.scale);
    print_kv("shift", "%.6g", cfg.shift);
    print_kv("ms", "%.3f", ms);
    print_kv("gelem_s", "%.3f", gelem_s);
    print_kv("gib_s", "%.3f", gib_s);

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_out));
}

template <typename T, typename LaunchFn>
static void run_bench_compute(
    const BenchConfig& cfg,
    const cudaDeviceProp& prop,
    uint64_t rng_iters,
    const char* label,
    LaunchFn&& launch_fn
) {
    int blocks = cfg.blocks;
    if (blocks <= 0) {
        if (cfg.blocks_per_sm <= 0) {
            std::fprintf(stderr, "blocks-per-sm must be > 0\n");
            std::exit(1);
        }
        if (prop.multiProcessorCount > 0 && cfg.blocks_per_sm > std::numeric_limits<int>::max() / prop.multiProcessorCount) {
            std::fprintf(stderr, "blocks-per-sm too large for this device\n");
            std::exit(1);
        }
        blocks = prop.multiProcessorCount * cfg.blocks_per_sm;
    }

    if (blocks <= 0 || blocks > prop.maxGridSize[0]) {
        std::fprintf(stderr, "Invalid block count: %d\n", blocks);
        std::exit(1);
    }
    if (cfg.threads <= 0 || cfg.threads > prop.maxThreadsPerBlock) {
        std::fprintf(stderr, "Invalid thread count: %d\n", cfg.threads);
        std::exit(1);
    }
    if (rng_iters == 0) {
        std::fprintf(stderr, "rng-iters must be > 0\n");
        std::exit(1);
    }

    uint64_t total_threads = static_cast<uint64_t>(blocks) * static_cast<uint64_t>(cfg.threads);
    if (total_threads == 0) {
        std::fprintf(stderr, "Total threads must be > 0\n");
        std::exit(1);
    }
    if (total_threads > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(curandStatePhilox4_32_10_t))) {
        std::fprintf(stderr, "State allocation too large\n");
        std::exit(1);
    }

    if (rng_iters > std::numeric_limits<uint64_t>::max() / 4) {
        std::fprintf(stderr, "rng-iters too large\n");
        std::exit(1);
    }
    uint64_t values_per_thread = rng_iters * 4;
    if (total_threads > std::numeric_limits<uint64_t>::max() / values_per_thread) {
        std::fprintf(stderr, "Total generated values overflow\n");
        std::exit(1);
    }

    uint64_t total_values = total_threads * values_per_thread;

    curandStatePhilox4_32_10_t* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, total_threads * sizeof(curandStatePhilox4_32_10_t)));

    philox_init<<<blocks, cfg.threads>>>(d_states, cfg.seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        launch_fn(d_states, rng_iters, blocks, cfg.threads);
    };

    float ms = benchmark_ms(cfg.warmup, cfg.iters, launch);

    if (total_values > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(sizeof(T))) {
        std::fprintf(stderr, "Total generated bytes overflow\n");
        std::exit(1);
    }
    uint64_t bytes_u64 = total_values * static_cast<uint64_t>(sizeof(T));
    SizeDisplay size = format_bytes_decimal(bytes_u64);
    double bytes = static_cast<double>(bytes_u64);
    double gelem_s = static_cast<double>(total_values) / (ms * 1e-3) / 1e9;
    double gib_s = bytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);

    std::printf("philox_bench\n");
    print_kv("device", "\"%s\"", prop.name);
    print_kv("device_id", "%d", cfg.device);
    print_kv("cc", "%d.%d", prop.major, prop.minor);
    print_kv("sms", "%d", prop.multiProcessorCount);
    print_kv("mode", "%s", label);
    print_kv("nostore", "%d", 1);
    print_kv("rng_iters", "%llu", static_cast<unsigned long long>(rng_iters));
    print_kv("values_per_thread", "%llu", static_cast<unsigned long long>(values_per_thread));
    print_kv("values", "%llu", static_cast<unsigned long long>(total_values));
    print_kv("size_bytes", "%llu", static_cast<unsigned long long>(bytes_u64));
    print_kv("size", "%.3f %s", size.value, size.unit);
    print_kv("threads", "%d", cfg.threads);
    print_kv("blocks", "%d", blocks);
    print_kv("blocks_per_sm", "%d", cfg.blocks_per_sm);
    print_kv("warmup", "%d", cfg.warmup);
    print_kv("iters", "%d", cfg.iters);
    print_kv("seed", "%llu", static_cast<unsigned long long>(cfg.seed));
    print_kv("ms", "%.3f", ms);
    print_kv("gelem_s", "%.3f", gelem_s);
    print_kv("gib_s", "%.3f", gib_s);

    CUDA_CHECK(cudaFree(d_states));
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
    if (cfg.iters < 1) {
        std::fprintf(stderr, "iters must be >= 1\n");
        return 1;
    }
    if (cfg.warmup < 0) {
        std::fprintf(stderr, "warmup must be >= 0\n");
        return 1;
    }
    if (cfg.device < 0) {
        std::fprintf(stderr, "device must be >= 0\n");
        return 1;
    }
    if (cfg.no_store) {
        if (cfg.rng_iters == 0) {
            std::fprintf(stderr, "rng-iters must be > 0\n");
            return 1;
        }
    } else {
        if (cfg.elements == 0 && (cfg.rows <= 0 || cfg.cols <= 0)) {
            std::fprintf(stderr, "rows and cols must be > 0\n");
            return 1;
        }
    }

    CUDA_CHECK(cudaSetDevice(cfg.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

    uint64_t total_elements = 0;
    if (!cfg.no_store) {
        total_elements = cfg.elements;
        if (total_elements == 0) {
            uint64_t rows = static_cast<uint64_t>(cfg.rows);
            uint64_t cols = static_cast<uint64_t>(cfg.cols);
            if (rows > 0 && cols > 0 && rows > std::numeric_limits<uint64_t>::max() / cols) {
                std::fprintf(stderr, "rows*cols overflows 64-bit size\n");
                return 1;
            }
            total_elements = rows * cols;
        } else {
            cfg.rows = static_cast<int64_t>(total_elements);
            cfg.cols = 1;
        }

        if (total_elements == 0) {
            std::fprintf(stderr, "elements must be > 0\n");
            return 1;
        }
    }

    const char* label = mode_label(cfg.mode);

    switch (cfg.mode) {
        case Mode::UNIFORM_F32: {
            if (cfg.no_store) {
                run_bench_compute<float>(cfg, prop, cfg.rng_iters, label, [&](curandStatePhilox4_32_10_t* states, uint64_t iters, int blocks, int threads) {
                    philox_compute_uniform_f32<<<blocks, threads>>>(states, iters);
                });
            } else {
                float scale = static_cast<float>(cfg.scale);
                float shift = static_cast<float>(cfg.shift);
                run_bench_store<float>(cfg, prop, total_elements, label, [&](float* out, uint64_t count, curandStatePhilox4_32_10_t* states, int blocks, int threads) {
                    philox_fill_uniform_f32<<<blocks, threads>>>(out, count, states, scale, shift);
                });
            }
            break;
        }
        case Mode::NORMAL_F32: {
            if (cfg.no_store) {
                run_bench_compute<float>(cfg, prop, cfg.rng_iters, label, [&](curandStatePhilox4_32_10_t* states, uint64_t iters, int blocks, int threads) {
                    philox_compute_normal_f32<<<blocks, threads>>>(states, iters);
                });
            } else {
                float scale = static_cast<float>(cfg.scale);
                float shift = static_cast<float>(cfg.shift);
                run_bench_store<float>(cfg, prop, total_elements, label, [&](float* out, uint64_t count, curandStatePhilox4_32_10_t* states, int blocks, int threads) {
                    philox_fill_normal_f32<<<blocks, threads>>>(out, count, states, scale, shift);
                });
            }
            break;
        }
        case Mode::U32: {
            if (cfg.no_store) {
                run_bench_compute<uint32_t>(cfg, prop, cfg.rng_iters, label, [&](curandStatePhilox4_32_10_t* states, uint64_t iters, int blocks, int threads) {
                    philox_compute_u32<<<blocks, threads>>>(states, iters);
                });
            } else {
                run_bench_store<uint32_t>(cfg, prop, total_elements, label, [&](uint32_t* out, uint64_t count, curandStatePhilox4_32_10_t* states, int blocks, int threads) {
                    philox_fill_u32<<<blocks, threads>>>(out, count, states);
                });
            }
            break;
        }
    }

    return 0;
}
