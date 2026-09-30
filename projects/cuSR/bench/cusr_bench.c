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
#define CUSR_AST_SASS_IMPLEMENTATION
#define CUSR_SASS_INSPECT_IMPLEMENTATION
#define CUSR_AST_SASS_PATCH_IMPLEMENTATION
#define CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION

#include <cusr_ast_sass_patch.h>
#include <cusr_ast_sass_cpu.h>
#include <cusr_settings.h>
#include <cusr_tile_static_mse_embedded.h>
#include <cusr_tile_static_mse_nvrtc.h>
#include <cusr_native_cuda_reference_gen.h>

#include <cuda.h>
#include <nvrtc.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUSR_BENCH_CTA_THREADS_DEFAULT 128u
#define CUSR_BENCH_TILE_ROWS_DEFAULT 64u
#define CUSR_BENCH_COLUMNS_DEFAULT 32u
#define CUSR_BENCH_SETTINGS_DEFAULT 1024u
#define CUSR_BENCH_MODULES_DEFAULT 1u
#define CUSR_BENCH_KERNELS_DEFAULT 32u
#define CUSR_BENCH_ASTS_DEFAULT 32u
#define CUSR_BENCH_AST_LENGTH_DEFAULT 16u
#define CUSR_BENCH_AST_STRIDE_DEFAULT 64u
#define CUSR_BENCH_FIRST_MARKER 0x7fc0ffeeu
#define CUSR_BENCH_RUN_ROWS_DEFAULT 1048576u
#define CUSR_BENCH_CHECK_ROWS_DEFAULT 0u
#define CUSR_BENCH_RUN_ITERS_DEFAULT 30u
#define CUSR_BENCH_RUNTIME_WARMUP_ITERS 2u
#define CUSR_BENCH_LEAF_WORDS_STRIDE 8u
#define CUSR_BENCH_OUTPUT_ATOL_DEFAULT 2.0e-3f
#define CUSR_BENCH_ERROR_RET(ans) do { int cusr_bench_result = (ans); return cusr_bench_result; } while (0)
#define CUSR_BENCH_CHECK_RET(ans) do { if (!(ans)) { CUSR_BENCH_ERROR_RET(0); } } while (0)

#ifndef CUSR_BENCH_SOURCE_DIR
#define CUSR_BENCH_SOURCE_DIR "."
#endif

typedef enum CusrBenchVariant {
    CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE = 0,
    CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE = 1
} CusrBenchVariant;

typedef enum CusrBenchAstMode {
    CUSR_BENCH_AST_MODE_SIMPLE = 0,
    CUSR_BENCH_AST_MODE_MUFU = 1,
    CUSR_BENCH_AST_MODE_MIXED = 2,
    CUSR_BENCH_AST_MODE_ALU_REDUCE = 3,
    CUSR_BENCH_AST_MODE_OFFSET_MINMAX = 4,
    CUSR_BENCH_AST_MODE_DEPTH3_ALU = 5,
    CUSR_BENCH_AST_MODE_DEPTH3_MUFU = 6
} CusrBenchAstMode;

typedef struct CusrBenchConfig {
    CusrBenchVariant variant;
    CusrBenchAstMode ast_mode;
    size_t num_modules;
    size_t kernels_per_module;
    size_t ast_capacity;
    size_t active_asts_per_kernel;
    size_t ast_length;
    size_t program_stride;
    uint32_t num_settings;
    uint32_t num_columns;
    uint32_t tile_rows;
    uint32_t cta_threads;
    uint32_t run_rows;
    uint32_t check_rows;
    uint32_t run_iters;
    float output_atol;
} CusrBenchConfig;

typedef struct CusrBenchCubin {
    unsigned char* data;
    size_t size;
} CusrBenchCubin;

typedef struct CusrBenchKernelSymbols {
    char** function_names;
    size_t count;
} CusrBenchKernelSymbols;

typedef struct CusrBenchLoadedModule {
    CUmodule module;
    CUfunction* functions;
} CusrBenchLoadedModule;

typedef struct CusrBenchDeviceRun {
    CUdeviceptr d_x;
    CUdeviceptr d_target;
    CUdeviceptr d_leaf_masks;
    CUdeviceptr d_leaf_words;
    CUdeviceptr* d_outputs;
    size_t output_bytes_per_module;
} CusrBenchDeviceRun;

static uint32_t cusr_bench_rng_state = 0x9e3779b9u;

static void
cusr_bench_set_runtime_defaults(CusrBenchConfig* config)
{
    config->variant = CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE;
    config->ast_mode = CUSR_BENCH_AST_MODE_DEPTH3_ALU;
    config->num_modules = CUSR_BENCH_MODULES_DEFAULT;
    config->kernels_per_module = CUSR_BENCH_KERNELS_DEFAULT;
    config->ast_capacity = CUSR_BENCH_ASTS_DEFAULT;
    config->active_asts_per_kernel = CUSR_BENCH_ASTS_DEFAULT;
    config->ast_length = CUSR_BENCH_AST_LENGTH_DEFAULT;
    config->program_stride = CUSR_BENCH_AST_STRIDE_DEFAULT;
    config->num_settings = CUSR_BENCH_SETTINGS_DEFAULT;
    config->num_columns = CUSR_BENCH_COLUMNS_DEFAULT;
    config->tile_rows = CUSR_BENCH_TILE_ROWS_DEFAULT;
    config->cta_threads = CUSR_BENCH_CTA_THREADS_DEFAULT;
    config->run_rows = CUSR_BENCH_RUN_ROWS_DEFAULT;
    config->check_rows = CUSR_BENCH_CHECK_ROWS_DEFAULT;
    config->run_iters = CUSR_BENCH_RUN_ITERS_DEFAULT;
    config->output_atol = CUSR_BENCH_OUTPUT_ATOL_DEFAULT;
}

static void
cusr_bench_set_patch_defaults(CusrBenchConfig* config)
{
    cusr_bench_set_runtime_defaults(config);

    config->variant = CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE;
    config->ast_mode = CUSR_BENCH_AST_MODE_ALU_REDUCE;
    config->num_modules = 32u;
    config->kernels_per_module = 8u;
    config->ast_capacity = CUSR_TILE_STATIC_MSE_EMBEDDED_AST_CAPACITY;
    config->active_asts_per_kernel = CUSR_TILE_STATIC_MSE_EMBEDDED_AST_CAPACITY;
    config->num_settings = 1u;
    config->run_rows = 1u;
    config->check_rows = 0u;
    config->run_iters = 1u;
}

static double
cusr_bench_now_seconds(void)
{
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static uint32_t
cusr_bench_rand_u32(void)
{
    cusr_bench_rng_state = cusr_bench_rng_state * 1664525u + 1013904223u;
    return cusr_bench_rng_state;
}

static float
cusr_bench_rand_unit_f32(void)
{
    return (float)(cusr_bench_rand_u32() >> 8) * (1.0f / 16777216.0f);
}

static float
cusr_bench_rand_f32(float scale)
{
    return (cusr_bench_rand_unit_f32() * 2.0f - 1.0f) * scale;
}

static uint32_t
cusr_bench_f32_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t
cusr_bench_hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint32_t
cusr_bench_reduce_leaf(uint32_t global_ast_idx, uint32_t term)
{
    const uint32_t leaf_count = 8u;
    const uint32_t offset = global_ast_idx % leaf_count;
    return (term + offset) % leaf_count;
}

static uint32_t
cusr_bench_reduce_op(uint32_t global_ast_idx, uint32_t term)
{
    return cusr_bench_hash32(global_ast_idx * 0xc2b2ae35u + term * 0x27d4eb2fu) & 3u;
}

static int
cusr_bench_parse_size(const char* text, size_t* value_ret)
{
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    *value_ret = (size_t)value;
    return 1;
}

static int
cusr_bench_parse_u32(const char* text, uint32_t* value_ret)
{
    size_t value = 0u;

    if (!cusr_bench_parse_size(text, &value) || value > UINT32_MAX) {
        return 0;
    }

    *value_ret = (uint32_t)value;
    return 1;
}

static int
cusr_bench_parse_float(const char* text, float* value_ret)
{
    char* end = NULL;
    double value;

    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    *value_ret = (float)value;
    return 1;
}

static const char*
cusr_bench_variant_name(CusrBenchVariant variant)
{
    if (variant == CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE) {
        return "native_cuda_reference";
    }
    if (variant == CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE) {
        return "tile_static_mse_template";
    }
    return "unknown";
}

static const char*
cusr_bench_ast_mode_name(CusrBenchAstMode mode)
{
    if (mode == CUSR_BENCH_AST_MODE_SIMPLE) {
        return "simple";
    }
    if (mode == CUSR_BENCH_AST_MODE_MUFU) {
        return "mufu";
    }
    if (mode == CUSR_BENCH_AST_MODE_ALU_REDUCE) {
        return "alu_reduce";
    }
    if (mode == CUSR_BENCH_AST_MODE_DEPTH3_ALU) {
        return "depth3_alu";
    }
    if (mode == CUSR_BENCH_AST_MODE_DEPTH3_MUFU) {
        return "depth3_mufu";
    }
    if (mode == CUSR_BENCH_AST_MODE_OFFSET_MINMAX) {
        return "offset_minmax";
    }
    return "mixed";
}

static const char*
cusr_bench_function_pattern(CusrBenchVariant variant)
{
    if (variant == CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE) {
        return "cusr_native_cuda_reference_f32_%03d";
    }
    return "cusr_tile_static_mse_f32_%03d";
}

static int
cusr_bench_variant_outputs_mse(CusrBenchVariant variant)
{
    (void)variant;
    return 1;
}

static int
cusr_bench_variant_needs_patch(CusrBenchVariant variant)
{
    return variant != CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE;
}

static unsigned
cusr_bench_native_reference_expr_mode(CusrBenchAstMode ast_mode)
{
    if (ast_mode == CUSR_BENCH_AST_MODE_ALU_REDUCE) {
        return CUSR_NATIVE_CUDA_REFERENCE_EXPR_MIXED_REDUCE;
    }
    if (ast_mode == CUSR_BENCH_AST_MODE_DEPTH3_ALU) {
        return CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_ALU;
    }
    if (ast_mode == CUSR_BENCH_AST_MODE_DEPTH3_MUFU) {
        return CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_MUFU;
    }

    return CUSR_NATIVE_CUDA_REFERENCE_EXPR_OFFSET_MINMAX;
}

static int
cusr_bench_parse_variant(const char* text, CusrBenchVariant* variant_ret)
{
    if (strcmp(text, "mse-template") == 0 || strcmp(text, "tile_static_mse_template") == 0) {
        *variant_ret = CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE;
        return 1;
    }
    if (strcmp(text, "native") == 0 || strcmp(text, "native_cuda_reference") == 0) {
        *variant_ret = CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE;
        return 1;
    }
    return 0;
}

static int
cusr_bench_parse_ast_mode(const char* text, CusrBenchAstMode* mode_ret)
{
    if (strcmp(text, "simple") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_SIMPLE;
        return 1;
    }
    if (strcmp(text, "mufu") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_MUFU;
        return 1;
    }
    if (strcmp(text, "mixed") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_MIXED;
        return 1;
    }
    if (strcmp(text, "alu-reduce") == 0 || strcmp(text, "alu_reduce") == 0 || strcmp(text, "reduce") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_ALU_REDUCE;
        return 1;
    }
    if (strcmp(text, "depth3-alu") == 0 || strcmp(text, "depth3_alu") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_DEPTH3_ALU;
        return 1;
    }
    if (strcmp(text, "depth3-mufu") == 0 || strcmp(text, "depth3_mufu") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_DEPTH3_MUFU;
        return 1;
    }
    if (strcmp(text, "offset-minmax") == 0 || strcmp(text, "offset_minmax") == 0) {
        *mode_ret = CUSR_BENCH_AST_MODE_OFFSET_MINMAX;
        return 1;
    }
    return 0;
}

static int
cusr_bench_apply_preset(CusrBenchConfig* config, const char* text)
{
    if (strcmp(text, "runtime") == 0 || strcmp(text, "run") == 0) {
        cusr_bench_set_runtime_defaults(config);
        return 1;
    }
    if (strcmp(text, "patch") == 0 || strcmp(text, "patch-rate") == 0 || strcmp(text, "patch_rate") == 0) {
        cusr_bench_set_patch_defaults(config);
        return 1;
    }
    return 0;
}

static void
cusr_bench_print_usage(const char* argv0)
{
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "\n"
        "options:\n"
        "  --preset runtime|patch\n"
        "  --variant tile_static_mse_template|native_cuda_reference\n"
        "  --ast-mode simple|mufu|mixed|alu-reduce|offset-minmax|depth3-alu|depth3-mufu\n"
        "  --modules N\n"
        "  --kernels-per-module N\n"
        "  --ast-capacity N\n"
        "  --asts-per-kernel N\n"
        "  --ast-length N\n"
        "  --program-stride N\n"
        "  --settings N\n"
        "  --columns N\n"
        "  --tile-rows 64|128|256\n"
        "  --cta-threads 64|128|256\n"
        "  --run-rows N\n"
        "  --check-rows N\n"
        "  --run-iters N\n"
        "  --atol X\n",
        argv0
    );
}

static int
cusr_bench_parse_args(CusrBenchConfig* config, int argc, char** argv)
{
    int i;
    int ast_capacity_set = 0;
    int active_asts_per_kernel_set = 0;

    cusr_bench_set_runtime_defaults(config);

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--preset") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --preset\n");
                return 0;
            }
            CUSR_BENCH_CHECK_RET(cusr_bench_apply_preset(config, argv[i + 1]));
        }
    }

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* value;

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            cusr_bench_print_usage(argv[0]);
            exit(0);
        }

        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", arg);
            return 0;
        }
        value = argv[++i];

        if (strcmp(arg, "--preset") == 0) {
            continue;
        } else if (strcmp(arg, "--variant") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_variant(value, &config->variant));
        } else if (strcmp(arg, "--ast-mode") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_ast_mode(value, &config->ast_mode));
        } else if (strcmp(arg, "--modules") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_size(value, &config->num_modules));
        } else if (strcmp(arg, "--kernels-per-module") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_size(value, &config->kernels_per_module));
        } else if (strcmp(arg, "--ast-capacity") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_size(value, &config->ast_capacity));
            ast_capacity_set = 1;
        } else if (strcmp(arg, "--asts-per-kernel") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_size(value, &config->active_asts_per_kernel));
            active_asts_per_kernel_set = 1;
        } else if (strcmp(arg, "--ast-length") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_size(value, &config->ast_length));
        } else if (strcmp(arg, "--program-stride") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_size(value, &config->program_stride));
        } else if (strcmp(arg, "--settings") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->num_settings));
        } else if (strcmp(arg, "--columns") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->num_columns));
        } else if (strcmp(arg, "--tile-rows") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->tile_rows));
        } else if (strcmp(arg, "--cta-threads") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->cta_threads));
        } else if (strcmp(arg, "--run-rows") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->run_rows));
        } else if (strcmp(arg, "--check-rows") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->check_rows));
        } else if (strcmp(arg, "--run-iters") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_u32(value, &config->run_iters));
        } else if (strcmp(arg, "--atol") == 0) {
            CUSR_BENCH_CHECK_RET(cusr_bench_parse_float(value, &config->output_atol));
        } else {
            fprintf(stderr, "unknown option %s\n", arg);
            return 0;
        }
    }

    if (active_asts_per_kernel_set && !ast_capacity_set) {
        config->ast_capacity = config->active_asts_per_kernel;
    } else if (ast_capacity_set && !active_asts_per_kernel_set) {
        config->active_asts_per_kernel = config->ast_capacity;
    }

    if (config->num_columns > 32u) {
        fprintf(stderr, "tile-static kernels support at most 32 columns\n");
        return 0;
    }

    if (config->num_modules == 0u ||
        config->kernels_per_module == 0u ||
        config->kernels_per_module >
            ((size_t)UINT32_MAX - (size_t)CUSR_BENCH_FIRST_MARKER) / CUSR_SASS_INSPECT_MARKER_STRIDE + 1u ||
        config->ast_capacity == 0u ||
        config->active_asts_per_kernel == 0u ||
        config->active_asts_per_kernel > config->ast_capacity ||
        config->num_settings == 0u ||
        config->num_columns == 0u ||
        (config->tile_rows != 64u && config->tile_rows != 128u && config->tile_rows != 256u) ||
        (config->cta_threads != 64u && config->cta_threads != 128u && config->cta_threads != 256u) ||
        config->ast_length < 2u ||
        config->program_stride < config->ast_length ||
        config->program_stride > CUSR_AST_MAX_PROGRAM_INSTRUCTIONS ||
        config->run_iters == 0u) {
        fprintf(stderr, "invalid configuration\n");
        return 0;
    }

    if (config->variant == CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE &&
        config->active_asts_per_kernel != config->ast_capacity) {
        fprintf(stderr, "native CUDA reference requires active ASTs to equal capacity\n");
        return 0;
    }

    if (config->variant == CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE &&
        config->ast_capacity != 8u && config->ast_capacity != 16u && config->ast_capacity != 32u) {
        fprintf(stderr, "template kernel capacity must be 8, 16, or 32\n");
        return 0;
    }

    if (config->variant == CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE &&
        config->ast_mode != CUSR_BENCH_AST_MODE_OFFSET_MINMAX &&
        config->ast_mode != CUSR_BENCH_AST_MODE_ALU_REDUCE &&
        config->ast_mode != CUSR_BENCH_AST_MODE_DEPTH3_ALU &&
        config->ast_mode != CUSR_BENCH_AST_MODE_DEPTH3_MUFU) {
        config->ast_mode = CUSR_BENCH_AST_MODE_OFFSET_MINMAX;
    }

    if (config->ast_mode == CUSR_BENCH_AST_MODE_OFFSET_MINMAX && config->program_stride < 32u) {
        fprintf(stderr, "offset-minmax requires --program-stride >= 32\n");
        return 0;
    }

    if (config->ast_mode == CUSR_BENCH_AST_MODE_DEPTH3_MUFU && config->program_stride < 64u) {
        fprintf(stderr, "depth3-mufu requires --program-stride >= 64\n");
        return 0;
    }

    return 1;
}

static int
cusr_bench_check_cuda(CUresult result, const char* label)
{
    if (result != CUDA_SUCCESS) {
        const char* name = "unknown";
        const char* description = "unknown";
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);
        fprintf(stderr, "%s failed: %s (%s)\n", label, name, description);
        return 0;
    }

    return 1;
}

static int
cusr_bench_check_nvrtc(nvrtcResult result, const char* label)
{
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "%s failed: %s\n", label, nvrtcGetErrorString(result));
        return 0;
    }

    return 1;
}

static int
cusr_bench_device_arch(char* arch, size_t arch_size, unsigned* sass_arch_ret, CUcontext* context_ret)
{
    CUdevice device = 0;
    int major = 0;
    int minor = 0;
    int written;

    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuInit(0u), "cuInit"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor"));
#if CUDA_VERSION >= 13000
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuCtxCreate(context_ret, NULL, 0u, device), "cuCtxCreate"));
#else
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuCtxCreate(context_ret, 0u, device), "cuCtxCreate"));
#endif

    written = snprintf(arch, arch_size, "--gpu-architecture=sm_%d%d", major, minor);
    CUSR_BENCH_CHECK_RET(written > 0 && (size_t)written < arch_size);
    *sass_arch_ret = (unsigned)(major * 10 + minor);
    return 1;
}

static size_t
cusr_bench_asts_per_module(const CusrBenchConfig* config)
{
    return config->kernels_per_module * config->active_asts_per_kernel;
}

static size_t
cusr_bench_total_asts(const CusrBenchConfig* config)
{
    return config->num_modules * cusr_bench_asts_per_module(config);
}

static int
cusr_bench_uses_embedded_cubin(const CusrBenchConfig* config, unsigned sass_arch)
{
    return
        config->variant == CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE &&
        (config->kernels_per_module == 8u ||
         config->kernels_per_module == 32u ||
         config->kernels_per_module == 128u) &&
        config->ast_capacity == CUSR_TILE_STATIC_MSE_EMBEDDED_AST_CAPACITY &&
        config->tile_rows == CUSR_TILE_STATIC_MSE_EMBEDDED_TILE_ROWS &&
        config->cta_threads == CUSR_TILE_STATIC_MSE_EMBEDDED_CTA_THREADS &&
        sass_arch == cusr_tile_static_mse_embedded_sass_arch();
}

static size_t
cusr_bench_num_tiles(const CusrBenchConfig* config, uint32_t rows)
{
    return ((size_t)rows + config->tile_rows - 1u) / config->tile_rows;
}

static size_t
cusr_bench_output_elems_per_module(const CusrBenchConfig* config, uint32_t rows)
{
    const size_t asts_per_module = cusr_bench_asts_per_module(config);
    const size_t settings = config->num_settings;
    const size_t tiles = cusr_bench_num_tiles(config, rows);

    (void)tiles;
    return asts_per_module * settings;
}

static size_t
cusr_bench_output_bytes_per_module(const CusrBenchConfig* config, uint32_t rows)
{
    return cusr_bench_output_elems_per_module(config, rows) * sizeof(float);
}

static int
cusr_bench_generate_source(const CusrBenchConfig* config, char** source_ret)
{
    size_t size = 0u;
    size_t written = 0u;
    char* source;

    *source_ret = NULL;

    CUSR_BENCH_CHECK_RET(config->variant == CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE);
    CUSR_BENCH_CHECK_RET(cusr_native_cuda_reference_gen_cuda_size(
        config->kernels_per_module,
        config->ast_capacity,
        config->tile_rows,
        config->cta_threads,
        cusr_bench_native_reference_expr_mode(config->ast_mode),
        &size) == CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);

    source = (char*)malloc(size);
    CUSR_BENCH_CHECK_RET(source != NULL);

    CUSR_BENCH_CHECK_RET(cusr_native_cuda_reference_gen_cuda(
        source,
        size,
        config->kernels_per_module,
        config->ast_capacity,
        config->tile_rows,
        config->cta_threads,
        cusr_bench_native_reference_expr_mode(config->ast_mode),
        &written) == CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);

    (void)written;
    *source_ret = source;
    return 1;
}

static char*
cusr_bench_copy_string(const char* source)
{
    const size_t size = strlen(source) + 1u;
    char* copy = (char*)malloc(size);

    if (copy != NULL) {
        memcpy(copy, source, size);
    }

    return copy;
}

static void
cusr_bench_destroy_kernel_symbols(CusrBenchKernelSymbols* symbols)
{
    size_t kernel_idx;

    if (symbols == NULL) {
        return;
    }

    for (kernel_idx = 0u; kernel_idx < symbols->count; ++kernel_idx) {
        free(symbols->function_names != NULL ? symbols->function_names[kernel_idx] : NULL);
    }

    free(symbols->function_names);
    memset(symbols, 0, sizeof(*symbols));
}

static int
cusr_bench_make_kernel_symbols(const CusrBenchConfig* config, CusrBenchKernelSymbols* symbols)
{
    size_t kernel_idx;

    memset(symbols, 0, sizeof(*symbols));
    symbols->count = config->kernels_per_module;
    symbols->function_names = (char**)calloc(symbols->count, sizeof(char*));
    if (symbols->function_names == NULL) {
        cusr_bench_destroy_kernel_symbols(symbols);
        CUSR_BENCH_ERROR_RET(0);
    }

    for (kernel_idx = 0u; kernel_idx < symbols->count; ++kernel_idx) {
        char buffer[256];
        int written;

        written = snprintf(buffer, sizeof(buffer), cusr_bench_function_pattern(config->variant), (int)kernel_idx);
        if (written <= 0 || (size_t)written >= sizeof(buffer)) {
            cusr_bench_destroy_kernel_symbols(symbols);
            CUSR_BENCH_ERROR_RET(0);
        }
        symbols->function_names[kernel_idx] = cusr_bench_copy_string(buffer);
        if (symbols->function_names[kernel_idx] == NULL) {
            cusr_bench_destroy_kernel_symbols(symbols);
            CUSR_BENCH_ERROR_RET(0);
        }
    }

    return 1;
}

static int
cusr_bench_compile_source_impl(
    const char* source,
    const char* arch_option,
    CusrBenchCubin* cubin_ret,
    nvrtcProgram* program_ret)
{
    char include_kernels[1024];
    char include_src[1024];
    const char* options[8];
    nvrtcResult compile_result;
    size_t log_size = 0u;
    size_t cubin_size = 0u;
    unsigned char* cubin;

    snprintf(include_kernels, sizeof(include_kernels), "-I%s/kernels", CUSR_BENCH_SOURCE_DIR);
    snprintf(include_src, sizeof(include_src), "-I%s/src", CUSR_BENCH_SOURCE_DIR);

    options[0] = arch_option;
    options[1] = "--std=c++11";
    options[2] = "--device-as-default-execution-space";
    options[3] = include_kernels;
    options[4] = include_src;
    options[5] = "--ptxas-options=-v";
    options[6] = "--ptxas-options=-warn-spills";
    options[7] = "--ptxas-options=-Werror";

    CUSR_BENCH_CHECK_RET(cusr_bench_check_nvrtc(nvrtcCreateProgram(program_ret, source, "cusr_bench_generated.cu", 0, NULL, NULL), "nvrtcCreateProgram"));
    compile_result = nvrtcCompileProgram(*program_ret, 8, options);

    if (nvrtcGetProgramLogSize(*program_ret, &log_size) == NVRTC_SUCCESS && log_size > 1u) {
        char* log = (char*)malloc(log_size);
        if (log != NULL) {
            if (nvrtcGetProgramLog(*program_ret, log) == NVRTC_SUCCESS && compile_result != NVRTC_SUCCESS) {
                fprintf(stderr, "%s\n", log);
            }
            free(log);
        }
    }

    CUSR_BENCH_CHECK_RET(cusr_bench_check_nvrtc(compile_result, "nvrtcCompileProgram"));

    CUSR_BENCH_CHECK_RET(cusr_bench_check_nvrtc(nvrtcGetCUBINSize(*program_ret, &cubin_size), "nvrtcGetCUBINSize"));

    cubin = (unsigned char*)malloc(cubin_size);
    CUSR_BENCH_CHECK_RET(cubin != NULL);
    CUSR_BENCH_CHECK_RET(cusr_bench_check_nvrtc(nvrtcGetCUBIN(*program_ret, (char*)cubin), "nvrtcGetCUBIN"));

    cubin_ret->data = cubin;
    cubin_ret->size = cubin_size;
    return 1;
}

static int
cusr_bench_compile_source(const char* source, const char* arch_option, CusrBenchCubin* cubin_ret)
{
    nvrtcProgram program = NULL;
    int ok;

    cubin_ret->data = NULL;
    cubin_ret->size = 0u;

    ok = cusr_bench_compile_source_impl(source, arch_option, cubin_ret, &program);
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }

    return ok;
}

static void
cusr_bench_print_template_nvrtc_log(const CusrTileStaticMseNvrtcHandle* handle)
{
    size_t log_size = 0u;
    char* log;

    if (handle == NULL ||
        cusr_tile_static_mse_nvrtc_log_size(handle, &log_size) != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS ||
        log_size <= 1u) {
        return;
    }

    log = (char*)malloc(log_size);
    if (log != NULL) {
        if (cusr_tile_static_mse_nvrtc_get_log(handle, log, log_size) == CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
            fprintf(stderr, "%s\n", log);
        }
        free(log);
    }
}

static int
cusr_bench_compile_template_impl(
    const CusrBenchConfig* config,
    unsigned device_sass_arch,
    CusrBenchCubin* cubin_ret,
    CusrTileStaticMseNvrtcHandle** handle_ret)
{
    CusrTileStaticMseNvrtcResult result;
    size_t cubin_size = 0u;
    unsigned char* cubin;

    result = cusr_tile_static_mse_nvrtc_create(
        config->kernels_per_module,
        config->ast_capacity,
        config->tile_rows,
        config->cta_threads,
        device_sass_arch / 10u,
        device_sass_arch % 10u,
        handle_ret);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
        cusr_bench_print_template_nvrtc_log(*handle_ret);
        fprintf(stderr, "cusr_tile_static_mse_nvrtc_create failed: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_BENCH_ERROR_RET(0);
    }

    result = cusr_tile_static_mse_nvrtc_cubin_size(*handle_ret, &cubin_size);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_mse_nvrtc_cubin_size failed: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_BENCH_ERROR_RET(0);
    }

    cubin = (unsigned char*)malloc(cubin_size);
    CUSR_BENCH_CHECK_RET(cubin != NULL);
    cubin_ret->data = cubin;
    cubin_ret->size = cubin_size;

    result = cusr_tile_static_mse_nvrtc_get_cubin(*handle_ret, cubin, cubin_size);
    if (result != CUSR_TILE_STATIC_MSE_NVRTC_SUCCESS) {
        fprintf(stderr, "cusr_tile_static_mse_nvrtc_get_cubin failed: %s\n", cusr_tile_static_mse_nvrtc_result_to_string(result));
        CUSR_BENCH_ERROR_RET(0);
    }

    CUSR_BENCH_ERROR_RET(1);
}

static int
cusr_bench_compile_template(const CusrBenchConfig* config, unsigned device_sass_arch, CusrBenchCubin* cubin_ret)
{
    CusrTileStaticMseNvrtcHandle* handle = NULL;
    int ok = cusr_bench_compile_template_impl(config, device_sass_arch, cubin_ret, &handle);

    cusr_tile_static_mse_nvrtc_destroy(handle);
    if (!ok) {
        free(cubin_ret->data);
        cubin_ret->data = NULL;
        cubin_ret->size = 0u;
    }
    return ok;
}

static int
cusr_bench_copy_embedded_cubin(const CusrBenchConfig* config, CusrBenchCubin* cubin_ret)
{
    size_t cubin_size = 0u;
    const unsigned char* embedded = cusr_tile_static_mse_embedded_cubin(config->kernels_per_module, &cubin_size);

    CUSR_BENCH_CHECK_RET(embedded != NULL && cubin_size != 0u);
    cubin_ret->data = (unsigned char*)malloc(cubin_size);
    CUSR_BENCH_CHECK_RET(cubin_ret->data != NULL);
    memcpy(cubin_ret->data, embedded, cubin_size);
    cubin_ret->size = cubin_size;
    return 1;
}

static void
cusr_bench_emit_leaf(CusrAstInstruction* program, size_t* count)
{
    const uint32_t input_idx = cusr_bench_rand_u32() & 7u;
    const uint32_t use_constant = (cusr_bench_rand_u32() & 7u) == 0u;

    if (use_constant) {
        program[(*count)++] = cusr_ast_encode_constant_bits(cusr_bench_f32_bits(cusr_bench_rand_f32(0.25f)));
    } else {
        program[(*count)++] = cusr_ast_encode_input(input_idx);
    }
}

static void
cusr_bench_emit_simple_unary(CusrAstInstruction* program, size_t* count)
{
    if ((cusr_bench_rand_u32() & 1u) == 0u) {
        program[(*count)++] = cusr_ast_encode_neg;
    } else {
        program[(*count)++] = cusr_ast_encode_abs;
    }
}

static void
cusr_bench_emit_mufu_unary(CusrAstInstruction* program, size_t* count)
{
    switch (cusr_bench_rand_u32() % 4u) {
        case 0u:
            program[(*count)++] = cusr_ast_encode_sin;
            break;
        case 1u:
            program[(*count)++] = cusr_ast_encode_cos;
            break;
        case 2u:
            program[(*count)++] = cusr_ast_encode_ex2;
            break;
        default:
            program[(*count)++] = cusr_ast_encode_abs;
            program[(*count)++] = cusr_ast_encode_constant_bits(cusr_bench_f32_bits(0.25f));
            program[(*count)++] = cusr_ast_encode_add;
            program[(*count)++] = cusr_ast_encode_rsqrt;
            break;
    }
}

static void
cusr_bench_emit_binary(CusrAstInstruction* program, size_t* count)
{
    switch (cusr_bench_rand_u32() % 5u) {
        case 0u:
            program[(*count)++] = cusr_ast_encode_add;
            break;
        case 1u:
            program[(*count)++] = cusr_ast_encode_sub;
            break;
        case 2u:
            program[(*count)++] = cusr_ast_encode_mul;
            break;
        case 3u:
            program[(*count)++] = cusr_ast_encode_min;
            break;
        default:
            program[(*count)++] = cusr_ast_encode_max;
            break;
    }
}

static void
cusr_bench_emit_reduce_binary(CusrAstInstruction* program, size_t* count, uint32_t op)
{
    switch (op & 3u) {
        case 0u:
            program[(*count)++] = cusr_ast_encode_add;
            break;
        case 1u:
            program[(*count)++] = cusr_ast_encode_mul;
            break;
        case 2u:
            program[(*count)++] = cusr_ast_encode_min;
            break;
        default:
            program[(*count)++] = cusr_ast_encode_max;
            break;
    }
}

static uint32_t
cusr_bench_depth3_unary_op(uint32_t global_ast_idx, uint32_t term)
{
    return cusr_bench_hash32(global_ast_idx * 0x9e3779b9u + term * 0x85ebca6bu) & 3u;
}

static uint32_t
cusr_bench_depth3_binary_op(uint32_t global_ast_idx, uint32_t node_idx)
{
    return cusr_bench_hash32(global_ast_idx * 0xc2b2ae35u + node_idx * 0x27d4eb2fu) & 3u;
}

static void
cusr_bench_emit_depth3_leaf(
    CusrAstInstruction* program,
    size_t* count,
    size_t global_ast_idx,
    uint32_t term,
    int use_mufu)
{
    const uint32_t leaf = cusr_bench_reduce_leaf((uint32_t)global_ast_idx, term);

    program[(*count)++] = cusr_ast_encode_input(leaf);

    if (!use_mufu) {
        return;
    }

    switch (cusr_bench_depth3_unary_op((uint32_t)global_ast_idx, term)) {
        case 0u:
            program[(*count)++] = cusr_ast_encode_sin;
            break;
        case 1u:
            program[(*count)++] = cusr_ast_encode_cos;
            break;
        case 2u:
            program[(*count)++] = cusr_ast_encode_constant_bits(cusr_bench_f32_bits(0.125f));
            program[(*count)++] = cusr_ast_encode_mul;
            program[(*count)++] = cusr_ast_encode_ex2;
            break;
        default:
            program[(*count)++] = cusr_ast_encode_abs;
            program[(*count)++] = cusr_ast_encode_constant_bits(cusr_bench_f32_bits(0.25f));
            program[(*count)++] = cusr_ast_encode_add;
            program[(*count)++] = cusr_ast_encode_rsqrt;
            break;
    }
}

static void
cusr_bench_make_depth3_program(
    CusrAstInstruction* program,
    size_t program_stride,
    size_t global_ast_idx,
    int use_mufu)
{
    size_t count = 0u;
    uint32_t term = 0u;
    uint32_t node = 0u;

    memset(program, 0, program_stride * sizeof(*program));

    if (program_stride < (use_mufu ? 64u : 16u)) {
        return;
    }

    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));

    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));
    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));

    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));

    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_depth3_leaf(program, &count, global_ast_idx, term++, use_mufu);
    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));
    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));

    cusr_bench_emit_reduce_binary(program, &count, cusr_bench_depth3_binary_op((uint32_t)global_ast_idx, node++));
    program[count++] = cusr_ast_encode_return;
}

static void
cusr_bench_make_alu_reduce_program(CusrAstInstruction* program, size_t program_stride, size_t global_ast_idx)
{
    size_t count = 0u;
    uint32_t term;

    memset(program, 0, program_stride * sizeof(*program));

    if (program_stride < 16u) {
        return;
    }

    program[count++] = cusr_ast_encode_input(cusr_bench_reduce_leaf((uint32_t)global_ast_idx, 0u));
    for (term = 1u; term < 8u; ++term) {
        program[count++] = cusr_ast_encode_input(cusr_bench_reduce_leaf((uint32_t)global_ast_idx, term));
        cusr_bench_emit_reduce_binary(program, &count, cusr_bench_reduce_op((uint32_t)global_ast_idx, term));
    }
    program[count++] = cusr_ast_encode_return;
}

static void
cusr_bench_emit_offset_minmax_leaf(CusrAstInstruction* program, size_t* count, uint32_t input_idx, float offset)
{
    program[(*count)++] = cusr_ast_encode_input(input_idx);
    program[(*count)++] = cusr_ast_encode_constant_bits(cusr_bench_f32_bits(offset));
    program[(*count)++] = cusr_ast_encode_add;
}

static void
cusr_bench_make_offset_minmax_program(CusrAstInstruction* program, size_t program_stride, size_t global_ast_idx)
{
    size_t count = 0u;
    const float offset = (float)global_ast_idx;

    memset(program, 0, program_stride * sizeof(*program));

    if (program_stride < 32u) {
        return;
    }

    cusr_bench_emit_offset_minmax_leaf(program, &count, 0u, offset);
    cusr_bench_emit_offset_minmax_leaf(program, &count, 1u, offset);
    program[count++] = cusr_ast_encode_min;

    cusr_bench_emit_offset_minmax_leaf(program, &count, 2u, offset);
    cusr_bench_emit_offset_minmax_leaf(program, &count, 3u, offset);
    program[count++] = cusr_ast_encode_max;
    program[count++] = cusr_ast_encode_min;

    cusr_bench_emit_offset_minmax_leaf(program, &count, 4u, offset);
    cusr_bench_emit_offset_minmax_leaf(program, &count, 5u, offset);
    program[count++] = cusr_ast_encode_min;

    cusr_bench_emit_offset_minmax_leaf(program, &count, 6u, offset);
    cusr_bench_emit_offset_minmax_leaf(program, &count, 7u, offset);
    program[count++] = cusr_ast_encode_max;
    program[count++] = cusr_ast_encode_max;
    program[count++] = cusr_ast_encode_min;
    program[count++] = cusr_ast_encode_return;
}

static void
cusr_bench_make_program(CusrAstInstruction* program, size_t program_stride, size_t ast_length, CusrBenchAstMode mode, size_t global_ast_idx)
{
    size_t count = 0u;
    size_t stack_depth = 0u;

    if (mode == CUSR_BENCH_AST_MODE_ALU_REDUCE) {
        cusr_bench_make_alu_reduce_program(program, program_stride, global_ast_idx);
        return;
    }

    if (mode == CUSR_BENCH_AST_MODE_DEPTH3_ALU) {
        cusr_bench_make_depth3_program(program, program_stride, global_ast_idx, 0);
        return;
    }

    if (mode == CUSR_BENCH_AST_MODE_DEPTH3_MUFU) {
        cusr_bench_make_depth3_program(program, program_stride, global_ast_idx, 1);
        return;
    }

    if (mode == CUSR_BENCH_AST_MODE_OFFSET_MINMAX) {
        cusr_bench_make_offset_minmax_program(program, program_stride, global_ast_idx);
        return;
    }

    memset(program, 0, program_stride * sizeof(*program));

    while (count + 1u < ast_length) {
        const size_t remaining_before_return = ast_length - 1u - count;
        const int must_reduce = stack_depth > 1u && remaining_before_return <= stack_depth - 1u;
        const int can_leaf = stack_depth < 16u;
        const int can_binary = stack_depth >= 2u;
        const int can_unary = stack_depth >= 1u;
        const uint32_t choice = cusr_bench_rand_u32() % 100u;

        if (must_reduce && can_binary) {
            cusr_bench_emit_binary(program, &count);
            stack_depth -= 1u;
        } else if ((stack_depth == 0u || (!can_binary && can_leaf) || choice < 35u) && can_leaf) {
            cusr_bench_emit_leaf(program, &count);
            stack_depth += 1u;
        } else if (can_unary && choice < 65u) {
            if (mode == CUSR_BENCH_AST_MODE_MUFU || (mode == CUSR_BENCH_AST_MODE_MIXED && (cusr_bench_rand_u32() & 1u) != 0u)) {
                if (count + 5u <= ast_length) {
                    cusr_bench_emit_mufu_unary(program, &count);
                } else {
                    cusr_bench_emit_simple_unary(program, &count);
                }
            } else {
                cusr_bench_emit_simple_unary(program, &count);
            }
        } else if (can_binary) {
            cusr_bench_emit_binary(program, &count);
            stack_depth -= 1u;
        } else {
            cusr_bench_emit_leaf(program, &count);
            stack_depth += 1u;
        }
    }

    while (stack_depth > 1u && count + 1u < program_stride) {
        cusr_bench_emit_binary(program, &count);
        stack_depth -= 1u;
    }

    if (stack_depth == 0u && count + 1u < program_stride) {
        cusr_bench_emit_leaf(program, &count);
        stack_depth = 1u;
    }

    program[count++] = cusr_ast_encode_return;
}

static void
cusr_bench_make_programs(const CusrBenchConfig* config, CusrAstInstruction* programs)
{
    const size_t asts_per_module = cusr_bench_asts_per_module(config);
    const size_t total_asts = cusr_bench_total_asts(config);
    size_t i;

    for (i = 0u; i < total_asts; ++i) {
        const size_t ast_idx = config->variant == CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE ? i % asts_per_module : i;

        cusr_bench_make_program(
            programs + i * config->program_stride,
            config->program_stride,
            config->ast_length,
            config->ast_mode,
            ast_idx
        );
    }
}

static void
cusr_bench_make_program_ptrs(
    const CusrBenchConfig* config,
    const CusrAstInstruction* programs,
    const CusrAstInstruction** program_ptrs)
{
    const size_t total_asts = cusr_bench_total_asts(config);
    size_t i;

    for (i = 0u; i < total_asts; ++i) {
        program_ptrs[i] = programs + i * config->program_stride;
    }
}

static void
cusr_bench_fill_data(float* x, float* target, uint32_t rows, uint32_t columns, uint32_t leading_dim)
{
    uint32_t column;
    uint32_t row;

    for (column = 0u; column < columns; ++column) {
        for (row = 0u; row < rows; ++row) {
            x[(size_t)column * (size_t)leading_dim + row] =
                0.05f * (float)(column + 1u) + 0.00001f * (float)row + cusr_bench_rand_f32(0.125f);
        }
    }

    if (target != NULL) {
        for (row = 0u; row < rows; ++row) {
            const float x0 = x[row];
            const float x1 = columns > 1u ? x[(size_t)leading_dim + row] : x0;
            target[row] = 0.5f + 0.25f * x0 - 0.125f * x1;
        }
    }
}

static void
cusr_bench_fill_settings(CusrSettingF32* settings, uint32_t num_settings, uint32_t num_columns)
{
    uint32_t setting_idx;

    for (setting_idx = 0u; setting_idx < num_settings; ++setting_idx) {
        uint32_t input_idx;

        settings[setting_idx].column_mask = 0xffu;
        for (input_idx = 0u; input_idx < CUSR_NUM_INPUTS; ++input_idx) {
            settings[setting_idx].column_indices[input_idx] = (setting_idx * 3u + input_idx * 5u) % num_columns;
            settings[setting_idx].constants[input_idx] = cusr_bench_rand_f32(0.25f);
        }
    }
}

static void
cusr_bench_fill_leaf_settings(
    uint8_t* leaf_masks,
    uint32_t* leaf_words,
    size_t num_kernels,
    uint32_t num_settings,
    const CusrSettingF32* settings)
{
    size_t kernel_idx;

    for (kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
        uint32_t setting_idx;

        for (setting_idx = 0u; setting_idx < num_settings; ++setting_idx) {
            uint32_t input_idx;
            const size_t offset = kernel_idx * num_settings + setting_idx;

            leaf_masks[offset] = (uint8_t)settings[setting_idx].column_mask;
            for (input_idx = 0u; input_idx < CUSR_NUM_INPUTS; ++input_idx) {
                leaf_words[offset * CUSR_BENCH_LEAF_WORDS_STRIDE + input_idx] =
                    settings[setting_idx].column_indices[input_idx];
            }
        }
    }
}

static int
cusr_bench_inspect_base_cubin(
    const CusrBenchConfig* config,
    const CusrBenchCubin* base_cubin,
    const CusrBenchKernelSymbols* symbols,
    CusrSassInspectHandle* inspect_ret,
    void** workspace_ret)
{
    size_t workspace_size = 0u;
    CusrSassInspectResult inspect_result;

    inspect_result = cusr_sass_inspect_workspace_size(
        base_cubin->data,
        base_cubin->size,
        (const char* const*)symbols->function_names,
        config->kernels_per_module,
        config->ast_capacity,
        CUSR_BENCH_FIRST_MARKER,
        1u,
        &workspace_size
    );

    if (inspect_result != CUSR_SASS_INSPECT_SUCCESS) {
        fprintf(stderr, "inspect workspace failed: %s\n", cusr_sass_inspect_result_to_string(inspect_result));
        CUSR_BENCH_ERROR_RET(0);
    }

    *workspace_ret = malloc(workspace_size == 0u ? 1u : workspace_size);
    CUSR_BENCH_CHECK_RET(*workspace_ret != NULL);

    inspect_result = cusr_sass_inspect(
        base_cubin->data,
        base_cubin->size,
        (const char* const*)symbols->function_names,
        config->kernels_per_module,
        config->ast_capacity,
        CUSR_BENCH_FIRST_MARKER,
        1u,
        *workspace_ret,
        workspace_size,
        inspect_ret
    );

    if (inspect_result != CUSR_SASS_INSPECT_SUCCESS) {
        fprintf(stderr, "inspect failed: %s\n", cusr_sass_inspect_result_to_string(inspect_result));
        CUSR_BENCH_ERROR_RET(0);
    }

    return 1;
}

static int
cusr_bench_copy_base_cubins(const CusrBenchConfig* config, const CusrBenchCubin* base_cubin, CusrBenchCubin* cubins)
{
    size_t module_idx;

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        cubins[module_idx].data = (unsigned char*)malloc(base_cubin->size);
        CUSR_BENCH_CHECK_RET(cubins[module_idx].data != NULL);
        memcpy(cubins[module_idx].data, base_cubin->data, base_cubin->size);
        cubins[module_idx].size = base_cubin->size;
    }

    return 1;
}

static int
cusr_bench_patch_cubins(
    const CusrBenchConfig* config,
    const CusrSassInspectHandle* inspect,
    CusrBenchCubin* cubins,
    const CusrAstInstruction* const* program_ptrs,
    CusrAstSassPatchStats* stats_ret)
{
    const size_t asts_per_module = cusr_bench_asts_per_module(config);
    const uint32_t capability_major = inspect->sass_arch / 10u;
    const uint32_t capability_minor = inspect->sass_arch % 10u;
    size_t module_idx;

    if (stats_ret != NULL) {
        memset(stats_ret, 0, sizeof(*stats_ret));
    }

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        CusrAstSassPatchStats module_stats;
        CusrAstSassPatchResult patch_result = cusr_ast_sass_patch_cubin(
            inspect,
            capability_major,
            capability_minor,
            CUSR_AST_SASS_PATCH_EPILOGUE_SSE,
            NULL,
            0u,
            program_ptrs + module_idx * asts_per_module,
            asts_per_module,
            cubins[module_idx].data,
            cubins[module_idx].size,
            &module_stats
        );

        if (patch_result != CUSR_AST_SASS_PATCH_SUCCESS) {
            fprintf(stderr, "patch failed for module %zu: %s\n", module_idx, cusr_ast_sass_patch_result_to_string(patch_result));
            CUSR_BENCH_ERROR_RET(0);
        }

        if (stats_ret != NULL) {
            stats_ret->sites_patched += module_stats.sites_patched;
            stats_ret->asts_patched += module_stats.asts_patched;
            stats_ret->sass_instructions_written += module_stats.sass_instructions_written;
            stats_ret->sass_bytes_written += module_stats.sass_bytes_written;
            if (module_stats.max_original_register_count > stats_ret->max_original_register_count) {
                stats_ret->max_original_register_count = module_stats.max_original_register_count;
            }
            if (module_stats.max_expanded_register_count > stats_ret->max_expanded_register_count) {
                stats_ret->max_expanded_register_count = module_stats.max_expanded_register_count;
            }
            if (module_stats.max_patched_register_count > stats_ret->max_patched_register_count) {
                stats_ret->max_patched_register_count = module_stats.max_patched_register_count;
            }
        }
    }

    return 1;
}

static void
cusr_bench_unload_modules(const CusrBenchConfig* config, CusrBenchLoadedModule* modules)
{
    size_t module_idx;

    if (modules == NULL) {
        return;
    }

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        free(modules[module_idx].functions);
        modules[module_idx].functions = NULL;
        if (modules[module_idx].module != NULL) {
            (void)cuModuleUnload(modules[module_idx].module);
            modules[module_idx].module = NULL;
        }
    }
}

static int
cusr_bench_alloc_device_run(
    const CusrBenchConfig* config,
    uint32_t rows,
    const float* x,
    const float* target,
    const CusrSettingF32* settings,
    const uint8_t* leaf_masks,
    const uint32_t* leaf_words,
    CusrBenchDeviceRun* run_ret)
{
    const size_t x_bytes = (size_t)config->num_columns * (size_t)rows * sizeof(float);
    const size_t target_bytes = (size_t)rows * sizeof(float);
    const size_t leaf_masks_bytes = config->kernels_per_module * (size_t)config->num_settings * sizeof(uint8_t);
    const size_t leaf_words_bytes = config->kernels_per_module * (size_t)config->num_settings * CUSR_BENCH_LEAF_WORDS_STRIDE * sizeof(uint32_t);
    size_t module_idx;

    memset(run_ret, 0, sizeof(*run_ret));
    run_ret->output_bytes_per_module = cusr_bench_output_bytes_per_module(config, rows);
    run_ret->d_outputs = (CUdeviceptr*)calloc(config->num_modules, sizeof(CUdeviceptr));
    CUSR_BENCH_CHECK_RET(run_ret->d_outputs != NULL);

    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemAlloc(&run_ret->d_x, x_bytes), "cuMemAlloc x"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemcpyHtoD(run_ret->d_x, x, x_bytes), "cuMemcpyHtoD x"));

    if (cusr_bench_variant_outputs_mse(config->variant)) {
        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemAlloc(&run_ret->d_target, target_bytes), "cuMemAlloc target"));
        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemcpyHtoD(run_ret->d_target, target, target_bytes), "cuMemcpyHtoD target"));
    }

    (void)settings;
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemAlloc(&run_ret->d_leaf_masks, leaf_masks_bytes), "cuMemAlloc leaf masks"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemAlloc(&run_ret->d_leaf_words, leaf_words_bytes), "cuMemAlloc leaf words"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemcpyHtoD(run_ret->d_leaf_masks, leaf_masks, leaf_masks_bytes), "cuMemcpyHtoD leaf masks"));
    CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemcpyHtoD(run_ret->d_leaf_words, leaf_words, leaf_words_bytes), "cuMemcpyHtoD leaf words"));

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemAlloc(run_ret->d_outputs + module_idx, run_ret->output_bytes_per_module), "cuMemAlloc output"));
        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemsetD8(run_ret->d_outputs[module_idx], 0u, run_ret->output_bytes_per_module), "cuMemsetD8 output"));
    }

    return 1;
}

static void
cusr_bench_free_device_run(const CusrBenchConfig* config, CusrBenchDeviceRun* run)
{
    size_t module_idx;

    if (run == NULL) {
        return;
    }

    if (run->d_outputs != NULL) {
        for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
            if (run->d_outputs[module_idx] != 0u) {
                (void)cuMemFree(run->d_outputs[module_idx]);
            }
        }
        free(run->d_outputs);
    }

    if (run->d_leaf_words != 0u) (void)cuMemFree(run->d_leaf_words);
    if (run->d_leaf_masks != 0u) (void)cuMemFree(run->d_leaf_masks);
    if (run->d_target != 0u) (void)cuMemFree(run->d_target);
    if (run->d_x != 0u) (void)cuMemFree(run->d_x);
    memset(run, 0, sizeof(*run));
}

static int
cusr_bench_launch_one(
    const CusrBenchConfig* config,
    const CusrBenchLoadedModule* modules,
    const CusrBenchDeviceRun* run,
    uint32_t rows,
    size_t module_idx,
    size_t kernel_idx,
    CUstream stream)
{
    const unsigned int blocks = (unsigned int)cusr_bench_num_tiles(config, rows);
    const size_t num_rows = rows;
    const uint32_t num_columns = config->num_columns;
    const size_t leading_dim = rows;
    const uint32_t num_settings = config->num_settings;
    const uint32_t num_asts = (uint32_t)config->active_asts_per_kernel;
    const size_t leaf_words_stride = CUSR_BENCH_LEAF_WORDS_STRIDE;
    const uint32_t shared_stride = num_columns | 1u;
    const unsigned int mse_shared_bytes = (config->tile_rows * shared_stride + config->tile_rows) * (unsigned int)sizeof(float);
    const unsigned int mse_threads = config->cta_threads;
    CUfunction function = modules[module_idx].functions[kernel_idx];

    {
        const size_t per_kernel_output_count = config->active_asts_per_kernel * (size_t)num_settings;
        const size_t per_kernel_output_bytes = per_kernel_output_count * sizeof(float);
        const size_t settings_offset = kernel_idx * (size_t)num_settings;
        CUdeviceptr leaf_masks = run->d_leaf_masks + settings_offset * sizeof(uint8_t);
        CUdeviceptr leaf_words = run->d_leaf_words + settings_offset * CUSR_BENCH_LEAF_WORDS_STRIDE * sizeof(uint32_t);
        CUdeviceptr output = run->d_outputs[module_idx] + kernel_idx * per_kernel_output_bytes;
        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuMemsetD8Async(output, 0u, per_kernel_output_bytes, stream), "cuMemsetD8Async SSE"));
        if (config->variant != CUSR_BENCH_VARIANT_NATIVE_CUDA_REFERENCE) {
            void* args[] = {
                (void*)&run->d_x,
                (void*)&run->d_target,
                (void*)&num_rows,
                (void*)&num_columns,
                (void*)&leading_dim,
                (void*)&leaf_masks,
                (void*)&leaf_words,
                (void*)&leaf_words_stride,
                (void*)&num_settings,
                (void*)&num_asts,
                (void*)&output
            };

            CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuLaunchKernel(function, blocks, 1u, 1u, mse_threads, 1u, 1u, mse_shared_bytes, stream, args, NULL), "cuLaunchKernel tile_static_mse"));
        } else {
            void* args[] = {
                (void*)&run->d_x,
                (void*)&run->d_target,
                (void*)&num_rows,
                (void*)&num_columns,
                (void*)&leading_dim,
                (void*)&leaf_masks,
                (void*)&leaf_words,
                (void*)&leaf_words_stride,
                (void*)&num_settings,
                (void*)&output
            };

            CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuLaunchKernel(function, blocks, 1u, 1u, mse_threads, 1u, 1u, mse_shared_bytes, stream, args, NULL), "cuLaunchKernel native_cuda_reference"));
        }
    }

    return 1;
}

static int
cusr_bench_launch_all(
    const CusrBenchConfig* config,
    const CusrBenchLoadedModule* modules,
    const CusrBenchDeviceRun* run,
    uint32_t rows,
    CUstream stream)
{
    size_t module_idx;

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        size_t kernel_idx;

        for (kernel_idx = 0u; kernel_idx < config->kernels_per_module; ++kernel_idx) {
            CUSR_BENCH_CHECK_RET(cusr_bench_launch_one(config, modules, run, rows, module_idx, kernel_idx, stream));
        }
    }

    return 1;
}

static int
cusr_bench_launch_all_streamed(
    const CusrBenchConfig* config,
    const CusrBenchLoadedModule* modules,
    const CusrBenchDeviceRun* run,
    uint32_t rows,
    CUstream* streams)
{
    size_t module_idx;

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        size_t kernel_idx;

        for (kernel_idx = 0u; kernel_idx < config->kernels_per_module; ++kernel_idx) {
            const size_t stream_idx = module_idx * config->kernels_per_module + kernel_idx;
            CUSR_BENCH_CHECK_RET(cusr_bench_launch_one(config, modules, run, rows, module_idx, kernel_idx, streams[stream_idx]));
        }
    }

    return 1;
}

static void
cusr_bench_fill_expected_mse(
    const CusrBenchConfig* config,
    const float* cpu_eval,
    const float* target,
    uint32_t rows,
    float* expected)
{
    size_t kernel_idx;

    for (kernel_idx = 0u; kernel_idx < config->kernels_per_module; ++kernel_idx) {
        size_t local_ast;

        for (local_ast = 0u; local_ast < config->active_asts_per_kernel; ++local_ast) {
            uint32_t setting_idx;
            const size_t global_ast = kernel_idx * config->active_asts_per_kernel + local_ast;

            for (setting_idx = 0u; setting_idx < config->num_settings; ++setting_idx) {
                float sse = 0.0f;
                uint32_t row;
                const size_t output_index =
                    (kernel_idx * config->active_asts_per_kernel + local_ast) *
                    (size_t)config->num_settings + setting_idx;

                for (row = 0u; row < rows; ++row) {
                    const size_t eval_index =
                        (global_ast * (size_t)config->num_settings + setting_idx) * rows + row;
                    const float error = cpu_eval[eval_index] - target[row];
                    sse = error * error + sse;
                }

                expected[output_index] = sse / (float)rows;
            }
        }
    }
}

static int
cusr_bench_compare_output(const char* label, const float* actual, const float* expected, size_t count, float atol)
{
    float max_abs = -1.0f;
    size_t worst_idx = 0u;
    size_t i;

    for (i = 0u; i < count; ++i) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "%s verification failed: non-finite value at %zu actual=%g expected=%g\n", label, i, actual[i], expected[i]);
            return 0;
        }

        const float diff = fabsf(actual[i] - expected[i]);

        if (diff > max_abs || i == 0u) {
            max_abs = diff;
            worst_idx = i;
        }
    }

    printf("%s max_abs_error=%g worst_idx=%zu\n", label, max_abs, worst_idx);
    if (max_abs > atol) {
        fprintf(stderr, "%s verification failed: actual=%g expected=%g\n", label, actual[worst_idx], expected[worst_idx]);
        return 0;
    }

    return 1;
}

static int
cusr_bench_check_outputs(
    const CusrBenchConfig* config,
    const CusrBenchLoadedModule* modules,
    const CusrAstInstruction* programs,
    const CusrSettingF32* settings,
    const uint8_t* leaf_masks,
    const uint32_t* leaf_words)
{
    const size_t asts_per_module = cusr_bench_asts_per_module(config);
    const uint32_t rows = config->check_rows;
    const size_t x_bytes = (size_t)config->num_columns * (size_t)rows * sizeof(float);
    const size_t target_bytes = (size_t)rows * sizeof(float);
    const size_t cpu_eval_elems = asts_per_module * (size_t)config->num_settings * rows;
    const size_t output_bytes = cusr_bench_output_bytes_per_module(config, rows);
    float* x = NULL;
    float* target = NULL;
    float* cpu_eval = NULL;
    float* expected = NULL;
    float* actual = NULL;
    CusrBenchDeviceRun run;
    CUstream stream = NULL;
    size_t module_idx;
    int ok = 0;

    if (rows == 0u) {
        return 1;
    }

    memset(&run, 0, sizeof(run));

    x = (float*)malloc(x_bytes);
    target = (float*)malloc(target_bytes);
    cpu_eval = (float*)malloc(cpu_eval_elems * sizeof(float));
    expected = (float*)malloc(output_bytes);
    actual = (float*)malloc(output_bytes);

    if (x == NULL || target == NULL || cpu_eval == NULL || expected == NULL || actual == NULL) {
        fprintf(stderr, "check allocation failed\n");
    } else {
        cusr_bench_fill_data(x, target, rows, config->num_columns, rows);

        if (cusr_bench_check_cuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate check") &&
            cusr_bench_alloc_device_run(config, rows, x, target, settings, leaf_masks, leaf_words, &run) &&
            cusr_bench_launch_all(config, modules, &run, rows, stream) &&
            cusr_bench_check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize check")) {
            ok = 1;
        }

        for (module_idx = 0u; ok && module_idx < config->num_modules; ++module_idx) {
            char label[128];

            if (cusr_ast_sass_cpu_eval_programs_f32(
                    x,
                    rows,
                    rows,
                    settings,
                    config->num_settings,
                    NULL,
                    0u,
                    programs + module_idx * asts_per_module * config->program_stride,
                    asts_per_module,
                    config->program_stride,
                    cpu_eval) != CUSR_AST_SASS_CPU_SUCCESS) {
                fprintf(stderr, "cpu eval failed for module %zu\n", module_idx);
                ok = 0;
                break;
            }

            cusr_bench_fill_expected_mse(config, cpu_eval, target, rows, expected);

            if (!cusr_bench_check_cuda(cuMemcpyDtoH(actual, run.d_outputs[module_idx], output_bytes), "cuMemcpyDtoH check output")) {
                ok = 0;
                break;
            }

            if (cusr_bench_variant_outputs_mse(config->variant)) {
                size_t output_idx;
                const size_t output_count = output_bytes / sizeof(float);
                const float inverse_rows = 1.0f / (float)rows;

                for (output_idx = 0u; output_idx < output_count; ++output_idx) {
                    actual[output_idx] *= inverse_rows;
                }
            }

            snprintf(label, sizeof(label), "check module %zu", module_idx);
            if (!cusr_bench_compare_output(label, actual, expected, output_bytes / sizeof(float), config->output_atol)) {
                ok = 0;
                break;
            }
        }
    }

    if (stream != NULL) {
        (void)cuStreamDestroy(stream);
    }
    cusr_bench_free_device_run(config, &run);
    free(actual);
    free(expected);
    free(cpu_eval);
    free(target);
    free(x);
    return ok;
}

static int
cusr_bench_time_runtime(
    const CusrBenchConfig* config,
    const CusrBenchLoadedModule* modules,
    const CusrSettingF32* settings,
    const uint8_t* leaf_masks,
    const uint32_t* leaf_words,
    double* runtime_seconds_ret)
{
    const uint32_t rows = config->run_rows;
    const size_t x_bytes = (size_t)config->num_columns * (size_t)rows * sizeof(float);
    const size_t target_bytes = (size_t)rows * sizeof(float);
    const size_t stream_count = config->num_modules * config->kernels_per_module;
    float* x = NULL;
    float* target = NULL;
    CusrBenchDeviceRun run;
    CUstream timing_stream = NULL;
    CUstream* streams = NULL;
    CUevent* done_events = NULL;
    CUevent start = NULL;
    CUevent stop = NULL;
    float elapsed_ms = 0.0f;
    uint32_t iter;
    size_t stream_idx;
    int ok = 0;

    memset(&run, 0, sizeof(run));
    *runtime_seconds_ret = 0.0;

    x = (float*)malloc(x_bytes);
    target = (float*)malloc(target_bytes);
    streams = (CUstream*)calloc(stream_count, sizeof(CUstream));
    done_events = (CUevent*)calloc(stream_count, sizeof(CUevent));
    if (x == NULL || target == NULL || streams == NULL || done_events == NULL) {
        fprintf(stderr, "runtime allocation failed\n");
    } else {
        cusr_bench_fill_data(x, target, rows, config->num_columns, rows);

        if (cusr_bench_check_cuda(cuStreamCreate(&timing_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate runtime timing") &&
            cusr_bench_check_cuda(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate start") &&
            cusr_bench_check_cuda(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate stop") &&
            cusr_bench_alloc_device_run(config, rows, x, target, settings, leaf_masks, leaf_words, &run)) {
            ok = 1;
        }

        for (stream_idx = 0u; ok && stream_idx < stream_count; ++stream_idx) {
            ok = cusr_bench_check_cuda(cuStreamCreate(streams + stream_idx, CU_STREAM_NON_BLOCKING), "cuStreamCreate runtime worker") &&
                cusr_bench_check_cuda(cuEventCreate(done_events + stream_idx, CU_EVENT_DISABLE_TIMING), "cuEventCreate runtime done");
        }

        for (iter = 0u; ok && iter < CUSR_BENCH_RUNTIME_WARMUP_ITERS; ++iter) {
            ok = cusr_bench_launch_all_streamed(config, modules, &run, rows, streams);
        }

        if (ok) {
            ok = cusr_bench_check_cuda(cuCtxSynchronize(), "cuCtxSynchronize runtime warmup");
        }

        if (ok && cusr_bench_check_cuda(cuEventRecord(start, timing_stream), "cuEventRecord start")) {
            for (stream_idx = 0u; ok && stream_idx < stream_count; ++stream_idx) {
                ok = cusr_bench_check_cuda(cuStreamWaitEvent(streams[stream_idx], start, 0u), "cuStreamWaitEvent start");
            }
        }

        for (iter = 0u; ok && iter < config->run_iters; ++iter) {
            ok = cusr_bench_launch_all_streamed(config, modules, &run, rows, streams);
        }

        for (stream_idx = 0u; ok && stream_idx < stream_count; ++stream_idx) {
            ok = cusr_bench_check_cuda(cuEventRecord(done_events[stream_idx], streams[stream_idx]), "cuEventRecord done") &&
                cusr_bench_check_cuda(cuStreamWaitEvent(timing_stream, done_events[stream_idx], 0u), "cuStreamWaitEvent done");
        }

        if (ok &&
            cusr_bench_check_cuda(cuEventRecord(stop, timing_stream), "cuEventRecord stop") &&
            cusr_bench_check_cuda(cuEventSynchronize(stop), "cuEventSynchronize stop") &&
            cusr_bench_check_cuda(cuEventElapsedTime(&elapsed_ms, start, stop), "cuEventElapsedTime")) {
            *runtime_seconds_ret = (double)elapsed_ms * 1.0e-3;
        } else {
            ok = 0;
        }
    }

    if (stop != NULL) (void)cuEventDestroy(stop);
    if (start != NULL) (void)cuEventDestroy(start);
    if (done_events != NULL) {
        for (stream_idx = 0u; stream_idx < stream_count; ++stream_idx) {
            if (done_events[stream_idx] != NULL) {
                (void)cuEventDestroy(done_events[stream_idx]);
            }
        }
    }
    if (streams != NULL) {
        for (stream_idx = 0u; stream_idx < stream_count; ++stream_idx) {
            if (streams[stream_idx] != NULL) {
                (void)cuStreamDestroy(streams[stream_idx]);
            }
        }
    }
    if (timing_stream != NULL) (void)cuStreamDestroy(timing_stream);
    cusr_bench_free_device_run(config, &run);
    free(done_events);
    free(streams);
    free(target);
    free(x);
    return ok;
}

static int
cusr_bench_main_impl(const CusrBenchConfig* config, CUcontext* context_ret)
{
    char arch_option[64];
    char* source = NULL;
    CusrBenchCubin base_cubin = { NULL, 0u };
    CusrBenchKernelSymbols symbols;
    CusrBenchCubin* cubins = NULL;
    CusrBenchLoadedModule* modules = NULL;
    CusrAstInstruction* programs = NULL;
    const CusrAstInstruction** program_ptrs = NULL;
    CusrSettingF32* settings = NULL;
    uint8_t* leaf_masks = NULL;
    uint32_t* leaf_words = NULL;
    void* inspect_workspace = NULL;
    CusrSassInspectHandle inspect;
    CusrAstSassPatchStats patch_stats;
    const size_t total_asts = cusr_bench_total_asts(config);
    const size_t total_sites = config->num_modules * config->kernels_per_module;
    const size_t programs_bytes = total_asts * config->program_stride * sizeof(CusrAstInstruction);
    const size_t program_ptrs_bytes = total_asts * sizeof(*program_ptrs);
    const size_t settings_bytes = (size_t)config->num_settings * sizeof(CusrSettingF32);
    const size_t leaf_masks_bytes = config->kernels_per_module * (size_t)config->num_settings * sizeof(uint8_t);
    const size_t leaf_words_bytes = config->kernels_per_module * (size_t)config->num_settings * CUSR_BENCH_LEAF_WORDS_STRIDE * sizeof(uint32_t);
    double t0;
    double t1;
    double nvrtc_seconds = 0.0;
    double patch_seconds = 0.0;
    double module_load_seconds = 0.0;
    double runtime_seconds = 0.0;
    int active_blocks_per_sm = 0;
    unsigned device_sass_arch = 0u;
    int use_embedded_cubin = 0;
    size_t module_idx;
    int ok = 0;

    memset(&inspect, 0, sizeof(inspect));
    memset(&patch_stats, 0, sizeof(patch_stats));
    memset(&symbols, 0, sizeof(symbols));

    CUSR_BENCH_CHECK_RET(cusr_bench_device_arch(arch_option, sizeof(arch_option), &device_sass_arch, context_ret));
    use_embedded_cubin = cusr_bench_uses_embedded_cubin(config, device_sass_arch);
    CUSR_BENCH_CHECK_RET(cusr_bench_make_kernel_symbols(config, &symbols));

    if (use_embedded_cubin) {
        CUSR_BENCH_CHECK_RET(cusr_bench_copy_embedded_cubin(config, &base_cubin));
    } else {
        t0 = cusr_bench_now_seconds();
        if (config->variant == CUSR_BENCH_VARIANT_TILE_STATIC_MSE_TEMPLATE) {
            CUSR_BENCH_CHECK_RET(cusr_bench_compile_template(config, device_sass_arch, &base_cubin));
        } else {
            CUSR_BENCH_CHECK_RET(cusr_bench_generate_source(config, &source));
            CUSR_BENCH_CHECK_RET(cusr_bench_compile_source(source, arch_option, &base_cubin));
        }
        t1 = cusr_bench_now_seconds();
        nvrtc_seconds = t1 - t0;
    }

    if (cusr_bench_variant_needs_patch(config->variant)) {
        CUSR_BENCH_CHECK_RET(cusr_bench_inspect_base_cubin(config, &base_cubin, &symbols, &inspect, &inspect_workspace));
    }

    cubins = (CusrBenchCubin*)calloc(config->num_modules, sizeof(CusrBenchCubin));
    modules = (CusrBenchLoadedModule*)calloc(config->num_modules, sizeof(CusrBenchLoadedModule));
    programs = (CusrAstInstruction*)malloc(programs_bytes);
    program_ptrs = (const CusrAstInstruction**)malloc(program_ptrs_bytes);
    settings = (CusrSettingF32*)malloc(settings_bytes);
    leaf_masks = (uint8_t*)malloc(leaf_masks_bytes);
    leaf_words = (uint32_t*)malloc(leaf_words_bytes);
    CUSR_BENCH_CHECK_RET(cubins != NULL && modules != NULL && programs != NULL && program_ptrs != NULL && settings != NULL && leaf_masks != NULL && leaf_words != NULL);

    cusr_bench_make_programs(config, programs);
    cusr_bench_make_program_ptrs(config, programs, program_ptrs);
    cusr_bench_fill_settings(settings, config->num_settings, config->num_columns);
    cusr_bench_fill_leaf_settings(leaf_masks, leaf_words, config->kernels_per_module, config->num_settings, settings);
    CUSR_BENCH_CHECK_RET(cusr_bench_copy_base_cubins(config, &base_cubin, cubins));

    if (cusr_bench_variant_needs_patch(config->variant)) {
        t0 = cusr_bench_now_seconds();
        CUSR_BENCH_CHECK_RET(cusr_bench_patch_cubins(config, &inspect, cubins, program_ptrs, &patch_stats));
        t1 = cusr_bench_now_seconds();
        patch_seconds = t1 - t0;
    }

    t0 = cusr_bench_now_seconds();
    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuModuleLoadData(&modules[module_idx].module, cubins[module_idx].data), "cuModuleLoadData"));
    }
    t1 = cusr_bench_now_seconds();
    module_load_seconds = t1 - t0;

    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        size_t kernel_idx;

        modules[module_idx].functions = (CUfunction*)calloc(config->kernels_per_module, sizeof(CUfunction));
        CUSR_BENCH_CHECK_RET(modules[module_idx].functions != NULL);

        for (kernel_idx = 0u; kernel_idx < config->kernels_per_module; ++kernel_idx) {
            CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(
                cuModuleGetFunction(&modules[module_idx].functions[kernel_idx], modules[module_idx].module, symbols.function_names[kernel_idx]),
                "cuModuleGetFunction"));
        }

    }

    if (cusr_bench_variant_outputs_mse(config->variant)) {
        const uint32_t shared_stride = config->num_columns | 1u;
        const size_t dynamic_shared_bytes =
            (size_t)config->tile_rows * (shared_stride + 1u) * sizeof(float);
        const int cta_threads = (int)config->cta_threads;

        CUSR_BENCH_CHECK_RET(cusr_bench_check_cuda(cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            modules[0].functions[0],
            cta_threads,
            dynamic_shared_bytes), "cuOccupancyMaxActiveBlocksPerMultiprocessor"));
    }

    CUSR_BENCH_CHECK_RET(cusr_bench_check_outputs(config, modules, programs, settings, leaf_masks, leaf_words));
    CUSR_BENCH_CHECK_RET(cusr_bench_time_runtime(config, modules, settings, leaf_masks, leaf_words, &runtime_seconds));

    {
        const double asts_per_patch_second = patch_seconds > 0.0 ? (double)total_asts / patch_seconds : 0.0;
        const double sites_per_patch_second = patch_seconds > 0.0 ? (double)total_sites / patch_seconds : 0.0;
        const double patch_us_per_ast = total_asts != 0u ? patch_seconds * 1.0e6 / (double)total_asts : 0.0;
        const double load_asts_per_second = module_load_seconds > 0.0 ? (double)total_asts / module_load_seconds : 0.0;
        const double load_us_per_ast = total_asts != 0u ? module_load_seconds * 1.0e6 / (double)total_asts : 0.0;
        const double row_evals =
            (double)total_asts * (double)config->num_settings * (double)config->run_rows * (double)config->run_iters;
        const double row_evals_per_second = runtime_seconds > 0.0 ? row_evals / runtime_seconds : 0.0;

        const size_t output_bytes = cusr_bench_output_bytes_per_module(config, config->run_rows) * config->num_modules;

        printf("variant=%s ast_mode=%s modules=%zu kernels_per_module=%zu ast_capacity=%zu active_asts_per_kernel=%zu asts=%zu settings=%u tile_rows=%u cta_threads=%u run_rows=%u run_iters=%u\n",
            cusr_bench_variant_name(config->variant),
            cusr_bench_ast_mode_name(config->ast_mode),
            config->num_modules,
            config->kernels_per_module,
            config->ast_capacity,
            config->active_asts_per_kernel,
            total_asts,
            config->num_settings,
            config->tile_rows,
            config->cta_threads,
            config->run_rows,
            config->run_iters);
        printf("kernel_image=%s nvrtc_seconds=%.6f cubin_bytes=%zu inspect_sites=%zu symbol_resolution=%s\n",
            use_embedded_cubin ? "embedded_nvcc" : "runtime_nvrtc",
            nvrtc_seconds,
            base_cubin.size,
            inspect.num_sites,
            "extern_c");
        printf("patch_seconds=%.6f asts_per_second=%.3f sites_per_second=%.3f us_per_ast=%.3f\n",
            patch_seconds,
            asts_per_patch_second,
            sites_per_patch_second,
            patch_us_per_ast);
        printf("patch_output sites=%zu asts=%zu sass_instructions=%zu sass_bytes=%zu max_original=%u max_expanded=%u max_patched=%u\n",
            patch_stats.sites_patched,
            patch_stats.asts_patched,
            patch_stats.sass_instructions_written,
            patch_stats.sass_bytes_written,
            patch_stats.max_original_register_count,
            patch_stats.max_expanded_register_count,
            patch_stats.max_patched_register_count);
        printf("module_load_seconds=%.6f loaded_asts_per_second=%.3f load_us_per_ast=%.3f\n",
            module_load_seconds,
            load_asts_per_second,
            load_us_per_ast);
        printf("runtime_memory output_bytes=%zu\n", output_bytes);
        if (cusr_bench_variant_outputs_mse(config->variant)) {
            const uint32_t shared_stride = config->num_columns | 1u;
            const size_t dynamic_shared_bytes =
                (size_t)config->tile_rows * (shared_stride + 1u) * sizeof(float);
            const int cta_threads = (int)config->cta_threads;

            printf("runtime_resources dynamic_shared_bytes=%zu active_blocks_per_sm=%d active_warps_per_sm=%d\n",
                dynamic_shared_bytes,
                active_blocks_per_sm,
                active_blocks_per_sm * (cta_threads / 32));
        }
        printf("runtime_seconds=%.6f row_evals=%.0f row_evals_per_second=%.3e\n",
            runtime_seconds,
            row_evals,
            row_evals_per_second);
    }

    ok = 1;

    cusr_bench_unload_modules(config, modules);
    for (module_idx = 0u; module_idx < config->num_modules; ++module_idx) {
        free(cubins[module_idx].data);
    }
    free(leaf_words);
    free(leaf_masks);
    free(settings);
    free(program_ptrs);
    free(programs);
    free(modules);
    free(cubins);
    free(inspect_workspace);
    free(base_cubin.data);
    free(source);
    cusr_bench_destroy_kernel_symbols(&symbols);
    return ok;
}

int
main(int argc, char** argv)
{
    CusrBenchConfig config;
    CUcontext context = NULL;
    int ok = 0;

    if (!cusr_bench_parse_args(&config, argc, argv)) {
        cusr_bench_print_usage(argv[0]);
        return 1;
    }

    ok = cusr_bench_main_impl(&config, &context);
    if (context != NULL) {
        (void)cuCtxDestroy(context);
    }

    return ok ? 0 : 1;
}
