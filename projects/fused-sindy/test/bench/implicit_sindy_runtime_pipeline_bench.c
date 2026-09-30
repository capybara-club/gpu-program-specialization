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
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "implicit_sindy.h"
#include "stack_ptx_harness_common.h"

#include <cuda.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    ISRBP_FEATURES = 32,
    ISRBP_LEAVES = 8,
    ISRBP_DEFAULT_PRIMITIVE_COLS = 16
};

typedef enum {
    ISRBP_AST_MODE_POLYNOMIAL = 0,
    ISRBP_AST_MODE_SIMPLE = 1,
    ISRBP_AST_MODE_RANDOM = 2,
    ISRBP_AST_MODE_MAXIMAL = 3
} IsrbpAstMode;

typedef enum {
    ISRBP_SOLVE_POSV = 0,
    ISRBP_SOLVE_STLSQ = 1
} IsrbpSolveMode;

typedef struct {
    size_t modules;
    size_t kernels_per_module;
    size_t settings;
    size_t train_rows;
    size_t validation_rows;
    size_t primitive_cols;
    size_t rhs;
    size_t sweeps;
    size_t compile_workers;
    size_t compile_workspace_mb_per_worker;
    size_t repeats;
    size_t warmup;
    size_t seed;
    unsigned int sm_major;
    unsigned int sm_minor;
    int sm_set;
    int csv;
    IsrbpAstMode ast_mode;
    IsrbpSolveMode solve_mode;
} IsrbpOptions;

typedef struct {
    float train_gram_ms;
    float validation_gram_ms;
    float solve_ms;
    float mse_ms;
    float total_ms;
} IsrbpRunTimes;

typedef struct {
    double best_total_ms;
    double mean_total_ms;
    double stddev_total_ms;
    IsrbpRunTimes best_times;
} IsrbpStats;

typedef struct {
    CUdeviceptr primitive;
    CUdeviceptr targets;
    CUdeviceptr gram;
    CUdeviceptr x_sum;
    CUdeviceptr xty;
    CUdeviceptr y_sum;
    CUdeviceptr yy;
} IsrbpRawBuffers;

typedef struct {
    IsrbpOptions options;
    CUdevice device;
    CUstream stream;
    int owns_primary;
    unsigned int sm_major;
    unsigned int sm_minor;
    size_t cohort_count;
    size_t ast_count;
    size_t cubin_count;
    BinaryAST* asts;
    int32_t* leaf_masks_host;
    int32_t* leaf_words_host;
    float* train_primitive_host;
    float* validation_primitive_host;
    float* train_targets_host;
    float* validation_targets_host;
    float* alphas_host;
    float* thresholds_host;
    ImplicitSindyAstCompiler* compiler;
    void* compile_workspace;
    void** cubins;
    size_t* cubin_sizes;
    void** modules;
    ImplicitSindyGramModule** gram_modules;
    ImplicitFeatureRidgeSolve* solver;
    CUdeviceptr leaf_masks;
    CUdeviceptr leaf_words;
    CUdeviceptr alphas;
    CUdeviceptr thresholds;
    IsrbpRawBuffers train;
    IsrbpRawBuffers validation;
    CUdeviceptr x_mean;
    CUdeviceptr x_scale;
    CUdeviceptr y_mean;
    CUdeviceptr beta;
    CUdeviceptr solve_info;
    CUdeviceptr active_masks;
    CUdeviceptr active_counts;
    CUdeviceptr iteration_counts;
    CUdeviceptr mse;
} IsrbpState;

static void
isrbp_usage(
    const char* exe
) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --modules N       cubin modules to compile/run (default: 1)\n"
        "  --kernels N       Gram kernels/cohorts per module, 1..64 (default: 4)\n"
        "  --settings N      runtime settings per cohort (default: 2048)\n"
        "  --train-rows N    training rows (default: 131072)\n"
        "  --validation-rows N validation rows (default: 65536)\n"
        "  --primitive-cols N primitive feature columns, 2..32 (default: 16)\n"
        "  --rhs N           target RHS count, 1..32 (default: 1)\n"
        "  --sweeps N        ridge/STLSQ alpha sweeps (default: 4)\n"
        "  --solve MODE      posv or stlsq (default: stlsq)\n"
        "  --mode NAME       polynomial, simple, random, maximal (default: polynomial)\n"
        "  --compile-workers N compiler workers for setup (default: online CPUs, capped by modules)\n"
        "  --compile-workspace-mb N scratch MiB per compile worker (default: 256)\n"
        "  --repeats N       timed GPU repeats (default: 5)\n"
        "  --warmup N        untimed GPU warmups (default: 1)\n"
        "  --seed N          deterministic seed (default: 1)\n"
        "  --sm SM           override compile SM, e.g. 90, sm_90, compute_90\n"
        "  --csv             print one CSV row\n"
        "  -h, --help        show this help\n",
        exe
    );
}

static const char*
isrbp_ast_mode_name(
    IsrbpAstMode mode
) {
    switch (mode) {
        case ISRBP_AST_MODE_SIMPLE: return "simple";
        case ISRBP_AST_MODE_RANDOM: return "random";
        case ISRBP_AST_MODE_MAXIMAL: return "maximal";
        case ISRBP_AST_MODE_POLYNOMIAL:
        default: return "polynomial";
    }
}

static const char*
isrbp_solve_mode_name(
    IsrbpSolveMode mode
) {
    return mode == ISRBP_SOLVE_POSV ? "posv" : "stlsq";
}

static int
isrbp_parse_ast_mode(
    const char* text,
    IsrbpAstMode* out_mode
) {
    if (text == NULL || out_mode == NULL) return 0;
    if (strcmp(text, "polynomial") == 0 || strcmp(text, "poly") == 0) {
        *out_mode = ISRBP_AST_MODE_POLYNOMIAL;
        return 1;
    }
    if (strcmp(text, "simple") == 0) {
        *out_mode = ISRBP_AST_MODE_SIMPLE;
        return 1;
    }
    if (strcmp(text, "random") == 0) {
        *out_mode = ISRBP_AST_MODE_RANDOM;
        return 1;
    }
    if (strcmp(text, "maximal") == 0 || strcmp(text, "max") == 0) {
        *out_mode = ISRBP_AST_MODE_MAXIMAL;
        return 1;
    }
    return 0;
}

static int
isrbp_parse_solve_mode(
    const char* text,
    IsrbpSolveMode* out_mode
) {
    if (text == NULL || out_mode == NULL) return 0;
    if (strcmp(text, "posv") == 0) {
        *out_mode = ISRBP_SOLVE_POSV;
        return 1;
    }
    if (strcmp(text, "stlsq") == 0) {
        *out_mode = ISRBP_SOLVE_STLSQ;
        return 1;
    }
    return 0;
}

static size_t
isrbp_default_compile_workers(
    size_t modules
) {
    size_t workers = 1u;
#ifdef _SC_NPROCESSORS_ONLN
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0) workers = (size_t)nproc;
#endif
    if (workers > modules) workers = modules;
    if (workers == 0u) workers = 1u;
    return workers;
}

static int
isrbp_parse_options(
    int argc,
    char** argv,
    IsrbpOptions* options
) {
    int i;
    if (options == NULL) return 0;
    memset(options, 0, sizeof(*options));
    options->modules = 1u;
    options->kernels_per_module = 4u;
    options->settings = 2048u;
    options->train_rows = 131072u;
    options->validation_rows = 65536u;
    options->primitive_cols = ISRBP_DEFAULT_PRIMITIVE_COLS;
    options->rhs = 1u;
    options->sweeps = 4u;
    options->compile_workspace_mb_per_worker = 256u;
    options->repeats = 5u;
    options->warmup = 1u;
    options->seed = 1u;
    options->ast_mode = ISRBP_AST_MODE_POLYNOMIAL;
    options->solve_mode = ISRBP_SOLVE_STLSQ;

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            isrbp_usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--modules") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->modules)) return 0;
        } else if (strcmp(arg, "--kernels") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->kernels_per_module)) return 0;
        } else if (strcmp(arg, "--settings") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->settings)) return 0;
        } else if (strcmp(arg, "--train-rows") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->train_rows)) return 0;
        } else if (strcmp(arg, "--validation-rows") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->validation_rows)) return 0;
        } else if (strcmp(arg, "--primitive-cols") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->primitive_cols)) return 0;
        } else if (strcmp(arg, "--rhs") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->rhs)) return 0;
        } else if (strcmp(arg, "--sweeps") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->sweeps)) return 0;
        } else if (strcmp(arg, "--solve") == 0 && i + 1 < argc) {
            if (!isrbp_parse_solve_mode(argv[++i], &options->solve_mode)) return 0;
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            if (!isrbp_parse_ast_mode(argv[++i], &options->ast_mode)) return 0;
        } else if (strcmp(arg, "--compile-workers") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->compile_workers)) return 0;
        } else if (strcmp(arg, "--compile-workspace-mb") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->compile_workspace_mb_per_worker)) return 0;
        } else if (strcmp(arg, "--repeats") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->repeats)) return 0;
        } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->warmup)) return 0;
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->seed)) return 0;
        } else if (strcmp(arg, "--sm") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_sm(argv[++i], &options->sm_major, &options->sm_minor)) return 0;
            options->sm_set = 1;
        } else if (strcmp(arg, "--csv") == 0) {
            options->csv = 1;
        } else {
            return 0;
        }
    }

    if (options->modules == 0u ||
        options->kernels_per_module == 0u ||
        options->kernels_per_module > 64u ||
        options->settings == 0u ||
        options->train_rows == 0u ||
        options->validation_rows == 0u ||
        options->primitive_cols < 2u ||
        options->primitive_cols > 32u ||
        options->rhs == 0u ||
        options->rhs > 32u ||
        options->sweeps == 0u ||
        options->repeats == 0u ||
        options->compile_workspace_mb_per_worker == 0u) {
        return 0;
    }
    if (options->compile_workers == 0u) {
        options->compile_workers = isrbp_default_compile_workers(options->modules);
    }
    if (options->compile_workers > options->modules) {
        options->compile_workers = options->modules;
    }
    if (options->compile_workers == 0u) {
        options->compile_workers = 1u;
    }
    return 1;
}

static int
isrbp_mul_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) return 0;
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int
isrbp_add_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL || b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static uint32_t
isrbp_u32_mix(
    uint32_t x
) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static uint32_t
isrbp_f32_bits(
    float value
) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

static int
isrbp_cu(
    CUresult result,
    const char* label
) {
    if (result == CUDA_SUCCESS) return 1;
    fprintf(
        stderr,
        "%s failed: %s %s\n",
        label,
        stack_ptx_test_cu_name(result),
        stack_ptx_test_cu_string(result)
    );
    return 0;
}

static int
isrbp_device_alloc(
    CUdeviceptr* ptr,
    size_t bytes,
    const char* label
) {
    if (ptr == NULL || bytes == 0u) return 0;
    *ptr = 0;
    if (!isrbp_cu(cuMemAlloc(ptr, bytes), label)) return 0;
    return 1;
}

static int
isrbp_copy_htod(
    CUdeviceptr dst,
    const void* src,
    size_t bytes,
    const char* label
) {
    if (dst == 0 || src == NULL || bytes == 0u) return 0;
    return isrbp_cu(cuMemcpyHtoD(dst, src, bytes), label);
}

static float*
isrbp_device_f32(
    CUdeviceptr ptr
) {
    return (float*)(uintptr_t)ptr;
}

static int32_t*
isrbp_device_i32(
    CUdeviceptr ptr
) {
    return (int32_t*)(uintptr_t)ptr;
}

static uint32_t*
isrbp_device_u32(
    CUdeviceptr ptr
) {
    return (uint32_t*)(uintptr_t)ptr;
}

static BinaryAstUnaryOp
isrbp_polynomial_unary_op(
    size_t seed,
    size_t idx,
    size_t unary_idx
) {
    const size_t selector = (seed + idx * 11u + unary_idx * 5u) % 3u;
    switch (selector) {
        case 1u: return BINARY_AST_UNARY_SQUARE_F32;
        case 2u: return BINARY_AST_UNARY_CUBE_F32;
        default: return BINARY_AST_UNARY_IDENTITY;
    }
}

static BinaryAstBinaryOp
isrbp_polynomial_binary_op(
    size_t seed,
    size_t idx,
    size_t binary_idx
) {
    const size_t selector = (seed + idx * 7u + binary_idx * 3u) % 5u;
    switch (selector) {
        case 1u: return BINARY_AST_BINARY_SUB_FTZ_F32;
        case 2u: return BINARY_AST_BINARY_MUL_FTZ_F32;
        case 3u: return BINARY_AST_BINARY_KEEP_LEFT;
        case 4u: return BINARY_AST_BINARY_KEEP_RIGHT;
        default: return BINARY_AST_BINARY_ADD_FTZ_F32;
    }
}

static BinaryAstUnaryOp
isrbp_random_unary_op(
    size_t seed,
    size_t idx,
    size_t unary_idx
) {
    const uint32_t value = isrbp_u32_mix((uint32_t)seed ^ (uint32_t)(idx * 131u + unary_idx * 17u));
    switch (value % 12u) {
        case 0u: return BINARY_AST_UNARY_IDENTITY;
        case 1u: return BINARY_AST_UNARY_IDENTITY;
        case 2u: return BINARY_AST_UNARY_SQUARE_F32;
        case 3u: return BINARY_AST_UNARY_CUBE_F32;
        case 4u: return BINARY_AST_UNARY_NEG_FTZ_F32;
        case 5u: return BINARY_AST_UNARY_ABS_FTZ_F32;
        case 6u: return BINARY_AST_UNARY_SIN_APPROX_FTZ_F32;
        case 7u: return BINARY_AST_UNARY_COS_APPROX_FTZ_F32;
        case 8u: return BINARY_AST_UNARY_EX2_APPROX_FTZ_F32;
        case 9u: return BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32;
        case 10u: return BINARY_AST_UNARY_SAFE_RCP_F32;
        default: return BINARY_AST_UNARY_SAFE_SQRT_F32;
    }
}

static BinaryAstBinaryOp
isrbp_random_binary_op(
    size_t seed,
    size_t idx,
    size_t binary_idx
) {
    const uint32_t value = isrbp_u32_mix((uint32_t)(seed + 0x9e3779b9u) ^ (uint32_t)(idx * 73u + binary_idx * 29u));
    switch (value % 10u) {
        case 0u: return BINARY_AST_BINARY_KEEP_LEFT;
        case 1u: return BINARY_AST_BINARY_KEEP_RIGHT;
        case 2u: return BINARY_AST_BINARY_ADD_FTZ_F32;
        case 3u: return BINARY_AST_BINARY_SUB_FTZ_F32;
        case 4u: return BINARY_AST_BINARY_MUL_FTZ_F32;
        case 5u: return BINARY_AST_BINARY_DIV_APPROX_FTZ_F32;
        case 6u: return BINARY_AST_BINARY_MIN_FTZ_F32;
        case 7u: return BINARY_AST_BINARY_MAX_FTZ_F32;
        default: return BINARY_AST_BINARY_SAFE_DIV_F32;
    }
}

static void
isrbp_make_ast(
    const IsrbpOptions* options,
    size_t idx,
    BinaryAST* ast
) {
    size_t i;
    memset(ast, 0, sizeof(*ast));
    if (options->ast_mode == ISRBP_AST_MODE_SIMPLE) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = BINARY_AST_UNARY_IDENTITY;
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = BINARY_AST_BINARY_ADD_FTZ_F32;
        return;
    }
    if (options->ast_mode == ISRBP_AST_MODE_MAXIMAL) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
            ast->unary[i] = (i & 1u) ? BINARY_AST_UNARY_SAFE_EXP_F32 : BINARY_AST_UNARY_SAFE_LOG10_F32;
        }
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = BINARY_AST_BINARY_SAFE_DIV_F32;
        return;
    }
    if (options->ast_mode == ISRBP_AST_MODE_RANDOM) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = isrbp_random_unary_op(options->seed, idx, i);
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = isrbp_random_binary_op(options->seed, idx, i);
        return;
    }
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = isrbp_polynomial_unary_op(options->seed, idx, i);
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = isrbp_polynomial_binary_op(options->seed, idx, i);
}

static int
isrbp_prepare_host_inputs(
    IsrbpState* state
) {
    size_t primitive_train_count;
    size_t primitive_validation_count;
    size_t target_train_count;
    size_t target_validation_count;
    size_t leaf_mask_count;
    size_t leaf_word_count;
    size_t row;
    size_t col;
    size_t rhs;
    size_t cohort;
    size_t setting;
    size_t feature;
    size_t leaf;

    if (state == NULL) return 0;
    if (!isrbp_mul_size(state->options.modules, state->options.kernels_per_module, &state->cohort_count)) return 0;
    if (!isrbp_mul_size(state->cohort_count, ISRBP_FEATURES, &state->ast_count)) return 0;
    state->cubin_count = implicit_sindy_ast_cubin_count(state->ast_count, state->options.kernels_per_module);
    if (state->cohort_count == 0u || state->ast_count == 0u || state->cubin_count == 0u) return 0;

    if (!isrbp_mul_size(state->options.primitive_cols, state->options.train_rows, &primitive_train_count)) return 0;
    if (!isrbp_mul_size(state->options.primitive_cols, state->options.validation_rows, &primitive_validation_count)) return 0;
    if (!isrbp_mul_size(state->options.rhs, state->options.train_rows, &target_train_count)) return 0;
    if (!isrbp_mul_size(state->options.rhs, state->options.validation_rows, &target_validation_count)) return 0;
    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &leaf_mask_count)) return 0;
    if (!isrbp_mul_size(leaf_mask_count, ISRBP_FEATURES, &leaf_mask_count)) return 0;
    if (!isrbp_mul_size(leaf_mask_count, ISRBP_LEAVES, &leaf_word_count)) return 0;

    state->asts = (BinaryAST*)calloc(state->ast_count, sizeof(*state->asts));
    state->leaf_masks_host = (int32_t*)calloc(leaf_mask_count, sizeof(*state->leaf_masks_host));
    state->leaf_words_host = (int32_t*)calloc(leaf_word_count, sizeof(*state->leaf_words_host));
    state->train_primitive_host = (float*)calloc(primitive_train_count, sizeof(*state->train_primitive_host));
    state->validation_primitive_host = (float*)calloc(primitive_validation_count, sizeof(*state->validation_primitive_host));
    state->train_targets_host = (float*)calloc(target_train_count, sizeof(*state->train_targets_host));
    state->validation_targets_host = (float*)calloc(target_validation_count, sizeof(*state->validation_targets_host));
    state->alphas_host = (float*)calloc(state->options.sweeps, sizeof(*state->alphas_host));
    state->thresholds_host = (float*)calloc(state->options.sweeps, sizeof(*state->thresholds_host));
    if (state->asts == NULL ||
        state->leaf_masks_host == NULL ||
        state->leaf_words_host == NULL ||
        state->train_primitive_host == NULL ||
        state->validation_primitive_host == NULL ||
        state->train_targets_host == NULL ||
        state->validation_targets_host == NULL ||
        state->alphas_host == NULL ||
        state->thresholds_host == NULL) {
        return 0;
    }

    for (feature = 0u; feature < state->ast_count; ++feature) {
        isrbp_make_ast(&state->options, feature, &state->asts[feature]);
    }

    for (col = 0u; col < state->options.primitive_cols; ++col) {
        for (row = 0u; row < state->options.train_rows; ++row) {
            state->train_primitive_host[col * state->options.train_rows + row] =
                0.25f + 0.017f * (float)col + 0.00001f * (float)(row & 1023u) +
                0.0003f * (float)((row * 17u + col * 13u) % 31u);
        }
        for (row = 0u; row < state->options.validation_rows; ++row) {
            state->validation_primitive_host[col * state->options.validation_rows + row] =
                0.27f + 0.015f * (float)col + 0.000012f * (float)(row & 1023u) +
                0.0002f * (float)((row * 19u + col * 11u) % 29u);
        }
    }

    for (rhs = 0u; rhs < state->options.rhs; ++rhs) {
        for (row = 0u; row < state->options.train_rows; ++row) {
            const float x0 = state->train_primitive_host[row];
            const float x1 = state->train_primitive_host[state->options.train_rows + row];
            state->train_targets_host[rhs * state->options.train_rows + row] =
                (0.1f + 0.03f * (float)rhs) + 0.7f * x0 - 0.2f * x1 + 0.05f * x0 * x1;
        }
        for (row = 0u; row < state->options.validation_rows; ++row) {
            const float x0 = state->validation_primitive_host[row];
            const float x1 = state->validation_primitive_host[state->options.validation_rows + row];
            state->validation_targets_host[rhs * state->options.validation_rows + row] =
                (0.12f + 0.025f * (float)rhs) + 0.68f * x0 - 0.18f * x1 + 0.04f * x0 * x1;
        }
    }

    for (cohort = 0u; cohort < state->cohort_count; ++cohort) {
        for (setting = 0u; setting < state->options.settings; ++setting) {
            for (feature = 0u; feature < ISRBP_FEATURES; ++feature) {
                uint32_t mask = 0u;
                const size_t setting_feature =
                    (cohort * state->options.settings + setting) * ISRBP_FEATURES + feature;
                for (leaf = 0u; leaf < ISRBP_LEAVES; ++leaf) {
                    const uint32_t value = isrbp_u32_mix(
                        (uint32_t)state->options.seed ^
                        (uint32_t)(cohort * 1000003u + setting * 9176u + feature * 137u + leaf * 17u)
                    );
                    const size_t word_idx = setting_feature * ISRBP_LEAVES + leaf;
                    if ((value % 100u) < 82u) {
                        mask |= 1u << leaf;
                        state->leaf_words_host[word_idx] = (int32_t)(value % (uint32_t)state->options.primitive_cols);
                    } else {
                        const float constant = 0.25f + 0.05f * (float)(value % 9u);
                        state->leaf_words_host[word_idx] = (int32_t)isrbp_f32_bits(constant);
                    }
                }
                state->leaf_masks_host[setting_feature] = (int32_t)mask;
            }
        }
    }

    for (rhs = 0u; rhs < state->options.sweeps; ++rhs) {
        state->alphas_host[rhs] = 1.0e-6f * powf(10.0f, (float)rhs);
        state->thresholds_host[rhs] = rhs == 0u ? 1.0e-3f : 5.0e-3f * (float)rhs;
    }
    return 1;
}

static int
isrbp_compile_and_load(
    IsrbpState* state
) {
    ImplicitSindyResult result;
    ImplicitFeatureRidgeSolveResult solve_result;
    size_t per_worker_workspace_bytes;
    size_t workspace_bytes = 0u;
    size_t i;

    if (state == NULL) return 0;
    if (!stack_ptx_test_init_context(&state->device, &state->owns_primary)) return 0;
    if (state->options.sm_set) {
        state->sm_major = state->options.sm_major;
        state->sm_minor = state->options.sm_minor;
    } else if (!stack_ptx_test_device_sm(state->device, &state->sm_major, &state->sm_minor)) {
        return 0;
    }
    if (!isrbp_cu(cuStreamCreate(&state->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")) return 0;

    result = implicit_sindy_ast_compiler_create(
        state->options.kernels_per_module,
        state->sm_major,
        state->sm_minor,
        NULL,
        0u,
        &state->compiler
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "compiler create failed: %s\n", implicit_sindy_result_to_string(result));
        return 0;
    }

    if (!isrbp_mul_size(state->options.compile_workspace_mb_per_worker, 1024u * 1024u, &per_worker_workspace_bytes)) {
        fprintf(stderr, "compile workspace size overflow\n");
        return 0;
    }

    result = implicit_sindy_ast_compile_workspace_size(
        state->compiler,
        state->ast_count,
        state->options.compile_workers,
        per_worker_workspace_bytes,
        &workspace_bytes
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "workspace size failed: %s\n", implicit_sindy_result_to_string(result));
        return 0;
    }
    state->compile_workspace = malloc(workspace_bytes);
    state->cubins = (void**)calloc(state->cubin_count, sizeof(*state->cubins));
    state->cubin_sizes = (size_t*)calloc(state->cubin_count, sizeof(*state->cubin_sizes));
    state->modules = (void**)calloc(state->cubin_count, sizeof(*state->modules));
    state->gram_modules = (ImplicitSindyGramModule**)calloc(state->cubin_count, sizeof(*state->gram_modules));
    if (state->compile_workspace == NULL ||
        state->cubins == NULL ||
        state->cubin_sizes == NULL ||
        state->modules == NULL ||
        state->gram_modules == NULL) {
        return 0;
    }

    result = implicit_sindy_ast_compile_cubins(
        state->compiler,
        state->asts,
        state->ast_count,
        sizeof(state->asts[0]),
        state->options.compile_workers,
        state->compile_workspace,
        workspace_bytes,
        state->cubins,
        state->cubin_sizes
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "compile failed: %s\n", implicit_sindy_result_to_string(result));
        return 0;
    }

    result = implicit_sindy_load_cubin_modules(
        state->cubins,
        state->cubin_sizes,
        state->cubin_count,
        state->modules
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "module load failed: %s\n", implicit_sindy_result_to_string(result));
        return 0;
    }
    for (i = 0u; i < state->cubin_count; ++i) {
        result = implicit_sindy_gram_module_create(
            state->modules[i],
            state->options.kernels_per_module,
            &state->gram_modules[i]
        );
        if (result != IMPLICIT_SINDY_SUCCESS) {
            fprintf(stderr, "gram module create failed: %s\n", implicit_sindy_result_to_string(result));
            return 0;
        }
    }

    solve_result = implicit_feature_ridge_solve_create(&state->solver);
    if (solve_result != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) {
        fprintf(stderr, "solver create failed: %s\n", implicit_feature_ridge_solve_result_to_string(solve_result));
        return 0;
    }
    return 1;
}

static int
isrbp_alloc_raw_buffers(
    IsrbpState* state,
    IsrbpRawBuffers* buffers,
    size_t rows,
    const float* primitive_host,
    const float* targets_host,
    const char* label
) {
    size_t bytes;
    size_t raw_count;
    char name[128];

    if (state == NULL || buffers == NULL || primitive_host == NULL || targets_host == NULL || label == NULL) return 0;
    memset(buffers, 0, sizeof(*buffers));

    if (!isrbp_mul_size(state->options.primitive_cols, rows, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, sizeof(float), &bytes)) return 0;
    snprintf(name, sizeof(name), "%s primitive", label);
    if (!isrbp_device_alloc(&buffers->primitive, bytes, name)) return 0;
    if (!isrbp_copy_htod(buffers->primitive, primitive_host, bytes, name)) return 0;

    if (!isrbp_mul_size(state->options.rhs, rows, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, sizeof(float), &bytes)) return 0;
    snprintf(name, sizeof(name), "%s targets", label);
    if (!isrbp_device_alloc(&buffers->targets, bytes, name)) return 0;
    if (!isrbp_copy_htod(buffers->targets, targets_host, bytes, name)) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, ISRBP_FEATURES * ISRBP_FEATURES, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, sizeof(float), &bytes)) return 0;
    snprintf(name, sizeof(name), "%s gram", label);
    if (!isrbp_device_alloc(&buffers->gram, bytes, name)) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, ISRBP_FEATURES, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, sizeof(float), &bytes)) return 0;
    snprintf(name, sizeof(name), "%s x_sum", label);
    if (!isrbp_device_alloc(&buffers->x_sum, bytes, name)) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, state->options.rhs, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, ISRBP_FEATURES, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, sizeof(float), &bytes)) return 0;
    snprintf(name, sizeof(name), "%s xty", label);
    if (!isrbp_device_alloc(&buffers->xty, bytes, name)) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, state->options.rhs, &raw_count)) return 0;
    if (!isrbp_mul_size(raw_count, sizeof(float), &bytes)) return 0;
    snprintf(name, sizeof(name), "%s y_sum", label);
    if (!isrbp_device_alloc(&buffers->y_sum, bytes, name)) return 0;
    snprintf(name, sizeof(name), "%s yy", label);
    if (!isrbp_device_alloc(&buffers->yy, bytes, name)) return 0;

    return 1;
}

static int
isrbp_allocate_device_memory(
    IsrbpState* state
) {
    size_t count;
    size_t bytes;

    if (state == NULL) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &count)) return 0;
    if (!isrbp_mul_size(count, ISRBP_FEATURES, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(int32_t), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->leaf_masks, bytes, "leaf masks")) return 0;
    if (!isrbp_copy_htod(state->leaf_masks, state->leaf_masks_host, bytes, "leaf masks copy")) return 0;

    if (!isrbp_mul_size(count, ISRBP_LEAVES, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(int32_t), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->leaf_words, bytes, "leaf words")) return 0;
    if (!isrbp_copy_htod(state->leaf_words, state->leaf_words_host, bytes, "leaf words copy")) return 0;

    if (!isrbp_alloc_raw_buffers(
            state,
            &state->train,
            state->options.train_rows,
            state->train_primitive_host,
            state->train_targets_host,
            "train")) return 0;
    if (!isrbp_alloc_raw_buffers(
            state,
            &state->validation,
            state->options.validation_rows,
            state->validation_primitive_host,
            state->validation_targets_host,
            "validation")) return 0;

    if (!isrbp_mul_size(state->options.sweeps, sizeof(float), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->alphas, bytes, "alphas")) return 0;
    if (!isrbp_copy_htod(state->alphas, state->alphas_host, bytes, "alphas copy")) return 0;
    if (!isrbp_device_alloc(&state->thresholds, bytes, "thresholds")) return 0;
    if (!isrbp_copy_htod(state->thresholds, state->thresholds_host, bytes, "thresholds copy")) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &count)) return 0;
    if (!isrbp_mul_size(count, ISRBP_FEATURES, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->x_mean, bytes, "x_mean")) return 0;
    if (!isrbp_device_alloc(&state->x_scale, bytes, "x_scale")) return 0;

    if (!isrbp_mul_size(state->cohort_count, state->options.settings, &count)) return 0;
    if (!isrbp_mul_size(count, state->options.rhs, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->y_mean, bytes, "y_mean")) return 0;

    if (!isrbp_mul_size(state->options.sweeps, state->cohort_count, &count)) return 0;
    if (!isrbp_mul_size(count, state->options.settings, &count)) return 0;
    if (!isrbp_mul_size(count, state->options.rhs, &count)) return 0;
    if (!isrbp_mul_size(count, ISRBP_FEATURES, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->beta, bytes, "beta")) return 0;

    if (!isrbp_mul_size(state->options.sweeps, state->cohort_count, &count)) return 0;
    if (!isrbp_mul_size(count, state->options.settings, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(int32_t), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->solve_info, bytes, "solve_info")) return 0;

    if (!isrbp_mul_size(state->options.sweeps, state->cohort_count, &count)) return 0;
    if (!isrbp_mul_size(count, state->options.settings, &count)) return 0;
    if (!isrbp_mul_size(count, state->options.rhs, &count)) return 0;
    if (!isrbp_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrbp_device_alloc(&state->mse, bytes, "mse")) return 0;

    if (state->options.solve_mode == ISRBP_SOLVE_STLSQ) {
        if (!isrbp_mul_size(state->options.sweeps, state->cohort_count, &count)) return 0;
        if (!isrbp_mul_size(count, state->options.settings, &count)) return 0;
        if (!isrbp_mul_size(count, state->options.rhs, &count)) return 0;
        if (!isrbp_mul_size(count, sizeof(uint32_t), &bytes)) return 0;
        if (!isrbp_device_alloc(&state->active_masks, bytes, "active_masks")) return 0;
        if (!isrbp_device_alloc(&state->active_counts, bytes, "active_counts")) return 0;
        if (!isrbp_device_alloc(&state->iteration_counts, bytes, "iteration_counts")) return 0;
    }

    return isrbp_cu(cuStreamSynchronize(state->stream), "initial copy synchronize");
}

static size_t
isrbp_raw_setting_offset(
    const IsrbpState* state,
    size_t cohort
) {
    return cohort * state->options.settings;
}

static int
isrbp_launch_gram_phase(
    IsrbpState* state,
    IsrbpRawBuffers* buffers,
    size_t rows
) {
    size_t cohort;
    for (cohort = 0u; cohort < state->cohort_count; ++cohort) {
        const size_t module_idx = cohort / state->options.kernels_per_module;
        const size_t kernel_idx = cohort - module_idx * state->options.kernels_per_module;
        const size_t setting_offset = isrbp_raw_setting_offset(state, cohort);
        ImplicitSindyResult result = implicit_sindy_gram_launch(
            state->gram_modules[module_idx],
            kernel_idx,
            state->stream,
            (int64_t)state->options.settings,
            isrbp_device_f32(buffers->primitive),
            (int64_t)rows,
            (int64_t)rows,
            (int64_t)state->options.primitive_cols,
            isrbp_device_f32(buffers->targets),
            (int64_t)rows,
            (int64_t)state->options.rhs,
            isrbp_device_i32(state->leaf_masks + setting_offset * ISRBP_FEATURES * sizeof(int32_t)),
            isrbp_device_i32(state->leaf_words + setting_offset * ISRBP_FEATURES * ISRBP_LEAVES * sizeof(int32_t)),
            ISRBP_LEAVES,
            isrbp_device_f32(buffers->gram + setting_offset * ISRBP_FEATURES * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            isrbp_device_f32(buffers->x_sum + setting_offset * ISRBP_FEATURES * sizeof(float)),
            isrbp_device_f32(buffers->xty + setting_offset * state->options.rhs * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            isrbp_device_f32(buffers->y_sum + setting_offset * state->options.rhs * sizeof(float)),
            isrbp_device_f32(buffers->yy + setting_offset * state->options.rhs * sizeof(float))
        );
        if (result != IMPLICIT_SINDY_SUCCESS) {
            fprintf(stderr, "gram launch failed: %s\n", implicit_sindy_result_to_string(result));
            return 0;
        }
    }
    return 1;
}

static int
isrbp_launch_solve_phase(
    IsrbpState* state
) {
    size_t cohort;
    for (cohort = 0u; cohort < state->cohort_count; ++cohort) {
        const size_t setting_offset = isrbp_raw_setting_offset(state, cohort);
        const size_t solve_setting_offset = cohort * state->options.settings;
        const size_t rhs_setting_count = state->options.settings * state->options.rhs;
        const size_t beta_cohort_offset = cohort * state->options.settings * state->options.rhs * ISRBP_FEATURES;
        const size_t solve_info_offset = cohort * state->options.settings;
        const size_t active_offset = cohort * state->options.settings * state->options.rhs;
        if (state->options.solve_mode == ISRBP_SOLVE_POSV) {
            ImplicitFeatureRidgeSolveResult result = implicit_feature_ridge_solve_posv_sweep(
                state->solver,
                state->stream,
                (int64_t)state->options.train_rows,
                (int64_t)state->options.settings,
                (int64_t)state->options.rhs,
                (int64_t)state->options.sweeps,
                1.0e-2f,
                isrbp_device_f32(state->train.gram + setting_offset * ISRBP_FEATURES * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                ISRBP_FEATURES * ISRBP_FEATURES,
                isrbp_device_f32(state->train.x_sum + setting_offset * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                isrbp_device_f32(state->train.xty + setting_offset * state->options.rhs * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                (int64_t)(state->options.rhs * ISRBP_FEATURES),
                isrbp_device_f32(state->train.y_sum + setting_offset * state->options.rhs * sizeof(float)),
                1,
                (int64_t)state->options.rhs,
                isrbp_device_f32(state->alphas),
                1,
                isrbp_device_f32(state->x_mean + solve_setting_offset * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                isrbp_device_f32(state->x_scale + solve_setting_offset * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                isrbp_device_f32(state->y_mean + solve_setting_offset * state->options.rhs * sizeof(float)),
                1,
                (int64_t)state->options.rhs,
                isrbp_device_f32(state->beta + beta_cohort_offset * sizeof(float)),
                ISRBP_FEATURES,
                (int64_t)(state->options.rhs * ISRBP_FEATURES),
                (int64_t)(rhs_setting_count * ISRBP_FEATURES),
                isrbp_device_i32(state->solve_info + solve_info_offset * sizeof(int32_t)),
                (int64_t)state->options.settings,
                1
            );
            if (result != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) {
                fprintf(stderr, "posv solve failed: %s\n", implicit_feature_ridge_solve_result_to_string(result));
                return 0;
            }
        } else {
            ImplicitFeatureRidgeSolveResult result = implicit_feature_ridge_solve_stlsq_sweep(
                state->solver,
                state->stream,
                (int64_t)state->options.train_rows,
                (int64_t)state->options.settings,
                (int64_t)state->options.rhs,
                (int64_t)state->options.sweeps,
                1.0e-2f,
                isrbp_device_f32(state->train.gram + setting_offset * ISRBP_FEATURES * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                ISRBP_FEATURES * ISRBP_FEATURES,
                isrbp_device_f32(state->train.x_sum + setting_offset * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                isrbp_device_f32(state->train.xty + setting_offset * state->options.rhs * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                (int64_t)(state->options.rhs * ISRBP_FEATURES),
                isrbp_device_f32(state->train.y_sum + setting_offset * state->options.rhs * sizeof(float)),
                1,
                (int64_t)state->options.rhs,
                isrbp_device_f32(state->alphas),
                1,
                isrbp_device_f32(state->thresholds),
                1,
                isrbp_device_f32(state->x_mean + solve_setting_offset * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                isrbp_device_f32(state->x_scale + solve_setting_offset * ISRBP_FEATURES * sizeof(float)),
                ISRBP_FEATURES,
                isrbp_device_f32(state->y_mean + solve_setting_offset * state->options.rhs * sizeof(float)),
                1,
                (int64_t)state->options.rhs,
                isrbp_device_f32(state->beta + beta_cohort_offset * sizeof(float)),
                ISRBP_FEATURES,
                (int64_t)(state->options.rhs * ISRBP_FEATURES),
                (int64_t)(rhs_setting_count * ISRBP_FEATURES),
                isrbp_device_u32(state->active_masks + active_offset * sizeof(uint32_t)),
                1,
                (int64_t)state->options.rhs,
                (int64_t)rhs_setting_count,
                isrbp_device_i32(state->active_counts + active_offset * sizeof(int32_t)),
                1,
                (int64_t)state->options.rhs,
                (int64_t)rhs_setting_count,
                isrbp_device_i32(state->iteration_counts + active_offset * sizeof(int32_t)),
                1,
                (int64_t)state->options.rhs,
                (int64_t)rhs_setting_count,
                isrbp_device_i32(state->solve_info + solve_info_offset * sizeof(int32_t)),
                (int64_t)state->options.settings,
                1
            );
            if (result != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) {
                fprintf(stderr, "stlsq solve failed: %s\n", implicit_feature_ridge_solve_result_to_string(result));
                return 0;
            }
        }
    }
    return 1;
}

static int
isrbp_launch_mse_phase(
    IsrbpState* state
) {
    size_t cohort;
    for (cohort = 0u; cohort < state->cohort_count; ++cohort) {
        const size_t setting_offset = isrbp_raw_setting_offset(state, cohort);
        const size_t solve_setting_offset = cohort * state->options.settings;
        const size_t rhs_setting_count = state->options.settings * state->options.rhs;
        const size_t beta_cohort_offset = cohort * state->options.settings * state->options.rhs * ISRBP_FEATURES;
        const size_t solve_info_offset = cohort * state->options.settings;
        const size_t mse_offset = cohort * state->options.settings * state->options.rhs;
        ImplicitFeatureRidgeSolveResult result = implicit_feature_ridge_score_validation_mse(
            state->solver,
            state->stream,
            (int64_t)state->options.validation_rows,
            (int64_t)state->options.settings,
            (int64_t)state->options.rhs,
            (int64_t)state->options.sweeps,
            isrbp_device_f32(state->validation.gram + setting_offset * ISRBP_FEATURES * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            ISRBP_FEATURES * ISRBP_FEATURES,
            isrbp_device_f32(state->validation.x_sum + setting_offset * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            isrbp_device_f32(state->validation.xty + setting_offset * state->options.rhs * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            (int64_t)(state->options.rhs * ISRBP_FEATURES),
            isrbp_device_f32(state->validation.y_sum + setting_offset * state->options.rhs * sizeof(float)),
            1,
            (int64_t)state->options.rhs,
            isrbp_device_f32(state->validation.yy + setting_offset * state->options.rhs * sizeof(float)),
            1,
            (int64_t)state->options.rhs,
            isrbp_device_f32(state->beta + beta_cohort_offset * sizeof(float)),
            ISRBP_FEATURES,
            (int64_t)(state->options.rhs * ISRBP_FEATURES),
            (int64_t)(rhs_setting_count * ISRBP_FEATURES),
            isrbp_device_f32(state->x_mean + solve_setting_offset * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            isrbp_device_f32(state->x_scale + solve_setting_offset * ISRBP_FEATURES * sizeof(float)),
            ISRBP_FEATURES,
            isrbp_device_f32(state->y_mean + solve_setting_offset * state->options.rhs * sizeof(float)),
            1,
            (int64_t)state->options.rhs,
            isrbp_device_i32(state->solve_info + solve_info_offset * sizeof(int32_t)),
            (int64_t)state->options.settings,
            1,
            isrbp_device_f32(state->mse + mse_offset * sizeof(float)),
            1,
            (int64_t)state->options.rhs,
            (int64_t)rhs_setting_count
        );
        if (result != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) {
            fprintf(stderr, "mse failed: %s\n", implicit_feature_ridge_solve_result_to_string(result));
            return 0;
        }
    }
    return 1;
}

static int
isrbp_event_create(
    CUevent* event
) {
    return event != NULL && isrbp_cu(cuEventCreate(event, CU_EVENT_DEFAULT), "cuEventCreate");
}

static int
isrbp_elapsed_ms(
    CUevent start,
    CUevent end,
    float* out_ms
) {
    if (out_ms == NULL) return 0;
    return isrbp_cu(cuEventElapsedTime(out_ms, start, end), "cuEventElapsedTime");
}

static int
isrbp_run_gpu_once(
    IsrbpState* state,
    IsrbpRunTimes* out_times
) {
    CUevent total_start = NULL;
    CUevent train_end = NULL;
    CUevent validation_end = NULL;
    CUevent solve_end = NULL;
    CUevent mse_end = NULL;
    int ok = 0;

    if (state == NULL || out_times == NULL) return 0;
    memset(out_times, 0, sizeof(*out_times));
    if (!isrbp_event_create(&total_start) ||
        !isrbp_event_create(&train_end) ||
        !isrbp_event_create(&validation_end) ||
        !isrbp_event_create(&solve_end) ||
        !isrbp_event_create(&mse_end)) {
        goto cleanup;
    }

    if (!isrbp_cu(cuEventRecord(total_start, state->stream), "record total_start")) goto cleanup;
    if (!isrbp_launch_gram_phase(state, &state->train, state->options.train_rows)) goto cleanup;
    if (!isrbp_cu(cuEventRecord(train_end, state->stream), "record train_end")) goto cleanup;
    if (!isrbp_launch_gram_phase(state, &state->validation, state->options.validation_rows)) goto cleanup;
    if (!isrbp_cu(cuEventRecord(validation_end, state->stream), "record validation_end")) goto cleanup;
    if (!isrbp_launch_solve_phase(state)) goto cleanup;
    if (!isrbp_cu(cuEventRecord(solve_end, state->stream), "record solve_end")) goto cleanup;
    if (!isrbp_launch_mse_phase(state)) goto cleanup;
    if (!isrbp_cu(cuEventRecord(mse_end, state->stream), "record mse_end")) goto cleanup;
    if (!isrbp_cu(cuEventSynchronize(mse_end), "cuEventSynchronize")) goto cleanup;

    if (!isrbp_elapsed_ms(total_start, train_end, &out_times->train_gram_ms)) goto cleanup;
    if (!isrbp_elapsed_ms(train_end, validation_end, &out_times->validation_gram_ms)) goto cleanup;
    if (!isrbp_elapsed_ms(validation_end, solve_end, &out_times->solve_ms)) goto cleanup;
    if (!isrbp_elapsed_ms(solve_end, mse_end, &out_times->mse_ms)) goto cleanup;
    if (!isrbp_elapsed_ms(total_start, mse_end, &out_times->total_ms)) goto cleanup;
    ok = 1;

cleanup:
    if (mse_end != NULL) (void)cuEventDestroy(mse_end);
    if (solve_end != NULL) (void)cuEventDestroy(solve_end);
    if (validation_end != NULL) (void)cuEventDestroy(validation_end);
    if (train_end != NULL) (void)cuEventDestroy(train_end);
    if (total_start != NULL) (void)cuEventDestroy(total_start);
    return ok;
}

static int
isrbp_run_benchmark(
    IsrbpState* state,
    IsrbpStats* stats
) {
    IsrbpRunTimes* times = NULL;
    size_t i;
    double sum = 0.0;
    double variance_sum = 0.0;

    if (state == NULL || stats == NULL) return 0;
    memset(stats, 0, sizeof(*stats));
    stats->best_total_ms = DBL_MAX;
    times = (IsrbpRunTimes*)calloc(state->options.repeats, sizeof(*times));
    if (times == NULL) return 0;

    for (i = 0u; i < state->options.warmup; ++i) {
        IsrbpRunTimes warmup_times;
        if (!isrbp_run_gpu_once(state, &warmup_times)) {
            free(times);
            return 0;
        }
    }

    for (i = 0u; i < state->options.repeats; ++i) {
        if (!isrbp_run_gpu_once(state, &times[i])) {
            free(times);
            return 0;
        }
        sum += (double)times[i].total_ms;
        if ((double)times[i].total_ms < stats->best_total_ms) {
            stats->best_total_ms = (double)times[i].total_ms;
            stats->best_times = times[i];
        }
    }
    stats->mean_total_ms = sum / (double)state->options.repeats;
    for (i = 0u; i < state->options.repeats; ++i) {
        const double diff = (double)times[i].total_ms - stats->mean_total_ms;
        variance_sum += diff * diff;
    }
    stats->stddev_total_ms = sqrt(variance_sum / (double)state->options.repeats);
    free(times);
    return 1;
}

static void
isrbp_print_results(
    const IsrbpState* state,
    const IsrbpStats* stats
) {
    const double gram_ms = (double)stats->best_times.train_gram_ms + (double)stats->best_times.validation_gram_ms;
    const double total_row_settings = (double)state->cohort_count *
        (double)state->options.settings *
        (double)(state->options.train_rows + state->options.validation_rows);
    const double total_feature_row_settings = total_row_settings * (double)ISRBP_FEATURES;
    const double solve_candidates = (double)state->cohort_count *
        (double)state->options.settings *
        (double)state->options.sweeps *
        (double)state->options.rhs;
    const double gram_row_settings_per_sec = gram_ms > 0.0
        ? 1000.0 * total_row_settings / gram_ms
        : 0.0;
    const double feature_row_settings_per_sec = gram_ms > 0.0
        ? 1000.0 * total_feature_row_settings / gram_ms
        : 0.0;
    const double solve_candidates_per_sec = stats->best_times.solve_ms > 0.0f
        ? 1000.0 * solve_candidates / (double)stats->best_times.solve_ms
        : 0.0;
    const double mse_candidates_per_sec = stats->best_times.mse_ms > 0.0f
        ? 1000.0 * solve_candidates / (double)stats->best_times.mse_ms
        : 0.0;
    const double pipeline_settings_per_sec = stats->best_total_ms > 0.0
        ? 1000.0 * (double)(state->cohort_count * state->options.settings) / stats->best_total_ms
        : 0.0;

    if (state->options.csv) {
        printf(
            "mode,solve,modules,kernels,cohorts,settings,train_rows,validation_rows,rhs,sweeps,sm,"
            "warmup,repeats,total_ms_best,total_ms_mean,total_ms_stddev,train_gram_ms,"
            "validation_gram_ms,solve_ms,mse_ms,pipeline_settings_per_sec,"
            "gram_row_settings_per_sec,gram_feature_row_settings_per_sec,solve_candidates_per_sec,"
            "mse_candidates_per_sec\n"
        );
        printf(
            "%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%u%u,%zu,%zu,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3e,%.3e,%.3e,%.3e\n",
            isrbp_ast_mode_name(state->options.ast_mode),
            isrbp_solve_mode_name(state->options.solve_mode),
            state->options.modules,
            state->options.kernels_per_module,
            state->cohort_count,
            state->options.settings,
            state->options.train_rows,
            state->options.validation_rows,
            state->options.rhs,
            state->options.sweeps,
            state->sm_major,
            state->sm_minor,
            state->options.warmup,
            state->options.repeats,
            stats->best_total_ms,
            stats->mean_total_ms,
            stats->stddev_total_ms,
            (double)stats->best_times.train_gram_ms,
            (double)stats->best_times.validation_gram_ms,
            (double)stats->best_times.solve_ms,
            (double)stats->best_times.mse_ms,
            pipeline_settings_per_sec,
            gram_row_settings_per_sec,
            feature_row_settings_per_sec,
            solve_candidates_per_sec,
            mse_candidates_per_sec
        );
        return;
    }

    printf(
        "implicit_sindy_runtime_pipeline mode=%s solve=%s modules=%zu kernels=%zu cohorts=%zu "
        "settings=%zu train_rows=%zu validation_rows=%zu rhs=%zu sweeps=%zu sm_%u%u "
        "warmup=%zu repeats=%zu\n",
        isrbp_ast_mode_name(state->options.ast_mode),
        isrbp_solve_mode_name(state->options.solve_mode),
        state->options.modules,
        state->options.kernels_per_module,
        state->cohort_count,
        state->options.settings,
        state->options.train_rows,
        state->options.validation_rows,
        state->options.rhs,
        state->options.sweeps,
        state->sm_major,
        state->sm_minor,
        state->options.warmup,
        state->options.repeats
    );
    printf(
        "  total_ms_best=%.3f total_ms_mean=%.3f total_ms_stddev=%.3f\n",
        stats->best_total_ms,
        stats->mean_total_ms,
        stats->stddev_total_ms
    );
    printf(
        "  phases_best_ms: train_gram=%.3f validation_gram=%.3f solve=%.3f mse=%.3f\n",
        (double)stats->best_times.train_gram_ms,
        (double)stats->best_times.validation_gram_ms,
        (double)stats->best_times.solve_ms,
        (double)stats->best_times.mse_ms
    );
    printf(
        "  throughput: pipeline_settings_per_sec=%.3f gram_row_settings_per_sec=%.3e "
        "gram_feature_row_settings_per_sec=%.3e solve_candidates_per_sec=%.3e "
        "mse_candidates_per_sec=%.3e\n",
        pipeline_settings_per_sec,
        gram_row_settings_per_sec,
        feature_row_settings_per_sec,
        solve_candidates_per_sec,
        mse_candidates_per_sec
    );
}

static void
isrbp_free_raw_buffers(
    IsrbpRawBuffers* buffers
) {
    if (buffers == NULL) return;
    if (buffers->primitive != 0) (void)cuMemFree(buffers->primitive);
    if (buffers->targets != 0) (void)cuMemFree(buffers->targets);
    if (buffers->gram != 0) (void)cuMemFree(buffers->gram);
    if (buffers->x_sum != 0) (void)cuMemFree(buffers->x_sum);
    if (buffers->xty != 0) (void)cuMemFree(buffers->xty);
    if (buffers->y_sum != 0) (void)cuMemFree(buffers->y_sum);
    if (buffers->yy != 0) (void)cuMemFree(buffers->yy);
    memset(buffers, 0, sizeof(*buffers));
}

static void
isrbp_cleanup(
    IsrbpState* state
) {
    size_t i;
    if (state == NULL) return;
    if (state->stream != NULL) (void)cuStreamSynchronize(state->stream);
    if (state->mse != 0) (void)cuMemFree(state->mse);
    if (state->iteration_counts != 0) (void)cuMemFree(state->iteration_counts);
    if (state->active_counts != 0) (void)cuMemFree(state->active_counts);
    if (state->active_masks != 0) (void)cuMemFree(state->active_masks);
    if (state->solve_info != 0) (void)cuMemFree(state->solve_info);
    if (state->beta != 0) (void)cuMemFree(state->beta);
    if (state->y_mean != 0) (void)cuMemFree(state->y_mean);
    if (state->x_scale != 0) (void)cuMemFree(state->x_scale);
    if (state->x_mean != 0) (void)cuMemFree(state->x_mean);
    if (state->thresholds != 0) (void)cuMemFree(state->thresholds);
    if (state->alphas != 0) (void)cuMemFree(state->alphas);
    isrbp_free_raw_buffers(&state->validation);
    isrbp_free_raw_buffers(&state->train);
    if (state->leaf_words != 0) (void)cuMemFree(state->leaf_words);
    if (state->leaf_masks != 0) (void)cuMemFree(state->leaf_masks);
    if (state->solver != NULL) (void)implicit_feature_ridge_solve_destroy(state->solver);
    if (state->gram_modules != NULL) {
        for (i = 0u; i < state->cubin_count; ++i) {
            if (state->gram_modules[i] != NULL) {
                (void)implicit_sindy_gram_module_destroy(state->gram_modules[i]);
            }
        }
    }
    if (state->modules != NULL) (void)implicit_sindy_unload_modules(state->modules, state->cubin_count);
    implicit_sindy_free_cubins(state->cubins, state->cubin_sizes, state->cubin_count);
    if (state->compiler != NULL) (void)implicit_sindy_ast_compiler_destroy(state->compiler);
    if (state->stream != NULL) (void)cuStreamDestroy(state->stream);
    stack_ptx_test_release_context(state->device, state->owns_primary);
    free(state->gram_modules);
    free(state->modules);
    free(state->cubin_sizes);
    free(state->cubins);
    free(state->compile_workspace);
    free(state->thresholds_host);
    free(state->alphas_host);
    free(state->validation_targets_host);
    free(state->train_targets_host);
    free(state->validation_primitive_host);
    free(state->train_primitive_host);
    free(state->leaf_words_host);
    free(state->leaf_masks_host);
    free(state->asts);
}

int
main(
    int argc,
    char** argv
) {
    IsrbpState state;
    IsrbpStats stats;
    int ok = 0;

    memset(&state, 0, sizeof(state));
    memset(&stats, 0, sizeof(stats));
    if (!isrbp_parse_options(argc, argv, &state.options)) {
        isrbp_usage(argv[0]);
        return 1;
    }
    if (!isrbp_prepare_host_inputs(&state)) goto cleanup;
    if (!isrbp_compile_and_load(&state)) goto cleanup;
    if (!isrbp_allocate_device_memory(&state)) goto cleanup;
    if (!isrbp_run_benchmark(&state, &stats)) goto cleanup;
    isrbp_print_results(&state, &stats);
    ok = 1;

cleanup:
    isrbp_cleanup(&state);
    return ok ? 0 : 1;
}
