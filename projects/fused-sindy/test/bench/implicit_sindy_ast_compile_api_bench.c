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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    ISACB_FEATURES_PER_KERNEL = 32,
    ISACB_MAX_WORKER_SWEEP = 64
};

typedef enum {
    ISACB_AST_MODE_POLYNOMIAL = 0,
    ISACB_AST_MODE_SIMPLE = 1,
    ISACB_AST_MODE_RANDOM = 2,
    ISACB_AST_MODE_MAXIMAL = 3
} IsacbAstMode;

typedef struct {
    size_t modules;
    size_t kernels_per_module;
    size_t workers[ISACB_MAX_WORKER_SWEEP];
    size_t num_worker_values;
    size_t repeats;
    size_t warmup;
    size_t workspace_mb_per_worker;
    size_t seed;
    unsigned int sm_major;
    unsigned int sm_minor;
    int sm_set;
    int csv;
    IsacbAstMode ast_mode;
} IsacbOptions;

typedef struct {
    IsacbOptions options;
    ImplicitSindyAstCompiler* compiler;
    BinaryAST* asts;
    void** cubins;
    size_t* cubin_sizes;
    size_t ast_count;
    size_t cubin_count;
    CUdevice device;
    int owns_primary;
    unsigned int sm_major;
    unsigned int sm_minor;
} IsacbState;

typedef struct {
    double best_ms;
    double mean_ms;
    double stddev_ms;
} IsacbStats;

static void
isacb_usage(
    const char* exe
) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --modules N       cubins/modules to compile (default: 512)\n"
        "  --kernels N       generated Gram kernels per module, 1..64 (default: 4)\n"
        "  --workers LIST    worker sweep, comma-separated or single value (default: auto)\n"
        "  --repeats N       timed repeats per worker value (default: 1)\n"
        "  --warmup N        untimed warmup compiles per worker value (default: 0)\n"
        "  --workspace-mb N  scratch MiB per worker (default: 256)\n"
        "  --mode NAME       polynomial, simple, random, maximal (default: random)\n"
        "  --seed N          deterministic AST seed (default: 1)\n"
        "  --sm SM           override SM, e.g. 90, sm_90, compute_90\n"
        "  --csv             print CSV rows\n"
        "  -h, --help        show this help\n",
        exe
    );
}

static const char*
isacb_ast_mode_name(
    IsacbAstMode mode
) {
    switch (mode) {
        case ISACB_AST_MODE_SIMPLE: return "simple";
        case ISACB_AST_MODE_RANDOM: return "random";
        case ISACB_AST_MODE_MAXIMAL: return "maximal";
        case ISACB_AST_MODE_POLYNOMIAL:
        default: return "polynomial";
    }
}

static int
isacb_parse_ast_mode(
    const char* text,
    IsacbAstMode* out_mode
) {
    if (text == NULL || out_mode == NULL) return 0;
    if (strcmp(text, "polynomial") == 0 || strcmp(text, "poly") == 0) {
        *out_mode = ISACB_AST_MODE_POLYNOMIAL;
        return 1;
    }
    if (strcmp(text, "simple") == 0) {
        *out_mode = ISACB_AST_MODE_SIMPLE;
        return 1;
    }
    if (strcmp(text, "random") == 0) {
        *out_mode = ISACB_AST_MODE_RANDOM;
        return 1;
    }
    if (strcmp(text, "maximal") == 0 || strcmp(text, "max") == 0) {
        *out_mode = ISACB_AST_MODE_MAXIMAL;
        return 1;
    }
    return 0;
}

static size_t
isacb_online_cpus(void) {
    size_t cpus = 1u;
#ifdef _SC_NPROCESSORS_ONLN
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0) cpus = (size_t)nproc;
#endif
    return cpus;
}

static int
isacb_push_worker_value(
    IsacbOptions* options,
    size_t workers
) {
    size_t i;
    if (options == NULL || workers == 0u) return 0;
    if (workers > options->modules) workers = options->modules;
    for (i = 0u; i < options->num_worker_values; ++i) {
        if (options->workers[i] == workers) return 1;
    }
    if (options->num_worker_values == ISACB_MAX_WORKER_SWEEP) return 0;
    options->workers[options->num_worker_values++] = workers;
    return 1;
}

static int
isacb_parse_worker_list(
    const char* text,
    IsacbOptions* options
) {
    const char* p = text;
    char item[64];
    size_t item_len;
    size_t value;

    if (text == NULL || options == NULL) return 0;
    options->num_worker_values = 0u;
    while (*p != '\0') {
        item_len = 0u;
        while (*p != '\0' && *p != ',') {
            if (item_len + 1u >= sizeof(item)) return 0;
            item[item_len++] = *p++;
        }
        item[item_len] = '\0';
        if (item_len == 0u) return 0;
        if (!stack_ptx_test_parse_size(item, &value)) return 0;
        if (!isacb_push_worker_value(options, value)) return 0;
        if (*p == ',') ++p;
    }
    return options->num_worker_values != 0u;
}

static int
isacb_default_worker_list(
    IsacbOptions* options
) {
    static const size_t candidates[] = {
        1u, 2u, 4u, 8u, 12u, 16u, 24u, 32u, 48u, 64u, 96u, 128u
    };
    size_t max_workers;
    size_t i;

    if (options == NULL) return 0;
    max_workers = isacb_online_cpus();
    if (max_workers > options->modules) max_workers = options->modules;
    if (max_workers == 0u) max_workers = 1u;
    options->num_worker_values = 0u;
    for (i = 0u; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (candidates[i] <= max_workers && !isacb_push_worker_value(options, candidates[i])) return 0;
    }
    if (!isacb_push_worker_value(options, max_workers)) return 0;
    return options->num_worker_values != 0u;
}

static int
isacb_normalize_worker_list(
    IsacbOptions* options
) {
    size_t normalized[ISACB_MAX_WORKER_SWEEP];
    size_t normalized_count = 0u;
    size_t i;

    if (options == NULL || options->modules == 0u) return 0;
    for (i = 0u; i < options->num_worker_values; ++i) {
        size_t j;
        size_t workers = options->workers[i];
        int duplicate = 0;
        if (workers == 0u) return 0;
        if (workers > options->modules) workers = options->modules;
        for (j = 0u; j < normalized_count; ++j) {
            if (normalized[j] == workers) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            if (normalized_count == ISACB_MAX_WORKER_SWEEP) return 0;
            normalized[normalized_count++] = workers;
        }
    }
    if (normalized_count == 0u) return 0;
    memset(options->workers, 0, sizeof(options->workers));
    memcpy(options->workers, normalized, normalized_count * sizeof(normalized[0]));
    options->num_worker_values = normalized_count;
    return 1;
}

static int
isacb_parse_options(
    int argc,
    char** argv,
    IsacbOptions* options
) {
    int i;
    if (options == NULL) return 0;
    memset(options, 0, sizeof(*options));
    options->modules = 512u;
    options->kernels_per_module = 4u;
    options->repeats = 1u;
    options->workspace_mb_per_worker = 256u;
    options->seed = 1u;
    options->ast_mode = ISACB_AST_MODE_RANDOM;

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            isacb_usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--modules") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->modules)) return 0;
        } else if (strcmp(arg, "--kernels") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->kernels_per_module)) return 0;
        } else if (strcmp(arg, "--workers") == 0 && i + 1 < argc) {
            if (!isacb_parse_worker_list(argv[++i], options)) return 0;
        } else if (strcmp(arg, "--repeats") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->repeats)) return 0;
        } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->warmup)) return 0;
        } else if (strcmp(arg, "--workspace-mb") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->workspace_mb_per_worker)) return 0;
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            if (!isacb_parse_ast_mode(argv[++i], &options->ast_mode)) return 0;
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
        options->repeats == 0u ||
        options->workspace_mb_per_worker == 0u) {
        return 0;
    }
    if (options->num_worker_values == 0u && !isacb_default_worker_list(options)) return 0;
    return isacb_normalize_worker_list(options);
}

static uint32_t
isacb_u32_mix(
    uint32_t x
) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static BinaryAstUnaryOp
isacb_polynomial_unary_op(
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
isacb_polynomial_binary_op(
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
isacb_random_unary_op(
    size_t seed,
    size_t idx,
    size_t unary_idx
) {
    const uint32_t value = isacb_u32_mix((uint32_t)seed ^ (uint32_t)(idx * 131u + unary_idx * 17u));
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
isacb_random_binary_op(
    size_t seed,
    size_t idx,
    size_t binary_idx
) {
    const uint32_t value = isacb_u32_mix((uint32_t)(seed + 0x9e3779b9u) ^ (uint32_t)(idx * 73u + binary_idx * 29u));
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
isacb_make_ast(
    const IsacbOptions* options,
    size_t idx,
    BinaryAST* ast
) {
    size_t i;
    memset(ast, 0, sizeof(*ast));
    if (options->ast_mode == ISACB_AST_MODE_SIMPLE) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = BINARY_AST_UNARY_IDENTITY;
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = BINARY_AST_BINARY_ADD_FTZ_F32;
        return;
    }
    if (options->ast_mode == ISACB_AST_MODE_MAXIMAL) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
            ast->unary[i] = (i & 1u) ? BINARY_AST_UNARY_SAFE_EXP_F32 : BINARY_AST_UNARY_SAFE_LOG10_F32;
        }
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = BINARY_AST_BINARY_SAFE_DIV_F32;
        return;
    }
    if (options->ast_mode == ISACB_AST_MODE_RANDOM) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = isacb_random_unary_op(options->seed, idx, i);
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = isacb_random_binary_op(options->seed, idx, i);
        return;
    }
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = isacb_polynomial_unary_op(options->seed, idx, i);
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = isacb_polynomial_binary_op(options->seed, idx, i);
}

static int
isacb_setup_asts(
    IsacbState* state
) {
    size_t i;
    if (state == NULL) return 0;
    if (state->options.modules > SIZE_MAX / state->options.kernels_per_module) return 0;
    if (state->options.modules * state->options.kernels_per_module > SIZE_MAX / ISACB_FEATURES_PER_KERNEL) return 0;
    state->ast_count = state->options.modules * state->options.kernels_per_module * ISACB_FEATURES_PER_KERNEL;
    state->cubin_count = implicit_sindy_ast_cubin_count(state->ast_count, state->options.kernels_per_module);
    if (state->ast_count == 0u || state->cubin_count == 0u) return 0;
    state->asts = (BinaryAST*)calloc(state->ast_count, sizeof(*state->asts));
    state->cubins = (void**)calloc(state->cubin_count, sizeof(*state->cubins));
    state->cubin_sizes = (size_t*)calloc(state->cubin_count, sizeof(*state->cubin_sizes));
    if (state->asts == NULL || state->cubins == NULL || state->cubin_sizes == NULL) return 0;
    for (i = 0u; i < state->ast_count; ++i) {
        isacb_make_ast(&state->options, i, &state->asts[i]);
    }
    return 1;
}

static int
isacb_setup_context_and_compiler(
    IsacbState* state
) {
    ImplicitSindyResult result;
    if (state == NULL) return 0;
    if (!stack_ptx_test_init_context(&state->device, &state->owns_primary)) return 0;
    if (state->options.sm_set) {
        state->sm_major = state->options.sm_major;
        state->sm_minor = state->options.sm_minor;
    } else if (!stack_ptx_test_device_sm(state->device, &state->sm_major, &state->sm_minor)) {
        return 0;
    }
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
    return 1;
}

static void
isacb_reset_outputs(
    IsacbState* state
) {
    if (state == NULL) return;
    memset(state->cubins, 0, state->cubin_count * sizeof(*state->cubins));
    memset(state->cubin_sizes, 0, state->cubin_count * sizeof(*state->cubin_sizes));
}

static int
isacb_compile_once(
    IsacbState* state,
    size_t worker_count,
    void* workspace,
    size_t workspace_bytes,
    double* out_hot_ms
) {
    ImplicitSindyResult result;
    double start_ms;
    double hot_ms;
    if (state == NULL || workspace == NULL || out_hot_ms == NULL) return 0;

    isacb_reset_outputs(state);
    start_ms = stack_ptx_test_now_ms();
    result = implicit_sindy_ast_compile_cubins(
        state->compiler,
        state->asts,
        state->ast_count,
        sizeof(state->asts[0]),
        worker_count,
        workspace,
        workspace_bytes,
        state->cubins,
        state->cubin_sizes
    );
    hot_ms = stack_ptx_test_now_ms() - start_ms;
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "compile failed: %s\n", implicit_sindy_result_to_string(result));
        implicit_sindy_free_cubins(state->cubins, state->cubin_sizes, state->cubin_count);
        isacb_reset_outputs(state);
        return 0;
    }

    *out_hot_ms = hot_ms;
    implicit_sindy_free_cubins(state->cubins, state->cubin_sizes, state->cubin_count);
    isacb_reset_outputs(state);
    return 1;
}

static int
isacb_run_worker_value(
    IsacbState* state,
    size_t worker_count,
    IsacbStats* out_stats
) {
    void* workspace = NULL;
    double* times = NULL;
    size_t workspace_bytes = 0u;
    size_t i;
    double sum = 0.0;
    double variance_sum = 0.0;
    ImplicitSindyResult result;

    if (state == NULL || out_stats == NULL || worker_count == 0u) return 0;
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->best_ms = DBL_MAX;
    if (state->options.workspace_mb_per_worker > SIZE_MAX / (1024u * 1024u)) return 0;

    result = implicit_sindy_ast_compile_workspace_size(
        state->compiler,
        state->ast_count,
        worker_count,
        state->options.workspace_mb_per_worker * 1024u * 1024u,
        &workspace_bytes
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "workspace size failed: %s\n", implicit_sindy_result_to_string(result));
        return 0;
    }
    workspace = malloc(workspace_bytes);
    times = (double*)calloc(state->options.repeats, sizeof(*times));
    if (workspace == NULL || times == NULL) {
        free(workspace);
        free(times);
        return 0;
    }

    for (i = 0u; i < state->options.warmup; ++i) {
        double ignored_ms = 0.0;
        if (!isacb_compile_once(state, worker_count, workspace, workspace_bytes, &ignored_ms)) {
            free(workspace);
            free(times);
            return 0;
        }
    }

    for (i = 0u; i < state->options.repeats; ++i) {
        if (!isacb_compile_once(state, worker_count, workspace, workspace_bytes, &times[i])) {
            free(workspace);
            free(times);
            return 0;
        }
        sum += times[i];
        if (times[i] < out_stats->best_ms) out_stats->best_ms = times[i];
    }
    out_stats->mean_ms = sum / (double)state->options.repeats;
    for (i = 0u; i < state->options.repeats; ++i) {
        const double diff = times[i] - out_stats->mean_ms;
        variance_sum += diff * diff;
    }
    out_stats->stddev_ms = sqrt(variance_sum / (double)state->options.repeats);

    free(workspace);
    free(times);
    return 1;
}

static void
isacb_print_header(
    const IsacbState* state
) {
    if (state->options.csv) {
        printf(
            "mode,modules,kernels,workers,sm,warmup,repeats,hot_compile_ms_best,"
            "hot_compile_ms_mean,hot_compile_ms_stddev,modules_per_sec_best,"
            "gram_kernels_per_sec_best,asts_per_sec_best\n"
        );
        return;
    }
    printf(
        "implicit_sindy_ast_compile_api_hot mode=%s modules=%zu kernels=%zu asts=%zu "
        "sm_%u%u warmup=%zu repeats=%zu\n",
        isacb_ast_mode_name(state->options.ast_mode),
        state->options.modules,
        state->options.kernels_per_module,
        state->ast_count,
        state->sm_major,
        state->sm_minor,
        state->options.warmup,
        state->options.repeats
    );
    printf(
        "%8s %16s %16s %16s %16s %20s %16s\n",
        "workers",
        "best_ms",
        "mean_ms",
        "stddev_ms",
        "modules/s",
        "gram_kernels/s",
        "asts/s"
    );
}

static void
isacb_print_row(
    const IsacbState* state,
    size_t worker_count,
    const IsacbStats* stats
) {
    const double module_rate = 1000.0 * (double)state->options.modules / stats->best_ms;
    const double gram_kernel_rate = 1000.0 * (double)(state->options.modules * state->options.kernels_per_module) / stats->best_ms;
    const double ast_rate = 1000.0 * (double)state->ast_count / stats->best_ms;

    if (state->options.csv) {
        printf(
            "%s,%zu,%zu,%zu,%u%u,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            isacb_ast_mode_name(state->options.ast_mode),
            state->options.modules,
            state->options.kernels_per_module,
            worker_count,
            state->sm_major,
            state->sm_minor,
            state->options.warmup,
            state->options.repeats,
            stats->best_ms,
            stats->mean_ms,
            stats->stddev_ms,
            module_rate,
            gram_kernel_rate,
            ast_rate
        );
        return;
    }
    printf(
        "%8zu %16.3f %16.3f %16.3f %16.3f %20.3f %16.3f\n",
        worker_count,
        stats->best_ms,
        stats->mean_ms,
        stats->stddev_ms,
        module_rate,
        gram_kernel_rate,
        ast_rate
    );
}

static int
isacb_run(
    IsacbState* state
) {
    size_t i;
    if (state == NULL) return 0;
    if (!isacb_setup_asts(state)) return 0;
    if (!isacb_setup_context_and_compiler(state)) return 0;
    isacb_print_header(state);
    for (i = 0u; i < state->options.num_worker_values; ++i) {
        IsacbStats stats;
        const size_t worker_count = state->options.workers[i];
        if (!isacb_run_worker_value(state, worker_count, &stats)) return 0;
        isacb_print_row(state, worker_count, &stats);
        fflush(stdout);
    }
    return 1;
}

static void
isacb_cleanup(
    IsacbState* state
) {
    if (state == NULL) return;
    implicit_sindy_free_cubins(state->cubins, state->cubin_sizes, state->cubin_count);
    free(state->cubin_sizes);
    free(state->cubins);
    free(state->asts);
    if (state->compiler != NULL) (void)implicit_sindy_ast_compiler_destroy(state->compiler);
    stack_ptx_test_release_context(state->device, state->owns_primary);
}

int
main(
    int argc,
    char** argv
) {
    IsacbState state;
    int ok;
    memset(&state, 0, sizeof(state));
    if (!isacb_parse_options(argc, argv, &state.options)) {
        isacb_usage(argv[0]);
        return 1;
    }
    ok = isacb_run(&state);
    isacb_cleanup(&state);
    return ok ? 0 : 1;
}
