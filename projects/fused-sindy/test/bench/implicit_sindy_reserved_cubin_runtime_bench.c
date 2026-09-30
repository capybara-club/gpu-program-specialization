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

#define IMPLICIT_SINDY_AST_COMPILER_IMPLEMENTATION
#include "implicit_sindy_ast_compiler.h"
#include "stack_ptx_harness_common.h"

#include <cuda.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ISRCB_FEATURES = 32,
    ISRCB_LEAVES = 8,
    ISRCB_DEFAULT_PRIMITIVE_COLS = 16
};

typedef struct {
    size_t kernels_per_module;
    size_t settings;
    size_t rows;
    size_t primitive_cols;
    size_t rhs;
    size_t compile_workspace_mb;
    size_t launch_kernel;
    size_t repeats;
    size_t warmup;
    size_t seed;
    unsigned int sm_major;
    unsigned int sm_minor;
    int sm_set;
    int csv;
    int print_patch_report;
    int require_exact_tail;
    const char* dump_cubin_path;
} IsrcbOptions;

typedef enum {
    ISRCB_AST_MODE_RESERVED = 0,
    ISRCB_AST_MODE_SIMPLE = 1,
    ISRCB_AST_MODE_POLYNOMIAL = 2,
    ISRCB_AST_MODE_RANDOM = 3,
    ISRCB_AST_MODE_MAXIMAL = 4
} IsrcbAstMode;

typedef struct {
    CUdeviceptr primitive;
    CUdeviceptr targets;
    CUdeviceptr leaf_masks;
    CUdeviceptr leaf_words;
    CUdeviceptr gram;
    CUdeviceptr x_sum;
    CUdeviceptr xty;
    CUdeviceptr y_sum;
    CUdeviceptr yy;
} IsrcbDeviceBuffers;

typedef struct {
    IsrcbOptions options;
    CUdevice device;
    CUstream stream;
    int owns_primary;
    unsigned int sm_major;
    unsigned int sm_minor;
    ImplicitSindyAstCompiler* compiler;
    void* module;
    ImplicitSindyGramModule* gram_module;
    BinaryAST* asts;
    void* patched_cubin;
    size_t patched_cubin_bytes;
    float* primitive_host;
    float* targets_host;
    int32_t* leaf_masks_host;
    int32_t* leaf_words_host;
    IsrcbDeviceBuffers device_buffers;
    double create_ms;
    double compile_ms;
    size_t reserved_cubin_bytes;
    IsrcbAstMode ast_mode;
} IsrcbState;

typedef struct {
    double best_ms;
    double mean_ms;
    double stddev_ms;
} IsrcbStats;

static void
isrcb_usage(
    const char* exe
) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --kernels N        reserved gram kernels in the module, 1..64 (default: 4)\n"
        "  --settings N       runtime settings per kernel (default: 2048)\n"
        "  --rows N           rows per setting (default: 131072)\n"
        "  --primitive-cols N primitive feature columns, 2..32 (default: 16)\n"
        "  --rhs N            target RHS count, 1..32 (default: 1)\n"
        "  --mode NAME        reserved, simple, polynomial, random, maximal (default: reserved)\n"
        "  --compile-workspace-mb N single inline compile scratch MiB (default: 512)\n"
        "  --launch-kernel N  launch only one kernel from the module (default: all)\n"
        "  --print-patch-report print per-function reserved/replacement sizes\n"
        "  --require-exact-tail reject replacements smaller than reserved text\n"
        "  --dump-cubin PATH  write loaded cubin bytes before launching\n"
        "  --repeats N        timed repeats (default: 5)\n"
        "  --warmup N         untimed warmups (default: 1)\n"
        "  --seed N           deterministic seed (default: 1)\n"
        "  --sm SM            override compile SM, e.g. 90, sm_90, compute_90\n"
        "  --csv              print one CSV row\n"
        "  -h, --help         show this help\n",
        exe
    );
}

static const char*
isrcb_ast_mode_name(
    IsrcbAstMode mode
) {
    switch (mode) {
        case ISRCB_AST_MODE_SIMPLE: return "simple";
        case ISRCB_AST_MODE_POLYNOMIAL: return "polynomial";
        case ISRCB_AST_MODE_RANDOM: return "random";
        case ISRCB_AST_MODE_MAXIMAL: return "maximal";
        case ISRCB_AST_MODE_RESERVED:
        default: return "reserved";
    }
}

static int
isrcb_parse_ast_mode(
    const char* text,
    IsrcbAstMode* mode_out
) {
    if (text == NULL || mode_out == NULL) return 0;
    if (strcmp(text, "reserved") == 0) *mode_out = ISRCB_AST_MODE_RESERVED;
    else if (strcmp(text, "simple") == 0) *mode_out = ISRCB_AST_MODE_SIMPLE;
    else if (strcmp(text, "polynomial") == 0) *mode_out = ISRCB_AST_MODE_POLYNOMIAL;
    else if (strcmp(text, "random") == 0) *mode_out = ISRCB_AST_MODE_RANDOM;
    else if (strcmp(text, "maximal") == 0) *mode_out = ISRCB_AST_MODE_MAXIMAL;
    else return 0;
    return 1;
}

static int
isrcb_mul_size(
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
isrcb_write_file(
    const char* path,
    const void* data,
    size_t bytes
) {
    FILE* file;
    int ok = 0;

    if (path == NULL || data == NULL || bytes == 0u) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    if (fwrite(data, 1u, bytes, file) == bytes) ok = 1;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static uint32_t
isrcb_u32_mix(
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
isrcb_f32_bits(
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
isrcb_cu(
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
isrcb_device_alloc(
    CUdeviceptr* ptr,
    size_t bytes,
    const char* label
) {
    if (ptr == NULL || bytes == 0u) return 0;
    *ptr = 0;
    if (!isrcb_cu(cuMemAlloc(ptr, bytes), label)) return 0;
    return 1;
}

static int
isrcb_copy_htod(
    CUdeviceptr dst,
    const void* src,
    size_t bytes,
    const char* label
) {
    if (dst == 0 || src == NULL || bytes == 0u) return 0;
    return isrcb_cu(cuMemcpyHtoD(dst, src, bytes), label);
}

static float*
isrcb_device_f32(
    CUdeviceptr ptr
) {
    return (float*)(uintptr_t)ptr;
}

static int32_t*
isrcb_device_i32(
    CUdeviceptr ptr
) {
    return (int32_t*)(uintptr_t)ptr;
}

static int
isrcb_parse_options(
    int argc,
    char** argv,
    IsrcbState* state
) {
    IsrcbOptions* options;
    int i;
    if (state == NULL) return 0;
    options = &state->options;
    memset(options, 0, sizeof(*options));
    options->kernels_per_module = 4u;
    options->settings = 2048u;
    options->rows = 131072u;
    options->primitive_cols = ISRCB_DEFAULT_PRIMITIVE_COLS;
    options->rhs = 1u;
    options->compile_workspace_mb = 512u;
    options->launch_kernel = SIZE_MAX;
    options->repeats = 5u;
    options->warmup = 1u;
    options->seed = 1u;
    state->ast_mode = ISRCB_AST_MODE_RESERVED;

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) return 0;
        if (strcmp(arg, "--csv") == 0) {
            options->csv = 1;
        } else if (strcmp(arg, "--print-patch-report") == 0) {
            options->print_patch_report = 1;
        } else if (strcmp(arg, "--require-exact-tail") == 0) {
            options->require_exact_tail = 1;
        } else if (strcmp(arg, "--dump-cubin") == 0 && i + 1 < argc) {
            options->dump_cubin_path = argv[++i];
        } else if (strcmp(arg, "--kernels") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->kernels_per_module)) return 0;
        } else if (strcmp(arg, "--settings") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->settings)) return 0;
        } else if (strcmp(arg, "--rows") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->rows)) return 0;
        } else if (strcmp(arg, "--primitive-cols") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->primitive_cols)) return 0;
        } else if (strcmp(arg, "--rhs") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->rhs)) return 0;
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            if (!isrcb_parse_ast_mode(argv[++i], &state->ast_mode)) return 0;
        } else if (strcmp(arg, "--compile-workspace-mb") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->compile_workspace_mb)) return 0;
        } else if (strcmp(arg, "--launch-kernel") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->launch_kernel)) return 0;
        } else if (strcmp(arg, "--repeats") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->repeats)) return 0;
        } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->warmup)) return 0;
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_size(argv[++i], &options->seed)) return 0;
        } else if (strcmp(arg, "--sm") == 0 && i + 1 < argc) {
            if (!stack_ptx_test_parse_sm(argv[++i], &options->sm_major, &options->sm_minor)) return 0;
            options->sm_set = 1;
        } else {
            return 0;
        }
    }

    if (options->kernels_per_module == 0u || options->kernels_per_module > 64u) return 0;
    if (options->settings == 0u || options->rows == 0u) return 0;
    if (options->primitive_cols < 2u || options->primitive_cols > 32u) return 0;
    if (options->rhs == 0u || options->rhs > 32u) return 0;
    if (options->compile_workspace_mb == 0u) return 0;
    if (options->launch_kernel != SIZE_MAX && options->launch_kernel >= options->kernels_per_module) return 0;
    if (options->repeats == 0u) return 0;
    return 1;
}

static BinaryAstUnaryOp
isrcb_polynomial_unary_op(
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
isrcb_polynomial_binary_op(
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
isrcb_random_unary_op(
    size_t seed,
    size_t idx,
    size_t unary_idx
) {
    const uint32_t value = isrcb_u32_mix((uint32_t)seed ^ (uint32_t)(idx * 131u + unary_idx * 17u));
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
isrcb_random_binary_op(
    size_t seed,
    size_t idx,
    size_t binary_idx
) {
    const uint32_t value = isrcb_u32_mix((uint32_t)(seed + 0x9e3779b9u) ^ (uint32_t)(idx * 73u + binary_idx * 29u));
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
isrcb_make_ast(
    const IsrcbState* state,
    size_t idx,
    BinaryAST* ast
) {
    size_t i;
    if (state == NULL || ast == NULL) return;
    memset(ast, 0, sizeof(*ast));
    if (state->ast_mode == ISRCB_AST_MODE_SIMPLE) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = BINARY_AST_UNARY_IDENTITY;
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = BINARY_AST_BINARY_ADD_FTZ_F32;
        return;
    }
    if (state->ast_mode == ISRCB_AST_MODE_MAXIMAL) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) ast->unary[i] = BINARY_AST_UNARY_SAFE_EXP_F32;
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) ast->binary[i] = BINARY_AST_BINARY_SAFE_DIV_F32;
        return;
    }
    if (state->ast_mode == ISRCB_AST_MODE_RANDOM) {
        for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
            ast->unary[i] = isrcb_random_unary_op(state->options.seed, idx, i);
        }
        for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
            ast->binary[i] = isrcb_random_binary_op(state->options.seed, idx, i);
        }
        return;
    }
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
        ast->unary[i] = isrcb_polynomial_unary_op(state->options.seed, idx, i);
    }
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        ast->binary[i] = isrcb_polynomial_binary_op(state->options.seed, idx, i);
    }
}

static int
isrcb_prepare_asts(
    IsrcbState* state
) {
    size_t ast_count;
    size_t i;

    if (state == NULL || state->ast_mode == ISRCB_AST_MODE_RESERVED) return 1;
    if (!isrcb_mul_size(state->options.kernels_per_module, ISRCB_FEATURES, &ast_count)) return 0;
    state->asts = (BinaryAST*)calloc(ast_count, sizeof(*state->asts));
    if (state->asts == NULL) return 0;
    for (i = 0u; i < ast_count; ++i) {
        isrcb_make_ast(state, i, &state->asts[i]);
    }
    return 1;
}

static int
isrcb_compile_patched_cubin_inline(
    IsrcbState* state,
    size_t ast_count,
    ImplicitSindyArena* arena,
    void** cubin_out,
    size_t* cubin_bytes_out
) {
    char* eval_ptx = NULL;
    void* object = NULL;
    void* cubin = NULL;
    CubinFunctionPatchReport* reports = NULL;
    size_t eval_ptx_bytes = 0u;
    size_t object_bytes = 0u;
    size_t cubin_capacity = 0u;
    size_t cubin_bytes = 0u;
    CubinFunctionPatchResult patch_result = CUBIN_FUNCTION_PATCH_SUCCESS;
    ImplicitSindyResult result = IMPLICIT_SINDY_SUCCESS;
    size_t i;

    if (state == NULL || arena == NULL || cubin_out == NULL || cubin_bytes_out == NULL) return 0;
    *cubin_out = NULL;
    *cubin_bytes_out = 0u;

    result = implicit_sindy_build_eval_ptx_for_batch(
        state->compiler,
        state->asts,
        ast_count,
        sizeof(state->asts[0]),
        0u,
        arena,
        &eval_ptx,
        &eval_ptx_bytes
    );
    if (result == IMPLICIT_SINDY_SUCCESS) {
        result = implicit_sindy_compile_ptx_to_rdc(
            eval_ptx,
            eval_ptx_bytes,
            state->sm_major,
            state->sm_minor,
            state->compiler->nvptx_options,
            state->compiler->num_nvptx_options,
            arena,
            &object,
            &object_bytes
        );
    }
    if (result == IMPLICIT_SINDY_SUCCESS) {
        patch_result = cubin_function_patch_output_size(state->compiler->patch_handle, &cubin_capacity);
        if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS || cubin_capacity == 0u) {
            result = IMPLICIT_SINDY_ERROR_INTERNAL;
        }
    }
    if (result == IMPLICIT_SINDY_SUCCESS) {
        cubin = malloc(cubin_capacity);
        reports = (CubinFunctionPatchReport*)calloc(state->options.kernels_per_module, sizeof(*reports));
        if (cubin == NULL || reports == NULL) result = IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY;
    }
    if (result == IMPLICIT_SINDY_SUCCESS) {
        patch_result = cubin_function_patch_begin(
            state->compiler->patch_handle,
            cubin,
            cubin_capacity,
            &cubin_bytes
        );
        if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) result = IMPLICIT_SINDY_ERROR_INTERNAL;
    }
    if (result == IMPLICIT_SINDY_SUCCESS) {
        patch_result = cubin_function_patch_apply_all_in_place_ex(
            state->compiler->patch_handle,
            cubin,
            cubin_bytes,
            object,
            object_bytes,
            state->options.require_exact_tail
                ? CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE
                : CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
            reports,
            state->options.kernels_per_module
        );
        if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) result = IMPLICIT_SINDY_ERROR_INTERNAL;
    }

    if (state->options.print_patch_report && reports != NULL) {
        for (i = 0u; i < state->options.kernels_per_module; ++i) {
            fprintf(
                stderr,
                "patch_report[%zu] symbol=%s reserved=%zu replacement=%zu tail=%zu\n",
                i,
                reports[i].symbol_name != NULL ? reports[i].symbol_name : "(null)",
                reports[i].reserved_size,
                reports[i].replacement_size,
                reports[i].reserved_size >= reports[i].replacement_size
                    ? reports[i].reserved_size - reports[i].replacement_size
                    : 0u
            );
        }
    }

    free(reports);
    if (result != IMPLICIT_SINDY_SUCCESS || patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) {
        fprintf(
            stderr,
            "inline patch failed: %s patch=%s\n",
            implicit_sindy_result_to_string(result),
            cubin_function_patch_result_to_string(patch_result)
        );
        free(cubin);
        return 0;
    }

    *cubin_out = cubin;
    *cubin_bytes_out = cubin_bytes;
    return 1;
}

static int
isrcb_prepare_host_inputs(
    IsrcbState* state
) {
    size_t primitive_count;
    size_t target_count;
    size_t leaf_mask_count;
    size_t leaf_word_count;
    size_t row;
    size_t col;
    size_t rhs;
    size_t kernel;
    size_t setting;
    size_t feature;
    size_t leaf;

    if (state == NULL) return 0;
    if (!isrcb_mul_size(state->options.primitive_cols, state->options.rows, &primitive_count)) return 0;
    if (!isrcb_mul_size(state->options.rhs, state->options.rows, &target_count)) return 0;
    if (!isrcb_mul_size(state->options.kernels_per_module, state->options.settings, &leaf_mask_count)) return 0;
    if (!isrcb_mul_size(leaf_mask_count, ISRCB_FEATURES, &leaf_mask_count)) return 0;
    if (!isrcb_mul_size(leaf_mask_count, ISRCB_LEAVES, &leaf_word_count)) return 0;

    state->primitive_host = (float*)calloc(primitive_count, sizeof(*state->primitive_host));
    state->targets_host = (float*)calloc(target_count, sizeof(*state->targets_host));
    state->leaf_masks_host = (int32_t*)calloc(leaf_mask_count, sizeof(*state->leaf_masks_host));
    state->leaf_words_host = (int32_t*)calloc(leaf_word_count, sizeof(*state->leaf_words_host));
    if (state->primitive_host == NULL ||
        state->targets_host == NULL ||
        state->leaf_masks_host == NULL ||
        state->leaf_words_host == NULL) {
        return 0;
    }

    for (col = 0u; col < state->options.primitive_cols; ++col) {
        for (row = 0u; row < state->options.rows; ++row) {
            state->primitive_host[col * state->options.rows + row] =
                0.25f + 0.017f * (float)col + 0.00001f * (float)(row & 1023u) +
                0.0003f * (float)((row * 17u + col * 13u) % 31u);
        }
    }

    for (rhs = 0u; rhs < state->options.rhs; ++rhs) {
        for (row = 0u; row < state->options.rows; ++row) {
            const float x0 = state->primitive_host[row];
            const float x1 = state->primitive_host[state->options.rows + row];
            state->targets_host[rhs * state->options.rows + row] =
                (0.1f + 0.03f * (float)rhs) + 0.7f * x0 - 0.2f * x1 + 0.05f * x0 * x1;
        }
    }

    for (kernel = 0u; kernel < state->options.kernels_per_module; ++kernel) {
        for (setting = 0u; setting < state->options.settings; ++setting) {
            for (feature = 0u; feature < ISRCB_FEATURES; ++feature) {
                uint32_t mask = 0u;
                const size_t setting_feature =
                    (kernel * state->options.settings + setting) * ISRCB_FEATURES + feature;
                for (leaf = 0u; leaf < ISRCB_LEAVES; ++leaf) {
                    const uint32_t value = isrcb_u32_mix(
                        (uint32_t)state->options.seed ^
                        (uint32_t)(kernel * 1000003u + setting * 9176u + feature * 137u + leaf * 17u)
                    );
                    const size_t word_idx = setting_feature * ISRCB_LEAVES + leaf;
                    if ((value % 100u) < 82u) {
                        mask |= 1u << leaf;
                        state->leaf_words_host[word_idx] = (int32_t)(value % (uint32_t)state->options.primitive_cols);
                    } else {
                        const float constant = 0.25f + 0.05f * (float)(value % 9u);
                        state->leaf_words_host[word_idx] = (int32_t)isrcb_f32_bits(constant);
                    }
                }
                state->leaf_masks_host[setting_feature] = (int32_t)mask;
            }
        }
    }

    return 1;
}

static int
isrcb_alloc_device_buffers(
    IsrcbState* state
) {
    size_t count;
    size_t bytes;
    IsrcbDeviceBuffers* buffers;

    if (state == NULL) return 0;
    buffers = &state->device_buffers;

    if (!isrcb_mul_size(state->options.primitive_cols, state->options.rows, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->primitive, bytes, "primitive")) return 0;
    if (!isrcb_copy_htod(buffers->primitive, state->primitive_host, bytes, "primitive copy")) return 0;

    if (!isrcb_mul_size(state->options.rhs, state->options.rows, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->targets, bytes, "targets")) return 0;
    if (!isrcb_copy_htod(buffers->targets, state->targets_host, bytes, "targets copy")) return 0;

    if (!isrcb_mul_size(state->options.kernels_per_module, state->options.settings, &count)) return 0;
    if (!isrcb_mul_size(count, ISRCB_FEATURES, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(int32_t), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->leaf_masks, bytes, "leaf masks")) return 0;
    if (!isrcb_copy_htod(buffers->leaf_masks, state->leaf_masks_host, bytes, "leaf masks copy")) return 0;

    if (!isrcb_mul_size(count, ISRCB_LEAVES, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(int32_t), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->leaf_words, bytes, "leaf words")) return 0;
    if (!isrcb_copy_htod(buffers->leaf_words, state->leaf_words_host, bytes, "leaf words copy")) return 0;

    if (!isrcb_mul_size(state->options.kernels_per_module, state->options.settings, &count)) return 0;
    if (!isrcb_mul_size(count, ISRCB_FEATURES * ISRCB_FEATURES, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->gram, bytes, "gram")) return 0;

    if (!isrcb_mul_size(state->options.kernels_per_module, state->options.settings, &count)) return 0;
    if (!isrcb_mul_size(count, ISRCB_FEATURES, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->x_sum, bytes, "x_sum")) return 0;

    if (!isrcb_mul_size(state->options.kernels_per_module, state->options.settings, &count)) return 0;
    if (!isrcb_mul_size(count, state->options.rhs, &count)) return 0;
    if (!isrcb_mul_size(count, ISRCB_FEATURES, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->xty, bytes, "xty")) return 0;

    if (!isrcb_mul_size(state->options.kernels_per_module, state->options.settings, &count)) return 0;
    if (!isrcb_mul_size(count, state->options.rhs, &count)) return 0;
    if (!isrcb_mul_size(count, sizeof(float), &bytes)) return 0;
    if (!isrcb_device_alloc(&buffers->y_sum, bytes, "y_sum")) return 0;
    if (!isrcb_device_alloc(&buffers->yy, bytes, "yy")) return 0;

    return isrcb_cu(cuStreamSynchronize(state->stream), "initial copy synchronize");
}

static int
isrcb_create_and_load_reserved(
    IsrcbState* state
) {
    ImplicitSindyResult result;
    ImplicitSindyArena arena;
    void* workspace = NULL;
    void* cubins[1];
    size_t cubin_sizes[1];
    size_t ast_count = 0u;
    size_t workspace_bytes = 0u;
    double start_ms;
    int ok = 0;

    if (state == NULL) return 0;
    if (!stack_ptx_test_init_context(&state->device, &state->owns_primary)) return 0;
    if (state->options.sm_set) {
        state->sm_major = state->options.sm_major;
        state->sm_minor = state->options.sm_minor;
    } else if (!stack_ptx_test_device_sm(state->device, &state->sm_major, &state->sm_minor)) {
        return 0;
    }
    if (!isrcb_cu(cuStreamCreate(&state->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")) return 0;

    start_ms = stack_ptx_test_now_ms();
    result = implicit_sindy_ast_compiler_create(
        state->options.kernels_per_module,
        state->sm_major,
        state->sm_minor,
        NULL,
        0u,
        &state->compiler
    );
    state->create_ms = stack_ptx_test_now_ms() - start_ms;
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "compiler create failed: %s\n", implicit_sindy_result_to_string(result));
        return 0;
    }
    if (state->compiler->reserved_cubin == NULL || state->compiler->reserved_cubin_bytes == 0u) return 0;
    state->reserved_cubin_bytes = state->compiler->reserved_cubin_bytes;

    if (state->ast_mode != ISRCB_AST_MODE_RESERVED) {
        if (!isrcb_mul_size(state->options.kernels_per_module, ISRCB_FEATURES, &ast_count)) return 0;
        if (!isrcb_mul_size(state->options.compile_workspace_mb, 1024u * 1024u, &workspace_bytes)) return 0;
        workspace = malloc(workspace_bytes);
        if (workspace == NULL) return 0;
        arena.base = (unsigned char*)workspace;
        arena.capacity = workspace_bytes;
        arena.offset = 0u;
        start_ms = stack_ptx_test_now_ms();
        result = isrcb_compile_patched_cubin_inline(
            state,
            ast_count,
            &arena,
            &state->patched_cubin,
            &state->patched_cubin_bytes
        ) ? IMPLICIT_SINDY_SUCCESS : IMPLICIT_SINDY_ERROR_INTERNAL;
        state->compile_ms = stack_ptx_test_now_ms() - start_ms;
        if (result != IMPLICIT_SINDY_SUCCESS) {
            fprintf(stderr, "inline patch compile failed\n");
            free(workspace);
            return 0;
        }
        cubins[0] = state->patched_cubin;
        cubin_sizes[0] = state->patched_cubin_bytes;
    } else {
        cubins[0] = state->compiler->reserved_cubin;
        cubin_sizes[0] = state->compiler->reserved_cubin_bytes;
    }

    if (state->options.dump_cubin_path != NULL &&
        !isrcb_write_file(state->options.dump_cubin_path, cubins[0], cubin_sizes[0])) {
        fprintf(stderr, "failed to write cubin: %s\n", state->options.dump_cubin_path);
        free(workspace);
        return 0;
    }

    result = implicit_sindy_load_cubin_modules(cubins, cubin_sizes, 1u, &state->module);
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "reserved module load failed: %s\n", implicit_sindy_result_to_string(result));
        free(workspace);
        return 0;
    }

    result = implicit_sindy_gram_module_create(
        state->module,
        state->options.kernels_per_module,
        &state->gram_module
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        fprintf(stderr, "gram module create failed: %s\n", implicit_sindy_result_to_string(result));
        free(workspace);
        return 0;
    }
    ok = 1;
    free(workspace);
    return ok;
}

static int
isrcb_launch_gram_once(
    IsrcbState* state
) {
    size_t kernel;
    size_t kernel_begin;
    size_t kernel_end;
    IsrcbDeviceBuffers* buffers;

    if (state == NULL) return 0;
    buffers = &state->device_buffers;
    if (state->options.launch_kernel == SIZE_MAX) {
        kernel_begin = 0u;
        kernel_end = state->options.kernels_per_module;
    } else {
        kernel_begin = state->options.launch_kernel;
        kernel_end = state->options.launch_kernel + 1u;
    }
    for (kernel = kernel_begin; kernel < kernel_end; ++kernel) {
        const size_t setting_offset = kernel * state->options.settings;
        ImplicitSindyResult result = implicit_sindy_gram_launch(
            state->gram_module,
            kernel,
            state->stream,
            (int64_t)state->options.settings,
            isrcb_device_f32(buffers->primitive),
            (int64_t)state->options.rows,
            (int64_t)state->options.rows,
            (int64_t)state->options.primitive_cols,
            isrcb_device_f32(buffers->targets),
            (int64_t)state->options.rows,
            (int64_t)state->options.rhs,
            isrcb_device_i32(buffers->leaf_masks + setting_offset * ISRCB_FEATURES * sizeof(int32_t)),
            isrcb_device_i32(buffers->leaf_words + setting_offset * ISRCB_FEATURES * ISRCB_LEAVES * sizeof(int32_t)),
            ISRCB_LEAVES,
            isrcb_device_f32(buffers->gram + setting_offset * ISRCB_FEATURES * ISRCB_FEATURES * sizeof(float)),
            ISRCB_FEATURES,
            isrcb_device_f32(buffers->x_sum + setting_offset * ISRCB_FEATURES * sizeof(float)),
            isrcb_device_f32(buffers->xty + setting_offset * state->options.rhs * ISRCB_FEATURES * sizeof(float)),
            ISRCB_FEATURES,
            isrcb_device_f32(buffers->y_sum + setting_offset * state->options.rhs * sizeof(float)),
            isrcb_device_f32(buffers->yy + setting_offset * state->options.rhs * sizeof(float))
        );
        if (result != IMPLICIT_SINDY_SUCCESS) {
            fprintf(stderr, "gram launch failed: %s\n", implicit_sindy_result_to_string(result));
            return 0;
        }
    }
    return 1;
}

static int
isrcb_event_create(
    CUevent* event
) {
    return event != NULL && isrcb_cu(cuEventCreate(event, CU_EVENT_DEFAULT), "cuEventCreate");
}

static int
isrcb_run_once(
    IsrcbState* state,
    double* out_ms
) {
    CUevent start = NULL;
    CUevent end = NULL;
    float elapsed_ms = 0.0f;
    int ok = 0;

    if (state == NULL || out_ms == NULL) return 0;
    if (!isrcb_event_create(&start) || !isrcb_event_create(&end)) {
        if (start != NULL) (void)cuEventDestroy(start);
        if (end != NULL) (void)cuEventDestroy(end);
        return 0;
    }

    if (isrcb_cu(cuEventRecord(start, state->stream), "record start") &&
        isrcb_launch_gram_once(state) &&
        isrcb_cu(cuEventRecord(end, state->stream), "record end") &&
        isrcb_cu(cuEventSynchronize(end), "cuEventSynchronize") &&
        isrcb_cu(cuEventElapsedTime(&elapsed_ms, start, end), "cuEventElapsedTime")) {
        *out_ms = (double)elapsed_ms;
        ok = 1;
    }

    (void)cuEventDestroy(end);
    (void)cuEventDestroy(start);
    return ok;
}

static int
isrcb_run_benchmark(
    IsrcbState* state,
    IsrcbStats* stats
) {
    double* times;
    double sum = 0.0;
    double variance_sum = 0.0;
    size_t i;

    if (state == NULL || stats == NULL) return 0;
    memset(stats, 0, sizeof(*stats));
    stats->best_ms = DBL_MAX;

    for (i = 0u; i < state->options.warmup; ++i) {
        double ignored_ms = 0.0;
        if (!isrcb_run_once(state, &ignored_ms)) return 0;
    }

    times = (double*)calloc(state->options.repeats, sizeof(*times));
    if (times == NULL) return 0;
    for (i = 0u; i < state->options.repeats; ++i) {
        if (!isrcb_run_once(state, &times[i])) {
            free(times);
            return 0;
        }
        if (times[i] < stats->best_ms) stats->best_ms = times[i];
        sum += times[i];
    }
    stats->mean_ms = sum / (double)state->options.repeats;
    for (i = 0u; i < state->options.repeats; ++i) {
        const double diff = times[i] - stats->mean_ms;
        variance_sum += diff * diff;
    }
    stats->stddev_ms = sqrt(variance_sum / (double)state->options.repeats);
    free(times);
    return 1;
}

static void
isrcb_print_results(
    const IsrcbState* state,
    const IsrcbStats* stats
) {
    const double cohorts = state->options.launch_kernel == SIZE_MAX ? (double)state->options.kernels_per_module : 1.0;
    const double total_settings = cohorts * (double)state->options.settings;
    const double row_settings = total_settings * (double)state->options.rows;
    const double feature_row_settings = row_settings * (double)ISRCB_FEATURES;
    const double settings_per_sec = stats->best_ms > 0.0 ? 1000.0 * total_settings / stats->best_ms : 0.0;
    const double row_settings_per_sec = stats->best_ms > 0.0 ? 1000.0 * row_settings / stats->best_ms : 0.0;
    const double feature_row_settings_per_sec =
        stats->best_ms > 0.0 ? 1000.0 * feature_row_settings / stats->best_ms : 0.0;
    const size_t cubin_bytes =
        state->ast_mode == ISRCB_AST_MODE_RESERVED ? state->reserved_cubin_bytes : state->patched_cubin_bytes;
    char launch_kernel_text[32];

    if (state->options.launch_kernel == SIZE_MAX) {
        strcpy(launch_kernel_text, "all");
    } else {
        snprintf(launch_kernel_text, sizeof(launch_kernel_text), "%zu", state->options.launch_kernel);
    }

    if (state->options.csv) {
        printf(
            "kind,kernels,launch_kernel,settings,rows,primitive_cols,rhs,sm,warmup,repeats,"
            "create_ms,compile_ms,cubin_bytes,best_ms,mean_ms,stddev_ms,settings_per_sec,"
            "row_settings_per_sec,feature_row_settings_per_sec\n"
        );
        printf(
            "%s,%zu,%s,%zu,%zu,%zu,%zu,%u%u,%zu,%zu,%.3f,%.3f,%zu,"
            "%.3f,%.3f,%.3f,%.3f,%.3e,%.3e\n",
            isrcb_ast_mode_name(state->ast_mode),
            state->options.kernels_per_module,
            launch_kernel_text,
            state->options.settings,
            state->options.rows,
            state->options.primitive_cols,
            state->options.rhs,
            state->sm_major,
            state->sm_minor,
            state->options.warmup,
            state->options.repeats,
            state->create_ms,
            state->compile_ms,
            cubin_bytes,
            stats->best_ms,
            stats->mean_ms,
            stats->stddev_ms,
            settings_per_sec,
            row_settings_per_sec,
            feature_row_settings_per_sec
        );
        return;
    }

    printf(
        "implicit_sindy_reserved_cubin_runtime mode=%s kernels=%zu launch_kernel=%s "
        "settings=%zu rows=%zu primitive_cols=%zu rhs=%zu sm_%u%u warmup=%zu repeats=%zu\n",
        isrcb_ast_mode_name(state->ast_mode),
        state->options.kernels_per_module,
        launch_kernel_text,
        state->options.settings,
        state->options.rows,
        state->options.primitive_cols,
        state->options.rhs,
        state->sm_major,
        state->sm_minor,
        state->options.warmup,
        state->options.repeats
    );
    printf(
        "  create_ms=%.3f compile_ms=%.3f cubin_bytes=%zu reserved_cubin_bytes=%zu\n",
        state->create_ms,
        state->compile_ms,
        cubin_bytes,
        state->reserved_cubin_bytes
    );
    printf(
        "  run_ms_best=%.3f run_ms_mean=%.3f run_ms_stddev=%.3f\n",
        stats->best_ms,
        stats->mean_ms,
        stats->stddev_ms
    );
    printf(
        "  throughput: settings_per_sec=%.3f row_settings_per_sec=%.3e "
        "feature_row_settings_per_sec=%.3e\n",
        settings_per_sec,
        row_settings_per_sec,
        feature_row_settings_per_sec
    );
}

static void
isrcb_free_device_buffers(
    IsrcbDeviceBuffers* buffers
) {
    if (buffers == NULL) return;
    if (buffers->yy != 0) (void)cuMemFree(buffers->yy);
    if (buffers->y_sum != 0) (void)cuMemFree(buffers->y_sum);
    if (buffers->xty != 0) (void)cuMemFree(buffers->xty);
    if (buffers->x_sum != 0) (void)cuMemFree(buffers->x_sum);
    if (buffers->gram != 0) (void)cuMemFree(buffers->gram);
    if (buffers->leaf_words != 0) (void)cuMemFree(buffers->leaf_words);
    if (buffers->leaf_masks != 0) (void)cuMemFree(buffers->leaf_masks);
    if (buffers->targets != 0) (void)cuMemFree(buffers->targets);
    if (buffers->primitive != 0) (void)cuMemFree(buffers->primitive);
    memset(buffers, 0, sizeof(*buffers));
}

static void
isrcb_cleanup(
    IsrcbState* state
) {
    if (state == NULL) return;
    if (state->stream != NULL) (void)cuStreamSynchronize(state->stream);
    isrcb_free_device_buffers(&state->device_buffers);
    if (state->gram_module != NULL) (void)implicit_sindy_gram_module_destroy(state->gram_module);
    if (state->module != NULL) (void)implicit_sindy_unload_modules(&state->module, 1u);
    free(state->patched_cubin);
    if (state->compiler != NULL) (void)implicit_sindy_ast_compiler_destroy(state->compiler);
    if (state->stream != NULL) (void)cuStreamDestroy(state->stream);
    stack_ptx_test_release_context(state->device, state->owns_primary);
    free(state->leaf_words_host);
    free(state->leaf_masks_host);
    free(state->targets_host);
    free(state->primitive_host);
    free(state->asts);
}

static int
isrcb_main_impl(
    IsrcbState* state,
    IsrcbStats* stats
) {
    if (state == NULL || stats == NULL) return 0;
    if (!isrcb_prepare_host_inputs(state)) return 0;
    if (!isrcb_prepare_asts(state)) return 0;
    if (!isrcb_create_and_load_reserved(state)) return 0;
    if (!isrcb_alloc_device_buffers(state)) return 0;
    if (!isrcb_run_benchmark(state, stats)) return 0;
    return 1;
}

int
main(
    int argc,
    char** argv
) {
    IsrcbState state;
    IsrcbStats stats;
    int ok;

    memset(&state, 0, sizeof(state));
    memset(&stats, 0, sizeof(stats));
    if (!isrcb_parse_options(argc, argv, &state)) {
        isrcb_usage(argv[0]);
        return 1;
    }

    ok = isrcb_main_impl(&state, &stats);
    if (ok) isrcb_print_results(&state, &stats);
    isrcb_cleanup(&state);
    return ok ? 0 : 1;
}
