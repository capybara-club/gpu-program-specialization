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
#include "implicit_sindy_cpu_gram_reference.h"
#include "stack_ptx_harness_common.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ISGCRT_FEATURES = 32,
    ISGCRT_LEAVES = 8,
    ISGCRT_SETTINGS = 3,
    ISGCRT_ROWS = 259,
    ISGCRT_PRIMITIVE_COLS = 9,
    ISGCRT_PRIMITIVE_STRIDE = 267,
    ISGCRT_RHS = 2,
    ISGCRT_TARGET_STRIDE = 263,
    ISGCRT_WORKERS = 2,
    ISGCRT_SCRATCH_BYTES_PER_WORKER = 64 * 1024 * 1024
};

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
} IsgcrtDeviceBuffers;

typedef struct {
    BinaryAST* asts;
    float* primitive;
    float* targets;
    int32_t* leaf_masks;
    int32_t* leaf_words;
    float* expected_gram;
    float* expected_x_sum;
    float* expected_xty;
    float* expected_y_sum;
    float* expected_yy;
    float* actual_gram;
    float* actual_x_sum;
    float* actual_xty;
    float* actual_y_sum;
    float* actual_yy;
    void* workspace;
    void** cubins;
    size_t* cubin_sizes;
    void** modules;
    IsgcrtDeviceBuffers device;
    ImplicitSindyAstCompiler* compiler;
    ImplicitSindyGramModule* gram_module;
} IsgcrtState;

static int
isgcrt_check(
    int condition,
    const char* label
) {
    if (condition) return 1;
    fprintf(stderr, "check failed: %s\n", label);
    return 0;
}

static void
isgcrt_stage(
    const char* label
) {
    if (getenv("IMPLICIT_SINDY_TEST_VERBOSE") == NULL) return;
    fprintf(stderr, "stage: %s\n", label);
    fflush(stderr);
}

static float*
isgcrt_device_f32(
    CUdeviceptr ptr
) {
    return (float*)(uintptr_t)ptr;
}

static int32_t*
isgcrt_device_i32(
    CUdeviceptr ptr
) {
    return (int32_t*)(uintptr_t)ptr;
}

static int32_t
isgcrt_f32_constant_word(
    float value
) {
    union {
        float f;
        int32_t i;
    } bits;
    bits.f = value;
    return bits.i;
}

static int
isgcrt_cuda(
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

static void
isgcrt_make_ast(
    size_t idx,
    BinaryAST* ast
) {
    static const BinaryAstUnaryOp unary_ops[] = {
        BINARY_AST_UNARY_IDENTITY,
        BINARY_AST_UNARY_SQUARE_F32,
        BINARY_AST_UNARY_CUBE_F32,
        BINARY_AST_UNARY_NEG_FTZ_F32,
        BINARY_AST_UNARY_ABS_FTZ_F32
    };
    static const BinaryAstBinaryOp binary_ops[] = {
        BINARY_AST_BINARY_ADD_FTZ_F32,
        BINARY_AST_BINARY_SUB_FTZ_F32,
        BINARY_AST_BINARY_MUL_FTZ_F32,
        BINARY_AST_BINARY_KEEP_LEFT,
        BINARY_AST_BINARY_KEEP_RIGHT,
        BINARY_AST_BINARY_MIN_FTZ_F32,
        BINARY_AST_BINARY_MAX_FTZ_F32
    };
    size_t i;

    memset(ast, 0, sizeof(*ast));
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
        ast->unary[i] = unary_ops[(idx + i * 3u) % (sizeof(unary_ops) / sizeof(unary_ops[0]))];
    }
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        ast->binary[i] = binary_ops[(idx * 5u + i * 2u) % (sizeof(binary_ops) / sizeof(binary_ops[0]))];
    }
}

static void
isgcrt_fill_inputs(
    IsgcrtState* state,
    size_t ast_count
) {
    size_t i;
    int setting;
    int feature;
    int leaf;
    int col;
    int row;
    int rhs;

    for (i = 0u; i < ast_count; ++i) {
        isgcrt_make_ast(i, state->asts + i);
    }

    for (col = 0; col < ISGCRT_PRIMITIVE_COLS; ++col) {
        for (row = 0; row < ISGCRT_PRIMITIVE_STRIDE; ++row) {
            float value = 0.0f;
            if (row < ISGCRT_ROWS) {
                const float centered = (float)(row - 129) * 0.0025f;
                const float ripple = (float)((row + col * 7) % 19) * 0.00125f;
                value = centered + ripple + (float)col * 0.03125f;
            }
            state->primitive[row + col * ISGCRT_PRIMITIVE_STRIDE] = value;
        }
    }

    for (rhs = 0; rhs < ISGCRT_RHS; ++rhs) {
        for (row = 0; row < ISGCRT_TARGET_STRIDE; ++row) {
            float value = 0.0f;
            if (row < ISGCRT_ROWS) {
                const float x0 = state->primitive[row];
                const float x1 = state->primitive[row + ISGCRT_PRIMITIVE_STRIDE];
                const float x2 = state->primitive[row + 2 * ISGCRT_PRIMITIVE_STRIDE];
                value = rhs == 0
                    ? x0 + 0.25f * x1 * x2
                    : x2 - 0.5f * x1;
            }
            state->targets[row + rhs * ISGCRT_TARGET_STRIDE] = value;
        }
    }

    for (setting = 0; setting < ISGCRT_SETTINGS; ++setting) {
        for (feature = 0; feature < ISGCRT_FEATURES; ++feature) {
            uint32_t mask = 0u;
            const int feature_offset = setting * ISGCRT_FEATURES + feature;
            for (leaf = 0; leaf < ISGCRT_LEAVES; ++leaf) {
                const int word_offset = feature_offset * ISGCRT_LEAVES + leaf;
                if (((setting + feature * 3 + leaf) % 4) != 0) {
                    mask |= 1u << leaf;
                    state->leaf_words[word_offset] =
                        (int32_t)((setting + feature + leaf * 2) % ISGCRT_PRIMITIVE_COLS);
                } else {
                    const float value =
                        -0.20f + 0.025f * (float)((setting * 11 + feature * 5 + leaf * 3) % 17);
                    state->leaf_words[word_offset] = isgcrt_f32_constant_word(value);
                }
            }
            state->leaf_masks[feature_offset] = (int32_t)mask;
        }
    }
}

static int
isgcrt_alloc_host(
    IsgcrtState* state,
    size_t ast_count
) {
    const size_t stats_count = ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_FEATURES;
    const size_t x_count = ISGCRT_SETTINGS * ISGCRT_FEATURES;
    const size_t rhs_stats_count = ISGCRT_SETTINGS * ISGCRT_RHS;
    const size_t xty_count = rhs_stats_count * ISGCRT_FEATURES;

    state->asts = (BinaryAST*)calloc(ast_count, sizeof(state->asts[0]));
    state->primitive = (float*)calloc(ISGCRT_PRIMITIVE_COLS * ISGCRT_PRIMITIVE_STRIDE, sizeof(float));
    state->targets = (float*)calloc(ISGCRT_RHS * ISGCRT_TARGET_STRIDE, sizeof(float));
    state->leaf_masks = (int32_t*)calloc(ISGCRT_SETTINGS * ISGCRT_FEATURES, sizeof(int32_t));
    state->leaf_words = (int32_t*)calloc(ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_LEAVES, sizeof(int32_t));
    state->expected_gram = (float*)calloc(stats_count, sizeof(float));
    state->expected_x_sum = (float*)calloc(x_count, sizeof(float));
    state->expected_xty = (float*)calloc(xty_count, sizeof(float));
    state->expected_y_sum = (float*)calloc(rhs_stats_count, sizeof(float));
    state->expected_yy = (float*)calloc(rhs_stats_count, sizeof(float));
    state->actual_gram = (float*)calloc(stats_count, sizeof(float));
    state->actual_x_sum = (float*)calloc(x_count, sizeof(float));
    state->actual_xty = (float*)calloc(xty_count, sizeof(float));
    state->actual_y_sum = (float*)calloc(rhs_stats_count, sizeof(float));
    state->actual_yy = (float*)calloc(rhs_stats_count, sizeof(float));

    return isgcrt_check(
        state->asts != NULL &&
        state->primitive != NULL &&
        state->targets != NULL &&
        state->leaf_masks != NULL &&
        state->leaf_words != NULL &&
        state->expected_gram != NULL &&
        state->expected_x_sum != NULL &&
        state->expected_xty != NULL &&
        state->expected_y_sum != NULL &&
        state->expected_yy != NULL &&
        state->actual_gram != NULL &&
        state->actual_x_sum != NULL &&
        state->actual_xty != NULL &&
        state->actual_y_sum != NULL &&
        state->actual_yy != NULL,
        "host allocations"
    );
}

static int
isgcrt_alloc_device(
    IsgcrtState* state
) {
    const size_t gram_bytes = ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_FEATURES * sizeof(float);
    const size_t x_sum_bytes = ISGCRT_SETTINGS * ISGCRT_FEATURES * sizeof(float);
    const size_t rhs_stats_bytes = ISGCRT_SETTINGS * ISGCRT_RHS * sizeof(float);
    const size_t xty_bytes = ISGCRT_SETTINGS * ISGCRT_RHS * ISGCRT_FEATURES * sizeof(float);

    if (!isgcrt_cuda(cuMemAlloc(
            &state->device.primitive,
            ISGCRT_PRIMITIVE_COLS * ISGCRT_PRIMITIVE_STRIDE * sizeof(float)
        ), "cuMemAlloc primitive")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(
            &state->device.targets,
            ISGCRT_RHS * ISGCRT_TARGET_STRIDE * sizeof(float)
        ), "cuMemAlloc targets")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(
            &state->device.leaf_masks,
            ISGCRT_SETTINGS * ISGCRT_FEATURES * sizeof(int32_t)
        ), "cuMemAlloc leaf_masks")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(
            &state->device.leaf_words,
            ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_LEAVES * sizeof(int32_t)
        ), "cuMemAlloc leaf_words")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(&state->device.gram, gram_bytes), "cuMemAlloc gram")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(&state->device.x_sum, x_sum_bytes), "cuMemAlloc x_sum")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(&state->device.xty, xty_bytes), "cuMemAlloc xty")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(&state->device.y_sum, rhs_stats_bytes), "cuMemAlloc y_sum")) return 0;
    if (!isgcrt_cuda(cuMemAlloc(&state->device.yy, rhs_stats_bytes), "cuMemAlloc yy")) return 0;
    return 1;
}

static int
isgcrt_copy_inputs_to_device(
    IsgcrtState* state
) {
    if (!isgcrt_cuda(cuMemcpyHtoD(
            state->device.primitive,
            state->primitive,
            ISGCRT_PRIMITIVE_COLS * ISGCRT_PRIMITIVE_STRIDE * sizeof(float)
        ), "cuMemcpyHtoD primitive")) return 0;
    if (!isgcrt_cuda(cuMemcpyHtoD(
            state->device.targets,
            state->targets,
            ISGCRT_RHS * ISGCRT_TARGET_STRIDE * sizeof(float)
        ), "cuMemcpyHtoD targets")) return 0;
    if (!isgcrt_cuda(cuMemcpyHtoD(
            state->device.leaf_masks,
            state->leaf_masks,
            ISGCRT_SETTINGS * ISGCRT_FEATURES * sizeof(int32_t)
        ), "cuMemcpyHtoD leaf_masks")) return 0;
    if (!isgcrt_cuda(cuMemcpyHtoD(
            state->device.leaf_words,
            state->leaf_words,
            ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_LEAVES * sizeof(int32_t)
        ), "cuMemcpyHtoD leaf_words")) return 0;
    return 1;
}

static int
isgcrt_compile(
    IsgcrtState* state,
    size_t kernels_per_module,
    size_t ast_count,
    unsigned int sm_major,
    unsigned int sm_minor
) {
    ImplicitSindyResult result;
    size_t workspace_bytes = 0u;
    size_t cubin_count;

    result = implicit_sindy_ast_compiler_create(
        kernels_per_module,
        sm_major,
        sm_minor,
        NULL,
        0u,
        &state->compiler
    );
    if (!isgcrt_check(result == IMPLICIT_SINDY_SUCCESS, "compiler create")) return 0;

    result = implicit_sindy_ast_compile_workspace_size(
        state->compiler,
        ast_count,
        ISGCRT_WORKERS,
        ISGCRT_SCRATCH_BYTES_PER_WORKER,
        &workspace_bytes
    );
    if (!isgcrt_check(result == IMPLICIT_SINDY_SUCCESS, "workspace size")) return 0;

    state->workspace = malloc(workspace_bytes);
    if (!isgcrt_check(state->workspace != NULL, "workspace allocation")) return 0;

    cubin_count = implicit_sindy_ast_cubin_count(
        ast_count,
        kernels_per_module
    );
    if (!isgcrt_check(cubin_count == 1u, "one cubin expected")) return 0;

    state->cubins = (void**)calloc(cubin_count, sizeof(state->cubins[0]));
    state->cubin_sizes = (size_t*)calloc(cubin_count, sizeof(state->cubin_sizes[0]));
    state->modules = (void**)calloc(cubin_count, sizeof(state->modules[0]));
    if (!isgcrt_check(
            state->cubins != NULL && state->cubin_sizes != NULL && state->modules != NULL,
            "cubin/module arrays")) return 0;

    result = implicit_sindy_ast_compile_cubins(
        state->compiler,
        state->asts,
        ast_count,
        sizeof(state->asts[0]),
        ISGCRT_WORKERS,
        state->workspace,
        workspace_bytes,
        state->cubins,
        state->cubin_sizes
    );
    if (!isgcrt_check(result == IMPLICIT_SINDY_SUCCESS, "compile cubins")) return 0;
    if (!isgcrt_check(state->cubins[0] != NULL && state->cubin_sizes[0] != 0u, "compiled cubin")) return 0;

    result = implicit_sindy_load_cubin_modules(
        state->cubins,
        state->cubin_sizes,
        cubin_count,
        state->modules
    );
    if (!isgcrt_check(result == IMPLICIT_SINDY_SUCCESS, "load cubin modules")) return 0;

    result = implicit_sindy_gram_module_create(
        state->modules[0],
        kernels_per_module,
        &state->gram_module
    );
    if (!isgcrt_check(result == IMPLICIT_SINDY_SUCCESS, "gram module create")) return 0;

    return 1;
}

static int
isgcrt_copy_outputs_to_host(
    IsgcrtState* state
) {
    if (!isgcrt_cuda(cuMemcpyDtoH(
            state->actual_gram,
            state->device.gram,
            ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_FEATURES * sizeof(float)
        ), "cuMemcpyDtoH gram")) return 0;
    if (!isgcrt_cuda(cuMemcpyDtoH(
            state->actual_x_sum,
            state->device.x_sum,
            ISGCRT_SETTINGS * ISGCRT_FEATURES * sizeof(float)
        ), "cuMemcpyDtoH x_sum")) return 0;
    if (!isgcrt_cuda(cuMemcpyDtoH(
            state->actual_xty,
            state->device.xty,
            ISGCRT_SETTINGS * ISGCRT_RHS * ISGCRT_FEATURES * sizeof(float)
        ), "cuMemcpyDtoH xty")) return 0;
    if (!isgcrt_cuda(cuMemcpyDtoH(
            state->actual_y_sum,
            state->device.y_sum,
            ISGCRT_SETTINGS * ISGCRT_RHS * sizeof(float)
        ), "cuMemcpyDtoH y_sum")) return 0;
    if (!isgcrt_cuda(cuMemcpyDtoH(
            state->actual_yy,
            state->device.yy,
            ISGCRT_SETTINGS * ISGCRT_RHS * sizeof(float)
        ), "cuMemcpyDtoH yy")) return 0;
    return 1;
}

static int
isgcrt_compare_array(
    const char* label,
    const float* expected,
    const float* actual,
    size_t count,
    float abs_tol,
    float rel_tol
) {
    size_t i;
    float max_diff = 0.0f;
    size_t max_idx = 0u;

    for (i = 0u; i < count; ++i) {
        const float diff = fabsf(expected[i] - actual[i]);
        const float limit = abs_tol + rel_tol * fabsf(expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (!(diff <= limit)) {
            fprintf(
                stderr,
                "%s mismatch at %zu: expected %.9g actual %.9g diff %.9g limit %.9g\n",
                label,
                i,
                expected[i],
                actual[i],
                diff,
                limit
            );
            return 0;
        }
    }

    printf("%s max_diff %.9g at %zu\n", label, max_diff, max_idx);
    return 1;
}

static int
isgcrt_run_kernel_compare(
    IsgcrtState* state,
    size_t kernel_index
) {
    const size_t ast_offset = kernel_index * ISGCRT_FEATURES;
    const size_t gram_count = ISGCRT_SETTINGS * ISGCRT_FEATURES * ISGCRT_FEATURES;
    const size_t x_count = ISGCRT_SETTINGS * ISGCRT_FEATURES;
    const size_t rhs_count = ISGCRT_SETTINGS * ISGCRT_RHS;
    const size_t xty_count = rhs_count * ISGCRT_FEATURES;
    ImplicitSindyResult result;
    int reference_result;

    memset(state->expected_gram, 0, gram_count * sizeof(float));
    memset(state->expected_x_sum, 0, x_count * sizeof(float));
    memset(state->expected_xty, 0, xty_count * sizeof(float));
    memset(state->expected_y_sum, 0, rhs_count * sizeof(float));
    memset(state->expected_yy, 0, rhs_count * sizeof(float));
    memset(state->actual_gram, 0, gram_count * sizeof(float));
    memset(state->actual_x_sum, 0, x_count * sizeof(float));
    memset(state->actual_xty, 0, xty_count * sizeof(float));
    memset(state->actual_y_sum, 0, rhs_count * sizeof(float));
    memset(state->actual_yy, 0, rhs_count * sizeof(float));

    isgcrt_stage("cpu gram reference");
    reference_result = implicit_sindy_cpu_gram_reference_stats_many_settings(
        state->asts + ast_offset,
        ISGCRT_FEATURES,
        ISGCRT_SETTINGS,
        state->primitive,
        ISGCRT_ROWS,
        ISGCRT_PRIMITIVE_STRIDE,
        ISGCRT_PRIMITIVE_COLS,
        state->targets,
        ISGCRT_TARGET_STRIDE,
        ISGCRT_RHS,
        state->leaf_masks,
        state->leaf_words,
        ISGCRT_LEAVES,
        state->expected_gram,
        ISGCRT_FEATURES,
        state->expected_x_sum,
        state->expected_xty,
        ISGCRT_FEATURES,
        state->expected_y_sum,
        state->expected_yy
    );
    if (!isgcrt_check(
            reference_result == IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS,
            "CPU gram reference")) return 0;

    isgcrt_stage("gram launch");
    result = implicit_sindy_gram_launch(
        state->gram_module,
        kernel_index,
        NULL,
        ISGCRT_SETTINGS,
        isgcrt_device_f32(state->device.primitive),
        ISGCRT_ROWS,
        ISGCRT_PRIMITIVE_STRIDE,
        ISGCRT_PRIMITIVE_COLS,
        isgcrt_device_f32(state->device.targets),
        ISGCRT_TARGET_STRIDE,
        ISGCRT_RHS,
        isgcrt_device_i32(state->device.leaf_masks),
        isgcrt_device_i32(state->device.leaf_words),
        ISGCRT_LEAVES,
        isgcrt_device_f32(state->device.gram),
        ISGCRT_FEATURES,
        isgcrt_device_f32(state->device.x_sum),
        isgcrt_device_f32(state->device.xty),
        ISGCRT_FEATURES,
        isgcrt_device_f32(state->device.y_sum),
        isgcrt_device_f32(state->device.yy)
    );
    if (!isgcrt_check(result == IMPLICIT_SINDY_SUCCESS, "gram launch")) return 0;
    isgcrt_stage("context synchronize");
    if (!isgcrt_cuda(cuCtxSynchronize(), "cuCtxSynchronize")) return 0;
    isgcrt_stage("copy outputs");
    if (!isgcrt_copy_outputs_to_host(state)) return 0;

    if (!isgcrt_compare_array("gram", state->expected_gram, state->actual_gram, gram_count, 3.0e-3f, 4.0e-4f)) return 0;
    if (!isgcrt_compare_array("x_sum", state->expected_x_sum, state->actual_x_sum, x_count, 2.0e-4f, 2.0e-4f)) return 0;
    if (!isgcrt_compare_array("xty", state->expected_xty, state->actual_xty, xty_count, 6.0e-4f, 4.0e-4f)) return 0;
    if (!isgcrt_compare_array("y_sum", state->expected_y_sum, state->actual_y_sum, rhs_count, 2.0e-5f, 2.0e-5f)) return 0;
    if (!isgcrt_compare_array("yy", state->expected_yy, state->actual_yy, rhs_count, 2.0e-5f, 2.0e-5f)) return 0;
    return 1;
}

static void
isgcrt_destroy_state(
    IsgcrtState* state
) {
    if (state == NULL) return;
    if (state->gram_module != NULL) {
        (void)implicit_sindy_gram_module_destroy(state->gram_module);
    }
    if (state->modules != NULL) {
        (void)implicit_sindy_unload_modules(state->modules, 1u);
    }
    if (state->cubins != NULL || state->cubin_sizes != NULL) {
        implicit_sindy_free_cubins(state->cubins, state->cubin_sizes, 1u);
    }
    if (state->compiler != NULL) {
        (void)implicit_sindy_ast_compiler_destroy(state->compiler);
    }
    if (state->device.primitive != 0) (void)cuMemFree(state->device.primitive);
    if (state->device.targets != 0) (void)cuMemFree(state->device.targets);
    if (state->device.leaf_masks != 0) (void)cuMemFree(state->device.leaf_masks);
    if (state->device.leaf_words != 0) (void)cuMemFree(state->device.leaf_words);
    if (state->device.gram != 0) (void)cuMemFree(state->device.gram);
    if (state->device.x_sum != 0) (void)cuMemFree(state->device.x_sum);
    if (state->device.xty != 0) (void)cuMemFree(state->device.xty);
    if (state->device.y_sum != 0) (void)cuMemFree(state->device.y_sum);
    if (state->device.yy != 0) (void)cuMemFree(state->device.yy);
    free(state->workspace);
    free(state->cubins);
    free(state->cubin_sizes);
    free(state->modules);
    free(state->asts);
    free(state->primitive);
    free(state->targets);
    free(state->leaf_masks);
    free(state->leaf_words);
    free(state->expected_gram);
    free(state->expected_x_sum);
    free(state->expected_xty);
    free(state->expected_y_sum);
    free(state->expected_yy);
    free(state->actual_gram);
    free(state->actual_x_sum);
    free(state->actual_xty);
    free(state->actual_y_sum);
    free(state->actual_yy);
}

static int
isgcrt_run_case_impl(
    size_t kernels_per_module,
    size_t kernel_index,
    unsigned int sm_major,
    unsigned int sm_minor
) {
    IsgcrtState state;
    const size_t ast_count = kernels_per_module * ISGCRT_FEATURES;
    int ok;

    memset(&state, 0, sizeof(state));
    isgcrt_stage("alloc host");
    ok = isgcrt_alloc_host(&state, ast_count);
    if (ok) {
        isgcrt_stage("alloc device");
        ok = isgcrt_alloc_device(&state);
    }
    if (ok) {
        isgcrt_stage("fill inputs");
        isgcrt_fill_inputs(&state, ast_count);
    }
    if (ok) {
        isgcrt_stage("copy inputs");
        ok = isgcrt_copy_inputs_to_device(&state);
    }
    if (ok) {
        isgcrt_stage("compile");
        ok = isgcrt_compile(&state, kernels_per_module, ast_count, sm_major, sm_minor);
    }
    if (ok) {
        isgcrt_stage("run compare");
        ok = isgcrt_run_kernel_compare(&state, kernel_index);
    }
    isgcrt_destroy_state(&state);
    return ok;
}

int
main(void)
{
    CUdevice device;
    int owns_primary = 0;
    unsigned int sm_major = 0u;
    unsigned int sm_minor = 0u;

    if (!stack_ptx_test_init_context(&device, &owns_primary)) return 1;
    if (!stack_ptx_test_device_sm(device, &sm_major, &sm_minor)) {
        stack_ptx_test_release_context(device, owns_primary);
        return 1;
    }

    if (!isgcrt_run_case_impl(1u, 0u, sm_major, sm_minor)) {
        stack_ptx_test_release_context(device, owns_primary);
        return 1;
    }
    if (!isgcrt_run_case_impl(4u, 0u, sm_major, sm_minor)) {
        stack_ptx_test_release_context(device, owns_primary);
        return 1;
    }
    if (!isgcrt_run_case_impl(4u, 3u, sm_major, sm_minor)) {
        stack_ptx_test_release_context(device, owns_primary);
        return 1;
    }

    stack_ptx_test_release_context(device, owns_primary);
    printf("PASS implicit_sindy_gram_cpu_reference_test\n");
    return 0;
}
