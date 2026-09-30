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
    ISACRT_FEATURES = 32,
    ISACRT_LEAVES = 8,
    ISACRT_SETTINGS = 3,
    ISACRT_ROWS = 257,
    ISACRT_PRIMITIVE_COLS = 10,
    ISACRT_PRIMITIVE_STRIDE = 263,
    ISACRT_SINGLE_KERNELS_PER_MODULE = 4,
    ISACRT_COLUMNS_KERNELS_PER_MODULE = 1,
    ISACRT_WORKERS = 4,
    ISACRT_SCRATCH_BYTES_PER_WORKER = 64 * 1024 * 1024
};

typedef struct {
    CUdeviceptr primitive;
    CUdeviceptr leaf_masks;
    CUdeviceptr leaf_words;
    CUdeviceptr single_columns;
    CUdeviceptr tiled_columns;
} IsacrtDeviceBuffers;

typedef struct {
    BinaryAST asts[ISACRT_FEATURES];
    float* primitive;
    int32_t* leaf_masks;
    int32_t* leaf_words;
    float* expected_columns;
    float* single_columns;
    float* tiled_columns;
    void* single_workspace;
    size_t single_workspace_bytes;
    void** single_cubins;
    size_t* single_cubin_sizes;
    size_t single_cubin_count;
    void** single_modules;
    ImplicitSindyAstColumnModule** single_module_handles;
    void* tiled_workspace;
    size_t tiled_workspace_bytes;
    void** tiled_cubins;
    size_t* tiled_cubin_sizes;
    size_t tiled_cubin_count;
    void** tiled_modules;
    ImplicitSindyColumnsModule** tiled_module_handles;
    ImplicitSindyAstColumnCompiler* single_compiler;
    ImplicitSindyAstCompiler* tiled_compiler;
    IsacrtDeviceBuffers device;
} IsacrtState;

static int
isacrt_check(
    int condition,
    const char* label
) {
    if (condition) return 1;
    fprintf(stderr, "check failed: %s\n", label);
    return 0;
}

static int
isacrt_write_file(
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

static int
isacrt_dump_cubins_if_requested(
    const char* kind,
    void* const* cubins,
    const size_t* cubin_sizes,
    size_t cubin_count
) {
    const char* prefix = getenv("IMPLICIT_SINDY_TEST_DUMP_PREFIX");
    size_t i;

    if (prefix == NULL || prefix[0] == '\0') return 1;
    for (i = 0u; i < cubin_count; ++i) {
        char path[512];
        snprintf(
            path,
            sizeof(path),
            "%s_%s_c%zu.cubin",
            prefix,
            kind,
            i
        );
        if (!isacrt_write_file(path, cubins[i], cubin_sizes[i])) {
            fprintf(stderr, "failed to write cubin dump: %s\n", path);
            return 0;
        }
        fprintf(stderr, "wrote cubin dump: %s (%zu bytes)\n", path, cubin_sizes[i]);
    }
    return 1;
}

static float*
isacrt_device_f32(
    CUdeviceptr ptr
) {
    return (float*)(uintptr_t)ptr;
}

static int32_t*
isacrt_device_i32(
    CUdeviceptr ptr
) {
    return (int32_t*)(uintptr_t)ptr;
}

static int32_t
isacrt_f32_constant_word(
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
isacrt_cuda(
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
isacrt_make_ast(
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
        ast->unary[i] = unary_ops[(idx * 7u + i * 3u) % (sizeof(unary_ops) / sizeof(unary_ops[0]))];
    }
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        ast->binary[i] = binary_ops[(idx * 5u + i * 2u) % (sizeof(binary_ops) / sizeof(binary_ops[0]))];
    }
}

static void
isacrt_fill_inputs(
    IsacrtState* state
) {
    int setting;
    int feature;
    int leaf;
    int col;
    int row;

    for (feature = 0; feature < ISACRT_FEATURES; ++feature) {
        isacrt_make_ast((size_t)feature, state->asts + feature);
    }

    for (col = 0; col < ISACRT_PRIMITIVE_COLS; ++col) {
        for (row = 0; row < ISACRT_PRIMITIVE_STRIDE; ++row) {
            float value = 0.0f;
            if (row < ISACRT_ROWS) {
                const float centered = (float)(row - 128) * 0.0020f;
                const float ripple = (float)((row * 3 + col * 11) % 23) * 0.00075f;
                value = centered + ripple + (float)col * 0.0234375f + 0.35f;
            }
            state->primitive[row + col * ISACRT_PRIMITIVE_STRIDE] = value;
        }
    }

    for (setting = 0; setting < ISACRT_SETTINGS; ++setting) {
        for (feature = 0; feature < ISACRT_FEATURES; ++feature) {
            uint32_t mask = 0u;
            const int feature_offset = setting * ISACRT_FEATURES + feature;
            for (leaf = 0; leaf < ISACRT_LEAVES; ++leaf) {
                const int word_offset = feature_offset * ISACRT_LEAVES + leaf;
                if (((setting * 5 + feature * 3 + leaf) % 5) != 0) {
                    mask |= 1u << leaf;
                    state->leaf_words[word_offset] =
                        (int32_t)((setting + feature * 2 + leaf * 3) % ISACRT_PRIMITIVE_COLS);
                } else {
                    const float value =
                        0.125f + 0.0175f * (float)((setting * 13 + feature * 7 + leaf * 5) % 19);
                    state->leaf_words[word_offset] = isacrt_f32_constant_word(value);
                }
            }
            state->leaf_masks[feature_offset] = (int32_t)mask;
        }
    }
}

static int
isacrt_alloc_host(
    IsacrtState* state
) {
    const size_t column_count = ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_ROWS;

    state->primitive = (float*)calloc(ISACRT_PRIMITIVE_COLS * ISACRT_PRIMITIVE_STRIDE, sizeof(float));
    state->leaf_masks = (int32_t*)calloc(ISACRT_SETTINGS * ISACRT_FEATURES, sizeof(int32_t));
    state->leaf_words = (int32_t*)calloc(ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_LEAVES, sizeof(int32_t));
    state->expected_columns = (float*)calloc(column_count, sizeof(float));
    state->single_columns = (float*)calloc(column_count, sizeof(float));
    state->tiled_columns = (float*)calloc(column_count, sizeof(float));

    return isacrt_check(
        state->primitive != NULL &&
        state->leaf_masks != NULL &&
        state->leaf_words != NULL &&
        state->expected_columns != NULL &&
        state->single_columns != NULL &&
        state->tiled_columns != NULL,
        "host allocations"
    );
}

static int
isacrt_alloc_device(
    IsacrtState* state
) {
    const size_t columns_bytes = ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_ROWS * sizeof(float);

    if (!isacrt_cuda(cuMemAlloc(
            &state->device.primitive,
            ISACRT_PRIMITIVE_COLS * ISACRT_PRIMITIVE_STRIDE * sizeof(float)
        ), "cuMemAlloc primitive")) return 0;
    if (!isacrt_cuda(cuMemAlloc(
            &state->device.leaf_masks,
            ISACRT_SETTINGS * ISACRT_FEATURES * sizeof(int32_t)
        ), "cuMemAlloc leaf_masks")) return 0;
    if (!isacrt_cuda(cuMemAlloc(
            &state->device.leaf_words,
            ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_LEAVES * sizeof(int32_t)
        ), "cuMemAlloc leaf_words")) return 0;
    if (!isacrt_cuda(cuMemAlloc(&state->device.single_columns, columns_bytes), "cuMemAlloc single columns")) return 0;
    if (!isacrt_cuda(cuMemAlloc(&state->device.tiled_columns, columns_bytes), "cuMemAlloc tiled columns")) return 0;
    return 1;
}

static int
isacrt_copy_inputs_to_device(
    IsacrtState* state
) {
    if (!isacrt_cuda(cuMemcpyHtoD(
            state->device.primitive,
            state->primitive,
            ISACRT_PRIMITIVE_COLS * ISACRT_PRIMITIVE_STRIDE * sizeof(float)
        ), "cuMemcpyHtoD primitive")) return 0;
    if (!isacrt_cuda(cuMemcpyHtoD(
            state->device.leaf_masks,
            state->leaf_masks,
            ISACRT_SETTINGS * ISACRT_FEATURES * sizeof(int32_t)
        ), "cuMemcpyHtoD leaf_masks")) return 0;
    if (!isacrt_cuda(cuMemcpyHtoD(
            state->device.leaf_words,
            state->leaf_words,
            ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_LEAVES * sizeof(int32_t)
        ), "cuMemcpyHtoD leaf_words")) return 0;
    return 1;
}

static int
isacrt_build_expected_columns(
    IsacrtState* state
) {
    const int result = implicit_sindy_cpu_gram_reference_columns_many_settings(
        state->asts,
        ISACRT_FEATURES,
        ISACRT_SETTINGS,
        state->primitive,
        ISACRT_ROWS,
        ISACRT_PRIMITIVE_STRIDE,
        ISACRT_PRIMITIVE_COLS,
        state->leaf_masks,
        state->leaf_words,
        ISACRT_LEAVES,
        state->expected_columns,
        ISACRT_FEATURES * ISACRT_ROWS,
        ISACRT_ROWS
    );
    return isacrt_check(
        result == IMPLICIT_SINDY_CPU_GRAM_REFERENCE_SUCCESS,
        "cpu reference columns"
    );
}

static int
isacrt_compile_single_columns(
    IsacrtState* state,
    unsigned int sm_major,
    unsigned int sm_minor
) {
    ImplicitSindyResult result;
    size_t workspace_bytes = 0u;
    size_t cubin_count;
    size_t i;

    result = implicit_sindy_ast_column_compiler_create(
        ISACRT_SINGLE_KERNELS_PER_MODULE,
        sm_major,
        sm_minor,
        NULL,
        0u,
        &state->single_compiler
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;

    result = implicit_sindy_ast_column_compile_workspace_size(
        state->single_compiler,
        ISACRT_FEATURES,
        ISACRT_WORKERS,
        ISACRT_SCRATCH_BYTES_PER_WORKER,
        &workspace_bytes
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;

    state->single_workspace = malloc(workspace_bytes);
    if (!isacrt_check(state->single_workspace != NULL, "single workspace allocation")) return 0;
    state->single_workspace_bytes = workspace_bytes;

    cubin_count = implicit_sindy_ast_column_cubin_count(
        ISACRT_FEATURES,
        ISACRT_SINGLE_KERNELS_PER_MODULE
    );
    if (!isacrt_check(cubin_count == ISACRT_FEATURES / ISACRT_SINGLE_KERNELS_PER_MODULE, "single cubin count")) {
        return 0;
    }
    state->single_cubin_count = cubin_count;

    state->single_cubins = (void**)calloc(cubin_count, sizeof(state->single_cubins[0]));
    state->single_cubin_sizes = (size_t*)calloc(cubin_count, sizeof(state->single_cubin_sizes[0]));
    state->single_modules = (void**)calloc(cubin_count, sizeof(state->single_modules[0]));
    state->single_module_handles =
        (ImplicitSindyAstColumnModule**)calloc(cubin_count, sizeof(state->single_module_handles[0]));
    if (!isacrt_check(
            state->single_cubins != NULL &&
            state->single_cubin_sizes != NULL &&
            state->single_modules != NULL &&
            state->single_module_handles != NULL,
            "single cubin/module arrays")) return 0;

    result = implicit_sindy_ast_column_compile_cubins(
        state->single_compiler,
        state->asts,
        ISACRT_FEATURES,
        sizeof(state->asts[0]),
        ISACRT_WORKERS,
        state->single_workspace,
        workspace_bytes,
        state->single_cubins,
        state->single_cubin_sizes
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;
    if (!isacrt_dump_cubins_if_requested(
            "single_columns",
            state->single_cubins,
            state->single_cubin_sizes,
            cubin_count)) return 0;

    result = implicit_sindy_load_cubin_modules(
        state->single_cubins,
        state->single_cubin_sizes,
        cubin_count,
        state->single_modules
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;

    for (i = 0u; i < cubin_count; ++i) {
        result = implicit_sindy_ast_column_module_create(
            state->single_modules[i],
            ISACRT_SINGLE_KERNELS_PER_MODULE,
            &state->single_module_handles[i]
        );
        if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;
    }
    return 1;
}

static int
isacrt_compile_tiled_columns(
    IsacrtState* state,
    unsigned int sm_major,
    unsigned int sm_minor
) {
    ImplicitSindyResult result;
    size_t workspace_bytes = 0u;
    size_t cubin_count;
    size_t i;

    result = implicit_sindy_ast_compiler_create_columns(
        ISACRT_COLUMNS_KERNELS_PER_MODULE,
        sm_major,
        sm_minor,
        NULL,
        0u,
        &state->tiled_compiler
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;

    result = implicit_sindy_ast_compile_workspace_size(
        state->tiled_compiler,
        ISACRT_FEATURES,
        ISACRT_WORKERS,
        ISACRT_SCRATCH_BYTES_PER_WORKER,
        &workspace_bytes
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;

    state->tiled_workspace = malloc(workspace_bytes);
    if (!isacrt_check(state->tiled_workspace != NULL, "tiled workspace allocation")) return 0;
    state->tiled_workspace_bytes = workspace_bytes;

    cubin_count = implicit_sindy_ast_cubin_count(
        ISACRT_FEATURES,
        ISACRT_COLUMNS_KERNELS_PER_MODULE
    );
    if (!isacrt_check(cubin_count == 1u, "tiled cubin count")) return 0;
    state->tiled_cubin_count = cubin_count;

    state->tiled_cubins = (void**)calloc(cubin_count, sizeof(state->tiled_cubins[0]));
    state->tiled_cubin_sizes = (size_t*)calloc(cubin_count, sizeof(state->tiled_cubin_sizes[0]));
    state->tiled_modules = (void**)calloc(cubin_count, sizeof(state->tiled_modules[0]));
    state->tiled_module_handles =
        (ImplicitSindyColumnsModule**)calloc(cubin_count, sizeof(state->tiled_module_handles[0]));
    if (!isacrt_check(
            state->tiled_cubins != NULL &&
            state->tiled_cubin_sizes != NULL &&
            state->tiled_modules != NULL &&
            state->tiled_module_handles != NULL,
            "tiled cubin/module arrays")) return 0;

    result = implicit_sindy_ast_compile_cubins(
        state->tiled_compiler,
        state->asts,
        ISACRT_FEATURES,
        sizeof(state->asts[0]),
        ISACRT_WORKERS,
        state->tiled_workspace,
        workspace_bytes,
        state->tiled_cubins,
        state->tiled_cubin_sizes
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;
    if (!isacrt_dump_cubins_if_requested(
            "tiled_columns",
            state->tiled_cubins,
            state->tiled_cubin_sizes,
            cubin_count)) return 0;

    result = implicit_sindy_load_cubin_modules(
        state->tiled_cubins,
        state->tiled_cubin_sizes,
        cubin_count,
        state->tiled_modules
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;

    for (i = 0u; i < cubin_count; ++i) {
        result = implicit_sindy_columns_module_create(
            state->tiled_modules[i],
            ISACRT_COLUMNS_KERNELS_PER_MODULE,
            &state->tiled_module_handles[i]
        );
        if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;
    }
    return 1;
}

static int
isacrt_launch_single_columns(
    IsacrtState* state
) {
    size_t feature;

    for (feature = 0u; feature < ISACRT_FEATURES; ++feature) {
        ImplicitSindyResult result;
        const size_t module_idx = feature / ISACRT_SINGLE_KERNELS_PER_MODULE;
        const size_t kernel_idx = feature % ISACRT_SINGLE_KERNELS_PER_MODULE;
        const CUdeviceptr output_ptr =
            state->device.single_columns + feature * ISACRT_ROWS * sizeof(float);
        result = implicit_sindy_ast_column_launch(
            state->single_module_handles[module_idx],
            kernel_idx,
            NULL,
            (int64_t)feature,
            ISACRT_SETTINGS,
            isacrt_device_f32(state->device.primitive),
            ISACRT_ROWS,
            ISACRT_PRIMITIVE_STRIDE,
            ISACRT_PRIMITIVE_COLS,
            isacrt_device_i32(state->device.leaf_masks),
            isacrt_device_i32(state->device.leaf_words),
            ISACRT_LEAVES,
            isacrt_device_f32(output_ptr),
            ISACRT_FEATURES * ISACRT_ROWS
        );
        if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;
    }
    if (!isacrt_cuda(cuCtxSynchronize(), "cuCtxSynchronize single columns")) return 0;
    return 1;
}

static int
isacrt_launch_tiled_columns(
    IsacrtState* state
) {
    ImplicitSindyResult result = implicit_sindy_columns_launch(
        state->tiled_module_handles[0],
        0u,
        NULL,
        ISACRT_SETTINGS,
        isacrt_device_f32(state->device.primitive),
        ISACRT_ROWS,
        ISACRT_PRIMITIVE_STRIDE,
        ISACRT_PRIMITIVE_COLS,
        isacrt_device_i32(state->device.leaf_masks),
        isacrt_device_i32(state->device.leaf_words),
        ISACRT_LEAVES,
        isacrt_device_f32(state->device.tiled_columns),
        ISACRT_FEATURES * ISACRT_ROWS,
        ISACRT_ROWS
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) return 0;
    if (!isacrt_cuda(cuCtxSynchronize(), "cuCtxSynchronize tiled columns")) return 0;
    return 1;
}

static int
isacrt_copy_outputs_to_host(
    IsacrtState* state
) {
    const size_t columns_bytes = ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_ROWS * sizeof(float);

    if (!isacrt_cuda(cuMemcpyDtoH(
            state->single_columns,
            state->device.single_columns,
            columns_bytes
        ), "cuMemcpyDtoH single columns")) return 0;
    if (!isacrt_cuda(cuMemcpyDtoH(
            state->tiled_columns,
            state->device.tiled_columns,
            columns_bytes
        ), "cuMemcpyDtoH tiled columns")) return 0;
    return 1;
}

static int
isacrt_compare_array(
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
isacrt_compare_columns(
    IsacrtState* state
) {
    const size_t column_count = ISACRT_SETTINGS * ISACRT_FEATURES * ISACRT_ROWS;

    if (!isacrt_compare_array(
            "single ast columns vs cpu columns",
            state->expected_columns,
            state->single_columns,
            column_count,
            1.5e-4f,
            2.0e-4f)) return 0;
    if (!isacrt_compare_array(
            "tiled 32-ast columns vs cpu columns",
            state->expected_columns,
            state->tiled_columns,
            column_count,
            1.5e-4f,
            2.0e-4f)) return 0;
    return 1;
}

static int
isacrt_check_single_padding_slots(
    IsacrtState* state
) {
    enum {
        PADDING_AST_COUNT = 2,
        PADDING_FIRST_EMPTY_KERNEL = 2
    };
    ImplicitSindyResult result;
    void* cubins[1];
    size_t cubin_sizes[1];
    void* modules[1];
    ImplicitSindyAstColumnModule* column_module = NULL;
    CUdeviceptr output = 0;
    float* host_output = NULL;
    int ok = 0;
    size_t count;
    size_t i;
    size_t slot;

    cubins[0] = NULL;
    cubin_sizes[0] = 0u;
    modules[0] = NULL;

    if (!isacrt_check(
            implicit_sindy_ast_column_cubin_count(
                PADDING_AST_COUNT,
                ISACRT_SINGLE_KERNELS_PER_MODULE
            ) == 1u,
            "padding cubin count")) return 0;

    result = implicit_sindy_ast_column_compile_cubins(
        state->single_compiler,
        state->asts,
        PADDING_AST_COUNT,
        sizeof(state->asts[0]),
        ISACRT_WORKERS,
        state->single_workspace,
        state->single_workspace_bytes,
        cubins,
        cubin_sizes
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) goto done;

    result = implicit_sindy_load_cubin_modules(cubins, cubin_sizes, 1u, modules);
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) goto done;

    result = implicit_sindy_ast_column_module_create(
        modules[0],
        ISACRT_SINGLE_KERNELS_PER_MODULE,
        &column_module
    );
    if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) goto done;

    count = ISACRT_SETTINGS * ISACRT_ROWS;
    host_output = (float*)malloc(count * sizeof(float));
    if (!isacrt_check(host_output != NULL, "padding host output")) goto done;
    if (!isacrt_cuda(cuMemAlloc(&output, count * sizeof(float)), "cuMemAlloc padding output")) goto done;

    for (slot = PADDING_FIRST_EMPTY_KERNEL; slot < ISACRT_SINGLE_KERNELS_PER_MODULE; ++slot) {
        for (i = 0u; i < count; ++i) host_output[i] = 17.0f;
        if (!isacrt_cuda(cuMemcpyHtoD(output, host_output, count * sizeof(float)), "cuMemcpyHtoD padding output")) {
            goto done;
        }

        result = implicit_sindy_ast_column_launch(
            column_module,
            slot,
            NULL,
            0,
            ISACRT_SETTINGS,
            isacrt_device_f32(state->device.primitive),
            ISACRT_ROWS,
            ISACRT_PRIMITIVE_STRIDE,
            ISACRT_PRIMITIVE_COLS,
            isacrt_device_i32(state->device.leaf_masks),
            isacrt_device_i32(state->device.leaf_words),
            ISACRT_LEAVES,
            isacrt_device_f32(output),
            ISACRT_ROWS
        );
        if (!isacrt_check(result == IMPLICIT_SINDY_SUCCESS, implicit_sindy_result_to_string(result))) goto done;
        if (!isacrt_cuda(cuCtxSynchronize(), "cuCtxSynchronize padding column")) goto done;
        if (!isacrt_cuda(cuMemcpyDtoH(host_output, output, count * sizeof(float)), "cuMemcpyDtoH padding output")) {
            goto done;
        }
        for (i = 0u; i < count; ++i) {
            if (host_output[i] != 0.0f) {
                fprintf(
                    stderr,
                    "padding slot %zu output[%zu] expected 0 actual %.9g\n",
                    slot,
                    i,
                    host_output[i]
                );
                goto done;
            }
        }
    }

    printf("single ast padding slots produced zero columns\n");
    ok = 1;

done:
    if (output != 0) (void)cuMemFree(output);
    free(host_output);
    if (column_module != NULL) (void)implicit_sindy_ast_column_module_destroy(column_module);
    if (modules[0] != NULL) (void)implicit_sindy_unload_modules(modules, 1u);
    if (cubins[0] != NULL || cubin_sizes[0] != 0u) {
        implicit_sindy_free_cubins(cubins, cubin_sizes, 1u);
    }
    return ok;
}

static void
isacrt_destroy_state(
    IsacrtState* state
) {
    size_t i;
    if (state == NULL) return;
    if (state->single_module_handles != NULL) {
        for (i = 0u; i < state->single_cubin_count; ++i) {
            if (state->single_module_handles[i] != NULL) {
                (void)implicit_sindy_ast_column_module_destroy(state->single_module_handles[i]);
            }
        }
    }
    if (state->tiled_module_handles != NULL) {
        for (i = 0u; i < state->tiled_cubin_count; ++i) {
            if (state->tiled_module_handles[i] != NULL) {
                (void)implicit_sindy_columns_module_destroy(state->tiled_module_handles[i]);
            }
        }
    }
    if (state->single_modules != NULL) {
        (void)implicit_sindy_unload_modules(state->single_modules, state->single_cubin_count);
    }
    if (state->tiled_modules != NULL) {
        (void)implicit_sindy_unload_modules(state->tiled_modules, state->tiled_cubin_count);
    }
    implicit_sindy_free_cubins(state->single_cubins, state->single_cubin_sizes, state->single_cubin_count);
    implicit_sindy_free_cubins(state->tiled_cubins, state->tiled_cubin_sizes, state->tiled_cubin_count);
    if (state->single_compiler != NULL) {
        (void)implicit_sindy_ast_column_compiler_destroy(state->single_compiler);
    }
    if (state->tiled_compiler != NULL) {
        (void)implicit_sindy_ast_compiler_destroy(state->tiled_compiler);
    }
    if (state->device.primitive != 0) (void)cuMemFree(state->device.primitive);
    if (state->device.leaf_masks != 0) (void)cuMemFree(state->device.leaf_masks);
    if (state->device.leaf_words != 0) (void)cuMemFree(state->device.leaf_words);
    if (state->device.single_columns != 0) (void)cuMemFree(state->device.single_columns);
    if (state->device.tiled_columns != 0) (void)cuMemFree(state->device.tiled_columns);
    free(state->single_workspace);
    free(state->tiled_workspace);
    free(state->single_cubins);
    free(state->single_cubin_sizes);
    free(state->single_modules);
    free(state->single_module_handles);
    free(state->tiled_cubins);
    free(state->tiled_cubin_sizes);
    free(state->tiled_modules);
    free(state->tiled_module_handles);
    free(state->primitive);
    free(state->leaf_masks);
    free(state->leaf_words);
    free(state->expected_columns);
    free(state->single_columns);
    free(state->tiled_columns);
}

static int
isacrt_run_case(
    unsigned int sm_major,
    unsigned int sm_minor
) {
    IsacrtState state;
    int ok;

    memset(&state, 0, sizeof(state));
    ok =
        isacrt_alloc_host(&state) &&
        isacrt_alloc_device(&state) &&
        (isacrt_fill_inputs(&state), 1) &&
        isacrt_build_expected_columns(&state) &&
        isacrt_copy_inputs_to_device(&state) &&
        isacrt_compile_single_columns(&state, sm_major, sm_minor) &&
        isacrt_compile_tiled_columns(&state, sm_major, sm_minor) &&
        isacrt_launch_single_columns(&state) &&
        isacrt_launch_tiled_columns(&state) &&
        isacrt_copy_outputs_to_host(&state) &&
        isacrt_compare_columns(&state) &&
        isacrt_check_single_padding_slots(&state);
    isacrt_destroy_state(&state);
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

    if (!isacrt_run_case(sm_major, sm_minor)) {
        stack_ptx_test_release_context(device, owns_primary);
        return 1;
    }

    stack_ptx_test_release_context(device, owns_primary);
    printf("PASS implicit_sindy_ast_columns_cpu_reference_test\n");
    return 0;
}
