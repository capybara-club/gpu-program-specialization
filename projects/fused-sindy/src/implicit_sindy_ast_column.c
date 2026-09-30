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
#include "implicit_sindy_internal_constants.h"
#include "binary_ast_to_stack_ptx.h"
#include "ptx_inject.h"
#include "stack_ptx.h"

#include <stack_ptx_descriptions.h>

#include <cuda.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>

#include <pthread.h>
#include <stdarg.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ISAC_ERROR_RET(ans)                         \
    do {                                           \
        ImplicitSindyResult isac_result__ = (ans); \
        return isac_result__;                      \
    } while (0)

#define ISAC_CHECK_RET(ans)                         \
    do {                                           \
        ImplicitSindyResult isac_result__ = (ans); \
        if (isac_result__ != IMPLICIT_SINDY_SUCCESS) { \
            ISAC_ERROR_RET(isac_result__);         \
        }                                          \
    } while (0)

#define ISAC_CHECK_CUDA_RET(ans)                    \
    do {                                           \
        CUresult isac_cuda_result__ = (ans);       \
        if (isac_cuda_result__ != CUDA_SUCCESS) {  \
            ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_CUDA); \
        }                                          \
    } while (0)

enum {
    ISAC_THREADS = 128,
    ISAC_PROGRAM_STRIDE = 1024,
    ISAC_MAX_KERNELS_PER_MODULE = 64
};

typedef struct {
    unsigned char* base;
    size_t capacity;
    size_t offset;
} IsacArena;

typedef struct {
    char* data;
    size_t size;
    size_t capacity;
} IsacString;

struct ImplicitSindyAstColumnCompiler {
    PtxInjectHandle inject;
    char* template_ptx;
    size_t template_ptx_bytes;
    StackPtxCompilerInfo compiler_info;
    size_t stack_workspace_size;
    size_t program_stride;
    size_t kernels_per_module;
    size_t* inject_to_kernel_idx;
    size_t num_injects;
    unsigned int sm_major;
    unsigned int sm_minor;
    const char* const* nvptx_options;
    size_t num_nvptx_options;
};

struct ImplicitSindyAstColumnModule {
    CUmodule module;
    size_t kernels_per_module;
    CUfunction functions[];
};

typedef struct {
    const ImplicitSindyAstColumnCompiler* compiler;
    const BinaryAST* asts;
    size_t num_asts;
    size_t ast_stride_bytes;
    void** out_cubins;
    size_t* out_cubin_sizes;
    IsacArena arena;
    size_t batch_begin;
    size_t batch_end;
    ImplicitSindyResult result;
} IsacWorker;

static const StackPtxRegister isac_registers[] = {
    { "_x1", STACK_PTX_STACK_TYPE_F32 },
    { "_x2", STACK_PTX_STACK_TYPE_F32 },
    { "_x3", STACK_PTX_STACK_TYPE_F32 },
    { "_x4", STACK_PTX_STACK_TYPE_F32 },
    { "_x5", STACK_PTX_STACK_TYPE_F32 },
    { "_x6", STACK_PTX_STACK_TYPE_F32 },
    { "_x7", STACK_PTX_STACK_TYPE_F32 },
    { "_x8", STACK_PTX_STACK_TYPE_F32 },
    { "_x0", STACK_PTX_STACK_TYPE_F32 }
};

static const size_t isac_requests[] = { BINARY_AST_NUM_INPUTS };

static ImplicitSindyResult
isac_mul_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (a != 0u && b > SIZE_MAX / a) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out = a * b;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_add_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (b > SIZE_MAX - a) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out = a + b;
    return IMPLICIT_SINDY_SUCCESS;
}

static size_t
isac_align_up_size(
    size_t value,
    size_t alignment
) {
    if (alignment == 0u) return value;
    return ((value + alignment - 1u) / alignment) * alignment;
}

static uintptr_t
isac_align_up_uintptr(
    uintptr_t value,
    uintptr_t alignment
) {
    if (alignment == 0u) return value;
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static ImplicitSindyResult
isac_arena_alloc(
    IsacArena* arena,
    size_t size,
    size_t alignment,
    void** out_ptr
) {
    size_t aligned_offset;
    if (arena == NULL || arena->base == NULL || out_ptr == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    aligned_offset = isac_align_up_size(arena->offset, alignment);
    if (aligned_offset > arena->capacity || size > arena->capacity - aligned_offset) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    *out_ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_arena_remaining(
    IsacArena* arena,
    size_t alignment,
    void** out_ptr,
    size_t* out_bytes
) {
    size_t aligned_offset;
    if (arena == NULL || arena->base == NULL || out_ptr == NULL || out_bytes == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    aligned_offset = isac_align_up_size(arena->offset, alignment);
    if (aligned_offset > arena->capacity) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    arena->offset = aligned_offset;
    *out_ptr = arena->base + aligned_offset;
    *out_bytes = arena->capacity - aligned_offset;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_arena_commit(
    IsacArena* arena,
    size_t size
) {
    if (arena == NULL || size > arena->capacity - arena->offset) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    arena->offset += size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_result_from_binary_ast(
    BinaryAstResult result
) {
    switch (result) {
        case BINARY_AST_SUCCESS:
            return IMPLICIT_SINDY_SUCCESS;
        case BINARY_AST_ERROR_INVALID_VALUE:
            return IMPLICIT_SINDY_ERROR_INVALID_VALUE;
        case BINARY_AST_ERROR_INSUFFICIENT_BUFFER:
            return IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED;
    }
    return IMPLICIT_SINDY_ERROR_INTERNAL;
}

static ImplicitSindyResult
isac_result_from_stack_ptx(
    StackPtxResult result
) {
    return result == STACK_PTX_SUCCESS
        ? IMPLICIT_SINDY_SUCCESS
        : IMPLICIT_SINDY_ERROR_STACK_PTX;
}

static ImplicitSindyResult
isac_result_from_ptx_inject(
    PtxInjectResult result
) {
    return result == PTX_INJECT_SUCCESS
        ? IMPLICIT_SINDY_SUCCESS
        : IMPLICIT_SINDY_ERROR_PTX_INJECT;
}

static size_t
isac_default_worker_count(
    size_t requested_worker_count,
    size_t num_batches
) {
    size_t workers = requested_worker_count;
    if (workers == 0u) {
        workers = 1u;
#ifdef _SC_NPROCESSORS_ONLN
        {
            long nproc = sysconf(_SC_NPROCESSORS_ONLN);
            if (nproc > 0) workers = (size_t)nproc;
        }
#endif
    }
    if (num_batches != 0u && workers > num_batches) workers = num_batches;
    if (workers == 0u) workers = 1u;
    return workers;
}

static size_t
isac_kernels_per_module_count(
    size_t kernels_per_module
) {
    if (kernels_per_module == 0u || kernels_per_module > ISAC_MAX_KERNELS_PER_MODULE) return 0u;
    return kernels_per_module;
}

static size_t
isac_cubin_count(
    size_t num_asts,
    size_t kernels_per_module
) {
    size_t kernel_count = isac_kernels_per_module_count(kernels_per_module);
    if (num_asts == 0u || kernel_count == 0u) return 0u;
    if (num_asts > SIZE_MAX - (kernel_count - 1u)) return 0u;
    return (num_asts + kernel_count - 1u) / kernel_count;
}

static const BinaryAST*
isac_ast_at(
    const BinaryAST* asts,
    size_t ast_stride_bytes,
    size_t idx
) {
    const unsigned char* base = (const unsigned char*)asts;
    return (const BinaryAST*)(const void*)(base + idx * ast_stride_bytes);
}

static void
isac_make_empty_ast(
    BinaryAST* ast
) {
    size_t i;
    if (ast == NULL) return;
    memset(ast, 0, sizeof(*ast));
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
        ast->unary[i] = BINARY_AST_UNARY_ZERO_F32;
    }
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        ast->binary[i] = BINARY_AST_BINARY_KEEP_LEFT;
    }
}

static void
isac_string_free(
    IsacString* string
) {
    if (string == NULL) return;
    free(string->data);
    string->data = NULL;
    string->size = 0u;
    string->capacity = 0u;
}

static ImplicitSindyResult
isac_string_reserve(
    IsacString* string,
    size_t needed
) {
    char* next;
    size_t next_capacity;
    if (string == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (needed <= string->capacity) return IMPLICIT_SINDY_SUCCESS;
    next_capacity = string->capacity != 0u ? string->capacity : 4096u;
    while (next_capacity < needed) {
        if (next_capacity > SIZE_MAX / 2u) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
        next_capacity *= 2u;
    }
    next = (char*)realloc(string->data, next_capacity);
    if (next == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    string->data = next;
    string->capacity = next_capacity;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_string_append(
    IsacString* string,
    const char* text
) {
    size_t len;
    if (string == NULL || text == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    len = strlen(text);
    ISAC_CHECK_RET(isac_string_reserve(string, string->size + len + 1u));
    memcpy(string->data + string->size, text, len + 1u);
    string->size += len;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_string_appendf(
    IsacString* string,
    const char* format,
    ...
) {
    va_list args;
    va_list args_copy;
    int needed_i;
    size_t needed;

    if (string == NULL || format == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    va_start(args, format);
    va_copy(args_copy, args);
    needed_i = vsnprintf(NULL, 0u, format, args_copy);
    va_end(args_copy);
    if (needed_i < 0) {
        va_end(args);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
    }
    needed = (size_t)needed_i;
    ISAC_CHECK_RET(isac_string_reserve(string, string->size + needed + 1u));
    if (vsnprintf(string->data + string->size, string->capacity - string->size, format, args) != needed_i) {
        va_end(args);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
    }
    va_end(args);
    string->size += needed;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_append_column_kernel_source(
    IsacString* source,
    size_t kernel_count,
    size_t kernel_idx
) {
    if (source == NULL || kernel_idx >= kernel_count) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    ISAC_CHECK_RET(isac_string_append(
        source,
        "extern \"C\" __global__\n"
        "__launch_bounds__(128)\n"
        "void\n"
    ));
    if (kernel_count == 1u) {
        ISAC_CHECK_RET(isac_string_append(source, "implicit_sindy_ast_column_kernel(\n"));
    } else {
        ISAC_CHECK_RET(isac_string_appendf(
            source,
            "implicit_sindy_ast_column_kernel_%zu(\n",
            kernel_idx
        ));
    }
    ISAC_CHECK_RET(isac_string_append(
        source,
        "    const float* primitive_features,\n"
        "    int64_t row_count,\n"
        "    int64_t primitive_feature_stride,\n"
        "    int64_t num_primitive_features,\n"
        "    int64_t feature_index,\n"
        "    const int32_t* leaf_masks,\n"
        "    const int32_t* leaf_words,\n"
        "    int64_t leaf_words_feature_stride,\n"
        "    float* output,\n"
        "    int64_t output_setting_stride\n"
        ") {\n"
        "    __shared__ float leaf_constants[8];\n"
        "    __shared__ const float* leaf_ptrs[8];\n"
        "    __shared__ int leaf_strides[8];\n"
        "    const int tid = static_cast<int>(threadIdx.x);\n"
        "    const int setting_idx = static_cast<int>(blockIdx.x);\n"
        "    const int64_t row = static_cast<int64_t>(blockIdx.y) * 128ll + static_cast<int64_t>(tid);\n"
        "    if (feature_index < 0 || feature_index >= 32) return;\n"
        "    if (tid < 8) {\n"
        "        const int leaf = tid;\n"
        "        const int64_t setting_feature = static_cast<int64_t>(setting_idx) * 32ll + feature_index;\n"
        "        const uint32_t mask = static_cast<uint32_t>(leaf_masks[setting_feature]);\n"
        "        const uint32_t word = static_cast<uint32_t>(\n"
        "            leaf_words[setting_feature * leaf_words_feature_stride + static_cast<int64_t>(leaf)]);\n"
        "        if (((mask >> leaf) & 1u) != 0u) {\n"
        "            const int col = static_cast<int>(word);\n"
        "            if (col >= 0 && col < static_cast<int>(num_primitive_features)) {\n"
        "                leaf_ptrs[leaf] = primitive_features + static_cast<int64_t>(col) * primitive_feature_stride;\n"
        "                leaf_strides[leaf] = 1;\n"
        "            } else {\n"
        "                leaf_constants[leaf] = 0.0f;\n"
        "                leaf_ptrs[leaf] = leaf_constants + leaf;\n"
        "                leaf_strides[leaf] = 0;\n"
        "            }\n"
        "        } else {\n"
        "            leaf_constants[leaf] = isac_u32_as_f32(word);\n"
        "            leaf_ptrs[leaf] = leaf_constants + leaf;\n"
        "            leaf_strides[leaf] = 0;\n"
        "        }\n"
        "    }\n"
        "    __syncthreads();\n"
        "    if (row < row_count) {\n"
        "        float value = 0.0f;\n"
        "        const float leaf0 = leaf_ptrs[0][row * static_cast<int64_t>(leaf_strides[0])];\n"
        "        const float leaf1 = leaf_ptrs[1][row * static_cast<int64_t>(leaf_strides[1])];\n"
        "        const float leaf2 = leaf_ptrs[2][row * static_cast<int64_t>(leaf_strides[2])];\n"
        "        const float leaf3 = leaf_ptrs[3][row * static_cast<int64_t>(leaf_strides[3])];\n"
        "        const float leaf4 = leaf_ptrs[4][row * static_cast<int64_t>(leaf_strides[4])];\n"
        "        const float leaf5 = leaf_ptrs[5][row * static_cast<int64_t>(leaf_strides[5])];\n"
        "        const float leaf6 = leaf_ptrs[6][row * static_cast<int64_t>(leaf_strides[6])];\n"
        "        const float leaf7 = leaf_ptrs[7][row * static_cast<int64_t>(leaf_strides[7])];\n"
        "        asm volatile(\n"
        "            \"{\\n\\t\"\n"
        "            \".reg .f32 %%_x0;\\n\\t\"\n"
        "            \".reg .f32 %%_x1;\\n\\t\"\n"
        "            \".reg .f32 %%_x2;\\n\\t\"\n"
        "            \".reg .f32 %%_x3;\\n\\t\"\n"
        "            \".reg .f32 %%_x4;\\n\\t\"\n"
        "            \".reg .f32 %%_x5;\\n\\t\"\n"
        "            \".reg .f32 %%_x6;\\n\\t\"\n"
        "            \".reg .f32 %%_x7;\\n\\t\"\n"
        "            \".reg .f32 %%_x8;\\n\\t\"\n"
        "            \"mov.f32 %%_x0, %0;\\n\\t\"\n"
        "            \"mov.f32 %%_x1, %1;\\n\\t\"\n"
        "            \"mov.f32 %%_x2, %2;\\n\\t\"\n"
        "            \"mov.f32 %%_x3, %3;\\n\\t\"\n"
        "            \"mov.f32 %%_x4, %4;\\n\\t\"\n"
        "            \"mov.f32 %%_x5, %5;\\n\\t\"\n"
        "            \"mov.f32 %%_x6, %6;\\n\\t\"\n"
        "            \"mov.f32 %%_x7, %7;\\n\\t\"\n"
        "            \"mov.f32 %%_x8, %8;\\n\\t\"\n"
    ));
    ISAC_CHECK_RET(isac_string_appendf(
        source,
        "            \"// PTX_INJECT_START implicit_ast_column_%zu\\n\\t\"\n",
        kernel_idx
    ));
    ISAC_CHECK_RET(isac_string_append(
        source,
        "            \"// _x0 m f32 F32 value\\n\\t\"\n"
        "            \"// _x1 i f32 F32 leaf0\\n\\t\"\n"
        "            \"// _x2 i f32 F32 leaf1\\n\\t\"\n"
        "            \"// _x3 i f32 F32 leaf2\\n\\t\"\n"
        "            \"// _x4 i f32 F32 leaf3\\n\\t\"\n"
        "            \"// _x5 i f32 F32 leaf4\\n\\t\"\n"
        "            \"// _x6 i f32 F32 leaf5\\n\\t\"\n"
        "            \"// _x7 i f32 F32 leaf6\\n\\t\"\n"
        "            \"// _x8 i f32 F32 leaf7\\n\\t\"\n"
        "            \"// PTX_INJECT_END\\n\\t\"\n"
        "            \"mov.f32 %0, %%_x0;\\n\\t\"\n"
        "            \"}\"\n"
        "            : \"+f\"(value)\n"
        "            : \"f\"(leaf0), \"f\"(leaf1), \"f\"(leaf2), \"f\"(leaf3),\n"
        "              \"f\"(leaf4), \"f\"(leaf5), \"f\"(leaf6), \"f\"(leaf7));\n"
        "        output[static_cast<int64_t>(setting_idx) * output_setting_stride + row] = value;\n"
        "    }\n"
        "}\n\n"
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_generate_column_template_source(
    size_t kernel_count,
    char** source_out
) {
    IsacString source;
    size_t kernel_idx;

    if (source_out == NULL || kernel_count == 0u) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *source_out = NULL;
    memset(&source, 0, sizeof(source));

    ISAC_CHECK_RET(isac_string_append(
        &source,
        "extern \"C\" {\n"
        "typedef int int32_t;\n"
        "typedef unsigned int uint32_t;\n"
        "typedef long long int64_t;\n"
        "}\n\n"
        "__device__ __forceinline__\n"
        "float\n"
        "isac_u32_as_f32(\n"
        "    uint32_t bits\n"
        ") {\n"
        "    union {\n"
        "        uint32_t u;\n"
        "        float f;\n"
        "    } value;\n"
        "    value.u = bits;\n"
        "    return value.f;\n"
        "}\n\n"
    ));
    for (kernel_idx = 0u; kernel_idx < kernel_count; ++kernel_idx) {
        ISAC_CHECK_RET(isac_append_column_kernel_source(
            &source,
            kernel_count,
            kernel_idx
        ));
    }

    *source_out = source.data;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_compile_cuda_to_ptx(
    const char* source,
    const char* program_name,
    unsigned int sm_major,
    unsigned int sm_minor,
    char** ptx_out,
    size_t* ptx_bytes_out
) {
    nvrtcProgram program = NULL;
    nvrtcResult result;
    char arch_option[64];
    const char* options[4];
    int option_count = 0;
    size_t ptx_size = 0u;
    char* ptx = NULL;

    if (source == NULL || program_name == NULL || ptx_out == NULL || ptx_bytes_out == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *ptx_out = NULL;
    *ptx_bytes_out = 0u;

    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%u%u", sm_major, sm_minor);
    options[option_count++] = arch_option;
    options[option_count++] = "--std=c++17";
    options[option_count++] = "--use_fast_math";

    result = nvrtcCreateProgram(&program, source, program_name, 0, NULL, NULL);
    if (result != NVRTC_SUCCESS) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    result = nvrtcCompileProgram(program, option_count, options);
    if (result != NVRTC_SUCCESS) {
        size_t log_size = 0u;
        (void)nvrtcGetProgramLogSize(program, &log_size);
        if (log_size > 1u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    fprintf(stderr, "implicit-sindy ast-column nvrtc log:\n%s\n", log);
                }
                free(log);
            }
        }
        (void)nvrtcDestroyProgram(&program);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    if (nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS || ptx_size == 0u) {
        (void)nvrtcDestroyProgram(&program);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    ptx = (char*)malloc(ptx_size);
    if (ptx == NULL) {
        (void)nvrtcDestroyProgram(&program);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    }
    if (nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) {
        free(ptx);
        (void)nvrtcDestroyProgram(&program);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    (void)nvrtcDestroyProgram(&program);
    *ptx_out = ptx;
    *ptx_bytes_out = ptx_size != 0u ? ptx_size - 1u : 0u;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_compile_ptx_to_malloced_cubin(
    const ImplicitSindyAstColumnCompiler* compiler,
    const char* ptx,
    size_t ptx_bytes,
    void** cubin_out,
    size_t* cubin_bytes_out
) {
    nvPTXCompilerHandle ptx_compiler = NULL;
    nvPTXCompileResult result;
    const char* options[80];
    char arch_option[64];
    size_t option_count = 0u;
    size_t image_size = 0u;
    void* image = NULL;
    size_t i;

    if (compiler == NULL || ptx == NULL || ptx_bytes == 0u || cubin_out == NULL || cubin_bytes_out == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *cubin_out = NULL;
    *cubin_bytes_out = 0u;
    if (compiler->num_nvptx_options + 3u > sizeof(options) / sizeof(options[0])) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    snprintf(arch_option, sizeof(arch_option), "--gpu-name=sm_%u%u", compiler->sm_major, compiler->sm_minor);
    options[option_count++] = arch_option;
    options[option_count++] = "--opt-level=3";
    options[option_count++] = "--allow-expensive-optimizations=false";
    for (i = 0u; i < compiler->num_nvptx_options; ++i) {
        options[option_count++] = compiler->nvptx_options[i];
    }

    result = nvPTXCompilerCreate(&ptx_compiler, ptx_bytes, ptx);
    if (result != NVPTXCOMPILE_SUCCESS) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    result = nvPTXCompilerCompile(ptx_compiler, (int)option_count, options);
    if (result != NVPTXCOMPILE_SUCCESS) {
        size_t log_size = 0u;
        (void)nvPTXCompilerGetErrorLogSize(ptx_compiler, &log_size);
        if (log_size > 1u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                if (nvPTXCompilerGetErrorLog(ptx_compiler, log) == NVPTXCOMPILE_SUCCESS) {
                    fprintf(stderr, "implicit-sindy ast-column nvPTXCompiler log:\n%s\n", log);
                }
                free(log);
            }
        }
        (void)nvPTXCompilerDestroy(&ptx_compiler);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    if (nvPTXCompilerGetCompiledProgramSize(ptx_compiler, &image_size) != NVPTXCOMPILE_SUCCESS ||
        image_size == 0u) {
        (void)nvPTXCompilerDestroy(&ptx_compiler);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    image = malloc(image_size);
    if (image == NULL) {
        (void)nvPTXCompilerDestroy(&ptx_compiler);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    }
    if (nvPTXCompilerGetCompiledProgram(ptx_compiler, image) != NVPTXCOMPILE_SUCCESS) {
        free(image);
        (void)nvPTXCompilerDestroy(&ptx_compiler);
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    (void)nvPTXCompilerDestroy(&ptx_compiler);
    *cubin_out = image;
    *cubin_bytes_out = image_size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_parse_column_site_name(
    const char* name,
    size_t* site_idx_out
) {
    const char prefix[] = "implicit_ast_column_";
    const char* p;
    size_t value = 0u;

    if (name == NULL || site_idx_out == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (strncmp(name, prefix, sizeof(prefix) - 1u) != 0) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
    }
    p = name + sizeof(prefix) - 1u;
    if (*p == '\0') ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
    while (*p != '\0') {
        if (*p < '0' || *p > '9') ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
        if (value > (SIZE_MAX - (size_t)(*p - '0')) / 10u) {
            ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
        }
        value = value * 10u + (size_t)(*p - '0');
        ++p;
    }
    *site_idx_out = value;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_setup_inject_order(
    ImplicitSindyAstColumnCompiler* compiler
) {
    size_t i;

    if (compiler == NULL || compiler->inject == NULL || compiler->kernels_per_module == 0u) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    ISAC_CHECK_RET(isac_result_from_ptx_inject(ptx_inject_num_injects(
        compiler->inject,
        &compiler->num_injects
    )));
    if (compiler->num_injects != compiler->kernels_per_module) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
    }
    compiler->inject_to_kernel_idx = (size_t*)calloc(
        compiler->num_injects,
        sizeof(*compiler->inject_to_kernel_idx)
    );
    if (compiler->inject_to_kernel_idx == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    for (i = 0u; i < compiler->num_injects; ++i) {
        const char* name = NULL;
        size_t kernel_idx = 0u;
        ISAC_CHECK_RET(isac_result_from_ptx_inject(ptx_inject_inject_info_by_index(
            compiler->inject,
            i,
            &name,
            NULL,
            NULL
        )));
        ISAC_CHECK_RET(isac_parse_column_site_name(name, &kernel_idx));
        if (kernel_idx >= compiler->kernels_per_module) {
            ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
        }
        compiler->inject_to_kernel_idx[i] = kernel_idx;
    }
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_emit_stack_ptx_stub(
    const ImplicitSindyAstColumnCompiler* compiler,
    const BinaryAST* ast,
    IsacArena* arena,
    void* stack_workspace,
    StackPtxInstruction* program,
    char** stub_out
) {
    BinaryAstResult binary_result;
    StackPtxResult stack_result;
    char* stub = NULL;
    size_t stub_capacity = 0u;
    size_t instruction_count = 0u;
    size_t written_instructions = 0u;
    size_t written_ptx = 0u;

    if (compiler == NULL || ast == NULL || arena == NULL || stack_workspace == NULL ||
        program == NULL || stub_out == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *stub_out = NULL;
    binary_result = binary_ast_stack_ptx_instruction_count(ast, &instruction_count);
    ISAC_CHECK_RET(isac_result_from_binary_ast(binary_result));
    if (instruction_count == 0u || instruction_count > compiler->program_stride) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_STACK_PTX);
    }
    binary_result = binary_ast_write_stack_ptx(
        ast,
        program,
        compiler->program_stride,
        &written_instructions
    );
    ISAC_CHECK_RET(isac_result_from_binary_ast(binary_result));
    if (written_instructions == 0u) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_STACK_PTX);

    ISAC_CHECK_RET(isac_arena_remaining(arena, 1u, (void**)&stub, &stub_capacity));
    stack_result = stack_ptx_compile(
        &compiler->compiler_info,
        &stack_ptx_stack_info,
        program,
        isac_registers,
        sizeof(isac_registers) / sizeof(isac_registers[0]),
        binary_ast_stack_ptx_routines,
        binary_ast_stack_ptx_num_routines,
        isac_requests,
        sizeof(isac_requests) / sizeof(isac_requests[0]),
        compiler->program_stride,
        stack_workspace,
        compiler->stack_workspace_size,
        stub,
        stub_capacity,
        &written_ptx
    );
    ISAC_CHECK_RET(isac_result_from_stack_ptx(stack_result));
    if (written_ptx + 1u > stub_capacity) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    stub[written_ptx] = '\0';
    ISAC_CHECK_RET(isac_arena_commit(arena, written_ptx + 1u));
    *stub_out = stub;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_build_ptx_for_batch(
    const ImplicitSindyAstColumnCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t batch_idx,
    IsacArena* arena,
    char** ptx_out,
    size_t* ptx_bytes_out
) {
    const char** stubs_by_kernel = NULL;
    const char** stubs_by_inject = NULL;
    void* stack_workspace = NULL;
    StackPtxInstruction* program = NULL;
    char* rendered_ptx = NULL;
    size_t rendered_capacity = 0u;
    size_t rendered_bytes = 0u;
    BinaryAST empty_ast;
    size_t kernel_idx;
    size_t inject_idx;

    if (compiler == NULL || asts == NULL || arena == NULL || ptx_out == NULL || ptx_bytes_out == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *ptx_out = NULL;
    *ptx_bytes_out = 0u;
    arena->offset = 0u;
    isac_make_empty_ast(&empty_ast);

    ISAC_CHECK_RET(isac_arena_alloc(
        arena,
        compiler->kernels_per_module * sizeof(*stubs_by_kernel),
        16u,
        (void**)&stubs_by_kernel
    ));
    ISAC_CHECK_RET(isac_arena_alloc(
        arena,
        compiler->num_injects * sizeof(*stubs_by_inject),
        16u,
        (void**)&stubs_by_inject
    ));

    ISAC_CHECK_RET(isac_arena_alloc(
        arena,
        compiler->stack_workspace_size,
        64u,
        &stack_workspace
    ));
    ISAC_CHECK_RET(isac_arena_alloc(
        arena,
        compiler->program_stride * sizeof(*program),
        64u,
        (void**)&program
    ));
    for (kernel_idx = 0u; kernel_idx < compiler->kernels_per_module; ++kernel_idx) {
        const size_t ast_idx = batch_idx * compiler->kernels_per_module + kernel_idx;
        const BinaryAST* ast = ast_idx < num_asts
            ? isac_ast_at(asts, ast_stride_bytes, ast_idx)
            : &empty_ast;
        ISAC_CHECK_RET(isac_emit_stack_ptx_stub(
            compiler,
            ast,
            arena,
            stack_workspace,
            program,
            (char**)&stubs_by_kernel[kernel_idx]
        ));
    }
    for (inject_idx = 0u; inject_idx < compiler->num_injects; ++inject_idx) {
        const size_t kernel = compiler->inject_to_kernel_idx[inject_idx];
        if (kernel >= compiler->kernels_per_module) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
        stubs_by_inject[inject_idx] = stubs_by_kernel[kernel];
    }
    ISAC_CHECK_RET(isac_arena_remaining(
        arena,
        1u,
        (void**)&rendered_ptx,
        &rendered_capacity
    ));
    ISAC_CHECK_RET(isac_result_from_ptx_inject(ptx_inject_render_ptx(
        compiler->inject,
        stubs_by_inject,
        compiler->num_injects,
        rendered_ptx,
        rendered_capacity,
        &rendered_bytes
    )));
    if (rendered_bytes + 1u > rendered_capacity) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    rendered_ptx[rendered_bytes] = '\0';
    ISAC_CHECK_RET(isac_arena_commit(arena, rendered_bytes + 1u));
    *ptx_out = rendered_ptx;
    *ptx_bytes_out = rendered_bytes;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
isac_compile_batch(
    const ImplicitSindyAstColumnCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t batch_idx,
    IsacArena* arena,
    void** out_cubins,
    size_t* out_cubin_sizes
) {
    char* ptx = NULL;
    size_t ptx_bytes = 0u;

    if (compiler == NULL || asts == NULL || arena == NULL || out_cubins == NULL || out_cubin_sizes == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    ISAC_CHECK_RET(isac_build_ptx_for_batch(
        compiler,
        asts,
        num_asts,
        ast_stride_bytes,
        batch_idx,
        arena,
        &ptx,
        &ptx_bytes
    ));
    ISAC_CHECK_RET(isac_compile_ptx_to_malloced_cubin(
        compiler,
        ptx,
        ptx_bytes,
        &out_cubins[batch_idx],
        &out_cubin_sizes[batch_idx]
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static void*
isac_worker_main(
    void* userdata
) {
    IsacWorker* worker = (IsacWorker*)userdata;
    size_t batch_idx;

    if (worker == NULL) return NULL;
    worker->result = IMPLICIT_SINDY_SUCCESS;
    for (batch_idx = worker->batch_begin; batch_idx < worker->batch_end; ++batch_idx) {
        worker->result = isac_compile_batch(
            worker->compiler,
            worker->asts,
            worker->num_asts,
            worker->ast_stride_bytes,
            batch_idx,
            &worker->arena,
            worker->out_cubins,
            worker->out_cubin_sizes
        );
        if (worker->result != IMPLICIT_SINDY_SUCCESS) break;
    }
    return NULL;
}

static ImplicitSindyResult
isac_prepare_workers(
    const ImplicitSindyAstColumnCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t num_batches,
    size_t ast_stride_bytes,
    void** out_cubins,
    size_t* out_cubin_sizes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    pthread_t** out_threads,
    IsacWorker** out_workers
) {
    uintptr_t p;
    uintptr_t end;
    size_t worker_bytes;
    size_t thread_bytes;
    size_t arena_per_worker;
    size_t i;
    IsacWorker* workers = NULL;
    pthread_t* threads = NULL;

    if (compiler == NULL || asts == NULL || num_batches == 0u || worker_memory == NULL ||
        out_threads == NULL || out_workers == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_threads = NULL;
    *out_workers = NULL;
    ISAC_CHECK_RET(isac_mul_size(worker_count, sizeof(*workers), &worker_bytes));
    ISAC_CHECK_RET(isac_mul_size(worker_count, sizeof(*threads), &thread_bytes));
    p = isac_align_up_uintptr((uintptr_t)worker_memory, 64u);
    end = (uintptr_t)worker_memory + worker_memory_size;
    if (end < (uintptr_t)worker_memory || p > end) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    if (worker_bytes > (size_t)(end - p)) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    workers = (IsacWorker*)p;
    memset(workers, 0, worker_bytes);
    p = isac_align_up_uintptr(p + worker_bytes, 64u);
    if (p > end || thread_bytes > (size_t)(end - p)) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    threads = (pthread_t*)p;
    memset(threads, 0, thread_bytes);
    p = isac_align_up_uintptr(p + thread_bytes, 64u);
    if (p > end) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    arena_per_worker = (size_t)(end - p) / worker_count;
    if (arena_per_worker == 0u) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);

    for (i = 0u; i < worker_count; ++i) {
        workers[i].compiler = compiler;
        workers[i].asts = asts;
        workers[i].num_asts = num_asts;
        workers[i].ast_stride_bytes = ast_stride_bytes;
        workers[i].out_cubins = out_cubins;
        workers[i].out_cubin_sizes = out_cubin_sizes;
        workers[i].arena.base = (unsigned char*)p + i * arena_per_worker;
        workers[i].arena.capacity = arena_per_worker;
        workers[i].arena.offset = 0u;
        workers[i].batch_begin = num_batches * i / worker_count;
        workers[i].batch_end = num_batches * (i + 1u) / worker_count;
        workers[i].result = IMPLICIT_SINDY_SUCCESS;
    }

    *out_threads = threads;
    *out_workers = workers;
    return IMPLICIT_SINDY_SUCCESS;
}

size_t
implicit_sindy_ast_column_cubin_count(
    size_t num_asts,
    size_t kernels_per_module
) {
    return isac_cubin_count(num_asts, kernels_per_module);
}

ImplicitSindyResult
implicit_sindy_ast_column_compiler_create(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstColumnCompiler** out_compiler
) {
    ImplicitSindyAstColumnCompiler* compiler;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;
    StackPtxResult stack_result;
    char* source = NULL;

    if (out_compiler == NULL || isac_kernels_per_module_count(kernels_per_module) == 0u) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_compiler = NULL;
    compiler = (ImplicitSindyAstColumnCompiler*)calloc(1u, sizeof(*compiler));
    if (compiler == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    compiler->program_stride = ISAC_PROGRAM_STRIDE;
    compiler->kernels_per_module = kernels_per_module;
    compiler->sm_major = sm_major;
    compiler->sm_minor = sm_minor;
    compiler->nvptx_options = nvptx_options;
    compiler->num_nvptx_options = num_nvptx_options;
    compiler->compiler_info.max_ast_size = compiler->program_stride;
    compiler->compiler_info.max_ast_to_visit_stack_depth = compiler->program_stride;
    compiler->compiler_info.stack_size = 256u;
    compiler->compiler_info.max_frame_depth = 8u;
    compiler->compiler_info.store_size = 32u;

    stack_result = stack_ptx_compile_workspace_size(
        &compiler->compiler_info,
        &stack_ptx_stack_info,
        &compiler->stack_workspace_size
    );
    error_ret = isac_result_from_stack_ptx(stack_result);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = isac_generate_column_template_source(
            compiler->kernels_per_module,
            &source
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = isac_compile_cuda_to_ptx(
            source,
            "implicit_sindy_ast_column_template.cu",
            sm_major,
            sm_minor,
            &compiler->template_ptx,
            &compiler->template_ptx_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = isac_result_from_ptx_inject(ptx_inject_create(
            &compiler->inject,
            compiler->template_ptx
        ));
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = isac_setup_inject_order(compiler);
    }

    free(source);
    if (error_ret != IMPLICIT_SINDY_SUCCESS) {
        if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
        free(compiler->inject_to_kernel_idx);
        free(compiler->template_ptx);
        free(compiler);
        ISAC_ERROR_RET(error_ret);
    }

    *out_compiler = compiler;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_column_compiler_destroy(
    ImplicitSindyAstColumnCompiler* compiler
) {
    if (compiler == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
    free(compiler->inject_to_kernel_idx);
    free(compiler->template_ptx);
    free(compiler);
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_column_compile_workspace_size(
    const ImplicitSindyAstColumnCompiler* compiler,
    size_t num_asts,
    size_t worker_count,
    size_t scratch_bytes_per_worker,
    size_t* out_bytes
) {
    size_t workers;
    size_t worker_bytes;
    size_t thread_bytes;
    size_t scratch_total;
    size_t num_batches;
    size_t total = 128u;

    if (compiler == NULL || out_bytes == NULL || num_asts == 0u || scratch_bytes_per_worker == 0u) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_bytes = 0u;
    num_batches = isac_cubin_count(num_asts, compiler->kernels_per_module);
    if (num_batches == 0u) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    workers = isac_default_worker_count(worker_count, num_batches);
    ISAC_CHECK_RET(isac_mul_size(workers, sizeof(IsacWorker), &worker_bytes));
    ISAC_CHECK_RET(isac_mul_size(workers, sizeof(pthread_t), &thread_bytes));
    ISAC_CHECK_RET(isac_mul_size(workers, scratch_bytes_per_worker, &scratch_total));
    ISAC_CHECK_RET(isac_add_size(total, isac_align_up_size(worker_bytes, 64u), &total));
    ISAC_CHECK_RET(isac_add_size(total, isac_align_up_size(thread_bytes, 64u), &total));
    ISAC_CHECK_RET(isac_add_size(total, scratch_total, &total));
    *out_bytes = total;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_column_compile_cubins(
    const ImplicitSindyAstColumnCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    void** out_cubins,
    size_t* out_cubin_sizes
) {
    pthread_t* threads = NULL;
    IsacWorker* workers = NULL;
    ImplicitSindyResult result = IMPLICIT_SINDY_SUCCESS;
    size_t workers_count;
    size_t num_batches;
    size_t i;

    if (compiler == NULL || asts == NULL || num_asts == 0u ||
        ast_stride_bytes < sizeof(BinaryAST) || worker_memory == NULL ||
        worker_memory_size == 0u || out_cubins == NULL || out_cubin_sizes == NULL) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    num_batches = isac_cubin_count(num_asts, compiler->kernels_per_module);
    if (num_batches == 0u) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    workers_count = isac_default_worker_count(worker_count, num_batches);
    for (i = 0u; i < num_batches; ++i) {
        out_cubins[i] = NULL;
        out_cubin_sizes[i] = 0u;
    }

    ISAC_CHECK_RET(isac_prepare_workers(
        compiler,
        asts,
        num_asts,
        num_batches,
        ast_stride_bytes,
        out_cubins,
        out_cubin_sizes,
        workers_count,
        worker_memory,
        worker_memory_size,
        &threads,
        &workers
    ));

    for (i = 0u; i < workers_count; ++i) {
        if (pthread_create(&threads[i], NULL, isac_worker_main, &workers[i]) != 0) {
            result = IMPLICIT_SINDY_ERROR_THREAD;
            workers_count = i;
            break;
        }
    }
    for (i = 0u; i < workers_count; ++i) {
        if (pthread_join(threads[i], NULL) != 0 && result == IMPLICIT_SINDY_SUCCESS) {
            result = IMPLICIT_SINDY_ERROR_THREAD;
        }
        if (workers[i].result != IMPLICIT_SINDY_SUCCESS && result == IMPLICIT_SINDY_SUCCESS) {
            result = workers[i].result;
        }
    }
    if (result != IMPLICIT_SINDY_SUCCESS) ISAC_ERROR_RET(result);
    for (i = 0u; i < num_batches; ++i) {
        if (out_cubins[i] == NULL || out_cubin_sizes[i] == 0u) {
            ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
        }
    }
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_column_module_create(
    void* module,
    size_t kernels_per_module,
    ImplicitSindyAstColumnModule** out_column_module
) {
    ImplicitSindyAstColumnModule* column_module;
    size_t kernel_count;
    size_t allocation_size;
    size_t i;

    if (module == NULL || out_column_module == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out_column_module = NULL;
    kernel_count = isac_kernels_per_module_count(kernels_per_module);
    if (kernel_count == 0u ||
        kernel_count > (SIZE_MAX - sizeof(*column_module)) / sizeof(column_module->functions[0])) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    allocation_size = sizeof(*column_module) + kernel_count * sizeof(column_module->functions[0]);
    column_module = (ImplicitSindyAstColumnModule*)calloc(1u, allocation_size);
    if (column_module == NULL) ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    column_module->module = (CUmodule)module;
    column_module->kernels_per_module = kernel_count;
    for (i = 0u; i < kernel_count; ++i) {
        char symbol[64];
        if (kernel_count == 1u) {
            strcpy(symbol, "implicit_sindy_ast_column_kernel");
        } else {
            int bytes_written = snprintf(
                symbol,
                sizeof(symbol),
                "implicit_sindy_ast_column_kernel_%zu",
                i
            );
            if (bytes_written < 0 || (size_t)bytes_written >= sizeof(symbol)) {
                free(column_module);
                ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
            }
        }
        if (cuModuleGetFunction(
                &column_module->functions[i],
                column_module->module,
                symbol
            ) != CUDA_SUCCESS) {
            free(column_module);
            ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_CUDA);
        }
    }
    *out_column_module = column_module;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_column_module_destroy(
    ImplicitSindyAstColumnModule* column_module
) {
    free(column_module);
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_column_launch(
    ImplicitSindyAstColumnModule* column_module,
    size_t kernel_index,
    void* stream,
    int64_t feature_index,
    int64_t num_settings,
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* output,
    int64_t output_setting_stride
) {
    void* args[10];
    uint64_t row_blocks_u64;
    unsigned int row_blocks;

    if (column_module == NULL ||
        kernel_index >= column_module->kernels_per_module ||
        column_module->functions[kernel_index] == NULL ||
        feature_index < 0 ||
        feature_index >= IMPLICIT_SINDY_INTERNAL_FEATURES ||
        num_settings <= 0 ||
        num_settings > (int64_t)UINT_MAX ||
        primitive_features == NULL ||
        row_count <= 0 ||
        primitive_feature_stride <= 0 ||
        num_primitive_features <= 0 ||
        num_primitive_features > IMPLICIT_SINDY_INTERNAL_PRIMITIVE_FEATURES_MAX ||
        leaf_masks == NULL ||
        leaf_words == NULL ||
        leaf_words_feature_stride < IMPLICIT_SINDY_INTERNAL_LEAVES ||
        output == NULL ||
        output_setting_stride < row_count) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    row_blocks_u64 = ((uint64_t)row_count + (uint64_t)ISAC_THREADS - 1u) / (uint64_t)ISAC_THREADS;
    if (row_blocks_u64 == 0u || row_blocks_u64 > (uint64_t)UINT_MAX) {
        ISAC_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    row_blocks = (unsigned int)row_blocks_u64;

    args[0] = (void*)&primitive_features;
    args[1] = &row_count;
    args[2] = &primitive_feature_stride;
    args[3] = &num_primitive_features;
    args[4] = &feature_index;
    args[5] = (void*)&leaf_masks;
    args[6] = (void*)&leaf_words;
    args[7] = &leaf_words_feature_stride;
    args[8] = &output;
    args[9] = &output_setting_stride;

    ISAC_CHECK_CUDA_RET(cuLaunchKernel(
        column_module->functions[kernel_index],
        (unsigned int)num_settings,
        row_blocks,
        1u,
        ISAC_THREADS,
        1u,
        1u,
        0u,
        (CUstream)stream,
        args,
        NULL
    ));

    return IMPLICIT_SINDY_SUCCESS;
}
