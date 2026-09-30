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
#ifndef IMPLICIT_SINDY_AST_COMPILER_H
#define IMPLICIT_SINDY_AST_COMPILER_H

#include "implicit_sindy.h"

/*
 * Single-header C99 API for compiling fixed-depth BinaryAST cohorts into
 * patched implicit-feature Gram cubins.
 *
 * Usage:
 *
 *   #define IMPLICIT_SINDY_AST_COMPILER_IMPLEMENTATION
 *   #include "implicit_sindy_ast_compiler.h"
 *
 * The implementation depends on CUDA Driver, NVRTC, nvPTXCompiler, nvJitLink,
 * ptx_inject.h, stack_ptx.h, binary_ast_to_stack_ptx.h, cubin_function_patch.h,
 * and the generated stack_ptx_descriptions.h include.
 */

IMPLICIT_SINDY_PUBLIC_DEC const char* implicit_sindy_result_to_string(ImplicitSindyResult result);
IMPLICIT_SINDY_PUBLIC_DEC size_t implicit_sindy_ast_cubin_count(size_t num_asts, size_t kernels_per_module);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_cubin(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_cubin,
    size_t gram_template_cubin_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_ptx(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_ptx,
    size_t gram_template_ptx_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compiler_create_columns(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_ast_compiler_destroy(ImplicitSindyAstCompiler* compiler);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compile_workspace_size(
    const ImplicitSindyAstCompiler* compiler,
    size_t num_asts,
    size_t worker_count,
    size_t scratch_bytes_per_worker,
    size_t* out_bytes
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_ast_compile_cubins(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    void** out_cubins,
    size_t* out_cubin_sizes
);

IMPLICIT_SINDY_PUBLIC_DEC void implicit_sindy_free_cubins(void** cubins, size_t* cubin_sizes, size_t num_cubins);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult
implicit_sindy_load_cubin_modules(
    void* const* cubins,
    const size_t* cubin_sizes,
    size_t num_cubins,
    void** out_modules
);

IMPLICIT_SINDY_PUBLIC_DEC ImplicitSindyResult implicit_sindy_unload_modules(void** modules, size_t num_modules);

#endif

#ifdef IMPLICIT_SINDY_AST_COMPILER_IMPLEMENTATION
#ifndef IMPLICIT_SINDY_AST_COMPILER_IMPLEMENTATION_ONCE
#define IMPLICIT_SINDY_AST_COMPILER_IMPLEMENTATION_ONCE

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#define BINARY_AST_TO_STACK_PTX_IMPLEMENTATION
#define CUBIN_FUNCTION_PATCH_IMPLEMENTATION
#define PTX_INJECT_IMPLEMENTATION
#define STACK_PTX_IMPLEMENTATION

#ifndef IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INCLUDE
#error "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INCLUDE must point to the public include directory"
#endif

#ifndef IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INTERNAL
#error "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INTERNAL must point to the internal source include directory"
#endif

#ifndef IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_CUDA
#error "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_CUDA must point to the CUDA source include directory"
#endif

#include "binary_ast_to_stack_ptx.h"
#include "ptx_inject.h"
#include "stack_ptx.h"

#include <stack_ptx_descriptions.h>

#include <cuda.h>
#include <cubin_function_patch.h>
#include <nvJitLink.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define IMPLICIT_SINDY_AST_PROGRAM_STRIDE 1024u
#define IMPLICIT_SINDY_AST_FEATURES_PER_KERNEL 32u
#define IMPLICIT_SINDY_AST_LEAVES_PER_FEATURE 8u
#define IMPLICIT_SINDY_AST_EVAL_PTX_CAPACITY (8u * 1024u * 1024u)
#define IMPLICIT_SINDY_AST_MAX_KERNELS_PER_MODULE 64u

#define IMPLICIT_SINDY_ERROR_RET(ans)              \
    do {                                           \
        ImplicitSindyResult is_result__ = (ans);   \
        return is_result__;                        \
    } while (0)

#define IMPLICIT_SINDY_CHECK_RET(ans)              \
    do {                                           \
        ImplicitSindyResult is_result__ = (ans);   \
        if (is_result__ != IMPLICIT_SINDY_SUCCESS) { \
            IMPLICIT_SINDY_ERROR_RET(is_result__); \
        }                                          \
    } while (0)

typedef struct {
    unsigned char* base;
    size_t capacity;
    size_t offset;
} ImplicitSindyArena;

typedef struct {
    char* data;
    size_t size;
    size_t capacity;
} ImplicitSindyString;

struct ImplicitSindyAstCompiler {
    PtxInjectHandle inject;
    CubinFunctionPatchHandle* patch_handle;
    void* patch_handle_memory;
    void* reserved_cubin;
    size_t reserved_cubin_bytes;
    char* eval_template_ptx;
    size_t eval_template_ptx_bytes;
    size_t* inject_to_feature_idx;
    size_t num_injects;
    StackPtxCompilerInfo compiler_info;
    size_t stack_workspace_size;
    size_t kernels_per_module;
    size_t kernel_count;
    size_t features_per_cubin;
    size_t program_stride;
    unsigned int sm_major;
    unsigned int sm_minor;
    const char* const* nvptx_options;
    size_t num_nvptx_options;
};

typedef struct {
    const ImplicitSindyAstCompiler* compiler;
    const BinaryAST* asts;
    size_t num_asts;
    size_t ast_stride_bytes;
    void** out_cubins;
    size_t* out_cubin_sizes;
    ImplicitSindyArena arena;
    size_t batch_begin;
    size_t batch_end;
    ImplicitSindyResult result;
} ImplicitSindyAstWorker;

static const StackPtxRegister implicit_sindy_ast_registers[] = {
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

static const size_t implicit_sindy_ast_requests[] = { BINARY_AST_NUM_INPUTS };

static ImplicitSindyResult
implicit_sindy_mul_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (a != 0u && b > SIZE_MAX / a) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out = a * b;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_add_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (b > SIZE_MAX - a) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out = a + b;
    return IMPLICIT_SINDY_SUCCESS;
}

static size_t
implicit_sindy_align_up_size(
    size_t value,
    size_t alignment
) {
    if (alignment == 0u) return value;
    return ((value + alignment - 1u) / alignment) * alignment;
}

static uintptr_t
implicit_sindy_align_up_uintptr(
    uintptr_t value,
    uintptr_t alignment
) {
    if (alignment == 0u) return value;
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static ImplicitSindyResult
implicit_sindy_arena_alloc(
    ImplicitSindyArena* arena,
    size_t size,
    size_t alignment,
    void** out_ptr
) {
    size_t aligned_offset;
    if (arena == NULL || arena->base == NULL || out_ptr == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    aligned_offset = implicit_sindy_align_up_size(arena->offset, alignment);
    if (aligned_offset > arena->capacity || size > arena->capacity - aligned_offset) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    *out_ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_arena_remaining(
    ImplicitSindyArena* arena,
    size_t alignment,
    void** out_ptr,
    size_t* out_bytes
) {
    size_t aligned_offset;
    if (arena == NULL || arena->base == NULL || out_ptr == NULL || out_bytes == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    aligned_offset = implicit_sindy_align_up_size(arena->offset, alignment);
    if (aligned_offset > arena->capacity) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    arena->offset = aligned_offset;
    *out_ptr = arena->base + aligned_offset;
    *out_bytes = arena->capacity - aligned_offset;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_arena_commit(
    ImplicitSindyArena* arena,
    size_t size
) {
    if (arena == NULL || size > arena->capacity - arena->offset) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    arena->offset += size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_result_from_binary_ast(
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
implicit_sindy_result_from_stack_ptx(
    StackPtxResult result
) {
    return result == STACK_PTX_SUCCESS
        ? IMPLICIT_SINDY_SUCCESS
        : IMPLICIT_SINDY_ERROR_STACK_PTX;
}

static ImplicitSindyResult
implicit_sindy_result_from_ptx_inject(
    PtxInjectResult result
) {
    return result == PTX_INJECT_SUCCESS
        ? IMPLICIT_SINDY_SUCCESS
        : IMPLICIT_SINDY_ERROR_PTX_INJECT;
}

static size_t
implicit_sindy_kernels_per_module_count(
    size_t kernels_per_module
) {
    if (kernels_per_module == 0u ||
        kernels_per_module > IMPLICIT_SINDY_AST_MAX_KERNELS_PER_MODULE) {
        return 0u;
    }
    return kernels_per_module;
}

static size_t
implicit_sindy_default_worker_count(
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

static const BinaryAST*
implicit_sindy_ast_at(
    const BinaryAST* asts,
    size_t ast_stride_bytes,
    size_t idx
) {
    const unsigned char* base = (const unsigned char*)asts;
    return (const BinaryAST*)(const void*)(base + idx * ast_stride_bytes);
}

static void
implicit_sindy_ast_make_padding_ast(
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
implicit_sindy_ast_make_max_ast(
    BinaryAST* ast
) {
    size_t i;
    if (ast == NULL) return;
    memset(ast, 0, sizeof(*ast));
    for (i = 0u; i < BINARY_AST_NUM_UNARY_OPS; ++i) {
        ast->unary[i] = BINARY_AST_UNARY_SAFE_EXP_F32;
    }
    for (i = 0u; i < BINARY_AST_NUM_BINARY_OPS; ++i) {
        ast->binary[i] = BINARY_AST_BINARY_SAFE_DIV_F32;
    }
}

static void
implicit_sindy_string_free(
    ImplicitSindyString* string
) {
    if (string == NULL) return;
    free(string->data);
    string->data = NULL;
    string->size = 0u;
    string->capacity = 0u;
}

static ImplicitSindyResult
implicit_sindy_string_reserve(
    ImplicitSindyString* string,
    size_t needed
) {
    char* next;
    size_t next_capacity;
    if (string == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (needed <= string->capacity) return IMPLICIT_SINDY_SUCCESS;
    next_capacity = string->capacity != 0u ? string->capacity : 4096u;
    while (next_capacity < needed) {
        if (next_capacity > SIZE_MAX / 2u) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
        next_capacity *= 2u;
    }
    next = (char*)realloc(string->data, next_capacity);
    if (next == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    string->data = next;
    string->capacity = next_capacity;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_string_append(
    ImplicitSindyString* string,
    const char* text
) {
    size_t len;
    if (string == NULL || text == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    len = strlen(text);
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_reserve(string, string->size + len + 1u));
    memcpy(string->data + string->size, text, len + 1u);
    string->size += len;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_string_appendf(
    ImplicitSindyString* string,
    const char* format,
    ...
) {
    va_list args;
    va_list args_copy;
    int needed_i;
    size_t needed;

    if (string == NULL || format == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    va_start(args, format);
    va_copy(args_copy, args);
    needed_i = vsnprintf(NULL, 0u, format, args_copy);
    va_end(args_copy);
    if (needed_i < 0) {
        va_end(args);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
    }
    needed = (size_t)needed_i;
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_reserve(string, string->size + needed + 1u));
    if (vsnprintf(string->data + string->size, string->capacity - string->size, format, args) != needed_i) {
        va_end(args);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
    }
    va_end(args);
    string->size += needed;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_compile_cuda_to_ptx(
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
    char include_option_0[1024];
    char include_option_1[1024];
    char include_option_2[1024];
    const char* options[8];
    int option_count = 0;
    size_t ptx_size = 0u;
    char* ptx = NULL;

    if (source == NULL || program_name == NULL || ptx_out == NULL || ptx_bytes_out == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *ptx_out = NULL;
    *ptx_bytes_out = 0u;

    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%u%u", sm_major, sm_minor);
    snprintf(include_option_0, sizeof(include_option_0), "--include-path=%s", IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INCLUDE);
    snprintf(include_option_1, sizeof(include_option_1), "--include-path=%s", IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INTERNAL);
    snprintf(include_option_2, sizeof(include_option_2), "--include-path=%s", IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_CUDA);
    options[option_count++] = arch_option;
    options[option_count++] = "--std=c++17";
    options[option_count++] = "--device-c";
    options[option_count++] = "--use_fast_math";
    options[option_count++] = include_option_0;
    options[option_count++] = include_option_1;
    options[option_count++] = include_option_2;

    result = nvrtcCreateProgram(&program, source, program_name, 0, NULL, NULL);
    if (result != NVRTC_SUCCESS) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    result = nvrtcCompileProgram(program, option_count, options);
    if (result != NVRTC_SUCCESS) {
        size_t log_size = 0u;
        (void)nvrtcGetProgramLogSize(program, &log_size);
        if (log_size > 1u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    fprintf(stderr, "implicit-sindy nvrtc log:\n%s\n", log);
                }
                free(log);
            }
        }
        (void)nvrtcDestroyProgram(&program);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    if (nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS || ptx_size == 0u) {
        (void)nvrtcDestroyProgram(&program);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    ptx = (char*)malloc(ptx_size);
    if (ptx == NULL) {
        (void)nvrtcDestroyProgram(&program);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    }
    if (nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) {
        free(ptx);
        (void)nvrtcDestroyProgram(&program);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    (void)nvrtcDestroyProgram(&program);
    *ptx_out = ptx;
    *ptx_bytes_out = ptx_size != 0u ? ptx_size - 1u : 0u;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_compile_ptx_to_rdc(
    const char* ptx,
    size_t ptx_bytes,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* extra_options,
    size_t num_extra_options,
    ImplicitSindyArena* arena,
    void** object_out,
    size_t* object_bytes_out
) {
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult result;
    const char* options[80];
    char arch_option[64];
    size_t option_count = 0u;
    size_t image_size = 0u;
    void* image = NULL;
    size_t i;

    if (ptx == NULL || ptx_bytes == 0u || arena == NULL || object_out == NULL || object_bytes_out == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *object_out = NULL;
    *object_bytes_out = 0u;
    if (num_extra_options + 4u > sizeof(options) / sizeof(options[0])) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    snprintf(arch_option, sizeof(arch_option), "--gpu-name=sm_%u%u", sm_major, sm_minor);
    options[option_count++] = arch_option;
    options[option_count++] = "--compile-only";
    options[option_count++] = "--opt-level=3";
    options[option_count++] = "--allow-expensive-optimizations=false";
    for (i = 0u; i < num_extra_options; ++i) {
        options[option_count++] = extra_options[i];
    }

    result = nvPTXCompilerCreate(&compiler, ptx_bytes, ptx);
    if (result != NVPTXCOMPILE_SUCCESS) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    result = nvPTXCompilerCompile(compiler, (int)option_count, options);
    if (result != NVPTXCOMPILE_SUCCESS) {
        size_t log_size = 0u;
        (void)nvPTXCompilerGetErrorLogSize(compiler, &log_size);
        if (log_size > 1u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                if (nvPTXCompilerGetErrorLog(compiler, log) == NVPTXCOMPILE_SUCCESS) {
                    fprintf(stderr, "implicit-sindy nvPTXCompiler log:\n%s\n", log);
                }
                free(log);
            }
        }
        (void)nvPTXCompilerDestroy(&compiler);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    if (nvPTXCompilerGetCompiledProgramSize(compiler, &image_size) != NVPTXCOMPILE_SUCCESS || image_size == 0u) {
        (void)nvPTXCompilerDestroy(&compiler);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_alloc(arena, image_size, 64u, &image));
    if (nvPTXCompilerGetCompiledProgram(compiler, image) != NVPTXCOMPILE_SUCCESS) {
        (void)nvPTXCompilerDestroy(&compiler);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    (void)nvPTXCompilerDestroy(&compiler);
    *object_out = image;
    *object_bytes_out = image_size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_link_reserved_cubin(
    const char* gram_ptx,
    size_t gram_ptx_bytes,
    const void* eval_object,
    size_t eval_object_bytes,
    unsigned int sm_major,
    unsigned int sm_minor,
    void** cubin_out,
    size_t* cubin_bytes_out
) {
    nvJitLinkHandle link = NULL;
    nvJitLinkResult result;
    const char* options[3];
    char arch_option[64];
    size_t cubin_size = 0u;
    void* cubin = NULL;

    if (gram_ptx == NULL || gram_ptx_bytes == 0u || eval_object == NULL ||
        eval_object_bytes == 0u || cubin_out == NULL || cubin_bytes_out == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *cubin_out = NULL;
    *cubin_bytes_out = 0u;

    snprintf(arch_option, sizeof(arch_option), "-arch=sm_%u%u", sm_major, sm_minor);
    options[0] = arch_option;
    options[1] = "-O3";
    options[2] = "-no-cache";
    result = nvJitLinkCreate(&link, 3u, options);
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(
            link,
            NVJITLINK_INPUT_PTX,
            (void*)gram_ptx,
            gram_ptx_bytes,
            "implicit_feature_gram_patch.ptx"
        );
    }
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(
            link,
            NVJITLINK_INPUT_CUBIN,
            (void*)eval_object,
            eval_object_bytes,
            "implicit_feature_eval_reserved.cubin"
        );
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkComplete(link);
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubinSize(link, &cubin_size);
    if (result == NVJITLINK_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin == NULL) {
            if (link != NULL) (void)nvJitLinkDestroy(&link);
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
        }
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubin(link, cubin);
    if (result != NVJITLINK_SUCCESS) {
        size_t log_size = 0u;
        if (link != NULL && nvJitLinkGetErrorLogSize(link, &log_size) == NVJITLINK_SUCCESS && log_size > 1u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                if (nvJitLinkGetErrorLog(link, log) == NVJITLINK_SUCCESS) {
                    fprintf(stderr, "implicit-sindy nvJitLink log:\n%s\n", log);
                }
                free(log);
            }
        }
        if (link != NULL) (void)nvJitLinkDestroy(&link);
        free(cubin);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    if (link != NULL) (void)nvJitLinkDestroy(&link);
    *cubin_out = cubin;
    *cubin_bytes_out = cubin_size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_link_reserved_cubin_from_gram_cubin(
    const void* gram_cubin,
    size_t gram_cubin_bytes,
    const void* eval_object,
    size_t eval_object_bytes,
    unsigned int sm_major,
    unsigned int sm_minor,
    void** cubin_out,
    size_t* cubin_bytes_out
) {
    nvJitLinkHandle link = NULL;
    nvJitLinkResult result;
    const char* options[3];
    char arch_option[64];
    size_t cubin_size = 0u;
    void* cubin = NULL;

    if (gram_cubin == NULL || gram_cubin_bytes == 0u || eval_object == NULL ||
        eval_object_bytes == 0u || cubin_out == NULL || cubin_bytes_out == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *cubin_out = NULL;
    *cubin_bytes_out = 0u;

    snprintf(arch_option, sizeof(arch_option), "-arch=sm_%u%u", sm_major, sm_minor);
    options[0] = arch_option;
    options[1] = "-O3";
    options[2] = "-no-cache";
    result = nvJitLinkCreate(&link, 3u, options);
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(
            link,
            NVJITLINK_INPUT_CUBIN,
            (void*)gram_cubin,
            gram_cubin_bytes,
            "implicit_feature_gram_template.cubin"
        );
    }
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(
            link,
            NVJITLINK_INPUT_CUBIN,
            (void*)eval_object,
            eval_object_bytes,
            "implicit_feature_eval_reserved.cubin"
        );
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkComplete(link);
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubinSize(link, &cubin_size);
    if (result == NVJITLINK_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin == NULL) {
            if (link != NULL) (void)nvJitLinkDestroy(&link);
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
        }
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubin(link, cubin);
    if (result != NVJITLINK_SUCCESS) {
        size_t log_size = 0u;
        if (link != NULL && nvJitLinkGetErrorLogSize(link, &log_size) == NVJITLINK_SUCCESS && log_size > 1u) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                if (nvJitLinkGetErrorLog(link, log) == NVJITLINK_SUCCESS) {
                    fprintf(stderr, "implicit-sindy nvJitLink log:\n%s\n", log);
                }
                free(log);
            }
        }
        if (link != NULL) (void)nvJitLinkDestroy(&link);
        free(cubin);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_NVPTX);
    }
    if (link != NULL) (void)nvJitLinkDestroy(&link);
    *cubin_out = cubin;
    *cubin_bytes_out = cubin_size;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_append_eval_site(
    ImplicitSindyString* source,
    size_t feature_idx,
    size_t site_idx
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
        source,
        "                case %zu: {\n"
        "                    asm volatile(\n"
        "                        \"{\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x0;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x1;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x2;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x3;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x4;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x5;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x6;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x7;\\n\\t\"\n"
        "                        \".reg .f32 %%%%_x8;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x0, %%0;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x1, %%1;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x2, %%2;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x3, %%3;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x4, %%4;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x5, %%5;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x6, %%6;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x7, %%7;\\n\\t\"\n"
        "                        \"mov.f32 %%%%_x8, %%8;\\n\\t\"\n"
        "                        \"// PTX_INJECT_START implicit_feature_%zu\\n\\t\"\n"
        "                        \"// _x0 m f32 F32 value\\n\\t\"\n"
        "                        \"// _x1 i f32 F32 leaf0\\n\\t\"\n"
        "                        \"// _x2 i f32 F32 leaf1\\n\\t\"\n"
        "                        \"// _x3 i f32 F32 leaf2\\n\\t\"\n"
        "                        \"// _x4 i f32 F32 leaf3\\n\\t\"\n"
        "                        \"// _x5 i f32 F32 leaf4\\n\\t\"\n"
        "                        \"// _x6 i f32 F32 leaf5\\n\\t\"\n"
        "                        \"// _x7 i f32 F32 leaf6\\n\\t\"\n"
        "                        \"// _x8 i f32 F32 leaf7\\n\\t\"\n"
        "                        \"// PTX_INJECT_END\\n\\t\"\n"
        "                        \"mov.f32 %%0, %%%%_x0;\\n\\t\"\n"
        "                        \"}\"\n"
        "                        : \"+f\"(value)\n"
        "                        : \"f\"(leaf0), \"f\"(leaf1), \"f\"(leaf2), \"f\"(leaf3),\n"
        "                          \"f\"(leaf4), \"f\"(leaf5), \"f\"(leaf6), \"f\"(leaf7));\n"
        "                    break;\n"
        "                }\n",
        feature_idx,
        site_idx
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_generate_eval_template_source(
    size_t kernel_count,
    char** source_out
) {
    ImplicitSindyString source;
    size_t kernel_idx;
    size_t feature_idx;

    if (source_out == NULL || kernel_count == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *source_out = NULL;
    memset(&source, 0, sizeof(source));

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        &source,
        "extern \"C\" {\n"
        "typedef long long int64_t;\n"
        "}\n\n"
    ));

    for (kernel_idx = 0u; kernel_idx < kernel_count; ++kernel_idx) {
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
            &source,
            "extern \"C\" __device__ __noinline__\n"
            "void\n"
            "implicit_feature_eval_%zu(\n"
            "    float* implicit_panel,\n"
            "    const float* const* leaf_ptrs,\n"
            "    const int* leaf_strides,\n"
            "    int active_rows\n"
            ") {\n"
            "    const int tid = static_cast<int>(threadIdx.x);\n"
            "    for (int linear = tid; linear < 32 * 128; linear += 128) {\n"
            "        const int feature = linear >> 7;\n"
            "        const int row = linear & 127;\n"
            "        float value = 0.0f;\n"
            "        if (row < active_rows) {\n"
            "            const int base = feature * 8;\n"
            "            const float leaf0 = leaf_ptrs[base + 0][row * leaf_strides[base + 0]];\n"
            "            const float leaf1 = leaf_ptrs[base + 1][row * leaf_strides[base + 1]];\n"
            "            const float leaf2 = leaf_ptrs[base + 2][row * leaf_strides[base + 2]];\n"
            "            const float leaf3 = leaf_ptrs[base + 3][row * leaf_strides[base + 3]];\n"
            "            const float leaf4 = leaf_ptrs[base + 4][row * leaf_strides[base + 4]];\n"
            "            const float leaf5 = leaf_ptrs[base + 5][row * leaf_strides[base + 5]];\n"
            "            const float leaf6 = leaf_ptrs[base + 6][row * leaf_strides[base + 6]];\n"
            "            const float leaf7 = leaf_ptrs[base + 7][row * leaf_strides[base + 7]];\n"
            "            switch (feature) {\n",
            kernel_idx
        ));
        for (feature_idx = 0u; feature_idx < IMPLICIT_SINDY_AST_FEATURES_PER_KERNEL; ++feature_idx) {
            IMPLICIT_SINDY_CHECK_RET(implicit_sindy_append_eval_site(
                &source,
                feature_idx,
                kernel_idx * IMPLICIT_SINDY_AST_FEATURES_PER_KERNEL + feature_idx
            ));
        }
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
            &source,
            "                default: value = 0.0f; break;\n"
            "            }\n"
            "        }\n"
            "        implicit_panel[linear] = value;\n"
            "    }\n"
            "}\n\n"
        ));
    }

    *source_out = source.data;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_append_kernel_args(
    ImplicitSindyString* source
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        source,
        "    const float* primitive_features,\n"
        "    int64_t row_count,\n"
        "    int64_t primitive_feature_stride,\n"
        "    int64_t num_primitive_features,\n"
        "    const float* targets,\n"
        "    int64_t target_rhs_stride,\n"
        "    int64_t num_target_rhs,\n"
        "    const int32_t* leaf_masks,\n"
        "    const int32_t* leaf_words,\n"
        "    int64_t leaf_words_feature_stride,\n"
        "    float* gram,\n"
        "    int64_t gram_col_stride,\n"
        "    float* x_sum,\n"
        "    float* xty,\n"
        "    int64_t xty_rhs_stride,\n"
        "    float* y_sum,\n"
        "    float* yy\n"
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_append_kernel_forward_args(
    ImplicitSindyString* source
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        source,
        "        primitive_features,\n"
        "        row_count,\n"
        "        primitive_feature_stride,\n"
        "        num_primitive_features,\n"
        "        targets,\n"
        "        target_rhs_stride,\n"
        "        num_target_rhs,\n"
        "        leaf_masks,\n"
        "        leaf_words,\n"
        "        leaf_words_feature_stride,\n"
        "        gram,\n"
        "        gram_col_stride,\n"
        "        x_sum,\n"
        "        xty,\n"
        "        xty_rhs_stride,\n"
        "        y_sum,\n"
        "        yy\n"
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_generate_gram_template_source(
    size_t kernel_count,
    char** source_out
) {
    ImplicitSindyString source;
    size_t kernel_idx;

    if (source_out == NULL || kernel_count == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *source_out = NULL;
    memset(&source, 0, sizeof(source));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        &source,
        "#define IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL 1\n"
        "#include \"implicit_feature_gram.cuh\"\n\n"
    ));

    for (kernel_idx = 0u; kernel_idx < kernel_count; ++kernel_idx) {
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
            &source,
            "extern \"C\" __device__ __noinline__\n"
            "void\n"
            "implicit_feature_eval_%zu(\n"
            "    float* implicit_panel,\n"
            "    const float* const* leaf_ptrs,\n"
            "    const int* leaf_strides,\n"
            "    int active_rows\n"
            ");\n\n"
            "namespace implicit_sindy_feature_gram_kernel {\n"
            "template <>\n"
            "__device__ __forceinline__\n"
            "void\n"
            "implicit_feature_eval_selector<%zu>(\n"
            "    float* implicit_panel,\n"
            "    const float* const* leaf_ptrs,\n"
            "    const int* leaf_strides,\n"
            "    int active_rows\n"
            ") {\n"
            "    ::implicit_feature_eval_%zu(\n"
            "        implicit_panel,\n"
            "        leaf_ptrs,\n"
            "        leaf_strides,\n"
            "        active_rows\n"
            "    );\n"
            "}\n"
            "}  // namespace implicit_sindy_feature_gram_kernel\n\n",
            kernel_idx,
            kernel_idx,
            kernel_idx
        ));
    }

    for (kernel_idx = 0u; kernel_idx < kernel_count; ++kernel_idx) {
        if (kernel_count == 1u) {
            IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
                &source,
                "extern \"C\" __global__\n"
                "__launch_bounds__(implicit_sindy_feature_gram_kernel::kThreads)\n"
                "void\n"
                "implicit_feature_gram_kernel(\n"
            ));
        } else {
            IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
                &source,
                "extern \"C\" __global__\n"
                "__launch_bounds__(implicit_sindy_feature_gram_kernel::kThreads)\n"
                "void\n"
                "implicit_feature_gram_kernel_%zu(\n",
                kernel_idx
            ));
        }
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_append_kernel_args(&source));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
            &source,
            ") {\n"
            "    implicit_sindy_feature_gram_kernel::implicit_feature_gram_kernel_device<%zu>(\n",
            kernel_idx
        ));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_append_kernel_forward_args(&source));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
            &source,
            "    );\n"
            "}\n\n"
        ));
    }

    *source_out = source.data;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_append_columns_kernel_args(
    ImplicitSindyString* source
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        source,
        "    const float* primitive_features,\n"
        "    int64_t row_count,\n"
        "    int64_t primitive_feature_stride,\n"
        "    int64_t num_primitive_features,\n"
        "    const int32_t* leaf_masks,\n"
        "    const int32_t* leaf_words,\n"
        "    int64_t leaf_words_feature_stride,\n"
        "    float* output,\n"
        "    int64_t output_setting_stride,\n"
        "    int64_t output_feature_stride\n"
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_append_columns_kernel_forward_args(
    ImplicitSindyString* source
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        source,
        "        primitive_features,\n"
        "        row_count,\n"
        "        primitive_feature_stride,\n"
        "        num_primitive_features,\n"
        "        leaf_masks,\n"
        "        leaf_words,\n"
        "        leaf_words_feature_stride,\n"
        "        output,\n"
        "        output_setting_stride,\n"
        "        output_feature_stride\n"
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_generate_columns_template_source(
    size_t kernel_count,
    char** source_out
) {
    ImplicitSindyString source;
    size_t kernel_idx;

    if (source_out == NULL || kernel_count == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *source_out = NULL;
    memset(&source, 0, sizeof(source));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
        &source,
        "#include \"implicit_feature_columns.cuh\"\n\n"
    ));

    for (kernel_idx = 0u; kernel_idx < kernel_count; ++kernel_idx) {
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
            &source,
            "extern \"C\" __device__ __noinline__\n"
            "void\n"
            "implicit_feature_eval_%zu(\n"
            "    float* implicit_panel,\n"
            "    const float* const* leaf_ptrs,\n"
            "    const int* leaf_strides,\n"
            "    int active_rows\n"
            ");\n\n"
            "namespace implicit_sindy_feature_gram_kernel {\n"
            "template <>\n"
            "__device__ __forceinline__\n"
            "void\n"
            "implicit_feature_eval_selector<%zu>(\n"
            "    float* implicit_panel,\n"
            "    const float* const* leaf_ptrs,\n"
            "    const int* leaf_strides,\n"
            "    int active_rows\n"
            ") {\n"
            "    ::implicit_feature_eval_%zu(\n"
            "        implicit_panel,\n"
            "        leaf_ptrs,\n"
            "        leaf_strides,\n"
            "        active_rows\n"
            "    );\n"
            "}\n"
            "}  // namespace implicit_sindy_feature_gram_kernel\n\n",
            kernel_idx,
            kernel_idx,
            kernel_idx
        ));
    }

    for (kernel_idx = 0u; kernel_idx < kernel_count; ++kernel_idx) {
        if (kernel_count == 1u) {
            IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
                &source,
                "extern \"C\" __global__\n"
                "__launch_bounds__(implicit_sindy_feature_gram_kernel::kThreads)\n"
                "void\n"
                "implicit_feature_columns_kernel(\n"
            ));
        } else {
            IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
                &source,
                "extern \"C\" __global__\n"
                "__launch_bounds__(implicit_sindy_feature_gram_kernel::kThreads)\n"
                "void\n"
                "implicit_feature_columns_kernel_%zu(\n",
                kernel_idx
            ));
        }
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_append_columns_kernel_args(&source));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_appendf(
            &source,
            ") {\n"
            "    implicit_sindy_feature_gram_kernel::implicit_feature_columns_kernel_device<%zu>(\n",
            kernel_idx
        ));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_append_columns_kernel_forward_args(&source));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_string_append(
            &source,
            "    );\n"
            "}\n\n"
        ));
    }

    *source_out = source.data;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_parse_feature_site_name(
    const char* name,
    size_t* site_idx_out
) {
    const char prefix[] = "implicit_feature_";
    const char* p;
    size_t value = 0u;
    if (name == NULL || site_idx_out == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (strncmp(name, prefix, sizeof(prefix) - 1u) != 0) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
    }
    p = name + sizeof(prefix) - 1u;
    if (*p == '\0') IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
    while (*p != '\0') {
        if (*p < '0' || *p > '9') IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
        if (value > (SIZE_MAX - (size_t)(*p - '0')) / 10u) {
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
        }
        value = value * 10u + (size_t)(*p - '0');
        ++p;
    }
    *site_idx_out = value;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_setup_inject_order(
    ImplicitSindyAstCompiler* compiler
) {
    size_t i;
    if (compiler == NULL || compiler->inject == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_ptx_inject(ptx_inject_num_injects(
        compiler->inject,
        &compiler->num_injects
    )));
    if (compiler->num_injects != compiler->features_per_cubin) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
    }
    compiler->inject_to_feature_idx = (size_t*)calloc(compiler->num_injects, sizeof(*compiler->inject_to_feature_idx));
    if (compiler->inject_to_feature_idx == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    for (i = 0u; i < compiler->num_injects; ++i) {
        const char* name = NULL;
        size_t feature_idx = 0u;
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_ptx_inject(ptx_inject_inject_info_by_index(
            compiler->inject,
            i,
            &name,
            NULL,
            NULL
        )));
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_parse_feature_site_name(name, &feature_idx));
        if (feature_idx >= compiler->features_per_cubin) {
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_PTX_INJECT);
        }
        compiler->inject_to_feature_idx[i] = feature_idx;
    }
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_emit_stack_ptx_stub(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* ast,
    ImplicitSindyArena* arena,
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
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *stub_out = NULL;
    binary_result = binary_ast_stack_ptx_instruction_count(ast, &instruction_count);
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_binary_ast(binary_result));
    if (instruction_count == 0u || instruction_count > compiler->program_stride) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_STACK_PTX);
    }
    binary_result = binary_ast_write_stack_ptx(
        ast,
        program,
        compiler->program_stride,
        &written_instructions
    );
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_binary_ast(binary_result));
    if (written_instructions == 0u) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_STACK_PTX);

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_remaining(arena, 1u, (void**)&stub, &stub_capacity));
    stack_result = stack_ptx_compile(
        &compiler->compiler_info,
        &stack_ptx_stack_info,
        program,
        implicit_sindy_ast_registers,
        sizeof(implicit_sindy_ast_registers) / sizeof(implicit_sindy_ast_registers[0]),
        binary_ast_stack_ptx_routines,
        binary_ast_stack_ptx_num_routines,
        implicit_sindy_ast_requests,
        sizeof(implicit_sindy_ast_requests) / sizeof(implicit_sindy_ast_requests[0]),
        compiler->program_stride,
        stack_workspace,
        compiler->stack_workspace_size,
        stub,
        stub_capacity,
        &written_ptx
    );
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_stack_ptx(stack_result));
    if (written_ptx + 1u > stub_capacity) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    stub[written_ptx] = '\0';
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_commit(arena, written_ptx + 1u));
    *stub_out = stub;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_build_eval_ptx_for_batch(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t batch_idx,
    ImplicitSindyArena* arena,
    char** ptx_out,
    size_t* ptx_bytes_out
) {
    const char** stubs_by_feature = NULL;
    const char** stubs_by_inject = NULL;
    void* stack_workspace = NULL;
    StackPtxInstruction* program = NULL;
    char* rendered_ptx = NULL;
    size_t rendered_capacity = 0u;
    size_t rendered_bytes = 0u;
    BinaryAST padding_ast;
    size_t feature_idx;
    size_t inject_idx;

    if (compiler == NULL || asts == NULL || arena == NULL || ptx_out == NULL || ptx_bytes_out == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *ptx_out = NULL;
    *ptx_bytes_out = 0u;
    arena->offset = 0u;
    implicit_sindy_ast_make_padding_ast(&padding_ast);

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_alloc(
        arena,
        compiler->features_per_cubin * sizeof(*stubs_by_feature),
        16u,
        (void**)&stubs_by_feature
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_alloc(
        arena,
        compiler->num_injects * sizeof(*stubs_by_inject),
        16u,
        (void**)&stubs_by_inject
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_alloc(
        arena,
        compiler->stack_workspace_size,
        64u,
        &stack_workspace
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_alloc(
        arena,
        compiler->program_stride * sizeof(*program),
        64u,
        (void**)&program
    ));

    for (feature_idx = 0u; feature_idx < compiler->features_per_cubin; ++feature_idx) {
        size_t ast_idx = batch_idx * compiler->features_per_cubin + feature_idx;
        const BinaryAST* ast;
        ast = ast_idx < num_asts
            ? implicit_sindy_ast_at(asts, ast_stride_bytes, ast_idx)
            : &padding_ast;
        IMPLICIT_SINDY_CHECK_RET(implicit_sindy_emit_stack_ptx_stub(
            compiler,
            ast,
            arena,
            stack_workspace,
            program,
            (char**)&stubs_by_feature[feature_idx]
        ));
    }

    for (inject_idx = 0u; inject_idx < compiler->num_injects; ++inject_idx) {
        const size_t feature = compiler->inject_to_feature_idx[inject_idx];
        if (feature >= compiler->features_per_cubin) {
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
        }
        stubs_by_inject[inject_idx] = stubs_by_feature[feature];
    }

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_remaining(
        arena,
        1u,
        (void**)&rendered_ptx,
        &rendered_capacity
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_ptx_inject(ptx_inject_render_ptx(
        compiler->inject,
        stubs_by_inject,
        compiler->num_injects,
        rendered_ptx,
        rendered_capacity,
        &rendered_bytes
    )));
    if (rendered_bytes + 1u > rendered_capacity) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    rendered_ptx[rendered_bytes] = '\0';
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_arena_commit(arena, rendered_bytes + 1u));
    *ptx_out = rendered_ptx;
    *ptx_bytes_out = rendered_bytes;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_patch_object_to_malloced_cubin(
    const ImplicitSindyAstCompiler* compiler,
    const void* object,
    size_t object_bytes,
    void** cubin_out,
    size_t* cubin_bytes_out
) {
    CubinFunctionPatchResult patch_result;
    void* cubin = NULL;
    size_t cubin_capacity = 0u;
    size_t cubin_bytes = 0u;

    if (compiler == NULL || compiler->patch_handle == NULL || object == NULL || object_bytes == 0u ||
        cubin_out == NULL || cubin_bytes_out == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *cubin_out = NULL;
    *cubin_bytes_out = 0u;
    patch_result = cubin_function_patch_output_size(compiler->patch_handle, &cubin_capacity);
    if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS || cubin_capacity == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
    }
    cubin = malloc(cubin_capacity);
    if (cubin == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    patch_result = cubin_function_patch_begin(
        compiler->patch_handle,
        cubin,
        cubin_capacity,
        &cubin_bytes
    );
    if (patch_result == CUBIN_FUNCTION_PATCH_SUCCESS) {
        patch_result = cubin_function_patch_apply_all_in_place(
            compiler->patch_handle,
            cubin,
            cubin_bytes,
            object,
            object_bytes,
            NULL,
            0u
        );
    }
    if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) {
        free(cubin);
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
    }
    *cubin_out = cubin;
    *cubin_bytes_out = cubin_bytes;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compile_batch(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t batch_idx,
    ImplicitSindyArena* arena,
    void** out_cubins,
    size_t* out_cubin_sizes
) {
    char* eval_ptx = NULL;
    size_t eval_ptx_bytes = 0u;
    void* object = NULL;
    size_t object_bytes = 0u;

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_build_eval_ptx_for_batch(
        compiler,
        asts,
        num_asts,
        ast_stride_bytes,
        batch_idx,
        arena,
        &eval_ptx,
        &eval_ptx_bytes
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_compile_ptx_to_rdc(
        eval_ptx,
        eval_ptx_bytes,
        compiler->sm_major,
        compiler->sm_minor,
        compiler->nvptx_options,
        compiler->num_nvptx_options,
        arena,
        &object,
        &object_bytes
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_patch_object_to_malloced_cubin(
        compiler,
        object,
        object_bytes,
        &out_cubins[batch_idx],
        &out_cubin_sizes[batch_idx]
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static void*
implicit_sindy_ast_worker_main(
    void* userdata
) {
    ImplicitSindyAstWorker* worker = (ImplicitSindyAstWorker*)userdata;
    size_t batch_idx;

    if (worker == NULL) return NULL;
    worker->result = IMPLICIT_SINDY_SUCCESS;
    for (batch_idx = worker->batch_begin; batch_idx < worker->batch_end; ++batch_idx) {
        worker->result = implicit_sindy_ast_compile_batch(
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

const char*
implicit_sindy_result_to_string(
    ImplicitSindyResult result
) {
    switch (result) {
        case IMPLICIT_SINDY_SUCCESS:
            return "IMPLICIT_SINDY_SUCCESS";
        case IMPLICIT_SINDY_ERROR_INVALID_VALUE:
            return "IMPLICIT_SINDY_ERROR_INVALID_VALUE";
        case IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY:
            return "IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY";
        case IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED:
            return "IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED";
        case IMPLICIT_SINDY_ERROR_PTX_INJECT:
            return "IMPLICIT_SINDY_ERROR_PTX_INJECT";
        case IMPLICIT_SINDY_ERROR_STACK_PTX:
            return "IMPLICIT_SINDY_ERROR_STACK_PTX";
        case IMPLICIT_SINDY_ERROR_NVPTX:
            return "IMPLICIT_SINDY_ERROR_NVPTX";
        case IMPLICIT_SINDY_ERROR_THREAD:
            return "IMPLICIT_SINDY_ERROR_THREAD";
        case IMPLICIT_SINDY_ERROR_CUDA:
            return "IMPLICIT_SINDY_ERROR_CUDA";
        case IMPLICIT_SINDY_ERROR_INTERNAL:
            return "IMPLICIT_SINDY_ERROR_INTERNAL";
    }
    return "IMPLICIT_SINDY_ERROR_UNKNOWN";
}

static size_t
implicit_sindy_ast_features_per_cubin_internal(
    size_t kernels_per_module
) {
    size_t kernel_count = implicit_sindy_kernels_per_module_count(kernels_per_module);
    if (kernel_count == 0u) return 0u;
    if (kernel_count > SIZE_MAX / IMPLICIT_SINDY_AST_FEATURES_PER_KERNEL) return 0u;
    return kernel_count * IMPLICIT_SINDY_AST_FEATURES_PER_KERNEL;
}

size_t
implicit_sindy_ast_cubin_count(
    size_t num_asts,
    size_t kernels_per_module
) {
    size_t features_per_cubin = implicit_sindy_ast_features_per_cubin_internal(kernels_per_module);
    if (num_asts == 0u || features_per_cubin == 0u) return 0u;
    if (num_asts > SIZE_MAX - (features_per_cubin - 1u)) return 0u;
    return (num_asts + features_per_cubin - 1u) / features_per_cubin;
}

static ImplicitSindyResult
implicit_sindy_create_patch_symbols(
    size_t kernel_count,
    char*** symbols_out
) {
    char** symbols;
    size_t i;
    if (symbols_out == NULL || kernel_count == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *symbols_out = NULL;
    symbols = (char**)calloc(kernel_count, sizeof(*symbols));
    if (symbols == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    for (i = 0u; i < kernel_count; ++i) {
        char buffer[64];
        const size_t symbol_idx = i;
        snprintf(buffer, sizeof(buffer), "implicit_feature_eval_%zu", symbol_idx);
        symbols[i] = (char*)malloc(strlen(buffer) + 1u);
        if (symbols[i] == NULL) {
            size_t j;
            for (j = 0u; j < i; ++j) free(symbols[j]);
            free(symbols);
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
        }
        strcpy(symbols[i], buffer);
    }
    *symbols_out = symbols;
    return IMPLICIT_SINDY_SUCCESS;
}

static void
implicit_sindy_free_patch_symbols(
    char** symbols,
    size_t count
) {
    size_t i;
    if (symbols == NULL) return;
    for (i = 0u; i < count; ++i) {
        free(symbols[i]);
    }
    free(symbols);
}

static ImplicitSindyResult
implicit_sindy_prepare_eval_template(
    ImplicitSindyAstCompiler* compiler
) {
    char* eval_source = NULL;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;

    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);

    error_ret = implicit_sindy_generate_eval_template_source(compiler->kernel_count, &eval_source);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_cuda_to_ptx(
            eval_source,
            "implicit_feature_eval_template.cu",
            compiler->sm_major,
            compiler->sm_minor,
            &compiler->eval_template_ptx,
            &compiler->eval_template_ptx_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_result_from_ptx_inject(ptx_inject_create(
            &compiler->inject,
            compiler->eval_template_ptx
        ));
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_setup_inject_order(compiler);
    }

    free(eval_source);
    IMPLICIT_SINDY_CHECK_RET(error_ret);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_setup_patch_handle(
    ImplicitSindyAstCompiler* compiler
) {
    char** patch_symbols = NULL;
    void* handle_memory = NULL;
    size_t handle_memory_bytes = 0u;
    CubinFunctionPatchResult patch_result;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;

    if (compiler == NULL || compiler->reserved_cubin == NULL || compiler->reserved_cubin_bytes == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }

    error_ret = implicit_sindy_create_patch_symbols(compiler->kernel_count, &patch_symbols);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        patch_result = cubin_function_patch_handle_size(
            (const char* const*)patch_symbols,
            compiler->kernel_count,
            &handle_memory_bytes
        );
        if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) error_ret = IMPLICIT_SINDY_ERROR_INTERNAL;
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        handle_memory = malloc(handle_memory_bytes);
        if (handle_memory == NULL) error_ret = IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY;
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        patch_result = cubin_function_patch_create(
            compiler->reserved_cubin,
            compiler->reserved_cubin_bytes,
            (const char* const*)patch_symbols,
            compiler->kernel_count,
            handle_memory,
            handle_memory_bytes,
            &compiler->patch_handle
        );
        if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) {
            error_ret = IMPLICIT_SINDY_ERROR_INTERNAL;
        } else {
            compiler->patch_handle_memory = handle_memory;
            handle_memory = NULL;
        }
    }

    free(handle_memory);
    implicit_sindy_free_patch_symbols(patch_symbols, compiler->kernel_count);
    IMPLICIT_SINDY_CHECK_RET(error_ret);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_make_reserved_cubin(
    ImplicitSindyAstCompiler* compiler
) {
    char* gram_source = NULL;
    char* gram_ptx = NULL;
    size_t gram_ptx_bytes = 0u;
    ImplicitSindyArena arena;
    void* eval_object = NULL;
    size_t eval_object_bytes = 0u;
    unsigned char* temp_memory = NULL;
    size_t temp_memory_bytes = 128u * 1024u * 1024u;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;
    BinaryAST max_ast;
    char* max_eval_ptx = NULL;
    size_t max_eval_ptx_bytes = 0u;

    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    temp_memory = (unsigned char*)malloc(temp_memory_bytes);
    if (temp_memory == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    arena.base = temp_memory;
    arena.capacity = temp_memory_bytes;
    arena.offset = 0u;

    error_ret = implicit_sindy_prepare_eval_template(compiler);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        BinaryAST* max_asts = NULL;
        size_t i;
        max_asts = (BinaryAST*)calloc(compiler->features_per_cubin, sizeof(*max_asts));
        if (max_asts == NULL) {
            error_ret = IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY;
        } else {
            for (i = 0u; i < compiler->features_per_cubin; ++i) {
                implicit_sindy_ast_make_max_ast(&max_ast);
                max_asts[i] = max_ast;
            }
            error_ret = implicit_sindy_build_eval_ptx_for_batch(
                compiler,
                max_asts,
                compiler->features_per_cubin,
                sizeof(*max_asts),
                0u,
                &arena,
                &max_eval_ptx,
                &max_eval_ptx_bytes
            );
            free(max_asts);
        }
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_ptx_to_rdc(
            max_eval_ptx,
            max_eval_ptx_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            compiler->nvptx_options,
            compiler->num_nvptx_options,
            &arena,
            &eval_object,
            &eval_object_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_generate_gram_template_source(compiler->kernel_count, &gram_source);
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_cuda_to_ptx(
            gram_source,
            "implicit_feature_gram_patch_template.cu",
            compiler->sm_major,
            compiler->sm_minor,
            &gram_ptx,
            &gram_ptx_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_link_reserved_cubin(
            gram_ptx,
            gram_ptx_bytes,
            eval_object,
            eval_object_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            &compiler->reserved_cubin,
            &compiler->reserved_cubin_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_setup_patch_handle(compiler);
    }

    free(gram_ptx);
    free(gram_source);
    free(temp_memory);
    IMPLICIT_SINDY_CHECK_RET(error_ret);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_make_reserved_columns_cubin(
    ImplicitSindyAstCompiler* compiler
) {
    char* columns_source = NULL;
    char* columns_ptx = NULL;
    size_t columns_ptx_bytes = 0u;
    ImplicitSindyArena arena;
    void* eval_object = NULL;
    size_t eval_object_bytes = 0u;
    unsigned char* temp_memory = NULL;
    size_t temp_memory_bytes = 128u * 1024u * 1024u;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;
    BinaryAST max_ast;
    char* max_eval_ptx = NULL;
    size_t max_eval_ptx_bytes = 0u;

    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    temp_memory = (unsigned char*)malloc(temp_memory_bytes);
    if (temp_memory == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    arena.base = temp_memory;
    arena.capacity = temp_memory_bytes;
    arena.offset = 0u;

    error_ret = implicit_sindy_prepare_eval_template(compiler);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        BinaryAST* max_asts = NULL;
        size_t i;
        max_asts = (BinaryAST*)calloc(compiler->features_per_cubin, sizeof(*max_asts));
        if (max_asts == NULL) {
            error_ret = IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY;
        } else {
            for (i = 0u; i < compiler->features_per_cubin; ++i) {
                implicit_sindy_ast_make_max_ast(&max_ast);
                max_asts[i] = max_ast;
            }
            error_ret = implicit_sindy_build_eval_ptx_for_batch(
                compiler,
                max_asts,
                compiler->features_per_cubin,
                sizeof(*max_asts),
                0u,
                &arena,
                &max_eval_ptx,
                &max_eval_ptx_bytes
            );
            free(max_asts);
        }
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_ptx_to_rdc(
            max_eval_ptx,
            max_eval_ptx_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            compiler->nvptx_options,
            compiler->num_nvptx_options,
            &arena,
            &eval_object,
            &eval_object_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_generate_columns_template_source(
            compiler->kernel_count,
            &columns_source
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_cuda_to_ptx(
            columns_source,
            "implicit_feature_columns_patch_template.cu",
            compiler->sm_major,
            compiler->sm_minor,
            &columns_ptx,
            &columns_ptx_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_link_reserved_cubin(
            columns_ptx,
            columns_ptx_bytes,
            eval_object,
            eval_object_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            &compiler->reserved_cubin,
            &compiler->reserved_cubin_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_setup_patch_handle(compiler);
    }

    free(columns_ptx);
    free(columns_source);
    free(temp_memory);
    IMPLICIT_SINDY_CHECK_RET(error_ret);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_make_reserved_cubin_from_gram_cubin(
    ImplicitSindyAstCompiler* compiler,
    const void* gram_template_cubin,
    size_t gram_template_cubin_bytes
) {
    ImplicitSindyArena arena;
    void* eval_object = NULL;
    size_t eval_object_bytes = 0u;
    unsigned char* temp_memory = NULL;
    size_t temp_memory_bytes = 128u * 1024u * 1024u;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;
    BinaryAST max_ast;
    char* max_eval_ptx = NULL;
    size_t max_eval_ptx_bytes = 0u;

    if (compiler == NULL || gram_template_cubin == NULL || gram_template_cubin_bytes == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    temp_memory = (unsigned char*)malloc(temp_memory_bytes);
    if (temp_memory == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    arena.base = temp_memory;
    arena.capacity = temp_memory_bytes;
    arena.offset = 0u;

    error_ret = implicit_sindy_prepare_eval_template(compiler);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        BinaryAST* max_asts = NULL;
        size_t i;
        max_asts = (BinaryAST*)calloc(compiler->features_per_cubin, sizeof(*max_asts));
        if (max_asts == NULL) {
            error_ret = IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY;
        } else {
            for (i = 0u; i < compiler->features_per_cubin; ++i) {
                implicit_sindy_ast_make_max_ast(&max_ast);
                max_asts[i] = max_ast;
            }
            error_ret = implicit_sindy_build_eval_ptx_for_batch(
                compiler,
                max_asts,
                compiler->features_per_cubin,
                sizeof(*max_asts),
                0u,
                &arena,
                &max_eval_ptx,
                &max_eval_ptx_bytes
            );
            free(max_asts);
        }
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_ptx_to_rdc(
            max_eval_ptx,
            max_eval_ptx_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            compiler->nvptx_options,
            compiler->num_nvptx_options,
            &arena,
            &eval_object,
            &eval_object_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_link_reserved_cubin_from_gram_cubin(
            gram_template_cubin,
            gram_template_cubin_bytes,
            eval_object,
            eval_object_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            &compiler->reserved_cubin,
            &compiler->reserved_cubin_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_setup_patch_handle(compiler);
    }

    free(temp_memory);
    IMPLICIT_SINDY_CHECK_RET(error_ret);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_make_reserved_cubin_from_gram_ptx(
    ImplicitSindyAstCompiler* compiler,
    const void* gram_template_ptx,
    size_t gram_template_ptx_bytes
) {
    ImplicitSindyArena arena;
    void* eval_object = NULL;
    size_t eval_object_bytes = 0u;
    unsigned char* temp_memory = NULL;
    size_t temp_memory_bytes = 128u * 1024u * 1024u;
    ImplicitSindyResult error_ret = IMPLICIT_SINDY_SUCCESS;
    BinaryAST max_ast;
    char* max_eval_ptx = NULL;
    size_t max_eval_ptx_bytes = 0u;

    if (compiler == NULL || gram_template_ptx == NULL || gram_template_ptx_bytes == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    temp_memory = (unsigned char*)malloc(temp_memory_bytes);
    if (temp_memory == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);
    arena.base = temp_memory;
    arena.capacity = temp_memory_bytes;
    arena.offset = 0u;

    error_ret = implicit_sindy_prepare_eval_template(compiler);
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        BinaryAST* max_asts = NULL;
        size_t i;
        max_asts = (BinaryAST*)calloc(compiler->features_per_cubin, sizeof(*max_asts));
        if (max_asts == NULL) {
            error_ret = IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY;
        } else {
            for (i = 0u; i < compiler->features_per_cubin; ++i) {
                implicit_sindy_ast_make_max_ast(&max_ast);
                max_asts[i] = max_ast;
            }
            error_ret = implicit_sindy_build_eval_ptx_for_batch(
                compiler,
                max_asts,
                compiler->features_per_cubin,
                sizeof(*max_asts),
                0u,
                &arena,
                &max_eval_ptx,
                &max_eval_ptx_bytes
            );
            free(max_asts);
        }
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_compile_ptx_to_rdc(
            max_eval_ptx,
            max_eval_ptx_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            compiler->nvptx_options,
            compiler->num_nvptx_options,
            &arena,
            &eval_object,
            &eval_object_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_link_reserved_cubin(
            (const char*)gram_template_ptx,
            gram_template_ptx_bytes,
            eval_object,
            eval_object_bytes,
            compiler->sm_major,
            compiler->sm_minor,
            &compiler->reserved_cubin,
            &compiler->reserved_cubin_bytes
        );
    }
    if (error_ret == IMPLICIT_SINDY_SUCCESS) {
        error_ret = implicit_sindy_setup_patch_handle(compiler);
    }

    free(temp_memory);
    IMPLICIT_SINDY_CHECK_RET(error_ret);
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compiler_init_common(
    ImplicitSindyAstCompiler* compiler,
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options
) {
    StackPtxResult stack_result;

    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    compiler->kernel_count = implicit_sindy_kernels_per_module_count(kernels_per_module);
    if (compiler->kernel_count == 0u) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    compiler->kernels_per_module = kernels_per_module;
    compiler->features_per_cubin = implicit_sindy_ast_features_per_cubin_internal(kernels_per_module);
    compiler->program_stride = IMPLICIT_SINDY_AST_PROGRAM_STRIDE;
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
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_result_from_stack_ptx(stack_result));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compiler_create_impl(
    ImplicitSindyAstCompiler* compiler,
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_ast_compiler_init_common(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        nvptx_options,
        num_nvptx_options
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_make_reserved_cubin(compiler));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_cubin_impl(
    ImplicitSindyAstCompiler* compiler,
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_cubin,
    size_t gram_template_cubin_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_ast_compiler_init_common(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        nvptx_options,
        num_nvptx_options
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_make_reserved_cubin_from_gram_cubin(
        compiler,
        gram_template_cubin,
        gram_template_cubin_bytes
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_ptx_impl(
    ImplicitSindyAstCompiler* compiler,
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_ptx,
    size_t gram_template_ptx_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_ast_compiler_init_common(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        nvptx_options,
        num_nvptx_options
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_make_reserved_cubin_from_gram_ptx(
        compiler,
        gram_template_ptx,
        gram_template_ptx_bytes
    ));
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compiler_create_columns_impl(
    ImplicitSindyAstCompiler* compiler,
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options
) {
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_ast_compiler_init_common(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        nvptx_options,
        num_nvptx_options
    ));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_make_reserved_columns_cubin(compiler));
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compiler_create(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
) {
    ImplicitSindyAstCompiler* compiler;
    ImplicitSindyResult result;

    if (out_compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out_compiler = NULL;
    compiler = (ImplicitSindyAstCompiler*)calloc(1u, sizeof(*compiler));
    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    result = implicit_sindy_ast_compiler_create_impl(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        nvptx_options,
        num_nvptx_options
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
        if (compiler->patch_handle != NULL) cubin_function_patch_destroy(compiler->patch_handle);
        free(compiler->patch_handle_memory);
        free(compiler->reserved_cubin);
        free(compiler->eval_template_ptx);
        free(compiler->inject_to_feature_idx);
        free(compiler);
        IMPLICIT_SINDY_ERROR_RET(result);
    }

    *out_compiler = compiler;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compiler_create_columns(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
) {
    ImplicitSindyAstCompiler* compiler;
    ImplicitSindyResult result;

    if (out_compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    *out_compiler = NULL;
    compiler = (ImplicitSindyAstCompiler*)calloc(1u, sizeof(*compiler));
    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    result = implicit_sindy_ast_compiler_create_columns_impl(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        nvptx_options,
        num_nvptx_options
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
        if (compiler->patch_handle != NULL) cubin_function_patch_destroy(compiler->patch_handle);
        free(compiler->patch_handle_memory);
        free(compiler->reserved_cubin);
        free(compiler->eval_template_ptx);
        free(compiler->inject_to_feature_idx);
        free(compiler);
        IMPLICIT_SINDY_ERROR_RET(result);
    }

    *out_compiler = compiler;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_cubin(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_cubin,
    size_t gram_template_cubin_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
) {
    ImplicitSindyAstCompiler* compiler;
    ImplicitSindyResult result;

    if (out_compiler == NULL || gram_template_cubin == NULL || gram_template_cubin_bytes == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_compiler = NULL;
    compiler = (ImplicitSindyAstCompiler*)calloc(1u, sizeof(*compiler));
    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    result = implicit_sindy_ast_compiler_create_with_gram_cubin_impl(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        gram_template_cubin,
        gram_template_cubin_bytes,
        nvptx_options,
        num_nvptx_options
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
        if (compiler->patch_handle != NULL) cubin_function_patch_destroy(compiler->patch_handle);
        free(compiler->patch_handle_memory);
        free(compiler->reserved_cubin);
        free(compiler->eval_template_ptx);
        free(compiler->inject_to_feature_idx);
        free(compiler);
        IMPLICIT_SINDY_ERROR_RET(result);
    }

    *out_compiler = compiler;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compiler_create_with_gram_ptx(
    size_t kernels_per_module,
    unsigned int sm_major,
    unsigned int sm_minor,
    const void* gram_template_ptx,
    size_t gram_template_ptx_bytes,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    ImplicitSindyAstCompiler** out_compiler
) {
    ImplicitSindyAstCompiler* compiler;
    ImplicitSindyResult result;

    if (out_compiler == NULL || gram_template_ptx == NULL || gram_template_ptx_bytes == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_compiler = NULL;
    compiler = (ImplicitSindyAstCompiler*)calloc(1u, sizeof(*compiler));
    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_OUT_OF_MEMORY);

    result = implicit_sindy_ast_compiler_create_with_gram_ptx_impl(
        compiler,
        kernels_per_module,
        sm_major,
        sm_minor,
        gram_template_ptx,
        gram_template_ptx_bytes,
        nvptx_options,
        num_nvptx_options
    );
    if (result != IMPLICIT_SINDY_SUCCESS) {
        if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
        if (compiler->patch_handle != NULL) cubin_function_patch_destroy(compiler->patch_handle);
        free(compiler->patch_handle_memory);
        free(compiler->reserved_cubin);
        free(compiler->eval_template_ptx);
        free(compiler->inject_to_feature_idx);
        free(compiler);
        IMPLICIT_SINDY_ERROR_RET(result);
    }

    *out_compiler = compiler;
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compiler_destroy(
    ImplicitSindyAstCompiler* compiler
) {
    if (compiler == NULL) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    if (compiler->inject != NULL) (void)ptx_inject_destroy(compiler->inject);
    if (compiler->patch_handle != NULL) cubin_function_patch_destroy(compiler->patch_handle);
    free(compiler->patch_handle_memory);
    free(compiler->reserved_cubin);
    free(compiler->eval_template_ptx);
    free(compiler->inject_to_feature_idx);
    free(compiler);
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compile_workspace_size(
    const ImplicitSindyAstCompiler* compiler,
    size_t num_asts,
    size_t worker_count,
    size_t scratch_bytes_per_worker,
    size_t* out_bytes
) {
    size_t num_cubins;
    size_t workers;
    size_t worker_bytes;
    size_t thread_bytes;
    size_t scratch_total;
    size_t total = 128u;

    if (compiler == NULL || out_bytes == NULL || num_asts == 0u || scratch_bytes_per_worker == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_bytes = 0u;
    num_cubins = implicit_sindy_ast_cubin_count(num_asts, compiler->kernels_per_module);
    workers = implicit_sindy_default_worker_count(worker_count, num_cubins);
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_mul_size(workers, sizeof(ImplicitSindyAstWorker), &worker_bytes));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_mul_size(workers, sizeof(pthread_t), &thread_bytes));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_mul_size(workers, scratch_bytes_per_worker, &scratch_total));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_add_size(total, implicit_sindy_align_up_size(worker_bytes, 64u), &total));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_add_size(total, implicit_sindy_align_up_size(thread_bytes, 64u), &total));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_add_size(total, scratch_total, &total));
    *out_bytes = total;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_prepare_workers(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    void** out_cubins,
    size_t* out_cubin_sizes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    pthread_t** out_threads,
    ImplicitSindyAstWorker** out_workers
) {
    uintptr_t p;
    uintptr_t end;
    size_t num_cubins;
    size_t worker_bytes;
    size_t thread_bytes;
    size_t arena_per_worker;
    size_t i;
    ImplicitSindyAstWorker* workers = NULL;
    pthread_t* threads = NULL;

    if (compiler == NULL || asts == NULL || worker_memory == NULL ||
        out_threads == NULL || out_workers == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    *out_threads = NULL;
    *out_workers = NULL;
    num_cubins = implicit_sindy_ast_cubin_count(num_asts, compiler->kernels_per_module);
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_mul_size(worker_count, sizeof(*workers), &worker_bytes));
    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_mul_size(worker_count, sizeof(*threads), &thread_bytes));
    p = implicit_sindy_align_up_uintptr((uintptr_t)worker_memory, 64u);
    end = (uintptr_t)worker_memory + worker_memory_size;
    if (end < (uintptr_t)worker_memory || p > end) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    if (worker_bytes > (size_t)(end - p)) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    workers = (ImplicitSindyAstWorker*)p;
    memset(workers, 0, worker_bytes);
    p = implicit_sindy_align_up_uintptr(p + worker_bytes, 64u);
    if (p > end || thread_bytes > (size_t)(end - p)) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    }
    threads = (pthread_t*)p;
    memset(threads, 0, thread_bytes);
    p = implicit_sindy_align_up_uintptr(p + thread_bytes, 64u);
    if (p > end) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);
    arena_per_worker = (size_t)(end - p) / worker_count;
    if (arena_per_worker == 0u) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_WORKSPACE_EXHAUSTED);

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
        workers[i].batch_begin = num_cubins * i / worker_count;
        workers[i].batch_end = num_cubins * (i + 1u) / worker_count;
        workers[i].result = IMPLICIT_SINDY_SUCCESS;
    }

    *out_threads = threads;
    *out_workers = workers;
    return IMPLICIT_SINDY_SUCCESS;
}

static ImplicitSindyResult
implicit_sindy_ast_compile_cubins_impl(
    const ImplicitSindyAstCompiler* compiler,
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
    ImplicitSindyAstWorker* workers = NULL;
    size_t started_threads = 0u;
    size_t i;

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_ast_prepare_workers(
        compiler,
        asts,
        num_asts,
        ast_stride_bytes,
        out_cubins,
        out_cubin_sizes,
        worker_count,
        worker_memory,
        worker_memory_size,
        &threads,
        &workers
    ));

    for (i = 0u; i < worker_count; ++i) {
        if (pthread_create(&threads[i], NULL, implicit_sindy_ast_worker_main, &workers[i]) != 0) {
            size_t join_idx;
            for (join_idx = 0u; join_idx < started_threads; ++join_idx) {
                pthread_join(threads[join_idx], NULL);
            }
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_THREAD);
        }
        ++started_threads;
    }
    for (i = 0u; i < started_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    for (i = 0u; i < worker_count; ++i) {
        IMPLICIT_SINDY_CHECK_RET(workers[i].result);
    }
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_ast_compile_cubins(
    const ImplicitSindyAstCompiler* compiler,
    const BinaryAST* asts,
    size_t num_asts,
    size_t ast_stride_bytes,
    size_t worker_count,
    void* worker_memory,
    size_t worker_memory_size,
    void** out_cubins,
    size_t* out_cubin_sizes
) {
    size_t num_cubins;
    size_t workers;
    size_t i;

    if (compiler == NULL || asts == NULL || num_asts == 0u ||
        worker_memory == NULL || worker_memory_size == 0u ||
        out_cubins == NULL || out_cubin_sizes == NULL) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    if (ast_stride_bytes == 0u) ast_stride_bytes = sizeof(*asts);
    if (ast_stride_bytes < sizeof(*asts)) IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    num_cubins = implicit_sindy_ast_cubin_count(num_asts, compiler->kernels_per_module);
    workers = implicit_sindy_default_worker_count(worker_count, num_cubins);
    for (i = 0u; i < num_cubins; ++i) {
        out_cubins[i] = NULL;
        out_cubin_sizes[i] = 0u;
    }

    IMPLICIT_SINDY_CHECK_RET(implicit_sindy_ast_compile_cubins_impl(
        compiler,
        asts,
        num_asts,
        ast_stride_bytes,
        workers,
        worker_memory,
        worker_memory_size,
        out_cubins,
        out_cubin_sizes
    ));
    for (i = 0u; i < num_cubins; ++i) {
        if (out_cubins[i] == NULL || out_cubin_sizes[i] == 0u) {
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INTERNAL);
        }
    }
    return IMPLICIT_SINDY_SUCCESS;
}

void
implicit_sindy_free_cubins(
    void** cubins,
    size_t* cubin_sizes,
    size_t num_cubins
) {
    size_t i;
    if (cubins == NULL) return;
    for (i = 0u; i < num_cubins; ++i) {
        free(cubins[i]);
        cubins[i] = NULL;
        if (cubin_sizes != NULL) cubin_sizes[i] = 0u;
    }
}

ImplicitSindyResult
implicit_sindy_load_cubin_modules(
    void* const* cubins,
    const size_t* cubin_sizes,
    size_t num_cubins,
    void** out_modules
) {
    size_t i;
    if (cubins == NULL || cubin_sizes == NULL || out_modules == NULL || num_cubins == 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    for (i = 0u; i < num_cubins; ++i) {
        out_modules[i] = NULL;
    }
    for (i = 0u; i < num_cubins; ++i) {
        CUmodule module = NULL;
        CUresult result;
        if (cubins[i] == NULL || cubin_sizes[i] == 0u) {
            (void)implicit_sindy_unload_modules(out_modules, i);
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
        }
        result = cuModuleLoadData(&module, cubins[i]);
        if (result != CUDA_SUCCESS) {
            if (getenv("IMPLICIT_SINDY_CUDA_DIAGNOSTICS") != NULL) {
                const char* name = NULL;
                const char* string = NULL;
                (void)cuGetErrorName(result, &name);
                (void)cuGetErrorString(result, &string);
                fprintf(
                    stderr,
                    "implicit_sindy_load_cubin_modules: cuModuleLoadData failed for cubin %zu: %s %s\n",
                    i,
                    name != NULL ? name : "CUDA_ERROR_UNKNOWN",
                    string != NULL ? string : ""
                );
            }
            (void)implicit_sindy_unload_modules(out_modules, i);
            IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_CUDA);
        }
        out_modules[i] = (void*)module;
    }
    return IMPLICIT_SINDY_SUCCESS;
}

ImplicitSindyResult
implicit_sindy_unload_modules(
    void** modules,
    size_t num_modules
) {
    size_t i;
    ImplicitSindyResult result = IMPLICIT_SINDY_SUCCESS;
    if (modules == NULL && num_modules != 0u) {
        IMPLICIT_SINDY_ERROR_RET(IMPLICIT_SINDY_ERROR_INVALID_VALUE);
    }
    for (i = 0u; i < num_modules; ++i) {
        if (modules[i] != NULL) {
            if (cuModuleUnload((CUmodule)modules[i]) != CUDA_SUCCESS && result == IMPLICIT_SINDY_SUCCESS) {
                result = IMPLICIT_SINDY_ERROR_CUDA;
            }
            modules[i] = NULL;
        }
    }
    IMPLICIT_SINDY_CHECK_RET(result);
    return IMPLICIT_SINDY_SUCCESS;
}

#endif
#endif
