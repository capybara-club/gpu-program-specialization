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
#include "stack_ptx_compiler.h"
#include "stack_ptx_compiler_backend_internal.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <nvPTXCompiler.h>
#include <ptx_inject.h>
#include <stack_ptx.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>

#ifndef STACK_PTX_COMPILER_OPENMP_ENABLED
#define STACK_PTX_COMPILER_OPENMP_ENABLED 0
#endif

#if STACK_PTX_COMPILER_OPENMP_ENABLED
#include <omp.h>
#endif

#ifndef STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS
#define STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS 0
#endif

enum {
    REGISTER_X = 0,
    REGISTER_Y = 1
};

static const char* kDefaultInjectPrefix = "func_";
static const char* kDefaultInputRegisterName = "x";
static const char* kDefaultOutputRegisterPrefix = "y";
static const char* kDefaultKernelNameFormat = "kernel_%06zu";

static const StackPtxCompilerInfo kCompilerInfo = {
    512,
    512,
    256,
    4,
    16
};

typedef struct {
    unsigned char* base;
    size_t capacity;
    size_t offset;
} StackPtxCompilerArena;

static size_t stack_ptx_compiler_align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static int stack_ptx_compiler_arena_alloc(
    StackPtxCompilerArena* arena,
    size_t size,
    size_t alignment,
    void** out_ptr
) {
    if (!arena || !out_ptr || !arena->base) {
        return 0;
    }
    size_t aligned_offset = stack_ptx_compiler_align_up(arena->offset, alignment);
    if (aligned_offset > arena->capacity || size > arena->capacity - aligned_offset) {
        return 0;
    }
    *out_ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return 1;
}

static const char* stack_ptx_compiler_nvptx_result_to_string(nvPTXCompileResult result) {
    switch (result) {
        case NVPTXCOMPILE_SUCCESS:
            return "NVPTXCOMPILE_SUCCESS";
        case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
            return "NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE";
        case NVPTXCOMPILE_ERROR_INVALID_INPUT:
            return "NVPTXCOMPILE_ERROR_INVALID_INPUT";
        case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
            return "NVPTXCOMPILE_ERROR_COMPILATION_FAILURE";
        case NVPTXCOMPILE_ERROR_INTERNAL:
            return "NVPTXCOMPILE_ERROR_INTERNAL";
        case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
            return "NVPTXCOMPILE_ERROR_OUT_OF_MEMORY";
        case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
            return "NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE";
        case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
            return "NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION";
        case NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC:
            return "NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC";
        case NVPTXCOMPILE_ERROR_CANCELLED:
            return "NVPTXCOMPILE_ERROR_CANCELLED";
    }
    return "NVPTXCOMPILE_ERROR_UNKNOWN";
}

static void stack_ptx_compiler_log_ptx_inject_error(const char* step, PtxInjectResult result) {
    const char* error_name = ptx_inject_result_to_string(result);
    fprintf(stderr, "ptx_inject %s failed: %s\n", step ? step : "step", error_name);
    if (result == PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED) {
        fprintf(stderr, "ptx_inject unique inject sites exceed configured max\n");
#ifdef PTX_INJECT_MAX_UNIQUE_INJECTS
        fprintf(stderr, "ptx_inject max unique inject sites: %d\n", PTX_INJECT_MAX_UNIQUE_INJECTS);
#endif
    }
}

static void stack_ptx_compiler_nvptx_log(nvPTXCompilerHandle compiler, const char* step, nvPTXCompileResult result) {
    fprintf(stderr, "nvPTXCompiler %s failed: %s\n",
        step ? step : "step", stack_ptx_compiler_nvptx_result_to_string(result));
    if (!compiler) {
        return;
    }
    size_t error_size = 0;
    if (nvPTXCompilerGetErrorLogSize(compiler, &error_size) == NVPTXCOMPILE_SUCCESS && error_size > 1) {
        char* error_log = (char*)calloc(error_size + 1, 1);
        if (error_log) {
            if (nvPTXCompilerGetErrorLog(compiler, error_log) == NVPTXCOMPILE_SUCCESS) {
                fprintf(stderr, "nvPTXCompiler error log:\n%s\n", error_log);
            }
            free(error_log);
        }
    }
    size_t info_size = 0;
    if (nvPTXCompilerGetInfoLogSize(compiler, &info_size) == NVPTXCOMPILE_SUCCESS && info_size > 1) {
        char* info_log = (char*)calloc(info_size + 1, 1);
        if (info_log) {
            if (nvPTXCompilerGetInfoLog(compiler, info_log) == NVPTXCOMPILE_SUCCESS) {
                fprintf(stderr, "nvPTXCompiler info log:\n%s\n", info_log);
            }
            free(info_log);
        }
    }
}

typedef struct {
    const StackPtxInstruction** instruction_stubs;
    const char** ptx_stubs;
} StackPtxCompilerThreadScratch;

typedef struct StackPtxCompilerLocal StackPtxCompilerLocal;

struct StackPtxCompilerLocal {
    PtxInjectHandle* ptx_injects;
    size_t num_ptx_injects;
    StackPtxRegister* registers;
    size_t num_registers;
    size_t* requests;
    size_t num_requests;
    const size_t** request_stubs;
    size_t* request_stub_sizes;
    size_t* reverse_indices;
    size_t num_injects;
    size_t input_dims;
    size_t embed_dims;
    size_t individuals_per_kernel;
    size_t individuals_per_module;
    size_t kernel_num_kernels;
    size_t kernel_groups_per_kernel;
    size_t execution_limit;
    char* kernel_name_format;
    int compute_major;
    int compute_minor;
    StackPtxCompilerThreadScratch* thread_scratch;
    void** workspaces;
    size_t capabilities;
    size_t queue_capacity;
    StackPtxCompilerWork* work_queue;
    size_t work_capacity;
    size_t work_head;
    size_t work_tail;
    size_t work_count;
    StackPtxCompilerOutput* results;
    size_t result_capacity;
    size_t result_head;
    size_t result_tail;
    size_t result_count;
    size_t workspace_bytes;
    pthread_mutex_t work_mutex;
    pthread_mutex_t result_mutex;
    int work_mutex_initialized;
    int result_mutex_initialized;
    pthread_t worker_thread;
    int worker_thread_started;
    atomic_int worker_stop;
    const StackPtxStackInfo* stack_info;
    const StackPtxCompilerInfo* compiler_info;
};

static int stack_ptx_compiler_should_stop(StackPtxCompilerLocal* compiler) {
    if (!compiler) {
        return 1;
    }
    return atomic_load(&compiler->worker_stop) != 0;
}

static void stack_ptx_compiler_sleep_short(void) {
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 1000000;
    nanosleep(&ts, NULL);
}

static StackPtxCompilerResult stack_ptx_compiler_work_push(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work
) {
    if (!compiler || !work) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (compiler->work_capacity == 0 || !compiler->work_queue) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    pthread_mutex_lock(&compiler->work_mutex);
    if (compiler->work_count >= compiler->work_capacity) {
        pthread_mutex_unlock(&compiler->work_mutex);
        return STACK_PTX_COMPILER_ERROR_QUEUE_FULL;
    }
    compiler->work_queue[compiler->work_tail] = *work;
    compiler->work_tail = (compiler->work_tail + 1) % compiler->work_capacity;
    compiler->work_count += 1;
    pthread_mutex_unlock(&compiler->work_mutex);
    return STACK_PTX_COMPILER_SUCCESS;
}

static int stack_ptx_compiler_work_pop(
    StackPtxCompilerLocal* compiler,
    StackPtxCompilerWork* out_work
) {
    int has_work = 0;
    if (!compiler || !out_work) {
        return 0;
    }
    if (!compiler->work_queue || compiler->work_capacity == 0) {
        return 0;
    }

    pthread_mutex_lock(&compiler->work_mutex);
    if (compiler->work_count > 0) {
        *out_work = compiler->work_queue[compiler->work_head];
        compiler->work_head = (compiler->work_head + 1) % compiler->work_capacity;
        compiler->work_count -= 1;
        has_work = 1;
    }
    pthread_mutex_unlock(&compiler->work_mutex);
    return has_work;
}

static void stack_ptx_compiler_result_push(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerOutput* output
) {
    if (!compiler || !output) {
        return;
    }
    if (compiler->result_capacity == 0 || !compiler->results) {
        return;
    }

    for (;;) {
        pthread_mutex_lock(&compiler->result_mutex);
        if (compiler->result_count < compiler->result_capacity) {
            compiler->results[compiler->result_tail] = *output;
            compiler->result_tail = (compiler->result_tail + 1) % compiler->result_capacity;
            compiler->result_count += 1;
            pthread_mutex_unlock(&compiler->result_mutex);
            return;
        }
        pthread_mutex_unlock(&compiler->result_mutex);
        if (stack_ptx_compiler_should_stop(compiler)) {
            if (output->cubin) {
                free(output->cubin);
            }
            return;
        }
        stack_ptx_compiler_sleep_short();
    }
}

static int stack_ptx_compiler_result_pop(
    StackPtxCompilerLocal* compiler,
    StackPtxCompilerOutput* out_result
) {
    int has_result = 0;
    if (!compiler || !out_result) {
        return 0;
    }
    if (!compiler->results || compiler->result_capacity == 0) {
        return 0;
    }
    pthread_mutex_lock(&compiler->result_mutex);
    if (compiler->result_count > 0) {
        *out_result = compiler->results[compiler->result_head];
        compiler->result_head = (compiler->result_head + 1) % compiler->result_capacity;
        compiler->result_count -= 1;
        has_result = 1;
    }
    pthread_mutex_unlock(&compiler->result_mutex);
    return has_result;
}

static void stack_ptx_compiler_clear_results(StackPtxCompilerLocal* compiler) {
    if (!compiler || !compiler->results) {
        return;
    }
    const int use_lock = compiler->result_mutex_initialized;
    if (use_lock) {
        pthread_mutex_lock(&compiler->result_mutex);
    }
    for (size_t i = 0; i < compiler->result_count; ++i) {
        size_t idx = (compiler->result_head + i) % compiler->result_capacity;
        StackPtxCompilerOutput* output = &compiler->results[idx];
        if (output->cubin) {
            free(output->cubin);
            output->cubin = NULL;
            output->cubin_size = 0;
        }
    }
    compiler->result_head = 0;
    compiler->result_tail = 0;
    compiler->result_count = 0;
    if (use_lock) {
        pthread_mutex_unlock(&compiler->result_mutex);
    }
}

static char* stack_ptx_compiler_strdup(const char* s) {
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s) + 1;
    char* out = (char*)malloc(len);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len);
    return out;
}

static int stack_ptx_compiler_str_starts_with(const char* s, const char* prefix) {
    if (!s || !prefix) {
        return 0;
    }
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static int stack_ptx_compiler_parse_size_t(const char* s, size_t* out) {
    if (!s || !*s) {
        return EINVAL;
    }

    errno = 0;
    char* end = NULL;
    uintmax_t v = strtoumax(s, &end, 10);

    if (errno == ERANGE) {
        return ERANGE;
    }
    if (end == s) {
        return EINVAL;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n') {
        end++;
    }
    if (*end != '\0') {
        return EINVAL;
    }
    if (v > (uintmax_t)SIZE_MAX) {
        return ERANGE;
    }

    *out = (size_t)v;
    return 0;
}

static int stack_ptx_compiler_stack_idx_from_data_type(
    const StackPtxStackInfo* info,
    const char* data_type,
    size_t* out_idx
) {
    if (!info || !data_type || !out_idx) {
        return EINVAL;
    }
    char lowered[32];
    size_t len = strlen(data_type);
    if (len == 0 || len >= sizeof(lowered)) {
        return EINVAL;
    }
    for (size_t i = 0; i < len; ++i) {
        lowered[i] = (char)tolower((unsigned char)data_type[i]);
    }
    lowered[len] = '\0';
    for (size_t i = 0; i < info->num_stacks; ++i) {
        const char* prefix = info->stack_literal_prefixes[i];
        if (!prefix) {
            continue;
        }
        if (strcmp(prefix, lowered) == 0) {
            *out_idx = i;
            return 0;
        }
    }
    return EINVAL;
}

static double stack_ptx_compiler_elapsed_ms(const struct timespec* start,
    const struct timespec* end) {
    const double sec = (double)(end->tv_sec - start->tv_sec) * 1000.0;
    const double nsec = (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
    return sec + nsec;
}

static StackPtxCompilerResult stack_ptx_compiler_build_instruction_stubs(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work,
    const StackPtxInstruction** instruction_stubs,
    size_t* out_population_modules
) {
    if (!compiler || !work || !instruction_stubs || !out_population_modules) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (!work->population || work->population_instructions == 0 || work->gene_length == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (compiler->individuals_per_module == 0 || compiler->num_injects == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (work->population_instructions % work->gene_length != 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const size_t population_individuals = work->population_instructions / work->gene_length;
    if (population_individuals == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (population_individuals % compiler->individuals_per_module != 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const size_t num_modules = population_individuals / compiler->individuals_per_module;
    if (work->module_idx >= num_modules) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    for (size_t i = 0; i < compiler->num_injects; ++i) {
        const size_t func_idx = compiler->reverse_indices[i];
        if (func_idx >= compiler->individuals_per_module) {
            return STACK_PTX_COMPILER_ERROR_INTERNAL;
        }
        const size_t population_idx = work->module_idx * compiler->individuals_per_module + func_idx;
        const size_t instruction_idx = population_idx * work->gene_length;
        instruction_stubs[i] = &work->population[instruction_idx];
    }

    *out_population_modules = num_modules;
    return STACK_PTX_COMPILER_SUCCESS;
}

StackPtxCompilerResult stack_ptx_compiler_compile_module(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work,
    size_t thread_idx,
    void** out_cubin,
    size_t* out_cubin_size,
    double* out_stub_ms,
    double* out_render_ms,
    double* out_cubin_ms,
    double* out_compile_ms
) {
    if (!compiler || !work) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (out_cubin && !out_cubin_size) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    if (out_cubin) {
        *out_cubin = NULL;
    }
    if (out_cubin_size) {
        *out_cubin_size = 0;
    }
    if (out_stub_ms) {
        *out_stub_ms = 0.0;
    }
    if (out_render_ms) {
        *out_render_ms = 0.0;
    }
    if (out_cubin_ms) {
        *out_cubin_ms = 0.0;
    }
    if (out_compile_ms) {
        *out_compile_ms = 0.0;
    }

    if (thread_idx >= compiler->capabilities) {
        thread_idx = 0;
    }
    if (!compiler->thread_scratch || !compiler->workspaces || !compiler->ptx_injects) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    StackPtxCompilerThreadScratch* scratch = &compiler->thread_scratch[thread_idx];
    const StackPtxInstruction** instruction_stubs = scratch->instruction_stubs;
    const char** ptx_stubs = scratch->ptx_stubs;
    PtxInjectHandle ptx_inject = compiler->ptx_injects[thread_idx];
    void* workspace = compiler->workspaces[thread_idx];
    if (!instruction_stubs || !ptx_stubs || !ptx_inject || !workspace) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    size_t num_modules = 0;
    nvPTXCompilerHandle nvptx_compiler = NULL;
    nvPTXCompileResult compile_result = NVPTXCOMPILE_SUCCESS;
    size_t cubin_size = 0;
    void* cubin = NULL;
    int return_cubin = (out_cubin != NULL);
    double stub_ms = 0.0;
    double render_ms = 0.0;
    double cubin_ms = 0.0;
    StackPtxCompilerResult result =
        stack_ptx_compiler_build_instruction_stubs(
            compiler,
            work,
            instruction_stubs,
            &num_modules
        );
    if (result != STACK_PTX_COMPILER_SUCCESS) {
        return result;
    }
    (void)num_modules;

    struct timespec compile_start;
    struct timespec compile_end;
    struct timespec stub_start;
    struct timespec stub_end;
    struct timespec render_start;
    struct timespec render_end;
    struct timespec cubin_start;
    struct timespec cubin_end;
    clock_gettime(CLOCK_MONOTONIC, &compile_start);

    StackPtxCompilerArena arena;
    size_t saved_offset = 0;
    void* stack_ptx_workspace_ptr = NULL;
    char* rendered_ptx_ptr = NULL;
    size_t rendered_required = 0;
    size_t rendered_capacity = 0;
    char compile_line_buffer[32];
    const char* ptx_compile_options[1];
    if (!workspace || compiler->workspace_bytes == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    arena.base = (unsigned char*)workspace;
    arena.capacity = compiler->workspace_bytes;
    arena.offset = 0;
    saved_offset = arena.offset;

    size_t stack_ptx_workspace_size = 0;
    if (stack_ptx_compile_workspace_size(compiler->compiler_info, compiler->stack_info,
            &stack_ptx_workspace_size) != STACK_PTX_SUCCESS) {
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }

    if (!stack_ptx_compiler_arena_alloc(
            &arena,
            stack_ptx_workspace_size,
            64,
            &stack_ptx_workspace_ptr
        )) {
        result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    clock_gettime(CLOCK_MONOTONIC, &stub_start);
    for (size_t i = 0; i < compiler->num_injects; ++i) {
        size_t required = 0;
        StackPtxResult stack_ptx_result = stack_ptx_compile(
                compiler->compiler_info,
                compiler->stack_info,
                instruction_stubs[i],
                compiler->registers,
                compiler->num_registers,
                NULL,
                0,
                compiler->request_stubs[i],
                compiler->request_stub_sizes[i],
                compiler->execution_limit,
                stack_ptx_workspace_ptr,
                stack_ptx_workspace_size,
                NULL,
                0,
                &required
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            fprintf(stderr, "stack_ptx_compile (size) failed: %s\n",
                stack_ptx_result_to_string(stack_ptx_result));
            result = STACK_PTX_COMPILER_ERROR_INTERNAL;
            goto cleanup;
        }

        const size_t capacity = required + 1;
        char* ptx_stub_ptr = NULL;
        if (!stack_ptx_compiler_arena_alloc(
                &arena,
                capacity,
                1,
                (void**)&ptx_stub_ptr
            )) {
            result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
            goto cleanup;
        }

        stack_ptx_result = stack_ptx_compile(
                compiler->compiler_info,
                compiler->stack_info,
                instruction_stubs[i],
                compiler->registers,
                compiler->num_registers,
                NULL,
                0,
                compiler->request_stubs[i],
                compiler->request_stub_sizes[i],
                compiler->execution_limit,
                stack_ptx_workspace_ptr,
                stack_ptx_workspace_size,
                ptx_stub_ptr,
                capacity,
                &required
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            fprintf(stderr, "stack_ptx_compile (render) failed: %s\n",
                stack_ptx_result_to_string(stack_ptx_result));
            result = STACK_PTX_COMPILER_ERROR_INTERNAL;
            goto cleanup;
        }
        if (required < capacity) {
            ptx_stub_ptr[required] = '\0';
        }
        ptx_stubs[i] = ptx_stub_ptr;
    }
    clock_gettime(CLOCK_MONOTONIC, &stub_end);
    stub_ms = stack_ptx_compiler_elapsed_ms(&stub_start, &stub_end);

    clock_gettime(CLOCK_MONOTONIC, &render_start);
    PtxInjectResult ptx_inject_result = ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            compiler->num_injects,
            NULL,
            0,
            &rendered_required
        );
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_render_ptx (size) failed: %s\n",
            ptx_inject_result_to_string(ptx_inject_result));
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }

    rendered_capacity = rendered_required + 1;
    if (!stack_ptx_compiler_arena_alloc(
            &arena,
            rendered_capacity,
            1,
            (void**)&rendered_ptx_ptr
        )) {
        result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    ptx_inject_result = ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            compiler->num_injects,
            rendered_ptx_ptr,
            rendered_capacity,
            &rendered_required
        );
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_render_ptx (render) failed: %s\n",
            ptx_inject_result_to_string(ptx_inject_result));
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }
    if (rendered_required < rendered_capacity) {
        rendered_ptx_ptr[rendered_required] = '\0';
    }
    clock_gettime(CLOCK_MONOTONIC, &render_end);
    render_ms = stack_ptx_compiler_elapsed_ms(&render_start, &render_end);

    snprintf(compile_line_buffer, sizeof(compile_line_buffer),
        "--gpu-name=sm_%d%d", compiler->compute_major, compiler->compute_minor);
    ptx_compile_options[0] = compile_line_buffer;

    clock_gettime(CLOCK_MONOTONIC, &cubin_start);
    compile_result = nvPTXCompilerCreate(&nvptx_compiler, rendered_required, rendered_ptx_ptr);
    if (compile_result != NVPTXCOMPILE_SUCCESS) {
        stack_ptx_compiler_nvptx_log(nvptx_compiler, "create", compile_result);
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup_nvptx;
    }

    compile_result = nvPTXCompilerCompile(nvptx_compiler, 1, ptx_compile_options);
    if (compile_result != NVPTXCOMPILE_SUCCESS) {
        stack_ptx_compiler_nvptx_log(nvptx_compiler, "compile", compile_result);
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup_nvptx;
    }

    compile_result = nvPTXCompilerGetCompiledProgramSize(nvptx_compiler, &cubin_size);
    if (compile_result != NVPTXCOMPILE_SUCCESS) {
        stack_ptx_compiler_nvptx_log(nvptx_compiler, "get-program-size", compile_result);
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup_nvptx;
    }

    if (return_cubin) {
        cubin = malloc(cubin_size);
        if (!cubin) {
            result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
            goto cleanup_nvptx;
        }
    } else {
        if (!stack_ptx_compiler_arena_alloc(&arena, cubin_size, 1, &cubin)) {
            result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
            goto cleanup_nvptx;
        }
    }

    compile_result = nvPTXCompilerGetCompiledProgram(nvptx_compiler, cubin);
    if (compile_result != NVPTXCOMPILE_SUCCESS) {
        stack_ptx_compiler_nvptx_log(nvptx_compiler, "get-program", compile_result);
        if (return_cubin && cubin) {
            free(cubin);
        }
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup_nvptx;
    }
    clock_gettime(CLOCK_MONOTONIC, &cubin_end);
    cubin_ms = stack_ptx_compiler_elapsed_ms(&cubin_start, &cubin_end);

    if (return_cubin && out_cubin) {
        *out_cubin = cubin;
    }
    if (out_cubin_size) {
        *out_cubin_size = cubin_size;
    }

cleanup_nvptx:
    if (nvptx_compiler) {
        (void)nvPTXCompilerDestroy(&nvptx_compiler);
    }

cleanup:
    arena.offset = saved_offset;
    clock_gettime(CLOCK_MONOTONIC, &compile_end);
    if (out_stub_ms) {
        *out_stub_ms = stub_ms;
    }
    if (out_render_ms) {
        *out_render_ms = render_ms;
    }
    if (out_cubin_ms) {
        *out_cubin_ms = cubin_ms;
    }
    if (out_compile_ms) {
        *out_compile_ms = stack_ptx_compiler_elapsed_ms(&compile_start, &compile_end);
    }

    return result;
}

StackPtxCompilerResult stack_ptx_compiler_dump_module_ptx(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work,
    size_t thread_idx,
    const char* path,
    size_t* out_ptx_bytes
) {
    if (!compiler || !work || !path || !path[0]) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (out_ptx_bytes) {
        *out_ptx_bytes = 0;
    }

    if (thread_idx >= compiler->capabilities) {
        thread_idx = 0;
    }
    if (!compiler->thread_scratch || !compiler->workspaces || !compiler->ptx_injects) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    StackPtxCompilerThreadScratch* scratch = &compiler->thread_scratch[thread_idx];
    const StackPtxInstruction** instruction_stubs = scratch->instruction_stubs;
    const char** ptx_stubs = scratch->ptx_stubs;
    PtxInjectHandle ptx_inject = compiler->ptx_injects[thread_idx];
    void* workspace = compiler->workspaces[thread_idx];
    if (!instruction_stubs || !ptx_stubs || !ptx_inject || !workspace) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    size_t num_modules = 0;
    StackPtxCompilerResult result =
        stack_ptx_compiler_build_instruction_stubs(
            compiler,
            work,
            instruction_stubs,
            &num_modules
        );
    if (result != STACK_PTX_COMPILER_SUCCESS) {
        return result;
    }
    (void)num_modules;

    StackPtxCompilerArena arena;
    size_t saved_offset = 0;
    void* stack_ptx_workspace_ptr = NULL;
    char* rendered_ptx_ptr = NULL;
    size_t rendered_required = 0;
    size_t rendered_capacity = 0;
    if (!workspace || compiler->workspace_bytes == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    arena.base = (unsigned char*)workspace;
    arena.capacity = compiler->workspace_bytes;
    arena.offset = 0;
    saved_offset = arena.offset;

    size_t stack_ptx_workspace_size = 0;
    if (stack_ptx_compile_workspace_size(compiler->compiler_info, compiler->stack_info,
            &stack_ptx_workspace_size) != STACK_PTX_SUCCESS) {
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }

    if (!stack_ptx_compiler_arena_alloc(
            &arena,
            stack_ptx_workspace_size,
            64,
            &stack_ptx_workspace_ptr
        )) {
        result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (size_t i = 0; i < compiler->num_injects; ++i) {
        size_t required = 0;
        StackPtxResult stack_ptx_result = stack_ptx_compile(
                compiler->compiler_info,
                compiler->stack_info,
                instruction_stubs[i],
                compiler->registers,
                compiler->num_registers,
                NULL,
                0,
                compiler->request_stubs[i],
                compiler->request_stub_sizes[i],
                compiler->execution_limit,
                stack_ptx_workspace_ptr,
                stack_ptx_workspace_size,
                NULL,
                0,
                &required
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            fprintf(stderr, "stack_ptx_compile (size) failed: %s\n",
                stack_ptx_result_to_string(stack_ptx_result));
            result = STACK_PTX_COMPILER_ERROR_INTERNAL;
            goto cleanup;
        }

        const size_t capacity = required + 1;
        char* ptx_stub_ptr = NULL;
        if (!stack_ptx_compiler_arena_alloc(
                &arena,
                capacity,
                1,
                (void**)&ptx_stub_ptr
            )) {
            result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
            goto cleanup;
        }

        stack_ptx_result = stack_ptx_compile(
                compiler->compiler_info,
                compiler->stack_info,
                instruction_stubs[i],
                compiler->registers,
                compiler->num_registers,
                NULL,
                0,
                compiler->request_stubs[i],
                compiler->request_stub_sizes[i],
                compiler->execution_limit,
                stack_ptx_workspace_ptr,
                stack_ptx_workspace_size,
                ptx_stub_ptr,
                capacity,
                &required
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            fprintf(stderr, "stack_ptx_compile (render) failed: %s\n",
                stack_ptx_result_to_string(stack_ptx_result));
            result = STACK_PTX_COMPILER_ERROR_INTERNAL;
            goto cleanup;
        }
        if (required < capacity) {
            ptx_stub_ptr[required] = '\0';
        }
        ptx_stubs[i] = ptx_stub_ptr;
    }

    PtxInjectResult ptx_inject_result = ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            compiler->num_injects,
            NULL,
            0,
            &rendered_required
        );
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_render_ptx (size) failed: %s\n",
            ptx_inject_result_to_string(ptx_inject_result));
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }

    rendered_capacity = rendered_required + 1;
    if (!stack_ptx_compiler_arena_alloc(
            &arena,
            rendered_capacity,
            1,
            (void**)&rendered_ptx_ptr
        )) {
        result = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    ptx_inject_result = ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            compiler->num_injects,
            rendered_ptx_ptr,
            rendered_capacity,
            &rendered_required
        );
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_render_ptx (render) failed: %s\n",
            ptx_inject_result_to_string(ptx_inject_result));
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }
    if (rendered_required < rendered_capacity) {
        rendered_ptx_ptr[rendered_required] = '\0';
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for writing\n", path);
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }
    size_t written = fwrite(rendered_ptx_ptr, 1, rendered_required, fp);
    if (written != rendered_required) {
        fprintf(stderr, "failed to write %zu bytes to %s\n", rendered_required, path);
        fclose(fp);
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "failed to close %s\n", path);
        result = STACK_PTX_COMPILER_ERROR_INTERNAL;
        goto cleanup;
    }

    if (out_ptx_bytes) {
        *out_ptx_bytes = rendered_required;
    }

cleanup:
    arena.offset = saved_offset;
    return result;
}

static void stack_ptx_compiler_execute_work(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work
) {
    if (!compiler || !work) {
        return;
    }

    StackPtxCompilerOutput output;
    memset(&output, 0, sizeof(output));
    output.job_id = work->job_id;
    output.module_idx = work->module_idx;

    size_t thread_idx = 0;
#if STACK_PTX_COMPILER_OPENMP_ENABLED
    thread_idx = (size_t)omp_get_thread_num();
#endif
    if (thread_idx >= compiler->capabilities) {
        thread_idx = 0;
    }

    output.status = stack_ptx_compiler_compile_module(
        compiler,
        work,
        thread_idx,
        &output.cubin,
        &output.cubin_size,
        NULL,
        NULL,
        NULL,
        &output.compile_ms
    );

    if (output.status != STACK_PTX_COMPILER_SUCCESS && output.cubin) {
        free(output.cubin);
        output.cubin = NULL;
        output.cubin_size = 0;
    }

    stack_ptx_compiler_result_push(compiler, &output);
}

static void* stack_ptx_compiler_worker_main(void* userdata) {
    StackPtxCompilerLocal* compiler = (StackPtxCompilerLocal*)userdata;
    if (!compiler) {
        return NULL;
    }

#if STACK_PTX_COMPILER_OPENMP_ENABLED
    omp_set_dynamic(0);
    if (compiler->capabilities > 0) {
        omp_set_num_threads((int)compiler->capabilities);
    }
    if (compiler->capabilities <= 1) {
        while (!stack_ptx_compiler_should_stop(compiler)) {
            StackPtxCompilerWork work;
            if (!stack_ptx_compiler_work_pop(compiler, &work)) {
                if (stack_ptx_compiler_should_stop(compiler)) {
                    break;
                }
                stack_ptx_compiler_sleep_short();
                continue;
            }
            stack_ptx_compiler_execute_work(compiler, &work);
        }
        return NULL;
    }

#pragma omp parallel num_threads((int)compiler->capabilities)
    {
#pragma omp single
        {
            while (!stack_ptx_compiler_should_stop(compiler)) {
                StackPtxCompilerWork work;
                if (!stack_ptx_compiler_work_pop(compiler, &work)) {
                    if (stack_ptx_compiler_should_stop(compiler)) {
                        break;
                    }
                    stack_ptx_compiler_sleep_short();
                    continue;
                }
#pragma omp task firstprivate(work)
                { stack_ptx_compiler_execute_work(compiler, &work); }
            }
#pragma omp taskwait
        }
    }

    return NULL;
#else
    while (!stack_ptx_compiler_should_stop(compiler)) {
        StackPtxCompilerWork work;
        if (!stack_ptx_compiler_work_pop(compiler, &work)) {
            if (stack_ptx_compiler_should_stop(compiler)) {
                break;
            }
            stack_ptx_compiler_sleep_short();
            continue;
        }
        stack_ptx_compiler_execute_work(compiler, &work);
    }
    return NULL;
#endif
}

static int stack_ptx_compiler_parse_sm(const char* s, unsigned int* major_out, unsigned int* minor_out) {
    if (!s || !s[0] || !major_out || !minor_out) {
        return 0;
    }

    while (*s == ' ' || *s == '\t' || *s == '\n') {
        ++s;
    }
    if ((s[0] == 's' || s[0] == 'S') && (s[1] == 'm' || s[1] == 'M') && s[2] == '_') {
        s += 3;
    }

    char* end = NULL;
    errno = 0;
    unsigned long major = strtoul(s, &end, 10);
    if (errno != 0 || end == s) {
        return 0;
    }

    unsigned long minor = 0;
    if (*end == '.') {
        const char* minor_s = end + 1;
        char* minor_end = NULL;
        errno = 0;
        minor = strtoul(minor_s, &minor_end, 10);
        if (errno != 0 || minor_end == minor_s) {
            return 0;
        }
        end = minor_end;
    } else if (*end == '\0') {
        if (major >= 10 && major <= 99) {
            minor = major % 10;
            major = major / 10;
        } else {
            minor = 0;
        }
    }

    while (*end == ' ' || *end == '\t' || *end == '\n') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }

    if (major == 0 || minor > 9) {
        return 0;
    }

    *major_out = (unsigned int)major;
    *minor_out = (unsigned int)minor;
    return 1;
}

static void stack_ptx_compiler_default_sm(unsigned int* major_out, unsigned int* minor_out) {
    if (!major_out || !minor_out) {
        return;
    }
    *major_out = 8;
    *minor_out = 0;
}

static void stack_ptx_compiler_get_sm_fallback(unsigned int* major_out, unsigned int* minor_out) {
    const char* env = getenv("STACK_PTX_COMPILER_SM");
    if (stack_ptx_compiler_parse_sm(env, major_out, minor_out)) {
        return;
    }
    env = getenv("STACK_PTX_NNG_SM");
    if (stack_ptx_compiler_parse_sm(env, major_out, minor_out)) {
        return;
    }
    stack_ptx_compiler_default_sm(major_out, minor_out);
}

static StackPtxCompilerResult stack_ptx_compiler_local_destroy(StackPtxCompilerLocal* compiler);

static StackPtxCompilerResult stack_ptx_compiler_local_create(
    const StackPtxCompilerHandleConfig* config,
    int start_worker_thread,
    StackPtxCompilerLocal** out_compiler,
    size_t* out_capabilities,
    size_t* out_queue_slots
) {
    if (out_capabilities) {
        *out_capabilities = 0;
    }
    if (out_queue_slots) {
        *out_queue_slots = 0;
    }
    if (!config || !out_compiler) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    *out_compiler = NULL;
    if (!config->kernel_ptx || !config->kernel_ptx[0]) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (!config->stack_info) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (config->kernel_num_kernels == 0 ||
        config->kernel_groups_per_kernel == 0 ||
        config->execution_limit == 0 ||
        config->workspace_bytes == 0) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    size_t capabilities = 1;
#if STACK_PTX_COMPILER_OPENMP_ENABLED
    int max_threads = omp_get_max_threads();
    if (max_threads > 0) {
        capabilities = (size_t)max_threads;
    }
    if (STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS > 0 &&
        capabilities > (size_t)STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS) {
        capabilities = (size_t)STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS;
    }
#endif
    if (capabilities < 1) {
        capabilities = 1;
    }
    if (config->requested_capabilities > 0) {
#if STACK_PTX_COMPILER_OPENMP_ENABLED
        capabilities = config->requested_capabilities;
        if (capabilities < 1) {
            capabilities = 1;
        }
        if (STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS > 0 &&
            capabilities > (size_t)STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS) {
            capabilities = (size_t)STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS;
        }
#else
        capabilities = 1;
#endif
    }

    size_t queue_capacity = capabilities * 2;
    if (queue_capacity < 1) {
        queue_capacity = 1;
    }

    StackPtxCompilerLocal* compiler = (StackPtxCompilerLocal*)calloc(1, sizeof(*compiler));
    if (!compiler) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }
    compiler->capabilities = capabilities;
    compiler->queue_capacity = queue_capacity;
    compiler->work_capacity = queue_capacity;
    compiler->result_capacity = queue_capacity;
    compiler->workspace_bytes = config->workspace_bytes;
    compiler->worker_thread_started = 0;
    atomic_init(&compiler->worker_stop, 0);

    if (pthread_mutex_init(&compiler->work_mutex, NULL) != 0) {
        free(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }
    compiler->work_mutex_initialized = 1;
    if (pthread_mutex_init(&compiler->result_mutex, NULL) != 0) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }
    compiler->result_mutex_initialized = 1;

    const char* inject_prefix = config->inject_prefix ? config->inject_prefix : kDefaultInjectPrefix;
    const char* input_register_name = config->input_register_name ?
        config->input_register_name : kDefaultInputRegisterName;
    const char* output_register_prefix = config->output_register_prefix ?
        config->output_register_prefix : kDefaultOutputRegisterPrefix;
    const char* kernel_name_format = (config->kernel_name_format && config->kernel_name_format[0])
        ? config->kernel_name_format
        : kDefaultKernelNameFormat;
    compiler->stack_info = config->stack_info;
    compiler->compiler_info = config->compiler_info ? config->compiler_info : &kCompilerInfo;

    compiler->kernel_num_kernels = config->kernel_num_kernels;
    compiler->kernel_groups_per_kernel = config->kernel_groups_per_kernel;
    compiler->execution_limit = config->execution_limit;
    compiler->individuals_per_kernel = compiler->kernel_groups_per_kernel;
    compiler->individuals_per_module = compiler->kernel_num_kernels * compiler->individuals_per_kernel;
    if (compiler->individuals_per_module == 0) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    compiler->kernel_name_format = stack_ptx_compiler_strdup(kernel_name_format);
    if (!compiler->kernel_name_format) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }

    compiler->ptx_injects = (PtxInjectHandle*)calloc(capabilities, sizeof(*compiler->ptx_injects));
    if (!compiler->ptx_injects) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }
    compiler->num_ptx_injects = capabilities;
    for (size_t i = 0; i < capabilities; ++i) {
        PtxInjectResult inject_rc = ptx_inject_create(&compiler->ptx_injects[i], config->kernel_ptx);
        if (inject_rc != PTX_INJECT_SUCCESS) {
            stack_ptx_compiler_log_ptx_inject_error("create", inject_rc);
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
        }
    }

    PtxInjectResult inject_rc = ptx_inject_num_injects(compiler->ptx_injects[0], &compiler->num_injects);
    if (inject_rc != PTX_INJECT_SUCCESS) {
        stack_ptx_compiler_log_ptx_inject_error("num_injects", inject_rc);
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }

    if (compiler->num_injects != compiler->individuals_per_module) {
        fprintf(stderr,
            "ptx_inject num_injects mismatch: ptx=%zu expected=%zu\n",
            compiler->num_injects, compiler->individuals_per_module);
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    char inject_name[64];
    snprintf(inject_name, sizeof(inject_name), "%s0", inject_prefix);
    size_t func_idx = 0;
    size_t num_args = 0;
    inject_rc = ptx_inject_inject_info_by_name(
            compiler->ptx_injects[0],
            inject_name,
            &func_idx,
            &num_args,
            NULL
        );
    if (inject_rc != PTX_INJECT_SUCCESS) {
        stack_ptx_compiler_log_ptx_inject_error("inject_info_by_name", inject_rc);
        fprintf(stderr, "ptx_inject inject name: %s\n", inject_name);
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }
    if (num_args < 2) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    const char* input_prefix = (input_register_name && input_register_name[0])
        ? input_register_name
        : NULL;
    const char* output_prefix = (output_register_prefix && output_register_prefix[0])
        ? output_register_prefix
        : NULL;
    size_t input_count = 0;
    size_t output_count = 0;
    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        const char* var_name = NULL;
        PtxInjectMutType mut_type = PTX_INJECT_MUT_TYPE_IN;
        inject_rc = ptx_inject_variable_info_by_index(
                compiler->ptx_injects[0],
                func_idx,
                arg_idx,
                &var_name,
                NULL,
                &mut_type,
                NULL,
                NULL
            );
        if (inject_rc != PTX_INJECT_SUCCESS) {
            stack_ptx_compiler_log_ptx_inject_error("variable_info_by_index", inject_rc);
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
        }
        if (mut_type == PTX_INJECT_MUT_TYPE_IN) {
            if (!input_prefix || stack_ptx_compiler_str_starts_with(var_name, input_prefix)) {
                input_count += 1;
            }
        } else if (mut_type == PTX_INJECT_MUT_TYPE_OUT || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
            if (!output_prefix || stack_ptx_compiler_str_starts_with(var_name, output_prefix)) {
                output_count += 1;
            }
        }
    }

    if (input_count == 0 || output_count == 0) {
        fprintf(stderr,
            "ptx_inject arg classification failed: inputs=%zu outputs=%zu\n",
            input_count, output_count);
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    compiler->input_dims = input_count;
    compiler->embed_dims = output_count;
    compiler->num_registers = input_count + output_count;
    compiler->num_requests = output_count;

    compiler->registers = (StackPtxRegister*)calloc(
        compiler->num_registers,
        sizeof(*compiler->registers)
    );
    compiler->requests = (size_t*)calloc(compiler->num_requests, sizeof(*compiler->requests));
    compiler->request_stubs = (const size_t**)calloc(
        compiler->num_injects,
        sizeof(*compiler->request_stubs)
    );
    compiler->request_stub_sizes = (size_t*)calloc(
        compiler->num_injects,
        sizeof(*compiler->request_stub_sizes)
    );
    compiler->reverse_indices = (size_t*)calloc(compiler->num_injects, sizeof(*compiler->reverse_indices));

    if (!compiler->registers || !compiler->requests || !compiler->request_stubs ||
        !compiler->request_stub_sizes || !compiler->reverse_indices) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }

    size_t input_written = 0;
    size_t output_written = 0;
    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        const char* var_name = NULL;
        const char* reg_name = NULL;
        PtxInjectMutType mut_type = PTX_INJECT_MUT_TYPE_IN;
        const char* data_type = NULL;
        inject_rc = ptx_inject_variable_info_by_index(
                compiler->ptx_injects[0],
                func_idx,
                arg_idx,
                &var_name,
                &reg_name,
                &mut_type,
                NULL,
                &data_type
            );
        if (inject_rc != PTX_INJECT_SUCCESS) {
            stack_ptx_compiler_log_ptx_inject_error("variable_info_by_index", inject_rc);
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
        }
        if (mut_type == PTX_INJECT_MUT_TYPE_IN) {
            if (input_prefix && !stack_ptx_compiler_str_starts_with(var_name, input_prefix)) {
                continue;
            }
            if (input_written >= input_count) {
                stack_ptx_compiler_local_destroy(compiler);
                _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
            }
            size_t stack_idx = 0;
            if (stack_ptx_compiler_stack_idx_from_data_type(
                    compiler->stack_info, data_type, &stack_idx) != 0) {
                stack_ptx_compiler_local_destroy(compiler);
                _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
            }
            compiler->registers[input_written].stack_idx = stack_idx;
            compiler->registers[input_written].name = reg_name;
            input_written += 1;
        } else if (mut_type == PTX_INJECT_MUT_TYPE_OUT || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
            if (output_prefix && !stack_ptx_compiler_str_starts_with(var_name, output_prefix)) {
                continue;
            }
            if (output_written >= output_count) {
                stack_ptx_compiler_local_destroy(compiler);
                _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
            }
            size_t stack_idx = 0;
            if (stack_ptx_compiler_stack_idx_from_data_type(
                    compiler->stack_info, data_type, &stack_idx) != 0) {
                stack_ptx_compiler_local_destroy(compiler);
                _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
            }
            const size_t reg_idx = input_count + output_written;
            compiler->registers[reg_idx].stack_idx = stack_idx;
            compiler->registers[reg_idx].name = reg_name;
            compiler->requests[output_written] = reg_idx;
            output_written += 1;
        }
    }
    if (input_written != input_count || output_written != output_count) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }

    for (size_t i = 0; i < compiler->num_injects; ++i) {
        compiler->request_stubs[i] = compiler->requests;
        compiler->request_stub_sizes[i] = compiler->num_requests;
    }

    for (size_t i = 0; i < compiler->num_injects; ++i) {
        const char* inject_site_name = NULL;
        if (ptx_inject_inject_info_by_index(
                compiler->ptx_injects[0],
                i,
                &inject_site_name,
                NULL,
                NULL
            ) != PTX_INJECT_SUCCESS) {
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
        }
        if (!inject_site_name || !stack_ptx_compiler_str_starts_with(inject_site_name, inject_prefix)) {
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
        }
        size_t func_value = 0;
        if (stack_ptx_compiler_parse_size_t(
                inject_site_name + strlen(inject_prefix),
                &func_value
            ) != 0) {
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
        }
        compiler->reverse_indices[i] = func_value;
    }

    if (config->device_capability_major > 0 && config->device_capability_minor <= 9) {
        compiler->compute_major = (int)config->device_capability_major;
        compiler->compute_minor = (int)config->device_capability_minor;
    } else {
        unsigned int sm_major = 0;
        unsigned int sm_minor = 0;
        stack_ptx_compiler_get_sm_fallback(&sm_major, &sm_minor);
        compiler->compute_major = (int)sm_major;
        compiler->compute_minor = (int)sm_minor;
    }

    compiler->thread_scratch = (StackPtxCompilerThreadScratch*)calloc(
        capabilities,
        sizeof(*compiler->thread_scratch)
    );
    compiler->workspaces = (void**)calloc(capabilities, sizeof(*compiler->workspaces));
    if (!compiler->thread_scratch || !compiler->workspaces) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }
    for (size_t i = 0; i < capabilities; ++i) {
        compiler->thread_scratch[i].instruction_stubs =
            (const StackPtxInstruction**)calloc(
                compiler->num_injects,
                sizeof(*compiler->thread_scratch[i].instruction_stubs)
            );
        compiler->thread_scratch[i].ptx_stubs =
            (const char**)calloc(
                compiler->num_injects,
                sizeof(*compiler->thread_scratch[i].ptx_stubs)
            );
        compiler->workspaces[i] = malloc(compiler->workspace_bytes);
        if (!compiler->thread_scratch[i].instruction_stubs ||
            !compiler->thread_scratch[i].ptx_stubs ||
            !compiler->workspaces[i]) {
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
        }
    }

    compiler->work_queue = (StackPtxCompilerWork*)calloc(
        compiler->work_capacity,
        sizeof(*compiler->work_queue)
    );
    compiler->results = (StackPtxCompilerOutput*)calloc(
        compiler->result_capacity,
        sizeof(*compiler->results)
    );
    if (!compiler->work_queue || !compiler->results) {
        stack_ptx_compiler_local_destroy(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }

    if (start_worker_thread) {
        if (pthread_create(&compiler->worker_thread, NULL, stack_ptx_compiler_worker_main, compiler) != 0) {
            stack_ptx_compiler_local_destroy(compiler);
            _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
        }
        compiler->worker_thread_started = 1;
    }

    *out_compiler = compiler;
    if (out_capabilities) {
        *out_capabilities = capabilities;
    }
    if (out_queue_slots) {
        *out_queue_slots = queue_capacity;
    }
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_local_destroy(StackPtxCompilerLocal* compiler) {
    if (!compiler) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    atomic_store(&compiler->worker_stop, 1);
    if (compiler->worker_thread_started) {
        pthread_join(compiler->worker_thread, NULL);
        compiler->worker_thread_started = 0;
    }

    stack_ptx_compiler_clear_results(compiler);
    if (compiler->ptx_injects) {
        for (size_t i = 0; i < compiler->num_ptx_injects; ++i) {
            if (compiler->ptx_injects[i]) {
                (void)ptx_inject_destroy(compiler->ptx_injects[i]);
            }
        }
    }
    free(compiler->ptx_injects);
    compiler->ptx_injects = NULL;
    free(compiler->registers);
    free(compiler->requests);
    free(compiler->request_stubs);
    free(compiler->request_stub_sizes);
    free(compiler->reverse_indices);
    free(compiler->kernel_name_format);
    if (compiler->thread_scratch) {
        for (size_t i = 0; i < compiler->capabilities; ++i) {
            free(compiler->thread_scratch[i].instruction_stubs);
            free(compiler->thread_scratch[i].ptx_stubs);
        }
    }
    free(compiler->thread_scratch);
    if (compiler->workspaces) {
        for (size_t i = 0; i < compiler->capabilities; ++i) {
            free(compiler->workspaces[i]);
        }
    }
    free(compiler->workspaces);
    free(compiler->work_queue);
    free(compiler->results);
    if (compiler->work_mutex_initialized) {
        pthread_mutex_destroy(&compiler->work_mutex);
        compiler->work_mutex_initialized = 0;
    }
    if (compiler->result_mutex_initialized) {
        pthread_mutex_destroy(&compiler->result_mutex);
        compiler->result_mutex_initialized = 0;
    }
    free(compiler);
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_local_submit(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work
) {
    if (!compiler || !work) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (stack_ptx_compiler_should_stop(compiler)) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    StackPtxCompilerResult rc = stack_ptx_compiler_work_push(compiler, work);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        _STACK_PTX_COMPILER_ERROR(rc);
    }
    return rc;
}

static StackPtxCompilerResult stack_ptx_compiler_local_poll(
    StackPtxCompilerLocal* compiler,
    StackPtxCompilerOutput* out_result,
    int* out_has_result
) {
    if (!compiler || !out_result || !out_has_result) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    *out_has_result = 0;
    if (stack_ptx_compiler_result_pop(compiler, out_result)) {
        *out_has_result = 1;
    }
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_backend_local_destroy(void* impl) {
    return stack_ptx_compiler_local_destroy((StackPtxCompilerLocal*)impl);
}

static StackPtxCompilerResult stack_ptx_compiler_backend_local_submit(void* impl, const StackPtxCompilerWork* work) {
    return stack_ptx_compiler_local_submit((StackPtxCompilerLocal*)impl, work);
}

static StackPtxCompilerResult stack_ptx_compiler_backend_local_poll(
    void* impl,
    StackPtxCompilerOutput* out_result,
    int* out_has_result
) {
    return stack_ptx_compiler_local_poll((StackPtxCompilerLocal*)impl, out_result, out_has_result);
}

static const StackPtxCompilerVTable kStackPtxCompilerLocalVTable = {
    stack_ptx_compiler_backend_local_destroy,
    stack_ptx_compiler_backend_local_submit,
    stack_ptx_compiler_backend_local_poll,
};

StackPtxCompilerResult stack_ptx_compiler_backend_local_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerBackendCreateOut* out
) {
    if (!config || !out) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    memset(out, 0, sizeof(*out));

    StackPtxCompilerLocal* compiler = NULL;
    size_t capabilities = 0;
    size_t queue_slots = 0;
    StackPtxCompilerResult rc = stack_ptx_compiler_local_create(
        config,
        1,
        &compiler,
        &capabilities,
        &queue_slots
    );
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        return rc;
    }

    out->impl = compiler;
    out->vtable = &kStackPtxCompilerLocalVTable;
    out->capabilities = capabilities;
    out->queue_slots = queue_slots;
    return STACK_PTX_COMPILER_SUCCESS;
}

StackPtxCompilerResult stack_ptx_compiler_backend_local_create_no_worker(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerBackendCreateOut* out
) {
    if (!config || !out) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    memset(out, 0, sizeof(*out));

    StackPtxCompilerLocal* compiler = NULL;
    size_t capabilities = 0;
    size_t queue_slots = 0;
    StackPtxCompilerResult rc = stack_ptx_compiler_local_create(
        config,
        0,
        &compiler,
        &capabilities,
        &queue_slots
    );
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        return rc;
    }

    out->impl = compiler;
    out->vtable = &kStackPtxCompilerLocalVTable;
    out->capabilities = capabilities;
    out->queue_slots = queue_slots;
    return STACK_PTX_COMPILER_SUCCESS;
}
