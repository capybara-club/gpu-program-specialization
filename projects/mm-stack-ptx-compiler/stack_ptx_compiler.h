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
#ifndef STACK_PTX_COMPILER_H
#define STACK_PTX_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#include <stack_ptx.h>

#ifdef __cplusplus
#define STACK_PTX_COMPILER_PUBLIC_DEC extern "C"
#define STACK_PTX_COMPILER_PUBLIC_DEF extern "C"
#else
#define STACK_PTX_COMPILER_PUBLIC_DEC extern
#define STACK_PTX_COMPILER_PUBLIC_DEF
#endif

#define STACK_PTX_COMPILER_VERSION_MAJOR ((uint16_t)1u)
#define STACK_PTX_COMPILER_VERSION_MINOR ((uint16_t)0u)
#define STACK_PTX_COMPILER_VERSION_PATCH ((uint16_t)0u)

typedef enum {
    STACK_PTX_COMPILER_SUCCESS = 0,
    STACK_PTX_COMPILER_ERROR_INVALID_VALUE = 1,
    STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY = 2,
    STACK_PTX_COMPILER_ERROR_QUEUE_FULL = 3,
    STACK_PTX_COMPILER_ERROR_INTERNAL = 4,
    STACK_PTX_COMPILER_ERROR_VERSION = 5,
    STACK_PTX_COMPILER_RESULT_NUM_ENUMS
} StackPtxCompilerResult;

typedef enum {
    STACK_PTX_KERNEL_GEN_SUCCESS = 0,
    STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE = 1,
    STACK_PTX_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER = 2,
    STACK_PTX_KERNEL_GEN_ERROR_OUT_OF_MEMORY = 3,
    STACK_PTX_KERNEL_GEN_ERROR_FORMAT = 4
} StackPtxKernelGenResult;

typedef enum {
    STACK_PTX_COMPILER_BACKEND_LOCAL = 0,
    STACK_PTX_COMPILER_BACKEND_NNG = 1,
    STACK_PTX_COMPILER_BACKEND_NUM_ENUMS
} StackPtxCompilerBackend;

#if defined(STACK_PTX_COMPILER_DEBUG) || defined(ELITE_NLE_COMPILER_DEBUG)
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define _STACK_PTX_COMPILER_ERROR(ans)                                                      \
    do {                                                                                    \
        StackPtxCompilerResult _result = (ans);                                             \
        const char* error_name = stack_ptx_compiler_result_to_string(_result);              \
        fprintf(stderr, "STACK_PTX_COMPILER_ERROR: %s \n  %s %d\n",                          \
                error_name, __FILE__, __LINE__);                                            \
        assert(0);                                                                          \
        exit(1);                                                                            \
    } while (0)

#define _STACK_PTX_COMPILER_CHECK_RET(ans)                                                  \
    do {                                                                                    \
        StackPtxCompilerResult _result = (ans);                                             \
        if (_result != STACK_PTX_COMPILER_SUCCESS) {                                        \
            const char* error_name = stack_ptx_compiler_result_to_string(_result);          \
            fprintf(stderr, "STACK_PTX_COMPILER_CHECK: %s \n  %s %d\n",                      \
                    error_name, __FILE__, __LINE__);                                        \
            assert(0);                                                                      \
            exit(1);                                                                        \
            return _result;                                                                 \
        }                                                                                   \
    } while (0)
#else
#define _STACK_PTX_COMPILER_ERROR(ans)                                                      \
    do {                                                                                    \
        StackPtxCompilerResult _result = (ans);                                             \
        return _result;                                                                     \
    } while (0)

#define _STACK_PTX_COMPILER_CHECK_RET(ans)                                                  \
    do {                                                                                    \
        StackPtxCompilerResult _result = (ans);                                             \
        if (_result != STACK_PTX_COMPILER_SUCCESS) return _result;                          \
    } while (0)
#endif

struct StackPtxCompilerHandleImpl;
typedef struct StackPtxCompilerHandleImpl StackPtxCompilerHandle;

typedef struct {
    const StackPtxInstruction* population;
    size_t population_instructions;
    size_t gene_length;
    size_t module_idx;
    uint64_t job_id;
} StackPtxCompilerWork;

typedef struct {
    uint64_t job_id;
    size_t module_idx;
    void* cubin;
    size_t cubin_size;
    double compile_ms;
    StackPtxCompilerResult status;
} StackPtxCompilerOutput;

typedef struct {
    const char* kernel_ptx;
    const char* kernel_name_format;
    size_t kernel_num_kernels;
    size_t kernel_groups_per_kernel;
    size_t execution_limit;
    size_t max_results;
    size_t workspace_bytes;
    const StackPtxStackInfo* stack_info;
    const StackPtxCompilerInfo* compiler_info;
    const char* inject_prefix;
    const char* input_register_name;
    const char* output_register_prefix;

    StackPtxCompilerBackend backend;
    size_t requested_capabilities;
    const char* nng_addr;

    unsigned int device_capability_major;
    unsigned int device_capability_minor;
} StackPtxCompilerHandleConfig;

STACK_PTX_COMPILER_PUBLIC_DEC const char* stack_ptx_compiler_result_to_string(StackPtxCompilerResult result);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerHandle** out_compiler,
    size_t* out_capabilities,
    size_t* out_queue_slots
);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_destroy(StackPtxCompilerHandle* compiler);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_submit(
    StackPtxCompilerHandle* compiler,
    const StackPtxCompilerWork* work
);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_poll(
    StackPtxCompilerHandle* compiler,
    StackPtxCompilerOutput* out_result,
    int* out_has_result
);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_nng_connected_server_count(
    StackPtxCompilerHandle* compiler,
    size_t* out_num_servers
);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_nng_connected_server_addr(
    StackPtxCompilerHandle* compiler,
    size_t server_idx,
    const char** out_addr
);

STACK_PTX_COMPILER_PUBLIC_DEC StackPtxCompilerResult stack_ptx_compiler_nng_connected_server_conn_count(
    StackPtxCompilerHandle* compiler,
    size_t server_idx,
    size_t* out_num_conns
);

#endif // STACK_PTX_COMPILER_H
