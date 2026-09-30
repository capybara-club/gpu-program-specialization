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
#pragma once

#include "stack_ptx_compiler.h"

typedef struct StackPtxCompilerLocal StackPtxCompilerLocal;

typedef struct {
    StackPtxCompilerResult (*destroy)(void* impl);
    StackPtxCompilerResult (*submit)(void* impl, const StackPtxCompilerWork* work);
    StackPtxCompilerResult (*poll)(void* impl, StackPtxCompilerOutput* out_result, int* out_has_result);
} StackPtxCompilerVTable;

typedef struct {
    void* impl;
    const StackPtxCompilerVTable* vtable;
    size_t capabilities;
    size_t queue_slots;
} StackPtxCompilerBackendCreateOut;

StackPtxCompilerResult stack_ptx_compiler_backend_local_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerBackendCreateOut* out
);

StackPtxCompilerResult stack_ptx_compiler_backend_local_create_no_worker(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerBackendCreateOut* out
);

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
);

StackPtxCompilerResult stack_ptx_compiler_dump_module_ptx(
    StackPtxCompilerLocal* compiler,
    const StackPtxCompilerWork* work,
    size_t thread_idx,
    const char* path,
    size_t* out_ptx_bytes
);

#ifdef STACK_PTX_COMPILER_ENABLE_NNG
StackPtxCompilerResult stack_ptx_compiler_backend_nng_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerBackendCreateOut* out
);

StackPtxCompilerResult stack_ptx_compiler_backend_nng_connected_server_count(
    void* impl,
    size_t* out_num_servers
);

StackPtxCompilerResult stack_ptx_compiler_backend_nng_connected_server_addr(
    void* impl,
    size_t server_idx,
    const char** out_addr
);

StackPtxCompilerResult stack_ptx_compiler_backend_nng_connected_server_conn_count(
    void* impl,
    size_t server_idx,
    size_t* out_num_conns
);
#endif
