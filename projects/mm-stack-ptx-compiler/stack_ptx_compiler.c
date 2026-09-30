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

#include <stdlib.h>
#include <string.h>

struct StackPtxCompilerHandleImpl {
    const StackPtxCompilerVTable* vtable;
    void* impl;
    size_t capabilities;
    size_t queue_slots;
    StackPtxCompilerBackend backend;
};

STACK_PTX_COMPILER_PUBLIC_DEF const char*
stack_ptx_compiler_result_to_string(StackPtxCompilerResult result) {
    switch (result) {
        case STACK_PTX_COMPILER_SUCCESS:
            return "STACK_PTX_COMPILER_SUCCESS";
        case STACK_PTX_COMPILER_ERROR_INVALID_VALUE:
            return "STACK_PTX_COMPILER_ERROR_INVALID_VALUE";
        case STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY:
            return "STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY";
        case STACK_PTX_COMPILER_ERROR_QUEUE_FULL:
            return "STACK_PTX_COMPILER_ERROR_QUEUE_FULL";
        case STACK_PTX_COMPILER_ERROR_INTERNAL:
            return "STACK_PTX_COMPILER_ERROR_INTERNAL";
        case STACK_PTX_COMPILER_ERROR_VERSION:
            return "STACK_PTX_COMPILER_ERROR_VERSION";
        case STACK_PTX_COMPILER_RESULT_NUM_ENUMS:
            return "STACK_PTX_COMPILER_RESULT_NUM_ENUMS";
    }
    return "STACK_PTX_COMPILER_ERROR_UNKNOWN";
}

StackPtxCompilerResult stack_ptx_compiler_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerHandle** out_compiler,
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

    StackPtxCompilerBackend backend = config->backend;
    if (backend >= STACK_PTX_COMPILER_BACKEND_NUM_ENUMS) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }

    StackPtxCompilerBackendCreateOut created;
    memset(&created, 0, sizeof(created));

    StackPtxCompilerResult rc = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    switch (backend) {
        case STACK_PTX_COMPILER_BACKEND_LOCAL:
            rc = stack_ptx_compiler_backend_local_create(config, &created);
            break;
        case STACK_PTX_COMPILER_BACKEND_NNG:
#ifdef STACK_PTX_COMPILER_ENABLE_NNG
            rc = stack_ptx_compiler_backend_nng_create(config, &created);
#else
            rc = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
#endif
            break;
        default:
            rc = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
            break;
    }

    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        _STACK_PTX_COMPILER_ERROR(rc);
    }
    if (!created.impl || !created.vtable) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }

    StackPtxCompilerHandle* compiler = (StackPtxCompilerHandle*)calloc(1, sizeof(*compiler));
    if (!compiler) {
        created.vtable->destroy(created.impl);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY);
    }

    compiler->vtable = created.vtable;
    compiler->impl = created.impl;
    compiler->capabilities = created.capabilities;
    compiler->queue_slots = created.queue_slots;
    compiler->backend = backend;

    *out_compiler = compiler;
    if (out_capabilities) {
        *out_capabilities = created.capabilities;
    }
    if (out_queue_slots) {
        *out_queue_slots = created.queue_slots;
    }
    return STACK_PTX_COMPILER_SUCCESS;
}

StackPtxCompilerResult stack_ptx_compiler_destroy(StackPtxCompilerHandle* compiler) {
    if (!compiler) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (!compiler->vtable || !compiler->impl) {
        free(compiler);
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INTERNAL);
    }

    StackPtxCompilerResult rc = compiler->vtable->destroy(compiler->impl);
    free(compiler);
    return rc;
}

StackPtxCompilerResult stack_ptx_compiler_submit(
    StackPtxCompilerHandle* compiler,
    const StackPtxCompilerWork* work
) {
    if (!compiler || !compiler->vtable || !compiler->impl || !work) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    StackPtxCompilerResult rc = compiler->vtable->submit(compiler->impl, work);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        _STACK_PTX_COMPILER_ERROR(rc);
    }
    return rc;
}

StackPtxCompilerResult stack_ptx_compiler_poll(
    StackPtxCompilerHandle* compiler,
    StackPtxCompilerOutput* out_result,
    int* out_has_result
) {
    if (!compiler || !compiler->vtable || !compiler->impl || !out_result || !out_has_result) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    return compiler->vtable->poll(compiler->impl, out_result, out_has_result);
}

StackPtxCompilerResult stack_ptx_compiler_nng_connected_server_count(
    StackPtxCompilerHandle* compiler,
    size_t* out_num_servers
) {
    if (out_num_servers) {
        *out_num_servers = 0;
    }
    if (!compiler || !out_num_servers) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (compiler->backend != STACK_PTX_COMPILER_BACKEND_NNG) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
#ifdef STACK_PTX_COMPILER_ENABLE_NNG
    return stack_ptx_compiler_backend_nng_connected_server_count(compiler->impl, out_num_servers);
#else
    _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
#endif
}

StackPtxCompilerResult stack_ptx_compiler_nng_connected_server_addr(
    StackPtxCompilerHandle* compiler,
    size_t server_idx,
    const char** out_addr
) {
    if (out_addr) {
        *out_addr = NULL;
    }
    if (!compiler || !out_addr) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (compiler->backend != STACK_PTX_COMPILER_BACKEND_NNG) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
#ifdef STACK_PTX_COMPILER_ENABLE_NNG
    return stack_ptx_compiler_backend_nng_connected_server_addr(compiler->impl, server_idx, out_addr);
#else
    _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
#endif
}

StackPtxCompilerResult stack_ptx_compiler_nng_connected_server_conn_count(
    StackPtxCompilerHandle* compiler,
    size_t server_idx,
    size_t* out_num_conns
) {
    if (out_num_conns) {
        *out_num_conns = 0;
    }
    if (!compiler || !out_num_conns) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
    if (compiler->backend != STACK_PTX_COMPILER_BACKEND_NNG) {
        _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
    }
#ifdef STACK_PTX_COMPILER_ENABLE_NNG
    return stack_ptx_compiler_backend_nng_connected_server_conn_count(compiler->impl, server_idx, out_num_conns);
#else
    _STACK_PTX_COMPILER_ERROR(STACK_PTX_COMPILER_ERROR_INVALID_VALUE);
#endif
}
