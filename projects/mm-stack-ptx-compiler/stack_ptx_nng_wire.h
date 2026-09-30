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

#include <stddef.h>
#include <stdint.h>

#include "stack_ptx_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define STACK_PTX_NNG_WIRE_MAGIC ((uint32_t)0x58545053u) /* 'SPTX' */
#define STACK_PTX_NNG_WIRE_VERSION_MAJOR ((uint16_t)2u)
#define STACK_PTX_NNG_WIRE_VERSION_MINOR ((uint16_t)0u)

typedef enum {
    STACK_PTX_NNG_MSG_COMPILE_REQUEST = 1,
    STACK_PTX_NNG_MSG_COMPILE_RESPONSE = 2,
    STACK_PTX_NNG_MSG_INIT_REQUEST = 3,
    STACK_PTX_NNG_MSG_INIT_RESPONSE = 4,
    STACK_PTX_NNG_MSG_EVICT_REQUEST = 5,
    STACK_PTX_NNG_MSG_EVICT_RESPONSE = 6,
    STACK_PTX_NNG_MSG_STATUS_REQUEST = 7,
    STACK_PTX_NNG_MSG_STATUS_RESPONSE = 8,
} StackPtxNngMsgType;

typedef enum {
    STACK_PTX_NNG_WIRE_SUCCESS = 0,
    STACK_PTX_NNG_WIRE_ERROR_INVALID = 1,
    STACK_PTX_NNG_WIRE_ERROR_SIZE = 2,
    STACK_PTX_NNG_WIRE_ERROR_VERSION = 3,
    STACK_PTX_NNG_WIRE_ERROR_MAGIC = 4,
    STACK_PTX_NNG_WIRE_ERROR_TYPE = 5,
} StackPtxNngWireResult;

typedef struct {
    uint64_t job_id;
    uint64_t module_idx;
    const uint8_t* instructions_wire;
    size_t instructions_wire_size;
    uint64_t session_id;
} StackPtxNngCompileRequestView;

typedef struct {
    uint64_t job_id;
    uint64_t module_idx;
    uint64_t session_id;
    StackPtxCompilerResult status;
    double compile_ms;
    double total_ms;
    const uint8_t* cubin;
    size_t cubin_size;
} StackPtxNngCompileResponseView;

typedef struct {
    uint16_t compiler_version_major;
    uint16_t compiler_version_minor;
    uint16_t compiler_version_patch;
    const uint8_t* compiler_state_wire;
    size_t compiler_state_wire_size;
} StackPtxNngInitRequestView;

typedef struct {
    uint64_t session_id;
    StackPtxCompilerResult status;
    uint16_t server_version_major;
    uint16_t server_version_minor;
    uint16_t server_version_patch;
    uint64_t server_capabilities;
    uint64_t server_queue_slots;
    uint64_t num_injects;
} StackPtxNngInitResponseView;

typedef struct {
    uint64_t session_id;
} StackPtxNngEvictRequestView;

typedef struct {
    uint64_t session_id;
    StackPtxCompilerResult status;
} StackPtxNngEvictResponseView;

typedef struct {
    StackPtxCompilerResult status;
    uint64_t server_requested_capabilities;
    uint64_t num_sessions;
    uint64_t total_compiles;
    double uptime_ms;
    uint64_t total_in_flight;
    StackPtxCompilerResult last_error;
    double last_error_uptime_ms;
} StackPtxNngStatusResponseView;

size_t stack_ptx_nng_init_request_wire_size(size_t compiler_state_wire_size);

StackPtxNngWireResult stack_ptx_nng_write_init_request(
    uint8_t* dst,
    size_t dst_size,
    const uint8_t* compiler_state_wire,
    size_t compiler_state_wire_size
);

StackPtxNngWireResult stack_ptx_nng_read_init_request(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngInitRequestView* out_view
);

size_t stack_ptx_nng_init_response_wire_size(void);

StackPtxNngWireResult stack_ptx_nng_write_init_response(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    StackPtxCompilerResult status,
    uint64_t server_capabilities,
    uint64_t server_queue_slots,
    uint64_t num_injects
);

StackPtxNngWireResult stack_ptx_nng_read_init_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngInitResponseView* out_view
);

size_t stack_ptx_nng_evict_request_wire_size(void);

StackPtxNngWireResult stack_ptx_nng_write_evict_request(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id
);

StackPtxNngWireResult stack_ptx_nng_read_evict_request(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngEvictRequestView* out_view
);

size_t stack_ptx_nng_evict_response_wire_size(void);

StackPtxNngWireResult stack_ptx_nng_write_evict_response(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    StackPtxCompilerResult status
);

StackPtxNngWireResult stack_ptx_nng_read_evict_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngEvictResponseView* out_view
);

size_t stack_ptx_nng_status_request_wire_size(void);

StackPtxNngWireResult stack_ptx_nng_write_status_request(uint8_t* dst, size_t dst_size);

StackPtxNngWireResult stack_ptx_nng_read_status_request(const uint8_t* src, size_t src_size);

size_t stack_ptx_nng_status_response_wire_size(void);

StackPtxNngWireResult stack_ptx_nng_write_status_response(
    uint8_t* dst,
    size_t dst_size,
    StackPtxCompilerResult status,
    uint64_t server_requested_capabilities,
    uint64_t num_sessions,
    uint64_t total_compiles,
    double uptime_ms,
    uint64_t total_in_flight,
    StackPtxCompilerResult last_error,
    double last_error_uptime_ms
);

StackPtxNngWireResult stack_ptx_nng_read_status_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngStatusResponseView* out_view
);

size_t stack_ptx_nng_request_wire_size(size_t instructions_wire_size);

StackPtxNngWireResult stack_ptx_nng_write_request(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    uint64_t job_id,
    uint64_t module_idx,
    const uint8_t* instructions_wire,
    size_t instructions_wire_size
);

StackPtxNngWireResult stack_ptx_nng_read_request(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngCompileRequestView* out_view
);

size_t stack_ptx_nng_response_wire_size(size_t cubin_size);

StackPtxNngWireResult stack_ptx_nng_write_response(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    uint64_t job_id,
    uint64_t module_idx,
    StackPtxCompilerResult status,
    double compile_ms,
    double total_ms,
    const uint8_t* cubin,
    size_t cubin_size
);

StackPtxNngWireResult stack_ptx_nng_read_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngCompileResponseView* out_view
);

#ifdef __cplusplus
} /* extern "C" */
#endif
