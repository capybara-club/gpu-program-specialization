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
#include "stack_ptx_nng_wire.h"

#include <string.h>

#define STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#include <stack_ptx_inject_serialize.h>

enum { STACK_PTX_NNG_HEADER_BYTES = 4u + 2u + 2u + 4u };

static StackPtxNngWireResult stack_ptx_nng_write_bytes(
    uint8_t** p,
    const uint8_t* end,
    const void* src,
    size_t n
) {
    if (!p || !*p || !end) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if ((size_t)(end - *p) < n) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    if (n > 0 && !src) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (n > 0) {
        memcpy(*p, src, n);
        *p += n;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

static StackPtxNngWireResult stack_ptx_nng_read_bytes(
    const uint8_t** p,
    const uint8_t* end,
    void* dst,
    size_t n
) {
    if (!p || !*p || !end) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if ((size_t)(end - *p) < n) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    if (n > 0 && !dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (n > 0) {
        memcpy(dst, *p, n);
        *p += n;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

static StackPtxNngWireResult stack_ptx_nng_write_u16(uint8_t** p, const uint8_t* end, uint16_t v) {
    return stack_ptx_nng_write_bytes(p, end, &v, sizeof(v));
}
static StackPtxNngWireResult stack_ptx_nng_write_u32(uint8_t** p, const uint8_t* end, uint32_t v) {
    return stack_ptx_nng_write_bytes(p, end, &v, sizeof(v));
}
static StackPtxNngWireResult stack_ptx_nng_write_u64(uint8_t** p, const uint8_t* end, uint64_t v) {
    return stack_ptx_nng_write_bytes(p, end, &v, sizeof(v));
}
static StackPtxNngWireResult stack_ptx_nng_write_double(uint8_t** p, const uint8_t* end, double v) {
    return stack_ptx_nng_write_bytes(p, end, &v, sizeof(v));
}

static StackPtxNngWireResult stack_ptx_nng_read_u16(const uint8_t** p, const uint8_t* end, uint16_t* out) {
    return stack_ptx_nng_read_bytes(p, end, out, sizeof(*out));
}
static StackPtxNngWireResult stack_ptx_nng_read_u32(const uint8_t** p, const uint8_t* end, uint32_t* out) {
    return stack_ptx_nng_read_bytes(p, end, out, sizeof(*out));
}
static StackPtxNngWireResult stack_ptx_nng_read_u64(const uint8_t** p, const uint8_t* end, uint64_t* out) {
    return stack_ptx_nng_read_bytes(p, end, out, sizeof(*out));
}
static StackPtxNngWireResult stack_ptx_nng_read_double(const uint8_t** p, const uint8_t* end, double* out) {
    return stack_ptx_nng_read_bytes(p, end, out, sizeof(*out));
}

size_t stack_ptx_nng_init_request_wire_size(size_t compiler_state_wire_size) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + (sizeof(uint16_t) * 3u) + sizeof(uint64_t) + compiler_state_wire_size;
}

StackPtxNngWireResult stack_ptx_nng_write_init_request(
    uint8_t* dst,
    size_t dst_size,
    const uint8_t* compiler_state_wire,
    size_t compiler_state_wire_size
) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_init_request_wire_size(compiler_state_wire_size)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_INIT_REQUEST);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_COMPILER_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_COMPILER_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_COMPILER_VERSION_PATCH);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, (uint64_t)compiler_state_wire_size);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_bytes(&p, end, compiler_state_wire, compiler_state_wire_size);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_init_request(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngInitRequestView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < STACK_PTX_NNG_HEADER_BYTES + sizeof(uint64_t)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_INIT_REQUEST) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    uint16_t compiler_version_major = 0;
    uint16_t compiler_version_minor = 0;
    uint16_t compiler_version_patch = 0;
    rc = stack_ptx_nng_read_u16(&p, end, &compiler_version_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &compiler_version_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &compiler_version_patch);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t compiler_state_size_u64 = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &compiler_state_size_u64);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (compiler_state_size_u64 > (uint64_t)(end - p)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    size_t compiler_state_wire_size = (size_t)compiler_state_size_u64;
    const uint8_t* compiler_state_wire = p;
    p += compiler_state_wire_size;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->compiler_version_major = compiler_version_major;
    out_view->compiler_version_minor = compiler_version_minor;
    out_view->compiler_version_patch = compiler_version_patch;
    out_view->compiler_state_wire = compiler_state_wire;
    out_view->compiler_state_wire_size = compiler_state_wire_size;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_init_response_wire_size(void) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + sizeof(uint64_t) + sizeof(int32_t) +
        (sizeof(uint16_t) * 3u) + (sizeof(uint64_t) * 3u);
}

StackPtxNngWireResult stack_ptx_nng_write_init_response(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    StackPtxCompilerResult status,
    uint64_t server_capabilities,
    uint64_t server_queue_slots,
    uint64_t num_injects
) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_init_response_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_INIT_RESPONSE);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = (int32_t)status;
    rc = stack_ptx_nng_write_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_COMPILER_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_COMPILER_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_COMPILER_VERSION_PATCH);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, server_capabilities);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, server_queue_slots);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, num_injects);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_init_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngInitResponseView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < stack_ptx_nng_init_response_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_INIT_RESPONSE) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    uint64_t session_id = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = 0;
    rc = stack_ptx_nng_read_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint16_t server_version_major = 0;
    uint16_t server_version_minor = 0;
    uint16_t server_version_patch = 0;
    rc = stack_ptx_nng_read_u16(&p, end, &server_version_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &server_version_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &server_version_patch);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t server_capabilities = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &server_capabilities);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t server_queue_slots = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &server_queue_slots);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    uint64_t num_injects = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &num_injects);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->session_id = session_id;
    out_view->status = (StackPtxCompilerResult)status_i32;
    out_view->server_version_major = server_version_major;
    out_view->server_version_minor = server_version_minor;
    out_view->server_version_patch = server_version_patch;
    out_view->server_capabilities = server_capabilities;
    out_view->server_queue_slots = server_queue_slots;
    out_view->num_injects = num_injects;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_evict_request_wire_size(void) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + sizeof(uint64_t);
}

StackPtxNngWireResult stack_ptx_nng_write_evict_request(uint8_t* dst, size_t dst_size, uint64_t session_id) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_evict_request_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_EVICT_REQUEST);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_evict_request(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngEvictRequestView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < stack_ptx_nng_evict_request_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_EVICT_REQUEST) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    uint64_t session_id = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->session_id = session_id;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_evict_response_wire_size(void) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + sizeof(uint64_t) + sizeof(int32_t);
}

StackPtxNngWireResult stack_ptx_nng_write_evict_response(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    StackPtxCompilerResult status
) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_evict_response_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_EVICT_RESPONSE);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = (int32_t)status;
    rc = stack_ptx_nng_write_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_evict_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngEvictResponseView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < stack_ptx_nng_evict_response_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_EVICT_RESPONSE) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    uint64_t session_id = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = 0;
    rc = stack_ptx_nng_read_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->session_id = session_id;
    out_view->status = (StackPtxCompilerResult)status_i32;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_status_request_wire_size(void) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES;
}

StackPtxNngWireResult stack_ptx_nng_write_status_request(uint8_t* dst, size_t dst_size) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_status_request_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_STATUS_REQUEST);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_status_request(const uint8_t* src, size_t src_size) {
    if (!src) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < stack_ptx_nng_status_request_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_STATUS_REQUEST) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_status_response_wire_size(void) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + sizeof(int32_t) + (sizeof(uint64_t) * 4u) + (sizeof(double) * 2u) +
        sizeof(int32_t);
}

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
) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_status_response_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_STATUS_RESPONSE);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = (int32_t)status;
    rc = stack_ptx_nng_write_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, server_requested_capabilities);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, num_sessions);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, total_compiles);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_double(&p, end, uptime_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, total_in_flight);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t last_error_i32 = (int32_t)last_error;
    rc = stack_ptx_nng_write_bytes(&p, end, &last_error_i32, sizeof(last_error_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_double(&p, end, last_error_uptime_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_status_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngStatusResponseView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < stack_ptx_nng_status_response_wire_size()) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_STATUS_RESPONSE) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    int32_t status_i32 = 0;
    rc = stack_ptx_nng_read_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t server_requested_capabilities = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &server_requested_capabilities);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t num_sessions = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &num_sessions);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t total_compiles = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &total_compiles);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    double uptime_ms = 0.0;
    rc = stack_ptx_nng_read_double(&p, end, &uptime_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t total_in_flight = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &total_in_flight);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t last_error_i32 = 0;
    rc = stack_ptx_nng_read_bytes(&p, end, &last_error_i32, sizeof(last_error_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    double last_error_uptime_ms = 0.0;
    rc = stack_ptx_nng_read_double(&p, end, &last_error_uptime_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->status = (StackPtxCompilerResult)status_i32;
    out_view->server_requested_capabilities = server_requested_capabilities;
    out_view->num_sessions = num_sessions;
    out_view->total_compiles = total_compiles;
    out_view->uptime_ms = uptime_ms;
    out_view->total_in_flight = total_in_flight;
    out_view->last_error = (StackPtxCompilerResult)last_error_i32;
    out_view->last_error_uptime_ms = last_error_uptime_ms;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_request_wire_size(size_t instructions_wire_size) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + (sizeof(uint64_t) * 4u) + instructions_wire_size;
}

StackPtxNngWireResult stack_ptx_nng_write_request(
    uint8_t* dst,
    size_t dst_size,
    uint64_t session_id,
    uint64_t job_id,
    uint64_t module_idx,
    const uint8_t* instructions_wire,
    size_t instructions_wire_size
) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_request_wire_size(instructions_wire_size)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_COMPILE_REQUEST);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, job_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, module_idx);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, (uint64_t)instructions_wire_size);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_bytes(&p, end, instructions_wire, instructions_wire_size);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_request(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngCompileRequestView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < STACK_PTX_NNG_HEADER_BYTES + (sizeof(uint64_t) * 4u)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_COMPILE_REQUEST) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    uint64_t session_id = 0;
    uint64_t job_id = 0;
    uint64_t module_idx = 0;
    uint64_t instructions_size_u64 = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u64(&p, end, &job_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u64(&p, end, &module_idx);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u64(&p, end, &instructions_size_u64);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (instructions_size_u64 > (uint64_t)(end - p)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    size_t instructions_wire_size = (size_t)instructions_size_u64;
    const uint8_t* instructions_wire = p;
    p += instructions_wire_size;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->session_id = session_id;
    out_view->job_id = job_id;
    out_view->module_idx = module_idx;
    out_view->instructions_wire = instructions_wire;
    out_view->instructions_wire_size = instructions_wire_size;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

size_t stack_ptx_nng_response_wire_size(size_t cubin_size) {
    return (size_t)STACK_PTX_NNG_HEADER_BYTES + (sizeof(uint64_t) * 4u) + sizeof(int32_t) + (sizeof(double) * 2u) +
        cubin_size;
}

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
) {
    if (!dst) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (dst_size < stack_ptx_nng_response_wire_size(cubin_size)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    if (cubin_size > 0 && !cubin) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }

    uint8_t* p = dst;
    const uint8_t* end = dst + dst_size;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_write_u32(&p, end, STACK_PTX_NNG_WIRE_MAGIC);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MAJOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u16(&p, end, STACK_PTX_NNG_WIRE_VERSION_MINOR);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u32(&p, end, (uint32_t)STACK_PTX_NNG_MSG_COMPILE_RESPONSE);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, job_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_u64(&p, end, module_idx);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = (int32_t)status;
    rc = stack_ptx_nng_write_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_double(&p, end, compile_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_double(&p, end, total_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    rc = stack_ptx_nng_write_u64(&p, end, (uint64_t)cubin_size);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_write_bytes(&p, end, cubin, cubin_size);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }
    return STACK_PTX_NNG_WIRE_SUCCESS;
}

StackPtxNngWireResult stack_ptx_nng_read_response(
    const uint8_t* src,
    size_t src_size,
    StackPtxNngCompileResponseView* out_view
) {
    if (!src || !out_view) {
        return STACK_PTX_NNG_WIRE_ERROR_INVALID;
    }
    if (src_size < STACK_PTX_NNG_HEADER_BYTES + (sizeof(uint64_t) * 4u) + sizeof(int32_t) + (sizeof(double) * 2u)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    const uint8_t* p = src;
    const uint8_t* end = src + src_size;

    uint32_t magic = 0;
    uint16_t ver_major = 0;
    uint16_t ver_minor = 0;
    uint32_t msg_type = 0;

    StackPtxNngWireResult rc = STACK_PTX_NNG_WIRE_SUCCESS;
    rc = stack_ptx_nng_read_u32(&p, end, &magic);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (magic != STACK_PTX_NNG_WIRE_MAGIC) return STACK_PTX_NNG_WIRE_ERROR_MAGIC;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_major);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u16(&p, end, &ver_minor);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (ver_major != STACK_PTX_NNG_WIRE_VERSION_MAJOR || ver_minor != STACK_PTX_NNG_WIRE_VERSION_MINOR) {
        return STACK_PTX_NNG_WIRE_ERROR_VERSION;
    }
    rc = stack_ptx_nng_read_u32(&p, end, &msg_type);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (msg_type != (uint32_t)STACK_PTX_NNG_MSG_COMPILE_RESPONSE) {
        return STACK_PTX_NNG_WIRE_ERROR_TYPE;
    }

    uint64_t session_id = 0;
    uint64_t job_id = 0;
    uint64_t module_idx = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &session_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u64(&p, end, &job_id);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    rc = stack_ptx_nng_read_u64(&p, end, &module_idx);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    int32_t status_i32 = 0;
    rc = stack_ptx_nng_read_bytes(&p, end, &status_i32, sizeof(status_i32));
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    double compile_ms = 0.0;
    rc = stack_ptx_nng_read_double(&p, end, &compile_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    double total_ms = 0.0;
    rc = stack_ptx_nng_read_double(&p, end, &total_ms);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;

    uint64_t cubin_size_u64 = 0;
    rc = stack_ptx_nng_read_u64(&p, end, &cubin_size_u64);
    if (rc != STACK_PTX_NNG_WIRE_SUCCESS) return rc;
    if (cubin_size_u64 > (uint64_t)(end - p)) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    size_t cubin_size = (size_t)cubin_size_u64;
    const uint8_t* cubin = p;
    p += cubin_size;
    if (p != end) {
        return STACK_PTX_NNG_WIRE_ERROR_SIZE;
    }

    out_view->session_id = session_id;
    out_view->job_id = job_id;
    out_view->module_idx = module_idx;
    out_view->status = (StackPtxCompilerResult)status_i32;
    out_view->compile_ms = compile_ms;
    out_view->total_ms = total_ms;
    out_view->cubin = cubin;
    out_view->cubin_size = cubin_size;
    return STACK_PTX_NNG_WIRE_SUCCESS;
}
