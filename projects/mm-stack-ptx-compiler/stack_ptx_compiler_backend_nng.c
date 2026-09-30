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

#include "stack_ptx_nng_wire.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <nng/nng.h>
#include <nng/protocol/reqrep0/req.h>
#include <ptx_inject.h>
#include <stdio.h>
#include <stack_ptx.h>
#include <stack_ptx_inject_serialize.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void stack_ptx_compiler_sleep_ms(int ms) {
    if (ms <= 0) {
        return;
    }
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static uint64_t stack_ptx_compiler_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000u);
}

static const char* kDefaultNngAddr = "tcp://127.0.0.1:5555";
static const char* kDefaultInjectPrefix = "func_";
static const char* kDefaultInputRegisterName = "x";
static const char* kDefaultOutputRegisterPrefix = "y";

static const StackPtxCompilerInfo kDefaultCompilerInfo = {
    512,
    512,
    256,
    4,
    16
};

static int stack_ptx_compiler_version_matches(uint16_t major, uint16_t minor, uint16_t patch) {
    return major == STACK_PTX_COMPILER_VERSION_MAJOR &&
        minor == STACK_PTX_COMPILER_VERSION_MINOR &&
        patch == STACK_PTX_COMPILER_VERSION_PATCH;
}

typedef struct StackPtxCompilerNng StackPtxCompilerNng;

typedef struct {
    char* addr;
    uint64_t session_id;
    uint64_t server_capabilities;
    uint64_t server_queue_slots;
    size_t num_conns;
    uint64_t local_in_flight;

    nng_socket status_sock;
    int status_sock_open;
    int status_in_flight;
    uint64_t status_last_send_ms;
    uint64_t status_last_recv_ms;
    int status_valid;
    StackPtxNngStatusResponseView status;
} StackPtxCompilerNngServer;

typedef struct {
    nng_socket sock;
    int sock_open;
    int in_flight;
    uint64_t in_flight_job_id;
    uint64_t in_flight_module_idx;
    size_t server_idx;
} StackPtxCompilerNngConn;

struct StackPtxCompilerNng {
    StackPtxCompilerNngServer* servers;
    size_t num_servers;
    StackPtxCompilerNngConn* conns;
    size_t num_conns;
    size_t rr_conn_idx;

    uint8_t* compiler_state_wire;
    size_t compiler_state_wire_size;

    size_t num_injects;
    size_t individuals_per_module;
    const StackPtxInstruction** instruction_stubs;
};

static int stack_ptx_compiler_str_starts_with(const char* s, const char* prefix) {
    if (!s || !prefix) {
        return 0;
    }
    return strncmp(s, prefix, strlen(prefix)) == 0;
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

static StackPtxCompilerResult stack_ptx_compiler_build_instruction_stubs(
    StackPtxCompilerNng* compiler,
    const StackPtxCompilerWork* work,
    const StackPtxInstruction** instruction_stubs_out
) {
    if (!compiler || !work || !instruction_stubs_out) {
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
        const size_t population_idx = work->module_idx * compiler->individuals_per_module + i;
        const size_t instruction_idx = population_idx * work->gene_length;
        instruction_stubs_out[i] = &work->population[instruction_idx];
    }

    return STACK_PTX_COMPILER_SUCCESS;
}

static const char* stack_ptx_compiler_nng_addr(const StackPtxCompilerHandleConfig* config) {
    if (config && config->nng_addr && config->nng_addr[0]) {
        return config->nng_addr;
    }
    const char* env = getenv("STACK_PTX_NNG_ADDR");
    if (!env || !env[0]) {
        return kDefaultNngAddr;
    }
    return env;
}

static int stack_ptx_compiler_is_addr_sep(char c) {
    return c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static void stack_ptx_compiler_free_addrs(char** addrs, size_t n) {
    if (!addrs) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        free(addrs[i]);
    }
    free(addrs);
}

static StackPtxCompilerResult stack_ptx_compiler_split_addrs(const char* s, char*** out_addrs, size_t* out_n) {
    if (out_addrs) {
        *out_addrs = NULL;
    }
    if (out_n) {
        *out_n = 0;
    }
    if (!out_addrs || !out_n) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (!s || !s[0]) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    size_t count = 0;
    const char* p = s;
    for (;;) {
        while (*p && stack_ptx_compiler_is_addr_sep(*p)) {
            ++p;
        }
        if (!*p) {
            break;
        }
        count += 1;
        while (*p && !stack_ptx_compiler_is_addr_sep(*p)) {
            ++p;
        }
    }
    if (count == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    char** addrs = (char**)calloc(count, sizeof(*addrs));
    if (!addrs) {
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    size_t written = 0;
    p = s;
    while (*p && written < count) {
        while (*p && stack_ptx_compiler_is_addr_sep(*p)) {
            ++p;
        }
        if (!*p) {
            break;
        }
        const char* start = p;
        while (*p && !stack_ptx_compiler_is_addr_sep(*p)) {
            ++p;
        }
        const size_t len = (size_t)(p - start);
        int has_scheme = 0;
        if (len >= 3) {
            for (size_t j = 0; (j + 2) < len; ++j) {
                if (start[j] == ':' && start[j + 1] == '/' && start[j + 2] == '/') {
                    has_scheme = 1;
                    break;
                }
            }
        }

        const char* prefix = has_scheme ? "" : "tcp://";
        const size_t prefix_len = has_scheme ? 0 : 6;

        char* addr = (char*)malloc(prefix_len + len + 1);
        if (!addr) {
            stack_ptx_compiler_free_addrs(addrs, written);
            return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
        }
        memcpy(addr, prefix, prefix_len);
        memcpy(addr + prefix_len, start, len);
        addr[prefix_len + len] = '\0';
        addrs[written++] = addr;
    }

    if (written == 0) {
        stack_ptx_compiler_free_addrs(addrs, written);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    *out_addrs = addrs;
    *out_n = written;
    return STACK_PTX_COMPILER_SUCCESS;
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
    // CUDA toolchains can drop support for older SM versions; pick something modern-ish by default.
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

static StackPtxCompilerResult stack_ptx_compiler_nng_destroy(StackPtxCompilerNng* compiler);

static StackPtxCompilerResult stack_ptx_compiler_nng_send_init(
    nng_socket sock,
    const uint8_t* compiler_state_wire,
    size_t compiler_state_wire_size,
    StackPtxNngInitResponseView* out_resp
) {
    if (out_resp) {
        memset(out_resp, 0, sizeof(*out_resp));
    }
    if (!compiler_state_wire || compiler_state_wire_size == 0 || !out_resp) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const size_t init_wire_size = stack_ptx_nng_init_request_wire_size(compiler_state_wire_size);
    nng_msg* msg = NULL;
    int rv = nng_msg_alloc(&msg, init_wire_size);
    if (rv != 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    StackPtxNngWireResult wrc = stack_ptx_nng_write_init_request(
        (uint8_t*)nng_msg_body(msg),
        init_wire_size,
        compiler_state_wire,
        compiler_state_wire_size
    );
    if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
        nng_msg_free(msg);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    rv = nng_sendmsg(sock, msg, 0);
    if (rv != 0) {
        nng_msg_free(msg);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    nng_msg* resp_msg = NULL;
    rv = nng_recvmsg(sock, &resp_msg, 0);
    if (rv != 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    wrc = stack_ptx_nng_read_init_response((const uint8_t*)nng_msg_body(resp_msg), nng_msg_len(resp_msg), out_resp);
    nng_msg_free(resp_msg);
    resp_msg = NULL;
    if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    if (!stack_ptx_compiler_version_matches(out_resp->server_version_major,
            out_resp->server_version_minor,
            out_resp->server_version_patch)) {
        return STACK_PTX_COMPILER_ERROR_VERSION;
    }
    if (out_resp->status != STACK_PTX_COMPILER_SUCCESS || out_resp->session_id == 0) {
        return out_resp->status != STACK_PTX_COMPILER_SUCCESS ? out_resp->status : STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    return STACK_PTX_COMPILER_SUCCESS;
}

static void stack_ptx_compiler_nng_best_effort_evict(nng_socket sock, uint64_t session_id) {
    if (session_id == 0) {
        return;
    }
    const size_t req_size = stack_ptx_nng_evict_request_wire_size();
    nng_msg* msg = NULL;
    if (nng_msg_alloc(&msg, req_size) != 0) {
        return;
    }

    StackPtxNngWireResult wrc = stack_ptx_nng_write_evict_request((uint8_t*)nng_msg_body(msg), req_size, session_id);
    if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
        nng_msg_free(msg);
        return;
    }

    int rv = nng_sendmsg(sock, msg, 0);
    if (rv != 0) {
        nng_msg_free(msg);
        return;
    }

    for (int i = 0; i < 10; ++i) {
        nng_msg* resp = NULL;
        rv = nng_recvmsg(sock, &resp, NNG_FLAG_NONBLOCK);
        if (rv == NNG_EAGAIN) {
            stack_ptx_compiler_sleep_ms(5);
            continue;
        }
        if (rv == 0 && resp) {
            nng_msg_free(resp);
        }
        break;
    }
}

enum {
    STACK_PTX_COMPILER_NNG_STATUS_POLL_INTERVAL_MS = 250u,
    STACK_PTX_COMPILER_NNG_STATUS_STALE_MS = 1000u,
};

static void stack_ptx_compiler_nng_conn_mark_idle(StackPtxCompilerNng* compiler, StackPtxCompilerNngConn* conn) {
    if (!compiler || !conn) {
        return;
    }
    if (!conn->in_flight) {
        conn->in_flight = 0;
        return;
    }
    if (compiler->servers && conn->server_idx < compiler->num_servers) {
        StackPtxCompilerNngServer* server = &compiler->servers[conn->server_idx];
        if (server->local_in_flight > 0) {
            server->local_in_flight -= 1;
        }
    }
    conn->in_flight = 0;
}

static void stack_ptx_compiler_nng_status_tick(StackPtxCompilerNng* compiler) {
    if (!compiler || !compiler->servers || compiler->num_servers == 0) {
        return;
    }
    const uint64_t now_ms = stack_ptx_compiler_now_ms();
    const size_t req_size = stack_ptx_nng_status_request_wire_size();

    for (size_t i = 0; i < compiler->num_servers; ++i) {
        StackPtxCompilerNngServer* server = &compiler->servers[i];
        if (!server->status_sock_open) {
            continue;
        }

        if (server->status_in_flight) {
            nng_msg* resp = NULL;
            const int rv = nng_recvmsg(server->status_sock, &resp, NNG_FLAG_NONBLOCK);
            if (rv == NNG_EAGAIN) {
                continue;
            }
            server->status_in_flight = 0;
            if (rv != 0 || !resp) {
                server->status_valid = 0;
                continue;
            }

            StackPtxNngStatusResponseView view;
            memset(&view, 0, sizeof(view));
            const StackPtxNngWireResult wrc =
                stack_ptx_nng_read_status_response((const uint8_t*)nng_msg_body(resp), nng_msg_len(resp), &view);
            nng_msg_free(resp);
            if (wrc != STACK_PTX_NNG_WIRE_SUCCESS || view.status != STACK_PTX_COMPILER_SUCCESS) {
                server->status_valid = 0;
                continue;
            }

            server->status = view;
            server->status_last_recv_ms = now_ms;
            server->status_valid = 1;
            continue;
        }

        if ((now_ms - server->status_last_send_ms) < (uint64_t)STACK_PTX_COMPILER_NNG_STATUS_POLL_INTERVAL_MS) {
            continue;
        }

        nng_msg* msg = NULL;
        int rv = nng_msg_alloc(&msg, req_size);
        if (rv != 0) {
            server->status_last_send_ms = now_ms;
            continue;
        }

        const StackPtxNngWireResult wrc = stack_ptx_nng_write_status_request((uint8_t*)nng_msg_body(msg), req_size);
        if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
            nng_msg_free(msg);
            server->status_last_send_ms = now_ms;
            server->status_valid = 0;
            continue;
        }

        rv = nng_sendmsg(server->status_sock, msg, NNG_FLAG_NONBLOCK);
        if (rv != 0) {
            nng_msg_free(msg);
            server->status_last_send_ms = now_ms;
            server->status_valid = 0;
            continue;
        }

        server->status_last_send_ms = now_ms;
        server->status_in_flight = 1;
    }
}

static size_t stack_ptx_compiler_nng_choose_conn(StackPtxCompilerNng* compiler) {
    if (!compiler || !compiler->conns || compiler->num_conns == 0 || !compiler->servers || compiler->num_servers == 0) {
        return SIZE_MAX;
    }

    const uint64_t now_ms = stack_ptx_compiler_now_ms();
    uint64_t best_load = UINT64_MAX;
    size_t best_conn = SIZE_MAX;

    for (size_t i = 0; i < compiler->num_conns; ++i) {
        const size_t idx = (compiler->rr_conn_idx + i) % compiler->num_conns;
        StackPtxCompilerNngConn* conn = &compiler->conns[idx];
        if (!conn->sock_open || conn->in_flight) {
            continue;
        }
        if (conn->server_idx >= compiler->num_servers) {
            continue;
        }

        const StackPtxCompilerNngServer* server = &compiler->servers[conn->server_idx];
        uint64_t load = server->local_in_flight;
        if (server->status_valid && (now_ms - server->status_last_recv_ms) <= (uint64_t)STACK_PTX_COMPILER_NNG_STATUS_STALE_MS) {
            load = server->status.total_in_flight;
        }

        if (load < best_load) {
            best_load = load;
            best_conn = idx;
            if (best_load == 0) {
                break;
            }
        }
    }

    return best_conn;
}

static StackPtxCompilerResult stack_ptx_compiler_nng_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerNng** out_compiler,
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
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    *out_compiler = NULL;

    if (!config->kernel_ptx || !config->kernel_ptx[0]) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (!config->stack_info) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (config->kernel_num_kernels == 0 || config->kernel_groups_per_kernel == 0 ||
        config->execution_limit == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    StackPtxCompilerNng* compiler = (StackPtxCompilerNng*)calloc(1, sizeof(*compiler));
    if (!compiler) {
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    const char* inject_prefix = config->inject_prefix ? config->inject_prefix : kDefaultInjectPrefix;
    const char* input_register_name = config->input_register_name ? config->input_register_name : kDefaultInputRegisterName;
    const char* output_register_prefix =
        config->output_register_prefix ? config->output_register_prefix : kDefaultOutputRegisterPrefix;
    const StackPtxCompilerInfo* compiler_info = config->compiler_info ? config->compiler_info : &kDefaultCompilerInfo;

    compiler->individuals_per_module =
        config->kernel_num_kernels * config->kernel_groups_per_kernel;
    if (compiler->individuals_per_module == 0) {
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    PtxInjectHandle ptx_inject = NULL;
    if (ptx_inject_create(&ptx_inject, config->kernel_ptx) != PTX_INJECT_SUCCESS) {
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    if (ptx_inject_num_injects(ptx_inject, &compiler->num_injects) != PTX_INJECT_SUCCESS) {
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    if (compiler->num_injects != compiler->individuals_per_module) {
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    compiler->instruction_stubs =
        (const StackPtxInstruction**)calloc(compiler->num_injects, sizeof(*compiler->instruction_stubs));
    if (!compiler->instruction_stubs) {
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    char inject_name[64];
    snprintf(inject_name, sizeof(inject_name), "%s0", inject_prefix);
    size_t inject_idx = 0;
    size_t num_args = 0;
    if (ptx_inject_inject_info_by_name(ptx_inject, inject_name, &inject_idx, &num_args, NULL) != PTX_INJECT_SUCCESS) {
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (num_args < 2) {
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const char* input_prefix = (input_register_name && input_register_name[0]) ? input_register_name : NULL;
    const char* output_prefix = (output_register_prefix && output_register_prefix[0]) ? output_register_prefix : NULL;
    size_t input_count = 0;
    size_t output_count = 0;
    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        const char* var_name = NULL;
        PtxInjectMutType mut_type = PTX_INJECT_MUT_TYPE_IN;
        if (ptx_inject_variable_info_by_index(ptx_inject, inject_idx, arg_idx, &var_name, NULL, &mut_type, NULL, NULL) !=
            PTX_INJECT_SUCCESS) {
            ptx_inject_destroy(ptx_inject);
            stack_ptx_compiler_nng_destroy(compiler);
            return STACK_PTX_COMPILER_ERROR_INTERNAL;
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
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const size_t num_registers = input_count + output_count;
    const size_t num_requests = output_count;

    StackPtxRegister* registers = (StackPtxRegister*)calloc(num_registers, sizeof(*registers));
    size_t* requests = (size_t*)calloc(num_requests, sizeof(*requests));
    const size_t** request_stubs = (const size_t**)calloc(compiler->num_injects, sizeof(*request_stubs));
    size_t* request_stub_sizes = (size_t*)calloc(compiler->num_injects, sizeof(*request_stub_sizes));
    if (!registers || !requests || !request_stubs || !request_stub_sizes) {
        free(registers);
        free(requests);
        free(request_stubs);
        free(request_stub_sizes);
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    size_t input_written = 0;
    size_t output_written = 0;
    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        const char* var_name = NULL;
        const char* reg_name = NULL;
        PtxInjectMutType mut_type = PTX_INJECT_MUT_TYPE_IN;
        const char* data_type = NULL;
        if (ptx_inject_variable_info_by_index(ptx_inject, inject_idx, arg_idx, &var_name, &reg_name, &mut_type, NULL,
                &data_type) != PTX_INJECT_SUCCESS) {
            free(registers);
            free(requests);
            free(request_stubs);
            free(request_stub_sizes);
            ptx_inject_destroy(ptx_inject);
            stack_ptx_compiler_nng_destroy(compiler);
            return STACK_PTX_COMPILER_ERROR_INTERNAL;
        }

        size_t stack_idx = 0;
        if (stack_ptx_compiler_stack_idx_from_data_type(config->stack_info, data_type, &stack_idx) != 0) {
            free(registers);
            free(requests);
            free(request_stubs);
            free(request_stub_sizes);
            ptx_inject_destroy(ptx_inject);
            stack_ptx_compiler_nng_destroy(compiler);
            return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
        }

        if (mut_type == PTX_INJECT_MUT_TYPE_IN) {
            if (input_prefix && !stack_ptx_compiler_str_starts_with(var_name, input_prefix)) {
                continue;
            }
            if (input_written >= input_count) {
                free(registers);
                free(requests);
                free(request_stubs);
                free(request_stub_sizes);
                ptx_inject_destroy(ptx_inject);
                stack_ptx_compiler_nng_destroy(compiler);
                return STACK_PTX_COMPILER_ERROR_INTERNAL;
            }
            registers[input_written].stack_idx = stack_idx;
            registers[input_written].name = reg_name;
            input_written += 1;
        } else if (mut_type == PTX_INJECT_MUT_TYPE_OUT || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
            if (output_prefix && !stack_ptx_compiler_str_starts_with(var_name, output_prefix)) {
                continue;
            }
            if (output_written >= output_count) {
                free(registers);
                free(requests);
                free(request_stubs);
                free(request_stub_sizes);
                ptx_inject_destroy(ptx_inject);
                stack_ptx_compiler_nng_destroy(compiler);
                return STACK_PTX_COMPILER_ERROR_INTERNAL;
            }
            const size_t reg_idx = input_count + output_written;
            registers[reg_idx].stack_idx = stack_idx;
            registers[reg_idx].name = reg_name;
            requests[output_written] = reg_idx;
            output_written += 1;
        }
    }
    if (input_written != input_count || output_written != output_count) {
        free(registers);
        free(requests);
        free(request_stubs);
        free(request_stub_sizes);
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    for (size_t i = 0; i < compiler->num_injects; ++i) {
        request_stubs[i] = requests;
        request_stub_sizes[i] = num_requests;
    }

    unsigned int sm_major = 0;
    unsigned int sm_minor = 0;
    if (config->device_capability_major > 0 && config->device_capability_minor <= 9) {
        sm_major = config->device_capability_major;
        sm_minor = config->device_capability_minor;
    } else {
        stack_ptx_compiler_get_sm_fallback(&sm_major, &sm_minor);
    }

    StackPtxExtraInfo extra = {
        .device_capability_major = sm_major,
        .device_capability_minor = sm_minor,
        .execution_limit = config->execution_limit,
    };

    StackPtxInjectCompilerStateSerialize state = {
        .annotated_ptx = config->kernel_ptx,
        .compiler_info = compiler_info,
        .stack_info = config->stack_info,
        .extra = &extra,
        .registers = registers,
        .num_registers = num_registers,
        .request_stubs = request_stubs,
        .request_stub_sizes = request_stub_sizes,
        .num_request_stubs = compiler->num_injects,
    };

    size_t compiler_state_wire_size = 0;
    StackPtxInjectSerializeResult sres = stack_ptx_inject_compiler_state_serialize(
        &state, NULL, 0, &compiler_state_wire_size
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS || compiler_state_wire_size == 0) {
        free(registers);
        free(requests);
        free(request_stubs);
        free(request_stub_sizes);
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    compiler->compiler_state_wire = (uint8_t*)malloc(compiler_state_wire_size);
    if (!compiler->compiler_state_wire) {
        free(registers);
        free(requests);
        free(request_stubs);
        free(request_stub_sizes);
        ptx_inject_destroy(ptx_inject);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }
    compiler->compiler_state_wire_size = compiler_state_wire_size;

    sres = stack_ptx_inject_compiler_state_serialize(
        &state, compiler->compiler_state_wire, compiler->compiler_state_wire_size, &compiler->compiler_state_wire_size
    );

    free(registers);
    free(requests);
    free(request_stubs);
    free(request_stub_sizes);
    ptx_inject_destroy(ptx_inject);

    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS) {
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    char** addrs = NULL;
    size_t num_addrs = 0;
    StackPtxCompilerResult split_rc = stack_ptx_compiler_split_addrs(stack_ptx_compiler_nng_addr(config), &addrs, &num_addrs);
    if (split_rc != STACK_PTX_COMPILER_SUCCESS) {
        stack_ptx_compiler_nng_destroy(compiler);
        return split_rc;
    }

    compiler->servers = (StackPtxCompilerNngServer*)calloc(num_addrs, sizeof(*compiler->servers));
    if (!compiler->servers) {
        stack_ptx_compiler_free_addrs(addrs, num_addrs);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    nng_socket* init_socks = (nng_socket*)calloc(num_addrs, sizeof(*init_socks));
    int* init_socks_open = (int*)calloc(num_addrs, sizeof(*init_socks_open));
    if (!init_socks || !init_socks_open) {
        free(init_socks);
        free(init_socks_open);
        stack_ptx_compiler_free_addrs(addrs, num_addrs);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    size_t servers_written = 0;
    for (size_t i = 0; i < num_addrs; ++i) {
        if (!addrs[i] || !addrs[i][0]) {
            continue;
        }

        nng_socket sock;
        int rv = nng_req0_open(&sock);
        if (rv != 0) {
            continue;
        }
        rv = nng_dial(sock, addrs[i], NULL, 0);
        if (rv != 0) {
            nng_close(sock);
            continue;
        }

        StackPtxNngInitResponseView init_resp;
        StackPtxCompilerResult init_rc =
            stack_ptx_compiler_nng_send_init(sock, compiler->compiler_state_wire, compiler->compiler_state_wire_size, &init_resp);
        if (init_rc != STACK_PTX_COMPILER_SUCCESS) {
            nng_close(sock);
            continue;
        }
        if (init_resp.num_injects != (uint64_t)compiler->num_injects) {
            nng_close(sock);
            continue;
        }

        size_t desired_conns = 0;
        if (config->requested_capabilities > 0) {
            desired_conns = config->requested_capabilities;
        } else if (init_resp.server_queue_slots > 0) {
            desired_conns = (size_t)init_resp.server_queue_slots;
        } else {
            desired_conns = (size_t)init_resp.server_capabilities;
        }
        if (desired_conns < 1) {
            desired_conns = 1;
        }
        if (init_resp.server_queue_slots > 0 && desired_conns > (size_t)init_resp.server_queue_slots) {
            desired_conns = (size_t)init_resp.server_queue_slots;
        }
        if (desired_conns < 1) {
            desired_conns = 1;
        }

        StackPtxCompilerNngServer* server = &compiler->servers[servers_written];
        server->addr = addrs[i];
        addrs[i] = NULL;
        server->session_id = init_resp.session_id;
        server->server_capabilities = init_resp.server_capabilities;
        server->server_queue_slots = init_resp.server_queue_slots;
        server->num_conns = desired_conns;

        init_socks[servers_written] = sock;
        init_socks_open[servers_written] = 1;
        servers_written += 1;
    }

    stack_ptx_compiler_free_addrs(addrs, num_addrs);
    addrs = NULL;
    num_addrs = 0;

    if (servers_written == 0) {
        free(init_socks);
        free(init_socks_open);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    compiler->num_servers = servers_written;

    size_t max_conns = 0;
    for (size_t i = 0; i < compiler->num_servers; ++i) {
        max_conns += compiler->servers[i].num_conns;
    }
    if (max_conns == 0) {
        free(init_socks);
        free(init_socks_open);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    compiler->conns = (StackPtxCompilerNngConn*)calloc(max_conns, sizeof(*compiler->conns));
    if (!compiler->conns) {
        for (size_t i = 0; i < compiler->num_servers; ++i) {
            if (init_socks_open[i]) {
                nng_close(init_socks[i]);
                init_socks_open[i] = 0;
            }
        }
        free(init_socks);
        free(init_socks_open);
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    size_t conns_written = 0;
    for (size_t server_idx = 0; server_idx < compiler->num_servers; ++server_idx) {
        StackPtxCompilerNngServer* server = &compiler->servers[server_idx];
        const size_t desired = server->num_conns;
        size_t created = 0;

        if (init_socks_open[server_idx]) {
            StackPtxCompilerNngConn* conn = &compiler->conns[conns_written++];
            conn->sock = init_socks[server_idx];
            conn->sock_open = 1;
            conn->in_flight = 0;
            conn->in_flight_job_id = 0;
            conn->in_flight_module_idx = 0;
            conn->server_idx = server_idx;
            init_socks_open[server_idx] = 0;
            created += 1;
        }

        for (size_t i = 1; i < desired; ++i) {
            nng_socket sock;
            int rv = nng_req0_open(&sock);
            if (rv != 0) {
                break;
            }
            rv = nng_dial(sock, server->addr, NULL, 0);
            if (rv != 0) {
                nng_close(sock);
                break;
            }

            StackPtxCompilerNngConn* conn = &compiler->conns[conns_written++];
            conn->sock = sock;
            conn->sock_open = 1;
            conn->in_flight = 0;
            conn->in_flight_job_id = 0;
            conn->in_flight_module_idx = 0;
            conn->server_idx = server_idx;
            created += 1;
        }

        server->num_conns = created;
    }

    for (size_t i = 0; i < compiler->num_servers; ++i) {
        if (init_socks_open[i]) {
            nng_close(init_socks[i]);
            init_socks_open[i] = 0;
        }
    }
    free(init_socks);
    free(init_socks_open);
    init_socks = NULL;
    init_socks_open = NULL;

    if (conns_written == 0) {
        stack_ptx_compiler_nng_destroy(compiler);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    compiler->num_conns = conns_written;
    compiler->rr_conn_idx = 0;

    for (size_t i = 0; i < compiler->num_servers; ++i) {
        StackPtxCompilerNngServer* server = &compiler->servers[i];
        server->local_in_flight = 0;
        server->status_sock_open = 0;
        server->status_in_flight = 0;
        server->status_last_send_ms = 0;
        server->status_last_recv_ms = 0;
        server->status_valid = 0;
        memset(&server->status, 0, sizeof(server->status));

        nng_socket sock;
        int rv = nng_req0_open(&sock);
        if (rv != 0) {
            continue;
        }
        rv = nng_dial(sock, server->addr, NULL, 0);
        if (rv != 0) {
            nng_close(sock);
            continue;
        }
        server->status_sock = sock;
        server->status_sock_open = 1;
    }

    *out_compiler = compiler;
    if (out_capabilities) {
        *out_capabilities = compiler->num_conns;
    }
    if (out_queue_slots) {
        *out_queue_slots = compiler->num_conns;
    }
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_nng_destroy(StackPtxCompilerNng* compiler) {
    if (!compiler) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    if (compiler->servers && compiler->conns) {
        for (size_t server_idx = 0; server_idx < compiler->num_servers; ++server_idx) {
            const uint64_t session_id = compiler->servers[server_idx].session_id;
            if (session_id == 0) {
                continue;
            }
            for (size_t conn_idx = 0; conn_idx < compiler->num_conns; ++conn_idx) {
                StackPtxCompilerNngConn* conn = &compiler->conns[conn_idx];
                if (!conn->sock_open || conn->in_flight || conn->server_idx != server_idx) {
                    continue;
                }
                stack_ptx_compiler_nng_best_effort_evict(conn->sock, session_id);
                break;
            }
        }
    }

    if (compiler->conns) {
        for (size_t i = 0; i < compiler->num_conns; ++i) {
            if (compiler->conns[i].sock_open) {
                nng_close(compiler->conns[i].sock);
                compiler->conns[i].sock_open = 0;
            }
        }
    }
    free(compiler->conns);
    compiler->conns = NULL;
    compiler->num_conns = 0;
    compiler->rr_conn_idx = 0;

    if (compiler->servers) {
        for (size_t i = 0; i < compiler->num_servers; ++i) {
            if (compiler->servers[i].status_sock_open) {
                nng_close(compiler->servers[i].status_sock);
                compiler->servers[i].status_sock_open = 0;
            }
            free(compiler->servers[i].addr);
        }
    }
    free(compiler->servers);
    compiler->servers = NULL;
    compiler->num_servers = 0;

    free(compiler->compiler_state_wire);
    compiler->compiler_state_wire = NULL;
    compiler->compiler_state_wire_size = 0;
    free((void*)compiler->instruction_stubs);
    compiler->instruction_stubs = NULL;
    free(compiler);
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_nng_submit(
    StackPtxCompilerNng* compiler,
    const StackPtxCompilerWork* work
) {
    if (!compiler || !work) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (!compiler->conns || compiler->num_conns == 0 || !compiler->servers || compiler->num_servers == 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    if (!compiler->compiler_state_wire || compiler->compiler_state_wire_size == 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    stack_ptx_compiler_nng_status_tick(compiler);

    StackPtxCompilerResult rc = stack_ptx_compiler_build_instruction_stubs(compiler, work, compiler->instruction_stubs);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        return rc;
    }

    size_t instructions_wire_size = 0;
    StackPtxInjectSerializeResult sres = stack_ptx_instructions_serialize(
        compiler->instruction_stubs, compiler->num_injects, NULL, 0, &instructions_wire_size
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    uint8_t* instructions_wire = (uint8_t*)nng_alloc(instructions_wire_size);
    if (!instructions_wire) {
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    sres = stack_ptx_instructions_serialize(
        compiler->instruction_stubs, compiler->num_injects, instructions_wire, instructions_wire_size, &instructions_wire_size
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS) {
        nng_free(instructions_wire, instructions_wire_size);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    const size_t conn_idx = stack_ptx_compiler_nng_choose_conn(compiler);
    if (conn_idx == SIZE_MAX) {
        nng_free(instructions_wire, instructions_wire_size);
        return STACK_PTX_COMPILER_ERROR_QUEUE_FULL;
    }

    StackPtxCompilerNngConn* conn = &compiler->conns[conn_idx];
    if (conn->server_idx >= compiler->num_servers) {
        nng_free(instructions_wire, instructions_wire_size);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    const uint64_t session_id = compiler->servers[conn->server_idx].session_id;
    if (session_id == 0) {
        nng_free(instructions_wire, instructions_wire_size);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    const size_t req_wire_size =
        stack_ptx_nng_request_wire_size(instructions_wire_size);
    nng_msg* msg = NULL;
    int rv = nng_msg_alloc(&msg, req_wire_size);
    if (rv != 0) {
        nng_free(instructions_wire, instructions_wire_size);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    uint8_t* body = (uint8_t*)nng_msg_body(msg);
    StackPtxNngWireResult wrc = stack_ptx_nng_write_request(
        body,
        req_wire_size,
        session_id,
        work->job_id,
        (uint64_t)work->module_idx,
        instructions_wire,
        instructions_wire_size
    );

    nng_free(instructions_wire, instructions_wire_size);
    instructions_wire = NULL;
    instructions_wire_size = 0;

    if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
        nng_msg_free(msg);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    rv = nng_sendmsg(conn->sock, msg, 0);
    if (rv != 0) {
        nng_msg_free(msg);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    conn->in_flight = 1;
    conn->in_flight_job_id = work->job_id;
    conn->in_flight_module_idx = work->module_idx;
    if (conn->server_idx < compiler->num_servers) {
        compiler->servers[conn->server_idx].local_in_flight += 1;
    }
    compiler->rr_conn_idx = (conn_idx + 1) % compiler->num_conns;
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_nng_poll(
    StackPtxCompilerNng* compiler,
    StackPtxCompilerOutput* out_result,
    int* out_has_result
) {
    if (!compiler || !out_result || !out_has_result) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    *out_has_result = 0;
    memset(out_result, 0, sizeof(*out_result));

    if (!compiler->conns || compiler->num_conns == 0 || !compiler->servers || compiler->num_servers == 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    stack_ptx_compiler_nng_status_tick(compiler);

    for (size_t i = 0; i < compiler->num_conns; ++i) {
        const size_t idx = (compiler->rr_conn_idx + i) % compiler->num_conns;
        StackPtxCompilerNngConn* conn = &compiler->conns[idx];
        if (!conn->sock_open || !conn->in_flight) {
            continue;
        }

        nng_msg* msg = NULL;
        int rv = nng_recvmsg(conn->sock, &msg, NNG_FLAG_NONBLOCK);
        if (rv == NNG_EAGAIN) {
            continue;
        }

        if (rv != 0) {
            out_result->job_id = conn->in_flight_job_id;
            out_result->module_idx = (size_t)conn->in_flight_module_idx;
            out_result->status = STACK_PTX_COMPILER_ERROR_INTERNAL;
            out_result->cubin = NULL;
            out_result->cubin_size = 0;
            out_result->compile_ms = 0.0;
            *out_has_result = 1;
            stack_ptx_compiler_nng_conn_mark_idle(compiler, conn);
            compiler->rr_conn_idx = (idx + 1) % compiler->num_conns;
            return STACK_PTX_COMPILER_SUCCESS;
        }

        StackPtxNngCompileResponseView resp;
        memset(&resp, 0, sizeof(resp));
        StackPtxNngWireResult wrc =
            stack_ptx_nng_read_response((const uint8_t*)nng_msg_body(msg), nng_msg_len(msg), &resp);
        if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
            nng_msg_free(msg);
            out_result->job_id = conn->in_flight_job_id;
            out_result->module_idx = (size_t)conn->in_flight_module_idx;
            out_result->status = STACK_PTX_COMPILER_ERROR_INTERNAL;
            *out_has_result = 1;
            stack_ptx_compiler_nng_conn_mark_idle(compiler, conn);
            compiler->rr_conn_idx = (idx + 1) % compiler->num_conns;
            return STACK_PTX_COMPILER_SUCCESS;
        }

        if (conn->server_idx >= compiler->num_servers ||
            resp.session_id != compiler->servers[conn->server_idx].session_id ||
            resp.job_id != conn->in_flight_job_id ||
            (size_t)resp.module_idx != conn->in_flight_module_idx) {
            nng_msg_free(msg);
            out_result->job_id = conn->in_flight_job_id;
            out_result->module_idx = (size_t)conn->in_flight_module_idx;
            out_result->status = STACK_PTX_COMPILER_ERROR_INTERNAL;
            *out_has_result = 1;
            stack_ptx_compiler_nng_conn_mark_idle(compiler, conn);
            compiler->rr_conn_idx = (idx + 1) % compiler->num_conns;
            return STACK_PTX_COMPILER_SUCCESS;
        }

        out_result->job_id = resp.job_id;
        out_result->module_idx = (size_t)resp.module_idx;
        out_result->status = resp.status;
        out_result->compile_ms = resp.compile_ms;

        if (resp.cubin_size > 0) {
            void* cubin = malloc(resp.cubin_size);
            if (!cubin) {
                out_result->status = STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
                out_result->cubin = NULL;
                out_result->cubin_size = 0;
            } else {
                memcpy(cubin, resp.cubin, resp.cubin_size);
                out_result->cubin = cubin;
                out_result->cubin_size = resp.cubin_size;
            }
        }

        nng_msg_free(msg);
        *out_has_result = 1;
        stack_ptx_compiler_nng_conn_mark_idle(compiler, conn);
        compiler->rr_conn_idx = (idx + 1) % compiler->num_conns;
        return STACK_PTX_COMPILER_SUCCESS;
    }

    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_compiler_backend_nng_destroy(void* impl) {
    return stack_ptx_compiler_nng_destroy((StackPtxCompilerNng*)impl);
}

static StackPtxCompilerResult stack_ptx_compiler_backend_nng_submit(void* impl, const StackPtxCompilerWork* work) {
    return stack_ptx_compiler_nng_submit((StackPtxCompilerNng*)impl, work);
}

static StackPtxCompilerResult stack_ptx_compiler_backend_nng_poll(
    void* impl,
    StackPtxCompilerOutput* out_result,
    int* out_has_result
) {
    return stack_ptx_compiler_nng_poll((StackPtxCompilerNng*)impl, out_result, out_has_result);
}

static const StackPtxCompilerVTable kStackPtxCompilerNngVTable = {
    stack_ptx_compiler_backend_nng_destroy,
    stack_ptx_compiler_backend_nng_submit,
    stack_ptx_compiler_backend_nng_poll,
};

StackPtxCompilerResult stack_ptx_compiler_backend_nng_create(
    const StackPtxCompilerHandleConfig* config,
    StackPtxCompilerBackendCreateOut* out
) {
    if (!config || !out) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    memset(out, 0, sizeof(*out));

    StackPtxCompilerNng* compiler = NULL;
    size_t capabilities = 0;
    size_t queue_slots = 0;
    StackPtxCompilerResult rc = stack_ptx_compiler_nng_create(config, &compiler, &capabilities, &queue_slots);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        return rc;
    }

    out->impl = compiler;
    out->vtable = &kStackPtxCompilerNngVTable;
    out->capabilities = capabilities;
    out->queue_slots = queue_slots;
    return STACK_PTX_COMPILER_SUCCESS;
}

StackPtxCompilerResult stack_ptx_compiler_backend_nng_connected_server_count(
    void* impl,
    size_t* out_num_servers
) {
    if (out_num_servers) {
        *out_num_servers = 0;
    }
    if (!impl || !out_num_servers) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    const StackPtxCompilerNng* compiler = (const StackPtxCompilerNng*)impl;
    *out_num_servers = compiler->num_servers;
    return STACK_PTX_COMPILER_SUCCESS;
}

StackPtxCompilerResult stack_ptx_compiler_backend_nng_connected_server_addr(
    void* impl,
    size_t server_idx,
    const char** out_addr
) {
    if (out_addr) {
        *out_addr = NULL;
    }
    if (!impl || !out_addr) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    const StackPtxCompilerNng* compiler = (const StackPtxCompilerNng*)impl;
    if (!compiler->servers || server_idx >= compiler->num_servers) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    const char* addr = compiler->servers[server_idx].addr;
    if (!addr || !addr[0]) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    *out_addr = addr;
    return STACK_PTX_COMPILER_SUCCESS;
}

StackPtxCompilerResult stack_ptx_compiler_backend_nng_connected_server_conn_count(
    void* impl,
    size_t server_idx,
    size_t* out_num_conns
) {
    if (out_num_conns) {
        *out_num_conns = 0;
    }
    if (!impl || !out_num_conns) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    const StackPtxCompilerNng* compiler = (const StackPtxCompilerNng*)impl;
    if (!compiler->servers || server_idx >= compiler->num_servers) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (!compiler->conns || compiler->num_conns == 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    size_t n = 0;
    for (size_t i = 0; i < compiler->num_conns; ++i) {
        const StackPtxCompilerNngConn* conn = &compiler->conns[i];
        if (conn->sock_open && conn->server_idx == server_idx) {
            n += 1;
        }
    }
    *out_num_conns = n;
    return STACK_PTX_COMPILER_SUCCESS;
}
