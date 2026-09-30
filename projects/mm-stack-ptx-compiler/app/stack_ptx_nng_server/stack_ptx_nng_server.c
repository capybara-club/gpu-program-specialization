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

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <nvPTXCompiler.h>
#include <nng/nng.h>
#include <nng/protocol/reqrep0/rep.h>
#include <pthread.h>
#include <ptx_inject.h>
#include <stack_ptx_inject_serialize.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct StackPtxNngServerSession StackPtxNngServerSession;

typedef struct {
    int used;
    uint64_t job_id;
    StackPtxCompilerResult status;
    double compile_ms;
    void* cubin;
    size_t cubin_size;
} StackPtxNngServerCompletedSlot;

struct StackPtxNngServerSession {
    uint64_t session_id;
    StackPtxCompilerHandle* compiler;

    uint8_t* compiler_state_buf;
    size_t compiler_state_buf_size;
    StackPtxInjectCompilerStateDeserialize* compiler_state;
    size_t num_injects;

    size_t capabilities;
    size_t queue_slots;
    uint64_t last_used_ms;

    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int mutex_initialized;
    int cond_initialized;
    int polling;
    size_t in_flight;
    uint64_t next_local_job_id;
    StackPtxNngServerCompletedSlot* completed;
    size_t completed_capacity;

    size_t refcount;
    StackPtxNngServerSession* next;
};

typedef struct {
    pthread_mutex_t mutex;
    int mutex_initialized;

    StackPtxNngServerSession* sessions;
    uint64_t next_session_id;
    size_t requested_capabilities;
    size_t workspace_bytes;

    struct timespec start_time;
    uint64_t total_compiles;

    uint64_t session_ttl_ms;
    size_t max_sessions;
    uint64_t janitor_interval_ms;
    pthread_t janitor_thread;
    int janitor_thread_started;
    StackPtxCompilerResult last_error;
    double last_error_uptime_ms;
    int verbose;
} StackPtxNngServer;

typedef struct {
    StackPtxNngServer* server;
    nng_socket sock;
    nng_ctx ctx;
    nng_aio* aio;
} StackPtxNngServerWorker;

static const StackPtxInstruction kStackPtxReturnInstruction = { .instruction_type = STACK_PTX_INSTRUCTION_TYPE_RETURN };

static int stack_ptx_compiler_version_matches(uint16_t major, uint16_t minor, uint16_t patch) {
    return major == STACK_PTX_COMPILER_VERSION_MAJOR &&
        minor == STACK_PTX_COMPILER_VERSION_MINOR &&
        patch == STACK_PTX_COMPILER_VERSION_PATCH;
}

static void stack_ptx_nng_sleep_ms(int ms) {
    if (ms <= 0) {
        return;
    }
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static double stack_ptx_nng_elapsed_ms(const struct timespec* start, const struct timespec* end) {
    const double sec = (double)(end->tv_sec - start->tv_sec) * 1000.0;
    const double nsec = (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
    return sec + nsec;
}

static uint64_t stack_ptx_nng_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000u);
}

static void stack_ptx_nng_server_note_error(StackPtxNngServer* server, StackPtxCompilerResult status) {
    if (!server || !server->mutex_initialized || status == STACK_PTX_COMPILER_SUCCESS) {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    const double uptime_ms = stack_ptx_nng_elapsed_ms(&server->start_time, &now);
    pthread_mutex_lock(&server->mutex);
    server->last_error = status;
    server->last_error_uptime_ms = uptime_ms;
    pthread_mutex_unlock(&server->mutex);
}

static size_t stack_ptx_nng_server_default_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) {
        return 1;
    }
    if ((uintmax_t)n > (uintmax_t)SIZE_MAX) {
        return SIZE_MAX;
    }
    return (size_t)n;
}

static void stack_ptx_nng_server_log_ptx_versions(void) {
    unsigned int api_major = 0;
    unsigned int api_minor = 0;
    nvPTXCompileResult rc = nvPTXCompilerGetVersion(&api_major, &api_minor);
    if (rc == NVPTXCOMPILE_SUCCESS) {
        fprintf(stderr, "stack_ptx_nng_server nvPTXCompiler api version: %u.%u\n", api_major, api_minor);
    } else {
        fprintf(stderr, "stack_ptx_nng_server nvPTXCompiler api version: unavailable\n");
    }
#ifdef PTX_INJECT_MAX_UNIQUE_INJECTS
    fprintf(stderr, "stack_ptx_nng_server ptx_inject max unique sites: %d\n", PTX_INJECT_MAX_UNIQUE_INJECTS);
#else
    fprintf(stderr, "stack_ptx_nng_server ptx_inject max unique sites: unknown\n");
#endif
}

static int stack_ptx_nng_server_parse_size_t(const char* s, size_t* out) {
    if (!s || !*s || !out) {
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

static int stack_ptx_nng_server_peek_msg_type(const uint8_t* body, size_t body_size, uint32_t* out_msg_type) {
    if (out_msg_type) {
        *out_msg_type = 0;
    }
    if (!body || body_size < 12 || !out_msg_type) {
        return 0;
    }
    uint32_t msg_type = 0;
    memcpy(&msg_type, body + 8, sizeof(msg_type));
    *out_msg_type = msg_type;
    return 1;
}

static void stack_ptx_nng_server_session_destroy(StackPtxNngServerSession* session) {
    if (!session) {
        return;
    }
    if (session->compiler) {
        (void)stack_ptx_compiler_destroy(session->compiler);
        session->compiler = NULL;
    }
    if (session->completed) {
        for (size_t i = 0; i < session->completed_capacity; ++i) {
            if (session->completed[i].used && session->completed[i].cubin) {
                free(session->completed[i].cubin);
            }
        }
    }
    free(session->completed);
    session->completed = NULL;
    session->completed_capacity = 0;
    free(session->compiler_state_buf);
    session->compiler_state_buf = NULL;
    session->compiler_state_buf_size = 0;
    session->compiler_state = NULL;
    session->num_injects = 0;
    if (session->cond_initialized) {
        pthread_cond_destroy(&session->cond);
        session->cond_initialized = 0;
    }
    if (session->mutex_initialized) {
        pthread_mutex_destroy(&session->mutex);
        session->mutex_initialized = 0;
    }
    free(session);
}

static size_t stack_ptx_nng_server_count_sessions_locked(const StackPtxNngServer* server) {
    size_t n = 0;
    for (const StackPtxNngServerSession* s = server ? server->sessions : NULL; s; s = s->next) {
        n += 1;
    }
    return n;
}

static StackPtxNngServerSession* stack_ptx_nng_server_acquire_session(StackPtxNngServer* server, uint64_t session_id) {
    if (!server || session_id == 0 || !server->mutex_initialized) {
        return NULL;
    }
    pthread_mutex_lock(&server->mutex);
    for (StackPtxNngServerSession* s = server->sessions; s; s = s->next) {
        if (s->session_id == session_id) {
            s->refcount += 1;
            s->last_used_ms = stack_ptx_nng_now_ms();
            pthread_mutex_unlock(&server->mutex);
            return s;
        }
    }
    pthread_mutex_unlock(&server->mutex);
    return NULL;
}

static void stack_ptx_nng_server_release_session(StackPtxNngServer* server, StackPtxNngServerSession* session) {
    if (!server || !session || !server->mutex_initialized) {
        return;
    }
    pthread_mutex_lock(&server->mutex);
    if (session->refcount > 0) {
        session->refcount -= 1;
    }
    pthread_mutex_unlock(&server->mutex);
}

static StackPtxCompilerResult stack_ptx_nng_server_evict_session(StackPtxNngServer* server, uint64_t session_id) {
    if (!server || session_id == 0 || !server->mutex_initialized) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    StackPtxNngServerSession* victim = NULL;

    pthread_mutex_lock(&server->mutex);
    StackPtxNngServerSession** prev = &server->sessions;
    for (StackPtxNngServerSession* s = server->sessions; s; s = s->next) {
        if (s->session_id == session_id) {
            victim = s;
            break;
        }
        prev = &s->next;
    }
    if (!victim) {
        pthread_mutex_unlock(&server->mutex);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (victim->refcount != 0) {
        pthread_mutex_unlock(&server->mutex);
        return STACK_PTX_COMPILER_ERROR_QUEUE_FULL;
    }
    if (victim->mutex_initialized) {
        pthread_mutex_lock(&victim->mutex);
    }
    const size_t in_flight = victim->in_flight;
    if (victim->mutex_initialized) {
        pthread_mutex_unlock(&victim->mutex);
    }
    if (in_flight != 0) {
        pthread_mutex_unlock(&server->mutex);
        return STACK_PTX_COMPILER_ERROR_QUEUE_FULL;
    }

    *prev = victim->next;
    pthread_mutex_unlock(&server->mutex);

    stack_ptx_nng_server_session_destroy(victim);
    return STACK_PTX_COMPILER_SUCCESS;
}

static void stack_ptx_nng_server_cleanup_sessions(StackPtxNngServer* server) {
    if (!server || !server->mutex_initialized) {
        return;
    }
    if (server->session_ttl_ms == 0 && server->max_sessions == 0) {
        return;
    }

    const uint64_t now_ms = stack_ptx_nng_now_ms();

    for (;;) {
        StackPtxNngServerSession* victim = NULL;
        StackPtxNngServerSession** victim_prev = NULL;

        pthread_mutex_lock(&server->mutex);

        if (server->session_ttl_ms > 0) {
            StackPtxNngServerSession** prev = &server->sessions;
            for (StackPtxNngServerSession* s = server->sessions; s; s = s->next) {
                if (s->refcount != 0) {
                    prev = &s->next;
                    continue;
                }
                if (now_ms >= s->last_used_ms && (now_ms - s->last_used_ms) >= server->session_ttl_ms) {
                    size_t in_flight = 0;
                    if (s->mutex_initialized) {
                        pthread_mutex_lock(&s->mutex);
                        in_flight = s->in_flight;
                        pthread_mutex_unlock(&s->mutex);
                    }
                    if (in_flight == 0) {
                        victim = s;
                        victim_prev = prev;
                        break;
                    }
                }
                prev = &s->next;
            }
        }

        if (!victim && server->max_sessions > 0) {
            const size_t session_count = stack_ptx_nng_server_count_sessions_locked(server);
            if (session_count > server->max_sessions) {
                uint64_t best_last_used = UINT64_MAX;
                StackPtxNngServerSession* best = NULL;
                StackPtxNngServerSession** best_prev = NULL;

                StackPtxNngServerSession** prev = &server->sessions;
                for (StackPtxNngServerSession* s = server->sessions; s; s = s->next) {
                    if (s->refcount != 0) {
                        prev = &s->next;
                        continue;
                    }
                    size_t in_flight = 0;
                    if (s->mutex_initialized) {
                        pthread_mutex_lock(&s->mutex);
                        in_flight = s->in_flight;
                        pthread_mutex_unlock(&s->mutex);
                    }
                    if (in_flight == 0 && s->last_used_ms <= best_last_used) {
                        best_last_used = s->last_used_ms;
                        best = s;
                        best_prev = prev;
                    }
                    prev = &s->next;
                }
                if (best) {
                    victim = best;
                    victim_prev = best_prev;
                }
            }
        }

        if (!victim || !victim_prev) {
            pthread_mutex_unlock(&server->mutex);
            return;
        }

        *victim_prev = victim->next;
        pthread_mutex_unlock(&server->mutex);

        stack_ptx_nng_server_session_destroy(victim);
    }
}

static void* stack_ptx_nng_server_janitor_main(void* arg) {
    StackPtxNngServer* server = (StackPtxNngServer*)arg;
    if (!server) {
        return NULL;
    }
    for (;;) {
        const uint64_t interval_ms = server->janitor_interval_ms;
        const int sleep_ms = (interval_ms > (uint64_t)INT_MAX) ? INT_MAX : (int)interval_ms;
        stack_ptx_nng_sleep_ms(sleep_ms);
        stack_ptx_nng_server_cleanup_sessions(server);
    }
}

static int stack_ptx_nng_server_stub_length(
    const StackPtxInstruction* stub,
    const uint8_t* buf_start,
    const uint8_t* buf_end,
    size_t* out_len
) {
    if (out_len) {
        *out_len = 0;
    }
    if (!stub || !buf_start || !buf_end || buf_end <= buf_start || !out_len) {
        return 0;
    }

    const uint8_t* stub_bytes = (const uint8_t*)stub;
    if (stub_bytes < buf_start || stub_bytes >= buf_end) {
        return 0;
    }

    const size_t remaining_bytes = (size_t)(buf_end - stub_bytes);
    const size_t remaining_instructions = remaining_bytes / sizeof(StackPtxInstruction);
    if (remaining_instructions == 0) {
        return 0;
    }

    for (size_t i = 0; i < remaining_instructions; ++i) {
        if (stub[i].instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN) {
            *out_len = i + 1;
            return 1;
        }
    }
    return 0;
}

static int stack_ptx_nng_server_session_pop_completed(
    StackPtxNngServerSession* session,
    uint64_t job_id,
    StackPtxCompilerResult* out_status,
    double* out_compile_ms,
    void** out_cubin,
    size_t* out_cubin_size
) {
    if (out_status) {
        *out_status = STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    if (out_compile_ms) {
        *out_compile_ms = 0.0;
    }
    if (out_cubin) {
        *out_cubin = NULL;
    }
    if (out_cubin_size) {
        *out_cubin_size = 0;
    }
    if (!session || !out_status || !out_compile_ms || !out_cubin || !out_cubin_size) {
        return 0;
    }

    for (size_t i = 0; i < session->completed_capacity; ++i) {
        StackPtxNngServerCompletedSlot* slot = &session->completed[i];
        if (!slot->used || slot->job_id != job_id) {
            continue;
        }

        *out_status = slot->status;
        *out_compile_ms = slot->compile_ms;
        *out_cubin = slot->cubin;
        *out_cubin_size = slot->cubin_size;

        slot->used = 0;
        slot->job_id = 0;
        slot->status = STACK_PTX_COMPILER_ERROR_INTERNAL;
        slot->compile_ms = 0.0;
        slot->cubin = NULL;
        slot->cubin_size = 0;
        return 1;
    }

    return 0;
}

static StackPtxCompilerResult stack_ptx_nng_server_session_stash_completed(
    StackPtxNngServerSession* session,
    uint64_t job_id,
    StackPtxCompilerResult status,
    double compile_ms,
    void* cubin,
    size_t cubin_size
) {
    if (!session || !session->completed || session->completed_capacity == 0) {
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }

    for (size_t i = 0; i < session->completed_capacity; ++i) {
        StackPtxNngServerCompletedSlot* slot = &session->completed[i];
        if (slot->used) {
            continue;
        }
        slot->used = 1;
        slot->job_id = job_id;
        slot->status = status;
        slot->compile_ms = compile_ms;
        slot->cubin = cubin;
        slot->cubin_size = cubin_size;
        return STACK_PTX_COMPILER_SUCCESS;
    }

    if (cubin) {
        free(cubin);
    }
    return STACK_PTX_COMPILER_ERROR_INTERNAL;
}

static StackPtxCompilerResult stack_ptx_nng_server_session_wait_for_job(
    StackPtxNngServerSession* session,
    uint64_t job_id,
    StackPtxCompilerResult* out_status,
    double* out_compile_ms,
    void** out_cubin,
    size_t* out_cubin_size
) {
    if (out_status) {
        *out_status = STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    if (out_compile_ms) {
        *out_compile_ms = 0.0;
    }
    if (out_cubin) {
        *out_cubin = NULL;
    }
    if (out_cubin_size) {
        *out_cubin_size = 0;
    }
    if (!session || !session->compiler || !out_status || !out_compile_ms || !out_cubin || !out_cubin_size ||
        !session->mutex_initialized || !session->cond_initialized) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    for (;;) {
        pthread_mutex_lock(&session->mutex);

        if (stack_ptx_nng_server_session_pop_completed(
                session, job_id, out_status, out_compile_ms, out_cubin, out_cubin_size
            )) {
            pthread_mutex_unlock(&session->mutex);
            return STACK_PTX_COMPILER_SUCCESS;
        }

        if (!session->polling) {
            session->polling = 1;
            pthread_mutex_unlock(&session->mutex);

            StackPtxCompilerOutput out;
            memset(&out, 0, sizeof(out));
            int has_result = 0;
            StackPtxCompilerResult rc = stack_ptx_compiler_poll(session->compiler, &out, &has_result);

            pthread_mutex_lock(&session->mutex);
            session->polling = 0;
            pthread_cond_broadcast(&session->cond);
            pthread_mutex_unlock(&session->mutex);

            if (rc != STACK_PTX_COMPILER_SUCCESS) {
                return rc;
            }

            if (!has_result) {
                stack_ptx_nng_sleep_ms(1);
                continue;
            }

            if (out.job_id == job_id) {
                *out_status = out.status;
                *out_compile_ms = out.compile_ms;
                *out_cubin = out.cubin;
                *out_cubin_size = out.cubin_size;
                return STACK_PTX_COMPILER_SUCCESS;
            }

            pthread_mutex_lock(&session->mutex);
            rc = stack_ptx_nng_server_session_stash_completed(
                session, out.job_id, out.status, out.compile_ms, out.cubin, out.cubin_size
            );
            pthread_cond_broadcast(&session->cond);
            pthread_mutex_unlock(&session->mutex);

            if (rc != STACK_PTX_COMPILER_SUCCESS) {
                return rc;
            }

            continue;
        }

        pthread_cond_wait(&session->cond, &session->mutex);
        pthread_mutex_unlock(&session->mutex);
    }
}

static StackPtxCompilerResult stack_ptx_nng_server_create_session(
    StackPtxNngServer* server,
    const StackPtxNngInitRequestView* req,
    uint64_t* out_session_id,
    uint64_t* out_server_capabilities,
    uint64_t* out_server_queue_slots,
    uint64_t* out_num_injects
) {
    if (out_session_id) {
        *out_session_id = 0;
    }
    if (out_server_capabilities) {
        *out_server_capabilities = 0;
    }
    if (out_server_queue_slots) {
        *out_server_queue_slots = 0;
    }
    if (out_num_injects) {
        *out_num_injects = 0;
    }
    if (!server || !server->mutex_initialized || !req || !req->compiler_state_wire || req->compiler_state_wire_size == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }
    if (!stack_ptx_compiler_version_matches(req->compiler_version_major,
            req->compiler_version_minor,
            req->compiler_version_patch)) {
        return STACK_PTX_COMPILER_ERROR_VERSION;
    }

    StackPtxInjectCompilerStateDeserialize* compiler_state = NULL;
    size_t compiler_state_wire_used = 0;
    size_t compiler_state_buf_size = 0;
    StackPtxInjectSerializeResult sres = stack_ptx_inject_compiler_state_deserialize(
        req->compiler_state_wire,
        req->compiler_state_wire_size,
        &compiler_state_wire_used,
        NULL,
        0,
        &compiler_state_buf_size,
        &compiler_state
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS || compiler_state_buf_size == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    uint8_t* compiler_state_buf = (uint8_t*)malloc(compiler_state_buf_size);
    if (!compiler_state_buf) {
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    sres = stack_ptx_inject_compiler_state_deserialize(
        req->compiler_state_wire,
        req->compiler_state_wire_size,
        &compiler_state_wire_used,
        compiler_state_buf,
        compiler_state_buf_size,
        &compiler_state_buf_size,
        &compiler_state
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS ||
        compiler_state_wire_used != req->compiler_state_wire_size ||
        !compiler_state) {
        free(compiler_state_buf);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    if (!compiler_state->annotated_ptx ||
        !compiler_state->annotated_ptx[0] ||
        !compiler_state->compiler_info ||
        !compiler_state->stack_info ||
        !compiler_state->extra ||
        compiler_state->extra->device_capability_major == 0 ||
        compiler_state->extra->execution_limit == 0 ||
        compiler_state->num_request_stubs == 0) {
        free(compiler_state_buf);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    StackPtxCompilerHandleConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.kernel_ptx = compiler_state->annotated_ptx;
    cfg.kernel_name_format = "kernel_%06zu";
    cfg.kernel_num_kernels = 1;
    cfg.kernel_groups_per_kernel = compiler_state->num_request_stubs;
    cfg.execution_limit = compiler_state->extra->execution_limit;
    cfg.max_results = 1;
    cfg.workspace_bytes = server->workspace_bytes ? server->workspace_bytes : (1024u * 1024u);
    cfg.stack_info = compiler_state->stack_info;
    cfg.compiler_info = compiler_state->compiler_info;

    cfg.backend = STACK_PTX_COMPILER_BACKEND_LOCAL;
    cfg.requested_capabilities = server->requested_capabilities;
    cfg.device_capability_major = compiler_state->extra->device_capability_major;
    cfg.device_capability_minor = compiler_state->extra->device_capability_minor;

    size_t capabilities = 0;
    size_t queue_slots = 0;
    StackPtxCompilerHandle* compiler = NULL;
    StackPtxCompilerResult rc = stack_ptx_compiler_create(&cfg, &compiler, &capabilities, &queue_slots);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        free(compiler_state_buf);
        return rc;
    }

    StackPtxNngServerSession* session = (StackPtxNngServerSession*)calloc(1, sizeof(*session));
    if (!session) {
        (void)stack_ptx_compiler_destroy(compiler);
        free(compiler_state_buf);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    session->session_id = 0;
    session->compiler = compiler;
    session->compiler_state_buf = compiler_state_buf;
    session->compiler_state_buf_size = compiler_state_buf_size;
    session->compiler_state = compiler_state;
    session->num_injects = compiler_state->num_request_stubs;
    session->capabilities = capabilities;
    session->queue_slots = queue_slots;

    session->mutex_initialized = 0;
    session->cond_initialized = 0;
    session->polling = 0;
    session->in_flight = 0;
    session->next_local_job_id = 1;
    session->completed = NULL;
    session->completed_capacity = 0;
    session->refcount = 0;
    session->last_used_ms = 0;
    session->next = NULL;

    if (pthread_mutex_init(&session->mutex, NULL) != 0) {
        stack_ptx_nng_server_session_destroy(session);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    session->mutex_initialized = 1;
    if (pthread_cond_init(&session->cond, NULL) != 0) {
        stack_ptx_nng_server_session_destroy(session);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    session->cond_initialized = 1;

    if (queue_slots == 0) {
        stack_ptx_nng_server_session_destroy(session);
        return STACK_PTX_COMPILER_ERROR_INTERNAL;
    }
    session->completed_capacity = queue_slots;
    session->completed = (StackPtxNngServerCompletedSlot*)calloc(session->completed_capacity, sizeof(*session->completed));
    if (!session->completed) {
        stack_ptx_nng_server_session_destroy(session);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    pthread_mutex_lock(&server->mutex);
    uint64_t session_id = server->next_session_id;
    if (session_id == 0) {
        session_id = 1;
    }
    server->next_session_id = session_id + 1;

    session->session_id = session_id;
    session->last_used_ms = stack_ptx_nng_now_ms();
    session->next = server->sessions;
    server->sessions = session;
    pthread_mutex_unlock(&server->mutex);

    if (out_session_id) {
        *out_session_id = session_id;
    }
    if (out_server_capabilities) {
        *out_server_capabilities = (uint64_t)capabilities;
    }
    if (out_server_queue_slots) {
        *out_server_queue_slots = (uint64_t)queue_slots;
    }
    if (out_num_injects) {
        *out_num_injects = (uint64_t)session->num_injects;
    }
    return STACK_PTX_COMPILER_SUCCESS;
}

static StackPtxCompilerResult stack_ptx_nng_server_compile(
    StackPtxNngServerSession* session,
    const StackPtxNngCompileRequestView* req,
    void** cubin_out,
    size_t* cubin_size_out,
    double* compile_ms_out,
    double* total_ms_out
) {
    if (cubin_out) {
        *cubin_out = NULL;
    }
    if (cubin_size_out) {
        *cubin_size_out = 0;
    }
    if (compile_ms_out) {
        *compile_ms_out = 0.0;
    }
    if (total_ms_out) {
        *total_ms_out = 0.0;
    }
    if (!session || !session->compiler || !req || !cubin_out || !cubin_size_out) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    uint64_t local_job_id = 0;
    if (session->mutex_initialized) {
        pthread_mutex_lock(&session->mutex);
        local_job_id = session->next_local_job_id;
        session->next_local_job_id += 1;
        if (session->next_local_job_id == 0) {
            session->next_local_job_id = 1;
        }
        pthread_mutex_unlock(&session->mutex);
    } else {
        local_job_id = req->job_id;
        if (local_job_id == 0) {
            local_job_id = 1;
        }
    }

    StackPtxInstruction** instruction_stubs = NULL;
    size_t num_instruction_stubs = 0;
    uint8_t* instruction_buf = NULL;
    size_t instruction_buf_size = 0;
    size_t instructions_wire_used = 0;

    StackPtxInjectSerializeResult sres = stack_ptx_instructions_deserialize(
        req->instructions_wire,
        req->instructions_wire_size,
        &instructions_wire_used,
        NULL,
        0,
        &instruction_buf_size,
        &instruction_stubs,
        &num_instruction_stubs
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS || instruction_buf_size == 0) {
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    instruction_buf = (uint8_t*)malloc(instruction_buf_size);
    if (!instruction_buf) {
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    sres = stack_ptx_instructions_deserialize(
        req->instructions_wire,
        req->instructions_wire_size,
        &instructions_wire_used,
        instruction_buf,
        instruction_buf_size,
        &instruction_buf_size,
        &instruction_stubs,
        &num_instruction_stubs
    );
    if (sres != STACK_PTX_INJECT_SERIALIZE_SUCCESS || instructions_wire_used != req->instructions_wire_size) {
        free(instruction_buf);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    if (session->num_injects == 0 || num_instruction_stubs != session->num_injects) {
        free(instruction_buf);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const uint8_t* instruction_buf_start = instruction_buf;
    const uint8_t* instruction_buf_end = instruction_buf + instruction_buf_size;

    size_t gene_length = 0;
    for (size_t i = 0; i < num_instruction_stubs; ++i) {
        size_t stub_len = 0;
        if (!stack_ptx_nng_server_stub_length(
                instruction_stubs[i],
                instruction_buf_start,
                instruction_buf_end,
                &stub_len
            )) {
            free(instruction_buf);
            return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
        }
        if (stub_len > gene_length) {
            gene_length = stub_len;
        }
    }
    if (gene_length == 0 || gene_length > (SIZE_MAX / num_instruction_stubs)) {
        free(instruction_buf);
        return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
    }

    const size_t population_instructions = gene_length * num_instruction_stubs;
    StackPtxInstruction* population = (StackPtxInstruction*)malloc(population_instructions * sizeof(*population));
    if (!population) {
        free(instruction_buf);
        return STACK_PTX_COMPILER_ERROR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < population_instructions; ++i) {
        population[i] = kStackPtxReturnInstruction;
    }

    for (size_t i = 0; i < num_instruction_stubs; ++i) {
        size_t stub_len = 0;
        if (!stack_ptx_nng_server_stub_length(
                instruction_stubs[i],
                instruction_buf_start,
                instruction_buf_end,
                &stub_len
            )) {
            free(population);
            free(instruction_buf);
            return STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
        }
        memcpy(&population[i * gene_length], instruction_stubs[i], stub_len * sizeof(*population));
    }

    StackPtxCompilerWork work;
    memset(&work, 0, sizeof(work));
    work.population = population;
    work.population_instructions = population_instructions;
    work.gene_length = gene_length;
    work.module_idx = 0;
    work.job_id = local_job_id;

    struct timespec total_start;
    struct timespec total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    StackPtxCompilerResult rc = stack_ptx_compiler_submit(session->compiler, &work);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        free(population);
        free(instruction_buf);
        return rc;
    }

    if (session->mutex_initialized) {
        pthread_mutex_lock(&session->mutex);
        session->in_flight += 1;
        pthread_mutex_unlock(&session->mutex);
    }

    StackPtxCompilerResult job_status = STACK_PTX_COMPILER_ERROR_INTERNAL;
    double job_compile_ms = 0.0;
    void* job_cubin = NULL;
    size_t job_cubin_size = 0;
    rc = stack_ptx_nng_server_session_wait_for_job(
        session,
        local_job_id,
        &job_status,
        &job_compile_ms,
        &job_cubin,
        &job_cubin_size
    );

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    if (total_ms_out) {
        *total_ms_out = stack_ptx_nng_elapsed_ms(&total_start, &total_end);
    }

    if (session->mutex_initialized) {
        pthread_mutex_lock(&session->mutex);
        if (session->in_flight > 0) {
            session->in_flight -= 1;
        }
        pthread_cond_broadcast(&session->cond);
        pthread_mutex_unlock(&session->mutex);
    }

    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        if (job_cubin) {
            free(job_cubin);
        }
        free(population);
        free(instruction_buf);
        return rc;
    }

    if (compile_ms_out) {
        *compile_ms_out = job_compile_ms;
    }

    if (job_status == STACK_PTX_COMPILER_SUCCESS) {
        *cubin_out = job_cubin;
        *cubin_size_out = job_cubin_size;
    } else if (job_cubin) {
        free(job_cubin);
    }

    free(population);
    free(instruction_buf);
    return job_status;
}

static int stack_ptx_nng_server_worker_send(StackPtxNngServerWorker* worker, nng_msg* resp) {
    if (!worker || !worker->aio || !resp) {
        return NNG_EINVAL;
    }

    nng_aio_set_msg(worker->aio, resp);
    nng_ctx_send(worker->ctx, worker->aio);
    nng_aio_wait(worker->aio);
    const int rv = nng_aio_result(worker->aio);
    if (rv != 0) {
        nng_msg_free(resp);
        return rv;
    }
    return 0;
}

static void* stack_ptx_nng_server_worker_main(void* arg) {
    StackPtxNngServerWorker* worker = (StackPtxNngServerWorker*)arg;
    if (!worker || !worker->server || !worker->aio) {
        return NULL;
    }

    for (;;) {
        nng_aio_set_msg(worker->aio, NULL);
        nng_ctx_recv(worker->ctx, worker->aio);
        nng_aio_wait(worker->aio);
        int rv = nng_aio_result(worker->aio);
        if (rv != 0) {
            fprintf(stderr, "nng_ctx_recv failed: %s\n", nng_strerror(rv));
            continue;
        }

        nng_msg* msg = nng_aio_get_msg(worker->aio);
        if (!msg) {
            continue;
        }

        const uint8_t* body = (const uint8_t*)nng_msg_body(msg);
        const size_t body_size = nng_msg_len(msg);
        uint32_t msg_type = 0;
        (void)stack_ptx_nng_server_peek_msg_type(body, body_size, &msg_type);

        if (msg_type == (uint32_t)STACK_PTX_NNG_MSG_STATUS_REQUEST) {
            StackPtxNngWireResult wrc = stack_ptx_nng_read_status_request(body, body_size);
            nng_msg_free(msg);
            msg = NULL;

            StackPtxNngStatusResponseView status_view;
            memset(&status_view, 0, sizeof(status_view));
            status_view.status = (wrc == STACK_PTX_NNG_WIRE_SUCCESS) ? STACK_PTX_COMPILER_SUCCESS
                                                                     : STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
            stack_ptx_nng_server_note_error(worker->server, status_view.status);

            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double uptime_ms = 0.0;
            if (worker->server) {
                uptime_ms = stack_ptx_nng_elapsed_ms(&worker->server->start_time, &now);
            }

            uint64_t num_sessions = 0;
            uint64_t total_compiles = 0;
            uint64_t requested_capabilities = 0;
            uint64_t total_in_flight = 0;
            StackPtxCompilerResult last_error = STACK_PTX_COMPILER_SUCCESS;
            double last_error_uptime_ms = 0.0;
            if (worker->server && worker->server->mutex_initialized) {
                pthread_mutex_lock(&worker->server->mutex);
                num_sessions = (uint64_t)stack_ptx_nng_server_count_sessions_locked(worker->server);
                total_compiles = worker->server->total_compiles;
                requested_capabilities = (uint64_t)worker->server->requested_capabilities;
                last_error = worker->server->last_error;
                last_error_uptime_ms = worker->server->last_error_uptime_ms;
                for (StackPtxNngServerSession* s = worker->server->sessions; s; s = s->next) {
                    if (s->mutex_initialized) {
                        pthread_mutex_lock(&s->mutex);
                        total_in_flight += (uint64_t)s->in_flight;
                        pthread_mutex_unlock(&s->mutex);
                    }
                }
                pthread_mutex_unlock(&worker->server->mutex);
            }

            const size_t resp_size = stack_ptx_nng_status_response_wire_size();
            nng_msg* resp = NULL;
            rv = nng_msg_alloc(&resp, resp_size);
            if (rv != 0) {
                fprintf(stderr, "nng_msg_alloc failed: %s\n", nng_strerror(rv));
                continue;
            }
            wrc = stack_ptx_nng_write_status_response(
                (uint8_t*)nng_msg_body(resp),
                resp_size,
                status_view.status,
                requested_capabilities,
                num_sessions,
                total_compiles,
                uptime_ms,
                total_in_flight,
                last_error,
                last_error_uptime_ms
            );
            if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
                fprintf(stderr, "failed to write status response: wire_rc=%d\n", (int)wrc);
                nng_msg_free(resp);
                continue;
            }

            rv = stack_ptx_nng_server_worker_send(worker, resp);
            if (rv != 0) {
                fprintf(stderr, "nng_ctx_send failed: %s\n", nng_strerror(rv));
            }
            continue;
        }

        if (msg_type == (uint32_t)STACK_PTX_NNG_MSG_EVICT_REQUEST) {
            StackPtxNngEvictRequestView ev;
            memset(&ev, 0, sizeof(ev));
            StackPtxNngWireResult wrc = stack_ptx_nng_read_evict_request(body, body_size, &ev);
            nng_msg_free(msg);
            msg = NULL;

            StackPtxCompilerResult status = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
            if (wrc == STACK_PTX_NNG_WIRE_SUCCESS) {
                status = stack_ptx_nng_server_evict_session(worker->server, ev.session_id);
            }
            stack_ptx_nng_server_note_error(worker->server, status);

            const size_t resp_size = stack_ptx_nng_evict_response_wire_size();
            nng_msg* resp = NULL;
            rv = nng_msg_alloc(&resp, resp_size);
            if (rv != 0) {
                fprintf(stderr, "nng_msg_alloc failed: %s\n", nng_strerror(rv));
                continue;
            }
            wrc = stack_ptx_nng_write_evict_response(
                (uint8_t*)nng_msg_body(resp),
                resp_size,
                ev.session_id,
                status
            );
            if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
                fprintf(stderr, "failed to write evict response: wire_rc=%d\n", (int)wrc);
                nng_msg_free(resp);
                continue;
            }

            rv = stack_ptx_nng_server_worker_send(worker, resp);
            if (rv != 0) {
                fprintf(stderr, "nng_ctx_send failed: %s\n", nng_strerror(rv));
            }
            continue;
        }

        if (msg_type == (uint32_t)STACK_PTX_NNG_MSG_INIT_REQUEST) {
            StackPtxNngInitRequestView init_view;
            memset(&init_view, 0, sizeof(init_view));
            StackPtxNngWireResult wrc = stack_ptx_nng_read_init_request(body, body_size, &init_view);

            uint64_t session_id = 0;
            uint64_t server_capabilities = 0;
            uint64_t server_queue_slots = 0;
            uint64_t num_injects = 0;
            StackPtxCompilerResult status = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;

            if (wrc == STACK_PTX_NNG_WIRE_SUCCESS) {
                status = stack_ptx_nng_server_create_session(
                    worker->server,
                    &init_view,
                    &session_id,
                    &server_capabilities,
                    &server_queue_slots,
                    &num_injects
                );
                if (status == STACK_PTX_COMPILER_SUCCESS) {
                    fprintf(stderr,
                        "INIT session_id=%" PRIu64 " num_injects=%" PRIu64 " caps=%" PRIu64 " queue=%" PRIu64 "\n",
                        session_id,
                        num_injects,
                        server_capabilities,
                        server_queue_slots);
                } else {
                    fprintf(stderr, "INIT failed: %s\n", stack_ptx_compiler_result_to_string(status));
                }
            } else {
                fprintf(stderr, "invalid init request: wire_rc=%d\n", (int)wrc);
            }

            nng_msg_free(msg);
            msg = NULL;
            stack_ptx_nng_server_note_error(worker->server, status);

            const size_t resp_size = stack_ptx_nng_init_response_wire_size();
            nng_msg* resp = NULL;
            rv = nng_msg_alloc(&resp, resp_size);
            if (rv != 0) {
                fprintf(stderr, "nng_msg_alloc failed: %s\n", nng_strerror(rv));
                continue;
            }
            wrc = stack_ptx_nng_write_init_response(
                (uint8_t*)nng_msg_body(resp),
                resp_size,
                session_id,
                status,
                server_capabilities,
                server_queue_slots,
                num_injects
            );
            if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
                fprintf(stderr, "failed to write init response: wire_rc=%d\n", (int)wrc);
                nng_msg_free(resp);
                continue;
            }

            rv = stack_ptx_nng_server_worker_send(worker, resp);
            if (rv != 0) {
                fprintf(stderr, "nng_ctx_send failed: %s\n", nng_strerror(rv));
            }
            continue;
        }

        StackPtxNngCompileRequestView req_view;
        memset(&req_view, 0, sizeof(req_view));
        StackPtxNngWireResult wrc = stack_ptx_nng_read_request(body, body_size, &req_view);

        void* cubin = NULL;
        size_t cubin_size = 0;
        double compile_ms = 0.0;
        double total_ms = 0.0;
        StackPtxCompilerResult status = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;

        if (wrc == STACK_PTX_NNG_WIRE_SUCCESS) {
            if (worker->server && worker->server->verbose) {
                fprintf(
                    stderr,
                    "REQ session=%" PRIu64 " job=%" PRIu64 " module=%" PRIu64 " bytes=%zu\n",
                    req_view.session_id,
                    req_view.job_id,
                    req_view.module_idx,
                    req_view.instructions_wire_size
                );
            }
            if (worker->server && worker->server->mutex_initialized) {
                pthread_mutex_lock(&worker->server->mutex);
                worker->server->total_compiles += 1;
                pthread_mutex_unlock(&worker->server->mutex);
            }

            StackPtxNngServerSession* session = stack_ptx_nng_server_acquire_session(worker->server, req_view.session_id);
            if (!session) {
                status = STACK_PTX_COMPILER_ERROR_INVALID_VALUE;
            } else {
                status = stack_ptx_nng_server_compile(session, &req_view, &cubin, &cubin_size, &compile_ms, &total_ms);
                stack_ptx_nng_server_release_session(worker->server, session);
            }
        } else {
            fprintf(stderr, "invalid compile request: wire_rc=%d msg_type=%" PRIu32 "\n", (int)wrc, msg_type);
        }
        stack_ptx_nng_server_note_error(worker->server, status);

        if (worker->server && worker->server->verbose) {
            fprintf(
                stderr,
                "RESP session=%" PRIu64 " job=%" PRIu64 " module=%" PRIu64 " status=%s compile_ms=%.2f total_ms=%.2f cubin=%zu\n",
                req_view.session_id,
                req_view.job_id,
                req_view.module_idx,
                stack_ptx_compiler_result_to_string(status),
                compile_ms,
                total_ms,
                cubin_size
            );
        }

        const size_t resp_size = stack_ptx_nng_response_wire_size(cubin_size);
        nng_msg* resp = NULL;
        rv = nng_msg_alloc(&resp, resp_size);
        if (rv != 0) {
            fprintf(stderr, "nng_msg_alloc failed: %s\n", nng_strerror(rv));
            free(cubin);
            nng_msg_free(msg);
            msg = NULL;
            continue;
        }

        wrc = stack_ptx_nng_write_response(
            (uint8_t*)nng_msg_body(resp),
            resp_size,
            req_view.session_id,
            req_view.job_id,
            req_view.module_idx,
            status,
            compile_ms,
            total_ms,
            (const uint8_t*)cubin,
            cubin_size
        );
        free(cubin);
        cubin = NULL;

        if (wrc != STACK_PTX_NNG_WIRE_SUCCESS) {
            fprintf(stderr, "failed to write response: wire_rc=%d\n", (int)wrc);
            nng_msg_free(resp);
            nng_msg_free(msg);
            msg = NULL;
            continue;
        }

        nng_msg_free(msg);
        msg = NULL;

        rv = stack_ptx_nng_server_worker_send(worker, resp);
        if (rv != 0) {
            fprintf(stderr, "nng_ctx_send failed: %s\n", nng_strerror(rv));
        }
    }
}

int main(int argc, char** argv) {
    const char* listen_addr = "tcp://127.0.0.1:5555";
    size_t threads = stack_ptx_nng_server_default_threads();
    size_t session_ttl_ms = 5u * 60u * 1000u;
    size_t max_sessions = 1024u;
    size_t janitor_interval_ms = 1000u;
    size_t workspace_mib = 16u;
    int verbose = 0;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!arg || !arg[0]) {
            continue;
        }
        if ((strcmp(arg, "-h") == 0) || (strcmp(arg, "--help") == 0)) {
            fprintf(stderr, "usage: %s [listen_addr] [--threads N]\n", argv[0] ? argv[0] : "stack_ptx_nng_server");
            fprintf(stderr, "  [--session-ttl-ms N] (0 disables)\n");
            fprintf(stderr, "  [--max-sessions N] (0 disables)\n");
            fprintf(stderr, "  [--janitor-interval-ms N]\n");
            fprintf(stderr, "  [--workspace-mib N]\n");
            fprintf(stderr, "  [--verbose|-v]\n");
            fprintf(stderr, "  default threads: all online CPUs\n");
            return 0;
        }
        if ((strcmp(arg, "--verbose") == 0) || (strcmp(arg, "-v") == 0)) {
            verbose = 1;
            continue;
        }
        if (strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--threads requires an argument\n");
                return 1;
            }
            size_t parsed = 0;
            if (stack_ptx_nng_server_parse_size_t(argv[i + 1], &parsed) != 0 || parsed < 1) {
                fprintf(stderr, "invalid --threads value: %s\n", argv[i + 1]);
                return 1;
            }
            threads = parsed;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--session-ttl-ms") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--session-ttl-ms requires an argument\n");
                return 1;
            }
            size_t parsed = 0;
            if (stack_ptx_nng_server_parse_size_t(argv[i + 1], &parsed) != 0) {
                fprintf(stderr, "invalid --session-ttl-ms value: %s\n", argv[i + 1]);
                return 1;
            }
            session_ttl_ms = parsed;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--max-sessions") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--max-sessions requires an argument\n");
                return 1;
            }
            size_t parsed = 0;
            if (stack_ptx_nng_server_parse_size_t(argv[i + 1], &parsed) != 0) {
                fprintf(stderr, "invalid --max-sessions value: %s\n", argv[i + 1]);
                return 1;
            }
            max_sessions = parsed;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--janitor-interval-ms") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--janitor-interval-ms requires an argument\n");
                return 1;
            }
            size_t parsed = 0;
            if (stack_ptx_nng_server_parse_size_t(argv[i + 1], &parsed) != 0 || parsed < 1) {
                fprintf(stderr, "invalid --janitor-interval-ms value: %s\n", argv[i + 1]);
                return 1;
            }
            janitor_interval_ms = parsed;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--workspace-mib") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--workspace-mib requires an argument\n");
                return 1;
            }
            size_t parsed = 0;
            if (stack_ptx_nng_server_parse_size_t(argv[i + 1], &parsed) != 0 || parsed < 1) {
                fprintf(stderr, "invalid --workspace-mib value: %s\n", argv[i + 1]);
                return 1;
            }
            workspace_mib = parsed;
            i += 1;
            continue;
        }
        if (arg[0] != '-') {
            listen_addr = arg;
            continue;
        }
        fprintf(stderr, "unknown argument: %s\n", arg);
        return 1;
    }

    StackPtxNngServer server;
    memset(&server, 0, sizeof(server));
    server.sessions = NULL;
    server.next_session_id = 1;
    server.requested_capabilities = threads;
    if (workspace_mib > (SIZE_MAX >> 20)) {
        fprintf(stderr, "invalid --workspace-mib value: %zu (too large)\n", workspace_mib);
        return 1;
    }
    server.workspace_bytes = workspace_mib << 20;
    clock_gettime(CLOCK_MONOTONIC, &server.start_time);
    server.total_compiles = 0;
    server.session_ttl_ms = (uint64_t)session_ttl_ms;
    server.max_sessions = max_sessions;
    server.janitor_interval_ms = (uint64_t)janitor_interval_ms;
    server.janitor_thread_started = 0;
    server.last_error = STACK_PTX_COMPILER_SUCCESS;
    server.last_error_uptime_ms = 0.0;
    server.verbose = verbose;

    if (pthread_mutex_init(&server.mutex, NULL) != 0) {
        fprintf(stderr, "pthread_mutex_init failed\n");
        return 1;
    }
    server.mutex_initialized = 1;

    nng_socket sock;
    int rv = nng_rep0_open(&sock);
    if (rv != 0) {
        fprintf(stderr, "nng_rep0_open failed: %s\n", nng_strerror(rv));
        return 1;
    }

    rv = nng_listen(sock, listen_addr, NULL, 0);
    if (rv != 0) {
        fprintf(stderr, "nng_listen(%s) failed: %s\n", listen_addr, nng_strerror(rv));
        nng_close(sock);
        return 1;
    }

    stack_ptx_nng_server_log_ptx_versions();
    fprintf(stderr, "stack_ptx_nng_server listening on %s\n", listen_addr);
    fprintf(stderr, "stack_ptx_nng_server compiler threads: %zu\n", threads);
    fprintf(stderr, "stack_ptx_nng_server max concurrent requests: %zu\n", threads);
    fprintf(stderr, "stack_ptx_nng_server workspace (MiB): %zu\n", workspace_mib);
    fprintf(stderr, "stack_ptx_nng_server session ttl (ms): %zu\n", (size_t)server.session_ttl_ms);
    fprintf(stderr, "stack_ptx_nng_server max sessions: %zu\n", server.max_sessions);
    fprintf(stderr, "stack_ptx_nng_server janitor interval (ms): %zu\n", (size_t)server.janitor_interval_ms);

    if (pthread_create(&server.janitor_thread, NULL, stack_ptx_nng_server_janitor_main, &server) == 0) {
        server.janitor_thread_started = 1;
    } else {
        fprintf(stderr, "pthread_create(janitor) failed\n");
    }

    pthread_t* worker_threads = (pthread_t*)calloc(threads, sizeof(*worker_threads));
    StackPtxNngServerWorker* workers = (StackPtxNngServerWorker*)calloc(threads, sizeof(*workers));
    if (!worker_threads || !workers) {
        fprintf(stderr, "failed to allocate worker threads\n");
        free(worker_threads);
        free(workers);
        nng_close(sock);
        return 1;
    }

    for (size_t i = 0; i < threads; ++i) {
        StackPtxNngServerWorker* worker = &workers[i];
        memset(worker, 0, sizeof(*worker));
        worker->server = &server;
        worker->sock = sock;
        worker->aio = NULL;
        rv = nng_aio_alloc(&worker->aio, NULL, NULL);
        if (rv != 0) {
            fprintf(stderr, "nng_aio_alloc failed: %s\n", nng_strerror(rv));
            return 1;
        }
        rv = nng_ctx_open(&worker->ctx, sock);
        if (rv != 0) {
            fprintf(stderr, "nng_ctx_open failed: %s\n", nng_strerror(rv));
            return 1;
        }
        if (pthread_create(&worker_threads[i], NULL, stack_ptx_nng_server_worker_main, worker) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 1;
        }
    }

    for (size_t i = 0; i < threads; ++i) {
        pthread_join(worker_threads[i], NULL);
    }

    free(worker_threads);
    free(workers);
    nng_close(sock);
    return 0;
}
