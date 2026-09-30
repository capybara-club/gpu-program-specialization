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
/* Runs the actual public runner and worker pipeline against a deterministic
 * driver model. No test hooks or indirection are compiled into the library. */
#include "s_runner_internal.h"
#include <errno.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                                          \
            abort();                                                                                         \
        }                                                                                                    \
    } while (0)
#define LIMIT 32
static struct Fault {
    const char *name;
    unsigned calls, fail_at;
    int persistent;
} faults[] = {{"malloc", 0, 0, 0},
              {"calloc", 0, 0, 0},
              {"pthread_create", 0, 0, 0},
              {"pthread_join", 0, 0, 0},
              {"pthread_mutex_init", 0, 0, 0},
              {"pthread_cond_init", 0, 0, 0},
              {"pthread_mutex_destroy", 0, 0, 0},
              {"pthread_cond_destroy", 0, 0, 0},
              {"cuCtxGetCurrent", 0, 0, 0},
              {"cuModuleGetLoadingMode", 0, 0, 0},
              {"cuStreamCreate", 0, 0, 0},
              {"cuStreamDestroy", 0, 0, 0},
              {"cuStreamSynchronize", 0, 0, 0},
              {"cuStreamWaitEvent", 0, 0, 0},
              {"cuEventCreate", 0, 0, 0},
              {"cuEventDestroy", 0, 0, 0},
              {"cuEventRecord", 0, 0, 0},
              {"cuEventSynchronize", 0, 0, 0},
              {"cuEventElapsedTime", 0, 0, 0},
              {"cuModuleLoadData", 0, 0, 0},
              {"cuModuleUnload", 0, 0, 0},
              {"cuModuleGetFunction", 0, 0, 0},
              {"cuLaunchKernel", 0, 0, 0},
              {"cuMemsetD2D32Async", 0, 0, 0}};
#define NFAULTS (sizeof(faults) / sizeof(faults[0]))
static unsigned scenarios;
static struct Fault *fault(const char *name) {
    size_t i;
    for (i = 0; i < NFAULTS; ++i)
        if (!strcmp(name, faults[i].name))
            return &faults[i];
    abort();
}
static int fail(const char *name) {
    struct Fault *f = fault(name);
    ++f->calls;
    return f->fail_at && (f->calls == f->fail_at || (f->persistent && f->calls >= f->fail_at));
}
static void arm(const char *name, unsigned at, int persistent) {
    struct Fault *f = fault(name);
    f->calls = 0;
    f->fail_at = at;
    f->persistent = persistent;
}
static void reset_faults(void) {
    size_t i;
    for (i = 0; i < NFAULTS; ++i)
        faults[i].calls = faults[i].fail_at = faults[i].persistent = 0;
}
#define DRIVER()                                                                                             \
    do {                                                                                                     \
        if (fail(__func__))                                                                                  \
            return CUDA_ERROR_UNKNOWN;                                                                       \
    } while (0)
static size_t allocations, threads, mutexes, conditions, stream_count, event_count, module_count;
static unsigned next_stream, complete[LIMIT];
struct TestStream {
    unsigned id, clock[LIMIT];
};
struct TestEvent {
    unsigned clock[LIMIT];
    int recorded;
};
struct TestModule {
    unsigned clock[LIMIT];
};
static CUstream streams[LIMIT];
static int wrong_context, lazy;
static atomic_int bad_specialization;
static atomic_int hold_specialization, entered_specialization;
static int idle_clock(const unsigned *clock) {
    unsigned i;
    for (i = 0; i < LIMIT; ++i)
        if (clock[i] > complete[i])
            return 0;
    return 1;
}
static void merge(unsigned *dst, const unsigned *src) {
    unsigned i;
    for (i = 0; i < LIMIT; ++i)
        if (src[i] > dst[i])
            dst[i] = src[i];
}
static void idle(void) {
    unsigned i;
    for (i = 0; i < LIMIT; ++i)
        if (streams[i])
            CHECK(idle_clock(streams[i]->clock));
}
static void clean(void) {
    CHECK(!allocations && !threads && !mutexes && !conditions && !stream_count && !event_count &&
          !module_count);
    next_stream = 0;
    memset(complete, 0, sizeof(complete));
    ++scenarios;
}
void *test_malloc(size_t n) {
    void *p;
    if (fail("malloc"))
        return NULL;
    p = malloc(n);
    CHECK(p);
    ++allocations;
    return p;
}
void *test_calloc(size_t n, size_t sz) {
    void *p;
    if (fail("calloc"))
        return NULL;
    p = calloc(n, sz);
    CHECK(p);
    ++allocations;
    return p;
}
void test_free(void *p) {
    if (p) {
        CHECK(allocations);
        --allocations;
        free(p);
    }
}
int test_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    int e;
    if (fail("pthread_create"))
        return EAGAIN;
    e = pthread_create(t, a, fn, arg);
    if (!e)
        ++threads;
    return e;
}
int test_pthread_join(pthread_t t, void **p) {
    int e;
    if (fail("pthread_join"))
        return EBUSY;
    e = pthread_join(t, p);
    if (!e)
        --threads;
    return e;
}
int test_pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
    int e;
    if (fail("pthread_mutex_init"))
        return EAGAIN;
    e = pthread_mutex_init(m, a);
    if (!e)
        ++mutexes;
    return e;
}
int test_pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) {
    int e;
    if (fail("pthread_cond_init"))
        return EAGAIN;
    e = pthread_cond_init(c, a);
    if (!e)
        ++conditions;
    return e;
}
int test_pthread_mutex_destroy(pthread_mutex_t *m) {
    int e;
    if (fail("pthread_mutex_destroy"))
        return EBUSY;
    e = pthread_mutex_destroy(m);
    if (!e)
        --mutexes;
    return e;
}
int test_pthread_cond_destroy(pthread_cond_t *c) {
    int e;
    if (fail("pthread_cond_destroy"))
        return EBUSY;
    e = pthread_cond_destroy(c);
    if (!e)
        --conditions;
    return e;
}
CUresult cuCtxGetCurrent(CUcontext *c) {
    DRIVER();
    *c = (void *)(uintptr_t)(wrong_context ? 2 : 1);
    return 0;
}
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *m) {
    DRIVER();
    *m = lazy ? 0 : CU_MODULE_EAGER_LOADING;
    return 0;
}
CUresult cuStreamCreate(CUstream *s, unsigned flags) {
    (void)flags;
    DRIVER();
    CHECK(next_stream < LIMIT);
    *s = calloc(1, sizeof(**s));
    CHECK(*s);
    (*s)->id = next_stream++;
    streams[(*s)->id] = *s;
    ++stream_count;
    return 0;
}
CUresult cuStreamDestroy(CUstream s) {
    DRIVER();
    CHECK(idle_clock(s->clock));
    streams[s->id] = NULL;
    free(s);
    --stream_count;
    return 0;
}
CUresult cuStreamSynchronize(CUstream s) {
    DRIVER();
    merge(complete, s->clock);
    return 0;
}
CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned flags) {
    (void)flags;
    DRIVER();
    CHECK(e->recorded);
    merge(s->clock, e->clock);
    return 0;
}
CUresult cuEventCreate(CUevent *e, unsigned flags) {
    (void)flags;
    DRIVER();
    *e = calloc(1, sizeof(**e));
    CHECK(*e);
    ++event_count;
    return 0;
}
CUresult cuEventDestroy(CUevent e) {
    DRIVER();
    free(e);
    --event_count;
    return 0;
}
CUresult cuEventRecord(CUevent e, CUstream s) {
    DRIVER();
    memcpy(e->clock, s->clock, sizeof(e->clock));
    e->recorded = 1;
    return 0;
}
CUresult cuEventSynchronize(CUevent e) {
    DRIVER();
    CHECK(e->recorded);
    merge(complete, e->clock);
    return 0;
}
CUresult cuEventElapsedTime(float *ms, CUevent start, CUevent stop) {
    DRIVER();
    CHECK(start->recorded && stop->recorded && idle_clock(stop->clock));
    *ms = 1;
    return 0;
}
CUresult cuModuleLoadData(CUmodule *m, const void *image) {
    (void)image;
    DRIVER();
    *m = calloc(1, sizeof(**m));
    CHECK(*m);
    ++module_count;
    return 0;
}
CUresult cuModuleUnload(CUmodule m) {
    DRIVER();
    CHECK(idle_clock(m->clock));
    free(m);
    --module_count;
    return 0;
}
CUresult cuModuleGetFunction(CUfunction *fn, CUmodule m, const char *name) {
    (void)name;
    DRIVER();
    *fn = m;
    return 0;
}
CUresult cuLaunchKernel(CUfunction fn, unsigned x, unsigned y, unsigned z, unsigned bx, unsigned by,
                        unsigned bz, unsigned shared, CUstream s, void **args, void **extra) {
    (void)x;
    (void)y;
    (void)z;
    (void)bx;
    (void)by;
    (void)bz;
    (void)shared;
    (void)args;
    (void)extra;
    DRIVER();
    ++s->clock[s->id];
    merge(fn->clock, s->clock);
    return 0;
}
CUresult cuMemsetD2D32Async(CUdeviceptr ptr, size_t pitch, unsigned value, size_t width, size_t height,
                            CUstream s) {
    (void)ptr;
    (void)pitch;
    (void)value;
    (void)width;
    (void)height;
    DRIVER();
    ++s->clock[s->id];
    return 0;
}
SecantResult secant_cubin_specialize_into(const SecantCubinPlan *p, const SecantAstProgramSet *a, void *image,
                                          size_t n) {
    (void)p;
    (void)a;
    (void)image;
    (void)n;
    atomic_store(&entered_specialization, 1);
    while (atomic_load(&hold_specialization)) {
        struct timespec pause = {0, 1000000};
        nanosleep(&pause, NULL);
    }
    return atomic_load(&bad_specialization) ? SECANT_ERROR_BAD_PROGRAM : SECANT_SUCCESS;
}
static struct SecantCubinPlan plan;
static unsigned char cubin[16];
static const SecantAstInstruction ast[] = {SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32, 0,
                                           SECANT_AST_INSTRUCTION_TYPE_RETURN_F32};
static const SecantAstInstruction *asts[13];
static SecantCubinToggleSSERun run;
static SecantResult create(SecantCubinRunner *r) {
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    options.num_workers = 3;
    options.num_streams = 3;
    return secant_cubin_runner_create(&plan, cubin, sizeof(cubin), &options, r);
}
static SecantResult execute(SecantCubinRunner r) {
    SecantCubinToggleSSERun second = run;
    const SecantCubinRunHeader *batch[] = {&run.header, &second.header};
    SecantRunnerStats stats = secant_runner_stats_init();
    second.output.address += 0x10000;
    return secant_cubin_runner_run_batch(r, batch, 2, &stats);
}
static void setup(void) {
    size_t i;
    memset(&plan, 0, sizeof(plan));
    plan.shape = _SECANT_CUBIN_SHAPE_TOGGLE_SSE;
    plan.cubin_size = sizeof(cubin);
    plan.num_kernels = 3;
    plan.asts_per_kernel = 2;
    plan.num_input_registers = 1;
    plan.num_targets = 1;
    plan.tile_rows = 32;
    plan.threads_per_block = 32;
    for (i = 0; i < 13; ++i)
        asts[i] = ast;
    run = secant_cubin_toggle_sse_run_init();
    run.programs.asts.items = asts;
    run.programs.asts.count = 13;
    run.input.address = 0x10000;
    run.input.leading_dimension = 32;
    run.input.num_elements = 32;
    run.targets.address = 0x20000;
    run.targets.leading_dimension = 32;
    run.targets.num_elements = 32;
    run.output.address = 0x30000;
    run.output.leading_dimension = 1;
    run.output.num_elements = 13;
    run.num_rows = 32;
    run.num_targets = 1;
    run.num_banks = 1;
}
static SecantResult expected_error(const char *name) {
    if (!strcmp(name, "malloc") || !strcmp(name, "calloc"))
        return SECANT_ERROR_ALLOCATION_FAILED;
    if (!strncmp(name, "pthread_", 8))
        return SECANT_ERROR_THREAD_FAILED;
    if (!strcmp(name, "cuCtxGetCurrent"))
        return SECANT_ERROR_INVALID_STATE;
    return SECANT_ERROR_DRIVER_FAILED;
}
static void sweep(void) {
    SecantCubinRunner r = NULL;
    unsigned counts[NFAULTS], phase, n;
    size_t i;
    /* Independently fail every operation reached in successful create/run/destroy. */
    for (phase = 0; phase < 3; ++phase) {
        reset_faults();
        if (phase)
            CHECK(create(&r) == SECANT_SUCCESS);
        if (phase == 2)
            CHECK(execute(r) == SECANT_SUCCESS);
        reset_faults();
        if (phase == 0)
            CHECK(create(&r) == SECANT_SUCCESS);
        else if (phase == 1)
            CHECK(execute(r) == SECANT_SUCCESS);
        else {
            CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
            r = NULL;
        }
        for (i = 0; i < NFAULTS; ++i)
            counts[i] = faults[i].calls;
        if (r)
            CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
        clean();
        for (i = 0; i < NFAULTS; ++i)
            for (n = 1; n <= counts[i]; ++n) {
                SecantResult result;
                reset_faults();
                r = NULL;
                if (phase)
                    CHECK(create(&r) == SECANT_SUCCESS);
                if (phase == 2)
                    CHECK(execute(r) == SECANT_SUCCESS);
                reset_faults();
                arm(faults[i].name, n, 0);
                if (phase == 0) {
                    result = create(&r);
                    CHECK(result != SECANT_SUCCESS && !r);
                } else if (phase == 1) {
                    result = execute(r);
                    CHECK(result != SECANT_SUCCESS);
                    idle();
                    reset_faults();
                    if (r->failed)
                        CHECK(execute(r) == SECANT_ERROR_INVALID_STATE);
                    else
                        CHECK(execute(r) == SECANT_SUCCESS);
                } else {
                    result = secant_cubin_runner_destroy(r);
                    CHECK(result != SECANT_SUCCESS);
                    reset_faults();
                    if (r->shutdown)
                        CHECK(execute(r) == SECANT_ERROR_INVALID_STATE);
                }
                CHECK(result == expected_error(faults[i].name));
                reset_faults();
                if (r)
                    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
                clean();
            }
    }
}
static void compound(void) {
    SecantCubinRunner r = NULL;
    unsigned which;
    /* Failure to establish completion retains the live module and blocks reuse. */
    for (which = 0; which < 2; ++which) {
        reset_faults();
        CHECK(create(&r) == SECANT_SUCCESS);
        arm("cuEventRecord", 2, 1);
        arm("cuStreamSynchronize", 1, 1);
        CHECK(execute(r) == SECANT_ERROR_COMPLETION_UNKNOWN);
        CHECK(r->failed && r->work_pending && r->module && module_count == 1);
        CHECK(execute(r) == SECANT_ERROR_INVALID_STATE);
        CHECK(secant_cubin_runner_destroy(r) == SECANT_ERROR_COMPLETION_UNKNOWN);
        CHECK(module_count == 1 && allocations && !threads);
        reset_faults();
        if (which)
            arm("cuModuleUnload", 1, 0);
        if (which) {
            CHECK(secant_cubin_runner_destroy(r) == SECANT_ERROR_DRIVER_FAILED);
            CHECK(module_count == 1);
        }
        CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
        clean();
    }
    /* Event failure falls back only to owned streams, and healthy runners recover. */
    reset_faults();
    CHECK(create(&r) == SECANT_SUCCESS);
    arm("cuEventRecord", 2, 1);
    CHECK(execute(r) == SECANT_ERROR_DRIVER_FAILED);
    CHECK(fault("cuStreamSynchronize")->calls == 3);
    idle();
    reset_faults();
    CHECK(execute(r) == SECANT_SUCCESS);
    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
    clean();
    /* Failed construction plus failed cleanup must return an owned destroy-only handle. */
    reset_faults();
    arm("pthread_create", 2, 0);
    arm("pthread_join", 1, 0);
    CHECK(create(&r) == SECANT_ERROR_THREAD_FAILED && r && threads == 1);
    CHECK(execute(r) == SECANT_ERROR_INVALID_STATE);
    reset_faults();
    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
    clean();
    reset_faults();
    arm("cuEventCreate", 2, 0);
    arm("cuStreamDestroy", 1, 0);
    CHECK(create(&r) == SECANT_ERROR_DRIVER_FAILED && r && stream_count == 1);
    reset_faults();
    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
    clean();
    reset_faults();
    CHECK(create(&r) == SECANT_SUCCESS);
    atomic_store(&bad_specialization, 1);
    CHECK(execute(r) == SECANT_ERROR_BAD_PROGRAM);
    idle();
    atomic_store(&bad_specialization, 0);
    CHECK(execute(r) == SECANT_SUCCESS);
    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
    clean();
    reset_faults();
    CHECK(create(&r) == SECANT_SUCCESS);
    wrong_context = 1;
    CHECK(execute(r) == SECANT_ERROR_INVALID_STATE);
    CHECK(secant_cubin_runner_destroy(r) == SECANT_ERROR_INVALID_STATE);
    wrong_context = 0;
    CHECK(execute(r) == SECANT_SUCCESS);
    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
    clean();
    reset_faults();
    lazy = 1;
    CHECK(create(&r) == SECANT_ERROR_EAGER_LOADING_REQUIRED && !r);
    lazy = 0;
    clean();
}
static void *execute_thread(void *opaque) {
    CHECK(execute((SecantCubinRunner)opaque) == SECANT_SUCCESS);
    return NULL;
}
static void busy_runner(void) {
    SecantCubinRunner r;
    pthread_t thread;
    reset_faults();
    CHECK(create(&r) == SECANT_SUCCESS);
    atomic_store(&entered_specialization, 0);
    atomic_store(&hold_specialization, 1);
    CHECK(!pthread_create(&thread, NULL, execute_thread, r));
    while (!atomic_load(&entered_specialization)) {
        struct timespec pause = {0, 1000000};
        nanosleep(&pause, NULL);
    }
    CHECK(execute(r) == SECANT_ERROR_INVALID_STATE);
    CHECK(secant_cubin_runner_destroy(r) == SECANT_ERROR_INVALID_STATE);
    atomic_store(&hold_specialization, 0);
    CHECK(!pthread_join(thread, NULL));
    CHECK(execute(r) == SECANT_SUCCESS);
    CHECK(secant_cubin_runner_destroy(r) == SECANT_SUCCESS);
    clean();
}
int main(void) {
    setup();
    sweep();
    compound();
    busy_runner();
    printf("runner failure coverage passed: %u scenarios\n", scenarios);
    return 0;
}
