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
#ifndef SECANT_RUNNER_INTERNAL_H_INCLUDED
#define SECANT_RUNNER_INTERNAL_H_INCLUDED
#include "s_cubin_internal.h"
#include "s_toggle_internal.h"
#include <cuda.h>
#include <pthread.h>

typedef struct SRun {
    SecantAstProgramSet programs;
    SecantDeviceMatrixF32 input, targets, output;
    SecantDeviceConstantBanks banks;
    size_t rows, targets_count, configurations, output_count, output_width;
    size_t input_span, target_span, bank_span, output_span;
    uint32_t toggle_bits;
} SRun;
typedef struct SSlot {
    unsigned char *cubin;
    size_t module_index, ast_count;
    int state; /* 0 free, 1 specializing, 2 ready, 3 in use */
    SecantResult result;
} SSlot;
typedef struct SWorker {
    struct SecantCubinRunnerImpl *runner;
    pthread_t thread;
    int started;
    double work_seconds;
} SWorker;
struct SecantCubinRunnerImpl {
    const SecantCubinPlan *plan;
    unsigned char *template_cubin;
    size_t module_capacity, slot_count, worker_count, stream_count;
    SSlot *slots;
    SWorker *workers;
    CUcontext context;
    CUstream *streams;
    CUevent *done;
    CUevent start, stop;
    CUfunction *functions;
    /* Retained until completion and unload succeed, including failed calls. */
    CUmodule module;
    int work_pending, failed;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int mutex_ready, condition_ready, shutdown, active, cancel;
    size_t next_module, module_count, compiling, compiled;
    SecantAstProgramSet programs;
    double started_at, compiled_at;
};

double s_runner_seconds(void);
SecantResult s_pipeline_create(SecantCubinRunner runner);
SecantResult s_pipeline_destroy(SecantCubinRunner runner);
SecantResult s_pipeline_execute(SecantCubinRunner runner, const SRun *runs, size_t count,
                                SecantRunnerStats *stats);
#endif
