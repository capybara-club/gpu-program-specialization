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
#ifndef SECANT_AMD_RUNNER_INTERNAL_H_INCLUDED
#define SECANT_AMD_RUNNER_INTERNAL_H_INCLUDED

#include "secant.h"
#include "secant_runner.h"

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>
#include <pthread.h>

#include <stddef.h>
#include <stdint.h>

typedef enum _SecantAmdRunnerShape {
    _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE = 1,
    _SECANT_AMD_RUNNER_SHAPE_SSE = 2
} _SecantAmdRunnerShape;

typedef enum _SecantAmdSlotState {
    _SECANT_AMD_SLOT_AVAILABLE = 0,
    _SECANT_AMD_SLOT_COMPILING = 1,
    _SECANT_AMD_SLOT_READY = 2,
    _SECANT_AMD_SLOT_LOADED = 3
} _SecantAmdSlotState;

typedef struct _SecantAmdRunnerSlot {
    size_t module_idx;
    _SecantAmdSlotState state;
    void* artifact;
    const void* binary;
    size_t binary_size;
    hipModule_t module;
    hipFunction_t* functions;
} _SecantAmdRunnerSlot;

typedef struct _SecantAmdRunner _SecantAmdRunner;

typedef SecantResult (*_SecantAmdCompileFn)(
    void* backend,
    size_t worker_idx,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    _SecantAmdRunnerSlot* slot
);

typedef SecantResult (*_SecantAmdSlotCreateFn)(
    void* backend,
    size_t slot_idx,
    _SecantAmdRunnerSlot* slot
);

typedef void (*_SecantAmdSlotReleaseFn)(
    void* backend,
    _SecantAmdRunnerSlot* slot
);

typedef struct _SecantAmdWorker {
    _SecantAmdRunner* runner;
    pthread_t thread;
    size_t worker_idx;
    double compile_work_seconds;
    int started;
} _SecantAmdWorker;

struct _SecantAmdRunner {
    _SecantAmdRunnerShape shape;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t asts_per_module;
    size_t num_inputs;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t num_workers;
    size_t num_streams;
    size_t num_slots;
    const char* function_name_prefix;
    hipStream_t* streams;
    hipEvent_t* done_events;
    hipEvent_t start_event;
    hipEvent_t stop_event;
    _SecantAmdRunnerSlot* slots;
    size_t* ready_slots;
    size_t ready_head;
    size_t ready_tail;
    size_t ready_count;
    _SecantAmdWorker* workers;
    pthread_mutex_t mutex;
    pthread_cond_t work_condition;
    pthread_cond_t ready_condition;
    pthread_cond_t state_condition;
    int mutex_initialized;
    int work_condition_initialized;
    int ready_condition_initialized;
    int state_condition_initialized;
    int shutdown;
    int run_active;
    int failed;
    SecantResult failure_result;
    size_t next_module_idx;
    size_t num_modules;
    size_t compile_completed;
    size_t active_compiles;
    double run_begin_seconds;
    double compile_finish_seconds;
    double compile_work_seconds;
    const SecantAstInstruction* const* routines;
    size_t num_routines;
    const char* const* routine_names;
    const SecantAstInstruction* const* asts;
    void* backend;
    _SecantAmdCompileFn compile;
    _SecantAmdSlotCreateFn slot_create;
    _SecantAmdSlotReleaseFn slot_release;
};

SecantResult _secant_amd_runner_create(
    _SecantAmdRunnerShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t num_workers,
    size_t num_streams,
    const char* function_name_prefix,
    void* backend,
    _SecantAmdCompileFn compile,
    _SecantAmdSlotCreateFn slot_create,
    _SecantAmdSlotReleaseFn slot_release,
    _SecantAmdRunner** runner_ret
);

SecantResult _secant_amd_materialize_runner_run_all(
    _SecantAmdRunner* runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_device_address,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_rows,
    uintptr_t output_device_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    size_t output_module_stride,
    SecantRunnerStats* stats_ret
);

SecantResult _secant_amd_sse_runner_run_all(
    _SecantAmdRunner* runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_device_address,
    size_t input_num_elements,
    size_t input_leading_dimension,
    uintptr_t targets_device_address,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    uintptr_t output_device_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    SecantRunnerStats* stats_ret
);

SecantResult _secant_amd_runner_destroy(
    _SecantAmdRunner* runner
);

#endif /* SECANT_AMD_RUNNER_INTERNAL_H_INCLUDED */
