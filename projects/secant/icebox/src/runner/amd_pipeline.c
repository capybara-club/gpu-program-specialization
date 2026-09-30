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
#include "amd_internal.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define _SECANT_AMD_RUNNER_NAME_BYTES 256u
#define _SECANT_AMD_RUNNER_ERROR_RET(ans) do { SecantResult _secant_amd_runner_result = (ans); return _secant_amd_runner_result; } while (0)

static double
_secant_amd_runner_seconds(void) {
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int
_secant_amd_runner_checked_add(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static int
_secant_amd_runner_checked_mul(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
_secant_amd_runner_span_required(
    size_t outer_count,
    size_t leading_dimension,
    size_t inner_count,
    size_t* required_ret
) {
    size_t offset;

    return outer_count != 0u && inner_count != 0u &&
        _secant_amd_runner_checked_mul(
            outer_count - 1u,
            leading_dimension,
            &offset) &&
        _secant_amd_runner_checked_add(
            offset,
            inner_count,
            required_ret);
}

static void
_secant_amd_runner_failure_set(
    _SecantAmdRunner* runner,
    SecantResult result
) {
    (void)pthread_mutex_lock(&runner->mutex);
    if (!runner->failed) {
        runner->failed = 1;
        runner->failure_result = result;
    }
    (void)pthread_cond_broadcast(&runner->work_condition);
    (void)pthread_cond_broadcast(&runner->ready_condition);
    (void)pthread_cond_broadcast(&runner->state_condition);
    (void)pthread_mutex_unlock(&runner->mutex);
}

static size_t
_secant_amd_runner_available_slot_find(
    const _SecantAmdRunner* runner
) {
    size_t slot_idx;

    for (slot_idx = 0u; slot_idx < runner->num_slots; ++slot_idx) {
        if (runner->slots[slot_idx].state ==
            _SECANT_AMD_SLOT_AVAILABLE) {
            return slot_idx;
        }
    }
    return SIZE_MAX;
}

static void*
_secant_amd_runner_worker_main(void* argument) {
    _SecantAmdWorker* worker = (_SecantAmdWorker*)argument;
    _SecantAmdRunner* runner = worker->runner;

    for (;;) {
        _SecantAmdRunnerSlot* slot;
        const SecantAstInstruction* const* asts;
        size_t module_idx;
        size_t slot_idx;
        double begin;
        double end;
        SecantResult result;

        (void)pthread_mutex_lock(&runner->mutex);
        for (;;) {
            slot_idx = _secant_amd_runner_available_slot_find(
                runner);
            if (runner->shutdown) {
                (void)pthread_mutex_unlock(&runner->mutex);
                return NULL;
            }
            if (runner->run_active && !runner->failed &&
                runner->next_module_idx < runner->num_modules &&
                slot_idx != SIZE_MAX) {
                break;
            }
            (void)pthread_cond_wait(
                &runner->work_condition,
                &runner->mutex);
        }
        module_idx = runner->next_module_idx++;
        slot = runner->slots + slot_idx;
        slot->module_idx = module_idx;
        slot->state = _SECANT_AMD_SLOT_COMPILING;
        runner->active_compiles += 1u;
        asts =
            runner->asts + module_idx * runner->asts_per_module;
        (void)pthread_mutex_unlock(&runner->mutex);

        begin = _secant_amd_runner_seconds();
        result = runner->compile(
            runner->backend,
            worker->worker_idx,
            runner->routines,
            runner->num_routines,
            runner->routine_names,
            asts,
            slot);
        end = _secant_amd_runner_seconds();

        (void)pthread_mutex_lock(&runner->mutex);
        runner->active_compiles -= 1u;
        runner->compile_completed += 1u;
        runner->compile_work_seconds += end - begin;
        worker->compile_work_seconds += end - begin;
        if (end > runner->compile_finish_seconds) {
            runner->compile_finish_seconds = end;
        }
        if (result == SECANT_SUCCESS && !runner->failed) {
            slot->state = _SECANT_AMD_SLOT_READY;
            runner->ready_slots[runner->ready_tail] = slot_idx;
            runner->ready_tail =
                (runner->ready_tail + 1u) % runner->num_slots;
            runner->ready_count += 1u;
            (void)pthread_cond_signal(&runner->ready_condition);
        } else {
            if (result == SECANT_SUCCESS) {
                runner->slot_release(runner->backend, slot);
            }
            slot->state = _SECANT_AMD_SLOT_AVAILABLE;
            if (!runner->failed) {
                runner->failed = 1;
                runner->failure_result = result;
            }
            (void)pthread_cond_broadcast(&runner->ready_condition);
        }
        (void)pthread_cond_broadcast(&runner->state_condition);
        (void)pthread_cond_broadcast(&runner->work_condition);
        (void)pthread_mutex_unlock(&runner->mutex);
    }
}

static SecantResult
_secant_amd_runner_ready_pop(
    _SecantAmdRunner* runner,
    size_t* slot_idx_ret
) {
    (void)pthread_mutex_lock(&runner->mutex);
    while (runner->ready_count == 0u &&
           runner->compile_completed < runner->num_modules &&
           !runner->failed) {
        (void)pthread_cond_wait(
            &runner->ready_condition,
            &runner->mutex);
    }
    if (runner->failed) {
        const SecantResult result = runner->failure_result;

        (void)pthread_mutex_unlock(&runner->mutex);
        return result;
    }
    if (runner->ready_count == 0u) {
        (void)pthread_mutex_unlock(&runner->mutex);
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_STATE);
    }
    *slot_idx_ret = runner->ready_slots[runner->ready_head];
    runner->ready_head =
        (runner->ready_head + 1u) % runner->num_slots;
    runner->ready_count -= 1u;
    (void)pthread_mutex_unlock(&runner->mutex);
    return SECANT_SUCCESS;
}

static SecantResult
_secant_amd_runner_slot_release(
    _SecantAmdRunner* runner,
    _SecantAmdRunnerSlot* slot
) {
    SecantResult result = SECANT_SUCCESS;

    if (slot->module != NULL) {
        if (hipModuleUnload(slot->module) != hipSuccess) {
            result = SECANT_ERROR_DRIVER_FAILED;
        }
        slot->module = NULL;
    }
    memset(
        slot->functions,
        0,
        runner->num_kernels * sizeof(*slot->functions));
    runner->slot_release(runner->backend, slot);
    (void)pthread_mutex_lock(&runner->mutex);
    slot->state = _SECANT_AMD_SLOT_AVAILABLE;
    (void)pthread_cond_broadcast(&runner->work_condition);
    (void)pthread_mutex_unlock(&runner->mutex);
    return result;
}

static SecantResult
_secant_amd_runner_module_load(
    _SecantAmdRunner* runner,
    _SecantAmdRunnerSlot* slot,
    SecantRunnerStats* stats
) {
    double begin = _secant_amd_runner_seconds();
    size_t function_idx;

    if (slot->binary == NULL || slot->binary_size == 0u ||
        hipModuleLoadData(&slot->module, slot->binary) != hipSuccess) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    for (function_idx = 0u;
         function_idx < runner->num_kernels;
         ++function_idx) {
        char name[_SECANT_AMD_RUNNER_NAME_BYTES];
        const int name_bytes = snprintf(
            name,
            sizeof(name),
            "%s_%03zu",
            runner->function_name_prefix,
            function_idx);

        if (name_bytes < 0 ||
            (size_t)name_bytes >= sizeof(name) ||
            hipModuleGetFunction(
                slot->functions + function_idx,
                slot->module,
                name) != hipSuccess) {
            (void)hipModuleUnload(slot->module);
            slot->module = NULL;
            _SECANT_AMD_RUNNER_ERROR_RET(
                name_bytes < 0
                    ? SECANT_ERROR_FORMAT
                    : SECANT_ERROR_FUNCTION_NOT_FOUND);
        }
    }
    (void)pthread_mutex_lock(&runner->mutex);
    slot->state = _SECANT_AMD_SLOT_LOADED;
    (void)pthread_mutex_unlock(&runner->mutex);
    stats->module_load_seconds +=
        _secant_amd_runner_seconds() - begin;
    stats->modules_loaded += 1u;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_amd_runner_module_launch(
    _SecantAmdRunner* runner,
    const _SecantAmdRunnerSlot* slot,
    uintptr_t input_device_address,
    size_t input_leading_dimension,
    uintptr_t targets_device_address,
    size_t targets_leading_dimension,
    size_t num_rows,
    uintptr_t output_device_address,
    size_t output_leading_dimension,
    size_t output_module_stride
) {
    const float* input = (const float*)input_device_address;
    const float* targets = (const float*)targets_device_address;
    size_t kernel_idx;
    size_t stream_idx;

    if (hipEventRecord(
            runner->start_event,
            runner->streams[0]) != hipSuccess) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    for (stream_idx = 1u;
         stream_idx < runner->num_streams;
         ++stream_idx) {
        if (hipStreamWaitEvent(
                runner->streams[stream_idx],
                runner->start_event,
                0u) != hipSuccess) {
            _SECANT_AMD_RUNNER_ERROR_RET(
                SECANT_ERROR_DRIVER_FAILED);
        }
    }
    for (kernel_idx = 0u;
         kernel_idx < runner->num_kernels;
         ++kernel_idx) {
        const size_t module_output_idx =
            slot->module_idx * output_module_stride;
        const size_t kernel_output_idx =
            module_output_idx +
            kernel_idx * runner->asts_per_kernel *
                output_leading_dimension;
        float* output =
            (float*)output_device_address + kernel_output_idx;
        void* arguments[7];
        unsigned int grid_x;

        stream_idx = kernel_idx % runner->num_streams;
        arguments[0] = &input;
        arguments[1] = &input_leading_dimension;
        if (runner->shape == _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE) {
            const unsigned int block_x = 256u;

            grid_x =
                (unsigned int)((num_rows + block_x - 1u) / block_x);
            arguments[2] = &num_rows;
            arguments[3] = &output;
            arguments[4] = &output_leading_dimension;
            if (hipModuleLaunchKernel(
                    slot->functions[kernel_idx],
                    grid_x, 1u, 1u,
                    block_x, 1u, 1u,
                    0u,
                    runner->streams[stream_idx],
                    arguments,
                    NULL) != hipSuccess) {
                _SECANT_AMD_RUNNER_ERROR_RET(
                    SECANT_ERROR_DRIVER_FAILED);
            }
        } else {
            grid_x = (unsigned int)(
                num_rows / runner->tile_rows +
                (num_rows % runner->tile_rows != 0u ? 1u : 0u));
            arguments[2] = &targets;
            arguments[3] = &targets_leading_dimension;
            arguments[4] = &num_rows;
            arguments[5] = &output;
            arguments[6] = &output_leading_dimension;
            if (hipModuleLaunchKernel(
                    slot->functions[kernel_idx],
                    grid_x, 1u, 1u,
                    (unsigned int)runner->threads_per_block,
                    1u, 1u,
                    0u,
                    runner->streams[stream_idx],
                    arguments,
                    NULL) != hipSuccess) {
                _SECANT_AMD_RUNNER_ERROR_RET(
                    SECANT_ERROR_DRIVER_FAILED);
            }
        }
    }
    for (stream_idx = 0u;
         stream_idx < runner->num_streams;
         ++stream_idx) {
        if (hipEventRecord(
                runner->done_events[stream_idx],
                runner->streams[stream_idx]) != hipSuccess) {
            _SECANT_AMD_RUNNER_ERROR_RET(
                SECANT_ERROR_DRIVER_FAILED);
        }
    }
    for (stream_idx = 1u;
         stream_idx < runner->num_streams;
         ++stream_idx) {
        if (hipStreamWaitEvent(
                runner->streams[0],
                runner->done_events[stream_idx],
                0u) != hipSuccess) {
            _SECANT_AMD_RUNNER_ERROR_RET(
                SECANT_ERROR_DRIVER_FAILED);
        }
    }
    if (hipEventRecord(
            runner->stop_event,
            runner->streams[0]) != hipSuccess) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_amd_runner_module_finish(
    _SecantAmdRunner* runner,
    _SecantAmdRunnerSlot* slot,
    SecantRunnerStats* stats
) {
    float milliseconds = 0.0f;
    double begin;
    SecantResult result;

    begin = _secant_amd_runner_seconds();
    if (hipEventSynchronize(runner->stop_event) != hipSuccess) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    stats->completion_wait_seconds +=
        _secant_amd_runner_seconds() - begin;
    if (hipEventElapsedTime(
            &milliseconds,
            runner->start_event,
            runner->stop_event) != hipSuccess) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    stats->runtime_seconds += (double)milliseconds * 1.0e-3;
    begin = _secant_amd_runner_seconds();
    result = _secant_amd_runner_slot_release(runner, slot);
    stats->module_unload_seconds +=
        _secant_amd_runner_seconds() - begin;
    return result;
}

static void
_secant_amd_runner_run_abort(
    _SecantAmdRunner* runner
) {
    size_t slot_idx;
    size_t stream_idx;

    (void)pthread_mutex_lock(&runner->mutex);
    runner->failed = 1;
    if (runner->failure_result == SECANT_SUCCESS) {
        runner->failure_result = SECANT_ERROR_INVALID_STATE;
    }
    (void)pthread_cond_broadcast(&runner->work_condition);
    while (runner->active_compiles != 0u) {
        (void)pthread_cond_wait(
            &runner->state_condition,
            &runner->mutex);
    }
    (void)pthread_mutex_unlock(&runner->mutex);

    for (stream_idx = 0u;
         stream_idx < runner->num_streams;
         ++stream_idx) {
        (void)hipStreamSynchronize(runner->streams[stream_idx]);
    }
    for (slot_idx = 0u; slot_idx < runner->num_slots; ++slot_idx) {
        if (runner->slots[slot_idx].state !=
            _SECANT_AMD_SLOT_AVAILABLE) {
            (void)_secant_amd_runner_slot_release(
                runner,
                runner->slots + slot_idx);
        }
    }
}

static SecantResult
_secant_amd_runner_execute(
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
    size_t output_module_stride,
    SecantRunnerStats* stats_ret
) {
    SecantRunnerStats stats;
    size_t required_input;
    size_t required_targets = 0u;
    size_t required_output;
    size_t current_slot_idx = SIZE_MAX;
    size_t consumed_modules = 0u;
    size_t output_inner_count;
    size_t output_module_span;
    size_t output_bytes;
    size_t worker_idx;
    SecantResult result = SECANT_SUCCESS;

    memset(&stats, 0, sizeof(stats));
    if (runner == NULL || asts == NULL || num_asts == 0u ||
        input_device_address == 0u || output_device_address == 0u ||
        num_rows == 0u || input_leading_dimension < num_rows ||
        num_asts % runner->asts_per_module != 0u ||
        (num_routines != 0u && routines == NULL)) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_amd_runner_span_required(
            runner->num_inputs,
            input_leading_dimension,
            num_rows,
            &required_input) ||
        required_input > input_num_elements) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (runner->shape == _SECANT_AMD_RUNNER_SHAPE_SSE) {
        if (targets_device_address == 0u ||
            targets_leading_dimension < num_rows ||
            output_leading_dimension < runner->num_targets ||
            !_secant_amd_runner_span_required(
                runner->num_targets,
                targets_leading_dimension,
                num_rows,
                &required_targets) ||
            required_targets > targets_num_elements) {
            _SECANT_AMD_RUNNER_ERROR_RET(
                SECANT_ERROR_INVALID_VALUE);
        }
        output_inner_count = runner->num_targets;
    } else {
        if (output_leading_dimension < num_rows) {
            _SECANT_AMD_RUNNER_ERROR_RET(
                SECANT_ERROR_INVALID_VALUE);
        }
        output_inner_count = num_rows;
    }
    if (!_secant_amd_runner_span_required(
            runner->asts_per_module,
            output_leading_dimension,
            output_inner_count,
            &output_module_span)) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    if (output_module_stride != 0u &&
        output_module_stride < output_module_span) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_amd_runner_span_required(
            num_asts / runner->asts_per_module,
            output_module_stride,
            output_module_span,
            &required_output) ||
        required_output > output_num_elements ||
        !_secant_amd_runner_checked_mul(
            required_output,
            sizeof(float),
            &output_bytes) ||
        (runner->shape == _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE &&
         num_rows > (size_t)UINT_MAX * 256u) ||
        (runner->shape == _SECANT_AMD_RUNNER_SHAPE_SSE &&
         (num_rows / runner->tile_rows +
          (num_rows % runner->tile_rows != 0u ? 1u : 0u)) >
            UINT_MAX)) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }

    (void)pthread_mutex_lock(&runner->mutex);
    if (runner->run_active || runner->shutdown) {
        (void)pthread_mutex_unlock(&runner->mutex);
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_STATE);
    }
    runner->run_active = 1;
    runner->failed = 0;
    runner->failure_result = SECANT_SUCCESS;
    runner->next_module_idx = 0u;
    runner->num_modules = num_asts / runner->asts_per_module;
    runner->compile_completed = 0u;
    runner->active_compiles = 0u;
    runner->ready_head = 0u;
    runner->ready_tail = 0u;
    runner->ready_count = 0u;
    runner->compile_work_seconds = 0.0;
    runner->compile_finish_seconds = 0.0;
    for (worker_idx = 0u;
         worker_idx < runner->num_workers;
         ++worker_idx) {
        runner->workers[worker_idx].compile_work_seconds = 0.0;
    }
    runner->routines = routines;
    runner->num_routines = num_routines;
    runner->routine_names = routine_names;
    runner->asts = asts;
    runner->run_begin_seconds = _secant_amd_runner_seconds();
    stats.num_modules = runner->num_modules;
    stats.num_asts = num_asts;
    if (runner->shape == _SECANT_AMD_RUNNER_SHAPE_SSE &&
        hipMemsetAsync(
            (void*)output_device_address,
            0,
            output_bytes,
            runner->streams[0]) != hipSuccess) {
        runner->run_active = 0;
        (void)pthread_mutex_unlock(&runner->mutex);
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    (void)pthread_cond_broadcast(&runner->work_condition);
    (void)pthread_mutex_unlock(&runner->mutex);

    while (consumed_modules < stats.num_modules) {
        size_t next_slot_idx = SIZE_MAX;

        result = _secant_amd_runner_ready_pop(
            runner,
            &next_slot_idx);
        if (result != SECANT_SUCCESS) {
            break;
        }
        if (current_slot_idx != SIZE_MAX) {
            result = _secant_amd_runner_module_finish(
                runner,
                runner->slots + current_slot_idx,
                &stats);
            current_slot_idx = SIZE_MAX;
            if (result != SECANT_SUCCESS) {
                break;
            }
        }
        result = _secant_amd_runner_module_load(
            runner,
            runner->slots + next_slot_idx,
            &stats);
        if (result != SECANT_SUCCESS) {
            break;
        }
        result = _secant_amd_runner_module_launch(
            runner,
            runner->slots + next_slot_idx,
            input_device_address,
            input_leading_dimension,
            targets_device_address,
            targets_leading_dimension,
            num_rows,
            output_device_address,
            output_leading_dimension,
            output_module_stride);
        if (result != SECANT_SUCCESS) {
            break;
        }
        current_slot_idx = next_slot_idx;
        consumed_modules += 1u;
    }
    if (result == SECANT_SUCCESS && current_slot_idx != SIZE_MAX) {
        result = _secant_amd_runner_module_finish(
            runner,
            runner->slots + current_slot_idx,
            &stats);
        current_slot_idx = SIZE_MAX;
    }
    if (result != SECANT_SUCCESS) {
        _secant_amd_runner_failure_set(runner, result);
        _secant_amd_runner_run_abort(runner);
    }

    (void)pthread_mutex_lock(&runner->mutex);
    while (runner->active_compiles != 0u) {
        (void)pthread_cond_wait(
            &runner->state_condition,
            &runner->mutex);
    }
    if (result == SECANT_SUCCESS && runner->failed) {
        result = runner->failure_result;
    }
    stats.compile_work_seconds = runner->compile_work_seconds;
    for (worker_idx = 0u;
         worker_idx < runner->num_workers;
         ++worker_idx) {
        if (runner->workers[worker_idx].compile_work_seconds >
            stats.compile_critical_seconds) {
            stats.compile_critical_seconds =
                runner->workers[worker_idx].compile_work_seconds;
        }
    }
    if (runner->compile_finish_seconds >=
        runner->run_begin_seconds) {
        stats.compile_window_seconds =
            runner->compile_finish_seconds -
            runner->run_begin_seconds;
    }
    stats.total_seconds =
        _secant_amd_runner_seconds() - runner->run_begin_seconds;
    runner->run_active = 0;
    runner->routines = NULL;
    runner->num_routines = 0u;
    runner->routine_names = NULL;
    runner->asts = NULL;
    (void)pthread_cond_broadcast(&runner->work_condition);
    (void)pthread_mutex_unlock(&runner->mutex);

    if (stats_ret != NULL) {
        *stats_ret = stats;
    }
    if (result == SECANT_SUCCESS &&
        (consumed_modules != stats.num_modules ||
         stats.modules_loaded != stats.num_modules)) {
        result = SECANT_ERROR_INVALID_STATE;
    }
    return result;
}

static SecantResult
_secant_amd_runner_resources_create(
    _SecantAmdRunner* runner
) {
    size_t slot_idx;
    size_t stream_idx;
    size_t worker_idx;
    SecantResult result = SECANT_SUCCESS;

    if (pthread_mutex_init(&runner->mutex, NULL) != 0) {
        return SECANT_ERROR_THREAD_FAILED;
    }
    runner->mutex_initialized = 1;
    if (pthread_cond_init(&runner->work_condition, NULL) != 0) {
        return SECANT_ERROR_THREAD_FAILED;
    }
    runner->work_condition_initialized = 1;
    if (pthread_cond_init(&runner->ready_condition, NULL) != 0) {
        return SECANT_ERROR_THREAD_FAILED;
    }
    runner->ready_condition_initialized = 1;
    if (pthread_cond_init(&runner->state_condition, NULL) != 0) {
        return SECANT_ERROR_THREAD_FAILED;
    }
    runner->state_condition_initialized = 1;
    if (hipEventCreate(&runner->start_event) != hipSuccess ||
        hipEventCreate(&runner->stop_event) != hipSuccess) {
        return SECANT_ERROR_DRIVER_FAILED;
    }
    for (stream_idx = 0u;
         stream_idx < runner->num_streams;
         ++stream_idx) {
        if (hipStreamCreateWithFlags(
                runner->streams + stream_idx,
                hipStreamNonBlocking) != hipSuccess ||
            hipEventCreateWithFlags(
                runner->done_events + stream_idx,
                hipEventDisableTiming) != hipSuccess) {
            return SECANT_ERROR_DRIVER_FAILED;
        }
    }
    for (slot_idx = 0u; slot_idx < runner->num_slots; ++slot_idx) {
        runner->slots[slot_idx].functions =
            (hipFunction_t*)calloc(
                runner->num_kernels,
                sizeof(*runner->slots[slot_idx].functions));
        if (runner->slots[slot_idx].functions == NULL) {
            return SECANT_ERROR_ALLOCATION_FAILED;
        }
        result = runner->slot_create(
            runner->backend,
            slot_idx,
            runner->slots + slot_idx);
        if (result != SECANT_SUCCESS) {
            return result;
        }
    }
    for (worker_idx = 0u;
         worker_idx < runner->num_workers;
         ++worker_idx) {
        runner->workers[worker_idx].runner = runner;
        runner->workers[worker_idx].worker_idx = worker_idx;
        if (pthread_create(
                &runner->workers[worker_idx].thread,
                NULL,
                _secant_amd_runner_worker_main,
                runner->workers + worker_idx) != 0) {
            return SECANT_ERROR_THREAD_FAILED;
        }
        runner->workers[worker_idx].started = 1;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_amd_runner_create(
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
) {
    _SecantAmdRunner* runner;
    size_t asts_per_module;
    SecantResult result;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_inputs == 0u || num_workers == 0u ||
        num_streams == 0u || function_name_prefix == NULL ||
        function_name_prefix[0] == '\0' || backend == NULL ||
        compile == NULL || slot_create == NULL ||
        slot_release == NULL || runner_ret == NULL ||
        num_workers > SIZE_MAX - 2u ||
        !_secant_amd_runner_checked_mul(
            num_kernels,
            asts_per_kernel,
            &asts_per_module)) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if ((shape != _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE &&
         shape != _SECANT_AMD_RUNNER_SHAPE_SSE) ||
        (shape == _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE &&
         (num_targets != 0u || tile_rows != 0u ||
          threads_per_block != 0u)) ||
        (shape == _SECANT_AMD_RUNNER_SHAPE_SSE &&
         (num_targets == 0u || tile_rows == 0u ||
          threads_per_block < 64u ||
          threads_per_block > tile_rows ||
          threads_per_block > 1024u ||
          threads_per_block % 64u != 0u))) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *runner_ret = NULL;
    runner = (_SecantAmdRunner*)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_ALLOCATION_FAILED);
    }
    runner->shape = shape;
    runner->num_kernels = num_kernels;
    runner->asts_per_kernel = asts_per_kernel;
    runner->asts_per_module = asts_per_module;
    runner->num_inputs = num_inputs;
    runner->num_targets = num_targets;
    runner->tile_rows = tile_rows;
    runner->threads_per_block = threads_per_block;
    runner->num_workers = num_workers;
    runner->num_streams = num_streams;
    runner->num_slots = num_workers + 2u;
    runner->function_name_prefix = function_name_prefix;
    runner->backend = backend;
    runner->compile = compile;
    runner->slot_create = slot_create;
    runner->slot_release = slot_release;
    runner->streams = (hipStream_t*)calloc(
        num_streams,
        sizeof(*runner->streams));
    runner->done_events = (hipEvent_t*)calloc(
        num_streams,
        sizeof(*runner->done_events));
    runner->slots = (_SecantAmdRunnerSlot*)calloc(
        runner->num_slots,
        sizeof(*runner->slots));
    runner->ready_slots = (size_t*)malloc(
        runner->num_slots * sizeof(*runner->ready_slots));
    runner->workers = (_SecantAmdWorker*)calloc(
        num_workers,
        sizeof(*runner->workers));
    if (runner->streams == NULL || runner->done_events == NULL ||
        runner->slots == NULL || runner->ready_slots == NULL ||
        runner->workers == NULL) {
        (void)_secant_amd_runner_destroy(runner);
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_ALLOCATION_FAILED);
    }
    result = _secant_amd_runner_resources_create(runner);
    if (result != SECANT_SUCCESS) {
        (void)_secant_amd_runner_destroy(runner);
        return result;
    }
    *runner_ret = runner;
    return SECANT_SUCCESS;
}

SecantResult
_secant_amd_materialize_runner_run_all(
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
) {
    if (runner == NULL ||
        runner->shape != _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    return _secant_amd_runner_execute(
        runner,
        routines,
        num_routines,
        routine_names,
        asts,
        num_asts,
        input_device_address,
        input_num_elements,
        input_leading_dimension,
        0u,
        0u,
        0u,
        num_rows,
        output_device_address,
        output_num_elements,
        output_leading_dimension,
        output_module_stride,
        stats_ret);
}

SecantResult
_secant_amd_sse_runner_run_all(
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
) {
    size_t output_module_stride;

    if (runner == NULL ||
        runner->shape != _SECANT_AMD_RUNNER_SHAPE_SSE) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_amd_runner_checked_mul(
            runner->asts_per_module,
            output_leading_dimension,
            &output_module_stride)) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    return _secant_amd_runner_execute(
        runner,
        routines,
        num_routines,
        routine_names,
        asts,
        num_asts,
        input_device_address,
        input_num_elements,
        input_leading_dimension,
        targets_device_address,
        targets_num_elements,
        targets_leading_dimension,
        num_rows,
        output_device_address,
        output_num_elements,
        output_leading_dimension,
        output_module_stride,
        stats_ret);
}

SecantResult
_secant_amd_runner_destroy(
    _SecantAmdRunner* runner
) {
    SecantResult result = SECANT_SUCCESS;
    size_t slot_idx;
    size_t stream_idx;
    size_t worker_idx;

    if (runner == NULL) {
        _SECANT_AMD_RUNNER_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (runner->mutex_initialized) {
        (void)pthread_mutex_lock(&runner->mutex);
        if (runner->run_active) {
            (void)pthread_mutex_unlock(&runner->mutex);
            _SECANT_AMD_RUNNER_ERROR_RET(
                SECANT_ERROR_INVALID_STATE);
        }
        runner->shutdown = 1;
        (void)pthread_cond_broadcast(&runner->work_condition);
        (void)pthread_mutex_unlock(&runner->mutex);
    }
    for (worker_idx = 0u;
         worker_idx < runner->num_workers;
         ++worker_idx) {
        if (runner->workers != NULL &&
            runner->workers[worker_idx].started &&
            pthread_join(
                runner->workers[worker_idx].thread,
                NULL) != 0) {
            result = SECANT_ERROR_THREAD_FAILED;
        }
    }
    for (slot_idx = 0u;
         slot_idx < runner->num_slots;
         ++slot_idx) {
        if (runner->slots != NULL) {
            if (runner->slots[slot_idx].module != NULL &&
                hipModuleUnload(
                    runner->slots[slot_idx].module) != hipSuccess &&
                result == SECANT_SUCCESS) {
                result = SECANT_ERROR_DRIVER_FAILED;
            }
            if (runner->slot_release != NULL) {
                runner->slot_release(
                    runner->backend,
                    runner->slots + slot_idx);
            }
            free(runner->slots[slot_idx].functions);
        }
    }
    for (stream_idx = 0u;
         stream_idx < runner->num_streams;
         ++stream_idx) {
        if (runner->done_events != NULL &&
            runner->done_events[stream_idx] != NULL &&
            hipEventDestroy(
                runner->done_events[stream_idx]) != hipSuccess &&
            result == SECANT_SUCCESS) {
            result = SECANT_ERROR_DRIVER_FAILED;
        }
        if (runner->streams != NULL &&
            runner->streams[stream_idx] != NULL &&
            hipStreamDestroy(
                runner->streams[stream_idx]) != hipSuccess &&
            result == SECANT_SUCCESS) {
            result = SECANT_ERROR_DRIVER_FAILED;
        }
    }
    if (runner->stop_event != NULL &&
        hipEventDestroy(runner->stop_event) != hipSuccess &&
        result == SECANT_SUCCESS) {
        result = SECANT_ERROR_DRIVER_FAILED;
    }
    if (runner->start_event != NULL &&
        hipEventDestroy(runner->start_event) != hipSuccess &&
        result == SECANT_SUCCESS) {
        result = SECANT_ERROR_DRIVER_FAILED;
    }
    if (runner->state_condition_initialized) {
        (void)pthread_cond_destroy(&runner->state_condition);
    }
    if (runner->ready_condition_initialized) {
        (void)pthread_cond_destroy(&runner->ready_condition);
    }
    if (runner->work_condition_initialized) {
        (void)pthread_cond_destroy(&runner->work_condition);
    }
    if (runner->mutex_initialized) {
        (void)pthread_mutex_destroy(&runner->mutex);
    }
    free(runner->workers);
    free(runner->ready_slots);
    free(runner->slots);
    free(runner->done_events);
    free(runner->streams);
    free(runner);
    return result;
}

#undef _SECANT_AMD_RUNNER_ERROR_RET
