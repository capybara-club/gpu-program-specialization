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
#include "s_runner_internal.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double s_runner_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now))
        return 0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}
static void *s_worker_main(void *opaque) {
    SWorker *worker = (SWorker *)opaque;
    SecantCubinRunner r = worker->runner;
    pthread_mutex_lock(&r->mutex);
    for (;;) {
        SSlot *slot = NULL;
        size_t i, first;
        SecantAstProgramSet programs;
        double begin;
        if (r->shutdown)
            break;
        if (r->active && !r->cancel && r->next_module < r->module_count)
            for (i = 0; i < r->slot_count; ++i)
                if (!r->slots[i].state) {
                    slot = r->slots + i;
                    break;
                }
        if (!slot) {
            pthread_cond_wait(&r->condition, &r->mutex);
            continue;
        }
        slot->state = 1;
        slot->module_index = r->next_module++;
        ++r->compiling;
        first = slot->module_index * r->module_capacity;
        programs = r->programs;
        programs.asts.items += first;
        programs.asts.count -= first;
        if (programs.asts.count > r->module_capacity)
            programs.asts.count = r->module_capacity;
        slot->ast_count = programs.asts.count;
        pthread_mutex_unlock(&r->mutex);
        begin = s_runner_seconds();
        /* A failed patch must never contaminate the next specialization. */
        memcpy(slot->cubin, r->template_cubin, r->plan->cubin_size);
        slot->result = secant_cubin_specialize_into(r->plan, &programs, slot->cubin, r->plan->cubin_size);
        pthread_mutex_lock(&r->mutex);
        worker->work_seconds += s_runner_seconds() - begin;
        slot->state = 2;
        --r->compiling;
        ++r->compiled;
        r->compiled_at = s_runner_seconds();
        pthread_cond_broadcast(&r->condition);
    }
    pthread_mutex_unlock(&r->mutex);
    return NULL;
}
SecantResult s_pipeline_create(SecantCubinRunner r) {
    size_t i;
    CUmoduleLoadingMode mode;
    if (cuCtxGetCurrent(&r->context) != CUDA_SUCCESS || !r->context)
        return SECANT_ERROR_INVALID_STATE;
    if (cuModuleGetLoadingMode(&mode) != CUDA_SUCCESS)
        return SECANT_ERROR_DRIVER_FAILED;
    if (mode != CU_MODULE_EAGER_LOADING)
        return SECANT_ERROR_EAGER_LOADING_REQUIRED;
    if (pthread_mutex_init(&r->mutex, NULL))
        return SECANT_ERROR_THREAD_FAILED;
    r->mutex_ready = 1;
    if (pthread_cond_init(&r->condition, NULL))
        return SECANT_ERROR_THREAD_FAILED;
    r->condition_ready = 1;
    r->slots = (SSlot *)calloc(r->slot_count, sizeof(*r->slots));
    r->workers = (SWorker *)calloc(r->worker_count, sizeof(*r->workers));
    r->streams = (CUstream *)calloc(r->stream_count, sizeof(*r->streams));
    r->done = (CUevent *)calloc(r->stream_count, sizeof(*r->done));
    r->functions = (CUfunction *)calloc(r->plan->num_kernels, sizeof(*r->functions));
    if (!r->slots || !r->workers || !r->streams || !r->done || !r->functions)
        return SECANT_ERROR_ALLOCATION_FAILED;
    for (i = 0; i < r->slot_count; ++i) {
        r->slots[i].cubin = (unsigned char *)malloc(r->plan->cubin_size);
        if (!r->slots[i].cubin)
            return SECANT_ERROR_ALLOCATION_FAILED;
    }
    for (i = 0; i < r->stream_count; ++i)
        if (cuStreamCreate(r->streams + i, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
            cuEventCreate(r->done + i, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
            return SECANT_ERROR_DRIVER_FAILED;
    if (cuEventCreate(&r->start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuEventCreate(&r->stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
        return SECANT_ERROR_DRIVER_FAILED;
    for (i = 0; i < r->worker_count; ++i) {
        r->workers[i].runner = r;
        if (pthread_create(&r->workers[i].thread, NULL, s_worker_main, r->workers + i))
            return SECANT_ERROR_THREAD_FAILED;
        r->workers[i].started = 1;
    }
    return SECANT_SUCCESS;
}
/* Recovery only: prove completion on every owned stream. If the event path is
 * broken, stream synchronization is the last resort; never synchronize the
 * whole context. A failed proof retains both module and caller buffer ownership. */
static SecantResult s_drain(SecantCubinRunner r) {
    size_t i;
    int complete = 1;
    if (!r->work_pending)
        return SECANT_SUCCESS;
    for (i = 0; i < r->stream_count; ++i) {
        if (cuEventRecord(r->done[i], r->streams[i]) == CUDA_SUCCESS &&
            cuEventSynchronize(r->done[i]) == CUDA_SUCCESS)
            continue;
        if (cuStreamSynchronize(r->streams[i]) != CUDA_SUCCESS)
            complete = 0;
    }
    if (!complete) {
        r->failed = 1;
        return SECANT_ERROR_COMPLETION_UNKNOWN;
    }
    r->work_pending = 0;
    return SECANT_SUCCESS;
}
static SecantResult s_unload(SecantCubinRunner r) {
    if (r->module) {
        if (cuModuleUnload(r->module) != CUDA_SUCCESS) {
            r->failed = 1;
            return SECANT_ERROR_DRIVER_FAILED;
        }
        r->module = NULL;
    }
    return SECANT_SUCCESS;
}
SecantResult s_pipeline_destroy(SecantCubinRunner r) {
    size_t i;
    SecantResult result = SECANT_SUCCESS;
    if (!r)
        return SECANT_SUCCESS;
    if (r->mutex_ready) {
        pthread_mutex_lock(&r->mutex);
        r->shutdown = 1;
        if (r->condition_ready)
            pthread_cond_broadcast(&r->condition);
        pthread_mutex_unlock(&r->mutex);
    }
    if (r->workers)
        for (i = 0; i < r->worker_count; ++i)
            if (r->workers[i].started) {
                if (pthread_join(r->workers[i].thread, NULL))
                    result = SECANT_ERROR_THREAD_FAILED;
                else
                    r->workers[i].started = 0;
            }
    /* A failed join must never free memory a worker could still be using. */
    if (result != SECANT_SUCCESS)
        return result;
    result = s_drain(r);
    if (result != SECANT_SUCCESS)
        return result;
    result = s_unload(r);
    if (result != SECANT_SUCCESS)
        return result;
#define DESTROY(handle, call)                                                                                \
    do {                                                                                                     \
        if (handle) {                                                                                        \
            if (call(handle) != CUDA_SUCCESS)                                                                \
                result = SECANT_ERROR_DRIVER_FAILED;                                                         \
            else                                                                                             \
                (handle) = NULL;                                                                             \
        }                                                                                                    \
    } while (0)
    if (r->done)
        for (i = 0; i < r->stream_count; ++i)
            DESTROY(r->done[i], cuEventDestroy);
    if (r->streams)
        for (i = 0; i < r->stream_count; ++i)
            DESTROY(r->streams[i], cuStreamDestroy);
    DESTROY(r->start, cuEventDestroy);
    DESTROY(r->stop, cuEventDestroy);
#undef DESTROY
    /* Failed handles remain owned and can be retried, including partial create. */
    if (result != SECANT_SUCCESS)
        return result;
    if (r->condition_ready) {
        if (pthread_cond_destroy(&r->condition))
            return SECANT_ERROR_THREAD_FAILED;
        r->condition_ready = 0;
    }
    if (r->mutex_ready) {
        if (pthread_mutex_destroy(&r->mutex))
            return SECANT_ERROR_THREAD_FAILED;
        r->mutex_ready = 0;
    }
    if (r->slots)
        for (i = 0; i < r->slot_count; ++i)
            free(r->slots[i].cubin);
    free(r->slots);
    free(r->workers);
    free(r->streams);
    free(r->done);
    free(r->functions);
    free(r->template_cubin);
    free(r);
    return SECANT_SUCCESS;
}
static const char *s_kernel_prefix(_SecantCubinShape shape) {
    switch (shape) {
    case _SECANT_CUBIN_SHAPE_MATERIALIZE:
        return "secant_cubin_materialize";
    case _SECANT_CUBIN_SHAPE_SSE:
        return "secant_cubin_sse";
    case _SECANT_CUBIN_SHAPE_AFFINE_STATS:
        return "secant_cubin_affine_stats";
    case _SECANT_CUBIN_SHAPE_GRAM_STATS:
        return "secant_cubin_gram_stats";
    case _SECANT_CUBIN_SHAPE_TOGGLE_SSE:
        return "secant_cubin_toggle_sse";
    default:
        return NULL;
    }
}
static SecantResult s_launch(SecantCubinRunner r, const SSlot *slot, const SRun *v, size_t kernel) {
    const SecantCubinPlan *p = r->plan;
    size_t first = slot->module_index * r->module_capacity + kernel * p->asts_per_kernel;
    size_t active = slot->ast_count - kernel * p->asts_per_kernel;
    size_t output_first = first, rows = v->rows, targets_count = v->targets_count;
    size_t input_ld = v->input.leading_dimension, target_ld = v->targets.leading_dimension,
           output_ld = v->output.leading_dimension;
    CUdeviceptr input = v->input.address, targets = v->targets.address, output, banks;
    size_t bank_stride = v->banks.bank_stride, configs = v->configurations;
    uint32_t bits = v->toggle_bits;
    unsigned block = p->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ? 256u : (unsigned)p->threads_per_block;
    size_t tile = p->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ? block : p->tile_rows;
    unsigned grid_x = (unsigned)((rows - 1) / tile + 1), grid_y = 1;
    CUstream stream = r->streams[kernel % r->stream_count];
    size_t active_asts = slot->ast_count;
    void *args[12] = {&input,  &input_ld,      &targets, &target_ld, &rows,
                      &active, &targets_count, &output,  &output_ld};
    if (active > p->asts_per_kernel)
        active = p->asts_per_kernel;
    if (p->shape == _SECANT_CUBIN_SHAPE_GRAM_STATS)
        output_first = first / p->asts_per_kernel;
    if (p->shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE) {
        /* Output offsets include the kernel index; bank vectors are shared. */
        output_first = slot->module_index * r->module_capacity * targets_count;
        banks = v->banks.address;
        output = v->output.address + output_first * output_ld * sizeof(float);
        args[4] = &banks;
        args[5] = &bank_stride;
        args[6] = &rows;
        args[7] = &configs;
        args[8] = &bits;
        args[9] = &output;
        args[10] = &output_ld;
        args[11] = &active_asts;
        grid_y = (unsigned)(((configs - 1) / block + 1) > 65535 ? 65535 : ((configs - 1) / block + 1));
    } else {
        output = v->output.address + output_first * output_ld * sizeof(float);
        if (p->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE) {
            args[2] = &rows;
            args[3] = &active;
            args[4] = &output;
            args[5] = &output_ld;
        }
    }
    r->work_pending = 1;
    return cuLaunchKernel(r->functions[kernel], grid_x, grid_y, 1, block, 1, 1, 0, stream, args, NULL) ==
                   CUDA_SUCCESS
               ? SECANT_SUCCESS
               : SECANT_ERROR_DRIVER_FAILED;
}
static SecantResult s_finish_events(SecantCubinRunner r, SecantRunnerStats *stats) {
    size_t i;
    double begin;
    float milliseconds;
    for (i = 0; i < r->stream_count; ++i) {
        if (cuEventRecord(r->done[i], r->streams[i]) != CUDA_SUCCESS ||
            cuStreamWaitEvent(r->streams[0], r->done[i], 0) != CUDA_SUCCESS)
            return SECANT_ERROR_DRIVER_FAILED;
    }
    if (cuEventRecord(r->stop, r->streams[0]) != CUDA_SUCCESS)
        return SECANT_ERROR_DRIVER_FAILED;
    begin = s_runner_seconds();
    if (cuEventSynchronize(r->stop) != CUDA_SUCCESS)
        return SECANT_ERROR_DRIVER_FAILED;
    r->work_pending = 0;
    stats->completion_wait_seconds += s_runner_seconds() - begin;
    if (cuEventElapsedTime(&milliseconds, r->start, r->stop) != CUDA_SUCCESS)
        return SECANT_ERROR_DRIVER_FAILED;
    stats->runtime_seconds += milliseconds * 1e-3;
    return SECANT_SUCCESS;
}
SecantResult s_pipeline_execute(SecantCubinRunner r, const SRun *runs, size_t count,
                                SecantRunnerStats *stats) {
    size_t i, j, completed = 0;
    SecantResult result = SECANT_SUCCESS;
    double begin = s_runner_seconds();
    pthread_mutex_lock(&r->mutex);
    r->programs = runs[0].programs;
    r->next_module = r->compiled = r->compiling = 0;
    r->module_count = (r->programs.asts.count - 1) / r->module_capacity + 1;
    r->started_at = r->compiled_at = begin;
    r->cancel = 0;
    for (i = 0; i < r->worker_count; ++i)
        r->workers[i].work_seconds = 0;
    pthread_cond_broadcast(&r->condition);
    pthread_mutex_unlock(&r->mutex);
    stats->num_asts = runs[0].programs.asts.count;
    stats->num_modules = r->module_count;
    /* Clear all active output bins once, then order every scoring stream after it. */
    for (i = 0; result == SECANT_SUCCESS && i < count; ++i) {
        if (r->plan->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE)
            continue;
        /* Preserve padding between active output rows. */
        r->work_pending = 1;
        if (cuMemsetD2D32Async(runs[i].output.address, runs[i].output.leading_dimension * sizeof(float), 0,
                               runs[i].output_width, runs[i].output_count, r->streams[0]) != CUDA_SUCCESS)
            result = SECANT_ERROR_DRIVER_FAILED;
    }
    while (result == SECANT_SUCCESS && completed < r->module_count) {
        SSlot *slot = NULL;
        double tick;
        SecantResult finished;
        pthread_mutex_lock(&r->mutex);
        while (!slot) {
            for (i = 0; i < r->slot_count; ++i)
                if (r->slots[i].state == 2) {
                    slot = r->slots + i;
                    slot->state = 3;
                    break;
                }
            if (!slot)
                pthread_cond_wait(&r->condition, &r->mutex);
        }
        pthread_mutex_unlock(&r->mutex);
        result = slot->result;
        tick = s_runner_seconds();
        if (result == SECANT_SUCCESS && cuModuleLoadData(&r->module, slot->cubin) != CUDA_SUCCESS)
            result = SECANT_ERROR_DRIVER_FAILED;
        if (r->module) {
            ++stats->modules_loaded;
            for (i = 0; result == SECANT_SUCCESS && i < r->plan->num_kernels; ++i) {
                char name[96];
                snprintf(name, sizeof(name), "%s_%03zu", s_kernel_prefix(r->plan->shape), i);
                if (cuModuleGetFunction(r->functions + i, r->module, name) != CUDA_SUCCESS)
                    result = SECANT_ERROR_DRIVER_FAILED;
            }
        }
        stats->module_load_seconds += s_runner_seconds() - tick;
        if (result == SECANT_SUCCESS && cuEventRecord(r->start, r->streams[0]) != CUDA_SUCCESS)
            result = SECANT_ERROR_DRIVER_FAILED;
        for (i = 1; result == SECANT_SUCCESS && i < r->stream_count; ++i)
            if (cuStreamWaitEvent(r->streams[i], r->start, 0) != CUDA_SUCCESS)
                result = SECANT_ERROR_DRIVER_FAILED;
        for (j = 0; result == SECANT_SUCCESS && j < count; ++j)
            for (i = 0; result == SECANT_SUCCESS && i * r->plan->asts_per_kernel < slot->ast_count; ++i)
                result = s_launch(r, slot, runs + j, i);
        /* Unload only after completion is proven. Failed event operations get
         * a separate recovery drain; its failure takes precedence over other errors. */
        if (r->module && result == SECANT_SUCCESS)
            result = s_finish_events(r, stats);
        finished = s_drain(r);
        if (finished != SECANT_SUCCESS)
            result = finished;
        else {
            tick = s_runner_seconds();
            finished = s_unload(r);
            if (result == SECANT_SUCCESS)
                result = finished;
            stats->module_unload_seconds += s_runner_seconds() - tick;
        }
        pthread_mutex_lock(&r->mutex);
        slot->state = 0;
        pthread_cond_broadcast(&r->condition);
        pthread_mutex_unlock(&r->mutex);
        ++completed;
    }
    pthread_mutex_lock(&r->mutex);
    r->cancel = 1;
    while (r->compiling)
        pthread_cond_wait(&r->condition, &r->mutex);
    for (i = 0; i < r->slot_count; ++i)
        r->slots[i].state = 0;
    r->next_module = r->module_count = 0;
    stats->compile_window_seconds = r->compiled_at - begin;
    for (i = 0; i < r->worker_count; ++i) {
        stats->compile_work_seconds += r->workers[i].work_seconds;
        if (r->workers[i].work_seconds > stats->compile_critical_seconds)
            stats->compile_critical_seconds = r->workers[i].work_seconds;
    }
    pthread_mutex_unlock(&r->mutex);
    /* Also covers a partial batch clear or a specialization failure before load. */
    if (r->work_pending && !r->failed) {
        SecantResult drained = s_drain(r);
        if (drained != SECANT_SUCCESS)
            result = drained;
    }
    stats->total_seconds = s_runner_seconds() - begin;
    return result;
}
