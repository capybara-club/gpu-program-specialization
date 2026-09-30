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
#define _POSIX_C_SOURCE 200809L

#include "ssid_internal.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SSID_CUDA_SUCCESS 0
#define SSID_CUDA_ERROR_NOT_READY 600
#define SSID_CU_STREAM_NON_BLOCKING 1u
#define SSID_CU_EVENT_DISABLE_TIMING 2u
#define SSID_WINNER_REDUCTION_KERNEL "ssid_reduce_winners"
#define SSID_TRAJECTORY_LM_KERNEL "secant_cubin_materialize_000"
#define SSID_TICKET_FEDBATCH 0u
#define SSID_TICKET_TRAJECTORY_LM 1u

typedef int ssid_cu_result;
typedef int ssid_cu_device;
typedef uint64_t ssid_cu_deviceptr;
typedef void *ssid_cu_context;
typedef void *ssid_cu_module;
typedef void *ssid_cu_function;
typedef void *ssid_cu_stream;
typedef void *ssid_cu_event;

typedef struct ssid_cuda_driver {
    void *library;
    ssid_cu_result (*cuInit)(unsigned int);
    ssid_cu_result (*cuDeviceGet)(ssid_cu_device *, int);
    ssid_cu_result (*cuCtxCreate_v2)(ssid_cu_context *, unsigned int, ssid_cu_device);
    ssid_cu_result (*cuCtxDestroy_v2)(ssid_cu_context);
    ssid_cu_result (*cuCtxSetCurrent)(ssid_cu_context);
    ssid_cu_result (*cuModuleLoadData)(ssid_cu_module *, const void *);
    ssid_cu_result (*cuModuleUnload)(ssid_cu_module);
    ssid_cu_result (*cuModuleGetFunction)(ssid_cu_function *, ssid_cu_module, const char *);
    ssid_cu_result (*cuLaunchKernel)(ssid_cu_function, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, ssid_cu_stream, void **, void **);
    ssid_cu_result (*cuStreamCreate)(ssid_cu_stream *, unsigned int);
    ssid_cu_result (*cuStreamWaitEvent)(ssid_cu_stream, ssid_cu_event, unsigned int);
    ssid_cu_result (*cuStreamSynchronize)(ssid_cu_stream);
    ssid_cu_result (*cuStreamDestroy_v2)(ssid_cu_stream);
    ssid_cu_result (*cuEventCreate)(ssid_cu_event *, unsigned int);
    ssid_cu_result (*cuEventRecord)(ssid_cu_event, ssid_cu_stream);
    ssid_cu_result (*cuEventQuery)(ssid_cu_event);
    ssid_cu_result (*cuEventDestroy_v2)(ssid_cu_event);
    ssid_cu_result (*cuMemAlloc_v2)(ssid_cu_deviceptr *, size_t);
    ssid_cu_result (*cuMemFree_v2)(ssid_cu_deviceptr);
    ssid_cu_result (*cuMemcpyHtoD_v2)(ssid_cu_deviceptr, const void *, size_t);
    ssid_cu_result (*cuMemcpyDtoH_v2)(void *, ssid_cu_deviceptr, size_t);
    ssid_cu_result (*cuGetErrorName)(ssid_cu_result, const char **);
    ssid_cu_result (*cuGetErrorString)(ssid_cu_result, const char **);
} ssid_cuda_driver;

typedef struct ssid_worker ssid_worker;

struct ssid_ticket {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    ssid_genome_batch batch;
    ssid_fedbatch_launch launch;
    ssid_trajectory_lm_launch lm_launch;
    const uint8_t *source_cubin;
    size_t source_cubin_byte_count;
    uint32_t kind;
    ssid_ticket_result result;
    double submitted_at;
    int complete;
    char error[512];
    ssid_worker *worker;
    ssid_cu_module module;
    ssid_cu_event event;
    ssid_cu_event *dependency_events;
    uint32_t dependency_event_count;
    uint32_t stream_index;
    struct ssid_ticket *next;
};

struct ssid_worker {
    struct ssid_pipeline *pipeline;
    pthread_t thread;
    pthread_cond_t image_condition;
    uint8_t *image;
    uint32_t index;
    int image_released;
    int started;
};

struct ssid_pipeline {
    const ssid_template *template_value;
    size_t cubin_byte_count;
    ssid_pipeline_config config;
    pthread_mutex_t mutex;
    pthread_mutex_t cuda_mutex;
    pthread_cond_t condition;
    pthread_cond_t worker_condition;
    ssid_worker *workers;
    uint32_t worker_count;
    pthread_t cuda_thread;
    int cuda_thread_started;
    int cuda_initialized;
    int cuda_initialization_status;
    char cuda_initialization_error[512];
    int stopping;
    uint32_t active_jobs;
    ssid_ticket *pending_head;
    ssid_ticket *pending_tail;
    ssid_ticket *ready_head;
    ssid_ticket *ready_tail;
    ssid_ticket *inflight_head;
    ssid_ticket *inflight_tail;
    uint32_t inflight_count;
    ssid_cuda_driver cuda;
    ssid_cu_context context;
    ssid_cu_stream *streams;
    uint32_t stream_count;
    uint32_t next_stream_index;
};

static void ssid_queue_append(ssid_ticket **head, ssid_ticket **tail, ssid_ticket *ticket) {
    ticket->next = NULL;
    if (*tail == NULL) *head = ticket;
    else (*tail)->next = ticket;
    *tail = ticket;
}

static ssid_ticket *ssid_queue_pop(ssid_ticket **head, ssid_ticket **tail) {
    ssid_ticket *result = *head;
    if (result == NULL) return NULL;
    *head = result->next;
    if (*head == NULL) *tail = NULL;
    result->next = NULL;
    return result;
}

static void ssid_ticket_complete(ssid_pipeline *pipeline, ssid_ticket *ticket, int status, const char *error) {
    pthread_mutex_lock(&ticket->mutex);
    ticket->result.status = status;
    ticket->result.total_seconds = ssid_monotonic_seconds() - ticket->submitted_at;
    if (error != NULL && error[0] != '\0') snprintf(ticket->error, sizeof(ticket->error), "%s", error);
    ticket->complete = 1;
    pthread_cond_broadcast(&ticket->condition);
    pthread_mutex_unlock(&ticket->mutex);
    pthread_mutex_lock(&pipeline->mutex);
    if (pipeline->active_jobs != 0) pipeline->active_jobs -= 1;
    if (pipeline->active_jobs == 0) pthread_cond_broadcast(&pipeline->condition);
    pthread_mutex_unlock(&pipeline->mutex);
}

static int ssid_cuda_symbol(ssid_cuda_driver *cuda, void *target, const char *name, int required) {
    void *symbol = dlsym(cuda->library, name);
    if (symbol == NULL && required) {
        ssid_set_error("CUDA driver symbol %s was not found", name);
        return SSID_CUDA_ERROR;
    }
    memcpy(target, &symbol, sizeof(symbol));
    return SSID_OK;
}

#define SSID_CUDA_SYMBOL(cuda_value, member, required_value) ssid_cuda_symbol((cuda_value), &(cuda_value)->member, #member, (required_value))

static int ssid_cuda_open(ssid_cuda_driver *cuda) {
    int status;
    memset(cuda, 0, sizeof(*cuda));
    cuda->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (cuda->library == NULL) cuda->library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (cuda->library == NULL) {
        ssid_set_error("the NVIDIA CUDA driver library was not found");
        return SSID_CUDA_ERROR;
    }
    if ((status = SSID_CUDA_SYMBOL(cuda, cuInit, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuDeviceGet, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuCtxCreate_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuCtxDestroy_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuCtxSetCurrent, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuModuleLoadData, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuModuleUnload, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuModuleGetFunction, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuLaunchKernel, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuStreamCreate, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuStreamWaitEvent, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuStreamSynchronize, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuStreamDestroy_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuEventCreate, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuEventRecord, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuEventQuery, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuEventDestroy_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuMemAlloc_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuMemFree_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuMemcpyHtoD_v2, 1)) != SSID_OK || (status = SSID_CUDA_SYMBOL(cuda, cuMemcpyDtoH_v2, 1)) != SSID_OK) {
        dlclose(cuda->library);
        memset(cuda, 0, sizeof(*cuda));
        return status;
    }
    SSID_CUDA_SYMBOL(cuda, cuGetErrorName, 0);
    SSID_CUDA_SYMBOL(cuda, cuGetErrorString, 0);
    return SSID_OK;
}

static void ssid_cuda_close(ssid_cuda_driver *cuda) {
    if (cuda->library != NULL) dlclose(cuda->library);
    memset(cuda, 0, sizeof(*cuda));
}

static void ssid_cuda_error_text(ssid_cuda_driver *cuda, ssid_cu_result result, const char *operation, char *destination, size_t destination_size) {
    const char *name = NULL;
    const char *description = NULL;
    if (cuda->cuGetErrorName != NULL) cuda->cuGetErrorName(result, &name);
    if (cuda->cuGetErrorString != NULL) cuda->cuGetErrorString(result, &description);
    snprintf(destination, destination_size, "%s failed with CUDA result %d%s%s%s%s", operation, result, name == NULL ? "" : " (", name == NULL ? "" : name, name == NULL ? "" : ")", description == NULL ? "" : description);
}

static int ssid_cuda_initialize(ssid_pipeline *pipeline) {
    ssid_cu_result result;
    ssid_cu_device device;
    uint32_t stream_index;
    int status = ssid_cuda_open(&pipeline->cuda);
    if (status != SSID_OK) return status;
    result = pipeline->cuda.cuInit(0);
    if (result != SSID_CUDA_SUCCESS) goto cuda_failure;
    result = pipeline->cuda.cuDeviceGet(&device, pipeline->config.device_ordinal);
    if (result != SSID_CUDA_SUCCESS) goto cuda_failure;
    result = pipeline->cuda.cuCtxCreate_v2(&pipeline->context, 0, device);
    if (result != SSID_CUDA_SUCCESS) goto cuda_failure;
    for (stream_index = 0; stream_index < pipeline->stream_count; ++stream_index) {
        result = pipeline->cuda.cuStreamCreate(
            &pipeline->streams[stream_index], SSID_CU_STREAM_NON_BLOCKING);
        if (result != SSID_CUDA_SUCCESS) goto cuda_failure;
    }
    return SSID_OK;

cuda_failure:
    ssid_cuda_error_text(&pipeline->cuda, result, "CUDA pipeline initialization", pipeline->cuda_initialization_error, sizeof(pipeline->cuda_initialization_error));
    for (stream_index = 0; stream_index < pipeline->stream_count; ++stream_index) {
        if (pipeline->streams[stream_index] != NULL) {
            pipeline->cuda.cuStreamDestroy_v2(pipeline->streams[stream_index]);
            pipeline->streams[stream_index] = NULL;
        }
    }
    if (pipeline->context != NULL) pipeline->cuda.cuCtxDestroy_v2(pipeline->context);
    pipeline->context = NULL;
    ssid_cuda_close(&pipeline->cuda);
    return SSID_CUDA_ERROR;
}

static void ssid_release_worker_image(ssid_pipeline *pipeline, ssid_ticket *ticket) {
    pthread_mutex_lock(&pipeline->mutex);
    ticket->worker->image_released = 1;
    pthread_cond_signal(&ticket->worker->image_condition);
    pthread_mutex_unlock(&pipeline->mutex);
}

static void ssid_destroy_dependency_events(
    ssid_pipeline *pipeline, ssid_ticket *ticket) {
    uint32_t index;
    for (index = 0; index < ticket->dependency_event_count; ++index) {
        if (ticket->dependency_events[index] != NULL) {
            pipeline->cuda.cuEventDestroy_v2(ticket->dependency_events[index]);
        }
    }
    free(ticket->dependency_events);
    ticket->dependency_events = NULL;
    ticket->dependency_event_count = 0;
}

static void ssid_synchronize_ticket_streams(
    ssid_pipeline *pipeline, const ssid_ticket *ticket) {
    uint32_t index;
    uint32_t count = ticket->dependency_event_count == 0
        ? 1u : ticket->dependency_event_count;
    for (index = 0; index < count; ++index) {
        uint32_t stream_index =
            (ticket->stream_index + index) % pipeline->stream_count;
        pipeline->cuda.cuStreamSynchronize(pipeline->streams[stream_index]);
    }
}

static int ssid_launch_loaded_module(ssid_pipeline *pipeline, ssid_ticket *ticket, char *error, size_t error_size) {
    const ssid_template *template_value = pipeline->template_value;
    ssid_cu_stream completion_stream = pipeline->streams[ticket->stream_index];
    ssid_cu_function *functions;
    ssid_cu_function reduction_function = NULL;
    uint32_t active_count = 0;
    uint32_t scoring_kernel_count;
    uint32_t used_stream_count;
    uint32_t active_sequence = 0;
    uint32_t kernel_index;
    uint64_t setting_tiles_64;
    ssid_cu_result cuda_result;
    double started;
    functions = (ssid_cu_function *)calloc(template_value->kernel_count, sizeof(ssid_cu_function));
    if (functions == NULL) {
        snprintf(error, error_size, "could not allocate CUDA function handles");
        return SSID_OUT_OF_MEMORY;
    }
    started = ssid_monotonic_seconds();
    for (kernel_index = 0; kernel_index < template_value->kernel_count; ++kernel_index) {
        const ssid_kernel_plan *kernel = &template_value->kernels[kernel_index];
        if (kernel->genome_base >= ticket->batch.genome_count) continue;
        cuda_result = pipeline->cuda.cuModuleGetFunction(&functions[kernel_index], ticket->module, kernel->name);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuModuleGetFunction", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
        active_count += 1;
    }
    scoring_kernel_count = active_count;
    if (scoring_kernel_count == 0) {
        snprintf(error, error_size, "loaded module has no active scoring kernels");
        free(functions);
        return SSID_INVALID_ARGUMENT;
    }
    if (ticket->launch.output_mode == SSID_OUTPUT_GENOME_WINNERS) {
        cuda_result = pipeline->cuda.cuModuleGetFunction(&reduction_function, ticket->module, SSID_WINNER_REDUCTION_KERNEL);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuModuleGetFunction(ssid_reduce_winners)", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
        active_count += 1;
    }
    ticket->result.function_lookup_seconds = ssid_monotonic_seconds() - started;
    cuda_result = pipeline->cuda.cuEventCreate(&ticket->event, SSID_CU_EVENT_DISABLE_TIMING);
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuEventCreate", error, error_size);
        free(functions);
        return SSID_CUDA_ERROR;
    }
    used_stream_count = scoring_kernel_count < pipeline->stream_count
        ? scoring_kernel_count : pipeline->stream_count;
    ticket->dependency_events = (ssid_cu_event *)calloc(
        used_stream_count, sizeof(ssid_cu_event));
    if (ticket->dependency_events == NULL) {
        snprintf(error, error_size, "could not allocate per-stream dependency events");
        free(functions);
        return SSID_OUT_OF_MEMORY;
    }
    ticket->dependency_event_count = used_stream_count;
    for (kernel_index = 0; kernel_index < used_stream_count; ++kernel_index) {
        cuda_result = pipeline->cuda.cuEventCreate(
            &ticket->dependency_events[kernel_index], SSID_CU_EVENT_DISABLE_TIMING);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result,
                "cuEventCreate(dependency)", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
    }
    started = ssid_monotonic_seconds();
    setting_tiles_64 = (ticket->launch.num_settings + ticket->launch.threads_per_block - 1u) / ticket->launch.threads_per_block;
    if (setting_tiles_64 > UINT32_MAX) {
        snprintf(error, error_size, "setting tile count exceeds the CUDA grid limit");
        free(functions);
        return SSID_OUT_OF_RANGE;
    }
    for (kernel_index = 0; kernel_index < template_value->kernel_count; ++kernel_index) {
        const ssid_kernel_plan *kernel = &template_value->kernels[kernel_index];
        uint32_t active_genomes;
        uint32_t genome_groups;
        uint64_t settings_device = ticket->launch.settings_device;
        uint64_t settings_ld = ticket->launch.settings_leading_dimension;
        uint64_t bindings_device = ticket->launch.bindings_device;
        uint64_t bindings_ld = ticket->launch.bindings_leading_dimension;
        uint64_t num_settings = ticket->launch.num_settings;
        uint32_t num_genomes = ticket->launch.num_genomes;
        uint64_t reference_device = ticket->launch.reference_device;
        uint32_t steps = ticket->launch.steps_per_observation;
        uint64_t mse_device = ticket->launch.mse_device;
        uint64_t cta_score_device = ticket->launch.cta_score_device;
        uint64_t cta_setting_device = ticket->launch.cta_setting_device;
        void *mse_parameters[9] = {&settings_device, &settings_ld, &bindings_device, &bindings_ld, &num_settings, &num_genomes, &reference_device, &steps, &mse_device};
        void *winner_parameters[10] = {&settings_device, &settings_ld, &bindings_device, &bindings_ld, &num_settings, &num_genomes, &reference_device, &steps, &cta_score_device, &cta_setting_device};
        void **parameters = ticket->launch.output_mode == SSID_OUTPUT_GENOME_WINNERS ? winner_parameters : mse_parameters;
        ssid_cu_stream kernel_stream;
        if (kernel->genome_base >= ticket->batch.genome_count) continue;
        kernel_stream = pipeline->streams[
            (ticket->stream_index + active_sequence) % pipeline->stream_count];
        active_sequence += 1;
        active_genomes = ticket->batch.genome_count - kernel->genome_base;
        if (active_genomes > kernel->genome_capacity) active_genomes = kernel->genome_capacity;
        genome_groups = (active_genomes + kernel->genomes_per_cta - 1u) / kernel->genomes_per_cta;
        cuda_result = pipeline->cuda.cuLaunchKernel(functions[kernel_index], (uint32_t)setting_tiles_64, genome_groups, 1, ticket->launch.threads_per_block, 1, 1, ticket->launch.shared_memory_bytes, kernel_stream, parameters, NULL);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuLaunchKernel", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
    }
    for (kernel_index = 0; kernel_index < used_stream_count; ++kernel_index) {
        ssid_cu_stream kernel_stream = pipeline->streams[
            (ticket->stream_index + kernel_index) % pipeline->stream_count];
        cuda_result = pipeline->cuda.cuEventRecord(
            ticket->dependency_events[kernel_index], kernel_stream);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result,
                "cuEventRecord(dependency)", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
        cuda_result = pipeline->cuda.cuStreamWaitEvent(
            completion_stream, ticket->dependency_events[kernel_index], 0u);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result,
                "cuStreamWaitEvent", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
    }
    if (ticket->launch.output_mode == SSID_OUTPUT_GENOME_WINNERS) {
        uint64_t cta_score_device = ticket->launch.cta_score_device;
        uint64_t cta_setting_device = ticket->launch.cta_setting_device;
        uint64_t num_settings = ticket->launch.num_settings;
        uint32_t num_genomes = ticket->launch.num_genomes;
        uint32_t setting_tiles = (uint32_t)setting_tiles_64;
        uint64_t winner_score_device = ticket->launch.winner_score_device;
        uint64_t winner_setting_device = ticket->launch.winner_setting_device;
        uint32_t reduction_threads = ticket->launch.reduction_threads;
        unsigned int reduction_shared_bytes = reduction_threads * (unsigned int)(sizeof(float) + sizeof(uint32_t));
        void *reduction_parameters[7] = {&cta_score_device, &cta_setting_device, &num_settings, &num_genomes, &setting_tiles, &winner_score_device, &winner_setting_device};
        cuda_result = pipeline->cuda.cuLaunchKernel(reduction_function, num_genomes, 1, 1, reduction_threads, 1, 1, reduction_shared_bytes, completion_stream, reduction_parameters, NULL);
        if (cuda_result != SSID_CUDA_SUCCESS) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuLaunchKernel(ssid_reduce_winners)", error, error_size);
            free(functions);
            return SSID_CUDA_ERROR;
        }
    }
    cuda_result = pipeline->cuda.cuEventRecord(ticket->event, completion_stream);
    ticket->result.launch_seconds = ssid_monotonic_seconds() - started;
    ticket->result.launched_kernel_count = active_count;
    free(functions);
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuEventRecord", error, error_size);
        return SSID_CUDA_ERROR;
    }
    return SSID_OK;
}

static int ssid_launch_trajectory_lm_module(ssid_pipeline *pipeline, ssid_ticket *ticket, char *error, size_t error_size) {
    ssid_trajectory_lm_launch *launch = &ticket->lm_launch;
    ssid_cu_stream stream = pipeline->streams[ticket->stream_index];
    ssid_cu_function function = NULL;
    ssid_cu_result cuda_result;
    uint64_t num_fits;
    uint64_t ctas;
    unsigned int shared_bytes;
    double started = ssid_monotonic_seconds();
    cuda_result = pipeline->cuda.cuModuleGetFunction(
        &function, ticket->module, SSID_TRAJECTORY_LM_KERNEL);
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result,
            "cuModuleGetFunction(secant_cubin_materialize_000)", error, error_size);
        return SSID_CUDA_ERROR;
    }
    ticket->result.function_lookup_seconds = ssid_monotonic_seconds() - started;
    cuda_result = pipeline->cuda.cuEventCreate(&ticket->event, SSID_CU_EVENT_DISABLE_TIMING);
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuEventCreate", error, error_size);
        return SSID_CUDA_ERROR;
    }
    num_fits = launch->num_settings * launch->starts_per_setting;
    ctas = 1u + (num_fits - 1u) / launch->threads_per_block;
    if (ctas > UINT32_MAX) {
        snprintf(error, error_size, "trajectory-LM CTA count exceeds the CUDA grid limit");
        return SSID_OUT_OF_RANGE;
    }
    shared_bytes = 12u * launch->threads_per_block * (unsigned int)sizeof(float);
    {
        uint64_t starts_device = launch->starts_device;
        uint64_t starts_ld = launch->starts_leading_dimension;
        uint64_t bindings_device = launch->bindings_device;
        uint64_t bindings_ld = launch->bindings_leading_dimension;
        uint64_t num_settings = launch->num_settings;
        uint32_t starts_per_setting = launch->starts_per_setting;
        uint64_t reference_device = launch->reference_device;
        uint32_t steps = launch->steps_per_observation;
        uint32_t iterations = launch->max_lm_iterations;
        uint32_t damping_attempts = launch->max_damping_attempts;
        float damping = launch->initial_damping;
        uint64_t constants_device = launch->constants_device;
        uint64_t constants_ld = launch->constants_leading_dimension;
        uint64_t mse_device = launch->mse_device;
        uint64_t iterations_device = launch->iterations_device;
        uint64_t accepted_steps_device = launch->accepted_steps_device;
        void *parameters[16] = {
            &starts_device, &starts_ld, &bindings_device, &bindings_ld,
            &num_settings, &starts_per_setting, &reference_device, &steps,
            &iterations, &damping_attempts, &damping, &constants_device,
            &constants_ld, &mse_device, &iterations_device, &accepted_steps_device
        };
        started = ssid_monotonic_seconds();
        cuda_result = pipeline->cuda.cuLaunchKernel(
            function, 1u, (uint32_t)ctas, 1u,
            launch->threads_per_block, 1u, 1u, shared_bytes,
            stream, parameters, NULL);
    }
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result,
            "cuLaunchKernel(trajectory LM)", error, error_size);
        return SSID_CUDA_ERROR;
    }
    cuda_result = pipeline->cuda.cuEventRecord(ticket->event, stream);
    ticket->result.launch_seconds = ssid_monotonic_seconds() - started;
    ticket->result.launched_kernel_count = 1u;
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuEventRecord", error, error_size);
        return SSID_CUDA_ERROR;
    }
    return SSID_OK;
}

static int ssid_load_and_launch(ssid_pipeline *pipeline, ssid_ticket *ticket, char *error, size_t error_size) {
    ssid_cu_result cuda_result;
    double started;
    int status;
    pthread_mutex_lock(&pipeline->cuda_mutex);
    pipeline->cuda.cuCtxSetCurrent(pipeline->context);
    started = ssid_monotonic_seconds();
    cuda_result = pipeline->cuda.cuModuleLoadData(&ticket->module, ticket->worker->image);
    ticket->result.module_load_seconds = ssid_monotonic_seconds() - started;
    ssid_release_worker_image(pipeline, ticket);
    if (cuda_result != SSID_CUDA_SUCCESS) {
        ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuModuleLoadData", error, error_size);
        pthread_mutex_unlock(&pipeline->cuda_mutex);
        return SSID_CUDA_ERROR;
    }
    ticket->stream_index = pipeline->next_stream_index;
    pipeline->next_stream_index =
        (pipeline->next_stream_index + 1u) % pipeline->stream_count;
    status = ticket->kind == SSID_TICKET_TRAJECTORY_LM
        ? ssid_launch_trajectory_lm_module(pipeline, ticket, error, error_size)
        : ssid_launch_loaded_module(pipeline, ticket, error, error_size);
    if (status != SSID_OK) {
        ssid_synchronize_ticket_streams(pipeline, ticket);
        if (ticket->event != NULL) pipeline->cuda.cuEventDestroy_v2(ticket->event);
        ssid_destroy_dependency_events(pipeline, ticket);
        pipeline->cuda.cuModuleUnload(ticket->module);
        ticket->event = NULL;
        ticket->module = NULL;
    }
    pthread_mutex_unlock(&pipeline->cuda_mutex);
    return status;
}

static int ssid_retire_one_completed(ssid_pipeline *pipeline) {
    ssid_ticket *ticket;
    ssid_ticket *previous;
    ssid_cu_result cuda_result;
    char error[512] = {0};
    pthread_mutex_lock(&pipeline->mutex);
    ticket = pipeline->inflight_head;
    pthread_mutex_unlock(&pipeline->mutex);
    while (ticket != NULL) {
        pthread_mutex_lock(&pipeline->cuda_mutex);
        pipeline->cuda.cuCtxSetCurrent(pipeline->context);
        cuda_result = pipeline->cuda.cuEventQuery(ticket->event);
        if (cuda_result == SSID_CUDA_SUCCESS) {
            pipeline->cuda.cuEventDestroy_v2(ticket->event);
            ssid_destroy_dependency_events(pipeline, ticket);
            pipeline->cuda.cuModuleUnload(ticket->module);
            ticket->event = NULL;
            ticket->module = NULL;
        } else if (cuda_result != SSID_CUDA_ERROR_NOT_READY) {
            ssid_cuda_error_text(&pipeline->cuda, cuda_result, "cuEventQuery", error, sizeof(error));
            ssid_synchronize_ticket_streams(pipeline, ticket);
            pipeline->cuda.cuEventDestroy_v2(ticket->event);
            ssid_destroy_dependency_events(pipeline, ticket);
            pipeline->cuda.cuModuleUnload(ticket->module);
            ticket->event = NULL;
            ticket->module = NULL;
        }
        pthread_mutex_unlock(&pipeline->cuda_mutex);
        if (cuda_result != SSID_CUDA_ERROR_NOT_READY) break;
        ticket = ticket->next;
    }
    if (ticket == NULL) return 0;
    pthread_mutex_lock(&pipeline->mutex);
    previous = NULL;
    {
        ssid_ticket *current = pipeline->inflight_head;
        while (current != NULL && current != ticket) {
            previous = current;
            current = current->next;
        }
        if (current == ticket) {
            if (previous == NULL) pipeline->inflight_head = ticket->next;
            else previous->next = ticket->next;
            if (pipeline->inflight_tail == ticket) pipeline->inflight_tail = previous;
            ticket->next = NULL;
            pipeline->inflight_count -= 1;
        }
    }
    pthread_mutex_unlock(&pipeline->mutex);
    ssid_ticket_complete(pipeline, ticket, cuda_result == SSID_CUDA_SUCCESS ? SSID_OK : SSID_CUDA_ERROR, error);
    return 1;
}

static void ssid_wait_one_millisecond(ssid_pipeline *pipeline) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    pthread_cond_timedwait(&pipeline->condition, &pipeline->mutex, &deadline);
}

static void *ssid_cuda_thread_main(void *argument) {
    ssid_pipeline *pipeline = (ssid_pipeline *)argument;
    int initialization = ssid_cuda_initialize(pipeline);
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->cuda_initialization_status = initialization;
    if (initialization != SSID_OK && pipeline->cuda_initialization_error[0] == '\0') snprintf(pipeline->cuda_initialization_error, sizeof(pipeline->cuda_initialization_error), "%s", ssid_last_error());
    pipeline->cuda_initialized = 1;
    pthread_cond_broadcast(&pipeline->condition);
    pthread_mutex_unlock(&pipeline->mutex);
    if (initialization != SSID_OK) return NULL;
    for (;;) {
        ssid_ticket *ticket = NULL;
        int should_stop;
        char error[512] = {0};
        int status;
        while (ssid_retire_one_completed(pipeline)) {}
        pthread_mutex_lock(&pipeline->mutex);
        if (pipeline->ready_head != NULL && pipeline->inflight_count < pipeline->config.maximum_loaded_modules) ticket = ssid_queue_pop(&pipeline->ready_head, &pipeline->ready_tail);
        should_stop = pipeline->stopping && pipeline->ready_head == NULL && pipeline->inflight_head == NULL;
        if (ticket == NULL && !should_stop) {
            if (pipeline->inflight_head == NULL) pthread_cond_wait(&pipeline->condition, &pipeline->mutex);
            else ssid_wait_one_millisecond(pipeline);
        }
        pthread_mutex_unlock(&pipeline->mutex);
        if (should_stop) break;
        if (ticket == NULL) continue;
        status = ssid_load_and_launch(pipeline, ticket, error, sizeof(error));
        if (status != SSID_OK) {
            ssid_ticket_complete(pipeline, ticket, status, error);
            continue;
        }
        pthread_mutex_lock(&pipeline->mutex);
        ssid_queue_append(&pipeline->inflight_head, &pipeline->inflight_tail, ticket);
        pipeline->inflight_count += 1;
        pthread_mutex_unlock(&pipeline->mutex);
    }
    pthread_mutex_lock(&pipeline->cuda_mutex);
    pipeline->cuda.cuCtxSetCurrent(pipeline->context);
    {
        uint32_t stream_index;
        for (stream_index = 0; stream_index < pipeline->stream_count; ++stream_index) {
            if (pipeline->streams[stream_index] != NULL) {
                pipeline->cuda.cuStreamDestroy_v2(pipeline->streams[stream_index]);
                pipeline->streams[stream_index] = NULL;
            }
        }
    }
    if (pipeline->context != NULL) pipeline->cuda.cuCtxDestroy_v2(pipeline->context);
    pipeline->context = NULL;
    pthread_mutex_unlock(&pipeline->cuda_mutex);
    ssid_cuda_close(&pipeline->cuda);
    return NULL;
}

static void *ssid_worker_main(void *argument) {
    ssid_worker *worker = (ssid_worker *)argument;
    ssid_pipeline *pipeline = worker->pipeline;
    for (;;) {
        ssid_ticket *ticket;
        ssid_specialization_stats stats = {0};
        double started;
        int status;
        pthread_mutex_lock(&pipeline->mutex);
        while (pipeline->pending_head == NULL && !pipeline->stopping) pthread_cond_wait(&pipeline->worker_condition, &pipeline->mutex);
        if (pipeline->pending_head == NULL && pipeline->stopping) {
            pthread_mutex_unlock(&pipeline->mutex);
            break;
        }
        ticket = ssid_queue_pop(&pipeline->pending_head, &pipeline->pending_tail);
        pthread_mutex_unlock(&pipeline->mutex);
        ticket->result.queue_wait_seconds = ssid_monotonic_seconds() - ticket->submitted_at;
        ticket->result.worker_index = worker->index;
        started = ssid_monotonic_seconds();
        if (ticket->kind == SSID_TICKET_TRAJECTORY_LM) {
            memcpy(worker->image, ticket->source_cubin, ticket->source_cubin_byte_count);
            status = SSID_OK;
        } else {
            status = ssid_specialize_module(pipeline->template_value, &ticket->batch,
                worker->image, pipeline->cubin_byte_count, &stats);
        }
        ticket->result.specialization_seconds = ssid_monotonic_seconds() - started;
        ticket->result.maximum_register_count = stats.maximum_register_count;
        ticket->result.sass_instruction_count = stats.sass_instruction_count;
        if (status != SSID_OK) {
            ssid_ticket_complete(pipeline, ticket, status, ssid_last_error());
            continue;
        }
        if (!pipeline->config.enable_cuda) {
            ssid_ticket_complete(pipeline, ticket, SSID_OK, NULL);
            continue;
        }
        pthread_mutex_lock(&pipeline->mutex);
        ticket->worker = worker;
        worker->image_released = 0;
        ssid_queue_append(&pipeline->ready_head, &pipeline->ready_tail, ticket);
        pthread_cond_signal(&pipeline->condition);
        while (!worker->image_released && !pipeline->stopping) pthread_cond_wait(&worker->image_condition, &pipeline->mutex);
        pthread_mutex_unlock(&pipeline->mutex);
    }
    return NULL;
}

static void ssid_pipeline_release_storage(ssid_pipeline *pipeline) {
    uint32_t index;
    if (pipeline == NULL) return;
    if (pipeline->workers != NULL) {
        for (index = 0; index < pipeline->worker_count; ++index) {
            pthread_cond_destroy(&pipeline->workers[index].image_condition);
            free(pipeline->workers[index].image);
        }
    }
    free(pipeline->workers);
    free(pipeline->streams);
    pthread_cond_destroy(&pipeline->worker_condition);
    pthread_cond_destroy(&pipeline->condition);
    pthread_mutex_destroy(&pipeline->cuda_mutex);
    pthread_mutex_destroy(&pipeline->mutex);
    free(pipeline);
}

static int ssid_pipeline_create_internal(const ssid_template *template_value,
    size_t cubin_byte_count, const ssid_pipeline_config *config, ssid_pipeline **result) {
    ssid_pipeline *pipeline;
    uint32_t index;
    if (result != NULL) *result = NULL;
    if (cubin_byte_count == 0u || config == NULL || result == NULL || config->specialization_threads == 0 || config->queue_capacity == 0 || config->execution_streams == 0 || (config->enable_cuda && (config->maximum_loaded_modules == 0 || config->maximum_loaded_modules < config->execution_streams))) {
        ssid_set_error("invalid specialization pipeline configuration");
        return SSID_INVALID_ARGUMENT;
    }
    pipeline = (ssid_pipeline *)calloc(1, sizeof(*pipeline));
    if (pipeline == NULL) return SSID_OUT_OF_MEMORY;
    pipeline->template_value = template_value;
    pipeline->cubin_byte_count = cubin_byte_count;
    pipeline->config = *config;
    pipeline->worker_count = config->specialization_threads;
    pipeline->stream_count = config->execution_streams;
    pthread_mutex_init(&pipeline->mutex, NULL);
    pthread_mutex_init(&pipeline->cuda_mutex, NULL);
    pthread_cond_init(&pipeline->condition, NULL);
    pthread_cond_init(&pipeline->worker_condition, NULL);
    pipeline->workers = (ssid_worker *)calloc(pipeline->worker_count, sizeof(ssid_worker));
    pipeline->streams = (ssid_cu_stream *)calloc(
        pipeline->stream_count, sizeof(ssid_cu_stream));
    if (pipeline->workers == NULL || pipeline->streams == NULL) {
        ssid_pipeline_release_storage(pipeline);
        return SSID_OUT_OF_MEMORY;
    }
    for (index = 0; index < pipeline->worker_count; ++index) {
        pipeline->workers[index].pipeline = pipeline;
        pipeline->workers[index].index = index;
        pipeline->workers[index].image = (uint8_t *)malloc(cubin_byte_count);
        pthread_cond_init(&pipeline->workers[index].image_condition, NULL);
        if (pipeline->workers[index].image == NULL) {
            ssid_set_error("could not allocate one CUBIN copy per specialization worker");
            pipeline->worker_count = index + 1;
            ssid_pipeline_release_storage(pipeline);
            return SSID_OUT_OF_MEMORY;
        }
    }
    if (config->enable_cuda) {
        if (pthread_create(&pipeline->cuda_thread, NULL, ssid_cuda_thread_main, pipeline) != 0) {
            ssid_set_error("could not create the CUDA-owner thread");
            ssid_pipeline_release_storage(pipeline);
            return SSID_INTERNAL_ERROR;
        }
        pipeline->cuda_thread_started = 1;
        pthread_mutex_lock(&pipeline->mutex);
        while (!pipeline->cuda_initialized) pthread_cond_wait(&pipeline->condition, &pipeline->mutex);
        pthread_mutex_unlock(&pipeline->mutex);
        if (pipeline->cuda_initialization_status != SSID_OK) {
            int initialization_status = pipeline->cuda_initialization_status;
            pthread_join(pipeline->cuda_thread, NULL);
            pipeline->cuda_thread_started = 0;
            ssid_set_error("%s", pipeline->cuda_initialization_error);
            ssid_pipeline_release_storage(pipeline);
            return initialization_status;
        }
    }
    for (index = 0; index < pipeline->worker_count; ++index) {
        if (pthread_create(&pipeline->workers[index].thread, NULL, ssid_worker_main, &pipeline->workers[index]) != 0) {
            uint32_t joined;
            pthread_mutex_lock(&pipeline->mutex);
            pipeline->stopping = 1;
            pthread_cond_broadcast(&pipeline->condition);
            pthread_cond_broadcast(&pipeline->worker_condition);
            pthread_mutex_unlock(&pipeline->mutex);
            for (joined = 0; joined < index; ++joined) pthread_join(pipeline->workers[joined].thread, NULL);
            if (pipeline->cuda_thread_started) pthread_join(pipeline->cuda_thread, NULL);
            ssid_set_error("could not create specialization worker %u", index);
            ssid_pipeline_release_storage(pipeline);
            return SSID_INTERNAL_ERROR;
        }
        pipeline->workers[index].started = 1;
    }
    *result = pipeline;
    return SSID_OK;
}

int ssid_pipeline_create(const ssid_template *template_value, const ssid_pipeline_config *config, ssid_pipeline **result) {
    if (template_value == NULL) return SSID_INVALID_ARGUMENT;
    return ssid_pipeline_create_internal(
        template_value, template_value->cubin_byte_count, config, result);
}

int ssid_module_pipeline_create(size_t cubin_byte_count, const ssid_pipeline_config *config, ssid_pipeline **result) {
    return ssid_pipeline_create_internal(NULL, cubin_byte_count, config, result);
}

static int ssid_validate_launch(const ssid_genome_batch *batch, const ssid_fedbatch_launch *launch) {
    if (launch == NULL || launch->settings_device == 0 || launch->bindings_device == 0 || launch->reference_device == 0 || launch->num_settings == 0 || launch->num_genomes != batch->genome_count || launch->threads_per_block == 0 || launch->threads_per_block > 1024 || launch->steps_per_observation == 0) {
        ssid_set_error("invalid fed-batch CUDA launch descriptor");
        return SSID_INVALID_ARGUMENT;
    }
    if (launch->output_mode == SSID_OUTPUT_FULL_MSE) {
        if (launch->mse_device == 0) {
            ssid_set_error("full-MSE launch requires an output buffer");
            return SSID_INVALID_ARGUMENT;
        }
    } else if (launch->output_mode == SSID_OUTPUT_GENOME_WINNERS) {
        if (launch->num_settings > UINT32_MAX || (launch->threads_per_block & (launch->threads_per_block - 1u)) != 0u || launch->reduction_threads == 0 || launch->reduction_threads > 1024 || (launch->reduction_threads & (launch->reduction_threads - 1u)) != 0u || launch->cta_score_device == 0 || launch->cta_setting_device == 0 || launch->winner_score_device == 0 || launch->winner_setting_device == 0) {
            ssid_set_error("winner launch requires power-of-two blocks and all reduction buffers");
            return SSID_INVALID_ARGUMENT;
        }
    } else {
        ssid_set_error("unknown fed-batch output mode %u", launch->output_mode);
        return SSID_INVALID_ARGUMENT;
    }
    return SSID_OK;
}

int ssid_pipeline_submit(ssid_pipeline *pipeline, const ssid_genome_batch *batch, const ssid_fedbatch_launch *launch, ssid_ticket **result) {
    ssid_ticket *ticket;
    int status;
    if (result != NULL) *result = NULL;
    if (pipeline == NULL || pipeline->template_value == NULL || result == NULL || (status = ssid_validate_batch(pipeline->template_value, batch)) != SSID_OK) return pipeline == NULL || pipeline->template_value == NULL || result == NULL ? SSID_INVALID_ARGUMENT : status;
    if (pipeline->config.enable_cuda && (status = ssid_validate_launch(batch, launch)) != SSID_OK) return status;
    ticket = (ssid_ticket *)calloc(1, sizeof(*ticket));
    if (ticket == NULL) return SSID_OUT_OF_MEMORY;
    pthread_mutex_init(&ticket->mutex, NULL);
    pthread_cond_init(&ticket->condition, NULL);
    ticket->batch = *batch;
    ticket->kind = SSID_TICKET_FEDBATCH;
    if (launch != NULL) ticket->launch = *launch;
    ticket->submitted_at = ssid_monotonic_seconds();
    pthread_mutex_lock(&pipeline->mutex);
    if (pipeline->stopping || pipeline->active_jobs >= pipeline->config.queue_capacity) {
        pthread_mutex_unlock(&pipeline->mutex);
        pthread_cond_destroy(&ticket->condition);
        pthread_mutex_destroy(&ticket->mutex);
        free(ticket);
        ssid_set_error(pipeline->stopping ? "pipeline is stopping" : "specialization queue is full");
        return pipeline->stopping ? SSID_INVALID_ARGUMENT : SSID_QUEUE_FULL;
    }
    pipeline->active_jobs += 1;
    ssid_queue_append(&pipeline->pending_head, &pipeline->pending_tail, ticket);
    pthread_cond_signal(&pipeline->worker_condition);
    pthread_mutex_unlock(&pipeline->mutex);
    *result = ticket;
    return SSID_OK;
}

static int ssid_validate_trajectory_lm_launch(const ssid_pipeline *pipeline,
    const uint8_t *cubin, size_t cubin_byte_count,
    const ssid_trajectory_lm_launch *launch) {
    uint64_t num_fits;
    if (pipeline == NULL || pipeline->template_value != NULL || cubin == NULL ||
        cubin_byte_count != pipeline->cubin_byte_count || launch == NULL ||
        launch->starts_device == 0u || launch->bindings_device == 0u ||
        launch->reference_device == 0u || launch->constants_device == 0u ||
        launch->mse_device == 0u || launch->iterations_device == 0u ||
        launch->accepted_steps_device == 0u || launch->num_settings == 0u ||
        launch->starts_per_setting == 0u || launch->steps_per_observation == 0u ||
        launch->max_lm_iterations == 0u || launch->max_damping_attempts == 0u ||
        !(launch->initial_damping > 0.0f) || launch->threads_per_block == 0u ||
        launch->threads_per_block > 1024u ||
        launch->num_settings > UINT64_MAX / launch->starts_per_setting) {
        ssid_set_error("invalid trajectory-LM queue launch descriptor");
        return SSID_INVALID_ARGUMENT;
    }
    num_fits = launch->num_settings * launch->starts_per_setting;
    if (1u + (num_fits - 1u) / launch->threads_per_block > UINT32_MAX) {
        ssid_set_error("trajectory-LM queue grid exceeds the CUDA y dimension");
        return SSID_OUT_OF_RANGE;
    }
    return SSID_OK;
}

int ssid_pipeline_submit_trajectory_lm(ssid_pipeline *pipeline,
    const uint8_t *cubin, size_t cubin_byte_count,
    const ssid_trajectory_lm_launch *launch, ssid_ticket **result) {
    ssid_ticket *ticket;
    int status;
    if (result != NULL) *result = NULL;
    if (result == NULL ||
        (status = ssid_validate_trajectory_lm_launch(
            pipeline, cubin, cubin_byte_count, launch)) != SSID_OK) {
        return result == NULL ? SSID_INVALID_ARGUMENT : status;
    }
    ticket = (ssid_ticket *)calloc(1, sizeof(*ticket));
    if (ticket == NULL) return SSID_OUT_OF_MEMORY;
    pthread_mutex_init(&ticket->mutex, NULL);
    pthread_cond_init(&ticket->condition, NULL);
    ticket->kind = SSID_TICKET_TRAJECTORY_LM;
    ticket->source_cubin = cubin;
    ticket->source_cubin_byte_count = cubin_byte_count;
    ticket->lm_launch = *launch;
    ticket->submitted_at = ssid_monotonic_seconds();
    pthread_mutex_lock(&pipeline->mutex);
    if (pipeline->stopping || pipeline->active_jobs >= pipeline->config.queue_capacity) {
        pthread_mutex_unlock(&pipeline->mutex);
        pthread_cond_destroy(&ticket->condition);
        pthread_mutex_destroy(&ticket->mutex);
        free(ticket);
        ssid_set_error(pipeline->stopping ? "pipeline is stopping" : "module queue is full");
        return pipeline->stopping ? SSID_INVALID_ARGUMENT : SSID_QUEUE_FULL;
    }
    pipeline->active_jobs += 1;
    ssid_queue_append(&pipeline->pending_head, &pipeline->pending_tail, ticket);
    pthread_cond_signal(&pipeline->worker_condition);
    pthread_mutex_unlock(&pipeline->mutex);
    *result = ticket;
    return SSID_OK;
}

int ssid_ticket_poll(ssid_ticket *ticket, int *complete, ssid_ticket_result *result) {
    if (ticket == NULL || complete == NULL) return SSID_INVALID_ARGUMENT;
    pthread_mutex_lock(&ticket->mutex);
    *complete = ticket->complete;
    if (ticket->complete && result != NULL) *result = ticket->result;
    pthread_mutex_unlock(&ticket->mutex);
    return SSID_OK;
}

int ssid_ticket_wait(ssid_ticket *ticket, ssid_ticket_result *result) {
    if (ticket == NULL) return SSID_INVALID_ARGUMENT;
    pthread_mutex_lock(&ticket->mutex);
    while (!ticket->complete) pthread_cond_wait(&ticket->condition, &ticket->mutex);
    if (result != NULL) *result = ticket->result;
    pthread_mutex_unlock(&ticket->mutex);
    return ticket->result.status;
}

const char *ssid_ticket_error(const ssid_ticket *ticket) {
    return ticket == NULL ? "invalid ticket" : ticket->error;
}

void ssid_ticket_destroy(ssid_ticket *ticket) {
    if (ticket == NULL) return;
    pthread_mutex_lock(&ticket->mutex);
    if (!ticket->complete) {
        pthread_mutex_unlock(&ticket->mutex);
        return;
    }
    pthread_mutex_unlock(&ticket->mutex);
    pthread_cond_destroy(&ticket->condition);
    pthread_mutex_destroy(&ticket->mutex);
    free(ticket);
}

int ssid_pipeline_wait_idle(ssid_pipeline *pipeline) {
    if (pipeline == NULL) return SSID_INVALID_ARGUMENT;
    pthread_mutex_lock(&pipeline->mutex);
    while (pipeline->active_jobs != 0) pthread_cond_wait(&pipeline->condition, &pipeline->mutex);
    pthread_mutex_unlock(&pipeline->mutex);
    return SSID_OK;
}

void ssid_pipeline_destroy(ssid_pipeline *pipeline) {
    uint32_t index;
    if (pipeline == NULL) return;
    ssid_pipeline_wait_idle(pipeline);
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stopping = 1;
    pthread_cond_broadcast(&pipeline->condition);
    pthread_cond_broadcast(&pipeline->worker_condition);
    for (index = 0; index < pipeline->worker_count; ++index) pthread_cond_broadcast(&pipeline->workers[index].image_condition);
    pthread_mutex_unlock(&pipeline->mutex);
    for (index = 0; index < pipeline->worker_count; ++index) if (pipeline->workers[index].started) pthread_join(pipeline->workers[index].thread, NULL);
    if (pipeline->cuda_thread_started) pthread_join(pipeline->cuda_thread, NULL);
    ssid_pipeline_release_storage(pipeline);
}

static int ssid_device_operation_begin(ssid_pipeline *pipeline) {
    ssid_cu_result result;
    if (pipeline == NULL || !pipeline->config.enable_cuda || pipeline->context == NULL) {
        ssid_set_error("device memory operations require a CUDA-enabled pipeline");
        return SSID_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pipeline->cuda_mutex);
    result = pipeline->cuda.cuCtxSetCurrent(pipeline->context);
    if (result != SSID_CUDA_SUCCESS) {
        char message[512];
        ssid_cuda_error_text(&pipeline->cuda, result, "cuCtxSetCurrent", message, sizeof(message));
        pthread_mutex_unlock(&pipeline->cuda_mutex);
        ssid_set_error("%s", message);
        return SSID_CUDA_ERROR;
    }
    return SSID_OK;
}

static int ssid_device_operation_end(ssid_pipeline *pipeline, ssid_cu_result result, const char *operation) {
    if (result != SSID_CUDA_SUCCESS) {
        char message[512];
        ssid_cuda_error_text(&pipeline->cuda, result, operation, message, sizeof(message));
        pthread_mutex_unlock(&pipeline->cuda_mutex);
        ssid_set_error("%s", message);
        return SSID_CUDA_ERROR;
    }
    pthread_mutex_unlock(&pipeline->cuda_mutex);
    return SSID_OK;
}

int ssid_device_alloc(ssid_pipeline *pipeline, size_t byte_count, uint64_t *device_pointer) {
    ssid_cu_result result;
    int status;
    if (byte_count == 0 || device_pointer == NULL) return SSID_INVALID_ARGUMENT;
    if ((status = ssid_device_operation_begin(pipeline)) != SSID_OK) return status;
    result = pipeline->cuda.cuMemAlloc_v2(device_pointer, byte_count);
    return ssid_device_operation_end(pipeline, result, "cuMemAlloc");
}

int ssid_device_free(ssid_pipeline *pipeline, uint64_t device_pointer) {
    ssid_cu_result result;
    int status;
    if (device_pointer == 0) return SSID_INVALID_ARGUMENT;
    if ((status = ssid_device_operation_begin(pipeline)) != SSID_OK) return status;
    result = pipeline->cuda.cuMemFree_v2(device_pointer);
    return ssid_device_operation_end(pipeline, result, "cuMemFree");
}

int ssid_device_upload(ssid_pipeline *pipeline, uint64_t destination, const void *source, size_t byte_count) {
    ssid_cu_result result;
    int status;
    if (destination == 0 || source == NULL || byte_count == 0) return SSID_INVALID_ARGUMENT;
    if ((status = ssid_device_operation_begin(pipeline)) != SSID_OK) return status;
    result = pipeline->cuda.cuMemcpyHtoD_v2(destination, source, byte_count);
    return ssid_device_operation_end(pipeline, result, "cuMemcpyHtoD");
}

int ssid_device_download(ssid_pipeline *pipeline, void *destination, uint64_t source, size_t byte_count) {
    ssid_cu_result result;
    int status;
    if (destination == NULL || source == 0 || byte_count == 0) return SSID_INVALID_ARGUMENT;
    if ((status = ssid_device_operation_begin(pipeline)) != SSID_OK) return status;
    result = pipeline->cuda.cuMemcpyDtoH_v2(destination, source, byte_count);
    return ssid_device_operation_end(pipeline, result, "cuMemcpyDtoH");
}
