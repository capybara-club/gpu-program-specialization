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

#include "o_odezza_internal.h"

#include "o_sha256.h"
#include "o_specialize_scoring_cubin.h"

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define O_SCORING_KERNEL_NAME "odezza_scoring"
#define O_ERROR_BYTES 512u

typedef CUresult OCuResult;
typedef CUmodule OCuModule;
typedef CUfunction OCuFunction;
typedef CUstream OCuStream;
typedef CUevent OCuEvent;

typedef struct OdezzaScoringPipelineConfig {
    uint32_t specialization_thread_count;
    uint32_t slots_per_thread;
    uint32_t queue_capacity;
    uint32_t maximum_loaded_modules;
    uint32_t execution_stream_count;
    uint32_t completion_poll_interval_ns;
    uint32_t threads_per_block;
} OdezzaScoringPipelineConfig;

typedef struct OdezzaScoringTicketResult {
    OdezzaResult status;
    uint32_t worker_index;
    uint32_t slot_index;
    uint32_t stream_index;
    OdezzaScoringSpecializationReport specialization;
    double queue_wait_seconds;
    double cubin_reset_seconds;
    double specialization_seconds;
    double module_load_seconds;
    double function_lookup_seconds;
    double launch_seconds;
    double completion_wait_seconds;
    double module_unload_seconds;
    double total_seconds;
    double gpu_begin_seconds,gpu_end_seconds;
    uint32_t shared_memory_bytes;
    int profiled;
} OdezzaScoringTicketResult;
typedef struct OScoringInterval { double begin,end; } OScoringInterval;

typedef struct OIndexRing {
    uint32_t *values;
    uint32_t capacity;
    uint32_t head;
    uint32_t count;
} OIndexRing;

typedef struct OScoringSlot {
    unsigned char *cubin;
    void *workspace;
    uint32_t ticket_index;
} OScoringSlot;

typedef struct OScoringWorker {
    struct OdezzaScoringPipeline *pipeline;
    pthread_t thread;
    uint32_t index;
    int started;
} OScoringWorker;

typedef struct OdezzaScoringTicket OdezzaScoringTicket;

struct OdezzaScoringTicket {
    struct OdezzaScoringPipeline *pipeline;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    const OdezzaScoringSystem *systems;
    size_t system_count;
    OdezzaScoringLaunch launch;
    OdezzaScoringTicketResult result;
    double submitted_at;
    double completion_wait_started;
    OCuModule module;
    OCuEvent event,profile_begin,profile_end;
    uint32_t index;
    uint32_t checked_out;
    uint32_t complete;
    char error[O_ERROR_BYTES];
};

struct OdezzaScoringPipeline {
    OdezzaScoringPipelineConfig config;
    char *generated_source;
    size_t generated_source_size;
    unsigned char *template_cubin;
    size_t cubin_size;
    void *inspection_storage;
    OdezzaScoringCubinInspection inspection;
    OdezzaScoringPrespecialization prespecialization;
    size_t workspace_size;

    OScoringWorker *workers;
    uint32_t worker_count;
    uint32_t started_worker_count;
    OScoringSlot *slots;
    uint32_t slot_count;
    unsigned char *slot_cubins;
    unsigned char *slot_workspaces;
    size_t slot_cubin_stride;
    size_t slot_workspace_stride;
    size_t run_workspace_size;
    OdezzaScoringTicket *tickets;
    OdezzaScoringTicket **run_tickets;
    OScoringInterval *intervals;
    OCuEvent profile_origin;
    uint32_t ticket_count;
    uint32_t initialized_ticket_count;

    uint32_t *free_slot_values;
    uint32_t *pending_ticket_values;
    uint32_t *ready_slot_values;
    uint32_t *free_ticket_values;
    uint32_t *inflight_ticket_values;
    OIndexRing free_slots;
    OIndexRing pending_tickets;
    OIndexRing ready_slots;
    OIndexRing free_tickets;
    uint32_t inflight_count;

    pthread_mutex_t mutex;
    pthread_cond_t worker_condition;
    pthread_cond_t ready_condition;
    int mutex_initialized;
    int worker_condition_initialized;
    int ready_condition_initialized;
    int stopping;
    int run_active;
    int creation_complete;
    uint32_t active_ticket_count;
    uint32_t checked_out_ticket_count;

    OCuStream *streams;
    uint32_t stream_count;
    uint32_t next_stream_index;
    uint32_t threads_per_block;

    OdezzaScoringPipelineStats stats;
    char error[O_ERROR_BYTES];
};

static int o_add_size(size_t lhs, size_t rhs, size_t *result) {
    if (result == NULL || lhs > SIZE_MAX - rhs) return 0;
    *result = lhs + rhs;
    return 1;
}

static int o_multiply_size(size_t lhs, size_t rhs, size_t *result) {
    if (result == NULL || (rhs != 0u && lhs > SIZE_MAX / rhs)) return 0;
    *result = lhs * rhs;
    return 1;
}

static int o_multiply_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
    if (result == NULL || (rhs != 0u && lhs > UINT64_MAX / rhs)) return 0;
    *result = lhs * rhs;
    return 1;
}

static int o_align_size(size_t value, size_t *result) {
    if (value > SIZE_MAX - 7u) return 0;
    *result = (value + 7u) & ~(size_t)7u;
    return 1;
}

static double o_monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + 1.0e-9 * (double)value.tv_nsec;
}

static void o_ring_initialize(OIndexRing *ring, uint32_t *values, uint32_t capacity) {
    ring->values = values;
    ring->capacity = capacity;
    ring->head = 0u;
    ring->count = 0u;
}

static int o_ring_push(OIndexRing *ring, uint32_t value) {
    uint32_t tail;
    if (ring == NULL || ring->count == ring->capacity) return 0;
    tail = (ring->head + ring->count) % ring->capacity;
    ring->values[tail] = value;
    ring->count += 1u;
    return 1;
}

static int o_ring_pop(OIndexRing *ring, uint32_t *value_ret) {
    if (ring == NULL || value_ret == NULL || ring->count == 0u) return 0;
    *value_ret = ring->values[ring->head];
    ring->head = (ring->head + 1u) % ring->capacity;
    ring->count -= 1u;
    return 1;
}

static void o_set_error(OdezzaScoringTicket *ticket, const char *message) {
    if (ticket == NULL) return;
    if (message == NULL) message = "";
    snprintf(ticket->error, sizeof(ticket->error), "%s", message);
}

static void o_set_pipeline_error(OdezzaScoringPipeline *pipeline, const char *format, ...) {
    va_list arguments;
    if (pipeline == NULL) return;
    if (format == NULL) {
        pipeline->error[0] = '\0';
        return;
    }
    va_start(arguments, format);
    (void)vsnprintf(pipeline->error, sizeof(pipeline->error), format, arguments);
    va_end(arguments);
}

static void o_cuda_error(OCuResult result, const char *operation, char *buffer, size_t buffer_size) {
    const char *name = NULL;
    const char *description = NULL;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
    snprintf(buffer, buffer_size, "%s failed with CUDA result %d%s%s%s%s%s", operation, result, name == NULL ? "" : " (", name == NULL ? "" : name,
             name == NULL ? "" : ")", description == NULL ? "" : ": ", description == NULL ? "" : description);
}

static OdezzaResult o_copy_array(
    unsigned char *storage,
    size_t storage_size,
    size_t *offset,
    const void *source,
    size_t count,
    size_t element_size,
    void **destination_ret
) {
    size_t aligned;
    size_t bytes;
    size_t end;
    if (offset == NULL || destination_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *destination_ret = NULL;
    if (count == 0u) return ODEZZA_SUCCESS;
    if (source == NULL || !o_align_size(*offset, &aligned) || !o_multiply_size(count, element_size, &bytes) || !o_add_size(aligned, bytes, &end))
        return ODEZZA_ERROR_OVERFLOW;
    if (storage != NULL) {
        if (end > storage_size) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
        memcpy(storage + aligned, source, bytes);
        *destination_ret = storage + aligned;
    }
    *offset = end;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_measure_inspection(
    const OdezzaScoringCubinInspection *source,
    size_t *required_ret
) {
    size_t required = 0u;
    void *pointer;
#define O_MEASURE(field, count, type) \
    do { \
        O_RETURN_IF_ERROR(o_copy_array(NULL, 0u, &required, source->field, (count), sizeof(type), &pointer)); \
    } while (0)
    if (source == NULL || required_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_MEASURE(register_count_file_offsets, source->register_count_file_offset_count, size_t);
    O_MEASURE(register_count_header_file_offsets, source->register_count_header_file_offset_count, size_t);
    O_MEASURE(dispatch_file_offsets, source->dispatch_instruction_count, size_t);
    O_MEASURE(dispatch_instructions, source->dispatch_instruction_count, OdezzaScoringInstruction);
    O_MEASURE(input_registers, source->input_count, uint8_t);
    O_MEASURE(output_registers, source->output_count, uint8_t);
    O_MEASURE(final_output_registers, source->output_count, uint8_t);
    O_MEASURE(output_materialization_file_offsets, source->output_count, size_t);
    O_MEASURE(available_registers, source->available_register_count, uint8_t);
    O_MEASURE(cleanup_file_offsets, source->cleanup_file_offset_count, size_t);
    O_MEASURE(target_table_file_offsets, source->system_capacity > 1u ? source->system_capacity : 0u, size_t);
    O_MEASURE(original_target_values, source->system_capacity > 1u ? source->system_capacity : 0u, uint32_t);
#undef O_MEASURE
    *required_ret = required;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_inspection(
    const OdezzaScoringCubinInspection *source,
    unsigned char *storage,
    OdezzaScoringCubinInspection *destination
) {
    size_t required = 0u;
    void *pointer;
    *destination = *source;
#define O_COPY(field, count, type) \
    do { \
        O_RETURN_IF_ERROR(o_copy_array(storage, SIZE_MAX, &required, source->field, (count), sizeof(type), &pointer)); \
        destination->field = (const type *)pointer; \
    } while (0)
    O_COPY(register_count_file_offsets, source->register_count_file_offset_count, size_t);
    O_COPY(register_count_header_file_offsets, source->register_count_header_file_offset_count, size_t);
    O_COPY(dispatch_file_offsets, source->dispatch_instruction_count, size_t);
    O_COPY(dispatch_instructions, source->dispatch_instruction_count, OdezzaScoringInstruction);
    O_COPY(input_registers, source->input_count, uint8_t);
    O_COPY(output_registers, source->output_count, uint8_t);
    O_COPY(final_output_registers, source->output_count, uint8_t);
    O_COPY(output_materialization_file_offsets, source->output_count, size_t);
    O_COPY(available_registers, source->available_register_count, uint8_t);
    O_COPY(cleanup_file_offsets, source->cleanup_file_offset_count, size_t);
    O_COPY(target_table_file_offsets, source->system_capacity > 1u ? source->system_capacity : 0u, size_t);
    O_COPY(original_target_values, source->system_capacity > 1u ? source->system_capacity : 0u, uint32_t);
#undef O_COPY
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_copy_inspection(
    const OdezzaScoringCubinInspection *source,
    void **storage_ret,
    OdezzaScoringCubinInspection *destination
) {
    size_t required;
    unsigned char *storage = NULL;
    OdezzaResult result;
    if (source == NULL || storage_ret == NULL || destination == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *storage_ret = NULL;
    O_RETURN_IF_ERROR(o_measure_inspection(source, &required));
    storage = (unsigned char *)malloc(required == 0u ? 1u : required);
    if (storage == NULL) return ODEZZA_ERROR_ALLOCATION;
    result = o_write_inspection(source, storage, destination);
    if (result != ODEZZA_SUCCESS) {
        if (storage != NULL) free(storage);
        return result;
    }
    *storage_ret = storage;
    return ODEZZA_SUCCESS;
}

static void o_pipeline_record_completion(
    OdezzaScoringPipeline *pipeline,
    OdezzaScoringTicket *ticket,
    OdezzaResult status,
    const char *error
) {
    pthread_mutex_lock(&ticket->mutex);
    ticket->result.status = status;
    ticket->result.total_seconds = o_monotonic_seconds() - ticket->submitted_at;
    o_set_error(ticket, error);
    ticket->complete = 1u;
    pthread_cond_broadcast(&ticket->condition);
    pthread_mutex_unlock(&ticket->mutex);

    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stats.completed_ticket_count += 1u;
    if (status == ODEZZA_ERROR_CUDA) pipeline->stats.cuda_failure_count += 1u;
    if (pipeline->active_ticket_count != 0u) pipeline->active_ticket_count -= 1u;
    pthread_cond_signal(&pipeline->ready_condition);
    pthread_mutex_unlock(&pipeline->mutex);
}

static void o_pipeline_release_slot(OdezzaScoringPipeline *pipeline, uint32_t slot_index) {
    pthread_mutex_lock(&pipeline->mutex);
    (void)o_ring_push(&pipeline->free_slots, slot_index);
    pthread_cond_signal(&pipeline->worker_condition);
    pthread_mutex_unlock(&pipeline->mutex);
}

static OdezzaResult o_cuda_resources_destroy(OdezzaScoringPipeline *pipeline, char error[O_ERROR_BYTES]) {
    uint32_t index;
    OCuResult cuda_result;
    OdezzaResult result = ODEZZA_SUCCESS;
    if (pipeline == NULL) return ODEZZA_SUCCESS;
    if(pipeline->profile_origin) {
        cuda_result=cuEventDestroy(pipeline->profile_origin);
        if(cuda_result==CUDA_SUCCESS)pipeline->profile_origin=NULL;
        else {o_cuda_error(cuda_result,"cuEventDestroy(profile_origin)",error,O_ERROR_BYTES);result=ODEZZA_ERROR_CUDA;}
    }
    for (index = 0u; pipeline->tickets != NULL && index < pipeline->initialized_ticket_count; ++index) {
        OCuEvent *timing[]={&pipeline->tickets[index].profile_begin,&pipeline->tickets[index].profile_end};
        unsigned k;
        for(k=0;k<2;k++)if(*timing[k]) {
            cuda_result=cuEventDestroy(*timing[k]);
            if(cuda_result==CUDA_SUCCESS)*timing[k]=NULL;
            else {o_cuda_error(cuda_result,"cuEventDestroy(profile)",error,O_ERROR_BYTES);result=ODEZZA_ERROR_CUDA;}
        }
        if (pipeline->tickets[index].event != NULL) {
            cuda_result = cuEventDestroy(pipeline->tickets[index].event);
            if (cuda_result != CUDA_SUCCESS && result == ODEZZA_SUCCESS) {
                o_cuda_error(cuda_result, "cuEventDestroy", error, O_ERROR_BYTES);
                result = ODEZZA_ERROR_CUDA;
            }
            if (cuda_result == CUDA_SUCCESS) pipeline->tickets[index].event = NULL;
        }
    }
    for (index = 0u; pipeline->streams != NULL && index < pipeline->stream_count; ++index) {
        if (pipeline->streams[index] != NULL) {
            cuda_result = cuStreamDestroy(pipeline->streams[index]);
            if (cuda_result != CUDA_SUCCESS && result == ODEZZA_SUCCESS) {
                o_cuda_error(cuda_result, "cuStreamDestroy", error, O_ERROR_BYTES);
                result = ODEZZA_ERROR_CUDA;
            }
            if (cuda_result == CUDA_SUCCESS) pipeline->streams[index] = NULL;
        }
    }
    return result;
}

static OdezzaResult o_cuda_call(OCuResult cuda_result, const char *label, char error[O_ERROR_BYTES]) {
    if (cuda_result == CUDA_SUCCESS) return ODEZZA_SUCCESS;
    o_cuda_error(cuda_result, label, error, O_ERROR_BYTES);
    return ODEZZA_ERROR_CUDA;
}

static OdezzaResult o_cuda_resources_initialize(OdezzaScoringPipeline *pipeline, char error[O_ERROR_BYTES]) {
    uint32_t index;
    CUmoduleLoadingMode loading_mode = CU_MODULE_LAZY_LOADING;

    O_RETURN_IF_ERROR(o_cuda_call(cuModuleGetLoadingMode(&loading_mode), "cuModuleGetLoadingMode", error));
    if (loading_mode != CU_MODULE_EAGER_LOADING) {
        snprintf(
            error,
            O_ERROR_BYTES,
            "CUDA module loading is not eager; the context owner must set CUDA_MODULE_LOADING=EAGER before CUDA initialization"
        );
        return ODEZZA_ERROR_UNSUPPORTED;
    }
    for (index = 0u; index < pipeline->stream_count; ++index) {
        O_RETURN_IF_ERROR(o_cuda_call(cuStreamCreate(&pipeline->streams[index], CU_STREAM_NON_BLOCKING), "cuStreamCreate", error));
    }
    O_RETURN_IF_ERROR(o_cuda_call(cuEventCreate(&pipeline->profile_origin,0),"cuEventCreate(profile_origin)",error));
    for (index = 0u; index < pipeline->initialized_ticket_count; ++index) {
        O_RETURN_IF_ERROR(o_cuda_call(cuEventCreate(&pipeline->tickets[index].profile_begin,0),"cuEventCreate(profile_begin)",error));
        O_RETURN_IF_ERROR(o_cuda_call(cuEventCreate(&pipeline->tickets[index].profile_end,0),"cuEventCreate(profile_end)",error));
        O_RETURN_IF_ERROR(o_cuda_call(cuEventCreate(&pipeline->tickets[index].event, CU_EVENT_DISABLE_TIMING), "cuEventCreate", error));
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_cuda_resources_create(OdezzaScoringPipeline *pipeline, char error[O_ERROR_BYTES]) {
    char cleanup_error[O_ERROR_BYTES] = {0};
    OdezzaResult result = o_cuda_resources_initialize(pipeline, error);
    if (result != ODEZZA_SUCCESS) (void)o_cuda_resources_destroy(pipeline, cleanup_error);
    return result;
}

static OdezzaResult o_validate_launch(
    const OdezzaScoringPipeline *pipeline,
    const OdezzaScoringLaunch *launch,
    size_t system_count,
    uint64_t *configuration_count_ret,
    uint32_t *grid_x_ret,
    uint32_t *shared_memory_bytes_ret,
    char error[O_ERROR_BYTES]
) {
    uint64_t configurations;
    uint64_t grid_x;
    size_t shared_words;
    size_t shared_memory_bytes;
    if ((pipeline->prespecialization.constant_count != 0u && launch->constant_banks_device == 0u && launch->sampled_parameters_device == 0u) || launch->constant_bank_count == 0u ||
        launch->reference_data_device == 0u ||
        launch->trajectory_offsets_device == 0u || launch->trajectory_times_device == 0u || launch->trajectory_count == 0u ||
        launch->trajectory_point_count < launch->trajectory_count || launch->steps_per_observation == 0u ||
        launch->mse_output_device == 0u || launch->active_toggle_count > ODEZZA_MAX_TOGGLE_BITS || launch->allow_missing_observations > 1u
    ) {
        snprintf(error, O_ERROR_BYTES, "scoring launch contains a null device pointer or zero dimension");
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (pipeline->prespecialization.state_count > UINT32_MAX / launch->trajectory_point_count) {
        snprintf(error, O_ERROR_BYTES, "runtime reference layout overflows the kernel index ABI");
        return ODEZZA_ERROR_OVERFLOW;
    }
    configurations = (uint64_t)launch->constant_bank_count * (UINT64_C(1) << launch->active_toggle_count);
    grid_x = (configurations + pipeline->threads_per_block - 1u) / pipeline->threads_per_block;
    if (grid_x == 0u || grid_x > INT32_MAX || system_count > UINT16_MAX) {
        snprintf(error, O_ERROR_BYTES, "scoring launch exceeds the supported grid dimensions");
        return ODEZZA_ERROR_UNSUPPORTED;
    }
    if (!o_multiply_size(pipeline->prespecialization.state_count, launch->trajectory_point_count, &shared_words) ||
        !o_add_size(shared_words, launch->trajectory_point_count, &shared_words) ||
        !o_add_size(shared_words, (size_t)launch->trajectory_count + 1u, &shared_words) ||
        !o_multiply_size(shared_words, sizeof(uint32_t), &shared_memory_bytes) || shared_memory_bytes > INT32_MAX
    ) {
        snprintf(error, O_ERROR_BYTES, "runtime trajectory shared-memory size overflows the launch ABI");
        return ODEZZA_ERROR_OVERFLOW;
    }
    *configuration_count_ret = configurations;
    *grid_x_ret = (uint32_t)grid_x;
    *shared_memory_bytes_ret = (uint32_t)shared_memory_bytes;
    return ODEZZA_SUCCESS;
}

/* Check complete byte spans, including the last module's accesses. The caller
 * still owns allocation sizes and pointer validity in its current context. */
static int o_device_span_valid(uint64_t address, uint64_t count, uint64_t element_size) {
    uint64_t bytes;
    return o_multiply_u64(count, element_size, &bytes) &&
           (bytes == 0u || (address % element_size == 0u && bytes - 1u <= UINT64_MAX - address));
}

static OdezzaResult o_load_and_launch(OdezzaScoringPipeline *pipeline, uint32_t slot_index, char error[O_ERROR_BYTES]) {
    OScoringSlot *slot = &pipeline->slots[slot_index];
    OdezzaScoringTicket *ticket = &pipeline->tickets[slot->ticket_index];
    OCuFunction function = NULL;
    OCuResult cuda_result;
    OCuStream stream;
    uint64_t configurations = 0u;
    uint32_t grid_x = 0u;
    uint32_t shared_memory_bytes = 0u;
    double started;
    OdezzaResult result;
    uint64_t constant_banks_device;
    uint32_t constant_bank_count;
    uint64_t trajectory_offsets_device;
    uint64_t trajectory_times_device;
    uint64_t reference_data_device;
    uint32_t trajectory_count;
    uint32_t trajectory_point_count;
    uint32_t active_state_count;
    uint32_t active_constant_count;
    uint32_t active_toggle_count;
    uint32_t active_system_count;
    uint32_t steps_per_observation;
    uint64_t mse_output_device;
    uint64_t sampled_parameters_device, uniform_pool_device, normal_pool_device, rng_pool_size;
    uint32_t allow_missing_observations;
    void *arguments[18];

    result = o_validate_launch(pipeline, &ticket->launch, ticket->system_count, &configurations, &grid_x, &shared_memory_bytes, error);
    (void)configurations;
    if (result != ODEZZA_SUCCESS) {
        o_pipeline_release_slot(pipeline, slot_index);
        return result;
    }

    started = o_monotonic_seconds();
    cuda_result = cuModuleLoadData(&ticket->module, slot->cubin);
    ticket->result.module_load_seconds = o_monotonic_seconds() - started;
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stats.module_load_seconds += ticket->result.module_load_seconds;
    pthread_mutex_unlock(&pipeline->mutex);

    /* cuModuleLoadData has consumed the image; release this exact slot now. */
    o_pipeline_release_slot(pipeline, slot_index);
    if (cuda_result != CUDA_SUCCESS) {
        o_cuda_error(cuda_result, "cuModuleLoadData", error, O_ERROR_BYTES);
        return ODEZZA_ERROR_CUDA;
    }

    started = o_monotonic_seconds();
    cuda_result = cuModuleGetFunction(&function, ticket->module, O_SCORING_KERNEL_NAME);
    ticket->result.function_lookup_seconds = o_monotonic_seconds() - started;
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stats.function_lookup_seconds += ticket->result.function_lookup_seconds;
    pthread_mutex_unlock(&pipeline->mutex);
    if (cuda_result != CUDA_SUCCESS) {
        o_cuda_error(cuda_result, "cuModuleGetFunction", error, O_ERROR_BYTES);
        (void)cuModuleUnload(ticket->module);
        ticket->module = NULL;
        return ODEZZA_ERROR_CUDA;
    }
    if (shared_memory_bytes > 48u * 1024u) {
        cuda_result = cuFuncSetAttribute(
            function,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            (int)shared_memory_bytes
        );
        if (cuda_result != CUDA_SUCCESS) {
            o_cuda_error(cuda_result, "cuFuncSetAttribute", error, O_ERROR_BYTES);
            (void)cuModuleUnload(ticket->module);
            ticket->module = NULL;
            return ODEZZA_ERROR_CUDA;
        }
    }

    ticket->result.stream_index = pipeline->next_stream_index;
    pipeline->next_stream_index = (pipeline->next_stream_index + 1u) % pipeline->stream_count;
    stream = pipeline->streams[ticket->result.stream_index];
    constant_banks_device = ticket->launch.constant_banks_device;
    constant_bank_count = ticket->launch.constant_bank_count;
    trajectory_offsets_device = ticket->launch.trajectory_offsets_device;
    trajectory_times_device = ticket->launch.trajectory_times_device;
    reference_data_device = ticket->launch.reference_data_device;
    trajectory_count = ticket->launch.trajectory_count;
    trajectory_point_count = ticket->launch.trajectory_point_count;
    active_state_count = (uint32_t)pipeline->prespecialization.state_count;
    active_constant_count = pipeline->prespecialization.constant_count;
    active_toggle_count = ticket->launch.active_toggle_count;
    active_system_count = (uint32_t)ticket->system_count;
    steps_per_observation = ticket->launch.steps_per_observation;
    mse_output_device = ticket->launch.mse_output_device;
    arguments[0] = &constant_banks_device;
    arguments[1] = &constant_bank_count;
    arguments[2] = &trajectory_offsets_device;
    arguments[3] = &trajectory_times_device;
    arguments[4] = &reference_data_device;
    arguments[5] = &trajectory_count;
    arguments[6] = &trajectory_point_count;
    arguments[7] = &active_state_count;
    arguments[8] = &active_constant_count;
    arguments[9] = &active_toggle_count;
    arguments[10] = &active_system_count;
    arguments[11] = &steps_per_observation;
    arguments[12] = &mse_output_device;
    sampled_parameters_device=ticket->launch.sampled_parameters_device;
    uniform_pool_device=ticket->launch.uniform_pool_device;
    normal_pool_device=ticket->launch.normal_pool_device;
    rng_pool_size=ticket->launch.rng_pool_size;
    arguments[13]=&sampled_parameters_device;arguments[14]=&uniform_pool_device;
    arguments[15]=&normal_pool_device;arguments[16]=&rng_pool_size;
    allow_missing_observations=ticket->launch.allow_missing_observations;arguments[17]=&allow_missing_observations;
    if(ticket->launch.input_ready_event) {
        cuda_result=cuStreamWaitEvent(stream,ticket->launch.input_ready_event,0);
        if(cuda_result!=CUDA_SUCCESS) {
            o_cuda_error(cuda_result,"cuStreamWaitEvent(input_ready)",error,O_ERROR_BYTES);
            (void)cuModuleUnload(ticket->module);ticket->module=NULL;return ODEZZA_ERROR_CUDA;
        }
    }
    ticket->result.shared_memory_bytes=shared_memory_bytes;
    if(ticket->launch.profile_timing) {
        cuda_result=cuEventRecord(ticket->profile_begin,stream);
        if(cuda_result!=CUDA_SUCCESS) {
            o_cuda_error(cuda_result,"cuEventRecord(profile_begin)",error,O_ERROR_BYTES);
            (void)cuModuleUnload(ticket->module);ticket->module=NULL;return ODEZZA_ERROR_CUDA;
        }
    }
    started = o_monotonic_seconds();
    cuda_result = cuLaunchKernel(
        function,
        grid_x,
        active_system_count,
        1u,
        pipeline->threads_per_block,
        1u,
        1u,
        shared_memory_bytes,
        stream,
        arguments,
        NULL
    );
    if(cuda_result==CUDA_SUCCESS&&ticket->launch.profile_timing)cuda_result=cuEventRecord(ticket->profile_end,stream);
    if (cuda_result == CUDA_SUCCESS) {
        cuda_result = cuEventRecord(ticket->event, stream);
    }
    ticket->result.launch_seconds = o_monotonic_seconds() - started;
    ticket->completion_wait_started = o_monotonic_seconds();
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stats.launch_seconds += ticket->result.launch_seconds;
    pthread_mutex_unlock(&pipeline->mutex);
    if (cuda_result != CUDA_SUCCESS) {
        o_cuda_error(cuda_result, "cuLaunchKernel/cuEventRecord", error, O_ERROR_BYTES);
        (void)cuStreamSynchronize(stream);
        (void)cuModuleUnload(ticket->module);
        ticket->module = NULL;
        return ODEZZA_ERROR_CUDA;
    }

    pthread_mutex_lock(&pipeline->mutex);
    pipeline->inflight_ticket_values[pipeline->inflight_count++] = ticket->index;
    if (pipeline->stats.peak_inflight_module_count < pipeline->inflight_count) {
        pipeline->stats.peak_inflight_module_count = pipeline->inflight_count;
    }
    pthread_mutex_unlock(&pipeline->mutex);
    return ODEZZA_SUCCESS;
}

static int o_retire_completed(OdezzaScoringPipeline *pipeline) {
    uint32_t position;
    uint32_t ticket_index = UINT32_MAX;
    OCuResult cuda_result = CUDA_ERROR_NOT_READY;
    OdezzaResult completion_status = ODEZZA_SUCCESS;
    char error[O_ERROR_BYTES] = {0};
    OdezzaScoringTicket *ticket;
    double started;

    pthread_mutex_lock(&pipeline->mutex);
    for (position = 0u; position < pipeline->inflight_count; ++position) {
        ticket_index = pipeline->inflight_ticket_values[position];
        ticket = &pipeline->tickets[ticket_index];
        cuda_result = cuEventQuery(ticket->event);
        if (cuda_result != CUDA_ERROR_NOT_READY) {
            pipeline->inflight_ticket_values[position] = pipeline->inflight_ticket_values[pipeline->inflight_count - 1u];
            pipeline->inflight_count -= 1u;
            break;
        }
    }
    pthread_mutex_unlock(&pipeline->mutex);
    if (cuda_result == CUDA_ERROR_NOT_READY || ticket_index == UINT32_MAX) {
        return 0;
    }

    ticket = &pipeline->tickets[ticket_index];
    ticket->result.completion_wait_seconds = o_monotonic_seconds() - ticket->completion_wait_started;
    if (cuda_result != CUDA_SUCCESS) {
        o_cuda_error(cuda_result, "cuEventQuery", error, sizeof(error));
        (void)cuStreamSynchronize(pipeline->streams[ticket->result.stream_index]);
        completion_status = ODEZZA_ERROR_CUDA;
    }
    if(cuda_result==CUDA_SUCCESS&&ticket->launch.profile_timing) {
        float begin_ms,end_ms;
        cuda_result=cuEventElapsedTime(&begin_ms,pipeline->profile_origin,ticket->profile_begin);
        if(cuda_result==CUDA_SUCCESS)cuda_result=cuEventElapsedTime(&end_ms,pipeline->profile_origin,ticket->profile_end);
        if(cuda_result==CUDA_SUCCESS) {
            ticket->result.gpu_begin_seconds=begin_ms*.001;
            ticket->result.gpu_end_seconds=end_ms*.001;
            ticket->result.profiled=1;
        } else {
            o_cuda_error(cuda_result,"cuEventElapsedTime(scoring)",error,sizeof(error));
            completion_status=ODEZZA_ERROR_CUDA;
        }
    }
    started = o_monotonic_seconds();
    cuda_result = cuModuleUnload(ticket->module);
    ticket->result.module_unload_seconds = o_monotonic_seconds() - started;
    ticket->module = NULL;
    if (cuda_result != CUDA_SUCCESS && completion_status == ODEZZA_SUCCESS) {
        o_cuda_error(cuda_result, "cuModuleUnload", error, sizeof(error));
        completion_status = ODEZZA_ERROR_CUDA;
    }
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stats.completion_wait_seconds += ticket->result.completion_wait_seconds;
    pipeline->stats.module_unload_seconds += ticket->result.module_unload_seconds;
    pthread_mutex_unlock(&pipeline->mutex);
    o_pipeline_record_completion(pipeline, ticket, completion_status, error);
    return 1;
}

static void o_wait_for_cuda_work(OdezzaScoringPipeline *pipeline) {
    if (pipeline->config.completion_poll_interval_ns == 0u) {
        pthread_mutex_unlock(&pipeline->mutex);
        sched_yield();
        pthread_mutex_lock(&pipeline->mutex);
    } else {
        struct timespec deadline;
        uint64_t nanoseconds;
        (void)clock_gettime(CLOCK_REALTIME, &deadline);
        nanoseconds = (uint64_t)deadline.tv_nsec + pipeline->config.completion_poll_interval_ns;
        deadline.tv_sec += (time_t)(nanoseconds / UINT64_C(1000000000));
        deadline.tv_nsec = (long)(nanoseconds % UINT64_C(1000000000));
        (void)pthread_cond_timedwait(&pipeline->ready_condition, &pipeline->mutex, &deadline);
    }
}

static OdezzaResult o_drive_cuda_until_idle(OdezzaScoringPipeline *pipeline) {
    for (;;) {
        uint32_t slot_index = UINT32_MAX;
        int complete;
        char error[O_ERROR_BYTES] = {0};
        OdezzaResult result;
        OdezzaScoringTicket *ticket;

        while (o_retire_completed(pipeline)) {
        }
        pthread_mutex_lock(&pipeline->mutex);
        if (pipeline->ready_slots.count != 0u && pipeline->inflight_count < pipeline->config.maximum_loaded_modules) {
            (void)o_ring_pop(&pipeline->ready_slots, &slot_index);
        }
        complete = pipeline->active_ticket_count == 0u;
        if (slot_index == UINT32_MAX && !complete) {
            if (pipeline->inflight_count == 0u) {
                (void)pthread_cond_wait(&pipeline->ready_condition, &pipeline->mutex);
            } else {
                o_wait_for_cuda_work(pipeline);
            }
        }
        pthread_mutex_unlock(&pipeline->mutex);
        if (complete) return ODEZZA_SUCCESS;
        if (slot_index == UINT32_MAX) continue;

        ticket = &pipeline->tickets[pipeline->slots[slot_index].ticket_index];
        result = o_load_and_launch(pipeline, slot_index, error);
        if (result != ODEZZA_SUCCESS) {
            o_pipeline_record_completion(pipeline, ticket, result, error);
        }
    }
}

static void *o_specialization_worker_main(void *argument) {
    OScoringWorker *worker = (OScoringWorker *)argument;
    OdezzaScoringPipeline *pipeline = worker->pipeline;
    for (;;) {
        uint32_t ticket_index;
        uint32_t slot_index;
        OScoringSlot *slot;
        OdezzaScoringTicket *ticket;
        OdezzaResult result;
        double started;
        char specialization_error[O_ERROR_BYTES];

        pthread_mutex_lock(&pipeline->mutex);
        while (!pipeline->stopping && (pipeline->pending_tickets.count == 0u || pipeline->free_slots.count == 0u)) {
            if (pipeline->pending_tickets.count != 0u && pipeline->free_slots.count == 0u) {
                pipeline->stats.worker_slot_wait_count += 1u;
            }
            (void)pthread_cond_wait(&pipeline->worker_condition, &pipeline->mutex);
        }
        if (pipeline->stopping && pipeline->pending_tickets.count == 0u) {
            pthread_mutex_unlock(&pipeline->mutex);
            break;
        }
        if (!o_ring_pop(&pipeline->pending_tickets, &ticket_index) || !o_ring_pop(&pipeline->free_slots, &slot_index)) {
            pthread_mutex_unlock(&pipeline->mutex);
            continue;
        }
        pthread_mutex_unlock(&pipeline->mutex);

        ticket = &pipeline->tickets[ticket_index];
        slot = &pipeline->slots[slot_index];
        slot->ticket_index = ticket_index;
        ticket->result.worker_index = worker->index;
        ticket->result.slot_index = slot_index;
        ticket->result.queue_wait_seconds = o_monotonic_seconds() - ticket->submitted_at;

        started = o_monotonic_seconds();
        memcpy(slot->cubin, pipeline->template_cubin, pipeline->cubin_size);
        ticket->result.cubin_reset_seconds = o_monotonic_seconds() - started;
        pthread_mutex_lock(&pipeline->mutex);
        pipeline->stats.cubin_reset_count += 1u;
        pipeline->stats.cubin_reset_byte_count += pipeline->cubin_size;
        pipeline->stats.cubin_reset_seconds += ticket->result.cubin_reset_seconds;
        pthread_mutex_unlock(&pipeline->mutex);

        started = o_monotonic_seconds();
        result = o_specialize_scoring_cubin_systems_fresh(
            slot->cubin,
            pipeline->cubin_size,
            &pipeline->inspection,
            &pipeline->prespecialization,
            ticket->launch.active_toggle_count,
            ticket->systems,
            ticket->system_count,
            slot->workspace,
            pipeline->workspace_size,
            &ticket->result.specialization
        );
        ticket->result.specialization_seconds = o_monotonic_seconds() - started;
        pthread_mutex_lock(&pipeline->mutex);
        pipeline->stats.specialization_seconds += ticket->result.specialization_seconds;
        if (result != ODEZZA_SUCCESS) pipeline->stats.specialization_failure_count += 1u;
        pthread_mutex_unlock(&pipeline->mutex);
        if (result != ODEZZA_SUCCESS) {
            (void)snprintf(
                specialization_error,
                sizeof(specialization_error),
                "scoring CUBIN specialization failed with result %d",
                (int)result
            );
            o_pipeline_release_slot(pipeline, slot_index);
            o_pipeline_record_completion(pipeline, ticket, result, specialization_error);
            continue;
        }

        pthread_mutex_lock(&pipeline->mutex);
        (void)o_ring_push(&pipeline->ready_slots, slot_index);
        if (pipeline->stats.peak_ready_slot_count < pipeline->ready_slots.count) {
            pipeline->stats.peak_ready_slot_count = pipeline->ready_slots.count;
        }
        pthread_cond_signal(&pipeline->ready_condition);
        pthread_mutex_unlock(&pipeline->mutex);
    }
    return NULL;
}

static void o_pipeline_stop_threads(OdezzaScoringPipeline *pipeline) {
    uint32_t index;
    if (pipeline == NULL || !pipeline->mutex_initialized) return;
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->stopping = 1;
    if (pipeline->worker_condition_initialized) {
        pthread_cond_broadcast(&pipeline->worker_condition);
    }
    if (pipeline->ready_condition_initialized) {
        pthread_cond_broadcast(&pipeline->ready_condition);
    }
    pthread_mutex_unlock(&pipeline->mutex);
    for (index = 0u; index < pipeline->started_worker_count; ++index) {
        (void)pthread_join(pipeline->workers[index].thread, NULL);
        pipeline->workers[index].started = 0;
    }
    pipeline->started_worker_count = 0u;
}

static OdezzaResult o_pipeline_release_members(OdezzaScoringPipeline *pipeline) {
    uint32_t index;
    OdezzaResult cuda_result;
    char cuda_error[O_ERROR_BYTES] = {0};
    if (pipeline == NULL) return ODEZZA_SUCCESS;
    cuda_result = o_cuda_resources_destroy(pipeline, cuda_error);
    for (index = 0u; pipeline->tickets != NULL && index < pipeline->initialized_ticket_count; ++index) {
        (void)pthread_cond_destroy(&pipeline->tickets[index].condition);
        (void)pthread_mutex_destroy(&pipeline->tickets[index].mutex);
    }
    if (pipeline->streams != NULL) free(pipeline->streams);
    if (pipeline->inflight_ticket_values != NULL) free(pipeline->inflight_ticket_values);
    if (pipeline->free_ticket_values != NULL) free(pipeline->free_ticket_values);
    if (pipeline->ready_slot_values != NULL) free(pipeline->ready_slot_values);
    if (pipeline->pending_ticket_values != NULL) free(pipeline->pending_ticket_values);
    if (pipeline->free_slot_values != NULL) free(pipeline->free_slot_values);
    if (pipeline->tickets != NULL) free(pipeline->tickets);
    if (pipeline->run_tickets != NULL) free(pipeline->run_tickets);
    free(pipeline->intervals);pipeline->intervals=NULL;
    if (pipeline->slots != NULL) free(pipeline->slots);
    if (pipeline->workers != NULL) free(pipeline->workers);
    if (pipeline->inspection_storage != NULL) free(pipeline->inspection_storage);
    if (pipeline->template_cubin != NULL) free(pipeline->template_cubin);
    if (pipeline->generated_source != NULL) free(pipeline->generated_source);
    pipeline->streams = NULL;
    pipeline->inflight_ticket_values = NULL;
    pipeline->free_ticket_values = NULL;
    pipeline->ready_slot_values = NULL;
    pipeline->pending_ticket_values = NULL;
    pipeline->free_slot_values = NULL;
    pipeline->tickets = NULL;
    pipeline->run_tickets = NULL;
    pipeline->slots = NULL;
    pipeline->workers = NULL;
    pipeline->inspection_storage = NULL;
    pipeline->template_cubin = NULL;
    pipeline->generated_source = NULL;
    pipeline->slot_cubins = NULL;
    pipeline->slot_workspaces = NULL;
    pipeline->initialized_ticket_count = 0u;
    if (pipeline->ready_condition_initialized) {
        (void)pthread_cond_destroy(&pipeline->ready_condition);
        pipeline->ready_condition_initialized = 0;
    }
    if (pipeline->worker_condition_initialized) {
        (void)pthread_cond_destroy(&pipeline->worker_condition);
        pipeline->worker_condition_initialized = 0;
    }
    if (pipeline->mutex_initialized) {
        (void)pthread_mutex_destroy(&pipeline->mutex);
        pipeline->mutex_initialized = 0;
    }
    return cuda_result;
}

static OdezzaResult o_pipeline_validate_create(
    const void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    const OdezzaScoringPipelineConfig *config,
    uint32_t *slot_count_ret
) {
    unsigned char digest[32];
    OSha256 sha;
    uint64_t slots;
    if (cubin == NULL || cubin_size == 0u || inspection == NULL || prespecialization == NULL || config == NULL || slot_count_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (inspection->cubin_size != cubin_size || memcmp(inspection->template_id, prespecialization->template_id, 32u) != 0 || inspection->architecture == 0u ||
        prespecialization->state_count == 0u || prespecialization->state_count > 128u
    ) {
        return ODEZZA_ERROR_FORMAT;
    }
    if (config->specialization_thread_count == 0u || config->slots_per_thread == 0u || config->queue_capacity == 0u) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    slots = (uint64_t)config->specialization_thread_count * config->slots_per_thread;
    if (slots == 0u || slots > UINT32_MAX || slots > config->queue_capacity) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (config->execution_stream_count == 0u || config->maximum_loaded_modules == 0u || config->maximum_loaded_modules < config->execution_stream_count ||
        config->maximum_loaded_modules > config->queue_capacity
    ) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (o_sha256_init(&sha) != ODEZZA_SUCCESS || o_sha256_update(&sha, cubin, cubin_size) != ODEZZA_SUCCESS || o_sha256_final(&sha, digest) != ODEZZA_SUCCESS) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    if (memcmp(digest, prespecialization->cubin_id, 32u) != 0) {
        return ODEZZA_ERROR_FORMAT;
    }
    *slot_count_ret = (uint32_t)slots;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_scoring_pipeline_initialize_members(
    OdezzaScoringPipeline *pipeline,
    const void *prespecialized_cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    const OdezzaScoringPipelineConfig *config
) {
    OdezzaResult result;
    uint32_t slot_count;
    uint32_t index;
    size_t slot_cubin_bytes;
    size_t slot_workspace_bytes;
    size_t aligned;
    double cuda_resource_started;
    char cuda_error[O_ERROR_BYTES] = {0};

    if (pipeline == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    result = o_pipeline_validate_create(
        prespecialized_cubin,
        cubin_size,
        inspection,
        prespecialization,
        config,
        &slot_count
    );
    if (result != ODEZZA_SUCCESS) return result;
    if (!o_multiply_size(slot_count, cubin_size, &slot_cubin_bytes)) {
        return ODEZZA_ERROR_OVERFLOW;
    }

    pipeline->config = *config;
    pipeline->cubin_size = cubin_size;
    pipeline->prespecialization = *prespecialization;
    pipeline->worker_count = config->specialization_thread_count;
    pipeline->slot_count = slot_count;
    pipeline->ticket_count = config->queue_capacity;
    pipeline->stream_count = config->execution_stream_count;
    pipeline->threads_per_block = config->threads_per_block;

    if (pthread_mutex_init(&pipeline->mutex, NULL) != 0) {
        return ODEZZA_ERROR_THREAD;
    }
    pipeline->mutex_initialized = 1;
    if (pthread_cond_init(&pipeline->worker_condition, NULL) != 0) {
        return ODEZZA_ERROR_THREAD;
    }
    pipeline->worker_condition_initialized = 1;
    if (pthread_cond_init(&pipeline->ready_condition, NULL) != 0) {
        return ODEZZA_ERROR_THREAD;
    }
    pipeline->ready_condition_initialized = 1;
    pipeline->template_cubin = (unsigned char *)malloc(cubin_size);
    if (pipeline->template_cubin == NULL) {
        return ODEZZA_ERROR_ALLOCATION;
    }
    memcpy(pipeline->template_cubin, prespecialized_cubin, cubin_size);
    O_RETURN_IF_ERROR(o_copy_inspection(inspection, &pipeline->inspection_storage, &pipeline->inspection));
    O_RETURN_IF_ERROR(odezza_scoring_specialization_workspace_size(&pipeline->inspection, &pipeline->workspace_size));
    if (!o_align_size(cubin_size, &aligned)) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    pipeline->slot_cubin_stride = aligned;
    if (!o_align_size(pipeline->workspace_size, &aligned)) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    pipeline->slot_workspace_stride = aligned;
    if (!o_multiply_size(slot_count, pipeline->slot_workspace_stride, &slot_workspace_bytes)) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    if (!o_multiply_size(slot_count, pipeline->slot_cubin_stride, &slot_cubin_bytes) ||
        !o_add_size(slot_cubin_bytes, slot_workspace_bytes, &pipeline->run_workspace_size)
    ) {
        return ODEZZA_ERROR_OVERFLOW;
    }

    pipeline->workers = (OScoringWorker *)calloc(pipeline->worker_count, sizeof(*pipeline->workers));
    pipeline->slots = (OScoringSlot *)calloc(slot_count, sizeof(*pipeline->slots));
    pipeline->tickets = (OdezzaScoringTicket *)calloc(pipeline->ticket_count, sizeof(*pipeline->tickets));
    pipeline->run_tickets = (OdezzaScoringTicket **)calloc(pipeline->ticket_count, sizeof(*pipeline->run_tickets));
    pipeline->intervals=calloc(pipeline->ticket_count,sizeof(*pipeline->intervals));
    pipeline->free_slot_values = (uint32_t *)malloc(slot_count * sizeof(uint32_t));
    pipeline->ready_slot_values = (uint32_t *)malloc(slot_count * sizeof(uint32_t));
    pipeline->pending_ticket_values = (uint32_t *)malloc(pipeline->ticket_count * sizeof(uint32_t));
    pipeline->free_ticket_values = (uint32_t *)malloc(pipeline->ticket_count * sizeof(uint32_t));
    pipeline->inflight_ticket_values = (uint32_t *)malloc(pipeline->ticket_count * sizeof(uint32_t));
    if (pipeline->stream_count != 0u) {
        pipeline->streams = (OCuStream *)calloc(pipeline->stream_count, sizeof(*pipeline->streams));
    }
    if (pipeline->workers == NULL || pipeline->slots == NULL || pipeline->tickets == NULL || pipeline->run_tickets == NULL || pipeline->intervals == NULL ||
        pipeline->free_slot_values == NULL || pipeline->ready_slot_values == NULL || pipeline->pending_ticket_values == NULL ||
        pipeline->free_ticket_values == NULL || pipeline->inflight_ticket_values == NULL || (pipeline->stream_count != 0u && pipeline->streams == NULL)
    ) {
        return ODEZZA_ERROR_ALLOCATION;
    }

    o_ring_initialize(&pipeline->free_slots, pipeline->free_slot_values, slot_count);
    o_ring_initialize(&pipeline->ready_slots, pipeline->ready_slot_values, slot_count);
    o_ring_initialize(&pipeline->pending_tickets, pipeline->pending_ticket_values, pipeline->ticket_count);
    o_ring_initialize(&pipeline->free_tickets, pipeline->free_ticket_values, pipeline->ticket_count);
    for (index = 0u; index < pipeline->ticket_count; ++index) {
        OdezzaScoringTicket *ticket = &pipeline->tickets[index];
        ticket->pipeline = pipeline;
        ticket->index = index;
        if (pthread_mutex_init(&ticket->mutex, NULL) != 0) {
            return ODEZZA_ERROR_THREAD;
        }
        if (pthread_cond_init(&ticket->condition, NULL) != 0) {
            (void)pthread_mutex_destroy(&ticket->mutex);
            return ODEZZA_ERROR_THREAD;
        }
        pipeline->initialized_ticket_count = index + 1u;
        (void)o_ring_push(&pipeline->free_tickets, index);
    }
    cuda_resource_started = o_monotonic_seconds();
    result = o_cuda_resources_create(pipeline, cuda_error);
    pipeline->stats.cuda_resource_create_seconds = o_monotonic_seconds() - cuda_resource_started;
    if (result != ODEZZA_SUCCESS) {
        o_set_pipeline_error(pipeline, "%s", cuda_error[0] == '\0' ? "CUDA stream/event creation failed" : cuda_error);
        return result;
    }
    for (index = 0u; index < pipeline->worker_count; ++index) {
        pipeline->workers[index].pipeline = pipeline;
        pipeline->workers[index].index = index;
    }

    for (index = 0u; index < pipeline->worker_count; ++index) {
        if (pthread_create(&pipeline->workers[index].thread, NULL, o_specialization_worker_main, &pipeline->workers[index]) != 0) {
            return ODEZZA_ERROR_THREAD;
        }
        pipeline->workers[index].started = 1;
        pipeline->started_worker_count = index + 1u;
    }
    pipeline->creation_complete = 1;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_scoring_pipeline_initialize(
    OdezzaScoringPipeline *pipeline,
    const void *prespecialized_cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    const OdezzaScoringPipelineConfig *config
) {
    OdezzaResult result = o_scoring_pipeline_initialize_members(pipeline, prespecialized_cubin, cubin_size, inspection, prespecialization, config);
    if (result != ODEZZA_SUCCESS) {
        o_pipeline_stop_threads(pipeline);
        (void)o_pipeline_release_members(pipeline);
    }
    return result;
}

static OdezzaResult o_scoring_pipeline_build(
    const OdezzaScoringPipelineCreateInfo *create_info,
    const OdezzaScoringPipelineTuning *tuning,
    OdezzaScoringPipeline *pipeline,
    const OdezzaScoringTemplate *value,
    unsigned char **cubin_ret,
    void **specialization_workspace_ret
) {
    OdezzaResult result;
    OdezzaScoringPipelineConfig config;
    const OdezzaScoringCubinInspection *inspection = NULL;
    OdezzaScoringPrespecialization prespecialization;
    size_t cubin_size = 0u;
    size_t specialization_workspace_size = 0u;
    uint64_t slot_count = 0u;
    uint32_t queue_capacity;
    uint32_t maximum_loaded_modules;
    uint32_t execution_stream_count;
    uint32_t completion_poll_interval_ns;

    if (pipeline == NULL || cubin_ret == NULL || specialization_workspace_ret == NULL
    ) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (create_info == NULL || create_info->state_count == 0u || create_info->state_capacity < create_info->state_count ||
        create_info->state_capacity > 128u || create_info->constant_capacity < create_info->constant_count ||
        create_info->system_capacity == 0u || create_info->shared_patch_capacity == 0u || create_info->system_patch_capacity == 0u ||
        create_info->worker_count == 0u || create_info->cubin_slots_per_worker == 0u ||
        (uint64_t)create_info->worker_count * create_info->cubin_slots_per_worker > UINT32_MAX ||
        (create_info->fixed_rhs_count != 0u && create_info->fixed_rhs == NULL)
    ) {
        o_set_pipeline_error(pipeline, "invalid scoring-pipeline creation description");
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (create_info->sm_version != 89u && create_info->sm_version != 90u && create_info->sm_version != 120u) {
        o_set_pipeline_error(pipeline, "unsupported scoring architecture sm_%u", create_info->sm_version);
        return ODEZZA_ERROR_UNSUPPORTED;
    }
    slot_count = (uint64_t)create_info->worker_count * create_info->cubin_slots_per_worker;
    if (tuning == NULL) {
        queue_capacity = slot_count > 64u ? (uint32_t)slot_count : 64u;
        maximum_loaded_modules = slot_count > 1u ? 2u : 1u;
        execution_stream_count = maximum_loaded_modules;
        completion_poll_interval_ns = 0u;
    } else {
        queue_capacity = tuning->queue_capacity;
        maximum_loaded_modules = tuning->maximum_loaded_modules;
        execution_stream_count = tuning->execution_stream_count;
        completion_poll_interval_ns = tuning->completion_poll_interval_ns;
    }
    if (queue_capacity == 0u || queue_capacity < slot_count || maximum_loaded_modules == 0u || execution_stream_count == 0u ||
        maximum_loaded_modules < execution_stream_count || maximum_loaded_modules > queue_capacity
    ) {
        o_set_pipeline_error(pipeline, "invalid private scoring-pipeline tuning");
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!value || !value->valid || value->info.sm_version != create_info->sm_version ||
        value->info.state_capacity != create_info->state_capacity ||
        value->info.constant_capacity != create_info->constant_capacity ||
        value->info.system_capacity != create_info->system_capacity ||
        value->info.shared_patch_capacity != create_info->shared_patch_capacity ||
        value->info.system_patch_capacity != create_info->system_patch_capacity) {
        o_set_pipeline_error(pipeline, "prepared scoring template is invalid or has a different shape");
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    cubin_size = value->cubin_size;
    inspection = value->inspection;
    *cubin_ret = malloc(cubin_size);
    pipeline->generated_source = malloc(value->source_size + 1u);
    if (!*cubin_ret || !pipeline->generated_source) {
        o_set_pipeline_error(pipeline, "prepared scoring template copy allocation failed");
        return ODEZZA_ERROR_ALLOCATION;
    }
    memcpy(*cubin_ret, value->cubin, cubin_size);
    memcpy(pipeline->generated_source, value->source, value->source_size + 1u);
    pipeline->generated_source_size = value->source_size;
    result = odezza_scoring_specialization_workspace_size(inspection, &specialization_workspace_size);
    if (result != ODEZZA_SUCCESS) {
        o_set_pipeline_error(pipeline, "static specialization workspace measurement failed");
        return result;
    }
    *specialization_workspace_ret = malloc(specialization_workspace_size == 0u ? 1u : specialization_workspace_size);
    if (*specialization_workspace_ret == NULL) {
        result = ODEZZA_ERROR_ALLOCATION;
        o_set_pipeline_error(pipeline, "could not allocate static specialization workspace");
        return result;
    }
    result = odezza_prespecialize_scoring_cubin(
        *cubin_ret,
        cubin_size,
        inspection,
        create_info->state_count,
        create_info->constant_count,
        create_info->fixed_rhs,
        create_info->fixed_rhs_count,
        *specialization_workspace_ret,
        specialization_workspace_size,
        &prespecialization
    );
    if (result != ODEZZA_SUCCESS) {
        o_set_pipeline_error(pipeline, "fixed RHS specialization failed with result %d", (int)result);
        return result;
    }
    memset(&config, 0, sizeof(config));
    config.specialization_thread_count = create_info->worker_count;
    config.slots_per_thread = create_info->cubin_slots_per_worker;
    config.queue_capacity = queue_capacity;
    config.maximum_loaded_modules = maximum_loaded_modules;
    config.execution_stream_count = execution_stream_count;
    config.completion_poll_interval_ns = completion_poll_interval_ns;
    config.threads_per_block = 128u;
    result = o_scoring_pipeline_initialize(pipeline, *cubin_ret, cubin_size, inspection, &prespecialization, &config);
    if (result != ODEZZA_SUCCESS) {
        if (pipeline->error[0] == '\0') {
            o_set_pipeline_error(pipeline, "pipeline resource creation failed with result %d", (int)result);
        }
        return result;
    }
    o_set_pipeline_error(pipeline, NULL);
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_pipeline_create_prepared(
    const OdezzaScoringPipelineCreateInfo *info, const OdezzaScoringPipelineTuning *tuning,
    const OdezzaScoringTemplate *value, int compile, OdezzaScoringPipeline **out
) {
    OdezzaScoringPipeline *pipeline;
    OdezzaScoringTemplate *owned = NULL;
    OdezzaScoringTemplateInfo shape;
    unsigned char *cubin = NULL;
    void *workspace = NULL;
    OdezzaResult result = ODEZZA_SUCCESS;
    if (!out) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out = NULL;
    pipeline = calloc(1u, sizeof(*pipeline));
    if (!pipeline) return ODEZZA_ERROR_ALLOCATION;
    *out = pipeline;
    /* Validate active counts before invoking a potentially expensive compiler. */
    if (!info || !info->state_count || info->state_capacity < info->state_count ||
        info->state_capacity > 128u || info->constant_capacity < info->constant_count ||
        !info->worker_count || !info->cubin_slots_per_worker ||
        (uint64_t)info->worker_count * info->cubin_slots_per_worker > UINT32_MAX ||
        (info->fixed_rhs_count && !info->fixed_rhs)) {
        o_set_pipeline_error(pipeline, "invalid scoring-pipeline creation description");
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (compile) {
        shape.sm_version=info->sm_version; shape.state_capacity=(uint32_t)info->state_capacity;
        shape.constant_capacity=info->constant_capacity; shape.system_capacity=info->system_capacity;
        shape.shared_patch_capacity=info->shared_patch_capacity; shape.system_patch_capacity=info->system_patch_capacity;
        result=odezza_scoring_template_create(&shape,&owned);
        value=owned;
        if (result) o_set_pipeline_error(pipeline,"%s",owned ? owned->error : "template allocation failed");
    }
    if (!result) result=o_scoring_pipeline_build(info,tuning,pipeline,value,&cubin,&workspace);
    free(workspace); free(cubin);
    (void)odezza_scoring_template_destroy(owned);
    return result;
}

OdezzaResult o_scoring_pipeline_create_with_tuning(
    const OdezzaScoringPipelineCreateInfo *info, const OdezzaScoringPipelineTuning *tuning,
    OdezzaScoringPipeline **out
) {
    return o_pipeline_create_prepared(info,tuning,NULL,1,out);
}
OdezzaResult odezza_scoring_pipeline_create(const OdezzaScoringPipelineCreateInfo *info,
    OdezzaScoringPipeline **out
) {
    return o_pipeline_create_prepared(info,NULL,NULL,1,out);
}
OdezzaResult odezza_scoring_pipeline_create_with_template(const OdezzaScoringPipelineCreateInfo *info,
    const OdezzaScoringTemplate *value, OdezzaScoringPipeline **out
) {
    return o_pipeline_create_prepared(info,NULL,value,0,out);
}

static OdezzaResult o_scoring_pipeline_submit(
    OdezzaScoringPipeline *pipeline,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    const OdezzaScoringLaunch *launch,
    OdezzaScoringTicket **ticket_ret
) {
    uint32_t ticket_index;
    OdezzaScoringTicket *ticket;
    if (ticket_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *ticket_ret = NULL;
    if (pipeline == NULL || systems == NULL || system_count == 0u || system_count > pipeline->inspection.system_capacity || launch == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pipeline->mutex);
    if (pipeline->stopping) {
        pthread_mutex_unlock(&pipeline->mutex);
        return ODEZZA_ERROR_BUSY;
    }
    if (!o_ring_pop(&pipeline->free_tickets, &ticket_index)) {
        pthread_mutex_unlock(&pipeline->mutex);
        return ODEZZA_ERROR_QUEUE_FULL;
    }
    ticket = &pipeline->tickets[ticket_index];
    pthread_mutex_lock(&ticket->mutex);
    ticket->systems = systems;
    ticket->system_count = system_count;
    if (launch != NULL)
        ticket->launch = *launch;
    else
        memset(&ticket->launch, 0, sizeof(ticket->launch));
    memset(&ticket->result, 0, sizeof(ticket->result));
    ticket->result.status = ODEZZA_ERROR_BUSY;
    ticket->result.worker_index = UINT32_MAX;
    ticket->result.slot_index = UINT32_MAX;
    ticket->result.stream_index = UINT32_MAX;
    ticket->submitted_at = o_monotonic_seconds();
    ticket->completion_wait_started = 0.0;
    ticket->module = NULL;
    ticket->checked_out = 1u;
    ticket->complete = 0u;
    ticket->error[0] = '\0';
    pthread_mutex_unlock(&ticket->mutex);
    if (!o_ring_push(&pipeline->pending_tickets, ticket_index)) {
        (void)o_ring_push(&pipeline->free_tickets, ticket_index);
        pthread_mutex_unlock(&pipeline->mutex);
        return ODEZZA_ERROR_QUEUE_FULL;
    }
    pipeline->active_ticket_count += 1u;
    pipeline->checked_out_ticket_count += 1u;
    pipeline->stats.submitted_ticket_count += 1u;
    if (pipeline->stats.peak_pending_ticket_count < pipeline->pending_tickets.count) {
        pipeline->stats.peak_pending_ticket_count = pipeline->pending_tickets.count;
    }
    pthread_cond_signal(&pipeline->worker_condition);
    pthread_mutex_unlock(&pipeline->mutex);
    *ticket_ret = ticket;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_scoring_ticket_wait(OdezzaScoringTicket *ticket, OdezzaScoringTicketResult *result_ret) {
    if (ticket == NULL || result_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    pthread_mutex_lock(&ticket->mutex);
    while (!ticket->complete) {
        (void)pthread_cond_wait(&ticket->condition, &ticket->mutex);
    }
    *result_ret = ticket->result;
    pthread_mutex_unlock(&ticket->mutex);
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_scoring_ticket_write_error(
    const OdezzaScoringTicket *ticket,
    char *buffer,
    size_t buffer_size,
    size_t *error_bytes_ret
) {
    OdezzaScoringTicket *mutable_ticket = (OdezzaScoringTicket *)ticket;
    size_t bytes;
    if (ticket == NULL || error_bytes_ret == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    pthread_mutex_lock(&mutable_ticket->mutex);
    bytes = strlen(ticket->error);
    *error_bytes_ret = bytes;
    if (buffer != NULL) {
        if (buffer_size <= bytes) {
            pthread_mutex_unlock(&mutable_ticket->mutex);
            return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
        }
        memcpy(buffer, ticket->error, bytes + 1u);
    }
    pthread_mutex_unlock(&mutable_ticket->mutex);
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_scoring_ticket_destroy(OdezzaScoringTicket *ticket) {
    OdezzaScoringPipeline *pipeline;
    if (ticket == NULL || ticket->pipeline == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    pipeline = ticket->pipeline;
    pthread_mutex_lock(&ticket->mutex);
    if (!ticket->checked_out || !ticket->complete) {
        pthread_mutex_unlock(&ticket->mutex);
        return ODEZZA_ERROR_BUSY;
    }
    ticket->checked_out = 0u;
    ticket->systems = NULL;
    ticket->system_count = 0u;
    pthread_mutex_unlock(&ticket->mutex);
    pthread_mutex_lock(&pipeline->mutex);
    if (pipeline->checked_out_ticket_count != 0u) {
        pipeline->checked_out_ticket_count -= 1u;
    }
    (void)o_ring_push(&pipeline->free_tickets, ticket->index);
    pthread_mutex_unlock(&pipeline->mutex);
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_pipeline_bind_workspace(OdezzaScoringPipeline *pipeline, void *workspace, size_t workspace_size) {
    unsigned char *bytes = (unsigned char *)workspace;
    size_t cubin_bytes;
    uint32_t index;
    if (pipeline == NULL || workspace == NULL || ((uintptr_t)workspace & 7u) != 0u || workspace_size < pipeline->run_workspace_size ||
        !o_multiply_size(pipeline->slot_count, pipeline->slot_cubin_stride, &cubin_bytes)
    ) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pipeline->mutex);
    if (pipeline->run_active || pipeline->active_ticket_count != 0u || pipeline->checked_out_ticket_count != 0u) {
        pthread_mutex_unlock(&pipeline->mutex);
        return ODEZZA_ERROR_BUSY;
    }
    pipeline->slot_cubins = bytes;
    pipeline->slot_workspaces = bytes + cubin_bytes;
    o_ring_initialize(&pipeline->free_slots, pipeline->free_slot_values, pipeline->slot_count);
    o_ring_initialize(&pipeline->ready_slots, pipeline->ready_slot_values, pipeline->slot_count);
    o_ring_initialize(&pipeline->pending_tickets, pipeline->pending_ticket_values, pipeline->ticket_count);
    for (index = 0u; index < pipeline->slot_count; ++index) {
        pipeline->slots[index].cubin = pipeline->slot_cubins + (size_t)index * pipeline->slot_cubin_stride;
        pipeline->slots[index].workspace = pipeline->slot_workspaces + (size_t)index * pipeline->slot_workspace_stride;
        (void)o_ring_push(&pipeline->free_slots, index);
    }
    pipeline->run_active = 1;
    pthread_mutex_unlock(&pipeline->mutex);
    return ODEZZA_SUCCESS;
}

static void o_pipeline_unbind_workspace(OdezzaScoringPipeline *pipeline) {
    uint32_t index;
    pthread_mutex_lock(&pipeline->mutex);
    for (index = 0u; index < pipeline->slot_count; ++index) {
        pipeline->slots[index].cubin = NULL;
        pipeline->slots[index].workspace = NULL;
    }
    pipeline->slot_cubins = NULL;
    pipeline->slot_workspaces = NULL;
    pipeline->free_slots.count = 0u;
    pipeline->ready_slots.count = 0u;
    pipeline->pending_tickets.count = 0u;
    pipeline->run_active = 0;
    pthread_mutex_unlock(&pipeline->mutex);
}

OdezzaResult odezza_scoring_pipeline_workspace_requirements(
    const OdezzaScoringPipeline *pipeline,
    size_t *workspace_size_ret,
    size_t *workspace_alignment_ret
) {
    if (pipeline == NULL || workspace_size_ret == NULL || workspace_alignment_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!pipeline->creation_complete) return ODEZZA_ERROR_BUSY;
    *workspace_size_ret = pipeline->run_workspace_size;
    *workspace_alignment_ret = 8u;
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_scoring_pipeline_run(
    OdezzaScoringPipeline *pipeline,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    const OdezzaScoringLaunch *launch,
    void *workspace,
    size_t workspace_size,
    OdezzaScoringRunReport *report_ret
) {
    OdezzaResult result = ODEZZA_SUCCESS;
    OdezzaResult first_failure = ODEZZA_SUCCESS;
    uint64_t configurations_per_system;
    uint64_t total_configurations;
    uint64_t constant_offset;
    uint64_t output_offset;
    uint64_t constant_elements;
    uint64_t reference_elements;
    uint32_t grid_x;
    uint32_t shared_memory_bytes;
    char launch_error[O_ERROR_BYTES];
    size_t module_count;
    size_t next_module = 0u;
    size_t next_system = 0u;
    size_t submitted;
    size_t index;
    double started,gpu_first=-1,gpu_last=0;

    if (report_ret != NULL) memset(report_ret, 0, sizeof(*report_ret));
    if (pipeline == NULL || !pipeline->creation_complete || systems == NULL || system_count == 0u || launch == NULL ||
        (pipeline->prespecialization.constant_count != 0u && launch->constant_banks_device == 0u && launch->sampled_parameters_device == 0u) || launch->constant_bank_count == 0u ||
        launch->reference_data_device == 0u || launch->trajectory_offsets_device == 0u ||
        launch->trajectory_times_device == 0u || launch->trajectory_count == 0u || launch->trajectory_point_count < launch->trajectory_count ||
        launch->steps_per_observation == 0u || launch->mse_output_device == 0u || launch->active_toggle_count > ODEZZA_MAX_TOGGLE_BITS || launch->allow_missing_observations > 1u || launch->profile_timing > 1u
    ) {
        if (pipeline != NULL) {
            o_set_pipeline_error(pipeline, !pipeline->creation_complete ? "the scoring handle did not complete creation"
                                                                        : "invalid bulk scoring arguments or launch description");
        }
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    result = o_validate_launch(pipeline, launch,
                               system_count < pipeline->inspection.system_capacity ? system_count : pipeline->inspection.system_capacity,
                               &configurations_per_system, &grid_x, &shared_memory_bytes, launch_error);
    if (result != ODEZZA_SUCCESS) {
        o_set_pipeline_error(pipeline, "%s", launch_error);
        return result;
    }
    if ((uint64_t)system_count > UINT64_MAX / configurations_per_system) {
        o_set_pipeline_error(pipeline, "bulk scoring configuration count overflows");
        return ODEZZA_ERROR_OVERFLOW;
    }
    total_configurations = (uint64_t)system_count * configurations_per_system;
    if (system_count > SIZE_MAX / sizeof(*systems) ||
        !o_multiply_u64(system_count, launch->constant_bank_count, &constant_elements) ||
        !o_multiply_u64(constant_elements, pipeline->prespecialization.constant_count, &constant_elements) ||
        !o_multiply_u64(pipeline->prespecialization.state_count, launch->trajectory_point_count, &reference_elements) ||
        !o_device_span_valid(launch->constant_banks_device, launch->sampled_parameters_device?0u:constant_elements, sizeof(float)) ||
        !o_device_span_valid(launch->mse_output_device, total_configurations, sizeof(float)) ||
        !o_device_span_valid(launch->reference_data_device, reference_elements, sizeof(float)) ||
        !o_device_span_valid(launch->trajectory_times_device, launch->trajectory_point_count, sizeof(float)) ||
        !o_device_span_valid(launch->trajectory_offsets_device, (uint64_t)launch->trajectory_count + 1u, sizeof(uint32_t))
    ) {
        o_set_pipeline_error(pipeline, "bulk scoring buffer span overflows or a device pointer is misaligned");
        return ODEZZA_ERROR_OVERFLOW;
    }
    if (launch->sampled_parameters_device) {
        uint64_t descriptors;
        if (!launch->rng_pool_size || launch->sampled_parameters_device%8u ||
            (!launch->uniform_pool_device && !launch->normal_pool_device) ||
            !o_multiply_u64(system_count,pipeline->prespecialization.constant_count,&descriptors) ||
            !o_device_span_valid(launch->sampled_parameters_device,descriptors,sizeof(OdezzaSampledParameter)) ||
            (launch->uniform_pool_device && !o_device_span_valid(launch->uniform_pool_device,launch->rng_pool_size,sizeof(float))) ||
            (launch->normal_pool_device && !o_device_span_valid(launch->normal_pool_device,launch->rng_pool_size,sizeof(float)))) {
            o_set_pipeline_error(pipeline,"invalid sampled-constant descriptor or pool span");return ODEZZA_ERROR_INVALID_ARGUMENT;
        }
    }
    module_count = 1u + (system_count - 1u) / pipeline->inspection.system_capacity;
    result = o_pipeline_bind_workspace(pipeline, workspace, workspace_size);
    if (result != ODEZZA_SUCCESS) {
        o_set_pipeline_error(pipeline, result == ODEZZA_ERROR_BUSY ? "the scoring handle is already running"
                                                                   : "the supplied scoring workspace is null, undersized, or misaligned");
        return result;
    }
    o_set_pipeline_error(pipeline, NULL);
    started = o_monotonic_seconds();
    if(launch->profile_timing) {
        CUresult cr=cuEventRecord(pipeline->profile_origin,pipeline->streams[0]);
        for(index=1;cr==CUDA_SUCCESS&&index<pipeline->stream_count;index++)cr=cuStreamWaitEvent(pipeline->streams[index],pipeline->profile_origin,0);
        if(cr!=CUDA_SUCCESS) {
            o_set_pipeline_error(pipeline,"scoring profiling origin could not be recorded");
            o_pipeline_unbind_workspace(pipeline);return ODEZZA_ERROR_CUDA;
        }
    }
    while (next_module < module_count && first_failure == ODEZZA_SUCCESS) {
        size_t wave_count = module_count - next_module,interval_count=0;
        if (wave_count > pipeline->ticket_count) wave_count = pipeline->ticket_count;
        submitted = 0u;
        for (index = 0u; index < wave_count; ++index) {
            OdezzaScoringLaunch batch_launch = *launch;
            size_t batch_system_count = system_count - next_system;
            uint64_t constant_stride;
            uint64_t output_stride;
            if (batch_system_count > pipeline->inspection.system_capacity) {
                batch_system_count = pipeline->inspection.system_capacity;
            }
            if (!o_multiply_u64(launch->constant_bank_count, pipeline->prespecialization.constant_count, &constant_stride) ||
                !o_multiply_u64(constant_stride, sizeof(float), &constant_stride) ||
                !o_multiply_u64(configurations_per_system, sizeof(float), &output_stride) ||
                !o_multiply_u64((uint64_t)next_system, constant_stride, &constant_offset) ||
                !o_multiply_u64((uint64_t)next_system, output_stride, &output_offset)
            ) {
                first_failure = ODEZZA_ERROR_OVERFLOW;
                o_set_pipeline_error(pipeline, "bulk scoring device-pointer offset overflows");
                break;
            }
            if (UINT64_MAX - launch->constant_banks_device < constant_offset || UINT64_MAX - launch->mse_output_device < output_offset) {
                first_failure = ODEZZA_ERROR_OVERFLOW;
                o_set_pipeline_error(pipeline, "bulk scoring device pointer wraps its address space");
                break;
            }
            if (launch->sampled_parameters_device) {
                batch_launch.sampled_parameters_device += (uint64_t)next_system*pipeline->prespecialization.constant_count*sizeof(OdezzaSampledParameter);
            } else batch_launch.constant_banks_device += constant_offset;
            batch_launch.mse_output_device += output_offset;
            result = o_scoring_pipeline_submit(
                pipeline,
                systems + next_system,
                batch_system_count,
                &batch_launch,
                &pipeline->run_tickets[submitted]
            );
            if (result != ODEZZA_SUCCESS) {
                first_failure = result;
                o_set_pipeline_error(pipeline, "internal scoring submission failed with result %d", (int)result);
                break;
            }
            submitted += 1u;
            next_system += batch_system_count;
            next_module += 1u;
        }
        (void)o_drive_cuda_until_idle(pipeline);
        for (index = 0u; index < submitted; ++index) {
            OdezzaScoringTicketResult ticket_result;
            result = o_scoring_ticket_wait(pipeline->run_tickets[index], &ticket_result);
            if(result==ODEZZA_SUCCESS&&report_ret) {
#define O_SUM(field) report_ret->field+=ticket_result.field
                O_SUM(queue_wait_seconds);O_SUM(cubin_reset_seconds);O_SUM(specialization_seconds);
                O_SUM(module_load_seconds);O_SUM(function_lookup_seconds);O_SUM(launch_seconds);
                O_SUM(completion_wait_seconds);O_SUM(module_unload_seconds);
#undef O_SUM
                if(ticket_result.specialization.register_count>report_ret->maximum_register_count)report_ret->maximum_register_count=ticket_result.specialization.register_count;
                if(ticket_result.shared_memory_bytes>report_ret->maximum_shared_memory_bytes)report_ret->maximum_shared_memory_bytes=ticket_result.shared_memory_bytes;
                if(ticket_result.profiled) {
                    OScoringInterval *v=&pipeline->intervals[interval_count++];
                    v->begin=ticket_result.gpu_begin_seconds;v->end=ticket_result.gpu_end_seconds;
                    report_ret->gpu_sum_seconds+=v->end-v->begin;
                    ++report_ret->profiled_modules;
                }
            }
            if (result == ODEZZA_SUCCESS && ticket_result.status != ODEZZA_SUCCESS) {
                char ticket_error[O_ERROR_BYTES];
                size_t error_size = 0u;
                result = ticket_result.status;
                if (o_scoring_ticket_write_error(pipeline->run_tickets[index], ticket_error, sizeof(ticket_error), &error_size) == ODEZZA_SUCCESS &&
                    error_size != 0u
                ) {
                    o_set_pipeline_error(
                        pipeline,
                        "module %zu failed: %s",
                        next_module - submitted + index,
                        ticket_error
                    );
                } else {
                    o_set_pipeline_error(
                        pipeline,
                        "module %zu failed with result %d",
                        next_module - submitted + index,
                        (int)result
                    );
                }
            }
            if (result != ODEZZA_SUCCESS && first_failure == ODEZZA_SUCCESS) {
                first_failure = result;
            }
            (void)o_scoring_ticket_destroy(pipeline->run_tickets[index]);
            pipeline->run_tickets[index] = NULL;
        }
        if(interval_count) {
            double begin,end;
            /* Small bounded ticket wave; insertion sort cannot call an allocator. */
            for(index=1;index<interval_count;index++) {
                OScoringInterval v=pipeline->intervals[index];size_t at=index;
                while(at&&pipeline->intervals[at-1].begin>v.begin) {
                    pipeline->intervals[at]=pipeline->intervals[at-1];--at;
                }
                pipeline->intervals[at]=v;
            }
            begin=pipeline->intervals[0].begin;end=pipeline->intervals[0].end;
            if(gpu_first<0)gpu_first=begin;
            for(index=1;index<interval_count;index++) {
                OScoringInterval v=pipeline->intervals[index];
                if(v.begin>end) {report_ret->gpu_active_seconds+=end-begin;begin=v.begin;}
                if(v.end>end)end=v.end;
            }
            report_ret->gpu_active_seconds+=end-begin;gpu_last=end;
        }
    }
    if (report_ret != NULL) {
        report_ret->gpu_span_seconds=gpu_first<0?0:gpu_last-gpu_first;
        report_ret->system_count = next_system;
        report_ret->module_count = next_module;
        report_ret->configuration_count = first_failure == ODEZZA_SUCCESS ? total_configurations : (uint64_t)next_system * configurations_per_system;
        report_ret->total_seconds = o_monotonic_seconds() - started;
    }
    o_pipeline_unbind_workspace(pipeline);
    return first_failure;
}

OdezzaResult odezza_scoring_pipeline_write_error(
    const OdezzaScoringPipeline *pipeline,
    char *buffer,
    size_t buffer_size,
    size_t *error_bytes_ret
) {
    size_t bytes;
    if (pipeline == NULL || error_bytes_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    bytes = strlen(pipeline->error);
    *error_bytes_ret = bytes;
    if (buffer != NULL) {
        if (buffer_size <= bytes) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
        memcpy(buffer, pipeline->error, bytes + 1u);
    }
    return ODEZZA_SUCCESS;
}

OdezzaResult odezza_scoring_pipeline_destroy(OdezzaScoringPipeline *pipeline) {
    OdezzaResult result;
    if (pipeline == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (pipeline->mutex_initialized) {
        pthread_mutex_lock(&pipeline->mutex);
        if (pipeline->run_active || pipeline->active_ticket_count != 0u || pipeline->checked_out_ticket_count != 0u) {
            o_set_pipeline_error(pipeline, "the scoring handle cannot be destroyed while a run is active");
            pthread_mutex_unlock(&pipeline->mutex);
            return ODEZZA_ERROR_BUSY;
        }
        pthread_mutex_unlock(&pipeline->mutex);
    }
    o_pipeline_stop_threads(pipeline);
    result = o_pipeline_release_members(pipeline);
    if (pipeline != NULL) free(pipeline);
    return result;
}

OdezzaResult o_scoring_pipeline_generated_source(
    const OdezzaScoringPipeline *pipeline,
    const char **source_ret,
    size_t *source_size_ret
) {
    if (pipeline == NULL || source_ret == NULL || source_size_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!pipeline->creation_complete) return ODEZZA_ERROR_BUSY;
    *source_ret = pipeline->generated_source;
    *source_size_ret = pipeline->generated_source_size;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_scoring_pipeline_cubin_inspection(
    const OdezzaScoringPipeline *pipeline,
    const OdezzaScoringCubinInspection **inspection_ret
) {
    if (pipeline == NULL || inspection_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!pipeline->creation_complete) return ODEZZA_ERROR_BUSY;
    *inspection_ret = &pipeline->inspection;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_scoring_pipeline_prespecialized_cubin(
    const OdezzaScoringPipeline *pipeline,
    const void **cubin_ret,
    size_t *cubin_size_ret
) {
    if (pipeline == NULL || cubin_ret == NULL || cubin_size_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!pipeline->creation_complete) return ODEZZA_ERROR_BUSY;
    *cubin_ret = pipeline->template_cubin;
    *cubin_size_ret = pipeline->cubin_size;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_scoring_pipeline_prespecialization(
    const OdezzaScoringPipeline *pipeline,
    OdezzaScoringPrespecialization *prespecialization_ret
) {
    if (pipeline == NULL || prespecialization_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!pipeline->creation_complete) return ODEZZA_ERROR_BUSY;
    *prespecialization_ret = pipeline->prespecialization;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_scoring_pipeline_stats(const OdezzaScoringPipeline *pipeline, OdezzaScoringPipelineStats *stats_ret) {
    OdezzaScoringPipeline *mutable_pipeline = (OdezzaScoringPipeline *)pipeline;
    if (pipeline == NULL || stats_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (!pipeline->creation_complete) return ODEZZA_ERROR_BUSY;
    pthread_mutex_lock(&mutable_pipeline->mutex);
    *stats_ret = pipeline->stats;
    pthread_mutex_unlock(&mutable_pipeline->mutex);
    return ODEZZA_SUCCESS;
}
