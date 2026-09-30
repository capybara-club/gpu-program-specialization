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
#include "secant_cuda_runner.h"
#include "lm_optimizer_runner_internal.h"

#include "s_runner_internal.h"

#include <cuda.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define S_CUDA_LM_RUN_ERROR_RET(ans) do { \
    SecantResult lm_run_result_ = (ans); \
    return lm_run_result_; \
} while (0)

enum {
    S_CUDA_LM_RUN_PARAMETERS = SECANT_CUDA_LM_OPTIMIZER_PARAMETERS,
    S_CUDA_LM_RUN_STATISTICS = SECANT_CUDA_LM_OPTIMIZER_STATISTICS
};

struct SecantCUDALMOptimizerRunnerImpl {
    CUcontext context;
    size_t max_num_asts;
    size_t num_input_columns;
    size_t num_static_input_columns;
    size_t tile_rows;
    size_t threads_per_block;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
    _SecantRunnerOptions options;
    unsigned char* compile_scratch;
    size_t compile_scratch_size;
    CUstream* streams;
    CUevent* done_events;
    size_t num_streams;
    CUevent ready_event;
    CUevent start_event;
    CUevent stop_event;
};

typedef struct SCUDALMRunPreflight {
    size_t evaluated_elements;
    size_t evaluated_bytes;
    unsigned int statistics_grid_x;
    unsigned int statistics_grid_y;
    unsigned int solver_grid_x;
    unsigned int dynamic_shared_bytes;
} SCUDALMRunPreflight;

static double
s_cuda_lm_run_seconds(void) {
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int
s_cuda_lm_checked_add(size_t lhs, size_t rhs, size_t* result_ret) {
    if (result_ret == NULL || rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static int
s_cuda_lm_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (result_ret == NULL || (lhs != 0u && rhs > SIZE_MAX / lhs)) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
s_cuda_lm_span(
    size_t outer_count,
    size_t leading_dimension,
    size_t inner_count,
    size_t* required_ret
) {
    size_t offset;

    return outer_count != 0u && inner_count != 0u &&
        s_cuda_lm_checked_mul(outer_count - 1u, leading_dimension, &offset) &&
        s_cuda_lm_checked_add(offset, inner_count, required_ret);
}

static int
s_cuda_lm_range(
    uintptr_t address,
    size_t num_elements,
    uintptr_t* end_ret
) {
    size_t bytes;

    return address != 0u && end_ret != NULL &&
        s_cuda_lm_checked_mul(num_elements, sizeof(uint32_t), &bytes) &&
        bytes <= UINTPTR_MAX - address &&
        (*end_ret = address + bytes) != 0u;
}

static int
s_cuda_lm_ranges_disjoint(
    const uintptr_t* starts,
    const uintptr_t* ends,
    size_t count
) {
    size_t lhs;

    for (lhs = 0u; lhs < count; ++lhs) {
        size_t rhs;

        for (rhs = lhs + 1u; rhs < count; ++rhs) {
            if (starts[lhs] < ends[rhs] && starts[rhs] < ends[lhs]) {
                return 0;
            }
        }
    }
    return 1;
}

static SecantResult
s_cuda_lm_result_map(SecantCUDAResult result) {
    switch (result) {
        case SECANT_CUDA_SUCCESS:
            return SECANT_SUCCESS;
        case SECANT_CUDA_ERROR_INVALID_VALUE:
            return SECANT_ERROR_INVALID_VALUE;
        case SECANT_CUDA_ERROR_OVERFLOW:
            return SECANT_ERROR_OVERFLOW;
        case SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER:
            return SECANT_ERROR_INSUFFICIENT_BUFFER;
        case SECANT_CUDA_ERROR_FORMAT:
            return SECANT_ERROR_FORMAT;
        case SECANT_CUDA_ERROR_BAD_PROGRAM:
            return SECANT_ERROR_BAD_PROGRAM;
        case SECANT_CUDA_ERROR_STACK_OVERFLOW:
            return SECANT_ERROR_STACK_OVERFLOW;
        case SECANT_CUDA_ERROR_STACK_UNDERFLOW:
            return SECANT_ERROR_STACK_UNDERFLOW;
        case SECANT_CUDA_ERROR_TOO_MANY_ARGS:
            return SECANT_ERROR_TOO_MANY_ARGS;
        case SECANT_CUDA_ERROR_UNSUPPORTED_OP:
            return SECANT_ERROR_UNSUPPORTED_OP;
        case SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS:
            return SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS;
        case SECANT_CUDA_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS:
            return SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS;
        case SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED:
            return SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED;
        case SECANT_CUDA_ERROR_ALLOCATION_FAILED:
            return SECANT_ERROR_ALLOCATION_FAILED;
        case SECANT_CUDA_ERROR_COMPILE_FAILED:
            return SECANT_ERROR_COMPILE_FAILED;
        case SECANT_CUDA_ERROR_INVALID_STATE:
            return SECANT_ERROR_INVALID_STATE;
        case SECANT_CUDA_ERROR_UNSUPPORTED_SHAPE:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
        case SECANT_CUDA_RESULT_NUM_ENUMS:
        default:
            return SECANT_ERROR_INVALID_STATE;
    }
}

static void
s_cuda_lm_stats_write(
    SecantRunnerStats* stats_ret,
    const SecantRunnerStats* stats
) {
    uint32_t struct_size;

    if (stats_ret == NULL || stats == NULL) {
        return;
    }
    struct_size = stats_ret->struct_size;
    *stats_ret = *stats;
    stats_ret->struct_size = struct_size;
}

static void
s_cuda_lm_runner_release(SecantCUDALMOptimizerRunner runner) {
    size_t stream_idx;

    if (runner == NULL) {
        return;
    }
    if (runner->stop_event != NULL) {
        (void)cuEventDestroy(runner->stop_event);
    }
    if (runner->start_event != NULL) {
        (void)cuEventDestroy(runner->start_event);
    }
    if (runner->ready_event != NULL) {
        (void)cuEventDestroy(runner->ready_event);
    }
    if (runner->done_events != NULL) {
        for (stream_idx = 0u; stream_idx < runner->num_streams; ++stream_idx) {
            if (runner->done_events[stream_idx] != NULL) {
                (void)cuEventDestroy(runner->done_events[stream_idx]);
            }
        }
    }
    if (runner->streams != NULL) {
        for (stream_idx = 0u; stream_idx < runner->num_streams; ++stream_idx) {
            if (runner->streams[stream_idx] != NULL) {
                (void)cuStreamDestroy(runner->streams[stream_idx]);
            }
        }
    }
    free(runner->done_events);
    free(runner->streams);
    free(runner->compile_scratch);
    _secant_runner_options_destroy(&runner->options);
    free(runner);
}

SecantResult
secant_cuda_lm_optimizer_runner_create(
    size_t max_num_asts,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_streams,
    SecantCUDALMOptimizerRunner* runner_ret
) {
    SecantCUDALMOptimizerRunner runner = NULL;
    CUcontext context = NULL;
    size_t stream_bytes;
    size_t event_bytes;
    size_t stream_idx;
    SecantResult result;

    if (max_num_asts == 0u || num_input_columns == 0u ||
        num_input_columns > SECANT_AST_MAX_INPUTS ||
        num_static_input_columns != num_input_columns ||
        num_static_input_columns + S_CUDA_LM_RUN_PARAMETERS > SECANT_AST_MAX_INPUTS ||
        tile_rows == 0u || threads_per_block == 0u || threads_per_block > 1024u ||
        compute_capability_major == 0u || compile_scratch_size == 0u ||
        num_streams == 0u || runner_ret == NULL ||
        (num_nvrtc_options != 0u && nvrtc_options == NULL) ||
        !s_cuda_lm_checked_mul(num_streams, sizeof(*runner->streams), &stream_bytes) ||
        !s_cuda_lm_checked_mul(num_streams, sizeof(*runner->done_events), &event_bytes)) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *runner_ret = NULL;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == NULL) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    runner = (SecantCUDALMOptimizerRunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_ALLOCATION_FAILED);
    }
    runner->context = context;
    runner->max_num_asts = max_num_asts;
    runner->num_input_columns = num_input_columns;
    runner->num_static_input_columns = num_static_input_columns;
    runner->tile_rows = tile_rows;
    runner->threads_per_block = threads_per_block;
    runner->compute_capability_major = compute_capability_major;
    runner->compute_capability_minor = compute_capability_minor;
    runner->compile_scratch_size = compile_scratch_size;
    runner->num_streams = num_streams;
    result = _secant_runner_options_copy(
        nvrtc_options,
        num_nvrtc_options,
        &runner->options);
    if (result != SECANT_SUCCESS) {
        s_cuda_lm_runner_release(runner);
        return result;
    }
    runner->compile_scratch = (unsigned char*)malloc(compile_scratch_size);
    runner->streams = (CUstream*)calloc(1u, stream_bytes);
    runner->done_events = (CUevent*)calloc(1u, event_bytes);
    if (runner->compile_scratch == NULL || runner->streams == NULL ||
        runner->done_events == NULL) {
        s_cuda_lm_runner_release(runner);
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_ALLOCATION_FAILED);
    }
    if (cuEventCreate(&runner->ready_event, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuEventCreate(&runner->start_event, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuEventCreate(&runner->stop_event, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
        s_cuda_lm_runner_release(runner);
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    for (stream_idx = 0u; stream_idx < num_streams; ++stream_idx) {
        if (cuStreamCreate(
                runner->streams + stream_idx,
                CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
            cuEventCreate(
                runner->done_events + stream_idx,
                CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
            s_cuda_lm_runner_release(runner);
            S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
        }
    }
    *runner_ret = runner;
    return SECANT_SUCCESS;
}

static SecantResult
s_cuda_lm_run_preflight(
    const SecantCUDALMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    SCUDALMRunPreflight* preflight_ret
) {
    uintptr_t starts[11];
    uintptr_t ends[11];
    size_t range_count = 0u;
    size_t ast_settings;
    size_t ast_statistics;
    size_t required_input;
    size_t required_masks;
    size_t required_words;
    size_t required_constants;
    size_t required_evaluated;
    size_t required_accepted;
    size_t required_damping;
    size_t required_predicted;
    size_t required_status = 0u;
    size_t shared_stride;
    size_t shared_elements;
    size_t shared_bytes;
    size_t grid_x;
    size_t grid_y;
    size_t solver_grid_x;

    if (runner == NULL || run == NULL || preflight_ret == NULL ||
        run->asts.count == 0u || run->asts.count > runner->max_num_asts ||
        run->asts.items == NULL ||
        (run->routines.count != 0u && run->routines.items == NULL) ||
        run->num_input_columns == 0u ||
        run->num_input_columns > runner->num_input_columns ||
        run->num_rows == 0u || run->num_settings == 0u ||
        run->num_settings > UINT_MAX || run->settings_per_cta == 0u ||
        run->settings_per_cta > run->num_settings ||
        run->settings_per_cta > UINT_MAX || run->num_iterations == 0u ||
        run->input.address == 0u || run->target.address == 0u ||
        run->leaf_masks.address == 0u || run->leaf_words.address == 0u ||
        run->current_constants.address == 0u ||
        run->proposal_constants.address == 0u ||
        run->evaluated_statistics.address == 0u ||
        run->accepted_statistics.address == 0u ||
        run->damping.address == 0u || run->predicted_reduction.address == 0u ||
        run->input.leading_dimension < run->num_rows ||
        run->leaf_masks.leading_dimension < run->num_settings ||
        run->leaf_words.leading_dimension < S_CUDA_LM_RUN_PARAMETERS ||
        run->current_constants.leading_dimension < S_CUDA_LM_RUN_PARAMETERS ||
        run->proposal_constants.leading_dimension !=
            run->current_constants.leading_dimension ||
        run->evaluated_statistics.leading_dimension < run->num_settings ||
        run->accepted_statistics.leading_dimension !=
            run->evaluated_statistics.leading_dimension ||
        run->damping.leading_dimension < run->num_settings ||
        run->predicted_reduction.leading_dimension != run->damping.leading_dimension ||
        (run->status.address != 0u &&
         run->status.leading_dimension != run->damping.leading_dimension) ||
        run->max_damping_attempts == 0u ||
        run->max_damping_attempts > (UINT_MAX >> 8u) ||
        !isfinite(run->damping_up) || !(run->damping_up > 1.0f) ||
        !isfinite(run->damping_down) || !(run->damping_down > 0.0f) ||
        run->damping_down > 1.0f ||
        !isfinite(run->minimum_damping) || !(run->minimum_damping > 0.0f) ||
        !isfinite(run->maximum_damping) ||
        run->maximum_damping < run->minimum_damping ||
        !isfinite(run->diagonal_floor) || !(run->diagonal_floor > 0.0f) ||
        !isfinite(run->pivot_floor) || !(run->pivot_floor > 0.0f) ||
        !s_cuda_lm_checked_mul(run->asts.count, run->num_settings, &ast_settings) ||
        !s_cuda_lm_checked_mul(
            run->asts.count, S_CUDA_LM_RUN_STATISTICS, &ast_statistics)) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!s_cuda_lm_span(
            run->num_input_columns,
            run->input.leading_dimension,
            run->num_rows,
            &required_input) ||
        required_input > run->input.num_elements ||
        run->target.num_elements < run->num_rows ||
        !s_cuda_lm_span(
            run->asts.count,
            run->leaf_masks.leading_dimension,
            run->num_settings,
            &required_masks) ||
        required_masks > run->leaf_masks.num_elements ||
        !s_cuda_lm_span(
            ast_settings,
            run->leaf_words.leading_dimension,
            S_CUDA_LM_RUN_PARAMETERS,
            &required_words) ||
        required_words > run->leaf_words.num_elements ||
        !s_cuda_lm_span(
            ast_settings,
            run->current_constants.leading_dimension,
            S_CUDA_LM_RUN_PARAMETERS,
            &required_constants) ||
        required_constants > run->current_constants.num_elements ||
        required_constants > run->proposal_constants.num_elements ||
        !s_cuda_lm_span(
            ast_statistics,
            run->evaluated_statistics.leading_dimension,
            run->num_settings,
            &required_evaluated) ||
        required_evaluated > run->evaluated_statistics.num_elements ||
        !s_cuda_lm_span(
            ast_statistics,
            run->accepted_statistics.leading_dimension,
            run->num_settings,
            &required_accepted) ||
        required_accepted > run->accepted_statistics.num_elements ||
        !s_cuda_lm_span(
            run->asts.count,
            run->damping.leading_dimension,
            run->num_settings,
            &required_damping) ||
        required_damping > run->damping.num_elements ||
        !s_cuda_lm_span(
            run->asts.count,
            run->predicted_reduction.leading_dimension,
            run->num_settings,
            &required_predicted) ||
        required_predicted > run->predicted_reduction.num_elements ||
        (run->status.address != 0u &&
         (!s_cuda_lm_span(
              run->asts.count,
              run->status.leading_dimension,
              run->num_settings,
              &required_status) ||
          required_status > run->status.num_elements))) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
    }

#define S_CUDA_LM_ADD_RANGE(view_, required_) do { \
    starts[range_count] = (view_).address; \
    if (!s_cuda_lm_range(starts[range_count], (required_), ends + range_count)) { \
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_OVERFLOW); \
    } \
    ++range_count; \
} while (0)
    S_CUDA_LM_ADD_RANGE(run->input, required_input);
    S_CUDA_LM_ADD_RANGE(run->target, run->num_rows);
    S_CUDA_LM_ADD_RANGE(run->leaf_masks, required_masks);
    S_CUDA_LM_ADD_RANGE(run->leaf_words, required_words);
    S_CUDA_LM_ADD_RANGE(run->current_constants, required_constants);
    S_CUDA_LM_ADD_RANGE(run->proposal_constants, required_constants);
    S_CUDA_LM_ADD_RANGE(run->evaluated_statistics, required_evaluated);
    S_CUDA_LM_ADD_RANGE(run->accepted_statistics, required_accepted);
    S_CUDA_LM_ADD_RANGE(run->damping, required_damping);
    S_CUDA_LM_ADD_RANGE(run->predicted_reduction, required_predicted);
    if (run->status.address != 0u) {
        S_CUDA_LM_ADD_RANGE(run->status, required_status);
    }
#undef S_CUDA_LM_ADD_RANGE
    if (!s_cuda_lm_ranges_disjoint(starts, ends, range_count)) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }

    grid_x = run->num_rows / runner->tile_rows +
        (run->num_rows % runner->tile_rows != 0u ? 1u : 0u);
    grid_y = run->num_settings / run->settings_per_cta +
        (run->num_settings % run->settings_per_cta != 0u ? 1u : 0u);
    solver_grid_x = ast_settings / runner->threads_per_block +
        (ast_settings % runner->threads_per_block != 0u ? 1u : 0u);
    shared_stride = runner->num_input_columns | 1u;
    if (grid_x == 0u || grid_x > UINT_MAX || grid_y == 0u || grid_y > UINT_MAX ||
        solver_grid_x == 0u || solver_grid_x > UINT_MAX ||
        !s_cuda_lm_checked_add(shared_stride, 1u, &shared_elements) ||
        !s_cuda_lm_checked_mul(shared_elements, runner->tile_rows, &shared_elements) ||
        !s_cuda_lm_checked_mul(shared_elements, sizeof(float), &shared_bytes) ||
        shared_bytes > UINT_MAX ||
        !s_cuda_lm_checked_mul(required_evaluated, sizeof(float),
                               &preflight_ret->evaluated_bytes)) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    preflight_ret->evaluated_elements = required_evaluated;
    preflight_ret->statistics_grid_x = (unsigned int)grid_x;
    preflight_ret->statistics_grid_y = (unsigned int)grid_y;
    preflight_ret->solver_grid_x = (unsigned int)solver_grid_x;
    preflight_ret->dynamic_shared_bytes = (unsigned int)shared_bytes;
    return SECANT_SUCCESS;
}

static SecantResult
s_cuda_lm_functions_load(
    const SecantCUDALMOptimizerRunner runner,
    const void* binary,
    size_t binary_size,
    size_t num_asts,
    unsigned int dynamic_shared_bytes,
    CUmodule* module_ret,
    CUfunction* statistics_functions,
    CUfunction* solver_ret
) {
    CUmodule module = NULL;
    CUdevice device;
    int normal_shared = 0;
    int optin_shared = 0;
    size_t ast_idx;
    if (binary == NULL || binary_size == 0u) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (cuModuleLoadData(&module, binary) != CUDA_SUCCESS ||
        cuModuleGetFunction(solver_ret, module, "secant_lm_solve_f32") != CUDA_SUCCESS ||
        cuCtxGetDevice(&device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &normal_shared,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &optin_shared,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device) != CUDA_SUCCESS) {
        if (module != NULL) {
            (void)cuModuleUnload(module);
        }
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    if (dynamic_shared_bytes > (unsigned int)optin_shared) {
        (void)cuModuleUnload(module);
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    for (ast_idx = 0u; ast_idx < num_asts; ++ast_idx) {
        char name[64];
        int name_size = snprintf(
            name,
            sizeof(name),
            "secant_lm_statistics_%03zu",
            ast_idx);

        if (name_size < 0 || (size_t)name_size >= sizeof(name) ||
            cuModuleGetFunction(statistics_functions + ast_idx, module, name) != CUDA_SUCCESS ||
            (dynamic_shared_bytes > (unsigned int)normal_shared &&
             cuFuncSetAttribute(
                 statistics_functions[ast_idx],
                 CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                 (int)dynamic_shared_bytes) != CUDA_SUCCESS)) {
            (void)cuModuleUnload(module);
            S_CUDA_LM_RUN_ERROR_RET(
                name_size < 0 ? SECANT_ERROR_FORMAT : SECANT_ERROR_DRIVER_FAILED);
        }
    }
    *module_ret = module;
    (void)runner;
    return SECANT_SUCCESS;
}

static SecantResult
s_cuda_lm_statistics_launch(
    const SecantCUDALMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    const SCUDALMRunPreflight* preflight,
    const CUfunction* functions,
    CUdeviceptr constants
) {
    const CUdeviceptr input = (CUdeviceptr)run->input.address;
    const CUdeviceptr target = (CUdeviceptr)run->target.address;
    const size_t num_input_columns = run->num_input_columns;
    const size_t input_leading_dimension = run->input.leading_dimension;
    const size_t leaf_words_leading_dimension = run->leaf_words.leading_dimension;
    const size_t constants_leading_dimension = run->current_constants.leading_dimension;
    const size_t statistics_leading_dimension =
        run->evaluated_statistics.leading_dimension;
    const unsigned int num_settings = (unsigned int)run->num_settings;
    const unsigned int settings_per_cta = (unsigned int)run->settings_per_cta;
    size_t stream_idx;
    size_t ast_idx;

    if (cuMemsetD8Async(
            (CUdeviceptr)run->evaluated_statistics.address,
            0u,
            preflight->evaluated_bytes,
            runner->streams[0]) != CUDA_SUCCESS ||
        cuEventRecord(runner->ready_event, runner->streams[0]) != CUDA_SUCCESS) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    for (stream_idx = 1u; stream_idx < runner->num_streams; ++stream_idx) {
        if (cuStreamWaitEvent(
                runner->streams[stream_idx],
                runner->ready_event,
                0u) != CUDA_SUCCESS) {
            S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
        }
    }
    for (ast_idx = 0u; ast_idx < run->asts.count; ++ast_idx) {
        const size_t setting_row = ast_idx * run->num_settings;
        const size_t statistics_row = ast_idx * S_CUDA_LM_RUN_STATISTICS;
        const CUdeviceptr leaf_masks = (CUdeviceptr)run->leaf_masks.address +
            ast_idx * run->leaf_masks.leading_dimension * sizeof(uint32_t);
        const CUdeviceptr leaf_words = (CUdeviceptr)run->leaf_words.address +
            setting_row * run->leaf_words.leading_dimension * sizeof(uint32_t);
        const CUdeviceptr ast_constants = constants +
            setting_row * constants_leading_dimension * sizeof(float);
        const CUdeviceptr statistics =
            (CUdeviceptr)run->evaluated_statistics.address +
            statistics_row * statistics_leading_dimension * sizeof(float);
        void* arguments[] = {
            (void*)&input,
            (void*)&num_input_columns,
            (void*)&input_leading_dimension,
            (void*)&leaf_masks,
            (void*)&leaf_words,
            (void*)&leaf_words_leading_dimension,
            (void*)&ast_constants,
            (void*)&constants_leading_dimension,
            (void*)&num_settings,
            (void*)&settings_per_cta,
            (void*)&target,
            (void*)&run->num_rows,
            (void*)&statistics,
            (void*)&statistics_leading_dimension
        };

        stream_idx = ast_idx % runner->num_streams;
        if (cuLaunchKernel(
                functions[ast_idx],
                preflight->statistics_grid_x,
                preflight->statistics_grid_y,
                1u,
                (unsigned int)runner->threads_per_block,
                1u,
                1u,
                preflight->dynamic_shared_bytes,
                runner->streams[stream_idx],
                arguments,
                NULL) != CUDA_SUCCESS) {
            S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
        }
    }
    for (stream_idx = 0u; stream_idx < runner->num_streams; ++stream_idx) {
        if (cuEventRecord(
                runner->done_events[stream_idx],
                runner->streams[stream_idx]) != CUDA_SUCCESS) {
            S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
        }
    }
    for (stream_idx = 1u; stream_idx < runner->num_streams; ++stream_idx) {
        if (cuStreamWaitEvent(
                runner->streams[0],
                runner->done_events[stream_idx],
                0u) != CUDA_SUCCESS) {
            S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
s_cuda_lm_solver_launch(
    const SecantCUDALMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    const SCUDALMRunPreflight* preflight,
    CUfunction solver,
    unsigned int initialize,
    unsigned int produce_proposal
) {
    const CUdeviceptr evaluated =
        (CUdeviceptr)run->evaluated_statistics.address;
    const CUdeviceptr accepted =
        (CUdeviceptr)run->accepted_statistics.address;
    const CUdeviceptr current_constants =
        (CUdeviceptr)run->current_constants.address;
    const CUdeviceptr proposal_constants =
        (CUdeviceptr)run->proposal_constants.address;
    const CUdeviceptr damping = (CUdeviceptr)run->damping.address;
    const CUdeviceptr predicted =
        (CUdeviceptr)run->predicted_reduction.address;
    const CUdeviceptr status = (CUdeviceptr)run->status.address;
    const size_t statistics_leading_dimension =
        run->evaluated_statistics.leading_dimension;
    const size_t constants_leading_dimension =
        run->current_constants.leading_dimension;
    const size_t state_leading_dimension = run->damping.leading_dimension;
    const size_t num_asts = run->asts.count;
    void* arguments[] = {
        (void*)&evaluated,
        (void*)&accepted,
        (void*)&statistics_leading_dimension,
        (void*)&current_constants,
        (void*)&proposal_constants,
        (void*)&constants_leading_dimension,
        (void*)&damping,
        (void*)&predicted,
        (void*)&status,
        (void*)&state_leading_dimension,
        (void*)&num_asts,
        (void*)&run->num_settings,
        (void*)&initialize,
        (void*)&produce_proposal,
        (void*)&run->max_damping_attempts,
        (void*)&run->damping_up,
        (void*)&run->damping_down,
        (void*)&run->minimum_damping,
        (void*)&run->maximum_damping,
        (void*)&run->diagonal_floor,
        (void*)&run->pivot_floor
    };

    if (cuLaunchKernel(
            solver,
            preflight->solver_grid_x,
            1u,
            1u,
            (unsigned int)runner->threads_per_block,
            1u,
            1u,
            0u,
            runner->streams[0],
            arguments,
            NULL) != CUDA_SUCCESS) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_DRIVER_FAILED);
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cuda_lm_optimizer_runner_execute_binary(
    SecantCUDALMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    const void* binary,
    size_t binary_size,
    SecantRunnerStats* stats_ret
) {
    SecantRunnerStats stats = secant_runner_stats_init();
    SCUDALMRunPreflight preflight;
    CUmodule module = NULL;
    CUfunction* statistics_functions = NULL;
    CUfunction solver = NULL;
    CUcontext current_context = NULL;
    const double total_begin = s_cuda_lm_run_seconds();
    double begin;
    size_t iteration_idx;
    float runtime_ms = 0.0f;
    SecantResult result;

    memset(&preflight, 0, sizeof(preflight));
    if (runner == NULL || run == NULL || binary == NULL || binary_size == 0u ||
        cuCtxGetCurrent(&current_context) != CUDA_SUCCESS ||
        current_context != runner->context) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    result = s_cuda_lm_run_preflight(runner, run, &preflight);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    stats.num_modules = 1u;
    stats.num_asts = run->asts.count;
    statistics_functions = (CUfunction*)calloc(
        run->asts.count,
        sizeof(*statistics_functions));
    if (statistics_functions == NULL) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_ALLOCATION_FAILED);
    }

    begin = s_cuda_lm_run_seconds();
    result = s_cuda_lm_functions_load(
        runner,
        binary,
        binary_size,
        run->asts.count,
        preflight.dynamic_shared_bytes,
        &module,
        statistics_functions,
        &solver);
    stats.module_load_seconds = s_cuda_lm_run_seconds() - begin;
    if (result != SECANT_SUCCESS) {
        goto finish;
    }
    stats.modules_loaded = 1u;

    result = cuEventRecord(runner->start_event, runner->streams[0]) == CUDA_SUCCESS
        ? SECANT_SUCCESS
        : SECANT_ERROR_DRIVER_FAILED;
    if (result == SECANT_SUCCESS) {
        result = s_cuda_lm_statistics_launch(
            runner,
            run,
            &preflight,
            statistics_functions,
            (CUdeviceptr)run->current_constants.address);
    }
    if (result == SECANT_SUCCESS) {
        result = s_cuda_lm_solver_launch(
            runner,
            run,
            &preflight,
            solver,
            1u,
            1u);
    }
    for (iteration_idx = 0u;
         result == SECANT_SUCCESS && iteration_idx < run->num_iterations;
         ++iteration_idx) {
        result = s_cuda_lm_statistics_launch(
            runner,
            run,
            &preflight,
            statistics_functions,
            (CUdeviceptr)run->proposal_constants.address);
        if (result == SECANT_SUCCESS) {
            result = s_cuda_lm_solver_launch(
                runner,
                run,
                &preflight,
                solver,
                0u,
                iteration_idx + 1u < run->num_iterations ? 1u : 0u);
        }
    }
    if (result == SECANT_SUCCESS &&
        cuEventRecord(runner->stop_event, runner->streams[0]) != CUDA_SUCCESS) {
        result = SECANT_ERROR_DRIVER_FAILED;
    }
    begin = s_cuda_lm_run_seconds();
    if (result == SECANT_SUCCESS) {
        if (cuEventSynchronize(runner->stop_event) != CUDA_SUCCESS ||
            cuEventElapsedTime(
                &runtime_ms,
                runner->start_event,
                runner->stop_event) != CUDA_SUCCESS) {
            result = SECANT_ERROR_DRIVER_FAILED;
        } else {
            stats.runtime_seconds = (double)runtime_ms * 1.0e-3;
        }
    } else {
        (void)cuCtxSynchronize();
    }
    stats.completion_wait_seconds = s_cuda_lm_run_seconds() - begin;

finish:
    if (module != NULL) {
        begin = s_cuda_lm_run_seconds();
        if (cuModuleUnload(module) != CUDA_SUCCESS && result == SECANT_SUCCESS) {
            result = SECANT_ERROR_DRIVER_FAILED;
        }
        stats.module_unload_seconds = s_cuda_lm_run_seconds() - begin;
    }
    free(statistics_functions);
    stats.total_seconds = s_cuda_lm_run_seconds() - total_begin;
    s_cuda_lm_stats_write(stats_ret, &stats);
    return result;
}

SecantResult
secant_cuda_lm_optimizer_runner_run(
    SecantCUDALMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    SecantRunnerStats* stats_ret
) {
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantRunnerStats execution_stats = secant_runner_stats_init();
    SCUDALMRunPreflight preflight;
    SecantCUDACompiled compiled = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    CUcontext current_context = NULL;
    const double total_begin = s_cuda_lm_run_seconds();
    double compile_begin;
    SecantCUDAResult cuda_result;
    SecantResult result;

    memset(&preflight, 0, sizeof(preflight));
    if (runner == NULL || run == NULL ||
        cuCtxGetCurrent(&current_context) != CUDA_SUCCESS ||
        current_context != runner->context) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    result = s_cuda_lm_run_preflight(runner, run, &preflight);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    compile_begin = s_cuda_lm_run_seconds();
    cuda_result = secant_cuda_lm_optimizer_compile(
        run->asts.count,
        runner->num_input_columns,
        runner->num_static_input_columns,
        runner->tile_rows,
        runner->threads_per_block,
        run->routines.items,
        run->routines.count,
        run->asts.items,
        runner->compute_capability_major,
        runner->compute_capability_minor,
        runner->options.values,
        runner->options.count,
        false,
        runner->compile_scratch,
        runner->compile_scratch_size,
        NULL,
        0u,
        NULL,
        &compiled);
    stats.compile_window_seconds = s_cuda_lm_run_seconds() - compile_begin;
    stats.compile_critical_seconds = stats.compile_window_seconds;
    stats.compile_work_seconds = stats.compile_window_seconds;
    if (cuda_result != SECANT_CUDA_SUCCESS) {
        result = s_cuda_lm_result_map(cuda_result);
    } else {
        cuda_result = secant_cuda_compiled_binary_get(
            compiled,
            &binary,
            &binary_size);
        result = cuda_result == SECANT_CUDA_SUCCESS
            ? _secant_cuda_lm_optimizer_runner_execute_binary(
                runner,
                run,
                binary,
                binary_size,
                &execution_stats)
            : s_cuda_lm_result_map(cuda_result);
    }
    if (compiled != NULL) {
        (void)secant_cuda_compiled_destroy(compiled);
    }
    if (execution_stats.num_asts != 0u) {
        const double compile_window_seconds = stats.compile_window_seconds;
        const double compile_critical_seconds = stats.compile_critical_seconds;
        const double compile_work_seconds = stats.compile_work_seconds;

        stats = execution_stats;
        stats.compile_window_seconds = compile_window_seconds;
        stats.compile_critical_seconds = compile_critical_seconds;
        stats.compile_work_seconds = compile_work_seconds;
    } else {
        stats.num_modules = 1u;
        stats.num_asts = run->asts.count;
    }
    stats.total_seconds = s_cuda_lm_run_seconds() - total_begin;
    s_cuda_lm_stats_write(stats_ret, &stats);
    return result;
}

SecantResult
secant_cuda_lm_optimizer_runner_destroy(
    SecantCUDALMOptimizerRunner runner
) {
    CUcontext current_context = NULL;

    if (runner == NULL || cuCtxGetCurrent(&current_context) != CUDA_SUCCESS ||
        current_context != runner->context) {
        S_CUDA_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    s_cuda_lm_runner_release(runner);
    return SECANT_SUCCESS;
}

#undef S_CUDA_LM_RUN_ERROR_RET
