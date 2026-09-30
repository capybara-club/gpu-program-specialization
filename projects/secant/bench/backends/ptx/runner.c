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
#include "secant_ptx_runner.h"

#include "s_runner_internal.h"
#include "../cuda/lm_optimizer_runner_internal.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

struct SecantPTXRunnerImpl {
    _SecantNvidiaRunner* runner;
    _SecantNvidiaRunnerShape shape;
    SecantPTXHandle handle;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
    _SecantRunnerOptions options;
    unsigned char* compile_scratch;
    size_t compile_scratch_size;
    const SecantAstInstruction** compile_asts;
};

struct SecantPTXLMOptimizerRunnerImpl {
    SecantPTXHandle handle;
    SecantCUDALMOptimizerRunner executor;
    size_t num_asts;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
    _SecantRunnerOptions options;
    unsigned char* compile_scratch;
    size_t compile_scratch_size;
};

static double
_secant_ptx_lm_seconds(void) {
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void
_secant_ptx_lm_stats_write(
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

static const SecantAstInstruction _secant_ptx_dummy_static_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction _secant_ptx_dummy_dynamic_leaf_ast[] = {
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction _secant_ptx_dummy_dynamic_constant_ast[] = {
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_return_f32
};

static SecantResult
_secant_ptx_runner_result_map(
    SecantPTXResult result
) {
    switch (result) {
        case SECANT_PTX_SUCCESS:
            return SECANT_SUCCESS;
        case SECANT_PTX_ERROR_INVALID_VALUE:
            return SECANT_ERROR_INVALID_VALUE;
        case SECANT_PTX_ERROR_OVERFLOW:
            return SECANT_ERROR_OVERFLOW;
        case SECANT_PTX_ERROR_INSUFFICIENT_BUFFER:
            return SECANT_ERROR_INSUFFICIENT_BUFFER;
        case SECANT_PTX_ERROR_FORMAT:
            return SECANT_ERROR_FORMAT;
        case SECANT_PTX_ERROR_BAD_PROGRAM:
            return SECANT_ERROR_BAD_PROGRAM;
        case SECANT_PTX_ERROR_STACK_OVERFLOW:
            return SECANT_ERROR_STACK_OVERFLOW;
        case SECANT_PTX_ERROR_STACK_UNDERFLOW:
            return SECANT_ERROR_STACK_UNDERFLOW;
        case SECANT_PTX_ERROR_TOO_MANY_ARGS:
            return SECANT_ERROR_TOO_MANY_ARGS;
        case SECANT_PTX_ERROR_UNSUPPORTED_OP:
            return SECANT_ERROR_UNSUPPORTED_OP;
        case SECANT_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS:
            return SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS;
        case SECANT_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS:
            return SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS;
        case SECANT_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED:
            return SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED;
        case SECANT_PTX_ERROR_ALLOCATION_FAILED:
            return SECANT_ERROR_ALLOCATION_FAILED;
        case SECANT_PTX_ERROR_COMPILE_FAILED:
            return SECANT_ERROR_COMPILE_FAILED;
        case SECANT_PTX_ERROR_INVALID_STATE:
            return SECANT_ERROR_INVALID_STATE;
        case SECANT_PTX_ERROR_UNSUPPORTED_SHAPE:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
        case SECANT_PTX_RESULT_NUM_ENUMS:
        default:
            return SECANT_ERROR_INVALID_STATE;
    }
}

static SecantResult
_secant_ptx_runner_slot_create(
    void* backend,
    size_t slot_idx,
    _SecantNvidiaRunnerSlot* slot
) {
    (void)backend;
    (void)slot_idx;
    if (slot == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return SECANT_SUCCESS;
}

static void
_secant_ptx_runner_slot_release(
    void* backend,
    _SecantNvidiaRunnerSlot* slot
) {
    (void)backend;
    if (slot != NULL && slot->artifact != NULL) {
        (void)secant_ptx_compiled_destroy(
            (SecantPTXCompiled)slot->artifact);
        slot->artifact = NULL;
        slot->binary = NULL;
        slot->binary_size = 0u;
    }
}

static SecantResult
_secant_ptx_runner_compile(
    void* backend,
    size_t worker_idx,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const float* current_constants,
    const float* current_constant_scales,
    size_t current_constants_leading_dimension,
    _SecantNvidiaRunnerSlot* slot
) {
    SecantPTXRunner runner = (SecantPTXRunner)backend;
    SecantPTXCompiled compiled = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    size_t log_size = 0u;
    size_t ast_idx;
    SecantPTXResult ptx_result;

    (void)routine_names;
    if (runner == NULL || worker_idx >= runner->runner->num_workers ||
        asts == NULL || slot == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if (slot->num_asts < runner->runner->asts_per_module) {
        const SecantAstInstruction* filler = runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE
            ? _secant_ptx_dummy_dynamic_leaf_ast
            : runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE
                ? _secant_ptx_dummy_dynamic_constant_ast
                : _secant_ptx_dummy_static_ast;
        const SecantAstInstruction** compile_asts =
            runner->compile_asts + worker_idx * runner->runner->asts_per_module;

        memcpy(compile_asts, asts, slot->num_asts * sizeof(*compile_asts));
        for (ast_idx = slot->num_asts; ast_idx < runner->runner->asts_per_module; ++ast_idx) {
            compile_asts[ast_idx] = filler;
        }
        asts = compile_asts;
        if (runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE) {
            float* constants = (float*)current_constants;
            float* scales = (float*)current_constant_scales;
            size_t constant_idx;

            for (ast_idx = slot->num_asts; ast_idx < runner->runner->asts_per_module; ++ast_idx) {
                for (constant_idx = 0u; constant_idx < runner->runner->num_input_constants; ++constant_idx) {
                    constants[ast_idx * current_constants_leading_dimension + constant_idx] = 0.0f;
                    scales[ast_idx * current_constants_leading_dimension + constant_idx] = 1.0f;
                }
            }
        }
    }
    if (runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE) {
        ptx_result = secant_ptx_packed_constant_optimizer_sse_compile(
            runner->handle,
            routines,
            num_routines,
            asts,
            current_constants,
            current_constant_scales,
            current_constants_leading_dimension,
            runner->compute_capability_major,
            runner->compute_capability_minor,
            runner->options.values,
            runner->options.count,
            false,
            runner->compile_scratch + worker_idx * runner->compile_scratch_size,
            runner->compile_scratch_size,
            NULL,
            0u,
            &log_size,
            &compiled);
    } else {
        ptx_result = secant_ptx_compile(
            runner->handle,
            routines,
            num_routines,
            asts,
            runner->compute_capability_major,
            runner->compute_capability_minor,
            runner->options.values,
            runner->options.count,
            false,
            runner->compile_scratch + worker_idx * runner->compile_scratch_size,
            runner->compile_scratch_size,
            NULL,
            0u,
            &log_size,
            &compiled);
    }
    if (ptx_result != SECANT_PTX_SUCCESS) {
        if (compiled != NULL) {
            (void)secant_ptx_compiled_destroy(compiled);
        }
        return _secant_ptx_runner_result_map(ptx_result);
    }
    ptx_result = secant_ptx_compiled_binary_get(
        compiled,
        &binary,
        &binary_size);
    if (ptx_result != SECANT_PTX_SUCCESS) {
        (void)secant_ptx_compiled_destroy(compiled);
        return _secant_ptx_runner_result_map(ptx_result);
    }
    slot->artifact = compiled;
    slot->binary = binary;
    slot->binary_size = binary_size;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_ptx_runner_create(
    _SecantNvidiaRunnerShape shape,
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
) {
    SecantPTXRunner runner = NULL;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_input_constants = 0u;
    size_t num_runtime_input_columns = 0u;
    size_t num_targets = 0u;
    size_t tile_rows = 0u;
    size_t threads_per_block = 0u;
    size_t scratch_bytes;
    size_t ast_pointer_count;
    SecantPTXResult ptx_result;
    SecantResult result;

    if (handle == NULL || compute_capability_major == 0u ||
        compile_scratch_size == 0u || num_workers == 0u ||
        num_streams == 0u || runner_ret == NULL ||
        num_workers > SIZE_MAX / compile_scratch_size) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *runner_ret = NULL;
    if (shape == _SECANT_NVIDIA_RUNNER_SHAPE_MATERIALIZE) {
        ptx_result = secant_ptx_materialize_info_get(
            handle,
            &num_kernels,
            &asts_per_kernel,
            &num_inputs);
    } else if (shape == _SECANT_NVIDIA_RUNNER_SHAPE_SSE) {
        ptx_result = secant_ptx_sse_info_get(
            handle,
            &num_kernels,
            &asts_per_kernel,
            &num_inputs,
            &num_targets,
            &tile_rows,
            &threads_per_block);
    } else if (shape == _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE) {
        ptx_result = secant_ptx_dynamic_leaf_sse_info_get(
            handle,
            &num_kernels,
            &asts_per_kernel,
            &num_runtime_input_columns,
            &num_inputs,
            &num_input_constants,
            &num_targets,
            &tile_rows,
            &threads_per_block);
    } else if (shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE) {
        ptx_result = secant_ptx_packed_constant_optimizer_sse_info_get(
            handle,
            &num_kernels,
            &asts_per_kernel,
            &num_inputs,
            &num_input_constants,
            &tile_rows,
            &threads_per_block);
        num_targets = 1u;
    } else {
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    if (ptx_result != SECANT_PTX_SUCCESS) {
        return _secant_ptx_runner_result_map(ptx_result);
    }
    if (num_kernels > SIZE_MAX / asts_per_kernel ||
        num_workers > SIZE_MAX / (num_kernels * asts_per_kernel) ||
        num_workers * num_kernels * asts_per_kernel > SIZE_MAX / sizeof(*runner->compile_asts)) {
        return SECANT_ERROR_OVERFLOW;
    }
    scratch_bytes = num_workers * compile_scratch_size;
    ast_pointer_count = num_workers * num_kernels * asts_per_kernel;
    runner = (SecantPTXRunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    runner->shape = shape;
    runner->handle = handle;
    runner->compute_capability_major = compute_capability_major;
    runner->compute_capability_minor = compute_capability_minor;
    runner->compile_scratch_size = compile_scratch_size;
    result = _secant_runner_options_copy(
        nvptx_options,
        num_nvptx_options,
        &runner->options);
    if (result == SECANT_SUCCESS) {
        runner->compile_scratch = (unsigned char*)malloc(scratch_bytes);
        runner->compile_asts = (const SecantAstInstruction**)malloc(
            ast_pointer_count * sizeof(*runner->compile_asts));
        if (runner->compile_scratch == NULL || runner->compile_asts == NULL) {
            result = SECANT_ERROR_ALLOCATION_FAILED;
        }
    }
    if (result == SECANT_SUCCESS) {
        result = _secant_nvidia_runner_create(
            shape,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_input_constants,
            num_runtime_input_columns,
            num_targets,
            tile_rows,
            threads_per_block,
            num_workers,
            num_streams,
            shape == _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE ||
                shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE,
            runner,
            _secant_ptx_runner_compile,
            _secant_ptx_runner_slot_create,
            _secant_ptx_runner_slot_release,
            &runner->runner);
        if (result == SECANT_SUCCESS && shape == _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE) {
            runner->runner->dynamic_leaf_setting_partitioned = 1;
        }
    }
    if (result != SECANT_SUCCESS) {
        free(runner->compile_scratch);
        free(runner->compile_asts);
        _secant_runner_options_destroy(&runner->options);
        free(runner);
        return result;
    }
    *runner_ret = runner;
    return SECANT_SUCCESS;
}

SecantResult
secant_ptx_materialize_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
) {
    return _secant_ptx_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_MATERIALIZE,
        handle,
        compute_capability_major,
        compute_capability_minor,
        nvptx_options,
        num_nvptx_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_ptx_sse_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
) {
    return _secant_ptx_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_SSE,
        handle,
        compute_capability_major,
        compute_capability_minor,
        nvptx_options,
        num_nvptx_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_ptx_dynamic_leaf_sse_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
) {
    return _secant_ptx_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE,
        handle,
        compute_capability_major,
        compute_capability_minor,
        nvptx_options,
        num_nvptx_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_ptx_packed_constant_optimizer_sse_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
) {
    return _secant_ptx_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE,
        handle,
        compute_capability_major,
        compute_capability_minor,
        nvptx_options,
        num_nvptx_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

typedef struct _SecantPTXPackedRunContext {
    SecantPTXRunner runner;
    const SecantCubinPackedConstantOptimizerSSERun* run;
} _SecantPTXPackedRunContext;

static SecantResult
_secant_ptx_packed_run_get(
    const void* runs,
    size_t run_idx,
    _SecantNvidiaRunData* run_ret
) {
    const _SecantPTXPackedRunContext* context = (const _SecantPTXPackedRunContext*)runs;
    const SecantCubinPackedConstantOptimizerSSERun* run;

    if (context == NULL || context->runner == NULL || context->run == NULL ||
        run_ret == NULL || run_idx != 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    run = context->run;
    if (context->runner->runner->asts_per_module > SIZE_MAX / run->sse.leading_dimension ||
        context->runner->runner->asts_per_module > SIZE_MAX / run->best.leading_dimension) {
        return SECANT_ERROR_OVERFLOW;
    }
    memset(run_ret, 0, sizeof(*run_ret));
    run_ret->input_device_address = run->input.address;
    run_ret->input_num_elements = run->input.num_elements;
    run_ret->input_leading_dimension = run->input.leading_dimension;
    run_ret->num_settings = run->num_settings;
    run_ret->num_iterations = run->num_iterations;
    run_ret->seed = run->seed;
    run_ret->generation = run->generation;
    run_ret->iteration = run->iteration;
    run_ret->constant_optimizer_update_mode = run->update_mode;
    run_ret->num_elites = run->num_elites;
    run_ret->current_constant_scales = run->programs.current_constant_scales.data;
    run_ret->current_constant_scales_num_elements = run->programs.current_constant_scales.num_elements;
    run_ret->current_constant_scales_leading_dimension =
        run->programs.current_constant_scales.leading_dimension;
    run_ret->current_constant_velocities = run->current_constant_velocities.data;
    run_ret->current_constant_velocities_num_elements = run->current_constant_velocities.num_elements;
    run_ret->current_constant_velocities_leading_dimension =
        run->current_constant_velocities.leading_dimension;
    run_ret->current_sse = run->current_sse.data;
    run_ret->current_sse_num_elements = run->current_sse.num_elements;
    run_ret->momentum = run->momentum;
    run_ret->scale_learning_rate = run->scale_learning_rate;
    run_ret->scale_failure_decay = run->scale_failure_decay;
    run_ret->minimum_scale = run->minimum_scale;
    run_ret->maximum_scale = run->maximum_scale;
    run_ret->runtime_num_targets = 1u;
    run_ret->targets_device_address = run->target.address;
    run_ret->targets_num_elements = run->target.num_elements;
    run_ret->targets_leading_dimension = run->target.leading_dimension;
    run_ret->num_rows = run->num_rows;
    run_ret->output_device_address = run->sse.address;
    run_ret->output_num_elements = run->sse.num_elements;
    run_ret->output_leading_dimension = run->sse.leading_dimension;
    run_ret->output_module_stride =
        context->runner->runner->asts_per_module * run->sse.leading_dimension;
    run_ret->result_device_address = run->best.address;
    run_ret->result_num_elements = run->best.num_elements;
    run_ret->result_leading_dimension = run->best.leading_dimension;
    run_ret->result_module_stride =
        context->runner->runner->asts_per_module * run->best.leading_dimension;
    return SECANT_SUCCESS;
}

SecantResult
secant_ptx_materialize_runner_run_all(
    SecantPTXRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
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
    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_nvidia_materialize_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        NULL,
        asts,
        num_asts,
        input_device_address,
        input_num_elements,
        input_leading_dimension,
        num_rows,
        output_device_address,
        output_num_elements,
        output_leading_dimension,
        output_module_stride,
        stats_ret);
}

SecantResult
secant_ptx_sse_runner_run_all(
    SecantPTXRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
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
    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_nvidia_sse_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        NULL,
        asts,
        num_asts,
        input_device_address,
        input_num_elements,
        input_leading_dimension,
        runner->runner->num_targets,
        targets_device_address,
        targets_num_elements,
        targets_leading_dimension,
        num_rows,
        output_device_address,
        output_num_elements,
        output_leading_dimension,
        stats_ret);
}

SecantResult
secant_ptx_dynamic_leaf_sse_runner_run_all(
    SecantPTXRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_device_address,
    size_t input_num_elements,
    size_t num_input_columns,
    size_t input_leading_dimension,
    uintptr_t leaf_masks_device_address,
    size_t leaf_masks_num_elements,
    uintptr_t leaf_words_device_address,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t settings_per_cta,
    size_t num_targets,
    uintptr_t targets_device_address,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    uintptr_t output_device_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    SecantRunnerStats* stats_ret
) {
    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_nvidia_dynamic_leaf_sse_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        NULL,
        asts,
        num_asts,
        input_device_address,
        input_num_elements,
        num_input_columns,
        input_leading_dimension,
        leaf_masks_device_address,
        leaf_masks_num_elements,
        leaf_words_device_address,
        leaf_words_num_elements,
        leaf_words_leading_dimension,
        num_settings,
        settings_per_cta,
        num_targets,
        targets_device_address,
        targets_num_elements,
        targets_leading_dimension,
        num_rows,
        output_device_address,
        output_num_elements,
        output_leading_dimension,
        stats_ret);
}

SecantResult
secant_ptx_packed_constant_optimizer_sse_runner_run(
    SecantPTXRunner runner,
    const SecantCubinPackedConstantOptimizerSSERun* run,
    SecantRunnerStats* stats_ret
) {
    _SecantPTXPackedRunContext context;

    if (runner == NULL || run == NULL ||
        run->header.shape != SECANT_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE_F32) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    context.runner = runner;
    context.run = run;
    return _secant_nvidia_runner_run_batch(
        runner->runner,
        run->programs.routines.items,
        run->programs.routines.count,
        NULL,
        run->programs.asts.items,
        run->programs.asts.count,
        run->programs.current_constants.data,
        run->programs.current_constants.num_elements,
        run->programs.current_constants.leading_dimension,
        &context,
        1u,
        _secant_ptx_packed_run_get,
        stats_ret);
}

SecantResult
secant_ptx_lm_optimizer_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_streams,
    SecantPTXLMOptimizerRunner* runner_ret
) {
    SecantPTXLMOptimizerRunner runner = NULL;
    size_t num_kernels;
    size_t num_input_columns;
    size_t num_static_input_columns;
    size_t tile_rows;
    size_t threads_per_block;
    SecantPTXResult ptx_result;
    SecantResult result;

    if (handle == NULL || compute_capability_major == 0u ||
        compile_scratch_size == 0u || num_streams == 0u || runner_ret == NULL ||
        (num_nvptx_options != 0u && nvptx_options == NULL)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *runner_ret = NULL;
    ptx_result = secant_ptx_lm_optimizer_info_get(
        handle,
        &num_kernels,
        &num_input_columns,
        &num_static_input_columns,
        &tile_rows,
        &threads_per_block);
    if (ptx_result != SECANT_PTX_SUCCESS) {
        return _secant_ptx_runner_result_map(ptx_result);
    }
    runner = (SecantPTXLMOptimizerRunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    runner->handle = handle;
    runner->num_asts = num_kernels;
    runner->compute_capability_major = compute_capability_major;
    runner->compute_capability_minor = compute_capability_minor;
    runner->compile_scratch_size = compile_scratch_size;
    result = _secant_runner_options_copy(
        nvptx_options,
        num_nvptx_options,
        &runner->options);
    if (result == SECANT_SUCCESS) {
        runner->compile_scratch = (unsigned char*)malloc(compile_scratch_size);
        if (runner->compile_scratch == NULL) {
            result = SECANT_ERROR_ALLOCATION_FAILED;
        }
    }
    if (result == SECANT_SUCCESS) {
        result = secant_cuda_lm_optimizer_runner_create(
            num_kernels,
            num_input_columns,
            num_static_input_columns,
            tile_rows,
            threads_per_block,
            compute_capability_major,
            compute_capability_minor,
            NULL,
            0u,
            1u,
            num_streams,
            &runner->executor);
    }
    if (result != SECANT_SUCCESS) {
        free(runner->compile_scratch);
        _secant_runner_options_destroy(&runner->options);
        free(runner);
        return result;
    }
    *runner_ret = runner;
    return SECANT_SUCCESS;
}

SecantResult
secant_ptx_lm_optimizer_runner_run(
    SecantPTXLMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    SecantRunnerStats* stats_ret
) {
    SecantRunnerStats stats = secant_runner_stats_init();
    SecantRunnerStats execution_stats = secant_runner_stats_init();
    SecantPTXCompiled compiled = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    const double total_begin = _secant_ptx_lm_seconds();
    double compile_begin;
    SecantPTXResult ptx_result;
    SecantResult result;

    if (runner == NULL || run == NULL || run->asts.items == NULL ||
        run->asts.count != runner->num_asts ||
        (run->routines.count != 0u && run->routines.items == NULL)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    compile_begin = _secant_ptx_lm_seconds();
    ptx_result = secant_ptx_compile(
        runner->handle,
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
    stats.compile_window_seconds = _secant_ptx_lm_seconds() - compile_begin;
    stats.compile_critical_seconds = stats.compile_window_seconds;
    stats.compile_work_seconds = stats.compile_window_seconds;
    if (ptx_result != SECANT_PTX_SUCCESS) {
        result = _secant_ptx_runner_result_map(ptx_result);
    } else {
        ptx_result = secant_ptx_compiled_binary_get(
            compiled,
            &binary,
            &binary_size);
        result = ptx_result == SECANT_PTX_SUCCESS
            ? _secant_cuda_lm_optimizer_runner_execute_binary(
                runner->executor,
                run,
                binary,
                binary_size,
                &execution_stats)
            : _secant_ptx_runner_result_map(ptx_result);
    }
    if (compiled != NULL) {
        (void)secant_ptx_compiled_destroy(compiled);
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
    stats.total_seconds = _secant_ptx_lm_seconds() - total_begin;
    _secant_ptx_lm_stats_write(stats_ret, &stats);
    return result;
}

SecantResult
secant_ptx_lm_optimizer_runner_destroy(
    SecantPTXLMOptimizerRunner runner
) {
    SecantResult result;

    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    result = secant_cuda_lm_optimizer_runner_destroy(runner->executor);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    free(runner->compile_scratch);
    _secant_runner_options_destroy(&runner->options);
    free(runner);
    return SECANT_SUCCESS;
}

SecantResult
secant_ptx_runner_destroy(
    SecantPTXRunner runner
) {
    SecantResult result;
    int destroyed = 0;

    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    result = _secant_nvidia_runner_destroy(
        runner->runner,
        &destroyed);
    if (!destroyed) {
        return result;
    }
    free(runner->compile_scratch);
    free(runner->compile_asts);
    _secant_runner_options_destroy(&runner->options);
    free(runner);
    return result;
}
