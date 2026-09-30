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

#include "s_runner_internal.h"

#include <stdlib.h>
#include <string.h>

struct SecantCUDARunnerImpl {
    _SecantNvidiaRunner* runner;
    _SecantNvidiaRunnerShape shape;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_input_constants;
    size_t num_runtime_input_columns;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    int dynamic_leaf_warp_owned;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
    _SecantRunnerOptions options;
    unsigned char* compile_scratch;
    size_t compile_scratch_size;
    const SecantAstInstruction** compile_asts;
};

static const SecantAstInstruction _secant_cuda_dummy_static_ast[] = {
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction _secant_cuda_dummy_dynamic_leaf_ast[] = {
    secant_ast_encode_dynamic_constant_or_column_input_f32(0u),
    secant_ast_encode_return_f32
};

static const SecantAstInstruction _secant_cuda_dummy_dynamic_constant_ast[] = {
    secant_ast_encode_dynamic_constant_input_f32(0u),
    secant_ast_encode_return_f32
};

static SecantResult
_secant_cuda_runner_result_map(
    SecantCUDAResult result
) {
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

static SecantResult
_secant_cuda_runner_slot_create(
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
_secant_cuda_runner_slot_release(
    void* backend,
    _SecantNvidiaRunnerSlot* slot
) {
    (void)backend;
    if (slot != NULL && slot->artifact != NULL) {
        (void)secant_cuda_compiled_destroy(
            (SecantCUDACompiled)slot->artifact);
        slot->artifact = NULL;
        slot->binary = NULL;
        slot->binary_size = 0u;
    }
}

static SecantResult
_secant_cuda_runner_compile(
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
    SecantCUDARunner runner = (SecantCUDARunner)backend;
    SecantCUDACompiled compiled = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    size_t log_size = 0u;
    size_t ast_idx;
    SecantCUDAResult cuda_result;

    if (runner == NULL || worker_idx >= runner->runner->num_workers ||
        asts == NULL || slot == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if (slot->num_asts < runner->runner->asts_per_module) {
        const SecantAstInstruction* filler = runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE
            ? _secant_cuda_dummy_dynamic_leaf_ast
            : runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE
                ? _secant_cuda_dummy_dynamic_constant_ast
                : _secant_cuda_dummy_static_ast;
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
                for (constant_idx = 0u; constant_idx < runner->num_input_constants; ++constant_idx) {
                    constants[ast_idx * current_constants_leading_dimension + constant_idx] = 0.0f;
                    scales[ast_idx * current_constants_leading_dimension + constant_idx] = 1.0f;
                }
            }
        }
    }
    if (runner->shape ==
        _SECANT_NVIDIA_RUNNER_SHAPE_MATERIALIZE) {
        cuda_result = secant_cuda_materialize_compile(
            runner->num_kernels,
            runner->asts_per_kernel,
            runner->num_inputs,
            routines,
            num_routines,
            routine_names,
            asts,
            runner->compute_capability_major,
            runner->compute_capability_minor,
            runner->options.values,
            runner->options.count,
            false,
            runner->compile_scratch +
                worker_idx * runner->compile_scratch_size,
            runner->compile_scratch_size,
            NULL,
            0u,
            &log_size,
            &compiled);
    } else if (runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_SSE) {
        cuda_result = secant_cuda_sse_compile(
            runner->num_kernels,
            runner->asts_per_kernel,
            runner->num_inputs,
            runner->num_targets,
            runner->tile_rows,
            runner->threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts,
            runner->compute_capability_major,
            runner->compute_capability_minor,
            runner->options.values,
            runner->options.count,
            false,
            runner->compile_scratch +
                worker_idx * runner->compile_scratch_size,
            runner->compile_scratch_size,
            NULL,
            0u,
            &log_size,
            &compiled);
    } else if (runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE) {
        cuda_result = runner->dynamic_leaf_warp_owned
            ? secant_cuda_warp_dynamic_leaf_sse_compile(
                runner->num_kernels,
                runner->asts_per_kernel,
                runner->num_runtime_input_columns,
                runner->num_inputs,
                runner->num_input_constants,
                runner->num_targets,
                runner->tile_rows,
                runner->threads_per_block,
                routines,
                num_routines,
                routine_names,
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
                &compiled)
            : secant_cuda_dynamic_leaf_sse_compile(
                runner->num_kernels,
                runner->asts_per_kernel,
                runner->num_runtime_input_columns,
                runner->num_inputs,
                runner->num_input_constants,
                runner->num_targets,
                runner->tile_rows,
                runner->threads_per_block,
                routines,
                num_routines,
                routine_names,
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
    } else if (runner->shape == _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE) {
        cuda_result = secant_cuda_packed_constant_optimizer_sse_compile(
            runner->num_kernels,
            runner->asts_per_kernel,
            runner->num_inputs,
            runner->num_input_constants,
            runner->tile_rows,
            runner->threads_per_block,
            routines,
            num_routines,
            routine_names,
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
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    if (cuda_result != SECANT_CUDA_SUCCESS) {
        if (compiled != NULL) {
            (void)secant_cuda_compiled_destroy(compiled);
        }
        return _secant_cuda_runner_result_map(cuda_result);
    }
    cuda_result = secant_cuda_compiled_binary_get(
        compiled,
        &binary,
        &binary_size);
    if (cuda_result != SECANT_CUDA_SUCCESS) {
        (void)secant_cuda_compiled_destroy(compiled);
        return _secant_cuda_runner_result_map(cuda_result);
    }
    slot->artifact = compiled;
    slot->binary = binary;
    slot->binary_size = binary_size;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cuda_runner_create(
    _SecantNvidiaRunnerShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_input_constants,
    size_t num_runtime_input_columns,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    int dynamic_leaf_warp_owned,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantCUDARunner* runner_ret
) {
    SecantCUDARunner runner = NULL;
    size_t scratch_bytes;
    size_t ast_pointer_count;
    SecantResult result;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        (num_inputs == 0u && num_input_constants == 0u) || compute_capability_major == 0u ||
        compile_scratch_size == 0u || num_workers == 0u ||
        num_streams == 0u || runner_ret == NULL ||
        num_workers > SIZE_MAX / compile_scratch_size ||
        num_kernels > SIZE_MAX / asts_per_kernel ||
        num_workers > SIZE_MAX / (num_kernels * asts_per_kernel) ||
        num_workers * num_kernels * asts_per_kernel >
            SIZE_MAX / sizeof(*runner->compile_asts)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *runner_ret = NULL;
    scratch_bytes = num_workers * compile_scratch_size;
    ast_pointer_count = num_workers * num_kernels * asts_per_kernel;
    runner = (SecantCUDARunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    runner->shape = shape;
    runner->num_kernels = num_kernels;
    runner->asts_per_kernel = asts_per_kernel;
    runner->num_inputs = num_inputs;
    runner->num_input_constants = num_input_constants;
    runner->num_runtime_input_columns = num_runtime_input_columns;
    runner->num_targets = num_targets;
    runner->tile_rows = tile_rows;
    runner->threads_per_block = threads_per_block;
    runner->dynamic_leaf_warp_owned = dynamic_leaf_warp_owned;
    runner->compute_capability_major = compute_capability_major;
    runner->compute_capability_minor = compute_capability_minor;
    runner->compile_scratch_size = compile_scratch_size;
    result = _secant_runner_options_copy(
        nvrtc_options,
        num_nvrtc_options,
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
            _secant_cuda_runner_compile,
            _secant_cuda_runner_slot_create,
            _secant_cuda_runner_slot_release,
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
secant_cuda_materialize_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantCUDARunner* runner_ret
) {
    return _secant_cuda_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_MATERIALIZE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        0u,
        0u,
        0u,
        0u,
        0,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_cuda_sse_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantCUDARunner* runner_ret
) {
    return _secant_cuda_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        0u,
        num_targets,
        tile_rows,
        threads_per_block,
        0,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_cuda_dynamic_leaf_sse_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantCUDARunner* runner_ret
) {
    return _secant_cuda_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE,
        num_kernels,
        asts_per_kernel,
        num_static_input_columns,
        num_dynamic_leaves,
        num_input_columns,
        num_targets,
        tile_rows,
        threads_per_block,
        0,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_cuda_warp_dynamic_leaf_sse_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantCUDARunner* runner_ret
) {
    if (threads_per_block % 32u != 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_cuda_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_DYNAMIC_LEAF_SSE,
        num_kernels,
        asts_per_kernel,
        num_static_input_columns,
        num_dynamic_leaves,
        num_input_columns,
        num_targets,
        tile_rows,
        threads_per_block,
        1,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_cuda_packed_constant_optimizer_sse_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantCUDARunner* runner_ret
) {
    return _secant_cuda_runner_create(
        _SECANT_NVIDIA_RUNNER_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_input_constants,
        0u,
        1u,
        tile_rows,
        threads_per_block,
        0,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        compile_scratch_size,
        num_workers,
        num_streams,
        runner_ret);
}

typedef struct _SecantCUDAPackedRunContext {
    SecantCUDARunner runner;
    const SecantCubinPackedConstantOptimizerSSERun* run;
} _SecantCUDAPackedRunContext;

static SecantResult
_secant_cuda_packed_run_get(
    const void* runs,
    size_t run_idx,
    _SecantNvidiaRunData* run_ret
) {
    const _SecantCUDAPackedRunContext* context = (const _SecantCUDAPackedRunContext*)runs;
    const SecantCubinPackedConstantOptimizerSSERun* run;
    size_t output_module_stride;
    size_t result_module_stride;

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
    run_ret->result_device_address = run->best.address;
    run_ret->result_num_elements = run->best.num_elements;
    run_ret->result_leading_dimension = run->best.leading_dimension;
    output_module_stride = context->runner->runner->asts_per_module * run->sse.leading_dimension;
    result_module_stride = context->runner->runner->asts_per_module * run->best.leading_dimension;
    run_ret->output_module_stride = output_module_stride;
    run_ret->result_module_stride = result_module_stride;
    return SECANT_SUCCESS;
}

SecantResult
secant_cuda_materialize_runner_run_all(
    SecantCUDARunner runner,
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
        (num_routines != 0u && routine_names == NULL)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_nvidia_materialize_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        routine_names,
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
secant_cuda_sse_runner_run_all(
    SecantCUDARunner runner,
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
    if (runner == NULL ||
        (num_routines != 0u && routine_names == NULL)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_nvidia_sse_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        routine_names,
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
secant_cuda_dynamic_leaf_sse_runner_run_all(
    SecantCUDARunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
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
    if (runner == NULL || (num_routines != 0u && routine_names == NULL)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_nvidia_dynamic_leaf_sse_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        routine_names,
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
secant_cuda_packed_constant_optimizer_sse_runner_run(
    SecantCUDARunner runner,
    const SecantCubinPackedConstantOptimizerSSERun* run,
    const char* const* routine_names,
    SecantRunnerStats* stats_ret
) {
    _SecantCUDAPackedRunContext context;

    if (runner == NULL || run == NULL ||
        run->header.shape != SECANT_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE_F32 ||
        (run->programs.routines.count != 0u &&
         (run->programs.routines.items == NULL || routine_names == NULL))) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    context.runner = runner;
    context.run = run;
    return _secant_nvidia_runner_run_batch(
        runner->runner,
        run->programs.routines.items,
        run->programs.routines.count,
        routine_names,
        run->programs.asts.items,
        run->programs.asts.count,
        run->programs.current_constants.data,
        run->programs.current_constants.num_elements,
        run->programs.current_constants.leading_dimension,
        &context,
        1u,
        _secant_cuda_packed_run_get,
        stats_ret);
}

SecantResult
secant_cuda_runner_destroy(
    SecantCUDARunner runner
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
