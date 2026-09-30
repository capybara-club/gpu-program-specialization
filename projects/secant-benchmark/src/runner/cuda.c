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

struct SecantCUDARunnerImpl {
    _SecantNvidiaRunner* runner;
    _SecantNvidiaRunnerShape shape;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
    _SecantRunnerOptions options;
    unsigned char* compile_scratch;
    size_t compile_scratch_size;
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
    _SecantNvidiaRunnerSlot* slot
) {
    SecantCUDARunner runner = (SecantCUDARunner)backend;
    SecantCUDACompiled compiled = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    size_t log_size = 0u;
    SecantCUDAResult cuda_result;

    if (runner == NULL || worker_idx >= runner->runner->num_workers ||
        asts == NULL || slot == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
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
    } else {
        cuda_result = secant_cuda_sse_compile(
            runner->num_kernels,
            runner->asts_per_kernel,
            runner->num_inputs,
            runner->num_targets,
            runner->tile_rows,
            runner->threads_per_block,
            SECANT_SSE_REDUCTION_MODE_ATOMIC,
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
    SecantCUDARunner runner = NULL;
    size_t scratch_bytes;
    SecantResult result;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        num_inputs == 0u || compute_capability_major == 0u ||
        compile_scratch_size == 0u || num_workers == 0u ||
        num_streams == 0u || runner_ret == NULL ||
        num_workers > SIZE_MAX / compile_scratch_size) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *runner_ret = NULL;
    scratch_bytes = num_workers * compile_scratch_size;
    runner = (SecantCUDARunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    runner->shape = shape;
    runner->num_kernels = num_kernels;
    runner->asts_per_kernel = asts_per_kernel;
    runner->num_inputs = num_inputs;
    runner->num_targets = num_targets;
    runner->tile_rows = tile_rows;
    runner->threads_per_block = threads_per_block;
    runner->compute_capability_major =
        compute_capability_major;
    runner->compute_capability_minor =
        compute_capability_minor;
    runner->compile_scratch_size = compile_scratch_size;
    result = _secant_runner_options_copy(
        nvrtc_options,
        num_nvrtc_options,
        &runner->options);
    if (result == SECANT_SUCCESS) {
        runner->compile_scratch =
            (unsigned char*)malloc(scratch_bytes);
        if (runner->compile_scratch == NULL) {
            result = SECANT_ERROR_ALLOCATION_FAILED;
        }
    }
    if (result == SECANT_SUCCESS) {
        result = _secant_nvidia_runner_create(
            shape,
            num_kernels,
            asts_per_kernel,
            num_inputs,
            num_targets,
            tile_rows,
            threads_per_block,
            num_workers,
            num_streams,
            runner,
            _secant_cuda_runner_compile,
            _secant_cuda_runner_slot_create,
            _secant_cuda_runner_slot_release,
            &runner->runner);
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
        num_targets,
        tile_rows,
        threads_per_block,
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
    _secant_runner_options_destroy(&runner->options);
    free(runner);
    return result;
}
