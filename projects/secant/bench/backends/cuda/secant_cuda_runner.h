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
#ifndef SECANT_CUDA_RUNNER_H_INCLUDED
#define SECANT_CUDA_RUNNER_H_INCLUDED

#include "secant_cuda.h"
#include "secant.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_CUDA_RUNNER_DEC extern "C"
#else
#define SECANT_CUDA_RUNNER_DEC extern
#endif

/** Comparison-only bulk runner for the CUDA C++ compiler baseline. */
typedef struct SecantCUDARunnerImpl* SecantCUDARunner;

/** Persistent CUDA resources for the fixed-schedule LM optimizer. */
typedef struct SecantCUDALMOptimizerRunnerImpl* SecantCUDALMOptimizerRunner;

/**
 * One blocking fixed-schedule LM optimization request.
 *
 * There is one generated statistics function per AST. `leaf_masks` is
 * `[ast][setting]`, while `leaf_words`, `current_constants`, and
 * `proposal_constants` are `[ast * setting][8]`. A dynamic-constant AST input
 * with index `i` always reads constants column `i`. A dynamic
 * constant-or-column input with index `i` reads the same constant when mask
 * bit `i` is zero, otherwise leaf word `i` is the runtime input-column index.
 * Column bindings contribute zero derivative for that parameter.
 *
 * `evaluated_statistics` and `accepted_statistics` are
 * `[ast * 45][setting]`; every vector is SSE, eight J^T r entries, then the
 * packed upper triangle of J^T J. Damping, predicted reduction, and optional
 * status are `[ast][setting]`. Status bit 0 means the last evaluated proposal
 * was accepted, bit 1 means a next proposal was produced, and bit 2 means no
 * numerically valid proposal could be produced. Bits 8 and above contain the
 * number of damping attempts.
 *
 * The runner evaluates the initial constants once, then evaluates exactly
 * `num_iterations` proposals. Each proposal is followed by one solve launch.
 * The final solve accepts or rejects the final proposal but does not create an
 * unevaluated next proposal. All AST functions and the solver remain in one
 * loaded module for the complete sequence.
 * `num_input_columns` is the active runtime prefix and may not exceed the
 * runner capacity; generated shared tiles retain the full configured width.
 */
typedef struct SecantCUDALMOptimizerRun {
    SecantAstProgramList routines;
    SecantAstProgramList asts;
    SecantDeviceMatrixF32 input;
    size_t num_input_columns;
    SecantDeviceMatrixU32 leaf_masks;
    SecantDeviceMatrixU32 leaf_words;
    SecantDeviceMatrixF32 target;
    size_t num_rows;
    size_t num_settings;
    size_t settings_per_cta;
    size_t num_iterations;
    SecantDeviceMatrixF32 current_constants;
    SecantDeviceMatrixF32 proposal_constants;
    SecantDeviceMatrixF32 evaluated_statistics;
    SecantDeviceMatrixF32 accepted_statistics;
    SecantDeviceMatrixF32 damping;
    SecantDeviceMatrixF32 predicted_reduction;
    SecantDeviceMatrixU32 status;
    uint32_t max_damping_attempts;
    float damping_up;
    float damping_down;
    float minimum_damping;
    float maximum_damping;
    float diagonal_floor;
    float pivot_floor;
} SecantCUDALMOptimizerRun;

/** Returns an LM request with conservative damping defaults. */
static inline SecantCUDALMOptimizerRun secant_cuda_lm_optimizer_run_init(void) {
#ifdef __cplusplus
    SecantCUDALMOptimizerRun value = {};
#else
    SecantCUDALMOptimizerRun value = {0};
#endif
    value.max_damping_attempts = 8u;
    value.damping_up = 10.0f;
    value.damping_down = 0.33333334f;
    value.minimum_damping = 1.0e-7f;
    value.maximum_damping = 1.0e7f;
    value.diagonal_floor = 1.0e-6f;
    value.pivot_floor = 1.0e-12f;
    return value;
}

/**
 * Creates a native CUDA materialize runner.
 *
 * NVRTC options are copied during creation. compile_scratch_size bytes are
 * allocated for every worker. The caller's current CUDA context is used
 * directly and must remain current for create, run, and destroy. CUDA eager
 * module loading is required.
 */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_materialize_runner_create(
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
);

/** Creates a native CUDA SSE runner. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_sse_runner_create(
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
);

/** Creates a native CUDA mixed static/dynamic-leaf SSE runner. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_dynamic_leaf_sse_runner_create(
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
);

/** Creates a native CUDA warp-owned mixed dynamic-leaf SSE runner. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_warp_dynamic_leaf_sse_runner_create(
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
);

/** Creates a native CUDA packed shared-jitter constant-optimizer runner. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_packed_constant_optimizer_sse_runner_create(
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
);

/**
 * Creates the fixed-eight thread-owned LM runner.
 *
 * `max_num_asts` is the maximum number of one-AST functions compiled into one
 * module. `tile_rows` controls the shared row tile independently of
 * `threads_per_block`. The caller's current CUDA context must remain current
 * for create, run, and destroy.
 */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_lm_optimizer_runner_create(
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
);

/**
 * Compiles, loads, and runs complete materialize modules in parallel.
 *
 * output_module_stride is measured in float elements. A zero stride reuses
 * one module-sized output region for every module. A nonzero stride must be at
 * least one complete module span. A runner does not support concurrent run or
 * destroy calls.
 */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_materialize_runner_run_all(
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
);

/** Compiles, loads, and runs complete SSE modules in parallel. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_sse_runner_run_all(
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
);

/**
 * Compiles, loads, and runs complete mixed dynamic-leaf SSE modules.
 *
 * `settings_per_cta` must be in [1, num_settings]. Settings are partitioned
 * across grid Y while grid X continues to partition row tiles.
 */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_dynamic_leaf_sse_runner_run_all(
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
);

/** Runs the packed optimizer using the common Secant optimizer request. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_packed_constant_optimizer_sse_runner_run(
    SecantCUDARunner runner,
    const SecantCubinPackedConstantOptimizerSSERun* run,
    const char* const* routine_names,
    SecantRunnerStats* stats_ret
);

/** Compiles, loads, and executes one fixed-schedule LM request. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_lm_optimizer_runner_run(
    SecantCUDALMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    SecantRunnerStats* stats_ret
);

/** Releases the LM runner's streams, events, options, and scratch storage. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_lm_optimizer_runner_destroy(
    SecantCUDALMOptimizerRunner runner
);

/** Stops workers and releases internal CUDA resources. */
SECANT_CUDA_RUNNER_DEC SecantResult secant_cuda_runner_destroy(
    SecantCUDARunner runner
);

#endif /* SECANT_CUDA_RUNNER_H_INCLUDED */
