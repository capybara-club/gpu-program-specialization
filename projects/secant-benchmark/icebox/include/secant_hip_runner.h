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
#ifndef SECANT_HIP_RUNNER_H_INCLUDED
#define SECANT_HIP_RUNNER_H_INCLUDED

#include "secant_hip.h"
#include "secant_runner.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_HIP_RUNNER_DEC extern "C"
#else
#define SECANT_HIP_RUNNER_DEC extern
#endif

/** Comparison-only bulk runner for the HIP C++ compiler baseline. */
typedef struct SecantHIPRunnerImpl* SecantHIPRunner;

/**
 * Creates a native HIP materialize runner.
 *
 * HIPRTC options and architecture are copied during creation.
 * compile_scratch_size bytes are allocated for every worker. The caller's
 * current HIP device must remain current for create, run, and destroy.
 */
SECANT_HIP_RUNNER_DEC SecantResult secant_hip_materialize_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const char* architecture,
    const char* const* hiprtc_options,
    size_t num_hiprtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantHIPRunner* runner_ret
);

/** Creates a native HIP SSE runner under the same ownership contract. */
SECANT_HIP_RUNNER_DEC SecantResult secant_hip_sse_runner_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const char* architecture,
    const char* const* hiprtc_options,
    size_t num_hiprtc_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantHIPRunner* runner_ret
);

/**
 * Compiles, loads, and runs complete materialize modules.
 *
 * ASTs are divided into complete template-sized modules. Device addresses use
 * uintptr_t so this header does not depend on HIP headers. Output is laid out
 * [ast][row]. output_module_stride is measured in float elements; zero reuses
 * one module-sized output region for every module. A nonzero stride must be at
 * least one complete module span. A runner does not support concurrent run or
 * destroy calls.
 */
SECANT_HIP_RUNNER_DEC SecantResult secant_hip_materialize_runner_run_all(
    SecantHIPRunner runner,
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

/**
 * Compiles, loads, and runs complete SSE modules.
 *
 * The runner clears the full [ast][target] output before launching.
 */
SECANT_HIP_RUNNER_DEC SecantResult secant_hip_sse_runner_run_all(
    SecantHIPRunner runner,
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

/** Stops workers and releases internal HIP runtime resources. */
SECANT_HIP_RUNNER_DEC SecantResult secant_hip_runner_destroy(
    SecantHIPRunner runner
);

#endif /* SECANT_HIP_RUNNER_H_INCLUDED */
