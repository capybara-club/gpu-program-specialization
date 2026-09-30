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
#ifndef SECANT_PTX_RUNNER_H_INCLUDED
#define SECANT_PTX_RUNNER_H_INCLUDED

#include "secant_ptx.h"
#include "secant.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_PTX_RUNNER_DEC extern "C"
#else
#define SECANT_PTX_RUNNER_DEC extern
#endif

/** Comparison-only bulk runner for the PTX compiler baseline. */
typedef struct SecantPTXRunnerImpl* SecantPTXRunner;

/**
 * Creates a materialize runner borrowing one prepared PTX handle.
 *
 * NVPTXCompiler options are copied during creation. The PTX handle must remain
 * alive until runner destruction. The caller's CUDA context must remain
 * current for create, run, and destroy. CUDA eager module loading is required.
 */
SECANT_PTX_RUNNER_DEC SecantResult secant_ptx_materialize_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
);

/** Creates an SSE runner borrowing one prepared PTX handle. */
SECANT_PTX_RUNNER_DEC SecantResult secant_ptx_sse_runner_create(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    size_t compile_scratch_size,
    size_t num_workers,
    size_t num_streams,
    SecantPTXRunner* runner_ret
);

/**
 * Compiles, loads, and runs complete PTX materialize modules.
 *
 * output_module_stride is measured in float elements. A zero stride reuses
 * one module-sized output region for every module. A nonzero stride must be at
 * least one complete module span. A runner does not support concurrent run or
 * destroy calls.
 */
SECANT_PTX_RUNNER_DEC SecantResult secant_ptx_materialize_runner_run_all(
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
);

/** Compiles, loads, and runs complete PTX SSE modules. */
SECANT_PTX_RUNNER_DEC SecantResult secant_ptx_sse_runner_run_all(
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
);

/** Stops workers and releases internal CUDA resources. */
SECANT_PTX_RUNNER_DEC SecantResult secant_ptx_runner_destroy(
    SecantPTXRunner runner
);

#endif /* SECANT_PTX_RUNNER_H_INCLUDED */
