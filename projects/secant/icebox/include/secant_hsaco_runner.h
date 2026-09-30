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
#ifndef SECANT_HSACO_RUNNER_H_INCLUDED
#define SECANT_HSACO_RUNNER_H_INCLUDED

#include "secant_hsaco.h"
#include "secant_runner.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_HSACO_RUNNER_DEC extern "C"
#else
#define SECANT_HSACO_RUNNER_DEC extern
#endif

/** Persistent specialization workers and HIP resources for one HSACO shape. */
typedef struct SecantHsacoRunnerImpl* SecantHsacoRunner;

/**
 * Creates a materialize runner from one inspected HSACO template.
 *
 * The plan is borrowed until destroy. The template is copied into reusable
 * worker slots during creation. The caller's current HIP device must remain
 * current for create, run, and destroy.
 */
SECANT_HSACO_RUNNER_DEC SecantResult secant_hsaco_materialize_runner_create(
    const SecantHsacoPlan* plan,
    const void* hsaco,
    size_t hsaco_size,
    size_t num_workers,
    size_t num_streams,
    SecantHsacoRunner* runner_ret
);

/** Creates an SSE runner under the same ownership and device contract. */
SECANT_HSACO_RUNNER_DEC SecantResult secant_hsaco_sse_runner_create(
    const SecantHsacoPlan* plan,
    const void* hsaco,
    size_t hsaco_size,
    size_t num_workers,
    size_t num_streams,
    SecantHsacoRunner* runner_ret
);

/**
 * Specializes, loads, and runs complete materialize modules.
 *
 * output_module_stride is measured in float elements. A zero stride reuses
 * one module-sized output region for every module. A nonzero stride must be at
 * least one complete module span. A runner does not support concurrent run or
 * destroy calls.
 */
SECANT_HSACO_RUNNER_DEC SecantResult secant_hsaco_materialize_runner_run_all(
    SecantHsacoRunner runner,
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

/** Specializes, loads, and runs complete SSE modules. */
SECANT_HSACO_RUNNER_DEC SecantResult secant_hsaco_sse_runner_run_all(
    SecantHsacoRunner runner,
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

/** Stops workers and releases internal HIP runtime resources. */
SECANT_HSACO_RUNNER_DEC SecantResult secant_hsaco_runner_destroy(
    SecantHsacoRunner runner
);

#endif /* SECANT_HSACO_RUNNER_H_INCLUDED */
