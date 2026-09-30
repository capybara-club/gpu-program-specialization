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
#ifndef SECANT_CUBIN_LM_OPTIMIZER_RUNNER_H_INCLUDED
#define SECANT_CUBIN_LM_OPTIMIZER_RUNNER_H_INCLUDED

#include "secant_cuda_runner.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_CUBIN_LM_RUNNER_DEC extern "C"
#else
#define SECANT_CUBIN_LM_RUNNER_DEC extern
#endif

/** Persistent template and CUDA resources for native-SASS LM execution. */
typedef struct SecantCubinLMOptimizerRunnerImpl*
    SecantCubinLMOptimizerRunner;

/**
 * Creates a thread-owned, one-AST-per-kernel native-SASS LM runner.
 *
 * The fixed statistics skeleton and shared solve kernel are compiled once at
 * creation. Each run only specializes AST value/gradient SASS into a private
 * copy of that template before executing the same schedule as the CUDA and PTX
 * comparison runners.
 */
SECANT_CUBIN_LM_RUNNER_DEC SecantResult
secant_cubin_lm_optimizer_runner_create(
    size_t max_num_asts,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_parameters,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    size_t compile_scratch_size,
    size_t num_streams,
    SecantCubinLMOptimizerRunner* runner_ret
);

/** Specializes the requested ASTs and runs the fixed LM schedule. */
SECANT_CUBIN_LM_RUNNER_DEC SecantResult
secant_cubin_lm_optimizer_runner_run(
    SecantCubinLMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    SecantRunnerStats* stats_ret
);

/** Releases the template, plan storage, and CUDA execution resources. */
SECANT_CUBIN_LM_RUNNER_DEC SecantResult
secant_cubin_lm_optimizer_runner_destroy(
    SecantCubinLMOptimizerRunner runner
);

#undef SECANT_CUBIN_LM_RUNNER_DEC

#endif
