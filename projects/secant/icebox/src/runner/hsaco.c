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
#include "secant_hsaco_runner.h"

#include "amd_internal.h"

#include <stdlib.h>
#include <string.h>

struct SecantHsacoRunnerImpl {
    _SecantAmdRunner* runner;
    const SecantHsacoPlan* plan;
    unsigned char* slot_hsacos;
    size_t hsaco_size;
    size_t num_slots;
};

static SecantResult
_secant_hsaco_runner_slot_create(
    void* backend,
    size_t slot_idx,
    _SecantAmdRunnerSlot* slot
) {
    SecantHsacoRunner runner = (SecantHsacoRunner)backend;

    if (runner == NULL || slot == NULL ||
        slot_idx >= runner->num_slots) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    slot->artifact =
        runner->slot_hsacos + slot_idx * runner->hsaco_size;
    return SECANT_SUCCESS;
}

static void
_secant_hsaco_runner_slot_release(
    void* backend,
    _SecantAmdRunnerSlot* slot
) {
    (void)backend;
    if (slot != NULL) {
        slot->binary = NULL;
        slot->binary_size = 0u;
    }
}

static SecantResult
_secant_hsaco_runner_compile(
    void* backend,
    size_t worker_idx,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    _SecantAmdRunnerSlot* slot
) {
    SecantHsacoRunner runner = (SecantHsacoRunner)backend;
    SecantResult result;

    (void)worker_idx;
    (void)routine_names;
    if (runner == NULL || asts == NULL || slot == NULL ||
        slot->artifact == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    result = secant_hsaco_specialize_into(
        runner->plan,
        routines,
        num_routines,
        asts,
        runner->runner->asts_per_module,
        slot->artifact,
        runner->hsaco_size);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    slot->binary = slot->artifact;
    slot->binary_size = runner->hsaco_size;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_runner_create(
    _SecantAmdRunnerShape shape,
    const SecantHsacoPlan* plan,
    const void* hsaco,
    size_t hsaco_size,
    size_t num_workers,
    size_t num_streams,
    SecantHsacoRunner* runner_ret
) {
    SecantHsacoRunner runner;
    const char* function_name_prefix;
    size_t inspected_hsaco_size;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_targets = 0u;
    size_t tile_rows = 0u;
    size_t threads_per_block = 0u;
    size_t slot_bytes;
    size_t slot_idx;
    SecantResult result;

    if (plan == NULL || hsaco == NULL || hsaco_size == 0u ||
        num_workers == 0u || num_streams == 0u ||
        runner_ret == NULL || num_workers > SIZE_MAX - 2u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *runner_ret = NULL;
    if (shape == _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE) {
        result = secant_hsaco_materialize_plan_info_get(
            plan,
            &inspected_hsaco_size,
            &num_kernels,
            &asts_per_kernel,
            &num_inputs);
        function_name_prefix = "secant_hsaco_materialize";
    } else {
        result = secant_hsaco_sse_plan_info_get(
            plan,
            &inspected_hsaco_size,
            &num_kernels,
            &asts_per_kernel,
            &num_inputs,
            &num_targets,
            &tile_rows,
            &threads_per_block);
        function_name_prefix = "secant_hsaco_sse";
    }
    if (result != SECANT_SUCCESS) {
        return result;
    }
    if (inspected_hsaco_size != hsaco_size) {
        return SECANT_ERROR_BAD_BINARY;
    }
    runner = (SecantHsacoRunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    runner->plan = plan;
    runner->hsaco_size = hsaco_size;
    runner->num_slots = num_workers + 2u;
    if (runner->num_slots > SIZE_MAX / hsaco_size) {
        free(runner);
        return SECANT_ERROR_OVERFLOW;
    }
    slot_bytes = runner->num_slots * hsaco_size;
    runner->slot_hsacos =
        (unsigned char*)malloc(slot_bytes);
    if (runner->slot_hsacos == NULL) {
        free(runner);
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    for (slot_idx = 0u; slot_idx < runner->num_slots; ++slot_idx) {
        memcpy(
            runner->slot_hsacos + slot_idx * hsaco_size,
            hsaco,
            hsaco_size);
    }
    result = _secant_amd_runner_create(
        shape,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        tile_rows,
        threads_per_block,
        num_workers,
        num_streams,
        function_name_prefix,
        runner,
        _secant_hsaco_runner_compile,
        _secant_hsaco_runner_slot_create,
        _secant_hsaco_runner_slot_release,
        &runner->runner);
    if (result != SECANT_SUCCESS) {
        free(runner->slot_hsacos);
        free(runner);
        return result;
    }
    *runner_ret = runner;
    return SECANT_SUCCESS;
}

SecantResult
secant_hsaco_materialize_runner_create(
    const SecantHsacoPlan* plan,
    const void* hsaco,
    size_t hsaco_size,
    size_t num_workers,
    size_t num_streams,
    SecantHsacoRunner* runner_ret
) {
    return _secant_hsaco_runner_create(
        _SECANT_AMD_RUNNER_SHAPE_MATERIALIZE,
        plan,
        hsaco,
        hsaco_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_hsaco_sse_runner_create(
    const SecantHsacoPlan* plan,
    const void* hsaco,
    size_t hsaco_size,
    size_t num_workers,
    size_t num_streams,
    SecantHsacoRunner* runner_ret
) {
    return _secant_hsaco_runner_create(
        _SECANT_AMD_RUNNER_SHAPE_SSE,
        plan,
        hsaco,
        hsaco_size,
        num_workers,
        num_streams,
        runner_ret);
}

SecantResult
secant_hsaco_materialize_runner_run_all(
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
) {
    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_amd_materialize_runner_run_all(
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
secant_hsaco_sse_runner_run_all(
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
) {
    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_amd_sse_runner_run_all(
        runner->runner,
        routines,
        num_routines,
        NULL,
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
secant_hsaco_runner_destroy(
    SecantHsacoRunner runner
) {
    SecantResult result;

    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    result = _secant_amd_runner_destroy(runner->runner);
    if (result == SECANT_ERROR_INVALID_STATE) {
        return result;
    }
    free(runner->slot_hsacos);
    free(runner);
    return result;
}
