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
#include "secant_cubin_lm_optimizer_runner.h"

#include "lm_optimizer_runner_internal.h"
#include "secant.h"

#include <nvrtc.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define S_CUBIN_LM_RUN_ERROR_RET(ans) do { \
    SecantResult cubin_lm_result_ = (ans); \
    return cubin_lm_result_; \
} while (0)

struct SecantCubinLMOptimizerRunnerImpl {
    SecantCUDALMOptimizerRunner executor;
    SecantCubinPlan* plan;
    void* plan_storage;
    unsigned char* template_cubin;
    size_t cubin_size;
};

static double
s_cubin_lm_seconds(void) {
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void
s_cubin_lm_release(SecantCubinLMOptimizerRunner runner) {
    if (runner == NULL) {
        return;
    }
    if (runner->executor != NULL) {
        (void)secant_cuda_lm_optimizer_runner_destroy(runner->executor);
    }
    free(runner->template_cubin);
    free(runner->plan_storage);
    free(runner);
}

static SecantResult
s_cubin_lm_template_source(
    const SecantCubinLMOptimizerRecipe* recipe,
    char** source_ret
) {
    char* source;
    size_t statistics_size = 0u;
    size_t solver_size = 0u;
    size_t source_size;

    if (recipe == NULL || source_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *source_ret = NULL;
    if (secant_cubin_source_size(&recipe->header, &statistics_size) !=
            SECANT_SUCCESS ||
        secant_cuda_lm_solver_source_generate(NULL, 0u, &solver_size) !=
            SECANT_CUDA_SUCCESS ||
        statistics_size == 0u || solver_size == 0u ||
        statistics_size - 1u > SIZE_MAX - solver_size) {
        return SECANT_ERROR_OVERFLOW;
    }
    source_size = statistics_size - 1u + solver_size;
    source = (char*)malloc(source_size);
    if (source == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    if (secant_cubin_source_write(
            &recipe->header,
            source,
            statistics_size) != SECANT_SUCCESS ||
        secant_cuda_lm_solver_source_generate(
            source + statistics_size - 1u,
            solver_size,
            &solver_size) != SECANT_CUDA_SUCCESS) {
        free(source);
        return SECANT_ERROR_FORMAT;
    }
    *source_ret = source;
    return SECANT_SUCCESS;
}

static void
s_cubin_lm_stats_write(
    SecantRunnerStats* stats_ret,
    const SecantRunnerStats* stats,
    double specialize_seconds
) {
    uint32_t struct_size;

    if (stats_ret == NULL || stats == NULL) {
        return;
    }
    struct_size = stats_ret->struct_size;
    *stats_ret = *stats;
    stats_ret->struct_size = struct_size;
    stats_ret->compile_window_seconds = specialize_seconds;
    stats_ret->compile_critical_seconds = specialize_seconds;
    stats_ret->compile_work_seconds = specialize_seconds;
    stats_ret->total_seconds += specialize_seconds;
}

static SecantResult
s_cubin_lm_template_compile(
    const SecantCubinLMOptimizerRecipe* recipe,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char architecture[64];
    char* source = NULL;
    const char** options = NULL;
    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_result;
    size_t cubin_size = 0u;
    size_t option_idx;
    int architecture_size;
    SecantResult result;

    if (recipe == NULL || compute_capability_major == 0u ||
        (num_nvrtc_options != 0u && nvrtc_options == NULL) ||
        cubin_ret == NULL || cubin_size_ret == NULL ||
        num_nvrtc_options > (size_t)INT_MAX - 3u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    architecture_size = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-architecture=sm_%u%u",
        compute_capability_major,
        compute_capability_minor);
    if (architecture_size < 0 ||
        (size_t)architecture_size >= sizeof(architecture)) {
        return SECANT_ERROR_FORMAT;
    }
    result = s_cubin_lm_template_source(recipe, &source);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    options = (const char**)malloc(
        (num_nvrtc_options + 3u) * sizeof(*options));
    if (options == NULL) {
        free(source);
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=3";
    for (option_idx = 0u; option_idx < num_nvrtc_options; ++option_idx) {
        options[option_idx + 3u] = nvrtc_options[option_idx];
    }
    nvrtc_result = nvrtcCreateProgram(
        &program,
        source,
        "secant_cubin_lm_optimizer_template.cu",
        0,
        NULL,
        NULL);
    if (nvrtc_result == NVRTC_SUCCESS) {
        nvrtc_result = nvrtcCompileProgram(
            program,
            (int)(num_nvrtc_options + 3u),
            options);
    }
    if (nvrtc_result == NVRTC_SUCCESS) {
        nvrtc_result = nvrtcGetCUBINSize(program, &cubin_size);
    }
    if (nvrtc_result == NVRTC_SUCCESS && cubin_size != 0u) {
        *cubin_ret = (unsigned char*)malloc(cubin_size);
        if (*cubin_ret == NULL) {
            result = SECANT_ERROR_ALLOCATION_FAILED;
        } else if (nvrtcGetCUBIN(program, (char*)*cubin_ret) !=
                   NVRTC_SUCCESS) {
            result = SECANT_ERROR_COMPILE_FAILED;
        } else {
            *cubin_size_ret = cubin_size;
            result = SECANT_SUCCESS;
        }
    } else {
        result = SECANT_ERROR_COMPILE_FAILED;
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    free(options);
    free(source);
    if (result != SECANT_SUCCESS) {
        free(*cubin_ret);
        *cubin_ret = NULL;
        *cubin_size_ret = 0u;
    }
    return result;
}

SecantResult
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
) {
    SecantCubinLMOptimizerRunner runner = NULL;
    SecantCubinLMOptimizerRecipe recipe =
        secant_cubin_lm_optimizer_recipe_init();
    size_t plan_storage_size = 0u;
    SecantResult result;

    if (runner_ret == NULL) {
        S_CUBIN_LM_RUN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *runner_ret = NULL;
    recipe.num_kernels = max_num_asts;
    recipe.num_input_columns = num_input_columns;
    recipe.num_static_input_columns = num_static_input_columns;
    recipe.num_parameters = num_parameters;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    result = secant_cubin_recipe_validate(&recipe.header);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    runner = (SecantCubinLMOptimizerRunner)calloc(1u, sizeof(*runner));
    if (runner == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    result = secant_cuda_lm_optimizer_runner_create(
        max_num_asts,
        num_input_columns,
        num_static_input_columns,
        tile_rows,
        threads_per_block,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        compile_scratch_size,
        num_streams,
        &runner->executor);
    if (result == SECANT_SUCCESS) {
        result = s_cubin_lm_template_compile(
            &recipe,
            compute_capability_major,
            compute_capability_minor,
            nvrtc_options,
            num_nvrtc_options,
            &runner->template_cubin,
            &runner->cubin_size);
    }
    if (result == SECANT_SUCCESS) {
        result = secant_cubin_plan_storage_size(
            &recipe.header,
            runner->template_cubin,
            runner->cubin_size,
            &plan_storage_size);
    }
    if (result == SECANT_SUCCESS) {
        runner->plan_storage = malloc(plan_storage_size);
        result = runner->plan_storage == NULL
            ? SECANT_ERROR_ALLOCATION_FAILED
            : secant_cubin_plan_init(
                &recipe.header,
                runner->template_cubin,
                runner->cubin_size,
                runner->plan_storage,
                plan_storage_size,
                &runner->plan);
    }
    if (result != SECANT_SUCCESS) {
        s_cubin_lm_release(runner);
        return result;
    }
    *runner_ret = runner;
    return SECANT_SUCCESS;
}

SecantResult
secant_cubin_lm_optimizer_runner_run(
    SecantCubinLMOptimizerRunner runner,
    const SecantCUDALMOptimizerRun* run,
    SecantRunnerStats* stats_ret
) {
    SecantRunnerStats execution_stats = secant_runner_stats_init();
    SecantAstProgramSet programs;
    unsigned char* cubin;
    const double begin = s_cubin_lm_seconds();
    double specialize_seconds;
    SecantResult result;

    if (runner == NULL || run == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    cubin = (unsigned char*)malloc(runner->cubin_size);
    if (cubin == NULL) {
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    memcpy(cubin, runner->template_cubin, runner->cubin_size);
    memset(&programs, 0, sizeof(programs));
    programs.routines = run->routines;
    programs.asts = run->asts;
    result = secant_cubin_specialize_into(
        runner->plan,
        &programs,
        cubin,
        runner->cubin_size);
    specialize_seconds = s_cubin_lm_seconds() - begin;
    if (result == SECANT_SUCCESS) {
        result = _secant_cuda_lm_optimizer_runner_execute_binary(
            runner->executor,
            run,
            cubin,
            runner->cubin_size,
            &execution_stats);
    }
    s_cubin_lm_stats_write(
        stats_ret,
        &execution_stats,
        specialize_seconds);
    free(cubin);
    return result;
}

SecantResult
secant_cubin_lm_optimizer_runner_destroy(
    SecantCubinLMOptimizerRunner runner
) {
    if (runner == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    s_cubin_lm_release(runner);
    return SECANT_SUCCESS;
}

#undef S_CUBIN_LM_RUN_ERROR_RET
