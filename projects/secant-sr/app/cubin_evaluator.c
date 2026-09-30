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
#define _POSIX_C_SOURCE 200809L

#include "cubin_evaluator.h"

#include <nvrtc.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
secant_sr_cubin_seconds_get(void) {
    struct timespec value;

    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int
secant_sr_size_mul(size_t left, size_t right, size_t* result_ret) {
    if (left != 0u && right > SIZE_MAX / left) {
        return 0;
    }
    *result_ret = left * right;
    return 1;
}

int
secant_sr_cuda_session_create(SecantSRCudaSession* session) {
    CUdevice device;

    if (session == NULL) {
        return 0;
    }
    memset(session, 0, sizeof(*session));
    return cuInit(0u) == CUDA_SUCCESS &&
        cuDeviceGet(&device, 0) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(
            &session->compute_capability_major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(
            &session->compute_capability_minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device) == CUDA_SUCCESS &&
        cuCtxCreate(&session->context, NULL, 0u, device) == CUDA_SUCCESS;
}

void
secant_sr_cuda_session_destroy(SecantSRCudaSession* session) {
    if (session == NULL) {
        return;
    }
    if (session->context != NULL) {
        (void)cuCtxDestroy(session->context);
    }
    memset(session, 0, sizeof(*session));
}

static int
secant_sr_nvrtc_compile(
    const char* source,
    int major,
    int minor,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret
) {
    char architecture[64];
    const char* options[4];
    nvrtcProgram program = NULL;
    nvrtcResult result;
    unsigned char* cubin = NULL;
    size_t cubin_size = 0u;
    int success = 0;

    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    if (snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d", major, minor) < 0) {
        return 0;
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    options[2] = "--ptxas-options=--opt-level=1";
    options[3] = "--no-cache";
    result = nvrtcCreateProgram(&program, source, "secant_sr_sse.cu", 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, 4, options);
    }
    if (result == NVRTC_SUCCESS) {
        result = nvrtcGetCUBINSize(program, &cubin_size);
    }
    if (result == NVRTC_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin != NULL && nvrtcGetCUBIN(program, (char*)cubin) == NVRTC_SUCCESS) {
            success = 1;
        }
    }
    if (!success && program != NULL) {
        size_t log_size = 0u;

        if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1u) {
            char* log = malloc(log_size);

            if (log != NULL) {
                if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
                    fprintf(stderr, "%s\n", log);
                }
                free(log);
            }
        }
    }
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    if (!success) {
        free(cubin);
        return 0;
    }
    *cubin_ret = cubin;
    *cubin_size_ret = cubin_size;
    return 1;
}

static void
secant_sr_cubin_template_trace(
    const SecantCubinRecipeHeader* recipe,
    int major,
    int minor,
    size_t source_size,
    size_t cubin_size
) {
    const char* trace = getenv("SECANT_SR_TEMPLATE_TRACE");
    int nvrtc_major = 0;
    int nvrtc_minor = 0;

    if (trace == NULL || trace[0] == '\0' || strcmp(trace, "0") == 0) {
        return;
    }
    (void)nvrtcVersion(&nvrtc_major, &nvrtc_minor);
    printf(
        "template_generated secant_version=%s recipe_version=%u recipe_flags=%u cc_major=%d cc_minor=%d "
        "nvrtc_major=%d nvrtc_minor=%d ptxas_opt_level=1 nvrtc_no_cache=1 "
        "source_bytes=%zu cubin_bytes=%zu ",
        SECANT_VERSION_STRING,
        recipe->version,
        recipe->flags,
        major,
        minor,
        nvrtc_major,
        nvrtc_minor,
        source_size,
        cubin_size);
    switch (recipe->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32: {
            const SecantCubinMaterializeRecipe* value =
                (const SecantCubinMaterializeRecipe*)(const void*)recipe;

            printf(
                "shape=static_materialize_f32 num_kernels=%zu asts_per_kernel=%zu num_inputs=%zu "
                "patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_inputs,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32: {
            const SecantCubinSSERecipe* value = (const SecantCubinSSERecipe*)(const void*)recipe;

            printf(
                "shape=static_sse_f32 num_kernels=%zu asts_per_kernel=%zu num_inputs=%zu num_targets=%zu "
                "tile_rows=%zu threads_per_block=%zu patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_inputs,
                value->num_targets,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32: {
            const SecantCubinAffineStatsRecipe* value =
                (const SecantCubinAffineStatsRecipe*)(const void*)recipe;

            printf(
                "shape=static_affine_stats_f32 num_kernels=%zu asts_per_kernel=%zu num_inputs=%zu "
                "num_targets=%zu tile_rows=%zu threads_per_block=%zu patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_inputs,
                value->num_targets,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32: {
            const SecantCubinGramStatsRecipe* value =
                (const SecantCubinGramStatsRecipe*)(const void*)recipe;

            printf(
                "shape=static_gram_stats_f32 num_kernels=%zu asts_per_kernel=%zu num_inputs=%zu "
                "num_targets=%zu tile_rows=%zu threads_per_block=%zu patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_inputs,
                value->num_targets,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE_F32: {
            const SecantCubinDynamicConstantSSERecipe* value =
                (const SecantCubinDynamicConstantSSERecipe*)(const void*)recipe;

            printf(
                "shape=dynamic_constant_sse_f32 num_kernels=%zu asts_per_kernel=%zu "
                "num_input_columns=%zu num_input_constants=%zu num_targets=%zu tile_rows=%zu "
                "threads_per_block=%zu patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_input_columns,
                value->num_input_constants,
                value->num_targets,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_CONSTANT_OPTIMIZER_SSE_F32: {
            const SecantCubinConstantOptimizerSSERecipe* value =
                (const SecantCubinConstantOptimizerSSERecipe*)(const void*)recipe;

            printf(
                "shape=constant_optimizer_sse_f32 num_kernels=%zu num_input_columns=%zu "
                "num_input_constants=%zu tile_rows=%zu threads_per_block=%zu "
                "patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->num_input_columns,
                value->num_input_constants,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE_F32: {
            const SecantCubinPackedConstantOptimizerSSERecipe* value =
                (const SecantCubinPackedConstantOptimizerSSERecipe*)(const void*)recipe;

            printf(
                "shape=packed_constant_optimizer_sse_f32 num_kernels=%zu asts_per_kernel=%zu "
                "num_input_columns=%zu num_input_constants=%zu tile_rows=%zu threads_per_block=%zu "
                "patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_input_columns,
                value->num_input_constants,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        case SECANT_KERNEL_SHAPE_DYNAMIC_LEAF_SSE_F32: {
            const SecantCubinDynamicLeafSSERecipe* value =
                (const SecantCubinDynamicLeafSSERecipe*)(const void*)recipe;

            printf(
                "shape=dynamic_leaf_sse_f32 num_kernels=%zu asts_per_kernel=%zu num_input_columns=%zu "
                "num_static_input_columns=%zu num_dynamic_leaves=%zu num_targets=%zu tile_rows=%zu "
                "threads_per_block=%zu patch_capacity_instructions=%zu\n",
                value->num_kernels,
                value->asts_per_kernel,
                value->num_input_columns,
                value->num_static_input_columns,
                value->num_dynamic_leaves,
                value->num_targets,
                value->tile_rows,
                value->threads_per_block,
                value->patch_capacity_instructions);
            break;
        }
        default:
            printf("shape=unknown shape_value=%u\n", recipe->shape);
            break;
    }
    fflush(stdout);
}

static int
secant_sr_cubin_source_prepare(
    const SecantCubinRecipeHeader* recipe,
    SecantSRCubinEvaluator* evaluator
) {
    size_t required_size;

    if (evaluator->source != NULL) {
        return 1;
    }
    if (secant_cubin_source_size(recipe, &required_size) != SECANT_SUCCESS) {
        return 0;
    }
    evaluator->source = malloc(required_size);
    evaluator->source_size = required_size;
    return evaluator->source != NULL &&
        secant_cubin_source_write(recipe, evaluator->source, required_size) == SECANT_SUCCESS;
}

static int
secant_sr_cubin_evaluator_prepare(
    const SecantCubinRecipeHeader* recipe,
    size_t num_workers,
    size_t num_streams,
    int major,
    int minor,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    const double prepare_begin = secant_sr_cubin_seconds_get();
    double stored_compile_seconds = 0.0;
    double operation_begin;
    size_t required_size;
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    int cache_hit = 0;
    int cache_store_pending = 0;
    SecantResult plan_result;

    if (nvrtcVersion(&nvrtc_major, &nvrtc_minor) != NVRTC_SUCCESS) {
        return 0;
    }
    if (cache != NULL) {
        operation_begin = secant_sr_cubin_seconds_get();
        if (!secant_sr_cubin_cache_lookup(
                cache,
                recipe,
                major,
                minor,
                nvrtc_major,
                nvrtc_minor,
                1,
                1,
                &evaluator->cubin,
                &evaluator->cubin_size,
                &stored_compile_seconds,
                &cache_hit)) {
            return 0;
        }
        evaluator->template_stats.cache_lookup_seconds += secant_sr_cubin_seconds_get() - operation_begin;
    }
    if (!cache_hit) {
        operation_begin = secant_sr_cubin_seconds_get();
        if (!secant_sr_cubin_source_prepare(recipe, evaluator)) {
            return 0;
        }
        evaluator->template_stats.source_seconds += secant_sr_cubin_seconds_get() - operation_begin;
        operation_begin = secant_sr_cubin_seconds_get();
        if (!secant_sr_nvrtc_compile(evaluator->source, major, minor, &evaluator->cubin, &evaluator->cubin_size)) {
            return 0;
        }
        stored_compile_seconds = secant_sr_cubin_seconds_get() - operation_begin;
        evaluator->template_stats.compile_seconds += stored_compile_seconds;
        evaluator->template_stats.estimated_uncached_compile_seconds += stored_compile_seconds;
        if (cache != NULL) {
            evaluator->template_stats.cache_misses += 1u;
            cache_store_pending = 1;
        }
        secant_sr_cubin_template_trace(recipe, major, minor, evaluator->source_size, evaluator->cubin_size);
    } else {
        evaluator->template_stats.cache_hits += 1u;
    }
    operation_begin = secant_sr_cubin_seconds_get();
    plan_result = secant_cubin_plan_storage_size(recipe, evaluator->cubin, evaluator->cubin_size, &required_size);
    evaluator->template_stats.inspect_plan_seconds += secant_sr_cubin_seconds_get() - operation_begin;
    if (plan_result != SECANT_SUCCESS) {
        if (!cache_hit) {
            return 0;
        }
        fprintf(stderr, "cached CUBIN failed inspection; recompiling its exact source\n");
        evaluator->template_stats.cache_invalidations += 1u;
        free(evaluator->cubin);
        evaluator->cubin = NULL;
        evaluator->cubin_size = 0u;
        operation_begin = secant_sr_cubin_seconds_get();
        if (!secant_sr_cubin_source_prepare(recipe, evaluator)) {
            return 0;
        }
        evaluator->template_stats.source_seconds += secant_sr_cubin_seconds_get() - operation_begin;
        operation_begin = secant_sr_cubin_seconds_get();
        if (!secant_sr_nvrtc_compile(evaluator->source, major, minor, &evaluator->cubin, &evaluator->cubin_size)) {
            return 0;
        }
        stored_compile_seconds = secant_sr_cubin_seconds_get() - operation_begin;
        evaluator->template_stats.compile_seconds += stored_compile_seconds;
        cache_store_pending = 1;
        secant_sr_cubin_template_trace(recipe, major, minor, evaluator->source_size, evaluator->cubin_size);
        operation_begin = secant_sr_cubin_seconds_get();
        plan_result = secant_cubin_plan_storage_size(recipe, evaluator->cubin, evaluator->cubin_size, &required_size);
        evaluator->template_stats.inspect_plan_seconds += secant_sr_cubin_seconds_get() - operation_begin;
        if (plan_result != SECANT_SUCCESS) {
            return 0;
        }
        cache_hit = 0;
    }
    if (cache_hit) {
        evaluator->template_stats.estimated_uncached_compile_seconds += stored_compile_seconds;
    } else if (evaluator->template_stats.estimated_uncached_compile_seconds == 0.0) {
        evaluator->template_stats.estimated_uncached_compile_seconds += stored_compile_seconds;
    }
    evaluator->plan_storage = malloc(required_size);
    evaluator->plan_storage_size = required_size;
    options.num_workers = num_workers;
    options.num_streams = num_streams;
    operation_begin = secant_sr_cubin_seconds_get();
    if (evaluator->plan_storage == NULL ||
        secant_cubin_plan_init(
            recipe,
            evaluator->cubin,
            evaluator->cubin_size,
            evaluator->plan_storage,
            evaluator->plan_storage_size,
            &evaluator->plan) != SECANT_SUCCESS) {
        return 0;
    }
    evaluator->template_stats.inspect_plan_seconds += secant_sr_cubin_seconds_get() - operation_begin;
    if (cache_store_pending) {
        operation_begin = secant_sr_cubin_seconds_get();
        if (!secant_sr_cubin_cache_store(
                cache,
                recipe,
                major,
                minor,
                nvrtc_major,
                nvrtc_minor,
                1,
                1,
                evaluator->cubin,
                evaluator->cubin_size,
                stored_compile_seconds)) {
            return 0;
        }
        evaluator->template_stats.cache_store_seconds += secant_sr_cubin_seconds_get() - operation_begin;
    }
    operation_begin = secant_sr_cubin_seconds_get();
    if (secant_cubin_runner_create(
            evaluator->plan,
            evaluator->cubin,
            evaluator->cubin_size,
            &options,
            &evaluator->runner) != SECANT_SUCCESS) {
        return 0;
    }
    evaluator->template_stats.runner_create_seconds += secant_sr_cubin_seconds_get() - operation_begin;
    evaluator->last_stats = secant_runner_stats_init();
    evaluator->template_stats.prepare_seconds = secant_sr_cubin_seconds_get() - prepare_begin;
    return 1;
}

static int
secant_sr_cubin_evaluator_create_inner(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t population_size,
    size_t num_rows,
    CUcontext context,
    int major,
    int minor,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinSSERecipe recipe = secant_cubin_sse_recipe_init();
    size_t asts_per_module;
    size_t input_elements;
    size_t output_bytes;

    if (population_size == 0u || num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u ||
        num_rows == 0u || !secant_sr_size_mul(num_kernels, asts_per_kernel, &asts_per_module) ||
        !secant_sr_size_mul(num_inputs, num_rows, &input_elements) ||
        !secant_sr_size_mul(population_size, sizeof(float), &output_bytes)) {
        return 0;
    }
    evaluator->context = context;
    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_inputs = num_inputs;
    recipe.num_targets = 1u;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (!secant_sr_cubin_evaluator_prepare(
            &recipe.header, num_workers, num_streams, major, minor, cache, evaluator)) {
        return 0;
    }

    {
        const double device_prepare_begin = secant_sr_cubin_seconds_get();

        if (input_elements > SIZE_MAX / sizeof(float) ||
            cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->output, output_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->target, target, num_rows * sizeof(float)) != CUDA_SUCCESS) {
            return 0;
        }
        evaluator->template_stats.device_prepare_seconds += secant_sr_cubin_seconds_get() - device_prepare_begin;
    }
    evaluator->population_size = population_size;
    evaluator->num_inputs = num_inputs;
    evaluator->num_rows = num_rows;
    return 1;
}

int
secant_sr_cubin_evaluator_routines_set(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* routines,
    size_t num_routines
) {
    if (evaluator == NULL || num_routines > SECANT_AST_MAX_ROUTINES ||
        (num_routines != 0u && routines == NULL)) {
        return 0;
    }
    evaluator->routines = routines;
    evaluator->num_routines = num_routines;
    return 1;
}

static int
secant_sr_cubin_materialize_evaluator_create_inner(
    size_t num_inputs,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    size_t num_rows,
    CUcontext context,
    int major,
    int minor,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();
    size_t input_elements;

    if (num_inputs == 0u || num_rows == 0u || !secant_sr_size_mul(num_inputs, num_rows, &input_elements)) {
        return 0;
    }
    evaluator->context = context;
    recipe.num_kernels = 1u;
    recipe.asts_per_kernel = 1u;
    recipe.num_inputs = num_inputs;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (!secant_sr_cubin_evaluator_prepare(
            &recipe.header, num_workers, num_streams, major, minor, cache, evaluator)) {
        return 0;
    }
    {
        const double device_prepare_begin = secant_sr_cubin_seconds_get();

        if (input_elements > SIZE_MAX / sizeof(float) || num_rows > SIZE_MAX / sizeof(float) ||
            cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->output, num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS) {
            return 0;
        }
        evaluator->template_stats.device_prepare_seconds += secant_sr_cubin_seconds_get() - device_prepare_begin;
    }
    evaluator->population_size = 1u;
    evaluator->num_inputs = num_inputs;
    evaluator->num_rows = num_rows;
    evaluator->materialize = 1;
    return 1;
}

int
secant_sr_cubin_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    if (evaluator == NULL || input == NULL || target == NULL || session == NULL || session->context == NULL ||
        num_workers == 0u || num_streams == 0u) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    if (!secant_sr_cubin_evaluator_create_inner(
        num_kernels,
        asts_per_kernel,
        num_inputs,
        tile_rows,
        threads_per_block,
        patch_capacity_instructions,
        num_workers,
        num_streams,
        input,
        target,
        population_size,
        num_rows,
        session->context,
        session->compute_capability_major,
        session->compute_capability_minor,
        cache,
        evaluator)) {
        secant_sr_cubin_evaluator_destroy(evaluator);
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_evaluator_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* sse
) {
    SecantCubinSSERun run = secant_cubin_sse_run_init();
    SecantResult result;

    if (evaluator == NULL || evaluator->runner == NULL || evaluator->dynamic_leaf || evaluator->dynamic_constant ||
        asts == NULL || sse == NULL || num_asts == 0u || num_asts > evaluator->population_size) {
        return 0;
    }
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.programs.routines.items = evaluator->routines;
    run.programs.routines.count = evaluator->num_routines;
    run.input.address = (uintptr_t)evaluator->input;
    run.input.num_elements = evaluator->num_inputs * evaluator->num_rows;
    run.input.leading_dimension = evaluator->num_rows;
    run.targets.address = (uintptr_t)evaluator->target;
    run.targets.num_elements = evaluator->num_rows;
    run.targets.leading_dimension = evaluator->num_rows;
    run.num_rows = evaluator->num_rows;
    run.num_targets = 1u;
    run.output.address = (uintptr_t)evaluator->output;
    run.output.num_elements = num_asts;
    run.output.leading_dimension = 1u;
    evaluator->last_stats = secant_runner_stats_init();
    result = secant_cubin_runner_run_sse(evaluator->runner, &run, &evaluator->last_stats);
    return result == SECANT_SUCCESS &&
        cuMemcpyDtoH(sse, evaluator->output, num_asts * sizeof(float)) == CUDA_SUCCESS;
}

static int
secant_sr_cubin_dynamic_constant_evaluator_create_inner(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    const float* constant_settings,
    size_t constant_settings_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    CUcontext context,
    int major,
    int minor,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinDynamicConstantSSERecipe recipe = secant_cubin_dynamic_constant_sse_recipe_init();
    size_t input_elements;
    size_t constant_elements;
    size_t output_elements;

    if (population_size == 0u || num_kernels == 0u || asts_per_kernel == 0u || num_input_columns == 0u ||
        num_input_constants == 0u || num_settings == 0u || num_rows == 0u ||
        constant_settings_leading_dimension < num_settings ||
        !secant_sr_size_mul(num_input_columns, num_rows, &input_elements) ||
        !secant_sr_size_mul(num_input_constants, constant_settings_leading_dimension, &constant_elements) ||
        !secant_sr_size_mul(population_size, num_settings, &output_elements)) {
        return 0;
    }
    evaluator->context = context;
    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_input_columns = num_input_columns;
    recipe.num_input_constants = num_input_constants;
    recipe.num_targets = 1u;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (!secant_sr_cubin_evaluator_prepare(
            &recipe.header, num_workers, num_streams, major, minor, cache, evaluator)) {
        return 0;
    }
    {
        const double device_prepare_begin = secant_sr_cubin_seconds_get();

        if (input_elements > SIZE_MAX / sizeof(float) || constant_elements > SIZE_MAX / sizeof(float) ||
            output_elements > SIZE_MAX / sizeof(float) ||
            cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->constant_settings, constant_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->output, output_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->target, target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(
                evaluator->constant_settings,
                constant_settings,
                constant_elements * sizeof(float)) != CUDA_SUCCESS) {
            return 0;
        }
        evaluator->template_stats.device_prepare_seconds += secant_sr_cubin_seconds_get() - device_prepare_begin;
    }
    evaluator->population_size = population_size;
    evaluator->num_inputs = num_input_columns;
    evaluator->num_rows = num_rows;
    evaluator->num_input_constants = num_input_constants;
    evaluator->num_settings = num_settings;
    evaluator->leaf_words_leading_dimension = constant_settings_leading_dimension;
    evaluator->dynamic_constant = 1;
    return 1;
}

int
secant_sr_cubin_dynamic_constant_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    const float* constant_settings,
    size_t constant_settings_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    if (evaluator == NULL || input == NULL || target == NULL || constant_settings == NULL || session == NULL ||
        session->context == NULL || num_workers == 0u || num_streams == 0u) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    if (!secant_sr_cubin_dynamic_constant_evaluator_create_inner(
            num_kernels,
            asts_per_kernel,
            num_input_columns,
            num_input_constants,
            tile_rows,
            threads_per_block,
            patch_capacity_instructions,
            num_workers,
            num_streams,
            input,
            target,
            constant_settings,
            constant_settings_leading_dimension,
            num_settings,
            population_size,
            num_rows,
            session->context,
            session->compute_capability_major,
            session->compute_capability_minor,
            cache,
            evaluator)) {
        secant_sr_cubin_evaluator_destroy(evaluator);
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_dynamic_constant_evaluator_run_device(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts
) {
    SecantCubinDynamicConstantSSERun run = secant_cubin_dynamic_constant_sse_run_init();
    SecantResult result;
    size_t output_elements;

    if (evaluator == NULL || evaluator->runner == NULL || !evaluator->dynamic_constant || asts == NULL ||
        num_asts == 0u || num_asts > evaluator->population_size ||
        !secant_sr_size_mul(num_asts, evaluator->num_settings, &output_elements)) {
        return 0;
    }
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.programs.routines.items = evaluator->routines;
    run.programs.routines.count = evaluator->num_routines;
    run.input.address = (uintptr_t)evaluator->input;
    run.input.num_elements = evaluator->num_inputs * evaluator->num_rows;
    run.input.leading_dimension = evaluator->num_rows;
    run.constant_settings.address = (uintptr_t)evaluator->constant_settings;
    run.constant_settings.num_elements = evaluator->num_input_constants * evaluator->leaf_words_leading_dimension;
    run.constant_settings.leading_dimension = evaluator->leaf_words_leading_dimension;
    run.targets.address = (uintptr_t)evaluator->target;
    run.targets.num_elements = evaluator->num_rows;
    run.targets.leading_dimension = evaluator->num_rows;
    run.num_rows = evaluator->num_rows;
    run.num_settings = evaluator->num_settings;
    run.num_targets = 1u;
    run.output.address = (uintptr_t)evaluator->output;
    run.output.num_elements = output_elements;
    run.output.leading_dimension = evaluator->num_settings;
    evaluator->last_stats = secant_runner_stats_init();
    result = secant_cubin_runner_run_dynamic_constant_sse(evaluator->runner, &run, &evaluator->last_stats);
    return result == SECANT_SUCCESS;
}

int
secant_sr_cubin_dynamic_constant_evaluator_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* sse
) {
    size_t output_elements;

    if (sse == NULL || !secant_sr_size_mul(num_asts, evaluator != NULL ? evaluator->num_settings : 0u, &output_elements) ||
        !secant_sr_cubin_dynamic_constant_evaluator_run_device(evaluator, asts, num_asts)) {
        return 0;
    }
    return cuMemcpyDtoH(sse, evaluator->output, output_elements * sizeof(float)) == CUDA_SUCCESS;
}

int
secant_sr_cubin_constant_optimizer_evaluator_create(
    size_t num_kernels,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinConstantOptimizerSSERecipe recipe = secant_cubin_constant_optimizer_sse_recipe_init();
    size_t input_elements;
    size_t constant_elements;
    size_t output_elements;
    int success = 0;

    if (evaluator == NULL || input == NULL || target == NULL || session == NULL || session->context == NULL ||
        num_kernels == 0u || num_input_columns == 0u || num_input_constants == 0u || num_settings == 0u ||
        population_size == 0u || num_rows == 0u || num_workers == 0u || num_streams == 0u ||
        !secant_sr_size_mul(num_input_columns, num_rows, &input_elements) ||
        !secant_sr_size_mul(population_size, num_input_constants, &constant_elements) ||
        !secant_sr_size_mul(population_size, num_settings, &output_elements) ||
        input_elements > SIZE_MAX / sizeof(float) || constant_elements > SIZE_MAX / sizeof(float) ||
        output_elements > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    evaluator->context = session->context;
    recipe.num_kernels = num_kernels;
    recipe.num_input_columns = num_input_columns;
    recipe.num_input_constants = num_input_constants;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (!secant_sr_cubin_evaluator_prepare(
            &recipe.header,
            num_workers,
            num_streams,
            session->compute_capability_major,
            session->compute_capability_minor,
            cache,
            evaluator)) {
        fprintf(stderr, "constant optimizer prepare failed\n");
        goto cleanup;
    }
    if (cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->constant_settings, constant_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->output, output_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->target, target, num_rows * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "constant optimizer device allocation or copy failed\n");
        goto cleanup;
    }
    evaluator->population_size = population_size;
    evaluator->num_inputs = num_input_columns;
    evaluator->num_rows = num_rows;
    evaluator->num_input_constants = num_input_constants;
    evaluator->num_settings = num_settings;
    evaluator->leaf_words_leading_dimension = num_input_constants;
    evaluator->constant_optimizer = 1;
    success = 1;

cleanup:
    if (!success) {
        secant_sr_cubin_evaluator_destroy(evaluator);
    }
    return success;
}

int
secant_sr_cubin_constant_optimizer_constants_upload(
    SecantSRCubinEvaluator* evaluator,
    const float* current_constants,
    size_t num_asts
) {
    size_t num_elements;

    return evaluator != NULL && evaluator->constant_optimizer && current_constants != NULL &&
        num_asts != 0u && num_asts <= evaluator->population_size &&
        secant_sr_size_mul(num_asts, evaluator->num_input_constants, &num_elements) &&
        num_elements <= SIZE_MAX / sizeof(float) &&
        cuMemcpyHtoD(
            evaluator->constant_settings, current_constants, num_elements * sizeof(float)) == CUDA_SUCCESS;
}

int
secant_sr_cubin_constant_optimizer_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uint64_t seed,
    uint64_t generation,
    uint64_t iteration,
    uint64_t ast_index_base,
    float perturbation_scale,
    float perturbation_decay,
    size_t num_iterations
) {
    SecantCubinConstantOptimizerSSERun run = secant_cubin_constant_optimizer_sse_run_init();
    size_t output_elements;

    if (evaluator == NULL || !evaluator->constant_optimizer || evaluator->runner == NULL ||
        asts == NULL || num_asts == 0u || num_iterations == 0u ||
        num_asts > evaluator->population_size || num_asts > UINT_MAX ||
        !isfinite(perturbation_scale) || perturbation_scale < 0.0f ||
        !isfinite(perturbation_decay) || perturbation_decay <= 0.0f || perturbation_decay > 1.0f ||
        !secant_sr_size_mul(num_asts, evaluator->num_settings, &output_elements)) {
        return 0;
    }
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.programs.routines.items = evaluator->routines;
    run.programs.routines.count = evaluator->num_routines;
    run.input.address = (uintptr_t)evaluator->input;
    run.input.num_elements = evaluator->num_inputs * evaluator->num_rows;
    run.input.leading_dimension = evaluator->num_rows;
    run.current_constants.address = (uintptr_t)evaluator->constant_settings;
    run.current_constants.num_elements = num_asts * evaluator->num_input_constants;
    run.current_constants.leading_dimension = evaluator->num_input_constants;
    run.target.address = (uintptr_t)evaluator->target;
    run.target.num_elements = evaluator->num_rows;
    run.target.leading_dimension = evaluator->num_rows;
    run.num_rows = evaluator->num_rows;
    run.num_settings = evaluator->num_settings;
    run.num_iterations = num_iterations;
    run.seed = seed;
    run.generation = generation;
    run.iteration = iteration;
    run.ast_index_base = ast_index_base;
    run.perturbation_scale = perturbation_scale;
    run.perturbation_decay = perturbation_decay;
    run.output.address = (uintptr_t)evaluator->output;
    run.output.num_elements = output_elements;
    run.output.leading_dimension = evaluator->num_settings;
    evaluator->last_stats = secant_runner_stats_init();
    return secant_cubin_runner_run_constant_optimizer_sse(
        evaluator->runner, &run, &evaluator->last_stats) == SECANT_SUCCESS;
}

int
secant_sr_cubin_constant_optimizer_constants_download(
    SecantSRCubinEvaluator* evaluator,
    float* current_constants,
    size_t num_asts
) {
    size_t num_elements;

    return evaluator != NULL && evaluator->constant_optimizer && current_constants != NULL &&
        num_asts != 0u && num_asts <= evaluator->population_size &&
        secant_sr_size_mul(num_asts, evaluator->num_input_constants, &num_elements) &&
        num_elements <= SIZE_MAX / sizeof(float) &&
        cuMemcpyDtoH(
            current_constants, evaluator->constant_settings, num_elements * sizeof(float)) == CUDA_SUCCESS;
}

int
secant_sr_cubin_packed_constant_optimizer_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinPackedConstantOptimizerSSERecipe recipe =
        secant_cubin_packed_constant_optimizer_sse_recipe_init();
    size_t input_elements;
    size_t output_elements;
    size_t best_leading_dimension;
    size_t best_elements;
    int success = 0;

    if (evaluator == NULL || input == NULL || target == NULL || session == NULL || session->context == NULL ||
        num_kernels == 0u || asts_per_kernel == 0u || num_input_columns == 0u || num_input_constants == 0u ||
        num_settings == 0u || population_size == 0u || num_rows == 0u || num_workers == 0u || num_streams == 0u ||
        !secant_sr_size_mul(num_input_columns, num_rows, &input_elements) ||
        !secant_sr_size_mul(population_size, num_settings, &output_elements) ||
        num_input_constants > (SIZE_MAX - 1u) / 4u ||
        (best_leading_dimension = 1u + 4u * num_input_constants) == 0u ||
        !secant_sr_size_mul(population_size, best_leading_dimension, &best_elements) ||
        input_elements > SIZE_MAX / sizeof(float) || output_elements > SIZE_MAX / sizeof(float) ||
        best_elements > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    evaluator->context = session->context;
    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_input_columns = num_input_columns;
    recipe.num_input_constants = num_input_constants;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (!secant_sr_cubin_evaluator_prepare(
            &recipe.header,
            num_workers,
            num_streams,
            session->compute_capability_major,
            session->compute_capability_minor,
            cache,
            evaluator)) {
        fprintf(stderr, "packed constant optimizer prepare failed\n");
        goto cleanup;
    }
    if (cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->output, output_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->optimizer_best, best_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->target, target, num_rows * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "packed constant optimizer device allocation or copy failed\n");
        goto cleanup;
    }
    evaluator->population_size = population_size;
    evaluator->num_inputs = num_input_columns;
    evaluator->num_rows = num_rows;
    evaluator->num_input_constants = num_input_constants;
    evaluator->num_settings = num_settings;
    evaluator->leaf_words_leading_dimension = best_leading_dimension;
    evaluator->packed_constant_optimizer = 1;
    success = 1;

cleanup:
    if (!success) {
        secant_sr_cubin_evaluator_destroy(evaluator);
    }
    return success;
}

int
secant_sr_cubin_packed_constant_optimizer_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* current_constants,
    float* current_constant_scales,
    float* current_constant_velocities,
    float* current_sse,
    uint64_t seed,
    uint64_t generation,
    uint64_t iteration,
    size_t num_iterations,
    float momentum,
    float scale_learning_rate,
    float scale_failure_decay,
    float minimum_scale,
    float maximum_scale
) {
    SecantCubinPackedConstantOptimizerSSERun run = secant_cubin_packed_constant_optimizer_sse_run_init();
    size_t state_elements;
    size_t output_elements;
    size_t best_elements;

    if (evaluator == NULL || !evaluator->packed_constant_optimizer || evaluator->runner == NULL || asts == NULL ||
        current_constants == NULL || current_constant_scales == NULL || current_constant_velocities == NULL ||
        current_sse == NULL || num_asts == 0u || num_asts > evaluator->population_size || num_iterations == 0u ||
        !secant_sr_size_mul(num_asts, evaluator->num_input_constants, &state_elements) ||
        !secant_sr_size_mul(num_asts, evaluator->num_settings, &output_elements) ||
        !secant_sr_size_mul(num_asts, evaluator->leaf_words_leading_dimension, &best_elements)) {
        return 0;
    }
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.programs.routines.items = evaluator->routines;
    run.programs.routines.count = evaluator->num_routines;
    run.programs.current_constants =
        (SecantHostMatrixF32){current_constants, state_elements, evaluator->num_input_constants};
    run.programs.current_constant_scales =
        (SecantHostMatrixF32){current_constant_scales, state_elements, evaluator->num_input_constants};
    run.input.address = (uintptr_t)evaluator->input;
    run.input.num_elements = evaluator->num_inputs * evaluator->num_rows;
    run.input.leading_dimension = evaluator->num_rows;
    run.target.address = (uintptr_t)evaluator->target;
    run.target.num_elements = evaluator->num_rows;
    run.target.leading_dimension = evaluator->num_rows;
    run.num_rows = evaluator->num_rows;
    run.num_settings = evaluator->num_settings;
    run.num_iterations = num_iterations;
    run.seed = seed;
    run.generation = generation;
    run.iteration = iteration;
    run.update_mode = SECANT_CONSTANT_OPTIMIZER_UPDATE_WINNER;
    run.num_elites = 1u;
    run.current_constant_velocities =
        (SecantHostMatrixF32){current_constant_velocities, state_elements, evaluator->num_input_constants};
    run.current_sse = (SecantHostSpanF32){current_sse, num_asts};
    run.momentum = momentum;
    run.scale_learning_rate = scale_learning_rate;
    run.scale_failure_decay = scale_failure_decay;
    run.minimum_scale = minimum_scale;
    run.maximum_scale = maximum_scale;
    run.sse.address = (uintptr_t)evaluator->output;
    run.sse.num_elements = output_elements;
    run.sse.leading_dimension = evaluator->num_settings;
    run.best.address = (uintptr_t)evaluator->optimizer_best;
    run.best.num_elements = best_elements;
    run.best.leading_dimension = evaluator->leaf_words_leading_dimension;
    evaluator->last_stats = secant_runner_stats_init();
    return secant_cubin_runner_run_packed_constant_optimizer_sse(
        evaluator->runner, &run, &evaluator->last_stats) == SECANT_SUCCESS;
}

void
secant_sr_cubin_lm_evaluator_destroy(SecantSRCubinLMEvaluator* evaluator) {
    if (evaluator == NULL) {
        return;
    }
    if (evaluator->runner != NULL) {
        (void)secant_cubin_lm_optimizer_runner_destroy(evaluator->runner);
    }
    if (evaluator->status != 0u) {
        (void)cuMemFree(evaluator->status);
    }
    if (evaluator->predicted_reduction != 0u) {
        (void)cuMemFree(evaluator->predicted_reduction);
    }
    if (evaluator->damping != 0u) {
        (void)cuMemFree(evaluator->damping);
    }
    if (evaluator->accepted_statistics != 0u) {
        (void)cuMemFree(evaluator->accepted_statistics);
    }
    if (evaluator->evaluated_statistics != 0u) {
        (void)cuMemFree(evaluator->evaluated_statistics);
    }
    if (evaluator->proposal_constants != 0u) {
        (void)cuMemFree(evaluator->proposal_constants);
    }
    if (evaluator->current_constants != 0u) {
        (void)cuMemFree(evaluator->current_constants);
    }
    if (evaluator->leaf_words != 0u) {
        (void)cuMemFree(evaluator->leaf_words);
    }
    if (evaluator->leaf_masks != 0u) {
        (void)cuMemFree(evaluator->leaf_masks);
    }
    if (evaluator->target != 0u) {
        (void)cuMemFree(evaluator->target);
    }
    if (evaluator->input != 0u) {
        (void)cuMemFree(evaluator->input);
    }
    memset(evaluator, 0, sizeof(*evaluator));
}

int
secant_sr_cubin_lm_evaluator_create(
    size_t ast_capacity,
    size_t num_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_streams,
    const float* input,
    const float* target,
    size_t num_settings,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinLMEvaluator* evaluator
) {
    static const char* const nvrtc_options[] = {"--restrict", "--use_fast_math"};
    size_t input_elements;
    size_t ast_settings;
    size_t constant_elements;
    size_t statistic_rows;
    size_t statistic_elements;
    int success = 0;

    if (evaluator == NULL || session == NULL || session->context == NULL || input == NULL || target == NULL ||
        ast_capacity == 0u || num_input_columns == 0u || num_settings == 0u || num_rows == 0u ||
        tile_rows == 0u || threads_per_block == 0u || patch_capacity_instructions == 0u || num_streams == 0u ||
        !secant_sr_size_mul(num_input_columns, num_rows, &input_elements) ||
        !secant_sr_size_mul(ast_capacity, num_settings, &ast_settings) ||
        !secant_sr_size_mul(ast_settings, SECANT_CUDA_LM_OPTIMIZER_PARAMETERS, &constant_elements) ||
        !secant_sr_size_mul(ast_capacity, SECANT_CUDA_LM_OPTIMIZER_STATISTICS, &statistic_rows) ||
        !secant_sr_size_mul(statistic_rows, num_settings, &statistic_elements) ||
        input_elements > SIZE_MAX / sizeof(float) || ast_settings > SIZE_MAX / sizeof(uint32_t) ||
        constant_elements > SIZE_MAX / sizeof(float) || statistic_elements > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    evaluator->context = session->context;
    if (secant_cubin_lm_optimizer_runner_create(
            ast_capacity,
            num_input_columns,
            num_input_columns,
            SECANT_CUDA_LM_OPTIMIZER_PARAMETERS,
            tile_rows,
            threads_per_block,
            patch_capacity_instructions,
            (uint32_t)session->compute_capability_major,
            (uint32_t)session->compute_capability_minor,
            nvrtc_options,
            sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
            32u * 1024u * 1024u,
            num_streams,
            &evaluator->runner) != SECANT_SUCCESS) {
        fprintf(stderr, "LM runner creation failed\n");
        goto cleanup;
    }
    if (cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->leaf_masks, ast_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->leaf_words, constant_elements * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->current_constants, constant_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->proposal_constants, constant_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->evaluated_statistics, statistic_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->accepted_statistics, statistic_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->damping, ast_settings * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->predicted_reduction, ast_settings * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&evaluator->status, ast_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->target, target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
        cuMemsetD8(evaluator->leaf_masks, 0u, ast_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemsetD8(evaluator->leaf_words, 0u, constant_elements * sizeof(uint32_t)) != CUDA_SUCCESS) {
        fprintf(stderr, "LM device allocation or copy failed\n");
        goto cleanup;
    }
    evaluator->ast_capacity = ast_capacity;
    evaluator->num_input_columns = num_input_columns;
    evaluator->num_rows = num_rows;
    evaluator->num_settings = num_settings;
    success = 1;

cleanup:
    if (!success) {
        secant_sr_cubin_lm_evaluator_destroy(evaluator);
    }
    return success;
}

int
secant_sr_cubin_lm_evaluator_routines_set(
    SecantSRCubinLMEvaluator* evaluator,
    const SecantAstInstruction* const* routines,
    size_t num_routines
) {
    if (evaluator == NULL || evaluator->runner == NULL ||
        (num_routines != 0u && routines == NULL)) {
        return 0;
    }
    evaluator->routines = routines;
    evaluator->num_routines = num_routines;
    return 1;
}

int
secant_sr_cubin_lm_evaluator_run_mixed(
    SecantSRCubinLMEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    const float* initial_constants,
    size_t settings_per_cta,
    size_t num_iterations,
    float initial_damping,
    float* optimized_constants,
    float* accepted_sse
) {
    SecantCUDALMOptimizerRun run = secant_cuda_lm_optimizer_run_init();
    size_t ast_settings;
    size_t constant_elements;
    size_t statistic_rows;
    size_t statistic_elements;
    uint32_t damping_bits;
    SecantResult lm_result;

    if (evaluator == NULL || evaluator->runner == NULL || asts == NULL || leaf_masks == NULL ||
        leaf_words == NULL || initial_constants == NULL || optimized_constants == NULL ||
        accepted_sse == NULL || num_asts == 0u ||
        num_asts > evaluator->ast_capacity || settings_per_cta == 0u ||
        settings_per_cta > evaluator->num_settings || num_iterations == 0u ||
        !isfinite(initial_damping) || initial_damping <= 0.0f ||
        !secant_sr_size_mul(num_asts, evaluator->num_settings, &ast_settings) ||
        !secant_sr_size_mul(ast_settings, SECANT_CUDA_LM_OPTIMIZER_PARAMETERS, &constant_elements) ||
        !secant_sr_size_mul(num_asts, SECANT_CUDA_LM_OPTIMIZER_STATISTICS, &statistic_rows) ||
        !secant_sr_size_mul(statistic_rows, evaluator->num_settings, &statistic_elements)) {
        return 0;
    }
    memcpy(&damping_bits, &initial_damping, sizeof(damping_bits));
    if (cuMemcpyHtoD(
            evaluator->current_constants,
            initial_constants,
            constant_elements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(
            evaluator->leaf_masks,
            leaf_masks,
            ast_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(
            evaluator->leaf_words,
            leaf_words,
            constant_elements * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemsetD32(evaluator->damping, damping_bits, ast_settings) != CUDA_SUCCESS) {
        return 0;
    }
    run.routines.items = evaluator->routines;
    run.routines.count = evaluator->num_routines;
    run.asts.items = asts;
    run.asts.count = num_asts;
    run.input = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->input,
        evaluator->num_input_columns * evaluator->num_rows,
        evaluator->num_rows};
    run.num_input_columns = evaluator->num_input_columns;
    run.leaf_masks = (SecantDeviceMatrixU32){
        (uintptr_t)evaluator->leaf_masks,
        evaluator->ast_capacity * evaluator->num_settings,
        evaluator->num_settings};
    run.leaf_words = (SecantDeviceMatrixU32){
        (uintptr_t)evaluator->leaf_words,
        evaluator->ast_capacity * evaluator->num_settings * SECANT_CUDA_LM_OPTIMIZER_PARAMETERS,
        SECANT_CUDA_LM_OPTIMIZER_PARAMETERS};
    run.target = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->target,
        evaluator->num_rows,
        evaluator->num_rows};
    run.num_rows = evaluator->num_rows;
    run.num_settings = evaluator->num_settings;
    run.settings_per_cta = settings_per_cta;
    run.num_iterations = num_iterations;
    run.current_constants = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->current_constants,
        evaluator->ast_capacity * evaluator->num_settings * SECANT_CUDA_LM_OPTIMIZER_PARAMETERS,
        SECANT_CUDA_LM_OPTIMIZER_PARAMETERS};
    run.proposal_constants = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->proposal_constants,
        evaluator->ast_capacity * evaluator->num_settings * SECANT_CUDA_LM_OPTIMIZER_PARAMETERS,
        SECANT_CUDA_LM_OPTIMIZER_PARAMETERS};
    run.evaluated_statistics = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->evaluated_statistics,
        evaluator->ast_capacity * SECANT_CUDA_LM_OPTIMIZER_STATISTICS * evaluator->num_settings,
        evaluator->num_settings};
    run.accepted_statistics = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->accepted_statistics,
        evaluator->ast_capacity * SECANT_CUDA_LM_OPTIMIZER_STATISTICS * evaluator->num_settings,
        evaluator->num_settings};
    run.damping = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->damping,
        evaluator->ast_capacity * evaluator->num_settings,
        evaluator->num_settings};
    run.predicted_reduction = (SecantDeviceMatrixF32){
        (uintptr_t)evaluator->predicted_reduction,
        evaluator->ast_capacity * evaluator->num_settings,
        evaluator->num_settings};
    run.status = (SecantDeviceMatrixU32){
        (uintptr_t)evaluator->status,
        evaluator->ast_capacity * evaluator->num_settings,
        evaluator->num_settings};
    evaluator->last_stats = secant_runner_stats_init();
    lm_result = secant_cubin_lm_optimizer_runner_run(
        evaluator->runner, &run, &evaluator->last_stats);
    if (lm_result != SECANT_SUCCESS) {
        fprintf(stderr, "LM runner failed: %s\n", secant_result_to_string(lm_result));
        return 0;
    }
    if (cuMemcpyDtoH(
            optimized_constants,
            evaluator->current_constants,
            constant_elements * sizeof(float)) != CUDA_SUCCESS) {
        return 0;
    }
    {
        CUDA_MEMCPY2D copy;

        memset(&copy, 0, sizeof(copy));
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = evaluator->accepted_statistics;
        copy.srcPitch = SECANT_CUDA_LM_OPTIMIZER_STATISTICS *
            evaluator->num_settings * sizeof(float);
        copy.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy.dstHost = accepted_sse;
        copy.dstPitch = evaluator->num_settings * sizeof(float);
        copy.WidthInBytes = evaluator->num_settings * sizeof(float);
        copy.Height = num_asts;
        if (cuMemcpy2D(&copy) != CUDA_SUCCESS) {
            return 0;
        }
    }
    (void)statistic_elements;
    return 1;
}

static int
secant_sr_cubin_dynamic_leaf_evaluator_create_inner(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    size_t num_inputs,
    size_t num_static_input_columns,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    CUcontext context,
    int major,
    int minor,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    size_t input_elements;
    size_t leaf_word_elements;
    size_t output_elements;
    size_t asts_per_module;

    if (population_size == 0u || num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u ||
        (num_static_input_columns != 0u && num_static_input_columns != num_inputs) ||
        num_dynamic_leaves == 0u || num_dynamic_leaves > SECANT_AST_MAX_DYNAMIC_LEAVES || num_settings == 0u ||
        leaf_words_leading_dimension < num_dynamic_leaves ||
        num_rows == 0u || !secant_sr_size_mul(num_kernels, asts_per_kernel, &asts_per_module) ||
        !secant_sr_size_mul(num_inputs, num_rows, &input_elements) ||
        !secant_sr_size_mul(num_settings, leaf_words_leading_dimension, &leaf_word_elements) ||
        !secant_sr_size_mul(population_size, num_settings, &output_elements)) {
        return 0;
    }
    evaluator->context = context;
    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_input_columns = num_inputs;
    recipe.num_static_input_columns = num_static_input_columns;
    recipe.num_dynamic_leaves = num_dynamic_leaves;
    recipe.num_targets = 1u;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    if (!secant_sr_cubin_evaluator_prepare(
            &recipe.header, num_workers, num_streams, major, minor, cache, evaluator)) {
        return 0;
    }
    {
        const double device_prepare_begin = secant_sr_cubin_seconds_get();

        if (input_elements > SIZE_MAX / sizeof(float) || leaf_word_elements > SIZE_MAX / sizeof(uint32_t) ||
            output_elements > SIZE_MAX / sizeof(float) || num_settings > SIZE_MAX / sizeof(uint32_t) ||
            cuMemAlloc(&evaluator->input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->leaf_masks, num_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->leaf_words, leaf_word_elements * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemAlloc(&evaluator->output, output_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->input, input, input_elements * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->target, target, num_rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->leaf_masks, leaf_masks, num_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(evaluator->leaf_words, leaf_words, leaf_word_elements * sizeof(uint32_t)) != CUDA_SUCCESS) {
            return 0;
        }
        evaluator->template_stats.device_prepare_seconds += secant_sr_cubin_seconds_get() - device_prepare_begin;
    }
    evaluator->population_size = population_size;
    evaluator->num_inputs = num_inputs;
    evaluator->num_static_input_columns = num_static_input_columns;
    evaluator->num_rows = num_rows;
    evaluator->num_dynamic_leaves = num_dynamic_leaves;
    evaluator->num_settings = num_settings;
    evaluator->leaf_words_leading_dimension = leaf_words_leading_dimension;
    evaluator->dynamic_leaf = 1;
    return 1;
}

int
secant_sr_cubin_dynamic_leaf_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    size_t num_inputs,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    if (evaluator == NULL || input == NULL || target == NULL || leaf_masks == NULL || leaf_words == NULL ||
        session == NULL || session->context == NULL || num_workers == 0u || num_streams == 0u) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    if (!secant_sr_cubin_dynamic_leaf_evaluator_create_inner(
        num_kernels,
        asts_per_kernel,
        num_dynamic_leaves,
        tile_rows,
        threads_per_block,
        patch_capacity_instructions,
        num_workers,
        num_streams,
        input,
        num_inputs,
        0u,
        target,
        leaf_masks,
        leaf_words,
        leaf_words_leading_dimension,
        num_settings,
        population_size,
        num_rows,
        session->context,
        session->compute_capability_major,
        session->compute_capability_minor,
        cache,
        evaluator)) {
        secant_sr_cubin_evaluator_destroy(evaluator);
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_mixed_dynamic_leaf_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    size_t num_inputs,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t population_size,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinEvaluator* evaluator
) {
    if (evaluator == NULL || input == NULL || target == NULL || leaf_masks == NULL || leaf_words == NULL ||
        session == NULL || session->context == NULL || num_workers == 0u || num_streams == 0u) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    if (!secant_sr_cubin_dynamic_leaf_evaluator_create_inner(
            num_kernels,
            asts_per_kernel,
            num_dynamic_leaves,
            tile_rows,
            threads_per_block,
            patch_capacity_instructions,
            num_workers,
            num_streams,
            input,
            num_inputs,
            num_inputs,
            target,
            leaf_masks,
            leaf_words,
            leaf_words_leading_dimension,
            num_settings,
            population_size,
            num_rows,
            session->context,
            session->compute_capability_major,
            session->compute_capability_minor,
            cache,
            evaluator)) {
        secant_sr_cubin_evaluator_destroy(evaluator);
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_dynamic_leaf_settings_update(
    SecantSRCubinEvaluator* evaluator,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_settings
) {
    size_t leaf_word_elements;

    if (evaluator == NULL || !evaluator->dynamic_leaf || leaf_masks == NULL || leaf_words == NULL ||
        num_settings != evaluator->num_settings ||
        leaf_words_leading_dimension != evaluator->leaf_words_leading_dimension ||
        !secant_sr_size_mul(num_settings, leaf_words_leading_dimension, &leaf_word_elements) ||
        num_settings > SIZE_MAX / sizeof(uint32_t) || leaf_word_elements > SIZE_MAX / sizeof(uint32_t)) {
        return 0;
    }
    if (cuMemcpyHtoD(evaluator->leaf_masks, leaf_masks, num_settings * sizeof(uint32_t)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(evaluator->leaf_words, leaf_words, leaf_word_elements * sizeof(uint32_t)) != CUDA_SUCCESS) {
        fprintf(stderr, "dynamic-leaf setting update failed\n");
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_dynamic_leaf_evaluator_run_device(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts
) {
    SecantCubinDynamicLeafSSERun run = secant_cubin_dynamic_leaf_sse_run_init();
    SecantResult result;
    size_t output_elements;

    if (evaluator == NULL || evaluator->runner == NULL || !evaluator->dynamic_leaf || asts == NULL ||
        num_asts == 0u || num_asts > evaluator->population_size ||
        !secant_sr_size_mul(num_asts, evaluator->num_settings, &output_elements)) {
        return 0;
    }
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.programs.routines.items = evaluator->routines;
    run.programs.routines.count = evaluator->num_routines;
    run.input.address = (uintptr_t)evaluator->input;
    run.input.num_elements = evaluator->num_inputs * evaluator->num_rows;
    run.input.leading_dimension = evaluator->num_rows;
    run.num_input_columns = evaluator->num_inputs;
    run.leaf_masks.address = (uintptr_t)evaluator->leaf_masks;
    run.leaf_masks.num_elements = evaluator->num_settings;
    run.leaf_words.address = (uintptr_t)evaluator->leaf_words;
    run.leaf_words.num_elements = evaluator->num_settings * evaluator->leaf_words_leading_dimension;
    run.leaf_words.leading_dimension = evaluator->leaf_words_leading_dimension;
    run.targets.address = (uintptr_t)evaluator->target;
    run.targets.num_elements = evaluator->num_rows;
    run.targets.leading_dimension = evaluator->num_rows;
    run.num_rows = evaluator->num_rows;
    run.num_settings = evaluator->num_settings;
    run.num_targets = 1u;
    run.output.address = (uintptr_t)evaluator->output;
    run.output.num_elements = output_elements;
    run.output.leading_dimension = evaluator->num_settings;
    evaluator->last_stats = secant_runner_stats_init();
    result = secant_cubin_runner_run_dynamic_leaf_sse(evaluator->runner, &run, &evaluator->last_stats);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr,
            "dynamic-leaf runner failed: %s modules=%zu loaded=%zu compile=%.9f load=%.9f runtime=%.9f\n",
            secant_result_to_string(result),
            evaluator->last_stats.num_modules,
            evaluator->last_stats.modules_loaded,
            evaluator->last_stats.compile_window_seconds,
            evaluator->last_stats.module_load_seconds,
            evaluator->last_stats.runtime_seconds);
        return 0;
    }
    return result == SECANT_SUCCESS;
}

int
secant_sr_cubin_dynamic_leaf_evaluator_run(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    float* sse
) {
    size_t output_elements;

    if (sse == NULL || !secant_sr_size_mul(num_asts, evaluator != NULL ? evaluator->num_settings : 0u, &output_elements) ||
        !secant_sr_cubin_dynamic_leaf_evaluator_run_device(evaluator, asts, num_asts)) {
        return 0;
    }
    if (cuMemcpyDtoH(sse, evaluator->output, output_elements * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "dynamic-leaf output copy failed\n");
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_staged_evaluator_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_dynamic_leaves,
    size_t static_tile_rows,
    size_t dynamic_tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    size_t num_workers,
    size_t num_streams,
    const float* input,
    const float* target,
    const uint32_t* leaf_masks,
    const uint32_t* leaf_words,
    size_t leaf_words_leading_dimension,
    size_t num_leaf_settings,
    const float* constant_settings,
    size_t num_input_constants,
    size_t constant_settings_leading_dimension,
    size_t num_constant_settings,
    const float* materialize_input,
    size_t materialize_num_rows,
    size_t population_size,
    size_t dynamic_ast_capacity,
    size_t num_rows,
    const SecantSRCudaSession* session,
    SecantSRCubinCache* cache,
    SecantSRCubinStagedEvaluator* evaluator
) {
    if (evaluator == NULL || input == NULL || target == NULL || leaf_masks == NULL || leaf_words == NULL ||
        (num_constant_settings != 0u && constant_settings == NULL) ||
        (num_constant_settings == 0u && constant_settings != NULL) ||
        ((materialize_input == NULL) != (materialize_num_rows == 0u)) || dynamic_ast_capacity == 0u ||
        session == NULL || session->context == NULL ||
        num_workers == 0u || num_streams == 0u) {
        return 0;
    }
    memset(evaluator, 0, sizeof(*evaluator));
    if (!secant_sr_cubin_evaluator_create_inner(
            num_kernels,
            asts_per_kernel,
            num_inputs,
            static_tile_rows,
            threads_per_block,
            patch_capacity_instructions,
            num_workers,
            num_streams,
            input,
            target,
            population_size,
            num_rows,
            session->context,
            session->compute_capability_major,
            session->compute_capability_minor,
            cache,
            &evaluator->static_sse) ||
        !secant_sr_cubin_dynamic_leaf_evaluator_create_inner(
            num_kernels,
            asts_per_kernel,
            num_dynamic_leaves,
            dynamic_tile_rows,
            threads_per_block,
            patch_capacity_instructions,
            num_workers,
            num_streams,
            input,
            num_inputs,
            0u,
            target,
            leaf_masks,
            leaf_words,
            leaf_words_leading_dimension,
            num_leaf_settings,
            dynamic_ast_capacity,
            num_rows,
            session->context,
            session->compute_capability_major,
            session->compute_capability_minor,
            cache,
            &evaluator->dynamic_leaf) ||
        (num_constant_settings != 0u &&
         !secant_sr_cubin_dynamic_constant_evaluator_create_inner(
             num_kernels,
             asts_per_kernel,
             num_inputs,
             num_input_constants,
             dynamic_tile_rows,
             threads_per_block,
             patch_capacity_instructions,
             num_workers,
             num_streams,
             input,
             target,
             constant_settings,
             constant_settings_leading_dimension,
             num_constant_settings,
             dynamic_ast_capacity,
             num_rows,
             session->context,
             session->compute_capability_major,
             session->compute_capability_minor,
             cache,
             &evaluator->dynamic_constant)) ||
        (materialize_input != NULL &&
         !secant_sr_cubin_materialize_evaluator_create_inner(
             num_inputs,
             patch_capacity_instructions,
             num_workers,
             num_streams,
             materialize_input,
             materialize_num_rows,
             session->context,
             session->compute_capability_major,
             session->compute_capability_minor,
             cache,
             &evaluator->materialize))) {
        secant_sr_cubin_staged_evaluator_destroy(evaluator);
        return 0;
    }
    return 1;
}

int
secant_sr_cubin_materialize_evaluator_run_one(
    SecantSRCubinEvaluator* evaluator,
    const SecantAstInstruction* ast,
    float* output
) {
    const SecantAstInstruction* asts[] = {ast};
    SecantCubinMaterializeRun run = secant_cubin_materialize_run_init();
    SecantResult result;

    if (evaluator == NULL || evaluator->runner == NULL || !evaluator->materialize || ast == NULL || output == NULL) {
        return 0;
    }
    run.programs.asts.items = asts;
    run.programs.asts.count = 1u;
    run.programs.routines.items = evaluator->routines;
    run.programs.routines.count = evaluator->num_routines;
    run.input.address = (uintptr_t)evaluator->input;
    run.input.num_elements = evaluator->num_inputs * evaluator->num_rows;
    run.input.leading_dimension = evaluator->num_rows;
    run.num_rows = evaluator->num_rows;
    run.output.address = (uintptr_t)evaluator->output;
    run.output.num_elements = evaluator->num_rows;
    run.output.leading_dimension = evaluator->num_rows;
    run.output_module_stride = 0u;
    evaluator->last_stats = secant_runner_stats_init();
    result = secant_cubin_runner_run_materialize(evaluator->runner, &run, &evaluator->last_stats);
    return result == SECANT_SUCCESS &&
        cuMemcpyDtoH(output, evaluator->output, evaluator->num_rows * sizeof(float)) == CUDA_SUCCESS;
}

void
secant_sr_cubin_staged_evaluator_destroy(SecantSRCubinStagedEvaluator* evaluator) {
    if (evaluator == NULL) {
        return;
    }
    secant_sr_cubin_evaluator_destroy(&evaluator->materialize);
    secant_sr_cubin_evaluator_destroy(&evaluator->constant_optimizer);
    secant_sr_cubin_evaluator_destroy(&evaluator->dynamic_constant);
    secant_sr_cubin_evaluator_destroy(&evaluator->dynamic_leaf);
    secant_sr_cubin_evaluator_destroy(&evaluator->static_sse);
    memset(evaluator, 0, sizeof(*evaluator));
}

void
secant_sr_cubin_evaluator_destroy(SecantSRCubinEvaluator* evaluator) {
    if (evaluator == NULL) {
        return;
    }
    if (evaluator->runner != NULL) {
        (void)secant_cubin_runner_destroy(evaluator->runner);
    }
    if (evaluator->output != 0u) {
        (void)cuMemFree(evaluator->output);
    }
    if (evaluator->optimizer_best != 0u) {
        (void)cuMemFree(evaluator->optimizer_best);
    }
    if (evaluator->leaf_words != 0u) {
        (void)cuMemFree(evaluator->leaf_words);
    }
    if (evaluator->constant_settings != 0u) {
        (void)cuMemFree(evaluator->constant_settings);
    }
    if (evaluator->leaf_masks != 0u) {
        (void)cuMemFree(evaluator->leaf_masks);
    }
    if (evaluator->target != 0u) {
        (void)cuMemFree(evaluator->target);
    }
    if (evaluator->input != 0u) {
        (void)cuMemFree(evaluator->input);
    }
    free(evaluator->plan_storage);
    free(evaluator->cubin);
    free(evaluator->source);
    memset(evaluator, 0, sizeof(*evaluator));
}
