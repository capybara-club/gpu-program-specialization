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
#ifndef SECANT_BENCH_SECANT_API_H_INCLUDED
#define SECANT_BENCH_SECANT_API_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>

static inline SecantResult
secant_bench_cubin_source_generate(
    const void* concrete_recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
    const SecantCubinRecipeHeader* recipe = (const SecantCubinRecipeHeader*)concrete_recipe;
    SecantResult result;

    if (required_size_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if (output == NULL) {
        return secant_cubin_source_size(recipe, required_size_ret);
    }
    result = secant_cubin_source_write(recipe, output, output_size);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    return secant_cubin_source_size(recipe, required_size_ret);
}

static inline SecantResult
secant_bench_cubin_inspect(
    const void* concrete_recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantCubinPlan** plan_ret
) {
    const SecantCubinRecipeHeader* recipe = (const SecantCubinRecipeHeader*)concrete_recipe;
    SecantResult result;

    if (required_plan_storage_ret == NULL || plan_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *plan_ret = NULL;
    result = secant_cubin_plan_storage_size(recipe, cubin, cubin_size, required_plan_storage_ret);
    if (result != SECANT_SUCCESS || plan_storage == NULL) {
        return result;
    }
    return secant_cubin_plan_init(
        recipe,
        cubin,
        cubin_size,
        plan_storage,
        plan_storage_size,
        plan_ret);
}

static inline SecantResult
secant_bench_cubin_specialize_into(
    const SecantCubinPlan* plan,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    void* cubin,
    size_t cubin_size
) {
    SecantAstProgramSet programs = {0};

    programs.routines.items = routines;
    programs.routines.count = num_routines;
    programs.asts.items = asts;
    programs.asts.count = num_asts;
    return secant_cubin_specialize_into(plan, &programs, cubin, cubin_size);
}

static inline SecantResult
secant_bench_cpu_materialize_run(
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_rows,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    SecantCpuMaterializeRun run = secant_cpu_materialize_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.num_inputs = num_inputs;
    run.input.data = input;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.num_rows = num_rows;
    run.output.data = output;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cpu_run(&run.header);
}

static inline SecantResult
secant_bench_cpu_sse_run(
    size_t num_inputs,
    size_t num_targets,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    SecantCpuSSERun run = secant_cpu_sse_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.num_inputs = num_inputs;
    run.num_targets = num_targets;
    run.input.data = input;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.targets.data = targets;
    run.targets.num_elements = targets_num_elements;
    run.targets.leading_dimension = targets_leading_dimension;
    run.num_rows = num_rows;
    run.output.data = output;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cpu_run(&run.header);
}

static inline SecantResult
secant_bench_cpu_dynamic_constant_sse_run(
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* constant_settings,
    size_t constant_settings_num_elements,
    size_t constants_leading_dimension,
    size_t num_settings,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    SecantCpuDynamicConstantSSERun run = secant_cpu_dynamic_constant_sse_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.num_input_columns = num_input_columns;
    run.num_input_constants = num_input_constants;
    run.num_targets = num_targets;
    run.input.data = input;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.constant_settings.data = constant_settings;
    run.constant_settings.num_elements = constant_settings_num_elements;
    run.constant_settings.leading_dimension = constants_leading_dimension;
    run.targets.data = targets;
    run.targets.num_elements = targets_num_elements;
    run.targets.leading_dimension = targets_leading_dimension;
    run.num_rows = num_rows;
    run.num_settings = num_settings;
    run.output.data = output;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cpu_run(&run.header);
}

static inline SecantResult
secant_bench_cpu_dynamic_leaf_sse_run(
    size_t num_dynamic_leaves,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_targets,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const uint32_t* leaf_masks,
    size_t leaf_masks_num_elements,
    const uint32_t* leaf_words,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    const float* targets,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    float* output,
    size_t output_num_elements,
    size_t output_leading_dimension
) {
    SecantCpuDynamicLeafSSERun run = secant_cpu_dynamic_leaf_sse_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.num_dynamic_leaves = num_dynamic_leaves;
    run.num_input_columns = num_input_columns;
    run.num_static_input_columns = num_static_input_columns;
    run.num_targets = num_targets;
    run.input.data = input;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.leaf_masks.data = leaf_masks;
    run.leaf_masks.num_elements = leaf_masks_num_elements;
    run.leaf_words.data = leaf_words;
    run.leaf_words.num_elements = leaf_words_num_elements;
    run.leaf_words.leading_dimension = leaf_words_leading_dimension;
    run.targets.data = targets;
    run.targets.num_elements = targets_num_elements;
    run.targets.leading_dimension = targets_leading_dimension;
    run.num_rows = num_rows;
    run.num_settings = num_settings;
    run.output.data = output;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cpu_run(&run.header);
}

static inline SecantResult
secant_bench_cubin_runner_create(
    const SecantCubinPlan* plan,
    const void* cubin,
    size_t cubin_size,
    size_t num_workers,
    size_t num_streams,
    SecantCubinRunner* runner_ret
) {
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();

    options.num_workers = num_workers;
    options.num_streams = num_streams;
    return secant_cubin_runner_create(plan, cubin, cubin_size, &options, runner_ret);
}

static inline SecantResult
secant_bench_cubin_materialize_runner_run(
    SecantCubinRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_address,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_rows,
    uintptr_t output_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    size_t output_module_stride,
    SecantRunnerStats* stats_ret
) {
    SecantCubinMaterializeRun run = secant_cubin_materialize_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.input.address = input_address;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.num_rows = num_rows;
    run.output.address = output_address;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    run.output_module_stride = output_module_stride;
    return secant_cubin_runner_run(runner, &run.header, stats_ret);
}

static inline SecantResult
secant_bench_cubin_sse_runner_run(
    SecantCubinRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_address,
    size_t input_num_elements,
    size_t input_leading_dimension,
    size_t num_targets,
    uintptr_t targets_address,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    uintptr_t output_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    SecantRunnerStats* stats_ret
) {
    SecantCubinSSERun run = secant_cubin_sse_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.input.address = input_address;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.targets.address = targets_address;
    run.targets.num_elements = targets_num_elements;
    run.targets.leading_dimension = targets_leading_dimension;
    run.num_rows = num_rows;
    run.num_targets = num_targets;
    run.output.address = output_address;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cubin_runner_run(runner, &run.header, stats_ret);
}

static inline SecantResult
secant_bench_cubin_dynamic_constant_sse_runner_run(
    SecantCubinRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_address,
    size_t input_num_elements,
    size_t input_leading_dimension,
    uintptr_t settings_address,
    size_t settings_num_elements,
    size_t settings_leading_dimension,
    size_t num_settings,
    size_t num_targets,
    uintptr_t targets_address,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    uintptr_t output_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    SecantRunnerStats* stats_ret
) {
    SecantCubinDynamicConstantSSERun run = secant_cubin_dynamic_constant_sse_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.input.address = input_address;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.constant_settings.address = settings_address;
    run.constant_settings.num_elements = settings_num_elements;
    run.constant_settings.leading_dimension = settings_leading_dimension;
    run.targets.address = targets_address;
    run.targets.num_elements = targets_num_elements;
    run.targets.leading_dimension = targets_leading_dimension;
    run.num_rows = num_rows;
    run.num_settings = num_settings;
    run.num_targets = num_targets;
    run.output.address = output_address;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cubin_runner_run(runner, &run.header, stats_ret);
}

static inline SecantResult
secant_bench_cubin_dynamic_leaf_sse_runner_run(
    SecantCubinRunner runner,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    uintptr_t input_address,
    size_t input_num_elements,
    size_t num_input_columns,
    size_t input_leading_dimension,
    uintptr_t leaf_masks_address,
    size_t leaf_masks_num_elements,
    uintptr_t leaf_words_address,
    size_t leaf_words_num_elements,
    size_t leaf_words_leading_dimension,
    size_t num_settings,
    size_t num_targets,
    uintptr_t targets_address,
    size_t targets_num_elements,
    size_t targets_leading_dimension,
    size_t num_rows,
    uintptr_t output_address,
    size_t output_num_elements,
    size_t output_leading_dimension,
    SecantRunnerStats* stats_ret
) {
    SecantCubinDynamicLeafSSERun run = secant_cubin_dynamic_leaf_sse_run_init();

    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = num_asts;
    run.input.address = input_address;
    run.input.num_elements = input_num_elements;
    run.input.leading_dimension = input_leading_dimension;
    run.num_input_columns = num_input_columns;
    run.leaf_masks.address = leaf_masks_address;
    run.leaf_masks.num_elements = leaf_masks_num_elements;
    run.leaf_words.address = leaf_words_address;
    run.leaf_words.num_elements = leaf_words_num_elements;
    run.leaf_words.leading_dimension = leaf_words_leading_dimension;
    run.targets.address = targets_address;
    run.targets.num_elements = targets_num_elements;
    run.targets.leading_dimension = targets_leading_dimension;
    run.num_rows = num_rows;
    run.num_settings = num_settings;
    run.num_targets = num_targets;
    run.output.address = output_address;
    run.output.num_elements = output_num_elements;
    run.output.leading_dimension = output_leading_dimension;
    return secant_cubin_runner_run(runner, &run.header, stats_ret);
}

#endif
