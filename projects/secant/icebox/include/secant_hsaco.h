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
#ifndef SECANT_HSACO_H_INCLUDED
#define SECANT_HSACO_H_INCLUDED

#include "secant.h"

#include <stddef.h>

#ifdef __cplusplus
#define SECANT_HSACO_DEC extern "C"
#else
#define SECANT_HSACO_DEC extern
#endif

/** Parameters shared by materialize source generation and HSACO inspection. */
typedef struct SecantHsacoMaterializeRecipe {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t patch_capacity_instructions;
} SecantHsacoMaterializeRecipe;

/** Parameters shared by SSE source generation and HSACO inspection. */
typedef struct SecantHsacoSSERecipe {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
    SecantSSEReductionMode reduction_mode;
} SecantHsacoSSERecipe;

/** Parameters shared by dynamic-constant SSE generation and inspection. */
typedef struct SecantHsacoDynamicConstantSSERecipe {
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_input_columns;
    size_t num_input_constants;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
    SecantSSEReductionMode reduction_mode;
} SecantHsacoDynamicConstantSSERecipe;

/**
 * Immutable metadata parsed from one generated HSACO template.
 *
 * The plan resides in caller-owned storage, retains no inspected binary, and
 * contains no mutable specialization scratch.
 */
typedef struct SecantHsacoPlan SecantHsacoPlan;

/** Generates static-column materialize HIP with one patch island per kernel. */
SECANT_HSACO_DEC SecantResult secant_hsaco_materialize_source_generate(
    const SecantHsacoMaterializeRecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

/**
 * Generates static-column SSE HIP with one patch island per kernel.
 *
 * Atomic mode also emits secant_hsaco_sse_compact. Callers may give the SSE
 * kernels a padded output leading dimension, then use this kernel to restore
 * the ordinary dense AST-major output layout.
 *
 * Workspace mode also emits secant_hsaco_sse_reduce for reducing
 * result-major tile partials into the atomic-mode output layout.
 */
SECANT_HSACO_DEC SecantResult secant_hsaco_sse_source_generate(
    const SecantHsacoSSERecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

/**
 * Generates tile-static HIP SSE with per-thread constant settings.
 *
 * Workspace mode also emits secant_hsaco_dynamic_constant_sse_reduce.
 */
SECANT_HSACO_DEC SecantResult
secant_hsaco_dynamic_constant_sse_source_generate(
    const SecantHsacoDynamicConstantSSERecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

/**
 * Measures or constructs a materialize HSACO patch plan.
 *
 * hsaco must be aligned to at least eight bytes. Direct ISA specialization
 * currently supports gfx1200 and gfx1201.
 */
SECANT_HSACO_DEC SecantResult secant_hsaco_materialize_inspect(
    const SecantHsacoMaterializeRecipe* recipe,
    const void* hsaco,
    size_t hsaco_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantHsacoPlan** plan_ret
);

/**
 * Measures or constructs an SSE HSACO patch plan.
 *
 * The alignment and architecture contract matches materialize inspection.
 */
SECANT_HSACO_DEC SecantResult secant_hsaco_sse_inspect(
    const SecantHsacoSSERecipe* recipe,
    const void* hsaco,
    size_t hsaco_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantHsacoPlan** plan_ret
);

/** Measures or constructs a dynamic-constant SSE HSACO patch plan. */
SECANT_HSACO_DEC SecantResult secant_hsaco_dynamic_constant_sse_inspect(
    const SecantHsacoDynamicConstantSSERecipe* recipe,
    const void* hsaco,
    size_t hsaco_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantHsacoPlan** plan_ret
);

/** Reads immutable materialize shape metadata from a validated plan. */
SECANT_HSACO_DEC SecantResult secant_hsaco_materialize_plan_info_get(
    const SecantHsacoPlan* plan,
    size_t* hsaco_size_ret,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret
);

/** Reads immutable SSE shape metadata from a validated plan. */
SECANT_HSACO_DEC SecantResult secant_hsaco_sse_plan_info_get(
    const SecantHsacoPlan* plan,
    size_t* hsaco_size_ret,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
);

/** Reads immutable dynamic-constant SSE metadata from a validated plan. */
SECANT_HSACO_DEC SecantResult
secant_hsaco_dynamic_constant_sse_plan_info_get(
    const SecantHsacoPlan* plan,
    size_t* hsaco_size_ret,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_input_columns_ret,
    size_t* num_input_constants_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
);

/**
 * Writes AST AMDGPU instructions directly into an HSACO template copy.
 *
 * The function allocates no memory. On error, hsaco may be partially
 * specialized and must not be loaded.
 */
SECANT_HSACO_DEC SecantResult secant_hsaco_specialize_into(
    const SecantHsacoPlan* plan,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    void* hsaco,
    size_t hsaco_size
);

#endif /* SECANT_HSACO_H_INCLUDED */
