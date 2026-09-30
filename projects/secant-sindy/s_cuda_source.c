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
#include "s_internal.h"

#include <stdio.h>
#include <string.h>

#define INCBIN_PREFIX secant_sindy_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include "thirdparty/incbin.h"

#ifndef SECANT_SINDY_CUDA_TARGET_STATS_SOURCE_PATH
#error "SECANT_SINDY_CUDA_TARGET_STATS_SOURCE_PATH must name the target-statistics CUDA source"
#endif
#ifndef SECANT_SINDY_CUDA_SOLVER_SOURCE_PATH
#error "SECANT_SINDY_CUDA_SOLVER_SOURCE_PATH must name the solver CUDA source"
#endif

INCTXT(embedded_target_stats, SECANT_SINDY_CUDA_TARGET_STATS_SOURCE_PATH);
INCTXT(embedded_solver, SECANT_SINDY_CUDA_SOLVER_SOURCE_PATH);

static size_t secant_sindy_padded_targets_get(size_t max_targets) {
    size_t padded = 1u;

    while (padded < max_targets) {
        padded *= 2u;
    }
    return padded;
}

static int secant_sindy_solver_prefix_write(const SecantSindyCudaSolverRecipe* recipe, char* output,
                                            size_t output_size) {
    const size_t padded_targets = secant_sindy_padded_targets_get(recipe->max_targets);
    const uint32_t solver_sm = (recipe->compute_capability_major * 10u + recipe->compute_capability_minor) * 10u;

    return snprintf(
        output,
        output_size,
        "#define SECANT_SINDY_FEATURE_CAPACITY %zu\n"
        "#define SECANT_SINDY_PADDED_TARGETS %zu\n"
        "#define SECANT_SINDY_STLSQ_MAX_ITERATIONS %zu\n"
        "#define SECANT_SINDY_CUSOLVERDX_SM %u\n",
        recipe->feature_capacity,
        padded_targets,
        recipe->max_stlsq_iterations,
        solver_sm);
}

SecantSindyResult secant_sindy_cuda_solver_recipe_validate(const SecantSindyCudaSolverRecipe* recipe) {
    if (recipe == NULL) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    if (recipe->version != SECANT_SINDY_CUDA_SOLVER_RECIPE_VERSION_1) {
        return SECANT_SINDY_ERROR_UNSUPPORTED_VERSION;
    }
    if (recipe->struct_size < sizeof(*recipe) || recipe->flags != 0u || recipe->reserved != 0u) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    if (recipe->feature_capacity == 0u || recipe->feature_capacity > SECANT_SINDY_MAX_FEATURE_CAPACITY ||
        recipe->max_targets == 0u || recipe->max_targets > SECANT_SINDY_MAX_TARGETS ||
        recipe->max_stlsq_iterations == 0u || recipe->max_stlsq_iterations > recipe->feature_capacity ||
        recipe->compute_capability_minor > 9u ||
        (recipe->compute_capability_major != 8u && recipe->compute_capability_major != 9u &&
         recipe->compute_capability_major != 10u && recipe->compute_capability_major != 12u)) {
        return SECANT_SINDY_ERROR_UNSUPPORTED_SHAPE;
    }
    return SECANT_SINDY_SUCCESS;
}

const char* secant_sindy_cuda_target_stats_source_get(size_t* source_size_ret) {
    if (source_size_ret != NULL) {
        *source_size_ret = secant_sindy_embedded_target_stats_size;
    }
    return secant_sindy_embedded_target_stats_data;
}

SecantSindyResult secant_sindy_cuda_solver_source_size(const SecantSindyCudaSolverRecipe* recipe,
                                                       size_t* source_size_ret) {
    SecantSindyResult result;
    int prefix_size;

    if (source_size_ret == NULL) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    *source_size_ret = 0u;
    result = secant_sindy_cuda_solver_recipe_validate(recipe);
    if (result != SECANT_SINDY_SUCCESS) {
        return result;
    }
    prefix_size = secant_sindy_solver_prefix_write(recipe, NULL, 0u);
    if (prefix_size < 0 || !secant_sindy_checked_add((size_t)prefix_size, secant_sindy_embedded_solver_size,
                                                     source_size_ret)) {
        return SECANT_SINDY_ERROR_OVERFLOW;
    }
    return SECANT_SINDY_SUCCESS;
}

SecantSindyResult secant_sindy_cuda_solver_source_write(const SecantSindyCudaSolverRecipe* recipe, char* source,
                                                        size_t source_size) {
    SecantSindyResult result;
    size_t required_size;
    int prefix_size;

    if (source == NULL) {
        return SECANT_SINDY_ERROR_INVALID_VALUE;
    }
    result = secant_sindy_cuda_solver_source_size(recipe, &required_size);
    if (result != SECANT_SINDY_SUCCESS) {
        return result;
    }
    if (source_size < required_size) {
        return SECANT_SINDY_ERROR_INSUFFICIENT_BUFFER;
    }
    prefix_size = secant_sindy_solver_prefix_write(recipe, source, source_size);
    if (prefix_size < 0 || (size_t)prefix_size >= source_size) {
        return SECANT_SINDY_ERROR_FORMAT;
    }
    memcpy(source + prefix_size, secant_sindy_embedded_solver_data, secant_sindy_embedded_solver_size);
    return SECANT_SINDY_SUCCESS;
}
