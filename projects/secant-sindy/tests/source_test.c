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
#include "secant_sindy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int target_source_check(void) {
    size_t size = 0u;
    const char* source = secant_sindy_cuda_target_stats_source_get(&size);

    return source != NULL && size >= 2u && source[size - 1u] == '\0' &&
        strstr(source, "secant_sindy_cuda_target_stats_f32") != NULL;
}

static int solver_source_check(uint32_t compute_capability_major, uint32_t compute_capability_minor,
                               const char* expected_sm) {
    SecantSindyCudaSolverRecipe recipe = secant_sindy_cuda_solver_recipe_init();
    SecantSindyResult result;
    size_t size = 0u;
    char* allocated_source = NULL;

    recipe.feature_capacity = 8u;
    recipe.max_targets = 3u;
    recipe.max_stlsq_iterations = 7u;
    recipe.compute_capability_major = compute_capability_major;
    recipe.compute_capability_minor = compute_capability_minor;
    result = secant_sindy_cuda_solver_source_size(&recipe, &size);
    if (result != SECANT_SINDY_SUCCESS || size < 2u) {
        return 0;
    }
    allocated_source = (char*)malloc(size);
    if (allocated_source == NULL) {
        return 0;
    }
    result = secant_sindy_cuda_solver_source_write(&recipe, allocated_source, size);
    if (result != SECANT_SINDY_SUCCESS || allocated_source[size - 1u] != '\0' ||
        strstr(allocated_source, "#define SECANT_SINDY_FEATURE_CAPACITY 8") == NULL ||
        strstr(allocated_source, "#define SECANT_SINDY_PADDED_TARGETS 4") == NULL ||
        strstr(allocated_source, expected_sm) == NULL ||
        strstr(allocated_source, "secant_sindy_cuda_ridge_solve_f32") == NULL ||
        strstr(allocated_source, "secant_sindy_cuda_stlsq_solve_f32") == NULL) {
        free(allocated_source);
        return 0;
    }
    free(allocated_source);
    return 1;
}

int main(void) {
    SecantSindyCudaSolverRecipe invalid = secant_sindy_cuda_solver_recipe_init();

    invalid.compute_capability_major = 11u;
    if (!target_source_check() || !solver_source_check(8u, 0u, "#define SECANT_SINDY_CUSOLVERDX_SM 800") ||
        !solver_source_check(9u, 0u, "#define SECANT_SINDY_CUSOLVERDX_SM 900") ||
        !solver_source_check(10u, 0u, "#define SECANT_SINDY_CUSOLVERDX_SM 1000") ||
        !solver_source_check(12u, 0u, "#define SECANT_SINDY_CUSOLVERDX_SM 1200") ||
        secant_sindy_cuda_solver_recipe_validate(&invalid) != SECANT_SINDY_ERROR_UNSUPPORTED_SHAPE) {
        fprintf(stderr, "source generation test failed\n");
        return 1;
    }
    printf("secant-sindy source generation test passed\n");
    return 0;
}
