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
#include "secant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t
test_count(const char* text, const char* pattern) {
    const size_t pattern_size = strlen(pattern);
    size_t count = 0u;

    while ((text = strstr(text, pattern)) != NULL) {
        ++count;
        text += pattern_size;
    }
    return count;
}

int
main(void) {
    SecantCubinLMOptimizerRecipe recipe =
        secant_cubin_lm_optimizer_recipe_init();
    SecantCubinLMOptimizerRecipe invalid_recipe;
    const char* predicated_load;
    const char* static_load;
    const char* specialization_site;
    const char* address_increment;
    char* source = NULL;
    size_t source_size = 0u;
    int success = 1;

    recipe.num_kernels = 2u;
    recipe.num_input_columns = 32u;
    recipe.num_static_input_columns = recipe.num_input_columns;
    recipe.num_parameters = SECANT_LM_OPTIMIZER_PARAMETERS;
    recipe.tile_rows = 128u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = 256u;

    invalid_recipe = recipe;
    invalid_recipe.num_static_input_columns = 0u;
    if (secant_cubin_recipe_validate(&invalid_recipe.header) !=
            SECANT_ERROR_INVALID_VALUE ||
        secant_cubin_recipe_validate(&recipe.header) != SECANT_SUCCESS ||
        secant_cubin_source_size(&recipe.header, &source_size) !=
            SECANT_SUCCESS ||
        source_size == 0u) {
        fprintf(stderr, "LM recipe did not enforce the mixed-column contract\n");
        return 1;
    }
    source = (char*)malloc(source_size);
    if (source == NULL ||
        secant_cubin_source_write(&recipe.header, source, source_size) !=
            SECANT_SUCCESS) {
        free(source);
        return 1;
    }

    predicated_load = strstr(
        source,
        "setp.ne.u32 p, %2, 0; @p ld.shared.f32 %0, [%1]");
    static_load = strstr(source, "float static_input31 =");
    specialization_site = static_load == NULL
        ? NULL
        : strstr(static_load, "brkpt;");
    address_increment = specialization_site == NULL
        ? NULL
        : strstr(specialization_site, "address7 += shared_stride_bytes;");
    if (predicated_load == NULL || static_load == NULL ||
        specialization_site == NULL || address_increment == NULL ||
        !(predicated_load < static_load && static_load < specialization_site &&
          specialization_site < address_increment) ||
        strstr(source, "\"+f\"(static_input31)") == NULL ||
        strstr(source, "\"+f\"(constant7)") == NULL ||
        strstr(source, "\"+f\"(mixed7)") == NULL ||
        strstr(source, "\"+f\"(mixed_gradient7)") == NULL ||
        test_count(source, "float static_input31 =") != recipe.num_kernels ||
        test_count(source, "address7 += shared_stride_bytes;") !=
            recipe.num_kernels ||
        test_count(source, "@p ld.shared.f32") !=
            recipe.num_kernels * recipe.num_parameters) {
        fprintf(stderr, "LM source lost the mixed SSE row-load shape\n");
        success = 0;
    }

    free(source);
    return success ? 0 : 1;
}
