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

#include "s_cubin_internal.h"

SecantResult
_secant_cubin_recipe_header_validate(
    const SecantCubinRecipeHeader* recipe
) {
    size_t required_size;

    if (recipe == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if ((size_t)recipe->struct_size < sizeof(*recipe)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if (recipe->version != SECANT_CUBIN_RECIPE_VERSION_3) {
        return SECANT_ERROR_UNSUPPORTED_VERSION;
    }
    if (recipe->flags != 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    switch (recipe->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32:
            required_size = sizeof(SecantCubinMaterializeRecipe);
            break;
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32:
            required_size = sizeof(SecantCubinSSERecipe);
            break;
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32:
            required_size = sizeof(SecantCubinAffineStatsRecipe);
            break;
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32:
            required_size = sizeof(SecantCubinGramStatsRecipe);
            break;
        case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32:
            required_size = sizeof(SecantCubinToggleSSERecipe); break;
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    if ((size_t)recipe->struct_size < required_size) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return SECANT_SUCCESS;
}

SecantResult
_secant_cubin_recipe_validate(
    const SecantCubinRecipeHeader* recipe
) {
    SecantResult result = _secant_cubin_recipe_header_validate(recipe);

    if (result != SECANT_SUCCESS) {
        return result;
    }
    switch (recipe->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32:
            return _secant_cubin_materialize_recipe_validate(
                (const SecantCubinMaterializeRecipe*)(const void*)recipe);
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32:
            return _secant_cubin_sse_recipe_validate(
                (const SecantCubinSSERecipe*)(const void*)recipe);
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32:
            return _secant_cubin_affine_stats_recipe_validate(
                (const SecantCubinAffineStatsRecipe*)(const void*)recipe);
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32:
            return _secant_cubin_gram_stats_recipe_validate(
                (const SecantCubinGramStatsRecipe*)(const void*)recipe);
        case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32:
            return _secant_cubin_toggle_sse_recipe_validate((const SecantCubinToggleSSERecipe*)recipe);
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
}

static SecantResult
_secant_cubin_source_generate(
    const SecantCubinRecipeHeader* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
    SecantResult result = _secant_cubin_recipe_validate(recipe);

    if (result != SECANT_SUCCESS) {
        return result;
    }
    if (required_size_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    switch (recipe->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32:
            return _secant_cubin_materialize_source_generate(
                (const SecantCubinMaterializeRecipe*)(const void*)recipe,
                output,
                output_size,
                required_size_ret);
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32:
            return _secant_cubin_sse_source_generate(
                (const SecantCubinSSERecipe*)(const void*)recipe,
                output,
                output_size,
                required_size_ret);
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32:
            return _secant_cubin_affine_stats_source_generate(
                (const SecantCubinAffineStatsRecipe*)(const void*)recipe,
                output,
                output_size,
                required_size_ret);
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32:
            return _secant_cubin_gram_stats_source_generate(
                (const SecantCubinGramStatsRecipe*)(const void*)recipe,
                output,
                output_size,
                required_size_ret);
        case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32:
            return _secant_cubin_toggle_sse_source_generate((const SecantCubinToggleSSERecipe*)recipe, output, output_size, required_size_ret);
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
}

SecantResult
secant_cubin_recipe_validate(
    const SecantCubinRecipeHeader* recipe
) {
    return _secant_cubin_recipe_validate(recipe);
}

SecantResult
secant_cubin_source_size(
    const SecantCubinRecipeHeader* recipe,
    size_t* required_size_ret
) {
    if (required_size_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *required_size_ret = 0u;
    return _secant_cubin_source_generate(recipe, NULL, 0u, required_size_ret);
}

SecantResult
secant_cubin_source_write(
    const SecantCubinRecipeHeader* recipe,
    char* output,
    size_t output_size
) {
    size_t required_size;

    if (output == NULL || output_size == 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_cubin_source_generate(recipe, output, output_size, &required_size);
}
