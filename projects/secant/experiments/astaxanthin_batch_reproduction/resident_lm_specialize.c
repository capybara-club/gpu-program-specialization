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

static const SecantAstInstruction glucose_rate[] = {
    secant_ast_encode_static_column_input_f32(4u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_static_column_input_f32(5u),
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_static_column_input_f32(6u),
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_div_f32,
    secant_ast_encode_return_f32
};

static const SecantAstInstruction sucrose_rate[] = {
    secant_ast_encode_static_column_input_f32(7u),
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_static_column_input_f32(2u),
    secant_ast_encode_static_column_input_f32(8u),
    secant_ast_encode_static_column_input_f32(0u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
    secant_ast_encode_static_column_input_f32(9u),
    secant_ast_encode_static_column_input_f32(1u),
    secant_ast_encode_mul_f32,
    secant_ast_encode_add_f32,
    secant_ast_encode_mul_f32,
    secant_ast_encode_div_f32,
    secant_ast_encode_return_f32
};

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s template.cubin specialized.cubin\n", argv[0]);
        return 2;
    }

    FILE *input = fopen(argv[1], "rb");
    if (input == NULL || fseek(input, 0, SEEK_END) != 0) return 3;
    const long bytes = ftell(input);
    if (bytes <= 0 || fseek(input, 0, SEEK_SET) != 0) return 4;
    unsigned char *cubin = (unsigned char *)malloc((size_t)bytes);
    if (cubin == NULL || fread(cubin, 1u, (size_t)bytes, input) != (size_t)bytes) {
        return 5;
    }
    fclose(input);

    SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();
    recipe.num_kernels = 1u;
    recipe.asts_per_kernel = 2u;
    recipe.num_inputs = 12u;
    recipe.patch_capacity_instructions = 128u;

    size_t plan_bytes = 0u;
    SecantResult result = secant_cubin_plan_storage_size(
        &recipe.header, cubin, (size_t)bytes, &plan_bytes);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr, "CUBIN inspection failed: %s\n", secant_result_to_string(result));
        return 6;
    }
    void *storage = malloc(plan_bytes);
    SecantCubinPlan *plan = NULL;
    result = secant_cubin_plan_init(
        &recipe.header, cubin, (size_t)bytes, storage, plan_bytes, &plan);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr, "plan initialization failed: %s\n", secant_result_to_string(result));
        return 7;
    }

    const SecantAstInstruction *asts[] = {glucose_rate, sucrose_rate};
    SecantAstProgramSet programs = {0};
    programs.asts.items = asts;
    programs.asts.count = 2u;
    result = secant_cubin_specialize_into(plan, &programs, cubin, (size_t)bytes);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr, "specialization failed: %s\n", secant_result_to_string(result));
        return 8;
    }

    FILE *output = fopen(argv[2], "wb");
    if (output == NULL || fwrite(cubin, 1u, (size_t)bytes, output) != (size_t)bytes) {
        return 9;
    }
    fclose(output);
    free(storage);
    free(cubin);
    return 0;
}
