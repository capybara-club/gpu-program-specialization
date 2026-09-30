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
#include "toggle_fixture.h"
#include "s_toggle_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c)                                                                                             \
    do {                                                                                                     \
        if (!(c)) {                                                                                          \
            fprintf(stderr, "line %d: %s\n", __LINE__, #c);                                                  \
            return 1;                                                                                        \
        }                                                                                                    \
    } while (0)
int main(void) {
    float input[3 * 17], targets[2 * 17], banks[38], output[7 * 2 * 109];
    size_t a, b, p, t, row, i, configs, span;
    int pass;
    SecantCpuToggleSSERun run = secant_cpu_toggle_sse_run_init();
    for (i = 0; i < 3 * 17; ++i)
        input[i] = (float)i * .013f - 0.3f;
    for (i = 0; i < 2 * 17; ++i)
        targets[i] = (float)i * .007f;
    for (i = 0; i < 38; ++i)
        banks[i] = (float)i * .01f - .6f;
    run.programs.asts.items = toggle_asts;
    run.programs.asts.count = 7;
    run.num_inputs = 3;
    run.num_constants = 2;
    run.num_rows = 17;
    run.num_targets = 2;
    run.num_banks = 13;
    run.toggle_bits = 3;
    run.input.data = input;
    run.input.num_elements = 51;
    run.input.leading_dimension = 17;
    run.targets.data = targets;
    run.targets.num_elements = 34;
    run.targets.leading_dimension = 17;
    run.constants.data = banks;
    run.constants.num_elements = 38;
    run.constants.bank_stride = 3;
    run.output.data = output;
    run.output.num_elements = 7 * 2 * 109;
    run.output.leading_dimension = 109;
    for (pass = 0; pass < 2; ++pass) {
        if (pass)
            for (i = 0; i < 38; ++i)
                banks[i] += .137f;
        for (i = 0; i < 7 * 2 * 109; ++i)
            output[i] = -123.f;
        CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_SUCCESS);
        for (a = 0; a < 7; ++a)
            for (t = 0; t < 2; ++t)
                for (b = 0; b < 13; ++b)
                    for (p = 0; p < 8; ++p) {
                        float expected = 0;
                        const float *c = banks + b * 3;
                        for (row = 0; row < 17; ++row) {
                            float x[3] = {input[row], input[17 + row], input[34 + row]};
                            float err = toggle_expected(a, (unsigned)p, x, c) - targets[t * 17 + row];
                            expected += err * err;
                        }
                        CHECK(fabsf(output[(a * 2 + t) * 109 + b * 8 + p] - expected) <
                              1e-5f * (1 + fabsf(expected)));
                    }
        for (a = 0; a < 14; ++a)
            for (i = 104; i < 109; ++i)
                CHECK(output[a * 109 + i] == -123.f);
    }
    for (run.header.version = 1; run.header.version < 3; ++run.header.version)
        CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_UNSUPPORTED_VERSION);
    run.header.version = SECANT_CPU_RUN_VERSION_3;
    output[0] = 1234;
    run.toggle_bits = 2;
    CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_BAD_PROGRAM);
    CHECK(output[0] == 1234);
    run.toggle_bits = 3;
    run.constants.num_elements = 37;
    CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_INSUFFICIENT_BUFFER);
    run.constants.num_elements = 38;
    run.constants.bank_stride = 1;
    CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_INVALID_VALUE);
    run.constants.bank_stride = 3;
    run.output.data = input;
    CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_INVALID_VALUE);
    run.output.data = output;
    CHECK(s_toggle_layout(1, 1, 32, 0, 0, &configs, &span) == SECANT_SUCCESS &&
          configs == UINT64_C(4294967296));
    CHECK(s_toggle_layout(1, SIZE_MAX, 32, 0, 0, &configs, &span) == SECANT_ERROR_OVERFLOW);
    {
        const uint8_t obsolete[] = {0xb7, 0, RET};
        const uint8_t *asts[] = {obsolete};
        run.programs.asts.items = asts;
        run.programs.asts.count = 1;
        CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_BAD_PROGRAM);
    }
    {
        const uint8_t invalid[] = {COL(0), COL(1), secant_ast_encode_add_f32, BANK(0), TOG(0), RET};
        const uint8_t *asts[] = {invalid};
        run.programs.asts.items = asts;
        CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_BAD_PROGRAM);
    }
    {
        const uint8_t invalid[] = {COL(0), COL(1), BANK(0), BANK(1), TOG4(0, 0), RET};
        const uint8_t *asts[] = {invalid};
        run.programs.asts.items = asts;
        CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_ERROR_BAD_PROGRAM);
    }
    {
        const uint8_t literal[] = {secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_TWO), RET};
        const uint8_t *asts[] = {literal};
        run.programs.asts.items = asts;
        run.num_constants = 0;
        run.constants.data = NULL;
        run.constants.bank_stride = run.constants.num_elements = 0;
        run.toggle_bits = 0;
        run.num_banks = 1;
        CHECK(secant_cpu_run_toggle_sse(&run) == SECANT_SUCCESS);
    }
    {
        SecantCubinToggleSSERecipe r = secant_cubin_toggle_sse_recipe_init();
        size_t size;
        char *source;
        r.num_kernels = 3;
        r.asts_per_kernel = 32;
        r.num_inputs = 3;
        r.num_constants = 8;
        r.num_targets = 2;
        r.tile_rows = 31;
        r.threads_per_block = 128;
        r.patch_capacity_instructions = 256;
        CHECK(secant_cubin_source_size(&r.header, &size) == SECANT_SUCCESS);
        source = (char *)malloc(size);
        CHECK(source);
        CHECK(secant_cubin_source_write(&r.header, source, size - 1) == SECANT_ERROR_INSUFFICIENT_BUFFER);
        CHECK(secant_cubin_source_write(&r.header, source, size) == SECANT_SUCCESS);
        CHECK(strstr(source, "configuration >> toggle_bits") && !strstr(source, "leaf_mask"));
        /* Exactly K bank loads per function, independent of packed AST count. */
        {
            size_t loads = 0;
            const char *cursor = source;
            while ((cursor = strstr(cursor, "= banks[bank * bank_stride + ")) != NULL) {
                ++loads;
                ++cursor;
            }
            CHECK(loads == r.num_kernels * r.num_constants);
        }
        free(source);
        for (r.header.version = 1; r.header.version < 3; ++r.header.version)
            CHECK(secant_cubin_source_size(&r.header, &size) == SECANT_ERROR_UNSUPPORTED_VERSION);
    }
    {
        uint8_t expression[24];
        const uint8_t *ast = expression;
        float x = 5.f, target = 0.f, bank[] = {-.25f, .5f}, values[4];
        SecantCpuToggleSSERun affine = secant_cpu_toggle_sse_run_init();
        secant_ast_affine_bank_write(expression, 0, 2.f, 3.f);
        expression[10] = SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32; expression[11] = 0;
        expression[12] = SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32; expression[13] = 0;
        expression[14] = SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
        affine.programs.asts.items=&ast; affine.programs.asts.count=1;
        affine.num_inputs=affine.num_constants=affine.num_targets=affine.num_rows=1;
        affine.num_banks=2; affine.toggle_bits=1;
        affine.input=(SecantConstHostMatrixF32){&x,1,1};
        affine.targets=(SecantConstHostMatrixF32){&target,1,1};
        affine.constants=(SecantConstHostConstantBanks){bank,2,1};
        affine.output=(SecantHostMatrixF32){values,4,4};
        CHECK(secant_cpu_run_toggle_sse(&affine)==SECANT_SUCCESS);
        CHECK(values[0]==6.25f && values[1]==25.f && values[2]==16.f && values[3]==25.f);
        secant_ast_affine_bank_write(expression, 0, NAN, 0.f);
        values[0]=1234;
        CHECK(secant_cpu_run_toggle_sse(&affine)==SECANT_ERROR_BAD_PROGRAM && values[0]==1234);
    }
    puts("toggle Cartesian CPU oracle and validation passed");
    return 0;
}
