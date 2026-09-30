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
#include "o_odezza_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                                  \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
static double evaluate(const OLmExpressions *c, uint32_t root, const double input[6], uint32_t permutation) {
    double values[O_LM_MAX_NODES];
    uint32_t i;
    for (i = 0; i <= root; ++i) {
        const OLmNode *n = &c->nodes[i];
        double a = n->arity ? values[n->child[0]] : 0;
        double b = n->arity > 1 ? values[n->child[1]] : 0;
        switch (n->opcode) {
        case ODEZZA_AST_STATE_F32:
            values[i] = input[n->value];
            break;
        case ODEZZA_AST_CONSTANT_F32:
            values[i] = input[3 + n->value];
            break;
        case ODEZZA_AST_LITERAL_F32: {
            float f;
            memcpy(&f, &n->value, 4);
            values[i] = f;
            break;
        }
        case ODEZZA_AST_ADD_F32:
            values[i] = a + b;
            break;
        case ODEZZA_AST_SUB_F32:
            values[i] = a - b;
            break;
        case ODEZZA_AST_MUL_F32:
            values[i] = a * b;
            break;
        case ODEZZA_AST_DIV_F32:
            values[i] = a / b;
            break;
        case ODEZZA_AST_NEG_F32:
            values[i] = -a;
            break;
        case ODEZZA_AST_SQRT_F32:
            values[i] = sqrt(a);
            break;
        case ODEZZA_AST_RCP_F32:
            values[i] = 1 / a;
            break;
        case ODEZZA_AST_SIN_F32:
            values[i] = sin(a);
            break;
        case ODEZZA_AST_COS_F32:
            values[i] = cos(a);
            break;
        case ODEZZA_AST_EX2_F32:
            values[i] = exp2(a);
            break;
        case ODEZZA_AST_LG2_F32:
            values[i] = log2(a);
            break;
        case ODEZZA_AST_RSQRT_F32:
            values[i] = 1 / sqrt(a);
            break;
        case ODEZZA_AST_TANH_F32:
            values[i] = tanh(a);
            break;
        case ODEZZA_AST_EXP_F32:
            values[i] = exp(a);
            break;
        case ODEZZA_AST_LOG_F32:
            values[i] = log(a);
            break;
        case ODEZZA_AST_FMA_F32:
            values[i] = a * b + values[n->child[2]];
            break;
        case ODEZZA_AST_TOGGLE2_F32:
            values[i] = values[n->child[(permutation >> n->value) & 1u]];
            break;
        default:
            CHECK(0);
        }
    }
    return values[root];
}
static void check_derivatives(OLmExpressions *c, const OdezzaLmShape *shape) {
    unsigned int op, permutation, j;
    for (op = 0; op < 12; ++op) {
        static const unsigned char operators[] = {0x94, 0x95, 0x96, 0x9b, 0x9c, 0x9d,
                                                  0x9e, 0x9f, 0xa0, 0xa1, 0xa2, 0x90};
        unsigned char bytes[] = {0x81, 0, 0x81, 2, 0x84, 0,    0x82, 0,   0x92,
                                 0x81, 1, 0x82, 1, 0x92, 0x90, 0x9b, 0x80};
        OdezzaAstProgram programs[3];
        if (operators[op] == 0x90)
            continue;
        bytes[15] = operators[op];
        for (j = 0; j < 3; ++j) {
            programs[j].bytes = bytes;
            programs[j].byte_count = sizeof(bytes);
        }
        CHECK(o_lm_expressions(programs, shape, 1, c) == ODEZZA_SUCCESS);
        for (permutation = 0; permutation < 2; ++permutation)
            for (j = 0; j < 6; ++j) {
                double input[6] = {.4, .7, .6, .3, .2, .8}, lo, hi, actual, expected;
                input[j] -= 1e-6;
                lo = evaluate(c, c->roots[0], input, permutation);
                input[j] += 2e-6;
                hi = evaluate(c, c->roots[0], input, permutation);
                input[j] -= 1e-6;
                actual = evaluate(c, c->roots[3 + j * 3], input, permutation);
                expected = (hi - lo) / 2e-6;
                CHECK(isfinite(actual) && fabs(actual - expected) < 2e-7 * (1 + fabs(expected)));
            }
    }
    puts("132 native derivative/finite-difference checks passed");
}
int main(int argc, char **argv) {
    OdezzaLmShape shape = {3, 3, 256, 1};
    size_t size;
    char *source;
    OLmInspection *inspection = malloc(sizeof(*inspection));
    OLmExpressions *expressions = malloc(sizeof(*expressions));
    unsigned char code[] = {0x81, 0, 0x82, 0, 0x92, 0x80};
    OdezzaAstProgram rhs[3] = {{code, 6}, {code, 6}, {code, 6}};
    CHECK(inspection && expressions);
    CHECK(o_lm_generate_cuda(&shape, NULL, 0, &size) == 0);
    source = malloc(size);
    CHECK(source);
    CHECK(o_lm_generate_cuda(&shape, source, size - 1, &size) == ODEZZA_ERROR_INSUFFICIENT_BUFFER);
    CHECK(o_lm_generate_cuda(&shape, source, size, &size) == 0);
    CHECK(strstr(source, "odezza_trajectory_lm") && strstr(source, "0f4fc40000"));
    CHECK(o_lm_expressions(rhs, &shape, 0, expressions) == 0);
    CHECK(expressions->nodes[expressions->roots[3]].opcode == ODEZZA_AST_CONSTANT_F32);
    CHECK(expressions->nodes[expressions->roots[3 + 3 * 3]].opcode == ODEZZA_AST_STATE_F32);
    CHECK(o_lm_inspect("invalid", 7, &shape, inspection) == ODEZZA_ERROR_FORMAT);
    if (argc > 1) {
        FILE *f = fopen(argv[1], "wb");
        CHECK(f);
        CHECK(fwrite(source, 1, size - 1, f) == size - 1);
        CHECK(!fclose(f));
    }
    if (argc > 2) {
        FILE *f = fopen(argv[2], "rb");
        long n;
        void *data, *patched;
        OSassInstruction *codebuf;
        uint32_t regs;
        CHECK(f);
        CHECK(!fseek(f, 0, SEEK_END));
        n = ftell(f);
        CHECK(n > 0);
        rewind(f);
        data = malloc((size_t)n);
        patched = malloc((size_t)n);
        codebuf = malloc(shape.site_patch_capacity * sizeof(*codebuf));
        CHECK(data && patched && codebuf);
        CHECK(fread(data, 1, (size_t)n, f) == (size_t)n);
        CHECK(!fclose(f));
        CHECK(o_lm_inspect(data, (size_t)n, &shape, inspection) == 0);
        CHECK(o_lm_specialize(data, (size_t)n, inspection, rhs, 0, expressions, codebuf, patched, &regs) ==
              0);
        printf("inspected and specialized %u sites, registers %u\n", inspection->site_count, regs);
        free(data);
        free(patched);
        free(codebuf);
    }
    check_derivatives(expressions, &shape);
    free(source);
    free(inspection);
    free(expressions);
    puts("native_lm_host_contract passed");
    return 0;
}
