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
#ifndef SECANT_TEST_TOGGLE_FIXTURE_H
#define SECANT_TEST_TOGGLE_FIXTURE_H
#include "secant.h"
#define COL(i) secant_ast_encode_column_f32(i)
#define BANK(i) secant_ast_encode_bank_constant_f32(i)
#define TOG(i) secant_ast_encode_toggle2_f32(i)
#define TOG4(i, j) secant_ast_encode_toggle4_f32(i, j)
#define RET secant_ast_encode_return_f32
static const uint8_t tf0[] = {COL(0), COL(1), TOG(0), BANK(0), secant_ast_encode_mul_f32, RET};
static const uint8_t tf1[] = {COL(2), BANK(1), TOG(1), COL(0), BANK(0), TOG(0), secant_ast_encode_add_f32,
                              RET};
static const uint8_t tf2[] = {COL(0), BANK(0), COL(2), BANK(1), TOG4(0, 2), RET};
static const uint8_t tf3[] = {BANK(0), BANK(1), TOG(2), BANK(1), BANK(0), TOG(2), secant_ast_encode_sub_f32,
                              RET};
static const uint8_t tf4[] = {COL(0),
                              BANK(0),
                              TOG(0),
                              COL(1),
                              BANK(1),
                              TOG(1),
                              secant_ast_encode_mul_f32,
                              COL(2),
                              BANK(0),
                              TOG(2),
                              secant_ast_encode_add_f32,
                              RET};
static const uint8_t tf5[] = {secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE), BANK(1), TOG(1), RET};
static const uint8_t tf6[] = {
    BANK(0), BANK(1), COL(0), COL(1), TOG4(2, 0), BANK(0), secant_ast_encode_add_f32, RET};
static const uint8_t *const toggle_asts[] = {tf0, tf1, tf2, tf3, tf4, tf5, tf6};
static inline float toggle_expected(size_t ast, unsigned p, const float *x, const float *c) {
    switch (ast) {
    case 0:
        return ((p & 1) ? x[1] : x[0]) * c[0];
    case 1:
        return ((p & 2) ? c[1] : x[2]) + ((p & 1) ? c[0] : x[0]);
    case 2: {
        float a[4] = {x[0], c[0], x[2], c[1]};
        return a[(p & 1) | ((p >> 1) & 2)];
    }
    case 3:
        return ((p & 4) ? c[1] : c[0]) - ((p & 4) ? c[0] : c[1]);
    case 4:
        return ((p & 1) ? c[0] : x[0]) * ((p & 2) ? c[1] : x[1]) + ((p & 4) ? c[0] : x[2]);
    case 5:
        return (p & 2) ? c[1] : 1.f;
    default: {
        float a[4] = {c[0], c[1], x[0], x[1]};
        return a[((p >> 2) & 1) | ((p & 1) << 1)] + c[0];
    }
    }
}
#endif
