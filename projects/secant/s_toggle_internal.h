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
#ifndef SECANT_TOGGLE_INTERNAL_H
#define SECANT_TOGGLE_INTERNAL_H
#include "s_ast_internal.h"

/* Checked dimensions shared by host and device launch adapters. */
static inline int s_size_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a)
        return 0;
    *out = a + b;
    return 1;
}
static inline int s_size_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a)
        return 0;
    *out = a * b;
    return 1;
}
static inline int s_span_size(size_t count, size_t stride, size_t width, size_t *out) {
    size_t offset;
    if (!count || !width) {
        *out = 0;
        return 1;
    }
    return stride >= width && s_size_mul(count - 1, stride, &offset) && s_size_add(offset, width, out);
}
static inline SecantResult s_toggle_layout(size_t asts, size_t banks, unsigned bits, size_t constants,
                                           size_t bank_stride, size_t *configurations,
                                           size_t *bank_elements) {
    if (!asts || !banks || bits > 32 || constants > SECANT_AST_MAX_INPUTS)
        return SECANT_ERROR_INVALID_VALUE;
    if ((uint64_t)SIZE_MAX < (UINT64_C(1) << bits) ||
        !s_size_mul(banks, (size_t)(UINT64_C(1) << bits), configurations))
        return SECANT_ERROR_OVERFLOW;
    if (!constants) {
        if (bank_stride)
            return SECANT_ERROR_INVALID_VALUE;
        *bank_elements = 0;
        return SECANT_SUCCESS;
    }
    if (bank_stride < constants)
        return SECANT_ERROR_INVALID_VALUE;
    if (!s_span_size(banks, bank_stride, constants, bank_elements))
        return SECANT_ERROR_OVERFLOW;
    return SECANT_SUCCESS;
}
#endif
