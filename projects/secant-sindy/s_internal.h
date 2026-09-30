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
#ifndef SECANT_SINDY_INTERNAL_H_INCLUDED
#define SECANT_SINDY_INTERNAL_H_INCLUDED

#include "secant_sindy.h"

#include <stdint.h>

static inline int secant_sindy_checked_add(size_t a, size_t b, size_t* result_ret) {
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *result_ret = a + b;
    return 1;
}

static inline int secant_sindy_checked_mul(size_t a, size_t b, size_t* result_ret) {
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *result_ret = a * b;
    return 1;
}

static inline int secant_sindy_span_extent(size_t outer_count, size_t leading_dimension, size_t inner_count,
                                           size_t* extent_ret) {
    size_t offset;

    if (outer_count == 0u || inner_count == 0u) {
        *extent_ret = 0u;
        return 1;
    }
    return secant_sindy_checked_mul(outer_count - 1u, leading_dimension, &offset) &&
        secant_sindy_checked_add(offset, inner_count, extent_ret);
}

#endif /* SECANT_SINDY_INTERNAL_H_INCLUDED */
