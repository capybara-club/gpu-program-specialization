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
#ifndef CUSR_TILE_STATIC_MSE_EMBEDDED_H_INCLUDED
#define CUSR_TILE_STATIC_MSE_EMBEDDED_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUSR_TILE_STATIC_MSE_EMBEDDED_AST_CAPACITY 32u
#define CUSR_TILE_STATIC_MSE_EMBEDDED_TILE_ROWS 64u
#define CUSR_TILE_STATIC_MSE_EMBEDDED_CTA_THREADS 128u
#define CUSR_TILE_STATIC_MSE_EMBEDDED_FUNCTION_PATTERN "cusr_tile_static_mse_f32_%03d"

/* Embedded module shapes contain 8, 32, or 128 kernels. */
const unsigned char*
cusr_tile_static_mse_embedded_cubin(size_t num_kernels, size_t* cubin_size_ret);

unsigned
cusr_tile_static_mse_embedded_sass_arch(void);

#ifdef __cplusplus
}
#endif

#endif /* CUSR_TILE_STATIC_MSE_EMBEDDED_H_INCLUDED */
