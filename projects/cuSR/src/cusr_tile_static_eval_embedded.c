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
#include "cusr_tile_static_eval_embedded.h"

#ifndef CUSR_TILE_STATIC_EVAL_8K_CUBIN_PATH
#error "CUSR_TILE_STATIC_EVAL_8K_CUBIN_PATH must name the generated 8-kernel cubin"
#endif

#ifndef CUSR_TILE_STATIC_EVAL_32K_CUBIN_PATH
#error "CUSR_TILE_STATIC_EVAL_32K_CUBIN_PATH must name the generated 32-kernel cubin"
#endif

#ifndef CUSR_TILE_STATIC_EVAL_EMBEDDED_SASS_ARCH
#error "CUSR_TILE_STATIC_EVAL_EMBEDDED_SASS_ARCH must name the cubin SASS architecture"
#endif

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX cusr_eval_embedded_
#include <incbin.h>

INCBIN(tile_static_eval_8k_cubin, CUSR_TILE_STATIC_EVAL_8K_CUBIN_PATH);
INCBIN(tile_static_eval_32k_cubin, CUSR_TILE_STATIC_EVAL_32K_CUBIN_PATH);

const unsigned char*
cusr_tile_static_eval_embedded_cubin(size_t num_kernels, size_t* cubin_size_ret)
{
    if (cubin_size_ret == NULL) {
        return NULL;
    }

    if (num_kernels == 8u) {
        *cubin_size_ret = (size_t)cusr_eval_embedded_tile_static_eval_8k_cubin_size;
        return cusr_eval_embedded_tile_static_eval_8k_cubin_data;
    }

    if (num_kernels == 32u) {
        *cubin_size_ret = (size_t)cusr_eval_embedded_tile_static_eval_32k_cubin_size;
        return cusr_eval_embedded_tile_static_eval_32k_cubin_data;
    }

    *cubin_size_ret = 0u;
    return NULL;
}

unsigned
cusr_tile_static_eval_embedded_sass_arch(void)
{
    return CUSR_TILE_STATIC_EVAL_EMBEDDED_SASS_ARCH;
}
