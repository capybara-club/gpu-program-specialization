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
#include "bench_cubin_embedded.h"

#ifndef SECANT_EMBEDDED_CUBIN_PATH
#error "SECANT_EMBEDDED_CUBIN_PATH must name the generated CUBIN"
#endif

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX secant_embedded_
#include <incbin.h>

INCBIN(cubin_sse_template, SECANT_EMBEDDED_CUBIN_PATH);

const unsigned char*
secant_bench_cubin_embedded_template(
    SecantBenchShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    uint32_t target_sm,
    size_t* cubin_size_ret
) {
    if (cubin_size_ret == NULL) {
        return NULL;
    }
    *cubin_size_ret = 0u;
    if (shape != SECANT_BENCH_SHAPE_SSE ||
        num_kernels != SECANT_EMBEDDED_CUBIN_KERNELS ||
        asts_per_kernel != SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL ||
        num_inputs != SECANT_EMBEDDED_CUBIN_INPUTS ||
        num_targets != SECANT_EMBEDDED_CUBIN_TARGETS ||
        tile_rows != SECANT_EMBEDDED_CUBIN_TILE_ROWS ||
        threads_per_block != SECANT_EMBEDDED_CUBIN_THREADS ||
        patch_capacity_instructions !=
            SECANT_EMBEDDED_CUBIN_PATCH_CAPACITY ||
        target_sm != SECANT_EMBEDDED_CUBIN_SM) {
        return NULL;
    }
    *cubin_size_ret =
        (size_t)secant_embedded_cubin_sse_template_size;
    return secant_embedded_cubin_sse_template_data;
}
