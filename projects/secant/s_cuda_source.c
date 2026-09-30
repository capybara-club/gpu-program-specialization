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

#define INCBIN_PREFIX secant_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include "thirdparty/incbin.h"

#ifndef SECANT_CUDA_TARGET_STATS_F32_SOURCE_PATH
#error "SECANT_CUDA_TARGET_STATS_F32_SOURCE_PATH must name the embedded CUDA source"
#endif

INCTXT(cuda_target_stats_f32_source, SECANT_CUDA_TARGET_STATS_F32_SOURCE_PATH);

const char *secant_cuda_target_stats_f32_source_get(size_t *size) {
    if (size) *size = secant_cuda_target_stats_f32_source_size;
    return secant_cuda_target_stats_f32_source_data;
}
