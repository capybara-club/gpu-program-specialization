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
#include <stdio.h>
#include <ptx_inject.h>

extern "C"
__global__
void
kernel(
    int64_t num_rows, int64_t num_cols,
    float* in, int64_t ld_in, int64_t batch_stride_in,
    float* out, int64_t ld_out, int64_t batch_stride_out,
    int triangle // 0 -> FULL, 1 -> LOWER, 2 -> UPPER
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int batch_num = blockIdx.z;

    if (row >= num_rows || col >= num_cols) return;
    if (triangle == 1 && col > row) return;
    if (triangle == 2 && row > col) return;

    int64_t idx_in = batch_num * batch_stride_in + col * ld_in + row;
    int64_t idx_out = batch_num * batch_stride_out + col * ld_out + row;

    float v = in[idx_in];
    PTX_INJECT("func",
        PTX_IN(U32, row, row),
        PTX_IN(U32, col, col),
        PTX_MOD(F32, v, v)
    );
    out[idx_out] = v;
}
