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
#include <base_semiring_copy_grad.cuh>

extern "C"
__global__
__launch_bounds__(256)
void
cute_semiring(
    int64_t m, int64_t n, int64_t k, int64_t l,
    float const *A, int64_t ldA, int64_t batch_stride_A,
    float const *B, int64_t ldB, int64_t batch_stride_B,
    float       *C, int64_t ldC, int64_t batch_stride_C,
    float       *h0, int64_t batch_stride_h0,
    float       *h1, int64_t batch_stride_h1,
    float       *h2, int64_t batch_stride_h2,
    float       *h3, int64_t batch_stride_h3
) {
    return
        cute_semiring_copy_grad<
            Majorness::COL_MAJOR,
            Majorness::COL_MAJOR,
            Alignment::ALIGN_4,
            Alignment::ALIGN_4,
            KernelMatrixPackedType::UPPER_TRIANGLE
        > (
            m, n, k, l,
            A, ldA, batch_stride_A,
            B, ldB, batch_stride_B,
            C, ldC, batch_stride_C,
            h0, batch_stride_h0,
            h1, batch_stride_h1,
            h2, batch_stride_h2,
            h3, batch_stride_h3
        );
}
