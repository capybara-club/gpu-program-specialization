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
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(__AVX512F__)
#error "this peak benchmark requires AVX-512"
#endif

static __attribute__((noinline)) float run_fma_chains(unsigned long long iterations)
{
    const __m512 multiplier = _mm512_set1_ps(0.99999994f);
    const __m512 addend = _mm512_set1_ps(0.0000001f);
    __m512 x0 = _mm512_set1_ps(0.01f);
    __m512 x1 = _mm512_set1_ps(0.02f);
    __m512 x2 = _mm512_set1_ps(0.03f);
    __m512 x3 = _mm512_set1_ps(0.04f);
    __m512 x4 = _mm512_set1_ps(0.05f);
    __m512 x5 = _mm512_set1_ps(0.06f);
    __m512 x6 = _mm512_set1_ps(0.07f);
    __m512 x7 = _mm512_set1_ps(0.08f);
    __m512 x8 = _mm512_set1_ps(0.09f);
    __m512 x9 = _mm512_set1_ps(0.10f);
    __m512 x10 = _mm512_set1_ps(0.11f);
    __m512 x11 = _mm512_set1_ps(0.12f);
    __m512 x12 = _mm512_set1_ps(0.13f);
    __m512 x13 = _mm512_set1_ps(0.14f);
    __m512 x14 = _mm512_set1_ps(0.15f);
    __m512 x15 = _mm512_set1_ps(0.16f);

    for (unsigned long long iteration = 0; iteration < iterations; ++iteration) {
        x0 = _mm512_fmadd_ps(x0, multiplier, addend);
        x1 = _mm512_fmadd_ps(x1, multiplier, addend);
        x2 = _mm512_fmadd_ps(x2, multiplier, addend);
        x3 = _mm512_fmadd_ps(x3, multiplier, addend);
        x4 = _mm512_fmadd_ps(x4, multiplier, addend);
        x5 = _mm512_fmadd_ps(x5, multiplier, addend);
        x6 = _mm512_fmadd_ps(x6, multiplier, addend);
        x7 = _mm512_fmadd_ps(x7, multiplier, addend);
        x8 = _mm512_fmadd_ps(x8, multiplier, addend);
        x9 = _mm512_fmadd_ps(x9, multiplier, addend);
        x10 = _mm512_fmadd_ps(x10, multiplier, addend);
        x11 = _mm512_fmadd_ps(x11, multiplier, addend);
        x12 = _mm512_fmadd_ps(x12, multiplier, addend);
        x13 = _mm512_fmadd_ps(x13, multiplier, addend);
        x14 = _mm512_fmadd_ps(x14, multiplier, addend);
        x15 = _mm512_fmadd_ps(x15, multiplier, addend);
    }

    __m512 sum = _mm512_add_ps(x0, x1);
    sum = _mm512_add_ps(sum, x2);
    sum = _mm512_add_ps(sum, x3);
    sum = _mm512_add_ps(sum, x4);
    sum = _mm512_add_ps(sum, x5);
    sum = _mm512_add_ps(sum, x6);
    sum = _mm512_add_ps(sum, x7);
    sum = _mm512_add_ps(sum, x8);
    sum = _mm512_add_ps(sum, x9);
    sum = _mm512_add_ps(sum, x10);
    sum = _mm512_add_ps(sum, x11);
    sum = _mm512_add_ps(sum, x12);
    sum = _mm512_add_ps(sum, x13);
    sum = _mm512_add_ps(sum, x14);
    sum = _mm512_add_ps(sum, x15);
    return _mm512_reduce_add_ps(sum);
}

int main(int argc, char **argv)
{
    const unsigned int thread_count =
        argc > 1 ? (unsigned int)strtoul(argv[1], NULL, 10) : 12u;
    const unsigned long long iterations =
        argc > 2 ? strtoull(argv[2], NULL, 10) : 100000000ull;
    if (thread_count == 0u || iterations == 0ull) return 2;
    omp_set_dynamic(0);
    omp_set_num_threads((int)thread_count);

    float checksum = 0.0f;
#pragma omp parallel reduction(+:checksum)
    checksum += run_fma_chains(1000000ull);

    const double begin = omp_get_wtime();
#pragma omp parallel reduction(+:checksum)
    checksum += run_fma_chains(iterations);
    const double seconds = omp_get_wtime() - begin;
    const double operations =
        (double)thread_count * (double)iterations * 16.0 * 16.0 * 2.0;
    printf("isa=avx512 threads=%u iterations=%llu seconds=%.9f "
           "fp32_gflops=%.6f checksum=%.9g\n",
           thread_count, iterations, seconds, operations / seconds * 1.0e-9,
           checksum);
    return 0;
}
