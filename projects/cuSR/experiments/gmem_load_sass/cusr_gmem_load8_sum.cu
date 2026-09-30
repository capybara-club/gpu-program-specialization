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
#include <stddef.h>
#include <stdint.h>

/*
 * SASS inspection probe. The brkpt intentionally traps if this unpatched
 * kernel is launched.
 *
 * X is column-major and leading_dimension is measured in float elements:
 *
 *     X[column * leading_dimension + row]
 */
extern "C"
__global__
void cusr_gmem_load8_sum_f32(
    const float* __restrict__ X,
    size_t leading_dimension,
    uint32_t num_rows,
    float* __restrict__ output)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }

    float value0 = X[(size_t)row + 0u * leading_dimension];
    float value1 = X[(size_t)row + 1u * leading_dimension];
    float value2 = X[(size_t)row + 2u * leading_dimension];
    float value3 = X[(size_t)row + 3u * leading_dimension];
    float value4 = X[(size_t)row + 4u * leading_dimension];
    float value5 = X[(size_t)row + 5u * leading_dimension];
    float value6 = X[(size_t)row + 6u * leading_dimension];
    float value7 = X[(size_t)row + 7u * leading_dimension];

    asm volatile(
        "brkpt;"
        : "+f"(value0), "+f"(value1), "+f"(value2), "+f"(value3),
          "+f"(value4), "+f"(value5), "+f"(value6), "+f"(value7)
        :
        : "memory");

    const float sum01 = value0 + value1;
    const float sum23 = value2 + value3;
    const float sum45 = value4 + value5;
    const float sum67 = value6 + value7;
    output[row] = (sum01 + sum23) + (sum45 + sum67);
}

/*
 * Four independently delimited load anchors. Each inline PTX block forces its
 * input load to complete before a recognizable marker FADD can execute.
 */
extern "C"
__global__
void cusr_gmem_marked_load4_f32(
    const float* __restrict__ X,
    size_t leading_dimension,
    uint32_t num_rows,
    float* __restrict__ output)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }

    float value0 = X[(size_t)row + 0u * leading_dimension];
    asm volatile(
        "add.rn.ftz.f32 %0, %0, 0f7fc0ffee;\n\t"
        "brkpt;"
        : "+f"(value0)
        :
        : "memory");

    float value1 = X[(size_t)row + 1u * leading_dimension];
    asm volatile(
        "add.rn.ftz.f32 %0, %0, 0f7fc0ffef;\n\t"
        "brkpt;"
        : "+f"(value1)
        :
        : "memory");

    float value2 = X[(size_t)row + 2u * leading_dimension];
    asm volatile(
        "add.rn.ftz.f32 %0, %0, 0f7fc0fff0;\n\t"
        "brkpt;"
        : "+f"(value2)
        :
        : "memory");

    float value3 = X[(size_t)row + 3u * leading_dimension];
    asm volatile(
        "add.rn.ftz.f32 %0, %0, 0f7fc0fff1;\n\t"
        "brkpt;"
        : "+f"(value3)
        :
        : "memory");

    output[row] = (value0 + value1) + (value2 + value3);
}

/*
 * A grouped load region followed by one brkpt and a grouped marker region.
 * This shape tests whether ptxas keeps all loads on one side of the boundary
 * while preserving a recognizable source register for every loaded value.
 */
extern "C"
__global__
void cusr_gmem_load4_brkpt_markers_f32(
    const float* __restrict__ X,
    size_t leading_dimension,
    uint32_t num_rows,
    float* __restrict__ output)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }

    float value0 = X[(size_t)row + 0u * leading_dimension];
    float value1 = X[(size_t)row + 1u * leading_dimension];
    float value2 = X[(size_t)row + 2u * leading_dimension];
    float value3 = X[(size_t)row + 3u * leading_dimension];

    asm volatile(
        "brkpt;\n\t"
        "add.rn.ftz.f32 %0, %0, 0f7fc0ffee;\n\t"
        "add.rn.ftz.f32 %1, %1, 0f7fc0ffef;\n\t"
        "add.rn.ftz.f32 %2, %2, 0f7fc0fff0;\n\t"
        "add.rn.ftz.f32 %3, %3, 0f7fc0fff1;"
        : "+f"(value0), "+f"(value1), "+f"(value2), "+f"(value3)
        :
        : "memory");

    output[row] = (value0 + value1) + (value2 + value3);
}

/* Thirty-two simultaneously live column loads around one marker block. */
extern "C"
__global__
void cusr_gmem_load32_brkpt_markers_f32(
    const float* __restrict__ X,
    size_t leading_dimension,
    uint32_t num_rows,
    float* __restrict__ output)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }

    float value0 = X[(size_t)row + 0u * leading_dimension];
    float value1 = X[(size_t)row + 1u * leading_dimension];
    float value2 = X[(size_t)row + 2u * leading_dimension];
    float value3 = X[(size_t)row + 3u * leading_dimension];
    float value4 = X[(size_t)row + 4u * leading_dimension];
    float value5 = X[(size_t)row + 5u * leading_dimension];
    float value6 = X[(size_t)row + 6u * leading_dimension];
    float value7 = X[(size_t)row + 7u * leading_dimension];
    float value8 = X[(size_t)row + 8u * leading_dimension];
    float value9 = X[(size_t)row + 9u * leading_dimension];
    float value10 = X[(size_t)row + 10u * leading_dimension];
    float value11 = X[(size_t)row + 11u * leading_dimension];
    float value12 = X[(size_t)row + 12u * leading_dimension];
    float value13 = X[(size_t)row + 13u * leading_dimension];
    float value14 = X[(size_t)row + 14u * leading_dimension];
    float value15 = X[(size_t)row + 15u * leading_dimension];
    float value16 = X[(size_t)row + 16u * leading_dimension];
    float value17 = X[(size_t)row + 17u * leading_dimension];
    float value18 = X[(size_t)row + 18u * leading_dimension];
    float value19 = X[(size_t)row + 19u * leading_dimension];
    float value20 = X[(size_t)row + 20u * leading_dimension];
    float value21 = X[(size_t)row + 21u * leading_dimension];
    float value22 = X[(size_t)row + 22u * leading_dimension];
    float value23 = X[(size_t)row + 23u * leading_dimension];
    float value24 = X[(size_t)row + 24u * leading_dimension];
    float value25 = X[(size_t)row + 25u * leading_dimension];
    float value26 = X[(size_t)row + 26u * leading_dimension];
    float value27 = X[(size_t)row + 27u * leading_dimension];
    float value28 = X[(size_t)row + 28u * leading_dimension];
    float value29 = X[(size_t)row + 29u * leading_dimension];
    float value30 = X[(size_t)row + 30u * leading_dimension];
    float value31 = X[(size_t)row + 31u * leading_dimension];

    asm volatile(
        "brkpt;\n\t"
        "add.rn.ftz.f32 %0, %0, 0f7fc0ffee;\n\t"
        "add.rn.ftz.f32 %1, %1, 0f7fc0ffef;\n\t"
        "add.rn.ftz.f32 %2, %2, 0f7fc0fff0;\n\t"
        "add.rn.ftz.f32 %3, %3, 0f7fc0fff1;\n\t"
        "add.rn.ftz.f32 %4, %4, 0f7fc0fff2;\n\t"
        "add.rn.ftz.f32 %5, %5, 0f7fc0fff3;\n\t"
        "add.rn.ftz.f32 %6, %6, 0f7fc0fff4;\n\t"
        "add.rn.ftz.f32 %7, %7, 0f7fc0fff5;\n\t"
        "add.rn.ftz.f32 %8, %8, 0f7fc0fff6;\n\t"
        "add.rn.ftz.f32 %9, %9, 0f7fc0fff7;\n\t"
        "add.rn.ftz.f32 %10, %10, 0f7fc0fff8;\n\t"
        "add.rn.ftz.f32 %11, %11, 0f7fc0fff9;\n\t"
        "add.rn.ftz.f32 %12, %12, 0f7fc0fffa;\n\t"
        "add.rn.ftz.f32 %13, %13, 0f7fc0fffb;\n\t"
        "add.rn.ftz.f32 %14, %14, 0f7fc0fffc;\n\t"
        "add.rn.ftz.f32 %15, %15, 0f7fc0fffd;\n\t"
        "add.rn.ftz.f32 %16, %16, 0f7fc0fffe;\n\t"
        "add.rn.ftz.f32 %17, %17, 0f7fc0ffff;\n\t"
        "add.rn.ftz.f32 %18, %18, 0f7fc10000;\n\t"
        "add.rn.ftz.f32 %19, %19, 0f7fc10001;\n\t"
        "add.rn.ftz.f32 %20, %20, 0f7fc10002;\n\t"
        "add.rn.ftz.f32 %21, %21, 0f7fc10003;\n\t"
        "add.rn.ftz.f32 %22, %22, 0f7fc10004;\n\t"
        "add.rn.ftz.f32 %23, %23, 0f7fc10005;\n\t"
        "add.rn.ftz.f32 %24, %24, 0f7fc10006;\n\t"
        "add.rn.ftz.f32 %25, %25, 0f7fc10007;\n\t"
        "add.rn.ftz.f32 %26, %26, 0f7fc10008;\n\t"
        "add.rn.ftz.f32 %27, %27, 0f7fc10009;\n\t"
        "add.rn.ftz.f32 %28, %28, 0f7fc1000a;\n\t"
        "add.rn.ftz.f32 %29, %29, 0f7fc1000b;\n\t"
        "add.rn.ftz.f32 %30, %30, 0f7fc1000c;\n\t"
        "add.rn.ftz.f32 %31, %31, 0f7fc1000d;\n\t"
        "brkpt;"
        : "+f"(value0), "+f"(value1), "+f"(value2), "+f"(value3),
          "+f"(value4), "+f"(value5), "+f"(value6), "+f"(value7),
          "+f"(value8), "+f"(value9), "+f"(value10), "+f"(value11),
          "+f"(value12), "+f"(value13), "+f"(value14), "+f"(value15),
          "+f"(value16), "+f"(value17), "+f"(value18), "+f"(value19),
          "+f"(value20), "+f"(value21), "+f"(value22), "+f"(value23),
          "+f"(value24), "+f"(value25), "+f"(value26), "+f"(value27),
          "+f"(value28), "+f"(value29), "+f"(value30), "+f"(value31)
        :
        : "memory");

    const float sum0 = value0 + value1;
    const float sum1 = value2 + value3;
    const float sum2 = value4 + value5;
    const float sum3 = value6 + value7;
    const float sum4 = value8 + value9;
    const float sum5 = value10 + value11;
    const float sum6 = value12 + value13;
    const float sum7 = value14 + value15;
    const float sum8 = value16 + value17;
    const float sum9 = value18 + value19;
    const float sum10 = value20 + value21;
    const float sum11 = value22 + value23;
    const float sum12 = value24 + value25;
    const float sum13 = value26 + value27;
    const float sum14 = value28 + value29;
    const float sum15 = value30 + value31;
    const float sum16 = sum0 + sum1;
    const float sum17 = sum2 + sum3;
    const float sum18 = sum4 + sum5;
    const float sum19 = sum6 + sum7;
    const float sum20 = sum8 + sum9;
    const float sum21 = sum10 + sum11;
    const float sum22 = sum12 + sum13;
    const float sum23 = sum14 + sum15;
    const float sum24 = sum16 + sum17;
    const float sum25 = sum18 + sum19;
    const float sum26 = sum20 + sum21;
    const float sum27 = sum22 + sum23;
    output[row] = (sum24 + sum25) + (sum26 + sum27);
}
