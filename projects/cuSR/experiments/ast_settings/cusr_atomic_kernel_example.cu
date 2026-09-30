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
#include <math.h>

#define CUSR_EXAMPLE_ROWS 128u
#define CUSR_EXAMPLE_THREADS 128u

/*
 * output_sse must be cleared before launch. Launch one block for each 128-row
 * tile and allocate (128 * (num_columns | 1) + 128) floats of dynamic shared
 * memory.
 *
 * Four fixed ASTs are shown to keep the example readable. The fastest measured
 * specialization emits the same straight-line pattern for 32 ASTs.
 */
extern "C" __global__ __launch_bounds__(CUSR_EXAMPLE_THREADS)
void cusr_atomic_sse_example_f32(
    const float* __restrict__ X,
    const float* __restrict__ target,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t num_settings,
    float* __restrict__ output_sse)
{
    extern __shared__ float shared[];

    const uint32_t tid = threadIdx.x;
    const uint32_t shared_stride = num_columns | 1u;
    const size_t row_base = (size_t)blockIdx.x * CUSR_EXAMPLE_ROWS;
    const size_t owned_row = row_base + tid;
    const bool owned_row_valid = owned_row < num_rows;
    const uint32_t row_limit = row_base + CUSR_EXAMPLE_ROWS <= num_rows
        ? CUSR_EXAMPLE_ROWS
        : (row_base < num_rows ? (uint32_t)(num_rows - row_base) : 0u);
    float* const x_tile = shared;
    float* const target_tile = x_tile + (size_t)CUSR_EXAMPLE_ROWS * shared_stride;

    target_tile[tid] = owned_row_valid ? target[owned_row] : 0.0f;

    #pragma unroll 1
    for (uint32_t column = 0u; column < num_columns; ++column) {
        x_tile[(size_t)tid * shared_stride + column] = owned_row_valid
            ? X[(size_t)column * leading_dim + owned_row]
            : 0.0f;
    }

    __syncthreads();

    for (uint32_t setting_base = 0u; setting_base < num_settings; setting_base += CUSR_EXAMPLE_THREADS) {
        const uint32_t setting = setting_base + tid;

        if (setting < num_settings) {
            const uint8_t mask = leaf_masks[setting];
            const uint32_t* const words = leaf_words + (size_t)setting * leaf_words_stride;
            const uint32_t word0 = words[0];
            const uint32_t word1 = words[1];
            const uint32_t word2 = words[2];
            const uint32_t word3 = words[3];
            const uint32_t word4 = words[4];
            const uint32_t word5 = words[5];
            const uint32_t word6 = words[6];
            const uint32_t word7 = words[7];
            const float constant0 = __uint_as_float(word0);
            const float constant1 = __uint_as_float(word1);
            const float constant2 = __uint_as_float(word2);
            const float constant3 = __uint_as_float(word3);
            const float constant4 = __uint_as_float(word4);
            const float constant5 = __uint_as_float(word5);
            const float constant6 = __uint_as_float(word6);
            const float constant7 = __uint_as_float(word7);
            float sse0 = 0.0f;
            float sse1 = 0.0f;
            float sse2 = 0.0f;
            float sse3 = 0.0f;

            #pragma unroll 1
            for (uint32_t row = 0u; row < row_limit; ++row) {
                const float* const x_row = x_tile + (size_t)row * shared_stride;
                float value0 = constant0;
                float value1 = constant1;
                float value2 = constant2;
                float value3 = constant3;
                float value4 = constant4;
                float value5 = constant5;
                float value6 = constant6;
                float value7 = constant7;

                if ((mask & 0x01u) != 0u) value0 = x_row[word0];
                if ((mask & 0x02u) != 0u) value1 = x_row[word1];
                if ((mask & 0x04u) != 0u) value2 = x_row[word2];
                if ((mask & 0x08u) != 0u) value3 = x_row[word3];
                if ((mask & 0x10u) != 0u) value4 = x_row[word4];
                if ((mask & 0x20u) != 0u) value5 = x_row[word5];
                if ((mask & 0x40u) != 0u) value6 = x_row[word6];
                if ((mask & 0x80u) != 0u) value7 = x_row[word7];

                /* Four representative fixed ASTs. */
                const float prediction0 = (value0 + value1) * (value2 + value3) + fminf(value4, value5 + value6 * value7);
                const float prediction1 = fmaxf(value0 * value3, value1 + value6) + fminf(value2, value4 * value7) + value5;
                const float prediction2 = (value0 - value7) * (value1 + value4) + fmaxf(value2 * value5, value3 + value6);
                const float prediction3 = fminf(value0 + value2, value1 * value3) * fmaxf(value4 + value7, value5 * value6);
                const float y = target_tile[row];
                const float error0 = prediction0 - y;
                const float error1 = prediction1 - y;
                const float error2 = prediction2 - y;
                const float error3 = prediction3 - y;

                sse0 = error0 * error0 + sse0;
                sse1 = error1 * error1 + sse1;
                sse2 = error2 * error2 + sse2;
                sse3 = error3 * error3 + sse3;
            }

            atomicAdd(output_sse + (size_t)0u * num_settings + setting, sse0);
            atomicAdd(output_sse + (size_t)1u * num_settings + setting, sse1);
            atomicAdd(output_sse + (size_t)2u * num_settings + setting, sse2);
            atomicAdd(output_sse + (size_t)3u * num_settings + setting, sse3);
        }
    }
}

#undef CUSR_EXAMPLE_THREADS
#undef CUSR_EXAMPLE_ROWS
