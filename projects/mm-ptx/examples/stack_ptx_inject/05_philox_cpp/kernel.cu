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
#include <ptx_inject.h>
#include <philox.cuh>

#include <stdio.h>
#include <stdint.h>

#define PTX_TYPE_INFO_PHILOX PTX_TYPES_DESC(u32, u32, r, ID)

#define PTX_PHILOX(state)                           \
    PTX_IN (PHILOX, philox_key_x, state.key.x),     \
    PTX_IN (PHILOX, philox_key_y, state.key.y),     \
    PTX_MOD (PHILOX, philox_ctr_x, state.ctr.x),    \
    PTX_IN (PHILOX, philox_ctr_y, state.ctr.y),     \
    PTX_IN (PHILOX, philox_ctr_z, state.ctr.z),     \
    PTX_IN (PHILOX, philox_ctr_w, state.ctr.w)

extern "C"
__global__
void
kernel(
    float* out
) {
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t state = philox_init(0,1234,tid);

    float4 output;

    output = philox_curand_normal4(&state);
    output = philox_curand_normal4(&state);
    state.ctr.x = 0;

    PTX_INJECT("func",
        PTX_PHILOX(state),
        PTX_OUT (F32, x, x),
        PTX_OUT (F32, y, y),
        PTX_OUT (F32, z, z),
        PTX_OUT (F32, w, w)
    );

    printf("%u\n", state.ctr.x);
    printf("%f:%f\n", x, output.x);
    printf("%f:%f\n", y, output.y);
    printf("%f:%f\n", z, output.z);
    printf("%f:%f\n", w, output.w);

    output = philox_curand_normal4(&state);
    output = philox_curand_normal4(&state);
    state.ctr.x = 2;

    PTX_INJECT("func",
        PTX_PHILOX(state),
        PTX_OUT (F32, x, x),
        PTX_OUT (F32, y, y),
        PTX_OUT (F32, z, z),
        PTX_OUT (F32, w, w)
    );

    printf("%u\n", state.ctr.x);
    printf("%f:%f\n", x, output.x);
    printf("%f:%f\n", y, output.y);
    printf("%f:%f\n", z, output.z);
    printf("%f:%f\n", w, output.w);

    *out = z;
}
