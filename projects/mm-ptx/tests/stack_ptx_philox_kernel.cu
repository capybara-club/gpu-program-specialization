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

#include <stdint.h>

extern "C"
__global__
void
kernel(
    uint32_t* out_u32,
    uint32_t* ref_u32,
    float* out_uniform,
    float* ref_uniform,
    float* out_normal,
    float* ref_normal
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandStatePhilox4_32_10_t state = philox_init(0, 1234, tid);
    curandStatePhilox4_32_10_t state_native = state;

    uint32_t u32_x = 0;
    uint32_t u32_y = 0;
    uint32_t u32_z = 0;
    uint32_t u32_w = 0;

    float uniform_x = 0.0f;
    float uniform_y = 0.0f;
    float uniform_z = 0.0f;
    float uniform_w = 0.0f;

    float normal_x = 0.0f;
    float normal_y = 0.0f;
    float normal_z = 0.0f;
    float normal_w = 0.0f;

    for (int i = 0; i < 2; ++i) {
        int base = i * 4;

        uint4 native_u32 = philox_curand4(&state_native);
        ref_u32[base + 0] = native_u32.x;
        ref_u32[base + 1] = native_u32.y;
        ref_u32[base + 2] = native_u32.z;
        ref_u32[base + 3] = native_u32.w;

        PTX_INJECT("philox_u32",
            PTX_PHILOX(state),
            PTX_OUT(U32, u32_x, u32_x),
            PTX_OUT(U32, u32_y, u32_y),
            PTX_OUT(U32, u32_z, u32_z),
            PTX_OUT(U32, u32_w, u32_w)
        );

        out_u32[base + 0] = u32_x;
        out_u32[base + 1] = u32_y;
        out_u32[base + 2] = u32_z;
        out_u32[base + 3] = u32_w;

        float4 native_uniform = philox_curand_uniform4(&state_native);
        ref_uniform[base + 0] = native_uniform.x;
        ref_uniform[base + 1] = native_uniform.y;
        ref_uniform[base + 2] = native_uniform.z;
        ref_uniform[base + 3] = native_uniform.w;

        PTX_INJECT("philox_uniform",
            PTX_PHILOX(state),
            PTX_OUT(F32, uniform_x, uniform_x),
            PTX_OUT(F32, uniform_y, uniform_y),
            PTX_OUT(F32, uniform_z, uniform_z),
            PTX_OUT(F32, uniform_w, uniform_w)
        );

        out_uniform[base + 0] = uniform_x;
        out_uniform[base + 1] = uniform_y;
        out_uniform[base + 2] = uniform_z;
        out_uniform[base + 3] = uniform_w;

        float4 native_normal = philox_curand_normal4(&state_native);
        ref_normal[base + 0] = native_normal.x;
        ref_normal[base + 1] = native_normal.y;
        ref_normal[base + 2] = native_normal.z;
        ref_normal[base + 3] = native_normal.w;

        PTX_INJECT("philox_normal",
            PTX_PHILOX(state),
            PTX_OUT(F32, normal_x, normal_x),
            PTX_OUT(F32, normal_y, normal_y),
            PTX_OUT(F32, normal_z, normal_z),
            PTX_OUT(F32, normal_w, normal_w)
        );

        out_normal[base + 0] = normal_x;
        out_normal[base + 1] = normal_y;
        out_normal[base + 2] = normal_z;
        out_normal[base + 3] = normal_w;
    }
}
