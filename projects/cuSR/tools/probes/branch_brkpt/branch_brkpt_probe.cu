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
#include <stdint.h>

#include "../../../kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh"

#define CUSR_BPT_1 "brkpt;\n\t"
#define CUSR_BPT_2 CUSR_BPT_1 CUSR_BPT_1
#define CUSR_BPT_4 CUSR_BPT_2 CUSR_BPT_2
#define CUSR_BPT_8 CUSR_BPT_4 CUSR_BPT_4
#define CUSR_BPT_16 CUSR_BPT_8 CUSR_BPT_8
#define CUSR_BPT_32 CUSR_BPT_16 CUSR_BPT_16
#define CUSR_BPT_64 CUSR_BPT_32 CUSR_BPT_32

extern "C"
__global__
void cusr_branch_skip_004(float* out, float x, uint32_t flag)
{
    float y = x;
    asm volatile(
        "{\n\t"
        ".reg .pred %%p;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@%%p bra cusr_branch_skip_004_trap;\n\t"
        "add.rn.ftz.f32 %0, %0, 0f3f800000;\n\t"
        "bra cusr_branch_skip_004_done;\n\t"
        "cusr_branch_skip_004_trap:\n\t"
        CUSR_BPT_4
        "cusr_branch_skip_004_done:\n\t"
        "}\n"
        : "+f"(y)
        : "r"(flag)
        : "memory");
    out[threadIdx.x] = y;
}

extern "C"
__global__
void cusr_branch_skip_016(float* out, float x, uint32_t flag)
{
    float y = x;
    asm volatile(
        "{\n\t"
        ".reg .pred %%p;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@%%p bra cusr_branch_skip_016_trap;\n\t"
        "add.rn.ftz.f32 %0, %0, 0f40000000;\n\t"
        "bra cusr_branch_skip_016_done;\n\t"
        "cusr_branch_skip_016_trap:\n\t"
        CUSR_BPT_16
        "cusr_branch_skip_016_done:\n\t"
        "}\n"
        : "+f"(y)
        : "r"(flag)
        : "memory");
    out[threadIdx.x] = y;
}

extern "C"
__global__
void cusr_branch_skip_064(float* out, float x, uint32_t flag)
{
    float y = x;
    asm volatile(
        "{\n\t"
        ".reg .pred %%p;\n\t"
        "setp.ne.u32 %%p, %1, 0;\n\t"
        "@%%p bra cusr_branch_skip_064_trap;\n\t"
        "add.rn.ftz.f32 %0, %0, 0f40400000;\n\t"
        "bra cusr_branch_skip_064_done;\n\t"
        "cusr_branch_skip_064_trap:\n\t"
        CUSR_BPT_64
        "cusr_branch_skip_064_done:\n\t"
        "}\n"
        : "+f"(y)
        : "r"(flag)
        : "memory");
    out[threadIdx.x] = y;
}

extern "C"
__global__
void cusr_brkpt_site_anchor(float* out, float x)
{
    float y = x;
    asm volatile(
        "{\n\t"
        "brkpt;\n\t"
        "add.rn.ftz.f32 %0, %0, 0f7fc0ffee;\n\t"
        "brkpt;\n\t"
        "}\n"
        : "+f"(y)
        :
        : "memory");
    cusr_tile_static_mse_template_emit_brkpt_pad<16u>();
    out[threadIdx.x] = y;
}
