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

 /* Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
  * Copyright 2026 Charles Durham.  All rights reserved.
  *
  * This file includes substantial modifications from the original NVIDIA version.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#pragma once

#include <stdint.h>

#define QUALIFIERS static __forceinline__ __device__

#define PHILOX_W32_0   (0x9E3779B9)
#define PHILOX_W32_1   (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)

#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)

QUALIFIERS 
uint4 
_philox4x32round(
    uint4 ctr, 
    uint2 key
) {
   unsigned int hi0 = __umulhi(PHILOX_M4x32_0, ctr.x);
   unsigned int hi1 = __umulhi(PHILOX_M4x32_1, ctr.z);
   unsigned int lo0 = PHILOX_M4x32_0 * ctr.x;
   unsigned int lo1 = PHILOX_M4x32_1 * ctr.z;

   return {hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0};
}

QUALIFIERS 
uint4 
_curand_Philox4x32_10(
    uint4 c, 
    uint2 k
) {
   c = _philox4x32round(c, k);                           // 1 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 2
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 3 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 4 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 5 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 6 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 7 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 8 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 9 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 10
   return c;
}

QUALIFIERS 
float4 
_curand_uniform4(
    uint4 x
) {
    float4 y;
    y.x = x.x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.y = x.y * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.z = x.z * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.w = x.w * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    return y;
}

QUALIFIERS 
float2 
_curand_box_muller(
    unsigned int x, 
    unsigned int y
) {
    float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2);
    float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2);
    float s = sqrtf(-2.0f * logf(u));
    
    float2 result;
    result.x = __sinf(v) * s;
    result.y = __cosf(v) * s;
    return result;
}

struct curandStatePhilox4_32_10 {
   uint4 ctr;
   uint2 key;
};

typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

__device__
curandStatePhilox4_32_10_t
philox_init(
    uint64_t launch_id,
    uint64_t seed,
    uint32_t utid
) {
    uint4 ctr = {0, (unsigned int)launch_id, (unsigned int)(launch_id >> 32), utid};
    uint2 key = {(unsigned int)seed, (unsigned int)(seed>>32)};
    return {ctr, key};
}

QUALIFIERS 
uint4 
philox_curand4(
    curandStatePhilox4_32_10_t *state
) {    
    uint4 output = _curand_Philox4x32_10(state->ctr,state->key);
    state->ctr.x++;
    return output;
}

QUALIFIERS 
float4 
philox_curand_uniform4(
    curandStatePhilox4_32_10_t *state
) {
   return _curand_uniform4(philox_curand4(state));
}

QUALIFIERS 
float4 
philox_curand_normal4(
    curandStatePhilox4_32_10_t *state
) {
    float4 result;
    float2 _result;
    uint4 x = philox_curand4(state);
    _result = _curand_box_muller(x.x, x.y);
    result.x = _result.x;
    result.y = _result.y;
    _result = _curand_box_muller(x.z, x.w);
    result.z = _result.x;
    result.w = _result.y;
    return result;
}
