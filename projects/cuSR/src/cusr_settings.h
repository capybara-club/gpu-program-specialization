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
#ifndef CUSR_SETTINGS_H_INCLUDED
#define CUSR_SETTINGS_H_INCLUDED

#if defined(__CUDACC_RTC__)
typedef unsigned int CusrU32;
typedef unsigned long long CusrSize;
#else
#include <stddef.h>
#include <stdint.h>
typedef uint32_t CusrU32;
typedef size_t CusrSize;
#endif

#ifndef CUSR_NUM_INPUTS
#define CUSR_NUM_INPUTS 8u
#endif

#ifndef CUSR_ROW_TILE_ROWS
#define CUSR_ROW_TILE_ROWS 128u
#endif

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define CUSR_HOST_DEVICE __host__ __device__
#else
#define CUSR_HOST_DEVICE
#endif

typedef struct CusrSettingF32 {
    CusrU32 column_mask;
    CusrU32 column_indices[CUSR_NUM_INPUTS];
    float constants[CUSR_NUM_INPUTS];
} CusrSettingF32;

static CUSR_HOST_DEVICE inline float
cusr_setting_load_f32(
    const float* x,
    CusrU32 row,
    CusrU32 leading_dim,
    const CusrSettingF32* setting,
    CusrU32 input_idx
) {
    const CusrU32 bit = 1u << input_idx;

    if ((setting->column_mask & bit) != 0u) {
        return x[(CusrSize)setting->column_indices[input_idx] * (CusrSize)leading_dim + (CusrSize)row];
    }

    return setting->constants[input_idx];
}

#endif /* CUSR_SETTINGS_H_INCLUDED */
