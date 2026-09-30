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
#pragma once
// One directional derivative, propagated analytically through the fixed RHS.
struct Dual {
    float v, d;
    __device__ __forceinline__ Dual(float value=0, float derivative=0):v(value),d(derivative){}
};
__device__ __forceinline__ Dual operator+(Dual a,Dual b){return {a.v+b.v,a.d+b.d};}
__device__ __forceinline__ Dual operator-(Dual a,Dual b){return {a.v-b.v,a.d-b.d};}
__device__ __forceinline__ Dual operator-(Dual a){return {-a.v,-a.d};}
__device__ __forceinline__ Dual operator*(Dual a,Dual b){return {a.v*b.v,a.d*b.v+a.v*b.d};}
__device__ __forceinline__ Dual operator/(Dual a,Dual b){float v=a.v/b.v;return {v,(a.d-v*b.d)/b.v};}
__device__ __forceinline__ Dual sin(Dual a){return {sinf(a.v),cosf(a.v)*a.d};}
__device__ __forceinline__ Dual cos(Dual a){return {cosf(a.v),-sinf(a.v)*a.d};}
__device__ __forceinline__ Dual tanh(Dual a){float v=tanhf(a.v);return {v,(1-v*v)*a.d};}
__device__ __forceinline__ Dual exp(Dual a){float v=expf(a.v);return {v,v*a.d};}
