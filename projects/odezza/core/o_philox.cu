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
typedef unsigned long long U64;
__device__ uint4 philox(U64 block, U64 seed, U64 domain) {
    uint4 c=make_uint4((unsigned)block,(unsigned)(block>>32),(unsigned)domain,(unsigned)(domain>>32));
    unsigned k0=(unsigned)seed,k1=(unsigned)(seed>>32);
    for(unsigned r=0;r<10;++r) {
        unsigned hi0=__umulhi(0xd2511f53u,c.x),lo0=0xd2511f53u*c.x;
        unsigned hi1=__umulhi(0xcd9e8d57u,c.z),lo1=0xcd9e8d57u*c.z;
        c=make_uint4(hi1^c.y^k0,lo1,hi0^c.w^k1,lo0);
        k0+=0x9e3779b9u;k1+=0xbb67ae85u;
    }
    return c;
}
__device__ float uniform(unsigned x) { return ((float)(x>>9)+0.5f)*0.00000011920928955078125f; }
template<bool Normal> __device__ void fill(float *out,U64 size,U64 seed,U64 stream,U64 base) {
    U64 blocks=(base%4+size+3)/4,first=base/4;
    for(U64 b=(U64)blockIdx.x*blockDim.x+threadIdx.x;b<blocks;b+=(U64)gridDim.x*blockDim.x) {
        uint4 bits=philox(first+b,seed,(stream<<1)|(Normal?1u:0u));
        float u[4]={uniform(bits.x),uniform(bits.y),uniform(bits.z),uniform(bits.w)};
        if(Normal) {
            for(unsigned p=0;p<4;p+=2) {
                float radius=sqrtf(-2.0f*logf(u[p])),s,c;
                sincosf(6.2831853071795864769f*u[p+1],&s,&c);
                u[p]=radius*c;u[p+1]=radius*s;
            }
        }
        for(unsigned j=0;j<4;++j) {
            U64 relative=b*4+j;
            if(relative>=base%4 && relative-base%4<size) out[relative-base%4]=u[j];
        }
    }
}
extern "C" __global__ void fill_uniform(float *out,U64 size,U64 seed,U64 stream,U64 base) { fill<false>(out,size,seed,stream,base); }
extern "C" __global__ void fill_normal(float *out,U64 size,U64 seed,U64 stream,U64 base) { fill<true>(out,size,seed,stream,base); }
