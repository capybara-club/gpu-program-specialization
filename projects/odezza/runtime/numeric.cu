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
struct Slot { U64 data,divisor,count; unsigned kind; int a,b; float v0,v1; };
__device__ uint4 philox(U64 block,U64 seed,U64 domain) {
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
__device__ float unit(unsigned x) { return ((float)(x>>9)+0.5f)*0.00000011920928955078125f; }
extern "C" __global__ void grammar_rng(float *out,U64 count,U64 seed,U64 stream,unsigned normal) {
    for(U64 b=(U64)blockIdx.x*blockDim.x+threadIdx.x;b<(count+3)/4;b+=(U64)gridDim.x*blockDim.x) {
        uint4 bits=philox(b,seed,(stream<<1)|normal);
        float u[4]={unit(bits.x),unit(bits.y),unit(bits.z),unit(bits.w)};
        if(normal) for(unsigned p=0;p<4;p+=2) {
            float radius=sqrtf(-2.0f*logf(u[p])),s,c;
            sincosf(6.2831853071795864769f*u[p+1],&s,&c);
            u[p]=radius*c;u[p+1]=radius*s;
        }
        for(unsigned j=0;j<4;++j) if(4*b+j<count) out[4*b+j]=u[j];
    }
}
extern "C" __global__ void grammar_prelude(const Slot *slots,float *out,unsigned nslots,unsigned systems,U64 start,unsigned banks) {
    for(U64 item=(U64)blockIdx.x*blockDim.x+threadIdx.x;item<(U64)systems*banks;item+=(U64)gridDim.x*blockDim.x) {
        unsigned sys=item/banks;U64 bank=start+item%banks;float *row=out+item*nslots;
        // Dependencies are fixed/grid slots, and need not precede the RNG slot.
        for(unsigned s=0;s<nslots;++s) {
            Slot d=slots[sys*nslots+s];
            if(d.kind==0) row[s]=d.v0;
            else row[s]=((const float*)d.data)[(bank/d.divisor)%d.count];
        }
        for(unsigned s=0;s<nslots;++s) {
            Slot d=slots[sys*nslots+s];
            if(d.kind<2) continue;
            float u=row[s];
            float a=d.a<0?d.v0:row[d.a],b=d.b<0?d.v1:row[d.b];
            float v=u;
            if(d.kind==2) v=a*u+b;
            if(d.kind==3) v=(1.0f-u)*a+u*b;
            if(d.kind==4) v=a*u+b;
            if(d.kind==5) v=expf((1.0f-u)*logf(a)+u*logf(b));
            row[s]=v;
        }
    }
}
struct Winner { float mse; unsigned reserved; U64 index; };
extern "C" __global__ void grammar_consume(float *scores,const Winner *w,U64 count) {
    for(U64 i=(U64)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(U64)gridDim.x*blockDim.x)
        if(w[i].index!=~0ull) scores[w[i].index]=3.40282346638528859812e38f;
}
