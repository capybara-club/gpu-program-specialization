# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Check installed CUB block primitives with the repository's NVRTC route."""
import ctypes as C
import time
n=C.CDLL('/usr/local/cuda/lib64/libnvrtc.so')
p=C.c_void_p()
n.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
n.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
n.nvrtcGetProgramLogSize.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
n.nvrtcGetProgramLog.argtypes=[C.c_void_p,C.c_void_p]
s=b'''#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
struct Item {float f; unsigned flags; unsigned long long index;};
struct Input {using value_type=Item; const float *p; __device__ Item operator[](int i) const{return {p[i],0,(unsigned long long)i};}};
extern "C" __global__ void test(float *p) {
  using Load=cub::BlockLoad<Item,256,8,cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Sort=cub::BlockRadixSort<float,256,8,unsigned long long>;
  __shared__ union {Load::TempStorage load; Sort::TempStorage sort;} s;
  Item v[8]; float keys[8]; unsigned long long values[8];
  Load(s.load).Load(Input{p},v,2048,Item{}); __syncthreads();
  for(int i=0;i<8;i++){keys[i]=v[i].f; values[i]=v[i].index;}
  Sort(s.sort).Sort(keys,values);p[threadIdx.x]=keys[0];
}'''
assert n.nvrtcCreateProgram(C.byref(p),s,b'cub_probe.cu',0,None,None)==0
opts=(C.c_char_p*4)(b'--std=c++17',b'--gpu-architecture=sm_120',b'--include-path=/usr/local/cuda/include/cccl',b'--include-path=/usr/local/cuda/include')
t=time.monotonic();r=n.nvrtcCompileProgram(p,4,opts)
z=C.c_size_t();n.nvrtcGetProgramLogSize(p,C.byref(z));log=C.create_string_buffer(z.value);n.nvrtcGetProgramLog(p,log)
print(log.value.decode(),flush=True);print('result',r,'seconds',time.monotonic()-t,flush=True)
n.nvrtcDestroyProgram(C.byref(p))
raise SystemExit(r)
