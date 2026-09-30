#!/usr/bin/env python3
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
"""Check the actual CUDA reverse derivatives against central differences.

Uses the installed driver/NVRTC, not nvcc or a Python GPU package. No installs.
"""
import ctypes as C
import math
import re
from pathlib import Path

class Instruction(C.Structure):
    _fields_=[('op',C.c_uint),('a',C.c_uint),('b',C.c_uint),('value',C.c_float)]
class Program(C.Structure):
    _fields_=[('count',C.c_uint),('parameters',C.c_uint),('model',C.c_uint),('permutation',C.c_uint),
              ('ids',C.c_uint*8),('center',C.c_float*8),('code',Instruction*63)]

def check(code):
    if code: raise RuntimeError(f'CUDA/NVRTC error {code}')

def main():
    root=Path(__file__).resolve().parents[1]
    nv=C.CDLL('/usr/local/cuda/lib64/libnvrtc.so');cu=C.CDLL('libcuda.so.1')
    check(cu.cuInit(0));device=C.c_int();check(cu.cuDeviceGet(C.byref(device),0))
    ctx=C.c_void_p();check(cu.cuDevicePrimaryCtxRetain(C.byref(ctx),device));check(cu.cuCtxSetCurrent(ctx))
    major=C.c_int();minor=C.c_int()
    check(cu.cuDeviceGetAttribute(C.byref(major),75,device));check(cu.cuDeviceGetAttribute(C.byref(minor),76,device))
    source=(root/'kernels/lm.cu').read_bytes()+b'''
extern "C" __global__ void lm_probe(const SRLMProgram *p, float *out) {
  unsigned i=threadIdx.x; if(i>=16)return;
  float theta[8]={.7f,1.1f},x[1]={.3f},g[8],v;
  bool ok=evaluate_n<8>(p[i],theta,x,1,0,v,g);
  out[i*6]=ok?v:0;out[i*6+5]=ok?1:0;
  if(!ok)return;
  for(unsigned j=0;j<2;++j){
    out[i*6+1+j]=g[j];
    float old=theta[j],h=.001f*(1+fabsf(old)),a,b,temp[8];
    theta[j]=old+h;bool aok=evaluate_n<8>(p[i],theta,x,1,0,a,temp);
    theta[j]=old-h;bool bok=evaluate_n<8>(p[i],theta,x,1,0,b,temp);theta[j]=old;
    out[i*6+3+j]=aok&&bok?(a-b)/(2*h):__int_as_float(0x7fc00000);
  }
}
'''
    h=(root/'src/lm_layout.h').read_bytes();headers=(C.c_char_p*1)(h);names=(C.c_char_p*1)(b'lm_layout.h')
    compiler=C.c_void_p();check(nv.nvrtcCreateProgram(C.byref(compiler),source,b'lm_probe.cu',1,headers,names))
    opts=(C.c_char_p*5)(f'--gpu-architecture=sm_{major.value}{minor.value}'.encode(),b'--std=c++14',b'--use_fast_math',b'--fmad=false',b'--ptxas-options=-v')
    status=nv.nvrtcCompileProgram(compiler,5,opts)
    size=C.c_size_t();check(nv.nvrtcGetProgramLogSize(compiler,C.byref(size)));log=C.create_string_buffer(size.value);check(nv.nvrtcGetProgramLog(compiler,log))
    print(log.value.decode(),end='');check(status)
    for name in ['lm_statistics_1','lm_statistics_2','lm_statistics_4','lm_statistics_8','lm_step']:
        report=re.search(r"Function properties for "+name+r"\n(.*?)(?=ptxas info    : Used)",log.value.decode(),re.S)
        # NVRTC's compiler cache may return an empty log. The loaded-function
        # resource check below is mandatory whether compilation was cold or warm.
        if report:assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in report.group(1),(name,log.value.decode())
    check(nv.nvrtcGetCUBINSize(compiler,C.byref(size)));blob=C.create_string_buffer(size.value);check(nv.nvrtcGetCUBIN(compiler,blob));check(nv.nvrtcDestroyProgram(C.byref(compiler)))
    module=C.c_void_p();function=C.c_void_p();check(cu.cuModuleLoadData(C.byref(module),blob));check(cu.cuModuleGetFunction(C.byref(function),module,b'lm_probe'))
    for name in ['lm_statistics_1','lm_statistics_2','lm_statistics_4','lm_statistics_8','lm_step']:
        f=C.c_void_p();local=C.c_int();registers=C.c_int()
        check(cu.cuModuleGetFunction(C.byref(f),module,name.encode()))
        check(cu.cuFuncGetAttribute(C.byref(local),3,f));check(cu.cuFuncGetAttribute(C.byref(registers),4,f))
        assert local.value==0,(name,local.value)
        print(f'{name}: registers={registers.value}, local_bytes={local.value}')
    programs=(Program*16)();binary=[0x85,0x86,0x87,0x88,0x8d,0x8e];unary=[0x89,0x8a,0x8c,0x90,0x91,0x95,0xba,0xbb]
    for i,op in enumerate(binary+unary):
        p=programs[i];p.parameters=2
        p.code[0]=Instruction(2,0,0,0)
        if op in binary:
            p.code[1]=Instruction(2,1,0,0);p.code[2]=Instruction(op,0,1,0);p.count=3
        else:p.code[1]=Instruction(op,0,0,0);p.count=2
    p=programs[14];p.parameters=1;p.count=5
    # Same parameter appearing twice, through two nonlinear paths.
    p.code[0]=Instruction(2,0,0,0);p.code[1]=Instruction(0x90,0,0,0)
    p.code[2]=Instruction(2,0,0,0);p.code[3]=Instruction(0xbb,2,0,0);p.code[4]=Instruction(0x85,1,3,0)
    p=programs[15];p.count=2;p.parameters=0;p.code[0]=Instruction(0,0,0,-1);p.code[1]=Instruction(0x8a,0,0,0)
    dp=C.c_uint64();do=C.c_uint64();output=(C.c_float*96)()
    check(cu.cuMemAlloc_v2(C.byref(dp),C.c_size_t(C.sizeof(programs))));check(cu.cuMemAlloc_v2(C.byref(do),C.c_size_t(C.sizeof(output))))
    check(cu.cuMemcpyHtoD_v2(dp,programs,C.c_size_t(C.sizeof(programs))))
    args=(C.c_void_p*2)(C.addressof(dp),C.addressof(do))
    check(cu.cuLaunchKernel(function,1,1,1,32,1,1,0,None,args,None))
    check(cu.cuMemcpyDtoH_v2(output,do,C.c_size_t(C.sizeof(output))))
    for i in range(15):
        assert output[i*6+5]==1,(i,list(output[i*6:i*6+6]))
        for j in range(2):
            actual,expected=output[i*6+1+j],output[i*6+3+j]
            assert math.isfinite(actual) and abs(actual-expected)<.002*max(1,abs(expected)),(i,j,actual,expected)
    assert output[15*6+5]==0
    check(cu.cuMemFree_v2(dp));check(cu.cuMemFree_v2(do));check(cu.cuModuleUnload(module));check(cu.cuDevicePrimaryCtxRelease_v2(device))
    print('GPU derivatives passed: 14 operators, shared nonlinear parameter, invalid-domain rejection')

if __name__=='__main__':main()
