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
"""LM-specific buffers and argument packing, kept out of the generic runner."""
import ctypes as C
import struct
from pipeline_runner import Launch
from pipeline_runner.cuda import D,U

class PreparedFit:
    def __init__(self,owner,function,input_path,width):
        self.owner=owner;self.closed=False
        if width not in (1,2,4,8,16,32):raise ValueError('Invalid subgroup width')
        with input_path.open('rb') as f:
            count,tr,points,steps,iterations,n,p=struct.unpack('<7I',f.read(28))
            if not (0<count<=1000000 and 0<tr<=points<=100000 and 0<n<=32 and 0<p<=32 and steps>0 and iterations>0):raise ValueError('Invalid input dimensions')
            def read(fmt,count):return list(struct.unpack('<'+fmt*count,f.read(4*count)))
            offsets=read('I',tr+1);times=read('f',points);ref=read('f',n*points);weights=read('f',n*points);starts=read('f',count*p)
            if f.read(1):raise ValueError('Trailing input')
        self.count=count;self.states=n;self.parameters=p;self.width=width
        self.inputs=[owner.buffer(C.c_float,len(starts),starts),owner.buffer(U,len(offsets),offsets),
                     owner.buffer(C.c_float,len(times),times),owner.buffer(C.c_float,len(ref),ref),owner.buffer(C.c_float,len(weights),weights)]
        self.outputs=[owner.buffer(C.c_float,count*p),owner.buffer(C.c_float,count),owner.buffer(C.c_float,count),
                      owner.buffer(U,count),owner.buffer(U,count),owner.buffer(U,count)]
        self.shared=4*(2*n*points+points+tr+1)
        args=(self.inputs[0][0],D(count),*[x[0] for x in self.inputs[1:]],U(tr),U(points),U(0),U(steps),U(iterations),U(8),C.c_float(1e-3),*[x[0] for x in self.outputs])
        self.launch=Launch(function,((count*width+31)//32,1,1),(32,1,1),self.shared,args,
                           uploads=tuple(self.inputs),downloads=tuple(self.outputs),keepalive=(owner,self),
                           writes=tuple(x[0].value for x in self.inputs+self.outputs),label=f'LM n={n} p={p} width={width}')
        self.resources=owner.driver.resources(function,self.shared)
    def values(self):
        if self.closed:raise RuntimeError('Fit buffers have been released')
        if self.launch._active:raise RuntimeError('Wait for completion before reading results')
        return {k:list(b[2]) for k,b in zip(['coefficients','initial_mse','mse','iterations','accepted','attempts'],self.outputs)}
    def close(self):
        if self.closed:return
        if self.launch._active:raise RuntimeError('Cannot free an in-flight fit')
        for b in reversed(self.inputs+self.outputs):
            for name,value in [('cuMemFree_v2',b[0]),('cuMemFreeHost',b[1])]:
                self.owner.gpu.call(name,value);self.owner.gpu.resources.remove((name,value))
        self.closed=True
