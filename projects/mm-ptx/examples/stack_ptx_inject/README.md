<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Stack PTX Inject Examples
These examples fuse the Stack PTX and PTX Inject systems together to get fully functioning ptx kernels that run.

## 00_simple
Takes two inputs (in and mod) and one output from PTX Inject, uses Stack PTX to form and run three kernels.
* one that uses add.ftz.f32 to add the two inputs
* one that uses mul.ftz.f32 to multiply the two inputs
* one that mixes sin, cos, mul and add

## 01_gemm
Uses a f32 simt CuTe kernel from cutlass as a base. Adds separate injects for the multiply, accumulate and epilogue steps inside the kernel. Cmake processes the cuda with the ptxinject cli tool and nvcc is called on the output by cmake as well. The example then uses Stack PTX to change the kernel into a mma, L1 norm and an L2 norm operation, checking all three versions against their respective cpu versions.

## 02_bulk_rand_gemm
This example also uses a f32 simt CuTe kernel from Cutlass as a base. It also:
* Uses Stack PTX to generate hundreds of permutations of different multiply, accumulate and epilogue
operations in place of the usual mma. 
* Uses PTX Inject to generate PTX for each of these kernels
* Uses openMP for cpu parallel compilation with the nvPTXCompiler API to generate SASS.
* Uses nvJitLink to merge the sass/cubin into one cubin.
* Loads the cubin as a module and runs all kernels on Device:0.

The output of other runs are in the 02_bulk_rand_gemm folder but this is a 1024 kernel run in a GH200 system:

```
Device(0, sm_90)

CPU threads (64 found, 64 using)

PTX generation (StackPTX + PtxInject - nvPTXCompiler)
  (1024 kernels, 64 cpu threads):
	0.003140 seconds
	326115 kernels/second
	3.066 microseconds/kernel
SASS compilation (StackPTX + PtxInject + nvPTXCompiler)
  (1024 kernels, 64 cpu threads):
	3.313440 seconds
	309 kernels/second
	3235.781 microseconds/kernel
nvJitLink (1 cpu thread, 1024 kernels): 0.045750 seconds

Ran Random Gemm Kernels (1024 kernels, 1 cuda stream, 0.027861 seconds)
```

Keep in mind recompiling `examples/stack_ptx_inject/02_bulk_rand_gemm/kernel.cu` once to PTX takes around 3 seconds!
```
time nvcc examples/stack_ptx_inject/02_bulk_rand_gemm/kernel.cu -ptx -o build/kernel.ptx -arch=sm_89 -I thirdparty/cutlass/include/

________________________________________________________
Executed in    3.20 secs    fish           external
   usr time    3.09 secs    0.00 micros    3.09 secs
   sys time    0.07 secs  508.00 micros    0.07 secs


```

## 03_mma_sync
An example of running a mma.sync f32 tf32 instruction. Shows using convert instructions to pop from the f32 stack to populate the tf32 stack.

## 04_half_type
An example of running both f16 and f16x2 types. Shows using the f16x2 as a datatype and then representing it as a vector of 2 f16 types.

## 04_philox
An example that injects a philox PRNG routine library and compares output against native curand calls.

## 05_philox_cpp
A C++ version of the philox stack_ptx_inject example using the C++ routine library header.
