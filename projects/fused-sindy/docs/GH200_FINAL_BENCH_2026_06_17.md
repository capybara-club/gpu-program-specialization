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

# GH200 Final Bench - 2026-06-17

## Environment

Hostname: 192-222-50-14

### nvidia-smi

~~~text
Wed Jun 17 18:55:31 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GH200 480GB             On  |   00000000:DD:00.0 Off |                    0 |
| N/A   34C    P0            126W /  700W |    1130MiB /  97871MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            7201      C   ..._reserved_cubin_runtime_bench        554MiB |
|    0   N/A  N/A            7793      C   ..._reserved_cubin_runtime_bench        554MiB |
+-----------------------------------------------------------------------------------------+
~~~

### CPU

~~~text
Architecture:                            aarch64
CPU op-mode(s):                          64-bit
Byte Order:                              Little Endian
CPU(s):                                  64
On-line CPU(s) list:                     0-63
Vendor ID:                               ARM
Model name:                              Neoverse-V2
Model:                                   0
Thread(s) per core:                      1
Core(s) per socket:                      64
Socket(s):                               1
Stepping:                                r0p0
BogoMIPS:                                2000.00
Flags:                                   fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti
NUMA node(s):                            9
NUMA node0 CPU(s):                       0-63
NUMA node1 CPU(s):                       
NUMA node2 CPU(s):                       
NUMA node3 CPU(s):                       
NUMA node4 CPU(s):                       
NUMA node5 CPU(s):                       
NUMA node6 CPU(s):                       
NUMA node7 CPU(s):                       
NUMA node8 CPU(s):                       
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; CSV2, BHB
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected
~~~

### Toolchain

~~~text
cmake version 3.28.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:26:18_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
Python 3.12.3
~~~

## Tests

~~~text
Internal ctest changing into directory: /home/ubuntu/fused-sindy/build-verify-gh200-current
Test project /home/ubuntu/fused-sindy/build-verify-gh200-current
    Start 1: binary_ast_to_stack_ptx_test
1/5 Test #1: binary_ast_to_stack_ptx_test ......................   Passed    0.01 sec
    Start 2: implicit_sindy_ast_compile_api_test
2/5 Test #2: implicit_sindy_ast_compile_api_test ...............   Passed   18.00 sec
    Start 3: implicit_sindy_gram_cpu_reference_test
3/5 Test #3: implicit_sindy_gram_cpu_reference_test ............   Passed   19.62 sec
    Start 4: implicit_sindy_ast_columns_cpu_reference_test
4/5 Test #4: implicit_sindy_ast_columns_cpu_reference_test .....   Passed    0.67 sec
    Start 5: implicit_feature_ridge_solve_cpu_reference_test
5/5 Test #5: implicit_feature_ridge_solve_cpu_reference_test ...   Passed    0.40 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =  38.70 sec
~~~

## Compile Hot Path

~~~bash
./build-verify-gh200-current/implicit_sindy_ast_compile_api_bench --modules 512 --kernels 4 --workers 1,2,4,8,16,32,64 --mode polynomial --repeats 1 --warmup 0 --workspace-mb 256 --sm sm_90
~~~

~~~text
implicit_sindy_ast_compile_api_hot mode=polynomial modules=512 kernels=4 asts=65536 sm_90 warmup=0 repeats=1
 workers          best_ms          mean_ms        stddev_ms        modules/s       gram_kernels/s           asts/s
       1        38235.165        38235.165            0.000           13.391               53.563         1714.024
       2        19537.787        19537.787            0.000           26.206              104.823         3354.321
       4         9711.726         9711.726            0.000           52.720              210.879         6748.131
       8         4866.526         4866.526            0.000          105.209              420.834        13466.691
      16         2496.345         2496.345            0.000          205.100              820.400        26252.786
      32         1386.133         1386.133            0.000          369.373             1477.491        47279.724
      64          948.666          948.666            0.000          539.705             2158.822        69082.300
~~~

## Large Random Compile Hot Path

~~~bash
./build-verify-gh200-current/implicit_sindy_ast_compile_api_bench --modules 1024 --kernels 8 --workers 64 --mode random --repeats 1 --warmup 0 --workspace-mb 256 --sm sm_90
~~~

~~~text
implicit_sindy_ast_compile_api_hot mode=random modules=1024 kernels=8 asts=262144 sm_90 warmup=0 repeats=1
 workers          best_ms          mean_ms        stddev_ms        modules/s       gram_kernels/s           asts/s
      64         4170.445         4170.445            0.000          245.537             1964.299        62857.567
~~~

## Runtime Pipeline

~~~bash
./build-verify-gh200-current/implicit_sindy_runtime_pipeline_bench --modules 1 --kernels 4 --settings 2048 --train-rows 131072 --validation-rows 65536 --rhs 1 --sweeps 4 --solve stlsq --mode polynomial --compile-workers 64 --warmup 1 --repeats 5 --sm sm_90
~~~

~~~text
implicit_sindy_runtime_pipeline mode=polynomial solve=stlsq modules=1 kernels=4 cohorts=4 settings=2048 train_rows=131072 validation_rows=65536 rhs=1 sweeps=4 sm_90 warmup=1 repeats=5
  total_ms_best=8306.110 total_ms_mean=8306.287 total_ms_stddev=0.137
  phases_best_ms: train_gram=5535.444 validation_gram=2770.030 solve=0.394 mse=0.242
  throughput: pipeline_settings_per_sec=986.262 gram_row_settings_per_sec=1.939e+08 gram_feature_row_settings_per_sec=6.205e+09 solve_candidates_per_sec=8.308e+07 mse_candidates_per_sec=1.354e+08
~~~

## Gram-Only Runtime

~~~bash
./build-verify-gh200-current/implicit_sindy_reserved_cubin_runtime_bench --kernels 4 --settings 2048 --rows 131072 --primitive-cols 16 --rhs 1 --mode polynomial --warmup 1 --repeats 5 --sm sm_90
~~~

~~~text
implicit_sindy_reserved_cubin_runtime mode=polynomial kernels=4 launch_kernel=all settings=2048 rows=131072 primitive_cols=16 rhs=1 sm_90 warmup=1 repeats=5
  create_ms=8407.769 compile_ms=76.463 cubin_bytes=840784 reserved_cubin_bytes=840784
  run_ms_best=5535.124 run_ms_mean=5535.293 run_ms_stddev=0.146
  throughput: settings_per_sec=1480.003 row_settings_per_sec=1.940e+08 feature_row_settings_per_sec=6.208e+09
~~~

## Post-Bench Cleanup

Two stale debug benchmark processes from earlier GH200 debugging were still visible in the initial nvidia-smi snapshot. They were killed after the final report run. The clean post-bench GPU state was:

~~~text
Wed Jun 17 19:00:49 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GH200 480GB             On  |   00000000:DD:00.0 Off |                    0 |
| N/A   31C    P0             78W /  700W |       7MiB /  97871MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
~~~
