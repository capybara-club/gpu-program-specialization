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

# GH200 Lambda Test And Bench Results - 2026-06-17

This run used Lambda Labs host `ubuntu@192.222.50.14`.

## System

| Item | Value |
|---|---|
| Hostname | `192-222-50-14` |
| CPU | 64-core `Neoverse-V2`, 1 thread/core |
| CPU arch | `aarch64` |
| GPU | `NVIDIA GH200 480GB` |
| Compute capability | `sm_90` |
| Driver | `580.105.08` |
| CUDA toolkit | `12.8.93` |
| CMake | `3.28.3` |
| Python | `3.12.3` |

## Fix Verified

This run exposed a GH200/CUDA 12.8 failure in the hot-patched Gram compiler path:

- A 4-kernel module with only 33 ASTs, meaning most evaluator slots were padding ASTs, produced `CUDA_ERROR_INVALID_IMAGE`.
- A 4-kernel polynomial module could load but faulted at runtime with `CUDA_ERROR_MISALIGNED_ADDRESS`.

The fix keeps CUDA function metadata from the reserved linked cubin and makes constant-bank side-section fallback less order-sensitive:

- `cubin_function_patch.h` no longer records or overwrites `.nv.info.<function>` side sections.
- Missing replacement side sections now leave the reserved template bytes intact instead of zeroing them.
- Constant-bank fallback side-section selection now prefers a numeric suffix from symbols such as `implicit_feature_eval_3`.
- FusedSINDy now registers patch symbols in natural order: `implicit_feature_eval_0`, `implicit_feature_eval_1`, ...

The targeted failures passed after the fix:

```text
PASS implicit_sindy_ast_compile_api_test

implicit_sindy_reserved_cubin_runtime mode=polynomial kernels=4 launch_kernel=all settings=8 rows=1024 primitive_cols=16 rhs=1 sm_90 warmup=0 repeats=1
  create_ms=8292.890 compile_ms=75.264 cubin_bytes=840784 reserved_cubin_bytes=840784
  run_ms_best=1.147 run_ms_mean=1.147 run_ms_stddev=0.000
  throughput: settings_per_sec=27906.458 row_settings_per_sec=2.858e+07 feature_row_settings_per_sec=9.144e+08
```

## Test Results

Build command:

```bash
cmake -S . -B build-gh200 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DIMPLICIT_SINDY_BUILD_TESTS=ON \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DIMPLICIT_SINDY_BUILD_SOLVER=ON \
  -DIMPLICIT_SINDY_SOLVER_USE_NVRTC=ON
cmake --build build-gh200 -j$(nproc)
ctest --test-dir build-gh200 --output-on-failure
```

FusedSINDy C/CUDA tests:

| Test | Result | Time |
|---|---:|---:|
| `binary_ast_to_stack_ptx_test` | Passed | 0.01 s |
| `implicit_sindy_ast_compile_api_test` | Passed | 17.81 s |
| `implicit_sindy_gram_cpu_reference_test` | Passed | 19.19 s |
| `implicit_sindy_ast_columns_cpu_reference_test` | Passed | 0.67 s |
| `implicit_feature_ridge_solve_cpu_reference_test` | Passed | 0.40 s |

Total: 5/5 passed in 38.08 s.

The same FusedSINDy C/CUDA suite was rerun locally on the RTX 5090 system with CUDA 13.3 / `sm_120`; 5/5 passed in 18.75 s.

`cubin-function-patch` was also rerun after the header fix:

| System | Tests | Result |
|---|---:|---:|
| Local RTX 5090 / x86_64 / `sm_120` | 2/2 | Passed |
| Lambda GH200 / ARM SBSA / `sm_90` | 2/2 | Passed |

## Compile Hot Path

Command:

```bash
./build-gh200/implicit_sindy_ast_compile_api_bench \
  --modules 512 \
  --kernels 4 \
  --workers 1,2,4,8,16,32,64 \
  --mode polynomial \
  --repeats 1 \
  --warmup 0 \
  --workspace-mb 256
```

This measures the hot compile path for 512 cubin modules, 4 Gram kernels per module, and 65,536 total ASTs.

| Workers | Best ms | Modules/s | Gram kernels/s | ASTs/s |
|---:|---:|---:|---:|---:|
| 1 | 38867.629 | 13.173 | 52.692 | 1686.133 |
| 2 | 19777.698 | 25.888 | 103.551 | 3313.631 |
| 4 | 9809.181 | 52.196 | 208.784 | 6681.088 |
| 8 | 4891.298 | 104.676 | 418.703 | 13398.488 |
| 16 | 2512.029 | 203.819 | 815.277 | 26088.875 |
| 32 | 1364.533 | 375.220 | 1500.879 | 48028.140 |
| 64 | 954.223 | 536.562 | 2146.248 | 68679.927 |

## Runtime Pipeline

Command:

```bash
./build-gh200/implicit_sindy_runtime_pipeline_bench \
  --modules 1 \
  --kernels 4 \
  --settings 2048 \
  --train-rows 131072 \
  --validation-rows 65536 \
  --rhs 1 \
  --sweeps 4 \
  --solve stlsq \
  --mode polynomial \
  --compile-workers 64 \
  --warmup 1 \
  --repeats 5
```

Output:

```text
implicit_sindy_runtime_pipeline mode=polynomial solve=stlsq modules=1 kernels=4 cohorts=4 settings=2048 train_rows=131072 validation_rows=65536 rhs=1 sweeps=4 sm_90 warmup=1 repeats=5
  total_ms_best=8305.850 total_ms_mean=8305.953 total_ms_stddev=0.085
  phases_best_ms: train_gram=5535.169 validation_gram=2770.046 solve=0.392 mse=0.243
  throughput: pipeline_settings_per_sec=986.293 gram_row_settings_per_sec=1.939e+08 gram_feature_row_settings_per_sec=6.206e+09 solve_candidates_per_sec=8.355e+07 mse_candidates_per_sec=1.347e+08
```

As expected for this kernel shape, the solve and MSE stages are tiny relative to Gram generation.

## Gram-Only Runtime

Command:

```bash
./build-gh200/implicit_sindy_reserved_cubin_runtime_bench \
  --kernels 4 \
  --settings 2048 \
  --rows 131072 \
  --mode polynomial \
  --warmup 1 \
  --repeats 5
```

Output:

```text
implicit_sindy_reserved_cubin_runtime mode=polynomial kernels=4 launch_kernel=all settings=2048 rows=131072 primitive_cols=16 rhs=1 sm_90 warmup=1 repeats=5
  create_ms=8408.496 compile_ms=76.097 cubin_bytes=840784 reserved_cubin_bytes=840784
  run_ms_best=5535.216 run_ms_mean=5535.388 run_ms_stddev=0.105
  throughput: settings_per_sec=1479.978 row_settings_per_sec=1.940e+08 feature_row_settings_per_sec=6.207e+09
```

The Gram-only runtime matches the pipeline's `train_gram` phase, which is a useful sanity check.
