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

# Row Evaluation Baselines

Local benchmark snapshot from July 7, 2026.

> Historical benchmark log: entries labeled `score+reduce` describe the removed
> row-partial implementation. The current tile-static MSE kernels expose only
> direct atomic SSE.

Hardware and toolchain:

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GiB
- Driver: 595.71.05
- CUDA toolkit: 13.1.115
- CPU: AMD Ryzen 9 9900X, 12 cores / 24 threads

The focused 32-AST/kernel settings saturation measurements are in
[`RTX_5090_SETTINGS_SWEEP.md`](RTX_5090_SETTINGS_SWEEP.md).
The production direct-atomic SSE comparison is in
[`RTX_5090_ATOMIC_SSE.md`](RTX_5090_ATOMIC_SSE.md).

## cuSR

PySR-like eval topology, one setting, row-parallel, materialized outputs:

```bash
./build/cusr_runtime_pipeline 32 4 16 1048576 1 0 alu
```

Shape:

- 32 kernels/module
- 4 AST sites/kernel
- 16 modules
- 2048 ASTs total
- 1,048,576 rows
- 1 setting
- ALU-only 8-leaf ASTs

Verified output:

```text
kernel_run=11.522 ms
hot_patch=3.620 ms
module_load=1.745 ms
max_abs_error=0
```

Derived throughput:

```text
AST-row evals/s          = 1.864e11
physical row passes/s    = 4.660e10
hot patch                = 5.658e5 AST/s/core = 1.77 us/AST
module load              = 1.174e6 AST/s      = 0.85 us/AST
```

Current `tile_static_mse` AST-SASS score kernels now report score-only runtime
separately from score-plus-reduce runtime. On July 8, 2026, using 128 kernels,
128 settings, 262,144 rows, mixed 16-instruction ASTs, and 10 timed iterations:

```text
shape                    score-only row evals/s  score+reduce row evals/s
8 ASTs/kernel, 8 cols    2.673e12                2.551e12
16 ASTs/kernel, 8 cols   2.981e12                2.797e12
32 ASTs/kernel, 8 cols   2.514e12                2.381e12
16 ASTs/kernel, 32 cols  2.145e12                2.055e12
```

The comparable cuSR_old shared-settings PTX path, with 32 columns and score plus
reduce timed together, measured:

```text
shape                         row evals/s
8 ASTs/kernel, mixed_reduce    3.854e12
16 ASTs/kernel, mixed_reduce   4.138e12
32 ASTs/kernel, mixed_reduce   4.415e12
16 ASTs/kernel, mixed_tiny     9.673e12
```

That old number is not the same metric as the PySR-like eval kernel above. It
is the fused scoring topology where settings and ASTs are packed to reuse loaded
data and avoid materializing prediction vectors. The old shared-settings
generator also used a leaner PTX-compiled score body and separate score routines
per AST after one tile load, while the current AST-SASS path uses a larger
patchable skeleton and, in the latest `tile_static_mse` generator, keeps packed
AST SSE accumulators live inside one shared row loop.

The native CUDA reference can also emit the same mixed 8-leaf,
7-binary-operation reduction shape as `cuSR_old`. This isolates kernel topology
from SASS patching overhead and from heavier expression bodies. On July 8, 2026,
using 1 module, 1,048,576 rows, 1,024 settings, 32 columns, 256 total ASTs, and
50 timed iterations:

```text
shape                         score-only row evals/s  score+reduce row evals/s
8 ASTs/kernel, alu_reduce      3.977e12                3.747e12
64 ASTs/kernel, alu_reduce     6.294e12                5.771e12
```

The matching `cuSR_old` mixed-reduce run with 32 kernels, 8 ASTs/kernel,
1,024 settings, 1,048,576 rows, and 50 timed iterations measured:

```text
score+reduce row evals/s = 3.862e12
```

The main miss in the earlier static-kernel comparison was expression mismatch:
the previous fixed min/max baseline added the AST index to all 8 inputs before
reducing, while `cuSR_old` used only 8 leaves and 7 hash-selected binary
operations. Once the expression is matched, packing more ASTs per kernel lets
the static kernel exceed the older PTX path.

## Technical Thesis

The core result is not simply that symbolic regression evaluation can run on a
GPU. The important shift is that cuSR makes AST specialization cheap enough to
shape the runtime kernel around explicit data locality. The large speedups come
from loading a dataset tile once, keeping the active columns resident in
on-chip memory, and evaluating many AST+setting combinations while that data is
still local.

Traditional GPU symbolic-regression backends usually face an unfavorable trade:
either interpret expression structure inside a generic kernel, paying dynamic
dispatch and intermediate-memory traffic, or compile specialized kernels and pay
too much CPU-side compilation latency. The AST-SASS path changes that balance.
Once a patchable tile-static kernel skeleton has been compiled and inspected,
new ASTs can be inserted in roughly microseconds per AST on one CPU core. That
makes specialization cheap enough to keep expression evaluation fully fused
inside the scoring kernel.

The tile-static MSE shape then exploits that locality directly. A CTA loads a
fixed row tile and the active input columns once, keeps the tile in low-latency
on-chip memory, and streams many runtime settings and packed ASTs across that
resident data. Each setting changes leaf bindings and constants without forcing
a new global memory pass over the dataset. The system turns one tile load into
many candidate scores.

This is why settings matter. They give the kernel enough useful work to do while
the data is still cache-resident. Without many settings, cuSR still has a fast
specialized AST body, but it does not amortize the tile load as aggressively.
With many settings, the dominant work becomes arithmetic over cached tile values
and reduction into SSE/MSE, not prediction-vector materialization or
global-memory movement of intermediate expression results. In the measured
tile-static runs, the system is much closer to a compute-bound row-evaluation
engine than a memory-bandwidth benchmark. The breakthrough is the combination:

- near-zero dynamic AST specialization cost after the skeleton is prepared;
- fused scoring with no prediction buffer;
- tile reuse across many runtime settings;
- one pooled AST patch site per kernel that shares the loaded data and register
  setup;
- enough specialization to avoid interpreter-style dispatch inside the hot row
  loop.

## Depth-3 AST Range

Local RTX 5090 sweep from July 8, 2026. Raw logs are in
`docs/bench_logs/depth3_20260708_195210/`.

Shape:

```text
modules             = 1
kernels/module      = 8
ASTs/kernel         = 32
ASTs total          = 256
settings            = 1024
rows                = 1,048,576
columns             = 32
timed iterations    = 30
tree shape          = balanced depth-3, 8 leaves, 7 binary reductions
```

The `depth3_alu` programs use deterministic hash-selected `add`, `mul`, `min`,
and `max` binary reductions. The `depth3_mufu` programs apply deterministic leaf
unary operations from `sin`, `cos`, `ex2`, and `rsqrt(abs(x) + 0.25)` before the
same balanced reduction shape.

```text
path                         score-only row evals/s  score+reduce row evals/s
native CUDA depth3_alu        7.937e12                7.024e12
AST-SASS depth3_alu           3.160e12                3.017e12
AST-SASS depth3_mufu          9.263e11                9.214e11
native CUDA depth3_mufu       1.307e11                1.307e11
```

Patch and load costs for the AST-SASS runs:

```text
path                 patch AST/s/core  patch us/AST  module load us/AST  patched regs
AST-SASS depth3_alu   7.092e5          1.410         0.299               77
AST-SASS depth3_mufu  4.929e5          2.029         0.287               78
```

The historical `1.410 us/AST` depth-3 ALU result used an unoptimized host
build. Rebuilding commit `2d082a7` with `CMAKE_BUILD_TYPE=Release` on July 12,
2026 produced `1.794e6 ASTs/s/core`, or `0.557 us/AST`, for the identical
256-AST workload. The former full-padding pooled-site implementation measured
`0.476 us/AST`, which is 14.5% less patch time (17.0% more ASTs/s) than that fair
Release baseline. The current prefix-only patcher is measured separately in
[`RTX_5090_ACTIVE_CAPACITY.md`](RTX_5090_ACTIVE_CAPACITY.md).

Read:

- Native CUDA is the ceiling for ALU-only ASTs when ptxas sees the final tree.
- AST-SASS is about `38%` of native for this historical ALU depth-3 runtime
  shape. Its corrected Release patch cost is `0.557 us/AST`.
- For the MUFU-heavy shape, AST-SASS is faster than the native CUDA `sinf` /
  `cosf` / `exp2f` path measured here because the patch path emits direct
  approximate MUFU sequences. This is a throughput ceiling for approximate
  special functions, not a claim of bitwise equivalence with libdevice.

## A10 sm_86 Validation

Remote A10 test snapshot from July 8, 2026:

```text
GPU: NVIDIA A10, compute capability 8.6, 23 GiB
Driver: 580.105.08
CUDA toolkit: 12.8.93
CMake: 3.28.3
Build option: -DCUSR_SASS_GPU_NAME=sm_86
```

The AST-SASS branch skip encoding is architecture-specific. Using the original
`sm_120` branch field on `sm_86` produced `CUDA_ERROR_INVALID_PC`. Inspection
decodes the target SM from the cubin ELF flags; callers pass that compute
capability to the patcher as `major/minor`, and the patcher chooses the branch
encoding from that explicit capability. The A10 build used
`CUSR_SASS_GPU_NAME=sm_86` only to generate sm_86 cubins.

Validation:

```text
ctest --test-dir ~/code/build/cuSR --output-on-failure
  1/1 tests passed

~/code/build/cuSR/cusr_ast_sass_random_test
  1k_1s_input0 max_abs_error=0
  1k_1s_neg0 max_abs_error=0
  1k_1s_add0c max_abs_error=0
  1k_1s_add01 max_abs_error=0
  1k_1s_native_ops max_abs_error=3.57628e-07
  1k_1s max_abs_error=1.19209e-07
  2k_2s max_abs_error=4.76837e-07
```

Direct native-vs-SASS run on the A10, using 1 module, 8 kernels/module,
32 ASTs/kernel, 256 total ASTs, 1,024 settings, 32 columns, 1,048,576 rows,
and 25 timed iterations:

```text
path                         score-only row evals/s  score+reduce row evals/s
native CUDA, alu_reduce       5.695e11                5.278e11
AST-SASS injected, alu_reduce  2.846e11                2.698e11
```

Patch/load timing for the same AST-SASS A10 run:

```text
patch rate       = 3.227e5 AST/s/core = 3.098 us/AST
module load      = 1.790e5 AST/s      = 5.585 us/AST
patched regs     = original 96, expanded 97, patched 99
max_abs_error    = 2.67029e-05
```

Read: SASS injection is functional on `sm_86` with CUDA 12.8, including the
branch skip path. For this light ALU reduction, ptxas-scheduled native CUDA is
still about 2x faster at runtime on A10, while AST patching remains microsecond
scale.

## A100 sm_80 Validation

Remote A100 test snapshot from July 8, 2026:

```text
GPU: NVIDIA A100-SXM4-40GB, compute capability 8.0, 40 GiB
Driver: 580.105.08
CUDA toolkit: 12.8.93
CMake: 3.28.3
Build option: -DCUSR_SASS_GPU_NAME=sm_80
```

The same explicit-capability branch selection works for `sm_80`. One ELF
nuance: the SM80 cubin tested here used flags `0x500550`, where the low byte
encodes the SM as `0x50` / 80. The local SM120 cubin used flags `0x6007802`,
where the low byte is not the SM and the next byte is `0x78` / 120. The
inspector decodes both layouts so callers can pass the corresponding
capability to the patcher.

Validation:

```text
ctest --test-dir ~/code/build/cuSR --output-on-failure
  1/1 tests passed

~/code/build/cuSR/cusr_ast_sass_random_test
  1k_1s_input0 max_abs_error=0
  1k_1s_neg0 max_abs_error=0
  1k_1s_add0c max_abs_error=0
  1k_1s_add01 max_abs_error=0
  1k_1s_native_ops max_abs_error=3.57628e-07
  1k_1s max_abs_error=1.19209e-07
  2k_2s max_abs_error=4.76837e-07
```

Direct AST-SASS run on the A100, using 1 module, 4 kernels/module,
16 ASTs/kernel, 64 total ASTs, 1,024 settings, 32 columns, 524,288 rows, and 15
timed iterations:

```text
patch rate         = 7.820e4 AST/s/core = 12.787 us/AST
module load        = 3.705e5 AST/s      = 2.699 us/AST
patched regs       = original 64, expanded 65, patched 67
score-only         = 1.037e12 row evals/s
score+reduce       = 1.019e12 row evals/s
max_abs_error      = 7.43866e-05
```

Read: SASS injection is functional on `sm_80` with CUDA 12.8. The smaller 64-AST
run is still somewhat overhead-dominated for patch timing, but it confirms that
the same runtime-decoded branch path works on A100.

Additional native-vs-SASS comparison on the same A100, using 1 module,
8 kernels/module, 16 ASTs/kernel, 128 total ASTs, 1,024 settings, 32 columns,
1,048,576 rows, and 20 timed iterations:

```text
path                         score-only row evals/s  score+reduce row evals/s
native CUDA, alu_reduce       1.592e12                1.524e12
AST-SASS injected, alu_reduce  1.073e12                1.037e12
```

```text
native module load = 0.706 us/AST
SASS patch rate    = 7.781e4 AST/s/core = 12.852 us/AST
SASS module load   = 1.564 us/AST
max_abs_error      = 7.43866e-05
```

Larger native-only A100 sanity run with 4 copied modules, 8 kernels/module,
16 ASTs/kernel, 512 total ASTs, 1,024 settings, 1,048,576 rows, and 8 timed
iterations:

```text
native CUDA, alu_reduce score-only   = 1.603e12 row evals/s
native CUDA, alu_reduce score+reduce = 1.588e12 row evals/s
native module load                   = 0.396 us/AST
```

Large patch/load stress sweep on the same A100 host, using a Release host build
(`CMAKE_BUILD_TYPE=Release`, `-O3 -DNDEBUG`), 4,096 total ASTs, 16 ASTs/kernel,
`settings=1`, `run_rows=1`, `check_rows=0`, and 1 timed iteration:

Raw logs:

```text
docs/bench_logs/bench_a100_load_sweep_20260708_184803.txt
docs/bench_logs/bench_a100_load_sweep_release_20260708_184940.txt
```

```text
ASTs/module  modules  patch AST/s/core  patch us/AST  load us/AST
16           256      5.996e5           1.668         4.546
32           128      5.971e5           1.675         3.436
64           64       5.972e5           1.675         2.568
128          32       5.181e5           1.930         1.881
256          16       5.972e5           1.675         0.434
512          8        5.987e5           1.670         0.388
1024         4        5.984e5           1.671         0.353
```

The earlier `~7.8e4 AST/s/core` A100 patch number came from a default CMake host
build with an empty `CMAKE_BUILD_TYPE`, so the C patcher was compiled without the
Release `-O3` flags. With Release flags and a 4,096-AST timed patch section, the
A100 host reaches roughly 0.6M AST/s/core. That is slower than the 5090 host CPU
run, but it is not the order-of-magnitude failure implied by the unoptimized
number.

## H100 sm_90 Validation

Remote H100 test snapshot from July 9, 2026:

```text
GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 80 GiB
Driver: 580.105.08
CUDA toolkit: 12.8.93
CMake: 3.28.3
Build option: -DCUSR_SASS_GPU_NAME=sm_90
CMAKE_BUILD_TYPE=Release
```

Validation:

```text
ctest --test-dir ~/code/cuSR_h100_2d082a7/build --output-on-failure
  1/1 tests passed

~/code/cuSR_h100_2d082a7/build/cusr_ast_sass_random_test
  1k_1s_input0 max_abs_error=0
  1k_1s_neg0 max_abs_error=0
  1k_1s_add0c max_abs_error=0
  1k_1s_add01 max_abs_error=0
  1k_1s_native_ops max_abs_error=3.57628e-07
  1k_1s max_abs_error=1.19209e-07
  2k_2s max_abs_error=4.76837e-07
```

Direct native-vs-SASS `alu_reduce` run on the H100, using 1 module,
8 kernels/module, 16 ASTs/kernel, 128 total ASTs, 1,024 settings, 32 columns,
1,048,576 rows, and 30 timed iterations:

```text
path                         score-only row evals/s  score+reduce row evals/s
native CUDA, alu_reduce       2.929e12                2.755e12
AST-SASS injected, alu_reduce  1.732e12                1.674e12
```

Patch/load timing for the same AST-SASS H100 run:

```text
patch rate       = 7.563e5 AST/s/core = 1.322 us/AST
module load      = 7.784e5 AST/s      = 1.285 us/AST
patched regs     = original 62, expanded 63, patched 65
```

Depth-3 run on the H100, using 1 module, 8 kernels/module, 32 ASTs/kernel,
256 total ASTs, 1,024 settings, 32 columns, 1,048,576 rows, and
30 timed iterations:

```text
path                         score-only row evals/s  score+reduce row evals/s
native CUDA depth3_alu        4.773e12                4.495e12
AST-SASS depth3_alu           1.282e12                1.274e12
AST-SASS depth3_mufu          4.484e11                4.468e11
native CUDA depth3_mufu       8.897e10                8.918e10
```

Patch and load costs for the depth-3 AST-SASS runs:

```text
path                 patch AST/s/core  patch us/AST  module load us/AST  patched regs
AST-SASS depth3_alu   7.609e5          1.314         0.736               101
AST-SASS depth3_mufu  6.423e5          1.557         0.790               102
```

Large patch/load stress sweep on the same H100 host, using a Release host build,
4,096 total ASTs, 16 ASTs/kernel, `settings=1`, `run_rows=1`, `check_rows=0`,
and 1 timed iteration:

Raw log:

```text
docs/bench_logs/h100_20260709_111700/bench_h100_load_sweep_release.txt
```

```text
ASTs/module  modules  patch AST/s/core  patch us/AST  load us/AST
16           256      7.404e5           1.351         3.232
32           128      7.410e5           1.350         1.640
64           64       7.404e5           1.351         1.203
128          32       7.118e5           1.405         0.927
256          16       7.156e5           1.397         0.637
512          8        7.070e5           1.414         0.609
1024         4        6.826e5           1.465         0.610
```

Read: SASS injection is functional on `sm_90` with CUDA 12.8. H100 is much
stronger than A100/A10 for native tile-static scoring, but on the current
AST-SASS skeleton it is still well behind the local RTX 5090 for the depth-3
SASS path. The likely practical story is that H100 is a solid cloud baseline,
while high-end Blackwell gaming cards remain unusually strong value for this
cache-local, multi-setting workload.

## PySR

Command:

```bash
cd /home/cdurham/code/pysr_ast_bench
PYTHON_JULIACALL_HANDLE_SIGNALS=yes JULIA_NUM_THREADS=24 \
  .venv/bin/python bench_pysr_candidate_score.py
```

Versions:

- PySR: 1.5.10
- SymbolicRegression.jl: 1.11.3
- Julia: 1.12.6
- Julia threads: 24

This benchmark scores 4096 candidate trees over each dataset. It reports `median_row_evals_per_sec = candidates * rows / median_seconds`.

Best 100k-row threaded cases:

```text
opset    mode           median seconds  row evals/s
arith    eval_tree_sum  0.408068        1.004e9
arith    eval_loss      0.408253        1.003e9
arith    eval_cost      0.411726        9.948e8
special  eval_tree_sum  0.777536        5.268e8
special  eval_loss      0.768252        5.332e8
special  eval_cost      0.765611        5.350e8
```

Relative to the current cuSR PySR-like eval kernel:

```text
cuSR / PySR arith eval_loss   ~= 186x
cuSR / PySR special eval_loss ~= 350x
```

PySR `model.predict` style historical local results are lower for full predictions at 1M rows, around `2.1e8 rows/s` for the best `linear_fma` case. The candidate-score benchmark above is the more relevant comparison because it evaluates many ASTs.

## PySR 24h Equivalent

This is only a scoring-throughput equivalence. It does not claim identical
search behavior, equation quality, mutation policy, constant optimization, or
hall-of-fame dynamics.

Using the local PySR candidate-score rates above:

```text
PySR arith eval_loss, 24 threads   = 1.003e9 row evals/s
PySR special eval_loss, 24 threads = 5.332e8 row evals/s
```

A 24-hour PySR scoring budget corresponds to the following cuSR wall times. The
primary comparison is against PySR arith `eval_loss`; the PySR special column is
included because it is the closer comparison for MUFU-heavy expressions.

| system | measured path | row evals/s | time for PySR arith 24h | speedup vs PySR arith | time for PySR special 24h | speedup vs PySR special | AST+settings/s at 100k rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| PySR | arith/special `eval_loss`, 24 threads | `1.003e9` / `5.332e8` | `24 h` | `1.0x` | `24 h` | `1.0x` | `10.0K` / `5.3K` |
| native CUDA | depth3 ALU, score+reduce | `7.024e12` | `12.3 sec` | `7003x` | `6.6 sec` | `13173x` | `70.2M` |
| cuSR ALU | AST-SASS depth3 ALU, score+reduce | `3.017e12` | `28.7 sec` | `3008x` | `15.3 sec` | `5658x` | `30.2M` |
| cuSR MUFU | AST-SASS depth3 MUFU, score+reduce | `9.214e11` | `1.6 min` | `919x` | `50.0 sec` | `1728x` | `9.2M` |

The native CUDA row is the absolute best-case ceiling for this kernel topology:
ptxas sees the final expression directly and can schedule it normally. The cuSR
rows are the dynamic AST-SASS patch path.

At `7.024e12 row evals/s`, the native CUDA score+reduce ceiling gives:

```text
rows      native AST+settings/s
32,768    ~214.4M
65,536    ~107.2M
100,000   ~70.2M
```

The measured AST-SASS depth-3 paths give:

```text
rows      depth3_alu AST+settings/s  depth3_mufu AST+settings/s
32,768    ~92.1M                     ~28.1M
65,536    ~46.0M                     ~14.1M
100,000   ~30.2M                     ~9.2M
```

For a concrete row-size view at `1.0e12 row evals/s`:

```text
rows      cuSR AST+settings/s  PySR arith candidates/s  PySR special candidates/s
32,768    ~30.5M               ~30.6K                   ~16.3K
65,536    ~15.3M               ~15.3K                   ~8.1K
100,000   ~10.0M               ~10.0K                   ~5.3K
```

At 32k rows, one hour of PySR arith scoring is roughly `110M` candidate scores.
The `1.0e12` cuSR row-eval path processes roughly `110B` AST+setting scores in
the same hour.

## Break-Even Rows Versus PySR

Model:

```text
cuSR time/AST = compile_or_patch_time + module_load_time + rows / cuSR_row_rate
PySR time/AST = rows / PySR_row_rate
```

Rates used:

```text
cuSR PySR-like eval kernel        = 1.864e11 AST-row evals/s
PySR arith eval_loss, 24 threads  = 1.003e9 row evals/s
PySR special eval_loss, 24 threads = 5.332e8 row evals/s
```

Current AST-SASS hot path:

```text
hot patch only       = 1.77 us/AST
hot patch + load     = 2.62 us/AST
```

Break-even rows per AST:

```text
compile/load model   vs PySR arith   vs PySR special
patch only           ~1,800 rows     ~950 rows
patch + module load  ~2,600 rows     ~1,400 rows
```

Legacy nvPTXCompiler-style compile path, assuming `44k AST/s` across all 12 physical CPU cores:

```text
aggregate compile rate = 44,000 AST/s = 22.7 us/AST
per-core compile rate  = 3,667 AST/s  = 272.7 us/AST
```

Break-even rows per AST:

```text
compile model          vs PySR arith   vs PySR special
44k/s aggregate        ~23,000 rows    ~12,000 rows
44k/s divided by 12    ~275,000 rows   ~146,000 rows
```

Read: with AST-SASS patching, cuSR starts making sense over PySR at only a few thousand rows per AST even if module load is charged per AST. With the older compiler path charged per physical CPU core, the row count needed to amortize compilation is closer to hundreds of thousands of rows.

## Break-Even With Settings And Fused Scoring

The native cuSR setting-search topology changes the model because compile and
module load are paid per AST, while runtime scores many settings per AST:

```text
cuSR time/AST = compile_or_patch_time + module_load_time
              + settings * rows / cuSR_native_rate

PySR time     = settings * rows / PySR_row_rate
```

Using the historical cuSR_old tile-static MSE rate:

```text
cuSR native fused scoring = 5.0e12 AST-setting-row evals/s
```

Asymptotic runtime-only ratios:

```text
cuSR native / PySR arith eval_loss   ~= 4,985x
cuSR native / PySR special eval_loss ~= 9,377x
```

Break-even rows per AST, charging current AST-SASS patch plus module load:

```text
settings/AST  vs PySR arith  vs PySR special
1             ~2,600 rows    ~1,400 rows
4             ~660 rows      ~350 rows
16            ~165 rows      ~90 rows
32            ~85 rows       ~45 rows
128           ~25 rows       ~15 rows
1024          ~3 rows        ~2 rows
```

Break-even rows per AST, charging the older `44k AST/s` aggregate compiler path:

```text
settings/AST  vs PySR arith  vs PySR special
1             ~23,000 rows   ~12,000 rows
4             ~5,700 rows    ~3,000 rows
16            ~1,425 rows    ~760 rows
32            ~715 rows      ~380 rows
128           ~180 rows      ~95 rows
1024          ~25 rows       ~15 rows
```

Break-even rows per AST, charging the older compiler path divided by 12 physical CPU cores:

```text
settings/AST  vs PySR arith  vs PySR special
1             ~274,000 rows  ~145,000 rows
4             ~68,000 rows   ~36,000 rows
16            ~17,000 rows   ~9,100 rows
32            ~8,600 rows    ~4,600 rows
128           ~2,200 rows    ~1,200 rows
1024          ~270 rows      ~145 rows
```

Read: once settings are part of the search space, the amortization changes by
roughly `1 / settings`. With AST-SASS patching and fused scoring, cuSR should
beat PySR at extremely small row counts when evaluating many settings per AST.
The caveat is that this is an effective scoring metric: PySR does not have an
equivalent runtime setting dimension, so this should be reported separately from
plain AST-row evals/s.

## EvoGP

Command:

```bash
cd /home/cdurham/code/evogp_trial
PYTHONPATH=/home/cdurham/code/evogp_trial/evogp/src \
  .venv/bin/python bench_evogp_sr.py \
  --rows <rows> --cols 32 --pop-size 4096 --repeats <repeats> --max-tree-len 32
```

Versions:

- torch: 2.12.1+cu130
- numpy: 2.5.0
- EvoGP: local checkout, no package version string

Fresh runs:

```text
rows      pop_size  median eval ms  row evals/s
100000    4096      92.303277       4.438e9
1000000   4096      426.139546      9.612e9
```

Relative to the current cuSR PySR-like eval kernel:

```text
cuSR eval kernel / EvoGP 1M ~= 19.4x
```

Relative to the historical cuSR_old tile-static MSE effective scoring rate:

```text
5.0e12 / 9.612e9 ~= 520x
```

## Notes

- The cuSR eval-kernel number materializes one output per AST per row, so it is a conservative PySR-topology baseline.
- A fused MSE cuSR eval kernel should avoid the prediction buffer write and may improve the PySR-like scoring number.
- The tile-static MSE number is a different, cuSR-native metric: AST-setting-row evals/s. It should be reported separately from raw AST-row evals/s.
- PySR uses CPU evaluation and a mature symbolic search stack. The numbers above isolate row evaluation/scoring throughput, not end-to-end search quality.
- EvoGP is GPU based but uses a runtime tree/forest evaluation path rather than patched straight-line AST SASS.
