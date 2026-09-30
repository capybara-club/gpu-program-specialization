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

# SECANT Benchmarks

This directory contains SECANT's optional comparison compilers, runtime
baselines, benchmark campaigns, papers, and performance artifacts. The active
benchmark surface compares CUDA C++, PTX, and direct-CUBIN paths on NVIDIA
hardware.

CUDA and PTX are experimental baselines used to measure native compiler
throughput and runtime against SECANT's direct CUBIN specialization. They are
not SECANT deployment backends. Their sources, public benchmark-only headers,
and tests are isolated under [`backends`](backends); vendored dependencies are
under [`backends/thirdparty`](backends/thirdparty).

## Executables

| Executable | Purpose |
| --- | --- |
| `secant_compile_bench` | AST-to-binary compile throughput |
| `secant_runtime_bench` | Resident-kernel runtime throughput |
| `secant_pipeline_bench` | Bulk compile, load, launch, and unload pipeline |
| `secant_dynamic_constant_sse_bench` | Dynamic-constant SSE kernel shape |
| `secant_dynamic_leaf_sse_bench` | Dynamic column-or-constant leaf SSE, including thread- and warp-owned CUDA/PTX variants |
| `secant_dynamic_leaf_lm_bench` | Analytic LM normal-equation statistics with thread- and warp-owned CUDA/PTX variants |
| `secant_philox_dynamic_leaf_select_bench` | Native-CUDA Philox leaf sampling and thread-local candidate selection prototype |
| `secant_constant_optimizer_sse_bench` | Per-AST Philox constant optimizer, reducer, and resident-iteration pipeline |
| `secant_packed_constant_optimizer_sse_bench` | Packed shared-jitter optimizer with pinned iterative feedback and respecialization |
| `secant_evogp_interop` | Validate mapped live EvoGP populations against Secant CPU and CUBIN execution |
| `runtime_*_sweep.py` | Row-count, tile-size, and AST-packing sweeps |
| `dynamic_constant_sse_sweep.py` | CUDA/PTX/CUBIN dynamic-constant setting-count comparison |
| `secant_native_avx_bench` | Static native AVX CPU baseline |

Build in Release mode:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECANT_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The compile, runtime, and pipeline executables accept
`--backend cuda|ptx|cubin`. Use `--help` for each executable's complete
options.

The dynamic-constant SSE benchmark accepts `--cse shared|distinct` to measure
common-subexpression elimination (CSE) in both ALU and MUFU corpora.
Both modes emit one discriminator constant and one ALU operation before every
MUFU-producing AST operation. `shared` reuses class-specific discriminators so
native CUDA/PTX compilers can eliminate equivalent subexpressions;
`distinct` assigns every AST site a unique discriminator. The sweep records
both modes in the `cse` CSV column, allowing their runtime and generated
code costs to be compared without changing the source operation count.

`secant_packed_constant_optimizer_sse_bench` uses `--optimizer-iterations` for
adaptive rounds within each timed runner call; `--iterations` remains the number
of measured calls. `--scale` initializes every AST-local constant scale;
`--momentum`, `--scale-learning-rate`, `--failure-decay`, `--minimum-scale`, and
`--maximum-scale` configure the reducer's per-constant adaptive state. Reported
row evaluations include all adaptive rounds. The compile-window metric includes
feedback dependency and backpressure, while compile-critical and compile-work
metrics contain actual specialization work.
`--update winner --elites 1` selects the single-winner policy.
`--update elite --elites N` selects uniformly weighted elite-distribution
recombination for `N` in `[2, 16]`; elite mode requires `--momentum 0`.

`secant_dynamic_leaf_sse_bench` generates one mask and one row of raw 32-bit
leaf words per setting. Dynamic-only programs project every column occurrence
into an indexed runtime column-or-constant slot. Mixed programs use
`--dynamic-sites` to project only that many column occurrences while retaining
the remaining column references and all literal constants as static AST
instructions. Dynamic-only mode necessarily projects every column occurrence;
`--dynamic-sites` controls mixed mode. The benchmark verifies a reduced
row/setting subset against the CPU interpreter before reporting specialization,
module pipeline, and device-runtime throughput. `--dynamic-values mixed`
generates column, constant, and mixed settings; `--dynamic-values columns`
forces every dynamic slot to select a runtime column. `--ast-nodes N` replaces
the normal balanced corpus with exact-N-node left-deep ALU or MUFU programs,
excluding the terminating return instruction. These programs keep their AST
evaluation stack near depth two while stressing long serial expressions.
`--columns` selects the active input prefix;
`--column-capacity` independently selects the compiled tile capacity, so one
template can be measured with smaller runtime inputs. `--static-columns 0`
selects dynamic-only programs, while setting it to the compiled column
capacity selects mixed static/dynamic programs. CUDA and PTX also accept
`--setting-owner thread|warp`. The warp-owned form assigns one setting to each
warp, keeps the input tile resident, evaluates rows lane-strided, reduces SSE
within the warp, and emits one atomic add per AST/target/tile. The direct-CUBIN
backend currently supports thread ownership only. For example:

```sh
CUDA_MODULE_LOADING=EAGER ./build/secant_dynamic_leaf_sse_bench \
  --ast-mode alu --modules 8 --workers 24 --kernels 16 \
  --asts-per-kernel 32 --leaves 16 --columns 8 --settings 4096 \
  --rows 1048576 --tile-rows 64
```

`dynamic_leaf_sse_grid.py` runs a resumable cross-product over AST mode, AST
packing, active/compiled input widths, dynamic-leaf capacity, dynamic sites,
settings, and dynamic-only versus mixed programs. Its defaults compile eight
dynamic registers and exercise one, two, four, and eight dynamic column sites
in mixed ALU and MUFU ASTs. Every recorded row has passed the benchmark's
CPU/GPU comparison; the script stops on the first mismatch.

`secant_dynamic_leaf_lm_bench` specializes the same mixed dynamic leaves with
forward-mode derivatives and accumulates, per AST/target/setting, SSE,
`J^T r`, and packed-upper `J^T J`. Its output layout is
`[ast][target][statistic][setting]`, with the setting dimension contiguous and
statistics ordered as SSE, every `J^T r` entry, then row-major upper-triangle
entries of `J^T J`. A leaf bound to a runtime column has a zero derivative;
repeated occurrences of one dynamic-leaf index share one derivative. The
prototype supports up to eight active parameters (45 accumulated floats).
`--ast-mode square-cube` exercises repeated leaves through `z + z^2 + z^3`.
Runs of at most 65,536 rows are checked against a double-precision CPU oracle.
`--verify-settings N` limits that oracle to `N` settings spread across the
full range, including the first and last settings; the GPU launch still runs
the complete shape. Omitting it checks every setting.
For example:

```sh
CUDA_MODULE_LOADING=EAGER ./build/secant_dynamic_leaf_lm_bench \
  --backend ptx --setting-owner warp --ast-mode square-cube \
  --bindings mixed --parameters 8 --settings 8 --rows 65536
```

`dynamic_leaf_cta_sweep.py` runs the same setting-partition and row-tile grid
for both dynamic-leaf SSE and LM. It records setting CTAs, row CTAs, total
CTAs, hot-kernel latency, and row evaluations per second after every case.
`--sse-asts-per-kernel` additionally sweeps packed SSE ASTs while LM remains at
one AST because its current packed implementation keeps a separate LM register
state per AST.
The defaults compare 4,096 and 8,192 settings, settings-per-CTA from 32 through
the complete setting count, and 32- through 512-row tiles using the PTX
thread-owned kernels. LM checks eight settings distributed across each full
GPU result so large sweeps do not spend most of their time in the CPU oracle.

```sh
python3 bench/dynamic_leaf_cta_sweep.py \
  --output dynamic-leaf-cta.csv \
  --settings 4096 8192 \
  --settings-per-cta 64 128 256 512 \
  --tile-rows 64 128 256 \
  --sse-asts-per-kernel 1 2 4 8 16 32 \
  --threads 64 128 256 --trials 3
```

`secant_philox_dynamic_leaf_select_bench` generates dynamic column-or-constant
leaf settings inside the native CUDA kernel with Philox4x32-10. Each thread
evaluates every packed AST for `--passes` independent settings, keeps the
lowest finite SSE, and writes
`pass_index * asts_per_kernel + ast_index`. The optional `--write-sse 1`
output stores the winning SSE. A second kernel reconstructs the exact leaf
mask and leaf words from the output slot and encoded winner. Before timing,
the benchmark compares the selected candidates and reconstructed settings
against the CPU dynamic-leaf interpreter. `--static-columns 0` selects
dynamic-only programs; setting it to `--columns` retains static column leaves
and rewrites only `--dynamic-sites` leaves. The timed interval includes kernel
launch and completion but excludes native CUDA compilation and module load.

```sh
CUDA_MODULE_LOADING=EAGER ./build/secant_philox_dynamic_leaf_select_bench \
  --ast-mode alu --kernels 64 --asts-per-kernel 32 \
  --columns 8 --static-columns 8 --leaves 8 --dynamic-sites 4 \
  --rows 256 --tile-rows 256 --threads 128 \
  --setting-blocks 16 --passes 4 --streams 64 \
  --warmups 2 --iterations 5 --write-sse 1
```

`pipeline_seconds` measures one external wall-clock interval around each
public runner call, including preflight validation, specialization, eager
module load, execution, completion, and unload. `runner_pipeline_seconds`
reports the runner's internal campaign interval after preflight. Neither field
includes source generation, NVRTC template compilation, CUBIN inspection,
runner creation, allocation, upload, warmups, or correctness verification.

The defaults use the large-row tile-static regime rather than a launch-limited
correctness shape. On the RTX 5090, 1,048,576 rows and 4,096 settings measured
2.06e12 runtime row-evals/s for the distinct 16-leaf ALU corpus. An 8-leaf
simple corpus reached 1.095e13 row-evals/s. The generated kernel uses 64-row
tiles and atomically accumulates one tile partial per
`[ast][target][setting]`.

`secant_constant_optimizer_sse_bench` measures the one-AST-per-kernel constant
optimizer separately from symbolic-search policy. Its `--optimizer-iterations`
remain inside each loaded module, while `--iterations` repeats the complete
specialize/load/run/unload pipeline for timing. `--kernels` is therefore the
number of AST kernels packed into each module; this shape has no
ASTs-per-kernel setting. Use `constant_optimizer_sse_sweep.py` to hold total
ASTs fixed and measure where kernels-per-module reaches diminishing returns.

The benchmark programs also own portable expression corpora, PySR, EvoGP,
Kozax, Operon, and static AVX comparison drivers, source papers, reports, and
generated figures. The former HIP/HSACO benchmark implementation, campaign
scripts, reports, and raw results are archived under [`icebox`](icebox) and
are excluded from the active CMake build.
