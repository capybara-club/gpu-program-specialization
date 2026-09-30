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

# cuSR Tests

`cusr_nvrtc_ast_sass_gpu.c` generates tile-static CUDA at runtime, compiles it
with NVRTC, inspects its one-site-per-kernel ABI, patches AST SASS into the
cubin, and executes it through the CUDA Driver API.

The fixed AST set covers `sin`, `cos`, `tanh`, protected division, protected
`sqrt` and `rsqrt`, `min`, `max`, and `fma`. Protected operations are declared
as routines, exercising routine expansion in both SASS lowering and CPU
interpretation.

The test runs 64-, 128-, and 256-row tile variants and compares final SSE/MSE
against `cusr_ast_sass_cpu.h`. It also generates capacity for 9 ASTs, patches
only 4, launches with `num_asts = 4`, and validates the active subset. It also
compares 8 active ASTs in capacity-8 and capacity-32 kernels.
The three runtime settings mix column-bound and constant-bound leaves.

`compare_nvrtc_ast_sass.py` independently regenerates the deterministic input
data and settings, evaluates the same ASTs in Python, and compares every GPU
output. NumPy is optional; the standard-library fallback runs the same checks.

All comparisons reject non-finite values first. MSE comparisons use
`atol=5e-4` and `rtol=5e-5` to allow for approximate MUFU operations and
different floating-point reduction order.

`cusr_ast_sass_contracts.c` is CPU-only. It checks routine register
reclamation, identity-routine ownership, expression-only lowering, site program
counts, architecture matching, prefix-only writes, and repeated destructive
patching with regcount reset.

`cusr_tile_static_eval_nvrtc_smoke.c` exercises the standalone C99 eval NVRTC
wrapper, retrieves its cubin, and checks the ELF image. This is separate from
the cuda.core compiler path.

The tests under `python/tests` compile direct CUDA template instantiations with
cuda.core and parse their cubins using the independent, `pyelftools`-backed
Python inspector. When `build/cusr_inspect_sites_cubin` is available, the
Python and C inspection results are compared field-for-field for AST capacities
8, 16, and 32.
`python/tests/test_native_pipeline.py` validates the nanobind AST ABI, builds a
native patch layout from the Python inspection result, patches a bytearray in
place, loads it with cuda.core, and launches two kernels over PyTorch-owned CUDA
tensors. It tests both the SSE and direct-value patch epilogues. The eval case
dispatches kernels over two streams and checks every `[AST, setting, row]`
value against the independent NumPy runner. Four ASTs include MUFU operations
and protected routines.

```sh
ctest --test-dir build --output-on-failure

python3 tests/compare_nvrtc_ast_sass.py \
  --producer build/cusr_nvrtc_ast_sass_gpu \
  --out-dir build/nvrtc_ast_sass_compare
```
