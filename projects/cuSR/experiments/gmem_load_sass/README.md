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

# Eight-column Global-load SASS Probe

This kernel loads one row from eight column-major columns, crosses an inline
PTX `brkpt`, reduces the eight register values with floating-point adds, and
stores the result. The `brkpt` intentionally traps if the unpatched kernel is
launched; this probe is for disassembly inspection.

The same source also provides `cusr_gmem_marked_load4_f32`, which emits four
explicit `global load -> marker FADD -> brkpt` sequences. Its marker immediates
run from `0x7fc0ffee` through `0x7fc0fff1`.

`cusr_gmem_load4_brkpt_markers_f32` instead places all four global loads before
one `brkpt` and all four marker FADDs after it. This probes whether the loads can
remain grouped while their destination registers stay recoverable.

`cusr_gmem_load32_brkpt_markers_f32` expands the grouped form to 32 live column
loads and places breakpoints on both sides of its 32-marker block so register
pressure and scheduling can be inspected at the intended maximum column count.

Build and disassemble for the RTX 5090:

```sh
mkdir -p build/experiments/gmem_load_sass
nvcc experiments/gmem_load_sass/cusr_gmem_load8_sum.cu \
  -O3 \
  --gpu-architecture=sm_120 \
  -Xptxas=--opt-level=1,-warn-spills,-Werror \
  -cubin \
  -o build/experiments/gmem_load_sass/cusr_gmem_load8_sum_sm120.cubin
nvdisasm -c -hex \
  build/experiments/gmem_load_sass/cusr_gmem_load8_sum_sm120.cubin \
  > build/experiments/gmem_load_sass/cusr_gmem_load8_sum_sm120.sass
```
