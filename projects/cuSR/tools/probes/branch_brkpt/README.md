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

# Branch And BRKPT SASS Probe

This probe checks the SASS behavior that cuSR relies on for AST-SASS
patching:

- unconditional forward `BRA` encoding
- `BPT.TRAP` / `brkpt` opcode stability
- contiguous `brkpt` padding after a marker instruction

Run:

```sh
cd /home/cdurham/code/cuSR
python3 tools/probes/branch_brkpt/verify_branch_brkpt.py
```

The default architecture list is:

```text
sm_80 sm_86 sm_89 sm_90 sm_100 sm_120
```

The branch probe is deliberately written so ptxas keeps a reachable `brkpt`
trap block and emits a forward branch over it. ptxas currently emits a
predicated branch for that shape, so the verifier checks the branch target field
rather than requiring the exact unconditional `BRA` opcode bits used by the
AST-SASS writer.

Generated cubins and disassembly are written to:

```text
tools/probes/branch_brkpt/out/
```

Latest local run with CUDA 13.1 on the RTX 5090 host:

```text
sm_80  PASS  branch target fields match SM8x encoding; max BPT run 17
sm_86  PASS  branch target fields match SM8x encoding; max BPT run 17
sm_89  PASS  branch target fields match SM8x encoding; max BPT run 17
sm_90  PASS  branch target fields match SM90+ encoding; max BPT run 17
sm_100 PASS  branch target fields match SM90+ encoding; max BPT run 17
sm_120 PASS  branch target fields match SM90+ encoding; max BPT run 17
```

`sm_110` was also checked manually and follows the same SM90+ branch target
encoding.
