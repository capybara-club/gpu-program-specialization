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

# Fused RK4 specialization SASS inspection

The CUDA template in `secant_specialization_template.cu` was compiled with
CUDA 13.1 NVRTC on 2026-08-23 using `--use_fast_math`, line information, and
ptxas optimization level 3. The resulting CUBINs were accepted by Secant's
materialize-plan inspector with one kernel, two AST outputs, 12 inputs, and a
128-instruction patch capacity.

| Host | GPU | Target | Registers | Stack/local/spills | CUBIN | ptxas compile log |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Ada | RTX 4090 | `sm_89` | 48 | 0 bytes | 19,816 bytes | 12.877 ms |
| Rohini | RTX 5090 | `sm_120` | 48 | 0 bytes | 24,344 bytes | 9.321 ms |

Both skeletons contain exactly 131 `BPT` instructions: three structural marker
boundaries and 128 patch-capacity instructions. This establishes that NVRTC and
ptxas retained exactly one physical specialization island.

## Register contract

The marker scan exposes the following physical registers:

| Value | `sm_89` | `sm_120` |
| --- | --- | --- |
| `X` | `R28` | `R11` |
| `S1` | `R27` | `R10` |
| `S2` | `R26` | `R9` |
| `P` | `R25` | `R8` |
| `c0..c7` | `R2..R9` | `R29,R28,R27,R26,R25,R24,R23,R22` |
| `mu_1` output | `R29` | `R0` |
| `mu_2` output | `R30` | `R30` |

The identities are recovered from unique immediate sentinels in the binary;
`nvdisasm` renders all of those NaN payloads simply as `+QNAN`.

## Specialized rate-law island

For validation, the two ASTs were specialized as

```text
mu_1 = c0*S1 / ((S1 + c1*X) * (1 + c2*S2))
mu_2 = c3*S2 / ((S2 + c4*X) * (1 + c5*S1))
```

On `sm_120`, the live replacement begins as:

```text
/*0450*/ NOP
/*0460*/ FMUL.FTZ R0,  R29, R10
/*0470*/ FMUL.FTZ R31, R28, R11
/*0480*/ FADD     R31, R10, R31
/*0490*/ FMUL.FTZ R32, R27, R9
/*04a0*/ FADD.FTZ R32, R32, 1
/*04b0*/ FMUL.FTZ R31, R31, R32
/*04c0*/ MUFU.RCP R31, R31
/*04d0*/ FMUL.FTZ R0,  R0,  R31
...
/*0560*/ FMUL.FTZ R30, R30, R31
/*0580*/ BRA .L_after_patch
```

The first `NOP` replaces the load-fence `BPT`. The two expressions use only
register arithmetic and `MUFU.RCP`; the final branch skips the unreachable
remainder of the reserved island.

## RK4 control flow

The four-stage loop also survives compilation as one loop. On `sm_120`, its
header and back edge are:

```text
.L_stage:
/*0420*/ IADD R40, R40, -1
...
/*0450*/ NOP                 // specialization load fence
/*0460*/ ...                 // mu_1 and mu_2
...
/*10b0*/ FFMA.FTZ ...        // next RK4 stage state
/*10c0*/ FFMA.FTZ ...
/*10d0*/ FFMA.FTZ ...
/*10e0*/ FFMA.FTZ ...
/*10f0*/ @P1 BRA .L_stage
```

The `sm_89` CUBIN has the same organization, with its stage back edge at
`0x10d0` targeting the patch-loop header at `0x0460`. Thus the compiled shape is
one candidate-expression site executed four times per RK4 step, not four copied
sites.

These measurements describe the current ground-truth-sized AST pair. Larger
candidate expressions may increase the CUBIN register count during Secant
specialization. Occupancy and spill behavior must therefore also be measured at
the maximum supported candidate size before reporting final throughput.
