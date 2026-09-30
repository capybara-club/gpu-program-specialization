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

# Specialized toggle predicate latency

## Impact

The immediate-mask toggle specializer originally retained PTXAS's six-cycle control stall on each `LOP3.LUT` predicate test. On `sm_120`, an immediately following `FSEL` could consume the previous value of the shared predicate register. Toggle sites separated from their select by materializing instructions happened to work, so the existing one-toggle checks did not expose the defect.

Results produced by this immediate-mask implementation before the repair are not correctness-valid when a specialized AST contains multiple toggle sites whose tests can be adjacent to their selects. Single-toggle validation runs and historical runs using the older float-input toggle ABI do not exercise this defect. The pre-repair million-AST timing remains useful only as performance provenance, not as a validated scoring result.

The first observed failure was random depth-2 corpus system 486088, configuration 184. Its CPU MSE was `0.000225523778`; repeated packed GPU executions produced varying values around `0.0102` to `0.0110`. A one-system replay made the error deterministic and showed that 160 of its 256 configurations were wrong. The pass pattern demonstrated that an adjacent toggle reused the preceding toggle's predicate.

## Repair

Both the C99 and Python SASS writers now replace the toggle test's stall field with the conservative 12-cycle ALU stall already used by the specialized arithmetic instructions. This keeps the inspected opcode, predicate register, permutation register, and architecture-specific instruction fields while making the predicate dependency explicit. It does not add instructions or registers.

## Validation

- The exact failing AST passes all 256 configurations on the RTX 5090, with maximum CPU/GPU relative error `7.76e-7`.
- The one-million-AST, 256-configuration run passes 1,024 complete CPU trajectory replays; two CPU/GPU scores were mutually invalid and 1,405 approximate-math candidates were skipped under the existing policy. Maximum relative error among strict finite comparisons was `3.96e-6`.
- The repaired 64-instruction, 512-system pipeline sustains 2.26 million AST/s and 579 million configurations/s in the three-run measurement, unchanged within noise from the suspect pre-repair timing.
- A permanent GPU regression uses two adjacent toggle sites and verifies independent bit selection. Unit tests also assert the 12-cycle control field in both C and Python specializers.
- Rohini (`sm_120`) and Ada (`sm_89`) both pass all 14 C/GPU tests; the local Python suite passes all 18 tests.

The deviation is resolved for `sm_120` and `sm_89`. Before treating another architecture as validated, run the same multi-toggle GPU regression on that architecture.
