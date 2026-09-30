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

# Scoring benchmark correctness coverage

Date: 2026-09-03  
Status: resolved for representative configuration semantics; saturated output remains sampled

## Original limitation

The eight-state and state-capacity timing harness initially asserted only output zero: candidate system zero using constant bank zero. That proved the specialized equation, RK4 rollout, and MSE for one exact configuration, but did not independently establish constant-bank indexing, system indexing, or toggle mapping across the rest of the output tensor.

No mismatch was observed. The limitation was insufficient validation coverage, not evidence of incorrect scores. Throughput timings remained conditionally valid while the coverage gap was open.

## Resolution

The harness now independently replays candidate equations and the RK4/MSE loop on the CPU. It uses materially distinct per-system bias literals and checks constant banks and toggle permutations by their documented `[system, configuration]` output ordering.

- A representative eight-state test exhaustively replayed all 8,448 outputs from eight systems, 33 constant banks, and 32 toggle permutations. Every output passed; worst relative MSE error was `9.38e-6`.
- The exact 2,097,152-output saturated constant-bank shape was sampled at 16 boundary/interior locations, including the final bank. Every sample passed; worst relative error was `1.80e-6`.
- The 20-, 32-, and 46-active-state boundary shapes passed 16 multi-bank CPU samples each, with worst relative errors no larger than `7.45e-6`.

The small FP32 differences are consistent with GPU/CPU instruction ordering. The timing claims can now be treated as valid for the tested scoring semantics. They do not imply that every output of the saturated run was individually CPU replayed.
