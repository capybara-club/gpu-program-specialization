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

# Prototype boundaries and deviations

- Separate FP64 CUDA-source prototype, not the production FP32 SASS-specialized
  evaluator. Constants are compiled into each model, with no RNG/toggle search
  interface yet. Compile, first-call, and kernel time must remain separate.
- Uses the private known equations to audit integration. This is not recovery.
  Scores against published observations remain separate from integration error
  against accurate independent FP64 references. No benchmark is requalified here.
- Initial compilation failed on CUDA 13.1/glibc's incompatible `rsqrt` exception
  declarations. Apply the repository's existing `-U_GNU_SOURCE -D_DEFAULT_SOURCE`
  flags, also used by `scratch/robust_search_trial/lm_toggle/run.py`. This is a
  project compiler setting; no system headers, packages, or compilers are changed.
  Cache fingerprints include compiler version, architecture, flags and sources.
- One lane per trajectory; optional repetitions are duplicate work for measuring
  scaling, never additional unique ASTs or parameter configurations. Validation
  launches are tiny and cannot establish saturated search throughput.
- Measured register pressure: the 3-state Lorenz kernel uses 128 registers with
  a 144-byte stack and no compiler-reported spills; 10/13-state biological models
  use 255 registers with 1,792/2,528-byte stacks and nonzero spill traffic.
  Correctness results remain valid. A register-only interpretation and a blanket
  throughput claim are invalid. Reduce workspace liveness, remove constant-state
  rows from the linear solve, or compare a cooperative layout before production.
- Tight chaotic validation is expensive: Rosenbrock23 is second order and local
  error control does not bound long-horizon global error. Record all tolerance
  sweeps; selecting a tighter tolerance here is a private numerical investigation,
  not a search controller learning from its held-out data. No global rtol default
  is claimed to guarantee a given trajectory MSE.
- Runtime creates a stream and temporary allocations per call, and retains the
  loaded model library. It waits on the copy-completion event on success. Pageable
  host copies and allocations can block internally; this is not a fully pipelined
  resident worker. Production integration must address that explicitly.
- Adaptive Rosenbrock23 uses scalar tolerances over all integrated states,
  including unobserved/constant states; only observed values contribute to MSE.
  Supports autonomous systems with 1..16 states and the current safe arithmetic
  operators. Nonautonomous equations, DAEs and unsupported operators are rejected
  by the model compiler, never silently assigned a different solver.
