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

# CPU audit scope and deviations

- The subsequent CPU FP32 study intentionally keeps observation times, step
  control/error norms, original observations and MSE accumulation in FP64.
  States, coefficients, RHS/Jacobian, matrix solves, stage updates and embedded
  defect arithmetic use FP32, including float elementary functions. It is not
  an all-operations-FP32 or bitwise production-GPU replay. Both precisions use
  the same compiler flags without fast-math or contracted FMAs.
- Tightening the FP32 Rosenbrock tolerance to 1e-9 creates 627,934 rejected
  attempts versus 697 in FP64 and makes it about 30% slower on this CPU sweep.
  This is retained as a precision-floor diagnostic, not a recommended default.
  FP32 RK4 loses previously passing systems as substeps rise beyond 32; one
  source-discrepant case gains an accidental pass at 128 from error cancellation.
  Reports distinguish retained baseline passes, all sample passes, score
  reproduction and trajectory disagreement. No benchmark is requalified.

- User requested CPU implementation and reproduction of benchmark-sample MSE.
  Native Rosenbrock23 and RK4 were compiled with mac1's installed clang++ and run
  on mac1, FP64 throughout. There is no CUDA dependency or GPU execution. This
  supersedes the earlier GPU-focused interpretation for this request.
- The existing Rosenbrock numerical header gained a compiler-conditional `HD`
  annotation. Its arithmetic and error controller were not changed. The old GPU
  reports remain evidence of their recorded source hash; no GPU benchmark rerun
  or new GPU performance claim is made here.
- Native methods use only standard-library Python. Independent installed SciPy
  Radau/BDF/LSODA run on rack1 CPUs because that environment already exists.
  These timings are not compared as if measured on mac1 or using native compiled
  RHS callbacks. No dependencies were installed.
- Primary metrics are MSE against the original samples, on every split. Agreement
  with previously measured accurate FP64 sample MSE is a separate reproduction
  test, with explicit 1e-8 absolute + 1e-4 relative score tolerance. This is not
  a trajectory-accuracy certificate and does not relax the 1e-6 sample-fit gate.
  All tolerance/settings sweeps remain recorded; no source data were altered.
- The first independent SciPy launch hit an import-name collision between CPU
  and GPU folders. The shared codegen import now restores Python's search path.
  A subsequent launch failed to serialize NumPy int32 work counters returned by
  LSODA; counters now convert explicitly to Python int. That launch retained no
  successful reports and is not used. The corrected run is `scipy-v2`; native
  mac1 results are unaffected. These are reporting/startup repairs, not solver
  fallbacks or a change in the mathematical objective.
- Independent BDF at rtol=1e-12 fails on ODEBench 055 because its required step
  is below representable time spacing; a targeted 1e-13 run also fails. Both are
  retained. Runs at 1e-10 and 1e-11 complete and reproduce every split's reference
  MSE under the declared reproduction tolerance. The report explicitly selects
  the tightest successfully completed setting and does not call the original
  1e-12 setting successful. No other solver silently replaces BDF.
