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

# Grammar integration validation — 2026-09-11

Implementation lives in `python/odegrammar` and `python/odezza/grammar`.
Native code in `core/` and the installed GP service are unchanged. The native
library was built from the current mac1 source in the isolated rack1 checkout:
`/home/cdurham/odezza/scratch/grammar_handoff_20260911`.

## Evidence

### Combined submission and addressed results follow-up

- The current test target passes **83 compiler tests and 19 adapter tests**.
  New coverage includes acknowledgement while the worker is blocked, snapshotting
  caller-owned inputs, polling preparation and GPU setup, asynchronous validation
  failure, queued cancellation, restart interruption, combined-file CLI support,
  uint64 addresses beyond 2^53, and reconstruction after restart without enumeration.
- [Final native conformance](validation/conformance-unified-03/conformance.json)
  verifies 13 recorded cases plus direct assertions, including unchanged grouped
  retention, constants, uniform/normal RNG, toggles, chunk invariance and LM.
  Index replay checks every compact output candidate in the scored cases. An
  inline-problem RNG job produces identical winners to its prepared-problem job.
- [mac1-to-rack1 one-call MCP test](validation/mcp-unified-submit.json) sends known
  RHS, trajectories and grammar in one `odezza_submit`. The acknowledgement took
  about **1 ms** over an already established SSH/MCP connection, before the job's
  preparation and GPU setup finished. This is one small-request observation, not
  a latency bound including connection setup or large uploads. Full and compact
  results, ID replay and index replay agree, including independent CPU diagnostics.
- [Full population comparison](validation/unified-million-01.json): combined and
  prepared routes both complete **1,048,576 configurations**, with identical full
  global/family/tag winner records. All **31** distinct compact output candidates
  reconstruct from their addresses; three per job also pass CPU replay. Combined
  submission acknowledged in 0.605 ms in-process and the job took 10.53 s; the
  subsequent warm prepared job took 10.33 s. These are different cache histories,
  not a controlled speed comparison. The JSON result view shrank from 213,749 to
  50,221 bytes. MCP's text/structured wrappers add transport overhead to both.

The million-trial and first MCP evidence precede a final status-only correction
that labels GPU initialization `running/gpu_setup` instead of `generating`.
The final conformance and local regression tests verify that correction. The
unchanged native core/library is used throughout; intermediate materialization
and storage optimization remains pending.

The first follow-up conformance attempt (`conformance-unified-01` on rack1) scored
six configurations successfully, then its report consumer failed because a new
top-level `job_id` duplicated the one the consumer already supplied. The same
collision appeared in the CLI. Keeping the existing full-report shape resolved
it, and the combined-file CLI regression and subsequent conformance pass. Those
initial attempt logs are retained; there was no numerical fallback or scoring
failure. The pre-change adapter is saved as `pre-unified-submit-python.tar.gz`
in the isolated rack1 checkout.

### Original integration evidence

- [Compiler import checksums](validation/compiler-import.json): all seven Python
  source files exactly match the supplied archive.
- `make test-grammar`: **83 supplied compiler tests and 10 adapter tests pass**.
- The main repository Python suite: **60 tests pass**.
- [Native GPU conformance](validation/conformance-10/conformance.json): 12 recorded
  cases, plus direct uniform/prelude assertions, pass on rack1 SM120. The final source passes, including cancellation between repeated reduction passes.
- [Actual MCP round trip from mac1 over SSH](validation/mcp-smoke.json):
  initialization, tool discovery, preparation, asynchronous submission/status,
  results and independent CPU replay pass. It scores three configurations and
  retains the correct decay rate with MSE about `3.55e-15`.

Conformance covers irregular observation intervals, a completely unobserved
state with an explicit initial value, tied state toggles, mixed constant/RNG
axes, all five transforms, transform-only constant dependencies, exact uniform
agreement with `odezza_rng_uniform`, numeric chunk invariance, top-k >16,
duplicates that span reduction passes, late family/tag provenance, domain
failures, separate native LM, frozen RNG/constant values during log-parameter
fitting, and the eight-state/eight-parameter LM boundary. That boundary case
selects four cooperating lanes through native register fallback and reaches
MSE `7.27e-13`.

## Supplied populations

These use generated execution fixtures: two trajectories, five observations
each, duration 0.1, all states observed. The reference data is constructed from
the first grammar expansion using the handoff's SHA reference coefficients.
Production scoring uses the separately versioned Philox coefficients. These
are not biological data, blind discovery comparisons or claims of exact
structural recovery. Three retained winners per large job are checked with an
independent FP64 rollout using the actual gathered FP32 coefficients.

The shared-template run is in
[examples-04](validation/examples-04/examples-summary.json).

| Supplied grammar | Emitted skeletons | Completed configuration visits | Host plan s | Template setup s | Native scoring calls s | Whole job s |
|---|---:|---:|---:|---:|---:|---:|
| Adaptation with toggles | 64 | 2,048 | 0.05 | 13.16 | 0.0012 | 13.49 |
| Adaptation families | 8,192 | 1,048,576 | 3.36 | 0.03 | 0.1391 | 10.48 |
| Gene circuit | 10,000 | 1,000,000 | 5.61 | 13.13 | 0.2254 | 30.37 |
| Gene circuit plus constant grid | 10,000 | 1,000,000 | 5.79 | 0.03 | 0.2335 | 17.64 |
| Enzyme pathway | 15,625 | 1,000,000 | 8.88 | 0.04 | 0.2965 | 26.18 |

Every population completes: **4,050,624 configuration visits** in this five-job
run. Do not call the sum of emitted skeletons globally distinct: the two gene
requests overlap structurally. No numerical failures occur in these fixtures.
Example placeholder problem labels are explicitly rebound to the generated
prepared handles; the original labels remain in summary metadata. Grammar
operators, numeric domains and generation budgets are unchanged.

These timings include mixed cache histories, explicitly shown in template setup.
The first adapter compiled exact coefficient counts separately. For example,
the gene case prepared nine counts (19–27 slots), spending 117.64 seconds in
template creation out of 142.39 total. The new adapter shares compiled capacities
in groups of eight and reuses templates across active counts. The analogous
shared-template run spends 13.13 seconds in preparation, but these are **not
matched cold-cache measurements**. A later warm representative run completed
the 1,048,576-configuration adaptation case in 10.43 seconds, including 3.38
seconds in planning and 0.140 seconds in native scoring calls.

Earlier evidence remains in [examples-01](validation/examples-01/examples-summary.json),
[examples-03](validation/examples-03/examples-summary.json), and
[examples-05](validation/examples-05/examples-summary.json), and
[examples-06](validation/examples-06/examples-summary.json). Source/profile hashes
in authoritative job reports distinguish adapter revisions. `examples-04`
precedes the final scratch-budget and bounded-memory report guards; `examples-05`
adds those scratch guards. Final conformance and a representative million-trial
run verify the final streaming report implementation separately.

`scoring_pipeline_seconds` includes native specialization, module loading and
integration. The C99 scoring report does not expose separate stage timings.
Prelude/RNG/reducer fields are CUDA event durations. Host serialization, result
identity construction, storage, allocations, and uninstrumented initialization
remain a wall-time residual. No achieved occupancy or 768M-config/s claim is
made for these workloads.

## Compatibility decisions and deviations

| Requested/compiler feature | Actual execution contract | Status |
|---|---|---|
| SHA reference RNG recipe | Versioned Philox4x32-10 production mapping; preserve original recipe and exact winner rows | Implemented/tested; intentionally different numeric samples |
| Constant/RNG Cartesian axes | GPU mixed-radix prelude, bounded coefficient chunks, native toggles | Implemented/tested; no numeric AST expansion |
| Pre-RK4 scale/offset | Reusable GPU preparation kernel before scoring; scorer loads fixed values before RK4 | Implemented; extra device bank write/read is explicit |
| `dt` on irregular data | A maximum step resolved to a common subdivision count; land on observation times | Implemented/tested; effective count enters identity |
| Time leaf `t` | Native ABI lacks TIME | Rejected before launch |
| Powers | Expanded multiplication/reciprocal for integer exponents -16..16 | Larger exponents explicitly rejected |
| Stiff/adaptive integrator annotations | No executable fallback | Rejected before launch |
| LM `tolerance` | Target observed-entry MSE | Explicit mapping; not a gradient/parameter tolerance |
| Other parameter types/initial values to fit | Only named active `theta` slots, <=8 states/8 fitted values | Other requests rejected |
| Late membership and duplicate retention | Disk-backed variant survivors and bounded global/family/tag rankings | Implemented/tested |
| Multi-GPU single-job scheduling | One persistent GPU owner per service process; independent workers can choose devices | Not implemented in this adapter |
| Known equations prepared once | Context checked and full vectors assembled; common fixed-RHS patch not yet used | Correctness valid; additional specialization work remains |

No source or data was sent to third parties. No cloud service, system packages,
GP policy or core kernel changes were needed. Initial prelude NVRTC issues and
missing eager-loading setup failed with **zero completed configurations**;
their separate job reports remain on rack1. There was no CPU scoring fallback.

## Existing unrelated failing test

`make test-python` passes the 60 main tests, then the historical
`scratch/c99_cubin_runner/test_pack_input.py::test_system_major_constants_and_header`
fails: `pack_input.py:95` accesses the removed
`KernelShape.toggle_permutation_count` property. The other legacy runner test
passes. `python/odezza/model.py`, that packer, and its tests are unchanged from
HEAD. This failure is not suppressed and is not evidence against the new
grammar/native path; the broader target is not reported as fully passing.

## Next performance work

The [2026-09-11 overhead profile](../../benchmarks/grammar_overhead/REPORT.md)
completes the first attribution step: normal warm median 10.251s, with 3.495s
planning, 3.035s survivor persistence and 2.194s candidate identities/hashing in
coarse profiles. Nsight records 0.0492s of actual scoring kernels. All 13 jobs have
identical retained results; optimization remains pending. Use that report for
current causes rather than treating the older residual as unexplained.

1. Profile and reduce host lowering, identity construction and storage costs;
   native execution is already a small portion of these short-fixture jobs.
2. Hoist provably identical known RHS into the existing native fixed-RHS patch,
   with slot remapping and mixed-family compatibility tests.
3. Add persistent template artifact caching using the existing public API if
   process restarts dominate. Current reuse is within the process; NVRTC's own
   cache history is not a controlled experimental variable here.
4. Add a correlated native timeline before attributing module-load versus GPU
   execution time or claiming occupancy improvements.
5. Generalize dispatch to multiple GPU workers while preserving global indices,
   family coverage and exact winner replay. Keep it outside `core/`.
