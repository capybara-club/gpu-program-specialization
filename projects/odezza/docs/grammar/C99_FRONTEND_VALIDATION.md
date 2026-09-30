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

# C99 frontend implementation and validation — 2026-09-11

The separate [frontend](../../frontend/README.md) implements the requested arena
phases and direct native AST producer. It is validated locally on mac1 and in
`/home/cdurham/odezza/scratch/grammar_handoff_20260911` on rack1. Live GP/server
checkouts and campaigns were not changed.

## Implemented

- Strict JSON section parsing into caller-owned trajectory/static/grammar arenas;
  NULL measurement, insufficient-space results, alignment and overflow checks.
- Native FP32 ragged/irregular trajectories, state-major observations, complete
  ICs, missing-state masks, observation/spacing metadata and RK4 work validation.
- Static RHS native postorder programs and fixed-RHS prespecialization in the
  scoring bridge, independent of variable AST generation.
- Prepared grammar references, recursive parameterized productions, shared shapes,
  global/local coefficient slots, grids, RNG descriptions/transforms, tags,
  binary/quad state toggles and sampled/all/explicit toggle groups.
- Per-family reservations, deterministic division of request ceilings, conflict
  rejection, bounded forward cursors, exact in-memory dedup, duplicate-provenance
  callbacks, accepted indices and whole-bank-row partial allocations.
- C-owned binary template caching with shape/source/checksum validation, atomic
  writes, capacity growth, and separate NVRTC/template/prespecialization timings.
- CPU Make/CMake libraries and optional C scoring bridge. Only the compile timer
  and its accessor were added to core behavior; kernel mathematics and GP policy
  were not changed.

## Checks

| Check | Result and scope |
|---|---|
| CPU contracts | Passed on Apple Clang/mac1 and GCC/rack1: exact/short arenas, FP32 time collapse, masks, static ASTs, shared choices, cursor resume, local RNG bindings, allocation conflicts, partial failure prefixes, bounded empty yields, duplicate tag provenance |
| ASan + UBSan | All CPU tests and eight Python-oracle prefix comparisons passed on mac1 |
| Allocation/dependency audit | CPU archive has no heap allocator, filesystem/database or CUDA symbol dependencies |
| Enumeration oracle | First 32 native programs of seven supplied fixtures, plus the sole constant/RNG-product program, match Python's bytes, slot counts, toggle bits and Cartesian counts (225 programs total) |
| Larger fixture preparation | All 16 supplied scoring grammars, including four biology examples and recursive/sampled toggle examples, prepare and emit a prefix; stiff/LM/non-grammar documents are rejected by this scoring phase |
| Million AST stream | 1,000,000 accepted native candidates, 1,000,000 derivations, zero duplicates or prunes, 37,200,000 instruction bytes; identical checksum on both hosts |
| C-only GPU request | Parses combined JSON, prepares known RHS, expands variable RHS, packs native descriptors, resolves RK4 settings, uploads data/grid and scores on rack1; correct coefficient wins |
| Template growth/cache | Grows patch capacity 2→4; both shapes cached. Cold/warm score identical, MSE 8.881784197e-15; warm NVRTC time exactly zero |
| Native template API | Round trip, corruption/shape rejection, borrowed template lifetime, GPU replay and the new compile-time accessor passed |
| Public exports | 33 declared/exported core functions; internal helpers remain private |
| Existing downstream service | `tests/grammar/gpu_conformance.py` exits successfully, including masked/irregular scoring, chunk invariance, Philox/prelude checks, retention/provenance, index replay and LM |
| Python regression | 83 canonical compiler + 19 service tests, and 60 main Python tests pass |

The aggregate `make test-python` still reaches an unchanged legacy failure in
`scratch/c99_cubin_runner/test_pack_input.py`: `KernelShape.toggle_permutation_count`
no longer exists. None of that packer/test code was changed here. This is not a
passing claim for the repository's entire aggregate test target.

An initial native template test failed because its target omitted
`CUDA_MODULE_LOADING=EAGER`. Its diagnostic identified the context-owner setting;
the target was corrected and passes with the normal eager pipeline. The isolated
checkout also initially lacked the public-export checker; that test file was
copied and the check passes. Neither issue required a slower execution fallback.
The standalone CPU CMake build passes on mac1. Rack1 has no `cmake` executable;
its CUDA libraries and acceptance tests were built with the repository Makefiles.
The parent CUDA CMake target additions therefore have not been executed on rack1.

## Measurements

The [source manifest](native-validation/source-manifest.json) identifies the
implementation. The retained host-generation records are
[mac1](native-validation/mac1-generation.json) and
[rack1](native-validation/rack1-generation.json). They measure actual native AST
emission and exact deduplication in batches of 256. They include no per-AST JSON,
GPU execution, coefficient-pool generation, module loading or result reduction.
During validation, mac1 produced roughly **2.5–2.7 million ASTs/s**, and rack1
roughly **1.9–2.0 million ASTs/s**. These are host-generation measurements, not
end-to-end search or recovery rates. See the final raw records for exact runs.

The million-AST fixture uses a 128 MiB exact-key storage allowance, an index table
with two million entries, approximately 189 MiB total producer arena, a roughly
13 MiB output arena, and a roughly 10 KiB immutable grammar plan. Memory is
explicitly measured and supplied by the caller; it is not constant memory as the
requested exact-dedup population increases.

[Cold/warm template measurements](native-validation/rack1-cache.jsonl):

| Run | Cache hits/misses | Growths | Actual NVRTC | Template preparation | Pipeline preparation |
|---|---:|---:|---:|---:|---:|
| Fresh directory | 0 / 2 | 1 | 99.063 ms | 99.598 ms | 0.308 ms |
| Reuse directory | 2 / 0 | 1 | 0 ms | 0.412 ms | 0.124 ms |

The tiny GPU fixture validates correctness/cache behavior; it does not establish
GPU occupancy, sustained bandwidth or a speedup on the earlier million-config
Python job. The [downstream log](native-validation/downstream.log) belongs to the
existing Python service and is compatibility evidence, not timing for the new
C-only frontend.

## Material compatibility differences and open integration

These apply to the first C99 frontend implementation. Existing Python runs remain
valid under their original profile; they are not automatically equivalent to a
new C99-generated population.

| Intended behavior | Actual boundary / consequence | Follow-up |
|---|---|---|
| Preserve sampled populations | Native structural/toggle sampling reports `odezza.c99-grammar.splitmix64-fisher-yates.v1`; Python uses a different seed algorithm. Matching seeds do not mean matching AST sets. | Use frozen explicit populations for performance comparisons; retain profile with each job. |
| Generate at index x | Forward accepted-index resumption works without prefix replay; arbitrary seek/restore is rejected. | Add explicit checkpoints/indexing if independently addressed pages are needed. |
| All grammar semantics in one phase | Current scoring language supported; legacy `generate`/`budget` aliases, subtree annotation spans, LM and stiff methods are not implemented here. Flat tags and duplicate memberships are available. | Normalize or explicitly implement additional syntax; never silently translate unsupported solver settings. |
| Launch general numeric plans | Grids and RNG/transform dependencies are prepared as typed metadata. The C GPU test executes a grid; the existing Python service executes arbitrary RNG/prelude products. | Add a C-owned binding/uploader phase for arbitrary Cartesian RNG plans before migrating those requests. |
| Family fairness through completion | Reservations and bounded per-family cursors are implemented. Caller still owns interleaving, deadlines, native launch tiling and completion accounting. | Build scheduler around these cursors; keep requested/reserved/attempted/valid counts separate. |
| Compact native results | No new database is used. Existing reducer remains available; the new frontend does not yet own compact top-k reports. `retain` is preserved as policy metadata for that phase. | Connect C-owned reduction/retention and service polling. |
| Exactly one execution attempt | Accepted AST bytes are reused on patch growth, but another module may already have run before a capacity failure. | Count retries; discard failed-attempt scores. Consider preflight sizing to avoid repeated GPU work. |
| Persisting cache and grammar data | Kernel artifacts alone persist in the new C path; plans/cursors are arena pointers, not relocatable serialized records. | Add stable checkpoint formats only if requested. No plan/trajectory/survivor database. |

The highest-priority next step is to connect bounded family cursors, coefficient
binding and the existing reducer into one C-owned execution loop, then compare a
frozen large population end to end against the profiled Python route. That is
necessary before claiming priorities 1–3 are completed for the public MCP service.
