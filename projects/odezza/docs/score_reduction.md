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

# Score reduction and replay indices

Implemented 2026-09-06 in the C library and the isolated recovery trial. The scorer
still writes its bounded FP32 score buffer. A separately compiled, resident CUDA
module reduces that buffer; it does not inspect ASTs or change scorer topology.
The caller owns the CUDA context and all device storage.

## JSON

Explicit candidate scoring (`odezza.candidate-score.v1`):

```json
{
  "report": {
    "top_k": 20,
    "family_top_k": 5,
    "all_scores": false,
    "reduction": {"backend": "gpu", "group_by": "candidate", "top_k": 4}
  }
}
```

`report.top_k` still ranks candidates by their best configuration. The nested
`reduction.top_k` retains 1–16 configurations **per candidate**, available in each
candidate's `configuration_top_k`. Family membership and tags stay attached to
the candidate. `all_scores:true` additionally downloads the complete score vector
and checks reduced winners against it. No automatic CPU fallback occurs.

For a recovery campaign's native grammar screen:

```json
{
  "grammar_search": {
    "grammar": "R -> 'mul' A X\nA -> 'a'\nX -> 'x'",
    "families": [{"name": "linear", "start": "R", "accepted": 1}],
    "parameter_rows": [[-0.5], [-0.4]],
    "reduction": {"backend": "gpu", "group_by": "candidate", "top_k": 1}
  }
}
```

These fragments belong inside their normal complete requests. A standalone
`odezza.grammar-score.v1` request puts the same option in `report.reduction`.
Grammar reports currently retain one bank winner per AST before their existing
bounded global/family ranking; requesting a larger configuration top-k fails
validation. The frontend currently emits untoggled ASTs. JSON therefore accepts
only `group_by:"candidate"`; the C API already supports additional grouping.

Omitting the option preserves CPU reduction for compatibility and small banks.
Use GPU top-1 for broad screens with large banks. Small jobs may gain nothing;
large top-k increases reducer work. Refinement requires every probe score and
rejects an explicit reduction option, rather than silently ignoring it.

## Layout contract

For the completed bulk launch (which can span multiple packed modules):

```
permutations = 1ULL << launch.active_toggle_count
configurations = launch.constant_bank_count * permutations
score_index = system_index * configurations + bank_index * permutations + permutation
```

The original flat index is retained as a **64-bit integer**, never a float. Decode
using `odezza_score_decode_index`. System indices are local to the completed bulk
buffer, not global AST IDs or GPU thread IDs. The caller retains the batch-to-AST
mapping and sampling descriptor. Changing module packing does not change this
logical ordering; changing batching changes local indices, so indices alone are
not globally unique work IDs.

JSON reports include `score_location` (transport hash, local buffer index, system
index and bank width), the candidate identity, bank row, permutation, exact FP32
coefficient bytes, and existing replay work ID. Grammar locations additionally
identify the family and batch. Direct candidate transport indices identify the
candidate in the stable, equal-bank-length ordering. Existing old-worker CPU
reports can lack a physical score location; old transport formats remain readable.

The optional `odezza_score_reducer_gather` kernel copies exact rows from
`launch.constant_banks_device[system,bank,constant]` to `[group,k,constant]` using
only winning indices. The service already retains explicit host banks, so it
reconstructs report coefficients from those banks without another device gather.

**This does not add Philox generation.** Reconstructing future Philox samples also
requires a versioned generator descriptor: seed/key, logical sampling group or
stream, sample-base offset, component mapping, and uniform/normal transforms.
A physical output index supplies bank/permutation identity, not those missing
RNG semantics. Do not key randomness to module packing, CTA number or GPU number.

## C API

Include `o_score_reduce.h` and link `libodezza.a`.

1. With the caller's CUDA context current, create an `OdezzaScoreReducer` for SM
   version and immutable `k` (1–16). Keep it for repeated launches.
2. Query `odezza_score_reduction_requirements` for the actual scoring launch,
   bulk system count, grouping, and k. Allocate the reported workspace, winners
   and counts, separately from raw scores. Requirements/decode are CPU-only.
3. Complete scoring. Call `odezza_score_reducer_run` with these buffers. It is
   synchronous and performs no module loading or allocation in the hot path.
4. Download the small winner/count buffers or gather constants on device.
5. Destroy the reducer with the same caller context current. One caller at a
   time per handle; independent handles can serve different GPUs.

Groups are `ODEZZA_SCORE_BY_SYSTEM`, `ODEZZA_SCORE_BY_SYSTEM_PERMUTATION` (all banks
for each toggle permutation), or `ODEZZA_SCORE_GLOBAL`. Global reduction returns
a true global top-k; it does not preserve structural diversity. Per-system is the
normal search choice. Outputs are `[group,k]` records of `(float mse, reserved=0,
uint64_t score_index)`, plus `[group]` valid/invalid/negative counts.

Sorting is exact by `(mse, original_index)`, preserving FP32 bits and preferring
the earliest index on ties. NaN, infinity, FLT_MAX and negative scores are invalid;
finite negatives are counted separately so the worker can fail consistently with
its previous CPU behavior. Missing ranks are `(FLT_MAX, UINT64_MAX)`; gathering
those rows produces NaNs. No mean, median or approximate top-k is implemented.
Caller buffers must satisfy sizes/alignment and documented non-aliasing. Overflow
is rejected. Discard outputs after a failed run.

The first kernel uses up to 256 tiles per group. Threads retain local top-k lists,
merge them within a block, and write partial winners and counts. A second kernel
merges those partials exactly. This bounds scratch space independently of total
bank count, subject to the reported group count and launch-grid limits. Full MSE
writes to VRAM remain; this change removes bulk score downloads and CPU scans.

## Validation and measured scope

`tests/o_score_reduce_test.c` compares exact score/index pairs and gathered
constants against a CPU sort: all three grouping modes; k=1/4/16; invalid and
negative scores; signed-zero ties; all-invalid groups; odd bank widths and partial
tiles; packed system counts; 32-bit toggle decoding; overflow, aliasing and
undersized-buffer rejection. It runs on each Rack1 RTX 5080.

`scratch/recovery_trial/test_score_reduction.py` checks real trajectory scoring,
CPU/GPU top-k parity, full-vector cross-checks, variable per-candidate banks,
invalid ASTs, cached module reuse, grammar parity, and campaign-option propagation
to both devices. Existing trial tests cover refinement and workflow behavior.

Prepared benchmark: 128 prebuilt ASTs, 8,192 explicit banks each, three synthetic
analytic trajectories, four observations each, eight RK4 steps per interval;
1,048,576 configurations. AST parsing, bank generation and binary packing occur
before timing. Six warm measurements follow one cold request. Same frozen scoring
library, scorer shape and input values for both backends; all best records match.

| Device | CPU worker ms | GPU top-1 worker ms | Native scoring phase CPU → GPU ms |
|---|---:|---:|---:|
| Rack1 GPU 0 | 4.404 | 3.264 | 3.027 → 1.903 |
| Rack1 GPU 1 | 4.556 | 3.486 | 2.981 → 1.908 |

The worker measurement includes native binary input reading, host allocation and
report transport, but excludes Python AST/bank preparation and network service
admission. The native scoring phase includes uploads, scorer pipeline lifecycle,
reduction/download and native result printing. These are **not kernel-only** or
complete recovery measurements. Score return traffic falls from 4,194,304 to
5,120 bytes. The scorer pipeline time is essentially unchanged. At 192 banks,
worker times were approximately 0.77–0.80 ms for CPU/top-1 GPU; no meaningful total
speedup. Independent 2.10-million-score reduction tests show top-16 can cost about
as much as downloading all scores, despite much lower return traffic.

Cold reducer compilation/loading is reported separately in setup. Warm compatible
requests report zero reducer creation time. NVRTC/driver cache state can affect
cold cost. The live service smoke observed 0.105–0.115 seconds of reducer setup
on its first GPU requests; the warm benchmark setup often benefited from compiler/driver
caches. A Mac1-to-Rack1 JSON campaign verified successfully with GPU reduction
on both screen workers (campaign `31686ee1ed9f42a5f522cc0d268de4a5`).
Full evidence and the prepared benchmark script are in
`scratch/score_reduction/`; the measured inputs are synthetic, not recovery holdouts.

## Build and isolation

Normal library builds include `src/o_score_reduce.c`. `make test-cuda` also runs
the standalone reducer test. CMake registers `score_reduction_cuda_exact`.
`python3 tests/embed_score_reducer.py --check` verifies the embedded NVRTC source;
regenerate it after editing the `.cu` file.

The recovery trial keeps its own reducer source snapshot in `native/reducer` and
links it alongside its **unchanged frozen scorer library**. The trial native
Makefile is self-contained. Update both source copies together. Production port
8765 is outside this change; the trial service is port 8766. Deployment backups
and hashes are recorded in `scratch/score_reduction/deployment.json`.
