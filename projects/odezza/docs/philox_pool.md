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

# Philox pools, partial observations and completion events

The isolated recovery trial accepts reusable GPU coefficient pools in grammar requests and campaigns. Each worker keeps its CUDA context and cached pipelines alive. A matching pool is reused inside its pipeline cache entry; the cache has four entries. Pool creation is lazy on the first request, not a separate server initialization endpoint. Fixed pools are initialized by separate uniform and normal CUDA kernels, never by the specialized RHS kernel. The scoring CTA loads and transforms its constants before integrating a trajectory.

## Grammar request

Replace `parameter_rows` with:

```json
"rng": {
  "pool": {
    "size": 1048576,
    "seed": 20260906,
    "stream": 0,
    "sample_base": 0,
    "distributions": ["uniform", "normal"]
  },
  "bank_count": 8192,
  "parameters": {
    "a": {"distribution": "uniform", "min": -2, "max": 2},
    "b": {"distribution": "normal", "mean": 0, "stddev": 0.5, "clamp": [-2, 2]},
    "c": {"fixed": 0.3}
  }
}
```

These are fields within an existing `odezza.grammar-score.v1` request or campaign `grammar_search`. Every declared parameter needs a descriptor. A family may override descriptors by name with `parameter_sampling`. For example, `"parameter_sampling": {"a": {"distribution":"uniform","min":-0.5,"max":0.5}}`. Campaign fit bounds must contain the sampling ranges; a fixed fit parameter cannot be sampled.

`size` means FP32 samples **per distribution plane**: two planes of size N use 8N bytes. Current service limits are 16,777,216 samples per plane and 65,536 bank rows. Explicit candidate scoring and refinement still receive explicit rows; grammar winners provide those rows without exporting the pool.

For parameter slot j, default pool index is `j + bank_row * parameter_count`. Optional `offset` and positive `stride` override this mapping. Requests exceeding the pool fail validation; indices never wrap. Using the same offset/stride across ASTs deliberately compares structures on the same draws. Distinct offsets provide different draws when desired. Family scales may differ while sharing raw draws. Toggle bits select structural permutations; they do not change the bank's constants.

Uniform samples are in (0,1). Normal samples are N(0,1); `mean` and `stddev` are the shift and scale. `clamp` clips a normal sample; it does **not** sample a truncated normal distribution. Omitting it uses the finite FP32 range. CTA scaling performs a separately rounded FP32 multiply and add followed by clamping, shared with winner gathering.

## Identity and replay

Reports retain pool version, seed, stream, sample base, size, descriptors, descriptor hash, bank index and score location. With B bank rows and T active toggle bits:

```
configurations_per_system = B * 2^T
system = flat_score_index // configurations_per_system
configuration = flat_score_index % configurations_per_system
permutation = configuration % 2^T
bank_row = configuration >> T
absolute_sample = sample_base + offset + bank_row * stride
```

The flat index belongs to the reported completed worker batch; retain its batch/family identity as well. A score index alone is insufficient. Grammar ASTs currently have no toggle bits; the general C scorer supports them.

Philox4x32-10 uses key=64-bit seed, lower counter=absolute_sample/4, upper counter=`stream*2 + domain`, lane=absolute_sample%4. Domains 0 and 1 separate uniform and normal draws. Stream is limited to 63 bits. Uniform conversion uses 23-bit midpoint values, reproducible exactly with `odezza_rng_uniform`. Normal generation uses two Box–Muller pairs per Philox block. Transcendental rounding is not promised identical across future GPU/compiler versions; normal winners include exact GPU-gathered FP32 coefficients for replay. Same-version slices and both tested Rack1 GPUs replay identically.

The implementation follows the published [Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors). It does not claim cuRAND-compatible counter ordering or output conversion.

## Partially observed states

All state dynamics are still integrated. Use JSON `null` for unobserved values, including an entirely hidden state. Supply a complete finite `initial_state` vector per trajectory:

```json
{"times":[0,0.1,0.2],
 "initial_state":[1,2,3],
 "states":[[1,null,3],[1.2,null,3.1],[1.4,null,3.2]]}
```

The explicit initial vector is authoritative. Otherwise, a complete first observation supplies it. Missing initial conditions are rejected. Initial points never contribute to MSE. MSE is the sum of squared observed residuals divided by the number of observed scalar values; reports include total and per-state observation counts. Infinite observations, nonfinite integrated hidden states, and a dataset with no scored observations remain invalid.

Candidate scoring, grammar screening, native refinement, dataset storage, trajectory scopes and independent FP64 verification support this objective. A hidden state may be unidentifiable from the observed data; an excellent observed MSE does not establish that its true dynamics were uniquely recovered.

The prepared entry point accepts `--initial-states initial_states.json`, containing a mapping such as `{"trajectory-A":[1,2,3]}` with exactly the CSV's trajectory IDs. Keep all state columns in the CSV and leave missing observation cells blank. The sidecar is retained and hashed with the other inputs for resume.

## Event contract and ABI

`OdezzaScoringLaunch.input_ready_event` is optional. Record it after input uploads in their producer stream. Each scoring stream and the reducer stream issue `cuStreamWaitEvent`; the CPU need not wait for the upload. Keep the event alive until the synchronous operation returns. With no event, the caller must ensure input readiness.

The pipeline retires modules by their completion events. Reducer/gather completion and pool generation use recorded completion events. Full stream synchronization remains only in teardown or error cleanup, where queued work must finish before freeing resources. Public synchronous calls still wait for their own result; this change does not make them asynchronous APIs.

The scoring launch C ABI and generated kernel ABI changed (generator version 5). Rebuild all callers. The trial's Makefile links the new generator, pipeline and pool sources from `native/core` and reducer sources from `native/reducer` before the preserved baseline archive. This is a scoring implementation update even though that archive itself is unchanged.
