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

# First-pass GPU saturation and useful work

**Yes: 1,024 coefficient rows buy 33% more configurations for approximately 3%
more prepared-work time than 768. A second opportunity is larger: 39% of this
population has no active coefficients and was being evaluated repeatedly.**

The benchmark uses the exact 65,536 ordered ASTs from the first pass of the
2026-09-07 depth-3 game, with its known equations and training screen: six
trajectories, seven points each through t=0.6, 32 RK4 steps per interval. It uses
both RTX5080 GPUs on Rack1, EAGER module loading, 256-AST kernels, the current
Philox pool/CTA scaling and current GPU reducer/gather. No new solve or test-set
access was performed. AST enumeration and construction precede timed requests.

## Bank-size measurement

Median of three warm runs; reported time is the slower GPU worker's wall time.
Native pipeline includes specialization, loading/unloading, scoring, reduction
and winning-coefficient gather. Context, pipeline/RNG creation, AST preparation,
JSON, controller ranking and coefficient fitting are outside this timed region.
Family boundaries are preserved here, with module chunks redistributed over GPUs.

| Rows per AST | Actual evaluations | Prepared pipeline time | Relative time vs 768 |
| ---: | ---: | ---: | ---: |
| 256 | 16,777,216 | 0.612 s | 0.39× |
| 768 | 50,331,648 | 1.557 s | 1.00× |
| 1,024 | 67,108,864 | 1.599 s | 1.027× |
| 1,280 | 83,886,080 | 2.071 s | 1.33× |
| 1,536 | 100,663,296 | 2.551 s | 1.64× |
| 3,072 | 201,326,592 | 4.564 s | 2.93× |
| 6,144 | 402,653,184 | 8.613 s | 5.53× |

Larger banks contain fresh Philox samples. The original row prefix, descriptors,
seed, stream and sample base are preserved by extending the pool, not repeating
it. A bank of 1,024 is an empirically favorable point for this workload. This
is not evidence that arbitrary bank sizes or other trajectory costs scale alike.

A separate end-to-end service comparison used the existing synthetic calibration
input, three repetitions in alternating order, the same hard refinement bounds,
and a deliberately tiny 1e-30 target. It did not attempt to recover another RHS.

| Rows | Median screen phase | Median input to report |
| ---: | ---: | ---: |
| 768 | 2.282 s | 3.759 s |
| 1,024 | 1.950 s | 3.763 s |

All quotas completed. The service also changed family scheduling from four to
five native commands at 1,024 rows, so the faster screen is not a pure kernel
scaling measurement. Input-to-report variation was 3.70–4.02 seconds for 768 and
3.43–3.78 seconds for 1,024; do not interpret the 4 ms median difference as precise.
These synthetic timings are not directly interchangeable with the native table.

## Is the GPU saturated?

There is already substantial parallel work at 768 rows. The inspected kernel has
55 registers/thread, 128 threads/CTA and nine register-limited resident CTAs/SM.
The 5080 has 84 SMs: a full module launches 256 × 6 = 1,536 CTAs, about two waves
relative to 756 resident CTA slots. The theoretical thread-occupancy ceiling is
75% for this compiled shape. This is not measured achieved occupancy.

With all kernels resident, the same family-boundary workload's GPU event spans
were approximately 1.183 seconds at 768 rows and 8.331 seconds at 6,144 rows.
Thus 768 delivers about 88% of the measured high-bank resident-kernel throughput.
That is a comparison within this workload, not 88% of a GPU hardware peak.

At 768, normal-pipeline wall time was 1.562 seconds versus 1.187 seconds with
resident kernels: roughly 24% of that wall time can disappear in this resident
benchmark. At large banks execution dominates and this gap becomes small.
The individual module-load/unload/wait counters overlap and include waiting;
adding them would overstate the cost. The particular 768→1,024 plateau is visible
in pipeline timing while pure GPU spans still grow. Scheduling/overlap matters;
we have not attributed every part of the plateau to a specific CUDA mechanism.

No Nsight profiler was available; none was installed. A spot nvidia-smi sample
reported both GPUs busy during the larger-bank sweep, but that does not establish
SM efficiency or achieved occupancy. No clock/power limits were changed.

## Packing and redundant coefficient rows

All four requests together have this exact active-parameter histogram:

| Active parameters | ASTs |
| ---: | ---: |
| 0 | 25,659 |
| 1 | 27,208 |
| 2 | 10,697 |
| 3 | 1,852 |
| 4 | 120 |

The known equations contain no sampled constants. Therefore every coefficient
row of the 25,659 zero-parameter ASTs repeats its evaluation. This statement does
not rely on algebraically simplifying expressions, which could change invalid
arithmetic behavior. For more general jobs, inspect the whole system's active
parameters, toggles, initial-condition sampling and other varied inputs before
skipping any configuration.

The isolated benchmark groups coefficient-free ASTs together and gives them one
row. The other 39,877 ASTs retain the requested bank. It also packs across original
family boundaries; family labels have no effect on this identical scoring scope
and descriptors. This reduced 302 modules to 257 and balanced the two GPU shards.

| Strategy | Rows on parameterized ASTs | Actual evaluations | Prepared pipeline time |
| --- | ---: | ---: | ---: |
| Original boundaries, every AST gets full bank | 768 | 50,331,648 | 1.557 s |
| Original boundaries, every AST gets full bank | 1,024 | 67,108,864 | 1.599 s |
| Mixed packing, every AST gets full bank | 768 | 50,331,648 | 1.457 s |
| Mixed packing, every AST gets full bank | 1,024 | 67,108,864 | 1.490 s |
| Coefficient-free ASTs once, mixed packing | 768 | 30,651,195 | 1.085 s |
| Coefficient-free ASTs once, mixed packing | 1,024 | 40,859,707 | 1.109 s |
| Coefficient-free ASTs once, mixed packing | 1,536 | 61,276,731 | 1.666 s |

Compared with the original 768-row layout, the last row doubles the coefficient
samples for every parameterized AST for about **7% more wall time**, while still
scoring every coefficient-free AST. It does not execute 100.7 million configurations:
61.3 million are actually evaluated. The 1,024-row optimized variant offers 33%
more samples on parameterized ASTs and is approximately 29% faster than baseline.

Changing one loading worker/GPU to two with the mixed layout made no meaningful
improvement: 1.457→1.458 seconds at 768 and 1.490→1.492 seconds at 1,024. Adding
loading threads is not the first optimization suggested by these measurements.

## Correctness and integration

Each bank size was checked using full score-buffer checksums between normal
pipeline launches and preloaded launches. The original 768-row valid count
(49,410,022) matched exactly. Order-independent checksums of every AST's bytecode,
minimum FP32 score and winning bank index agree between full banks and the
coefficient-aware buckets at 768, 1,024 and 1,536 rows; weighted valid counts agree
as well. The winner comparisons do not silently replace old provenance.

The first attempt exited before scoring because the service had removed its
temporary packed transport files. Fixtures were reconstructed from retained
training request/context JSON with the existing packer. The first bucket logs
had a nominal `bench_run.configs` field; correct actual counts were already in
`bench_winners.evaluated`. The final benchmark fixes that field, includes a guard
against sampled constants in known equations, and passed a follow-up check.
The report tables always use actual counts, not the old nominal field.

The current service already accepts `--banks 1024` through the probe entry point,
with capacity explicitly pinned to 256. This task did not change service defaults
or scoring kernel code. The coefficient-aware bucket and mixed-packing paths are
implemented and validated **in the isolated benchmark**, not yet in the service.

For service integration:

1. Bucket prepared systems by effective bank requirement; start with the provably
   coefficient-free case. Preserve the original requested bank and actual launched
   bank separately in accounting and reports.
2. Allow compatible ASTs from multiple families to fill a module. Carry family
   memberships and original candidate ordinals through the mapping.
3. Derive score/index reconstruction from each actual launched layout. A skipped
   coefficient-free bank still has canonical winner index zero; arbitrary future
   pruning needs an explicit index map.
4. Keep per-family diversity in the returned report even when execution mixes
   families. Do not mix different trajectory objectives or RNG distributions.
5. Re-run service replay and end-to-end latency tests before claiming deployment
   gains. The native benchmark is evidence for the change, not a deployed result.

Files: `scratch/first_pass_saturation/{summary.json,cross-packing-replay.json}`,
per-run raw logs and exact packed fixtures. Reproduction scripts are
`prepare_fixture.py`, `run_benchmark.py`, and `summarize.py` in that directory.
Native benchmark: `scratch/recovery_trial/native/o_pool_saturation_benchmark.c`;
normal project Makefile target `../bin/odezza-pool-saturation-benchmark`.
