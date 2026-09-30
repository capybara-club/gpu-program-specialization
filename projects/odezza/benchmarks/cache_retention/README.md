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

# SQLite template cache and bounded retention

Scope: replace the C-owned compiled-template file cache with SQLite, preserve
pipeline-create fixed-RHS prespecialization, share retained structure/provenance,
and reject reports whose conservative size bound exceeds the delivery budget
before scoring. Loaded candidate-module reuse is outside item 4 by the user's
2026-09-12 scope revision.

The cache is independent of the hardened CUDA core. SQLite 3.53.4 is vendored
unchanged from the official amalgamation, with archive and file checksums in
[the dependency manifest](../../third_party/sqlite/manifest.json). Import was
explicitly authorized after the initial source-download restriction. The normal
build never downloads dependencies or falls back to a system SQLite library.
The linked runtime exports no SQLite symbols and has no system SQLite dependency.

Current tests: SQLite hit/miss/checksum invalidation/eviction/oversized artifacts/
concurrent process misses/schema mismatch under ASan/UBSan; shared retention
payloads and early/late duplicate tags under ASan/UBSan; conservative report bounds
for the six retained service fixtures and the million-AST request.

Material output change: native requests are checked against an 8 MiB report
ceiling (optionally lowered with execution.max_report_bytes). The conservative
bound covers all requested leaderboards, candidate programs, slots, provenance
and trajectory metadata. A rejected request can have a smaller actual final
report; no top-k or tag memberships are silently truncated. Rejection reports
state the bound/limit and zero completed configurations. Combined CPU/GPU validation passes; live deployment evidence is recorded below.

## Item 3 isolated GPU validation

Checkout: `rack1:/home/cdurham/odezza/scratch/service_trial_20260912/cache-retention-test`.
This historical isolated test retained the old file cache before the SQLite
import was authorized. Its results validate item 3 independently. The combined
build below validates both changes together.

Two RTX 5080s, pooled input/result storage, attempt-owned numeric data, identical
4,096 ASTs × 64 configurations (262,144 evaluations), 16 coefficient slots,
raw global/family/tag rankings and late duplicate tags:

| Configuration tile | Baseline median | Item 3 median | Peak retained before → after |
|---|---:|---:|---:|
| 4,096 | 0.4504 s | 0.1458 s | 682.6 MB → 34.2 MB |
| 1,024 | 0.5069 s | 0.1946 s | 682.6 MB → 33.4 MB |

Three repetitions per case. Candidates, exact coefficient bits, indices,
leaderboards and coverage counts match the old runtime. Memory figures are
tracked retained records, not total RSS or GPU memory. Reports remain within
their conservative size bounds; deliberate oversized bounds and tiny report
budgets reject before any configurations are scored.

An initial singleton-screen regression (1.290 s versus a prior 1.221 s) was
repaired by packing metadata inline when no provenance/tags are possible and
only singleton retention is required. The contemporaneous 1M AST × 2,048 test
then measured 1.2186 s versus 1.2150 s baseline (warm medians, three runs), with
all candidates/leaderboards/counts/family records exact. This is short-rollout
throughput, not a hidden-RHS solve claim.

Validation: eight CUB family/RNG/late-tag cases, 15 native conformance cases,
pooled/unpooled memory-budget and reuse checks, 159 robustness cases including
eight random systems and 67,428,341 configurations; final singleton adjustment
reran the 159-case suite. CPU checks use ASan/UBSan for cache and retention.
Frontend contracts, mutation/allocation audits and eight supplied grammar
comparisons also pass. See results/summary.json and deviations.json.

## Combined build validation

The Make build passes on rack1, including the separately built vendored SQLite.
The optional CMake build was not executed: rack1 has no `cmake` command. No missing
tool was installed and no alternate dependency build was used. The production
service uses the tested Make build.

- 159/159 robustness cases, eight random systems, 67,428,341 configurations.
- Eight CUB family/RNG/late-tag checks, 15 native conformance checks, and pooled/
  unpooled memory limits, failure recovery and repeatable report serialization.
- Exact old/new candidates, coefficients, indices, leaderboards and coverage on
  the tagged cases and the million-AST benchmark. Three selected winners per
  million-AST request also pass independent CPU replay.
- Concurrent cold processes compile each requested shape once. Warm sessions
  perform zero NVRTC compilation. Corrupted checksums and mismatched generator
  identities rebuild correctly. Capacity growth stores and reuses both shapes.
- CPU tests cover schema rejection, fresh concurrent initialization, LRU eviction,
  oversize artifacts and 200 writes with normal short-reader WAL reclamation.
  The payload budget is not a hard directory quota; SQLite overhead and externally
  pinned WAL snapshots are separate. Cache errors fail explicitly.

The combined warm million-AST median is **1.2171 s**, versus **1.2150 s** in the
matched old runtime. Each request evaluates 1,000,000 ASTs × 2,048 configurations
on two RTX 5080s, with three observations and four RK4 steps per evaluation.
This is throughput measurement, not an equation-recovery benchmark. The first
combined request took 3.2203 s native / 4.8107 s through the local Python wrapper,
including cold setup; it is excluded from the warm comparison.

Tagged retained storage is 34.18 MB / 33.42 MB for 4,096 / 1,024 configuration
tiles, versus 682.58 MB previously. Warm combined native times were 0.142–0.149 s
and 0.200–0.202 s respectively; the first 4,096-tile request was cold and is
excluded. These are retained-record measurements, not RSS or device usage.

See `results/combined-*.json`, `results/vendor-*.txt`, and `deviations.json`.

## Live deployment and rollback

Deployed to the existing rack1 worker checkout on 2026-09-12; mac3 retained its
service generation and direct tunnel. Core source/binary, two-GPU allocation,
startup pools and host limits are unchanged. Six reference fixtures, two tagged
requests, report preflight rejection, and four 2.048B-configuration requests pass
through the live route with one worker PID and exact reference winners/coverage.
The queue is empty after releasing all test jobs. Live warm million-request native
median is 1.2443 s; the isolated matched comparison above remains the appropriate
old/new comparison. See `results/live-verification.json` and `deployment.json`.

Before/after source, live binary snapshots and checksum receipts are preserved at:
- mac1/mac3: `/Users/cdurham/code/.odezza-checkpoints/20260912-cache-retention`
- rack1: `/home/cdurham/odezza-checkpoints/20260912-cache-retention`

The pre-SQLite cache is archived for rollback; new service artifacts are in
`/tmp/odezza-service-trial-cache/templates.sqlite3`. Cache loss only makes the next
request cold. To roll back, drain the supervisor, restore `before-live.tar.gz`
into its recorded service parent directory, and restart the same supervisor with
its documented pool/host settings. Old cache files are optional performance data;
`before-cache.tar.gz` preserves them if needed. Customer jobs/results are RAM-only
and are not backed up in these archives. These checkpoints are working-tree
snapshots, not a committed/tagged release (remaining item 10).
