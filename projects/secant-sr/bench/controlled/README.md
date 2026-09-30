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

# Settings / toggles controlled comparison

This is an isolated benchmark, not a replacement Secant or Secant-SR API. It
compiles the current GP controller unchanged and substitutes its scoring calls.
No production kernel, specializer, AST definition, GP, or refinement policy is
modified by this experiment.

## Questions and controls

1. **Engine:** replay identical captured populations, data, coefficient banks and
   permutations through old settings and new toggles. Compare every score, then
   time resident kernels and full batches separately.
2. **Search:** use the same current GP with settings versus toggles and refinement
   off versus on. Repeat identical seeds to expose floating-point selection
   variability. All arms for a case stay on the same GPU.
3. **Production control:** separately run the normal toggle pipeline at pack 8.
   This measures the effect of actual scheduling and packing. It is not the pure
   selector comparison.
4. **Evolution:** only after those controls, compare toggle-specific mutation and
   crossover. The initial pilot uses the same baseline variation policy throughout.

The backend comparison retains each engine's kernel topology and input
representation. It is not an instruction-only substitution of a toggle for an
indexing operation.

The current Philox proposal/acceptance refiner is used in both controlled engines.
This does **not** reproduce the old GP or its old optimizer. Consequently it does
not, alone, explain every difference in historical SRBench success rates.

## Common scheduler and packing

The common engine processes one module at a time, with independent CUDA graph
kernel nodes within each module. Both backends use the same best-score reducer,
event waits, input data, 128-thread blocks and 128-row tiles. The pilot uses two
ASTs per kernel and 16 kernels per module. This serial-module benchmark scheduler
is deliberately different from the concurrent production pipeline.

Old kernels share at most 32 dynamic leaf slots across packed ASTs. Two 31-node
trees fit without restricting their leaf choices. Arbitrary pack-8 populations
can exceed that limit; they are rejected rather than truncated or regrouped.
Old templates adapt among 4/8/16/32 slots, preserving the old small-template
optimization. Each kernel receives its own settings tables; the old CUDA kernels
are unchanged, but this is not a replay through the untouched old host API.

Settings table construction and upload are measured in the full batch. Resident
measurements exclude these costs. Resource-derived active blocks per SM are
recorded; they are **not a measurement of achieved occupancy**. No cross-AST
selector sharing or related-tree packing is introduced.

## Build and use

Use existing repository toolchains and an existing archived settings checkout:

```sh
python3 bench/controlled/build.py --secant ../secant --legacy /path/to/old/secant --output /path/to/bin
python3 -m unittest discover -s bench/controlled -p 'test_*.py'
```

The old implementation lives in a hidden-symbol shared library with a narrow,
versioned bridge. The executables statically link NVRTC and require the NVIDIA
driver, not a toolkit installation on each worker.

Select `SECANT_BENCH_MODE=settings`, `toggles`, or `native`. Settings also requires
`SECANT_BENCH_LEGACY=/absolute/path/legacy.so`. Always set
`CUDA_MODULE_LOADING=EAGER`. Native delegates to the production scorer.

`SECANT_BENCH_CAPTURE=/existing/directory` captures scoring calls 0, 8 and 32, plus
the first refinement call. Filenames refuse overwrite. The little-endian capture
contains versioned dimensions, column-major float32 data, targets, banks and
length-prefixed AST bytecode. It contains no pointer or raw C-struct serialization.

```sh
python3 bench/controlled/replay_sweep.py --captures captures --replay bin/replay \
  --legacy bin/legacy.so --output replays --gpu 0
python3 bench/controlled/campaign.py --manifest manifest.json --executable bin/search \
  --legacy bin/legacy.so --output shard0 --gpu 0 --shard 0 --shards 3
```

The campaign defaults to 60 seconds per fit, 8,192 ASTs, 64 banks × 64 permutations,
four coefficient slots, 31 logical nodes and 128 fitting starts with one round
when enabled. With 24 cases, six arms and two repeats, it runs 288 fits plus retained
warmups. The deadline is checked at generation boundaries; overruns and setup are
reported. Hard process deadline is 300 seconds. Parameter-capacity fitting skips
remain reported. No claim is made that every constant-bearing tree gets refined.

Resume requires matching hashes/configuration; a retained failure stops the shard
and cannot be silently retried. Results include held-out R² and a numerical success
threshold of R² > .999. Symbolic equivalence is not assessed. Twelve diagnostic
datasets and two official seeds are a pilot, not the full official SRBench protocol.

## Correctness and timing boundaries

Full-grid comparison requires identical finite/invalid classification and scaled
RMSE disagreement at most `2e-5`. Small atomic-reduction differences are expected.
Sampled winners and additional configurations are also resolved independently and
checked against CPU materialized predictions. CPU arithmetic uses float32 squared
residuals and float64 summation, to avoid attributing sequential-sum error to a
selector. SSE overflow is treated as invalid, as in the float32 GPU scorer.

If CPU and GPU disagree, the case stays explicitly flagged in `cpu_sensitive`.
A separate GPU materialization of the fully resolved expression then checks its
score with host float64 summation, using the same tolerance. `accepted` means all
checks passed through their recorded reference; `cpu_agreement_all=false` must
not be reported as complete CPU agreement. An unresolved discrepancy stops the
run. Validation never changes scores, kernels or the tolerance.

The first 8,192-AST I.39.11 generation-8 replay exposed six sampled CPU-sensitive
configurations. For example, `cos(x0)/log(x2)` amplifies CPU/CUDA math differences
near `x2=1`. All 33,554,432 settings/toggle scores agree, with zero finite-class
disagreements; the six resolved GPU checks also pass. Earlier failed CPU-only
checks are retained under `replay-I39-g8*` on rack1. This resolves the selector
check, not CPU/CUDA numerical identity. Owner/follow-up: inspect CPU-sensitive
counts across the other populations; do not weaken the production winner audit.

`resident_seconds` sums per-module medians of repeated graph execution; it does
not assume all modules are resident together. `wall_seconds` in pipeline samples
includes lowering, specialization, loading, transfers, execution, reduction and
unloading. Setup/NVRTC and post-run validation are separate. The repeated
full-grid validation pass includes extra repeats and score copies, so its wall
time must not be presented as ordinary pipeline throughput. Module-load timing
also includes graph construction. Process elapsed time includes all validation.

Native controls use the production pipeline at pack 8 and sampled winner checks;
their output is kept separately from the exhaustive pack-2 backend score audit.
Search winners retain the existing production CPU audit and coefficient/index
replay checks. Matching seeds do not guarantee matching search prefixes because
small float32 accumulation changes can alter selection.

## Supported comparison domain

1–32 input columns, 0–16 coefficient slots, 1–512 banks, 0–8 toggle bits; GP
leaf-level two/four-way selectors, state/bank/literal/affine-bank alternatives,
and current GP arithmetic operators. Selectors over compound subexpressions are
rejected. Replay bounds are 65,536 ASTs and one million rows. This benchmark is a
bounded diagnostic harness, not an untrusted-input service.

Campaigns run from mac1 tmux with SSH workers on rack1/ada. Mac1 must remain online;
the workers are not autonomous remote daemons. Source and binary hashes, driver,
data identities, failure logs and per-arm results are retained.

## Telegram notifications (default for future campaigns)

Launch one observer alongside the mac1 tmux workers. It sends one campaign start
and one final summary after all workers have exited and their result copies have
finished. Failures are reported explicitly; individual fits do not send messages.
The observer uses the existing private Telegram configuration through
`secant/scripts/secant-notify`, without copying credentials to GPU machines.

```sh
python3 python/notify_campaign.py --launch scratch/RUN/launch.json --label 'Run label'
```

Run this observer in its own tmux session so it outlives the assistant turn. For
an already-running campaign, add `--already-running`; its first message is labelled
as a late attachment rather than a new launch. Workers must write `worker-exit.txt`
after execution and `copy-exit.txt` after result transfer, under `shardN/` next to
`launch.json`. Reuse the existing worker scripts as the pattern.

`telegram-receipts.json` records accepted messages and suppresses ordinary
restart duplicates. Delivery failures are logged without credentials and retried.
An ambiguous network timeout can still duplicate a delivered message; Telegram's
send API does not provide this observer an idempotency key. Mac1 and the observer
must remain running. Explicit user requests can disable notifications for a run.
