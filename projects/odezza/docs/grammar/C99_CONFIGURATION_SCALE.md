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

# One million systems with 1,024 and 2,048 configurations

Measured on rack1, one RTX 5080, 2026-09-11 local time. Three repetitions per
setting, using the existing C runtime and cached templates. No kernel, runtime
or search-policy code changed for this experiment; only request configuration
and the benchmark harness changed.

## Results

Every job contains **one million distinct complete six-equation RHS vectors**.
They are the Cartesian product of ten alternatives per equation, not one million
unique individual RHS subtrees. A single coefficient in the first equation takes
32, 1,024 or 2,048 explicit FP32 grid values. There are no Philox samples or state
toggles in this fixture.

Median C job time across three repetitions:

| Configurations per system | Configuration limit per tile | Total configurations | Seconds | µs/system | Systems/s | Configurations/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1,048,576 | 32,000,000 | 2.533 | 2.533 | 394,830 | 12.63M |
| 1,024 | 1,048,576 | 1,024,000,000 | 2.785 | 2.785 | 359,039 | 367.66M |
| 2,048 | 1,048,576 (current default) | 2,048,000,000 | 5.696 | 5.696 | 175,549 | 359.53M |
| 2,048 | 2,097,152 (complete page) | 2,048,000,000 | **2.848** | **2.848** | **351,116** | **719.09M** |

At one million systems, numerical seconds per job equal microseconds per system.
The complete-page setting doubles the 1,024-row workload for **2.26% more time**.
Relative to 32 rows it does **64× the configuration work for 12.45% more time**.
The 2,048-row complete-page runs range from 2.827 to 2.868 seconds.

These are complete C job rates, including parsing, generation, pipeline execution,
reduction and retention. They exclude service initialization/transport/CPU replay.
Only the first baseline needs runtime initialization (131ms); all templates are
cache hits with zero NVRTC time. Future completion wakes the harness directly,
so polling does not add a one-second detection delay.

## Why the default limit costs twice as much

A producer page contains 1,024 systems. With 2,048 coefficient rows, its full work
is 1,024 × 2,048 = 2,097,152 configurations. The current 1,048,576 tile ceiling
splits it into two calls through scoring and reduction. This doubles completed
tiles from 977 to 1,954 and repeats specialization/module loading for the same
programs. Increasing the request limit lets each page use one pass.

The two 2,048-row settings have identical candidate addresses, exact coefficient
bits, resolved programs and FP32 scores throughout their retained global
leaderboards. This is a comparable execution change, not reduced coverage.

Stage medians, amortized per system:

| Stage | 32 rows | 1,024 rows | 2,048 rows, default tile | 2,048 rows, complete page |
|---|---:|---:|---:|---:|
| Parse/allocate | 24.2 ns | 25.7 ns | 27.2 ns | 27.0 ns |
| Cached template preparation | 5.37 ns | 5.39 ns | 5.38 ns | 5.38 ns |
| Generation, overlapping execution | 648 ns | 886 ns | 842 ns | 793 ns |
| Coefficient prelude | 4.01 ns | 10.0 ns | 20.0 ns | 16.1 ns |
| Scoring pipeline | 1.782 µs | 1.799 µs | 3.580 µs | 1.803 µs |
| Reduce/gather/transfer | 535 ns | 775 ns | 1.729 µs | 825 ns |
| Retention processing | 107 ns | 106 ns | 211 ns | 106 ns |

Generation overlaps execution, and stage medians are computed independently;
they should not be summed to reproduce the job median. Pipeline time includes
specialization and module loading, not just GPU arithmetic. The near-flat pipeline
time with larger complete pages demonstrates amortization for this workload;
it does not establish the device's occupancy or a universal peak rate.

## Scope and validation

- Same finite grammar, initial values, trajectories, RK4 subdivisions and retention
  policy throughout. One trajectory, six states, three timestamps, twelve scored
  scalars, four RK4 steps per configuration. This is a very short rollout fixture.
- All 12 jobs complete: **15.456 billion configuration evaluations**, zero invalid
  scores, zero reported retry configurations, complete retention. This is repeated
  work on the same one-million-system population, not 12M unique systems.
- All repetitions and both 2,048-row tile settings retain identical ordered global
  winners for their respective coefficient grids.
- The top three candidates from each job replay identically by retained address.
  Independent FP64 CPU rollout agrees within 3.62e-15 absolute MSE across all checked
  winners. The full 2,048-row leaderboard equality is an exact FP32 comparison.
- Both GPUs were idle when checked before the run; existing services remained
  resident. Only device 0 executed this benchmark. This was not an exclusive GPU
  reservation or a dense occupancy trace.
- The initial 2,024-row request finished before the user's correction was applied.
  Those preliminary runs are excluded from every table here.

The runtime default remains unchanged. For this tested workload, the faster
request setting is:

```json
{
  "execution": {
    "batch_variants": 1024,
    "module_systems": 32,
    "max_chunk_configurations": 2097152,
    "dedup_bytes_per_family": 134217728
  }
}
```

Next: choose the tile allowance from the prepared layout and device-memory budget,
while bounding cancellation latency and preserving family scheduling. Avoid a
fixed ceiling that repeatedly reloads a population when its coefficient work
would fit together. Broader trajectory lengths/state counts and blind recovery
remain separate validations; the short-fixture rate should not be projected onto
them without measurement.

Reproduce:

```sh
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python python3 tests/runtime/configuration_scale.py \
  --configurations 1024 2048 --repeats 3 \
  --output /tmp/odezza-runtime-1m-1024-2048.json
```

[Harness](../../tests/runtime/configuration_scale.py),
[requests, binary hashes, timings and exact replays](runtime-validation/odezza-runtime-1m-1024-2048.json),
[live tracker](../../scratch/structural_search_trial/TODO.md).
