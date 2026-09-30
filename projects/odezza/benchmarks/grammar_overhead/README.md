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

# Direct grammar overhead measurement

This benchmark profiles the unchanged direct grammar service and C99 library.
It uses the supplied adaptation grammar, 8,192 skeletons, 128 configurations per
skeleton, the original retention policy and the same short synthetic trajectories
as the integration validation. It is an execution benchmark, not blind recovery.

See [the measured findings](REPORT.md) and [raw evidence](results/20260911).

On a configured CUDA host, from the repository root:

```sh
PYTHONPATH=python python3 benchmarks/grammar_overhead/profile_run.py \
  --library build/libodezza.so --root /tmp/grammar-profile-new
```

The output root must be new. A first baseline job starts a new CUDA owner, then
the default sequence interleaves four warm baselines, two scoped timing runs and
one cProfile run. Existing disk/driver caches are not cleared. The profile runs
use temporary wrappers in the benchmark process; production source files and
kernel code are unchanged. Each whole job executes in the service worker thread,
so the profiler measures that thread rather than time waiting on its future.
Generator timing surrounds each `next`, excluding the consumer's execution time.

Every run must finish all 1,048,576 configurations, retain exactly the same full
leaderboards/provenance as the initial run and independently replay three winners
on CPU. Timing and partial/failure statuses remain explicit. Summary files include
source hashes, cold/warm context labels, process CPU time, counts, bytes and
profile details; per-job files preserve the original service reports.

For a CUDA timeline with an already installed Nsight Systems:

```sh
PYTHONPATH=python /opt/nvidia/nsight-systems/2025.5.2/bin/nsys profile \
  --trace=cuda,osrt --sample=none --cpuctxsw=none --cuda-event-trace=false \
  --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
  --output=/tmp/grammar-trace-new \
  python3 benchmarks/grammar_overhead/profile_run.py \
    --library build/libodezza.so --root /tmp/grammar-trace-jobs-new --modes trace

python3 benchmarks/grammar_overhead/analyze_trace.py /tmp/grammar-trace-new.sqlite \
  --output /tmp/grammar-trace-analysis.json
```

The trace starts in the warm worker after a full untraced warmup job. CPU sampling
and context-switch collection are disabled; no elevated profiler privileges or
new dependencies were needed. The path above is the installed rack1 version.

Instrumentation changes timing. Use ordinary baselines for normal latency;
use nested timers for coarse wall attribution, cProfile for call counts and
hotspots, and the CUDA trace for kernel/API durations. API sums across threads
can overlap; do not add them to obtain critical-path runtime. GPU kernel activity
is not a measurement of achieved SM occupancy.

The first recorded sweep used coarse spans without SQLite wrappers; the second
added connection wrappers for individual query/commit timings. Their harness
hashes differ and are preserved in the environment records. Both profile the
same unchanged adapter/compiler/native implementation. The original first-sweep
harness is also retained in rack1's `overhead-harness.tar.gz`.
