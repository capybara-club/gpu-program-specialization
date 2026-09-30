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

# Native search-policy trial

The five-arm matched experiment is described in
[the diagnostic report](../../docs/policy-repair-20260920.md). It uses the normal
Secant-SR CUDA binary; no legacy engine plugin, fallback scorer or CPU polishing.

On a prepared GPU worker:

```sh
python3 secant-sr/bench/policy/campaign.py \
  --manifest manifest.json --executable bin/search \
  --output shard0 --shard 0 --shards 3 --gpu 0
```

Use distinct shard/output/GPU assignments for concurrent workers. Resume requires
the same manifest, executable, runner, adapter and configuration. Failed fits are
retained and stop their shard rather than being silently retried. Prepared dataset
hashes, index replay and CPU/GPU score audits use the existing SRBench adapter.
Successful completion does not imply numerical recovery of every case.

The mac1 launch folder contains worker scripts, `launch.json`, copied results and
Telegram receipts. Run `python/notify_campaign.py` independently in tmux, with
`--launch` pointing to that launch file. Credentials stay on mac1.

After all workers and result copies finish, validate and summarize paired results:

```sh
python3 bench/policy/report.py \
  --run scratch/policy-repair-20260920 \
  --old-misses scratch/polish-20260920/manifest.json \
  --output docs/data/policy-repair-results-20260920.json
```

This checks the full matrix, identities, data/device matching, work counters,
index replay and retained numerical audits. It reports every case and both
reporting thresholds; numerical success does not establish symbolic equivalence.
