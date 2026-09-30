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

# Budgeted final-polishing experiment

See [design, boundaries and validation](../../docs/finalist-polishing-20260920.md).

Use the already-installed scientific Python environment. `prepare.py` reads
existing hash-verified datasets; it does not download data or solutions.

```sh
python3 bench/finalists/prepare.py --previous old-manifest.json --output run
python3 bench/finalists/campaign.py --manifest run/manifest.json \
  --executable build/search --replay build/replay --output run/shard0 \
  --shard 0 --shards 2 --gpu 0
```

`replay.c` links the native CPU evaluator and dataset reader. `campaign.py` wraps
the normal CUDA search executable; `final_polish.py` has no CUDA ownership or GP
feedback. Results retain the original `search_result` separately from `final_model`.
All phase costs count toward measured fit elapsed time. Inspect
`budget_overrun_seconds` because deadlines are cooperative, not preemptive.

Tests require NumPy/SciPy and a built native replay executable:

```sh
SECANT_REPLAY_TEST_BIN=build/replay python3 -m unittest discover -s bench/finalists -v
```
