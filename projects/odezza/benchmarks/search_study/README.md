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

# Search design experiments — September 10

The corrected-native random study is running in mac1 tmux as
`odezza-focused-night`. Its frozen inputs and runtime are unchanged.
This folder adds public-only benchmark trials and an automatically refreshed
report outside the core and outside that running controller.

## Fixed experiment

- Random: 33 accepted depth-three systems, eight policies, nine fitting cells
  across three hosts; 912 scheduled trials. See
  [the random experiment](../lm_tuning/FOCUSED_OVERNIGHT.md).
- Benchmarks: 12 qualified noiseless tasks, three fixed policies each, 180 seconds
  per trial and raw source-unit target MSE 1e-6. MDBench/ODEBench use four matched
  source models spanning one through four states. Four distinct biological
  models span three/five states. Selection uses public IDs and qualification,
  never hidden equations or results from this run.
- Benchmark policies: native LM, curvature, and native LM with up to 21 screening
  observations instead of seven. The supplied depth-three grammar and coefficient
  range [-100,100] stay fixed. More screen observations also change integration
  resolution per unit time; this is a whole screening-policy comparison.
- Benchmark work waits for every random worker to finish, and stays within the
  original nine-hour deadline. Whole policy groups require a drain margin before
  admission. Fewer remaining minutes can leave explicit unstarted benchmark work.

These are single-RHS completion adaptations with coupled trajectory scoring,
not official leaderboard protocols. No derivative targets or hidden equations
cross the worker input. A separate test ledger belongs to each predeclared
policy; no test-guided revisions or retries are made. Qualification excludes
noise, unresolved source-model discrepancy and unsupported cases. The native LM
API currently admits at most eight states; larger biological tasks are not
claimed covered. The full benchmark inventory remains in its original manifests.

## Entry points and evidence

```sh
python3 -B -m benchmarks.search_study.prepare --root benchmarks/search_study/runs/20260910-robustness --random-root benchmarks/lm_tuning/runs/focused-night-20260910
python3 -B -m benchmarks.search_study.manage deploy --root benchmarks/search_study/runs/20260910-robustness
python3 -B -m benchmarks.search_study.manage run --root benchmarks/search_study/runs/20260910-robustness
```

Run the final command in mac1 tmux. Deployment creates a separate supplement
directory on each GPU host; it reuses the frozen native library, backend and
search controller. The September 10 supplement uses the separately frozen
`runtime-v5` with the [candidate-domain repair](../../docs/incidents/2026-09-10-singular-lm-proposals.md);
the original random runtime is unchanged. No rebuild or core edit is required. Public task validation
and the existing conservative fixed-RHS patch-sizing helper are included by
source hash. Preparation is checked for every case/policy before GPU submission.

`REPORT.md` and `report.json` refresh during the campaign and at terminal status.
They contain paired recovery gains/losses, budget-penalized time, state/parameter/
observation strata, complete input/output-matched native width comparisons,
benchmark MSEs, failure/partial-work counts, replay rejections and work accounting.
Overlapping native timing spans are never presented as exclusive phases.
`recovery.csv` supports later analysis. Raw trial artifacts stay on their worker
hosts; summaries and public input identities are collected on mac1.

The original random coordinator sends its completion Telegram. The supplement
sends another when the combined report is terminal, explicitly identifying
incomplete/failed work. This is an exploratory report, followed by fresh-cohort
confirmation before promoting defaults or comparing against PySR.

The active mac1 report supervisor is `odezza-search-study`. It waits until all
original workers are terminal and their processes have exited. A known,
reproduced singular-candidate failure does not prevent the independent supplement
after its repair regression passes; any other unresolved failure blocks launch.
Both completed and unstarted work remain visible in the combined report.

Read [initial findings](INITIAL_FINDINGS.md) for the completed calibration and
provisional design implications. The live report is
`runs/20260910-robustness/REPORT.md`; it refreshes throughout the campaign.
