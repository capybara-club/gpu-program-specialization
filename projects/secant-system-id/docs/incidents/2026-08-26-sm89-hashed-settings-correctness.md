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

# sm_89 hashed-settings correctness incident

Status: resolved 2026-08-27; repaired production path reopened after repeated-launch, end-to-end replay, and matched-performance validation
Detected: 2026-08-26 EDT  
Affected hardware: Ada-family `sm_89` (tested on RTX 4090)  
Unaffected controls: materialized settings on RTX 4090; tested `sm_120` hashed runs that passed materialized replay

## Material finding

Before this repair, the hashed-incumbent GP score kernel could produce nondeterministic scores on `sm_89` when a specialized AST contained division. Identical CUBIN bytes, inputs, launch dimensions, and output initialization occasionally changed a small number of genome/setting scores. Some corruptions turned an invalid configuration into a falsely low finite MSE, so the search could select a false winner.

This is a correctness failure, not numerical noise and not a reporting-only defect.

## Evidence

- A representative hashed workload produced false winners in both fused-winner and full-MSE output modes.
- The same immutable specialized CUBIN and identical inputs failed across repeated direct launches, excluding GP bookkeeping, specialization nondeterminism, and the eager module queue as the cause.
- A 200-launch hashed stress test still had 22 changed launches after an explicit shared-memory handoff experiment; that experiment was rejected and reverted.
- The equivalent materialized-settings kernel completed 500 repeated identical launches with zero changed launches.
- Hashed runs without division completed the diagnostic sample without changes, localizing the trigger to the specialized division/MUFU schedule on `sm_89`.
- The same immutable-CUBIN division stress test completed 500 repeated `sm_120` launches with zero changed launches. The `sm_120` two-generation reproduction and completed Rohini campaigns also replayed their final winners through the materialized kernel within floating-point tolerance.

## Root cause

The packed specializer removed the input-marker instructions that originally carried the dependency waits for the shared-memory leaf loads. It then branched directly into PTXAS's preserved indirect-dispatch sequence. That sequence begins with an `LDC` which reused dependency barrier slot 0 before the specialized AST's first instruction applied the recovered wait mask. The wait was therefore correct in value but too late in program order. Division exposed the race because its reciprocal path was especially latency-sensitive; neither its scratch register nor its own MUFU write barrier was the root cause.

The repair moves the recovered six-slot wait mask onto the packed fast-path entry branch, before preserved dispatch, and starts each packed AST with no inherited pre-dispatch wait. Direct and split-LM marker sites keep their existing first-instruction wait because they do not execute an intervening dispatch sequence. Wait extraction is also narrowed from 12 control bits to the six actual dependency-barrier bits; the seventh observed bit is a PTXAS register-reuse flag, not a wait slot.

## Validation of the working-tree repair

- The former simple-division reproduction completed 2,000 identical high-occupancy launches on RTX 4090 with zero changed launches.
- The captured complex failing genome completed another 2,000 identical high-occupancy launches with zero changed launches.
- A forced fresh-register specialization completed 1,000 identical high-occupancy launches with zero changed launches.
- Python and C specializers produced byte-identical real Ada CUBINs for the captured complex genome. A genuinely empty scratch pool forced expansion to 76 registers, patched both Ada metadata locations consistently, and also matched byte-for-byte.
- That exact C-specialized, fresh-register complex CUBIN completed 2,000 identical high-occupancy launches with zero changed launches.
- All 79 repository tests pass on both macOS and Ada, including packed-boundary, empty-scratch, and Python/C byte-parity regressions.
- After the production guard was removed, the former simple-division and captured complex-genome reproductions each completed another 2,000 launches through the reopened C hashed pipeline with zero changed launches.
- A 500-generation production GP run on Ada evaluated 262,144,000 configurations through fused GPU winner selection at 12,509,931 configurations/s. Its final hashed winner replayed through the materialized kernel with zero absolute MSE difference, and the report recorded `result_validity: valid`, `all_final_winners_passed_materialized_replay: true`, and no fallback.
- Matched 100-generation full-MSE controls used the same seed, population of 1,024, and 512 settings. Hashed settings sustained 13,522,665 configurations/s in 3.877 seconds; materialized settings sustained 5,806,050 configurations/s in 9.030 seconds. Both reports were valid and replayed with zero absolute MSE difference. The repaired hashed path was 2.33 times as fast in this control.

The full production C hashed-settings mode is now reopened on `sm_89`. The C runtime guard and the CLI/campaign automatic materialized fallback were removed only after the gates above passed. No historical result has been reclassified.

## Secondary CUBIN metadata defect found during audit

Register-expanding specialization previously patched `EIATTR_REGCOUNT` in `.nv.info` but not the nonzero register count encoded in the high byte of the Ada `.text.*` section header's `sh_info`. That produced internally inconsistent CUBIN metadata for fresh-register and LM pressure-fallback paths. The working-tree repair records and patches both locations when the section-header count is present. This header count is toolchain/architecture-specific: checked `sm_90` and `sm_120` CUBINs leave the high byte at zero, so their existing `EIATTR_REGCOUNT`-only behavior is preserved. Inspection rejects conflicting nonzero counts.

The audit also found that Python could grow entirely from fresh registers while the C template descriptor rejected an empty exposed-scratch pool. The C runtime now accepts an empty pool and uses the same fresh-register allocator; a byte-parity regression covers it.

## Result classification

- Historical Ada reports with `gp_settings_mode: hashed_incumbent`: **invalid for quality claims**, even when a single final replay happened to match. The launch fault is intermittent and can alter selection earlier in a run.
- New Ada hashed reports produced by the repaired code are **valid only when their required materialized replay passes** and the report says `result_validity: valid`.
- Ada reports with `gp_settings_mode: materialized` and a passing replay: **valid**, but not performance-equivalent to Rohini's hashed mode. Their reports must be labeled as materialized.
- Rohini `sm_120` hashed reports with passing materialized replay: **valid under the tested replay evidence**. They are not evidence that the `sm_89` path is safe.
- Cross-box timing comparisons between Ada materialized and Rohini hashed settings: **non-equivalent** unless the different settings paths are explicitly part of the comparison.

## Containment and resolution

1. During containment, the C runtime refused hashed-incumbent templates on `sm_89`, and the CLI/campaign wrapper forced the full-MSE materialized path.
2. Those temporary restrictions were removed after the repaired production C path passed the repeated-launch, end-to-end, replay, and throughput gates above.
3. Recovery reports retain a top-level correctness verdict, fallback metadata, per-seed replay tolerance, and per-seed replay pass/fail.
4. A failed final materialized replay writes an invalid report, sets progress to `failed_correctness`, and exits nonzero instead of reporting completion.
5. Progress distinguishes searched seeds from verified/completed seeds. Telegram messages call the in-search MSE provisional and do not call a seed complete before materialized replay.

The materialized path has a real cost. The final matched 100-generation control measured 5.81 million configurations/s materialized versus 13.52 million configurations/s with repaired hashed settings. Treat the effective cost as workload- and machine-state-dependent and always report the actual settings path.

## Campaign disposition

- On 2026-08-27, both paused LM campaigns and the two older recovery tmux sessions were intentionally terminated at the user's request.
- The incomplete current seeds were discarded. Completed reports and previously completed seed artifacts remain on disk.
- No pre-fix Ada hashed report was promoted to valid as part of the repair.

The stopped Ada materialized and Rohini hashed campaigns must not be described as directly comparable settings-path performance.

## Remaining engineering work

- Add an in-campaign sampled differential check if hashed settings are enabled on additional architectures.
- Keep the fixed-CUBIN stress probe as a hardware regression test and run it for every supported architecture/operator family.
