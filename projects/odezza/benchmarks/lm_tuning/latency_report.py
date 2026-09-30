# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Report completed cache-controlled policy checks without treating counts as utility."""
import argparse
from pathlib import Path
import statistics
from .common import load,save

def report(root):
    hosts=('rack1','rohini','ada');data={h:load(root/(h+'-results.json')) for h in hosts}
    for h,d in data.items():
        if d['status']['status']!='complete' or len(d['results'])!=12:raise ValueError('Incomplete comparison: '+h)
        if not all(r['cache_control_valid'] for r in d['results']):raise ValueError('Unprepared timed templates: '+h)
    rows=[dict(r,host=h) for h,d in data.items() for r in d['results']]
    modes=('baseline','occupied','neighbors');summary={}
    for mode in modes:
        rs=[r for r in rows if r['mode']==mode];profiles=[p for r in rs for p in r['work_report']['native_lm_profiles']]
        summary[mode]=dict(verified=sum(r['verified'] for r in rs),attempts=len(rs),
            seconds=sum(r['total_seconds'] for r in rs),recorded_native_fits=sum(p['fit_count'] for p in profiles),
            incomplete_work_accounting=sum(not r['accounting_complete'] for r in rs),
            median_fits_per_call=statistics.median(p['fit_count'] for p in profiles))
    lines=['# Prepared scoring and useful fitting — September 10, 2026','',
        'The native template/cache and sampling repairs are enabled in this isolated build. '
        'Broad binding-neighbor expansion and occupied fitting remain explicit controls; '
        'the grammar and verification criteria are unchanged.','',
        '## Preparation and restart','',
        '| Host | Populate empty Odezza artifact cache (four shapes) | Restart and read artifacts | LM shape preparation |',
        '|---|---:|---:|---:|']
    for h,d in data.items():lines.append(f"| {h} | {d['cold']['seconds']:.3f} s | {d['restart']['seconds']:.3f} s | {d['lm']['seconds']:.3f} s |")
    lines+=['', 'Empty application cache does not imply an empty compiler/driver/OS cache. '
        'Restart preparation used disk artifacts. These are preparation times outside the '
        'following search budgets, not free work; a cold one-off use must include them. '
        'Templates have no equations or trajectory data. Service readiness declares its supported shapes.','',
        '## Complete recovery checks','',
        'Six retained public systems, two repeats per policy, 60-second search budgets. '
        'Every timed attempt restarts its C worker against the same prepared artifacts. '
        'Both LM lane policies are prepared before timing. All 36 attempts pass the checks '
        'for no unprepared scoring or LM templates. Trials are retrospective, not fresh blind evaluation. '
        'The three policies vary both fitting width and population/neighbor work; this does not '
        'isolate lane width alone.','',
        '| Host / case | Baseline median / solved | Occupied median / solved | Neighbors median / solved |',
        '|---|---:|---:|---:|']
    for h,case in sorted({(r['host'],r['case_id']) for r in rows}):
        values=[]
        for mode in modes:
            rs=[r for r in rows if r['host']==h and r['case_id']==case and r['mode']==mode]
            values.append(f"{statistics.median(r['total_seconds'] for r in rs):.2f} s / {sum(r['verified'] for r in rs)}/2")
        lines.append('| '+h+' / '+case+' | '+' | '.join(values)+' |')
    lines+=['', '| Policy | Verified attempts | Sum of attempt times | Recorded native fits | Median fits/call | Incomplete work accounting |',
        '|---|---:|---:|---:|---:|---:|']
    for mode,s in summary.items():
        lines.append(f"| {mode} | {s['verified']}/{s['attempts']} | {s['seconds']:.2f} s | {s['recorded_native_fits']:,} | {s['median_fits_per_call']:g} | {s['incomplete_work_accounting']} |")
    lines+=['', 'Timeouts include drain/report time. An incomplete accounting flag means a deadline-killed '
        'scoring command has unknown work; recorded counts are lower bounds. '
        'Native event, module load and completion-wait intervals overlap and cannot be added '
        'as exclusive costs. No achieved hardware occupancy claim is made.','',
        '## Interpretation and remaining work','',
        'The setup/reuse repair has a direct measured effect. More fitting work does not '
        'uniformly reduce recovery time. Broad neighbor expansion can consume the fitting '
        'budget and require another search wave. Keep both search controls explicit; use '
        'fresh systems to test selective augmentation rather than promoting this aggressive '
        'neighbor policy as the default. A timed-out case is still unresolved.','',
        'Native sampling is `semantic_candidate_starts_v2`: deadlines, device, width and '
        'tags no longer perturb original coefficient starts. The explicit seed, objective, '
        'program and initial coefficients do. Exact sampled rows remain in packing reports. '
        'Time budgets can still alter which later work is admitted.','',
        '## Validation and deviations','',
        '- Public C99 exports and native scoring tests pass on all three hosts. The new '
        'template test covers artifact round-trip, corruption/shape rejection, independent '
        'lifetime and GPU replay.',
        '- Thirty comparison-build Python/API tests per host pass, including RNG/reducer changes, '
        'worker restart, corrupted-artifact recovery, state-binding packing and sampling invariance. '
        'The existing seven planner tests also pass from their documented directory.',
        '- Final source adds a native/legacy default-sampling compatibility check: '
        '31 tests pass on each host. An isolated HTTP service on rack1 prepares both '
        'GPUs with one compile and one artifact read, restarts with two artifact reads, '
        'and produces bitwise identical winners in simultaneous scoring on both GPUs. '
        'This smoke test covers one declared shape; the four-shape timings above are separate. '
        'See `latency-20260910-final` for final source hashes and validation. '
        'Existing resident services were not replaced or restarted.',
        '- Initial combined test invocation used the wrong directory for that standalone '
        'planner suite; its failure and corrected run are retained under `latency-20260910b`.',
        '- Initial benchmark preparation used the wrong legacy LM marker layout for larger '
        'states and stopped before any searches. The helper now uses the same two-site layout '
        'as the adapter; a 36-shape regression check was added. See `latency-20260910c/d`.',
        '- rack1 has no tmux; all three experiment sessions are coordinated by installed '
        'mac1 tmux. No tooling was installed.',
        '- The first cache-controlled run used historical whole-request sampling and is '
        'preserved under `latency-20260910d`. Remaining wall time entered its seed. This '
        'report uses the separate corrected run; never pool the two sampling versions.',
        '- Runtime preparation/policy fixes are outside `core/`; the native addition is '
        'an opaque immutable template API, with unchanged scoring/LM math and descriptor layouts.','',
        'Source hashes, public plans, launch scripts, preparation records and per-host results '
        'are retained alongside this report. Worker reports and exact coefficient rows remain '
        'under the corresponding isolated remote experiment directory.','']
    (root/'REPORT.md').write_text('\n'.join(lines));save(root/'summary.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();print(report(a.root))
