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
"""Matched, all-outcome reporting; no hidden equations or test-driven proposals."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics as st
from benchmarks.lm_tuning.common import load, save

def rows_from(root):
    rows=[]
    for p in sorted((root/'hosts').glob('*/results.jsonl')):
        for line in p.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue  # A copy can end at an unfinished append.
            rows.append(dict(row,lane=p.parent.name))
    seen=set()
    for r in rows:
        key=r['lane'],r['group_id'],r['trial_id']
        if key in seen:raise ValueError('Duplicate completed trial '+str(key))
        seen.add(key)
    return rows

def success(row):
    return bool(row.get('verified')) and row['total_seconds']<=row['settings']['seconds']

def penalized(row):
    return row['total_seconds'] if success(row) else row['settings']['seconds']

def paired(rows,baseline):
    groups=defaultdict(dict)
    for r in rows:groups[r['lane'],r['case_id']][r['trial_id']]=r
    output=[]
    for policy in sorted({r['trial_id'] for r in rows}-{baseline}):
        pairs=[(rs[baseline],rs[policy]) for rs in groups.values() if baseline in rs and policy in rs]
        if not pairs:continue
        output.append(dict(policy=policy,paired_cases=len(pairs),
            gained=sum(success(b) and not success(a) for a,b in pairs),
            lost=sum(success(a) and not success(b) for a,b in pairs),
            baseline_mean_penalty=st.mean(penalized(a) for a,b in pairs),
            policy_mean_penalty=st.mean(penalized(b) for a,b in pairs)))
    return output

def report(root):
    plan=load(root/'plan.json');random_root=Path(plan['random_root'])
    summary=load(random_root/'summary.json') if (random_root/'summary.json').exists() else dict(rows=[],expected_trials=912,completed_trials=0)
    random=summary['rows'];search=[r for r in random if r['kind']=='search'];fits=[r for r in random if r['kind']=='fit']
    bench=rows_from(root);random_pairs=paired(search,'lm_baseline');bench_pairs=paired(bench,'native_lm')
    comparisons=[]
    fit_groups=defaultdict(dict)
    for r in fits:fit_groups[r['lane'],r['case_id']][r['trial_id']]=r
    for (lane,cid),peers in sorted(fit_groups.items()):
        for bank in (4,64,1024,16384):
            options=[peers[f'bank{bank}_lane{width}'] for width in (1,2,4,8)
                     if f'bank{bank}_lane{width}' in peers and peers[f'bank{bank}_lane{width}'].get('work_complete') is True]
            if len(options)<2:continue
            same=len({r['bank_sha256'] for r in options})==1 and len({r['candidate_output_sha256'] for r in options})==1
            best=min(options,key=lambda r:r['seconds'])
            comparisons.append(dict(lane=lane,case_id=cid,states=best['state_count'],parameters=best['declared_parameters'],bank=bank,
                same_input_output=same,available_widths=[r['settings']['lanes_per_fit'] for r in options],
                fastest_width=best['settings']['lanes_per_fit'] if same else None,
                fastest_seconds=best['seconds'] if same else None,
                slowest_over_fastest=max(r['seconds'] for r in options)/best['seconds'] if same else None))
    profiles=[p for r in fits for p in r.get('lm_profile') or []]
    search_profiles=[p for r in search+bench for p in r.get('work_report',{}).get('native_lm_profiles',[])]
    screen=Counter()
    for r in search+bench:screen.update(r.get('work_report',{}).get('screen',{}))
    terminal=(root/'completion.json').exists()
    report_data=dict(terminal=terminal,random_completed=len(random),random_expected=summary['expected_trials'],
        benchmark_completed=len(bench),benchmark_expected=plan['expected_trials'],
        random_policy_pairs=random_pairs,benchmark_policy_pairs=bench_pairs,matched_widths=comparisons,
        fit_statuses=dict(Counter(r['status'] for r in fits)),
        search_statuses=dict(Counter(r['status'] for r in search)),
        benchmark_statuses=dict(Counter(r['status'] for r in bench)),
        partial_fitting_trials=sum(r.get('work_complete') is False for r in fits),
        fit_trials_with_replay_rejections=sum(bool(r.get('replay_mismatches')) for r in fits),
        tiny_fit_calls=sum(p['fit_count']<=128 for p in profiles),fit_calls=len(profiles),
        tiny_search_fit_calls=sum(p['fit_count']<=128 for p in search_profiles),search_fit_calls=len(search_profiles),
        screen=dict(screen),benchmarks=bench)
    save(root/'report.json',report_data)
    lines=['# Odezza search design study','',
        '**'+('Terminal report' if terminal else 'Live, incomplete report')+'**. '
        f"Random study: {len(random)}/{summary['expected_trials']} trials reported. "
        f"Benchmark supplement: {len(bench)}/{plan['expected_trials']} trials reported.",'',
        '## Material incident','',
        'The original random runtime can abort an entire fitting batch when symbolic derivative constant folding '
        'encounters a literal zero denominator. Failed and unstarted trials remain in their original cohort; '
        'they are not retried or counted as timeouts. See `incident/reproduction.txt` and the repository incident note. '
        'The benchmark supplement uses a separately frozen Python repair after CPU and GPU regression validation; '
        'the native library, search budgets, and grammar stay unchanged. Cross-cohort timing is not a paired version comparison.','',
        '## Questions and interpretation','',
        'The random study compares eight fixed search allocations at the same initial configuration budget, '
        'plus separate planted-structure fitting calibration. Its generated systems use 3/6/8 states, '
        '1/3/6 declared coefficients, and dense or sparse/partially observed trajectories. '
        'Generation rejection and coefficient redundancy limit the scope of inference.','',
        'The benchmark supplement keeps source data, complete ICs, known equations, and the supplied grammar. '
        'Twelve qualified, noiseless, single-RHS completion tasks compare native LM, curvature, and a native '
        'screen using up to 21 rather than seven observations. Each has 180 seconds and target MSE 1e-6. '
        'Random recovery uses 240 seconds and 1e-10. Do not pool their success rates. '
        'MDBench/ODEBench pairs share equation ancestry. Biological tasks are different; larger than eight states '
        'and unqualified tasks remain outside this pilot. No derivatives or hidden equations enter search.','',
        'A success below means independently verified within its total trial wall budget. A late verified '
        'result remains in raw records but counts as a budget miss. A miss costs the full budget in the '
        'penalized-time column. Only finished matched policy pairs enter contrasts; there is no successful-only ranking.','',
        '## Random search allocation','', '| Policy | Matched cases | Gained / lost vs baseline | Mean penalized time: baseline → policy |',
        '|---|---:|---:|---:|']
    for r in random_pairs:lines.append(f"| {r['policy']} | {r['paired_cases']} | {r['gained']} / {r['lost']} | {r['baseline_mean_penalty']:.2f} → {r['policy_mean_penalty']:.2f} s |")
    if not random_pairs:lines.append('| Awaiting matched recovery pairs | — | — | — |')
    lines+=['','### Recovery by public problem stratum','', '| States | Coefficients | View | Policy | Verified within budget / reported |', '|---:|---:|---|---|---:|']
    strata=defaultdict(list)
    for r in search:strata[r['state_count'],r['declared_parameters'],r['view'],r['trial_id']].append(r)
    for (n,p,view,policy),rs in sorted(strata.items()):lines.append(f'| {n} | {p} | {view} | {policy} | {sum(success(r) for r in rs)} / {len(rs)} |')
    lines+=['','## Specialized fitting shape grid','',
        'Each row is one fixed candidate population on one host. Widths are compared only when complete '
        'populations use the same bank and produce identical retained candidate outputs. These are single '
        'measurements, not statistically established defaults; unavailable widths remain resource outcomes.','',
        '| GPU worker | States | Coefficients | Starts / AST | Matched widths | Fastest width | Fitting time | Slowest / fastest |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in comparisons:
        if r['same_input_output']:lines.append(f"| {r['lane']} | {r['states']} | {r['parameters']} | {r['bank']} | {r['available_widths']} | {r['fastest_width']} | {r['fastest_seconds']:.3f} s | {r['slowest_over_fastest']:.2f}× |")
        else:lines.append(f"| {r['lane']} | {r['states']} | {r['parameters']} | {r['bank']} | Output mismatch | — | — | — |")
    lines+=['','## Benchmark recovery','', '| Case | States | Policy | Status | Wall time | Final test MSE |', '|---|---:|---|---|---:|---:|']
    for r in bench:
        mse=r.get('heldout_mse');mse='—' if mse is None else f'{mse:.6g}'
        lines.append(f"| {r['case_id']} | {r['state_count']} | {r['trial_id']} | {r['status']} | {r['total_seconds']:.2f} s | {mse} |")
    if not bench:lines.append('| Waiting for random campaign to finish | — | — | — | — | — |')
    lines+=['','## Work, failures, and numerical qualifications','',
        f"- Fitting outcomes: `{report_data['fit_statuses']}`; partial populations: {report_data['partial_fitting_trials']}.",
        f"- Fitting trials rejecting at least one candidate during CPU replay: {report_data['fit_trials_with_replay_rejections']}.",
        f"- Native calls with at most 128 fits: {report_data['tiny_fit_calls']}/{len(profiles)} in calibration, "
        f"{report_data['tiny_search_fit_calls']}/{len(search_profiles)} in searches.",
        f"- Reported completed screening work: {screen['coefficient_trials']:,} configurations, {screen['rk4_steps']:,} scheduled RK4 steps. "
        f"Invalid configurations: {screen['invalid_configurations']:,}; padding: {screen['padding_configurations']:,}; "
        f"discarded deadline commands: {screen['deadline_discarded_commands']:,}.",
        '- Scheduled integration work can exit early. AST occurrences, structural identities, and algebraic equivalence are different quantities. '
        'Missing distinct counts stay unknown. Module-load calls and device-event intervals overlap; never add them as exclusive costs.',
        '- The readiness study found rational candidates whose coarse-step MSE reproduced on CPU but failed finer integration. '
        'Current independent replay remains mandatory; replay rejection counts alone do not prove every rejection has that cause.','',
        '## Design decisions these data support','',
        '1. Preserve structural coverage until a matched, fresh-cohort comparison shows that spending more on bindings or banks improves recovery. '
        'Use gained/lost cases and all-outcome time, then inspect the losing cases before selecting a policy.',
        '2. Choose native width using measured state/parameter/population cells and resource diagnostics. '
        'The fastest occupied width can differ from the first width that compiles. Confirm any dispatch rule on fresh ASTs.',
        '3. Treat frequent tiny fitting calls as a search-scheduling issue. Test useful additional starts or related candidates inside each module. '
        'The earlier two-versus-16-pack pilot showed no meaningful gain from a longer queue alone.',
        '4. If finer screening improves benchmark verification, add numerical-fidelity promotion before expensive refinement. '
        'Keep the grammar broad; separately investigate poles, large invalid fractions, sparse observability, and nonidentifiable coefficients.',
        '5. Reconcile every rejected admission, unsupported shape, partial bank, discarded command, and unreported trial before comparing throughput. '
        'These exploratory results do not establish superiority over PySR or reliable recovery of arbitrary systems.','',
        '## Provenance','',
        f'- Random inputs and reports: `{random_root}`.',
        '- `plan.json` freezes benchmark selection, settings, source discrepancy gates and public input hashes.',
        '- `execution.json`, `source-hashes.json`, and `repair-validation.json` identify the supplemental code, Python repair, and unchanged native binaries.',
        '- `report.json`, per-worker `results.jsonl`, and retained remote trial directories carry the detailed evidence.',
        '- No target-specific grammar changes or test-guided repairs are permitted during these fixed comparisons.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    flat=[dict(cohort='random',**r) for r in search]+[dict(cohort='benchmark',**r) for r in bench]
    with (root/'recovery.csv').open('w',newline='') as stream:
        names=['cohort','case_id','lane','trial_id','status','verified','total_seconds','heldout_mse','ast_occurrences','distinct_concrete_asts','configurations','accounting_complete']
        writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore');writer.writeheader();writer.writerows(flat)
    return report_data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();report(a.root)
