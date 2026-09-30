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
"""Summarize retained population curves and paired search integration checks."""
import argparse
from collections import Counter
from pathlib import Path
import statistics as st
from .common import load,save


def report(curves,searches):
    rows=[];checks=[];setup=[];screen_phases=[]
    for host in ('rack1','rohini','ada'):
        d=load(curves/(host+'-curve.json'))
        if d['status']['status']!='complete' or d['status']['output_mismatches']:
            raise ValueError('Incomplete or mismatched curve '+host)
        rows.extend(dict(r,host=host) for r in d['results'])
        d=load(searches/(host+'-search-check.json'))
        if d['status']['status']!='complete':raise ValueError('Incomplete search check '+host)
        checks.extend(dict(r,host=host) for r in d['results'])
        setup.extend(dict(r,host=host) for r in load(searches/(host+'-setup-failure.json'))['results'])
        attribution=searches/(host+'-all-screen-attribution.jsonl')
        if attribution.exists():
            import json
            for line in attribution.read_text().splitlines():
                r=json.loads(line)
                if Path(r['file']).parent.name!='w0000-000-gpu0':continue
                case,mode=r['case'].rsplit('-',1)
                run=next(c for c in checks if c['host']==host and c['case_id']==case and c['mode']==mode)
                screen_phases.append(dict(host=host,case=case,mode=mode,
                    pipeline_create_seconds=r['setup']['pipeline_create_seconds'],
                    pipeline_reused=r['setup']['pipeline_reused'],
                    family_seconds=sum(f['timing']['family_seconds'] for f in r['families']),
                    fit_wall_seconds=run['timing']['fit_wall_seconds'],total_seconds=run['total_seconds'],
                    source=r['file'],source_sha256=r['sha256']))
    timed=[r for r in rows if r['repeat']>=0]
    table=[]
    for host in ('rack1','rohini','ada'):
        for case in sorted({r['case_id'] for r in timed}):
            for width in (1,8):
                a=[r for r in timed if r['host']==host and r['case_id']==case and r['width']==width and r['target']==512]
                b=[r for r in timed if r['host']==host and r['case_id']==case and r['width']==width and r['target']==2048]
                old,new=st.median(r['seconds'] for r in a),st.median(r['seconds'] for r in b)
                table.append(dict(host=host,case=case,width=width,old_fits=a[0]['fits'],new_fits=b[0]['fits'],
                    old_seconds=old,new_seconds=new,time_increase_percent=100*(new/old-1),
                    old_native_seconds=st.median(r['native_seconds'] for r in a),
                    new_native_seconds=st.median(r['native_seconds'] for r in b)))
    native={}
    for mode in ('baseline','population'):
        rs=[r for r in checks if r['mode']==mode]
        ps=[p for r in rs for p in r['work_report'].get('native_lm_profiles',[])]
        native[mode]=dict(cases=len(rs),verified=sum(r['verified'] for r in rs),
            attempt_seconds=sum(r['total_seconds'] for r in rs),native_calls=len(ps),
            fits=sum(p['fit_count'] for p in ps),median_fits_per_call=st.median(p['fit_count'] for p in ps),
            calls_at_most_128=sum(p['fit_count']<=128 for p in ps),
            lane_masks=dict(Counter(p['shapes']['used_lanes_mask'] for p in ps)),
            outcomes=dict(Counter(r['status'] for r in rs)))
    prefix_worsening=[]
    for host in ('rack1','rohini','ada'):
        for case in sorted({r['case_id'] for r in timed}):
            base=next(r for r in timed if r['host']==host and r['case_id']==case and r['target']==0 and r['width']==1)
            original={r['candidate_id']:r['mse'] for r in base['winners']}
            for r in timed:
                if r['host']!=host or r['case_id']!=case:continue
                winners={r['candidate_id']:r['mse'] for r in r['winners']}
                for key,mse in original.items():
                    if key not in winners or winners[key]>mse:prefix_worsening.append((host,case,r['target'],r['width'],key))
    summary=dict(timed_runs=len(timed),warmups=len(rows)-len(timed),
        cpu_checks=sum(len(r['checks']) for r in rows),
        cpu_replay_rejections=sum(not c['matched'] for r in rows for c in r['checks']),
        prefix_worsening=prefix_worsening,curves=table,searches=native,
        setup_failures=[{k:r[k] for k in ('host','case_id','status','error','total_seconds')} for r in setup],
        screen_phases=screen_phases,
        search_comparison_status='Non-equivalent cold/warm pipeline preparation; not a causal population-policy speed comparison.',
        incomplete_work_accounting=[dict(host=r['host'],case=r['case_id'],mode=r['mode'])
            for r in checks if not r.get('accounting_complete',False)])
    save(curves/'summary.json',summary)
    lines=['# Native LM population and occupancy experiment — 2026-09-10','',
        'Bounded extra starts and actual-width launch geometry are implemented outside the native core. '
        'The full search API accepts the controls, including simple baseline campaigns. Defaults remain '
        'unchanged: more occupied fitting is not automatically the fastest recovery policy.','',
        '## Prepared-candidate results','',
        f"{summary['timed_runs']} timed adapter runs plus {summary['warmups']} warm-ups on RTX5080/5090/4090. "
        'Four public retained requests cover 3/6/8 states and a partially observed six-state case. '
        'One is a simple two-iteration baseline batch; the other three use 16 fitting iterations. '
        'At most eight original candidates/request were retained. The control uses the supplied '
        'coefficient rows explicitly (one generated start per input row), rather than reproducing '
        'the old adapter\'s extra random starts. This is a diagnostic population experiment, not an '
        'identical replay of the original search policy.','',
        'Widths 1 and 8, minimum fits/module 0/128/512/2048/8192, two timed repeats with rotated order. '
        'Total adapter time includes population preparation, JSON, allocations and transfers; one-time '
        'engine startup and CPU checks are outside it, and cold template warm-ups remain recorded. '
        'No independently authored CUDA or alternate fitting implementation was used.','',
        '### Eight-state batch: 4x fitting population','',
        '| GPU | Fits: 512 → 2048 per module | Adapter time | Time increase | Native runner time |',
        '|---|---:|---:|---:|---:|']
    for r in table:
        if r['case']=='case-008' and r['width']==8:
            lines.append(f"| {r['host']} | {r['old_fits']:,} → {r['new_fits']:,} | {r['old_seconds']:.3f} → {r['new_seconds']:.3f} s | {r['time_increase_percent']:.1f}% | {r['old_native_seconds']:.3f} → {r['new_native_seconds']:.3f} s |")
    lines+=['', 'At width eight, 512 and 2048 fits/module launch 128 and 512 one-warp CTAs/module. '
        'Both give much more parallel work than the historical four-fit module. Two modules remain '
        'the core concurrency limit. Whole-submission CTA counts are not simultaneous resident CTAs '
        'or achieved occupancy. No ncu/nsys executable was available on these hosts; no profiler was installed.','',
        f"All retained-output hashes agree across widths and repeats at each population. "
        f"{summary['cpu_checks']} selected CPU FP64 checks: {summary['cpu_replay_rejections']} rejections. "
        f"Candidate winners worsened against the preserved-start control: {len(prefix_worsening)}. "
        'CPU checks use the same RK4 resolution as the fitting objective; they do not replace finer-step '
        'or held-out recovery verification. Fitting MSE generally improved only slightly with extra starts.','',
        'The largest tested population is not universally best: width one wins several large-bank '
        'cells, while host sampling/serialization increasingly matters. The 512→2048 increment costs '
        'more on the cheap, already-near-target six-state batch. These measurements support configurable '
        'population/width policies, not an unconditional eight-lane or largest-bank default.','',
        '## Full search integration checks','',
        'Six previously used public systems, paired on the same host/seed with 60-second budgets. '
        'The experimental policy uses eight lanes and a 512-fit/module target with at most 65,536 '
        'extra fits/request. Baseline uses one preferred lane and the original starts. '
        'Case order alternates policies. These retrospective checks validate integration and show '
        'observed timings; they are not fresh blind evaluation or an isolated occupancy speedup. Template '
        'preparation/cache history, lane width, starts and subsequent search decisions can all differ.','',
        '| Host | Case | Baseline outcome / seconds | Population outcome / seconds |',
        '|---|---|---|---|']
    for host,case in sorted({(r['host'],r['case_id']) for r in checks}):
        rs={r['mode']:r for r in checks if r['host']==host and r['case_id']==case}
        a,b=rs['baseline'],rs['population']
        lines.append(f"| {host} | {case} | {a['status']} / {a['total_seconds']:.2f} | {b['status']} / {b['total_seconds']:.2f} |")
    if screen_phases:
        lines+=['', '### Timing attribution correction','',
            'Retained setup reports confirm a cold/warm confound. For case-024 on ada, population '
            'spent 25.65 s creating its scoring pipeline; baseline reused it. The actual screen '
            'family took 9.103 versus 9.088 s. Fitting took 3.255 versus 2.348 s. Thus the '
            '14.79→42.27 s difference is mostly setup, not evidence that population filling '
            'caused a large fitting or exploration regression. The apparent improvements in '
            'cases 001/008/028 favor the policy run second and are likewise mostly cold setup. '
            'Do not attribute the aggregate attempt-time difference to the population policy.','',
            '| Host / case | Policy | Initial pipeline creation | Initial screen family | All fitting | Total |',
            '|---|---|---:|---:|---:|---:|']
        for r in screen_phases:
            lines.append(f"| {r['host']} / {r['case']} | {r['mode']} | {r['pipeline_create_seconds']:.3f} s | {r['family_seconds']:.3f} s | {r['fit_wall_seconds']:.3f} s | {r['total_seconds']:.3f} s |")
        lines+=['', 'Screen-family timing includes pipeline dispatch/load/execution/report work, '
            'not GPU kernel time alone. Later waves are excluded from the initial-screen columns. '
            'Cold creation includes compilation, inspection, specialization and resource setup; '
            'the retained timer does not separately measure those components. '
            'See [timing review](../../OCCUPANCY_TIMING_REVIEW.md) for the cache boundary and next controls.']
    lines+=['', '| Policy | Verified | Sum of attempt times | Recorded native fits | Median fits/call | Calls with <=128 fits |',
            '|---|---:|---:|---:|---:|---:|']
    for mode,r in native.items():
        lines.append(f"| {mode} | {r['verified']}/{r['cases']} | {r['attempt_seconds']:.2f} s | {r['fits']:,} | {r['median_fits_per_call']:g} | {r['calls_at_most_128']}/{r['native_calls']} |")
    lines+=['', 'Timeout durations include draining admitted work and exceed 60 seconds slightly. '
        'Both policies retain the unchanged independent verification criteria. Three attempts '
        '(rack1 case-004 baseline; rohini case-032 both policies) mark accounting_complete=false: '
        'deadline-killed scoring work is unknown. Recorded work is not an exact total for those attempts. '
        'The paired outcomes are valid, but timing does not isolate the population policy.','',
        '## Deployment incident','',
        'The first full-search attempt on each host stopped because this isolated deployment omitted '
        '`libodezza_expand.so`. All three failed records/times are retained in the search-check directory. '
        'The existing validated library was copied, all three native binary hashes checked against the '
        'prior runtime, and the same frozen source/inputs reran into `searches-r2`. No kernel or algorithm '
        'repair was needed. These failed setup attempts are outside the paired successful-runtime table.','',
        '## Next decision','',
        'Keep population filling opt-in. First control cold/warm template preparation in both policies '
        'and retain separate first-request latency. Then compare a budgeted policy that applies larger populations to '
        'selected useful modules against blanket filling on fresh problems. Preserve structural exploration '
        'and family coverage, and measure time-to-verified-solution. Additional compatible toggle variants '
        'may spend the available parallel capacity better than still more starts around the same structure. '
        'Hardware occupancy and correlated host/device idle-time attribution remain unmeasured.','',
        '- API: `scratch/fitting_batch_trial/lm_toggle/POPULATION.md`.',
        '- Per-host curve JSONs, frozen source hashes, input fixtures and logs are alongside this report.',
        '- Paired search plans/results and deployment failure/repair records: `../occupancy-20260910b/`.','']
    (curves/'REPORT.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--curves',type=Path,required=True)
    p.add_argument('--searches',type=Path,required=True);a=p.parse_args()
    print(report(a.curves,a.searches)['searches'])
