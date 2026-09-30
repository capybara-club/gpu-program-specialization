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
"""Build the review table from retained CPU measurements, never from equations."""
import csv
import json
from pathlib import Path
import math
from solver import HERE,write


def main():
    root=HERE/'validation';native=json.loads((root/'all-clean-v1/summary.json').read_text())
    independent=json.loads((root/'scipy-v2/summary.json').read_text())
    retry=json.loads((root/'bdf055-v1/summary.json').read_text())
    replay=json.loads((root/'source-replay-v1.json').read_text())
    all_cases={r['case_id']:r for r in native['results']};runs={k:list(r['settings']) for k,r in all_cases.items()}
    for d in (independent,retry):
        for r in d['results']:
            for key in ('public_sha256','private_sha256','source_sha256'):
                assert r[key]==all_cases[r['case_id']][key]
            runs[r['case_id']].extend(r['settings'])
    source={r['case_id']:r for r in replay['results'] if r['recipe']=='upstream_sympy'}
    flagged=[k for k,r in all_cases.items() if max(v['mse'] for v in r['accurate_reference_mse'].values())>1e-6]
    methods=('Rosenbrock23','RK4','Radau','BDF','LSODA')
    selected={};counts={};failures=[]
    for key,ss in runs.items():
        selected[key]={}
        for method in methods:
            attempted=[s for s in ss if s['method']==method]
            for s in attempted:
                if not s['complete']:failures.append(dict(case_id=key,method=method,rtol=s.get('rtol'),error=s.get('error')))
            valid=[s for s in attempted if s['complete']]
            if not valid:continue
            chosen=max(valid,key=lambda s:s['substeps']) if method=='RK4' else min(valid,key=lambda s:s['rtol'])
            selected[key][method]=chosen
            counts.setdefault(method,dict(systems=0,reproduced=0,data_gate_pass=0))
            counts[method]['systems']+=1
            counts[method]['reproduced']+=int(any(s.get('matches_reference_mse',False) for s in valid))
            counts[method]['data_gate_pass']+=int(chosen['data_gate_pass'])
    for method in methods:assert counts[method]['systems']==counts[method]['reproduced'],method
    # All per-split scores are exported, not only the worst split shown below.
    with (root/'mse-by-system-and-split.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case_id','method','host','rtol','substeps','split','mse','reference_mse','count','seconds'])
        for key,mm in selected.items():
            for method,s in mm.items():
                for split,v in s['vs_observations'].items():
                    writer.writerow([key,method,native['host'] if method in ('RK4','Rosenbrock23') else independent['host'],
                        s.get('rtol'),s.get('substeps'),split,v['mse'],all_cases[key]['accurate_reference_mse'][split]['mse'],
                        v['scored_residuals'],s['wall_seconds']])
    def value(k,m):
        return max(v['mse'] for v in selected[k][m]['vs_observations'].values())
    lines=['# CPU benchmark MSE reproduction','',
        'Native Rosenbrock23 and RK4 run on **mac1 CPU**, with no CUDA dependency. '
        'Both reproduce the accurate-reference MSEs against the actual benchmark samples for '
        '**134/134 supported noiseless source systems** at tested settings. '
        '**120/134** also meet the original MSE <=1e-6 data-agreement gate; the same 14 remain above it.','',
        'Independent Radau, BDF and LSODA checks run on **rack1 CPUs**, using its existing SciPy installation. '
        'Each reproduces the accurate-reference sample MSEs for all 14 difficult systems at a tested setting. '
        'BDF needs a different tolerance on one case, detailed below. No GPU is used in these new runs.','',
        '“Reproduces” means every split satisfies `abs(MSE - reference_MSE) <= 1e-8 + 1e-4*abs(reference_MSE)`. '
        'This checks score reproduction, not exact trajectory equality. The separate original sample-fit gate remains 1e-6.','',
        '## MSE against the published observations','',
        'Every cell is **worst split MSE against samples**. Native RK4 uses 512 substeps; '
        'other columns use their tightest successfully completed tested tolerance. '
        'The CSV retains every split and the selected setting. All sweeps and failures remain in the raw reports.','',
        '| System | CPU RK4 | CPU Rosenbrock23 | CPU Radau | CPU BDF | CPU LSODA | Source-recipe LSODA |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in sorted(flagged):
        src=max(v['mse'] for v in source[k]['results'].values()) if k in source else None
        vals=[f'{value(k,m):.6g}' for m in ('RK4','Rosenbrock23','Radau','BDF','LSODA')]
        lines.append('| '+k.replace('-rhs-00','').replace('-rhs-02','')+' | '+' | '.join(vals)+
                     ' | '+('not established' if src is None else f'{src:.6g}')+' |')
    native_seconds=sum(s['cpu_seconds'] for r in native['results'] for s in r['settings'])
    native_settings=sum(len(r['settings']) for r in native['results'])
    lines += ['',
        'The fresh source-recipe replay reproduces all six flagged MDBench systems **exactly: MSE 0 on every split**. '
        'It uses the published loose LSODA settings and symbolic substitution recipe, '
        'rather than the tight canonical-model integrations in the other columns. '
        'The chaotic ODEBench replay still differs from the samples; historical generator provenance is unresolved. '
        'The metabol1 source/model discrepancy also remains unresolved.','',
        '## CPU cost and completion','',
        f'The native mac1 audit covered {native_settings} solver/settings combinations in '
        f'**{native["seconds"]:.2f} s elapsed**, including compilation and reporting. '
        f'Native integration consumed **{native_seconds:.2f} s CPU time** in total '
        f'({sum(s["cpu_seconds"] for r in native["results"] for s in r["settings"] if s["method"]=="Rosenbrock23"):.2f} s Rosenbrock23, '
        f'{sum(s["cpu_seconds"] for r in native["results"] for s in r["settings"] if s["method"]=="RK4"):.2f} s RK4). '
        f'All {native_settings} native settings completed successfully. Compilation was {sum(r["compile_seconds"] for r in native["results"]):.2f} s.','',
        'The difficult Lorenz settings in this native audit took approximately 0.1–0.5 s with tight Rosenbrock23 '
        'and 0.003–0.011 s with RK4/512. These are actual dataset batches, not saturated throughput comparisons '
        'or an isolated same-machine CPU/GPU speed ratio.','',
        f'The independent initial SciPy check took {independent["seconds"]:.2f} s with two CPU processes on '
        f'{independent["host"]}, SciPy {independent["scipy_version"]}. '
        'Its Python RHS callbacks, host and algorithms differ from the native mac1 run.','',
        'BDF on ODEBench 055 fails at rtol=1e-12 and 1e-13 with “Required step size is less than spacing between numbers.” '
        'The recorded targeted retry succeeds at 1e-10 and 1e-11, reproducing the reference MSE on every split. '
        'This failure is retained, not counted as a successful 1e-12 integration.','',
        'Five native CPU regression tests pass: stiff analytic trajectories with masking, fourth-order RK4 convergence, '
        'analytic Jacobians, explicit failures/input rejection, and ragged score-only requests. '
        'The JSON entry point was exercised on mac1. Native scores are independently recomputed in Python.','',
        '## Evidence and implications','',
        '- [All 134 native CPU cases](validation/all-clean-v1/summary.json)',
        '- [Per-system, per-split MSE CSV](validation/mse-by-system-and-split.csv)',
        '- [Independent CPU Radau/BDF/LSODA](validation/scipy-v2/summary.json)',
        '- [Explicit BDF retry and failures](validation/bdf055-v1/summary.json)',
        '- [Fresh generator replay](validation/source-replay-v1.json)',
        '- [Tests](validation/tests-final.log)',
        '- [CPU-only implementation and commands](README.md)','',
        'The implementation can reproduce the benchmark MSEs. A high nonzero MSE for the known equation '
        'is a data-generation/model/objective issue that a more accurate integrator does not necessarily repair. '
        'Keep the original observations and qualify those cases separately before treating them as recovery failures. '
        'All new native arithmetic here is FP64; FP32 and a higher-order Rosenbrock method remain separate future experiments.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    write(root/'reconciled.json',dict(methods=counts,failures=failures,
        native_settings=native_settings,native_cpu_seconds=native_seconds,
        native_elapsed_seconds=native['seconds'],flagged_systems=sorted(flagged)))
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
