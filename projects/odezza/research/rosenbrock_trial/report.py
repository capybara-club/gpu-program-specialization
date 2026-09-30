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
"""Reconcile retained audit sweeps; tolerance selection is PRIVATE validation."""
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent/'validation'


def main():
    names=('all-clean-v1','flagged-v1','chaotic-tight-v1','rossler058-tight-v1')
    reports={name:json.loads((ROOT/name/'summary.json').read_text()) for name in names}
    base=reports[names[0]]; systems={r['case_id']:dict(source=r,runs=[]) for r in base['results']}
    for name,report in reports.items():
        for r in report['results']:
            item=systems[r['case_id']]
            for k in ('public_sha256','private_sha256','source_sha256','model_sha256'):
                assert item['source'][k]==r[k], (r['case_id'],k)
            for s in r['settings']:
                assert s['complete'] and all(v['status']==0 for v in s['stats']),r['case_id']
                assert sum(v['count'] for v in s['stats'])==sum(v['scored_residuals'] for v in s['vs_observations'].values())
                item['runs'].append(dict(folder=name,**s))
    results=[]
    for name,item in systems.items():
        eligible=[s for s in item['runs'] if max(v['mse'] for v in s['vs_reference'].values())<=1e-8]
        assert eligible,name
        chosen=max(eligible,key=lambda s:s['rtol'])
        results.append(dict(case_id=name,states=item['source']['states'],folder=chosen['folder'],rtol=chosen['rtol'],
            worst_split_reference_mse=max(v['mse'] for v in chosen['vs_reference'].values()),
            worst_split_observation_mse=max(v['mse'] for v in chosen['vs_observations'].values()),
            kernel_ms=chosen['kernel_ms'],max_lane_accepted_steps=max(v['accepted'] for v in chosen['stats'])))
    result=dict(purpose=__doc__,criterion='worst split MSE against independent FP64 reference <=1e-8',
                all_integrations_complete=True,systems=len(results),
                collections=dict(Counter(v['case_id'].split('-')[0] for v in results)),
                loosest_tested_passing_rtol_counts=dict(Counter(str(v['rtol']) for v in results)),results=results)
    (ROOT/'reconciled.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))


if __name__=='__main__': main()
