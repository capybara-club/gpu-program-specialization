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
"""Audit saved live reports and write small, versionable measurement evidence.

Run after the documented calibration requests; this does not submit GPU work.
"""
import hashlib
import json
from pathlib import Path

from run import feedback

HERE=Path(__file__).resolve().parent


def main():
    rows=[]
    for directory in sorted((HERE/'runs').glob('*-result')):
        name=directory.name[:-7]
        # The first fixture predated the uniform filename convention.
        request_path=HERE/'runs'/(name+'.json')
        if not request_path.exists():
            raise ValueError(f'missing request for {directory}')
        request=json.loads(request_path.read_text())
        plan=json.loads(request_path.with_suffix('.plan.json').read_text())
        report_path=directory/'report.json'
        report=json.loads(report_path.read_text())
        summary=feedback(request,report,plan)
        assert report['status']=='complete' and report['retention_complete'],name
        sampled=request['grammar']['expansion']['strategy']=='sample'
        assert summary['all_families_exhausted'] != sampled,name
        bank_rows=request['grammar']['rng_banks']['draws']['count']
        planned={f['id']:f for f in plan['families']}
        for family in report['families']:
            if not sampled:
                assert family['completed_configurations']==planned[family['id']]['ordered_labelled_trees']*bank_rows,(name,family)
                assert family['pruned']==0 and family['configuration_limited_derivations']==0,(name,family)
        devices=report['execution']['devices']
        assert sum(d['completed_configurations'] for d in devices)==report['counts']['completed_configurations']
        saved=json.loads((directory/'feedback.json').read_text())
        summary['client_seconds']=saved['client_seconds']
        (directory/'feedback.json').write_text(json.dumps(summary,indent=2)+'\n')
        rows.append(dict(name=name,states=len(request['problem']['states']),
            trajectories=len(request['problem']['trajectories']),bank_rows=bank_rows,
            sampled_structures=sampled,exhaustive_families=sum(f['finite_language_exhausted'] for f in summary['families']),
            generated_asts=sum(f['generated_asts'] for f in report['families']),
            counts=report['counts'],timing=report['timing'],integration=report['integration'],
            client_seconds=summary['client_seconds'],
            underfilled_configurations=sum(d['sizing']['underfilled_configurations'] for d in devices),
            underfilled_fraction=sum(d['sizing']['underfilled_configurations'] for d in devices)/report['counts']['completed_configurations'],
            maximum_tile_systems=max(d['sizing']['maximum_tile_systems'] for d in devices),
            request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            families=[dict(id=f['coverage']['id'],coverage=f['coverage'],
                best_mse=f['winners'][0]['mse'] if f['winners'] else None) for f in summary['families']]))
    by_name={r['name']:r for r in rows}
    for a,b in [('depth2-3-512','depth2-3-2048'),('depth2-3-2048','depth2-3-8192'),
                ('depth2-6-128','depth2-6-512'),('depth2-6-512','depth2-6-2048'),
                ('system19-portfolio-512','system19-portfolio-2048'),('system19-subset-2048','system19-subset-8192')]:
        if a not in by_name or b not in by_name:raise ValueError('missing calibration pair')
        for x,y in zip(by_name[a]['families'],by_name[b]['families']):
            assert x['id']==y['id']
            assert y['best_mse']<=x['best_mse'],(a,b,x['id'])
    evidence=dict(version=1,date='2026-09-13',route='mac1 -> mac3 -> rack1, two RTX 5080',
        validation='Complete C enumeration counts match independent finite-language counts for every enumerated family; larger same-seed banks never worsen each family best MSE.',
        caveats=['Single measurements, not confidence intervals.',
            'Synthetic decay fixtures validate throughput and coverage, not blind recovery.',
            'System19 measurements reuse training observations; no recovery or validation claim.',
            'Subset objectives differ from full-data objectives; do not compare their MSE directly.',
            'Underfilled geometry is not measured SM occupancy; stage sums overlap across modules/devices.',
            'Nonzero NVRTC rows are cold and not a matched warm baseline.'],runs=rows)
    (HERE/'validation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(passed=True,runs=len(rows),configurations=sum(r['counts']['completed_configurations'] for r in rows))))


if __name__=='__main__':main()
