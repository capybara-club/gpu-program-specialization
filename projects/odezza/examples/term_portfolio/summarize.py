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
"""Audit the frozen local reports. Does not submit work or inspect prior answers."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    directory=HERE/'runs';screens=[];fits=[]
    for d in sorted(directory.glob('*-result')):
        name=d.name[:-7];raw=(d/'report.json').read_bytes();r=json.loads(raw)
        q=json.loads((directory/(name+'.json')).read_text())
        plan=json.loads((directory/(name+'.plan.json')).read_text())
        assert r['status']=='complete' and r['retention_complete']
        expected={f['id']:f for f in plan['families']}
        for f in r['families']:
            if 'expected_configurations' in expected[f['id']]:
                assert f['completed_configurations']==expected[f['id']]['expected_configurations']
            assert f['generation_stop']==1 and f['pruned']==0 and f['configuration_limited_derivations']==0
        devices=r['execution']['devices']
        screens.append(dict(name=name,counts=r['counts'],native_seconds=r['timing']['total_seconds'],
            client_seconds=json.loads((d/'feedback.json').read_text())['client_seconds'],
            generated_variants=sum(f['generated_asts'] for f in r['families']),
            underfilled_fraction=sum(v['sizing']['underfilled_configurations'] for v in devices)/r['counts']['completed_configurations'],
            public_motif=plan.get('public_motif'),coefficient_guidance=plan.get('coefficient_guidance'),
            report_sha256=hashlib.sha256(raw).hexdigest(),
            families=[dict(id=f['id'],counts=f,best_mse=min((r['candidates'][i]['mse'] for i in r['leaderboards']['families'][f['id']]),default=None)) for f in r['families']]))
    for p in sorted(directory.glob('*-fit.json')):
        r=json.loads(p.read_text());valid=[x for x in r['results'] if x.get('best')]
        fits.append(dict(name=p.stem,wall_seconds=r['elapsed_seconds'],count=len(r['results']),
            errors=[dict(candidate=x['candidate_id'],error=x['error']) for x in r['results'] if 'error' in x],
            best=min(valid,key=lambda x:x['best']['mse']),
            families={f:min(x['best']['mse'] for x in valid if x['family']==f) for f in {x['family'] for x in valid}}))
    result=dict(version=1,screens=screens,fits=fits,
        validation=json.loads((directory/'system20-validation.json').read_text()),
        total_configurations=sum(s['counts']['completed_configurations'] for s in screens),
        caveats=['Retrospective, not fresh blind recovery or a matched baseline comparison.',
            'Screens use four complete training trajectories; fitting uses 16.',
            'Continuous coefficients were sampled. Geometry does not measure SM occupancy.',
            'Fits have 12-second caps; timeouts retain best evaluated points but are not convergence proofs.',
            'Early shortlist policy allowed associative duplicates; final system19 follow-up canonicalized +/*.',
            'Public motifs were used only in explicitly named hint runs.',
            'Previous recovered equations were read only after final search/fitting froze, for a coverage audit.',
            'System19 is not recovered by the tested language: mixed product-plus-state arguments are absent.'])
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(screens=len(screens),fit_attempts=sum(f['count'] for f in fits),configurations=result['total_configurations'],native_seconds=sum(s['native_seconds'] for s in screens),fit_wall_seconds=sum(f['wall_seconds'] for f in fits))))


if __name__=='__main__':main()
