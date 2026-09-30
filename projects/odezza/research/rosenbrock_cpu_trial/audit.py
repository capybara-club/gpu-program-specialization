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
"""Recompute CPU rollout MSE against the actual benchmark samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from solver import Solver,write,HERE,SHARED

BENCH=HERE.parent/'trajectory_benchmarks'
sys.path.insert(0,str(BENCH))
from common import digest


def score(predictions,rows,groups):
    result={}
    for group,ids in groups.items():
        residuals=[];by_state=[[] for _ in rows[0]['initial_state']]
        for i in ids:
            if len(predictions[i])!=len(rows[i]['states']):raise ValueError('prediction length mismatch')
            for expected,pred in zip(rows[i]['states'][1:],predictions[i][1:]):
                for j,(a,b) in enumerate(zip(expected,pred)):
                    if a is not None:
                        if not math.isfinite(b):raise ValueError('nonfinite prediction')
                        residuals.append((a-b)**2);by_state[j].append((a-b)**2)
        if not residuals:raise ValueError('empty scoring split')
        result[group]=dict(mse=math.fsum(residuals)/len(residuals),scored_residuals=len(residuals),
            state_mse=[math.fsum(v)/len(v) if v else None for v in by_state])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--flagged-only',action='store_true');p.add_argument('--case',action='append')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((SHARED/'validation/all-clean-v1/summary.json').read_text())
    flagged={r['case_id'] for r in json.loads((SHARED/'validation/flagged-v1/summary.json').read_text())['results']}
    sources={}
    for collection in ('mdbench','odebench-v2','biological-v2'):
        folder=BENCH/'prepared'/collection
        for e in json.loads((folder/'qualified-manifest-v2.json').read_text())['cases']:
            if e.get('public'):sources[e['case_id']]=(folder,e)
    started=time.perf_counter();results=[]
    metadata=dict(purpose=__doc__,host=platform.node(),platform=platform.platform(),python=sys.version,
        execution='CPU only; no CUDA; serial native C++ trajectory loops',precision='FP64',
        data_gate_mse=1e-6,mse_reproduction_atol=1e-8,mse_reproduction_rtol=1e-4,
        comparison='all split MSEs vs retained accurate FP64 DOP853/Radau MSE against the same samples',
        baseline_sha256=digest(baseline),
        source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [HERE/'runtime.cpp',HERE/'solver.py',HERE/'audit.py',SHARED/'solver.cuh',SHARED/'codegen.py']})
    for old in baseline['results']:
        name=old['case_id']
        if a.flagged_only and name not in flagged:continue
        if a.case and not any(k in name for k in a.case):continue
        folder,entry=sources[name]
        case=json.loads((folder/entry['public']).read_text())
        private=json.loads((folder/'private'/(name+'.json')).read_text())
        for key,value in [('public_sha256',digest(case)),('private_sha256',digest(private)),('source_sha256',entry['source_sha256'])]:
            if value!=old[key]:raise ValueError('fingerprint mismatch '+name+' '+key)
        rows=[];groups={}
        for split,rr in case['datasets'].items():groups[split]=list(range(len(rows),len(rows)+len(rr)));rows.extend(rr)
        r=dict(case_id=name,public_sha256=digest(case),private_sha256=digest(private),source_sha256=entry['source_sha256'],
               states=len(case['problem']['states']),rows=len(rows),reference_method=old['reference_method'],
               accurate_reference_mse=old['reference_vs_observations'],settings=[])
        try:
            solver=Solver(private['canonical_rhs'],case['problem']['states'])
            r.update(compiled=solver.compiled,compile_seconds=solver.compile_seconds,load_seconds=solver.load_seconds,
                     model_sha256=solver.key)
            settings=[dict(method='Rosenbrock23',rtol=v,atol=v*.01) for v in (1e-8,1e-10,1e-12)]
            settings += [dict(method='RK4',substeps=n) for n in (32,128,512)]
            for setting in settings:
                run=solver.run(rows,**setting);pred=run.pop('predictions')
                if run['complete']:
                    measured=score(pred,rows,groups);run['vs_observations']=measured
                    run['matches_reference_mse']=all(abs(v['mse']-r['accurate_reference_mse'][k]['mse'])<=
                        1e-8+1e-4*abs(r['accurate_reference_mse'][k]['mse']) for k,v in measured.items())
                    run['data_gate_pass']=all(v['mse']<=1e-6 for v in measured.values())
                    run['mse_delta_from_reference']={k:v['mse']-r['accurate_reference_mse'][k]['mse'] for k,v in measured.items()}
                    if sum(v['scored_residuals'] for v in measured.values())!=sum(v['count'] for v in run['stats']):
                        raise ValueError('native residual count mismatch')
                    independent_rows=score(pred,rows,{str(i):[i] for i in range(len(rows))})
                    if any(not math.isclose(v['mse'],run['stats'][int(i)]['mse'],rel_tol=1e-12,abs_tol=1e-15)
                           for i,v in independent_rows.items()):raise ValueError('native score mismatch')
                    run['predictions_sha256']=digest(pred)
                r['settings'].append(run)
            # Tighten only numerical MSE reproduction failures, never modify data.
            last_ros=r['settings'][2]
            if not last_ros.get('matches_reference_mse',False):
                run=solver.run(rows,method='Rosenbrock23',rtol=1e-13,atol=1e-15,max_attempts=5000000)
                pred=run.pop('predictions')
                if run['complete']:
                    measured=score(pred,rows,groups);run['vs_observations']=measured
                    run['matches_reference_mse']=all(abs(v['mse']-r['accurate_reference_mse'][k]['mse'])<=
                        1e-8+1e-4*abs(r['accurate_reference_mse'][k]['mse']) for k,v in measured.items())
                    run['data_gate_pass']=all(v['mse']<=1e-6 for v in measured.values())
                    run['mse_delta_from_reference']={k:v['mse']-r['accurate_reference_mse'][k]['mse'] for k,v in measured.items()}
                    run['predictions_sha256']=digest(pred)
                r['settings'].append(run)
            r['status']='complete' if all(s['complete'] for s in r['settings']) else 'some_integrations_failed'
        except Exception as error:r.update(status='error',error=repr(error))
        results.append(r);write(a.out/(name+'.json'),r)
        write(a.out/'summary.json',dict(**metadata,seconds=time.perf_counter()-started,results=results))
        print(json.dumps(dict(case_id=name,status=r['status'],
            methods={m:any(s['method']==m and s.get('matches_reference_mse') for s in r['settings'])
                     for m in ('Rosenbrock23','RK4')})),flush=True)
    if not results:raise ValueError('no selected systems')


if __name__=='__main__':main()
