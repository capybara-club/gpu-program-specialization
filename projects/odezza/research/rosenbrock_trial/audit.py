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
"""PRIVATE known-equation integration audit, never symbolic recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from prototype import Solver

BENCH = Path(__file__).resolve().parents[1]/'trajectory_benchmarks'
sys.path.insert(0,str(BENCH))
from common import digest
from fp64_study import rhs_function, adaptive, difference


def write(path, data):
    # Failed numerical values must be explicit nulls, not nonstandard JSON NaN.
    def clean(x):
        if isinstance(x,float) and not np.isfinite(x): return None
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        return x
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')


def compare(pred, target, rows, groups):
    masked = [[[v if obs is not None else None for v,obs in zip(values,original)]
               for values,original in zip(tr,row['states'])] for tr,row in zip(target,rows)]
    return {s:difference([pred[i] for i in ids],[masked[i] for i in ids]) for s,ids in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--all-clean',action='store_true')
    p.add_argument('--rtols',default='1e-6,1e-8,1e-10'); p.add_argument('--device',type=int,default=0)
    p.add_argument('--case',action='append'); p.add_argument('--cpu-parity',action='store_true')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    rtols=[float(v) for v in a.rtols.split(',')]; started=time.perf_counter(); results=[]; seen=set()
    for collection in ('mdbench','odebench-v2','biological-v2'):
        folder=BENCH/'prepared'/collection
        manifest=json.loads((folder/'qualified-manifest-v2.json').read_text())
        for entry in manifest['cases']:
            system=entry['case_id'].rsplit('-rhs-',1)[0]
            if system in seen or entry['noise']!='none' or not entry.get('public'): continue
            if a.case and not any(v in entry['case_id'] for v in a.case): continue
            if not a.case and not a.all_clean and entry['status']!='needs_numerical_review': continue
            seen.add(system)
            case=json.loads((folder/entry['public']).read_text())
            private=json.loads((folder/'private'/(entry['case_id']+'.json')).read_text())
            if digest(case)!=entry['public_sha256']: raise ValueError('public fingerprint changed')
            record=dict(case_id=entry['case_id'],public_sha256=digest(case),private_sha256=digest(private),
                        source_sha256=entry['source_sha256'],settings=[])
            rows=[]; groups={}
            for split,rr in case['datasets'].items():
                groups[split]=list(range(len(rows),len(rows)+len(rr))); rows.extend(rr)
            try:
                solver=Solver(private['canonical_rhs'],case['problem']['states'])
                record.update(states=solver.n,trajectories=len(rows),model_sha256=solver.key,
                              compiled=solver.compiled,compile_seconds=solver.compile_seconds,load_seconds=solver.load_seconds)
                rhs=rhs_function(private['canonical_rhs'],case['problem']['states'])
                # Independent reference method, not the CPU build of this solver.
                cache=BENCH/'validation/fp64-flagged'/(entry['case_id']+'-predictions.json')
                reference_started=time.perf_counter()
                if cache.exists():
                    old=json.loads((cache.parent/(entry['case_id']+'.json')).read_text())
                    if any(old[k]!=record[k] for k in ('public_sha256','private_sha256','source_sha256')):
                        raise ValueError('reference fingerprint changed')
                    data=json.loads(cache.read_text())
                    if data['datasets']!=case['datasets']: raise ValueError('reference data changed')
                    reference=data['predictions']['fp64/DOP853/1e-12']; reference_kind='retained DOP853'
                else:
                    reference=[adaptive(rhs,r,1e-12,time.monotonic()+30,method='Radau')[0] for r in rows]
                    reference_kind='Radau'
                record.update(reference_method=reference_kind,reference_seconds=time.perf_counter()-reference_started,
                              reference_vs_observations=compare(reference,[r['states'] for r in rows],rows,groups))
                for rtol in rtols:
                    run=solver.run(rows,rtol=rtol,atol=rtol*.01,device=a.device)
                    pred=run.pop('predictions'); run['complete']=all(s['status']==0 for s in run['stats'])
                    if run['complete']:
                        run['vs_reference']=compare(pred,reference,rows,groups)
                        run['vs_observations']=compare(pred,[r['states'] for r in rows],rows,groups)
                        # Check the fused GPU score independently on the host.
                        host_scores=[difference([v],[r['states']]) for v,r in zip(pred,rows)]
                        run['score_agreement_max_abs']=max(abs(s['mse']-v['mse']) for s,v in zip(run['stats'],host_scores))
                    if a.cpu_parity:
                        cpu=solver.run(rows,rtol=rtol,atol=rtol*.01,cpu=True)
                        run['cpu_wall_seconds']=cpu['wall_seconds']
                        run['cpu_status']=[s['status'] for s in cpu['stats']]
                        if run['complete'] and all(s['status']==0 for s in cpu['stats']):
                            run['vs_same_algorithm_cpu']=compare(pred,cpu['predictions'],rows,groups)
                    write(a.out/(entry['case_id']+'-'+str(rtol)+'-predictions.json'),dict(predictions=pred))
                    record['settings'].append(run)
                record['status']='complete' if all(r['complete'] for r in record['settings']) else 'integration_failure'
            except Exception as error:
                record.update(status='error',error=repr(error))
            results.append(record); write(a.out/(entry['case_id']+'.json'),record)
            print(json.dumps(dict(case_id=entry['case_id'],status=record['status'],
                settings=[dict(rtol=r['rtol'],kernel_ms=r['kernel_ms'],complete=r['complete'],
                    mse=max(v['mse'] for v in r.get('vs_reference',{}).values()) if r.get('vs_reference') else None)
                    for r in record['settings']]),allow_nan=False),flush=True)
            write(a.out/'summary.json',dict(purpose=__doc__,seconds=time.perf_counter()-started,results=results,
                solver_sources={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest()
                                for f in ('solver.cuh','runtime.cu','codegen.py','prototype.py','audit.py')}))
    if not results: raise ValueError('no eligible systems')


if __name__=='__main__': main()
