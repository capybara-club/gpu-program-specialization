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
"""Independent installed SciPy stiff solvers on CPU; never initializes CUDA."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import platform
import time
from pathlib import Path
import sys
import scipy
from scipy.integrate import solve_ivp
from solver import HERE,SHARED,write
from audit import BENCH,score
from common import digest
from fp64_study import rhs_function


def check(job):
    folder,entry,old,methods,rtols=job;folder=Path(folder)
    case=json.loads((folder/entry['public']).read_text())
    private=json.loads((folder/'private'/(entry['case_id']+'.json')).read_text())
    if digest(case)!=old['public_sha256'] or digest(private)!=old['private_sha256'] or entry['source_sha256']!=old['source_sha256']:
        raise ValueError('fingerprint mismatch')
    rhs=rhs_function(private['canonical_rhs'],case['problem']['states'])
    rows=[];groups={}
    for split,rr in case['datasets'].items():groups[split]=list(range(len(rows),len(rows)+len(rr)));rows.extend(rr)
    r=dict(case_id=entry['case_id'],public_sha256=digest(case),private_sha256=digest(private),
        source_sha256=entry['source_sha256'],accurate_reference_mse=old['reference_vs_observations'],settings=[])
    for method in methods:
        for tol in rtols:
            start=time.perf_counter();s=dict(method=method,rtol=tol,atol=tol*.01,precision='FP64')
            try:
                pred=[];work=[]
                for row in rows:
                    deadline=time.monotonic()+30
                    def f(t,y):
                        if time.monotonic()>deadline:raise TimeoutError('30s per trajectory CPU reference budget')
                        return rhs(y)
                    sol=solve_ivp(f,(row['times'][0],row['times'][-1]),row['initial_state'],
                        method=method,rtol=tol,atol=tol*.01,t_eval=row['times'])
                    if not sol.success or len(sol.t)!=len(row['times']):raise ValueError(sol.message)
                    pred.append(sol.y.T.tolist());work.append(dict(nfev=int(sol.nfev),njev=int(sol.njev),nlu=int(sol.nlu)))
                measured=score(pred,rows,groups)
                s.update(complete=True,vs_observations=measured,work=work,predictions_sha256=digest(pred),
                    matches_reference_mse=all(abs(v['mse']-r['accurate_reference_mse'][k]['mse'])<=
                        1e-8+1e-4*abs(r['accurate_reference_mse'][k]['mse']) for k,v in measured.items()),
                    data_gate_pass=all(v['mse']<=1e-6 for v in measured.values()))
            except Exception as error:s.update(complete=False,error=repr(error))
            s['wall_seconds']=time.perf_counter()-start;r['settings'].append(s)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--case')
    p.add_argument('--methods',default='Radau,BDF,LSODA');p.add_argument('--rtols',default='1e-8,1e-12')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    methods=a.methods.split(',');rtols=[float(v) for v in a.rtols.split(',')]
    if any(m not in ('Radau','BDF','LSODA') for m in methods):raise ValueError('unsupported method')
    baseline=json.loads((SHARED/'validation/all-clean-v1/summary.json').read_text())
    flagged={r['case_id'] for r in json.loads((SHARED/'validation/flagged-v1/summary.json').read_text())['results']}
    sources={}
    for collection in ('mdbench','odebench-v2','biological-v2'):
        folder=BENCH/'prepared'/collection
        for e in json.loads((folder/'qualified-manifest-v2.json').read_text())['cases']:
            if e.get('public'):sources[e['case_id']]=(str(folder),e)
    jobs=[(*sources[r['case_id']],r,methods,rtols) for r in baseline['results']
          if r['case_id'] in flagged and (not a.case or a.case in r['case_id'])]
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(check,j) for j in jobs]):
            r=f.result();results.append(r);write(a.out/(r['case_id']+'.json'),r)
            write(a.out/'summary.json',dict(purpose=__doc__,execution='CPU only',host=platform.node(),
                scipy_version=scipy.__version__,python=sys.version,workers=a.workers,
                seconds=time.perf_counter()-started,results=results))
            print(json.dumps(dict(case_id=r['case_id'],complete=all(s['complete'] for s in r['settings']),
                methods={m:any(s['method']==m and s.get('matches_reference_mse') for s in r['settings'])
                    for m in ('Radau','BDF','LSODA')})),flush=True)


if __name__=='__main__':main()
