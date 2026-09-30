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
"""Paired CPU FP32/FP64 benchmark-sample MSE and rollout-precision study."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from solver import Solver,HERE,SHARED,write
from audit import BENCH,score,digest


def target_rows(rows,pred):
    return [dict(row,states=[[v if observed is not None else None for v,observed in zip(sample,obs)]
                for sample,obs in zip(values,row['states'])]) for row,values in zip(rows,pred)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',action='append');p.add_argument('--ros-rtols',default='1e-4,1e-5,1e-6,1e-7')
    p.add_argument('--rk-substeps',default='1,4,16,32,128,512')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((HERE/'validation/all-clean-v1/summary.json').read_text())
    inputs={}
    for collection in ('mdbench','odebench-v2','biological-v2'):
        folder=BENCH/'prepared'/collection
        for entry in json.loads((folder/'qualified-manifest-v2.json').read_text())['cases']:
            if entry.get('public'):inputs[entry['case_id']]=(folder,entry)
    configs=[dict(method='Rosenbrock23',rtol=t,atol=t*.01,max_attempts=500000)
             for t in [float(v) for v in a.ros_rtols.split(',') if v]]
    configs += [dict(method='RK4',substeps=int(n)) for n in a.rk_substeps.split(',') if n]
    started=time.perf_counter();results=[]
    metadata=dict(purpose=__doc__,host=platform.node(),platform=platform.platform(),
        precision_scope='FP32 states, literals, RHS, Jacobian, stages, LU; FP64 observation times, adaptive controller and MSE',
        primary_metric='MSE vs actual samples; every split retains original masks/ICs/times',
        diagnostic_reference='FP64 native RK4 with 512 substeps, not exact truth',
        sample_gate=1e-6,score_reproduction_atol=1e-8,score_reproduction_rtol=1e-4,
        settings=configs,prior_sha256=digest(prior),
        source_sha256={str(f.relative_to(HERE.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (HERE/'solver.py',HERE/'runtime.cpp',HERE/'solver32.hpp',SHARED/'codegen.py',SHARED/'solver.cuh',HERE/'precision_study.py')})
    for old in prior['results']:
        name=old['case_id']
        if a.case and not any(v in name for v in a.case):continue
        folder,entry=inputs[name];case=json.loads((folder/entry['public']).read_text())
        private=json.loads((folder/'private'/(name+'.json')).read_text())
        assert digest(case)==old['public_sha256'] and digest(private)==old['private_sha256']
        assert entry['source_sha256']==old['source_sha256']
        rows=[];groups={}
        for split,rr in case['datasets'].items():groups[split]=list(range(len(rows),len(rows)+len(rr)));rows.extend(rr)
        r=dict(case_id=name,states=len(case['problem']['states']),rows=len(rows),
            public_sha256=digest(case),private_sha256=digest(private),source_sha256=entry['source_sha256'],
            accurate_reference_mse=old['accurate_reference_mse'],settings=[])
        try:
            solvers={pr:Solver(private['canonical_rhs'],case['problem']['states'],precision=pr) for pr in ('FP64','FP32')}
            r['builds']={pr:dict(sha256=s.key,compiled=s.compiled,compile_seconds=s.compile_seconds) for pr,s in solvers.items()}
            reference=solvers['FP64'].run(rows,method='RK4',substeps=512)
            assert reference['complete'],'FP64 reference integration failed'
            refpred=reference.pop('predictions');refrows=target_rows(rows,refpred)
            refmse=score(refpred,rows,groups)
            oldrk=next(s for s in old['settings'] if s['method']=='RK4' and s['substeps']==512)
            assert all(math.isclose(v['mse'],oldrk['vs_observations'][k]['mse'],rel_tol=1e-12,abs_tol=1e-18)
                       for k,v in refmse.items()),'FP64 baseline changed'
            r['fp64_reference']=dict(**reference,vs_observations=refmse,predictions_sha256=digest(refpred))
            for config in configs:
                pair={};paired_pred={}
                for pr in ('FP64','FP32'):
                    run=solvers[pr].run(rows,**config);pred=run.pop('predictions')
                    if run['complete']:
                        measured=score(pred,rows,groups)
                        assert sum(v['scored_residuals'] for v in measured.values())==sum(v['count'] for v in run['stats'])
                        individual=score(pred,rows,{str(i):[i] for i in range(len(rows))})
                        assert all(math.isclose(v['mse'],run['stats'][int(i)]['mse'],rel_tol=1e-12,abs_tol=1e-15) for i,v in individual.items())
                        run.update(vs_observations=measured,vs_fp64_rk4_512=score(pred,refrows,groups),
                            predictions_sha256=digest(pred),data_gate_pass=all(v['mse']<=1e-6 for v in measured.values()),
                            matches_accurate_sample_mse=all(abs(v['mse']-old['accurate_reference_mse'][k]['mse'])<=
                                1e-8+1e-4*abs(old['accurate_reference_mse'][k]['mse']) for k,v in measured.items()))
                        paired_pred[pr]=pred
                    pair[pr]=run
                if len(paired_pred)==2:
                    pair['FP32']['vs_matched_fp64']=score(paired_pred['FP32'],target_rows(rows,paired_pred['FP64']),groups)
                r['settings'].append(dict(config=config,runs=pair))
            r['status']='complete'
        except Exception as error:r.update(status='error',error=repr(error))
        results.append(r);write(a.out/(name+'.json'),r)
        write(a.out/'summary.json',dict(**metadata,seconds=time.perf_counter()-started,results=results))
        print(json.dumps(dict(case_id=name,status=r['status'],errors=r.get('error'),
            fp32_completed=sum(s['runs']['FP32']['complete'] for s in r['settings']),settings=len(r['settings']))),flush=True)
    if not results:raise ValueError('no selected cases')


if __name__=='__main__':main()
