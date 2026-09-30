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
"""Post-freeze scoring stratified by private support size; never guides fitting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from report import load, average_precision


def measure(items, threshold, support_size=None):
    a=np.concatenate([r['truth'] for r in items]);s=np.concatenate([r['scores'] for r in items])
    keep=np.ones(len(a),dtype=bool) if support_size is None else a.sum(1)==support_size
    a,s=a[keep],s[keep]; selected=s>=threshold
    whole_system=[]
    for r in items:
        subset=np.ones(len(r['truth']),dtype=bool) if support_size is None else r['truth'].sum(1)==support_size
        whole_system.append(bool(np.all((r['scores'][subset]>=threshold)|~r['truth'][subset])))
    return {'rhs_count':len(a),'ap':float(np.mean([average_precision(t,v) for t,v in zip(a,s)])),
            'systems_all_inputs_retained':float(np.mean(whole_system)),
            'all_inputs_retained':float(np.all(selected|~a,axis=1).mean()),
            'states_retained':float(selected.sum(1).mean()),
            'exact_support':float(np.all(selected==a,axis=1).mean()),
            'edge_recall':float((selected&a).sum()/a.sum()),
            'edge_precision':float((selected&a).sum()/max(1,selected.sum()))}


def bootstrap_ap(items, k, rng):
    values=np.array([measure([r],0,k)['ap'] for r in items])
    samples=values[rng.integers(len(items),size=(1000,len(items)))].mean(1)
    return list(map(float,np.quantile(samples,[.025,.975])))


def calibrate(items, target):
    """Largest threshold meeting empirical all-input coverage on development only."""
    if not 0 < target <= 1: raise ValueError('Coverage target must be in (0,1]')
    a=np.concatenate([r['truth'] for r in items]);s=np.concatenate([r['scores'] for r in items])
    if not np.all(a.any(1)):raise ValueError('Scaling cohort must have nonempty supports')
    minima=np.min(np.where(a,s,np.inf),axis=1)
    threshold=float(np.sort(minima)[len(minima)-math.ceil(target*len(minima))])
    if measure(items,threshold)['all_inputs_retained']+1e-12<target:raise AssertionError('Calibration failed')
    return threshold


def write(out,name,rows):
    (out/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    if rows:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(root):
    root=Path(root);cfg=json.loads((root/'private/config.json').read_text())
    out=root/'report';out.mkdir(exist_ok=True)
    ranking=[];selection=[];topk=[];calibration=[];total_seconds=0.;total_cells=0
    rng=np.random.default_rng(8802)
    allrecords={}
    for n in cfg['state_counts']:
        data=root/f'n{n}';records=load(data,data/'results')
        expected=(cfg['systems']+cfg['development_systems'])*len(cfg['counts'])*3
        if len(records)!=expected:raise ValueError(f'n={n}: incomplete {len(records)}/{expected}')
        allrecords[n]=records
        raw=[json.loads(p.read_text()) for p in (data/'results').glob('*-m*-t*.json')]
        if len({r['fingerprint'] for r in raw})!=1:raise ValueError('Mixed model versions')
        total_seconds+=sum(r['seconds'] for r in raw);total_cells+=len(raw)
        for method in ['fixed','diagonal','full']:
            for count in cfg['counts']:
                group=[r for r in records if r['method']==method and r['count']==count]
                dev=[r for r in group if r['split']=='development'];test=[r for r in group if r['split']=='test']
                # Same threshold for every RHS: the learner is not given true k.
                for target in [.95,.99]:
                    threshold=calibrate(dev,target)
                    calibration.append(dict(states=n,method=method,count=count,target=target,threshold=threshold,
                                            **measure(dev,threshold)))
                    for k in cfg['support_sizes']+[None]:
                        selection.append(dict(states=n,method=method,count=count,support=k or 'all',target=target,
                                              threshold=threshold,**measure(test,threshold,k)))
                for k in cfg['support_sizes']+[None]:
                    scores=measure(test,0,k);ci=bootstrap_ap(test,k,rng)
                    harmonic=sum(1./i for i in range(1,n+1))
                    def chance(k):return harmonic/n+(k-1)*(n-harmonic)/(n*(n-1))
                    baseline=chance(k) if k else float(np.mean([chance(q) for q in cfg['support_sizes']]))
                    ranking.append(dict(states=n,method=method,count=count,support=k or 'all',ap=scores['ap'],
                        ap_ci_low=ci[0],ap_ci_high=ci[1],random_ap=baseline,
                        ap_above_random=(scores['ap']-baseline)/(1-baseline),
                        seconds_per_system=float(np.mean([r['seconds'] for r in test]))))
                    a=np.concatenate([r['truth'] for r in test]);s=np.concatenate([r['scores'] for r in test])
                    if k is not None:
                        idx=a.sum(1)==k;a,s=a[idx],s[idx]
                    order=np.argsort(-s,axis=1,kind='stable')
                    required=np.max(np.where(np.take_along_axis(a,order,axis=1),np.arange(1,n+1)[None,:],0),axis=1)
                    for budget in range(1,n+1):
                        topk.append(dict(states=n,method=method,count=count,support=k or 'all',retained=budget,
                                         all_inputs_retained=float(np.mean(required<=budget))))
    write(out,'ranking',ranking);write(out,'selection',selection);write(out,'calibration',calibration);write(out,'topk',topk)
    timing={'completed_cells':total_cells,'model_results':sum((cfg['systems']+cfg['development_systems'])*len(cfg['counts'])*3*n for n in cfg['state_counts']),
            'grader_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'calibration_rule':'exact-development-minimum-score-order-statistic-v2',
            'summed_cell_seconds':total_seconds,'generation':json.loads((root/'generation.json').read_text())}
    write(out,'accounting',[timing])
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,n in zip(axes,cfg['state_counts']):
        for method in ['fixed','diagonal','full']:
            r=[r for r in ranking if r['states']==n and r['method']==method and r['support']=='all']
            ax.plot([r['count'] for r in r],[r['ap'] for r in r],marker='o',label=method)
        ax.set(title=f'{n} states',xlabel='Independent initial conditions',ylabel='Mean average precision',ylim=(0,1.02),xscale='log')
        ax.set_xticks(cfg['counts'],labels=cfg['counts']);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'state_scaling.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,k in zip(axes,cfg['support_sizes']):
        for n in cfg['state_counts']:
            r=[r for r in ranking if r['states']==n and r['method']=='diagonal' and r['support']==k]
            ax.plot([r['count'] for r in r],[r['ap'] for r in r],marker='o',label=f'{n} states')
        ax.set(title=f'{k} true input'+('' if k==1 else 's')+' per RHS',xlabel='Independent initial conditions',ylabel='Diagonal RFM average precision',ylim=(0,1.02),xscale='log')
        ax.set_xticks(cfg['counts'],labels=cfg['counts']);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'support_scaling.png',dpi=180);plt.close(fig)
    print(json.dumps(timing))
    for r in ranking:
        if r['support']=='all':print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
