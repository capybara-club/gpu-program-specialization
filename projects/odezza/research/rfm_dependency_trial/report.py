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
"""Post-run grader: private truth enters here, never in learn.py."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def average_precision(truth, scores):
    if not truth.any(): return None
    order = np.argsort(-scores, kind='stable')
    y = truth[order]; s = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum(); precision = tp / (ends+1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def metrics(items, threshold):
    a = np.concatenate([x['truth'] for x in items])
    s = np.concatenate([x['scores'] for x in items])
    selected = s >= threshold
    counts = a.sum(1)
    complete = np.all(selected | ~a, axis=1)
    nonzero = counts > 0
    aps = [average_precision(t, v) for t, v in zip(a, s)]
    return {'macro_ap_nonzero': float(np.mean([x for x in aps if x is not None])),
            'all_inputs_retained': float(complete.mean()),
            'all_inputs_retained_nonzero': float(complete[nonzero].mean()),
            'states_retained': float(selected.sum(1).mean()),
            'states_retained_nonzero': float(selected[nonzero].sum(1).mean()),
            'edge_recall': float((selected & a).sum() / a.sum()),
            'edge_precision': float((selected & a).sum() / max(selected.sum(), 1)),
            'exact_support': float(np.all(selected == a, axis=1).mean()),
            'exact_support_nonzero': float(np.all(selected == a, axis=1)[nonzero].mean()),
            'zero_support_rhs': int((counts == 0).sum()), 'rhs_rows': len(a)}


def load(data, results):
    records = []
    for path in sorted(Path(results).glob('*-m*-t*.json')):
        r = json.loads(path.read_text())
        private = json.loads((Path(data) / 'private' / (r['id'] + '.json')).read_text())
        for mode, values in r['methods'].items():
            scores = np.diagonal(np.asarray(values['matrices']), axis1=1, axis2=2).copy()
            mx = scores.max(1)
            scores[mx < 1e-20] = 0
            scores /= np.where(mx < 1e-20, 1., mx)[:, None]
            records.append({'id': r['id'], 'split': r['split'], 'count': r['count'],
                            'duration': r['duration'], 'method': mode, 'scores': scores,
                            'truth': np.array(private['support'], dtype=bool),
                            'seconds': values.get('seconds_with_shared_initial', 0.),
                            'validation_mse': float(np.mean(values['validation_mse_scaled']))})
    return records


def report(data, results, out, equal_budget=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(data)/'manifest.json').read_text()); cfg = manifest['config']
    records = load(data, results)
    expected = len(manifest['cases'])*len(cfg['counts'])*(1 if equal_budget else len(cfg['durations']))*3
    if len(records) != expected: raise ValueError(f'Incomplete results: {len(records)}/{expected}')
    rows = []; calibration = []; curves = []
    rng = np.random.default_rng(8801)
    for method in ('fixed', 'diagonal', 'full'):
        for count in cfg['counts']:
            for duration in sorted(set(r['duration'] for r in records if r['count']==count)):
                group = [r for r in records if (r['method'],r['count'],r['duration']) == (method,count,duration)]
                dev = [r for r in group if r['split']=='development']
                test = [r for r in group if r['split']=='test']
                thresholds = np.r_[0., np.geomspace(1e-6, 1., 121)]
                threshold = max(float(t) for t in thresholds if metrics(dev,t)['all_inputs_retained_nonzero'] >= .95)
                calibration.append(dict(method=method,count=count,duration=duration,threshold=threshold,development=metrics(dev,threshold)))
                row = dict(method=method,count=count,duration=duration,threshold=threshold,**metrics(test,threshold))
                row['seconds_per_system'] = float(np.mean([r['seconds'] for r in test]))
                row['validation_mse_median'] = float(np.median([r['validation_mse'] for r in test]))
                per_system_ap = np.array([metrics([r],threshold)['macro_ap_nonzero'] for r in test])
                weights = np.array([r['truth'].any(1).sum() for r in test])
                ix = rng.integers(len(test),size=(1000,len(test)))
                boot = (per_system_ap[ix]*weights[ix]).sum(1)/weights[ix].sum(1)
                row['ap_ci_low'],row['ap_ci_high'] = map(float,np.quantile(boot,[.025,.975]))
                rows.append(row)
                for k in range(1,cfg['states']+1):
                    a = np.concatenate([r['truth'] for r in test]); s = np.concatenate([r['scores'] for r in test])
                    sel = np.zeros_like(a)
                    ix = np.argsort(-s,axis=1,kind='stable')[:,:k]
                    np.put_along_axis(sel,ix,True,axis=1)
                    active = a.any(1)
                    curves.append(dict(method=method,count=count,duration=duration,k=k,
                                       all_inputs_retained_nonzero=float(np.all(sel|~a,axis=1)[active].mean())))
    for name, values in [('summary',rows),('calibration',calibration),('topk',curves)]:
        (out/(name+'.json')).write_text(json.dumps(values,indent=2)+'\n')
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if equal_budget:
        print(json.dumps(rows))
        return
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,method in zip(axes,('fixed','diagonal','full')):
        grid=np.array([[next(r['macro_ap_nonzero'] for r in rows if (r['method'],r['count'],r['duration'])==(method,m,t)) for t in cfg['durations']] for m in cfg['counts']])
        im=ax.imshow(grid,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set(title=method,xlabel='Duration',ylabel='Trajectories',xticks=range(len(cfg['durations'])),xticklabels=cfg['durations'],yticks=range(len(cfg['counts'])),yticklabels=cfg['counts'])
        for i in range(len(cfg['counts'])):
            for j in range(len(cfg['durations'])): ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',color='white' if grid[i,j]<.6 else 'black')
    fig.colorbar(im,ax=axes,label='Mean average precision (nonzero RHSs)')
    fig.savefig(out/'dependency_accuracy.png',dpi=180); plt.close(fig)
    # First held-out system, fixed in advance by public ID order, not best case.
    first = min(r['id'] for r in records if r['split']=='test')
    example = [next(r for r in records if (r['id'],r['method'],r['count'],r['duration'])==(first,'full',m,t)) for m,t in [(1,4.),(16,.5)]]
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,mat,title in zip(axes,[example[0]['truth'],example[0]['scores'],example[1]['scores']],['True dependencies','1 trajectory, duration 4','16 trajectories, duration 0.5']):
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis')
        ax.set(title=title,xlabel='Input state',ylabel='RHS state',xticks=range(cfg['states']),yticks=range(cfg['states']))
    fig.colorbar(im,ax=axes,label='Truth or relative relevance (not probability)')
    fig.savefig(out/'example_matrices.png',dpi=180);plt.close(fig)
    private=[json.loads((Path(data)/'private'/(c['id']+'.json')).read_text()) for c in manifest['cases'] if c['split']=='test']
    support=np.concatenate([np.asarray(x['support']) for x in private])
    info={'completed_comparisons':len(records),'test_systems':len(private),'nonzero_rhs':int(support.any(1).sum()),
          'support_size_histogram':np.bincount(support.sum(1).astype(int),minlength=cfg['states']+1).tolist(),
          'generated':len(manifest['cases']),'rejected':len(manifest['rejections']),
          'generation_seconds':manifest['generation_seconds']}
    (out/'cohort.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))
    for r in rows:
        if (r['count'],r['duration']) in ((1,.5),(1,4.),(4,1.),(8,2.),(16,.5),(16,4.)): print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('results');p.add_argument('out')
    p.add_argument('--equal-budget',action='store_true')
    a=p.parse_args();report(a.data,a.results,a.out,a.equal_budget)
