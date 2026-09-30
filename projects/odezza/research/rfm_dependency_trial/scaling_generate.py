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
"""Generate matched abstract RHS templates across dimensions; private supports."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from generate import ast, symbolic, save


def leaves(tree):
    if tree[0] == 'x': return {tree[1]}
    if tree[0] == 'c': return set()
    return set().union(*(leaves(t) for t in tree[1:]))


def bind(tree, assignment):
    if tree[0] == 'x': return ['x', int(assignment[tree[1]])]
    if tree[0] == 'c': return tree.copy()
    return [tree[0]] + [bind(t,assignment) for t in tree[1:]]


def template(rng, cfg, k):
    local = dict(cfg, states=k)
    symbols = sp.symbols('x0:'+str(k))
    for attempt in range(100000):
        t = ast(rng,local,cfg['depth'])
        if len(leaves(t)) != k: continue
        f = sp.simplify(symbolic(t,symbols))
        if all(sp.simplify(sp.diff(f,s)) != 0 for s in symbols):
            return t, attempt+1
    raise RuntimeError('Support-conditioned template attempt budget exhausted')


def generate(config, out):
    cfg = json.loads(Path(config).read_text())
    if cfg['schema'] != 'odezza.rfm-scaling.v1': raise ValueError('Unknown schema')
    if cfg['depth'] != 2 or cfg['support_sizes'] != [1,2,4]: raise ValueError('This protocol requires depth two and support 1/2/4')
    if any(n%6 or n<4 for n in cfg['state_counts']): raise ValueError('State counts must be multiples of six')
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    save(out/'private/config.json',cfg)
    rng = np.random.default_rng(cfg['generation_seed'])
    manifests={}; symbols={}
    for n in cfg['state_counts']:
        public={k:v for k,v in cfg.items() if k not in ('generation_seed','support_sizes','state_counts')}
        public['states']=n
        manifests[n]={'config':public,'cases':[],'rejections':[],'complete':False}
        symbols[n]=sp.symbols('x0:'+str(n))
    times=np.arange(round(max(cfg['durations'])/cfg['spacing'])+1)*cfg['spacing']
    ntr=max(cfg['counts'])+cfg['validation_trajectories']
    start=time.perf_counter(); accepted=0; template_draws=0; rejections=[]
    for attempt in range(cfg['max_generation_attempts']):
        sizes=np.concatenate([rng.permutation([1,1,2,2,4,4]) for _ in range(max(cfg['state_counts'])//6)])
        templates=[]
        for k in sizes:
            t,draws=template(rng,cfg,int(k)); templates.append(t); template_draws+=draws
        master_ics=rng.uniform(*cfg['initial_range'],size=(ntr,max(cfg['state_counts'])))
        staged={}; failure=None
        for n in cfg['state_counts']:
            trees=[bind(t,rng.choice(n,int(k),replace=False)) for t,k in zip(templates[:n],sizes[:n])]
            expr=[cfg['rhs_scale']*symbolic(t,symbols[n]) for t in trees]
            funcs=[sp.lambdify(symbols[n],f,'numpy') for f in expr]
            def rhs(t,y):
                with np.errstate(over='ignore',invalid='ignore'):
                    return np.asarray([f(*y) for f in funcs],dtype=float)
            def bound(t,y): return cfg['state_bound']-np.max(np.abs(y))
            bound.terminal=True
            trajectories=[]
            for ic in master_ics[:,:n]:
                sol=solve_ivp(rhs,(0.,times[-1]),ic,t_eval=times,method='DOP853',rtol=1e-10,atol=1e-12,events=bound,max_step=.1)
                if not sol.success or sol.y.shape[1]!=len(times) or not np.isfinite(sol.y).all():
                    failure={'attempt':attempt,'states':n,'reason':'integration_failure_or_bound'};break
                trajectories.append(sol.y.T)
            if failure:break
            support=[[j in leaves(t) for j in range(n)] for t in trees]
            # Templates were symbolically checked; injective renaming preserves support.
            assert [sum(r) for r in support]==sizes[:n].tolist()
            staged[n]=(trees,expr,support,trajectories)
        if failure:
            rejections.append(failure);continue
        name=f'{accepted:03d}';split='development' if accepted<cfg['development_systems'] else 'test'
        for n,(trees,expr,support,trajectories) in staged.items():
            root=out/f'n{n}'
            data={'schema':'odezza.rfm-observations.v1','id':name,'split':split,'times':times.tolist(),
                  'observations':np.asarray(trajectories).tolist(),'training_trajectories':max(cfg['counts'])}
            save(root/f'public/{name}.json',data)
            save(root/f'private/{name}.json',{'asts':trees,'rhs':[str(f) for f in expr],'support':support,
                                            'support_sizes':sizes[:n].tolist(),'attempt':attempt})
            manifests[n]['cases'].append({'id':name,'split':split,'path':f'public/{name}.json',
                'sha256':hashlib.sha256((root/f'public/{name}.json').read_bytes()).hexdigest()})
        accepted+=1
        print(json.dumps({'replicate':name,'split':split,'attempt':attempt,'seconds':time.perf_counter()-start}),flush=True)
        if accepted==cfg['systems']+cfg['development_systems']:break
    for n,manifest in manifests.items():
        manifest.update(complete=accepted==cfg['systems']+cfg['development_systems'],rejections=rejections,
                        generation_seconds=time.perf_counter()-start)
        save(out/f'n{n}/manifest.json',manifest)
    save(out/'generation.json',{'accepted_replicates':accepted,'systems':accepted*len(manifests),
         'rejections':rejections,'template_draws':template_draws,'seconds':time.perf_counter()-start,
         'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if not all(m['complete'] for m in manifests.values()):raise RuntimeError('Generation budget exhausted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('out')
    a=p.parse_args();generate(a.config,a.out)
