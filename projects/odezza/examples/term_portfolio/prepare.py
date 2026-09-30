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
"""Prepare a versioned additive portfolio. Native C expands and scores its ASTs.

This experiment supports one unknown RHS in a three-state problem. Its public
input adapter reads known equations and observations only. All fitting is a
separate, explicitly bounded experiment outside Odezza's core.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def load_problem(known_path, trajectory_path):
    states=['x','y','z']
    known={}
    for line in Path(known_path).read_text().splitlines():
        match=re.fullmatch(r'd([xyz])/dt\s*=\s*(.+)',line)
        if match and match[2]!='???':known[match[1]]=match[2]
    groups={}
    with Path(trajectory_path).open() as f:
        for row in csv.DictReader(f):groups.setdefault(int(row['trajectory_id']),[]).append(row)
    trajectories=[]
    for ident,rows in sorted(groups.items()):
        times=[float(r['t']) for r in rows]
        values=[[float(r[s]) for s in states] for r in rows]
        trajectories.append(dict(initial=values[0],times=times,values=values))
    return dict(states=states,known_rhs=known,trajectories=trajectories)


def build_legacy(problem, *, rows=16384, seed=2026091401, indices=(0,4,8,12), dt=.125, hint=None):
    if problem['states']!=['x','y','z'] or set(problem['known_rhs'])!={'x','y'}:
        raise ValueError('this calibration requires x/y known and z unknown')
    if type(rows) is not int or rows<1:raise ValueError('positive integer rows required')
    if len(set(indices))!=len(indices) or not indices or any(i<0 or i>=len(problem['trajectories']) for i in indices):
        raise ValueError('invalid trajectory selection')
    grammar=dict(version=1,states=problem['states'],integration=dict(method='rk4',dt=dt),
        rules={},leaves={},rng={},rng_banks={'joint':dict(base='uniform01',count=rows,seed=seed,scope='run')},
        expansion=dict(strategy='enumerate',max_nodes=128,max_depth=64,max_expansion_depth=64),
        retain={'global':dict(k=32,unit='resolved_structure'),'per_family':dict(k=12,unit='resolved_structure')},families=[])
    def parameter(name,lo,hi):
        grammar['rng'][name]=dict(bank='joint',axis='trial',stream=name,
            transform=dict(kind='uniform',low=lo,high=hi))
        return 'rng.'+name
    def state(name):
        grammar['leaves'][name]=dict(states=['x','y'],arity=2,coverage='explicit',groups=[['x','y']])
        grammar['rules']['State'+name]=['leaf.'+name,'z']
        return 'State'+name
    def argument(name):
        s=state(name+'a');t=state(name+'b')
        a=parameter('freq'+name,-4,4);b=parameter('mix'+name,-4,4)
        rule='Argument'+name
        grammar['rules'][rule]=[f'{a}*{s}',f'{a}*({s}*{t})',f'({a}*{s}+{b}*{t})']
        return rule
    drift='+'.join(parameter('linear'+s,-1,1)+'*'+s for s in problem['states'])
    quad=parameter('quadratic',-.5,.5)+'*'+state('qa')+'*'+state('qb')
    a,b=argument('A'),argument('B')
    amp=parameter('amplitude',-1.5,1.5);amp_b=parameter('amplitudeB',-1.5,1.5)
    terms=[('sin',f'{amp}*sin({a})',21),('cos',f'{amp}*cos({a})',21)]
    for left,right in [('sin','sin'),('sin','cos'),('cos','cos')]:
        terms.append((f'{left}_times_{right}',f'{amp}*{left}({a})*{right}({b})',21*21))
        terms.append((f'{left}_plus_{right}',f'{amp}*{left}({a})+{amp_b}*{right}({b})',21*21))
    if hint:
        # An explicitly supplied public motif, not inferred from recovered truth.
        terms=[('hint',f'{amp}*({hint})',1),
            ('hint_times_sin',f'{amp}*({hint})*sin({a})',21),
            ('hint_times_cos',f'{amp}*({hint})*cos({a})',21),
            ('hint_plus_sin',f'{amp}*({hint})+{amp_b}*sin({a})',21),
            ('hint_plus_cos',f'{amp}*({hint})+{amp_b}*cos({a})',21)]
    families=[('linear',drift,1),('linear_quadratic',drift+'+'+quad,9)]
    for background in ['linear','linear_quadratic']:
        base=drift if background=='linear' else drift+'+'+quad
        for name,term,count in terms:
            families.append((name+'__'+background,base+'+'+term,count*(1 if background=='linear' else 9)))
    manifest=[]
    for name,expression,count in families:
        grammar['families'].append(dict(id=name,tags=['mechanism:'+name],rhs={'z':expression},
            limits=dict(max_skeletons=count+1,max_variants=count+1,max_configurations=(count+1)*rows,
                max_derivations=max(100,(count+1)*20),max_expansion_steps=max(10000,(count+1)*1000))))
        manifest.append(dict(id=name,resolved_syntax_count=count,expected_configurations=count*rows,coverage='finite_enumeration'))
    request=dict(problem=dict(problem,trajectories=[problem['trajectories'][i] for i in indices]),grammar=grammar,
        execution=dict(max_seconds=120,dedup_bytes_per_family=2<<20))
    plan=dict(version=1,scope='public-clue-assisted grammar' if hint else 'operator-only initial grammar, identical for both datasets',
        public_motif=hint,seed=seed,rows=rows,
        trajectory_indices=list(indices),families=manifest,
        expected_configurations=sum(f['expected_configurations'] for f in manifest),
        priors=['one unknown RHS; additive linear background',
            'optional one quadratic monomial; single, multiplied or added sin/cos terms',
            'arguments are scaled states, scaled products or weighted two-state sums',
            'No recovered coefficients, private truth or estimated derivatives used.',
            'Exact public motif is used.' if hint else 'No exact oscillation clues used.'])
    return request,plan


def build(problem, *, rows=None, seed=2026091401, indices=None, dt=.125, hint=None, profile=None):
    from sparse import build as sparse_build
    return sparse_build(problem,rows=rows,seed=seed,indices=indices,dt=dt,hint=hint,profile=profile)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--knowns',type=Path,required=True);p.add_argument('--trajectories',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',type=int,help='override every class bank size')
    p.add_argument('--seed',type=int,default=2026091401);p.add_argument('--indices',help='explicit whole-trajectory subset; default all')
    p.add_argument('--public-motif',help='optional expression explicitly supplied in the public clues')
    p.add_argument('--dt',type=float,default=.125)
    p.add_argument('--profile',type=Path,help='version-two allocation/range profile JSON')
    p.add_argument('--legacy',action='store_true',help='reproduce the original limited argument language')
    a=p.parse_args()
    if a.legacy and a.profile:p.error('--profile cannot be used with --legacy')
    plan_path=a.output.with_suffix('.plan.json')
    if a.output.exists() or plan_path.exists():p.error('output request or plan already exists')
    kwargs=dict(seed=a.seed,indices=tuple(map(int,a.indices.split(','))) if a.indices else None,dt=a.dt,hint=a.public_motif)
    if a.legacy:
        kwargs.update(rows=a.rows or 16384)
        if kwargs['indices'] is None:kwargs['indices']=(0,4,8,12)
        q,m=build_legacy(load_problem(a.knowns,a.trajectories),**kwargs)
    else:
        q,m=build(load_problem(a.knowns,a.trajectories),rows=a.rows,
                  profile=json.loads(a.profile.read_text()) if a.profile else None,**kwargs)
    m['inputs']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.knowns,a.trajectories]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(q,f,indent=2)
    with plan_path.open('x') as f:json.dump(m,f,indent=2)
    print(json.dumps(dict(families=len(m['families']),configurations=m['expected_configurations'])))


if __name__=='__main__':main()
