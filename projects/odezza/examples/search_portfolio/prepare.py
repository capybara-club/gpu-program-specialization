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
"""Prepare compact native JSON; the C worker expands ASTs and samples Philox.

No model fitting, derivative estimates, AST expansion, or private-truth input.
Input is a native {problem: ...} request or its standalone problem section.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import secrets

HERE=Path(__file__).resolve().parent


def tree_count(leaves, unary, binary, depth):
    count=leaves
    for _ in range(depth):count=leaves+unary*count+binary*count*count
    return count


def build(problem, *, dt, depth=2, rows=None, scale=2.0, seed=0,
          backgrounds=('none','self'), sample_asts=None, operators=None,
          family_k=8, max_configurations=1_000_000_000, trajectory_indices=None):
    p=deepcopy(problem.get('problem',problem));states=p['states']
    if not states or len(set(states))!=len(states):raise ValueError('unique state names required')
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',s) or s.startswith('Portfolio') for s in states):
        raise ValueError('state names must be identifiers outside the Portfolio prefix')
    unknown=[s for s in states if s not in p.get('known_rhs',{})]
    if len(unknown)!=1:raise ValueError('this prepared controller currently requires exactly one unknown RHS')
    original_trajectory_count=len(p['trajectories'])
    if trajectory_indices is not None:
        if not trajectory_indices or len(set(trajectory_indices))!=len(trajectory_indices) or any(
                type(i) is not int or not 0<=i<original_trajectory_count for i in trajectory_indices):
            raise ValueError('trajectory indices must be distinct valid zero-based indices')
        p['trajectories']=[p['trajectories'][i] for i in trajectory_indices]
    if rows is None:rows=2048 if len(states)<=3 else 512 if len(states)<=6 else 128
    if any(type(v) is not int for v in (depth,rows,seed,family_k,max_configurations)):
        raise ValueError('depth, rows, seed, family_k and max_configurations must be integers')
    if family_k<1 or max_configurations<1:raise ValueError('positive retention and work limits required')
    if depth not in (1,2,3) or rows<1 or not math.isfinite(dt) or not math.isfinite(scale) or dt<=0 or scale<=0:
        raise ValueError('invalid depth, rows, dt or coefficient scale')
    if not 0<=seed<2**64:raise ValueError('seed must be uint64')
    if sample_asts is not None and (type(sample_asts) is not int or sample_asts<1):
        raise ValueError('sample_asts must be a positive integer')
    if not backgrounds or any(b not in ('none','self') for b in backgrounds) or len(set(backgrounds))!=len(backgrounds):
        raise ValueError('backgrounds must be distinct none/self choices')
    ops=operators or json.loads((HERE/'operators.json').read_text())
    unary,binary=ops['unary'],ops['binary']
    if not unary or not binary:raise ValueError('nonempty operator catalog required')
    names=['leaf']+list(unary)+list(binary)
    if len(set(names))!=len(names) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',s) for s in names):
        raise ValueError('operator identifiers must be distinct, including the reserved leaf identifier')
    g=dict(version=1,states=states,integration=dict(method='rk4',dt=dt),rules={},shapes={},leaves={},rng={},
        rng_banks={'draws':dict(base='uniform01',count=rows,seed=seed,scope='run')},
        expansion=dict(strategy='sample' if sample_asts else 'enumerate',seed=seed,
            max_nodes=max(64,32*2**depth),max_depth=64,max_expansion_depth=64),
        retain={'global':{'k':32,'unit':'resolved_structure'},'per_family':{'k':family_k,'unit':'resolved_structure'}},families=[])
    def coefficient(name):
        # One zipped axis gives rows independent vectors, not rows^parameter_count.
        g['rng'][name]=dict(bank='draws',axis='trial',stream=name,
            transform=dict(kind='uniform',low=-scale,high=scale))
        return 'rng.'+name
    def terminal(path):
        key='PortfolioState'+path
        alternatives=[];at=0
        while at<len(states):
            remaining=len(states)-at
            arity=4 if remaining>=4 else 2 if remaining>=2 else 1
            group=states[at:at+arity]
            if arity==1:alternatives.append(group[0])
            else:
                axis='s'+path+'g'+str(at)
                g['leaves'][axis]=dict(states=group,arity=arity,coverage='explicit',groups=[group])
                alternatives.append('leaf.'+axis)
            at+=arity
        alternatives.append('1')
        g['rules'][key]=alternatives
        shape='term'+path;g['shapes'][shape]=coefficient('c'+path)+'*'+key
        return 'shape.'+shape
    def node(level,path):
        leaf=terminal(path)
        if not level:return leaf
        left=node(level-1,path+'L');right=node(level-1,path+'R')
        weight=coefficient('u'+path)
        alts=[leaf]
        alts.extend(weight+'*'+form.format(x=left) for form in unary.values())
        alts.extend(form.format(x=left,y=right) for form in binary.values())
        name='PortfolioNode'+path;g['rules'][name]=alts
        return name
    leaf=terminal('root');left=node(depth-1,'L');right=node(depth-1,'R')
    root_weight=coefficient('uRoot');damping=coefficient('drift')
    lower=tree_count(len(states)+1,len(unary),len(binary),depth-1)
    roots=[('leaf',leaf,len(states)+1)]
    roots.extend((name,root_weight+'*'+form.format(x=left),lower) for name,form in unary.items())
    roots.extend((name,form.format(x=left,y=right),lower*lower) for name,form in binary.items())
    manifest=[];total=0
    for background in backgrounds:
        for name,expr,trees in roots:
            ident=name+'__'+background
            if background=='self':expr+='+'+damping+'*'+unknown[0]
            # Loose finite upper bounds leave space to observe EXHAUSTED. These
            # are reservations, not promised unique AST or coefficient counts.
            variants=sample_asts if sample_asts else trees+1
            arity=4 if len(states)>=4 else 2 if len(states)>=2 else 1
            configurations=variants*rows*(arity**(2**depth) if sample_asts else 1)
            total+=configurations
            g['families'].append(dict(id=ident,tags=['root:'+name,'background:'+background],rhs={unknown[0]:expr},
                limits=dict(max_skeletons=variants,max_variants=variants,max_configurations=configurations,
                    max_derivations=max(100,variants*20),max_expansion_steps=max(10000,variants*1000))))
            manifest.append(dict(id=ident,ordered_labelled_trees=trees,reserved_configurations=configurations,
                coefficient_rows=rows,coverage='sampled' if sample_asts else 'finite_enumeration'))
    if total>max_configurations:
        raise ValueError(f'portfolio reserves {total:,} configurations, exceeding local ceiling {max_configurations:,}; lower depth/rows or explicitly sample structures')
    if len(manifest)>64:raise ValueError('portfolio exceeds the current service 64-family limit')
    # Per-family dedup is explicitly bounded, avoiding the 64 MiB-per-tiny-family default.
    request=dict(problem=p,grammar=g,execution=dict(max_seconds=120,dedup_bytes_per_family=8<<20))
    metadata=dict(version=1,unknown_rhs=unknown[0],depth=depth,seed=seed,coefficient_scale=scale,
        trajectory_selection=dict(original_count=original_trajectory_count,
            indices=list(trajectory_indices) if trajectory_indices is not None else list(range(original_trajectory_count))),
        abstract_tree_definition='ordered labelled states/constant leaves, declared operators, coefficient weights excluded from depth',
        families=manifest,reserved_configurations=total,
        caveats=['Tree counts are abstract syntax counts, not globally distinct numerical functions.',
                 'Exhaustive structure coverage is confirmed only after every family exhausts without pruning or budget truncation.',
                 'Continuous coefficients are sampled, not exhaustively optimized.',
                 'Only explicitly selected complete trajectories are used; their times and initial conditions are preserved. No automatic holdout or window selection.',
                 'The self background adds a linear target-state term beyond the stated core tree depth.'])
    return request,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('problem',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dt',type=float,required=True);p.add_argument('--depth',type=int,choices=[1,2,3],default=2)
    p.add_argument('--rows',type=int,help='default 2048 for <=3 states, 512 for <=6, otherwise 128')
    p.add_argument('--coefficient-scale',type=float,default=2)
    p.add_argument('--seed',type=int);p.add_argument('--sample-asts',type=int)
    p.add_argument('--trajectory-indices',help='explicit zero-based comma-separated subset; whole trajectories, unchanged times/ICs')
    p.add_argument('--backgrounds',default='none,self');p.add_argument('--operators',type=Path,default=HERE/'operators.json')
    p.add_argument('--max-configurations',type=int,default=1_000_000_000)
    a=p.parse_args();source=a.problem.read_bytes()
    q,m=build(json.loads(source),dt=a.dt,depth=a.depth,rows=a.rows,scale=a.coefficient_scale,
        seed=a.seed if a.seed is not None else secrets.randbits(63),backgrounds=tuple(a.backgrounds.split(',')),
        sample_asts=a.sample_asts,operators=json.loads(a.operators.read_text()),max_configurations=a.max_configurations,
        trajectory_indices=[int(i) for i in a.trajectory_indices.split(',')] if a.trajectory_indices is not None else None)
    m['source_sha256']=hashlib.sha256(source).hexdigest()
    if a.output.exists() or a.output.with_suffix('.plan.json').exists():
        p.error('request or plan output already exists; choose a new output name')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(q,f,indent=2,allow_nan=False);f.write('\n')
    with a.output.with_suffix('.plan.json').open('x') as f:json.dump(m,f,indent=2);f.write('\n')
    print(json.dumps(dict(request=str(a.output),families=len(m['families']),reserved_configurations=m['reserved_configurations'],seed=m['seed'])))


if __name__=='__main__':main()
