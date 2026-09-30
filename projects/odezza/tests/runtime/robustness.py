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
"""Reproducible native grammar robustness campaign, using only the standard library.

Synthetic truth is used to generate observations and to check controlled coverage;
this is execution/recovery calibration, not an unrestricted blind-search benchmark.
Run from an existing CUDA checkout with PYTHONPATH=python and EAGER module loading.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import struct
import time
import traceback
from unittest.mock import patch

from odezza.grammar.native_service import NativeService

ROOT = Path(__file__).resolve().parents[2]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def philox(seed, stream, index, normal):
    """Independent integer Philox and scalar Box-Muller reference."""
    mask = (1 << 32)-1
    block, lane = divmod(index, 4)
    domain = (stream << 1) | normal
    c = [block & mask, block >> 32, domain & mask, domain >> 32]
    k0, k1 = seed & mask, seed >> 32
    for _ in range(10):
        a, b = 0xd2511f53*c[0], 0xcd9e8d57*c[2]
        c = [(b >> 32)^c[1]^k0, b & mask, (a >> 32)^c[3]^k1, a & mask]
        k0, k1 = (k0+0x9e3779b9) & mask, (k1+0xbb67ae85) & mask
    u = [f32(((v >> 9)+.5)*2**-23) for v in c]
    if not normal:
        return u[lane]
    pair = lane//2*2
    radius = math.sqrt(f32(-2*f32(math.log(u[pair]))))
    angle = f32(f32(2*math.pi)*u[pair+1])
    return f32(radius*(math.cos(angle) if lane % 2 == 0 else math.sin(angle)))


def check_rng(candidate):
    values = [struct.unpack('<f', bytes.fromhex(x)[::-1])[0] for x in candidate['value_bits']]
    checked = 0
    for slot, actual in zip(candidate['slots'], values):
        if slot['kind'] < 2:
            continue
        raw = philox(int(slot['seed']), int(slot['stream']), int(slot['axis_index']), slot['kind'] == 3)
        a = values[slot['scale_slot']] if slot['scale_slot'] >= 0 else f32(slot['scale'])
        b = values[slot['shift_slot']] if slot['shift_slot'] >= 0 else f32(slot['shift'])
        transform = slot['transform']
        expected = raw
        if transform in (1, 3):
            expected = f32(f32(a*raw)+b)
        elif transform == 2:
            expected = f32(f32(f32(1-raw)*a)+f32(raw*b))
        elif transform == 4:
            expected = f32(math.exp(f32(f32(f32(1-raw)*f32(math.log(a)))+f32(raw*f32(math.log(b))))))
        assert abs(actual-expected) <= 3e-6*max(1, abs(expected)), (slot, actual, expected)
        checked += 1
    return checked


def observe(truth, *, trajectories=3, duration=.6, points=6, mask=False, seed=1):
    """Fine FP64 RK4 data from independent source expressions, not native bytecode."""
    rng = random.Random(seed)
    states = [f'x{i}' for i in range(len(truth))]
    codes = [compile(x, '<synthetic truth>', 'eval') for x in truth]
    functions = {name: getattr(math, name) for name in ('sin', 'cos', 'tanh', 'exp', 'log', 'sqrt')}
    functions.update(abs=abs, pow=pow)
    def rhs(y):
        env = dict(functions, **dict(zip(states, y)))
        return [eval(c, {'__builtins__': {}}, env) for c in codes]
    data = []
    for trajectory in range(trajectories):
        ts = [duration*(i/(points-1))**(1+.15*(trajectory % 3)) for i in range(points)]
        initial = [rng.uniform(.15, .8) for _ in states]
        y, values = initial[:], [initial[:]]
        for a, b in zip(ts, ts[1:]):
            count = math.ceil((b-a)/.001)
            dt = (b-a)/count
            for _ in range(count):
                k1 = rhs(y)
                k2 = rhs([v+dt*k/2 for v, k in zip(y, k1)])
                k3 = rhs([v+dt*k/2 for v, k in zip(y, k2)])
                k4 = rhs([v+dt*k for v, k in zip(y, k3)])
                y = [v+dt*(a+2*b+2*c+d)/6 for v,a,b,c,d in zip(y,k1,k2,k3,k4)]
            values.append([None if mask and (i+trajectory) % 3 == 1 else v for i,v in enumerate(y)])
        data.append(dict(initial=initial, times=ts, values=values))
    return dict(states=states, trajectories=data)


def base(n=2):
    return dict(version=1, states=[f'x{i}' for i in range(n)],
                integration=dict(method='rk4', dt=.01),
                rhs={f'x{i}': f'-x{i}' for i in range(n)},
                limits=dict(max_skeletons=256, max_variants=256, max_configurations=2000000,
                            max_derivations=20000, max_expansion_steps=2000000),
                retain={'global':8})


def grammar(n=2):
    g = base(n)
    g['retain'] = {'global': {'k':8, 'unit':'numeric_candidate'}}
    return g


def cases(seed, random_count):
    p = observe(['-x0+.2*x1', '-.4*x1'], seed=seed)
    g = grammar(); g['rhs']['x0'] = '-const.a*x0+const.b*x1'; g['rhs']['x1'] = '-theta.d*x1'
    g['constant_banks'] = {'a':[.5,1,1.5], 'b':[0,.2,.4]}
    g['constants'] = {'a':{'bank':'a'},'b':{'bank':'b'},'unused':{'value':123}}
    g['parameters'] = {'d':{'initial':.4}}
    yield dict(name='explicit_cartesian_recovery', problem=p, grammar=g, expected=9, solved=True, pair=True)
    for normal in (False, True):
        for transform in ('identity','affine', 'normal' if normal else 'uniform')+(() if normal else ('log_uniform',)):
            g = grammar(); g['rhs']['x0'] = '-rng.a*x0+.1*x1'
            g['rng_banks'] = {'b':{'base':'normal01' if normal else 'uniform01','count':257,'seed':2**64-1}}
            tr = {'identity':{'kind':'identity'}, 'affine':{'kind':'affine','scale':-.2,'shift':.7},
                  'normal':{'kind':'normal','std':.2,'mean':.7},
                  'uniform':{'kind':'uniform','low':.2,'high':1.2},
                  'log_uniform':{'kind':'log_uniform','low':.02,'high':2}}[transform]
            g['rng'] = {'a':{'bank':'b','transform':tr}}
            yield dict(name=f'{normal=}_{transform}', problem=p, grammar=g, expected=257, pair=True)
    for scope in ('run','skeleton'):
        for shared in (False,True):
            g = grammar(); g['rhs']['x0']='R'; g['rules']={'R':['-rng.u*x0+rng.n','-rng.u*sin(x0)+rng.n']}
            g['constants']={'lo':{'values':[.1,.2]},'hi':{'values':[.8,1.]},'sd':{'values':[0,.1]}}
            g['rng_banks']={'u':{'base':'uniform01','count':7,'seed':91,'scope':scope},
                            'n':{'base':'normal01','count':7 if shared else 5,'seed':82,'scope':scope}}
            g['rng']={'u':{'bank':'u','transform':{'kind':'uniform','low':'const.lo','high':'const.hi'}},
                      'n':{'bank':'n','transform':{'kind':'normal','std':'const.sd','mean':0}}}
            if shared:
                for v in g['rng'].values(): v['axis']='trial'
            yield dict(name=f'dependent_{scope}_{shared=}', problem=p, grammar=g, expected=2*8*7*(1 if shared else 5),pair=True)
    g=grammar();g['rhs']['x0']='R+R'
    g['rng_banks']={'u':{'base':'uniform01','count':5,'seed':51}}
    g['rules']={'R':{'expr':'-rng.u*x0+param.p','locals':{'constants':{'s':{'values':[.2,.4]}},
                  'parameters':{'p':{'initial':.01}},'rng':{'u':{'bank':'u','transform':{'kind':'affine','scale':'const.s'}}}}}}
    yield dict(name='independent_local_slots',problem=p,grammar=g,expected=100,pair=True)
    g=grammar();g['rhs']['x0']='shape.S+shape.S';g['shapes']={'S':['-.3*x0','-.5*x0','-.7*x0']}
    yield dict(name='shared_shape_choice',problem=p,grammar=g,expected=3,pair=True)
    for coverage in ('all','explicit','sample','joint'):
        g=grammar(6); g['rhs']['x0']='-.5*x0+.1*leaf.a+.02*leaf.b'
        g['leaves']={'a':{'states':g['states'],'arity':4},'b':{'states':g['states'],'arity':2}}
        variants=15*15
        if coverage=='explicit':
            g['leaves']['a'].update(coverage='explicit',groups=[['x5','x3','x1','x0']])
            g['leaves']['b'].update(coverage='explicit',groups=[['x4','x2']]);variants=1
        if coverage=='sample':
            for leaf in g['leaves'].values():leaf.update(coverage='sample',samples=3,seed=123)
            variants=9
        if coverage=='joint':g['toggle_sampling']={'count':7,'seed':321};variants=7
        yield dict(name=f'toggles_{coverage}',problem=observe([f'-x{i}' for i in range(6)],seed=seed),
                   grammar=g,expected=variants*8,pair=True)
    g=grammar();g['rhs']['x0']='R'
    g['rules']={'R':['-x0','sin(R)','tanh(R)','R+.1*R']}
    g['expansion']={'strategy':'sample','seed':seed,'max_nodes':23,'max_depth':7,'max_expansion_depth':5}
    g['limits']['max_skeletons']=g['limits']['max_variants']=48
    yield dict(name='recursive_sampled',problem=p,grammar=g,pair=True)
    g=grammar();g['rhs']['x0']='R';g['rules']={'R':['-x0','-x0',{'expr':'-x0','tags':['late','μ-tag']},'-2*x0']}
    g['families']=[{'id':'a','tags':['first']},{'id':'b','rules':{'R':['-x0','-.3*x0']},'tags':['second']}]
    g['retain']={'global':2,'per_family':2,'by_tag':{'late':2,'μ-tag':2,'second':2}}
    yield dict(name='family_overrides_late_tags',problem=p,grammar=g,expected=4,pair=True)
    for k in (0,1,17,32):
        g=grammar(1);g['rhs']['x0']='-const.c*x0';g['constants']={'c':{'values':[1]*20+[.1+.02*i for i in range(35)]}}
        g['retain']={'global':{'k':k,'unit':'numeric_candidate'}}
        yield dict(name=f'numeric_retention_{k}',problem=observe(['-x0'],seed=seed),grammar=g,expected=55,pair=True)
    g=grammar();g['rhs']['x0']='R'
    ops=['-x0', '-sin(x0)', '-cos(x0)', '-tanh(x0)', '-exp(-x0)', '-log(1+x0*x0)',
         '-sqrt(1+x0*x0)', '-abs(x0)', '-pow(x0,2)', '-pow(1+x0*x0,-2)', '-pow(x0,0)',
         '-pow(.2*x0,16)', '-x0/(1+x1*x1)', '-(-x0)', 'sqrt(-1)', 'log(-1)', '1/0']
    g['rules']={'R':ops};g['retain']={'global':{'k':32,'unit':'numeric_candidate'}}
    yield dict(name='operator_domains',problem=p,grammar=g,expected=len(ops),invalid=3,pair=True)
    g=grammar();g['rhs']['x0']='R';g['rules']={'R':['x0+x1','sin(x0+x1)']}
    g['expansion']={'max_nodes':1}
    yield dict(name='all_structures_pruned',problem=p,grammar=g,expected=0,pair=True)
    # Coupled fake unknown systems, randomized only from this seed on the host.
    rng=random.Random(seed)
    for i in range(random_count):
        n=(2,3,6,8)[i%4];unknown=i%n;a,b=(.25,.5,.75)[i%3],(.1,.2,.3)[(i//3)%3]
        expressions=[f'x{(unknown+1)%n}',f'sin(x{(unknown+1)%n})',
                     f'tanh(x{(unknown+1)%n}*x{(unknown+2)%n})',
                     f'exp(-x{(unknown+1)%n}*x{(unknown+1)%n})/(1+x{(unknown+2)%n}*x{(unknown+2)%n})']
        choice=rng.randrange(len(expressions))
        truth=[f'-.4*x{j}+.1*x{(j+1)%n}' for j in range(n)]
        truth[unknown]=f'-{a}*x{unknown}+{b}*({expressions[choice]})'
        p2=observe(truth,trajectories=2+i%3,duration=.3+.3*(i%4),points=3+i%5,mask=bool(i%2),seed=seed+i)
        p2['known_rhs']={f'x{j}':e for j,e in enumerate(truth) if j!=unknown}
        g=grammar(n);g['rhs']={f'x{unknown}':f'-const.a*x{unknown}+const.b*R'}
        g['constants']={'a':{'values':[.25,.5,.75]},'b':{'values':[.1,.2,.3]}}
        g['rules']={'R':[{'expr':e,'tags':[f'structure-{j}']} for j,e in enumerate(expressions)]}
        g['retain']={'global':{'k':4,'unit':'resolved_structure'},'per_family':4}
        yield dict(name=f'coupled_recovery_{i:02d}_{n}states',problem=p2,grammar=g,expected=36,solved=True,pair=True,truth=truth)
    for n in (1,3,6,12):
        for terms in (8,32,96):
            g=grammar(n);g['expansion']={'max_nodes':4096,'max_depth':128}
            term='sin(.1*x0)*cos(.1*x0)'
            def balanced(xs):
                if len(xs)==1:return xs[0]
                m=len(xs)//2;return '('+balanced(xs[:m])+'+'+balanced(xs[m:])+')'
            g['rhs']['x0']='-.5*x0+.001*'+balanced([term]*terms)
            p2=observe([f'-.5*x0+{terms}*.001*({term})']+[f'-x{i}' for i in range(1,n)],seed=seed)
            yield dict(name=f'ast_{terms}terms_{n}states',problem=p2,grammar=g,expected=1,solved=True,
                       execution={'patch_capacity':16},pair=True)
    # One job can require >4096 distinct tiny skeleton-scoped banks across pages.
    g=grammar(1);g['rhs']['x0']='R';g['rules']={'R':[f'-x0+{i}*.000001+.01*rng.u' for i in range(4200)]}
    g['rng_banks']={'u':{'base':'uniform01','count':5,'seed':71,'scope':'skeleton'}};g['rng']={'u':{'bank':'u'}}
    g['limits'].update(max_skeletons=4200,max_variants=4200)
    yield dict(name='pool_entry_pressure',problem=observe(['-x0'],points=3,duration=.05,seed=seed),grammar=g,expected=21000,
               execution={'batch_variants':64,'max_bank_bytes':1048576},evictions=True)
    yield dict(name='pool_byte_pressure',problem=observe(['-x0'],points=3,duration=.05,seed=seed),grammar=g,expected=21000,
               execution={'batch_variants':16,'max_bank_bytes':1024},evictions=True)
    # Fully known systems still need a scoring candidate even with an empty variable RHS.
    g=grammar();g['rhs']={};q=observe(['-x0','-.5*x1'],seed=seed);q['known_rhs']={'x0':'-x0','x1':'-.5*x1'}
    yield dict(name='all_known',problem=q,grammar=g,expected=1,solved=True,pair=True)
    # Exercise provided grammars on synthetic observations, preserving all operators,
    # banks and retention policies. Only population/work ceilings are bounded here.
    for path in sorted((ROOT/'examples/grammar').rglob('*.json')):
        supplied=json.loads(path.read_text())
        if not ('rhs' in supplied or 'families' in supplied) or supplied.get('kind','compile')!='compile' or supplied.get('integration',{}).get('stiff'):
            continue
        g=deepcopy(supplied);g['limits']=dict(max_skeletons=128,max_variants=128,max_configurations=262144,
                                            max_derivations=10000,max_expansion_steps=2000000,max_seconds=120)
        q=observe([f'-.1*x{i}' for i in range(len(g['states']))],trajectories=2,points=4,duration=.12,seed=seed)
        q['states']=g['states']
        yield dict(name='supplied_'+path.stem,problem=q,grammar=g,pair=True,
                   source=str(path.relative_to(ROOT)),original_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    g=grammar(6);g['limits'].update(max_skeletons=4096,max_variants=4096,max_configurations=100000000)
    g['rhs']={f'x{i}':f'-rng.d*x{i}+.1*R{i}' for i in range(6)}
    g['rules']={f'R{i}':[f'x{(i+1)%6}',f'sin(x{(i+2)%6})',f'tanh(x{(i+3)%6})',f'x{i}/(1+x{(i+1)%6}*x{(i+1)%6})'] for i in range(6)}
    g['rhs']['x0']+='+.01*leaf.s+rng.n';g['leaves']={'s':{'states':['x0','x1','x2','x3'],'arity':4}}
    g['constants']={'shift':{'values':[.2,.4]}}
    g['rng_banks']={'u':{'base':'uniform01','count':2048,'seed':7},'n':{'base':'normal01','count':2048,'seed':11}}
    g['rng']={'d':{'bank':'u','axis':'trial','transform':{'kind':'affine','scale':.5,'shift':'const.shift'}},
              'n':{'bank':'n','axis':'trial','transform':{'kind':'normal','std':.01,'mean':0}}}
    g['expansion']={'max_nodes':256,'max_depth':24}
    yield dict(name='mixed_saturation',problem=observe([f'-.5*x{i}+.1*x{(i+1)%6}' for i in range(6)],trajectories=2,points=4,duration=.12,seed=seed),
               grammar=g,expected=4096*4096*4,execution={'batch_variants':128,'max_chunk_configurations':2097152})


def invalid_cases(seed):
    p=observe(['-x0','-x1'],seed=seed)
    specs=[('time_leaf',lambda g,p:g['rhs'].update(x0='t')),
           ('unknown_operator',lambda g,p:g['rhs'].update(x0='erf(x0)')),
           ('large_power',lambda g,p:g['rhs'].update(x0='pow(x0,17)')),
           ('fractional_power',lambda g,p:g['rhs'].update(x0='pow(x0,.5)')),
           ('unknown_field',lambda g,p:g.update(garbage=True)),
           ('choice_wrapper_field',lambda g,p:(g.update(rules={'R':{'choices':['-x0'],'weights':[1]}}),g['rhs'].update(x0='R'))),
           ('stiff',lambda g,p:g['integration'].update(stiff=True)),
           ('adaptive',lambda g,p:g['integration'].update(rtol=1e-6)),
           ('empty_grid',lambda g,p:(g.update(constants={'a':{'values':[]}}),g['rhs'].update(x0='const.a*x0'))),
           ('missing_ic',lambda g,p:p['trajectories'][0].pop('initial')),
           ('bad_ic',lambda g,p:p['trajectories'][0].update(initial=[1,2])),
           ('duplicate_time',lambda g,p:p['trajectories'][0]['times'].__setitem__(1,0)),
           ('ragged_values',lambda g,p:p['trajectories'][0]['values'][1].pop()),
           ('unknown_fixed_state',lambda g,p:p.update(known_rhs={'nonesuch':'x0'})),
           ('duplicate_family',lambda g,p:g.update(families=[{'id':'a'},{'id':'a'}])),
           ('family_conflict',lambda g,p:(g.update(families=[{'id':'a','limits':{'max_configurations':10}}]),g['limits'].update(max_configurations=2))),
           ('bad_arity',lambda g,p:(g.update(leaves={'a':{'states':['x0','x1'],'arity':3}}),g['rhs'].update(x0='leaf.a'))),
           ('bad_retention',lambda g,p:g.update(retain={'global':257})),
           ('unsupported_allocation',lambda g,p:g.update(allocation={'redistribute_unused':True}))]
    for name,mutate in specs:
        g=grammar();q=deepcopy(p);mutate(g,q)
        yield dict(name='reject_'+name,problem=q,grammar=g,reject=True)
    for name, transform in [('negative_std',{'kind':'normal','std':-1}),
                             ('bad_uniform',{'kind':'uniform','low':2,'high':1}),
                             ('log_zero',{'kind':'log_uniform','low':0,'high':1}),
                             ('missing_dependency',{'kind':'affine','scale':'const.missing'})]:
        g=grammar();g['rhs']['x0']='rng.r*x0';g['rng_banks']={'b':{'base':'normal01' if name=='negative_std' else 'uniform01','count':7}}
        g['rng']={'r':{'bank':'b','transform':transform}}
        yield dict(name='reject_'+name,problem=p,grammar=g,reject=True)
    for name,opts in [('host_memory',{'max_host_bytes':512}),('device_memory',{'max_device_bytes':1}),
                      ('pool_memory',{'max_bank_bytes':8}),('dedup_memory',{'dedup_bytes_per_family':1})]:
        g=grammar();g['constants']={'a':{'values':[1,2,3]}};g['rhs']['x0']='const.a*x0'
        yield dict(name='reject_'+name,problem=p,grammar=g,execution=opts,reject=True)
    for name,extra in [('unequal_shared_axes',True),('bad_distribution_transform',False)]:
        g=grammar();g['rhs']['x0']='rng.a*x0+rng.b'
        g['rng_banks']={'a':{'base':'uniform01','count':3},'b':{'base':'normal01','count':4}}
        g['rng']={'a':{'bank':'a','axis':'trial'},'b':{'bank':'b','axis':'trial'}}
        if not extra:g['rng']['a']['transform']={'kind':'normal','std':1}
        yield dict(name='reject_'+name,problem=p,grammar=g,reject=True)


def signature(r):
    return [(k,r['candidates'][k]['mse'],r['candidates'][k]['resolved_programs'],
             r['candidates'][k]['value_bits'],sorted(r['candidates'][k]['tags']))
            for k in r['leaderboards']['global']],r['leaderboards']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='/tmp/odezza-robustness-cache')
    ap.add_argument('--output',required=True)
    ap.add_argument('--seed',type=int,default=20260911)
    ap.add_argument('--random-count',type=int,default=24)
    ap.add_argument('--only',default='')
    ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--devices',type=int,nargs='+')
    args=ap.parse_args()
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'device':args.device,'devices':args.devices,'host':platform.node(),
            'purpose':'controlled synthetic grammar robustness, not open-ended blind discovery',
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'binary_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'build/libodezza.so',ROOT/'build/runtime/libodezza_runtime.so')},
            'cases':[], 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ('frontend','runtime') for p in sorted((ROOT/folder).glob('*.[ch]'))}}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    def save():path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    service=NativeService(args.root,device=args.device,devices=args.devices,max_jobs=1024)
    def run(case, suffix='', override=None):
        entry={'name':case['name']+suffix,'request':{'problem':case['problem'],'grammar':case['grammar']}}
        execution=dict(batch_variants=32,module_systems=16,dedup_bytes_per_family=4*1024**2,max_seconds=120)
        execution.update(case.get('execution',{}));execution.update(override or {})
        entry['request']['execution']=execution
        if 'truth' in case:entry['truth']=case['truth']
        for key in ('source','original_sha256'):
            if key in case:entry[key]=case[key]
        result['cases'].append(entry);save()
        begin=time.monotonic()
        try:
            handle=service.submit(**entry['request']);entry['job_id']=handle['job_id']
            report=service.jobs[handle['job_id']].result(timeout=180)
            entry['wall_seconds']=time.monotonic()-begin;entry['report']=report;save()
            if case.get('reject'):
                assert report['status']=='failed' and report['error'],report['status']
                assert not report['runtime_quarantined'],report['error']
            else:
                assert report['status']=='complete',(report['status'],report['error'])
                assert report['retention_complete']
                counts=report['counts'];assert counts['valid']+counts['invalid']==counts['completed_configurations']
                if 'expected' in case:assert counts['completed_configurations']==case['expected'],counts
                if 'invalid' in case:assert counts['invalid']==case['invalid'],counts
                if case.get('evictions'):assert report['cache']['numeric_pool_evictions']>0,report['cache']
                entry['cpu_checked']=entry['rng_checked']=0;entry['max_cpu_error']=0.
                for identifier,c in report['candidates'].items():
                    replay=service.replay(handle['job_id'],identifier,cpu=True)
                    cpu=replay['cpu_reference']['mse'];error=abs(cpu-c['mse'])
                    assert math.isfinite(cpu) and error<=2e-9+5e-5*abs(cpu),(identifier,cpu,c['mse'])
                    entry['max_cpu_error']=max(entry['max_cpu_error'],error)
                    assert service.replay(handle['job_id'],address=c['origin'])['candidate']==c
                    entry['rng_checked']+=check_rng(c);entry['cpu_checked']+=1
                if case.get('solved'):
                    best=report['candidates'][report['leaderboards']['global'][0]]
                    assert best['mse']<1e-10,best['mse']
            entry['passed']=True
        except Exception:
            entry['passed']=False;entry['failure']=traceback.format_exc()
        print(entry['name'],'PASS' if entry['passed'] else 'FAIL',entry.get('report',{}).get('counts'),entry.get('failure','')[-500:],flush=True)
        save();return entry
    try:
        with patch('sqlite3.connect',side_effect=AssertionError('native execution opened SQLite')):
            for case in list(cases(args.seed,args.random_count))+list(invalid_cases(args.seed)):
                if args.only and args.only not in case['name']:continue
                if case['name']=='pool_entry_pressure' and args.devices and len(args.devices)>1:
                    case=deepcopy(case)
                    count=4200*len(args.devices)
                    case['grammar']['rules']['R']=[f'-x0+{i}*.000001+.01*rng.u' for i in range(count)]
                    case['grammar']['limits'].update(max_skeletons=count,max_variants=count)
                    case['expected']=count*5
                    case['name']+='_'+str(len(args.devices))+'devices'
                original=run(case)
                if case.get('pair') and original['passed']:
                    second=run(case,'_tiled',{'batch_variants':1,'max_chunk_configurations':1024})
                    if second['passed']:
                        same=signature(original['report'])==signature(second['report'])
                        second['chunk_invariant']=same
                        if not same:second.update(passed=False,failure='Chunking changed retained leaderboard')
                        save()
            # Ensure rejected jobs did not poison the persistent context.
            run(dict(name='final_context_reuse',problem=observe(['-x0'],seed=args.seed),grammar=grammar(1),expected=1,solved=True))
    finally:
        service.close()
        result['completed_utc']=datetime.now(timezone.utc).isoformat()
        result['summary']={'passed':sum(c.get('passed',False) for c in result['cases']),
                           'failed':sum(not c.get('passed',False) for c in result['cases']),
                           'configurations':sum(c.get('report',{}).get('counts',{}).get('completed_configurations',0) for c in result['cases'])}
        save()
    print(json.dumps(result['summary']),flush=True)
    raise SystemExit(bool(result['summary']['failed']))


if __name__=='__main__':main()
