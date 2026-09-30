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
"""Bounded CPU sensitivity fitting of retained experimental candidates.

The service currently exports resolved literals, not a parameterized program.
This experimental bridge accepts only unique slot-bit identities, rejects
ambiguity, and preserves the original record. It is not the public fitting API.
No symbolic-regression engine or estimated trajectory derivatives are used.
"""
import argparse
import ast
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import struct
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from prepare import load_problem


ZERO=('c',0.0)
ONE=('c',1.0)


def make(op,a,b=None):
    if op=='+':
        if a==ZERO:return b
        if b==ZERO:return a
    if op=='-':
        if b==ZERO:return a
    if op=='*':
        if a==ZERO or b==ZERO:return ZERO
        if a==ONE:return b
        if b==ONE:return a
    return (op,a) if b is None else (op,a,b)


def derivative(node,variable):
    op=node[0]
    if op in ('c','v'):return ONE if node==variable else ZERO
    a=node[1];da=derivative(a,variable)
    if op=='+':return make('+',da,derivative(node[2],variable))
    if op=='-':return make('-',da,derivative(node[2],variable))
    if op=='*':return make('+',make('*',da,node[2]),make('*',a,derivative(node[2],variable)))
    if op=='sin':return make('*',make('cos',a),da)
    if op=='cos':return make('*',('c',-1.0),make('*',make('sin',a),da))
    if op=='exp':return make('*',make('exp',a),da)
    raise ValueError(op)


def unpack(program,slot_bits=()):
    if len(set(slot_bits))!=len(slot_bits):raise ValueError('ambiguous equal-valued slots')
    slots={b:('v',i+3) for i,b in enumerate(slot_bits)}
    data=bytes.fromhex(program);i=0;stack=[];used=set()
    while i<len(data):
        op=data[i];i+=1
        if op==0x80:
            if i!=len(data) or len(stack)!=1:raise ValueError('malformed RETURN')
            if used!=set(slot_bits):raise ValueError('slot was absent from unknown program')
            return stack[0]
        if op==0x81:stack.append(('v',data[i]));i+=1
        elif op==0x83:
            bits=f'{struct.unpack_from("<I",data,i)[0]:08x}'
            value=struct.unpack_from('<f',data,i)[0];i+=4
            if bits in slots:
                # This grammar has no fixed nonzero literals in the unknown RHS.
                # Ambiguous zero values are rejected rather than rebound.
                if value==0:raise ValueError('ambiguous zero-valued coefficient')
                used.add(bits);stack.append(slots[bits])
            else:stack.append(('c',value))
        elif op==0x94:stack.append(make('*',('c',-1.0),stack.pop()))
        elif op in (0x9b,0x9c,0xa1):stack.append(make({0x9b:'sin',0x9c:'cos',0xa1:'exp'}[op],stack.pop()))
        else:
            b,a=stack.pop(),stack.pop()
            stack.append(make({0x90:'+',0x91:'-',0x92:'*'}[op],a,b))
    raise ValueError('missing RETURN')


def known_tree(text):
    def visit(n):
        if isinstance(n,ast.Name):return ('v',['x','y','z'].index(n.id))
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return ('c',float(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return make('*',('c',-1),visit(n.operand))
        if isinstance(n,ast.BinOp):return make({ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}[type(n.op)],visit(n.left),visit(n.right))
        raise ValueError('unsupported known RHS syntax')
    return visit(ast.parse(text,mode='eval').body)


def compile_model(trees,parameters):
    memo={};lines=[]
    def emit(n):
        if n in memo:return memo[n]
        op=n[0]
        if op=='c':return repr(n[1])
        if op=='v':return f's[:,{n[1]}]' if n[1]<3 else f'p[{n[1]-3}]'
        args=[emit(a) for a in n[1:]]
        expr=f'np.{op}({args[0]})' if op in ('sin','cos','exp') else f'({args[0]} {op} {args[1]})'
        name=f't{len(lines)}';lines.append(f'    {name}={expr}');memo[n]=name;return name
    values=[emit(n) for n in trees]
    gradients=[emit(derivative(n,('v',j))) for n in trees for j in range(3+parameters)]
    source='def compiled(s,p):\n'+'\n'.join(lines)+'\n'
    source+='    n=len(s)\n    value=np.column_stack(['+','.join(f'np.broadcast_to({e},(n,))' for e in values)+'])\n'
    source+='    grad=np.stack(['+','.join(f'np.broadcast_to({e},(n,))' for e in gradients)+f'],axis=1).reshape(n,3,{3+parameters})\n'
    source+='    return value,grad\n'
    ns={'np':np};exec(source,ns);return ns['compiled']


def model_for(candidate,known):
    if (set(candidate.get('_fixed_literal_bits',[])) | {'3f800000'}) & set(candidate['value_bits']):
        raise ValueError('a coefficient collides with a fixed public-motif literal')
    tree=unpack(candidate['resolved_programs'][2],candidate['value_bits'])
    return compile_model([known_tree(known['x']),known_tree(known['y']),tree],len(candidate['values']))


def structure_key(candidate):
    """Collapse commutative duplicates with the same named parameter roles."""
    def key(n):
        if n[0]=='v' and n[1]>=3:return ('parameter',candidate['slots'][n[1]-3]['name'])
        if n[0] in ('v','c'):return n
        children=[key(x) for x in n[1:]]
        if n[0] in ('+','*'):
            children=[v for child in children for v in (child[1:] if child[0]==n[0] else [child])]
            children.sort(key=repr)
        return (n[0],*children)
    return key(unpack(candidate['resolved_programs'][2],candidate['value_bits']))


def simulate(model,p,trajectories,*,deadline=None,rtol=2e-7,atol=1e-9):
    times=np.asarray(trajectories[0]['times']);n=len(trajectories);m=len(p)
    if any(not np.array_equal(t['times'],times) for t in trajectories):raise ValueError('common times required in this experiment')
    initial=np.array([t['initial'] for t in trajectories]);q=np.zeros((n,3+3*m));q[:,:3]=initial
    def rhs(t,flat):
        if deadline and time.perf_counter()>deadline:raise TimeoutError('fit time budget')
        q=flat.reshape(n,3+3*m);states=q[:,:3]
        if not np.isfinite(q).all() or np.max(abs(states))>100:raise FloatingPointError('divergent rollout')
        values,grad=model(states,p);sens=q[:,3:].reshape(n,3,m)
        out=np.empty_like(q);out[:,:3]=values
        out[:,3:]=(grad[:,:,:3]@sens+grad[:,:,3:]).reshape(n,3*m)
        return out.ravel()
    sol=solve_ivp(rhs,(times[0],times[-1]),q.ravel(),t_eval=times,method='DOP853',rtol=rtol,atol=atol)
    if not sol.success:raise RuntimeError(sol.message)
    arr=sol.y.reshape(n,3+3*m,len(times)).transpose(0,2,1)
    obs=np.asarray([t['values'] for t in trajectories])
    return (arr[:,1:,:3]-obs[:,1:]).ravel(),arr[:,1:,3:].reshape(-1,m)


def fit_one(job):
    candidate,known,trajectories,seconds=job
    started_at_unix=time.time()
    begin=time.perf_counter();deadline=begin+seconds;p0=np.array(candidate['values'])
    result=dict(candidate_id=candidate['candidate_id'],family=candidate['family'],
        source_origin=candidate['origin'],screen_mse=candidate['mse'],source_values=candidate['values'])
    calls=0;invalid=0;best=None;last=None;cached=None
    try:
        model=model_for(candidate,known)
        def evaluate(p):
            nonlocal calls,invalid,best,last,cached
            if last is not None and np.array_equal(p,last):return cached
            calls+=1
            try:r,j=simulate(model,p,trajectories,deadline=deadline)
            except (FloatingPointError,RuntimeError):
                invalid+=1;length=sum((len(t['times'])-1)*3 for t in trajectories)
                return np.full(length,1e6),np.zeros((length,len(p)))
            mse=float(np.mean(r*r))
            if best is None or mse<best['mse']:best=dict(mse=mse,parameters=p.tolist())
            last=p.copy();cached=(r,j);return cached
        initial=evaluate(p0)[0]
        if np.mean(initial*initial)>1e10:raise ValueError('initial CPU rollout diverged')
        result['initial_full_training_mse']=float(np.mean(initial*initial))
        fitted=least_squares(lambda p:evaluate(p)[0],p0,jac=lambda p:evaluate(p)[1],
            bounds=(-8,8),x_scale='jac',max_nfev=70,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        result.update(status=int(fitted.status),nfev=fitted.nfev)
    except Exception as error:result['error']=repr(error)
    result.update(seconds=time.perf_counter()-begin,calls=calls,invalid_trials=invalid,best=best,
                  started_at_unix=started_at_unix,finished_at_unix=time.time())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--knowns',type=Path,required=True);p.add_argument('--trajectories',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-family',type=int,default=2)
    p.add_argument('--seconds',type=float,default=12);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--fixed-literals',default='',help='known fixed numerical literals in the public motif, checked against slot collisions')
    p.add_argument('--families',help='optional comma-separated retained family IDs for a focused follow-up')
    a=p.parse_args();problem=load_problem(a.knowns,a.trajectories);report=json.loads(a.report.read_text())
    if a.per_family<1 or a.workers<1 or a.seconds<=0:raise ValueError('positive fit budgets required')
    if a.output.exists():raise ValueError('output already exists')
    candidates=[]
    fixed=[f'{struct.unpack("<I",struct.pack("<f",float(x)))[0]:08x}' for x in a.fixed_literals.split(',') if x]
    for family,ids in report['leaderboards']['families'].items():
        if a.families and family not in a.families.split(','):continue
        seen=set();selected=0
        for ident in ids:
            candidate=dict(report['candidates'][ident],candidate_id=ident,_fixed_literal_bits=fixed)
            key=structure_key(candidate)
            if key in seen:continue
            seen.add(key);candidates.append(candidate);selected+=1
            if selected>=a.per_family:break
    # Preserve all source records, including the numerical address provenance.
    source=a.output.with_suffix('.sources.json');source.write_text(json.dumps(candidates,indent=2))
    provenance=dict(source_report_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.knowns,a.trajectories)},
        fit_policy=dict(per_family=a.per_family,seconds_per_fit=a.seconds,workers=a.workers,
            shortlist='associative_commutative_named_parameters',fixed_literals=a.fixed_literals))
    begin=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(fit_one,(c,problem['known_rhs'],problem['trajectories'][:16],a.seconds)) for c in candidates]
        for future in as_completed(tasks):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
            a.output.write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-begin,training_indices=list(range(min(16,len(problem['trajectories'])))),
                validation_consumed=False,results=results,**provenance),indent=2))


if __name__=='__main__':main()
