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
"""Compile fixed ordinary CUDA RHS models and measure cooperative fitting.

No specialization library, patch-site generation or runtime AST evaluator is used.
Run with the repository's existing SciPy Python and installed NVRTC compiler.
"""
import argparse
import ast
import json
from pathlib import Path
import random
import shutil
import struct
import sys
import numpy as np
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'assignment_search'))
from structures import OPS,encode
from verify import expression,verify
import game_input as game
from run_gpu import run
from pipeline_runner.cuda import CudaOwner

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def parse(text):
    def visit(n):
        if isinstance(n,ast.Name):return ('state' if n.id[0]=='x' else 'parameter',int(n.id[1:]))
        if isinstance(n,ast.Constant):return ('constant',float(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return ('sub',('constant',0.),visit(n.operand))
        if isinstance(n,ast.BinOp):return ({ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div'}[type(n.op)],visit(n.left),visit(n.right))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('sin','cos','exp','tanh') and len(n.args)==1:return(n.func.id,visit(n.args[0]))
        raise ValueError('unsupported fixed expression')
    return visit(ast.parse(text,mode='eval').body)

def encoded(t):
    if t[0]=='state':return (0x81,t[1])
    if t[0]=='parameter':return (0x82,t[1])
    if t[0]=='constant':return (0x83,struct.unpack('<I',struct.pack('<f',t[1]))[0])
    return (OPS[t[0]][0],*(encoded(c) for c in t[1:]))
def code(t):return (encode(encoded(t))+b'\x80').hex()
def decode(hexcode):
    # The reference decoder preserves arbitrary parameter objects as constants.
    def transform(t):
        if t[0]=='constant' and isinstance(t[1],tuple):return t[1]
        if t[0] in ('constant','state'):return t
        return (t[0],*(transform(c) for c in t[1:]))
    return transform(expression(hexcode,[('parameter',i) for i in range(255)]))

def cexpr(t):
    if t[0]=='state':return f'x[{t[1]}]'
    if t[0]=='parameter':return f'p[{t[1]}]'
    if t[0]=='constant':return f'Dual({float(t[1]).hex()}f)'
    if t[0] in ('add','sub','mul','div'):return '('+cexpr(t[1])+{'add':'+','sub':'-','mul':'*','div':'/'}[t[0]]+cexpr(t[2])+')'
    return t[0]+'('+cexpr(t[1])+')'

def header(case):
    lines=['#pragma once','#include "dual.cuh"',f'#define MODEL_STATES {case["n"]}',f'#define MODEL_PARAMETERS {case["p"]}',f'#define MODEL_VARIANTS {len(case["variants"])}',
           '__device__ __forceinline__ void model_rhs(int variant,const Dual*x,const Dual*p,Dual*f){']
    lines += [f'f[{i}]={cexpr(decode(c))};' for i,c in enumerate(case['fixed'])]
    lines+=['switch(variant){']+[f'case {i}: f[MODEL_STATES-1]={cexpr(decode(c))};break;' for i,c in enumerate(case['variants'])]
    return '\n'.join(lines+['}','}'])+'''
extern "C" __global__ void model_jacobian(const float *values, float *out) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=MODEL_VARIANTS*(MODEL_STATES+MODEL_PARAMETERS))return;
    int variant=q/(MODEL_STATES+MODEL_PARAMETERS), direction=q%(MODEL_STATES+MODEL_PARAMETERS);
    Dual x[MODEL_STATES], p[MODEL_PARAMETERS], f[MODEL_STATES];
    for(int i=0;i<MODEL_STATES;++i)x[i]=Dual(values[i],direction==i?1.f:0.f);
    for(int i=0;i<MODEL_PARAMETERS;++i)p[i]=Dual(values[MODEL_STATES+i],direction==MODEL_STATES+i?1.f:0.f);
    model_rhs(variant,x,p,f);
    for(int i=0;i<MODEL_STATES;++i){out[q*MODEL_STATES*2+i]=f[i].v;out[q*MODEL_STATES*2+MODEL_STATES+i]=f[i].d;}
}
'''

def probe_input(case):
    # Independent central differences in FP64; never uses generation truth.
    rng=np.random.default_rng(81426);z=rng.uniform(.2,.7,case['n']+case['p']).astype(np.float32).astype(float);expected=[]
    def rhs(z,v):
        expr=[expression(c,z[case['n']:]) for c in case['fixed']+[case['variants'][v]]]
        return np.array([game.evaluate(t,z[:case['n']]) for t in expr])
    for v in range(len(case['variants'])):
        for j in range(len(z)):
            lo=z.copy();hi=z.copy();lo[j]-=1e-5;hi[j]+=1e-5
            expected.extend(rhs(z,v).tolist()+((rhs(hi,v)-rhs(lo,v))/2e-5).tolist())
    return z.tolist(),expected

def synthetic(n,parameters=6):
    if n<3 or parameters not in (3,6,8):raise ValueError('Control requires >=3 states and 3, 6 or 8 coefficients')
    fixed=[code(parse(f'-0.6*x{i}+0.06*sin(x{(i+1)%n})')) for i in range(n-1)]
    formula=f'(p0*sin(x0+p1*x2)+p2*cos(p3*x{3%n})+p4*x{4%n})/(1+x{5%n}*x{5%n})-p5*x{n-1}'
    truth=[-.4,.7,.3,-.6,.2,.55]
    if parameters==8:
        formula+=f'+p6*sin(x1)+p7*x0'
        truth += [.12,-.17]
    if parameters==3:
        formula=formula.replace('p1','0.7').replace('p3','(-0.6)').replace('p4','0.2').replace('p2','p1').replace('p5','p2')
        truth=[-.4,.3,.55]
    variant=code(parse(formula))
    truth=[struct.unpack('<f',struct.pack('<f',x))[0] for x in truth]
    expr=[expression(c,truth) for c in fixed+[variant]];rng=np.random.default_rng(6090926+n);rows=[]
    max_delta=0.
    for _ in range(8):
        ic=rng.uniform(-.7,.7,n);times=np.linspace(0,1,21)
        rhs=lambda t,y:[game.evaluate(e,y) for e in expr]
        a=solve_ivp(rhs,(0,1),ic,t_eval=times,method='DOP853',rtol=1e-11,atol=1e-13)
        b=solve_ivp(rhs,(0,1),ic,t_eval=times,method='Radau',rtol=1e-11,atol=1e-13)
        if not a.success or not b.success:raise ValueError('generation failed')
        max_delta=max(max_delta,float(np.max(np.abs(a.y-b.y))))
        rows.append(dict(initial_state=ic.tolist(),times=times.tolist(),states=a.y.T.tolist()))
    if max_delta>1e-9:raise ValueError('generation disagreement')
    return dict(name=f'nonlinear-n{n}-p{parameters}',n=n,p=parameters,fixed=fixed,variants=[variant],rows=rows[:4],validation=rows[4:6],
                reserved_test=rows[6:],reference_difference=max_delta,truth_for_calibration_only=truth)

def input_file(case,folder,count):
    rows=case['rows'];offsets=[0];times=[]
    for r in rows:times+=r['times'];offsets.append(len(times))
    points=len(times);ref=[];weights=[]
    for s in range(case['n']):
        for r in rows:
            for i,state in enumerate(r['states']):
                ref.append(r['initial_state'][s] if i==0 else 0. if state[s] is None else state[s])
                weights.append(0. if i==0 or state[s] is None else 1.)
    rng=random.Random(19092026);base=case.get('base',[.05]*case['p'])
    starts=[[max(-1.5,min(1.5,x+(0 if i<len(case['variants']) else rng.uniform(-.4,.4)))) for x in base] for i in range(count)]
    with (folder/'input.bin').open('wb') as f:
        f.write(struct.pack('<7I',count,len(rows),points,32,32,case['n'],case['p']))
        f.write(struct.pack('<'+'I'*len(offsets),*offsets))
        for values in (times,ref,weights,[x for row in starts for x in row]):f.write(struct.pack('<'+'f'*len(values),*values))
    save(folder/'starts.json',starts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--widths',type=int,nargs='+',default=[1,2,4,8]);p.add_argument('--count',type=int,default=67)
    p.add_argument('--parameters',type=int,nargs='+',choices=[3,6,8],default=[6])
    p.add_argument('--cases',nargs='+',default=['saved8','12','16']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cases=[]
    if 'saved8' in a.cases:
        name='n8-d3-p6-r1-dense';req=json.loads((a.inputs/'fit'/f'{name}.json').read_text());val=json.loads((a.inputs/'validation'/f'{name}.json').read_text())
        cases.append(dict(name=name,n=8,p=6,fixed=val['fixed_rhs'],variants=[c['program_hex'] for c in req['candidates']],rows=req['data']['trajectories'],validation=val['rows'],base=req['candidates'][0]['parameter_rows'][0]))
    for n in (3,6,8,12,16):
        if str(n) in a.cases:
            for parameters in a.parameters:cases.append(synthetic(n,parameters))
    summary=[]
    for case in cases:
        folder=a.out/case['name'];folder.mkdir();save(folder/'case.json',case);(folder/'model.cuh').write_text(header(case));input_file(case,folder,a.count)
        for name in ('dual.cuh','kernel.cuh'):shutil.copy2(HERE/name,folder/name)
        baseline=None
        for width in a.widths:
            print(json.dumps(dict(case=case['name'],width=width,stage='compile')),flush=True)
            source=f'#define ODEZZA_FIT_THREAD_COUNT {width}\n'+(folder/'kernel.cuh').read_text().replace('#include "model.cuh"',header(case).replace('#include "dual.cuh"',(HERE/'dual.cuh').read_text())).replace('#pragma once','')
            (folder/f'kernel-w{width}.cu').write_text(source)
            with CudaOwner() as owner:
                streams=owner.stream_pool(1)
                result,compiler=run(source,folder,width,probe_input(case),owner=owner,streams=streams)
            save(folder/f'compiler-w{width}.json',compiler);save(folder/f'gpu-w{width}.json',result)
            compile_seconds=compiler['seconds']
            best=min(range(a.count),key=lambda i:result['mse'][i]);params=result['coefficients'][best*case['p']:(best+1)*case['p']]
            candidate=dict(rhs=case['fixed']+[case['variants'][best%len(case['variants'])]],parameter_values=params,target=1e-10)
            checks={}
            for label,rows in [('training',case['rows']),('validation',case['validation'])]:
                try:checks[label]=verify(dict(candidate,rows=rows))
                except (ValueError,OverflowError,ZeroDivisionError,FloatingPointError) as e:checks[label]=dict(passed=False,error=str(e))
            delta=None
            if baseline:
                delta=dict(max_initial_mse_difference=max(abs(x-y) for x,y in zip(result['initial_mse'],baseline['initial_mse'])),
                           max_final_mse_difference=max(abs(x-y) for x,y in zip(result['mse'],baseline['mse'])),
                           max_parameter_difference=max(abs(x-y) for x,y in zip(result['coefficients'],baseline['coefficients'])),
                           same_iterations=result['iterations']==baseline['iterations'],same_accepted=result['accepted']==baseline['accepted'])
            else:baseline=result
            row=dict(case=case['name'],width=width,count=a.count,compile_seconds=compile_seconds,
                     **{k:result[k] for k in ['registers','local_bytes','shared_bytes','active_blocks_per_sm','kernel_ms','launch_wait_seconds','derivative_probe','module_load_seconds','resources']},
                     best_index=best,best_mse=result['mse'][best],best_parameters=params,checks=checks,comparison_first_width=delta)
            summary.append(row);save(a.out/'results.json',summary);print(json.dumps(row),flush=True)
    save(a.out/'complete.json',dict(complete=True,cells=len(summary)))

if __name__=='__main__':main()
