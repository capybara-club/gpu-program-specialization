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
import json,sys,itertools,time
import numpy as np
from scipy.integrate import solve_ivp
from adjoint_library import Library
from features import term,label,catalogue
from initial_features import fitpair
from prepare import ROOT,load
from common import descriptors
from native import make_base,rng,run,fit_report

def proposals(source):
 d=json.loads(source.read_text());t,ic,obs=load();lib=Library(d['base'],t,ic,obs);q=np.array(d['parameters']);p=q[:16];A=q[16:].reshape(-1,2)
 def fun(t,z):
  X=z.reshape(-1,6);f,_=lib.model(X,p);f[:,4:]+=np.einsum('nf,fr->nr',lib.features(X),A,optimize=False);return f.ravel()
 s=solve_ivp(fun,(0,t[-1]),ic.ravel(),t_eval=t,rtol=1e-8,atol=1e-10);X=s.y.T.reshape(-1,6);F=lib.features(X);Y=np.einsum('nf,fr->nr',F,A,optimize=False);cat=catalogue();groups=A.reshape(-1,3,2);out=[]
 for state in range(2):
  mag=np.sqrt(np.sum(groups[:,:,state]**2,axis=1));top=np.argsort(mag)[-12:][::-1];rows=[]
  for a,b in itertools.combinations(top,2):
   wa=np.array([.8,1.4,2.]);aa=groups[a,:,state].sum();bb=groups[b,:,state].sum();w1=np.sum(wa*abs(groups[a,:,state]))/max(1e-10,sum(abs(groups[a,:,state])));w2=np.sum(wa*abs(groups[b,:,state]))/max(1e-10,sum(abs(groups[b,:,state])));r=fitpair(X,Y[:,state],cat[a],cat[b],[aa,w1,bb,w2]);rows.append(r)
  rows.sort(key=lambda r:r['mse']);out.append(rows[:16]);print('hidden',state+4,rows[:3],flush=True)
 result=dict(source=source.name,observed_descriptors=descriptors(d['base']['expression'])[:4],observed_parameters=p[:8].tolist(),hidden_pairs=out,proposal_only=True);(ROOT/(source.stem+'-pairs.json')).write_text(json.dumps(result,indent=2));return result

def request(pairs,rows=131072):
 q=make_base(rows=rows,points=6,seed=271114);g=q['grammar'];g['expansion']['max_nodes']=128;g['retain']['global']['k']=32;g['retain']['per_family']['k']=8;rhs={}
 for state in [2,3]:
  parts=[]
  for j in range(2):
   slot=(state-2)*2+j;d=pairs['observed_descriptors'][slot];a,w=pairs['observed_parameters'][2*slot:2*slot+2];aa=rng(g,f'o{slot}a',max(-1,a-.02),min(1,a+.02));ww=rng(g,f'o{slot}w',max(.65,w-.04),min(2.3,w+.04));parts.append(aa+'*('+label(d).replace('w',ww)+')')
  rhs[f'x{state}']=f'-.35*x{state}+'+'+'.join(parts)
 # Four families each allocate a disjoint quarter of the pair cross-product.
 for family in range(4):
  rr=rhs.copy()
  for state in [4,5]:
   options=[];choices=pairs['hidden_pairs'][state-4][(family*4 if state==4 else 0):(family*4+4 if state==4 else 16)]
   for i,r in enumerate(choices):
    parts=[]
    for j,d in enumerate(r['terms']):
     a,w=r['parameters'][2*j:2*j+2];a=np.clip(a,-.97,.97);w=np.clip(w,.68,2.27);aa=rng(g,f'f{family}h{state}r{i}a{j}',max(-1,a-.12),min(1,a+.12));ww=rng(g,f'f{family}h{state}r{i}w{j}',max(.65,w-.20),min(2.3,w+.20));parts.append(aa+'*('+label(d).replace('w',ww)+')')
    options.append('+'.join(parts))
   name=f'H{state}f{family}';g['rules'][name]=options;rr[f'x{state}']=f'-.35*x{state}+{name}'
  g['families'].append(dict(id=f'pairs_{family}',rhs=rr,tags=[f'library:{pairs["source"]}'],limits=dict(max_skeletons=64,max_variants=64,max_configurations=64*rows,max_derivations=10000,max_expansion_steps=10000000)))
 return q
if __name__=='__main__':
 if (ROOT/'SOLVED.json').exists():
  print('Skipped: validation already succeeded',flush=True);raise SystemExit(0)
 variant=int(sys.argv[1]);stage=int(sys.argv[2]) if len(sys.argv)>2 else 0;source=ROOT/f'adjoint-sparse-{variant}-{stage}.json';p=proposals(source);report=run(f'focus-{variant}-{stage}',request(p));fit_report(report,f'focus-fit-{variant}-{stage}',32)
