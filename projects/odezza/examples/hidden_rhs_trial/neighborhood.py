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
"""Bounded automatic structural edits with GPU scoring and CPU promotion."""
import json,time
import numpy as np
from prepare import ROOT,load
from common import descriptors,batch
from features import catalogue,label
from native import make_base,rng,run,decode

def best_available():
 rows=[]
 for f in list(ROOT.glob('*fit*.json')):
  if f.name.endswith('-jobs.json'):continue
  
  try:d=json.loads(f.read_text())
  except (json.JSONDecodeError,OSError):continue
  rows.extend(r for r in d.get('results',[]) if r.get('best'))
 return min(rows,key=lambda r:r['best']['mse'])
def request(base,iteration):
 ds=descriptors(base['expression']);p=base['best']['parameters'];q=make_base(rows=16384,points=6,seed=282000+iteration);g=q['grammar'];g['expansion']['max_nodes']=128;g['retain']={'global':{'k':24,'unit':'resolved_structure'},'per_family':{'k':4,'unit':'resolved_structure'}}
 for slot in range(8):
  aa=rng(g,f'a{slot}',-1,1);ww=rng(g,f'w{slot}',.65,2.3);rule=f'E{slot}';g['rules'][rule]=[aa+'*('+label(d).replace('w',ww)+')' for d in catalogue()];rhs={}
  for state in range(2,6):
   terms=[]
   for j in range(2):
    k=(state-2)*2+j
    terms.append(rule if k==slot else repr(float(p[2*k]))+'*('+label(ds[k]).replace('w',repr(float(p[2*k+1])))+')')
   rhs[f'x{state}']=f'-.35*x{state}+'+'+'.join(terms)
  g['families'].append(dict(id=f'edit_{slot}',rhs=rhs,tags=[f'slot:{slot}',f'round:{iteration}'],limits=dict(max_skeletons=105,max_variants=105,max_configurations=105*16384,max_derivations=10000,max_expansion_steps=1000000)))
 return q

def fit_screen(r,name):
 ids=r['leaderboards']['global'][:4];families=list(r['leaderboards']['families'].values())
 for j in range(2):
  for v in families:ids.extend(v[j:j+1])
 jobs=[];seen=set();t,ic,obs=load()
 for ident in ids:
  c=r['candidates'][ident];e,p=decode(c);key=tuple(e)
  if key in seen:continue
  seen.add(key);p=np.clip(p,np.tile([-1,.65],8)+1e-7,np.tile([1,2.3],8)-1e-7);jobs.append((e,p,t,ic,obs,10,dict(candidate=ident,origin=c['origin'],screen_mse=c['mse'])))
  if len(jobs)>=16:break
 return batch(jobs,name,workers=4)
if __name__=='__main__':
 history=[]
 for iteration in range(6):
  base=best_available();m=base['best']['mse']
  if m<1e-15:break
  print('BASE',iteration,m,flush=True);(ROOT/f'edit-parent-{iteration}.json').write_text(json.dumps(base,indent=2));r=run(f'edit-screen-{iteration}',request(base,iteration));fit_screen(r,f'edit-fit-{iteration}');best=best_available();history.append(dict(iteration=iteration,parent_mse=m,best=best,at_unix=time.time()));(ROOT/'edit-history.json').write_text(json.dumps(history,indent=2));print('BEST',iteration,best['best']['mse'],flush=True)
  if best['best']['mse']>=m*(1-1e-6):break
