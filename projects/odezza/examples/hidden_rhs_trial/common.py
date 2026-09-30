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
import json,time,concurrent.futures
import numpy as np
from prepare import ROOT,load
from features import label,catalogue
from engine import fit

def build(rows):
 e=['-.2*x0-.8*x1+.35*x2','.8*x0-.2*x1+.35*x3'];p=[]
 for i,r in enumerate(rows,2):
  parts=[]
  for d,c in zip(r['terms'],[r['parameters'][:2],r['parameters'][2:]]):
   n=len(p);parts.append(f'p{n}*('+label(d).replace('w',f'p{n+1}')+')');p.extend(c)
  e.append(f'-.35*x{i}+'+'+'.join(parts))
 return e,p

def expression(ds):return build([dict(terms=ds[j:j+2],parameters=[.4,1.4,.4,1.4]) for j in range(0,8,2)])[0]
def descriptors(e):
 out=[]
 for s in range(8):
  z=[d for d in catalogue() if f'p{2*s}*('+label(d).replace('w',f'p{2*s+1}')+')' in e[2+s//2]]
  assert len(z)==1,(s,z,e);out.append(z[0])
 return out

def batch(jobs,name,workers=6):
 start=time.perf_counter();res=[]
 (ROOT/(name+'-jobs.json')).write_text(json.dumps([dict(expression=j[0],parameters=np.asarray(j[1]).tolist(),seconds=j[-2],id=j[-1]) for j in jobs],indent=2))
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
  fs=[pool.submit(fit,j) for j in jobs]
  for f in concurrent.futures.as_completed(fs):
   r=f.result();res.append(r);(ROOT/(name+'.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-start,results=res),indent=2));print(name,len(res),r['best']['mse'] if r.get('best') else r.get('error'),flush=True)
 return res
