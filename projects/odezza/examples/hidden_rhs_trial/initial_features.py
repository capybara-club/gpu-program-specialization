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
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from features import term,catalogue,label
ROOT=Path(__file__).resolve().parent

def fitpair(X,y,d1,d2,p0):
 def fun(p,jac=False):
  f,g,*_=term(X,d1,p[1],True);h,q,*_=term(X,d2,p[3],True)
  return np.column_stack([f,p[0]*g,h,p[2]*q]) if jac else p[0]*f+p[2]*h-y
 a=np.array(p0);a[[0,2]]=np.clip(a[[0,2]],-.95,.95)
 r=least_squares(fun,a,jac=lambda p:fun(p,True),bounds=([-1,.65,-1,.65],[1,2.3,1,2.3]),max_nfev=35)
 return dict(terms=[d1,d2],parameters=r.x.tolist(),mse=float(np.mean(r.fun**2)),description=[label(d1),label(d2)])

if __name__=='__main__':
 start=time.perf_counter();data=np.load(ROOT/'proposals.npz');X=data['ic'];desc=[(d,w) for d in catalogue() for w in np.linspace(.7,2.2,21)];F=np.array([term(X,d,w) for d,w in desc]).T
 G=np.einsum('ki,kj->ij',F,F,optimize=False);diag=np.diag(G);den=diag[:,None]*diag[None,:]-G**2;bad=den<1e-8;den=np.maximum(den,1e-30);rows={}
 for method in ['d3','d4']:
  for j in range(2):
   y=data[method][:,0,j]+.35*X[:,2+j];b=np.einsum('ki,k->i',F,y,optimize=False)
   aa=(b[:,None]*diag[None,:]-b[None,:]*G)/den;bb=(b[None,:]*diag[:,None]-b[:,None]*G)/den
   err=(np.sum(y*y)-aa*b[:,None]-bb*b[None,:])/len(y);err[bad]=1e9;err[np.tril_indices(len(desc))]=1e9;err[(abs(aa)>1.2)|(abs(bb)>1.2)]=1e9
   best=np.argpartition(err.ravel(),12000)[:12000];best=best[np.argsort(err.ravel()[best])];seen=set();fits=[]
   for idx in best:
    a,c=np.unravel_index(idx,err.shape);d1,w1=desc[a];d2,w2=desc[c];key=(d1,d2)
    if key in seen or d1==d2:continue
    seen.add(key);fits.append(fitpair(X,y,d1,d2,[aa[a,c],w1,bb[a,c],w2]))
    if len(fits)>=150:break
   fits.sort(key=lambda r:r['mse']);rows[method+'_'+str(j+2)]=fits
   print(method,j+2,'best',json.dumps(fits[:8]),flush=True)
 out=dict(seconds=time.perf_counter()-start,feature_columns=len(desc),results=rows);(ROOT/'initial-features.json').write_text(json.dumps(out,indent=2));print('seconds',out['seconds'])
