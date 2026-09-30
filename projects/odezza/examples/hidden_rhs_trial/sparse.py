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
import json,sys,time
import numpy as np
from scipy.optimize import minimize
from adjoint_library import Library
from prepare import ROOT,load
from common import build
if __name__=='__main__':
 variant=int(sys.argv[1]);t,ic,obs=load();pro=json.loads((ROOT/'initial-features.json').read_text())['results'];rows=[pro['d4_2'][variant],pro['d4_3'][0]]+[dict(terms=[(0,0,1),(1,2,3)],parameters=[0.,1.4,0.,1.4])]*2
 e,p=build(rows);base=dict(expression=e,best=dict(parameters=p));lib=Library(base,t,ic,obs);q=np.r_[p,np.zeros(lib.K*2)];bounds=[(-1,1),(.65,2.3)]*4+[(float(v),float(v)) for v in q[8:16]]+[(-1,1)]*(lib.K*2);history=[];start=time.perf_counter()
 def objective(q,penalty):
  mse,g=lib.evaluate(q);A=q[16:].reshape(-1,3,2);norm=np.sqrt(np.sum(A*A,axis=1)+1e-8);loss=mse+penalty*np.sum(norm-1e-4);g[16:]+=(penalty*A/norm[:,None,:]).ravel();history.append(dict(call=len(history)+1,mse=mse,objective=float(loss),seconds=time.perf_counter()-start))
  if len(history)%20==0:print(variant,history[-1],flush=True)
  return loss,g
 for stage,penalty in enumerate([5e-5,2e-5]):
  r=minimize(lambda q:objective(q,penalty),q,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=180,maxfun=220,ftol=2e-13,gtol=1e-7,maxls=15,maxcor=20));q=r.x
  out=dict(variant=variant,stage=stage,base=base,descriptors=lib.desc,parameters=q.tolist(),history=history,penalty=penalty,result=dict(success=bool(r.success),message=str(r.message),iterations=int(r.nit),calls=int(r.nfev)),proposal_only=True,training_indices=list(range(24)));(ROOT/f'adjoint-sparse-{variant}-{stage}.json').write_text(json.dumps(out,indent=2));print('STAGE',variant,stage,r.fun,flush=True)
