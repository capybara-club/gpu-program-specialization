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
"""CPU-only adjoint-assisted library proposals for hidden dynamics.

The dense library is an auxiliary proposal model, not a valid two-term solution.
"""
import json,sys,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from features import catalogue
from prepare import ROOT,load
from engine import compile_model
from mutate import descriptors

class Library:
 def __init__(self,base,t,ic,obs):
  self.base=base;self.model=compile_model(base['expression'],16);self.t=t;self.ic=ic;self.obs=obs;self.N=len(ic);self.K=len(catalogue())*3;self.desc=[(d,w) for d in catalogue() for w in [.8,1.4,2.]];a=np.array([d for d,w in self.desc]);self.kind=a[:,0];self.u=a[:,1];self.v=a[:,2];self.w=np.array([w for d,w in self.desc]);self.last=None;self.cache=None;self.calls=0;self.started=time.perf_counter();self.history=[]
 def features(self,X,grad=False):
  U=X[:,self.u];V=X[:,self.v];w=self.w[None,:];F=np.zeros_like(U);GU=np.zeros_like(U);GV=np.zeros_like(U)
  for k in range(4):
   z=self.kind==k;u=U[:,z];v=V[:,z];ww=w[:,z]
   if k==0:
    a=ww*u*v;F[:,z]=np.sin(a)
    if grad:GU[:,z]=np.cos(a)*ww*v;GV[:,z]=np.cos(a)*ww*u
   elif k==1:
    a=ww*u+v;F[:,z]=np.sin(a)
    if grad:GU[:,z]=np.cos(a)*ww;GV[:,z]=np.cos(a)
   elif k==2:
    a=ww*u*np.sin(v);F[:,z]=np.sin(a)
    if grad:GU[:,z]=np.cos(a)*ww*np.sin(v);GV[:,z]=np.cos(a)*ww*u*np.cos(v)
   else:
    F[:,z]=np.sin(ww*u)*np.sin(v)
    if grad:GU[:,z]=np.cos(ww*u)*ww*np.sin(v);GV[:,z]=np.sin(ww*u)*np.cos(v)
  return (F,GU,GV) if grad else F
 def evaluate(self,q,gradient=True):
  p=q[:16];A=q[16:].reshape(self.K,2)
  def fun(t,flat):
   X=flat.reshape(self.N,6);f,_=self.model(X,p);F=self.features(X);f[:,4:]+=np.einsum('nf,fr->nr',F,A,optimize=False);return f.ravel()
  sol=solve_ivp(fun,(0,self.t[-1]),self.ic.ravel(),dense_output=True,rtol=2e-7,atol=2e-9)
  if not sol.success:raise ValueError(sol.message)
  pred=sol.sol(self.t).reshape(self.N,6,len(self.t)).transpose(0,2,1);err=pred[:,1:,:4]-self.obs[:,1:];mse=float(np.mean(err**2))
  if not gradient:return mse
  z=np.zeros(self.N*6+len(q));norm=err.size
  def back(t,z):
   X=sol.sol(t).reshape(self.N,6);L=z[:self.N*6].reshape(self.N,6);_,g=self.model(X,p);J=g[:,:,:6].copy();F,GU,GV=self.features(X,True)
   for state in range(6):
    u=self.u==state;v=self.v==state
    J[:,4:,state]+=np.einsum('nf,fr->nr',GU[:,u],A[u],optimize=False)+np.einsum('nf,fr->nr',GV[:,v],A[v],optimize=False)
   dl=-np.einsum('nij,ni->nj',J,L,optimize=False);gp=-np.einsum('nip,ni->p',g[:,:,6:],L,optimize=False);ga=-np.einsum('nf,nr->fr',F,L[:,4:],optimize=False)
   return np.r_[dl.ravel(),gp,ga.ravel()]
  for j in range(len(self.t)-1,0,-1):
   z[:self.N*6].reshape(self.N,6)[:,:4]+=2*err[:,j-1]/norm
   b=solve_ivp(back,(self.t[j],self.t[j-1]),z,rtol=5e-7,atol=2e-9)
   if not b.success:raise ValueError(b.message)
   z=b.y[:,-1]
  return mse,z[self.N*6:]
 def objective(self,q,penalty):
  mse,g=self.evaluate(q);ix=np.r_[np.arange(8,16,2),np.arange(16,len(q))];v=q[ix];smooth=np.sqrt(v*v+1e-8);loss=mse+penalty*np.sum(smooth-1e-4);g[ix]+=penalty*v/smooth;self.calls+=1
  row=dict(call=self.calls,mse=mse,objective=float(loss),penalty=penalty,seconds=time.perf_counter()-self.started);self.history.append(row)
  if self.calls%10==0:print(json.dumps(row),flush=True)
  return loss,g

if __name__=='__main__':
 t,ic,obs=load();hist=json.loads((ROOT/'mutation-history.json').read_text());base=hist[-1]['best'];lib=Library(base,t,ic[::2],obs[::2]);q=np.r_[base['best']['parameters'],np.zeros(lib.K*2)];m,g=lib.evaluate(q);checks=[]
 for i in [0,3,8,15,20,79,222,600]:
  d=np.zeros_like(q);d[i]=2e-5;fd=(lib.evaluate(q+d,False)-lib.evaluate(q-d,False))/(4e-5);checks.append(dict(index=i,adjoint=float(g[i]),finite_difference=float(fd),error=float(abs(fd-g[i]))))
 err=max(c['error'] for c in checks);assert err<2e-6,err;(ROOT/'adjoint-check.json').write_text(json.dumps(dict(mse=m,max_error=err,checks=checks),indent=2));print('ADJOINT',err,m,flush=True)
 bounds=[(-1,1),(.65,2.3)]*8+[(-1,1)]*(lib.K*2)
 for stage,penalty in enumerate([2e-5,1e-4]):
  r=minimize(lambda q:lib.objective(q,penalty),q,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=90,maxfun=130,ftol=1e-12,gtol=1e-7,maxls=15));q=r.x;out=dict(stage=stage,penalty=penalty,base=base,descriptors=lib.desc,parameters=q.tolist(),history=lib.history,result=dict(success=bool(r.success),message=str(r.message),iterations=int(r.nit),function_calls=int(r.nfev)),proposal_only=True,training_indices=list(range(0,24,2)))
  (ROOT/f'adjoint-library-{stage}.json').write_text(json.dumps(out,indent=2));print('STAGE',stage,str(r.message),float(r.fun),flush=True)
