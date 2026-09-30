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
"""CPU-only constrained interpolation. Initial slopes are proposals, not truth."""
import json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline,CubicSpline
from scipy.linalg import null_space
ROOT=Path(__file__).resolve().parent

def mm(a,b):return np.einsum('ij,jk->ik',a,b,optimize=False) if b.ndim==2 else np.einsum('ij,j->i',a,b,optimize=False)
def pinv(A,rcond=1e-12):
 U,S,V=np.linalg.svd(A,full_matrices=False);inv=np.where(S>S[0]*rcond,1/np.maximum(S,1e-300),0.);return mm(V.T*inv[None,:],U.T)
def load():
 d=json.loads((ROOT/'public.json').read_text());r=d['train'][:24];t=np.array(d['observation_times']);ic=np.array([q['t0'] for q in r]);obs=np.array([[q['observations']['x'+str(i)] for i in range(4)] for q in r]).transpose(0,2,1)
 return t,ic,obs

def operator(t,derivative=3):
 knots=np.r_[np.repeat(t[0],6),np.repeat(t[1:-1],3),np.repeat(t[-1],6)];n=len(knots)-6;basis=BSpline(knots,np.eye(n),5);B=basis(t);gx,gw=np.polynomial.legendre.leggauss(28);weighted=[];E=[]
 for lo,hi in zip(t[:-1],t[1:]):
  h=hi-lo;tt=(lo+hi)/2+h*gx/2;tau=hi-tt;c=np.exp(-.2*tau)*np.cos(.8*tau);s=np.exp(-.2*tau)*np.sin(.8*tau);bb=basis(tt)*(.35*h/2*gw)[:,None]
  weighted.append(np.r_[mm(c[None,:],bb)[0],-mm(s[None,:],bb)[0]]);weighted.append(np.r_[mm(s[None,:],bb)[0],mm(c[None,:],bb)[0]])
  E.append(np.exp(-.2*h)*np.array([[np.cos(.8*h),-np.sin(.8*h)],[np.sin(.8*h),np.cos(.8*h)]]))
 A=np.r_[np.block([[B,np.zeros_like(B)],[np.zeros_like(B),B]]),np.array(weighted)];scale=np.linalg.norm(A,axis=1);AA=A/scale[:,None];pin=pinv(AA,rcond=1e-12);N=null_space(AA,rcond=1e-12);rr=basis(np.linspace(0,t[-1],501),nu=derivative);R=np.block([[rr,np.zeros_like(rr)],[np.zeros_like(rr),rr]]);mapcoef=pin-mm(N,mm(pinv(mm(R,N),rcond=1e-12),mm(R,pin)))
 return basis,mapcoef/scale[None,:],A,np.array(E)

if __name__=='__main__':
 start=time.perf_counter();t,ic,obs=load();out=[];err=[]
 for deriv in [3,4]:
  basis,P,A,E=operator(t,deriv);n=basis.c.shape[0];cc=[];dd=[]
  for o in obs:
   endpoint=o[1:,:2]-np.einsum('tij,tj->ti',E,o[:-1,:2],optimize=False);target=np.r_[o[:,2],o[:,3],endpoint.ravel()];c=mm(P,target);cc.append(c);err.append(float(np.max(abs(mm(A,c)-target))));dd.append(np.column_stack([mm(basis(t,1),c[:n]),mm(basis(t,1),c[n:])]))
  out.append(np.array(dd));np.savez(ROOT/f'splines{deriv}.npz',knots=basis.t,coefs=cc)
 cubic=np.array([CubicSpline(t,o,axis=0)(t,1)[:,2:] for o in obs]);np.savez(ROOT/'proposals.npz',ic=ic,t=t,obs=obs,d3=out[0],d4=out[1],cubic=cubic)
 meta=dict(seconds=time.perf_counter()-start,max_known_constraint_error=max(err),initial_slope_d3_d4_rms=np.sqrt(np.mean((out[0][:,0]-out[1][:,0])**2,axis=0)).tolist(),initial_slope_d3_cubic_rms=np.sqrt(np.mean((out[0][:,0]-cubic[:,0])**2,axis=0)).tolist())
 (ROOT/'preparation.json').write_text(json.dumps(meta,indent=2));print(meta)
