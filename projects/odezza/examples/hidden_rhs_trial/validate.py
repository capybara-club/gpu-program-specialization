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
"""Freeze on training, refine on training, independently validate untouched ICs."""
import json,time,hashlib,datetime
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from prepare import ROOT,load
from engine import compile_model,simulate
from common import descriptors
START=datetime.datetime.fromisoformat(json.loads((ROOT/'experiment.json').read_text())['started_utc']).timestamp()
DS=None
def rhs(t,x,p):
 import math
 out=np.array([-.2*x[0]-.8*x[1]+.35*x[2],.8*x[0]-.2*x[1]+.35*x[3],-.35*x[2],-.35*x[3],-.35*x[4],-.35*x[5]])
 for slot,(kind,u,v) in enumerate(DS):
  a,w=p[2*slot:2*slot+2];U,V=float(x[u]),float(x[v])
  if kind==0:f=math.sin(w*U*V)
  elif kind==1:f=math.sin(w*U+V)
  elif kind==2:f=math.sin(w*U*math.sin(V))
  elif kind==3:f=math.sin(w*U)*math.sin(V)
  else:raise ValueError(kind)
  out[2+slot//2]+=a*f
 return out

def score(records,p,t):
 predictions=[];rows=[]
 for r in records:
  sol=solve_ivp(lambda t,x:rhs(t,x,p),(0,t[-1]),r['t0'],t_eval=t,method='DOP853',rtol=2.3e-14,atol=2e-15)
  assert sol.success;pred=sol.y.T;predictions.append(dict(trajectory_id=r.get('trajectory_id',r.get('id')),times=t.tolist(),states={f'x{i}':pred[:,i].tolist() for i in range(6)}))
  if 'observations' in r:
   obs=np.array([r['observations'][f'x{i}'] for i in range(4)]).T;err=pred[1:,:4]-obs[1:];rows.append(dict(id=r.get('trajectory_id',r.get('id')),mse=float(np.mean(err**2)),max_absolute_error=float(np.max(abs(err))),state_mse=np.mean(err**2,axis=0).tolist()))
 return predictions,rows

if __name__=='__main__':
 begin=time.perf_counter();from neighborhood import best_available;best=best_available();assert best['best']['mse']<1e-15;DS=descriptors(best['expression'])
 (ROOT/'frozen-training-model.json').write_text(json.dumps(dict(frozen_at_unix=time.time(),model=best),indent=2));t,ic,obs=load();e=best['expression'];p=np.array(best['best']['parameters']);model=compile_model(e,16);last=None;cached=None;calls=0
 def evaluate(p):
  global last,cached,calls
  if last is None or not np.array_equal(p,last):cached=simulate(model,p,t,ic,obs,rtol=2.5e-13,atol=2e-15);last=p.copy();calls+=1
  return cached
 lower=np.tile([.25,.7],8);upper=np.tile([.8,2.2],8)
 for i in range(0,16,2):
  if p[i]<0:lower[i],upper[i]=-.8,-.25
 r=least_squares(lambda p:evaluate(p)[0],p,jac=lambda p:evaluate(p)[1],bounds=(lower,upper),x_scale='jac',max_nfev=30,ftol=1e-13,gtol=1e-13,xtol=1e-13);p=r.x;precision=dict(seconds=time.perf_counter()-begin,calls=calls,mse=float(np.mean(r.fun**2)),parameters=p.tolist(),expression=e,descriptors=descriptors(e),training_indices=list(range(24)))
 (ROOT/'precision.json').write_text(json.dumps(precision,indent=2))
 # Independent independent scalar descriptor evaluator against generic model at random states.
 X=np.random.default_rng(282828).uniform(-1.5,1.5,(128,6));manual=np.array([rhs(0,x,p) for x in X]);generic=model(X,p)[0];agree=float(np.max(abs(manual-generic)));assert agree<1e-13
 j=evaluate(p)[1];fd=[]
 for k in range(16):
  dp=np.zeros(16);dp[k]=1e-5;rp=simulate(model,p+dp,t[:2],ic[:2],obs[:2,:2],rtol=1e-11,atol=1e-13)[0];rm=simulate(model,p-dp,t[:2],ic[:2],obs[:2,:2],rtol=1e-11,atol=1e-13)[0];fd.append((rp-rm)/2e-5)
 _,jj=simulate(model,p,t[:2],ic[:2],obs[:2,:2],rtol=1e-11,atol=1e-13);jacerr=float(np.max(abs(jj-np.array(fd).T)));assert jacerr<1e-7
 d=json.loads((ROOT/'public.json').read_text());pred,rows=score(d['train'],p,t);hold,_=score(d['holdout'],p,t)
 results=dict(validated_at_unix=time.time(),time_to_first_accurate_seconds=best['best']['at_unix']-START,time_to_validated_seconds=time.time()-START,training_mse=float(np.mean([r['mse'] for r in rows[:24]])),internal_validation_mse=float(np.mean([r['mse'] for r in rows[24:]])),all_supplied_mse=float(np.mean([r['mse'] for r in rows])),max_absolute_error=max(r['max_absolute_error'] for r in rows),manual_rhs_agreement=agree,jacobian_max_absolute_error=jacerr,per_trajectory=rows,training_indices=list(range(24)),internal_validation_indices=list(range(24,32)),official_holdout_observations_available=False,parameters=p.tolist(),input_sha256=hashlib.sha256((ROOT/'public.json').read_bytes()).hexdigest())
 (ROOT/'validation.json').write_text(json.dumps(results,indent=2));(ROOT/'predictions.json').write_text(json.dumps(dict(supplied=pred,official_holdout=hold),indent=2));eq=[]
 for ex in e:
  import re
  eq.append(re.sub(r'\bp(\d+)\b',lambda m:repr(float(p[int(m[1])])),ex))
 (ROOT/'submission.json').write_text(json.dumps(dict(challenge_id=d['challenge_id'],rhs={f'x{i}':ex for i,ex in enumerate(eq)},parameters=p.tolist(),parameterized_rhs=e,metrics=results),indent=2));print(json.dumps({k:v for k,v in results.items() if k not in ['per_trajectory','parameters']},indent=2),flush=True)

 if results['internal_validation_mse']<1e-16:(ROOT/'SOLVED.json').write_text(json.dumps(dict(validated_at_unix=time.time(),validation_mse=results['internal_validation_mse'])))
