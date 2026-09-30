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
"""Standalone NumPy/SciPy expression differentiation and full-rollout least squares."""
import ast,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
Z=('c',0.);ONE=('c',1.)
def node(op,a,b=None):
 if op=='+':
  if a==Z:return b
  if b==Z:return a
 if op=='*':
  if a==Z or b==Z:return Z
  if a==ONE:return b
  if b==ONE:return a
 return (op,a) if b is None else (op,a,b)
def parse(text):
 def visit(n):
  if isinstance(n,ast.Constant):return ('c',float(n.value))
  if isinstance(n,ast.Name):return ('v',int(n.id[1:])) if n.id.startswith('x') else ('p',int(n.id[1:]))
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return node('*',('c',-1.),visit(n.operand))
  if isinstance(n,ast.BinOp):return node({ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}[type(n.op)],visit(n.left),visit(n.right))
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['sin','cos']:return node(n.func.id,visit(n.args[0]))
  raise ValueError(ast.dump(n))
 return visit(ast.parse(text,mode='eval').body)
def diff(n,v):
 op=n[0]
 if op in ['c','v','p']:return ONE if n==v else Z
 a=n[1];da=diff(a,v)
 if op in ['+','-']:return node(op,da,diff(n[2],v))
 if op=='*':return node('+',node('*',da,n[2]),node('*',a,diff(n[2],v)))
 if op=='sin':return node('*',node('cos',a),da)
 if op=='cos':return node('*',('c',-1.),node('*',node('sin',a),da))
 raise ValueError(op)
def compile_model(expressions,P):
 trees=[parse(e) for e in expressions];memo={};lines=[]
 def emit(n):
  if n in memo:return memo[n]
  op=n[0]
  if op=='c':return repr(n[1])
  if op=='v':return f's[:,{n[1]}]'
  if op=='p':return f'p[{n[1]}]'
  args=[emit(c) for c in n[1:]];expr=f'np.{op}({args[0]})' if op in ['sin','cos'] else f'({args[0]} {op} {args[1]})';name='v'+str(len(lines));lines.append('    '+name+'='+expr);memo[n]=name;return name
 vals=[emit(n) for n in trees];grads=[emit(diff(n,k)) for n in trees for k in [('v',i) for i in range(6)]+[('p',i) for i in range(P)]]
 source='def f(s,p):\n'+'\n'.join(lines)+'\n    N=len(s)\n    return np.column_stack(['+','.join(f'np.broadcast_to({e},(N,))' for e in vals)+']),np.stack(['+','.join(f'np.broadcast_to({e},(N,))' for e in grads)+f'],axis=1).reshape(N,6,{6+P})\n'
 ns={'np':np};exec(source,ns);return ns['f']
def simulate(model,p,t,ic,obs,deadline=None,rtol=1e-7,atol=1e-9):
 N=len(obs);P=len(p);W=6+6*P;q=np.zeros((N,W));q[:,:6]=ic
 def rhs(tm,flat):
  if deadline and time.perf_counter()>deadline:raise TimeoutError('fit budget')
  a=flat.reshape(N,W);s=a[:,:6]
  if not np.isfinite(a).all() or np.max(abs(s))>100:raise FloatingPointError('divergent rollout')
  val,g=model(s,p);out=np.empty_like(a);out[:,:6]=val;out[:,6:]=(np.einsum('nij,njk->nik',g[:,:,:6],a[:,6:].reshape(N,6,P),optimize=False)+g[:,:,6:]).reshape(N,6*P);return out.ravel()
 sol=solve_ivp(rhs,(t[0],t[-1]),q.ravel(),t_eval=t,method='DOP853',rtol=rtol,atol=atol)
 if not sol.success:raise RuntimeError(sol.message)
 v=sol.y.reshape(N,W,len(t)).transpose(0,2,1);return (v[:,1:,:4]-obs[:,1:]).ravel(),v[:,1:,6:].reshape(N,len(t)-1,6,P)[:,:,:4].reshape(-1,P)
def fit(job):
 expr,p0,t,ic,obs,seconds,ident=job;begin=time.perf_counter();started=time.time();deadline=begin+seconds;P=len(p0);model=compile_model(expr,P);best=None;calls=0;last=None;cached=None;invalid=0
 try:
  def evaluate(p):
   nonlocal best,calls,last,cached,invalid
   if last is not None and np.array_equal(last,p):return cached
   calls+=1
   try:r,j=simulate(model,p,t,ic,obs,deadline)
   except (FloatingPointError,RuntimeError):invalid+=1;return np.full(obs[:,1:].size,1e6),np.zeros((obs[:,1:].size,P))
   mse=float(np.mean(r*r))
   if best is None or mse<best['mse']:best=dict(mse=mse,parameters=p.tolist(),at_unix=time.time())
   last=p.copy();cached=r,j;return cached
  if np.mean(evaluate(np.array(p0))[0]**2)>1e10:raise ValueError('initial rollout divergent')
  f=least_squares(lambda p:evaluate(p)[0],p0,jac=lambda p:evaluate(p)[1],bounds=(np.tile([-1,.65],P//2),np.tile([1,2.3],P//2)),x_scale='jac',max_nfev=45,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  details=dict(status=int(f.status),nfev=f.nfev)
 except Exception as e:details=dict(error=repr(e))
 return dict(id=ident,expression=expr,initial_parameters=np.asarray(p0).tolist(),best=best,calls=calls,invalid=invalid,seconds=time.perf_counter()-begin,started_at_unix=started,finished_at_unix=time.time(),**details)
