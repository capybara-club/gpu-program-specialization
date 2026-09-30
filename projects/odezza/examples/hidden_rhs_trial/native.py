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
import json,sys,time,hashlib,struct
import numpy as np
from prepare import ROOT,load
from features import label,catalogue
from common import build,batch
import ast

def parse(text):
 """Read result expressions without algebraic folding of zero/one coefficients."""
 def visit(n):
  if isinstance(n,ast.Constant):return ('c',float(n.value))
  if isinstance(n,ast.Name):return ('v' if n.id.startswith('x') else 'p',int(n.id[1:]))
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return ('*',('c',-1.),visit(n.operand))
  if isinstance(n,ast.BinOp):return ({ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}[type(n.op)],visit(n.left),visit(n.right))
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='sin':return ('sin',visit(n.args[0]))
  raise ValueError(ast.dump(n))
 return visit(ast.parse(text,mode='eval').body)
sys.path.insert(0,str(ROOT.parents[1]))
from service_trial.client import Client

KNOWN={'x0':'-.2*x0-.8*x1+.35*x2','x1':'.8*x0-.2*x1+.35*x3'}
def make_base(rows=8192,indices=None,points=4,seed=280914):
 t,ic,obs=load();indices=indices if indices is not None else [0,4,8,12,16,20];traj=[]
 for i in indices:
  values=[ic[i].tolist()]+[v.tolist()+[None,None] for v in obs[i,1:points]]
  traj.append(dict(initial=ic[i].tolist(),times=t[:points].tolist(),values=values))
 g=dict(version=1,states=list('x'+str(i) for i in range(6)),integration=dict(method='rk4',dt=.05),expansion=dict(strategy='enumerate',max_nodes=128,max_depth=32),rules={},leaves={},rng={},rng_banks={'bank':dict(base='uniform01',count=rows,seed=seed,scope='run')},retain={'global':dict(k=48,unit='resolved_structure'),'per_family':dict(k=12,unit='resolved_structure')},families=[])
 return dict(problem=dict(states=g['states'],known_rhs=KNOWN,trajectories=traj),grammar=g,execution=dict(max_seconds=180,dedup_bytes_per_family=16777216,profile_timing=True))
def rng(g,name,lo,hi):
 g['rng'][name]=dict(bank='bank',axis='trial',stream=name,transform=dict(kind='uniform',low=float(lo),high=float(hi)))
 return 'rng.'+name

def run(name,q):
 out=ROOT/name;out.mkdir(exist_ok=False);raw=json.dumps(q).encode();(out/'request.json').write_bytes(raw);start=time.perf_counter();c=Client(timeout=30);h=None
 try:
  rec=c.submit(raw);h=rec['handle'];(out/'submitted.json').write_text(json.dumps(rec,indent=2));print('SUBMITTED',name,h,flush=True)
  while c.status(h)['state']!='terminal':
   if time.perf_counter()-start>240:c.rpc('cancel.'+h)
   time.sleep(.1)
  r=c.result(h);(out/'report.json').write_text(json.dumps(r,indent=2));(out/'timing.json').write_text(json.dumps(dict(client_seconds=time.perf_counter()-start,timing=r.get('timing'),counts=r.get('counts'),request_sha256=hashlib.sha256(raw).hexdigest()),indent=2));print(name,r.get('status'),r.get('counts'),r.get('timing'),r.get('error'),flush=True)
  if r['status']!='complete':raise ValueError(r.get('error'))
  return r
 finally:
  if h:c.rpc('release.'+h)
  c.close()

def unpack(c,program):
 bits=c['value_bits'];assert len(set(bits))==len(bits),'duplicate numeric slots';slots={b:'p'+str(i) for i,b in enumerate(bits)};data=bytes.fromhex(program);i=0;stack=[]
 while i<len(data):
  op=data[i];i+=1
  if op==0x80:assert len(stack)==1 and i==len(data);return stack[0]
  if op==0x81:stack.append('x'+str(data[i]));i+=1
  elif op==0x82:stack.append('p'+str(data[i]));i+=1
  elif op==0x83:
   b=f'{struct.unpack_from("<I",data,i)[0]:08x}';v=struct.unpack_from('<f',data,i)[0];i+=4;stack.append(slots.get(b,repr(v)))
  elif op==0x94:stack.append('(-'+stack.pop()+')')
  elif op==0x9b:stack.append('sin('+stack.pop()+')')
  else:
   b,a=stack.pop(),stack.pop();stack.append('('+a+{0x90:'+',0x91:'-',0x92:'*'}[op]+b+')')
 raise ValueError('no return')

def decode(c):
 def scalar(n):
  if n[0]=='p':return c['values'][n[1]]
  if n[0]=='c':return n[1]
  if n[0]=='*':return scalar(n[1])*scalar(n[2])
  raise ValueError(n)
 def leaves(n):
  if n[0] in ['c','p']:return [n]
  if n[0]=='v':return []
  return sum([leaves(v) for v in n[1:]],[])
 def canonical(n):
  if n[0] in ['c','p']:return ('p',1)
  if n[0]=='v':return n
  return (n[0],)+tuple(canonical(v) for v in n[1:])
 def adds(n):return adds(n[1])+adds(n[2]) if n[0]=='+' else [n]
 templates={parse(label(d).replace('w','p1')):d for d in catalogue()};rows=[]
 for program in c['resolved_programs'][2:]:
  e=unpack(c,program);terms=adds(parse(e))[1:];assert len(terms)==2,e;ds=[];p=[]
  for n in terms:
   assert n[0]=='*',n
   a=n[1];shape=n[2];key=canonical(shape);d=templates[key];ww=leaves(shape);assert len(ww)==1,ww;ds.append(d);p.extend([scalar(a),scalar(ww[0])])
  rows.append(dict(terms=ds,parameters=p))
 return build(rows)

def fit_report(report,name,limit=32):
 ids=[];families=list(report['leaderboards']['families'].values())
 # Eight per-family survivors, interleaved, before extra global winners.
 for depth in range(8):
  for group in families:ids.extend(group[depth:depth+1])
 ids+=report['leaderboards']['global']
 seen=set();jobs=[];t,ic,obs=load()
 for ident in ids:
  c=report['candidates'][ident];e,p=decode(c);key=tuple(e)
  if key in seen:continue
  seen.add(key);p=np.clip(p,np.tile([-1,.65],8)+1e-7,np.tile([1,2.3],8)-1e-7);jobs.append((e,p,t,ic,obs,12,dict(candidate=ident,screen_mse=c['mse'],origin=c['origin'])))
  if len(jobs)>=limit:break
 return batch(jobs,name,workers=5)

def initial():
 q=make_base(rows=8192);g=q['grammar'];pro=json.loads((ROOT/'initial-features.json').read_text())['results'];rs=np.random.default_rng(280914);cat=catalogue()
 # Independent random two-term pairs, sampled from the full public catalogue.
 hidden=[]
 for state in [4,5]:
  options=[]
  for j in range(128):
   pair=rs.choice(len(cat),2,replace=False);parts=[]
   for slot,d in enumerate([cat[k] for k in pair]):
    a=rng(g,f'h{state}a{slot}',-.8,.8);w=rng(g,f'h{state}w{slot}',.7,2.2);parts.append(a+'*('+label(d).replace('w',w)+')')
   options.append('+'.join(parts))
  g['rules'][f'H{state}']=options
 for family,rank2 in enumerate([0,1,2,3]):
  rhs={}
  for state,rank in [(2,rank2),(3,0)]:
   r=pro[f'd4_{state}'][rank];parts=[]
   for j,d in enumerate(r['terms']):
    a,w=r['parameters'][2*j:2*j+2];aa=rng(g,f'f{family}s{state}a{j}',max(-1,a-.06),min(1,a+.06));ww=rng(g,f'f{family}s{state}w{j}',max(.65,w-.12),min(2.3,w+.12));parts.append(aa+'*('+label(d).replace('w',ww)+')')
   rhs[f'x{state}']=f'-.35*x{state}+'+'+'.join(parts)
  for state in [4,5]:rhs[f'x{state}']=f'-.35*x{state}+H{state}'
  g['families'].append(dict(id=f'observed_{rank2}',rhs=rhs,tags=[f'initial_observed_rank:{rank2}'],limits=dict(max_skeletons=16384,max_variants=16384,max_configurations=16384*8192,max_derivations=100000,max_expansion_steps=100000000)))
 return q
if __name__=='__main__':
 r=run('initial-screen',initial());fit_report(r,'initial-fit',32)
