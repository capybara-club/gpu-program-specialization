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
"""Run the six-state public-template trial in a fresh output directory."""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCES=['prepare','features','initial_features','engine','common','mutate','adjoint_library','native','sparse','focus','neighborhood','validate']
def check(d):
 if d.get('states')!=[f'x{i}' for i in range(6)] or d.get('observed_states')!=['x0','x1','x2','x3'] or d.get('hidden_after_t0')!=['x4','x5']:raise ValueError('This trial requires six states, x0..x3 observed, x4/x5 hidden after full ICs.')
 expected=['sin(w*u*v)','sin(w*u+v)','sin(w*u*sin(v))','sin(w*u)*sin(v)']
 if [s.replace(' ','') for s in d.get('templates',[])]!=expected:raise ValueError('Unsupported template catalogue')
 expected_known=['-0.20*x0-0.80*x1+0.35*x2','0.80*x0-0.20*x1+0.35*x3']
 for i,e in enumerate(expected_known):
  if d.get('known_rhs',{}).get(f'x{i}','').split('=')[-1].replace(' ','')!=e:raise ValueError('Unsupported known linear subsystem')
 if d.get('known_damping')!=-.35 or d.get('amplitude_abs_range')!=[.25,.8] or d.get('frequency_range')!=[.7,2.2]:raise ValueError('Unsupported damping/coefficient profile')
 if d.get('observation_times')!=[0.,.7,1.4,2.2,3.1,4.1,5.2,6.4,8.] or len(d.get('train',[]))!=32:raise ValueError('Unsupported observation schedule/training count')
 for r in d['train']:
  if len(r['t0'])!=6 or any(len(r['observations'][f'x{i}'])!=9 for i in range(4)):raise ValueError('Invalid observations/IC dimensions')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('public',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--timeout',type=float,default=900);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raw=a.public.read_bytes();d=json.loads(raw);check(d)
 if a.check_only:print('Supported public profile');return 0
 if a.out is None:ap.error('--out is required for a run')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.time();(out/'public.json').write_bytes(raw)
 for name in SOURCES:shutil.copyfile(HERE/(name+'.py'),out/(name+'.py'))
 (out/'experiment.json').write_text(json.dumps(dict(started_utc=datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),input_sha256=hashlib.sha256(raw).hexdigest(),training_indices=list(range(24)),validation_indices=list(range(24,32)),entry_point=str(HERE/'run.py'),private_truth_used=False),indent=2))
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');env['PYTHONPATH']=str(REPO)+os.pathsep+env.get('PYTHONPATH','');children={};events=[]
 def event(kind,**kw):
  row=dict(event=kind,seconds=time.time()-started,**kw);events.append(row);(out/'controller-events.json').write_text(json.dumps(events,indent=2));print(json.dumps(row),flush=True)
 def launch(key,script,*args):
  log=open(out/(key+'.log'),'w');p=subprocess.Popen([sys.executable,str(out/(script+'.py')),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children[key]=p;event('start',job=key,pid=p.pid)
 def stop():
  for key,p in children.items():
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM);event('cancel',job=key)
  for p in children.values():
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  sys.path.insert(0,str(REPO))
  from service_trial.client import Client
  receipts=list(out.glob('*/submitted.json'))
  if receipts:
   c=Client(timeout=5)
   try:
    for receipt in receipts:
     h=json.loads(receipt.read_text())['handle']
     try:
      st=c.status(h)
     except RuntimeError as e:
      event('handle_unavailable',handle=h,error=str(e));continue
     if st['state']!='terminal':c.rpc('cancel.'+h)
     end=time.monotonic()+10
     while c.status(h)['state']!='terminal' and time.monotonic()<end:time.sleep(.1)
     if c.status(h)['state']=='terminal':
      (receipt.parent/'controller-stop-report.json').write_text(json.dumps(c.result(h),indent=2));c.rpc('release.'+h);event('released',handle=h)
     else:event('cleanup_pending',handle=h)
   finally:c.close()
 try:
  for script in ['prepare','initial_features']:
   launch(script,script);rc=children[script].wait()
   if rc:raise RuntimeError(f'{script} exited {rc}')
  launch('initial','native');launch('sparse0','sparse',0);launch('sparse2','sparse',2);completed=set();validation_started=False
  while time.time()-started<a.timeout:
   if (out/'SOLVED.json').exists():event('solved',validation=json.loads((out/'validation.json').read_text()));return 0
   for key,p in list(children.items()):
    rc=p.poll()
    if rc is None or key in completed:continue
    completed.add(key);event('finish',job=key,returncode=rc)
    if key=='initial' and rc==0:launch('edits','neighborhood')
    if key in ['sparse0','sparse2'] and rc==0:launch('focus'+key[-1],'focus',int(key[-1]),1)
    if key=='validate' and rc!=0:raise RuntimeError('Validation failed; retained artifacts require inspection')
   if not validation_started:
    winner=None
    for f in out.glob('*fit*.json'):
     if f.name.endswith('-jobs.json'):continue
     try:rows=json.loads(f.read_text()).get('results',[])
     except (OSError,json.JSONDecodeError):continue
     for r in rows:
      if r.get('best') and r['best']['mse']<1e-15:winner=r
    if winner is not None:validation_started=True;event('accurate_candidate',mse=winner['best']['mse'],first_accurate_unix=winner['best']['at_unix']);launch('validate','validate')
   if all(p.poll() is not None for p in children.values()):event('exhausted');return 2
   time.sleep(.25)
  event('timeout');return 2
 finally:stop()
if __name__=='__main__':raise SystemExit(main())
