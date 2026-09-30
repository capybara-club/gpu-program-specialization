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
"""Standard-library Python + native C++ CPU integrators; no NumPy or CUDA."""
from array import array
import ctypes as C
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SHARED=HERE.parent/'rosenbrock_trial'
sys.path.insert(0,str(SHARED))
from codegen import model
sys.path.pop(0)


class Stats(C.Structure):
    _fields_=[(k,C.c_int) for k in ('status','accepted','rejected','attempts','rhs','jac','lu','pivot_failures')]
    _fields_ += [('count',C.c_int64),('mse',C.c_double),('min_h',C.c_double),('max_h',C.c_double)]


DP=C.POINTER(C.c_double); IP=C.POINTER(C.c_int); SP=C.POINTER(Stats)


def pointer(a,kind):
    return C.cast(a.buffer_info()[0],kind)


def write(path,value):
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x): return None
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return [clean(v) for v in x]
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


class Solver:
    def __init__(self,expressions,states,cache=None,compiler=None,precision='FP64'):
        if precision not in ('FP32','FP64'):raise ValueError('precision must be FP32 or FP64')
        self.precision=precision
        self.n=len(states); generated=model(expressions,states,precision)
        compiler=compiler or shutil.which('clang++') or shutil.which('c++')
        if not compiler: raise RuntimeError('an installed C++17 compiler is required')
        flags=['-std=c++17','-O3','-ffp-contract=off','-fPIC',
               '-dynamiclib' if sys.platform=='darwin' else '-shared']
        if precision=='FP32':flags.append('-DODEZZA_CPU_FLOAT=1')
        version=subprocess.check_output([compiler,'--version'],text=True)
        content=(generated.encode()+(SHARED/'solver.cuh').read_bytes()+(HERE/'runtime.cpp').read_bytes()+(HERE/'solver32.hpp').read_bytes()
                 +repr(flags).encode()+version.encode()+sys.platform.encode())
        self.key=hashlib.sha256(content).hexdigest()
        folder=Path(cache or HERE/'build')/self.key;folder.mkdir(parents=True,exist_ok=True)
        library=folder/('model.dylib' if sys.platform=='darwin' else 'model.so')
        start=time.perf_counter();self.compiled=False
        with (folder/'build.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not library.exists():
                (folder/'model.cuh').write_text(generated)
                cmd=[compiler,*flags,'-I'+str(SHARED),'-I'+str(folder),str(HERE/'runtime.cpp'),'-o',str(library)+'.tmp']
                run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (folder/'compile.log').write_text(run.stdout)
                write(folder/'build.json',dict(command=cmd,compiler=version))
                if run.returncode: raise RuntimeError(run.stdout)
                Path(str(library)+'.tmp').rename(library);self.compiled=True
        self.compile_seconds=time.perf_counter()-start
        start=time.perf_counter();self.lib=C.CDLL(str(library));self.load_seconds=time.perf_counter()-start
        prefix=[C.c_int,IP,DP,DP,DP]
        self.lib.cpu_ros23.argtypes=prefix+[C.c_double,C.c_double,C.c_double,C.c_int,DP,SP]
        self.lib.cpu_ros23.restype=None
        self.lib.cpu_rk4.argtypes=prefix+[C.c_int,DP,SP];self.lib.cpu_rk4.restype=None
        self.lib.cpu_model.argtypes=[DP,DP,DP];self.lib.cpu_model.restype=None

    def evaluate(self,y):
        if len(y)!=self.n: raise ValueError('state dimension mismatch')
        y=array('d',y);rhs=array('d',[0.]*self.n);jac=array('d',[0.]*(self.n*self.n))
        self.lib.cpu_model(pointer(y,DP),pointer(rhs,DP),pointer(jac,DP))
        return list(rhs),[list(jac[j*self.n:(j+1)*self.n]) for j in range(self.n)]

    def run(self,rows,method='Rosenbrock23',rtol=None,atol=None,h0=0.,max_attempts=2000000,
            substeps=32,predictions=True):
        if rtol is None:rtol=1e-5 if self.precision=='FP32' else 1e-8
        if atol is None:atol=rtol*.01
        if method not in ('Rosenbrock23','RK4'): raise ValueError('unknown CPU method')
        if not rows: raise ValueError('empty trajectories')
        if not all(math.isfinite(x) and x>0 for x in (rtol,atol)) or not math.isfinite(h0) or h0<0:
            raise ValueError('invalid tolerances or initial step')
        if type(max_attempts)!=int or not 1<=max_attempts<=2**31-1: raise ValueError('invalid work limit')
        if type(substeps)!=int or not 1<=substeps<=2**31-1: raise ValueError('invalid substep count')
        off=array('i',[0]);tt=array('d');ic=array('d');obs=array('d')
        for row in rows:
            times=row['times'];initial=row['initial_state'];values=row['states']
            if (len(times)<2 or not all(math.isfinite(v) for v in times)
                    or not all(b>a for a,b in zip(times,times[1:]))
                    or len(initial)!=self.n or not all(math.isfinite(v) for v in initial)
                    or len(values)!=len(times) or any(len(v)!=self.n for v in values)):
                raise ValueError('invalid trajectory dimensions, times or initial state')
            tt.extend(times);ic.extend(initial)
            for value in (v for sample in values for v in sample):
                if value is not None and not math.isfinite(value): raise ValueError('nonfinite observation; use null for missing')
                obs.append(math.nan if value is None else value)
            if len(tt)*self.n>2**31-1 or (len(times)-1)*substeps>2**29: raise ValueError('index/work capacity exceeded')
            off.append(len(tt))
        pp=array('d',[math.nan])*len(obs) if predictions else None
        stats=(Stats*len(rows))()
        prefix=[len(rows),pointer(off,IP),pointer(tt,DP),pointer(ic,DP),pointer(obs,DP)]
        tail=[pointer(pp,DP) if pp is not None else None,stats]
        start=time.perf_counter(); cpu_start=time.process_time()
        if method=='Rosenbrock23':self.lib.cpu_ros23(*prefix,rtol,atol,h0,max_attempts,*tail)
        else:self.lib.cpu_rk4(*prefix,substeps,*tail)
        cpu_seconds=time.process_time()-cpu_start;seconds=time.perf_counter()-start
        output=None if pp is None else [[list(pp[i*self.n:(i+1)*self.n]) for i in range(a,b)] for a,b in zip(off,off[1:])]
        records=[{k:getattr(s,k) for k,_ in Stats._fields_} for s in stats]
        return dict(method=method,precision=self.precision,clock_precision='FP64',score_precision='FP64',execution='native CPU, serial trajectories',
                    rtol=rtol if method=='Rosenbrock23' else None,atol=atol if method=='Rosenbrock23' else None,
                    substeps=substeps if method=='RK4' else None,wall_seconds=seconds,cpu_seconds=cpu_seconds,
                    stats=records,predictions=output,complete=all(s['status']==0 for s in records),
                    model_sha256=self.key)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.request.read_text())
    if r.get('schema') not in ('odezza.rosenbrock-trial.v1','odezza.cpu-solver-trial.v1'):raise ValueError('unknown schema')
    if set(r)-{'schema','states','rhs','trajectories','solver'}:raise ValueError('unknown field')
    settings=dict(r.get('solver',{}));precision=settings.pop('precision','FP64')
    s=Solver(r['rhs'],r['states'],precision=precision);result=s.run(r['trajectories'],**settings)
    result.update(compile_seconds=s.compile_seconds,load_seconds=s.load_seconds)
    write(a.out,result);print(json.dumps(dict(complete=result['complete'],cpu_seconds=result['cpu_seconds'],output=str(a.out))))
    if not result['complete']:raise SystemExit(2)


if __name__=='__main__':main()
