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
"""Local experimental solver interface; NOT registered with the search service."""
import ctypes as C
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
from codegen import model

HERE = Path(__file__).resolve().parent


class Stats(C.Structure):
    _fields_ = [(k, C.c_int) for k in ('status', 'accepted', 'rejected', 'attempts', 'rhs', 'jac', 'lu', 'pivot_failures')]
    _fields_ += [('count', C.c_int64), ('mse', C.c_double), ('min_h', C.c_double), ('max_h', C.c_double)]


DP = C.POINTER(C.c_double)
IP = C.POINTER(C.c_int)
SP = C.POINTER(Stats)


class Solver:
    def __init__(self, expressions, states, cache=None, arch='sm_120', nvcc='/usr/local/cuda/bin/nvcc'):
        self.n = len(states)
        generated = model(expressions, states)
        flags = ['-std=c++17', '-O3', '--fmad=false', '-U_GNU_SOURCE', '-D_DEFAULT_SOURCE']
        content = generated.encode() + (HERE/'solver.cuh').read_bytes() + (HERE/'runtime.cu').read_bytes() + repr(flags).encode()
        version = subprocess.check_output([nvcc, '--version'], text=True)
        self.key = hashlib.sha256(content + arch.encode() + version.encode()).hexdigest()
        folder = Path(cache or HERE/'build')/self.key
        folder.mkdir(parents=True, exist_ok=True)
        library = folder/'model.so'
        started = time.perf_counter(); self.compiled = False
        with (folder/'build.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library.exists():
                (folder/'model.cuh').write_text(generated)
                cmd = [nvcc, *flags, '-arch='+arch, '--shared',
                       '-Xcompiler', '-fPIC', '-Xptxas', '-v', '-I'+str(folder), '-I'+str(HERE),
                       str(HERE/'runtime.cu'), '-o', str(library)+'.tmp']
                result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (folder/'compile.log').write_text(result.stdout)
                (folder/'build.json').write_text(json.dumps(dict(command=cmd, compiler=version)))
                if result.returncode: raise RuntimeError(result.stdout)
                Path(str(library)+'.tmp').rename(library); self.compiled = True
        self.compile_seconds = time.perf_counter()-started
        started = time.perf_counter(); self.lib = C.CDLL(str(library))
        self.load_seconds = time.perf_counter()-started
        common = [C.c_int, IP, DP, DP, DP, C.c_double, C.c_double, C.c_double, C.c_int, DP, SP]
        self.lib.cpu_run.argtypes = common; self.lib.cpu_run.restype = None
        self.lib.gpu_run.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int] + common[1:] + [C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_char_p, C.c_int]
        self.lib.gpu_run.restype = C.c_int
        self.lib.model_eval.argtypes = [DP, DP, DP]; self.lib.model_eval.restype = None

    def evaluate(self, y):
        y = np.ascontiguousarray(y, dtype=np.float64)
        if y.shape != (self.n,): raise ValueError('state dimension mismatch')
        rhs = np.empty(self.n); jac = np.empty((self.n,self.n))
        self.lib.model_eval(y.ctypes.data_as(DP), rhs.ctypes.data_as(DP), jac.ctypes.data_as(DP))
        return rhs, jac

    def run(self, rows, *, rtol=1e-8, atol=1e-10, h0=0., max_attempts=2000000,
            device=0, cpu=False, repetitions=1, predictions=True):
        if not rows: raise ValueError('empty trajectories')
        if not all(math.isfinite(x) and x>0 for x in (rtol,atol)) or not math.isfinite(h0) or h0<0:
            raise ValueError('finite positive tolerances and nonnegative initial step required')
        if type(max_attempts) is not int or not 1<=max_attempts<=2**31-1: raise ValueError('invalid attempt limit')
        if type(repetitions) is not int or not 1<=repetitions or repetitions*len(rows)>2**31-1:
            raise ValueError('invalid repetition count')
        if cpu and repetitions != 1: raise ValueError('CPU repetitions unsupported')
        offsets = [0]; times = []; initial = []; observed = []
        for row in rows:
            tt = np.asarray(row['times'], dtype=np.float64)
            ii = np.asarray(row['initial_state'], dtype=np.float64)
            oo = np.asarray(row['states'], dtype=np.float64) # None -> NaN means unobserved.
            if (tt.ndim != 1 or len(tt)<2 or not np.isfinite(tt).all() or not (np.diff(tt)>0).all()
                    or ii.shape != (self.n,) or not np.isfinite(ii).all() or oo.shape != (len(tt),self.n)):
                raise ValueError('invalid trajectory shape, times or initial state')
            if np.isinf(oo).any(): raise ValueError('infinite observations')
            times.extend(tt); initial.extend(ii); observed.extend(oo)
            offsets.append(len(times))
        if len(times)*self.n>2**31-1: raise ValueError('trajectory index capacity exceeded')
        off = np.asarray(offsets,dtype=np.int32); tt = np.asarray(times,dtype=np.float64)
        ii = np.asarray(initial,dtype=np.float64); oo = np.asarray(observed,dtype=np.float64)
        pp = np.full(oo.shape,np.nan) if predictions else None
        lanes = len(rows)*repetitions; stats = (Stats*lanes)()
        tail = [off.ctypes.data_as(IP),tt.ctypes.data_as(DP),ii.ctypes.data_as(DP),oo.ctypes.data_as(DP),
                rtol,atol,h0,max_attempts,pp.ctypes.data_as(DP) if predictions else None,stats]
        started = time.perf_counter(); kernel = C.c_float(); transfers = C.c_float()
        if cpu: self.lib.cpu_run(len(rows),*tail)
        else:
            error = C.create_string_buffer(2048)
            code = self.lib.gpu_run(device,len(rows),lanes,len(times),*tail,C.byref(kernel),C.byref(transfers),error,len(error))
            if code: raise RuntimeError(error.value.decode())
        elapsed = time.perf_counter()-started
        records = [{k:getattr(s,k) for k,_ in Stats._fields_} for s in stats]
        return dict(stats=records, predictions=None if pp is None else [pp[a:b].tolist() for a,b in zip(offsets,offsets[1:])],
                    wall_seconds=elapsed,kernel_ms=None if cpu else kernel.value,
                    transfer_interval_ms=None if cpu else transfers.value,lanes=lanes,repetitions=repetitions,
                    unique_trajectories=len(rows),unique_models=1,rtol=rtol,atol=atol,model_sha256=self.key)
