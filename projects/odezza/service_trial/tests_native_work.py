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
"""Direct-runtime safety backstop: no service admission involved."""
import ctypes as C
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
lib=C.CDLL(str(ROOT/'build/runtime/libodezza_runtime.so'))
P=C.c_void_p
lib.odz_runtime_create.argtypes=[C.c_uint,C.c_char_p,C.POINTER(P)]
lib.odz_runtime_destroy.argtypes=[P]
lib.odz_job_create.argtypes=[C.c_char_p,C.c_size_t,C.POINTER(P)]
lib.odz_job_run.argtypes=[P,P]
lib.odz_job_report.argtypes=[P,P,C.c_size_t,C.POINTER(C.c_size_t)]
lib.odz_job_destroy.argtypes=[P]
r=P();assert not lib.odz_runtime_create(0,b'/tmp/odezza-service-trial-cache',C.byref(r))
records=[]
try:
    for intervals,dt,accepted in ((1,1/8192,False),(17,1/4096,False),(1,1e-30,False),(65536,1,False),(16,1/4096,True),(1,.125,True)):
        request=dict(problem=dict(states=['x'],trajectories=[dict(initial=[0],times=list(range(intervals+1)),values=[[0]]*(intervals+1))]),grammar=dict(version=1,states=['x'],rhs={'x':'-x'},integration=dict(method='rk4',dt=dt)))
        raw=json.dumps(request).encode();job=P()
        assert not lib.odz_job_create(raw,len(raw),C.byref(job))
        lib.odz_job_run(r,job);size=C.c_size_t()
        assert not lib.odz_job_report(job,None,0,C.byref(size))
        out=C.create_string_buffer(size.value)
        assert not lib.odz_job_report(job,out,len(out),C.byref(size))
        report=json.loads(out.value);lib.odz_job_destroy(job)
        assert report['status']==('complete' if accepted else 'failed'),report.get('error')
        assert not report['runtime_quarantined']
        if not accepted:
            assert report['counts']['completed_configurations']==0
            assert report['timing']['nvrtc_seconds']==0
            assert report['cache']['misses']==0
        records.append(dict(intervals=intervals,dt=dt,accepted=accepted,report=report))
    (ROOT/'service_trial/validation/native-work-limits.json').write_text(json.dumps(dict(passed=True,records=records),indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(records))))
finally:lib.odz_runtime_destroy(r)
