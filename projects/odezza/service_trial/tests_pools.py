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
"""Real-CUDA pool capacity, reuse, bounds and legacy ownership comparisons."""
import copy
import ctypes as C
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
lib=C.CDLL(str(ROOT/'build/runtime/libodezza_runtime.so'))
P=C.c_void_p
lib.odz_runtime_create_devices.argtypes=[C.POINTER(C.c_uint),C.c_size_t,C.c_char_p,C.POINTER(P)]
lib.odz_runtime_attempt_inputs.argtypes=[P]
lib.odz_runtime_reserve_buffers.argtypes=[P,C.c_size_t,C.c_size_t,C.c_size_t]
lib.odz_job_create.argtypes=[C.c_char_p,C.c_size_t,C.POINTER(P)]
lib.odz_job_create_borrowed.argtypes=[C.c_char_p,C.c_size_t,P,C.c_size_t,C.POINTER(P)]
lib.odz_job_run.argtypes=[P,P]
lib.odz_job_report.argtypes=[P,P,C.c_size_t,C.POINTER(C.c_size_t)]
lib.odz_job_destroy.argtypes=lib.odz_runtime_destroy.argtypes=[P]
probe=C.CDLL(None)
probe.odz_probe_stop.restype=C.c_uint64

def runtime(pools=None,devices=(0,)):
    r=P()
    assert not lib.odz_runtime_create_devices((C.c_uint*len(devices))(*devices),len(devices),b'/tmp/odezza-service-trial-cache',C.byref(r))
    assert not lib.odz_runtime_attempt_inputs(r)
    if pools:assert not lib.odz_runtime_reserve_buffers(r,*pools)
    return r

def run(r,request,capacity=None):
    raw=json.dumps(request,separators=(',',':')).encode()
    job=P()
    arena=C.create_string_buffer(capacity+32) if capacity is not None else None
    if arena is not None:
        C.memset(C.addressof(arena),0x5a,len(arena))
        assert not lib.odz_job_create_borrowed(raw,len(raw),C.addressof(arena)+16,capacity,C.byref(job))
    else:assert not lib.odz_job_create(raw,len(raw),C.byref(job))
    lib.odz_job_run(r,job)
    size=C.c_size_t()
    assert not lib.odz_job_report(job,None,0,C.byref(size))
    output=C.create_string_buffer(size.value)
    assert not lib.odz_job_report(job,output,len(output),C.byref(size))
    result=json.loads(output.value)
    lib.odz_job_destroy(job)
    if arena is not None:
        assert arena.raw[:16]==b'Z'*16 and arena.raw[-16:]==b'Z'*16,'arena boundary overwritten'
    return result

def same(a,b):
    assert a['status']==b['status']=='complete',(a.get('error'),b.get('error'))
    for k in ('candidates','leaderboards','families','counts'):assert a[k]==b[k],k

def main():
    base=json.loads((ROOT/'service_trial/validation/requests/explicit_cartesian_recovery.json').read_text())
    literal=copy.deepcopy(base)
    for key in ('constants','constant_banks','parameters'):literal['grammar'].pop(key,None)
    literal['grammar']['rhs']={'x0':'-x0','x1':'-x1'}
    legacy=runtime()
    refs=[run(legacy,base),run(legacy,literal)]
    lib.odz_runtime_destroy(legacy)
    records=[]
    for devices in ((0,),(0,1)):
        r=runtime((1048576,1048576,1048576),devices)
        probe.odz_probe_start()
        same(refs[1],run(r,literal,1048576))
        assert probe.odz_probe_stop()==0,'first literal request allocated device storage'
        for i in range(3):
            same(refs[0],run(r,base,1048576))
            probe.odz_probe_start()
            report=run(r,literal,1048576)
            allocations=probe.odz_probe_stop()
            same(refs[1],report)
            assert allocations==0,allocations
            records.append(dict(check='reuse_and_no_explicit_cuda_allocations',devices=devices,iteration=i,allocations=allocations,report=report))
        assert lib.odz_runtime_reserve_buffers(r,1024,1024,1024),'late reservation accepted'
        short=run(r,base,16)
        assert short['status']=='failed' and 'CPU pool too small' in short['error'] and not short['runtime_quarantined'],short
        same(refs[0],run(r,base,1048576))
        records.append(dict(check='short_cpu_arena_then_reuse',devices=devices,report=short))
        lib.odz_runtime_destroy(r)
    r=runtime((1024,1048576,1048576))
    longer=copy.deepcopy(base)
    longer['problem']['trajectories']*=4
    bad=run(r,longer,1048576)
    assert bad['status']=='failed' and 'GPU pool too small' in bad['error'] and not bad['runtime_quarantined'],bad
    same(refs[0],run(r,base,1048576))
    records.append(dict(check='short_gpu_trajectory_pool_then_reuse',report=bad))
    lib.odz_runtime_destroy(r)
    r=runtime((1048576,32768,4096))
    toggle=json.loads((ROOT/'service_trial/validation/requests/toggles_all.json').read_text())
    references=json.loads((ROOT/'service_trial/validation/direct-reference.json').read_text())
    ref=next(x['report'] for x in references if x['name']=='toggles_all')
    actual=run(r,toggle,1048576)
    # Chunk count can change when the fixed pool requires smaller tiles.
    for k in ('candidates','leaderboards','families'):assert actual[k]==ref[k],(k,actual.get('error'))
    assert actual['counts']['completed_configurations']==ref['counts']['completed_configurations']
    assert actual['counts']['completed_chunks']>ref['counts']['completed_chunks']
    records.append(dict(check='bounded_tiles_preserve_all_work',report=actual))
    lib.odz_runtime_destroy(r)
    (ROOT/'service_trial/validation/buffer-pools.json').write_text(json.dumps(dict(passed=True,checks=records),indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(records))))

if __name__=='__main__':main()
