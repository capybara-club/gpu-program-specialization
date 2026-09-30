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
"""Replay frozen JSON through C, with exact retained-result comparisons.

Request timings include C parsing and one-time AST generation, measured
separately in reports. Scoring CUDA intervals start after inputs are prepared.
Optional prepared replay is instrumentation, excluded from normal timing runs.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def signature(report):
    payload = dict(candidates=report['candidates'], leaderboards=report['leaderboards'],
        counts={k: report['counts'][k] for k in ['completed_configurations','valid','invalid']})
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',required=True);p.add_argument('--request',required=True)
    p.add_argument('--output',required=True);p.add_argument('--devices',default='0')
    p.add_argument('--work-cap',type=int,default=67108864)
    p.add_argument('--family-configs',type=int);p.add_argument('--families',type=int)
    p.add_argument('--execution',default='{}');p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--profile',action='store_true');p.add_argument('--prepared',action='store_true')
    p.add_argument('--reference');p.add_argument('--cache',default='/tmp/odezza-long-rollout-probe-cache')
    a=p.parse_args()
    assert 1<=a.work_cap<=64*1024**3
    os.environ['CUDA_MODULE_LOADING']='EAGER'
    os.environ['ODZ_BENCH_TILE_WORK_UNITS']=str(a.work_cap)
    os.environ['ODZ_BENCH_TRACE']='1'
    if a.prepared:os.environ['ODZ_BENCH_PREPARED_REPEATS']='1'
    request_path=Path(a.request);raw=request_path.read_bytes();q=json.loads(raw)
    if a.families:q['grammar']['families']=q['grammar']['families'][:a.families]
    if a.family_configs:
        for family in q['grammar']['families']:
            family['limits']['max_configurations']=min(a.family_configs,family['limits']['max_configurations'])
    q['execution'].update(json.loads(a.execution))
    q['execution']['max_seconds']=600
    q['execution']['profile_timing']=a.profile or a.prepared
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists(),out
    devices=list(map(int,a.devices.split(',')))
    lib=C.CDLL(a.library);P=C.c_void_p
    lib.odz_runtime_create_devices.argtypes=[C.POINTER(C.c_uint),C.c_size_t,C.c_char_p,C.POINTER(P)]
    lib.odz_runtime_attempt_inputs.argtypes=[P]
    lib.odz_runtime_reserve_buffers.argtypes=[P,C.c_size_t,C.c_size_t,C.c_size_t]
    lib.odz_runtime_destroy.argtypes=[P]
    lib.odz_job_create.argtypes=[C.c_char_p,C.c_size_t,C.POINTER(P)]
    lib.odz_job_run.argtypes=[P,P]
    lib.odz_job_report.argtypes=[P,P,C.c_size_t,C.POINTER(C.c_size_t)]
    lib.odz_job_destroy.argtypes=[P]
    handle=P();begin=time.perf_counter()
    assert not lib.odz_runtime_create_devices((C.c_uint*len(devices))(*devices),len(devices),a.cache.encode(),C.byref(handle))
    assert not lib.odz_runtime_attempt_inputs(handle)
    assert not lib.odz_runtime_reserve_buffers(handle,64<<20,256<<20,64<<20)
    result=dict(source_request_sha256=hashlib.sha256(raw).hexdigest(),request=q,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        devices=devices,work_cap=a.work_cap,profile=a.profile,prepared=a.prepared,
        initialization_seconds=time.perf_counter()-begin,runs=[])
    expected=None
    if a.reference:
        previous=json.loads(Path(a.reference).read_text())
        expected=signature(previous['runs'][-1]['report'] if 'runs' in previous else previous)
    try:
        for repeat in range(a.repeats+1):
            raw=json.dumps(q,separators=(',',':')).encode();job=P()
            assert not lib.odz_job_create(raw,len(raw),C.byref(job))
            start=time.perf_counter()
            try:
                code=lib.odz_job_run(handle,job)
                size=C.c_size_t();assert not lib.odz_job_report(job,None,0,C.byref(size))
                buf=C.create_string_buffer(size.value)
                assert not lib.odz_job_report(job,buf,len(buf),C.byref(size))
                report=json.loads(buf.value)
                row=dict(repeat=repeat,cold=repeat==0,elapsed=time.perf_counter()-start,report=report)
                result['runs'].append(row)
                out.write_text(json.dumps(result,indent=2)+'\n')
                assert not code and report['status']=='complete' and report['retention_complete'],report.get('error')
                current=signature(report)
                if expected is None:expected=current
                assert current==expected,('retained outputs or coverage changed',current,expected)
                row['exact_signature']=current
                print(json.dumps(dict(repeat=repeat,wall=report['timing']['total_seconds'],
                    configs=report['counts']['completed_configurations'],
                    chunks=report['counts']['completed_chunks'],signature=current,
                    gpus=[dict(device=d['device'],gpu=d['pipeline_breakdown']['gpu_active_seconds'],
                        tile=d['largest_tile_configurations'],modules=d['pipeline_breakdown']['modules'],
                        capacity=d['pipeline_breakdown']['maximum_module_capacity']) for d in report['execution']['devices']])),flush=True)
            finally:lib.odz_job_destroy(job)
        result['passed']=True
        if a.repeats:result['warm_median_seconds']=statistics.median(r['report']['timing']['total_seconds'] for r in result['runs'][1:])
    finally:
        lib.odz_runtime_destroy(handle)
        out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
