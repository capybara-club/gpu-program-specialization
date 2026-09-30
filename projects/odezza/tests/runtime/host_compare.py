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
"""Matched single-device native-runtime timing using the fixed million-system fixture."""
import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from configuration_scale import request
from service_upgrade import signature
from odezza.grammar.native_service import NativeService

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--reference',required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cache',default='/tmp/odezza-native-host-compare-cache')
    a=p.parse_args()
    reference=json.loads(Path(a.reference).read_text())
    expected=next(x['report'] for x in reference['benchmarks'] if x['mode']=='automatic_single')
    spec=request(2048,1048576,120);spec['execution'].pop('max_chunk_configurations')
    nvrtc=ctypes.CDLL('/usr/local/cuda/lib64/libnvrtc.so');major=ctypes.c_int();minor=ctypes.c_int()
    assert nvrtc.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor))==0
    out=dict(started=datetime.now(timezone.utc).isoformat(),host=platform.node(),nvrtc=[major.value,minor.value],
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True).strip(),
        request=spec,request_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()) if (ROOT/'SOURCE_MANIFEST.json').exists() else None,
        binaries={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'build/libodezza.so',ROOT/'build/runtime/libodezza_runtime.so')},runs=[])
    dest=Path(a.output)
    def save():dest.write_text(json.dumps(out,indent=2)+'\n')
    service=NativeService(a.cache,device=a.device)
    try:
        for iteration in range(a.repeats+1):
            start=time.monotonic();job=service.submit(**spec)
            r=service.jobs[job['job_id']].result(timeout=180)
            row=dict(role='warmup' if not iteration else 'measured',report=r,end_to_end_seconds=time.monotonic()-start)
            out['runs'].append(row);save()
            assert r['status']=='complete',r.get('error')
            assert r['counts']['completed_configurations']==2_048_000_000
            assert sum(f['generated_asts'] for f in r['families'])==1_000_000
            assert signature(r)==signature(expected),'Cross-host candidate/leaderboard mismatch'
            row['exact_reference_match']=True
            row['cpu_replays']=[]
            if iteration==1:
                for identifier in r['leaderboards']['global'][:3]:
                    replay=service.replay(job['job_id'],identifier,cpu=True)
                    gpu=r['candidates'][identifier]['mse'];cpu=replay['cpu_reference']['mse']
                    assert abs(gpu-cpu)<3e-10+3e-5*abs(cpu)
                    row['cpu_replays'].append(replay)
            service.release(job_id=job['job_id']);save()
            print(iteration,row['role'],r['timing'],flush=True)
        rows=[r for r in out['runs'] if r['role']=='measured']
        t=statistics.median(r['report']['timing']['total_seconds'] for r in rows)
        out['summary']=dict(median_seconds=t,systems_per_second=1e6/t,configurations_per_second=2048e6/t,
                            all_exact_reference_match=True,measured_repeats=len(rows))
        out['completed']=datetime.now(timezone.utc).isoformat();save();print(out['summary'],flush=True)
    finally:service.close()


if __name__=='__main__':main()
