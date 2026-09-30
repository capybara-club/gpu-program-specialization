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
"""Longer rollouts, auto memory tiling, live polling/cancel and worker loss."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from configuration_scale import request
from service_upgrade import signature
from gpu_conformance import problem,grammar
from odezza.grammar.native_service import NativeService
from odezza.grammar.native_supervisor import SupervisedNativeService


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    records=[]
    def save(name,**fields):
        records.append(dict(name=name,**fields));Path(a.output).write_text(json.dumps(records,indent=2)+'\n');print('PASS',name,flush=True)
    spec=request(2048,1048576,120)
    spec['grammar']['limits'].update(max_skeletons=4096,max_variants=4096,max_configurations=4096*2048)
    spec['grammar']['integration']['dt']=.005
    ts=[i*.02 for i in range(26)]
    spec['problem']['trajectories']=[dict(initial=[initial]*6,times=ts,
            values=[[initial*math.exp(t)]*6 for t in ts]) for initial in (.1,.12,.14,.16)]
    service=NativeService('/tmp/odezza-runtime-cache',devices=[0,1])
    try:
        results=[]
        for mode in ('fixed','automatic','memory_bound'):
            q=deepcopy(spec)
            if mode!='fixed':q['execution'].pop('max_chunk_configurations')
            if mode=='memory_bound':q['execution']['max_device_bytes']=1024*1024
            start=time.monotonic();job=service.submit(**q);polls=[]
            while not service.jobs[job['job_id']].done():
                t=time.monotonic();status=service.status(job['job_id']);polls.append(time.monotonic()-t)
                time.sleep(.001)
            report=service.jobs[job['job_id']].result(timeout=60)
            save(mode+'_raw',report=report)
            assert report['status']=='complete',report
            assert report['counts']['completed_configurations']==4096*2048
            if results:assert signature(report)==signature(results[0])
            results.append(report)
            save(mode,report=report,poll_count=len(polls),max_poll_seconds=max(polls or [0]),wall_seconds=time.monotonic()-start)
            service.release(job_id=job['job_id'])
        q=request(2048,1048576,120)
        job=service.submit(**q)
        deadline=time.monotonic()+10
        while service.status(job['job_id'])['counts']['completed_configurations']==0 and time.monotonic()<deadline:time.sleep(.002)
        start=time.monotonic();service.cancel(job['job_id'])
        report=service.jobs[job['job_id']].result(timeout=10)
        assert report['status']=='cancelled' and report['counts']['completed_configurations']<2_048_000_000
        save('cancel while two GPUs score',cancel_seconds=time.monotonic()-start,report=report)
    finally:service.close()
    service=SupervisedNativeService('/tmp/odezza-runtime-cache',devices=[0,1])
    try:
        # Warm contexts, then lose a worker during an active multi-billion job.
        job=service.submit(problem=problem(),grammar=grammar());assert service.wait(job['job_id'],30)['status']=='complete'
        service.release(job_id=job['job_id'])
        from concurrent.futures import ThreadPoolExecutor
        def transport_payload(index):
            prepared=service.prepare(dict(test_payload='x'*(2*1024*1024),index=index))
            service.release(problem_id=prepared['problem_id'])
        with ThreadPoolExecutor(4) as callers:list(callers.map(transport_payload,range(8)))
        save('concurrent large control messages do not block reply reader')
        q=request(2048,1048576,120);job=service.submit(**q)
        deadline=time.monotonic()+10
        while service.status(job['job_id'])['status']=='queued' and time.monotonic()<deadline:time.sleep(.005)
        pid=service.process.pid;service.process.kill()
        deadline=time.monotonic()+5
        while service.process is not None and time.monotonic()<deadline:time.sleep(.01)
        failed=service.status(job['job_id'])
        assert failed['status']=='failed' and failed['error_code']=='worker_lost' and not failed['results_available'],failed
        replacement=service.submit(problem=problem(),grammar=grammar())
        assert service.wait(replacement['job_id'],30)['status']=='complete'
        assert replacement['job_id']!=job['job_id'] and service.process.pid!=pid
        # A new successful job must not turn the failed one into a success.
        assert service.status(job['job_id'])['error_code']=='worker_lost'
        service.release(job_id=job['job_id']);service.release(job_id=replacement['job_id'])
        save('active worker crash is explicit and never replayed',failed=failed)
    finally:service.close()


if __name__=='__main__':main()
