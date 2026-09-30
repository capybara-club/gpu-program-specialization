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
"""Lifecycle, batching, device-group and worker-recovery regression checks."""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import statistics
import threading
import time
from unittest.mock import patch

from odezza.grammar.native_service import NativeService
from odezza.grammar.native_supervisor import SupervisedNativeService, _rss_bytes
from odezza.grammar.mcp import serve, MAX_MESSAGE_BYTES
from gpu_conformance import problem, grammar
from configuration_scale import request


def signature(report):
    return {key:report[key] for key in ('candidates','leaderboards')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--soak-jobs',type=int,default=1000)
    parser.add_argument('--cache',default='/tmp/odezza-runtime-cache');parser.add_argument('--skip-scale',action='store_true')
    args=parser.parse_args();out=dict(checks=[],benchmarks=[])
    def check(name,**fields):
        out['checks'].append(dict(name=name,**fields));print('PASS',name,flush=True)
        Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    def run(service, spec=None):
        job=service.submit(**(spec or dict(problem=problem(),grammar=grammar())))
        report=service.jobs[job['job_id']].result(timeout=120)
        assert report['status']=='complete',report
        return job['job_id'],report
    service=NativeService(args.cache,max_jobs=2,max_pending=1)
    try:
        for i in range(50):
            identifier,report=run(service)
            assert report['counts']['completed_configurations']==6
            service.release(job_id=identifier)
        assert not service.handles and not service.requests and not service.jobs
        check('50 jobs through two-handle service with explicit release')
        gate=threading.Event();blocker=service.executor.submit(gate.wait)
        identifier=service.submit(problem=problem(),grammar=grammar(),idempotency_key='reuse')['job_id']
        assert service.submit(problem=problem(),grammar=grammar(),idempotency_key='reuse')['job_id']==identifier
        with patch.object(service.lib,'odz_job_report',side_effect=AssertionError('full report used for status')):
            latencies=[]
            for _ in range(100):
                start=time.perf_counter();assert service.status(identifier)['status']=='queued';latencies.append(time.perf_counter()-start)
        for operation in (lambda:service.release(job_id=identifier),lambda:service.submit(problem=problem(),grammar=grammar())):
            try:operation();raise AssertionError('active release/admission unexpectedly accepted')
            except ValueError:pass
        service.cancel(identifier);gate.set();blocker.result()
        assert service.jobs[identifier].result()['status']=='cancelled'
        service.release(job_id=identifier)
        identifier=service.submit(problem=problem(),grammar=grammar(),idempotency_key='reuse')['job_id']
        assert service.jobs[identifier].result()['status']=='complete';service.release(job_id=identifier)
        check('cheap status, pending admission, active release, cancellation and idempotency lifetime',status_median_seconds=statistics.median(latencies))
        prepared=service.prepare(problem())['problem_id'];service.release(problem_id=prepared)
        assert not service.problems
        check('prepared input release')
        before=_rss_bytes();start=time.monotonic()
        for i in range(args.soak_jobs):
            identifier,_=run(service)
            service.release(job_id=identifier)
        growth=_rss_bytes()-before
        assert growth<64*1024**2,(growth,args.soak_jobs)
        assert not any((service.requests,service.jobs,service.handles,service.completed,service.keys))
        check('repeated job release soak',jobs=args.soak_jobs,rss_growth_bytes=growth,seconds=time.monotonic()-start)
    finally:service.close()
    service=NativeService(args.cache,completed_ttl=.1,max_jobs=1)
    try:
        identifier,_=run(service)
        time.sleep(.25)
        assert identifier not in service.handles
        identifier,_=run(service)
        check('automatic idle expiration and handle reuse')
    finally:service.close()
    # One oversized message must not destroy the transport or eat the next line.
    source=io.StringIO(' '* (MAX_MESSAGE_BYTES+1)+'\n'+json.dumps(dict(jsonrpc='2.0',id=7,method='ping'))+'\n')
    output=io.StringIO();serve(object(),source,output)
    responses=[json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]['error']['data']['code']=='message_too_large' and responses[1]['id']==7
    check('oversized MCP message drains and next request succeeds')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='odezza-group-cold-') as cold:
        cold_service=NativeService(cold,devices=[0,1])
        try:
            spec=dict(problem=problem(),grammar=grammar(),execution={'batch_variants':1})
            spec['grammar']['rhs']['x0']='R'
            spec['grammar']['rules']={'R':['-const.k*leaf.s','-sin(const.k*leaf.s)']}
            identifier,r=run(cold_service,spec)
            assert r['cache']['misses']==1 and r['cache']['hits']==1,r['cache']
            check('cold scoring template compiled once for two GPUs',cache=r['cache'],timing=r['timing'])
        finally:cold_service.close()
    baseline=NativeService(args.cache)
    grouped=NativeService(args.cache,devices=[0,1])
    try:
        spec=dict(problem=problem(),grammar=grammar(),execution={'batch_variants':1})
        g=spec['grammar'];g['rules']={'R':['-const.k*leaf.s','-sin(const.k*leaf.s)','-tanh(const.k*leaf.s)',{'expr':'-const.k*leaf.s','tags':['late']}]}
        g['rhs']['x0']='R';g['retain']['by_tag']={'late':3}
        a,ra=run(baseline,spec);b,rb=run(grouped,spec)
        assert signature(ra)==signature(rb),(ra,rb)
        assert sum(d['completed_configurations'] for d in rb['execution']['devices'])==rb['counts']['completed_configurations']
        for identifier in rb['candidates']:grouped.replay(b,identifier,cpu=True)
        check('two-device exact winners and late tag provenance',devices=rb['execution']['devices'])
        baseline.release(job_id=a);grouped.release(job_id=b)
        from gpu_conformance import ROOT
        mixed=json.loads((ROOT/'examples/grammar/constant_rng_product.json').read_text())
        a,ra=run(baseline,dict(problem=problem(),grammar=mixed))
        b,rb=run(grouped,dict(problem=problem(),grammar=mixed,execution={'batch_variants':1,'max_chunk_configurations':17}))
        assert signature(ra)==signature(rb)
        check('Philox/grid/toggle snapshots independent of GPU dispatch and tiling')
        baseline.release(job_id=a);grouped.release(job_id=b)
        if not args.skip_scale:
            comparisons=[]
            for service,mode in ((baseline,'explicit_1m'),(baseline,'automatic_single'),(grouped,'automatic_dual')):
                for repeat in range(3):
                    spec=request(2048,1048576,120)
                    if mode!='explicit_1m':spec['execution'].pop('max_chunk_configurations')
                    identifier,report=run(service,spec)
                    assert report['counts']['completed_configurations']==2_048_000_000
                    assert sum(f['generated_asts'] for f in report['families'])==1_000_000
                    if comparisons:assert signature(report)==signature(comparisons[0])
                    comparisons.append(report)
                    out['benchmarks'].append(dict(mode=mode,repeat=repeat,report=report))
                    service.release(job_id=identifier)
                    check(mode+str(repeat),seconds=report['timing']['total_seconds'],devices=report['execution']['devices'])
    finally:baseline.close();grouped.close()
    supervised=SupervisedNativeService(args.cache,watchdog_seconds=2,deadline_grace_seconds=2,devices=[0,1])
    try:
        job=supervised.submit(problem=problem(),grammar=grammar())
        report=supervised.wait(job['job_id'],timeout=30);assert report['status']=='complete',report
        supervised.release(job_id=job['job_id'])
        check('supervised two-device submit/status/results/release')
        # Freeze only our own worker to reproduce an unresponsive CUDA/control process.
        pid=supervised.process.pid
        os.kill(pid,signal.SIGSTOP)
        deadline=time.monotonic()+8
        while supervised.process is not None and time.monotonic()<deadline:time.sleep(.05)
        assert supervised.process is None,'watchdog did not terminate frozen worker'
        job=supervised.submit(problem=problem(),grammar=grammar())
        report=supervised.wait(job['job_id'],timeout=30);assert report['status']=='complete',report
        assert supervised.process.pid!=pid
        check('watchdog kills stuck worker and next submission starts a fresh context')
        supervised.release(job_id=job['job_id'])
        supervised.max_worker_rss_bytes=1
        deadline=time.monotonic()+5
        while supervised.process is not None and time.monotonic()<deadline:time.sleep(.05)
        assert supervised.process is None,'RSS ceiling did not terminate worker'
        supervised.max_worker_rss_bytes=4*1024**3
        job=supervised.submit(problem=problem(),grammar=grammar())
        assert supervised.wait(job['job_id'],timeout=30)['status']=='complete'
        supervised.release(job_id=job['job_id'])
        check('process RSS ceiling and restart')
    finally:supervised.close()
    out['passed']=True;Path(args.output).write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
