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
"""Isolated real-CUDA/NATS disconnect, cancellation, and integration-cap checks.

Run on rack1 against an otherwise empty isolated service. Starts only its own
worker supervisor and TCP fault proxy. Baseline and fault workloads are identical.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import threading
import time
from client import Client
ROOT=Path(__file__).resolve().parents[1]

class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True
    def __init__(self, port, upstream):
        self.upstream=upstream
        self.lock=threading.Lock()
        self.peers=set()
        self.blocked=False
        super().__init__(('127.0.0.1',port),Handler)
    def outage(self,seconds):
        with self.lock:
            self.blocked=True
            for s in tuple(self.peers):
                try:s.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        time.sleep(seconds)
        with self.lock:self.blocked=False
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        proxy=self.server
        with proxy.lock:
            if proxy.blocked:return
            upstream=socket.create_connection(('127.0.0.1',proxy.upstream),3)
            proxy.peers.update((self.request,upstream))
        def pump(src,dst):
            try:
                while data:=src.recv(65536):dst.sendall(data)
            except OSError:pass
            finally:
                try:dst.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        thread=threading.Thread(target=pump,args=(self.request,upstream),daemon=True);thread.start()
        pump(upstream,self.request);thread.join(3)
        with proxy.lock:proxy.peers.difference_update((self.request,upstream))
        upstream.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=15223)
    p.add_argument('--proxy-port',type=int,default=15226)
    p.add_argument('--output',type=Path,default=ROOT/'service_trial/validation/control-gpu.json')
    a=p.parse_args();proxy=Proxy(a.proxy_port,a.port)
    thread=threading.Thread(target=proxy.serve_forever,daemon=True);thread.start()
    env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda/lib64',CUDA_MODULE_LOADING='EAGER')
    log=(ROOT/'build/service_trial/control-gpu-worker.log').open('wb')
    worker=subprocess.Popen(['python3','service_trial/supervise_worker.py','--devices','0,1','--url',f'nats://127.0.0.1:{a.proxy_port}'],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
    c=Client(port=a.port);handles=[];records=[]
    def record(row):
        records.append(row)
        a.output.write_text(json.dumps(dict(passed=False,in_progress=True,records=records),indent=2)+"\n")
        print(json.dumps(dict(check=row["check"],seconds=row.get("seconds"))),flush=True)
    def submit(request):
        h=c.submit(json.dumps(request,separators=(',',':')).encode())['handle'];handles.append(h);return h
    def wait(h,progress=False,timeout=90):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            s=c.status(h)
            if s['state']=='terminal':
                assert not progress,('completed before fault was applied',s)
                return s
            if progress and sum(s['observed_configurations_by_attempt'])>0:return s
            time.sleep(.02)
        raise AssertionError(('job did not finish',c.status(h)))
    def result(h):
        s=wait(h);r=c.result(h);c.rpc('release.'+h);handles.remove(h);return s,r
    try:
        # Exact published reference: original 2.048B configuration request.
        original=json.loads((ROOT/'service_trial/validation/million-2048-request.json').read_text())
        ref=json.loads((ROOT/'service_trial/validation/million-direct-attempt.json').read_text())[-1]['report']
        start=time.monotonic();s,r=result(submit(original))
        assert r['status']=='complete',r.get('error')
        for k in ('candidates','leaderboards'):assert r[k]==ref[k],k
        assert len(r['families'])==len(ref['families'])
        for actual,expected in zip(r['families'],ref['families']):
            assert {k:actual[k] for k in expected}==expected
            assert actual['max_configurations_per_skeleton']==0 and actual['configuration_limited_derivations']==0
        record(dict(check='original_2.048B_exact_reference',seconds=time.monotonic()-start,status=s,report=r))
        large=copy.deepcopy(original)
        large['grammar']['constants']['c']['values']*=32
        large['grammar']['limits']['max_configurations']*=32
        start=time.monotonic();s,baseline=result(submit(large))
        assert baseline['status']=='complete',baseline.get('error')
        record(dict(check='65.536B_fault_baseline',seconds=time.monotonic()-start,status=s,report=baseline))
        for seconds in (.15,6):
            start=time.monotonic();h=submit(large);before=wait(h,progress=True)
            proxy.outage(seconds)
            after,r=result(h)
            assert r['status']=='complete',r.get('error')
            for k in ('candidates','leaderboards','families'):assert r[k]==baseline[k],(seconds,k)
            assert r['counts']['completed_configurations']==baseline['counts']['completed_configurations']
            if seconds<1:
                assert after['attempts']==1 and before['worker_info']['pid']==after['worker_info']['pid'],after
            else:
                assert after['attempts']==2 and before['worker_info']['pid']!=after['worker_info']['pid'],after
            record(dict(check='disconnect_recovery',outage_seconds=seconds,seconds=time.monotonic()-start,before=before,after=after,report=r))
        h=submit(large);before=wait(h,progress=True);start=time.monotonic();c.rpc('cancel.'+h)
        after,r=result(h);assert r['status']=='cancelled',r
        record(dict(check='cancel_during_gpu_work',seconds=time.monotonic()-start,before=before,after=after,report=r))
        # Deliberately huge explicit configuration tile: work cap must split it.
        capped=dict(problem=dict(states=['x'],trajectories=[dict(initial=[0],times=[0,1],values=[[0],[0]])]),
                    grammar=dict(version=1,states=['x'],rhs={'x':'-rng.c*x'},rng_banks={'u':{'base':'uniform01','count':524288,'seed':713}},rng={'c':{'bank':'u'}},
                                 integration=dict(method='rk4',dt=1/4096),retain={'global':{'k':1,'unit':'numeric_candidate'}}),
                    execution=dict(max_chunk_configurations=4294967296,max_seconds=120))
        start=time.monotonic();s,r=result(submit(capped))
        assert r['status']=='complete',r.get('error')
        assert r['counts']['completed_configurations']==524288,r['counts']
        for d in r['execution']['devices']:
            assert d['largest_tile_configurations']<=d['sizing']['work_ceiling_configurations'],d
        assert next(iter(r['candidates'].values()))['mse']==0,r['candidates']
        record(dict(check='explicit_tile_override_still_bounded_all_configs_scored',seconds=time.monotonic()-start,status=s,report=r))
        s,r=result(submit(original));assert r['status']=='complete'
        for k in ('candidates','leaderboards','families'):assert r[k]==ref[k],k
        record(dict(check='worker_pool_reuse_after_faults_and_longer_rollout',status=s,report=r))
        a.output.write_text(json.dumps(dict(passed=True,records=records),indent=2)+'\n')
        print(json.dumps([dict(check=x['check'],seconds=x.get('seconds'),outage_seconds=x.get('outage_seconds')) for x in records]))
    except BaseException as e:
        a.output.write_text(json.dumps(dict(passed=False,error=str(e),records=records),indent=2)+"\n")
        raise
    finally:
        for h in handles:
            try:c.rpc('cancel.'+h)
            except Exception:pass
        worker.send_signal(signal.SIGTERM)
        try:worker.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        for h in handles:
            try:wait(h,timeout=25);c.rpc('release.'+h)
            except Exception:pass
        c.close();proxy.shutdown();proxy.server_close();log.close()
if __name__=='__main__':main()
