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
"""Process boundary for the native service. No search, grammar or cache policy.

The worker owns CUDA and all job data. The parent owns admission/handle lifetimes
and terminates a worker after an unfenced CUDA error, RSS ceiling or watchdog
expiry. Interrupted jobs fail explicitly and are never silently resubmitted.
"""
from __future__ import annotations
from concurrent.futures import Future, TimeoutError as FutureTimeout
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import threading
import time

from .native_service import NativeService, ServiceError, capabilities


def _rss_bytes():
    try:
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if __import__('sys').platform == 'darwin' else peak*1024)


def _worker(connection, root, options):
    service = None
    try:
        service = NativeService(root, **options)
        last_health = 0.
        while True:
            if connection.poll(.1):
                identifier, method, arguments = connection.recv()
                if method == '_close':
                    service.close()
                    connection.send(('reply', identifier, True, None))
                    break
                try:
                    value = getattr(service, method)(**arguments)
                    connection.send(('reply', identifier, True, value))
                except Exception as exc:
                    connection.send(('reply', identifier, False,
                                     dict(type=type(exc).__name__, message=str(exc),code=getattr(exc,"code",None) or ("unknown_handle" if isinstance(exc,KeyError) else "invalid_request"))))
            now = time.monotonic()
            if now-last_health>=.1:
                with service.lock:
                    statuses = {identifier: service.status(identifier) for identifier in service.jobs}
                connection.send(('health', dict(rss_bytes=_rss_bytes(), statuses=statuses)))
                last_health = now
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(('fatal', str(exc)))
        except (OSError, EOFError):
            pass
    finally:
        # Normal shutdown explicitly closes first. On an unexpected control-plane
        # failure, process exit fences ownership; do not block in CUDA cleanup.
        connection.close()
        if service is not None and not service.closed:
            os._exit(70)


class WorkerFailure(RuntimeError):
    """All live handles in the lost worker generation are invalid."""
    code = "worker_lost"


class SupervisedNativeService:
    supported_tools = NativeService.supported_tools

    def __init__(self, root, *, max_worker_rss_bytes=4*1024**3,
                 max_total_rss_bytes=8*1024**3, watchdog_seconds=30., deadline_grace_seconds=10., **options):
        if min(max_worker_rss_bytes,max_total_rss_bytes,watchdog_seconds,deadline_grace_seconds)<=0:
            raise ValueError('Supervisor limits must be positive')
        self.root,self.options=root,options
        self.max_worker_rss_bytes=max_worker_rss_bytes
        self.max_total_rss_bytes=max_total_rss_bytes
        self.watchdog_seconds,self.deadline_grace_seconds=watchdog_seconds,deadline_grace_seconds
        self.lock=threading.RLock()
        self.send_lock=threading.Lock()
        self.pending={}
        self.terminal={}
        self.deadlines={}
        self.errors={}
        self.error_times={}
        self.current_jobs=set()
        self.sequence=0
        self.generation=0
        self.process=None
        self.connection=None
        self.closed=False
        self.last_health=0.
        self.health={}
        self.stop=threading.Event()
        self._start()
        self.monitor=threading.Thread(target=self._monitor,name='odezza-supervisor',daemon=True)
        self.monitor.start()

    def capabilities(self):
        return dict(capabilities(), process_supervision=True, devices=self.options.get('devices') or
                    [self.options.get('device',0)], max_worker_rss_bytes=self.max_worker_rss_bytes, max_total_rss_bytes=self.max_total_rss_bytes,
                    completed_job_ttl_seconds=self.options.get('completed_ttl',3600),
                    max_pending_jobs=self.options.get('max_pending',8),
                    failure_policy='Lost worker handles fail; no automatic job replay; next submit restarts worker')

    def _start(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('Service is closed')
            context=mp.get_context('spawn')
            parent,child=context.Pipe()
            self.generation+=1
            generation=self.generation
            self.connection=parent
            self.process=context.Process(target=_worker,args=(child,self.root,self.options),
                                         name='odezza-native-worker',daemon=True)
            self.process.start()
            child.close()
            self.last_health=time.monotonic()
            self.health={}
            self.reader=threading.Thread(target=self._read,args=(parent,generation),daemon=True)
            self.reader.start()

    def _read(self, connection, generation):
        try:
            while True:
                message=connection.recv()
                with self.lock:
                    if generation!=self.generation:
                        return
                    if message[0]=='reply':
                        _,identifier,ok,value=message
                        future=self.pending.pop(identifier,None)
                        if future:
                            if ok:future.set_result(value)
                            else:future.set_exception(ServiceError(value['code'],value['message']))
                    elif message[0]=='health':
                        self.health=message[1]
                        self.last_health=time.monotonic()
                        for identifier,status in self.health['statuses'].items():
                            if status['status'] in ('complete','cancelled','failed','timeout'):
                                self.terminal[identifier]=status
                    else:
                        self._lost(message[1])
                        return
        except (EOFError,OSError):
            with self.lock:
                if generation==self.generation and not self.closed:
                    self._lost('Native worker exited unexpectedly')

    def _lost(self, reason):
        # Caller holds self.lock. Termination is confined to our own worker.
        if self.process is None:
            return
        process=self.process
        self.process=None
        if process.is_alive():process.kill()
        process.join(timeout=2)
        if self.connection:self.connection.close()
        self.connection=None
        self.generation+=1
        failure=WorkerFailure(reason)
        for future in self.pending.values():
            if not future.done():future.set_exception(failure)
        self.pending.clear()
        for identifier in self.current_jobs:
            self.errors[identifier]=reason
            self.error_times[identifier]=time.monotonic()
        self.current_jobs.clear()

    def _failure(self, identifier):
        previous=self.terminal.get(identifier,{})
        return dict(previous,job_id=identifier,status='failed',error=self.errors[identifier],
                    error_code='worker_lost',results_available=False)

    def _monitor(self):
        while not self.stop.wait(.1):
            with self.lock:
                if self.closed:continue
                now=time.monotonic()
                ttl=self.options.get('completed_ttl',3600)
                expired=[i for i,t in self.error_times.items() if now-t>=ttl]
                oldest=sorted(self.error_times,key=self.error_times.get)
                expired+=oldest[:max(0,len(oldest)-self.options.get('max_jobs',32))]
                live=self.health.get('statuses',{})
                expired += [i for i in self.terminal if i not in live and i not in self.errors]
                for identifier in expired:
                    self.current_jobs.discard(identifier)
                    for table in (self.terminal,self.deadlines,self.errors,self.error_times):table.pop(identifier,None)
                if self.process is None:continue
                reason=None
                if not self.process.is_alive():reason='Native worker exited unexpectedly'
                elif now-self.last_health>self.watchdog_seconds:reason='Native worker heartbeat deadline exceeded'
                elif self.health.get('rss_bytes',0)>self.max_worker_rss_bytes:reason='Native worker RSS ceiling exceeded'
                elif self.health.get('rss_bytes',0)+_rss_bytes()>self.max_total_rss_bytes:reason='Combined service RSS ceiling exceeded'
                for identifier,status in self.health.get('statuses',{}).items():
                    if status.get('runtime_quarantined'):reason='Native worker quarantined after an unfenced CUDA failure'
                    if status['status']=='initializing' and status['timing'].get('worker_seconds',0)>self.watchdog_seconds:reason='CUDA initialization exceeded its watchdog deadline'
                    if status['status'] in ('preparing','running') and status['timing']['total_seconds']>self.deadlines.get(identifier,600)+self.deadline_grace_seconds:
                        reason='Native job exceeded its hard deadline including cleanup grace'
                if reason:self._lost(reason)

    def _rpc(self, method, **arguments):
        with self.lock:
            if self.closed and method!='_close':raise RuntimeError('Service is closed')
            if self.process is None:
                if method in ('submit','prepare'):self._start()
                else:raise WorkerFailure('Worker unavailable; submit a new request to restart')
            self.sequence+=1
            identifier=self.sequence
            future=Future()
            self.pending[identifier]=future
            connection=self.connection
            generation=self.generation
        # Never hold the metadata lock while sending a potentially large payload:
        # the reader must be able to drain replies/heartbeats concurrently.
        try:
            with self.send_lock:connection.send((identifier,method,arguments))
        except (OSError,EOFError) as exc:
            with self.lock:
                if generation==self.generation:self._lost(str(exc))
                elif not future.done():future.set_exception(WorkerFailure(str(exc)))
        try:
            return future.result(timeout=self.watchdog_seconds+5)
        except FutureTimeout:
            with self.lock:
                if generation==self.generation:self._lost('Native control request exceeded its watchdog deadline')
            raise WorkerFailure('Native control request exceeded its watchdog deadline')

    def prepare(self, problem):return self._rpc('prepare',problem=problem)

    def submit(self, problem_id=None, grammar=None, execution=None, **kwargs):
        result=self._rpc('submit',problem_id=problem_id,grammar=grammar,execution=execution,**kwargs)
        identifier=result['job_id']
        with self.lock:
            self.current_jobs.add(identifier)
            limits=(grammar.get('limits') or {}) if isinstance(grammar,dict) else {}
            values=[600.]
            for source in (execution,limits):
                value=source.get('max_seconds') if isinstance(source,dict) else None
                if isinstance(value,(float,int)) and not isinstance(value,bool) and 0<value<float('inf'):
                    values.append(value)
            self.deadlines[identifier]=min(values)
        return result

    def status(self, job_id):
        with self.lock:
            if job_id in self.errors:return self._failure(job_id)
        return self._rpc('status',job_id=job_id)

    def results(self, job_id, format='compact'):
        with self.lock:
            if job_id in self.errors:return self._failure(job_id)
        return self._rpc('results',job_id=job_id,format=format)

    def wait(self, job_id, timeout=None):
        deadline=time.monotonic()+timeout if timeout is not None else float('inf')
        while time.monotonic()<deadline:
            if self.status(job_id)['status'] in ('complete','cancelled','failed','timeout'):
                return self.results(job_id)
            time.sleep(.02)
        raise TimeoutError('Job wait timed out')

    def cancel(self, job_id):return self._rpc('cancel',job_id=job_id)
    def replay(self, job_id, candidate_id=None, **kwargs):
        return self._rpc('replay',job_id=job_id,candidate_id=candidate_id,**kwargs)

    def release(self, job_id=None, problem_id=None):
        with self.lock:
            lost=job_id in self.errors if job_id is not None else False
        result=dict(released=True,job_id=job_id) if lost else self._rpc('release',job_id=job_id,problem_id=problem_id)
        if job_id is not None:
            with self.lock:
                self.current_jobs.discard(job_id)
                for table in (self.terminal,self.deadlines,self.errors,self.error_times):table.pop(job_id,None)
        return result

    def close(self):
        self.stop.set()
        with self.lock:
            if self.closed:return
            self.closed=True
            process=self.process
        if process is not None:
            try:self._rpc('_close')
            except Exception:pass
            with self.lock:self._lost('Service closed')
        self.monitor.join(timeout=2)
        with self.lock:
            for table in (self.terminal,self.deadlines,self.errors,self.error_times):table.clear()
            self.current_jobs.clear()
