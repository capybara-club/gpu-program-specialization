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
"""Bounded audit reproductions; records existing defects, not a passing CI suite.

CPU mode calls the real service methods with fake native handles. GPU mode uses
the real runtime and an explicitly preloaded, one-shot upload-failure shim.
"""
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes as C
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from odezza.grammar.native_service import NativeService, ServiceError
from odezza.grammar.native_supervisor import SupervisedNativeService


def controls():
    supervisor = SupervisedNativeService.__new__(SupervisedNativeService)
    supervisor.lock = threading.RLock()
    supervisor.current_jobs = set()
    supervisor.deadlines = {}
    supervisor._rpc = Mock(return_value={'job_id': 'audit'})
    supervisor.submit(problem={}, grammar={}, execution={'max_seconds': 1800})
    deadline = supervisor.deadlines['audit']
    supervisor.stop = Mock()
    supervisor.stop.wait.side_effect = [False, True]
    supervisor.closed = False
    supervisor.options = {}
    supervisor.error_times = {}
    supervisor.terminal = {}
    supervisor.errors = {}
    supervisor.process = Mock()
    supervisor.process.is_alive.return_value = True
    supervisor.last_health = time.monotonic()
    supervisor.watchdog_seconds = 30
    supervisor.deadline_grace_seconds = 10
    supervisor.max_worker_rss_bytes = supervisor.max_total_rss_bytes = 10**12
    supervisor.health = {'rss_bytes': 1024, 'statuses': {
        'audit': {'status': 'running', 'timing': {'total_seconds': 611}}}}
    supervisor._lost = Mock()
    with patch('odezza.grammar.native_supervisor._rss_bytes', return_value=1024):
        supervisor._monitor()

    service = NativeService.__new__(NativeService)
    service.lock = threading.RLock()
    service.handles = {'audit': 1}
    service.failures = {}
    service.worker_started = {}
    service.transport = {}
    service.jobs = {'audit': Future()}
    entered, release = threading.Event(), threading.Event()
    payload = b'{"status":"complete","timing":{}}'

    def native_report(handle, output, capacity, required):
        entered.set()
        if not release.wait(5):
            raise TimeoutError('audit gate')
        return native_status(handle, output, capacity, required)

    def native_status(handle, output, capacity, required):
        C.cast(required, C.POINTER(C.c_size_t))[0] = len(payload)+1
        if output is not None:
            C.memmove(output, payload+b'\0', len(payload)+1)
        return 0

    service.lib = SimpleNamespace(odz_job_report=native_report, odz_job_status=native_status)
    with ThreadPoolExecutor(2) as threads:
        report = threads.submit(service._report, 'audit')
        assert entered.wait(5)
        status_started = threading.Event()
        def status_call():
            status_started.set()
            return service.status('audit')
        status = threads.submit(status_call)
        assert status_started.wait(5)
        time.sleep(.1)
        blocked = not status.done()
        release.set()
        report.result(timeout=5)
        status_result = status.result(timeout=5)
    try:
        service.release(job_id='audit')
    except ServiceError as exc:
        release_error = exc.code
    else:
        release_error = None
    return dict(evidence='real Python methods; fake CUDA handles and one monitor tick',
                requested_seconds=1800, supervisor_deadline=deadline,
                killed_at_simulated_elapsed_611=supervisor._lost.called,
                kill_reason=supervisor._lost.call_args.args[0] if supervisor._lost.called else None,
                status_blocked_by_full_report=blocked,
                native_status=status_result['status'], immediate_release_error=release_error)


def gpu():
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root/'examples/grammar/submit.json').read_text())
    request['grammar']['constants']['k']['values'] = [.5+i/64 for i in range(65)]
    request['grammar']['retain'] = {'global': {'k': 1, 'unit': 'numeric_candidate'}}
    runs = []
    for fresh in (False, True):
        service = NativeService('/tmp/odezza-runtime-cache')
        try:
            for name in (['fresh_runtime_control'] if fresh else ['injected_upload_failure', 'same_runtime_after_failure']):
                job = service.submit(**request)
                report = service.jobs[job['job_id']].result(timeout=90)
                runs.append(dict(name=name, report=report))
        finally:
            service.close()
    return dict(evidence='real CUDA; one-shot 260-byte grid upload failure, zeroed destination', runs=runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = gpu() if args.gpu else controls()
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result if not args.gpu else [
        dict(name=r['name'], status=r['report']['status'], error=r['report']['error'],
             quarantined=r['report']['runtime_quarantined'], candidates=r['report']['candidates'])
        for r in result['runs']], indent=2))
