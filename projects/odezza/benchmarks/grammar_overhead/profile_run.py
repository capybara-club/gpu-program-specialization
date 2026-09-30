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
"""Matched direct-grammar profiles; production modules and native core are unchanged.

Run on a CUDA host with the repository's PYTHONPATH and shared library. Detailed
instrumentation is scoped to the service worker, including generator iteration.
Native CUDA driver calls internal to C99 require the separate Nsight trace.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import cProfile
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import pstats
import sqlite3
import time

from odezza.grammar import cuda, executor, native_scoring, prelude, service
from odegrammar import compiler

ROOT = Path(__file__).resolve().parents[2]


class Timings:
    """Nested synchronous wall spans; exclusive totals form a partition."""
    def __init__(self):
        self.stack, self.patches = [], []
        self.rows = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
        self.bytes = defaultdict(int)

    @contextmanager
    def span(self, name):
        frame = [time.perf_counter(), 0., name]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.rows[name]
            row['calls'] += 1
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += elapsed-frame[1]

    def wrap(self, owner, attribute, label, *, generator=False):
        original = getattr(owner, attribute)
        if generator:
            def wrapped(*args, **kwargs):
                iterator = original(*args, **kwargs)
                try:
                    while True:
                        with self.span(label):
                            try:
                                item = next(iterator)
                            except StopIteration:
                                return
                        yield item
                finally:
                    iterator.close()
        else:
            def wrapped(*args, **kwargs):
                name = label(*args, **kwargs) if callable(label) else label
                if owner is cuda.CUDA and attribute == 'call':
                    api = args[1]
                    if api in ('cuMemcpyHtoDAsync_v2', 'cuMemcpyDtoHAsync_v2'):
                        self.bytes[api] += int(args[4])
                    elif api in ('cuMemAlloc_v2', 'cuMemAllocHost_v2'):
                        self.bytes[api] += int(args[3])
                with self.span(name):
                    return original(*args, **kwargs)
        self.patches.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)

    def install(self):
        collector = self
        original_connect = sqlite3.connect

        class TimedConnection(sqlite3.Connection):
            def section(self):
                names = [frame[2] for frame in collector.stack]
                return 'retention' if 'plan.retain' in names else 'compile' if 'plan.compile' in names else 'other'

            def execute(self, sql, parameters=()):
                with collector.span('sqlite.'+self.section()+'.'+sql.lstrip().split()[0].lower()):
                    return super().execute(sql, parameters)

            def commit(self):
                with collector.span('sqlite.'+self.section()+'.commit'):
                    return super().commit()

        def connect(*args, **kwargs):
            kwargs.setdefault('factory', TimedConnection)
            return original_connect(*args, **kwargs)

        self.patches.append((sqlite3, 'connect', original_connect))
        sqlite3.connect = connect
        for owner, names, prefix in [
            (service.Plan, ['compile', 'retain', 'reports', 'skeleton', 'variant'], 'plan'),
            (executor.Executor, ['__init__', 'initialize', 'pipeline', 'data', 'score_batch'], 'executor'),
            (native_scoring.Pipeline, ['run'], 'native_pipeline'),
            (native_scoring.Reducer, ['__init__', 'run'], 'reducer'),
            (prelude.Prelude, ['__init__', 'descriptors', 'bank', 'run', 'trim'], 'prelude'),
            (cuda.Buffer, ['__init__', 'read', 'close'], 'buffer'),
        ]:
            for name in names:
                self.wrap(owner, name, prefix+'.'+name)
        self.wrap(service.Plan, 'batches', 'plan.batch_next', generator=True)
        self.wrap(compiler.Compiler, 'records', 'compiler.record_next', generator=True)
        self.wrap(service, 'lower_system', 'plan.lower_system')
        self.wrap(service, 'atomic', 'service.atomic_json_file')
        self.wrap(executor, 'identities', 'winner.identities')
        self.wrap(executor, 'content_id', 'executor.content_id')
        self.wrap(cuda, 'compile_cuda', 'prelude.nvrtc_compile')
        self.wrap(cuda.CUDA, 'call', lambda owner, name, *args: 'cuda.'+name)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)


def stats_json(profile):
    rows = []
    for (filename, line, function), (primitive, calls, exclusive, inclusive, callers) in pstats.Stats(profile).stats.items():
        rows.append(dict(file=filename, line=line, function=function, calls=calls,
                         primitive_calls=primitive, exclusive_seconds=exclusive,
                         inclusive_seconds=inclusive))
    return sorted(rows, key=lambda row: -row['exclusive_seconds'])


class MeasuredService(service.Service):
    mode = 'baseline'

    def _run(self, directory, payload, problem, event):
        mode = self.mode
        collector = Timings() if mode == 'spans' else None
        profiler = cProfile.Profile() if mode == 'cprofile' else None
        driver = None
        if mode == 'trace':
            # Nsight can capture exactly this job after a full warmup job.
            driver = ctypes.CDLL('libcuda.so.1')
            driver.cuProfilerStart.restype = driver.cuProfilerStop.restype = ctypes.c_int
            if driver.cuProfilerStart() != 0:
                raise RuntimeError('Could not start CUDA profiler capture')
        if collector:
            collector.install()
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            if profiler:
                profiler.enable()
            if collector:
                with collector.span('worker_job'):
                    report = super()._run(directory, payload, problem, event)
            else:
                report = super()._run(directory, payload, problem, event)
        finally:
            if profiler:
                profiler.disable()
            elapsed, cpu_elapsed = time.perf_counter()-wall, time.process_time()-cpu
            if collector:
                collector.restore()
            if driver and driver.cuProfilerStop() != 0:
                raise RuntimeError('Could not stop CUDA profiler capture')
        detail = dict(mode=mode, worker_seconds=elapsed, process_cpu_seconds=cpu_elapsed)
        if collector:
            detail.update(spans=dict(collector.rows), bytes=dict(collector.bytes))
        if profiler:
            profiler.dump_stats(str(directory/'worker.prof'))
            detail['functions'] = stats_json(profiler)
        service.atomic(directory/'measurement.json', detail)
        return report


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--modes', nargs='+', choices=['baseline', 'spans', 'cprofile', 'trace'],
                        default=['baseline', 'baseline', 'spans', 'baseline', 'spans', 'cprofile', 'baseline'])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('grammar_fixtures', ROOT/'tests/grammar/gpu_examples.py')
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    request = json.loads((ROOT/'examples/grammar/adaptation_search.json').read_text())
    problem = fixtures.fixture(request)
    options = dict(patch_capacity=1024, module_systems=32, max_seconds=1200)
    args.root.mkdir(parents=True, exist_ok=False)
    service.atomic(args.root/'inputs.json', dict(grammar=request, problem=problem, execution=options))
    service.atomic(args.root/'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ['python/odegrammar', 'python/odezza/grammar'] for p in sorted((ROOT/folder).glob('*.py'))},
        profiling='spans are nested wall timers; cProfile overhead is not normal runtime; trace uses CUDA profiler range',
        cache='first job starts new process/context; disk/driver caches are not flushed; later jobs reuse executor/templates'))
    results = []
    engine = MeasuredService(args.root/'service', library=args.library, device=args.device)
    try:
        handle = engine.prepare(problem)
        modes = ['baseline']+args.modes
        expected_hash = None
        for index, mode in enumerate(modes):
            engine.mode = mode
            job = engine.submit(handle['problem_id'], request, options)
            report = engine.jobs[job['job_id']].result()
            folder = engine.job_path(job['job_id'])
            measured = json.loads((folder/'measurement.json').read_text())
            winner_hash = digest(report.get('winners', {}))
            if expected_hash is None:
                expected_hash = winner_hash
            assert report['status'] == 'complete' and report['retention_complete'], report
            assert report['counts']['completed_configurations'] == 1048576, report['counts']
            assert winner_hash == expected_hash, 'Profiler changed retained winners/provenance'
            # Check the whole requested leaderboard, plus independent CPU replay of three winners.
            replay = []
            for candidate in report['winners']['global'][:3]:
                checked = engine.replay(job['job_id'], candidate['id'], cpu=True)
                cpu = checked['cpu_reference']['mse']
                error = abs(candidate['mse']-cpu)
                assert error < 1e-8 + 1e-4*abs(cpu), checked
                replay.append(dict(gpu=candidate['mse'], cpu=cpu, absolute_error=error))
            with sqlite3.connect(folder/'plan.sqlite') as db:
                stored = {table: db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
                          for table in ['skeletons', 'variants', 'provenance', 'candidates']}
            chunk_rows = [json.loads(line) for line in (folder/'chunks.jsonl').read_text().splitlines()]
            row = dict(index=index, mode=mode, cache='new_context' if index == 0 else 'warm', job_id=job['job_id'],
                status=report['status'], counts=report['counts'], timing=report['timing'],
                planning_seconds=report['generation']['adapter_compile_seconds'], measurement=measured,
                winner_sha256=winner_hash, stored=stored, report_bytes=(folder/'report.json').stat().st_size,
                plan_bytes=(folder/'plan.sqlite').stat().st_size, reducer_passes=sum(c['reducer_passes'] for c in chunk_rows),
                modules=sum(c['native']['modules'] for c in chunk_rows), cpu_replay=replay)
            results.append(row)
            service.atomic(args.root/'summary.json', results)
            print(json.dumps({k: row[k] for k in ['index', 'mode', 'cache', 'status', 'counts', 'timing', 'planning_seconds', 'stored', 'report_bytes']}), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
