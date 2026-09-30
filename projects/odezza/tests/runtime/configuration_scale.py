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
"""Compare constant-row throughput on one million fixed RHS-vector combinations.

No AST population is materialized by Python. Each request contains the same
finite grammar and trajectories; the C runtime expands and scores it.
"""
import argparse
from concurrent.futures import TimeoutError
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from odezza.grammar.native_service import NativeService

ROOT = Path(__file__).resolve().parents[2]


def request(configurations, chunk_configurations, max_seconds):
    grammar = json.loads((ROOT / 'examples/grammar/million.json').read_text())
    grammar['constants'] = {'c': {'values': [.25 + i / configurations for i in range(configurations)]}}
    grammar['rhs']['x0'] = 'const.c*(' + grammar['rhs']['x0'] + ')'
    grammar['limits']['max_configurations'] = 1_000_000 * configurations
    grammar['limits']['max_seconds'] = max_seconds
    problem = dict(states=grammar['states'], trajectories=[dict(
        initial=[.1]*6, times=[0, .01, .03],
        values=[[.1*math.exp(t)]*6 for t in [0, .01, .03]])])
    return dict(problem=problem, grammar=grammar, execution=dict(
        batch_variants=1024, module_systems=32, dedup_bytes_per_family=134217728,
        max_chunk_configurations=chunk_configurations, max_seconds=max_seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configurations', type=int, nargs='+', default=[1024, 2048])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--root', default='/tmp/odezza-runtime-cache')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if any(not 1 <= n <= 65536 for n in args.configurations) or not 1 <= args.repeats <= 5:
        parser.error('configurations must be in 1..65536 and repeats in 1..5')
    output = Path(args.output)
    cases = [('baseline_32', 32, 1048576)]
    for configurations in args.configurations:
        cases.append((f'configs_{configurations}_default_chunk', configurations, 1048576))
        if 1024*configurations > 1048576:
            cases.append((f'configs_{configurations}_complete_page', configurations, 1024*configurations))
    cases = [(name, n, chunk, repetition) for name, n, chunk in cases for repetition in range(args.repeats)]
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  population_unit='full RHS-vector/binding combination; 10 alternatives per each of 6 equations',
                  numeric_sampling='explicit grid on coefficient c; no Philox or state toggles',
                  cases=[], device=args.device,
                  gpu_inventory=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip(),
                  binary_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [ROOT/'build/libodezza.so', ROOT/'build/runtime/libodezza_runtime.so']})
    service = NativeService(args.root, device=args.device)
    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    try:
        for name, configurations, chunk, repetition in cases:
            inputs = request(configurations, chunk, args.max_seconds)
            entry = dict(name=name, repetition=repetition, configurations_per_system=configurations, request=inputs)
            result['cases'].append(entry)
            begin = time.perf_counter()
            job = service.submit(**inputs)
            entry['acknowledgement_seconds'] = time.perf_counter()-begin
            future = service.jobs[job['job_id']]
            while True:
                try:
                    report = future.result(timeout=1)
                    break
                except TimeoutError:
                    status = service.status(job['job_id'])
                    print(name, status['status'], status['counts']['completed_configurations'], flush=True)
            entry['end_to_end_seconds'] = time.perf_counter()-begin
            entry['report'] = report
            save()
            assert report['status'] == 'complete', (name, report['status'], report['error'])
            assert report['retention_complete']
            assert sum(f['generated_asts'] for f in report['families']) == 1_000_000
            expected = 1_000_000 * configurations
            assert report['counts']['completed_configurations'] == expected
            assert report['counts']['valid'] == expected and report['counts']['invalid'] == 0
            assert report['cache']['retry_requested_configurations'] == 0
            entry['replays'] = []
            for identifier in report['leaderboards']['global'][:3]:
                replay = service.replay(job['job_id'], identifier, cpu=True)
                indexed = service.replay(job['job_id'], address=replay['candidate']['origin'])
                assert indexed['candidate'] == replay['candidate']
                gpu, cpu = replay['candidate']['mse'], replay['cpu_reference']['mse']
                assert abs(gpu-cpu) < 3e-10+3e-5*abs(cpu), (name, gpu, cpu)
                entry['replays'].append(replay)
            seconds = report['timing']['total_seconds']
            entry['rates'] = dict(systems_per_second=1_000_000/seconds,
                                 configurations_per_second=expected/seconds,
                                 microseconds_per_system=seconds,
                                 nanoseconds_per_configuration=seconds/expected*1e9)
            print(name, json.dumps(dict(elapsed=entry['end_to_end_seconds'], rates=entry['rates'],
                                       timing=report['timing'], chunks=report['counts']['completed_chunks'])), flush=True)
            save()
        def winners(entry):
            r = entry['report']
            return [(i, r['candidates'][i]['value_bits'], r['candidates'][i]['resolved_programs'], r['candidates'][i]['mse'])
                    for i in r['leaderboards']['global']]
        for configurations in {entry['configurations_per_system'] for entry in result['cases']}:
            matching = [entry for entry in result['cases'] if entry['configurations_per_system'] == configurations]
            assert all(winners(entry) == winners(matching[0]) for entry in matching), 'Chunking/repetition changed retained results'
        result['chunk_invariant_leaderboards'] = True
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    finally:
        service.close()


if __name__ == '__main__':
    main()
