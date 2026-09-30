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
"""Compare combined and prepared submissions on the supplied million-trial case."""
import argparse
import json
from pathlib import Path
import time

from odegrammar.compiler import content_id
from odezza.grammar.service import Service, atomic
from gpu_examples import fixture, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', required=True); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    engine = Service(args.root, library=args.library)
    records = []
    try:
        grammar = json.loads((ROOT/'examples/grammar/adaptation_search.json').read_text())
        problem = fixture(grammar)
        execution = dict(patch_capacity=1024, module_systems=32, max_seconds=1200)
        expected = None
        for mode in ('combined', 'prepared'):
            if mode == 'prepared':
                prepared = engine.prepare(problem)
            start = time.monotonic()
            job = (engine.submit(problem=problem, grammar=grammar, execution=execution) if mode == 'combined'
                   else engine.submit(prepared['problem_id'], grammar, execution))
            acknowledgement = time.monotonic()-start
            report = engine.jobs[job['job_id']].result()
            assert report['status'] == 'complete' and report['retention_complete'], report
            assert report['counts']['completed_configurations'] == 1048576
            if expected is None: expected = report['winners']
            assert report['winners'] == expected, 'Combined and prepared results differ'
            compact = engine.results(job['job_id'], format='compact')
            assert compact['leaderboards']['global'] == [c['id'] for c in report['winners']['global']]
            for group in ('families', 'tags'):
                assert compact['leaderboards'][group] == {k: [c['id'] for c in v] for k, v in report['winners'][group].items()}
            for candidate in compact['candidates'].values():
                resolved = engine.replay(job['job_id'], address=candidate['origin'])
                assert resolved['id'] == candidate['id']
                assert resolved['mse'] == candidate['mse']
                assert resolved['named_values'] == candidate['named_values']
            checks = []
            for identifier in compact['leaderboards']['global'][:3]:
                candidate = compact['candidates'][identifier]
                replay = engine.replay(job['job_id'], address=candidate['origin'], cpu=True)
                cpu = replay['cpu_reference']['mse']
                assert cpu is not None and abs(cpu-candidate['mse']) < 1e-8+1e-4*abs(cpu)
                checks.append(dict(id=identifier, gpu=candidate['mse'], cpu=cpu))
            full_bytes = len(json.dumps(report, separators=(',', ':')).encode())
            compact_bytes = len(json.dumps(compact, separators=(',', ':')).encode())
            row = dict(mode=mode, job_id=job['job_id'], acknowledgement_seconds=acknowledgement,
                report=report, compact=compact, full_bytes=full_bytes, compact_bytes=compact_bytes,
                winner_sha256=content_id(report['winners']), indexed_cpu_checks=checks)
            records.append(row)
            atomic(args.root/'unified-summary.json', dict(status='passed', cases=records))
            print(json.dumps(dict(mode=mode, acknowledgement_seconds=acknowledgement,
                total_seconds=report['timing']['total_seconds'], configurations=report['counts']['completed_configurations'],
                full_bytes=full_bytes, compact_bytes=compact_bytes, indexed_candidates=len(compact['candidates']))), flush=True)
    finally:
        engine.close()


if __name__ == '__main__': main()
