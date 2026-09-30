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
"""Client-side GPU service validation, using exact direct-C reference reports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from client import Client


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--requests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--concurrent', action='store_true')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--port', type=int, default=14222)
    a = p.parse_args()
    references = {r['name']: r['report'] for r in json.loads(Path(a.reference).read_text())}

    def run(name):
        c = Client(port=a.port)
        handle = None
        try:
            start = time.perf_counter()
            admission = c.submit((Path(a.requests)/(name+'.json')).read_bytes())
            admitted = time.perf_counter()
            handle = admission['handle']
            deadline = time.monotonic()+180
            while True:
                status = c.status(handle)
                if status['state'] == 'terminal':
                    break
                assert time.monotonic() < deadline, status
                time.sleep(.02)
            complete = time.perf_counter()
            report = c.result(handle)
            finished = time.perf_counter()
            ref = references[name]
            assert report['status'] == 'complete', report
            assert report['numeric_input_scope'] == 'attempt'
            for key in ('candidates', 'leaderboards', 'families'):
                assert report[key] == ref[key], (name, key)
            assert report['counts']['completed_configurations'] == ref['counts']['completed_configurations']
            c.rpc('release.'+handle)
            return dict(name=name, service=status, report=report, admission_seconds=admitted-start,
                        admission_to_observed_terminal_seconds=complete-admitted,
                        result_seconds=finished-complete, end_to_end_seconds=finished-start)
        finally:
            c.close()

    tasks = list(references) * a.repeats
    if a.concurrent:
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(run, tasks))
    else:
        results = [run(name) for name in tasks]
    Path(a.output).write_text(json.dumps(dict(passed=True, concurrent=a.concurrent, runs=results), indent=2)+'\n')
    print(json.dumps([dict(name=r['name'], worker=r['service']['worker_info'],
                          end_to_end_seconds=r['end_to_end_seconds'], native_seconds=r['report']['timing']['total_seconds']) for r in results]))


if __name__ == '__main__':
    main()
