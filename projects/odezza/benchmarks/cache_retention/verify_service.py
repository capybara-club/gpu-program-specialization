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
"""Verify the deployed mac3/rack1 route with retained reference results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'service_trial'))
from client import Client
from run_cases import cases


def candidates(report):
    out = deepcopy(report['candidates'])
    for row in out.values(): row.pop('equations', None)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    rows = []; client = Client(); generation = client.health()['generation']
    def run(name, request, reference=None, reject=False):
        begin = time.perf_counter(); handle = client.submit(json.dumps(request).encode())['handle']
        deadline = time.monotonic()+120
        while (status := client.status(handle))['state'] != 'terminal':
            assert time.monotonic() < deadline, status
            time.sleep(.02)
        report = client.result(handle)
        rows.append(dict(name=name, elapsed_seconds=time.perf_counter()-begin, status=status, report=report))
        Path(a.output).write_text(json.dumps(dict(passed=False, generation=generation, runs=rows), indent=2)+'\n')
        assert status['worker_info']['host_limit_bytes'] == 2 << 30
        if reject:
            assert report['status'] == 'failed' and report['delivery']['preflight_rejected'], report
            assert report['counts']['completed_configurations'] == 0
        else:
            assert report['template_cache']['backend'] == 'sqlite'
            assert report['numeric_input_scope'] == 'attempt'
            assert report['status'] == 'complete', report
            assert report['buffer_storage']['scoring'] == 'reserved_slabs'
            assert len(json.dumps(report, separators=(',', ':')).encode()) < report['delivery']['conservative_bound_bytes']
            if reference:
                assert candidates(report) == candidates(reference), name
                for key in ('leaderboards','counts','families'): assert report[key] == reference[key], (name,key)
        client.rpc('release.'+handle)
        print(name,report['status'],report.get('timing',{}).get('total_seconds'),flush=True)
    try:
        for ref in json.loads((ROOT/'service_trial/validation/direct-reference.json').read_text()):
            request = json.loads((ROOT/f"service_trial/validation/requests/{ref['name']}.json").read_text())
            run(ref['name'],request,ref['report'])
        tagged = {r['name']:r['report'] for r in json.loads((ROOT/'benchmarks/cache_retention/results/before.json').read_text())}
        for name, request in cases(): run(name, request, tagged[name])
        req = next(cases())[1]; req['execution']['max_report_bytes'] = 4096
        run('report_preflight_rejection', req, reject=True)
        ref = json.loads((ROOT/'benchmarks/cache_retention/results/combined-million.json').read_text())
        for i in range(4): run('million_'+str(i),ref['request'],ref['runs'][1]['report'])
        assert len({r['status']['worker_info']['pid'] for r in rows}) == 1
        assert not json.loads(client.rpc('jobs'))['jobs']
        Path(a.output).write_text(json.dumps(dict(passed=True, generation=generation, runs=rows), indent=2)+'\n')
    finally: client.close()

if __name__ == '__main__': main()
