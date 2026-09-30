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
"""Non-disruptive post-deployment check on an otherwise idle live service."""
import argparse
import json
from pathlib import Path
import statistics
import time
from client import Client
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=14223);p.add_argument('--output',type=Path,default=ROOT/'service_trial/validation/control-live.json');a=p.parse_args()
c=Client(port=a.port);runs=[];h=None
try:
    health=c.health();assert health['integration_limits']['base_work_units_per_tile']==67108864
    assert health['integration_limits']['tile_policy']=='parallel-work-v1'
    raw=(ROOT/'service_trial/validation/million-2048-request.json').read_bytes()
    ref=json.loads((ROOT/'service_trial/validation/million-direct-attempt.json').read_text())[-1]['report']
    bad=json.loads(raw);bad['grammar']['integration']['dt']=1e-10
    h=c.submit(json.dumps(bad).encode())['handle'];status=c.status(h);rejected=c.result(h)
    assert status['attempts']==0 and status['state']=='terminal'
    assert rejected['error']=='integration_work_limit' and rejected['completed_configurations']==0
    c.rpc('release.'+h);h=None
    for i in range(3):
        start=time.monotonic();h=c.submit(raw)['handle'];admitted=time.monotonic()
        while (status:=c.status(h))['state']!='terminal':
            assert time.monotonic()-start<60;time.sleep(.02)
        observed=time.monotonic();report=c.result(h);end=time.monotonic()
        assert report['status']=='complete',report.get('error')
        for k in ('candidates','leaderboards'):assert report[k]==ref[k],k
        # The archived reference predates additive family sizing diagnostics.
        # Compare every historical field, and check the newer coverage fields.
        assert len(report['families'])==len(ref['families'])
        for actual,expected in zip(report['families'],ref['families']):
            assert {k:actual[k] for k in expected}==expected
            assert actual['batch_variants']==1024
            assert actual['max_configurations_per_skeleton']==0
            assert actual['configuration_limited_derivations']==0
        assert report['counts']['completed_configurations']==2048000000
        assert report['buffer_storage']['scoring']=='reserved_slabs'
        assert status['attempts']==1
        runs.append(dict(admission_seconds=admitted-start,result_seconds=end-observed,end_to_end_seconds=end-start,status=status,report=report))
        c.rpc('release.'+h);h=None
    result=dict(passed=True,health=health,rejected=rejected,runs=runs,median_seconds=statistics.median(x['end_to_end_seconds'] for x in runs))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,median_seconds=result['median_seconds'],times=[x['end_to_end_seconds'] for x in runs],native=[x['report']['timing']['total_seconds'] for x in runs],worker=runs[-1]['status']['worker_info'])))
finally:
    if h:
        try:c.rpc('cancel.'+h)
        except Exception:pass
    c.close()
