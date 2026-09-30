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
"""Bounded memory failures and recovery through the deployed service."""
import argparse
import json
from pathlib import Path
import time
from client import Client


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=14222)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    name='explicit_cartesian_recovery'
    request=json.loads((root/f'service_trial/validation/requests/{name}.json').read_text())
    reference=next(r['report'] for r in json.loads((root/'service_trial/validation/direct-reference.json').read_text()) if r['name']==name)
    records=[];client=Client(port=a.port)
    try:
        for case,cap in [('operator_ceiling',(2<<30)+1),('job_budget',1024),('recovery',None)]:
            req=json.loads(json.dumps(request))
            if cap is not None:req.setdefault('execution',{})['max_host_bytes']=cap
            handle=client.submit(json.dumps(req).encode())['handle']
            deadline=time.monotonic()+30
            while (status:=client.status(handle))['state']!='terminal':
                assert time.monotonic()<deadline,status
                time.sleep(.02)
            report=client.result(handle);client.rpc('release.'+handle)
            assert status['attempts']==1,status
            assert report['memory']['worker_host_ceiling_bytes']==2<<30
            assert not report['runtime_quarantined']
            if cap is not None:
                assert report['status']=='failed' and report['counts']['completed_configurations']==0
                if case=='operator_ceiling':assert 'worker host ceiling' in report['error']
                else:assert report['memory']['denied_reservations']>=1
            else:
                assert report['status']=='complete'
                for key in ('candidates','leaderboards','families'):assert report[key]==reference[key],key
            records.append(dict(name=case,status=status,report=report))
        assert len({r['status']['worker_info']['pid'] for r in records})==1
        result=dict(passed=True,runs=records)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,cases=len(records),worker=records[-1]['status']['worker_info'])))
    finally:client.close()


if __name__=='__main__':main()
