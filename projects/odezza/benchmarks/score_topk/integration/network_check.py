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
"""Verify the deployed worker against direct native request results."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'service_trial'))
from client import Client

def native_candidates(report):
    values=deepcopy(report['candidates'])
    # NativeService adds human-readable equations in Python. The C wire report
    # carries their authoritative postorder programs; compare all native fields.
    for c in values.values():c.pop('equations',None)
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=14222);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    base=Path(__file__).resolve().parent;results=[]
    for mode in ['after','raw']:
        reference=json.loads((base/('request-'+mode+'.json')).read_text())
        raw=json.dumps(reference['request'],separators=(',',':')).encode();ref=reference['runs'][1]['report']
        for repeat in range(a.repeats):
            c=Client(port=a.port)
            try:
                start=time.perf_counter();job=c.submit(raw);handle=job['handle'];deadline=time.monotonic()+120
                while True:
                    status=c.status(handle)
                    if status['state']=='terminal':break
                    assert time.monotonic()<deadline,status
                    time.sleep(.02)
                r=c.result(handle);elapsed=time.perf_counter()-start
                entry=dict(mode=mode,repeat=repeat,end_to_end_seconds=elapsed,status=status,report=r,passed=False)
                results.append(entry)
                def save():
                    (base/'live-million.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in results),
                        comparison='all native fields exact; Python-only formatted equations excluded',runs=results),indent=2)+'\n')
                save()
                assert r['status']=='complete',r
                assert r['numeric_input_scope']=='attempt'
                assert r['buffer_storage']['scoring']=='reserved_slabs'
                assert r['reduction']['backend']=='cub_block_hierarchy'
                assert native_candidates(r)==native_candidates(ref),(mode,'candidates')
                for k in ['leaderboards','families']:assert r[k]==ref[k],(mode,k)
                assert r['counts']['completed_configurations']==2048000000
                assert r['reduction']['continuation_passes']==0
                entry['passed']=True;save()
                print(mode,repeat,'native',r['timing']['total_seconds'],'network',elapsed,flush=True)
                c.rpc('release.'+handle)
            finally:c.close()
if __name__=='__main__':main()
