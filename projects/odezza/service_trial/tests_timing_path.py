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
"""Combined request contract, reference fixtures, capped search and large work."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from client import Client
ROOT=Path(__file__).resolve().parents[1]


def normalized(report):
    result=deepcopy(report['candidates'])
    for value in result.values():value.pop('equations',None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',required=True);a=p.parse_args();c=Client(port=a.port)
    results=[]
    def run(name,q,expected,reference=None):
        started=time.monotonic();h=c.submit(json.dumps(q).encode())['handle']
        try:
            while (status:=c.status(h))['state']!='terminal':
                assert time.monotonic()-started<120,status
                time.sleep(.02)
            r=c.result(h);assert r['status']=='complete',r
            assert r['counts']['completed_configurations']==expected
            assert r['buffer_storage']['scoring']=='reserved_slabs'
            assert r['numeric_input_scope']=='attempt'
            assert len(json.dumps(r,separators=(',',':')).encode())<r['delivery']['conservative_bound_bytes']
            if reference:
                assert normalized(r)==normalized(reference),(name,'candidates')
                assert r['leaderboards']==reference['leaderboards']
                assert r['counts']==reference['counts']
                for actual,old in zip(r['families'],reference['families']):
                    assert {k:actual[k] for k in old}==old
            for d in r['execution']['devices']:
                if q.get('execution',{}).get('profile_timing'):
                    b=d['pipeline_breakdown'];assert b['profiled_modules']==b['modules']
                    assert b['gpu_active_seconds']<=b['gpu_span_seconds']+1e-6
            results.append(dict(name=name,status=status,report=r,wall=time.monotonic()-started))
            Path(a.output).write_text(json.dumps(dict(passed=False,runs=results),indent=2)+'\n')
            print(name,r['timing']['total_seconds'],flush=True)
            return r
        finally:c.rpc('release.'+h)
    try:
        for item in json.loads((ROOT/'service_trial/validation/direct-reference.json').read_text()):
            q=json.loads((ROOT/f"service_trial/validation/requests/{item['name']}.json").read_text())
            q.setdefault('execution',{})['profile_timing']=True
            run(item['name'],q,item['report']['counts']['completed_configurations'],item['report'])
        q=json.loads((ROOT/'examples/grammar/submit.json').read_text())
        q['grammar']['limits']={'max_configurations_per_skeleton':2}
        q['grammar']['families']=[{'id':'a'},{'id':'b','limits':{'max_configurations_per_skeleton':1}}]
        q['execution']['profile_timing']=True
        r=run('family caps',q,3)
        assert [f['completed_configurations'] for f in r['families']]==[2,1]
        q=json.loads((ROOT/'service_trial/validation/million-2048-request.json').read_text())
        explicit=run('million explicit',q,2048000000)
        for field in ('batch_variants','module_systems','max_chunk_configurations'):q['execution'].pop(field,None)
        automatic=run('million automatic',q,2048000000)
        assert normalized(automatic)==normalized(explicit)
        assert automatic['leaderboards']==explicit['leaderboards']
        assert automatic['families'][0]['batch_variants']==1024
        q['execution']['profile_timing']=True
        profiled=run('million profiled',q,2048000000)
        assert normalized(profiled)==normalized(explicit)
        assert len({x['status']['worker_info']['pid'] for x in results})==1
        Path(a.output).write_text(json.dumps(dict(passed=True,health=c.health(),runs=results),indent=2)+'\n')
        print('PASS',len(results),'combined network/GPU requests')
    finally:c.close()


if __name__=='__main__':main()
