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
"""Validate retained outputs, memory and report admission with actual pooled CUDA."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/runtime'));sys.path.insert(0,str(Path(__file__).parent))
from memory_checks import Runtime
from run_cases import cases

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reference',required=True);p.add_argument('--library');p.add_argument('--baseline',action='store_true');p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
    reference={r['name']:r for r in json.loads(Path(a.reference).read_text())};rows=[];r=Runtime(devices=(0,1),library=a.library)
    try:
        for name,request in list(cases())*a.repeats:
            row=r.run(request);report=row['report'];assert report['status']=='complete',report
            old=reference[name]['report']
            assert report['leaderboards']==old['leaderboards']
            assert report['candidates']==old['candidates']
            assert report['counts']==old['counts']
            if not a.baseline:assert len(json.dumps(report,separators=(',',':')).encode())<report['delivery']['conservative_bound_bytes']
            row['name']=name;row['comparison']='exact candidates, leaderboards and counts; same device group and GPU pools';rows.append(row)
            print(name,report['timing']['total_seconds'],report['memory']['classes']['retained'],flush=True)
        for variant in ([] if a.baseline else ['large_bounds','tiny_report_budget']):
            request=next(cases())[1]
            if variant=='large_bounds':request['grammar']['expansion']['max_nodes']=4096
            else:request['execution']['max_report_bytes']=4096
            row=r.run(request);report=row['report']
            assert report['status']=='failed' and report['delivery']['preflight_rejected'],report
            assert report['counts']['completed_configurations']==0,report
            assert len(json.dumps(report).encode())<4096
            rows.append(dict(name=variant,**row));print('PASS preflight',variant,flush=True)
    finally:r.close()
    Path(a.output).write_text(json.dumps({'passed':True,'runs':rows},indent=2)+'\n')
if __name__=='__main__':main()
