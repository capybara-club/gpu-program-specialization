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
"""Optional profiling and skeleton caps preserve addresses/whole-toggle rows."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from odezza.grammar.native_service import NativeService
from gpu_conformance import problem,grammar


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    rows=[];reference=None
    for devices in ([0],[0,1]):
        s=NativeService('/tmp/odz-timing-allocation-cache',devices=devices)
        try:
            for profile in (False,True):
                for batch in (1,64):
                    g=grammar();g['rules']={'R':['-const.k*leaf.s','-sin(const.k*leaf.s)']};g['rhs']['x0']='R'
                    g['limits']={'max_configurations_per_skeleton':2}
                    g['families']=[{'id':'a','limits':{'max_configurations_per_skeleton':4}},{'id':'b'}]
                    g['retain']={'global':{'k':8,'unit':'evaluation_row'},'per_family':{'k':8,'unit':'evaluation_row'}}
                    h=s.submit(problem=problem(),grammar=g,execution=dict(profile_timing=profile,batch_variants=batch))
                    r=s.jobs[h['job_id']].result(timeout=30)
                    assert r['status']=='complete',r
                    assert r['counts']['completed_configurations']==12,r['families']
                    assert [f['reserved_configurations'] for f in r['families']]==[8,4]
                    assert [f['configuration_limited_derivations'] for f in r['families']]==[2,2]
                    sig={k:r[k] for k in ('candidates','leaderboards')}
                    if reference is None:reference=deepcopy(sig)
                    assert sig==reference
                    for d in r['execution']['devices']:
                        b=d['pipeline_breakdown']
                        assert (b['gpu_active_seconds'] is not None)==profile
                        if profile:
                            assert b['profiled_modules']==b['modules']
                            assert 0<=b['gpu_active_seconds']<=b['gpu_span_seconds']+1e-6
                    for ident in r['candidates']:
                        replay=s.replay(h['job_id'],ident,cpu=True)
                        assert abs(replay['cpu_reference']['mse']-r['candidates'][ident]['mse'])<3e-5
                    rows.append(r);s.release(job_id=h['job_id'])
            # A cap cannot split one toggle product. It skips, reports, and ends.
            g=grammar();g['limits']={'max_configurations_per_skeleton':1}
            h=s.submit(problem=problem(),grammar=g);r=s.jobs[h['job_id']].result(timeout=30)
            assert r['status']=='complete' and r['counts']['completed_configurations']==0
            assert r['families'][0]['configuration_limited_derivations']==1
            rows.append(r);s.release(job_id=h['job_id'])
        finally:s.close()
    Path(a.output).write_text(json.dumps(rows,indent=2)+'\n')
    print('PASS',len(rows),'timing/cap/chunk/device invariance and independent CPU replay cases')


if __name__=='__main__':main()
