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
"""Exact family/global raw-row retention, tiling, tags, mixed units and RNG replay."""
import argparse
from copy import deepcopy
import json
import struct
from pathlib import Path
from odezza.grammar.native_service import NativeService
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    rows=[]
    problem={'states':['x0'],'trajectories':[{'initial':[0.], 'times':[0.,.125,.25], 'values':[[0.],[0.],[0.]]}]}
    g={'version':1,'states':['x0'],'integration':{'method':'rk4','dt':.125},
       'constants':{'c':{'values':[(i-32)/32 for i in range(65)]}},
       'rules':{'R':['const.c','const.c+4','const.c+8','const.c+16']},'rhs':{'x0':'R'},
       'families':[{'id':n,'tags':['all'],'limits':{'max_skeletons':4,'max_variants':4,'max_configurations':budget}}
                   for n,budget in [('a',260),('b',197),('c',130)]],
       'retain':{'global':{'k':64,'unit':'evaluation_row'},'per_family':{'k':32,'unit':'evaluation_row'},
                 'by_tag':{'all':{'k':48,'unit':'evaluation_row'}}}}
    expected=[]
    for f,budget in enumerate([260,197,130]):
        for ast,offset in enumerate([0,4,8,16]):
            for bank in range(65):
                if ast*65+bank>=budget:continue
                c=(bank-32)/32+offset
                expected.append((c*c*.0390625,f,ast,bank,0))
    expected.sort()
    def check(report,gpu_rows,k):
        actual=[]
        for identifier in gpu_rows:
            c=report['candidates'][identifier];o=c['origin']
            actual.append((struct.unpack('f',struct.pack('f',c['mse']))[0],o['family_index'],int(o['ast_index']),int(o['bank_index']),o['permutation']))
        assert actual==k,(actual,k)
    reports=[]
    for devices,chunk,batch in [([0],100000,8),([0],51,2),([1],100000,8),([0,1],19,1),([0,1],100000,8)]:
        s=NativeService('/tmp/odezza-cub-family-cache',devices=devices)
        try:
            current=deepcopy(g)
            if devices==[0,1] and chunk==100000:
                current['retain'].pop('by_tag')
                current['retain']['global']['k']=256
                current['retain']['per_family']['k']=128
            j=s.submit(problem=problem,grammar=current,execution={'max_chunk_configurations':chunk,'batch_variants':batch})
            r=s.jobs[j['job_id']].result(timeout=120)
            assert r['status']=='complete',r
            assert r['counts']['completed_configurations']==587,r['counts']
            assert r['reduction']['backend']=='cub_block_hierarchy'
            assert r['reduction']['continuation_passes']==0
            assert r['reduction']['passes']==r['counts']['completed_chunks']
            check(r,r['leaderboards']['global'],expected[:current['retain']['global']['k']])
            for f,name in enumerate(['a','b','c']):
                check(r,r['leaderboards']['families'][name],[x for x in expected if x[1]==f][:current['retain']['per_family']['k']])
                assert all(r['candidates'][i]['origin']['ast_index']=='0' for i in r['leaderboards']['families'][name][:32])
            if 'by_tag' in current['retain']:check(r,r['leaderboards']['tags']['all'],expected[:48])
            reports.append(r)
            rows.append({'name':'analytic_family_rows','devices':devices,'chunk':chunk,'batch':batch,'report':r})
            print('PASS analytic families',devices,chunk,batch,flush=True)
        finally:s.close()
    # The same best numerical expression occupies many bank rows. Raw and
    # distinct policies must coexist without either hiding the other's winners.
    s=NativeService('/tmp/odezza-cub-mixed-cache',devices=[0,1])
    try:
        mg=deepcopy(g);mg.pop('families');mg['constants']['c']['values']=[0]*48+[1]*20+[2]*20
        mg['rules']['R']=['const.c',{'expr':'const.c','tags':['late']}]
        mg['retain']={'global':{'k':3,'unit':'numeric_candidate'},'per_family':{'k':32,'unit':'evaluation_row'},
                      'by_tag':{'late':{'k':24,'unit':'evaluation_row'}}}
        j=s.submit(problem=problem,grammar=mg,execution={'batch_variants':1,'max_chunk_configurations':88})
        r=s.jobs[j['job_id']].result(timeout=120);assert r['status']=='complete',r
        assert len(r['leaderboards']['global'])==3
        assert sorted(r['candidates'][i]['values'][0] for i in r['leaderboards']['global'])==[0,1,2]
        f=next(iter(r['leaderboards']['families'].values()))
        assert [int(r['candidates'][i]['origin']['bank_index']) for i in f]==list(range(32))
        assert [int(r['candidates'][i]['origin']['bank_index']) for i in r['leaderboards']['tags']['late']]==list(range(24))
        assert r['reduction']['continuation_passes']>0
        rows.append({'name':'mixed_units_late_tag','report':r});print('PASS mixed units and late tags',flush=True)
        rng=json.loads((ROOT/'examples/grammar/constant_rng_product.json').read_text())
        rng['retain']={'global':{'k':32,'unit':'evaluation_row'},'per_family':{'k':256,'unit':'evaluation_row'}}
        data={'states':rng['states'],'trajectories':[{'initial':[.2,.3],'times':[0.,.03,.1],
                                                   'values':[[.2,.3],[.21,.29],[.23,.26]]}]}
        baseline=None
        for chunk in [100000,37]:
            j=s.submit(problem=data,grammar=rng,execution={'max_chunk_configurations':chunk})
            r=s.jobs[j['job_id']].result(timeout=120);assert r['status']=='complete',r
            assert r['reduction']['continuation_passes']==0
            sig=(r['leaderboards'],r['candidates'])
            if baseline is None:baseline=sig
            else:assert sig==baseline,'RNG row rankings changed with tiling'
            for identifier in r['leaderboards']['global']:
                cpu=s.replay(j['job_id'],identifier,cpu=True)
                assert abs(cpu['cpu_reference']['mse']-r['candidates'][identifier]['mse'])<3e-5
                replay=s.replay(j['job_id'],address=r['candidates'][identifier]['origin'])
                assert replay['candidate']==cpu['candidate']
            rows.append({'name':'rng_rows_256','chunk':chunk,'report':r});print('PASS RNG rows',chunk,flush=True)
    finally:s.close()
    Path(a.output).write_text(json.dumps({'passed':True,'cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
