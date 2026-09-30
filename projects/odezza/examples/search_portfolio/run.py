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
"""Submit one prepared portfolio and produce compact, family-balanced feedback."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from service_trial.client import Client
from odezza.grammar.reference import expression


def feedback(request, report, plan=None, top_k=3):
    if top_k<1:raise ValueError('top_k must be positive')
    states=request['problem']['states'];families=[]
    unknown=[s for s in states if s not in request['problem'].get('known_rhs',{})]
    allocations={f['id']:f for f in plan.get('families',[])} if plan else {}
    enumeration=request['grammar'].get('expansion',{}).get('strategy')=='enumerate'
    if plan:
        enumeration=enumeration and all(x['coverage']=='finite_enumeration' for x in plan['families'])
    for f in report.get('families',[]):
        ids=report['leaderboards']['families'].get(f['id'],[])
        winners=[]
        for ident in ids[:top_k]:
            c=report['candidates'][ident]
            equations={s:expression(p,states) for s,p in zip(states,c['resolved_programs']) if s in unknown}
            support={s:sorted({node.id for node in ast.walk(ast.parse(equations[s],mode='eval'))
                if isinstance(node,ast.Name) and node.id in states}) for s in unknown}
            winners.append(dict(candidate_id=ident,mse=c['mse'],origin=c['origin'],
                equations=equations,state_support=support,parameter_slots=len(c['slots']),
                values=c['values'],value_bits=c['value_bits']))
        families.append(dict(coverage=f,allocation=allocations.get(f['id']),retained_winner_count=len(ids),winners=winners,
            finite_language_exhausted=enumeration and report['status']=='complete'
                and f['generation_stop']==1 and f['pruned']==0 and f['configuration_limited_derivations']==0
                and f['completed_configurations']==f['reserved_configurations']))
    expected={f['id'] for f in request['grammar']['families']}
    groups={}
    for dimension in ('argument_class','mechanism','background'):
        group={}
        for family in families:
            allocation=family['allocation'] or {}
            label=allocation.get(dimension)
            if label is None:continue
            entry=group.setdefault(label,dict(families=[],completed_configurations=0,valid=0,invalid=0,best_mse=None))
            entry['families'].append(family['coverage']['id'])
            for key in ('completed_configurations','valid','invalid'):
                entry[key]+=family['coverage'].get(key,0)
            for winner in family['winners']:
                score=winner['mse']
                if score is not None and (entry['best_mse'] is None or score<entry['best_mse']):entry['best_mse']=score
        if group:groups[dimension]=group
    return dict(status=report['status'],counts=report['counts'],timing=report['timing'],
        integration=report.get('integration'),devices=report.get('execution',{}).get('devices'),families=families,
        groups=groups,group_scope='Each grouping separately partitions families; do not add counts across grouping dimensions. Best MSE is a retained-winner statistic.',
        winner_detail='Display shortlist only; report.json retains every winner, fixed RHS, exact slots, RNG addresses and programs.',
        interpretation='Scores rank sampled coefficient vectors. Poor raw MSE does not exclude a structure before coefficient fitting.',
        all_families_exhausted=bool(families) and {f['coverage']['id'] for f in families}==expected
            and all(f['finite_language_exhausted'] for f in families))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path)
    p.add_argument('--port',type=int,default=14222);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    raw=a.request.read_bytes();request=json.loads(raw)
    planpath=a.request.with_suffix('.plan.json');plan=json.loads(planpath.read_text()) if planpath.exists() else None
    c=Client(port=a.port,timeout=30);handle=None;report=None
    start=time.monotonic()
    try:
        health=c.health();submitted=c.submit(raw);handle=submitted['handle']
        (a.output/'submitted.json').write_text(json.dumps(dict(health=health,submitted=submitted,request_sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')
        print(json.dumps(submitted),flush=True)
        deadline=start+request.get('execution',{}).get('max_seconds',600)+60
        while c.status(handle)['state']!='terminal':
            if time.monotonic()>deadline:
                c.rpc('cancel.'+handle)
                raise TimeoutError('Service exceeded request deadline; job handle retained in submitted.json')
            time.sleep(.05)
        report=c.result(handle)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        summary=feedback(request,report,plan);summary['client_seconds']=time.monotonic()-start
        (a.output/'feedback.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(status=report['status'],client_seconds=summary['client_seconds'],counts=report['counts'],
            all_families_exhausted=summary['all_families_exhausted'],families=[dict(id=f['coverage']['id'],
                best_mse=f['winners'][0]['mse'] if f['winners'] else None,exhausted=f['finite_language_exhausted']) for f in summary['families']])),flush=True)
        if report['status']!='complete':raise RuntimeError(report.get('error') or report['status'])
    finally:
        if handle:
            if report is not None:c.rpc('release.'+handle)
            else:c.rpc('cancel.'+handle)
        c.close()


if __name__=='__main__':main()
