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
"""Fresh systems with fixed state count/depth and varying dependencies per RHS.

Prepare on the generating host. Only plan.json is sent to GPU workers; private
generation seeds, solutions, calibration candidates and group labels stay local.
Execution reuses latency_compare with one predeclared policy. No core changes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from .common import save,load,digest,integer

def collect(root):
    """Read-only collection from the detached isolated workers."""
    from concurrent.futures import ThreadPoolExecutor
    from .deploy import ssh,python_for
    launches=load(root/'launch.json')
    def one(entry):
        code="""from pathlib import Path
import json
p=Path(%r)/'comparison'
def read(name):
 q=p/name
 return json.loads(q.read_text()) if q.exists() else None
rows=[json.loads(l) for l in (p/'results.jsonl').read_text().splitlines()] if (p/'results.jsonl').exists() else []
for row in rows:
 saved=read(row['case_id']+'-'+row['mode']+'-r'+str(row['repeat'])+'/result.json')
 if saved:row['final_training_best']=saved.get('best')
 campaign_root=p/(row['case_id']+'-'+row['mode']+'-r'+str(row['repeat']))/'store/campaigns'
 fits=[]
 for q in sorted(campaign_root.glob('*/result.json')):
  d=json.loads(q.read_text());best=d.get('best')
  if best:fits.append(dict(campaign_id=q.parent.name,status=d.get('status'),scope=d.get('current_scope'),
   best={k:best.get(k) for k in ('mse','objective_id','expression','program_hex','parameter_values')}))
 row['campaign_best_records']=fits
print(json.dumps(dict(status=read('status.json'),results=rows,cold=read('cold-preparation/readiness.json'),
 restart=read('restart-preparation/readiness.json'),lm=read('lm-readiness.json'))))
"""%entry['remote']
        result=json.loads(ssh(entry['host'],[python_for(entry['host']),'-c',code],timeout=30).stdout)
        save(root/(entry['host']+'-results.json'),result)
        return dict(host=entry['host'],status=result['status'],verified=sum(r['verified'] for r in result['results']))
    with ThreadPoolExecutor(max_workers=len(launches)) as pool:
        return list(pool.map(one,launches))

def prepare(spec,root):
    from .generate import generate,validate
    integer(spec['cases_per_level'],'cases_per_level',1,100)
    levels=spec['dependencies']
    if not isinstance(levels,list) or not levels or len(set(levels))!=len(levels):
        raise ValueError('Use a nonempty list of distinct dependency levels')
    base={k:spec[k] for k in ('states','depth','parameters','duration','samples','trajectories','view')}
    base.update(background='random',attempts=spec.get('generation_attempts',200),seconds=spec['generation_seconds'])
    for k in levels:validate(dict(base,dependencies=k,known_dependencies=k))
    root.mkdir(parents=True,exist_ok=False)
    save(root/'design.json',spec)
    entries=[];cases=[];started=time.time()
    for repeat in range(spec['cases_per_level']):
        for k in levels:
            cid=f'case-{len(entries):03d}'
            item=dict(id=cid,dependencies=k,replicate=repeat)
            settings=dict(base,dependencies=k,known_dependencies=k)
            result=generate(settings,root/'generation'/cid)
            item['generation']=result;entries.append(item)
            save(root/'catalog.json',entries)
            if result:
                if result['dependency_counts'] != [k]*spec['states']:
                    raise RuntimeError('Generated dependency counts differ from requested design')
                opts=dict(initial=65536,banks=1024,wave=65536,generation_limit=64,
                    offspring=4194304,parents=1024,candidates=64,starts=4,iterations=16,
                    fit_seconds=20,toggle_width=4,leaf_toggle_width=4,initial_damping=.001,
                    damping_attempts=8,max_step=.4,
                    seed=int(digest([spec['design_seed'],cid])[:8],16))
                cases.append(dict(id=cid,public=load(root/'generation'/cid/'public.json'),settings=opts))
            print(cid,'dependencies',k,'accepted' if result else 'generation_rejected',flush=True)
    save(root/'plan.json',dict(scope='fresh public-only blind recovery',seconds=spec['search_seconds'],
        modes=['occupied'],cases=cases))
    save(root/'preparation.json',dict(status='ready' if cases else 'failed',planned=len(entries),
        accepted=len(cases),rejected=len(entries)-len(cases),seconds=time.time()-started,
        private_boundary='Only plan.json is exported; generation seeds, solutions and level labels remain local',
        interpretation='Fixed state count, grammar and observations. Fresh random structures vary between levels; this is not an isolated causal coupling estimate.'))

def report(root):
    design=load(root/'design.json');catalog={r['id']:r for r in load(root/'catalog.json')}
    plan=load(root/'plan.json');expected={r['id'] for r in plan['cases']}
    rows=[];hosts={}
    for host in design['hosts']:
        data=load(root/(host+'-results.json'));hosts[host]=data
        seen=set()
        for r in data['results']:
            if r['case_id'] in seen or r['case_id'] not in expected:raise ValueError('Unexpected or duplicate attempt')
            seen.add(r['case_id'])
            rows.append(dict(r,host=host,dependencies=catalog[r['case_id']]['dependencies']))
        if data['status']['status']=='complete' and seen!=expected:raise ValueError('Missing completed attempts')
    groups=[]
    for k in design['dependencies']:
        rs=[r for r in rows if r['dependencies']==k]
        successes=[r for r in rs if r['verified']]
        groups.append(dict(dependencies=k,attempts=len(rs),verified=len(successes),
            distinct_verified_systems=len({r['case_id'] for r in successes}),
            generated=sum(bool(c['generation']) and c['dependencies']==k for c in catalog.values()),
            median_success_seconds=statistics.median(r['total_seconds'] for r in successes) if successes else None,
            median_all_seconds=statistics.median(r['total_seconds'] for r in rs) if rs else None,
            statuses=dict(Counter(r['status'] for r in rs)),
            unprepared_attempts=sum(not r['cache_control_valid'] for r in rs)))
    summary=dict(groups=groups,rows=rows,statuses={h:d['status'] for h,d in hosts.items()},
        reported_attempts=len(rows),expected_attempts=len(expected)*len(design['hosts']),
        generation_rejected=[c['id'] for c in catalog.values() if not c['generation']],
        recorded_ast_occurrences=sum(r['ast_occurrences'] for r in rows),
        recorded_configurations=sum(r['configurations'] for r in rows),
        incomplete_work_accounting=sum(not r['accounting_complete'] for r in rows),
        unprepared_attempts=sum(not r['cache_control_valid'] for r in rows),
        interpretation='Three machines repeat the same systems; attempts are correlated. Solutions verify trajectories, not symbolic uniqueness. Preparation is outside search timing; timeout accounting may be incomplete.')
    save(root/'summary.json',summary)
    lines=['# RHS dependency breadth stress test','',
        f"{summary['reported_attempts']}/{summary['expected_attempts']} attempts reported. "
        f"Generation rejected {len(summary['generation_rejected'])}/{len(catalog)} planned slots; those were never sent to search.", '',
        f"{design['states']} states; depth {design['depth']}; {design['parameters']} coefficient leaves in the blinded RHS. "
        'Every known RHS has the same requested dependency count. One RHS is blinded; all state observations are supplied. '
        'Search receives the unchanged broad grammar, never the hidden dependency subset or private equation seed.','',
        f"{design['trajectories']} noiseless trajectories, {design['samples']} samples over {design['duration']} time units; "
        f"{design['trajectories']-2} training, one validation and one held out. Expanded native LM with four-way state-assignment screening; "
        f"{design['search_seconds']}-second search budgets. Different dependency levels use different random systems. "
        'Depth constrains the possible operator/leaf mix, and generation rejects unstable or numerically inactive systems.','',
        '| Distinct inputs per RHS | Verified attempts | Systems solved on any host | Median successful time | Median all outcomes |',
        '|---:|---:|---:|---:|---:|']
    duration=lambda v:'—' if v is None else f'{v:.2f} s'
    for g in groups:
        lines.append(f"| {g['dependencies']} | {g['verified']}/{g['attempts']} | {g['distinct_verified_systems']}/{g['generated']} | {duration(g['median_success_seconds'])} | {duration(g['median_all_seconds'])} |")
    lines+=['',summary['interpretation'],'',
        '| Case | Inputs/RHS | '+' | '.join(design['hosts'])+' |','|---|---:|'+''.join('---|' for _ in design['hosts'])]
    for cid,c in catalog.items():
        values=[]
        for host in design['hosts']:
            rs=[r for r in rows if r['case_id']==cid and r['host']==host]
            r=rs[0] if rs else None
            values.append(('generation rejected' if not c['generation'] else 'unrun') if r is None else f"{r['total_seconds']:.2f} s, "+(f"MSE {r['heldout_mse']:.2g}" if r['verified'] else r['status']))
        lines.append('| '+cid+' | '+str(c['dependencies'])+' | '+' | '.join(values)+' |')
    lines+=['','## Preparation','', '| Host | Empty application scoring cache | Restart | LM templates |', '|---|---:|---:|---:|']
    for h,d in hosts.items():
        lines.append(f"| {h} | {d['cold']['seconds']:.2f} s | {d['restart']['seconds']:.2f} s | {d['lm']['seconds']:.2f} s |")
    lines+=['',f"Recorded work: {summary['recorded_ast_occurrences']:,} seed AST occurrences and "
        f"{summary['recorded_configurations']:,} configurations; {summary['incomplete_work_accounting']} attempts "
        'have incomplete work accounting, so these are lower bounds. Toggle-expanded distinct AST counts remain unknown. '
        f"Unprepared-template attempts: {summary['unprepared_attempts']}.", '',
        'No achieved-occupancy measurement or comparison against other solvers is implied. '
        'Exact coefficient rows, full worker reports and frozen public plans are retained in the isolated remote experiment trees.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return groups

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--spec',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('report');a.add_argument('root',type=Path)
    a=sub.add_parser('collect');a.add_argument('root',type=Path)
    args=p.parse_args()
    if args.command=='prepare':prepare(load(args.spec),args.out)
    elif args.command=='report':print(report(args.root))
    else:print(collect(args.root))
