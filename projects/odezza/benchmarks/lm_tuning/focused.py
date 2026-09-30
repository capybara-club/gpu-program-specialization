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
"""Prepare, deploy, run and summarize fixed overnight comparisons.

Run `python3 -m benchmarks.lm_tuning.focused --help`. Preparation does not launch
GPU work. The separate run command belongs in mac1 tmux. No production defaults
or core kernels are modified. Trials use the existing durable native worker.
"""
import argparse
from collections import defaultdict, Counter
import copy
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
from .common import REPO, save, load, digest, file_hash
from .deploy import lanes, ssh, deploy_frozen
from .focused_design import specifications, fit_variants, search_variants, group, reserved_seconds, calibration_subset


def prepare(root, pilot=False):
    root.mkdir(parents=True,exist_ok=False)
    study=dict(schema='odezza.focused-tuning.v1',name='focused_'+root.name.replace('-','_'),
        hosts=['rack1','rohini','ada'],hours=1 if pilot else 9,pilot=pilot,
        notify=not pilot,design_seed=2026091017,case_attempts=3,
        grammar_policy='Unchanged broad depth-three grammar; no true dependency subset sent to search',
        inference='Prepared full candidate pools contain planted structure; blind recovery receives public.json only',
        search_comparison='Fixed policies, same public inputs and seed on one GPU; adaptive paths differ',
        calibration_comparison='Same eight-candidate planted pool and bank prefixes; exact widths and explicit automatic modes',
        no_pysr='Excluded from this campaign; first resolve internal policy/throughput questions')
    if not study['name'].isidentifier():raise ValueError('Use an alphanumeric run directory name')
    specs=specifications()
    if pilot:
        specs=[next(s for s in specs if s['states']==n and s['parameters']==p) for n,p in [(3,3),(6,6),(8,6)]]
    save(root/'study.json',study)
    save(root/'planned-specifications.json',specs)
    entries=lanes(study['hosts']); schedule=[]; seen=set(); generation=[]
    from .generate import generate
    for index,spec in enumerate(specs):
        cid=f'case-{index:03d}'; case=None
        for attempt in range(study['case_attempts']):
            attempt_path=root/'generation'/cid/f'draw-{attempt}'
            result=generate(spec,attempt_path)
            generation.append(dict(case_id=cid,attempt=attempt,status='accepted' if result else 'rejected',
                                   path=str(attempt_path.relative_to(root)),specification=spec))
            save(root/'generation.json',generation)
            if result:
                case=root/'cases'/cid;case.mkdir(parents=True)
                for name in ('public.json','calibration.json','generation.json'):
                    (case/name).write_bytes((attempt_path/name).read_bytes())
                save(case/'origin.json',dict(path=str(attempt_path.relative_to(root)),specification=spec))
                break
        if case is None:continue
        seed=int(digest([study['design_seed'],cid])[:8],16)
        cell=(spec['states'],spec['parameters'])
        if cell not in seen:
            seen.add(cell)
            fv=fit_variants()
            if pilot:fv=[t for t in fv if t['trial_id'] in ('bank64_auto','bank64_queue16','bank64_toggle1','bank1024_lane2','bank16384_lane2')]
            payload=calibration_subset(load(case/'calibration.json'),load(attempt_path/'private/solution.json')['planted_program'])
            save(case/'prepared-calibration.json',payload)
            g=group(cid,payload,'fit',fv,index,seed,'calibration')
            save(root/'groups'/(g['group_id']+'.json'),g)
            for host in study['hosts']:
                entry=next(e for e in entries if e[0]==host and e[1]==(index%2 if host=='rack1' else 0))
                schedule.append(dict(lane=entry[2],group_id=g['group_id'],reserve_seconds=reserved_seconds(g),phase='calibration'))
        sv=search_variants(60 if pilot else 240)
        if pilot:sv=sv[:2]
        g=group(cid,load(case/'public.json'),'search',sv,index,seed,'recovery')
        save(root/'groups'/(g['group_id']+'.json'),g)
        schedule.append(dict(lane=entries[index%len(entries)][2],group_id=g['group_id'],reserve_seconds=reserved_seconds(g),phase='recovery'))
    schedule.sort(key=lambda r:r['phase']!='calibration')
    totals={entry[2]:sum(r['reserve_seconds'] for r in schedule if r['lane']==entry[2]) for entry in entries}
    if any(t>study['hours']*3600-120 for t in totals.values()):
        raise ValueError('Plan exceeds conservative admission budget: '+str(totals))
    plan=dict(schedule=schedule,reserved_seconds_by_lane=totals,
        accepted_cases=len(list((root/'cases').glob('case-*'))),planned_cases=len(specs),
        calibration_cells=[list(c) for c in sorted(seen)],generation_rejections=sum(r['status']=='rejected' for r in generation))
    plan['group_hashes']={p.name:file_hash(p) for p in (root/'groups').glob('*.json')}
    save(root/'plan.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k not in ('schedule','group_hashes')},indent=2))


def summarize(root):
    plan=load(root/'plan.json');rows=[]; seen=set(); expected={}
    for entry in plan['schedule']:
        g=load(root/'groups'/(entry['group_id']+'.json'))
        for t in g['trials']:expected[entry['lane'],g['group_id'],t['trial_id']]=t
    for p in (root/'hosts').glob('*/results.jsonl'):
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            key=(p.parent.name,r['group_id'],r['trial_id'])
            if key not in expected or key in seen:raise ValueError('Unexpected/duplicate trial '+str(key))
            seen.add(key)
            if r['settings']!=expected[key]['settings']:raise ValueError('Trial settings differ from frozen plan')
            r=dict(r,lane=p.parent.name)
            r['within_budget']=r.get('verified',False) and r['total_seconds']<=expected[key]['seconds']
            r['budget_penalized_seconds']=r['total_seconds'] if r['within_budget'] else expected[key]['seconds']
            origin=load(root/'cases'/r['case_id']/'origin.json')['specification']
            r['state_count']=origin['states'];r['declared_parameters']=origin['parameters']
            r['view']=origin['view'];rows.append(r)
    grouped=defaultdict(list)
    for r in rows:grouped[r['kind'],r['trial_id']].append(r)
    summary=[]
    for (kind,policy),rs in sorted(grouped.items()):
        profiles=[p for r in rs for p in r.get('lm_profile') or []]
        summary.append(dict(kind=kind,policy=policy,completed=len(rs),verified=sum(r.get('verified',False) for r in rs),
            within_budget=sum(r['within_budget'] for r in rs),statuses=dict(Counter(r['status'] for r in rs)),
            median_all_seconds=statistics.median(r['total_seconds'] for r in rs),
            mean_budget_penalized_seconds=statistics.mean(r['budget_penalized_seconds'] for r in rs),
            work_complete=sum(r.get('work_complete') is True for r in rs),
            configurations=sum(r.get('configurations',0) for r in rs),
            native_calls=len(profiles),calls_at_most_128_fits=sum(p['fit_count']<=128 for p in profiles),
            max_submitted_packs=max((p.get('submitted_packs',0) for p in profiles),default=0)))
    matches=[]
    for (lane,gid,_),_t in expected.items():
        peers={r['trial_id']:r for r in rows if r['lane']==lane and r['group_id']==gid}
        if not peers or next(iter(peers.values()))['kind']!='fit':continue
        if any(m['lane']==lane and m['group_id']==gid for m in matches):continue
        comparisons=[]
        for start in (4,64,1024,16384):
            ids=[f'bank{start}_lane{w}' for w in (1,2,4,8)]+[f'bank{start}_queue16',f'bank{start}_auto']
            if start==64:ids+=['bank64_toggle1']
            valid=[peers[k] for k in ids if k in peers and peers[k].get('work_complete')]
            if len(valid)>1:
                comparisons.append(dict(starts=start,policies=[r['trial_id'] for r in valid],
                    same_bank=len({r['bank_sha256'] for r in valid})==1,
                    same_candidate_outputs=len({r['candidate_output_sha256'] for r in valid})==1))
        matches.append(dict(lane=lane,group_id=gid,comparisons=comparisons))
    result=dict(expected_trials=len(expected),completed_trials=len(rows),unreported_trials=len(expected)-len(rows),
        summary=summary,rows=rows,matched_fit_checks=matches,
        interpretation='Policies are paired within case/GPU; repetitions across hosts are correlated. Unreported work is separate. No holdout-driven selection or PySR claim.')
    save(root/'summary.json',result)
    text=['# Focused tuning results','',f"{len(rows)}/{len(expected)} trials reported.",'',
          '| Work | Policy | Verified | Within wall budget | Median all outcomes |',
          '|---|---|---:|---:|---:|']
    for r in summary:text.append(f"| {r['kind']} | {r['policy']} | {r['verified']}/{r['completed']} | {r['within_budget']} | {r['median_all_seconds']:.2f} s |")
    text+=['',result['interpretation'],'',
           'The JSON retains per-case/state/parameter/view data, failures, complete-work checks, input/output identities and native profiles. '
           'Exact unavailable widths are expected resource outcomes; partial populations must not enter matched-work speed ratios.','']
    (root/'SUMMARY.md').write_text('\n'.join(text))
    return result


def run(root):
    from .coordinator import collect,upload,start_workers
    study=load(root/'study.json');plan=load(root/'plan.json')
    deployment=load(root/'source/deployment.json');entries=lanes(study['hosts']);remote=deployment['remote']
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'completion.json').exists():raise ValueError('Campaign already terminal; use summarize')
    for name,sha in plan['group_hashes'].items():
        if file_hash(root/'groups'/name)!=sha:raise ValueError('Frozen group changed: '+name)
    receipt=root/'launch.json'
    if not receipt.exists():
        for name,sha in load(root/'source/source-hashes.json').items():
            if Path(name).suffix=='.md':continue
            if file_hash(REPO/name)!=sha:raise ValueError('Local source changed since deployment: '+name)
        for entry in entries:ssh(entry[0],['mkdir','-p',remote+'/queues/'+entry[2]+'/inbox'])
        for sequence,item in enumerate(plan['schedule']):
            entry=next(e for e in entries if e[2]==item['lane'])
            upload(root,remote,entry,load(root/'groups'/(item['group_id']+'.json')),sequence)
        for entry in entries:ssh(entry[0],['touch',remote+'/queues/'+entry[2]+'/CLOSED'])
        save(receipt,dict(at=time.time(),deadline=time.time()+study['hours']*3600,plan_sha256=digest(plan),study_sha256=digest(study)))
    launch=load(receipt)
    if launch['plan_sha256']!=digest(plan):raise ValueError('Frozen plan changed since launch')
    if launch.get('study_sha256',digest(study))!=digest(study):raise ValueError('Frozen study changed since launch')
    start_workers(root,study,remote,entries,launch['deadline'])
    while True:
        states=collect(root,remote,entries);result=summarize(root)
        if all(s.get('finished_at') for s in states.values()):break
        if time.time()>launch['deadline']+90:break
        time.sleep(15)
    complete=all(s.get('finished_at') and s.get('status')=='complete' for s in states.values())
    save(root/'completion.json',dict(at=time.time(),status='complete' if complete else 'failed_or_unconfirmed',lanes=states,
        completed_trials=result['completed_trials'],unreported_trials=result['unreported_trials']))
    if study['notify']:
        sys.path.insert(0,str(REPO/'scratch/fitting_batch_trial'))
        from telegram_notify import send
        message=f"Odezza focused tuning {'finished' if complete else 'stopped with failures'}: {result['completed_trials']}/{result['expected_trials']} trials reported. Results: {root}/SUMMARY.md"
        for attempt in range(3):
            try:
                save(root/'telegram-complete.json',dict(message_id=send(message),message=message));break
            except RuntimeError as e:
                save(root/'telegram-error.json',dict(error=str(e),attempt=attempt));time.sleep(10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','deploy','run','summarize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pilot',action='store_true',help='Prepare a short integration trial, not a tuning result')
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root,a.pilot)
    elif a.action=='deploy':
        study=load(root/'study.json');source=root/'source';source.mkdir(exist_ok=False)
        deploy_frozen(source,study['name'],study['hosts'],[3,6,8])
    elif a.action=='run':run(root)
    else:summarize(root)


if __name__=='__main__':main()
