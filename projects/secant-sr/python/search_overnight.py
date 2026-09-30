#!/usr/bin/env python3
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
"""Frozen, checkpointed SRBench population/time/variation campaign.

Small factorial/ablation panel first, then all remaining datasets for each seed.
Settings is an isolated archived executable, never a compatibility mode in Secant.
"""
import argparse
import fcntl
import json
import math
import platform
import subprocess
import time
from pathlib import Path

from compare_search_policies import legacy
from run_srbench_toggle import DEFAULTS, atomic_json, execute, load_manifest, sha256

HYBRID = dict(align_crossover_bits=1, toggle_mutation_probability=.5, leaf_mix_probability=.25)


def arms_for(diagnostic, include_settings):
    def native(name, population, seconds, **options):
        return dict(name=name, engine='toggle', config=dict(DEFAULTS, population=population,
            seconds=seconds, generations=10000, refine_rounds=0, **options))
    arms = [native('base-p8192-t30',8192,30), native('base-p8192-t60',8192,60),
            native('base-p32768-t30',32768,30), native('base-p32768-t60',32768,60),
            native('hybrid-p32768-t60',32768,60,**HYBRID)]
    if diagnostic:
        arms += [native('hybrid-p8192-t60',8192,60,**HYBRID),
                 native('base-p65536-t60',65536,60),native('hybrid-p65536-t60',65536,60,**HYBRID),
                 native('align-p8192-t60',8192,60,align_crossover_bits=1),
                 native('mutate-p8192-t60',8192,60,toggle_mutation_probability=.5),
                 native('mix-p8192-t60',8192,60,leaf_mix_probability=.25)]
    if include_settings:
        arms.append(dict(name='settings-p8192-t60',engine='settings',seconds=60))
        if diagnostic:
            arms.append(dict(name='settings-p8192-t30',engine='settings',seconds=30))
    return arms


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','executable','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--settings',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=3)
    p.add_argument('--hours',type=float,default=8)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 0<=a.shard<a.shards or a.gpu<0 or not math.isfinite(a.hours) or a.hours<=0:
        p.error('invalid shard, GPU or hours')
    a.manifest=a.manifest.resolve();a.executable=a.executable.resolve();a.output=a.output.resolve()
    if a.settings:a.settings=a.settings.resolve()
    manifest=load_manifest(a.manifest)
    cases=[(i,j,arms_for(j['phase']=='diagnostic',bool(a.settings)))
           for i,j in enumerate(manifest['jobs']) if i%a.shards==a.shard]
    if not cases:p.error('empty shard')
    identity=dict(manifest_sha256=sha256(a.manifest),executable_sha256=sha256(a.executable),
        settings_sha256=sha256(a.settings) if a.settings else None,shard=a.shard,shards=a.shards,gpu=a.gpu,
        hours=a.hours,source_sha256={n:sha256(Path(__file__).with_name(n)) for n in
            ['search_overnight.py','compare_search_policies.py','run_srbench_toggle.py','secant_sr_ast.py','srbench_v2.py']},
        arms={'diagnostic':arms_for(True,bool(a.settings)),'main':arms_for(False,bool(a.settings))})
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_file=a.output/'plan.json'
        if plan_file.exists():
            plan=json.loads(plan_file.read_text())
            if not a.resume or plan['identity']!=identity:
                raise ValueError('resume requires identical code, data, binaries and configuration')
            plan['status']='running'
        else:
            if any(x.name!='.lock' for x in a.output.iterdir()):raise ValueError('nonempty output without identity')
            plan=dict(identity=identity,host=platform.node(),started_unix=time.time(),status='running',
                planned=sum(len(arms) for _,_,arms in cases),cases=len(cases),
                hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True),
                limitations=['Numerical held-out R2 > .999; no symbolic equivalence assessment',
                    'Two prepared official seeds; not official ten-seed aggregate',
                    'Settings has different GP, constant refinement, tree limits, batching and SQLite cache',
                    'No settings arm on ada: archived executable needs unavailable libnvrtc.so.13',
                    'Native refinement disabled to isolate population/time/variation; fitting remains separate work',
                    'Warmups excluded and retained; per-fit setup remains in the time budget',
                    'Generation-boundary budgets can overrun; hard subprocess watchdog retained',
                    'Configured campaign window; incomplete cases/arms remain explicit and are never called failures',
                    'Concurrent GPUs share host CPU resources; these are search measurements, not isolated kernel peaks',
                    'Aligned crossover preserves selected donor leaf identities, not donor bank values at a different bank row',
                    'Bit exhaustion may require external coupling; no donor bits merged; reuse counts are reported'])
            atomic_json(plan_file,plan)
        rows=[]
        for _,job,arms in cases:
            for arm in arms:
                f=a.output/arm['name']/'trials'/job['id']/'result.json'
                if f.exists():
                    r=json.loads(f.read_text())
                    if r['status'] not in ('completed','no_finite_model'):
                        raise RuntimeError('retained failed trial requires investigation; no automatic retry')
                    r.update(arm=arm['name'],phase=job['phase'],id=job['id']);rows.append(r)
        done={(r['id'],r['arm']) for r in rows}
        def save():
            atomic_json(a.output/'results.json',rows)
            summary={}
            for _,_,arms in cases:
                for arm in arms:
                    n=arm['name']
                    if n in summary:continue
                    records=[r for r in rows if r['arm']==n]
                    summary[n]=dict(finished=len(records),planned=sum(any(x['name']==n for x in aa) for _,_,aa in cases),
                        errors=sum(r['status'] not in ('completed','no_finite_model') for r in records),
                        successes=sum(r.get('accuracy_solution')==1 for r in records),
                        process_seconds=sum(r.get('process_wall_seconds',0) for r in records),
                        known_configurations=str(sum(int(r.get('total_configurations',r.get('configurations',0))) for r in records)),
                        missing_configuration_counts=sum('configurations' not in r for r in records))
            atomic_json(a.output/'progress.json',dict(status=plan['status'],finished=len(rows),planned=plan['planned'],
                elapsed_seconds=time.time()-plan['started_unix'],arms=summary,updated_unix=time.time()))
        def run_one(job,arm,warm=False):
            folder=a.output/('warmups' if warm else '')/arm['name']
            if arm['engine']=='settings':
                return legacy(job,a.manifest.parent,a.settings,folder/('trials/'+job['id']),
                              a.output/'settings-templates.sqlite',a.gpu,arm['seconds'],warm)
            config=arm['config']
            if warm:config=dict(config,population=256,generations=1,seconds=120)
            return execute(job,a.manifest.parent,a.executable,config,'cuda',a.gpu,folder,
                           max(180,2*config['seconds']+120))
        save()
        try:
            for index,job,arms in cases:
                remaining=[arm for arm in arms if (job['id'],arm['name']) not in done]
                if not remaining:continue
                reserve=sum(arm.get('seconds',arm.get('config',{}).get('seconds',0))+15 for arm in remaining)+120
                if time.time()+reserve > plan['started_unix']+a.hours*3600:
                    plan['status']='time_budget';break
                print(f"case {job['id']} phase={job['phase']} arms={len(remaining)}",flush=True)
                engines=set()
                for arm in remaining:
                    if arm['engine'] in engines:continue
                    engines.add(arm['engine'])
                    wf=a.output/'warmups'/arm['name']/'trials'/job['id']/'result.json'
                    warm=json.loads(wf.read_text()) if wf.exists() else run_one(job,arm,True)
                    if warm['status'] not in ('completed','no_finite_model'):
                        atomic_json(a.output/'failure.json',dict(stage='warmup',arm=arm['name'],result=warm))
                        raise RuntimeError('warmup failed; shard stopped')
                shift=(index+index//a.shards)%len(remaining)
                remaining=remaining[shift:]+remaining[:shift]
                for arm in remaining:
                    print(f"start {job['id']} {arm['name']}",flush=True)
                    result=run_one(job,arm)
                    result.update(id=job['id'],arm=arm['name'],phase=job['phase'])
                    rows.append(result);done.add((job['id'],arm['name']));save()
                    print(f"finish {len(rows)}/{plan['planned']} {job['id']} {arm['name']} "
                          f"{result['status']} R2={result.get('validation_r2')} "
                          f"seconds={result.get('process_wall_seconds')}",flush=True)
                    if result['status'] not in ('completed','no_finite_model'):
                        atomic_json(a.output/'failure.json',dict(stage='fit',result=result))
                        raise RuntimeError('fit failed; shard stopped without discarding evidence')
            else:plan['status']='complete'
        except BaseException:
            plan['status']='failed';plan['finished_unix']=time.time();atomic_json(plan_file,plan);save();raise
        plan['finished_unix']=time.time();atomic_json(plan_file,plan);save()
        print(json.dumps(dict(status=plan['status'],finished=len(rows),planned=plan['planned'])),flush=True)


if __name__=='__main__':main()
