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
"""Public retained LM batches: useful population/width curves, outside core.

Candidate/start preparation is frozen before timed fitting. This is a diagnostic
replay, not a new blind recovery. Full adapter latency includes population
generation, serialization, allocations and transfers. CPU checks are separate.
"""
import argparse
import copy
import itertools
import math
import os
from pathlib import Path
import random
import time

from .common import load, save, append, digest, file_hash
from .run_trial import Engine, Control, REPO, rf, cs, verify
from lm_toggle.native_service import Adapter


def run(fixtures, output, device, targets, repeats):
    output.mkdir(parents=True, exist_ok=False)
    engine=Engine(REPO/'scratch/fitting_batch_trial/bin/odezza-fit-core-run',[device])
    def init():
        adapter=Adapter(device)
        engine.lm_adapters[device]=adapter
        return adapter
    adapter=engine.pools[device].submit(init).result()
    save(output/'protocol.json',dict(fixtures_sha256=file_hash(fixtures), targets=targets,
        widths=[1,8],repeats=repeats,mode='retained public candidate replay',
        core_sha256=file_hash(Path(adapter.library)),hardware=engine.pools[device].submit(adapter.owner.driver.identity).result(),
        timing='full adapter call; template warm-up and CPU verification separate',
        samples='original rows preserved; additional distinct FP32 local/global uniform samples, not Philox'))
    state=dict(status='running',started_at=time.time(),completed=0)
    rows=[]
    try:
        for fixture in load(fixtures):
            spec=copy.deepcopy(fixture['spec']);spec['execution']['device']=device
            original=spec.pop('lm_dispatch',{})
            # Use original coefficient rows explicitly; no width-dependent
            # random-start prefix. All structural candidates are retained.
            spec['refinement']['wall_seconds']=60
            options=dict(starts_per_candidate=1,toggle_width=4,shape_fallback=True,
                         initial_damping=original.get('initial_damping',.001),
                         damping_attempts=original.get('damping_attempts',8))
            save(output/(fixture['id']+'-input.json'),spec)
            modes=list(itertools.product(targets,[1,8]))
            schedule=[(0,1,-1),(0,8,-1)]
            for repeat in range(repeats):
                order=modes[:];random.Random(714+repeat).shuffle(order)
                schedule.extend((target,width,repeat) for target,width in order)
            for target,width,repeat in schedule:
                label=f"{fixture['id']}-m{target}-w{width}-r{repeat}"
                settings=dict(options,lanes_per_fit=width,population=dict(minimum_fits_per_module=target,max_extra_fits=262144))
                state.update(current=label);save(output/'status.json',state)
                def execute():
                    started=time.monotonic()
                    report=adapter.run(spec,output/label,settings,time.monotonic()+60,Control(),engine.backend,engine.workers[device])
                    return report,time.monotonic()-started
                report,elapsed=engine.pools[device].submit(execute).result()
                profiles=report['lm_profile']
                winners=[r for r in report['candidates'] if r['mse'] is not None]
                checks=[]
                # One replay per distinct winner per mode; repeats still check
                # complete output digests. Replays use training observations only.
                if repeat==0:
                    seen=set()
                    for r in sorted(winners,key=lambda r:r['mse']):
                        identity=(r['program_hex'],r['parameter_values_f32_le_hex'])
                        if identity in seen:continue
                        seen.add(identity)
                        cpu=verify({k:spec[k] for k in cs.CONTEXT_FIELDS},r['program_hex'],r['parameter_values'],
                                   spec['execution']['steps_per_observation'],deadline=time.monotonic()+8)
                        checks.append(dict(candidate_id=r['candidate_id'],gpu_mse=r['mse'],cpu=cpu,
                            matched=cpu['status']=='complete' and abs(cpu['mse']-r['mse'])<=max(2e-8,.005*abs(cpu['mse']))))
                        if len(checks)>=2:break
                identity=sorted((r['candidate_id'],r['program_hex'],r['parameter_values'],r['mse']) for r in winners)
                row=dict(case_id=fixture['id'],target=target,width=width,repeat=repeat,seconds=elapsed,
                    status=report['status'],work_complete=report['work_complete'],counts=report['counts'],
                    population=report.get('population'),timing=report['timing'],profiles=profiles,checks=checks,
                    output_sha256=digest(identity),best_mse=min((r['mse'] for r in winners),default=None),
                    winners=[dict(candidate_id=r['candidate_id'],mse=r['mse']) for r in winners],
                    native_seconds=sum(p['total_seconds'] for p in profiles),
                    fits=sum(p['fit_count'] for p in profiles))
                append(output/'results.jsonl',row);rows.append(row)
                if not report['work_complete']:raise RuntimeError('Incomplete replay population; retain outcome and stop')
                state['completed']+=1
                print(label,round(elapsed,4),row['fits'],flush=True)
        # Hash equality across actual lane shapes and repeated runs is stronger
        # than agreement of the single global best candidate.
        mismatches=[]
        for fixture in load(fixtures):
            for target in targets:
                peers=[r for r in rows if r['case_id']==fixture['id'] and r['target']==target and r['repeat']>=0]
                if len({r['output_sha256'] for r in peers})!=1:mismatches.append((fixture['id'],target))
        state.update(status='complete',output_mismatches=mismatches,
                     cpu_replay_rejections=sum(not c['matched'] for r in rows for c in r['checks']))
    except Exception as error:
        state.update(status='error',error=str(error))
        raise
    finally:
        state['finished_at']=time.time();save(output/'status.json',state);engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--targets',type=int,nargs='+',default=[0,128,512,2048,8192])
    p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
    run(a.fixtures,a.output,a.device,a.targets,a.repeats)
