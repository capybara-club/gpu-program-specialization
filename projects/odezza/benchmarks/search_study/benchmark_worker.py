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
"""Public-only benchmark search through the frozen native engine; one GPU owner."""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import traceback
from benchmarks.lm_tuning.common import save, load, digest, append
from benchmarks.lm_tuning.run_trial import Engine, Store as BaseStore, Campaigns, Searches, DONE, REPO
from benchmarks.lm_tuning.work_report import collect
import run_search as gs
from odezza.native import NativeError

def helper(name):
    path=Path(__file__).parent/'helpers'/(name+'.py')
    spec=importlib.util.spec_from_file_location('study_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

contract=helper('common')
sizing=helper('fixed_capacity')

class Store(BaseStore):
    def context(self,problem,parameters,split,execution=None,owner=None):
        c=super().context(problem,parameters,split,execution,owner)
        v=gs.validate_common(c);size=sizing.shared_patch_sizing(v['fixed'],v['execution']['shared_patch_capacity'])
        return dict(c,execution=dict(c.get('execution',{}),shared_patch_capacity=size['selected']))
    def dataset(self,payload):
        gs.keys(payload,{'states','trajectories','evaluation_scope'})
        scope=payload['evaluation_scope']
        if not isinstance(scope,str) or len(scope)!=64 or any(c not in '0123456789abcdef' for c in scope):
            raise ValueError('Invalid task scope')
        super().dataset({k:payload[k] for k in ('states','trajectories')})
        return self.dataset_metadata(self.immutable('datasets',payload))
    def fingerprints(self,key):
        data=self.read('datasets',key)
        return [digest([data['evaluation_scope'],f]) for f in super().fingerprints(key)]
    def problem(self,payload):
        scope=contract.task_scope(payload)
        if any(self.read('datasets',key).get('evaluation_scope')!=scope for key in payload['data'].values()):
            raise ValueError('Dataset task scope mismatch')
        return super().problem(payload)

def trial(group,policy,root,engine,preflight=False):
    case=group['public'];contract.validate_case(case)
    if digest(case)!=group['public_sha256']:raise ValueError('Changed public problem')
    root.mkdir(parents=True,exist_ok=False);save(root/'public.json',case)
    store=Store(root/'store');campaigns=Campaigns(store,engine);searches=Searches(store,engine,campaigns)
    try:
        scope=contract.task_scope(case['problem'])
        data={split:store.dataset(dict(states=case['problem']['states'],trajectories=rows,evaluation_scope=scope))['dataset_id']
              for split,rows in case['datasets'].items()}
        pid=store.problem(dict(case['problem'],data=data))['problem_id'];s=policy['settings']
        fit=dict(backend=policy['backend'],candidates=s['candidates'],starts_per_candidate=s['starts'],
            iterations=s['iterations'],wall_seconds=s['fit_seconds'],toggle_width=s['toggle_width'],
            initial_damping=s['initial_damping'],damping_attempts=s['damping_attempts'],max_step=s['max_step'])
        if policy['backend']=='lm_toggle':fit.update(lanes_per_fit=1,shape_fallback=True)
        request=dict(request_id=group['case_id']+'-'+policy['trial_id'],problem_id=pid,grammar=case['grammar'],
            initial=dict(distinct_asts=s['initial']),expand=dict(new_distinct_asts=s['offspring'],report_every_new_asts=s['wave']),
            parents=dict(limit=s['parents']),bank_count=s['banks'],seed=group['seed'],fit=fit,
            budget=dict(wall_seconds=s['seconds'],devices=engine.devices),target_mse=s['target_mse'],baselines=True,
            screen=dict(trajectories=6,observations=s['observations'],steps_per_observation=s['steps']),
            islands=dict(count=1),leaf_toggles=dict(width=s['leaf_toggle_width'],alternatives='state_assignments'),
            adaptive_coverage=dict(enabled=False))
        save(root/'request.json',request)
        try:prepared=searches.prepare(request)[0]
        except ValueError as e:
            return dict(status='admission_rejected',verified=False,error=str(e),phase='API preparation')
        save(root/'resolved-plan.json',prepared)
        if preflight:return dict(status='prepared_without_submission',verified=False)
        state=searches.create(request)
        while state['status'] not in DONE:
            time.sleep(.25);state=searches.get(state['search_id'])
            save(root/'progress.json',{k:state.get(k) for k in ('status','elapsed_seconds','scored_new_asts','coefficient_trials','current_wave')})
        save(root/'service-result.json',state)
        metrics=(state.get('verification') or {}).get('test',{}).get('results',[])
        return dict(status=state['status'],verified=state['status']=='verified',seconds=state['elapsed_seconds'],
            ast_occurrences=state['scored_new_asts'],distinct_concrete_asts=state.get('cross_island_distinct_asts'),
            configurations=state['coefficient_trials'],accounting_complete=state.get('accounting_complete'),
            timing=state.get('timing'),work_report=collect(root),error=state.get('error'),
            heldout_mse=metrics[-1].get('mse') if metrics else None,
            verification=state.get('verification'),frontier=state.get('frontier'),waves=len(state.get('waves',[])))
    finally:searches.close();campaigns.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--device',type=int,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    root=a.root;root.mkdir(parents=True,exist_ok=True);lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'started.json').exists():raise ValueError('Worker already started; reconcile results before any retry')
    save(root/'started.json',dict(at=time.time(),preflight=a.preflight))
    engine=Engine(REPO/'scratch/fitting_batch_trial/bin/odezza-fit-core-run',[a.device]);fatal=False
    state=dict(status='running',started_at=time.time(),finished_trials=0,device=a.device)
    try:
        for file in sorted((root/'groups').glob('*.json')):
            group=load(file);assert digest({k:v for k,v in group.items() if k!='group_id'})[:24]==group['group_id']
            reserve=sum(t['settings']['seconds']+30 for t in group['trials'])+60
            if not a.preflight and time.time()+reserve>a.deadline:
                state['stop_reason']='insufficient_time_for_complete_policy_block';break
            for policy in group['trials']:
                state.update(case_id=group['case_id'],policy=policy['trial_id']);save(root/'status.json',state)
                path=root/'trials'/(group['group_id']+'-'+policy['trial_id']);began=time.monotonic()
                def expire():
                    save(root/'watchdog.json',dict(case_id=group['case_id'],policy=policy['trial_id'],at=time.time()));os._exit(71)
                timer=threading.Timer(policy['settings']['seconds']+120,expire);timer.daemon=True;timer.start()
                try:result=trial(group,policy,path,engine,a.preflight)
                except NativeError as e:
                    supported_failure=e.result in (7,11,12)
                    result=dict(status='unsupported_resource' if supported_failure else 'error',error=str(e),native_result=e.result,verified=False)
                    fatal=not supported_failure
                except Exception as e:
                    result=dict(status='error',error=str(e),verified=False,traceback=traceback.format_exc());fatal=True
                finally:timer.cancel()
                result.update(case_id=group['case_id'],corpus=group['corpus'],trial_id=policy['trial_id'],group_id=group['group_id'],
                    state_count=group['state_count'],device=a.device,settings=policy['settings'],total_seconds=time.monotonic()-began,
                    source_model_discrepancy=group['source_model_discrepancy'],public_sha256=group['public_sha256'])
                save(path/'summary.json',result);append(root/'results.jsonl',result);state['finished_trials']+=1
                if result['status'] in ('error','interrupted'):fatal=True;break
            if fatal:break
    finally:
        if not fatal:engine.close()
        state.update(status='failed' if fatal else 'complete',finished_at=time.time());save(root/'status.json',state)
    if fatal:os._exit(1)

if __name__=='__main__':main()
