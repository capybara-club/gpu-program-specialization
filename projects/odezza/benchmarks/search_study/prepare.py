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
"""Select benchmark tasks using public inventories, never hidden equations."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from benchmarks.lm_tuning.common import REPO, digest, file_hash, load, save

SEED = 2026091019

def key(case):
    return digest([SEED, case['case_id']])

def variants(index):
    base = dict(initial=65536, offspring=4194304, wave=65536, banks=1024,
        parents=1024, candidates=64, starts=4, iterations=16, fit_seconds=20,
        seconds=180, toggle_width=4, leaf_toggle_width=4, initial_damping=.001,
        damping_attempts=8, max_step=.4, target_mse=1e-6, observations=7, steps=32)
    policies = [('native_lm', 'lm_toggle', {}), ('curvature', 'curvature', {}),
                ('native_finer_screen', 'lm_toggle', {'observations':21})]
    result = [dict(trial_id=name, backend=backend, settings=dict(base, **extra))
              for name, backend, extra in policies]
    offset = index % len(result)
    return result[offset:] + result[:offset]

def select(manifests):
    """Four matched MDBench/ODEBench tasks, plus four biological problems."""
    eligible = {name:[c for c in m['cases'] if c['status']=='prepared'
                      and c['noise']=='none' and c['state_count']<=8]
                for name,m in manifests.items()}
    ode = {(c['case_id'].split('-')[1],c['unknown_rhs']):c
           for c in eligible['odebench-v2']}
    chosen = []
    for n in (1,2,3,4):
        pool = [c for c in eligible['mdbench'] if c['state_count']==n
                and (c['case_id'].split('-')[1],c['unknown_rhs']) in ode]
        if not pool:raise ValueError('No qualified matched benchmark pair for N='+str(n))
        first = min(pool,key=key)
        chosen += [('mdbench',first),('odebench-v2',ode[first['case_id'].split('-')[1],first['unknown_rhs']])]
    # Distinct biological source problems: two 3-state and two 5-state tasks.
    for n in (3,5):
        seen=set()
        for c in sorted((c for c in eligible['biological-v2'] if c['state_count']==n),key=key):
            family=c['case_id'].rsplit('-rhs-',1)[0]
            if family in seen:continue
            chosen.append(('biological-v2',c));seen.add(family)
            if len(seen)==2:break
        if len(seen)!=2:raise ValueError('Insufficient biological source problems')
    return chosen

def prepare(root, random_root):
    root.mkdir(parents=True,exist_ok=False)
    source=REPO/'scratch/trajectory_benchmarks/prepared'
    manifests={n:load(source/n/'qualified-manifest-v2.json')
               for n in ('mdbench','odebench-v2','biological-v2')}
    chosen=select(manifests);groups=[]
    for index,(corpus,entry) in enumerate(chosen):
        public=load(source/corpus/entry['public'])
        if digest(public)!=entry['public_sha256']:raise ValueError('Public source hash changed')
        reference=entry['reference_check']
        if not reference['complete'] or reference['maximum_split_mse']>1e-6:
            raise ValueError('Reference model does not qualify at frozen target')
        g=dict(case_id=entry['case_id'],corpus=corpus,public=public,public_sha256=digest(public),
            state_count=entry['state_count'],trajectory_count=entry['trajectory_count'],
            observations=entry['observations'],source_model_discrepancy=reference['maximum_split_mse'],
            seed=int(digest([SEED,entry['case_id']])[:8],16),trials=variants(index))
        # Each paired source task stays on the same GPU, but policies never share survivors.
        g['lane']=['rack1_0','rack1_1','rohini_0','ada_0'][index//2 if index<8 else index-8]
        g['group_id']=digest(g)[:24]
        save(root/'groups'/(g['group_id']+'.json'),g);groups.append(g)
    plan=dict(schema='odezza.search-study.v1',seed=SEED,random_root=str(random_root.resolve()),
        random_deadline=load(random_root/'launch.json')['deadline'],
        runtime_remote=load(random_root/'source/deployment.json')['remote'],
        cohorts={n:dict(Counter(c['status'] for c in m['cases'])) for n,m in manifests.items()},
        selections=[{k:g[k] for k in ('case_id','corpus','state_count','trajectory_count','observations','lane','group_id','source_model_discrepancy')} for g in groups],
        group_hashes={g['group_id']:file_hash(root/'groups'/(g['group_id']+'.json')) for g in groups},
        expected_trials=sum(len(g['trials']) for g in groups),
        selection_policy='Hash-ranked qualified public tasks; matched MDBench/ODEBench ancestry; four distinct biological sources',
        ordering='Wait for all random workers to finish. Keep benchmark work inside original nine-hour deadline.',
        interpretation='Exploratory single-RHS completion, not official leaderboard scores. No derivatives or hidden equations sent. Qualification/state-limit conditioned subset.',
        grammar_policy='Keep supplied depth-three operators and [-100,100] coefficient range for every benchmark policy.',
        fixed_test_policy='One independent sealed ledger per frozen policy; no test-guided revisions or retries.')
    save(root/'plan.json',plan)
    print(json.dumps(dict(cases=len(groups),trials=plan['expected_trials'],selected=plan['selections']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--random-root',type=Path,required=True)
    a=p.parse_args();prepare(a.root.resolve(),a.random_root.resolve())
