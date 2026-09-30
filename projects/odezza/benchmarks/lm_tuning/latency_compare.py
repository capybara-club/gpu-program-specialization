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
"""Cache-controlled recovery comparison on retained PUBLIC problems.

Preparation is measured separately, including restart/disk reuse. Each timed
attempt restarts its C scorer, uses identical saved artifacts, and retains both
policies' prepared LM shapes. No generator solutions enter this process.
"""
import argparse
import copy
from pathlib import Path
import tempfile
import time
from .common import load,save,append,REPO,digest
from .run_trial import Engine,search

MODES={
    'baseline':dict(lanes_per_fit=1,shape_fallback=True,population={'minimum_fits_per_module':0},state_variants=0),
    'occupied':dict(lanes_per_fit=8,shape_fallback=True,population={'minimum_fits_per_module':512,'max_extra_fits':65536,'global_fraction':.25},state_variants=0),
    'neighbors':dict(lanes_per_fit=8,shape_fallback=True,population={'minimum_fits_per_module':512,'max_extra_fits':65536,'global_fraction':.25},state_variants=3),
}

def public_shapes(cases):
    """Use the scorer's own fixed-RHS sizing on public inputs, without scoring."""
    from .run_trial import game
    import run_search as gs
    shapes={}
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'public.json'
        for case in cases:
            save(path,case['public'])
            with game.public_input(path) as source:
                config,problem,states,rows,splits=game.load_challenge(source)
            parameters=['p'+str(i) for i in range(config['grammar'].get('max_parameters',8))]
            context=dict(states=states,parameters=parameters,unknown_state=problem['unknown_rhs'][0],
                known_rhs=problem['known_rhs'],data={'trajectories':[rows[i] for i in splits['training']]})
            checked=gs.validate_common(context)
            for systems in (64,256):
                values=(len(states),len(parameters),systems,checked['execution']['shared_patch_capacity'],128)
                shapes[values]=dict(zip(('state_capacity','constant_capacity','system_capacity',
                    'shared_patch_capacity','system_patch_capacity'),values))
    return [shapes[key] for key in sorted(shapes)]

def run(plan_path,output,device,repeats):
    output.mkdir(parents=True,exist_ok=False);plan=load(plan_path);save(output/'plan.json',plan)
    modes=plan.get('modes',list(MODES))
    if not isinstance(modes,list) or not modes or any(m not in MODES for m in modes) or len(set(modes))!=len(modes):
        raise ValueError('Choose unique nonempty comparison modes')
    shapes=plan.get('scoring_shapes') or public_shapes(plan['cases'])
    save(output/'resolved-scoring-shapes.json',shapes)
    backend=REPO/'scratch/fitting_batch_trial/bin/odezza-fit-core-run'
    engine=Engine(backend,[device],scoring_cache=output/'artifacts')
    state=dict(status='preparing',started_at=time.time(),completed=0);save(output/'status.json',state)
    try:
        engine.prepare_scoring(shapes,output/'cold-preparation')
        engine.close();engine=Engine(backend,[device],scoring_cache=output/'artifacts')
        warm=engine.prepare_scoring(shapes,output/'restart-preparation')
        if any(r['template_source']!='disk' for r in warm['records']):raise RuntimeError('Restart did not use saved artifacts')
        from lm_toggle.native_service import Adapter
        def prepare_lm():
            adapter=Adapter(device);engine.lm_adapters[device]=adapter
            states=sorted({len(c['public']['problem']['states']) for c in plan['cases']})
            return adapter.prepare_shapes([(n,p,w) for n in states for p in range(3,9) for w in (1,8)])
        before=time.monotonic();lm=engine.pools[device].submit(prepare_lm).result()
        save(output/'lm-readiness.json',dict(seconds=time.monotonic()-before,shapes=lm))
        state['status']='running'
        for repeat in range(repeats):
            for index,case in enumerate(plan['cases']):
                order=list(modes);offset=(repeat+index)%len(order);order=order[offset:]+order[:offset]
                if repeat%2:order.reverse()
                for mode in order:
                    label=f'{case["id"]}-{mode}-r{repeat}';state['current']=label;save(output/'status.json',state)
                    # Same process-start boundary in every arm; no policy runs
                    # second against the first arm's problem-specific pipeline.
                    engine.workers[device].close()
                    opts=dict(case['settings'],seconds=plan['seconds'],**MODES[mode])
                    before=time.monotonic();r=search(case['public'],opts,'native_lm',output/label,engine)
                    r.update(case_id=case['id'],mode=mode,repeat=repeat,total_seconds=time.monotonic()-before,
                        public_sha256=digest(case['public']),options=opts)
                    setup=r['work_report']['scoring_setup']
                    r['unprepared_scoring_templates']=[s for s in setup if s.get('template_source')=='compiled']
                    r['unprepared_lm_batches']=[dict(report=p['report'],seconds=p.get('prepare_seconds'))
                        for p in r['work_report']['native_lm_profiles'] if not p.get('template_cache_hit')]
                    r['cache_control_valid']=not r['unprepared_scoring_templates'] and not r['unprepared_lm_batches']
                    append(output/'results.jsonl',r);state['completed']+=1
                    print(label,r['status'],round(r['total_seconds'],3),'cache_control',r['cache_control_valid'],flush=True)
                    if r['status']=='error' or not r['cache_control_valid']:
                        raise RuntimeError('Search or preparation boundary failed; retain records and repair before continuing')
        state['status']='complete'
    except Exception as error:
        state.update(status='error',error=str(error));raise
    finally:
        state['finished_at']=time.time();save(output/'status.json',state);engine.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();run(a.plan,a.output,a.device,a.repeats)
