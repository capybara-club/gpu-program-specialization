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
"""Fixed contrasts for the corrected specialized LM and blind search policies."""
import copy
from .common import SCHEMA, digest


def search_variants(seconds=240):
    base = dict(initial=65536, banks=1024, wave=65536, generation_limit=64,
        offspring=4194304, parents=1024, candidates=64, starts=4, iterations=16,
        fit_seconds=20, toggle_width=4, leaf_toggle_width=4,
        initial_damping=.001, damping_attempts=8, max_step=.4, seconds=seconds,
        lanes_per_fit=1, shape_fallback=True)
    contrasts = [
        ('lm_baseline', 'native_lm', {}),
        ('curvature', 'curvature', {}),
        ('more_structures', 'native_lm', dict(initial=262144,banks=256,wave=262144,offspring=10000000)),
        ('more_coefficients', 'native_lm', dict(initial=16384,banks=4096,wave=16384,offspring=1048576)),
        ('fewer_bindings', 'native_lm', dict(initial=131072,wave=131072,offspring=8388608,leaf_toggle_width=2)),
        ('fit_more_candidates', 'native_lm', dict(candidates=128,starts=2)),
        ('fit_fewer_candidates', 'native_lm', dict(candidates=16,starts=16)),
        ('fit_more_starts', 'native_lm', dict(starts=16)),
    ]
    return [dict(trial_id=name, method=method, settings=dict(base,**change), seconds=seconds)
            for name,method,change in contrasts]


def fit_variants(seconds=20):
    base = dict(candidates=8, starts=64, iterations=16, toggle_width=4,
        initial_damping=.001,damping_attempts=8,max_step=.4,batch_size=4096,
        seconds=seconds,submission_packs=2,buffer_fit_limit=131072,
        lanes_per_fit=1,shape_fallback=True)
    result=[]
    for starts in (4,64,1024,16384):
        for width in (1,2,4,8):
            result.append(dict(trial_id=f'bank{starts}_lane{width}',method='native_lm',
                settings=dict(base,starts=starts,lanes_per_fit=width,shape_fallback=False),seconds=seconds))
    for starts in (4,64,1024):
        result.append(dict(trial_id=f'bank{starts}_queue16',method='native_lm',
            settings=dict(base,starts=starts,submission_packs=16),seconds=seconds))
        result.append(dict(trial_id=f'bank{starts}_auto',method='native_lm',
            settings=dict(base,starts=starts),seconds=seconds))
    result.append(dict(trial_id='bank64_toggle1',method='native_lm',settings=dict(base,toggle_width=1),seconds=seconds))
    result.append(dict(trial_id='bank64_curvature',method='curvature',settings=dict(base),seconds=seconds))
    return result


def calibration_subset(payload, planted, count=8):
    """Explicit planted-structure calibration only; never called for search."""
    result=copy.deepcopy(payload)
    pool=payload['candidates']
    target=next(c for c in pool if c['program_hex']==planted)
    others=[c for c in pool if c['program_hex']!=planted][:count-1]
    chosen={c['id'] for c in [target,*others]}
    result['candidates']=[copy.deepcopy(c) for c in pool if c['id'] in chosen]
    return result


def specifications(repeats=4):
    """Balanced N/P cells; private equation seeds are generated separately."""
    result=[]
    for repetition in range(repeats):
        for n in (3,6,8):
            for p in (1,3,6):
                result.append(dict(states=n,depth=3,dependencies=1+repetition%2,
                    known_dependencies=2,parameters=p,background='random',
                    view='dense' if repetition%4<2 else 'sparse_hidden',
                    duration=.5,samples=21,trajectories=6,unary=['sin','cos'],
                    binary=['add','sub','mul','div'],constants={'min':-1.,'max':1.},
                    attempts=200,seconds=60,max_step=.005))
    return result


def group(case_id,payload,kind,variants,index,seed,phase):
    trials=copy.deepcopy(variants)
    # Rotate order within a matched block; each policy receives the same bank /
    # search seed. Retain the order effect rather than calling it fully warm.
    shift=index%len(trials)
    trials=trials[shift:]+trials[:shift]
    for trial in trials:trial['settings']['seed']=seed
    d=dict(schema=SCHEMA,case_id=case_id,kind=kind,phase=phase,
        replicate_index=index,payload=payload,trials=trials,
        comparison='Same case and search/bank seed, all policies on one GPU; fixed contrasts, no result-driven selection')
    d['group_id']=digest(d)[:24]
    return d


def reserved_seconds(group):
    return sum(t['seconds']+30 for t in group['trials'])+60
