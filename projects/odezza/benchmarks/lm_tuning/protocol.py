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
"""Versioned, result-independent case and hyperparameter sampling."""
import copy
import math
import random
from .common import SCHEMA, require_keys, integer, digest


def validate(value):
    require_keys(value, {'schema', 'name', 'deadline_utc', 'hosts', 'pilot', 'generation', 'fit', 'search', 'schedule'})
    if value['schema'] != SCHEMA:
        raise ValueError('Unknown protocol schema')
    if not value['name'].isidentifier():
        raise ValueError('Protocol name must be an identifier')
    if not value['hosts'] or set(value['hosts']) - {'rack1', 'rohini', 'ada'} or len(set(value['hosts'])) != len(value['hosts']):
        raise ValueError('Invalid hosts')
    from datetime import datetime
    deadline = datetime.fromisoformat(value['deadline_utc'].replace('Z', '+00:00'))
    if deadline.utcoffset() is None:
        raise ValueError('Deadline requires an explicit timezone')
    integer(value['pilot']['count'], 'pilot count', 1, 100)
    for section in ('fit', 'search'):
        for key, values in value[section].items():
            if not isinstance(values, list) or not values:
                raise ValueError(f'{section}.{key} must be a nonempty list')
    if any(n not in (2, 3, 4, 5, 6, 8) for n in value['generation']['states']):
        raise ValueError('This native LM campaign supports 2..8 states')
    if set(value['fit']['toggle_width']) - {1, 2, 4}:
        raise ValueError('Invalid LM toggle width')
    if any(n < 1 or n > 65536 for n in value['fit']['starts']):
        raise ValueError('Invalid prepared bank size')
    return value


def rng_for(seed, label):
    return random.Random(int(digest([seed, label]), 16))


def sample_case(protocol, index):
    """Design seed is public; equation generation uses a separate private seed."""
    rng = rng_for(protocol['schedule']['design_seed'], ['case', index])
    g = protocol['generation']
    for _ in range(1000):
        n = rng.choice(g['states'])
        d = rng.choice(g['depths'])
        k = rng.choice(g['dependencies'])
        p = rng.choice(g['parameters'])
        if k <= n and k + p <= 2 ** d:
            break
    else:
        raise ValueError('No feasible dependency/parameter/depth combination')
    return dict(states=n, depth=d, dependencies=k, parameters=p,
                known_dependencies=min(n, rng.choice(g['known_dependencies']), 2 ** d),
                background=rng.choice(g['backgrounds']), view=rng.choice(g['views']),
                duration=rng.choice(g['durations']), samples=21, trajectories=6,
                unary=g['unary'], binary=g['binary'], constants=g['constants'],
                attempts=g['attempts'], seconds=g['seconds_per_case'], max_step=.005)


def settings(protocol, kind, index):
    rng = rng_for(protocol['schedule']['design_seed'], [kind, index])
    selected = {key: rng.choice(values) for key, values in protocol[kind].items()}
    if kind == 'fit':
        # Keep total output bounded while permitting multiple resident waves.
        feasible = [n for n in protocol['fit']['candidates'] if n * selected['starts'] <= 4194304]
        selected['candidates'] = rng.choice(feasible)
    else:
        selected['offspring'] = min(10000000, selected['wave'] * selected['generation_limit'])
    return selected


def pair(case_id, payload, kind, settings_value, index, phase='overnight'):
    methods = ['curvature', 'native_lm']
    if index % 2:
        methods.reverse()
    trials = [dict(method=method, settings=copy.deepcopy(settings_value), seconds=settings_value['seconds']) for method in methods]
    result = dict(schema=SCHEMA, case_id=case_id, kind=kind, phase=phase, replicate_index=index,
                  payload=payload, trials=trials,
                  comparison='Same inputs and nominal settings; adaptive recovery paths and fitter work may differ')
    result['group_id'] = digest(result)[:24]
    return result
