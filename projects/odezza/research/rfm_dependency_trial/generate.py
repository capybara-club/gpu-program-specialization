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
"""Private synthetic generator; learner is a separate public-only process."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def ast(rng, cfg, depth):
    if depth == 0:
        return ['c', float(rng.uniform(-1, 1))] if rng.random() < cfg['constant_probability'] else ['x', int(rng.integers(cfg['states']))]
    op = str(rng.choice(cfg['operators']))
    args = [ast(rng, cfg, depth - 1)]
    if op in ('add', 'sub', 'mul'):
        args.append(ast(rng, cfg, depth - 1))
    return [op] + args


def symbolic(a, symbols):
    op = a[0]
    if op == 'x': return symbols[a[1]]
    if op == 'c': return sp.Float(a[1])
    args = [symbolic(x, symbols) for x in a[1:]]
    return {'add': lambda: args[0] + args[1], 'sub': lambda: args[0] - args[1],
            'mul': lambda: args[0] * args[1], 'sin': lambda: sp.sin(args[0]),
            'cos': lambda: sp.cos(args[0]), 'tanh': lambda: sp.tanh(args[0])}[op]()


def generate(config, out):
    cfg = json.loads(Path(config).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'private/config.json', cfg)
    rng = np.random.default_rng(cfg['generation_seed'])
    n = cfg['states']
    symbols = sp.symbols('x0:' + str(n))
    times = np.arange(round(max(cfg['durations']) / cfg['spacing']) + 1) * cfg['spacing']
    ntr = max(cfg['counts']) + cfg['validation_trajectories']
    public_cfg = {k: v for k, v in cfg.items() if k not in ('generation_seed',)}
    manifest = {'config': public_cfg, 'cases': [], 'rejections': [], 'started_unix': time.time()}
    start = time.perf_counter()
    for attempt in range(cfg['max_generation_attempts']):
        trees = [ast(rng, cfg, cfg['depth']) for _ in range(n)]
        expr = [sp.simplify(cfg['rhs_scale'] * symbolic(a, symbols)) for a in trees]
        functions = [sp.lambdify(symbols, f, 'numpy') for f in expr]
        def rhs(t, y):
            with np.errstate(over='ignore', invalid='ignore'):
                return np.asarray([f(*y) for f in functions], dtype=float)
        def bound(t, y): return cfg['state_bound'] - np.max(np.abs(y))
        bound.terminal = True
        ics = rng.uniform(*cfg['initial_range'], size=(ntr, n))
        trajectories = []
        rejected = None
        for ic in ics:
            sol = solve_ivp(rhs, (0., times[-1]), ic, t_eval=times, method='DOP853',
                            rtol=1e-10, atol=1e-12, events=bound, max_step=0.1)
            if not sol.success or sol.y.shape[1] != len(times) or not np.isfinite(sol.y).all():
                rejected = 'integration_failure_or_bound'
                break
            trajectories.append(sol.y.T)
        if rejected:
            manifest['rejections'].append({'attempt': attempt, 'reason': rejected})
            continue
        i = len(manifest['cases'])
        split = 'development' if i < cfg['development_systems'] else 'test'
        name = f'{i:03d}'
        support = [[bool(sp.simplify(sp.diff(f, x)) != 0) for x in symbols] for f in expr]
        data = {'schema': 'odezza.rfm-observations.v1', 'id': name, 'split': split,
                'times': times.tolist(), 'observations': np.asarray(trajectories).tolist(),
                'training_trajectories': max(cfg['counts'])}
        save(out / f'public/{name}.json', data)
        save(out / f'private/{name}.json', {'asts': trees, 'rhs': [str(f) for f in expr],
                                          'support': support, 'attempt': attempt})
        manifest['cases'].append({'id': name, 'split': split, 'path': f'public/{name}.json',
                                 'sha256': hashlib.sha256((out / f'public/{name}.json').read_bytes()).hexdigest()})
        print(json.dumps({'generated': name, 'attempt': attempt, 'split': split}), flush=True)
        if i + 1 == cfg['systems'] + cfg['development_systems']: break
    manifest['generation_seconds'] = time.perf_counter() - start
    manifest['complete'] = len(manifest['cases']) == cfg['systems'] + cfg['development_systems']
    save(out / 'manifest.json', manifest)
    if not manifest['complete']: raise RuntimeError('Generation attempt budget exhausted')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('out')
    a = p.parse_args(); generate(a.config, a.out)
