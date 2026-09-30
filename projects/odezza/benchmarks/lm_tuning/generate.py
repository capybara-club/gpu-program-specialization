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
"""Portable random systems with independently controlled RHS dependency breadth.

Only public.json goes to blind recovery. Calibration candidates are a separate,
explicitly planted-structure fitting task; coefficients and seeds stay private.
"""
import argparse
import copy
import math
from pathlib import Path
import random
import secrets
import struct
import sys
import time

from .common import REPO, save, load, integer, require_keys, digest
sys.path.insert(0, str(REPO))
import ode_game as game

OPCODES = {'add': 0x90, 'sub': 0x91, 'mul': 0x92, 'div': 0x93,
           'sin': 0x9b, 'cos': 0x9c, 'tanh': 0xa0, 'exp': 0xa1}


def leaves(tree, kind):
    if tree[0] in ('state', 'constant'):
        return [tree] if tree[0] == kind else []
    return [leaf for child in tree[1:] for leaf in leaves(child, kind)]


def depth(tree):
    return 0 if tree[0] in ('state', 'constant') else 1 + max(map(depth, tree[1:]))


def remap(tree, domain):
    if tree[0] == 'state':
        return ('state', domain[tree[1]])
    if tree[0] == 'constant':
        return tree
    return (tree[0], *(remap(child, domain) for child in tree[1:]))


def encode(tree, fitted=True):
    values = []
    def visit(node):
        if node[0] == 'state':
            return bytes([0x81, node[1]])
        if node[0] == 'constant':
            if not fitted:
                return b'\x83' + struct.pack('<f', node[1])
            index = len(values)
            values.append(node[1])
            return bytes([0x82, index])
        return b''.join(visit(child) for child in node[1:]) + bytes([OPCODES[node[0]]])
    return (visit(tree) + b'\x80').hex(), values


def paths(tree):
    return [()] + ([] if tree[0] in ('state', 'constant') else
                  [(i,) + p for i, child in enumerate(tree[1:], 1) for p in paths(child)])


def replace(tree, path, value):
    if not path:
        return value
    children = list(tree)
    children[path[0]] = replace(children[path[0]], path[1:], value)
    return tuple(children)


def subtree(tree, path):
    for i in path:
        tree = tree[i]
    return tree


def validate(spec):
    require_keys(spec, {'states', 'depth', 'dependencies', 'parameters'},
                 {'known_dependencies', 'background', 'unary', 'binary', 'constants',
                  'duration', 'samples', 'trajectories', 'view', 'attempts', 'seconds', 'max_step'})
    n = integer(spec['states'], 'states', 2, 16)
    d = integer(spec['depth'], 'depth', 1, 4)
    k = integer(spec['dependencies'], 'dependencies', 1, n)
    p = integer(spec['parameters'], 'parameters', 0, 12)
    integer(spec.get('known_dependencies', k), 'known_dependencies', 1, n)
    if k + p > 2 ** d:
        raise ValueError('Requested state and coefficient leaves cannot fit at this AST depth')
    if spec.get('background', 'random') not in ('random', 'linear_control'):
        raise ValueError('Unknown background distribution')
    if spec.get('view', 'dense') not in ('dense', 'sparse', 'sparse_hidden'):
        raise ValueError('Unknown observation view')
    integer(spec.get('samples', 21), 'samples', 7, 101)
    integer(spec.get('trajectories', 6), 'trajectories', 6, 12)
    integer(spec.get('attempts', 200), 'attempts', 1, 2000)
    for name in ('duration', 'seconds', 'max_step'):
        value = spec.get(name, {'duration': .5, 'seconds': 90., 'max_step': .005}[name])
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid ' + name)
    return spec


def sample_tree(cfg, rng, dependencies, parameters=None):
    domain = rng.sample(range(cfg['states']), dependencies)
    small = dict(cfg, states=dependencies, blinded_rhs=['x' + str(dependencies - 1)])
    total = game.counts(small)[-1]
    for _ in range(10000):
        tree = game.assign_constants(game.unrank(small, rng.randrange(total)), rng,
                                     game.bounds(cfg['grammar']['constants']))
        if len({leaf[1] for leaf in leaves(tree, 'state')}) != dependencies:
            continue
        if parameters is not None and len(leaves(tree, 'constant')) != parameters:
            continue
        return remap(tree, domain)
    raise ValueError('tree_acceptance_exhausted')


def active_dependencies(tree, n, rng):
    """Reject obvious cancellations; this is a probe diagnostic, not a proof."""
    declared = sorted({leaf[1] for leaf in leaves(tree, 'state')})
    active = set()
    for _ in range(12):
        point = [rng.uniform(-.6, .6) for _ in range(n)]
        for i in declared:
            plus, minus = point[:], point[:]
            plus[i] += 1e-5
            minus[i] -= 1e-5
            try:
                slope = (game.evaluate(tree, plus) - game.evaluate(tree, minus)) / 2e-5
                if math.isfinite(slope) and abs(slope) > 1e-7:
                    active.add(i)
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
    return sorted(active)


def candidates(tree, cfg, rng, count=64):
    result = {encode(tree)[0]: 'planted'}
    positions = paths(tree)
    for attempt in range(20000):
        if len(result) >= count:
            break
        path = rng.choice(positions)
        node = subtree(tree, path)
        if node[0] == 'state':
            changed = ('state', rng.randrange(cfg['states']))
        elif node[0] == 'constant':
            continue
        else:
            choices = cfg['grammar']['unary' if len(node) == 2 else 'binary']
            changed = (rng.choice(choices), *node[1:])
        variant = replace(tree, path, changed)
        # Several edits create a larger prepared pool without a target grammar.
        if attempt % 2:
            for other in rng.sample(positions, min(2, len(positions))):
                leaf = subtree(variant, other)
                if leaf[0] == 'state':
                    variant = replace(variant, other, ('state', rng.randrange(cfg['states'])))
        result.setdefault(encode(variant)[0], 'structural_neighbour')
    items = list(result)
    rng.shuffle(items)
    return items, result


def generate(spec, out, seed=None):
    validate(spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed = secrets.randbits(128) if seed is None else seed
    save(out / 'private/seed.json', {'seed': seed, 'specification': spec})
    rng = random.Random(seed)
    cfg = load(REPO / 'examples/grammar_game/depth3.json')
    n = spec['states']
    cfg.update(states=n, blinded_rhs=['x' + str(n - 1)])
    cfg['grammar'].update(max_depth=spec['depth'], unary=spec.get('unary', ['sin', 'cos']),
                          binary=spec.get('binary', ['add', 'sub', 'mul', 'div']),
                          constants=spec.get('constants', {'min': -1., 'max': 1.}))
    cfg['trajectories'].update(count=spec.get('trajectories', 6), duration=spec.get('duration', .5),
                               samples=spec.get('samples', 21))
    game.validate(cfg)
    started = time.monotonic()
    rejected = []
    accepted = False
    for attempt in range(spec.get('attempts', 200)):
        if time.monotonic() - started > spec.get('seconds', 90):
            break
        try:
            unknown = sample_tree(cfg, rng, spec['dependencies'], spec['parameters'])
            trees = [sample_tree(cfg, rng, spec.get('known_dependencies', spec['dependencies'])) for _ in range(n - 1)]
            if spec.get('background', 'random') == 'linear_control':
                trees = [('add', ('mul', ('constant', -rng.uniform(.4, .9)), ('state', i)),
                          ('mul', ('constant', rng.uniform(.03, .15)), ('state', (i + 1) % n))) for i in range(n - 1)]
            trees.append(unknown)
            matrix = [sorted({leaf[1] for leaf in leaves(t, 'state')}) for t in trees]
            detected = [active_dependencies(t, n, rng) for t in trees]
            if matrix != detected:
                raise ValueError('cancelled_or_numerically_inactive_dependency')
            records = []
            difference = 0.
            times = [cfg['trajectories']['duration'] * i / (cfg['trajectories']['samples'] - 1)
                     for i in range(cfg['trajectories']['samples'])]
            for i in range(cfg['trajectories']['count']):
                initial = [rng.uniform(-.5, .5) for _ in range(n)]
                coarse = game.integrate(trees, initial, times, spec.get('max_step', .005), 20.)
                fine = game.integrate(trees, initial, times, spec.get('max_step', .005) / 2, 20.)
                for a, b in zip(coarse, fine):
                    for x, y in zip(a, b):
                        difference = max(difference, abs(x - y))
                        if abs(x - y) > 1e-9 + 1e-8 * abs(y):
                            raise ValueError('reference_step_doubling_disagreement')
                if max(row[-1] for row in fine) - min(row[-1] for row in fine) < .001:
                    raise ValueError('unknown_rhs_trajectory_span')
                records.append(dict(trajectory_id=str(i), times=times, initial_state=initial, states=fine))
            accepted = True
            break
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            rejected.append(dict(attempt=attempt + 1, reason=str(error)))
            save(out / 'generation.json', {'status': 'generating', 'rejected': rejected})
    if not accepted:
        save(out / 'generation.json', {'status': 'rejected', 'attempts': rejected, 'seconds': time.monotonic() - started})
        return None
    states = ['x' + str(i) for i in range(n)]
    known = {s: game.expression(t) for s, t in zip(states[:-1], trees[:-1])}
    view = spec.get('view', 'dense')
    if view != 'dense':
        indices = sorted({round(i * (len(times) - 1) / 6) for i in range(7)})
        for row in records:
            row['times'] = [row['times'][i] for i in indices]
            row['states'] = [row['states'][i] for i in indices]
            if view == 'sparse_hidden':
                for point in row['states']:
                    point[0] = None
        cfg['trajectories'].update(samples=7, observed_states=states[1:] if view == 'sparse_hidden' else 'all')
    split = {'training': [str(i) for i in range(len(records) - 2)], 'validation': [str(len(records) - 2)], 'test': [str(len(records) - 1)]}
    public = dict(schema=game.PUBLIC_SCHEMA, grammar=cfg,
                  problem=dict(states=states, known_rhs=known, unknown_rhs=[states[-1]], noise='none'), splits=split,
                  knowns='\n'.join(f'd{s}/dt = {known.get(s, "???")}' for s in states),
                  ics=dict(columns=['trajectory_id', *states], rows=[[r['trajectory_id'], *r['initial_state']] for r in records]),
                  trajectories=dict(columns=['trajectory_id', 't', *states], rows=[[r['trajectory_id'], t, *y] for r in records for t, y in zip(r['times'], r['states'])]))
    public['manifest'] = {'content_sha256': game.digest(public)}
    save(out / 'public.json', public)
    programs, kinds = candidates(unknown, cfg, rng)
    code, coefficients = encode(unknown)
    calibration_rows = [{key: value for key, value in row.items() if key != 'trajectory_id'} for row in records]
    save(out / 'calibration.json', dict(scope='prepared_candidates_with_planted_structure_not_blind_recovery',
         states=states, parameters=['p' + str(i) for i in range(len(coefficients))], unknown_state=states[-1],
         known_rhs=known, data={'trajectories': calibration_rows[:-2]}, validation={'trajectories': calibration_rows[-2:-1]},
         test={'trajectories': calibration_rows[-1:]}, candidates=[dict(id=str(i), program_hex=c) for i, c in enumerate(programs)]))
    save(out / 'private/solution.json', dict(rhs=trees, dependency_matrix=matrix, active_probe_matrix=detected,
         coefficient_values=coefficients, planted_program=code, candidate_kinds=kinds, specification=spec))
    result = dict(status='accepted', states=n, depth=spec['depth'], actual_rhs_depths=list(map(depth, trees)),
                  parameters=len(coefficients), dependency_counts=list(map(len, matrix)), view=view,
                  background=spec.get('background', 'random'), attempts=attempt + 1, rejections=rejected,
                  seconds=time.monotonic() - started, reference='FP64 RK4 step doubling', reference_difference=difference,
                  public_sha256=digest(public), candidate_count=len(programs))
    save(out / 'generation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = generate(load(args.spec), args.out)
    print(__import__('json').dumps(result))
    if result is None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
