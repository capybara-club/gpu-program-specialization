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
"""Version-two sparse-quadratic grammar preparation; native C expands its ASTs.

This is a finite search profile, independent of the CUDA kernel implementation.
Allocation and numeric sampling are explicit; no private solution input exists.
"""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


PROFILE_PATH = Path(__file__).with_name('sparse-profile.json')
CATEGORIES = ('baseline', 'simple', 'compound', 'mixed', 'double_compound')


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def load_profile(profile=None, rows=None):
    result = deepcopy(profile) if profile is not None else json.loads(PROFILE_PATH.read_text())
    keys = {'version', 'coefficient_rows', 'ranges', 'retained_per_family', 'max_configurations', 'max_seconds'}
    if set(result) != keys or result['version'] != 2:
        raise ValueError('expected the version-two profile fields')
    if set(result['coefficient_rows']) != set(CATEGORIES):
        raise ValueError('coefficient_rows must allocate each argument class explicitly')
    if rows is not None:
        positive_int(rows, 'rows')
        result['coefficient_rows'] = dict.fromkeys(CATEGORIES, rows)
    for name, count in result['coefficient_rows'].items():
        positive_int(count, name)
    for key in ('retained_per_family', 'max_configurations', 'max_seconds'):
        positive_int(result[key], key)
    if set(result['ranges']) != {'linear', 'quadratic', 'argument', 'amplitude'}:
        raise ValueError('ranges must define linear, quadratic, argument and amplitude')
    for bounds in result['ranges'].values():
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds) or
                not bounds[0] < bounds[1]):
            raise ValueError('coefficient ranges must be finite increasing pairs')
    return result


def monomials(states):
    """Canonical monomials: constant, linear states, then unordered quadratics."""
    return [()] + [(s,) for s in states] + list(itertools.combinations_with_replacement(states, 2))


def text_monomial(monomial):
    return '*'.join(monomial) if monomial else '1'


class ArgumentCompiler:
    def __init__(self, grammar):
        self.grammar = grammar
        self.states = grammar['states']
        self.monomials = monomials(self.states)
        self.counter = 0

    def state_choices(self, values, role):
        """Disjoint 4/2/1 groups; never pad a toggle with repeated states."""
        choices = []
        offset = 0
        while offset < len(values):
            remaining = len(values) - offset
            width = 4 if remaining >= 4 else 2 if remaining >= 2 else 1
            group = list(values[offset:offset + width])
            if width == 1:
                choices.append(group[0])
            else:
                name = f'{role}_{self.counter}'
                self.counter += 1
                self.grammar['leaves'][name] = dict(states=group, arity=width, coverage='explicit', groups=[group])
                choices.append('leaf.' + name)
            offset += width
        return choices

    def packed_monomials(self, items, role):
        choices = ['1'] if () in items else []
        linear = [m[0] for m in items if len(m) == 1]
        choices.extend(self.state_choices(linear, role))
        for first in self.states:
            tails = [m[1] for m in items if len(m) == 2 and m[0] == first]
            choices.extend(f'({first}*{tail})' for tail in self.state_choices(tails, role))
        return choices

    def argument(self, role, kind):
        name = f'Argument{role}{kind.title()}'
        if name in self.grammar['rules']:
            return name
        first = 'rng.freq' + role
        if kind == 'simple':
            choices = [f'{first}*({m})' for m in self.packed_monomials(self.monomials, role)]
        else:
            choices = []
            # Distinct unordered monomial pairs. Coefficients remain independent,
            # so the reverse ordering has no additional structural support.
            for i, monomial in enumerate(self.monomials[:-1]):
                for tail in self.packed_monomials(self.monomials[i + 1:], role):
                    choices.append(f'({first}*({text_monomial(monomial)})+rng.mix{role}*({tail}))')
        self.grammar['rules'][name] = choices
        return name


def build(problem, *, rows=None, seed=2026091401, indices=None, dt=.125, hint=None, profile=None):
    profile = load_profile(profile, rows)
    states = problem['states']
    if (not states or len(states) != len(set(states)) or any(
            not isinstance(s, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', s) for s in states)):
        raise ValueError('unique identifier state names required')
    known = problem.get('known_rhs', {})
    if not set(known) <= set(states):
        raise ValueError('known RHS references an undeclared state')
    unknown = [s for s in states if s not in known]
    if len(unknown) != 1:
        raise ValueError('this portfolio requires exactly one unknown RHS')
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be uint64')
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if not problem['trajectories']:
        raise ValueError('at least one trajectory required')
    indices = list(range(len(problem['trajectories']))) if indices is None else list(indices)
    if (not indices or len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < len(problem['trajectories']) for i in indices)):
        raise ValueError('trajectory selection must contain distinct valid indices')
    if hint is not None and (not isinstance(hint, str) or not hint.strip()):
        raise ValueError('public motif must be a nonempty expression')

    grammar = dict(version=1, states=list(states), integration=dict(method='rk4', dt=dt),
        rules={}, leaves={}, rng_banks={}, families=[],
        expansion=dict(strategy='enumerate', max_nodes=160, max_depth=64, max_expansion_depth=64),
        retain={'global': dict(k=32, unit='resolved_structure'),
                'per_family': dict(k=profile['retained_per_family'], unit='resolved_structure')})
    compiler = ArgumentCompiler(grammar)
    m = len(compiler.monomials)
    sizes = {'simple': m, 'compound': m * (m - 1) // 2}
    args = {(role, kind): compiler.argument(role, kind) for role in ('A', 'B') for kind in sizes}
    quadratic = [q for q in compiler.monomials if len(q) == 2]
    grammar['rules']['Quadratic'] = compiler.packed_monomials(quadratic, 'Q')
    linear = '+'.join('rng.linear' + s + '*' + s for s in states)
    backgrounds = [('linear', linear, 1), ('linear_quadratic', linear + '+rng.quadratic*Quadratic', len(quadratic))]
    params = {**{'linear' + s: 'linear' for s in states}, 'quadratic': 'quadratic',
              'freqA': 'argument', 'mixA': 'argument', 'freqB': 'argument', 'mixB': 'argument',
              'amplitude': 'amplitude', 'amplitudeB': 'amplitude'}
    manifest = []

    def family(mechanism, background, expression, count, category, argument_roles):
        ident = mechanism + '__' + background + '__' + '_'.join(argument_roles.values())
        ident = ident.rstrip('_')
        bank = 'bank_' + category
        count_rows = profile['coefficient_rows'][category]
        grammar['rng_banks'][bank] = dict(base='uniform01', count=count_rows, seed=seed, scope='run')
        rng = {name: dict(bank=bank, axis='trial', stream=name,
                         transform=dict(kind='uniform', low=profile['ranges'][kind][0], high=profile['ranges'][kind][1]))
               for name, kind in params.items()}
        configs = count * count_rows
        grammar['families'].append(dict(id=ident,
            tags=['mechanism:' + mechanism, 'background:' + background, 'argument_class:' + category],
            rhs={unknown[0]: expression}, rng=rng,
            limits=dict(max_skeletons=count + 1, max_variants=count + 1, max_configurations=configs + count_rows,
                        max_derivations=max(100, (count + 1) * 20), max_expansion_steps=max(10000, (count + 1) * 1000))))
        manifest.append(dict(id=ident, mechanism=mechanism, background=background, argument_class=category,
            argument_roles=argument_roles, coefficient_rows=count_rows, resolved_syntax_count=count,
            expected_configurations=configs, coverage='finite_enumeration'))

    for background, base, bg_count in backgrounds:
        family('baseline', background, base, bg_count, 'baseline', {})
        if hint:
            family('hint', background, base + '+rng.amplitude*(' + hint + ')', bg_count, 'simple', {})
        for kind, arg_count in sizes.items():
            a = args['A', kind]
            for unary in ('sin', 'cos'):
                if hint:
                    for join, term in [('times', f'rng.amplitude*({hint})*{unary}({a})'),
                                       ('plus', f'rng.amplitude*({hint})+rng.amplitudeB*{unary}({a})')]:
                        family(f'hint_{join}_{unary}', background, base + '+' + term,
                               bg_count * arg_count, kind, {'A': kind})
                else:
                    family(unary, background, base + f'+rng.amplitude*{unary}({a})',
                           bg_count * arg_count, kind, {'A': kind})
        if hint:
            continue
        for ka, kb in itertools.product(sizes, repeat=2):
            a, b = args['A', ka], args['B', kb]
            category = 'simple' if ka == kb == 'simple' else 'double_compound' if ka == kb else 'mixed'
            for left, right in [('sin', 'sin'), ('sin', 'cos'), ('cos', 'cos')]:
                for join, term in [('times', f'rng.amplitude*{left}({a})*{right}({b})'),
                                   ('plus', f'rng.amplitude*{left}({a})+rng.amplitudeB*{right}({b})')]:
                    family(f'{left}_{join}_{right}', background, base + '+' + term,
                           bg_count * sizes[ka] * sizes[kb], category, {'A': ka, 'B': kb})
    if len(manifest) > 64:
        raise ValueError('portfolio exceeds the current 64-family service limit')
    allocated = sum(f['limits']['max_configurations'] for f in grammar['families'])
    if allocated > profile['max_configurations']:
        raise ValueError(f'portfolio reserves {allocated:,} configurations above {profile["max_configurations"]:,}; reduce coefficient_rows explicitly')
    selected = deepcopy(problem)
    selected['trajectories'] = [selected['trajectories'][i] for i in indices]
    request = dict(problem=selected, grammar=grammar,
                   execution=dict(max_seconds=profile['max_seconds'], dedup_bytes_per_family=2 << 20))
    plan = dict(version=2, profile=profile, seed=seed, public_motif=hint,
        scope='public-clue-assisted grammar' if hint else 'operator-only sparse-quadratic grammar',
        unknown_rhs=unknown[0], trajectory_indices=indices,
        monomial_catalog=[text_monomial(item) for item in compiler.monomials], argument_counts=sizes,
        families=manifest, expected_configurations=sum(f['expected_configurations'] for f in manifest),
        reserved_configurations=allocated,
        category_configurations={category: sum(f['expected_configurations'] for f in manifest if f['argument_class'] == category)
                                 for category in CATEGORIES},
        coverage='Finite structural enumeration; coefficients sampled. Syntax counts are not unique functions.',
        priors=['one unknown RHS; linear background and optional quadratic term',
                'one or two distinct monomials per argument, including a constant',
                'per-class numeric allocations; no implicit redistribution',
                'complete selected trajectories; no implicit holdout selection'])
    plan['grammar_sha256'] = hashlib.sha256(json.dumps(grammar, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return request, plan
