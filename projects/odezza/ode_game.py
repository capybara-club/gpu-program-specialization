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
"""Seeded ODE grammar game. Generation and compilation use only Python's stdlib.

Copy this file and a game JSON anywhere to generate a challenge. Only the
optional submit command imports the existing Odezza repository client.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import sys
import tempfile
import zipfile

SCHEMA = 'odezza.grammar_game.v1'
PUBLIC_SCHEMA = 'odezza.grammar_game.public.v1'
UNARY = {'sin': math.sin, 'cos': math.cos, 'tanh': math.tanh, 'exp': math.exp}
BINARY = {'add', 'sub', 'mul', 'div'}
PUBLIC_FILES = {'grammar.json', 'problem.json', 'splits.json', 'knowns.txt',
                'ics.csv', 'trajectories.csv', 'manifest.json'}


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value)-set(required)-set(optional) or set(required)-set(value):
        raise ValueError('Expected fields '+str(sorted(required))+'; optional '+str(sorted(optional)))


def integer(value, lo, hi, label):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{label} must be an integer in [{lo}, {hi}]')
    return value


def finite(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(label+' must be finite'+(' and positive' if positive else ''))
    return value


def bounds(value):
    fields(value, {'min', 'max'})
    a, b = finite(value['min'], 'min'), finite(value['max'], 'max')
    if not -1e6 <= a < b <= 1e6:
        raise ValueError('Bounds must satisfy -1e6 <= min < max <= 1e6')
    return a, b


def validate(config):
    fields(config, {'schema', 'states', 'grammar', 'blinded_rhs', 'trajectories', 'generation'})
    if config['schema'] != SCHEMA:
        raise ValueError('Unsupported game schema')
    states = ['x'+str(i) for i in range(integer(config['states'], 1, 32, 'states'))]
    g = config['grammar']; fields(g, {'max_depth', 'unary', 'binary', 'constants'})
    integer(g['max_depth'], 0, 4, 'max_depth')
    for key, allowed in [('unary', UNARY), ('binary', BINARY)]:
        ops = g[key]
        if not isinstance(ops, list) or any(type(x) is not str or x not in allowed for x in ops) or len(set(ops)) != len(ops):
            raise ValueError('Invalid or repeated '+key+' operators')
    bounds(g['constants'])
    blinded = config['blinded_rhs']
    if not isinstance(blinded, list) or not blinded or any(x not in states for x in blinded) or len(set(blinded)) != len(blinded):
        raise ValueError('blinded_rhs must list distinct declared states')
    t = config['trajectories']
    fields(t, {'count', 'duration', 'samples', 'sampling', 'observed_states', 'initial_values', 'noise_stddev'})
    integer(t['count'], 3, 1000, 'trajectory count'); integer(t['samples'], 2, 10000, 'samples')
    finite(t['duration'], 'duration', True)
    if t['sampling'] not in ('uniform', 'irregular'):
        raise ValueError('sampling must be uniform or irregular')
    observed = states if t['observed_states'] == 'all' else t['observed_states']
    if not isinstance(observed, list) or not observed or any(s not in states for s in observed) or len(set(observed)) != len(observed):
        raise ValueError('observed_states must be all or a nonempty list of distinct states')
    bounds(t['initial_values'])
    if finite(t['noise_stddev'], 'noise_stddev') < 0:
        raise ValueError('noise_stddev must be nonnegative')
    a = config['generation']
    fields(a, {'max_attempts', 'max_step', 'max_abs_state', 'min_observed_span', 'convergence_atol', 'convergence_rtol'})
    integer(a['max_attempts'], 1, 100000, 'max_attempts')
    for k in ('max_step', 'max_abs_state', 'convergence_atol', 'convergence_rtol'):
        finite(a[k], k, True)
    if finite(a['min_observed_span'], 'min_observed_span') < 0:
        raise ValueError('min_observed_span must be nonnegative')
    if math.ceil(t['duration']/a['max_step']) > 1000000:
        raise ValueError('More than one million integration steps per trajectory; adjust duration/max_step')
    return states, observed


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def counts(config):
    """Counts ordered symbolic skeletons, treating each constant leaf as one choice."""
    states, _ = validate(config)
    g = config['grammar']; n = len(states)+1; result = [n]
    for _ in range(g['max_depth']):
        result.append(n+len(g['unary'])*result[-1]+len(g['binary'])*result[-1]**2)
    return result


def unrank(config, index, depth=None):
    """Uniform ranks select skeletons; a constant value is assigned separately."""
    g = config['grammar']; depth = g['max_depth'] if depth is None else depth
    sizes = counts(config); n = config['states']; leaves = n+1
    if not 0 <= index < sizes[depth]:
        raise ValueError('AST rank outside grammar')
    def take(d, k):
        if k < n: return ('state', k)
        if k == n: return ('constant', None)
        k -= leaves
        child = sizes[d-1]
        if k < len(g['unary'])*child:
            op, rank = divmod(k, child)
            return (g['unary'][op], take(d-1, rank))
        k -= len(g['unary'])*child
        op, rank = divmod(k, child*child)
        left, right = divmod(rank, child)
        return (g['binary'][op], take(d-1, left), take(d-1, right))
    return take(depth, index)


def assign_constants(tree, rng, interval):
    if tree[0] == 'constant': return ('constant', rng.uniform(*interval))
    if tree[0] == 'state': return tree
    return (tree[0], *(assign_constants(c, rng, interval) for c in tree[1:]))


def expression(tree):
    op = tree[0]
    if op == 'state': return 'x'+str(tree[1])
    if op == 'constant': return repr(tree[1])
    if op in UNARY: return op+'('+expression(tree[1])+')'
    symbol = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[op]
    return '('+expression(tree[1])+symbol+expression(tree[2])+')'


def evaluate(tree, state):
    op = tree[0]
    if op == 'state': return state[tree[1]]
    if op == 'constant': return tree[1]
    if op in UNARY: return UNARY[op](evaluate(tree[1], state))
    a, b = evaluate(tree[1], state), evaluate(tree[2], state)
    if op == 'add': return a+b
    if op == 'sub': return a-b
    if op == 'mul': return a*b
    return a/b


def integrate(trees, initial, times, max_step, max_abs):
    """Unprotected real arithmetic and fixed maximum-step RK4; reject divergence."""
    def checked(values):
        if any(not math.isfinite(x) or abs(x) > max_abs for x in values):
            raise ValueError('state_bound')
        return values
    def rhs(y): return [evaluate(t, checked(y)) for t in trees]
    def plus(y, k, h): return [a+h*b for a, b in zip(y, k)]
    y = checked(list(initial)); result = [y[:]]
    for start, end in zip(times, times[1:]):
        steps = max(1, math.ceil((end-start)/max_step)); h = (end-start)/steps
        for _ in range(steps):
            k1 = rhs(y); k2 = rhs(plus(y, k1, h/2)); k3 = rhs(plus(y, k2, h/2)); k4 = rhs(plus(y, k3, h))
            y = checked([a+h*(b+2*c+2*d+e)/6 for a, b, c, d, e in zip(y, k1, k2, k3, k4)])
        result.append(y[:])
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def generate(config, seed=None, output=None, public_format='json', save_private=False, public_basename='public'):
    states, observed = validate(config)
    output = Path('.') if output is None else Path(output)
    if public_format not in ('json', 'zip', 'directory'): raise ValueError('Invalid public format')
    if not isinstance(public_basename, str) or not public_basename or Path(public_basename).name != public_basename or public_basename in ('.', '..'):
        raise ValueError('Public basename must be a filename without directories')
    if seed is None: seed = secrets.randbits(64)
    integer(seed, 0, 2**64-1, 'seed')
    destinations = [output/'public', output/(public_basename+'.'+public_format)]
    if save_private: destinations.append(output/'private')
    if any(p.exists() for p in destinations):
        raise ValueError('Output already exists; use a new directory to preserve challenges')
    g, t, limits = config['grammar'], config['trajectories'], config['generation']
    # Separate streams keep observation sparsity/noise from consuming equation draws.
    def stream(label, attempt): return random.Random(int(digest([SCHEMA, seed, label, attempt]), 16))
    failures = []; total = counts(config)[-1]
    for attempt in range(limits['max_attempts']):
        rng = stream('system', attempt)
        ranks = [rng.randrange(total) for _ in states]
        trees = [assign_constants(unrank(config, rank), rng, bounds(g['constants'])) for rank in ranks]
        ic_rng, time_rng = stream('initial', attempt), stream('times', attempt)
        records = []; max_difference = 0.
        try:
            for i in range(t['count']):
                initial = [ic_rng.uniform(*bounds(t['initial_values'])) for _ in states]
                if t['sampling'] == 'uniform':
                    times = [t['duration']*j/(t['samples']-1) for j in range(t['samples'])]
                else:
                    times = [0.]+sorted(time_rng.uniform(0., t['duration']) for _ in range(t['samples']-2))+[t['duration']]
                if any(a >= b for a, b in zip(times, times[1:])): raise ValueError('duplicate_times')
                coarse = integrate(trees, initial, times, limits['max_step'], limits['max_abs_state'])
                fine = integrate(trees, initial, times, limits['max_step']/2, limits['max_abs_state'])
                for a, b in zip(coarse, fine):
                    for x, y in zip(a, b):
                        error = abs(x-y); max_difference = max(error, max_difference)
                        if error > limits['convergence_atol']+limits['convergence_rtol']*abs(y):
                            raise ValueError('integration_convergence')
                span = max(max(row[states.index(s)] for row in fine)-min(row[states.index(s)] for row in fine) for s in observed)
                if span < limits['min_observed_span']: raise ValueError('observed_span')
                records.append(dict(trajectory_id=str(i), times=times, initial_state=initial, states=fine))
            break
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            failures.append(dict(attempt=attempt, reason=str(error) or type(error).__name__))
    else:
        raise ValueError('No acceptable system within max_attempts; no challenge written. Rejections: '+canonical(failures[-5:]))
    public, private = output/'public', output/'private'
    public.mkdir(parents=True)
    write_json(public/'grammar.json', config)
    known = {s: expression(tree) for s, tree in zip(states, trees) if s not in config['blinded_rhs']}
    problem = dict(states=states, known_rhs=known, unknown_rhs=config['blinded_rhs'], noise='none' if t['noise_stddev'] == 0 else 'unknown')
    write_json(public/'problem.json', problem)
    validation_count = max(1, t['count']//5); train_count = t['count']-2*validation_count
    ids = [str(i) for i in range(t['count'])]
    write_json(public/'splits.json', dict(training=ids[:train_count], validation=ids[train_count:train_count+validation_count], test=ids[train_count+validation_count:]))
    (public/'knowns.txt').write_text('# State symbols: '+', '.join(states)+'\n'+''.join(f'd{s}/dt = {known.get(s, "???")}\n' for s in states)+'\n# Grammar: grammar.json; complete initial states: ics.csv\n')
    with (public/'ics.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['trajectory_id', *states])
        writer.writerows([r['trajectory_id'], *r['initial_state']] for r in records)
    noise_rng = stream('noise', attempt)
    with (public/'trajectories.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['trajectory_id', 't', *states])
        for r in records:
            for j, (time, values) in enumerate(zip(r['times'], r['states'])):
                # Initial observations remain exact and agree with separately provided ICs.
                row = [values[k]+(noise_rng.gauss(0, t['noise_stddev']) if j else 0.) if s in observed else '' for k, s in enumerate(states)]
                writer.writerow([r['trajectory_id'], time, *row])
    solution = dict(schema=SCHEMA, seed=seed, attempt=attempt,
        grammar_sha256=digest(config), ast_ranks=ranks, rhs=dict(zip(states, map(expression, trees))),
        asts=trees, rejected_attempts=failures, convergence_max_abs_difference=max_difference,
        python=sys.version, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampling='Uniform over ordered skeleton ranks, then independent uniform constants; conditioned on acceptance filters')
    if save_private:
        private.mkdir()
        write_json(private/'solution.json', solution)
        write_json(private/'clean-trajectories.json', records)
        (private/'generator.py').write_bytes(Path(__file__).read_bytes())
    write_json(public/'manifest.json', dict(schema=SCHEMA, grammar_sha256=digest(config),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public.iterdir()) if p.is_file()},
        warning='Share only the public challenge. Seed, solution and rejection history are intentionally private.'))
    if public_format == 'json':
        bundle = dict(schema=PUBLIC_SCHEMA)
        for name in ('grammar', 'problem', 'splits'):
            bundle[name] = json.loads((public/(name+'.json')).read_text())
        bundle['knowns'] = (public/'knowns.txt').read_text()
        for name in ('ics', 'trajectories'):
            with (public/(name+'.csv')).open(newline='') as f:
                reader = csv.reader(f)
                bundle[name] = dict(columns=next(reader), rows=[
                    [row[0], *(float(v) if v else None for v in row[1:])] for row in reader])
        bundle['manifest'] = dict(content_sha256=digest(bundle))
        packed = output/(public_basename+'.json')
        write_json(packed, bundle)
    elif public_format == 'zip':
        archive = output/(public_basename+'.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in sorted(PUBLIC_FILES):
                # Fixed metadata makes explicit-seed replays byte-identical.
                info = zipfile.ZipInfo('public/'+name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, (public/name).read_bytes())
        packed = archive
    if public_format != 'directory':
        for name in PUBLIC_FILES: (public/name).unlink()
        public.rmdir()
        public = packed
    return dict(public=str(public), private=str(private) if save_private else None,
                seed=seed, rhs=solution['rhs'], states=len(states), skeletons_per_rhs=total)


@contextmanager
def public_input(source):
    """Open a public JSON, directory or ZIP without reading private sibling files."""
    source = Path(source)
    if source.is_dir():
        yield source
        return
    with tempfile.TemporaryDirectory(prefix='odezza-public-') as tmp:
        public = Path(tmp)
        if source.suffix.lower() == '.json':
            bundle = json.loads(source.read_text())
            fields(bundle, {'schema', 'grammar', 'problem', 'splits', 'knowns', 'ics', 'trajectories', 'manifest'})
            if bundle['schema'] != PUBLIC_SCHEMA: raise ValueError('Unsupported public JSON schema')
            manifest = bundle.pop('manifest')
            fields(manifest, {'content_sha256'})
            if digest(bundle) != manifest['content_sha256']: raise ValueError('Public JSON content hash mismatch')
            states, _ = validate(bundle['grammar'])
            if not isinstance(bundle['knowns'], str): raise ValueError('knowns must be text')
            for name in ('grammar', 'problem', 'splits'):
                write_json(public/(name+'.json'), bundle[name])
            (public/'knowns.txt').write_text(bundle['knowns'])
            for name in ('ics', 'trajectories'):
                table = bundle[name]; fields(table, {'columns', 'rows'})
                expected = ['trajectory_id', *(['t'] if name == 'trajectories' else []), *states]
                if table['columns'] != expected or not isinstance(table['rows'], list) or not table['rows']:
                    raise ValueError('Invalid '+name+' table')
                for row in table['rows']:
                    if not isinstance(row, list) or len(row) != len(expected) or not isinstance(row[0], str):
                        raise ValueError('Invalid '+name+' row')
                    for i, value in enumerate(row[1:], 1):
                        if value is None and name == 'trajectories' and i > 1: continue
                        finite(value, name+' value')
                with (public/(name+'.csv')).open('w', newline='') as f:
                    writer = csv.writer(f); writer.writerow(expected); writer.writerows(table['rows'])
            yield public
            return
        try:
            with zipfile.ZipFile(source) as z:
                members = {}
                for info in z.infolist():
                    # Finder may add metadata when users zip the old public directory.
                    if info.filename.startswith('__MACOSX/') or info.filename == 'public/': continue
                    if info.filename not in {'public/'+name for name in PUBLIC_FILES}:
                        raise ValueError('Unexpected public archive member: '+info.filename)
                    name = info.filename[len('public/'):]
                    if name in members: raise ValueError('Duplicate public archive member: '+name)
                    members[name] = info
                if set(members) != PUBLIC_FILES: raise ValueError('Public archive is missing required files')
                if sum(info.file_size for info in members.values()) > 1024**3:
                    raise ValueError('Public archive exceeds 1 GiB uncompressed')
                for name, info in members.items():
                    (public/name).write_bytes(z.read(info))
        except zipfile.BadZipFile as error:
            raise ValueError('Invalid public ZIP archive') from error
        manifest = json.loads((public/'manifest.json').read_text())
        if set(manifest['files']) != PUBLIC_FILES-{'manifest.json'}:
            raise ValueError('Unexpected public manifest file list')
        for name, expected in manifest['files'].items():
            if hashlib.sha256((public/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Public file hash mismatch: '+name)
        if digest(json.loads((public/'grammar.json').read_text())) != manifest['grammar_sha256']:
            raise ValueError('Public grammar hash mismatch')
        yield public


def compile_search(config, banks=256, search_seed=0, quota=None):
    """Lower the same position/depth grammar directly to native C grammar search."""
    states, _ = validate(config)
    if len(config['blinded_rhs']) != 1:
        raise ValueError('Generation supports multiple blinded RHSs; current Odezza recovery supports exactly one. No equation will be silently revealed.')
    integer(banks, 1, 65536, 'banks'); integer(search_seed, 0, 2**64-1, 'search seed')
    if quota is not None: integer(quota, 1, 2**63-1, 'AST quota')
    g = config['grammar']; depth = g['max_depth']; sizes = counts(config)
    parameters = ['p'+str(i) for i in range(2**depth)]
    lines = []; next_id = [0]
    def position(d, slot):
        name = 'N'+str(next_id[0]); next_id[0] += 1
        choices = [repr(s) for s in states]+[repr(parameters[slot])]
        if d and (g['unary'] or g['binary']):
            left, _ = position(d-1, slot)
            right = position(d-1, slot+2**(d-1))[0] if g['binary'] else None
            choices += [repr(op)+' '+left for op in g['unary']]
            choices += [repr(op)+' '+left+' '+right for op in g['binary']]
        lines.append(name+' -> '+' | '.join(choices))
        return name, choices
    # Build child positions once; separate root choices give native workers independent families.
    root, choices = position(depth, 0)
    families = []
    for i, choice in enumerate(choices):
        name = 'Root'+str(i); lines.append(name+' -> '+choice)
        count = 1 if i < len(states)+1 else sizes[depth-1] if i < len(states)+1+len(g['unary']) else sizes[depth-1]**2
        target = min(count, quota) if quota else count
        if target > 2**63-1: raise ValueError('Family population exceeds supported count; set a finite --quota')
        families.append(dict(name='root-'+choice.split()[0].strip("'"), start=name,
            accepted=target, work_limit=min(2**63-1, max(1000, target*100)), maximum_nodes=2**(depth+1)-1,
            maximum_parameters=len(parameters), maximum_function_depth=depth, batch_size=128,
            tags={'game_grammar_sha256': digest(config), 'root_choice': choice.split()[0],
                  'population': str(count), 'coverage': 'exhaustive_requested' if target == count else 'enumeration_prefix'}))
    # Actual reachable counts, including unary-only grammars; never drop operators.
    rules = {line.split(' -> ')[0]: [choice.split() for choice in line.split(' -> ')[1].split(' | ')] for line in lines}
    limits = []
    for family in families:
        visited, pending = set(), [family['start']]
        while pending:
            node = pending.pop()
            if node in visited: continue
            visited.add(node)
            pending.extend(child for rule in rules[node] for child in rule[1:])
        limits.append((len(visited), sum(len(rules[n]) for n in visited)))
    max_reachable, max_productions = max(n for n, _ in limits), max(p for _, p in limits)
    if max_reachable > 64 or max_productions > 512:
        raise ValueError(f'Grammar needs up to {max_reachable} nodes/{max_productions} productions per root; current native limits 64/512. Reduce depth or state count.')
    lo, hi = bounds(g['constants'])
    grammar_search = dict(grammar='\n'.join(lines), families=families,
        rng=dict(pool=dict(size=max(4096, banks*len(parameters)), seed=search_seed, distributions=['uniform']),
                 bank_count=banks, parameters={p: dict(distribution='uniform', min=lo, max=hi) for p in parameters}))
    # The game's constant range is part of its language, not just an
    # initialization preference. Keep native refinement inside it as well.
    return dict(parameters=parameters, grammar_search=grammar_search,
                fit=dict(bounds={p:[lo,hi] for p in parameters})), dict(
        schema=SCHEMA, grammar_sha256=digest(config), skeletons_per_rhs=str(sizes[-1]),
        requested_ast_occurrences=sum(f['accepted'] for f in families),
        requested_screen_configurations=sum(f['accepted'] for f in families)*banks,
        constant_slots=len(parameters), coefficient_rows=banks,
        counting='Ordered ASTs; constant values are configurations, not ASTs; no algebraic deduplication claim',
        sampling='Native exhaustive enumeration unless quota truncates a root family; independent Philox coefficient rows',
        families=families)


def campaign(fragment, seconds=300, devices=None, target_mse=1e-10):
    finite(seconds, 'seconds', True)
    finite(target_mse, 'target_mse', True)
    if seconds < 30: raise ValueError('Allow at least 30 seconds for a campaign')
    available = seconds-10
    return dict(schema='odezza.trial.campaign.v1', request_id='game-'+digest(fragment)[:16],
        problem_id='<registered problem ID>', **fragment,
        budget=dict(wall_seconds=seconds, verification_reserve_seconds=10, devices=devices or [0, 1]),
        workflow=dict(screen=dict(wall_seconds=available*.4), refine=dict(wall_seconds=available*.35),
                      full=dict(wall_seconds=available*.2), revision_wait_seconds=available*.05),
        policy_options=dict(cycles=1, family_survivors=2, global_survivors=32, maximum_survivors=128, target_mse=target_mse))


def target_for(config, target_mse):
    if target_mse is None and config['trajectories']['noise_stddev'] > 0:
        raise ValueError('Noisy games require explicit --target-mse; the service does not infer an acceptance threshold from noise metadata')
    return 1e-10 if target_mse is None else finite(target_mse, 'target_mse', True)


def submit(public, output, banks, search_seed, quota, seconds, host, plan_only=False, target_mse=None):
    with public_input(public) as directory:
        return submit_directory(directory, output, banks, search_seed, quota, seconds,
                                host, plan_only, target_mse)


def submit_directory(public, output, banks, search_seed, quota, seconds, host, plan_only=False, target_mse=None):
    # Only explicitly public files are read; never inspect sibling private/.
    config, problem, states, rows, splits = load_challenge(public)
    fragment, population = compile_search(config, banks, search_seed, quota)
    target_mse = target_for(config, target_mse)
    if output.exists(): raise ValueError('Submission output exists; use a new directory')
    from client import Client
    output.mkdir(parents=True); write_json(output/'population.json', population)
    with Client(host=host, notify=False) as client:
        datasets = {role: client.upload(states, [dict(trajectory_id=i, **rows[i]) for i in selected]) for role, selected in splits.items()}
        problem_id = client.problem(dict(problem, data=datasets))
        write_json(output/'registration.json', dict(problem_id=problem_id, datasets=datasets))
        request = campaign(fragment, seconds, target_mse=target_mse); request['problem_id'] = problem_id
        request['request_id'] = 'game-'+digest([fragment, problem_id])[:24]
        write_json(output/'request.json', request)
        result = client.plan(request) if plan_only else client.start(request)
        write_json(output/('plan.json' if plan_only else 'job.json'), result)
        return result


def load_challenge(public):
    """Validate public input once for submit and the bounded feedback entry point."""
    config = json.loads((public/'grammar.json').read_text())
    problem = json.loads((public/'problem.json').read_text())
    states, _ = validate(config)
    if problem['states'] != states or problem['unknown_rhs'] != config['blinded_rhs']:
        raise ValueError('Problem and grammar states/blinding disagree')
    sys.path.insert(0, str(Path(__file__).resolve().parent/'scratch'/'recovery_trial'))
    import bootstrap
    from csv_data import read_trajectories
    with (public/'ics.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['trajectory_id', *states]: raise ValueError('Unexpected IC columns')
        initial = {}
        for row in reader:
            if row['trajectory_id'] in initial: raise ValueError('Duplicate IC trajectory ID')
            initial[row['trajectory_id']] = [float(row[s]) for s in states]
    rows = read_trajectories(public/'trajectories.csv', states, initial_states=initial)
    splits = json.loads((public/'splits.json').read_text()); fields(splits, {'training', 'validation', 'test'})
    ids = [i for group in splits.values() for i in group]
    if len(ids) != len(set(ids)) or set(ids) != set(rows) or any(not group for group in splits.values()):
        raise ValueError('Splits must partition all trajectory IDs into three nonempty disjoint sets')
    return config, problem, states, rows, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate'); gen.add_argument('config', type=Path); gen.add_argument('--seed', type=int, help='Optional replay seed; defaults to operating-system randomness'); gen.add_argument('--out', type=Path, help='Output directory (default: current directory)')
    gen.add_argument('--public-format', choices=['json', 'zip', 'directory'], default='json', help='Public challenge format (default: readable JSON)')
    gen.add_argument('--save-private', action='store_true', help='Also save private solution, clean trajectories and generator snapshot')
    build = sub.add_parser('compile'); build.add_argument('config', type=Path); build.add_argument('--out', type=Path, required=True)
    send = sub.add_parser('submit'); send.add_argument('public', type=Path); send.add_argument('--out', type=Path, required=True); send.add_argument('--host', default='rack1'); send.add_argument('--plan-only', action='store_true')
    for p in [build, send]:
        p.add_argument('--banks', type=int, default=256); p.add_argument('--search-seed', type=int, default=0)
        p.add_argument('--quota', type=int); p.add_argument('--seconds', type=float, default=300)
        p.add_argument('--target-mse', type=float)
    args = parser.parse_args()
    try:
        if args.command == 'generate':
            basename = args.config.stem+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            result = generate(json.loads(args.config.read_text()), args.seed, args.out, args.public_format, args.save_private, basename)
            print('Seed: '+str(result['seed']))
            for state, rhs in result['rhs'].items(): print(f'd{state}/dt = {rhs}')
            print('Wrote '+result['public'], file=sys.stderr)
            return
        elif args.command == 'compile':
            if args.out.exists(): raise ValueError('Output exists; use a new directory')
            config = json.loads(args.config.read_text())
            fragment, population = compile_search(config, args.banks, args.search_seed, args.quota)
            request = campaign(fragment, args.seconds, target_mse=target_for(config, args.target_mse))
            args.out.mkdir(parents=True); write_json(args.out/'request.json', request)
            write_json(args.out/'population.json', population); (args.out/'grammar.txt').write_text(fragment['grammar_search']['grammar']+'\n')
            result = population
        else:
            result = submit(args.public, args.out, args.banks, args.search_seed, args.quota, args.seconds, args.host, args.plan_only, args.target_mse)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, 'error: '+str(error)+'\n')


if __name__ == '__main__': main()
