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
"""Host-local execution. Blind recovery consumes public data only."""
import copy
import array
import json
import math
from pathlib import Path
import random
import sys
import time
from .common import REPO, save, digest, file_hash

TRIAL = REPO / 'scratch/fitting_batch_trial'
sys.path.insert(0, str(TRIAL))
import bootstrap
from engine import Engine, Control
from storage import Store
from controller import Campaigns
from search_controller import Searches, DONE
import game_input as game
import refinement as rf
import candidate_score as cs
from run_search import f32
from trajectory_verify import verify


def search(payload, options, method, root, engine):
    root.mkdir(parents=True)
    public = root / 'public.json'
    save(public, payload)
    with game.public_input(public) as source:
        config, problem, states, rows, splits = game.load_challenge(source)
    store = Store(root / 'store')
    campaigns = Campaigns(store, engine)
    searches = Searches(store, engine, campaigns)
    try:
        data = {scope: store.dataset(dict(states=states, trajectories=[dict(trajectory_id=i, **rows[i]) for i in ids]))['dataset_id']
                for scope, ids in splits.items()}
        pid = store.problem(dict(problem, data=data))['problem_id']
        request = dict(request_id=root.name, problem_id=pid, grammar=config['grammar'],
                       initial={'distinct_asts': options['initial']},
                       expand={'new_distinct_asts': options['offspring'], 'report_every_new_asts': options['wave']},
                       parents={'limit': options['parents']}, bank_count=options['banks'], seed=options.get('seed', 0),
                       budget={'wall_seconds': options['seconds'], 'devices': engine.devices}, baselines=True,
                       fit=dict(backend='lm_toggle' if method == 'native_lm' else 'curvature',
                                candidates=options['candidates'], starts_per_candidate=options['starts'],
                                iterations=options['iterations'], wall_seconds=options['fit_seconds'],
                                toggle_width=options['toggle_width'], initial_damping=options['initial_damping'],
                                damping_attempts=options['damping_attempts'], max_step=options['max_step']),
                       islands={'count': 1}, target_mse=1e-10,
                       leaf_toggles={'width': options.get('leaf_toggle_width',4), 'alternatives': 'state_assignments'})
        if method == 'native_lm':
            for key in ('lanes_per_fit','shape_fallback','population','submission_packs','buffer_fit_limit','sampling_seed','state_variants'):
                if key in options:request['fit'][key]=options[key]
        save(root / 'request.json', request)
        save(root / 'resolved-plan.json', searches.prepare(request)[0])
        state = searches.create(request)
        while state['status'] not in DONE:
            time.sleep(.2)
            state = searches.get(state['search_id'])
            save(root / 'progress.json', {k: state.get(k) for k in ('status', 'elapsed_seconds', 'scored_new_asts', 'coefficient_trials', 'current_wave')})
        save(root / 'result.json', state)
        result = dict(status=state['status'], verified=state['status'] == 'verified',
                      seconds=state['elapsed_seconds'], ast_occurrences=state['scored_new_asts'],
                      distinct_concrete_asts=state.get('cross_island_distinct_asts'),
                      configurations=state['coefficient_trials'], accounting_complete=state.get('accounting_complete'),
                      waves=len(state.get('waves', [])), timing=state.get('timing'), error=state.get('error'))
        if result['verified']:
            metric = state['verification']['test']['results'][-1]
            result.update(heldout_mse=metric['mse'], max_abs=metric['max_abs'])
        if state['status'] == 'error':
            text = str(result['error'])
            if any(f'C99 result {code}:' in text for code in (7, 11, 12)):
                result['status'] = 'unsupported_resource'
        from .work_report import collect
        result['work_report']=collect(root)
        return result
    finally:
        searches.close()
        campaigns.close()


def fit(payload, options, method, root, engine):
    root.mkdir(parents=True)
    preparation_started = time.monotonic()
    device = engine.devices[0]
    parameters = payload['parameters']
    rng = random.Random(options.get('seed', 0))
    candidates = payload['candidates'][:options['candidates']]
    bank = [[f32(rng.uniform(-1., 1.)) for _ in parameters] for _ in range(options['starts'])]
    context = {key: copy.deepcopy(payload[key]) for key in ('states', 'parameters', 'unknown_state', 'known_rhs', 'data')}
    context['execution'] = dict(device=device, system_capacity=128, steps_per_observation=32, timeout_seconds=options['seconds'] + 10)
    spec = dict(context, schema=rf.SCHEMA,
                candidates=[dict(c, parameter_rows=[bank[0]]) for c in candidates],
                refinement=dict(iterations=options['iterations'], max_step=options['max_step'], patience=5,
                                target_mse=1e-10, wall_seconds=options['seconds'], stop_on_target=False,
                                bounds={p: [-1.5, 1.5] for p in parameters}))
    raw_bank = array.array('f', (value for row in bank for value in row))
    if sys.byteorder != 'little':
        raw_bank.byteswap()
    with (root / 'bank.f32').open('wb') as stream:
        raw_bank.tofile(stream)
    bank_descriptor = dict(file='bank.f32', rows=len(bank), columns=len(parameters), dtype='float32-le', sha256=file_hash(root / 'bank.f32'))
    save(root / 'prepared-request.json', dict(template_request=spec, shared_parameter_bank=bank_descriptor))
    preparation_seconds = time.monotonic() - preparation_started
    before = time.monotonic()
    deadline = before + options['seconds']
    if method == 'native_lm':
        from lm_toggle.native_service import Adapter
        from .prepared_native import run as run_prepared
        if device not in engine.lm_adapters:
            engine.lm_adapters[device] = Adapter(device)
        report = run_prepared(engine.lm_adapters[device], spec, bank, options, root / 'fit', deadline)
    else:
        spec['refinement']['batching'] = {'mode': 'shared_ast_banks', 'batch_size': options['batch_size']}
        report = dict(status='complete', candidates=[], work_complete=True, counts={'coefficient_trials': 0, 'starts': 0},
                      timing={'chunks': []}, execution_path='curvature_bounded_4096_start_chunks')
        retained = {}
        chunk_rows = max(1, 4096 // len(candidates))
        for begin in range(0, len(bank), chunk_rows):
            if time.monotonic() >= deadline:
                report['work_complete'] = False
                break
            chunk = copy.deepcopy(spec)
            for candidate in chunk['candidates']:
                candidate['parameter_rows'] = bank[begin:begin + chunk_rows]
            chunk['refinement']['wall_seconds'] = max(.001, deadline - time.monotonic())
            part = rf.run(chunk, engine.backend, root / 'fit' / f'chunk-{begin:08d}', engine.workers[device])
            if part['status'] != 'complete':
                raise RuntimeError(part.get('error', 'Curvature chunk failed'))
            report['counts']['coefficient_trials'] += part['counts']['coefficient_trials']
            report['counts']['starts'] += sum(row['coefficient_trials'] > 0 for row in part['candidates'])
            report['timing']['chunks'].append(part.get('timing'))
            for row in part['candidates']:
                row['start_index'] += begin
                if row['mse'] is not None and (row['candidate_id'] not in retained or row['mse'] < retained[row['candidate_id']]['mse']):
                    retained[row['candidate_id']] = row
            if part['reason'] == 'deadline':
                report['work_complete'] = False
                break
        report['candidates'] = list(retained.values())
        save(root / 'fit' / 'report.json', report)
    elapsed = time.monotonic() - before
    if report['status'] != 'complete':
        raise RuntimeError(report.get('error', 'Incomplete fitting report'))
    best = {}
    for row in report['candidates']:
        if row['mse'] is not None and (row['candidate_id'] not in best or row['mse'] < best[row['candidate_id']]['mse']):
            best[row['candidate_id']] = row
    # Select on validation only, then consume test once for that frozen winner.
    validation = []
    replay_mismatches = []
    for row in sorted(best.values(), key=lambda r: r['mse'])[:4]:
        training = verify(context, row['program_hex'], row['parameter_values'], 128, deadline=time.monotonic()+8)
        if training['status'] != 'complete' or abs(training['mse']-row['mse']) > max(2e-8, .005*abs(training['mse'])):
            replay_mismatches.append(dict(candidate_id=row['candidate_id'], gpu_mse=row['mse'], cpu=training))
            continue
        checks = []
        for steps in (64, 128):
            checks.append(verify(dict(context, data=payload['validation']), row['program_hex'], row['parameter_values'], steps,
                                 deadline=time.monotonic() + 8))
        if all(c['status'] == 'complete' for c in checks):
            validation.append((max(c['mse'] for c in checks), row, checks))
    result = dict(status='complete', seconds=elapsed, configurations=report['counts']['coefficient_trials'],
                  candidates_requested=options['candidates'], candidates_actual=len(candidates), starts=options['starts'],
                  fitted_parameters=len(parameters), counts=report['counts'], timing=report.get('timing'),
                  lm_profile=report.get('lm_profile'), scope=payload['scope'], verified=False,
                  work_complete=report.get('work_complete'), execution_path=report.get('execution_path'),
                  bank_preparation_seconds=preparation_seconds, requested_fits=len(bank)*len(candidates),
                  replay_mismatches=replay_mismatches, verification_policy='reject_replay_inconsistent_candidates',
                  bank_sha256=bank_descriptor['sha256'],
                  candidate_output_sha256=digest(sorted(
                      (r['candidate_id'],r['program_hex'],r['parameter_values'],r['mse'],r['start_index'])
                      for r in report['candidates'])),
                  prepared_data_sha256=digest({k: spec[k] for k in ('states', 'known_rhs', 'data', 'candidates')}))
    if validation:
        _, winner, checks = min(validation, key=lambda item: item[0])
        test = verify(dict(context, data=payload['test']), winner['program_hex'], winner['parameter_values'], 128,
                      deadline=time.monotonic() + 10)
        result.update(validation=checks, test=test, winner=winner, heldout_mse=test.get('mse'),
                      verified=test['status'] == 'complete' and test['mse'] <= 1e-10 and checks[-1]['mse'] <= 1e-10)
    save(root / 'result.json', result)
    return result
