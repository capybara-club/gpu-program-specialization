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
"""Admission, persistence and address semantics without requiring CUDA."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from odegrammar.compiler import content_id
from odezza.grammar.lowering import lower_system, identities
from odezza.grammar.mcp import Transport
from odezza.grammar.problem import prepare_problem
from odezza.grammar.results import compact_report, decimal_index, origin, reconstruct
from odezza.grammar.service import Plan, Service
from odezza.grammar.__main__ import main as cli
from test_integration import grammar, problem


class SubmissionTests(unittest.TestCase):
    def test_combined_file_cli_keeps_one_job_id(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory)/'request.json'
            request.write_text(json.dumps(dict(problem=problem(), grammar=grammar())))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli(['--backend', 'python', '--root', str(Path(directory)/'jobs'), 'plan', str(request)]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report['status'], 'planned')
            self.assertEqual(len(report['job_id']), 64)

    def test_restart_marks_preparation_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'jobs'/('a'*64); folder.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps(dict(status='preparing', completed_configurations=0)))
            engine = Service(directory)
            try:
                self.assertEqual(engine.status('a'*64)['status'], 'interrupted')
            finally: engine.close()

    def test_inline_admission_does_not_wait_for_worker_and_snapshots_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Service(directory); release = threading.Event()
            blocker = engine.worker.submit(release.wait)
            request, observations = grammar(), problem()
            observations['known_rhs'] = {'x1': '0'}
            del request['rhs']['x1']
            try:
                with patch('odezza.grammar.service.prepare_problem', wraps=prepare_problem) as prepare:
                    job = engine.submit(problem=observations, grammar=request, plan_only=True, idempotency_key='retry')
                    self.assertEqual(job['status'], 'queued')
                    self.assertFalse(prepare.called)
                    self.assertEqual(engine.status(job['job_id'])['status'], 'queued')
                    self.assertEqual(engine.results(job['job_id'])['status'], 'queued')
                    self.assertEqual(engine.submit(problem=observations, grammar=request, plan_only=True,
                                                   idempotency_key='retry')['job_id'], job['job_id'])
                    observations['trajectories'][0]['initial'].clear()
                    request['rhs']['x0'] = 't'
                    release.set(); blocker.result(timeout=5)
                    report = engine.jobs[job['job_id']].result(timeout=5)
                    self.assertEqual(report['status'], 'planned')
                    self.assertEqual(report['generation']['configuration_visits'], 6)
                    self.assertEqual(report['problem_id'], engine.status(job['job_id'])['problem_id'])
                    self.assertEqual(prepare.call_count, 1)
                    self.assertIsNone(engine.executor)
            finally:
                release.set(); engine.close()

    def test_preparing_status_is_pollable_and_bad_observations_fail_as_job(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Service(directory); entered, release = threading.Event(), threading.Event()
            def blocked_prepare(spec):
                entered.set()
                if not release.wait(5): raise TimeoutError('test worker release')
                raise ValueError('invalid initial values')
            try:
                with patch('odezza.grammar.service.prepare_problem', side_effect=blocked_prepare):
                    job = engine.submit(problem=problem(), grammar=grammar())
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(engine.results(job['job_id'])['status'], 'preparing')
                    release.set()
                    report = engine.jobs[job['job_id']].result(timeout=5)
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(report['counts']['completed_configurations'], 0)
                    self.assertFalse(report['gpu_executed'])
                    self.assertIn('invalid initial', report['error'])
                    self.assertIsNone(engine.executor)
            finally:
                release.set(); engine.close()

    def test_queued_cancel_skips_problem_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Service(directory); release = threading.Event()
            blocker = engine.worker.submit(release.wait)
            try:
                with patch('odezza.grammar.service.prepare_problem', side_effect=AssertionError('must not prepare')):
                    job = engine.submit(problem=problem(), grammar=grammar())
                    engine.cancel(job['job_id']); release.set(); blocker.result(timeout=5)
                    report = engine.jobs[job['job_id']].result(timeout=5)
                    self.assertEqual(report['status'], 'cancelled')
                    self.assertFalse(report['gpu_executed'])
            finally:
                release.set(); engine.close()

    def test_gpu_setup_is_reported_after_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Service(directory); entered, release = threading.Event(), threading.Event()
            def blocked_setup():
                entered.set()
                if not release.wait(5): raise TimeoutError('test worker release')
                raise RuntimeError('simulated setup failure')
            try:
                with patch.object(engine, 'gpu_executor', side_effect=blocked_setup):
                    job = engine.submit(problem=problem(), grammar=grammar())
                    self.assertTrue(entered.wait(5))
                    status = engine.status(job['job_id'])
                    self.assertEqual((status['status'], status['phase']), ('running', 'gpu_setup'))
                    release.set()
                    report = engine.jobs[job['job_id']].result(timeout=5)
                    self.assertEqual(report['status'], 'failed')
                    self.assertFalse(report['gpu_executed'])
            finally:
                release.set(); engine.close()

    def test_exclusive_inputs_and_mcp_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Service(directory)
            try:
                with self.assertRaises(ValueError): engine.submit(grammar=grammar())
                with self.assertRaises(ValueError): engine.submit('0'*64, grammar(), problem=problem())
                t = Transport(engine)
                t.dispatch(dict(jsonrpc='2.0', id=1, method='initialize'))
                t.dispatch(dict(jsonrpc='2.0', method='notifications/initialized'))
                def submit(args):
                    return t.dispatch(dict(jsonrpc='2.0', id=2, method='tools/call', params=dict(name='odezza_submit', arguments=args)))
                self.assertIn('error', submit(dict(problem=problem(), problem_id='0'*64, grammar=grammar())))
                response = submit(dict(problem=problem(), grammar=grammar(), plan_only=True))
                self.assertFalse(response['result']['isError'])
                job = response['result']['structuredContent']
                self.assertEqual(engine.jobs[job['job_id']].result(timeout=5)['status'], 'planned')
            finally: engine.close()

    def test_compact_address_reconstruction_survives_restart_without_enumeration(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Service(directory)
            try:
                job = engine.submit(problem=problem(), grammar=grammar(), plan_only=True)
                report = engine.jobs[job['job_id']].result(timeout=5)
                folder = engine.job_path(job['job_id'])
                manifest = json.loads((folder/'manifest.json').read_text())
                plan = Plan(folder/'plan.sqlite')
                try:
                    variant = plan.variant_at(0); skeleton = plan.skeleton(variant['skeleton_id'])
                    layout, _ = lower_system(skeleton, variant, manifest['states'])
                    values, bank, permutation = [1.], 1, 1
                    numeric_id, structure_id, programs = identities(skeleton, variant, manifest['states'], layout, values, permutation)
                    candidate = dict(id=content_id('unit-snapshot'), variant_id=variant['id'], skeleton_id=skeleton['id'],
                        problem_id=report['problem_id'], numeric_id=numeric_id, structure_id=structure_id,
                        configuration_index=layout.grammar_index(bank, permutation), bank_index=bank, permutation=permutation,
                        slots=[list(pair) for pair in layout.slots], values=values, resolved_programs=programs, mse=1.,
                        integration=report['integration'])
                    # An explicit unit fixture, not a claim of GPU scoring.
                    plan.retain([candidate], 1)
                    address = origin(plan, manifest, candidate)
                    compact = compact_report(dict(status='complete', winners={'global': [candidate], 'families': {'f': [candidate]}, 'tags': {'t': [candidate]}}), manifest, job['job_id'], plan)
                    self.assertEqual(list(compact['candidates']), [candidate['id']])
                    self.assertEqual(compact['leaderboards']['tags']['t'], [candidate['id']])
                    self.assertIsInstance(address['configuration_index'], str)
                    bad = dict(address, manifest_id='0'*64)
                    with self.assertRaises(ValueError): reconstruct(plan, manifest, bad)
                    with self.assertRaises(KeyError): reconstruct(plan, manifest, dict(address, configuration_index='0'))
                    with self.assertRaises(KeyError): reconstruct(plan, manifest, dict(address, variant_index='999'))
                finally: plan.close()
            finally: engine.close()
            engine = Service(directory)
            try:
                with patch('odegrammar.compiler.Compiler.records', side_effect=AssertionError('must not enumerate')):
                    replay = engine.replay(job['job_id'], address=address)
                self.assertEqual(replay['values'], values)
                self.assertEqual(replay['resolved_programs'], programs)
                self.assertEqual(replay['equations']['x0'], '-1 * x1')
                with self.assertRaises(ValueError): engine.replay(job['job_id'], candidate['id'], address=address)
                self.assertEqual(engine.replay(job['job_id'], candidate['id'])['values'], values)
            finally: engine.close()

    def test_decimal_addresses_preserve_full_uint64_and_reject_ambiguous_numbers(self):
        for number in [0, 2**53+1, 2**64-1]:
            self.assertEqual(decimal_index(json.loads(json.dumps(str(number))), 64), number)
        for bad in [True, 1., 1, '-1', '01', '1e3', str(2**64), '9'*100]:
            with self.assertRaises(ValueError): decimal_index(bad, 64)


if __name__ == '__main__': unittest.main()
