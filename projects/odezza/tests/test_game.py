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
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ode_game', ROOT/'ode_game.py')
game = importlib.util.module_from_spec(spec); spec.loader.exec_module(game)


class GameTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'examples/grammar_game/depth2.json').read_text())

    def test_sampler_and_compiler_have_identical_language(self):
        c = self.config(); c['states'] = 1; c['blinded_rhs'] = ['x0']
        c['grammar']['max_depth'] = 2
        c['grammar']['unary'] = ['sin']; c['grammar']['binary'] = ['add', 'mul']
        self.check_language(c)

    def test_depth3_language_and_full_preset(self):
        # Exhaust a smaller depth-3 grammar to check arbitrary nesting in both
        # the sampler and compiled search, without materializing billions of ASTs.
        c = self.config(); c['states'] = 1; c['blinded_rhs'] = ['x0']
        c['grammar'].update(max_depth=3, unary=['sin'], binary=['add'])
        self.assertEqual(game.counts(c), [2, 8, 74, 5552])
        self.check_language(c)
        c = json.loads((ROOT/'examples/grammar_game/depth3.json').read_text())
        fragment, report = game.compile_search(c)
        self.assertEqual(game.counts(c), [4, 76, 23260, 2164156924])
        self.assertEqual(report['requested_screen_configurations'], 554024172544)
        self.assertEqual(fragment['parameters'], ['p'+str(i) for i in range(8)])
        self.assertTrue(all(f['maximum_nodes'] == 15 and f['maximum_function_depth'] == 3
                            for f in fragment['grammar_search']['families']))
        # Exercise complete challenge generation and public JSON ingestion too.
        c['trajectories'].update(count=3, samples=5, duration=.2)
        c['generation']['min_observed_span'] = 0
        with tempfile.TemporaryDirectory() as tmp:
            result = game.generate(c, 991, Path(tmp)/'challenge')
            with game.public_input(result['public']) as public:
                emitted = json.loads((public/'grammar.json').read_text())
                self.assertEqual(emitted, c)
                self.assertEqual(game.compile_search(emitted)[1]['requested_ast_occurrences'], 2164156924)

    def check_language(self, c):
        fragment, report = game.compile_search(c, banks=3)
        rules = {}
        for line in fragment['grammar_search']['grammar'].splitlines():
            name, rhs = line.split(' -> ')
            rules[name] = [shlex.split(p) for p in rhs.split(' | ')]
        def expand(name):
            result = set()
            for rule in rules[name]:
                if len(rule) == 1:
                    token = rule[0]
                    result.add(('constant', None) if token.startswith('p') else ('state', int(token[1:])))
                else:
                    result.update((rule[0], *children) for children in itertools.product(*(expand(child) for child in rule[1:])))
            return result
        searched = set()
        for f in fragment['grammar_search']['families']:
            language = expand(f['start'])
            self.assertEqual(f['accepted'], len(language))
            self.assertFalse(searched & language)
            searched.update(language)
        generated = {game.unrank(c, i) for i in range(game.counts(c)[-1])}
        self.assertEqual(searched, generated)
        self.assertEqual(report['requested_screen_configurations'], 3*len(generated))

    def test_default_population_and_independent_slots(self):
        fragment, report = game.compile_search(self.config())
        self.assertEqual(game.counts(self.config()), [4, 76, 23260])
        self.assertEqual(report['requested_screen_configurations'], 5954560)
        self.assertEqual(fragment['parameters'], ['p0', 'p1', 'p2', 'p3'])
        # Four simultaneous leaf positions use four independently sampled slots.
        text = fragment['grammar_search']['grammar']
        for name in fragment['parameters']: self.assertIn(repr(name), text)

    def test_constant_range_is_also_a_refinement_bound(self):
        c=self.config();c['grammar']['constants']={'min':-.75,'max':.25}
        fragment,_=game.compile_search(c)
        expected={p:[-.75,.25] for p in fragment['parameters']}
        self.assertEqual(fragment['fit']['bounds'],expected)
        self.assertEqual(game.campaign(fragment)['fit']['bounds'],expected)

    def test_real_division_and_analytic_integration(self):
        tree = ('mul', ('constant', -.7), ('state', 0))
        times = [0., .25, .5, 1.]
        rows = game.integrate([tree], [1.2], times, .005, 10)
        for t, row in zip(times, rows): self.assertAlmostEqual(row[0], 1.2*game.math.exp(-.7*t), places=11)
        with self.assertRaises(ZeroDivisionError): game.evaluate(('div', ('state', 0), ('constant', 0)), [1])

    def test_public_private_reproducibility_and_hidden_observations(self):
        c = self.config(); c['grammar']['max_depth'] = 1; c['grammar']['binary'] = ['add']
        c['trajectories'].update(count=3, samples=5, duration=.2, sampling='irregular', observed_states=['x0', 'x2'])
        c['generation']['min_observed_span'] = 0
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            game.generate(c, 991, a, public_format='directory', save_private=True); game.generate(c, 991, b, public_format='directory', save_private=True)
            for p in (a/'public').iterdir(): self.assertEqual(p.read_bytes(), (b/'public'/p.name).read_bytes())
            self.assertEqual((a/'private/solution.json').read_bytes(), (b/'private/solution.json').read_bytes())
            public = '\n'.join(p.read_text() for p in (a/'public').iterdir())
            self.assertNotIn('"seed"', public); self.assertNotIn('ast_ranks', public)
            problem = json.loads((a/'public/problem.json').read_text())
            self.assertNotIn('x2', problem['known_rhs'])
            with (a/'public/trajectories.csv').open() as f:
                for row in game.csv.DictReader(f): self.assertEqual(row['x1'], '')
            with self.assertRaises(ValueError): game.generate(c, 991, a)

    def test_automatic_seed_json_replay_and_submission_input(self):
        c = self.config(); c['grammar']['max_depth'] = 1; c['grammar']['binary'] = ['add']
        c['trajectories'].update(count=3, samples=5, duration=.2, observed_states=['x0', 'x2'])
        c['generation']['min_observed_span'] = 0
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            with mock.patch.object(game.secrets, 'randbits', return_value=991) as entropy:
                result = game.generate(c, output=a)
            entropy.assert_called_once_with(64)
            self.assertEqual(result['public'], str(a/'public.json'))
            self.assertFalse((a/'public').exists())
            self.assertEqual({p.name for p in a.iterdir()}, {'public.json'})
            self.assertEqual(result['seed'], 991)
            game.generate(c, result['seed'], b)
            self.assertEqual((a/'public.json').read_bytes(), (b/'public.json').read_bytes())
            bundle = json.loads((a/'public.json').read_text())
            self.assertEqual(bundle['schema'], game.PUBLIC_SCHEMA)
            self.assertEqual(bundle['ics']['columns'], ['trajectory_id', 'x0', 'x1', 'x2'])
            self.assertTrue(all(row[3] is None for row in bundle['trajectories']['rows']))
            text = (a/'public.json').read_text()
            self.assertNotIn('"seed"', text); self.assertNotIn('ast_ranks', text)
            def consume(public, *args):
                self.assertNotIn('x2', json.loads((public/'problem.json').read_text())['known_rhs'])
                with (public/'trajectories.csv').open() as f:
                    self.assertTrue(all(row['x1'] == '' for row in game.csv.DictReader(f)))
                return 'accepted'
            with mock.patch.object(game, 'submit_directory', side_effect=consume):
                self.assertEqual(game.submit(a/'public.json', Path(tmp)/'run', 256, 0, None, 60, 'rack1'), 'accepted')
            # Exercise the actual CLI defaults without supplying --seed or format.
            config = Path(tmp)/'config.json'; config.write_text(json.dumps(c))
            output = Path(tmp)/'cli'; output.mkdir()
            command = [sys.executable, '-B', str(ROOT/'ode_game.py'), 'generate', str(config)]
            cli = subprocess.run(command, cwd=output, capture_output=True, text=True, check=True)
            written = list(output.iterdir())
            self.assertEqual(len(written), 1)
            public_file = written[0]
            self.assertRegex(public_file.name, r'^config-\d{8}T\d{6}\.\d{6}Z\.json$')
            lines = cli.stdout.splitlines()
            seed = int(lines[0].removeprefix('Seed: '))
            replay = game.generate(c, seed, Path(tmp)/'cli-replay')
            self.assertEqual(lines[1:], [f'd{state}/dt = {rhs}' for state, rhs in replay['rhs'].items()])
            self.assertEqual(public_file.read_bytes(), (Path(tmp)/'cli-replay/public.json').read_bytes())
            original = public_file.read_bytes()
            second = subprocess.run(command, cwd=output, capture_output=True, text=True, check=True)
            self.assertEqual(public_file.read_bytes(), original)
            self.assertEqual(len(list(output.iterdir())), 2)
            self.assertTrue(all(p.is_file() and p.suffix == '.json' for p in output.iterdir()))
            # Old optional formats still carry exactly the same public observations.
            for fmt in ('zip', 'directory'):
                legacy = game.generate(c, 991, Path(tmp)/fmt, public_format=fmt)
                with game.public_input(a/'public.json') as new, game.public_input(legacy['public']) as old:
                    for name in ('ics.csv', 'trajectories.csv', 'problem.json', 'splits.json', 'grammar.json'):
                        self.assertEqual((new/name).read_bytes(), (old/name).read_bytes())
            bundle['problem']['known_rhs']['x0'] = '0'
            bad = Path(tmp)/'bad.json'; game.write_json(bad, bundle)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                with game.public_input(bad): pass
            # A valid hash does not excuse an invalid/missing initial value.
            bundle['ics']['rows'][0][1] = None
            bundle.pop('manifest')
            bundle['manifest'] = {'content_sha256': game.digest(bundle)}
            game.write_json(bad, bundle)
            with self.assertRaisesRegex(ValueError, 'finite'):
                with game.public_input(bad): pass

    def test_archive_rejects_tampering_and_unexpected_members(self):
        # A minimal synthetic manifest is sufficient to test archive integrity;
        # no dependence on a prior user's challenge or a private solution.
        files = {name: b'{}' for name in game.PUBLIC_FILES}
        files['manifest.json'] = json.dumps({'files': {
            name: game.hashlib.sha256(data).hexdigest() for name, data in files.items()
            if name != 'manifest.json'}, 'grammar_sha256': game.digest({})}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            for label, extra, corrupt in [('traversal', 'public/../../escape', False),
                                           ('private', 'private/solution.json', False),
                                           ('duplicate', 'public/ics.csv', False),
                                           ('missing', None, False), ('hash', None, True)]:
                archive = Path(tmp)/(label+'.zip')
                with zipfile.ZipFile(archive, 'w') as z:
                    for name, data in files.items():
                        if label == 'missing' and name == 'ics.csv': continue
                        z.writestr('public/'+name, b'changed' if corrupt and name == 'ics.csv' else data)
                    if extra:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', UserWarning)
                            z.writestr(extra, b'bad')
                with self.subTest(label=label), self.assertRaises(ValueError):
                    with game.public_input(archive): pass

    def test_limits_and_explicit_partial_coverage(self):
        c = self.config(); c['blinded_rhs'] = ['x1', 'x2']
        game.validate(c)
        with self.assertRaisesRegex(ValueError, 'exactly one'): game.compile_search(c)
        c = self.config(); c['states'] = 32; c['grammar']['max_depth'] = 4
        with self.assertRaises(ValueError): game.compile_search(c, quota=10)
        c = self.config(); _, r = game.compile_search(c, quota=5)
        self.assertLess(r['requested_ast_occurrences'], int(r['skeletons_per_rhs']))
        self.assertTrue(any(f['tags']['coverage'] == 'enumeration_prefix' for f in r['families']))
        c['grammar']['mystery'] = True
        with self.assertRaises(ValueError): game.validate(c)

    def test_noise_threshold_and_unary_only_backend_counts(self):
        c = self.config(); c['trajectories']['noise_stddev'] = .01
        with self.assertRaises(ValueError): game.target_for(c, None)
        self.assertEqual(game.target_for(c, .0002), .0002)
        c = self.config(); c['states'] = 32; c['grammar']['max_depth'] = 4
        c['grammar']['binary'] = []
        fragment, _ = game.compile_search(c, quota=5)
        self.assertTrue(fragment['grammar_search']['families'])


if __name__ == '__main__': unittest.main()
