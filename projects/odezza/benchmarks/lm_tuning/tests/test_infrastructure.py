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
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.lm_tuning.common import can_admit, digest
from benchmarks.lm_tuning.generate import generate, sample_tree, validate, leaves
from benchmarks.lm_tuning.protocol import validate as validate_protocol, settings, sample_case, pair
from benchmarks.lm_tuning.completion import message

ROOT = Path(__file__).resolve().parents[1]


class InfrastructureTests(unittest.TestCase):
    def test_early_failure_does_not_claim_deadline(self):
        states = {'gpu0': {'status': 'failed', 'finished_at': 50}}
        early = message(states, 3, now=60, deadline=100)
        self.assertIn('stopped early', early)
        self.assertNotIn('deadline reached', early)
        self.assertIn('deadline reached', message(states, 3, now=101, deadline=100))

    def test_impossible_leaf_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            validate(dict(states=6, depth=2, dependencies=3, parameters=3))

    def test_controlled_generation_keeps_truth_private(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'case'
            result = generate(dict(states=6, depth=2, dependencies=2, parameters=1,
                                   background='linear_control', duration=.1, seconds=5), root, seed=37)
            self.assertIsNotNone(result)
            public = json.loads((root / 'public.json').read_text())
            private = json.loads((root / 'private/solution.json').read_text())
            self.assertEqual(len(private['dependency_matrix'][-1]), 2)
            self.assertEqual(result['parameters'], 1)
            self.assertNotIn('dependencies', public['grammar']['generation'])
            self.assertNotIn('seed', public)
            self.assertNotIn('x5', public['problem']['known_rhs'])
            calibration = json.loads((root / 'calibration.json').read_text())
            self.assertEqual(set(calibration['data']['trajectories'][0]), {'times', 'states', 'initial_state'})
            self.assertEqual(public['manifest']['content_sha256'], digest({k: v for k, v in public.items() if k != 'manifest'}))

    def test_pair_uses_same_inputs_and_rotates_order(self):
        a = pair('a', {'observations': [1, 2]}, 'search', {'seconds': 20}, 0)
        b = pair('a', {'observations': [1, 2]}, 'search', {'seconds': 20}, 1)
        self.assertEqual(a['trials'][0]['settings'], a['trials'][1]['settings'])
        self.assertNotEqual(a['trials'][0]['method'], b['trials'][0]['method'])
        self.assertFalse(can_admit(a, 159, now=0))
        self.assertTrue(can_admit(a, 161, now=0))

    def test_sampling_reproducible_and_bank_limits_explicit(self):
        protocol = validate_protocol(json.loads((ROOT / 'examples/overnight.json').read_text()))
        for i in range(100):
            chosen = settings(protocol, 'fit', i)
            self.assertLessEqual(chosen['candidates'] * chosen['starts'], 4194304)
            self.assertEqual(chosen, settings(protocol, 'fit', i))
            case = sample_case(protocol, i)
            self.assertLessEqual(case['dependencies'] + case['parameters'], 2 ** case['depth'])


if __name__ == '__main__':
    unittest.main()
