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
import array
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import campaign
from replay_pair import compare


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.binary = self.root / 'search'
        self.binary.write_bytes(b'binary')
        self.plugin = self.root / 'legacy.so'
        self.plugin.write_bytes(b'plugin')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(schema=1, suite='feynman',
            jobs=[dict(id='case', path='data.bin')])) )
        self.output = self.root / 'output'
        self.calls = []

    def execute(self, job, data, binary, config, backend, gpu, output, timeout):
        self.calls.append((str(output), config.copy(), os.environ['SECANT_BENCH_MODE']))
        r = dict(id=job['id'], status='completed', accuracy_solution=1,
                 total_configurations='100', process_wall_seconds=2)
        campaign.atomic_json(output / 'trials' / job['id'] / 'result.json', r)
        return r

    def run_campaign(self, resume=False, execute=None):
        args = ['test', '--manifest', str(self.manifest), '--executable', str(self.binary),
                '--legacy', str(self.plugin), '--output', str(self.output),
                '--shard', '0', '--shards', '1']
        if resume:
            args.append('--resume')
        with patch('sys.argv', args), patch.dict(os.environ), \
             patch.object(campaign.subprocess, 'check_output', return_value='GPU'), \
             patch.object(campaign, 'execute', side_effect=execute or self.execute), \
             contextlib.redirect_stdout(io.StringIO()):
            campaign.main()

    def test_matrix_resume_and_identity(self):
        self.run_campaign()
        self.assertEqual(len(self.calls), 15)  # 3 warmups + 2 repeats x 6 arms.
        measured = [x for x in self.calls if 'warmups' not in x[0]]
        for _, c, engine in measured:
            self.assertEqual((c['population'], c['banks'], c['toggle_bits']), (8192, 64, 6))
            self.assertEqual(c['pack'], 8 if engine == 'native' else 2)
        p = json.loads((self.output / 'progress.json').read_text())
        self.assertEqual((p['status'], p['finished'], p['planned']), ('complete', 12, 12))
        self.run_campaign(resume=True)
        self.assertEqual(len(self.calls), 15)
        self.plugin.write_bytes(b'changed plugin')
        with self.assertRaisesRegex(ValueError, 'identical'):
            self.run_campaign(resume=True)

    def test_failure_stops_and_cannot_resume_past_failure(self):
        def fail(*args):
            r = self.execute(*args)
            if 'warmups' not in str(args[6]):
                r['status'] = 'failed'
            return r
        with self.assertRaisesRegex(RuntimeError, 'fit failed'):
            self.run_campaign(execute=fail)
        self.assertEqual(len(self.calls), 4)
        p = json.loads((self.output / 'progress.json').read_text())
        self.assertEqual((p['status'], p['finished']), ('failed', 1))
        with self.assertRaisesRegex(RuntimeError, 'retained failure'):
            self.run_campaign(resume=True)
        self.assertEqual(len(self.calls), 4)


class GridTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.request = self.root / 'request.bin'
        # One AST, four configurations, two rows, one input and one coefficient.
        self.request.write_bytes(struct.pack('<10Q', 0x3159414c50455253, 1, 1, 1, 2, 1, 2, 1, 0, 0)
                                 + struct.pack('<4f', 0, 1, 1, 2))

    def compare(self, left, right):
        paths = [self.root / 'a', self.root / 'b']
        for path, values in zip(paths, [left, right]):
            path.write_bytes(array.array('f', values).tobytes())
        return compare(self.request, paths)

    def test_exact_and_numerical_agreement(self):
        r = self.compare([0, 1, 2, float('inf')], [0, 1.000001, 2, float('nan')])
        self.assertTrue(r['accepted'])
        self.assertEqual((r['checked'], r['finite'], r['exactly_equal_finite']), (4, 3, 2))

    def test_finite_mismatch_and_score_mismatch_reject(self):
        self.assertFalse(self.compare([0, 1, 2, 3], [0, 1, 2, float('inf')])['accepted'])
        self.assertFalse(self.compare([0, 1, 2, 3], [0, 1, 2, 4])['accepted'])

    def test_truncation_rejects(self):
        self.assertFalse(self.compare([0, 1], [0, 1])['accepted'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.compare([0, 1], [0])


if __name__ == '__main__':
    unittest.main()
