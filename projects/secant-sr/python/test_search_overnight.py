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
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import search_overnight as campaign


class OvernightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.executable = self.root / 'search'
        self.executable.write_bytes(b'frozen executable')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(schema=1, suite='feynman', jobs=[
            dict(id='case', path='data.bin', phase='main')])) )
        self.output = self.root / 'output'
        self.calls = []

    def execute(self, job, data, binary, config, backend, gpu, output, timeout):
        self.calls.append((str(output), config))
        result = dict(status='completed', id=job['id'], accuracy_solution=1,
                      configurations='123', process_wall_seconds=2)
        folder = output / 'trials' / job['id']
        folder.mkdir(parents=True)
        campaign.atomic_json(folder / 'result.json', result)
        return result

    def run_campaign(self, resume=False, hours=8, execute=None):
        args = ['test', '--manifest', str(self.manifest), '--executable', str(self.executable),
                '--output', str(self.output), '--shard', '0', '--shards', '1', '--hours', str(hours)]
        if resume:
            args.append('--resume')
        with patch('sys.argv', args), patch.object(campaign.subprocess, 'check_output', return_value='GPU'), \
             patch.object(campaign, 'execute', side_effect=execute or self.execute), \
             contextlib.redirect_stdout(io.StringIO()):
            campaign.main()

    def test_matrix_and_checkpoint_resume(self):
        self.assertEqual(len(campaign.arms_for(True, True)), 13)
        self.assertEqual(len(campaign.arms_for(True, False)), 11)
        self.run_campaign()
        self.assertEqual(len(self.calls), 6)  # Five measured arms plus one retained warmup.
        progress = json.loads((self.output / 'progress.json').read_text())
        self.assertEqual((progress['status'], progress['finished']), ('complete', 5))
        self.run_campaign(resume=True)
        self.assertEqual(len(self.calls), 6)
        self.executable.write_bytes(b'different executable')
        with self.assertRaisesRegex(ValueError, 'identical'):
            self.run_campaign(resume=True)

    def test_deadline_records_unstarted_work(self):
        self.run_campaign(hours=.00001)
        progress = json.loads((self.output / 'progress.json').read_text())
        self.assertEqual((progress['status'], progress['finished'], progress['planned']), ('time_budget', 0, 5))
        self.assertEqual(self.calls, [])

    def test_failure_stops_after_evidence_is_retained(self):
        def fail(*args):
            result = self.execute(*args)
            if 'warmups' not in str(args[6]):
                result['status'] = 'failed'
                result['error'] = 'intentional failure'
                campaign.atomic_json(args[6] / 'trials' / 'case' / 'result.json', result)
            return result
        with self.assertRaisesRegex(RuntimeError, 'fit failed'):
            self.run_campaign(execute=fail)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue((self.output / 'failure.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'failed trial'):
            self.run_campaign(resume=True)
        self.assertEqual(len(self.calls), 2)


if __name__ == '__main__':
    unittest.main()
