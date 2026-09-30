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
"""Campaign correctness without external benchmark packages or CUDA."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from run_srbench_toggle import execute, r2, sha256, summarize, configuration, command_for
from secant_sr import write_dataset
from srbench_v2 import SRBENCH_V2_PROTOCOL

EXECUTABLE = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
RUNNER = Path(__file__).with_name("run_srbench_toggle.py")


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        x = [[(i - 16) / 16, ((i * 7) % 13 - 6) / 8] for i in range(32)]
        y = [a + b for a, b in x]
        self.data = self.root / "data.bin"
        write_dataset(self.data, x, y, (x[::-1], y[::-1]))
        self.job = dict(id="fake-s23654", problem="fake", seed=23654,
                        path="data.bin", protocol=SRBENCH_V2_PROTOCOL, target_noise=0,
                        num_inputs=2, num_train_rows=32, num_validation_rows=32,
                        train_variance=statistics.pvariance(y), test_variance=statistics.pvariance(y),
                        source_sha256="fixture", prepared_sha256=sha256(self.data))

    def test_real_cli_resume_and_identity(self):
        config = self.root / "config.json"
        config.write_text(json.dumps(dict(population=64, elites=2, banks=4, toggle_bits=2,
                                         generations=3, seconds=5, operators="add,sub,mul",
                                         refine_rounds=1, refine_budget=8, refine_scale=1)))
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(schema=1, suite="feynman", jobs=[self.job])))
        output = self.root / "campaign"
        command = [sys.executable, str(RUNNER), "run", "--manifest", str(manifest),
                   "--executable", str(EXECUTABLE), "--output", str(output),
                   "--backend", "cpu", "--config", str(config)]
        first = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        result_path = output / "trials" / self.job["id"] / "result.json"
        result = json.loads(result_path.read_text())
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("--problem", result["command"])
        self.assertEqual(result["symbolic_solution"], "")
        self.assertIn('--refine-rounds', result['command'])
        self.assertEqual(int(result['total_configurations']),
                         int(result['configurations']) + int(result['refinement_configurations']))
        self.assertEqual(result['refinement_seconds'], result['native_result']['refinement']['seconds'])
        self.assertAlmostEqual(result["validation_r2"], r2(result["validation_mse"], self.job["test_variance"]))
        original_time = result_path.stat().st_mtime_ns
        resumed = subprocess.run(command + ["--resume"], text=True, capture_output=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(result_path.stat().st_mtime_ns, original_time)
        config.write_text(json.dumps(dict(population=65)))
        rejected = subprocess.run(command + ["--resume"], text=True, capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("identical", rejected.stderr)

    def test_tampered_dataset_is_retained_as_failure(self):
        self.data.write_bytes(self.data.read_bytes() + b"tampered")
        result = execute(self.job, self.root, EXECUTABLE, {"constants": 2}, "cpu", 0, self.root / "out", 5)
        self.assertEqual(result["status"], "failed")
        self.assertIn("hash changed", result["error"])
        self.assertNotIn("accuracy_solution", result)
        summary = summarize([self.job], {self.job["id"]: result})
        self.assertEqual((summary["completed"], summary["errors"], summary["pending"]), (0, 1, 0))

    def test_hard_deadline_preserves_logs(self):
        sleeper = self.root / "sleeper"
        sleeper.write_text(f"#!{sys.executable}\nimport time\nprint('partial',flush=True)\ntime.sleep(20)\n")
        sleeper.chmod(0o700)
        result = execute(self.job, self.root, sleeper, {"constants": 2}, "cpu", 0, self.root / "out", .1)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(result["process_wall_seconds"], 6)
        self.assertEqual(result["symbolic_status"], "not_assessed")
        self.assertTrue((self.root / "out/trials" / self.job["id"] / "attempt-1.jsonl").exists())

    def test_constant_target_metric(self):
        self.assertEqual(r2(0, 0), 1)
        self.assertEqual(r2(2, 0), 0)
        self.assertIsNone(r2(None, 1))

    def test_lm_configuration_and_no_cpu_fallback(self):
        path=self.root/'lm.json'
        path.write_text(json.dumps(dict(lm_iterations=4,lm_budget=128,lm_bindings=32,lm_starts=4,lm_parameters=8,lm_interval=1,lm_threads=64,lm_scale=1)))
        config=configuration(path)
        command=command_for(EXECUTABLE,self.data,self.job,config,'cpu')
        self.assertIn('--lm-iterations',command)
        run=subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn('LM requires CUDA',run.stderr)


if __name__ == "__main__":
    if EXECUTABLE is None:
        raise SystemExit("pass the native search executable")
    unittest.main()
