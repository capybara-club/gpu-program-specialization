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
"""Audit rescued fixed ASTs using native CPU materialization and CUDA scoring."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['manifest', 'results', 'data', 'probe', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    jobs = {j['job']['id']: j['job'] for j in json.loads(a.manifest.read_text())['jobs']}
    records = []
    with tempfile.TemporaryDirectory(prefix='polish-replay-') as directory:
        for case in json.loads(a.results.read_text())['cases']:
            result = case['modes']['tied']
            if not result['recovered']:
                continue
            job = jobs[case['id']]
            source = a.data/job['path']
            raw = source.read_bytes()
            magic, version, inputs, train, test = struct.unpack('<8sIIQQ', raw[:32])
            split = 32+(inputs+1)*train*4
            validation = Path(directory)/(job['id']+'.secsr')
            validation.write_bytes(struct.pack('<8sIIQQ', magic, version, inputs, test, train)+raw[split:]+raw[32:split])
            for name, dataset, rows, variance in [('train', source, train, job['train_variance']),
                                                  ('validation', validation, test, job['test_variance'])]:
                run = subprocess.run([str(a.probe.resolve()), str(dataset), result['resolved_ast_hex']],
                                     capture_output=True, text=True, check=True, timeout=60)
                r = json.loads(run.stdout.strip().splitlines()[-1])
                r.update(id=job['id'], split=name)
                expected = result[name+'_mse']
                cpu = r['cpu_predictions_sse_f64_sum']/rows
                r['cpu_r2'] = 1-cpu/variance
                r['gpu_r2'] = 1-r['gpu_scoring_sse']/rows/variance
                r['relative_rmse_gap'] = abs(math.sqrt(cpu)-math.sqrt(expected))/math.sqrt(variance)
                r['accepted'] = (r['cpu_r2'] > .999 and r['gpu_r2'] > .999 and
                                 r['relative_rmse_gap'] <= 2e-5 and r['relative_prediction_rmse'] <= 2e-5)
                records.append(r)
                a.output.write_text(json.dumps(records, indent=2)+'\n')
                if not r['accepted']:
                    raise RuntimeError('native replay rejected; inspect retained evidence')
    print(json.dumps(dict(checked=len(records), accepted=all(r['accepted'] for r in records))))


if __name__ == '__main__':
    main()
