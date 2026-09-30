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
"""Replay captured populations; stop on any disagreement and retain diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_pair import digest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ['captures', 'replay', 'legacy', 'output']:
        p.add_argument('--' + field, type=Path, required=True)
    p.add_argument('--gpu', default='0')
    a = p.parse_args()
    requests = sorted(a.captures.glob('*/generation-*.bin'))
    if not requests:
        p.error('no population captures')
    a.output.mkdir(parents=True, exist_ok=True)
    identity = dict(binary=digest(a.replay), legacy=digest(a.legacy), gpu=a.gpu,
                    requests={str(r): digest(r) for r in requests},
                    runner=digest(Path(__file__)), pair=digest(Path(__file__).with_name('replay_pair.py')))
    path = a.output/'identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('refusing to resume changed replay sweep')
    else:
        path.write_text(json.dumps(identity, indent=2)+'\n')
    for i, request in enumerate(requests):
        dest = a.output/(request.parent.name+'-'+request.stem)
        result = dest/'result.json'
        if result.exists():
            if json.loads(result.read_text())['status'] != 'complete':
                raise RuntimeError('retained failure requires investigation')
        else:
            print('start', request, flush=True)
            command = [sys.executable, str(Path(__file__).with_name('replay_pair.py')),
                '--request', str(request), '--replay', str(a.replay), '--legacy', str(a.legacy),
                '--output', str(dest), '--gpu', a.gpu, '--samples', '3']
            if i % 2:
                command.append('--reverse')
            with (a.output/(dest.name+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=log, check=True)
        # Separate production-pipeline control, never mixed into selector ratios.
        for repeat in range(3):
            path = dest/f'native-{repeat}.jsonl'
            if path.exists():
                records = [json.loads(line) for line in path.read_text().splitlines()]
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_MODULE_LOADING='EAGER',
                           SECANT_BENCH_MODE='native')
                for key in ['SECANT_BENCH_GRID', 'SECANT_BENCH_SAMPLES', 'SECANT_BENCH_STATS', 'SECANT_BENCH_CAPTURE']:
                    env.pop(key, None)
                with path.open('w') as out, path.with_suffix('.stderr').open('w') as err:
                    subprocess.run([str(a.replay.resolve()), str(request), '8'],
                                   env=env, stdout=out, stderr=err, check=True, timeout=600)
                records = [json.loads(line) for line in path.read_text().splitlines()]
            if not any(r.get('event') == 'cpu_audit' and r['accepted'] for r in records):
                raise RuntimeError('native replay audit missing/failed')
        print('complete', request, flush=True)

if __name__ == '__main__':
    main()
