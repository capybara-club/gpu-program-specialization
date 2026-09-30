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
"""Validate all replay scores, then compare resident and full batch timings."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def compare(request, paths):
    with request.open('rb') as f:
        header = struct.unpack('<10Q', f.read(80))
        _, _, columns, constants, banks, bits, rows, asts, _, _ = header
        f.seek(80 + rows * columns * 4)
        targets = array.array('f', f.read(rows * 4))
    target_rms = math.sqrt(sum(float(y)*y for y in targets) / rows)
    configs = banks * 2**bits
    expected = asts * configs
    streams = [p.open('rb') for p in paths]
    checked = mismatch = finite = equal = 0
    max_gap = 0.0
    worst = None
    try:
        while True:
            raw = [f.read(4*262144) for f in streams]
            if not raw[0] and not raw[1]:
                break
            if len(raw[0]) != len(raw[1]) or len(raw[0]) % 4:
                raise ValueError('incomplete score grids')
            a, b = [array.array('f', r) for r in raw]
            for x, y in zip(a, b):
                index = checked
                checked += 1
                fx, fy = math.isfinite(x) and x >= 0, math.isfinite(y) and y >= 0
                if fx != fy:
                    mismatch += 1
                    continue
                if not fx:
                    continue
                finite += 1
                if x == y:
                    equal += 1
                    continue
                rx, ry = math.sqrt(x/rows), math.sqrt(y/rows)
                gap = abs(rx-ry) / max(1, target_rms, rx, ry)
                if gap > max_gap:
                    max_gap, worst = gap, index
    finally:
        for f in streams:
            f.close()
    result = dict(checked=checked, expected=expected, finite=finite,
                  exactly_equal_finite=equal, finite_classification_mismatches=mismatch,
                  max_relative_rmse_gap=max_gap, worst_index=worst, tolerance=2e-5)
    result['accepted'] = checked == expected and mismatch == 0 and max_gap <= 2e-5
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--pack', type=int, default=2)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--reverse', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(request_sha256=digest(a.request), binary_sha256=digest(a.replay),
                  legacy_sha256=digest(a.legacy), packed=a.pack, gpu=a.gpu,
                  started_unix=time.time(), modes={})
    try:
        for mode in (['toggles','settings'] if a.reverse else ['settings','toggles']):
            folder = a.output / mode
            folder.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_MODULE_LOADING='EAGER',
                       SECANT_BENCH_MODE=mode, SECANT_BENCH_LEGACY=str(a.legacy.resolve()),
                       SECANT_BENCH_GRID=str((folder/'scores.f32').resolve()),
                       SECANT_BENCH_STATS=str((folder/'stats.jsonl').resolve()),
                       SECANT_BENCH_SAMPLES=str(a.samples))
            start = time.perf_counter()
            with (folder/'stdout.jsonl').open('w') as out, (folder/'stderr.txt').open('w') as err:
                subprocess.run([str(a.replay.resolve()), str(a.request.resolve()), str(a.pack)],
                               env=env, stdout=out, stderr=err, check=True, timeout=600)
            records = [json.loads(line) for line in (folder/'stdout.jsonl').read_text().splitlines()]
            audit = next(r for r in records if r['event']=='cpu_audit')
            if not audit['accepted']:
                raise ValueError('CPU audit failed')
            result['modes'][mode] = dict(replay=next(r for r in records if r['event']=='replay'),
                cpu_audit=audit, stats=json.loads((folder/'stats.jsonl').read_text()),
                process_seconds=time.perf_counter()-start)
        result['scores'] = compare(a.request, [a.output/m/'scores.f32' for m in ['settings','toggles']])
        if not result['scores']['accepted']:
            raise ValueError('full-grid agreement check failed')
        # Measure the complete batch again, without resident repeats, grid copies
        # or CPU validation inside the reported engine wall time.
        for mode in ['settings', 'toggles']:
            result['modes'][mode]['pipeline_samples'] = []
        for repeat in range(3):
            order = ['toggles', 'settings'] if (repeat + a.reverse) % 2 else ['settings', 'toggles']
            for mode in order:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_MODULE_LOADING='EAGER',
                           SECANT_BENCH_MODE=mode, SECANT_BENCH_LEGACY=str(a.legacy.resolve()))
                for name in ['SECANT_BENCH_GRID', 'SECANT_BENCH_SAMPLES', 'SECANT_BENCH_STATS']:
                    env.pop(name, None)
                done=subprocess.run([str(a.replay.resolve()),str(a.request.resolve()),str(a.pack)],
                                    env=env,text=True,capture_output=True,check=True,timeout=600)
                (a.output/mode/f'pipeline-{repeat}.jsonl').write_text(done.stdout)
                (a.output/mode/f'pipeline-{repeat}.stderr').write_text(done.stderr)
                records=[json.loads(line) for line in done.stdout.splitlines()]
                result['modes'][mode]['pipeline_samples'].append(next(r for r in records if r['event']=='replay'))
                if not next(r for r in records if r['event']=='cpu_audit')['accepted']:
                    raise ValueError('pipeline CPU audit failed')
        result['status']='complete'
    except BaseException as error:
        result.update(status='failed',error=str(error))
        raise
    finally:
        result['finished_unix']=time.time()
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
