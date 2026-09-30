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
"""Observation-only learner. No imports from generator/grader or private reads."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import rfm


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False) + '\n'); tmp.replace(path)


def view(data, count, duration, stride=1):
    times = np.asarray(data['times'])
    ids = np.flatnonzero(times <= duration + 1e-9)[::stride]
    t = times[ids]
    obs = np.asarray(data['observations'])[:, ids]
    nt = data['training_trajectories']
    selected = list(range(count)) + list(range(nt, len(obs)))
    x = obs[selected, 1:-1]
    dy = np.stack([CubicSpline(t, obs[k], axis=0)(t[1:-1], 1) for k in selected])
    return x[:count].reshape(-1, x.shape[-1]), dy[:count].reshape(-1, x.shape[-1]), x[count:].reshape(-1, x.shape[-1]), dy[count:].reshape(-1, x.shape[-1])


def run(args):
    torch.set_num_threads(4)
    root = Path(args.data); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text()); cfg = manifest['config']
    if not manifest['complete']: raise ValueError('Incomplete generation')
    device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    provenance = {'config': cfg, 'device': str(device), 'torch': torch.__version__,
                  'host': platform.node(), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  'learner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'rfm_sha256': hashlib.sha256(Path(rfm.__file__).read_bytes()).hexdigest()}
    provenance['equal_budget'] = args.equal_budget
    fingerprint = hashlib.sha256(json.dumps({k:v for k,v in provenance.items() if k not in ('device','gpu','host')},sort_keys=True).encode()).hexdigest()
    save(out / f'worker-{args.shard}.json', provenance)
    for ci, case in enumerate(manifest['cases']):
        if ci % args.shards != args.shard: continue
        if args.limit and ci >= args.limit: continue
        data_path = root / case['path']
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != case['sha256']: raise ValueError('Data hash mismatch')
        data = json.loads(data_path.read_text())
        for count in cfg['counts']:
            durations = cfg['durations']
            if args.equal_budget:
                if args.equal_budget % count or args.equal_budget // count < 5: raise ValueError('Budget needs at least five samples per trajectory and exact divisibility')
                durations = [(args.equal_budget // count - 1) * cfg['spacing']]
                if durations[0] > max(cfg['durations']): raise ValueError('Equal budget exceeds available duration')
            for duration in durations:
                path = out / f'{case["id"]}-m{count}-t{duration:g}.json'
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous.get('fingerprint') != fingerprint or previous['input_sha256'] != case['sha256']: raise ValueError('Resume provenance mismatch')
                    continue
                started = time.perf_counter()
                x, y, xv, yv = view(data, count, duration)
                xm = x.mean(0); xs = x.std(0); xs[xs < 1e-10] = 1.
                ym = y.mean(0); ys = y.std(0); ys[ys < 1e-10] = 1.
                arrays = [(x-xm)/xs, (y-ym)/ys, (xv-xm)/xs, (yv-ym)/ys]
                xt, yt, xvt, yvt = [torch.as_tensor(a, dtype=torch.float64, device=device) for a in arrays]
                prep = time.perf_counter() - started
                rows = [rfm.learn(xt, yt[:, i], xvt, yvt[:, i], cfg) for i in range(x.shape[1])]
                result = {'id': case['id'], 'split': case['split'], 'count': count, 'duration': duration,
                          'fingerprint': fingerprint, 'equal_budget_observations': args.equal_budget,
                          'samples': len(x), 'input_sha256': case['sha256'], 'methods': {},
                          'scales': {'x': xs.tolist(), 'y': ys.tolist()}, 'preprocessing_seconds': prep}
                for mode in ('fixed', 'diagonal', 'full'):
                    result['methods'][mode] = {'matrices': [r[mode]['matrix'] for r in rows],
                                              'validation_mse_scaled': [r[mode]['validation_mse_scaled'] for r in rows],
                                              'seconds_with_shared_initial': sum(r[mode]['incremental_seconds'] + r[mode]['shared_initial_seconds'] for r in rows)}
                result['seconds'] = time.perf_counter() - started
                save(path, result)
                print(json.dumps({k: result[k] for k in ('id', 'count', 'duration', 'samples', 'seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data'); p.add_argument('out'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--equal-budget', type=int, default=0)
    run(p.parse_args())
