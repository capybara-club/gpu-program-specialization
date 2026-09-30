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
"""mac1 supervisor: generate privately, admit paired public jobs, collect results.

Workers are detached user services. Restarting this supervisor never resubmits a
running group. Overnight admission requires the explicit RELEASE_OVERNIGHT file
after the pilot has been inspected. All decisions and seeds are retained.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from .common import REPO, save, load, digest, can_admit
from .deploy import ssh, lanes, python_for
from .protocol import validate, sample_case, settings, pair
from .summarize import summarize
from .completion import message as completion_message


def collect(root, remote, entries):
    states = {}
    for host, device, lane in entries:
        destination = root / 'hosts' / lane
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('status.json', 'results.jsonl', 'heartbeat.json', 'hardware.json'):
            try:
                result = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    host + ':' + remote + '/queues/' + lane + '/' + name, str(destination / (name + '.incoming'))],
                    capture_output=True, timeout=20)
            except subprocess.TimeoutExpired:
                save(destination / 'connection-error.json', dict(at=time.time(), reason='copy_timeout'))
                continue
            if result.returncode == 0:
                (destination / (name + '.incoming')).replace(destination / name)
        if (destination / 'status.json').exists():
            states[lane] = load(destination / 'status.json')
        else:
            states[lane] = {'status': 'starting'}
        receipt = destination / 'launch.json'
        if receipt.exists() and not states[lane].get('finished_at'):
            launch = load(receipt)
            try:
                unit = ssh(host, ['systemctl', '--user', 'show', launch['unit'], '--property=ActiveState,SubState,Result'], timeout=15).stdout
                if time.time()-launch['at'] > 30 and ('ActiveState=failed' in unit or 'ActiveState=inactive' in unit):
                    states[lane].update(status='failed', finished_at=time.time(), error='Worker exited before terminal status', unit_state=unit)
                    save(destination / 'status.json', states[lane])
            except subprocess.SubprocessError:
                save(destination / 'connection-error.json', dict(at=time.time(), reason='unit_status_unavailable'))
    save(root / 'status.json', dict(at=time.time(), lanes=states))
    return states


def upload(root, remote, entry, group, sequence):
    host, _, lane = entry
    name = f'{sequence:08d}-' + group['group_id'] + '.json'
    local = root / 'groups' / name
    if not local.exists():
        save(local, group)
    receipt = root / 'delivery' / lane / name
    if receipt.exists():
        return
    target = remote + '/queues/' + lane + '/inbox/' + name
    subprocess.run(['scp', '-q', str(local), host + ':' + target + '.upload'], check=True, timeout=30)
    ssh(host, ['mv', target + '.upload', target], timeout=20)
    save(receipt, dict(group_id=group['group_id'], sequence=sequence, delivered_at=time.time(), phase=group['phase']))


def start_workers(root, protocol, remote, entries, deadline):
    for host, device, lane in entries:
        receipt = root / 'hosts' / lane / 'launch.json'
        if receipt.exists():
            continue
        queue = remote + '/queues/' + lane
        ssh(host, ['mkdir', '-p', queue + '/inbox'])
        unit = protocol['name'].replace('_', '-') + '-' + lane.replace('_', '-')
        command = ['systemd-run', '--user', '--unit=' + unit,
                   '--property=RuntimeMaxSec=' + str(max(1, int(deadline - time.time() + 30))),
                   '--property=KillMode=control-group', '--property=TimeoutStopSec=20',
                   '--working-directory=' + remote, '--setenv=CUDA_MODULE_LOADING=EAGER',
                   '--setenv=ODEZZA_CORE_LIBRARY=' + remote + '/build/libodezza.so',
                   '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=ODEZZA_TELEGRAM_DISABLE=1',
                   python_for(host), '-B', '-m', 'benchmarks.lm_tuning.worker',
                   '--root', queue, '--device', str(device), '--deadline', str(deadline)]
        result = ssh(host, command)
        save(receipt, dict(at=time.time(), unit=unit, stdout=result.stdout, stderr=result.stderr))


def pilot_settings(protocol):
    return dict(initial=65536, offspring=1000000, wave=125000, banks=1024, parents=1024,
                candidates=64, starts=4, iterations=16, fit_seconds=20, toggle_width=4,
                initial_damping=.001, damping_attempts=8, max_step=.4, seed=0,
                seconds=protocol['pilot']['seconds'])


def generate_pilot(root, index):
    case = root / 'cases' / f'pilot-{index:02d}'
    if (case / 'public.json').exists():
        return load(case / 'public.json')
    case.mkdir(parents=True, exist_ok=True)
    private = case / 'private'
    private.mkdir(exist_ok=True)
    config = root / 'pilot-depth3.json'
    if not config.exists():
        config.write_bytes((REPO / 'examples/grammar_game/depth3.json').read_bytes())
    began = time.monotonic()
    with (private / 'solution.txt').open('w') as answer, (private / 'generation.log').open('w') as log:
        result = subprocess.run([sys.executable, '-B', str(REPO / 'ode_game.py'), 'generate', str(config), '--out', str(case)],
                                stdout=answer, stderr=log, timeout=180)
    if result.returncode:
        raise RuntimeError('Pilot generation failed; inspect retained rejection log')
    public, = case.glob('pilot-depth3-*.json')
    public.rename(case / 'public.json')
    save(case / 'generation.json', dict(status='accepted', seconds=time.monotonic() - began,
         source='unchanged ode_game.py depth3 grammar', public_sha256=digest(load(case / 'public.json'))))
    return load(case / 'public.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = validate(load(root / 'protocol.json'))
    remote = load(root / 'deployment.json')['remote']
    entries = lanes(protocol['hosts'])
    deadline = datetime.fromisoformat(protocol['deadline_utc'].replace('Z', '+00:00')).timestamp()
    start_workers(root, protocol, remote, entries, deadline)
    # Public challenges are shared across hosts; paired methods stay on one GPU.
    for index in range(1, protocol['pilot']['count'] + 1):
        payload = generate_pilot(root, index)
        for host in protocol['hosts']:
            entry = next(e for e in entries if e[0] == host and e[1] == ((index - 1) % 2 if host == 'rack1' else 0))
            group = pair(f'pilot-{index:02d}', payload, 'search', pilot_settings(protocol), index, phase='pilot')
            upload(root, remote, entry, group, index)
        save(root / 'pilot-generation.json', dict(generated=index, count=protocol['pilot']['count']))
    counter = load(root / 'coordinator-progress.json').get('next_case', 0) if (root / 'coordinator-progress.json').exists() else 0
    while time.time() < deadline:
        states = collect(root, remote, entries)
        summarize(root)
        if all(s.get('finished_at') for s in states.values()):
            break
        if not (root / 'RELEASE_OVERNIGHT').exists():
            time.sleep(20)
            continue
        for entry in entries:
            lane = entry[2]
            state = states[lane]
            if state.get('finished_at') or state['status'] == 'failed':
                continue
            delivered = list((root / 'delivery' / lane).glob('*.json'))
            if len(delivered) - state.get('finished_groups', 0) >= protocol['schedule']['prefetch_per_lane']:
                continue
            spec = sample_case(protocol, counter)
            case_id = f'fresh-{counter:06d}'
            case = root / 'cases' / case_id
            spec_path = root / 'specifications' / (case_id + '.json')
            save(spec_path, spec)
            # Case counter is committed before generation; exhausted/terminated
            # draws remain visible and are never mistaken for accepted systems.
            index = counter
            counter += 1
            save(root / 'coordinator-progress.json', dict(next_case=counter))
            try:
                with (spec_path.with_suffix('.log')).open('w') as log:
                    result = subprocess.run([sys.executable, '-B', '-m', 'benchmarks.lm_tuning.generate',
                        '--spec', str(spec_path), '--out', str(case)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                        timeout=spec['seconds'] + 60)
            except subprocess.TimeoutExpired:
                save(case / 'generation-timeout.json', dict(status='rejected', reason='generation_wall_timeout', seconds=spec['seconds']+60))
                continue
            if result.returncode == 2:
                continue
            if result.returncode:
                raise RuntimeError('Unexpected random generator failure: ' + case_id)
            for repeat in range(protocol['schedule']['fit_groups_per_case']):
                value = settings(protocol, 'fit', index * 10 + repeat)
                value['seed'] = index
                group = pair(case_id, load(case / 'calibration.json'), 'fit', value, index + repeat)
                if can_admit(group, deadline):
                    upload(root, remote, entry, group, 1000 + index * 10 + repeat)
            if index % protocol['schedule']['recovery_every_cases'] == 0:
                value = settings(protocol, 'search', index)
                value['seed'] = index
                group = pair(case_id, load(case / 'public.json'), 'search', value, index)
                if can_admit(group, deadline):
                    upload(root, remote, entry, group, 1000 + index * 10 + 9)
        time.sleep(10)
    # Final worker state may arrive just after the wall deadline.
    for _ in range(12):
        states = collect(root, remote, entries)
        if all(s.get('finished_at') for s in states.values()):
            break
        time.sleep(5)
    result = summarize(root)
    confirmed = all(s.get('finished_at') for s in states.values())
    successful = confirmed and all(s['status']=='complete' for s in states.values())
    save(root / 'completion.json', dict(at=time.time(), status='complete' if successful else 'failed_or_unconfirmed', lanes=states, completed_pairs=result['completed_pairs']))
    # Existing user authorization: one whole-campaign completion notice.
    sys.path.insert(0, str(REPO / 'scratch/fitting_batch_trial'))
    from telegram_notify import send
    message = completion_message(states, result['completed_pairs'], now=time.time(), deadline=deadline)
    for _ in range(12):
        try:
            mid = send(message)
            save(root / 'telegram-complete.json', dict(message_id=mid, at=time.time(), message=message))
            break
        except RuntimeError:
            time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Preserve failure for the external observer; submitted host work is
        # never silently cancelled or launched a second time.
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1])
            save(root / 'coordinator-error.json', dict(at=time.time(), error=type(error).__name__ + ': ' + str(error)))
        raise
