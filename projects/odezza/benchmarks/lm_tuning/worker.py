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
"""Durable single-GPU queue. One process owns one resident execution engine.

Pending groups resume; ambiguous running groups require explicit reconciliation.
The surrounding user systemd unit kills all native children on a fatal watchdog.
"""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
import time
import traceback
from .common import SCHEMA, REPO, save, load, append, file_hash, can_admit, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('inbox', 'running', 'done', 'trials'):
        (root / name).mkdir(exist_ok=True)
    if list((root / 'running').glob('*.json')):
        raise RuntimeError('Unreconciled running group: inspect its trial result before resuming')
    from .run_trial import Engine, fit, search
    from odezza.native import NativeError
    backend = REPO / 'scratch/fitting_batch_trial/bin/odezza-fit-core-run'
    library = Path(os.environ['ODEZZA_CORE_LIBRARY'])
    engine = Engine(backend, [args.device])
    completed = len(list((root / 'done').glob('*.json')))
    previous_trials = len((root / 'results.jsonl').read_text().splitlines()) if (root / 'results.jsonl').exists() else 0
    state = dict(status='waiting', started_at=time.time(), deadline=args.deadline,
                 device=args.device, host=platform.node(), finished_groups=completed, finished_trials=previous_trials)
    save(root / 'hardware.json', dict(device=args.device, host=platform.node(),
         native_library_sha256=file_hash(library), scoring_worker_sha256=file_hash(backend),
         eager=os.environ.get('CUDA_MODULE_LOADING'), execution='native C99 LM and native C99 scoring; single GPU per worker',
         gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'], text=True)))
    stopped = threading.Event()
    def telemetry():
        while not stopped.is_set():
            save(root / 'heartbeat.json', dict(at=time.time(), **state))
            try:
                data = subprocess.run(['nvidia-smi', '--id=' + str(args.device),
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
                append(root / 'gpu.jsonl', dict(at=time.time(), group=state.get('current_group'), values=data.stdout.strip()))
            except subprocess.SubprocessError:
                pass
            stopped.wait(10)
    thread = threading.Thread(target=telemetry, daemon=True)
    thread.start()
    save(root / 'status.json', state)
    fatal = False
    try:
        while time.time() < args.deadline:
            files = sorted((root / 'inbox').glob('*.json'))
            if not files:
                if (root / 'CLOSED').exists():
                    break
                stopped.wait(2)
                continue
            path = files[0]
            group = load(path)
            if group['schema'] != SCHEMA or digest({k: v for k, v in group.items() if k != 'group_id'})[:24] != group['group_id']:
                raise ValueError('Queue group identity mismatch')
            if not can_admit(group, args.deadline):
                state['status'] = 'deadline_admission_closed'
                break
            if (root / 'done' / path.name).exists():
                path.unlink()
                continue
            claimed = root / 'running' / path.name
            path.rename(claimed)
            state.update(status='running', current_group=group['group_id'], phase=group['phase'])
            save(root / 'status.json', state)
            results = []
            for index, trial in enumerate(group['trials']):
                state.update(current_method=trial['method'], current_started_at=time.time())
                save(root / 'status.json', state)
                trial_id = trial.get('trial_id', trial['method'])
                if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', trial_id):
                    raise ValueError('Invalid trial identity')
                directory = root / 'trials' / (group['group_id'] + '-' + trial_id)
                def watchdog():
                    save(root / 'watchdog.json', dict(group=group['group_id'], method=trial['method'], at=time.time()))
                    os._exit(71)
                timer = threading.Timer(trial['seconds'] + 120, watchdog)
                timer.daemon = True
                timer.start()
                began = time.monotonic()
                try:
                    if group['kind'] == 'fit':
                        result = engine.pools[args.device].submit(fit, group['payload'], trial['settings'], trial['method'], directory, engine).result()
                    else:
                        result = search(group['payload'], trial['settings'], trial['method'], directory, engine)
                except NativeError as error:
                    result = dict(status='unsupported_resource' if error.result in (7, 11, 12) else 'error',
                                  error=str(error), native_result=error.result, verified=False)
                    fatal = error.result not in (7, 11, 12)
                except Exception as error:
                    result = dict(status='error', error=str(error), verified=False)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'exception.txt').write_text(traceback.format_exc())
                    fatal = True
                finally:
                    timer.cancel()
                result.update(case_id=group['case_id'], group_id=group['group_id'], kind=group['kind'], phase=group['phase'],
                              method=trial['method'], settings=trial['settings'], device=args.device,
                              total_seconds=time.monotonic() - began, order=index, at=time.time())
                if 'trial_id' in trial:result['trial_id']=trial_id
                save(directory / 'summary.json', result)
                append(root / 'results.jsonl', result)
                results.append(result)
                state['finished_trials'] += 1
                if result['status'] in ('error', 'interrupted'):
                    fatal = True
                    break
            if fatal:
                state.update(status='failed', error=results[-1].get('error'))
                break
            save(root / 'done' / path.name, dict(group=group, results=results, finished_at=time.time()))
            claimed.unlink()
            state.update(status='waiting', current_group=None, current_method=None)
            state['finished_groups'] += 1
            save(root / 'status.json', state)
    except Exception as error:
        fatal = True
        state.update(status='failed', error=str(error))
        (root / 'exception.txt').write_text(traceback.format_exc())
    finally:
        if not fatal:
            engine.close()
        # Fatal CUDA errors are reclaimed by process/cgroup teardown, not unsafe frees.
        stopped.set()
        thread.join(timeout=12)
        state.update(finished_at=time.time(), unstarted_groups=len(list((root / 'inbox').glob('*.json'))))
        if not fatal:
            state['status'] = 'complete'
        save(root / 'status.json', state)
    if fatal:
        os._exit(1)


if __name__ == '__main__':
    main()
