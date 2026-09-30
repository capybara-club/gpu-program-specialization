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
"""Real worker cancellation/loss and shared two-GPU execution acceptance.

Starts/stops only this trial's worker executables. Does not touch existing services.
"""
import json
from pathlib import Path
import signal
import subprocess
import time
from client import Client

REMOTE = '/home/cdurham/odezza/scratch/service_trial_20260912/odezza-native-trial'


def remote(script):
    return subprocess.check_output(['ssh', 'rack1', 'python3', '-'], input=script.encode()).decode()


def start_worker(devices):
    assert devices in ('0', '1', '0,1')
    script = f'''
import subprocess,pathlib,os,json
root=pathlib.Path({REMOTE!r})
env=dict(os.environ,CUDA_MODULE_LOADING='EAGER',LD_LIBRARY_PATH='/usr/local/cuda/lib64')
log=(root/'build/service_trial/worker-{devices}.log').open('ab')
p=subprocess.Popen([str(root/'build/service_trial/worker'),'nats://127.0.0.1:14223',{devices!r},'/tmp/odezza-service-trial-cache'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'build/service_trial/worker-{devices}.pid').write_text(str(p.pid))
print(json.dumps(dict(pid=p.pid,devices={devices!r})))
'''
    return json.loads(remote(script))['pid']


def stop_worker(pid, sig=signal.SIGTERM):
    script = f'''
import os,pathlib,signal,time
pid={int(pid)}
proc=pathlib.Path('/proc')/str(pid)
if proc.exists():
 cmd=(proc/'cmdline').read_bytes()
 if cmd:
  assert b'build/service_trial/worker' in cmd,cmd
  assert str((proc/'cwd').resolve())=={REMOTE!r}
  os.kill(pid,{int(sig)})
  for _ in range(100):
   if not proc.exists() or not (proc/'cmdline').read_bytes():break
   time.sleep(.05)
  else:raise RuntimeError('trial worker did not finish draining')
'''
    remote(script)


def wait(c, handle, state='terminal', timeout=45):
    until = time.monotonic()+timeout
    while True:
        value = c.status(handle)
        if value['state'] == state:
            return value
        assert time.monotonic() < until, value
        time.sleep(.02)


def main():
    root = Path(__file__).resolve().parent
    raw = (root/'validation/million-2048-request.json').read_bytes()
    ref = json.loads((root/'validation/million-direct-attempt.json').read_text())[-1]['report']
    pids = [start_worker('0'), start_worker('1')]
    c = Client()
    out = {}
    try:
        admission = c.submit(raw)
        h = admission['handle']
        wait(c, h, 'running')
        begin = time.perf_counter()
        c.rpc('cancel.'+h)
        status = wait(c, h)
        report = c.result(h)
        assert report['status'] == 'cancelled', report['status']
        assert status['attempts'] == 1
        out['active_cancel'] = dict(seconds=time.perf_counter()-begin, service=status, report=report)
        c.rpc('release.'+h)

        h = c.submit(raw)['handle']
        active = wait(c, h, 'running')
        pid = active['worker_info']['pid']
        assert pid in pids
        begin = time.perf_counter()
        stop_worker(pid, signal.SIGKILL)
        pids.remove(pid)
        status = wait(c, h)
        report = c.result(h)
        assert status['attempts'] == 2, status
        assert report['status'] == 'complete'
        for key in ('candidates', 'leaderboards', 'families'):
            assert report[key] == ref[key], key
        out['worker_loss'] = dict(seconds=time.perf_counter()-begin, killed=active['worker_info'], service=status, report=report)
        c.rpc('release.'+h)

        for pid in pids:
            stop_worker(pid)
        pids.clear()
        time.sleep(.5)
        pids.append(start_worker('0,1'))
        runs = []
        out['dual_gpu_group'] = runs
        for _ in range(3):
            begin = time.perf_counter()
            h = c.submit(raw)['handle']
            status = wait(c, h)
            report = c.result(h)
            runs.append(dict(end_to_end_seconds=time.perf_counter()-begin, service=status, report=report))
            assert report['status'] == 'complete'
            assert report['numeric_input_scope'] == 'attempt'
            for key in ('candidates', 'leaderboards', 'families'):
                assert report[key] == ref[key], key
            devices = report['execution']['devices']
            assert len(devices) == 2 and all(d['completed_configurations'] > 0 for d in devices)
            c.rpc('release.'+h)
        out['dual_gpu_group'] = runs
        out['passed'] = True
        out['live_worker'] = dict(pid=pids[0], devices=[0, 1], host='rack1')
        (root/'validation/recovery-and-dual.json').write_text(json.dumps(out, indent=2)+'\n')
        print(json.dumps(dict(passed=True, cancel_seconds=out['active_cancel']['seconds'],
                              worker_loss_recovery_seconds=out['worker_loss']['seconds'],
                              dual_end_to_end_seconds=[r['end_to_end_seconds'] for r in runs], live_worker=out['live_worker'])))
    except BaseException:
        for pid in pids:
            stop_worker(pid)
        (root/'validation/recovery-failure.json').write_text(json.dumps(out, indent=2)+'\n')
        raise
    finally:
        c.close()


if __name__ == '__main__':
    main()
