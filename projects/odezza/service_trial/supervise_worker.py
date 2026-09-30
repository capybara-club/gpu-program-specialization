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
"""Process lifetime only: restart a failed C GPU executor with a fresh context.

No JSON parsing, AST generation, scheduling, CUDA calls, or result manipulation.
JetStream/mac3 own attempts. SIGTERM drains the current C child and stops restarting.
Repeated immediate startup failures trip a circuit breaker instead of spinning.
"""
import argparse
from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import time


def resident_bytes(pid):
    """Linux current RSS; missing process returns None, not a false zero."""
    try:
        return int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return None


def wait_worker(child, rss_limit):
    """External guard includes NVRTC/core allocations and works during C stalls.

    This is a 100 ms sampled kill threshold, not a no-overshoot allocation limit.
    """
    peak = 0
    while child.poll() is None:
        rss = resident_bytes(child.pid)
        if rss is None and child.poll() is None:
            print(f'executor_memory_unobservable pid={child.pid}', flush=True)
            child.kill()
            return child.wait(), peak, 'rss_unobservable'
        if rss is not None:
            peak = max(peak, rss)
            if rss > rss_limit:
                print(f'executor_memory_limit pid={child.pid} rss_bytes={rss} limit_bytes={rss_limit}', flush=True)
                child.kill()
                return child.wait(), peak, 'rss_limit'
        try:
            return child.wait(timeout=.1), peak, None
        except subprocess.TimeoutExpired:
            pass
    return child.wait(), peak, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', default='0,1')
    p.add_argument('--url', default='nats://127.0.0.1:14223')
    p.add_argument('--cache', default='/tmp/odezza-service-trial-cache')
    p.add_argument('--trajectory-pool-mib', type=int, default=64)
    p.add_argument('--tile-device-pool-mib', type=int, default=256)
    p.add_argument('--tile-host-pool-mib', type=int, default=64)
    p.add_argument('--job-host-limit-mib', type=int, default=2048)
    p.add_argument('--max-worker-rss-mib', type=int, default=4096)
    a = p.parse_args()
    if not all(1 <= n <= 8192 for n in (a.trajectory_pool_mib,a.tile_device_pool_mib,a.tile_host_pool_mib)):
        p.error('pool sizes must be in 1..8192 MiB')
    if not 1 <= a.job_host_limit_mib <= 8192 or not 1 <= a.max_worker_rss_mib <= 65536:
        p.error('job host limit must be 1..8192 MiB; worker RSS limit 1..65536 MiB')
    if not Path('/proc/self/statm').is_file():
        p.error('worker RSS supervision requires Linux /proc')
    devices = [int(x) for x in a.devices.split(',')]
    if not devices or len(devices) > 8 or len(set(devices)) != len(devices) or any(x < 0 or x > 7 for x in devices):
        p.error('devices must be distinct indices in 0..7')
    root = Path(__file__).resolve().parents[1]
    executable = root/'build/service_trial/worker'
    stop = False
    child = None

    def terminate(signum, frame):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    env = dict(os.environ, CUDA_MODULE_LOADING='EAGER')
    env.update(ODEZZA_TRAJECTORY_POOL_MIB=str(a.trajectory_pool_mib),
               ODEZZA_TILE_DEVICE_POOL_MIB=str(a.tile_device_pool_mib),
               ODEZZA_TILE_HOST_POOL_MIB=str(a.tile_host_pool_mib),
               ODEZZA_JOB_HOST_LIMIT_MIB=str(a.job_host_limit_mib))
    failures = deque()
    while not stop:
        start = time.monotonic()
        child = subprocess.Popen([str(executable), a.url, ','.join(map(str, devices)), a.cache],
                                 cwd=root, env=env, stdin=subprocess.DEVNULL)
        if stop:
            child.send_signal(signal.SIGTERM)
        print(f'executor_started pid={child.pid} devices={a.devices}', flush=True)
        code, peak, memory_reason = wait_worker(child, a.max_worker_rss_mib*1024*1024)
        elapsed = time.monotonic()-start
        print(f'executor_exited pid={child.pid} code={code} lifetime_seconds={elapsed:.3f} '
              f'peak_sampled_rss_bytes={peak} memory_reason={memory_reason}', flush=True)
        child = None
        if stop:
            break
        now = time.monotonic()
        while failures and now-failures[0] > 60:
            failures.popleft()
        if elapsed < 10:
            failures.append(now)
        if len(failures) >= 5:
            raise SystemExit('executor_startup_circuit_open: five short failures in 60 seconds')
        until = time.monotonic()+min(8, 1+len(failures))
        while not stop and time.monotonic() < until:
            time.sleep(.1)


if __name__ == '__main__':
    main()
