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
"""Run on mac3: keep a direct private connection to a GPU host alive.

The GPU host's loopback port forwards to the broker on this machine. No client
machine is involved. Requires existing authenticated SSH access and known hosts.
"""
import argparse
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', required=True)
    p.add_argument('--worker-port', type=int, default=14223)
    p.add_argument('--broker-port', type=int, default=4222)
    a = p.parse_args()
    if a.worker.startswith('-') or not all(0 < port < 65536 for port in (a.worker_port, a.broker_port)):
        p.error('invalid worker or port')
    stopped = False
    child = None

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopped:
        child = subprocess.Popen([
            'ssh', '-NT', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
            '-R', f'127.0.0.1:{a.worker_port}:127.0.0.1:{a.broker_port}', a.worker,
        ], stdin=subprocess.DEVNULL)
        if stopped:
            child.terminate()
        print(f'tunnel_started pid={child.pid} worker={a.worker} port={a.worker_port}', flush=True)
        code = child.wait()
        print(f'tunnel_exited code={code}', flush=True)
        child = None
        until = time.monotonic() + 2
        while not stopped and time.monotonic() < until:
            time.sleep(.1)


if __name__ == '__main__':
    main()
