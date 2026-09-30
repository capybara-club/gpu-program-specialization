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
"""Kill the active supervised GPU child and verify automatic replacement/replay."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from client import Client
from tests_recovery import stop_worker, wait


def main():
    root = Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'validation/supervised-recovery.json')
    parser.add_argument('--port',type=int,default=14222)
    parser.add_argument('--local-worker',action='store_true')
    args=parser.parse_args()
    raw = (root/'validation/million-2048-request.json').read_bytes()
    ref = json.loads((root/'validation/million-direct-attempt.json').read_text())[-1]['report']
    c = Client(port=args.port)
    try:
        h = c.submit(raw)['handle']
        before = wait(c, h, 'running')
        start = time.perf_counter()
        pid=before['worker_info']['pid']
        if args.local_worker:
            proc=Path('/proc')/str(pid)
            assert (proc/'cwd').resolve()==root.parent
            assert b'build/service_trial/worker' in (proc/'cmdline').read_bytes()
            os.kill(pid,signal.SIGKILL)
        else:
            stop_worker(pid, signal.SIGKILL)
        after = wait(c, h, timeout=45)
        report = c.result(h)
        assert after['attempts'] == 2
        assert after['worker_info']['pid'] != before['worker_info']['pid']
        assert report['status'] == 'complete'
        for key in ('candidates', 'leaderboards', 'families'):
            assert report[key] == ref[key], key
        result = dict(passed=True, recovery_seconds=time.perf_counter()-start,
                      before=before, after=after, report=report)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        c.rpc('release.'+h)
        print(json.dumps(dict(passed=True, seconds=result['recovery_seconds'], worker=after['worker_info'])))
    finally:
        c.close()


if __name__ == '__main__':
    main()
