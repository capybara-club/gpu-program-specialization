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
"""Run the diagnostic in mac1 tmux and deliver its own lifecycle notifications."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from notify_campaign import deliver

root=Path(sys.argv[1]).resolve()
remote='/home/cdurham/experiments/secant-polish-20260920'
sender=Path(__file__).resolve().parents[3]/'secant/scripts/secant-notify'
receipt=root/'telegram-receipts.json'
with (root/'.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(receipt.read_text()) if receipt.exists() else dict(sent={})
    if 'finish' in state['sent']:raise SystemExit('Diagnostic already completed')
    def notify(event,message):
        while not deliver(sender,receipt,state,event,message):time.sleep(20)
    notify('start','Secant STARTED\nCoefficient rescue diagnostic on rack1 CPU\n13 old-system wins/current misses; two fitting policies.\nRetained structures only, training-only coefficient fitting. No new structural search.')
    status=1
    try:
        with (root/'run.log').open('w') as log:
            run=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=15','rack1',
                f'cd {remote} && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/cdurham/odeformer-trial/.venv/bin/python bench/polish/polish.py --manifest manifest.json --data /home/cdurham/experiments/secant-search-night-20260919 --output run --seconds 15'],stdout=log,stderr=log)
        subprocess.run(['scp',f'rack1:{remote}/run/result.json',str(root/'result.json')],check=True)
        r=json.loads((root/'result.json').read_text())
        status=run.returncode
        if status or r['status']!='complete':raise RuntimeError('Diagnostic failed; inspect retained log/result')
        lines=['Secant FINISHED','Coefficient rescue diagnostic','Elapsed: %.1f seconds'%(r['finished_unix']-r['started_unix'])]
        for mode in ['tied','all_literals']:
            lines.append('%s: %d/%d retained winners rescued (held-out R2 > .999)'%
                         (mode,sum(c['modes'][mode]['recovered'] for c in r['cases']),len(r['cases'])))
        lines.append('Fixed-structure CPU diagnostic with extra fitting time; not an end-to-end search speed result. Postmortem follows separately.')
        notify('finish','\n'.join(lines))
    except BaseException:
        notify('finish','Secant STOPPED WITH ERROR\nCoefficient rescue diagnostic\nInspect '+str(root/'run.log')+'; no complete rescue result claimed.')
        status=1
        raise
    finally:
        (root/'worker-exit.txt').write_text(str(status)+'\n')
