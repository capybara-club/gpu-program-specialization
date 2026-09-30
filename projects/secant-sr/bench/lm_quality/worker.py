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
"""mac1 tmux worker: SSH experiment, retain exit state, then copy its artifacts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

def copy_results(host,remote,local):
    # Use existing SSH/tar; rack1 does not have rsync installed.
    with tempfile.TemporaryFile() as archive:
        code=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
            shlex.join(['tar','-cf','-','-C',remote,'.'])],stdout=archive).returncode
        if code:return code
        archive.seek(0)
        with tarfile.open(fileobj=archive,mode='r:') as t:
            for m in t.getmembers():
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts or not (m.isfile() or m.isdir()):
                    raise ValueError('unsafe artifact member')
            t.extractall(local)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    a=p.parse_args();launch=json.loads(a.launch.read_text())
    w=next(w for w in launch['workers'] if w['shard']==a.shard)
    local=a.launch.resolve().parent/f'shard{a.shard}';local.mkdir(parents=True,exist_ok=True)
    if (local/'worker-exit.txt').exists(): raise RuntimeError('worker already finished; use a new run directory')
    remote=launch['remote'];out=f"{remote}/{launch['directory']}/shard{a.shard}"
    command=['python3',f'{remote}/secant-sr/bench/lm_quality/campaign.py',
        '--manifest',launch['manifest'],'--executable',f'{remote}/bin/search','--output',out,
        '--gpu',str(w['gpu']),'--shard',str(a.shard),'--shards',str(len(launch['workers'])),
        '--seconds',str(launch['seconds']),'--repeats',str(launch['repeats']),
        '--iterations',str(launch.get('iterations',4)),'--budget',str(launch.get('budget',512))]
    if launch.get('problems'): command+=['--problems',*launch['problems']]
    code=127
    try:
        with (local/'run.log').open('w') as log:
            code=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',w['host'],shlex.join(command)],stdout=log,stderr=subprocess.STDOUT).returncode
    finally:
        (local/'worker-exit.txt').write_text(str(code)+'\n')
        copy=127
        try:copy=copy_results(w['host'],out,local)
        finally:(local/'copy-exit.txt').write_text(str(copy)+'\n')
    return code or copy
if __name__=='__main__':sys.exit(main())
