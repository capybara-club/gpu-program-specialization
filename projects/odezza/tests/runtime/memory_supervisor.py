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
"""Process guard checks, including a real disposable child on Linux."""
import importlib.util
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

path=Path(__file__).resolve().parents[2]/'service_trial/supervise_worker.py'
spec=importlib.util.spec_from_file_location('supervise_worker',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Child:
    pid=123
    code=None
    killed=False
    def poll(self): return self.code
    def kill(self): self.killed=True;self.code=-9
    def wait(self,timeout=None):
        if self.code is None:self.code=0
        return self.code


records=[]
for rss,reason in [(4096,None),(8193,'rss_limit'),(None,'rss_unobservable')]:
    c=Child()
    with redirect_stdout(sys.stderr), patch.object(module,'resident_bytes',return_value=rss):
        code,peak,actual=module.wait_worker(c,8192)
    assert actual==reason and c.killed==(reason is not None)
    records.append(dict(test='mock',rss=rss,reason=reason,code=code,peak=peak))
if Path('/proc/self/statm').exists():
    c=subprocess.Popen([sys.executable,'-c','import time; a=bytearray(32*1024*1024); time.sleep(10)'],stdout=subprocess.DEVNULL)
    begin=time.monotonic()
    try:
        with redirect_stdout(sys.stderr):
            code,peak,reason=module.wait_worker(c,24<<20)
        elapsed=time.monotonic()-begin
        assert reason=='rss_limit' and code!=0 and peak>24<<20 and elapsed<5
        records.append(dict(test='real_child',reason=reason,peak=peak,elapsed_seconds=elapsed))
    finally:
        if c.poll() is None:c.kill();c.wait()
print(json.dumps(dict(passed=True,records=records)))
