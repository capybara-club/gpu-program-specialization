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
import importlib.util,json,sys
from pathlib import Path
from odezza.grammar.native_service import NativeService
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fixtures',ROOT/'tests/grammar/gpu_examples.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
g=json.loads((ROOT/'examples/grammar/adaptation_search.json').read_text());problem=f.fixture(g)
s=NativeService('/tmp/odezza-runtime-cache')
try:
 j=s.submit(problem=problem,grammar=g,execution={'module_systems':32,'patch_capacity':1024})
 r=s.jobs[j['job_id']].result();Path('/tmp/odezza-runtime-adaptation.json').write_text(json.dumps(r,indent=2));print(r['status'],r['error'],r['counts'],r['timing'])
 if r['status']=='complete':
  for k in r['leaderboards']['global'][:3]:
   v=s.replay(j['job_id'],k,cpu=True);print(v['candidate']['mse'],v['cpu_reference'])
finally:s.close()
