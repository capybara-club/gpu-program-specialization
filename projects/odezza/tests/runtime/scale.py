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
"""One million complete native RHS-vector combinations scored without an intermediate AST JSON stream."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from odezza.grammar.native_service import NativeService
ROOT=Path(__file__).resolve().parents[2]
g=json.loads((ROOT/'examples/grammar/million.json').read_text())
g['constants']={'c':{'values':[.25+i/32 for i in range(32)]}}
g['rhs']['x0']='const.c*('+g['rhs']['x0']+')'
g['limits']['max_configurations']=32000000
g['limits']['max_seconds']=120
p=dict(states=g['states'],trajectories=[dict(initial=[.1]*6,times=[0,.01,.03],values=[[.1*math.exp(t)]*6 for t in [0,.01,.03]])])
s=NativeService('/tmp/odezza-runtime-cache')
try:
 start=time.monotonic();job=s.submit(problem=p,grammar=g,execution={'batch_variants':1024,'module_systems':32,'dedup_bytes_per_family':134217728,'max_seconds':120})
 while not s.jobs[job['job_id']].done():
  status=s.status(job['job_id']);print(status['status'],status['counts'],flush=True);time.sleep(1)
 report=s.jobs[job['job_id']].result();report['wall_seconds']=time.monotonic()-start
 if report['leaderboards']['global']:
  report['best_replay']=s.replay(job['job_id'],report['leaderboards']['global'][0],cpu=True)
 Path('/tmp/odezza-runtime-scale.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report['status'],report['error'],report['counts'],report['timing'],flush=True)
 assert report['status']=='complete' and sum(f['generated_asts'] for f in report['families'])==1000000
 assert report['counts']['completed_configurations']==32000000
finally:s.close()
