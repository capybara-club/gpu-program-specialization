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
"""Replay a retained failed preparation in a separate GPU process and output."""
import argparse
from pathlib import Path
import time
from benchmarks.lm_tuning.common import load, save, file_hash
from benchmarks.lm_tuning.run_trial import Engine, Control, REPO, rf, cs, verify
from lm_toggle.native_service import Adapter
from lm_toggle.service import prepare_groups

def run(request, output, device):
    output.mkdir(parents=True,exist_ok=False)
    spec=load(request);settings=spec.pop('lm_dispatch');spec['execution']['device']=device
    groups,fixed=prepare_groups(rf.validate(spec),settings,'cpu-regression')
    engine=Engine(REPO/'scratch/fitting_batch_trial/bin/odezza-fit-core-run',[device])
    def execute():
        adapter=Adapter(device);engine.lm_adapters[device]=adapter
        return adapter.run(spec,output/'native',settings,time.monotonic()+30,Control(),engine.backend,engine.workers[device])
    try:
        before=time.monotonic();report=engine.pools[device].submit(execute).result();elapsed=time.monotonic()-before
        checks=[];context={k:spec[k] for k in cs.CONTEXT_FIELDS}
        for row in report['candidates']:
            if row['mse'] is None:continue
            cpu=verify(context,row['program_hex'],row['parameter_values'],128,deadline=time.monotonic()+10)
            matched=cpu['status']=='complete' and abs(cpu['mse']-row['mse'])<=max(2e-8,.005*abs(cpu['mse']))
            checks.append(dict(candidate_id=row['candidate_id'],gpu_mse=row['mse'],cpu=cpu,matched=matched))
        passed=report['work_complete'] and report['counts']['invalid_candidate_asts']>0 and bool(checks) and all(c['matched'] for c in checks)
        result=dict(passed=passed,request_sha256=file_hash(request),cpu_groups=len(groups),cpu_fixed=len(fixed),
            status=report['status'],work_complete=report['work_complete'],counts=report['counts'],seconds=elapsed,
            checks=checks,lm_profile=report['lm_profile'],protocol='Original failed fitting request; independent diagnostic 30-second drain budget; no search or held-out data.')
        save(output/'result.json',result)
        if not passed:raise RuntimeError('GPU regression did not pass; retain evidence')
    finally:engine.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args()
    run(a.request,a.output,a.device)
