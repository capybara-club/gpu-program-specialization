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
"""Benchmark adapter using explicit caller-owned streams and contained execution."""
import ctypes as C
import math
from pathlib import Path
import sys
import time
import statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_runner import Pipeline,Launch
from pipeline_runner.cuda import compile_cuda
from pipeline_runner.occupancy import launch_metrics
from workload import PreparedFit

def run(source,folder,width,probe,*,owner,streams):
    cubin,compiler=compile_cuda(source,owner.gpu.arch,resource_diagnostics=True)
    (folder/f'kernel-w{width}.cubin').write_bytes(cubin)
    t=time.perf_counter();module,function=owner.module(cubin);load_seconds=time.perf_counter()-t
    fit=PreparedFit(owner,function,folder/'input.bin',width)
    values,expected=probe;pi=owner.buffer(C.c_float,len(values),values);po=owner.buffer(C.c_float,len(expected))
    probe_function=owner.function(module,'model_jacobian');directions=len(expected)//(2*fit.states)
    probe_job=Launch(probe_function,((directions+31)//32,1,1),(32,1,1),0,(pi[0],po[0]),
                     uploads=(pi,),downloads=(po,),keepalive=(owner,pi,po),writes=(pi[0].value,po[0].value),label='derivative probe')
    with Pipeline(owner.driver,streams=streams) as pipeline:
        pipeline.wait(pipeline.submit(probe_job))
        normalized=max(abs(x-y)/(1+abs(y)) if math.isfinite(x) and math.isfinite(y) else math.inf for x,y in zip(po[2],expected))
        if normalized>1e-4:raise ValueError(f'Analytic derivative/primal probe failed: {normalized}')
        ms=[];wall=[]
        # Serial repetitions intentionally measure single-launch latency.
        # Concurrent submission is exercised separately using the same runner.
        for _ in range(3):
            t=time.perf_counter();ticket=pipeline.submit(fit.launch);pipeline.wait(ticket)
            ms.append(ticket.kernel_seconds*1000);wall.append(time.perf_counter()-t)
    resources=owner.driver.resources(function,fit.shared,compiler)
    return dict(width=width,states=fit.states,parameters=fit.parameters,count=fit.count,
                registers=resources['registers_per_thread'],local_bytes=resources['local_bytes_per_thread'],
                shared_bytes=fit.shared,active_blocks_per_sm=resources['active_blocks_per_sm'],
                device=owner.driver.identity(),resources=resources,kernel_ms=ms,launch_wait_seconds=wall,module_load_seconds=load_seconds,
                occupancy_geometry=launch_metrics(fit.count,width,resources,statistics.median(ms)/1000),
                derivative_probe=dict(passed=True,values=len(expected),maximum_normalized_error=normalized,tolerance=1e-4),**fit.values()),compiler
