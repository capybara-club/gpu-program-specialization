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
"""Native C99 cooperative LM: toggle identity, bounds, ragged data and fallback.

Run on a CUDA host against the built library. No CUDA source is generated here.
"""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scratch/fitting_batch_trial'))
import bootstrap
import odezza
odezza.__path__.insert(0, str(ROOT / 'python/odezza'))
from odezza.native import AstProgram, LmFit, LmPipeline, LmShape, NativeError
from pipeline_runner.cuda import CudaOwner


def run(owner, library, n, p, width, fallback):
    stream = owner.stream_pool(1).streams[0]
    event = owner.driver.event_create()
    targets = [-.3-.05*i for i in range(p)]
    starts = [[-.1-.015*((i+j)%40) for j in range(p)] for i in range(17)]
    offsets = [0, 7, 16]
    times = [i*.05 for i in range(7)]+[i*.07 for i in range(9)]
    ref = []
    for s in range(n):
        rate = targets[s] if s < p else -.7
        ref.extend([(1.+.1*s)*math.exp(rate*t) for t in times[:7]])
        ref.extend([(1.2+.1*s)*math.exp(rate*t) for t in times[7:]])
    def literal(f): return bytes([0x83])+struct.pack('<f', f)
    codes = []
    for s in range(n):
        # Four permutations with a known, distinct optimum for each coefficient.
        coefficient = bytes([0x82,s]) if s < p else literal(-.7)
        scale = literal(1)+literal(2)+literal(3)+literal(4)+bytes([0x85,0,1])
        codes.append(bytes([0x81,s])+coefficient+bytes([0x92])+scale+bytes([0x92,0x80]))
    buffers = []
    def buffer(t, count, values=None):
        b = owner.buffer(t, count, values); buffers.append(b); return b
    ins = [buffer(C.c_float,17*p,[x for row in starts for x in row]),
           buffer(C.c_uint32,3,offsets), buffer(C.c_float,16,times),
           buffer(C.c_float,n*16,ref), buffer(C.c_float,n*16,[1.]*(n*16)),
           buffer(C.c_float,p,[-3.]*p), buffer(C.c_float,p,[0.]*p)]
    total = 17*4
    outs = [buffer(C.c_float,total*p),buffer(C.c_float,total),buffer(C.c_float,total),
            *[buffer(C.c_uint32,total) for _ in range(5)]]
    strings = [C.create_string_buffer(x) for x in codes]
    programs = (AstProgram*n)(*[AstProgram(C.addressof(s),len(s)-1) for s in strings])
    fit = LmFit(rhs=programs,start_count=17,toggle_bit_count=2,trajectory_count=2,point_count=16,
                steps_per_interval=4,max_iterations=32,max_damping_attempts=8,
                initial_damping=.001,max_step=1,target_mse=1e-13)
    for name,b in zip(('starts_device','offsets_device','times_device','reference_device','weights_device','lower_device','upper_device'),ins):setattr(fit,name,b[0].value)
    for name,b in zip(('parameters_device','initial_mse_device','mse_device','iterations_device','accepted_device','factorizations_device','evaluations_device','invalid_device'),outs):setattr(fit,name,b[0].value)
    try:
        with LmPipeline(library,int(owner.gpu.arch.split('_')[1]),LmShape(n,p,256,width),fallback_lanes=fallback) as pipeline:
            for b in ins:owner.driver.upload(b,stream)
            owner.driver.event_record(event,stream);fit.input_ready_event=event.value
            report=pipeline.run([fit])
            for b in outs:owner.driver.download(b,stream)
            owner.driver.event_record(event,stream);owner.driver.event_wait(event)
            co,initial,mse,iterations,accepted,factors,evals,invalid = [list(b[2]) for b in outs]
            # Other fixed RHSs are also scaled: only compare exact coefficients
            # when all state coefficients are fitted. All rows are replayed.
            worst = 0.
            for i in range(total):
                scale=1+i%4; row=co[i*p:(i+1)*p]
                error=0.; residuals=0
                for s in range(n):
                    rate=scale*(row[s] if s<p else -.7)
                    for begin,end in zip(offsets,offsets[1:]):
                        state=ref[s*16+begin]
                        for point in range(begin+1,end):
                            h=(times[point]-times[point-1])/4
                            # Independent double precision RK4 for a linear RHS.
                            for _ in range(4):
                                k1=rate*state;k2=rate*(state+h*k1/2);k3=rate*(state+h*k2/2);k4=rate*(state+h*k3)
                                state+=h*(k1+2*k2+2*k3+k4)/6
                            error+=(state-ref[s*16+point])**2;residuals+=1
                replay=error/residuals
                assert math.isclose(replay,mse[i],rel_tol=3e-4,abs_tol=3e-10),(n,p,width,i,replay,mse[i])
                worst=max(worst,abs(replay-mse[i]))
                assert all(-3<=v<=0 for v in row)
                if n==p or scale==1:
                    assert mse[i]<2e-10,(n,p,width,i,mse[i],row,iterations[i],accepted[i],factors[i],evals[i])
                    assert max(abs(v-targets[j]/scale) for j,v in enumerate(row[:min(n,p)]))<2e-4,(row,targets,scale)
            assert report['fit_count']==total and report['completed_system_count']==1
            # Invalid AST must not be treated as a register fallback.
            saved=programs[0].byte_count;programs[0].byte_count=1
            try:pipeline.run([fit]);raise AssertionError('Invalid AST accepted')
            except NativeError as e:assert e.result!=12
            programs[0].byte_count=saved
            pipeline.run([fit])  # Same handle remains usable after rejection.
            return dict(states=n,parameters=p,width=width,fallback=list(fallback),report=report,
                        max_cpu_mse_difference=worst,arrays=dict(coefficients=co,initial_mse=initial,mse=mse,
                        iterations=iterations,accepted=accepted,factorizations=factors,evaluations=evals,invalid=invalid))
    finally:
        owner.driver.event_destroy(event)
        for b in reversed(buffers):
            for name,value in [('cuMemFree_v2',b[0]),('cuMemFreeHost',b[1])]:
                owner.gpu.call(name,value);owner.gpu.resources.remove((name,value))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--library',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();rows=[]
    with CudaOwner(0) as owner:
        for n,p in [(3,3),(3,6),(6,6),(8,6),(8,8)]:
            for width in (1,2,4,8):
                try:row=run(owner,args.library,n,p,width,())
                except NativeError as e:
                    assert e.result==12,(n,p,width,str(e))
                    row=dict(states=n,parameters=p,width=width,status='register_pressure')
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('arrays','report')}),flush=True)
            rows.append(run(owner,args.library,n,p,1,(2,4,8)))
            args.out.write_text(json.dumps(dict(arch=owner.gpu.arch,rows=rows),indent=2)+'\n')
        assert any(r.get('report',{}).get('shapes',{}).get('used_lanes_mask',1)>1 for r in rows if r.get('fallback'))
    print('All available shapes and automatic fallbacks passed independent replay')

if __name__=='__main__':main()
