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
"""Measure occupancy limits and throughput scaling using already compiled ASTs.

The existing 1,024-start bank is tiled, preserving the exact mix of LM work.
No generation, NVRTC, module loading or copies enter kernel-event timing.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_runner import Pipeline
from pipeline_runner.cuda import CudaOwner
from pipeline_runner.occupancy import launch_metrics,plateau
from workload import PreparedFit

FIELDS=('coefficients','initial_mse','mse','iterations','accepted','attempts')

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def sha(data):return hashlib.sha256(data).hexdigest()
def telemetry():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True)
    return dict(returncode=r.returncode,output=r.stdout.strip(),error=r.stderr.strip())

def repeat_input(raw,count,path):
    header=list(struct.unpack_from('<7I',raw));base=header[0];p=header[6]
    if count<base or count%base:raise ValueError('Population must be an integer multiple of the original bank')
    starts_size=4*base*p;header[0]=count
    path.write_bytes(struct.pack('<7I',*header)+raw[28:-starts_size]+raw[-starts_size:]*(count//base))
    return base

def output_bytes(result):
    return [struct.pack('<'+('f' if i<3 else 'I')*len(result[k]),*result[k]) for i,k in enumerate(FIELDS)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',nargs='+');p.add_argument('--counts',type=int,nargs='+',default=[1024,8192,32768,65536,131072,262144])
    p.add_argument('--widths',type=int,nargs='+',default=[1,2,4,8]);p.add_argument('--repeats',type=int,default=3);p.add_argument('--device',type=int,default=0)
    p.add_argument('--profile-one',action='store_true',help='Exactly one measured launch after loading: for a separate profiler invocation')
    a=p.parse_args()
    if a.repeats<3 and not a.profile_one:raise ValueError('At least three repeats required for scaling evidence')
    if sorted(set(a.counts))!=a.counts:raise ValueError('Counts must be unique and increasing')
    a.out.mkdir(parents=True,exist_ok=False);allrows=[];summaries=[]
    original=json.loads((a.grid/'results.json').read_text())
    cases=a.cases or list(dict.fromkeys(r['case'] for r in original))
    start=time.monotonic()
    manifest=dict(cases=cases,counts=a.counts,widths=a.widths,repeats=a.repeats,device=a.device,
                  workload='Repeated original bank; logical fits are repeated work, not additional unique configurations',
                  timing='CUDA kernel events only, serial launches after preparation; three samples per population',
                  plateau_rule=dict(rate_range_fraction=.05,repeat_spread_fraction=.05,minimum_resident_waves=4,consecutive_populations=3),
                  achieved_occupancy=None,profiler_status='rack1 hardware counters denied: ERR_NVGPUCTRPERM',
                  pre_run_telemetry=telemetry())
    save(a.out/'manifest.json',manifest)
    for name in cases:
        folder=a.grid/name;destination=a.out/name;destination.mkdir()
        raw=(folder/'input.bin').read_bytes();base=struct.unpack_from('<I',raw)[0]
        baseline=json.loads((folder/'gpu-w1.json').read_text());baseline_bytes=output_bytes(baseline)
        case_rows=[];functions={};resources={};expected={};compatible={};completed=set()
        with CudaOwner(a.device) as owner:
            identity=owner.driver.identity();save(destination/'device.json',identity)
            if identity!=json.loads((a.grid/'device.json').read_text()):raise ValueError('Device or driver changed from validated grid')
            pool=owner.stream_pool(1)
            for width in a.widths:
                result=json.loads((folder/f'gpu-w{width}.json').read_text());compiler=json.loads((folder/f'compiler-w{width}.json').read_text())
                cubin=(folder/f'kernel-w{width}.cubin').read_bytes();_,function=owner.module(cubin);functions[width]=function
                expected[width]=output_bytes(result);compatible[width]=expected[width]==baseline_bytes
                resources[width]=owner.driver.resources(function,result['shared_bytes'],compiler)
                prior=next(r for r in original if r['case']==name and r['width']==width)
                if not all(c['passed'] for c in prior['checks'].values()) or not result['derivative_probe']['passed']:
                    raise ValueError('Original numerical validation failed')
                save(destination/f'provenance-w{width}.json',dict(input_sha256=sha(raw),cubin_sha256=sha(cubin),compiler=compiler,resources=resources[width],equivalent_to_one_lane=compatible[width]))
            with Pipeline(owner.driver,streams=pool) as runner:
                if not a.profile_one:
                    for width in a.widths:
                        warm=PreparedFit(owner,functions[width],folder/'input.bin',width)
                        try:runner.wait(runner.submit(warm.launch))
                        finally:warm.close()
                for count in a.counts:
                    input_path=destination/f'input-{count}.bin';repeat_input(raw,count,input_path)
                    order=[w for w in a.widths if w not in completed];random.Random(count).shuffle(order)
                    for width in order:
                        fit=PreparedFit(owner,functions[width],input_path,width)
                        try:
                            seconds=[]
                            before=telemetry()
                            for _ in range(1 if a.profile_one else a.repeats):
                                ticket=runner.submit(fit.launch);runner.wait(ticket);seconds.append(ticket.kernel_seconds)
                            outputs=[bytes(b[2]) for b in fit.outputs]
                            exact=outputs==[b*(count//base) for b in expected[width]]
                            row=dict(case=name,count=count,width=width,base_count=base,distinct_start_rows=base,
                                     kernel_seconds=seconds,median_seconds=statistics.median(seconds),
                                     geometry=launch_metrics(count,width,resources[width],statistics.median(seconds)),
                                     resources=resources[width],exact_tiled_outputs=exact,equivalent_to_one_lane=compatible[width],
                                     output_sha256=[sha(b) for b in outputs],before_telemetry=before,after_telemetry=telemetry())
                            case_rows.append(row);allrows.append(row);save(a.out/'results.json',allrows)
                            if not exact:raise RuntimeError(f'Tiled output mismatch: {name}, width {width}, count {count}')
                            evidence=plateau([r for r in case_rows if r['width']==width]);row['plateau']=evidence
                            if evidence['confirmed']:completed.add(width)
                            save(a.out/'results.json',allrows)
                            print(json.dumps(dict(case=name,width=width,count=count,ms=row['median_seconds']*1000,
                                                  fits_per_second=row['geometry']['fits_per_second'],waves=row['geometry']['resident_waves'],plateau=evidence['confirmed'])),flush=True)
                        finally:fit.close()
                    if len(completed)==len(a.widths):break
            for width in a.widths:
                rows=[r for r in case_rows if r['width']==width];last=rows[-1]
                summaries.append(dict(case=name,width=width,count=last['count'],plateau=plateau(rows),
                                      equivalent_to_one_lane=compatible[width],resources=resources[width],geometry=last['geometry']))
            save(a.out/'summary.json',summaries)
    save(a.out/'complete.json',dict(complete=True,cases=len(cases),cells=len(allrows),seconds=time.monotonic()-start,
                                   shapes_with_plateau=sum(s['plateau']['confirmed'] for s in summaries),shapes=len(summaries)))

if __name__=='__main__':main()
