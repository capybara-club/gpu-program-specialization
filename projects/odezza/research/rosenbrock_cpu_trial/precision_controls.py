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
"""CPU precision checks on truly stiff systems, separate from benchmark samples."""
import argparse
import math
from pathlib import Path
import statistics
from solver import Solver,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cases=[dict(name='decay-1e6',states=['x','y'],rhs=['-1000000*x','-y'],ic=[1.,2.],
                times=[0.]+[10**(-8+i*8/39) for i in range(40)]),
           dict(name='Robertson-1e4',states=['x','y','z'],rhs=['-0.04*x+10000*y*z','0.04*x-10000*y*z-30000000*y*y','30000000*y*y'],
                ic=[1.,0.,0.],times=[0.]+[10**(-6+i*10/34) for i in range(35)])]
    results=[]
    for c in cases:
        solvers={pr:Solver(c['rhs'],c['states'],precision=pr) for pr in ('FP32','FP64')}
        row=dict(times=c['times'],initial_state=c['ic'],states=[[0.]*len(c['ic']) for _ in c['times']])
        if c['name']=='decay-1e6':
            reference=[[math.exp(-1e6*t),2*math.exp(-t)] for t in c['times']];kind='analytic solution'
        else:
            ref=solvers['FP64'].run([row],rtol=1e-10,atol=1e-14)
            assert ref['complete'];reference=ref['predictions'][0]
            kind='native FP64 Rosenbrock23 at 1e-10/1e-14, previously validated against independent Radau'
        row['states']=reference;settings=[]
        for tol in (1e-4,1e-5,1e-6,1e-7):
            for pr,s in solvers.items():
                r=s.run([row],rtol=tol,atol=tol*1e-4,max_attempts=500000)
                pred=r.pop('predictions')[0]
                if r['complete']:
                    r['max_abs_reference_error']=max(abs(v-q) for vv,qq in zip(pred,reference) for v,q in zip(vv,qq))
                    if len(c['ic'])==3:r['mass_conservation_max_abs']=max(abs(sum(v)-1.) for v in pred)
                settings.append(r)
        results.append(dict(name=c['name'],reference=kind,settings=settings))
    write(a.out,dict(purpose=__doc__,results=results))


if __name__=='__main__':main()
