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
"""True stiff controls plus duplicated-work GPU scaling; not a search benchmark."""
import argparse
import math
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from prototype import Solver
from audit import write


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=0); a=p.parse_args()
    tt=np.r_[0.,np.geomspace(1e-6,1e4,35)]
    def robertson(t,y):
        x,b,z=y
        return [-.04*x+1e4*b*z,.04*x-1e4*b*z-3e7*b*b,3e7*b*b]
    cases=[dict(name='million-to-one-decay',states=['x','y'],rhs=['-1000000*x','-y'],
                ic=[1.,2.],times=np.r_[0.,np.geomspace(1e-8,1.,40)],f=lambda t,y: [-1e6*y[0],-y[1]]),
           dict(name='Robertson-1e4',states=['x','y','z'],
                rhs=['-0.04*x+10000*y*z','0.04*x-10000*y*z-30000000*y*y','30000000*y*y'],
                ic=[1.,0.,0.],times=tt,f=robertson)]
    records=[]
    for case in cases:
        solver=Solver(case['rhs'],case['states']); started=time.perf_counter()
        ref=solve_ivp(case['f'],(case['times'][0],case['times'][-1]),case['ic'],
                      method='Radau',rtol=1e-11,atol=1e-14,t_eval=case['times'])
        reference_seconds=time.perf_counter()-started
        if not ref.success: raise RuntimeError(ref.message)
        row=dict(times=case['times'].tolist(),initial_state=case['ic'],states=ref.y.T.tolist())
        settings=[]
        for tol in (1e-6,1e-8,1e-10):
            run=solver.run([row],rtol=tol,atol=tol*1e-4,device=a.device)
            pred=np.asarray(run.pop('predictions')[0]); run['max_abs_vs_radau']=float(abs(pred-ref.y.T).max())
            if case['name']=='million-to-one-decay':
                exact=np.array([[math.exp(-1e6*t),2*math.exp(-t)] for t in case['times']])
                run['max_abs_vs_analytic']=float(abs(pred-exact).max())
                # RK4's negative-real stability interval is approximately [-2.785,0].
                run['rk4_stability_minimum_steps_over_horizon']=math.ceil(1e6/2.785)
            settings.append(run)
        scale=[]
        # Same trajectories, coefficients, method and tolerance at every size.
        for copies in (1,128,4096):
            measurements=[]
            for repeat in range(3):
                run=solver.run([row],rtol=1e-6,atol=1e-10,device=a.device,repetitions=copies,predictions=False)
                if any(s['status'] for s in run['stats']): raise RuntimeError('scaling integration failure')
                measurements.append(dict(kernel_ms=run['kernel_ms'],wall_seconds=run['wall_seconds'],
                                         transfer_interval_ms=run['transfer_interval_ms']))
            scale.append(dict(repeated_lanes=copies,unique_trajectories=1,unique_models=1,
                              unique_configurations=1,measurements=measurements,stats_per_lane=run['stats'][0]))
        records.append(dict(name=case['name'],reference_method='Radau',reference_seconds=reference_seconds,
                            nfev=ref.nfev,njev=ref.njev,nlu=ref.nlu,settings=settings,scaling=scale))
    write(a.out,dict(purpose=__doc__,results=records))


if __name__=='__main__': main()
