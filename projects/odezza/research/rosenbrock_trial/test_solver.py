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
import math
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from prototype import Solver
from codegen import model


def row(times, initial, observations=None):
    return dict(times=list(times),initial_state=initial,
                states=observations if observations is not None else [[0.]*len(initial) for _ in times])


class SolverTests(unittest.TestCase):
    def test_restricted_language(self):
        for e in ('__import__("os")', 'x[0]', 'x.real', 'unknown(x)', 'x**2', '1e999'):
            with self.assertRaises(ValueError): model([e],['x'])

    def test_jacobian(self):
        solver=Solver(['sin(x*y)+exp(z)/(1+x*x)','cos(x-z)-tanh(y)','x*y*z'],['x','y','z'])
        y=np.array([.7,-.4,.2]); rhs,j=solver.evaluate(y)
        strided=np.array([.7,9.,-.4,9.,.2,9.])[::2]
        np.testing.assert_array_equal(solver.evaluate(strided)[0],rhs)
        for k in range(3):
            v=np.zeros(3); v[k]=1e-5
            numeric=(solver.evaluate(y+v)[0]-solver.evaluate(y-v)[0])/(2e-5)
            np.testing.assert_allclose(j[:,k],numeric,rtol=1e-8,atol=1e-9)

    def test_stiff_decay_mask_and_cpu_parity(self):
        solver=Solver(['-1000000*x','-y'],['x','y'])
        tt=[0,1e-6,1e-4,.1,1.]
        truth=[[math.exp(-1e6*t),2*math.exp(-t)] for t in tt]
        obs=[[None,v[1]] for v in truth]; rr=[row(tt,[1.,2.],obs)]
        gpu=solver.run(rr,rtol=1e-8,atol=1e-11,h0=.1)
        cpu=solver.run(rr,rtol=1e-8,atol=1e-11,h0=.1,cpu=True)
        self.assertEqual(gpu['stats'][0]['status'],0)
        self.assertGreater(gpu['stats'][0]['rejected'],0)
        self.assertEqual(gpu['stats'][0]['count'],4)
        np.testing.assert_allclose(gpu['predictions'][0],truth,atol=3e-6,rtol=3e-6)
        np.testing.assert_allclose(gpu['predictions'],cpu['predictions'],atol=1e-10,rtol=1e-8)
        expected=np.mean([(v[1]-q[1])**2 for v,q in zip(gpu['predictions'][0][1:],truth[1:])])
        self.assertAlmostEqual(gpu['stats'][0]['mse'],expected,places=20)

    def test_coupling_pivot_and_refinement(self):
        # Initial A requires a row pivot; a diagonal-only solve would be wrong.
        solver=Solver(['-x+100*y','-100*x-y'],['x','y'])
        tt=np.array([0.,.1]); rr=[row(tt,[1.,0.])]
        expected=np.array([[math.exp(-t)*math.cos(100*t),-math.exp(-t)*math.sin(100*t)] for t in tt])
        errors=[]
        for tol in (1e-5,1e-8):
            run=solver.run(rr,rtol=tol,atol=tol*.01,h0=.1)
            self.assertEqual(run['stats'][0]['status'],0)
            errors.append(np.max(np.abs(np.asarray(run['predictions'][0])-expected)))
        self.assertLess(errors[1],errors[0]/20)
        self.assertLess(errors[1],1e-5)

    def test_robertson(self):
        rhs=['-0.04*x+10000*y*z','0.04*x-10000*y*z-30000000*y*y','30000000*y*y']
        solver=Solver(rhs,['x','y','z']); tt=np.r_[0.,np.geomspace(1e-6,1e4,35)]
        # Independent RHS, not our code generator, in the reference solve.
        def robertson(t,y):
            x,b,z=y
            return [-.04*x+1e4*b*z,.04*x-1e4*b*z-3e7*b*b,3e7*b*b]
        reference=solve_ivp(robertson,(0,1e4),[1.,0.,0.],
            method='Radau',rtol=1e-11,atol=1e-14,t_eval=tt)
        self.assertTrue(reference.success)
        run=solver.run([row(tt,[1.,0.,0.])],rtol=1e-8,atol=1e-12)
        self.assertEqual(run['stats'][0]['status'],0)
        pred=np.array(run['predictions'][0])
        np.testing.assert_allclose(pred,reference.y.T,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(pred.sum(axis=1),1.,atol=1e-10)

    def test_limits_domains_and_validation(self):
        solver=Solver(['-x'],['x']); rr=[row([0.,1.],[1.])]
        run=solver.run(rr,max_attempts=1)
        self.assertEqual(run['stats'][0]['status'],1)
        self.assertTrue(math.isinf(run['stats'][0]['mse']))
        self.assertTrue(math.isnan(run['predictions'][0][-1][0]))
        with self.assertRaises(ValueError): solver.run([row([0,0],[1.])])
        with self.assertRaises(ValueError): solver.run(rr,rtol=0)
        invalid=Solver(['1/x'],['x']).run([row([0.,1.],[0.])])
        self.assertEqual(invalid['stats'][0]['status'],3)

    def test_score_only_repeated_lanes(self):
        solver=Solver(['-x'],['x'])
        rr=[row([0,.2,.5],[1.]),row([0,.1,.8],[2.])]
        run=solver.run(rr,repetitions=129,predictions=False)
        self.assertIsNone(run['predictions'])
        self.assertEqual(len(run['stats']),258)
        for i,s in enumerate(run['stats']):
            self.assertEqual(s,run['stats'][i%2])


if __name__=='__main__': unittest.main(verbosity=2)
