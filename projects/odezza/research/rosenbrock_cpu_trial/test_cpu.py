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
import struct
from solver import Solver


def row(times,ic,states=None):
    return dict(times=times,initial_state=ic,states=states or [[0.]*len(ic) for _ in times])


class Tests(unittest.TestCase):
    def test_fp32_arithmetic_and_precision(self):
        rhs=['(16777216+x)-16777216']
        f=Solver(rhs,['x'],precision='FP32');d=Solver(rhs,['x'])
        self.assertEqual(f.evaluate([1.])[0],[0.])
        self.assertEqual(d.evaluate([1.])[0],[1.])
        with self.assertRaises(ValueError):Solver(['1e100'],['x'],precision='FP32')
        with self.assertRaises(ValueError):Solver(['x'],['x'],precision='unknown')
        f=Solver(['-1000000*x','-y'],['x','y'],precision='FP32')
        tt=[0.,1e-6,1e-4,.1,1.];obs=[[None,2*math.exp(-t)] for t in tt]
        r=f.run([row(tt,[1.,2.],obs)],rtol=1e-5,atol=1e-8)
        self.assertTrue(r['complete']);self.assertEqual(r['stats'][0]['count'],4)
        self.assertLess(r['stats'][0]['mse'],1e-7)
        for v in r['predictions'][0]:
            for x in v:self.assertEqual(x,struct.unpack('f',struct.pack('f',x))[0])
        error=sum((v[1]-o[1])**2 for v,o in zip(r['predictions'][0][1:],obs[1:]))/4
        self.assertAlmostEqual(error,r['stats'][0]['mse'],places=20)

    def test_fp32_rk4_and_clock(self):
        f=Solver(['-x'],['x'],precision='FP32')
        r=f.run([row([1e8,1e8+.25,1e8+1.],[1.])],method='RK4',substeps=32)
        self.assertTrue(r['complete'])
        self.assertAlmostEqual(r['predictions'][0][-1][0],math.exp(-1),delta=1e-6)
    def test_cpu_stiff_and_mask(self):
        solver=Solver(['-1000000*x','-y'],['x','y'])
        tt=[0.,1e-6,1e-4,.1,1.]
        obs=[[None,2*math.exp(-t)] for t in tt]
        run=solver.run([row(tt,[1.,2.],obs)],rtol=1e-8,atol=1e-11,h0=.1)
        self.assertTrue(run['complete']);self.assertEqual(run['stats'][0]['count'],4)
        self.assertGreater(run['stats'][0]['rejected'],0)
        expected=sum((v[1]-q[1])**2 for v,q in zip(run['predictions'][0][1:],obs[1:]))/4
        self.assertAlmostEqual(run['stats'][0]['mse'],expected,places=20)
        for t,v in zip(tt,run['predictions'][0]):
            self.assertAlmostEqual(v[0],math.exp(-1e6*t),delta=3e-6)
            self.assertAlmostEqual(v[1],2*math.exp(-t),delta=3e-6)

    def test_rk4_fourth_order(self):
        solver=Solver(['-x'],['x']);errors=[]
        for n in (8,16):
            r=solver.run([row([0.,1.],[1.])],method='RK4',substeps=n)
            self.assertTrue(r['complete']);self.assertEqual(r['stats'][0]['rhs'],4*n)
            errors.append(abs(r['predictions'][0][-1][0]-math.exp(-1.)))
        self.assertGreater(errors[0]/errors[1],14)
        self.assertLess(errors[0]/errors[1],19)

    def test_jacobian(self):
        s=Solver(['sin(x*y)+exp(z)/(1+x*x)','cos(x-z)-tanh(y)','x*y*z'],['x','y','z'])
        y=[.7,-.4,.2];_,j=s.evaluate(y)
        for k in range(3):
            a=y[:];b=y[:];a[k]+=1e-5;b[k]-=1e-5
            aa,_=s.evaluate(a);bb,_=s.evaluate(b)
            for i in range(3):self.assertAlmostEqual(j[i][k],(aa[i]-bb[i])/2e-5,delta=1e-8)

    def test_failure_and_invalid_input(self):
        s=Solver(['-x'],['x']);r=s.run([row([0.,1.],[1.])],max_attempts=1)
        self.assertFalse(r['complete']);self.assertEqual(r['stats'][0]['status'],1)
        self.assertTrue(math.isnan(r['predictions'][0][-1][0]))
        with self.assertRaises(ValueError):s.run([row([0.,0.],[1.])])
        with self.assertRaises(ValueError):s.run([row([0.,1.],[1.])],method='unknown')
        with self.assertRaises(ValueError):Solver(['__import__("os")'],['x'])
        r=Solver(['1/x'],['x']).run([row([0.,1.],[0.])])
        self.assertFalse(r['complete']);self.assertEqual(r['stats'][0]['status'],3)

    def test_ragged_no_prediction(self):
        s=Solver(['-x'],['x']);rr=[row([0.,.1],[1.]),row([0.,.5,1.],[2.])]
        a=s.run(rr);b=s.run(rr,predictions=False)
        self.assertEqual(a['stats'],b['stats']);self.assertIsNone(b['predictions'])
        self.assertEqual([v['count'] for v in b['stats']],[1,2])


if __name__=='__main__':unittest.main(verbosity=2)
