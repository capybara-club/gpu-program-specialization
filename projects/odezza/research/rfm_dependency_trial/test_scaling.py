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
import unittest
import numpy as np
import sympy as sp
from scaling_generate import template, bind, leaves
from generate import symbolic
from scaling_report import calibrate, measure


class TestScaling(unittest.TestCase):
    def test_calibration_below_old_grid_floor(self):
        row={'truth':np.array([[True,True,False]]*4),
             'scores':np.array([[1.,v,.01] for v in [1e-10,2e-9,3e-8,.5]])}
        t=calibrate([row],.75)
        self.assertEqual(t,2e-9)
        self.assertEqual(measure([row],t)['all_inputs_retained'],.75)
        self.assertLess(measure([row],np.nextafter(t,np.inf))['all_inputs_retained'],.75)
        self.assertEqual(calibrate([row],1.),1e-10)

    def test_exact_support_and_depth(self):
        cfg={'operators':['add','sub','mul','sin','cos','tanh'],'constant_probability':.2,'depth':2}
        rng=np.random.default_rng(998)
        def depth(t):return 0 if t[0] in ('x','c') else 1+max(map(depth,t[1:]))
        symbols=sp.symbols('x0:24')
        for k in [1,2,4]:
            for _ in range(10):
                t,_=template(rng,cfg,k)
                mapping=rng.choice(24,k,replace=False)
                result=bind(t,mapping)
                self.assertEqual(depth(result),2)
                self.assertEqual(leaves(result),set(mapping))
                f=symbolic(result,symbols)
                actual={i for i,s in enumerate(symbols) if sp.simplify(sp.diff(f,s))!=0}
                self.assertEqual(actual,set(mapping))


if __name__=='__main__':unittest.main()
