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
from polish import Expression as E, prepare, evaluate, fit, materialize

class PolishTests(unittest.TestCase):
    def test_host_identities_are_not_runtime_slots(self):
        a=E('affine_bank',(),(0,0,1))
        b=E('affine_bank',(),(0,0,1))
        e=E('add',(a,b))
        metadata=[dict(node=0,alternative=0,slot=100,value=1),dict(node=1,alternative=0,slot=101,value=1)]
        _,theta=prepare(e,[],0,'tied',metadata)
        self.assertEqual(len(theta),2)
        metadata[1]['slot']=100
        _,theta=prepare(e,[],0,'tied',metadata)
        self.assertEqual(len(theta),1)
        with self.assertRaisesRegex(ValueError,'metadata'):
            prepare(e,[],0,'tied',[])

    def test_resolved_binding_and_parameter_sharing(self):
        a=E.toggle2(E.bank(0), E.bank(1), 0)
        e=E('add',(E('mul',(a,E.input(0))),E('add',(E.bank(0),E.constant(2)))))
        code, theta=prepare(e,[.3,.8],0,'tied')
        self.assertEqual(len(theta),1)
        self.assertEqual(materialize(code,theta),e.resolve([.3,.8],0))
        code, theta=prepare(e,[.3,.8],0,'all_literals')
        self.assertEqual(len(theta),3)

    def test_analytic_jacobian(self):
        e=E('div',(E('exp',(E('mul',(E.bank(0),E.input(0))),)),E('add',(E.bank(1),E('sin',(E.input(1),))))))
        code,theta=prepare(e,[.2,2],0,'tied')
        x=np.random.default_rng(1).uniform(.1,1,(30,2))
        _,jac=evaluate(code,x,theta,True)
        for k in range(2):
            d=np.zeros(2);d[k]=1e-6
            finite=(evaluate(code,x,theta+d)[0]-evaluate(code,x,theta-d)[0])/(2e-6)
            np.testing.assert_allclose(jac[:,k],finite,rtol=1e-7,atol=1e-9)

    def test_recovers_coefficients_and_retains_original(self):
        e=E('add',(E('mul',(E.bank(0),E.input(0))),E.bank(1)))
        code,theta=prepare(e,[1,0],0,'tied')
        x=np.linspace(-2,2,100).reshape(-1,1).astype(np.float32)
        y=2.75*x[:,0]-.125
        result=fit(code,theta,x,y,123,5,starts=2,evaluations=30)
        np.testing.assert_allclose(result['parameters'],[2.75,-.125],atol=1e-6)
        self.assertLess(result['train_mse'],1e-12)

if __name__=='__main__':unittest.main()
