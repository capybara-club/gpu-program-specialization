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
import struct
import unittest

import numpy as np

from prepare import build_legacy as build
from fit import compile_model, known_tree, make, model_for, simulate, structure_key, unpack


class PortfolioTests(unittest.TestCase):
    def problem(self):
        return dict(states=['x','y','z'],known_rhs={'x':'-y-z','y':'x'},trajectories=[
            dict(initial=[.1,.2,.3],times=[0,.1],values=[[.1,.2,.3],[.1,.2,.3]]) for _ in range(16)])

    def test_same_general_grammar(self):
        p=self.problem();a,m=build(p)
        p['known_rhs']['x']='-0.8*y-0.46*z';b,_=build(p)
        self.assertEqual(a['grammar'],b['grammar'])
        self.assertEqual(len(m['families']),18)
        self.assertEqual(m['expected_configurations'],440565760)
        self.assertIsNone(m['public_motif'])

    def test_hint_is_explicit_and_separate(self):
        q,m=build(self.problem(),hint='cos(2.4*x)',rows=65536)
        self.assertEqual(m['public_motif'],'cos(2.4*x)')
        self.assertEqual(m['expected_configurations'],56360960)
        self.assertEqual(len(q['grammar']['families']),12)

    def test_fixed_literal_collision_rejected(self):
        c={'_fixed_literal_bits':['3f800000'],'value_bits':['3f800000']}
        with self.assertRaisesRegex(ValueError,'collides'):model_for(c,{})

    def test_commutative_product_duplicates(self):
        coefficient=b'\x83'+struct.pack('<f',.1)
        def candidate(a,b):
            program=coefficient+bytes([0x81,a,0x92,0x81,b,0x92,0x80])
            return dict(resolved_programs=['','',program.hex()],value_bits=[struct.pack('>f',.1).hex()],slots=[dict(name='rng.q')])
        self.assertEqual(structure_key(candidate(0,1)),structure_key(candidate(1,0)))
        self.assertNotEqual(structure_key(candidate(0,1)),structure_key(candidate(0,2)))

    def test_generated_jacobian_and_rollout_sensitivity(self):
        x,y,z=[('v',i) for i in range(3)];a,b=[('v',i) for i in (3,4)]
        tree=make('+',make('*',a,make('sin',make('*',x,y))),make('*',b,z))
        f=compile_model([known_tree('-y-z'),known_tree('x'),tree],2)
        states=np.array([[.1,.2,.3],[-.4,.5,.6]]);p=np.array([.4,-.3]);eps=1e-6
        v,j=f(states,p)
        for k in range(5):
            sl=states.copy();sh=states.copy();pl=p.copy();ph=p.copy()
            if k<3:sl[:,k]-=eps;sh[:,k]+=eps
            else:pl[k-3]-=eps;ph[k-3]+=eps
            expected=(f(sh,ph)[0]-f(sl,pl)[0])/(2*eps)
            np.testing.assert_allclose(j[:,:,k],expected,rtol=1e-5,atol=2e-8)
        ts=[dict(initial=states[0],times=[0,.1,.2],values=[[0]*3]*3)]
        residual,jac=simulate(f,p,ts,rtol=1e-10,atol=1e-12)
        for k in range(2):
            lo=p.copy();hi=p.copy();lo[k]-=eps;hi[k]+=eps
            expected=(simulate(f,hi,ts,rtol=1e-10,atol=1e-12)[0]-simulate(f,lo,ts,rtol=1e-10,atol=1e-12)[0])/(2*eps)
            np.testing.assert_allclose(jac[:,k],expected,rtol=1e-4,atol=2e-8)

    def test_exp_program_and_sensitivity(self):
        # exp(p*x*x) tests both the native natural-exp opcode and its chain rule.
        program=b'\x83'+struct.pack('<f',-.3)+bytes([0x81,0,0x81,0,0x92,0x92,0xa1,0x80])
        tree=unpack(program.hex(),[struct.pack('>f',-.3).hex()])
        f=compile_model([known_tree('-y'),known_tree('x'),tree],1)
        state=np.array([[.4,-.1,.2],[-.7,.2,.5]]);p=np.array([-.3])
        value,jac=f(state,p);e=np.exp(p[0]*state[:,0]**2)
        np.testing.assert_allclose(value[:,2],e)
        np.testing.assert_allclose(jac[:,2,0],2*p[0]*state[:,0]*e)
        np.testing.assert_allclose(jac[:,2,3],state[:,0]**2*e)
        ts=[dict(initial=state[0],times=[0,.2,.5],values=[[0]*3]*3)]
        _,j=simulate(f,p,ts,rtol=1e-10,atol=1e-12)
        eps=1e-6
        diff=(simulate(f,p+eps,ts,rtol=1e-10,atol=1e-12)[0]-simulate(f,p-eps,ts,rtol=1e-10,atol=1e-12)[0])/(2*eps)
        np.testing.assert_allclose(j[:,0],diff,rtol=1e-5,atol=2e-8)


if __name__=='__main__':unittest.main()
