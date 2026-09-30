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
import torch
import sympy as sp
from learn import view
from generate import symbolic
from rfm import gradient, predict, update, learn
from report import average_precision, metrics


class TestTrial(unittest.TestCase):
    def test_ranking_ties_and_false_negative(self):
        self.assertEqual(average_precision(np.array([True,False,False]),np.ones(3)),1/3)
        self.assertEqual(average_precision(np.array([True,False,True]),np.array([3.,1.,2.])),1.)
        self.assertIsNone(average_precision(np.zeros(3,dtype=bool),np.ones(3)))
        row={'truth':np.array([[True,True,False],[False,False,False]]),
             'scores':np.array([[1.,.01,.02],[0.,0.,0.]])}
        result=metrics([row],.1)
        self.assertEqual(result['all_inputs_retained_nonzero'],0.)
        self.assertEqual(result['all_inputs_retained'],.5)
        self.assertEqual(result['zero_support_rhs'],1)

    def test_gradient_matches_autograd_and_difference(self):
        torch.manual_seed(17)
        x = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        c = torch.randn(13, 4, dtype=torch.float64)
        a = torch.randn(13, dtype=torch.float64)
        b = torch.randn(4, 4, dtype=torch.float64); m = b.T @ b + torch.eye(4)
        got = gradient(x, c, a, m, 2.)
        expected = torch.autograd.grad(predict(x, c, a, m, 2.).sum(), x)[0]
        torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)
        for j in range(4):
            d = torch.zeros_like(x); d[:, j] = 1e-5
            fd = (predict(x+d, c, a, m, 2.) - predict(x-d, c, a, m, 2.)) / 2e-5
            torch.testing.assert_close(got[:, j], fd, atol=1e-8, rtol=1e-7)

    def test_zero_rhs_and_metric(self):
        torch.manual_seed(18)
        x = torch.randn(30, 4, dtype=torch.float64); y = torch.zeros(30, dtype=torch.float64)
        cfg = dict(bandwidth_factor=1., ridge=1e-4, rfm_updates=3, metric_floor=1e-4)
        for r in learn(x, y, x, y, cfg).values():
            self.assertEqual(np.max(np.abs(r['matrix'])), 0.)
        m = update(torch.ones(4, 4, dtype=torch.float64), False, 1e-4)
        self.assertGreater(torch.linalg.eigvalsh(m).min(), 0.)

    def test_view_does_not_use_later_observations(self):
        t = np.linspace(0, 4, 81)
        obs = np.stack([np.stack([t*t, t**3], axis=-1) + i for i in range(5)])
        data = dict(times=t.tolist(), observations=obs.tolist(), training_trajectories=3)
        before = view(data, 1, 1.)
        obs[:, t>1] = 1e9; obs[1:3] = -1e9
        data['observations'] = obs.tolist()
        after = view(data, 1, 1.)
        for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(before[1][:,0], 2*t[1:20], atol=1e-12)

    def test_functional_cancellation(self):
        x = sp.symbols('x0:3')
        f = symbolic(['sub', ['x', 1], ['x', 1]], x)
        self.assertEqual(f, 0)
        f = symbolic(['mul', ['x', 0], ['x', 2]], x)
        self.assertEqual([sp.diff(f,s) != 0 for s in x], [True, False, True])


if __name__ == '__main__': unittest.main()
