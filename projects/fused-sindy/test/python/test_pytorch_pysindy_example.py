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
from __future__ import annotations

import unittest

import numpy as np

import fused_sindy as isindy


def _torch_or_skip(test: unittest.TestCase):
    try:
        import torch
    except Exception as exc:
        test.skipTest(f"PyTorch is not available: {exc}")
    if not torch.cuda.is_available():
        test.skipTest("PyTorch CUDA is not available")
    return torch


def _poly_expressions() -> list[isindy.Expr]:
    x = isindy.feature(0)
    y = isindy.feature(1)
    exprs = [
        isindy.const(1.0),
        x,
        y,
        x * x,
        x * y,
        y * y,
        x * x * x,
        x * x * y,
        x * y * y,
        y * y * y,
    ]
    while len(exprs) < 32:
        exprs.append(isindy.const(0.0))
    return exprs


def _poly_library(primitive: np.ndarray) -> np.ndarray:
    x = primitive[0].astype(np.float32)
    y = primitive[1].astype(np.float32)
    panel = np.zeros((32, primitive.shape[1]), dtype=np.float32)
    panel[0] = np.float32(1.0)
    panel[1] = x
    panel[2] = y
    panel[3] = (x * x).astype(np.float32)
    panel[4] = (x * y).astype(np.float32)
    panel[5] = (y * y).astype(np.float32)
    panel[6] = (x * x * x).astype(np.float32)
    panel[7] = (x * x * y).astype(np.float32)
    panel[8] = (x * y * y).astype(np.float32)
    panel[9] = (y * y * y).astype(np.float32)
    return panel


def _make_data() -> tuple[np.ndarray, np.ndarray]:
    rows = 4096
    rng = np.random.default_rng(41)
    x = rng.uniform(-1.0, 1.0, rows).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, rows).astype(np.float32)
    primitive = np.stack([x, y]).astype(np.float32)
    targets = np.stack([
        (np.float32(-2.0) * x).astype(np.float32),
        y.astype(np.float32),
    ]).astype(np.float32)
    return primitive, targets


def _raw_beta(beta_standardized, x_scale):
    return (beta_standardized / x_scale[None, :, None, :]).detach().cpu().numpy()


def _intercept(beta_standardized, x_mean, x_scale, y_mean):
    beta_raw = beta_standardized / x_scale[None, :, None, :]
    return (
        y_mean[None, :, :]
        - (x_mean[None, :, None, :] * beta_raw).sum(dim=-1)
    ).detach().cpu().numpy()


def _numpy_stlsq(panel: np.ndarray, targets: np.ndarray, threshold: float = 1.0e-3) -> np.ndarray:
    coefs = np.zeros((targets.shape[0], panel.shape[0]), dtype=np.float64)
    gram = panel @ panel.T
    xty = panel @ targets.T
    for rhs in range(targets.shape[0]):
        active = np.ones((panel.shape[0],), dtype=bool)
        coef = np.zeros((panel.shape[0],), dtype=np.float64)
        for _ in range(16):
            active_idx = np.flatnonzero(active)
            solved = np.linalg.lstsq(gram[np.ix_(active_idx, active_idx)], xty[active_idx, rhs], rcond=None)[0]
            next_coef = np.zeros_like(coef)
            next_coef[active_idx] = solved
            next_active = np.abs(next_coef) >= threshold
            coef = next_coef
            if np.array_equal(next_active, active):
                break
            active = next_active
        coefs[rhs] = coef
    return coefs


class PytorchPySindyExampleTest(unittest.TestCase):
    def test_linear_ode_recovery_with_expression_asts(self) -> None:
        torch = _torch_or_skip(self)
        primitive, targets = _make_data()
        panel = _poly_library(primitive)
        numpy_coef = _numpy_stlsq(panel, targets)

        exprs = _poly_expressions()
        asts, leaf_masks, leaf_words = isindy.expressions_to_asts_and_settings(exprs)
        primitive_t = torch.tensor(primitive, device="cuda")
        targets_t = torch.tensor(targets, device="cuda")
        leaf_masks_t = torch.tensor(leaf_masks, device="cuda", dtype=torch.int32)
        leaf_words_t = torch.tensor(leaf_words, device="cuda", dtype=torch.int32)
        alphas = torch.tensor([1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1], device="cuda", dtype=torch.float32)
        thresholds = torch.tensor([1.0e-3, 1.0e-3, 1.0e-2, 5.0e-2], device="cuda", dtype=torch.float32)

        with isindy.compile_gram_kernels(asts, kernels_per_module=isindy.KernelsPerModule.ONE, worker_count=2) as gram_kernels:
            raw = gram_kernels.run_gram(
                primitive_t,
                targets_t,
                leaf_masks_t,
                leaf_words_t,
            )

            with isindy.RidgeSolver() as solver:
                dense = solver.solve_posv(raw, alphas, row_count=primitive.shape[1])
                stlsq = solver.solve_stlsq(raw, alphas, thresholds, row_count=primitive.shape[1])
                dense_mse = solver.score_validation_mse(raw, dense, validation_row_count=primitive.shape[1])
                stlsq_mse = solver.score_validation_mse(raw, stlsq, validation_row_count=primitive.shape[1])

        torch.cuda.synchronize()
        dense_beta = _raw_beta(dense.beta_standardized, dense.x_scale)
        stlsq_beta = _raw_beta(stlsq.beta_standardized, stlsq.x_scale)
        dense_intercept = _intercept(dense.beta_standardized, dense.x_mean, dense.x_scale, dense.y_mean)
        stlsq_intercept = _intercept(stlsq.beta_standardized, stlsq.x_mean, stlsq.x_scale, stlsq.y_mean)
        dense_mse_np = dense_mse.detach().cpu().numpy()
        stlsq_mse_np = stlsq_mse.detach().cpu().numpy()

        np.testing.assert_allclose(numpy_coef[0, 1], -2.0, atol=2.0e-6, rtol=2.0e-6)
        np.testing.assert_allclose(numpy_coef[1, 2], 1.0, atol=2.0e-6, rtol=2.0e-6)
        np.testing.assert_array_equal(dense.solve_info.detach().cpu().numpy(), np.zeros((4, 1), dtype=np.int32))
        np.testing.assert_array_equal(stlsq.solve_info.detach().cpu().numpy(), np.zeros((4, 1), dtype=np.int32))
        np.testing.assert_allclose(dense_beta[0, 0, 0, 1], -2.0, atol=2.5e-3, rtol=2.5e-3)
        np.testing.assert_allclose(dense_beta[0, 0, 1, 2], 1.0, atol=2.5e-3, rtol=2.5e-3)
        np.testing.assert_allclose(stlsq_beta[0, 0, 0, 1], -2.0, atol=2.5e-4, rtol=2.5e-4)
        np.testing.assert_allclose(stlsq_beta[0, 0, 1, 2], 1.0, atol=2.5e-4, rtol=2.5e-4)
        np.testing.assert_allclose(stlsq_beta[0, 0, 0, :10], numpy_coef[0, :10], atol=2.5e-4, rtol=2.5e-4)
        np.testing.assert_allclose(stlsq_beta[0, 0, 1, :10], numpy_coef[1, :10], atol=2.5e-4, rtol=2.5e-4)
        np.testing.assert_allclose(dense_intercept[0, 0], [0.0, 0.0], atol=2.0e-3, rtol=0.0)
        np.testing.assert_allclose(stlsq_intercept[0, 0], [0.0, 0.0], atol=2.0e-4, rtol=0.0)
        self.assertLess(float(dense_mse_np[0, 0, 0]), 1.0e-6)
        self.assertLess(float(dense_mse_np[0, 0, 1]), 1.0e-6)
        self.assertLess(float(stlsq_mse_np[0, 0, 0]), 1.0e-8)
        self.assertLess(float(stlsq_mse_np[0, 0, 1]), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
