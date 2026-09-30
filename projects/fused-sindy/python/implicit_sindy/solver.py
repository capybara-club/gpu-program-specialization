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

from dataclasses import dataclass

from .gram import FEATURES, GramOutputs, _check_cuda_tensor, _check_same_device, _torch_module


@dataclass(frozen=True)
class RidgeSolveOutputs:
    beta_standardized: object
    solve_info: object
    x_mean: object
    x_scale: object
    y_mean: object
    active_masks: object | None = None
    active_counts: object | None = None
    iteration_counts: object | None = None


class RidgeSolver:
    def __init__(self, *, device=None) -> None:
        torch = _torch_module()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in PyTorch")
        if device is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device(device)
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        with torch.cuda.device(self.device):
            torch.empty((), device=self.device)
            from ._impl import _implicit_sindy as _native

            self._handle = _native.RidgeSolver()
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> RidgeSolver:
        if self._closed:
            raise RuntimeError("RidgeSolver is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def solve_posv(
        self,
        raw: GramOutputs,
        alphas,
        *,
        row_count: int,
        scale_epsilon: float = 1.0e-2,
        stream=None,
    ) -> RidgeSolveOutputs:
        torch = _torch_module()
        self._check_open()
        _check_cuda_tensor(torch, alphas, "alphas", torch.float32, 1)
        num_settings, num_rhs = self._raw_dimensions(raw)
        _check_same_device((alphas,), self.device)
        if not alphas.is_contiguous():
            raise ValueError("alphas must be contiguous")

        num_sweeps = int(alphas.shape[0])
        beta = torch.empty((num_sweeps, num_settings, num_rhs, FEATURES), device=self.device, dtype=torch.float32)
        solve_info = torch.empty((num_sweeps, num_settings), device=self.device, dtype=torch.int32)
        x_mean = torch.empty((num_settings, FEATURES), device=self.device, dtype=torch.float32)
        x_scale = torch.empty((num_settings, FEATURES), device=self.device, dtype=torch.float32)
        y_mean = torch.empty((num_settings, num_rhs), device=self.device, dtype=torch.float32)
        if stream is None:
            stream = torch.cuda.current_stream(device=self.device)

        self._handle.solve_posv_raw(
            int(stream.cuda_stream),
            int(row_count),
            num_settings,
            num_rhs,
            num_sweeps,
            float(scale_epsilon),
            int(raw.gram_storage.data_ptr()),
            FEATURES,
            FEATURES * FEATURES,
            int(raw.x_sum.data_ptr()),
            FEATURES,
            int(raw.xty.data_ptr()),
            FEATURES,
            num_rhs * FEATURES,
            int(raw.y_sum.data_ptr()),
            1,
            num_rhs,
            int(alphas.data_ptr()),
            1,
            int(x_mean.data_ptr()),
            FEATURES,
            int(x_scale.data_ptr()),
            FEATURES,
            int(y_mean.data_ptr()),
            1,
            num_rhs,
            int(beta.data_ptr()),
            FEATURES,
            num_rhs * FEATURES,
            num_settings * num_rhs * FEATURES,
            int(solve_info.data_ptr()),
            num_settings,
            1,
        )
        return RidgeSolveOutputs(
            beta_standardized=beta,
            solve_info=solve_info,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
        )

    def solve_stlsq(
        self,
        raw: GramOutputs,
        alphas,
        thresholds,
        *,
        row_count: int,
        scale_epsilon: float = 1.0e-2,
        stream=None,
    ) -> RidgeSolveOutputs:
        torch = _torch_module()
        self._check_open()
        _check_cuda_tensor(torch, alphas, "alphas", torch.float32, 1)
        _check_cuda_tensor(torch, thresholds, "thresholds", torch.float32, 1)
        num_settings, num_rhs = self._raw_dimensions(raw)
        _check_same_device((alphas, thresholds), self.device)
        if not alphas.is_contiguous() or not thresholds.is_contiguous():
            raise ValueError("alphas and thresholds must be contiguous")
        if alphas.shape != thresholds.shape:
            raise ValueError("alphas and thresholds must have matching shapes")

        num_sweeps = int(alphas.shape[0])
        beta = torch.empty((num_sweeps, num_settings, num_rhs, FEATURES), device=self.device, dtype=torch.float32)
        active_masks = torch.empty((num_sweeps, num_settings, num_rhs), device=self.device, dtype=torch.uint32)
        active_counts = torch.empty((num_sweeps, num_settings, num_rhs), device=self.device, dtype=torch.int32)
        iteration_counts = torch.empty((num_sweeps, num_settings, num_rhs), device=self.device, dtype=torch.int32)
        solve_info = torch.empty((num_sweeps, num_settings), device=self.device, dtype=torch.int32)
        x_mean = torch.empty((num_settings, FEATURES), device=self.device, dtype=torch.float32)
        x_scale = torch.empty((num_settings, FEATURES), device=self.device, dtype=torch.float32)
        y_mean = torch.empty((num_settings, num_rhs), device=self.device, dtype=torch.float32)
        if stream is None:
            stream = torch.cuda.current_stream(device=self.device)

        self._handle.solve_stlsq_raw(
            int(stream.cuda_stream),
            int(row_count),
            num_settings,
            num_rhs,
            num_sweeps,
            float(scale_epsilon),
            int(raw.gram_storage.data_ptr()),
            FEATURES,
            FEATURES * FEATURES,
            int(raw.x_sum.data_ptr()),
            FEATURES,
            int(raw.xty.data_ptr()),
            FEATURES,
            num_rhs * FEATURES,
            int(raw.y_sum.data_ptr()),
            1,
            num_rhs,
            int(alphas.data_ptr()),
            1,
            int(thresholds.data_ptr()),
            1,
            int(x_mean.data_ptr()),
            FEATURES,
            int(x_scale.data_ptr()),
            FEATURES,
            int(y_mean.data_ptr()),
            1,
            num_rhs,
            int(beta.data_ptr()),
            FEATURES,
            num_rhs * FEATURES,
            num_settings * num_rhs * FEATURES,
            int(active_masks.data_ptr()),
            1,
            num_rhs,
            num_settings * num_rhs,
            int(active_counts.data_ptr()),
            1,
            num_rhs,
            num_settings * num_rhs,
            int(iteration_counts.data_ptr()),
            1,
            num_rhs,
            num_settings * num_rhs,
            int(solve_info.data_ptr()),
            num_settings,
            1,
        )
        return RidgeSolveOutputs(
            beta_standardized=beta,
            solve_info=solve_info,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            active_masks=active_masks,
            active_counts=active_counts,
            iteration_counts=iteration_counts,
        )

    def score_validation_mse(
        self,
        validation: GramOutputs,
        solve: RidgeSolveOutputs,
        *,
        validation_row_count: int,
        stream=None,
    ):
        torch = _torch_module()
        self._check_open()
        num_settings, num_rhs = self._raw_dimensions(validation)
        num_sweeps = self._check_solve(solve, num_settings, num_rhs)
        mse = torch.empty((num_sweeps, num_settings, num_rhs), device=self.device, dtype=torch.float32)
        if stream is None:
            stream = torch.cuda.current_stream(device=self.device)

        self._handle.score_validation_mse_raw(
            int(stream.cuda_stream),
            int(validation_row_count),
            num_settings,
            num_rhs,
            num_sweeps,
            int(validation.gram_storage.data_ptr()),
            FEATURES,
            FEATURES * FEATURES,
            int(validation.x_sum.data_ptr()),
            FEATURES,
            int(validation.xty.data_ptr()),
            FEATURES,
            num_rhs * FEATURES,
            int(validation.y_sum.data_ptr()),
            1,
            num_rhs,
            int(validation.yy.data_ptr()),
            1,
            num_rhs,
            int(solve.beta_standardized.data_ptr()),
            FEATURES,
            num_rhs * FEATURES,
            num_settings * num_rhs * FEATURES,
            int(solve.x_mean.data_ptr()),
            FEATURES,
            int(solve.x_scale.data_ptr()),
            FEATURES,
            int(solve.y_mean.data_ptr()),
            1,
            num_rhs,
            int(solve.solve_info.data_ptr()),
            num_settings,
            1,
            int(mse.data_ptr()),
            1,
            num_rhs,
            num_settings * num_rhs,
        )
        return mse

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("RidgeSolver is closed")

    def _raw_dimensions(self, raw: GramOutputs) -> tuple[int, int]:
        torch = _torch_module()
        _check_cuda_tensor(torch, raw.gram_storage, "gram_storage", torch.float32, 3)
        _check_cuda_tensor(torch, raw.x_sum, "x_sum", torch.float32, 2)
        _check_cuda_tensor(torch, raw.xty, "xty", torch.float32, 3)
        _check_cuda_tensor(torch, raw.y_sum, "y_sum", torch.float32, 2)
        _check_cuda_tensor(torch, raw.yy, "yy", torch.float32, 2)
        _check_same_device((raw.gram_storage, raw.x_sum, raw.xty, raw.y_sum, raw.yy), self.device)
        if not raw.gram_storage.is_contiguous() or not raw.x_sum.is_contiguous() or not raw.xty.is_contiguous() or not raw.y_sum.is_contiguous() or not raw.yy.is_contiguous():
            raise ValueError("GramOutputs tensors must be contiguous")

        num_settings = int(raw.x_sum.shape[0])
        num_rhs = int(raw.xty.shape[1])
        if raw.gram_storage.shape != (num_settings, FEATURES, FEATURES):
            raise ValueError("gram_storage must have shape [settings, 32, 32]")
        if raw.x_sum.shape != (num_settings, FEATURES):
            raise ValueError("x_sum must have shape [settings, 32]")
        if raw.xty.shape != (num_settings, num_rhs, FEATURES):
            raise ValueError("xty must have shape [settings, rhs, 32]")
        if raw.y_sum.shape != (num_settings, num_rhs):
            raise ValueError("y_sum must have shape [settings, rhs]")
        if raw.yy.shape != (num_settings, num_rhs):
            raise ValueError("yy must have shape [settings, rhs]")
        return num_settings, num_rhs

    def _check_solve(self, solve: RidgeSolveOutputs, num_settings: int, num_rhs: int) -> int:
        torch = _torch_module()
        _check_cuda_tensor(torch, solve.beta_standardized, "beta_standardized", torch.float32, 4)
        _check_cuda_tensor(torch, solve.solve_info, "solve_info", torch.int32, 2)
        _check_cuda_tensor(torch, solve.x_mean, "x_mean", torch.float32, 2)
        _check_cuda_tensor(torch, solve.x_scale, "x_scale", torch.float32, 2)
        _check_cuda_tensor(torch, solve.y_mean, "y_mean", torch.float32, 2)
        _check_same_device((solve.beta_standardized, solve.solve_info, solve.x_mean, solve.x_scale, solve.y_mean), self.device)
        if not solve.beta_standardized.is_contiguous() or not solve.solve_info.is_contiguous() or not solve.x_mean.is_contiguous() or not solve.x_scale.is_contiguous() or not solve.y_mean.is_contiguous():
            raise ValueError("solve tensors must be contiguous")

        num_sweeps = int(solve.beta_standardized.shape[0])
        if solve.beta_standardized.shape != (num_sweeps, num_settings, num_rhs, FEATURES):
            raise ValueError("beta_standardized must have shape [sweeps, settings, rhs, 32]")
        if solve.solve_info.shape != (num_sweeps, num_settings):
            raise ValueError("solve_info must have shape [sweeps, settings]")
        if solve.x_mean.shape != (num_settings, FEATURES):
            raise ValueError("x_mean must have shape [settings, 32]")
        if solve.x_scale.shape != (num_settings, FEATURES):
            raise ValueError("x_scale must have shape [settings, 32]")
        if solve.y_mean.shape != (num_settings, num_rhs):
            raise ValueError("y_mean must have shape [settings, rhs]")
        return num_sweeps


def create_ridge_solver(*, device=None) -> RidgeSolver:
    return RidgeSolver(device=device)
