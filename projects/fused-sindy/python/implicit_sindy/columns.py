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

from typing import Iterable, Sequence

import numpy as np

from .ast import BinaryAst, ast_arrays
from .gram import (
    FEATURES,
    LEAVES,
    KernelsPerModule,
    _check_cuda_tensor,
    _check_same_device,
    _device_index,
    _torch_module,
)
from ._impl import _implicit_sindy as _native


class CompiledColumnKernels:
    def __init__(
        self,
        asts: Iterable[BinaryAst],
        *,
        kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
        device=None,
        worker_count: int = 0,
        scratch_bytes_per_worker: int = 64 * 1024 * 1024,
        nvptx_options: Sequence[str] = (),
    ) -> None:
        torch = _torch_module()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in PyTorch")

        self.kernels_per_module = int(kernels_per_module)
        if self.kernels_per_module < 1 or self.kernels_per_module > 64:
            raise ValueError("kernels_per_module must be in [1, 64]")
        self.device = torch.device("cuda", _device_index(torch, device))
        unary, binary = ast_arrays(asts)
        self.num_asts = int(unary.shape[0])
        self._num_cohorts = (self.num_asts + FEATURES - 1) // FEATURES
        features_per_cubin = self.kernels_per_module * FEATURES
        padded_num_asts = ((self.num_asts + features_per_cubin - 1) // features_per_cubin) * features_per_cubin
        if padded_num_asts != self.num_asts:
            pad_count = padded_num_asts - self.num_asts
            final_module_count = self.num_asts % features_per_cubin
            final_module_start = self.num_asts - final_module_count
            reps = (pad_count + final_module_count - 1) // final_module_count
            unary_pad = np.tile(unary[final_module_start:self.num_asts], (reps, 1))[:pad_count]
            binary_pad = np.tile(binary[final_module_start:self.num_asts], (reps, 1))[:pad_count]
            unary = np.concatenate((unary, unary_pad), axis=0)
            binary = np.concatenate((binary, binary_pad), axis=0)

        with torch.cuda.device(self.device):
            torch.empty((), device=self.device)
            self._handle = _native.ColumnsKernelSet(
                unary.tobytes(order="C"),
                binary.tobytes(order="C"),
                int(unary.shape[0]),
                self.kernels_per_module,
                int(worker_count),
                int(scratch_bytes_per_worker),
                list(nvptx_options),
            )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> CompiledColumnKernels:
        if self._closed:
            raise RuntimeError("CompiledColumnKernels is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def num_cohorts(self) -> int:
        if self._closed:
            return 0
        return int(self._num_cohorts)

    @property
    def num_cubins(self) -> int:
        if self._closed:
            return 0
        return int(self._handle.num_cubins)

    def run_columns(
        self,
        primitive_features,
        leaf_masks,
        leaf_words,
        *,
        cohort_index: int = 0,
        output=None,
        stream=None,
    ):
        torch = _torch_module()
        if self._closed:
            raise RuntimeError("CompiledColumnKernels is closed")
        _check_cuda_tensor(torch, primitive_features, "primitive_features", torch.float32, 2)
        _check_cuda_tensor(torch, leaf_masks, "leaf_masks", torch.int32, 2)
        _check_cuda_tensor(torch, leaf_words, "leaf_words", torch.int32, 3)
        _check_same_device(
            (primitive_features, leaf_masks, leaf_words),
            self.device,
        )

        if primitive_features.stride(1) != 1:
            raise ValueError("primitive_features must have stride 1 along the row dimension")
        if not leaf_masks.is_contiguous():
            raise ValueError("leaf_masks must be contiguous with shape [settings, 32]")
        if not leaf_words.is_contiguous():
            raise ValueError("leaf_words must be contiguous with shape [settings, 32, 8]")

        num_settings = int(leaf_masks.shape[0])
        num_primitive_features = int(primitive_features.shape[0])
        row_count = int(primitive_features.shape[1])
        if num_settings <= 0:
            raise ValueError("num_settings must be positive")
        if leaf_masks.shape != (num_settings, FEATURES):
            raise ValueError("leaf_masks must have shape [settings, 32]")
        if leaf_words.shape != (num_settings, FEATURES, LEAVES):
            raise ValueError("leaf_words must have shape [settings, 32, 8]")
        if cohort_index < 0 or cohort_index >= self._num_cohorts:
            raise ValueError("cohort_index is out of range")
        if num_primitive_features <= 0 or num_primitive_features > 32:
            raise ValueError("primitive feature count must be in [1, 32]")

        if output is None:
            output = torch.empty((num_settings, FEATURES, row_count), device=self.device, dtype=torch.float32)
        else:
            _check_cuda_tensor(torch, output, "output", torch.float32, 3)
            _check_same_device((output,), self.device)
            if output.shape != (num_settings, FEATURES, row_count):
                raise ValueError("output must have shape [settings, 32, rows]")
            if output.stride(2) != 1:
                raise ValueError("output must have stride 1 along the row dimension")
            if output.stride(1) < row_count:
                raise ValueError("output feature stride must be at least the row count")
            if output.stride(0) <= 0:
                raise ValueError("output setting stride must be positive")

        if stream is None:
            stream = torch.cuda.current_stream(device=self.device)
        stream_ptr = int(stream.cuda_stream)

        self._handle.launch_raw(
            int(cohort_index),
            stream_ptr,
            num_settings,
            int(primitive_features.data_ptr()),
            row_count,
            int(primitive_features.stride(0)),
            num_primitive_features,
            int(leaf_masks.data_ptr()),
            int(leaf_words.data_ptr()),
            int(leaf_words.stride(1)),
            int(output.data_ptr()),
            int(output.stride(0)),
            int(output.stride(1)),
        )
        valid_features = self.num_asts - int(cohort_index) * FEATURES
        if valid_features < FEATURES:
            with torch.cuda.stream(stream):
                output[:, valid_features:, :].zero_()
        return output


class CompiledSingleAstColumnKernels:
    def __init__(
        self,
        asts: Iterable[BinaryAst],
        *,
        kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
        device=None,
        worker_count: int = 0,
        scratch_bytes_per_worker: int = 64 * 1024 * 1024,
        nvptx_options: Sequence[str] = (),
    ) -> None:
        torch = _torch_module()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in PyTorch")

        self.kernels_per_module = int(kernels_per_module)
        if self.kernels_per_module < 1 or self.kernels_per_module > 64:
            raise ValueError("kernels_per_module must be in [1, 64]")
        self.device = torch.device("cuda", _device_index(torch, device))
        unary, binary = ast_arrays(asts)
        self.num_asts = int(unary.shape[0])

        with torch.cuda.device(self.device):
            torch.empty((), device=self.device)
            self._handle = _native.SingleAstColumnKernelSet(
                unary.tobytes(order="C"),
                binary.tobytes(order="C"),
                self.num_asts,
                self.kernels_per_module,
                int(worker_count),
                int(scratch_bytes_per_worker),
                list(nvptx_options),
            )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> CompiledSingleAstColumnKernels:
        if self._closed:
            raise RuntimeError("CompiledSingleAstColumnKernels is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def num_cubins(self) -> int:
        if self._closed:
            return 0
        return int(self._handle.num_cubins)

    def _check_inputs(
        self,
        primitive_features,
        leaf_masks,
        leaf_words,
    ) -> tuple[object, int, int, int]:
        torch = _torch_module()
        _check_cuda_tensor(torch, primitive_features, "primitive_features", torch.float32, 2)
        _check_cuda_tensor(torch, leaf_masks, "leaf_masks", torch.int32, 2)
        _check_cuda_tensor(torch, leaf_words, "leaf_words", torch.int32, 3)
        _check_same_device(
            (primitive_features, leaf_masks, leaf_words),
            self.device,
        )

        if primitive_features.stride(1) != 1:
            raise ValueError("primitive_features must have stride 1 along the row dimension")
        if not leaf_masks.is_contiguous():
            raise ValueError("leaf_masks must be contiguous with shape [settings, 32]")
        if not leaf_words.is_contiguous():
            raise ValueError("leaf_words must be contiguous with shape [settings, 32, 8]")

        num_settings = int(leaf_masks.shape[0])
        num_primitive_features = int(primitive_features.shape[0])
        row_count = int(primitive_features.shape[1])
        if num_settings <= 0:
            raise ValueError("num_settings must be positive")
        if leaf_masks.shape != (num_settings, FEATURES):
            raise ValueError("leaf_masks must have shape [settings, 32]")
        if leaf_words.shape != (num_settings, FEATURES, LEAVES):
            raise ValueError("leaf_words must have shape [settings, 32, 8]")
        if num_primitive_features <= 0 or num_primitive_features > 32:
            raise ValueError("primitive feature count must be in [1, 32]")

        return torch, num_settings, num_primitive_features, row_count

    def run_column(
        self,
        primitive_features,
        leaf_masks,
        leaf_words,
        *,
        ast_index: int = 0,
        feature_index: int | None = None,
        output=None,
        stream=None,
    ):
        if self._closed:
            raise RuntimeError("CompiledSingleAstColumnKernels is closed")
        torch, num_settings, num_primitive_features, row_count = self._check_inputs(
            primitive_features,
            leaf_masks,
            leaf_words,
        )
        ast_index = int(ast_index)
        if ast_index < 0 or ast_index >= self.num_asts:
            raise ValueError("ast_index is out of range")
        if feature_index is None:
            if ast_index >= FEATURES:
                raise ValueError("feature_index is required when ast_index is outside [0, 31]")
            feature_index = ast_index
        feature_index = int(feature_index)
        if feature_index < 0 or feature_index >= FEATURES:
            raise ValueError("feature_index must be in [0, 31]")

        if output is None:
            output = torch.empty((num_settings, row_count), device=self.device, dtype=torch.float32)
        else:
            _check_cuda_tensor(torch, output, "output", torch.float32, 2)
            _check_same_device((output,), self.device)
            if output.shape != (num_settings, row_count):
                raise ValueError("output must have shape [settings, rows]")
            if output.stride(1) != 1:
                raise ValueError("output must have stride 1 along the row dimension")
            if output.stride(0) < row_count:
                raise ValueError("output setting stride must be at least the row count")

        if stream is None:
            stream = torch.cuda.current_stream(device=self.device)
        stream_ptr = int(stream.cuda_stream)

        self._handle.launch_raw(
            ast_index,
            feature_index,
            stream_ptr,
            num_settings,
            int(primitive_features.data_ptr()),
            row_count,
            int(primitive_features.stride(0)),
            num_primitive_features,
            int(leaf_masks.data_ptr()),
            int(leaf_words.data_ptr()),
            int(leaf_words.stride(1)),
            int(output.data_ptr()),
            int(output.stride(0)),
        )
        return output

    def run_columns(
        self,
        primitive_features,
        leaf_masks,
        leaf_words,
        *,
        output=None,
        stream=None,
    ):
        if self._closed:
            raise RuntimeError("CompiledSingleAstColumnKernels is closed")
        if self.num_asts > FEATURES:
            raise ValueError("run_columns maps ast_index to feature_index and only supports up to 32 ASTs")
        torch, num_settings, _num_primitive_features, row_count = self._check_inputs(
            primitive_features,
            leaf_masks,
            leaf_words,
        )

        if output is None:
            output = torch.empty((num_settings, self.num_asts, row_count), device=self.device, dtype=torch.float32)
        else:
            _check_cuda_tensor(torch, output, "output", torch.float32, 3)
            _check_same_device((output,), self.device)
            if output.shape != (num_settings, self.num_asts, row_count):
                raise ValueError("output must have shape [settings, num_asts, rows]")
            if output.stride(2) != 1:
                raise ValueError("output must have stride 1 along the row dimension")
            if output.stride(1) < row_count:
                raise ValueError("output feature stride must be at least the row count")

        if stream is None:
            stream = torch.cuda.current_stream(device=self.device)

        for ast_index in range(self.num_asts):
            self.run_column(
                primitive_features,
                leaf_masks,
                leaf_words,
                ast_index=ast_index,
                feature_index=ast_index,
                output=output[:, ast_index, :],
                stream=stream,
            )
        return output


def compile_column_kernels(
    asts: Iterable[BinaryAst],
    *,
    kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
    device=None,
    worker_count: int = 0,
    scratch_bytes_per_worker: int = 64 * 1024 * 1024,
    nvptx_options: Sequence[str] = (),
) -> CompiledColumnKernels:
    return CompiledColumnKernels(
        asts,
        kernels_per_module=kernels_per_module,
        device=device,
        worker_count=worker_count,
        scratch_bytes_per_worker=scratch_bytes_per_worker,
        nvptx_options=nvptx_options,
    )


def compile_single_ast_column_kernels(
    asts: Iterable[BinaryAst],
    *,
    kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
    device=None,
    worker_count: int = 0,
    scratch_bytes_per_worker: int = 64 * 1024 * 1024,
    nvptx_options: Sequence[str] = (),
) -> CompiledSingleAstColumnKernels:
    return CompiledSingleAstColumnKernels(
        asts,
        kernels_per_module=kernels_per_module,
        device=device,
        worker_count=worker_count,
        scratch_bytes_per_worker=scratch_bytes_per_worker,
        nvptx_options=nvptx_options,
    )
