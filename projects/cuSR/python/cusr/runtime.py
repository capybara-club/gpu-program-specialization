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

from collections.abc import Sequence
from typing import Any

import numpy as np

from cuda.core import Buffer, LaunchConfig, ObjectCode, Stream, launch

from .compiler import TileStaticEvalInstantiation, TileStaticMseInstantiation


class TileStaticMseModule:
    def __init__(
        self,
        cubin: bytes | bytearray,
        lowered_names: Sequence[str],
        instantiations: Sequence[TileStaticMseInstantiation],
        *,
        device: int = 0,
    ) -> None:
        import torch

        self.instantiations = tuple(instantiations)
        self.lowered_names = tuple(lowered_names)
        if not self.instantiations or len(self.instantiations) != len(self.lowered_names):
            raise ValueError("one lowered kernel name is required per instantiation")
        if tuple(item.kernel_index for item in self.instantiations) != tuple(range(len(self.instantiations))):
            raise ValueError("kernel indices must be consecutive and ordered from zero")
        if len({item.ast_capacity for item in self.instantiations}) != 1:
            raise ValueError("every kernel in a patched module must have the same AST capacity")

        torch.cuda.set_device(device)
        torch.empty(0, device=f"cuda:{device}")
        self.device = device
        self.object_code = ObjectCode.from_cubin(bytes(cubin), name="cusr_tile_static_mse")
        self.kernels = tuple(self.object_code.get_kernel(name) for name in self.lowered_names)

    @staticmethod
    def _buffer(tensor: Any) -> Buffer:
        return Buffer.from_handle(tensor.data_ptr(), tensor.numel() * tensor.element_size(), owner=tensor)

    @property
    def ast_capacity(self) -> int:
        return self.instantiations[0].ast_capacity

    def __call__(
        self,
        x: Any,
        target: Any,
        leaf_masks: Any,
        leaf_words: Any,
        *,
        asts_per_kernel: int,
        output_sse: Any | None = None,
        stream: Any | None = None,
        streams: Sequence[Any] | None = None,
    ) -> Any:
        import torch

        tensors = (x, target, leaf_masks, leaf_words)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("x, target, leaf_masks, and leaf_words must be PyTorch tensors")
        if any(not tensor.is_cuda or tensor.device.index != self.device for tensor in tensors):
            raise ValueError("all kernel inputs must be CUDA tensors on the module device")
        if x.dtype != torch.float32 or x.ndim != 2 or not x.is_contiguous():
            raise ValueError("x must be a contiguous float32 [column, leading_dim] tensor")
        if not 1 <= x.shape[0] <= 32:
            raise ValueError("x must contain between 1 and 32 columns")
        if target.dtype != torch.float32 or target.ndim != 1 or not target.is_contiguous():
            raise ValueError("target must be a contiguous float32 tensor")
        if leaf_masks.dtype != torch.uint8 or leaf_masks.ndim != 1 or not leaf_masks.is_contiguous():
            raise ValueError("leaf_masks must be a contiguous uint8 tensor")
        if leaf_words.dtype != torch.uint32 or leaf_words.ndim != 2 or leaf_words.shape[0] != leaf_masks.numel() or leaf_words.shape[1] < 8 or leaf_words.stride(1) != 1:
            raise ValueError("leaf_words must be a uint32 [setting, stride>=8] tensor")
        if target.numel() == 0 or x.shape[1] < target.numel():
            raise ValueError("x leading dimension must cover every target row")
        if not 1 <= asts_per_kernel <= self.ast_capacity:
            raise ValueError("asts_per_kernel exceeds the compiled AST capacity")

        num_settings = leaf_masks.numel()
        if num_settings == 0 or num_settings > 0xFFFFFFFF:
            raise ValueError("num_settings must fit in a positive uint32")
        total_asts = len(self.kernels) * asts_per_kernel
        if output_sse is not None and (
            not isinstance(output_sse, torch.Tensor)
            or not output_sse.is_cuda
            or output_sse.device != x.device
            or output_sse.dtype != torch.float32
            or not output_sse.is_contiguous()
            or tuple(output_sse.shape) != (total_asts, num_settings)
        ):
            raise ValueError("output_sse must be contiguous float32 [total_asts, num_settings]")

        if stream is not None and streams is not None:
            raise ValueError("stream and streams are mutually exclusive")
        producer_stream = torch.cuda.current_stream(x.device)
        if streams is None:
            worker_streams = (producer_stream if stream is None else stream,)
        else:
            worker_streams = tuple(streams)
            if not worker_streams:
                raise ValueError("streams must not be empty")
        if any(not hasattr(worker_stream, "cuda_stream") for worker_stream in worker_streams):
            raise TypeError("every stream must be a PyTorch CUDA stream")

        with torch.cuda.stream(producer_stream):
            if output_sse is None:
                output_sse = torch.zeros((total_asts, num_settings), dtype=torch.float32, device=x.device)
            else:
                output_sse.zero_()

        for worker_stream in worker_streams:
            if worker_stream != producer_stream:
                worker_stream.wait_stream(producer_stream)

        launch_streams = tuple(Stream.from_handle(worker_stream.cuda_stream) for worker_stream in worker_streams)
        num_rows = target.numel()
        num_columns = x.shape[0]
        leading_dim = x.shape[1]
        leaf_words_stride = leaf_words.stride(0)
        x_buffer = self._buffer(x)
        target_buffer = self._buffer(target)
        leaf_masks_buffer = self._buffer(leaf_masks)
        leaf_words_buffer = self._buffer(leaf_words)

        for kernel_idx, (kernel, instantiation) in enumerate(zip(self.kernels, self.instantiations, strict=True)):
            worker_index = kernel_idx % len(worker_streams)
            grid = (num_rows + instantiation.tile_rows - 1) // instantiation.tile_rows
            shared_stride = num_columns | 1
            shared_bytes = (instantiation.tile_rows * shared_stride + instantiation.tile_rows) * np.dtype(np.float32).itemsize
            kernel_output = output_sse[kernel_idx * asts_per_kernel : (kernel_idx + 1) * asts_per_kernel]
            launch(
                launch_streams[worker_index],
                LaunchConfig(grid=grid, block=instantiation.threads_per_cta, shmem_size=shared_bytes),
                kernel,
                x_buffer,
                target_buffer,
                np.uint64(num_rows),
                np.uint32(num_columns),
                np.uint64(leading_dim),
                leaf_masks_buffer,
                leaf_words_buffer,
                np.uint64(leaf_words_stride),
                np.uint32(num_settings),
                np.uint32(asts_per_kernel),
                self._buffer(kernel_output),
            )

        for worker_stream in worker_streams:
            for tensor in (*tensors, output_sse):
                tensor.record_stream(worker_stream)
        return output_sse


class TileStaticEvalModule:
    def __init__(
        self,
        cubin: bytes | bytearray,
        lowered_names: Sequence[str],
        instantiations: Sequence[TileStaticEvalInstantiation],
        *,
        device: int = 0,
    ) -> None:
        import torch

        self.instantiations = tuple(instantiations)
        self.lowered_names = tuple(lowered_names)
        if not self.instantiations or len(self.instantiations) != len(self.lowered_names):
            raise ValueError("one lowered kernel name is required per instantiation")
        if tuple(item.kernel_index for item in self.instantiations) != tuple(range(len(self.instantiations))):
            raise ValueError("kernel indices must be consecutive and ordered from zero")
        if len({item.ast_capacity for item in self.instantiations}) != 1:
            raise ValueError("every kernel in a patched module must have the same AST capacity")

        torch.cuda.set_device(device)
        torch.empty(0, device=f"cuda:{device}")
        self.device = device
        self.object_code = ObjectCode.from_cubin(bytes(cubin), name="cusr_tile_static_eval")
        self.kernels = tuple(self.object_code.get_kernel(name) for name in self.lowered_names)

    @staticmethod
    def _buffer(tensor: Any) -> Buffer:
        return Buffer.from_handle(tensor.data_ptr(), tensor.numel() * tensor.element_size(), owner=tensor)

    @property
    def ast_capacity(self) -> int:
        return self.instantiations[0].ast_capacity

    def __call__(
        self,
        x: Any,
        leaf_masks: Any,
        leaf_words: Any,
        *,
        asts_per_kernel: int,
        num_rows: int | None = None,
        output_values: Any | None = None,
        stream: Any | None = None,
        streams: Sequence[Any] | None = None,
    ) -> Any:
        import torch

        tensors = (x, leaf_masks, leaf_words)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("x, leaf_masks, and leaf_words must be PyTorch tensors")
        if any(not tensor.is_cuda or tensor.device.index != self.device for tensor in tensors):
            raise ValueError("all kernel inputs must be CUDA tensors on the module device")
        if x.dtype != torch.float32 or x.ndim != 2 or not x.is_contiguous():
            raise ValueError("x must be a contiguous float32 [column, leading_dim] tensor")
        if not 1 <= x.shape[0] <= 32:
            raise ValueError("x must contain between 1 and 32 columns")
        if leaf_masks.dtype != torch.uint8 or leaf_masks.ndim != 1 or not leaf_masks.is_contiguous():
            raise ValueError("leaf_masks must be a contiguous uint8 tensor")
        if leaf_words.dtype != torch.uint32 or leaf_words.ndim != 2 or leaf_words.shape[0] != leaf_masks.numel() or leaf_words.shape[1] < 8 or leaf_words.stride(1) != 1:
            raise ValueError("leaf_words must be a uint32 [setting, stride>=8] tensor")
        if num_rows is None:
            num_rows = x.shape[1]
        if isinstance(num_rows, bool) or not isinstance(num_rows, int) or not 1 <= num_rows <= x.shape[1]:
            raise ValueError("num_rows must select a positive prefix of x")
        if not 1 <= asts_per_kernel <= self.ast_capacity:
            raise ValueError("asts_per_kernel exceeds the compiled AST capacity")

        num_settings = leaf_masks.numel()
        if num_settings == 0 or num_settings > 0xFFFFFFFF:
            raise ValueError("num_settings must fit in a positive uint32")
        total_asts = len(self.kernels) * asts_per_kernel
        expected_shape = (total_asts, num_settings, num_rows)
        if output_values is not None and (
            not isinstance(output_values, torch.Tensor)
            or not output_values.is_cuda
            or output_values.device != x.device
            or output_values.dtype != torch.float32
            or not output_values.is_contiguous()
            or tuple(output_values.shape) != expected_shape
        ):
            raise ValueError("output_values must be contiguous float32 [total_asts, num_settings, num_rows]")

        if stream is not None and streams is not None:
            raise ValueError("stream and streams are mutually exclusive")
        producer_stream = torch.cuda.current_stream(x.device)
        if streams is None:
            worker_streams = (producer_stream if stream is None else stream,)
        else:
            worker_streams = tuple(streams)
            if not worker_streams:
                raise ValueError("streams must not be empty")
        if any(not hasattr(worker_stream, "cuda_stream") for worker_stream in worker_streams):
            raise TypeError("every stream must be a PyTorch CUDA stream")

        with torch.cuda.stream(producer_stream):
            if output_values is None:
                output_values = torch.empty(expected_shape, dtype=torch.float32, device=x.device)

        for worker_stream in worker_streams:
            if worker_stream != producer_stream:
                worker_stream.wait_stream(producer_stream)

        launch_streams = tuple(Stream.from_handle(worker_stream.cuda_stream) for worker_stream in worker_streams)
        num_columns = x.shape[0]
        leading_dim = x.shape[1]
        leaf_words_stride = leaf_words.stride(0)
        x_buffer = self._buffer(x)
        leaf_masks_buffer = self._buffer(leaf_masks)
        leaf_words_buffer = self._buffer(leaf_words)

        for kernel_idx, (kernel, instantiation) in enumerate(zip(self.kernels, self.instantiations, strict=True)):
            worker_index = kernel_idx % len(worker_streams)
            grid = (num_rows + instantiation.tile_rows - 1) // instantiation.tile_rows
            shared_stride = num_columns | 1
            shared_bytes = instantiation.tile_rows * shared_stride * np.dtype(np.float32).itemsize
            kernel_output = output_values[kernel_idx * asts_per_kernel : (kernel_idx + 1) * asts_per_kernel]
            launch(
                launch_streams[worker_index],
                LaunchConfig(grid=grid, block=instantiation.threads_per_cta, shmem_size=shared_bytes),
                kernel,
                x_buffer,
                np.uint64(num_rows),
                np.uint32(num_columns),
                np.uint64(leading_dim),
                leaf_masks_buffer,
                leaf_words_buffer,
                np.uint64(leaf_words_stride),
                np.uint32(num_settings),
                np.uint32(asts_per_kernel),
                self._buffer(kernel_output),
            )

        for worker_stream in worker_streams:
            for tensor in (*tensors, output_values):
                tensor.record_stream(worker_stream)
        return output_values
