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
from time import perf_counter
from typing import Iterable

from cuda.core import Buffer, Device, LaunchConfig, ObjectCode, launch
import numpy as np

from .compiler import (
    AffineStatsRecipe,
    DynamicConstantSSERecipe,
    DynamicLeafSSERecipe,
    GramStatsRecipe,
    MaterializeRecipe,
    SSERecipe,
)

try:
    import _secant_native as _native
except ImportError as error:
    raise ImportError(
        "the Secant nanobind extension is unavailable; build with SECANT_BUILD_PYTHON_BINDINGS=ON"
    ) from error


def _f32_matrix(name: str, value: object, outer_size: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional float32 array")
    if outer_size is not None and array.shape[0] != outer_size:
        raise ValueError(f"{name} must have {outer_size} outer elements")
    return array


def _host_buffer(array: np.ndarray) -> Buffer:
    return Buffer.from_handle(array.ctypes.data, array.nbytes, owner=array)


def _device_slice(buffer: Buffer, byte_offset: int, byte_size: int) -> Buffer:
    return Buffer.from_handle(buffer.handle + byte_offset, byte_size, owner=buffer)


def _release(buffers: Iterable[Buffer], stream) -> None:
    for buffer in buffers:
        buffer.close(stream)
    stream.sync()
    stream.close()


@dataclass(frozen=True, slots=True)
class SSERunResult:
    """Result and timings from one resident SSE execution."""

    output: np.ndarray
    gpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class BulkSSERunResult:
    """Output and native pipeline statistics from one bulk SSE runner call."""

    output: np.ndarray
    stats: dict[str, int | float]


class _CubinModule:
    def __init__(self, cubin: bytes, kernel_prefix: str, num_kernels: int, device_id: int) -> None:
        if not cubin.startswith(b"\x7fELF"):
            raise ValueError("cubin must contain an ELF CUBIN image")
        self.device = Device(device_id)
        self.device.set_current()
        self.object_code = ObjectCode.from_cubin(cubin, name=f"secant_{kernel_prefix}")
        self.kernels = tuple(
            self.object_code.get_kernel(f"secant_cubin_{kernel_prefix}_{kernel_idx:03d}")
            for kernel_idx in range(num_kernels)
        )


class MaterializeModule(_CubinModule):
    def __init__(self, cubin: bytes, recipe: MaterializeRecipe, *, device_id: int = 0, threads_per_block: int = 128) -> None:
        super().__init__(cubin, "materialize", recipe.num_kernels, device_id)
        if threads_per_block <= 0 or threads_per_block > 1024:
            raise ValueError("threads_per_block must be in [1, 1024]")
        self.recipe = recipe
        self.threads_per_block = threads_per_block

    def run(self, input_columns: np.ndarray) -> np.ndarray:
        packed_input = _f32_matrix("input_columns", input_columns, self.recipe.num_inputs)
        num_rows = packed_input.shape[1]
        num_asts = self.recipe.num_kernels * self.recipe.asts_per_kernel
        output = np.empty((num_asts, num_rows), dtype=np.float32)
        stream = self.device.create_stream()
        buffers: list[Buffer] = []

        try:
            device_input = self.device.allocate(packed_input.nbytes, stream=stream)
            buffers.append(device_input)
            device_output = self.device.allocate(output.nbytes, stream=stream)
            buffers.append(device_output)
            device_input.copy_from(_host_buffer(packed_input), stream=stream)
            grid_size = (num_rows + self.threads_per_block - 1) // self.threads_per_block
            kernel_output_bytes = self.recipe.asts_per_kernel * num_rows * output.itemsize
            for kernel_idx, kernel in enumerate(self.kernels):
                output_slice = _device_slice(device_output, kernel_idx * kernel_output_bytes, kernel_output_bytes)
                launch(
                    stream,
                    LaunchConfig(grid=grid_size, block=self.threads_per_block),
                    kernel,
                    device_input,
                    np.uint64(num_rows),
                    np.uint64(num_rows),
                    np.uint64(self.recipe.asts_per_kernel),
                    output_slice,
                    np.uint64(num_rows),
                )
            device_output.copy_to(_host_buffer(output), stream=stream)
            stream.sync()
            return output
        finally:
            _release(buffers, stream)


class SSEModule(_CubinModule):
    def __init__(self, cubin: bytes, recipe: SSERecipe, *, device_id: int = 0) -> None:
        super().__init__(cubin, "sse", recipe.num_kernels, device_id)
        self.recipe = recipe

    def run(self, input_columns: np.ndarray, targets: np.ndarray) -> np.ndarray:
        with self.resident(input_columns, targets) as execution:
            return execution.run().output

    def resident(
        self,
        input_columns: np.ndarray,
        targets: np.ndarray,
        *,
        num_streams: int = 1,
    ) -> ResidentSSE:
        """Upload input data once and create a reusable multi-stream SSE execution."""

        return ResidentSSE(self, input_columns, targets, num_streams=num_streams)


class AffineStatsModule(_CubinModule):
    """One specialized static affine-statistics CUBIN module."""

    def __init__(self, cubin: bytes, recipe: AffineStatsRecipe, *, device_id: int = 0) -> None:
        super().__init__(cubin, "affine_stats", recipe.num_kernels, device_id)
        self.recipe = recipe

    def run(self, input_columns: np.ndarray, targets: np.ndarray) -> np.ndarray:
        packed_input = _f32_matrix("input_columns", input_columns, self.recipe.num_inputs)
        packed_targets = _f32_matrix("targets", targets)
        if packed_targets.shape[0] > self.recipe.num_targets:
            raise ValueError("targets exceeds the affine-statistics recipe capacity")
        if packed_targets.shape[1] != packed_input.shape[1]:
            raise ValueError("input and target row counts must match")
        num_rows = packed_input.shape[1]
        num_asts = self.recipe.num_kernels * self.recipe.asts_per_kernel
        statistic_count = 2 + packed_targets.shape[0]
        output = np.empty((num_asts, statistic_count), dtype=np.float32)
        stream = self.device.create_stream()
        buffers: list[Buffer] = []

        try:
            device_input = self.device.allocate(packed_input.nbytes, stream=stream)
            buffers.append(device_input)
            device_targets = self.device.allocate(packed_targets.nbytes, stream=stream)
            buffers.append(device_targets)
            device_output = self.device.allocate(output.nbytes, stream=stream)
            buffers.append(device_output)
            device_input.copy_from(_host_buffer(packed_input), stream=stream)
            device_targets.copy_from(_host_buffer(packed_targets), stream=stream)
            device_output.fill(0, stream=stream)
            grid_size = (num_rows + self.recipe.tile_rows - 1) // self.recipe.tile_rows
            kernel_output_bytes = self.recipe.asts_per_kernel * statistic_count * output.itemsize
            for kernel_idx, kernel in enumerate(self.kernels):
                output_slice = _device_slice(device_output, kernel_idx * kernel_output_bytes, kernel_output_bytes)
                launch(
                    stream,
                    LaunchConfig(grid=grid_size, block=self.recipe.threads_per_block),
                    kernel,
                    device_input,
                    np.uint64(num_rows),
                    device_targets,
                    np.uint64(num_rows),
                    np.uint64(num_rows),
                    np.uint64(self.recipe.asts_per_kernel),
                    np.uint64(packed_targets.shape[0]),
                    output_slice,
                    np.uint64(statistic_count),
                )
            device_output.copy_to(_host_buffer(output), stream=stream)
            stream.sync()
            return output
        finally:
            _release(buffers, stream)


class GramStatsModule(_CubinModule):
    """One specialized static cohort Gram-statistics CUBIN module."""

    def __init__(self, cubin: bytes, recipe: GramStatsRecipe, *, device_id: int = 0) -> None:
        super().__init__(cubin, "gram_stats", recipe.num_kernels, device_id)
        self.recipe = recipe

    def run(self, input_columns: np.ndarray, targets: np.ndarray) -> np.ndarray:
        packed_input = _f32_matrix("input_columns", input_columns, self.recipe.num_inputs)
        packed_targets = _f32_matrix("targets", targets)
        if packed_targets.shape[0] > self.recipe.num_targets:
            raise ValueError("targets exceeds the Gram-statistics recipe capacity")
        if packed_targets.shape[1] != packed_input.shape[1]:
            raise ValueError("input and target row counts must match")
        num_rows = packed_input.shape[1]
        capacity = self.recipe.asts_per_kernel
        statistic_count = capacity + capacity * capacity + capacity * packed_targets.shape[0]
        output = np.empty((self.recipe.num_kernels, statistic_count), dtype=np.float32)
        stream = self.device.create_stream()
        buffers: list[Buffer] = []

        try:
            device_input = self.device.allocate(packed_input.nbytes, stream=stream)
            buffers.append(device_input)
            device_targets = self.device.allocate(packed_targets.nbytes, stream=stream)
            buffers.append(device_targets)
            device_output = self.device.allocate(output.nbytes, stream=stream)
            buffers.append(device_output)
            device_input.copy_from(_host_buffer(packed_input), stream=stream)
            device_targets.copy_from(_host_buffer(packed_targets), stream=stream)
            device_output.fill(0, stream=stream)
            grid_size = (num_rows + self.recipe.tile_rows - 1) // self.recipe.tile_rows
            for kernel_idx, kernel in enumerate(self.kernels):
                output_slice = _device_slice(
                    device_output,
                    kernel_idx * statistic_count * output.itemsize,
                    statistic_count * output.itemsize,
                )
                launch(
                    stream,
                    LaunchConfig(grid=grid_size, block=self.recipe.threads_per_block),
                    kernel,
                    device_input,
                    np.uint64(num_rows),
                    device_targets,
                    np.uint64(num_rows),
                    np.uint64(num_rows),
                    np.uint64(capacity),
                    np.uint64(packed_targets.shape[0]),
                    output_slice,
                )
            device_output.copy_to(_host_buffer(output), stream=stream)
            stream.sync()
            return output
        finally:
            _release(buffers, stream)


class ResidentSSE:
    """Reusable SSE launch state with resident device inputs and event timing."""

    def __init__(
        self,
        module: SSEModule,
        input_columns: np.ndarray,
        targets: np.ndarray,
        *,
        num_streams: int,
    ) -> None:
        if isinstance(num_streams, bool) or not isinstance(num_streams, int) or num_streams <= 0:
            raise ValueError("num_streams must be a positive integer")

        packed_input = _f32_matrix("input_columns", input_columns, module.recipe.num_inputs)
        packed_targets = _f32_matrix("targets", targets, module.recipe.num_targets)
        if packed_targets.shape[1] != packed_input.shape[1]:
            raise ValueError("input and target row counts must match")

        self.module = module
        self.num_rows = packed_input.shape[1]
        self.num_asts = module.recipe.num_kernels * module.recipe.asts_per_kernel
        self.output = np.empty((self.num_asts, module.recipe.num_targets), dtype=np.float32)
        self.control_stream = module.device.create_stream()
        self.work_streams = tuple(module.device.create_stream() for _ in range(num_streams))
        self.buffers: list[Buffer] = []
        self.closed = False

        upload_start = perf_counter()
        try:
            self.device_input = module.device.allocate(packed_input.nbytes, stream=self.control_stream)
            self.buffers.append(self.device_input)
            self.device_targets = module.device.allocate(packed_targets.nbytes, stream=self.control_stream)
            self.buffers.append(self.device_targets)
            self.device_output = module.device.allocate(self.output.nbytes, stream=self.control_stream)
            self.buffers.append(self.device_output)
            self.device_input.copy_from(_host_buffer(packed_input), stream=self.control_stream)
            self.device_targets.copy_from(_host_buffer(packed_targets), stream=self.control_stream)
            self.device_output.fill(0, stream=self.control_stream)
            self.control_stream.sync()
        except Exception:
            self.close()
            raise
        self.upload_seconds = perf_counter() - upload_start

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("resident SSE execution is closed")

    def _launch(self, iterations: int) -> None:
        recipe = self.module.recipe
        grid_size = (self.num_rows + recipe.tile_rows - 1) // recipe.tile_rows
        kernel_output_bytes = recipe.asts_per_kernel * recipe.num_targets * self.output.itemsize

        for _ in range(iterations):
            for kernel_idx, kernel in enumerate(self.module.kernels):
                stream = self.work_streams[kernel_idx % len(self.work_streams)]
                output_slice = _device_slice(
                    self.device_output,
                    kernel_idx * kernel_output_bytes,
                    kernel_output_bytes,
                )
                launch(
                    stream,
                    LaunchConfig(grid=grid_size, block=recipe.threads_per_block),
                    kernel,
                    self.device_input,
                    np.uint64(self.num_rows),
                    self.device_targets,
                    np.uint64(self.num_rows),
                    np.uint64(self.num_rows),
                    np.uint64(recipe.asts_per_kernel),
                    np.uint64(recipe.num_targets),
                    output_slice,
                    np.uint64(recipe.num_targets),
                )

    def _zero_output(self) -> None:
        self.device_output.fill(0, stream=self.control_stream)
        self.control_stream.sync()

    def run(self, *, iterations: int = 1, warmups: int = 0) -> SSERunResult:
        """Run resident kernels and return normalized output plus GPU and wall time."""

        self._check_open()
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
            raise ValueError("warmups must be a nonnegative integer")

        if warmups:
            self._zero_output()
            self._launch(warmups)
            for stream in self.work_streams:
                stream.sync()

        wall_start = perf_counter()
        self._zero_output()
        start = self.module.device.create_event({"timing_enabled": True})
        stop = self.module.device.create_event({"timing_enabled": True})
        finished = tuple(self.module.device.create_event() for _ in self.work_streams)

        try:
            self.control_stream.record(start)
            for stream in self.work_streams:
                stream.wait(start)
            self._launch(iterations)
            for stream, event in zip(self.work_streams, finished):
                stream.record(event)
                self.control_stream.wait(event)
            self.control_stream.record(stop)
            stop.sync()
            gpu_seconds = float(stop - start) * 1.0e-3

            self.device_output.copy_to(_host_buffer(self.output), stream=self.control_stream)
            self.control_stream.sync()
            normalized_output = self.output.copy()
            if iterations != 1:
                normalized_output /= np.float32(iterations)
            wall_seconds = perf_counter() - wall_start
            return SSERunResult(normalized_output, gpu_seconds, wall_seconds)
        finally:
            start.close()
            stop.close()
            for event in finished:
                event.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for stream in self.work_streams:
            stream.sync()
        for buffer in self.buffers:
            buffer.close(self.control_stream)
        self.control_stream.sync()
        for stream in self.work_streams:
            stream.close()
        self.control_stream.close()
        self.buffers.clear()

    def __enter__(self) -> ResidentSSE:
        self._check_open()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


class SSEBulkRunner:
    """Bulk AST-to-CUBIN/load/execute pipeline backed by Secant's native workers."""

    def __init__(
        self,
        template,
        *,
        num_workers: int,
        num_streams: int,
        device_id: int = 0,
    ) -> None:
        if template.shape != "sse":
            raise ValueError("SSEBulkRunner requires an SSE compiled template")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")
        if isinstance(num_streams, bool) or not isinstance(num_streams, int) or num_streams <= 0:
            raise ValueError("num_streams must be a positive integer")
        self.template = template
        self.device = Device(device_id)
        self.device.set_current()
        self.native = _native.CubinSSERunner(template.plan, template.cubin, num_workers, num_streams)
        self.closed = False

    def run_all(
        self,
        ast_data: bytearray | bytes | memoryview,
        ast_offsets: np.ndarray,
        input_columns: np.ndarray,
        targets: np.ndarray,
        *,
        routines=(),
    ) -> BulkSSERunResult:
        """Run a nonempty packed AST payload with up to the template target capacity."""

        if self.closed:
            raise RuntimeError("bulk SSE runner is closed")
        packed_input = _f32_matrix("input_columns", input_columns, self.template.recipe.num_inputs)
        packed_targets = _f32_matrix("targets", targets)
        if packed_targets.shape[0] > self.template.recipe.num_targets:
            raise ValueError(
                f"targets has {packed_targets.shape[0]} columns; template capacity is "
                f"{self.template.recipe.num_targets}"
            )
        if packed_targets.shape[1] != packed_input.shape[1]:
            raise ValueError("input and target row counts must match")

        ast_bytes = np.frombuffer(ast_data, dtype=np.uint8)
        offsets = np.ascontiguousarray(ast_offsets, dtype=np.uint64)
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError("ast_offsets must contain at least a start and end offset")
        num_asts = offsets.size - 1
        output = np.empty((num_asts, packed_targets.shape[0]), dtype=np.float32)
        stream = self.device.create_stream()
        buffers: list[Buffer] = []
        packed_routines = tuple(bytes(routine) for routine in routines)

        try:
            device_input = self.device.allocate(packed_input.nbytes, stream=stream)
            buffers.append(device_input)
            device_targets = self.device.allocate(packed_targets.nbytes, stream=stream)
            buffers.append(device_targets)
            device_output = self.device.allocate(output.nbytes, stream=stream)
            buffers.append(device_output)
            device_input.copy_from(_host_buffer(packed_input), stream=stream)
            device_targets.copy_from(_host_buffer(packed_targets), stream=stream)
            stream.sync()

            stats = self.native.run_all(
                packed_routines,
                ast_bytes,
                offsets,
                int(device_input.handle),
                packed_input.size,
                packed_input.shape[1],
                packed_targets.shape[0],
                int(device_targets.handle),
                packed_targets.size,
                packed_targets.shape[1],
                packed_input.shape[1],
                int(device_output.handle),
                output.size,
                packed_targets.shape[0],
            )
            device_output.copy_to(_host_buffer(output), stream=stream)
            stream.sync()
            return BulkSSERunResult(output, dict(stats))
        finally:
            _release(buffers, stream)

    def close(self) -> None:
        if self.closed:
            return
        self.native.close()
        self.closed = True

    def __enter__(self) -> SSEBulkRunner:
        if self.closed:
            raise RuntimeError("bulk SSE runner is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


class DynamicConstantSSEModule(_CubinModule):
    def __init__(self, cubin: bytes, recipe: DynamicConstantSSERecipe, *, device_id: int = 0) -> None:
        super().__init__(cubin, "dynamic_constant_sse", recipe.num_kernels, device_id)
        self.recipe = recipe

    def run(
        self,
        input_columns: np.ndarray,
        constant_settings: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        packed_input = _f32_matrix("input_columns", input_columns, self.recipe.num_input_columns)
        packed_constants = _f32_matrix(
            "constant_settings",
            constant_settings,
            self.recipe.num_input_constants,
        )
        packed_targets = _f32_matrix("targets", targets, self.recipe.num_targets)
        if packed_targets.shape[1] != packed_input.shape[1]:
            raise ValueError("input and target row counts must match")
        num_rows = packed_input.shape[1]
        num_settings = packed_constants.shape[1]
        num_asts = self.recipe.num_kernels * self.recipe.asts_per_kernel
        output = np.empty((num_asts, self.recipe.num_targets, num_settings), dtype=np.float32)
        stream = self.device.create_stream()
        buffers: list[Buffer] = []

        try:
            device_input = self.device.allocate(packed_input.nbytes, stream=stream)
            buffers.append(device_input)
            device_constants = self.device.allocate(packed_constants.nbytes, stream=stream)
            buffers.append(device_constants)
            device_targets = self.device.allocate(packed_targets.nbytes, stream=stream)
            buffers.append(device_targets)
            device_output = self.device.allocate(output.nbytes, stream=stream)
            buffers.append(device_output)
            device_input.copy_from(_host_buffer(packed_input), stream=stream)
            device_constants.copy_from(_host_buffer(packed_constants), stream=stream)
            device_targets.copy_from(_host_buffer(packed_targets), stream=stream)
            device_output.fill(0, stream=stream)
            grid_size = (num_rows + self.recipe.tile_rows - 1) // self.recipe.tile_rows
            kernel_output_bytes = (
                self.recipe.asts_per_kernel * self.recipe.num_targets * num_settings * output.itemsize
            )
            for kernel_idx, kernel in enumerate(self.kernels):
                output_slice = _device_slice(device_output, kernel_idx * kernel_output_bytes, kernel_output_bytes)
                launch(
                    stream,
                    LaunchConfig(grid=grid_size, block=self.recipe.threads_per_block),
                    kernel,
                    device_input,
                    np.uint64(num_rows),
                    device_constants,
                    np.uint64(num_settings),
                    np.uint64(num_settings),
                    device_targets,
                    np.uint64(num_rows),
                    np.uint64(num_rows),
                    np.uint64(self.recipe.asts_per_kernel),
                    np.uint64(self.recipe.num_targets),
                    output_slice,
                    np.uint64(num_settings),
                )
            device_output.copy_to(_host_buffer(output), stream=stream)
            stream.sync()
            return output
        finally:
            _release(buffers, stream)


class DynamicLeafSSEModule(_CubinModule):
    def __init__(self, cubin: bytes, recipe: DynamicLeafSSERecipe, *, device_id: int = 0) -> None:
        super().__init__(cubin, "dynamic_leaf_sse", recipe.num_kernels, device_id)
        self.recipe = recipe

    def run(
        self,
        input_columns: np.ndarray,
        leaf_masks: np.ndarray,
        leaf_words: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        packed_input = _f32_matrix("input_columns", input_columns)
        if packed_input.shape[0] > self.recipe.num_input_columns:
            raise ValueError(
                f"input_columns has {packed_input.shape[0]} columns; template capacity is "
                f"{self.recipe.num_input_columns}"
            )
        packed_masks = np.ascontiguousarray(leaf_masks, dtype=np.uint32)
        packed_words = np.ascontiguousarray(leaf_words, dtype=np.uint32)
        packed_targets = _f32_matrix("targets", targets, self.recipe.num_targets)
        if packed_targets.shape[1] != packed_input.shape[1]:
            raise ValueError("input and target row counts must match")
        if packed_masks.ndim != 1 or packed_masks.size == 0:
            raise ValueError("leaf_masks must be a nonempty one-dimensional uint32 array")
        if packed_words.ndim != 2 or packed_words.shape != (packed_masks.size, self.recipe.num_dynamic_leaves):
            raise ValueError("leaf_words must have shape [num_settings, num_dynamic_leaves]")
        for setting, mask in enumerate(packed_masks):
            for leaf_index, word in enumerate(packed_words[setting]):
                if int(mask) & (1 << leaf_index) and int(word) >= packed_input.shape[0]:
                    raise ValueError(f"dynamic column leaf {leaf_index} is out of range at setting {setting}")

        num_rows = packed_input.shape[1]
        num_settings = packed_masks.size
        num_asts = self.recipe.num_kernels * self.recipe.asts_per_kernel
        output = np.empty((num_asts, self.recipe.num_targets, num_settings), dtype=np.float32)
        stream = self.device.create_stream()
        buffers: list[Buffer] = []

        try:
            device_input = self.device.allocate(packed_input.nbytes, stream=stream)
            buffers.append(device_input)
            device_masks = self.device.allocate(packed_masks.nbytes, stream=stream)
            buffers.append(device_masks)
            device_words = self.device.allocate(packed_words.nbytes, stream=stream)
            buffers.append(device_words)
            device_targets = self.device.allocate(packed_targets.nbytes, stream=stream)
            buffers.append(device_targets)
            device_output = self.device.allocate(output.nbytes, stream=stream)
            buffers.append(device_output)
            device_input.copy_from(_host_buffer(packed_input), stream=stream)
            device_masks.copy_from(_host_buffer(packed_masks), stream=stream)
            device_words.copy_from(_host_buffer(packed_words), stream=stream)
            device_targets.copy_from(_host_buffer(packed_targets), stream=stream)
            device_output.fill(0, stream=stream)
            grid_size = (num_rows + self.recipe.tile_rows - 1) // self.recipe.tile_rows
            shared_stride = self.recipe.num_input_columns | 1
            shared_bytes = (shared_stride + self.recipe.num_targets) * self.recipe.tile_rows * output.itemsize
            kernel_output_bytes = (
                self.recipe.asts_per_kernel * self.recipe.num_targets * num_settings * output.itemsize
            )
            for kernel_idx, kernel in enumerate(self.kernels):
                output_slice = _device_slice(device_output, kernel_idx * kernel_output_bytes, kernel_output_bytes)
                launch(
                    stream,
                    LaunchConfig(
                        grid=grid_size,
                        block=self.recipe.threads_per_block,
                        shmem_size=shared_bytes,
                    ),
                    kernel,
                    device_input,
                    np.uint64(packed_input.shape[0]),
                    np.uint64(num_rows),
                    device_masks,
                    device_words,
                    np.uint64(self.recipe.num_dynamic_leaves),
                    np.uint64(num_settings),
                    device_targets,
                    np.uint64(num_rows),
                    np.uint64(num_rows),
                    np.uint64(self.recipe.asts_per_kernel),
                    np.uint64(self.recipe.num_targets),
                    output_slice,
                    np.uint64(num_settings),
                )
            device_output.copy_to(_host_buffer(output), stream=stream)
            stream.sync()
            return output
        finally:
            _release(buffers, stream)
