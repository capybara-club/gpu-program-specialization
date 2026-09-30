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
from enum import IntEnum
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .ast import BinaryAst, ast_arrays
from ._impl import _implicit_sindy as _native


FEATURES = 32
LEAVES = 8


_INTERNAL_CONSTANTS_SOURCE = """
#ifndef IMPLICIT_SINDY_INTERNAL_CONSTANTS_H
#define IMPLICIT_SINDY_INTERNAL_CONSTANTS_H
enum {
    IMPLICIT_SINDY_INTERNAL_FEATURES = 32,
    IMPLICIT_SINDY_INTERNAL_LEAVES = 8,
    IMPLICIT_SINDY_INTERNAL_PRIMITIVE_FEATURES_MAX = 32,
    IMPLICIT_SINDY_INTERNAL_ROWS_PER_TILE = 128,
    IMPLICIT_SINDY_INTERNAL_TARGET_RHS_MAX = 32,
    IMPLICIT_SINDY_INTERNAL_STLSQ_MAX_ITERATIONS = 32
};
#endif
"""


class KernelsPerModule(IntEnum):
    ONE = 1
    FOUR = 4


@dataclass(frozen=True)
class GramOutputs:
    gram: object
    gram_storage: object
    x_sum: object
    xty: object
    y_sum: object
    yy: object


def _torch_module():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("implicit_sindy gram execution requires PyTorch") from exc
    return torch


def _cuda_core_module():
    try:
        from cuda.core import Device, Program, ProgramOptions
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FusedSINDy runtime Gram compilation requires cuda-core==1.0.1 "
            "and cuda-bindings matching the active CUDA/PyTorch environment. "
            "Install a CUDA-enabled PyTorch wheel first, or install "
            "fused-sindy[cu12] / fused-sindy[cu13] for a standalone "
            "CUDA Python bindings stack."
        ) from exc
    return Device, Program, ProgramOptions


def _read_gram_header_source() -> str:
    resource = resources.files("implicit_sindy") / "cuda" / "implicit_feature_gram.cuh"
    if resource.is_file():
        return resource.read_text(encoding="utf-8")

    source_tree_path = Path(__file__).resolve().parents[2] / "cuda" / "implicit_feature_gram.cuh"
    if source_tree_path.is_file():
        return source_tree_path.read_text(encoding="utf-8")

    raise RuntimeError("implicit_feature_gram.cuh is missing from the installed FusedSINDy package")


def _processed_gram_header_source() -> str:
    source = _read_gram_header_source()
    source = source.replace('#include "implicit_sindy_internal_constants.h"', _INTERNAL_CONSTANTS_SOURCE)
    return source


def _kernel_args_source() -> str:
    return """\
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const float* targets,
    int64_t target_rhs_stride,
    int64_t num_target_rhs,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* gram,
    int64_t gram_col_stride,
    float* x_sum,
    float* xty,
    int64_t xty_rhs_stride,
    float* y_sum,
    float* yy
"""


def _kernel_forward_args_source() -> str:
    return """\
        primitive_features,
        row_count,
        primitive_feature_stride,
        num_primitive_features,
        targets,
        target_rhs_stride,
        num_target_rhs,
        leaf_masks,
        leaf_words,
        leaf_words_feature_stride,
        gram,
        gram_col_stride,
        x_sum,
        xty,
        xty_rhs_stride,
        y_sum,
        yy
"""


def _generate_gram_template_source(kernels_per_module: int) -> str:
    if kernels_per_module < 1 or kernels_per_module > 64:
        raise ValueError("kernels_per_module must be in [1, 64]")

    parts: list[str] = [
        "#define IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL 1\n",
        _processed_gram_header_source(),
        "\n",
    ]

    for kernel_idx in range(kernels_per_module):
        parts.append(f"""\
extern "C" __device__ __noinline__
void
implicit_feature_eval_{kernel_idx}(
    float* implicit_panel,
    const float* const* leaf_ptrs,
    const int* leaf_strides,
    int active_rows
);

namespace implicit_sindy_feature_gram_kernel {{
template <>
__device__ __forceinline__
void
implicit_feature_eval_selector<{kernel_idx}>(
    float* implicit_panel,
    const float* const* leaf_ptrs,
    const int* leaf_strides,
    int active_rows
) {{
    ::implicit_feature_eval_{kernel_idx}(
        implicit_panel,
        leaf_ptrs,
        leaf_strides,
        active_rows
    );
}}
}}  // namespace implicit_sindy_feature_gram_kernel

""")

    for kernel_idx in range(kernels_per_module):
        kernel_name = "implicit_feature_gram_kernel" if kernels_per_module == 1 else f"implicit_feature_gram_kernel_{kernel_idx}"
        parts.append(f"""\
extern "C" __global__
__launch_bounds__(implicit_sindy_feature_gram_kernel::kThreads)
void
{kernel_name}(
""")
        parts.append(_kernel_args_source())
        parts.append(f""") {{
    implicit_sindy_feature_gram_kernel::implicit_feature_gram_kernel_device<{kernel_idx}>(
""")
        parts.append(_kernel_forward_args_source())
        parts.append("""\
    );
}

""")

    return "".join(parts)


def _compile_gram_template_ptx(
    kernels_per_module: int,
    device,
) -> bytes:
    Device, Program, ProgramOptions = _cuda_core_module()
    dev = Device(int(device.index))
    dev.set_current()
    source = _generate_gram_template_source(kernels_per_module)
    options = ProgramOptions(
        name=f"implicit_sindy_gram_template_{kernels_per_module}",
        arch=f"sm_{dev.arch}",
        relocatable_device_code=True,
        std="c++17",
        use_fast_math=True,
    )
    program = Program(source, code_type="c++", options=options)
    object_code = program.compile("ptx")
    return bytes(object_code.code)


def _device_index(torch, device) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return int(device)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    if torch_device.index is None:
        return int(torch.cuda.current_device())
    return int(torch_device.index)


def _check_cuda_tensor(torch, tensor, name: str, dtype, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")


def _check_same_device(tensors: Sequence[object], device) -> None:
    for tensor in tensors:
        if tensor.device != device:
            raise ValueError("all tensors must be on the compiled gram kernel device")


class CompiledGramKernels:
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
        self._init(
            asts,
            kernels_per_module=kernels_per_module,
            device=device,
            worker_count=worker_count,
            scratch_bytes_per_worker=scratch_bytes_per_worker,
            nvptx_options=nvptx_options,
            gram_template_ptx=None,
        )

    @classmethod
    def _from_gram_template_ptx(
        cls,
        asts: Iterable[BinaryAst],
        *,
        kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
        device=None,
        worker_count: int = 0,
        scratch_bytes_per_worker: int = 64 * 1024 * 1024,
        nvptx_options: Sequence[str] = (),
        gram_template_ptx: bytes,
    ) -> CompiledGramKernels:
        compiled = cls.__new__(cls)
        compiled._init(
            asts,
            kernels_per_module=kernels_per_module,
            device=device,
            worker_count=worker_count,
            scratch_bytes_per_worker=scratch_bytes_per_worker,
            nvptx_options=nvptx_options,
            gram_template_ptx=gram_template_ptx,
        )
        return compiled

    def _init(
        self,
        asts: Iterable[BinaryAst],
        *,
        kernels_per_module: KernelsPerModule | int,
        device,
        worker_count: int,
        scratch_bytes_per_worker: int,
        nvptx_options: Sequence[str],
        gram_template_ptx: bytes | None,
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
        if self.num_asts % FEATURES != 0:
            raise ValueError("asts length must be a multiple of 32")
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
            if gram_template_ptx is None:
                gram_template_ptx = _compile_gram_template_ptx(
                    self.kernels_per_module,
                    self.device,
                )
            self._handle = _native.GramKernelSet(
                unary.tobytes(order="C"),
                binary.tobytes(order="C"),
                gram_template_ptx,
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

    def __enter__(self) -> CompiledGramKernels:
        if self._closed:
            raise RuntimeError("CompiledGramKernels is closed")
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

    def run_gram(
        self,
        primitive_features,
        targets,
        leaf_masks,
        leaf_words,
        *,
        cohort_index: int = 0,
        stream=None,
    ) -> GramOutputs:
        torch = _torch_module()
        if self._closed:
            raise RuntimeError("CompiledGramKernels is closed")
        _check_cuda_tensor(torch, primitive_features, "primitive_features", torch.float32, 2)
        if targets.ndim == 1:
            targets = targets.reshape(1, targets.shape[0])
        _check_cuda_tensor(torch, targets, "targets", torch.float32, 2)
        _check_cuda_tensor(torch, leaf_masks, "leaf_masks", torch.int32, 2)
        _check_cuda_tensor(torch, leaf_words, "leaf_words", torch.int32, 3)
        _check_same_device(
            (primitive_features, targets, leaf_masks, leaf_words),
            self.device,
        )

        if primitive_features.stride(1) != 1:
            raise ValueError("primitive_features must have stride 1 along the row dimension")
        if targets.stride(1) != 1:
            raise ValueError("targets must have stride 1 along the row dimension")
        if not leaf_masks.is_contiguous():
            raise ValueError("leaf_masks must be contiguous with shape [settings, 32]")
        if not leaf_words.is_contiguous():
            raise ValueError("leaf_words must be contiguous with shape [settings, 32, 8]")

        num_settings = int(leaf_masks.shape[0])
        num_primitive_features = int(primitive_features.shape[0])
        row_count = int(primitive_features.shape[1])
        num_target_rhs = int(targets.shape[0])
        if num_settings <= 0:
            raise ValueError("num_settings must be positive")
        if leaf_masks.shape != (num_settings, FEATURES):
            raise ValueError("leaf_masks must have shape [settings, 32]")
        if leaf_words.shape != (num_settings, FEATURES, LEAVES):
            raise ValueError("leaf_words must have shape [settings, 32, 8]")
        if targets.shape[1] != row_count:
            raise ValueError("targets row count must match primitive_features")
        if cohort_index < 0 or cohort_index >= self._num_cohorts:
            raise ValueError("cohort_index is out of range")
        if num_primitive_features <= 0 or num_primitive_features > 32:
            raise ValueError("primitive feature count must be in [1, 32]")
        if num_target_rhs <= 0 or num_target_rhs > 32:
            raise ValueError("target RHS count must be in [1, 32]")

        gram_storage = torch.empty((num_settings, FEATURES, FEATURES), device=self.device, dtype=torch.float32)
        gram = gram_storage.as_strided(
            (num_settings, FEATURES, FEATURES),
            (FEATURES * FEATURES, 1, FEATURES),
        )
        x_sum = torch.empty((num_settings, FEATURES), device=self.device, dtype=torch.float32)
        xty = torch.empty((num_settings, num_target_rhs, FEATURES), device=self.device, dtype=torch.float32)
        y_sum = torch.empty((num_settings, num_target_rhs), device=self.device, dtype=torch.float32)
        yy = torch.empty((num_settings, num_target_rhs), device=self.device, dtype=torch.float32)

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
            int(targets.data_ptr()),
            int(targets.stride(0)),
            num_target_rhs,
            int(leaf_masks.data_ptr()),
            int(leaf_words.data_ptr()),
            int(leaf_words.stride(1)),
            int(gram_storage.data_ptr()),
            FEATURES,
            int(x_sum.data_ptr()),
            int(xty.data_ptr()),
            FEATURES,
            int(y_sum.data_ptr()),
            int(yy.data_ptr()),
        )

        return GramOutputs(
            gram=gram,
            gram_storage=gram_storage,
            x_sum=x_sum,
            xty=xty,
            y_sum=y_sum,
            yy=yy,
        )


def compile_gram_kernels(
    asts: Iterable[BinaryAst],
    *,
    kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
    device=None,
    worker_count: int = 0,
    scratch_bytes_per_worker: int = 64 * 1024 * 1024,
    nvptx_options: Sequence[str] = (),
) -> CompiledGramKernels:
    return CompiledGramKernels(
        asts,
        kernels_per_module=kernels_per_module,
        device=device,
        worker_count=worker_count,
        scratch_bytes_per_worker=scratch_bytes_per_worker,
        nvptx_options=nvptx_options,
    )


def _compile_gram_kernels_with_template_ptx(
    asts: Iterable[BinaryAst],
    gram_template_ptx: bytes,
    *,
    kernels_per_module: KernelsPerModule | int = KernelsPerModule.ONE,
    device=None,
    worker_count: int = 0,
    scratch_bytes_per_worker: int = 64 * 1024 * 1024,
    nvptx_options: Sequence[str] = (),
) -> CompiledGramKernels:
    return CompiledGramKernels._from_gram_template_ptx(
        asts,
        kernels_per_module=kernels_per_module,
        device=device,
        worker_count=worker_count,
        scratch_bytes_per_worker=scratch_bytes_per_worker,
        nvptx_options=nvptx_options,
        gram_template_ptx=gram_template_ptx,
    )
