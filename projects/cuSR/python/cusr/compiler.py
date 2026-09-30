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
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from cuda.core import Program, ProgramOptions


_FIRST_MARKER = 0x7FC0FFEE
_MARKER_STRIDE = 128
_MAX_KERNEL_INDEX = (0xFFFFFFFF - _FIRST_MARKER) // _MARKER_STRIDE
_AST_CAPACITIES = frozenset((8, 16, 32))
_TILE_ROWS = frozenset((64, 128, 256))
_CTA_THREADS = frozenset((64, 128, 256))
_SUPPORTED_ARCH_MAJORS = frozenset((8, 9, 10, 12))
_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "tile_static_mse_template"
    / "cusr_tile_static_mse_template.cuh"
)
_TEMPLATE_KERNEL = "cusr_tile_static_mse_template_kernel_f32"
_EVAL_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "tile_static_eval_template"
    / "cusr_tile_static_eval_template.cuh"
)
_EVAL_TEMPLATE_KERNEL = "cusr_tile_static_eval_template_kernel_f32"


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _validate_arch(arch: str | None) -> None:
    if arch is None:
        return
    if not isinstance(arch, str) or not arch.startswith("sm_") or not arch[3:].isdigit():
        raise ValueError("arch must have the form sm_<compute-capability>")

    capability = int(arch[3:])
    major = capability // 10
    minor = capability % 10
    if major not in _SUPPORTED_ARCH_MAJORS or minor >= 10:
        raise ValueError(f"unsupported cuSR architecture: {arch}")


@dataclass(frozen=True, slots=True)
class TileStaticMseInstantiation:
    kernel_index: int
    ast_capacity: int = 32
    tile_rows: int = 64
    threads_per_cta: int = 128

    def __post_init__(self) -> None:
        kernel_index = _require_int("kernel_index", self.kernel_index)
        ast_capacity = _require_int("ast_capacity", self.ast_capacity)
        tile_rows = _require_int("tile_rows", self.tile_rows)
        threads_per_cta = _require_int("threads_per_cta", self.threads_per_cta)

        if kernel_index < 0 or kernel_index > _MAX_KERNEL_INDEX:
            raise ValueError(f"kernel_index must be in [0, {_MAX_KERNEL_INDEX}]")
        if ast_capacity not in _AST_CAPACITIES:
            raise ValueError("ast_capacity must be 8, 16, or 32")
        if tile_rows not in _TILE_ROWS:
            raise ValueError("tile_rows must be 64, 128, or 256")
        if threads_per_cta not in _CTA_THREADS:
            raise ValueError("threads_per_cta must be 64, 128, or 256")

    @property
    def name_expression(self) -> str:
        return (
            f"{_TEMPLATE_KERNEL}<"
            f"{self.kernel_index}u, {self.ast_capacity}u, "
            f"{self.tile_rows}u, {self.threads_per_cta}u>"
        )


@dataclass(frozen=True, slots=True)
class CompiledTileStaticMseCubin:
    cubin: bytes
    name_expressions: tuple[str, ...]
    lowered_names: tuple[str, ...]
    log: str

    @property
    def size(self) -> int:
        return len(self.cubin)


@dataclass(frozen=True, slots=True)
class TileStaticEvalInstantiation:
    kernel_index: int
    ast_capacity: int = 32
    tile_rows: int = 64
    threads_per_cta: int = 128

    def __post_init__(self) -> None:
        kernel_index = _require_int("kernel_index", self.kernel_index)
        ast_capacity = _require_int("ast_capacity", self.ast_capacity)
        tile_rows = _require_int("tile_rows", self.tile_rows)
        threads_per_cta = _require_int("threads_per_cta", self.threads_per_cta)

        if kernel_index < 0 or kernel_index > _MAX_KERNEL_INDEX:
            raise ValueError(f"kernel_index must be in [0, {_MAX_KERNEL_INDEX}]")
        if ast_capacity not in _AST_CAPACITIES:
            raise ValueError("ast_capacity must be 8, 16, or 32")
        if tile_rows not in _TILE_ROWS:
            raise ValueError("tile_rows must be 64, 128, or 256")
        if threads_per_cta not in _CTA_THREADS:
            raise ValueError("threads_per_cta must be 64, 128, or 256")

    @property
    def name_expression(self) -> str:
        return (
            f"{_EVAL_TEMPLATE_KERNEL}<"
            f"{self.kernel_index}u, {self.ast_capacity}u, "
            f"{self.tile_rows}u, {self.threads_per_cta}u>"
        )


@dataclass(frozen=True, slots=True)
class CompiledTileStaticEvalCubin:
    cubin: bytes
    name_expressions: tuple[str, ...]
    lowered_names: tuple[str, ...]
    log: str

    @property
    def size(self) -> int:
        return len(self.cubin)


class CubinCompileError(RuntimeError):
    def __init__(self, message: str, log: str) -> None:
        super().__init__(message if not log else f"{message}\n{log.rstrip()}")
        self.log = log


def make_tile_static_mse_instantiations(
    num_kernels: int,
    *,
    ast_capacity: int = 32,
    tile_rows: int = 64,
    threads_per_cta: int = 128,
) -> tuple[TileStaticMseInstantiation, ...]:
    num_kernels = _require_int("num_kernels", num_kernels)
    if num_kernels <= 0:
        raise ValueError("num_kernels must be positive")
    if num_kernels - 1 > _MAX_KERNEL_INDEX:
        raise ValueError("kernel index range exceeds the marker encoding")

    return tuple(
        TileStaticMseInstantiation(
            kernel_index=offset,
            ast_capacity=ast_capacity,
            tile_rows=tile_rows,
            threads_per_cta=threads_per_cta,
        )
        for offset in range(num_kernels)
    )


def make_tile_static_eval_instantiations(
    num_kernels: int,
    *,
    ast_capacity: int = 32,
    tile_rows: int = 64,
    threads_per_cta: int = 128,
) -> tuple[TileStaticEvalInstantiation, ...]:
    num_kernels = _require_int("num_kernels", num_kernels)
    if num_kernels <= 0:
        raise ValueError("num_kernels must be positive")
    if num_kernels - 1 > _MAX_KERNEL_INDEX:
        raise ValueError("kernel index range exceeds the marker encoding")

    return tuple(
        TileStaticEvalInstantiation(
            kernel_index=offset,
            ast_capacity=ast_capacity,
            tile_rows=tile_rows,
            threads_per_cta=threads_per_cta,
        )
        for offset in range(num_kernels)
    )


def _decode_symbol(symbol: str | bytes) -> str:
    return symbol.decode("ascii") if isinstance(symbol, bytes) else symbol


def _compiler_log(log: StringIO) -> str:
    return log.getvalue().rstrip("\0")


def compile_tile_static_mse_cubin(
    instantiations: Sequence[TileStaticMseInstantiation],
    *,
    arch: str | None = None,
    template_path: str | Path = _TEMPLATE_PATH,
) -> CompiledTileStaticMseCubin:
    instantiations = tuple(instantiations)
    if not instantiations:
        raise ValueError("instantiations must not be empty")
    if any(not isinstance(item, TileStaticMseInstantiation) for item in instantiations):
        raise TypeError("every instantiation must be a TileStaticMseInstantiation")
    if tuple(item.kernel_index for item in instantiations) != tuple(range(len(instantiations))):
        raise ValueError("kernel_index values must be consecutive and ordered from zero")
    _validate_arch(arch)

    source_path = Path(template_path)
    source = source_path.read_text(encoding="ascii")
    name_expressions = tuple(item.name_expression for item in instantiations)
    log = StringIO()
    options = ProgramOptions(
        name=source_path.name,
        arch=arch,
        std="c++11",
        device_as_default_execution_space=True,
        ptxas_options=("--opt-level=1", "-v", "-warn-spills", "-Werror"),
        split_compile=1,
    )

    try:
        object_code = Program(source, "c++", options).compile(
            "cubin",
            name_expressions=name_expressions,
            logs=log,
        )
    except Exception as error:
        raise CubinCompileError("cuda.core failed to compile the tile-static MSE template", _compiler_log(log)) from error

    try:
        lowered_names = tuple(
            _decode_symbol(object_code.symbol_mapping[expression])
            for expression in name_expressions
        )
    except (KeyError, TypeError, UnicodeDecodeError) as error:
        raise CubinCompileError("cuda.core did not return every lowered template symbol", _compiler_log(log)) from error

    cubin = bytes(object_code.code)
    if not cubin.startswith(b"\x7fELF"):
        raise CubinCompileError("cuda.core returned an invalid cubin image", _compiler_log(log))

    return CompiledTileStaticMseCubin(
        cubin=cubin,
        name_expressions=name_expressions,
        lowered_names=lowered_names,
        log=_compiler_log(log),
    )


def compile_tile_static_eval_cubin(
    instantiations: Sequence[TileStaticEvalInstantiation],
    *,
    arch: str | None = None,
    template_path: str | Path = _EVAL_TEMPLATE_PATH,
) -> CompiledTileStaticEvalCubin:
    instantiations = tuple(instantiations)
    if not instantiations:
        raise ValueError("instantiations must not be empty")
    if any(not isinstance(item, TileStaticEvalInstantiation) for item in instantiations):
        raise TypeError("every instantiation must be a TileStaticEvalInstantiation")
    if tuple(item.kernel_index for item in instantiations) != tuple(range(len(instantiations))):
        raise ValueError("kernel_index values must be consecutive and ordered from zero")
    _validate_arch(arch)

    source_path = Path(template_path)
    source = source_path.read_text(encoding="ascii")
    name_expressions = tuple(item.name_expression for item in instantiations)
    log = StringIO()
    options = ProgramOptions(
        name=source_path.name,
        arch=arch,
        std="c++11",
        device_as_default_execution_space=True,
        ptxas_options=("--opt-level=1", "-v", "-warn-spills", "-Werror"),
        split_compile=1,
    )

    try:
        object_code = Program(source, "c++", options).compile(
            "cubin",
            name_expressions=name_expressions,
            logs=log,
        )
    except Exception as error:
        raise CubinCompileError("cuda.core failed to compile the tile-static eval template", _compiler_log(log)) from error

    try:
        lowered_names = tuple(
            _decode_symbol(object_code.symbol_mapping[expression])
            for expression in name_expressions
        )
    except (KeyError, TypeError, UnicodeDecodeError) as error:
        raise CubinCompileError("cuda.core did not return every lowered eval template symbol", _compiler_log(log)) from error

    cubin = bytes(object_code.code)
    if not cubin.startswith(b"\x7fELF"):
        raise CubinCompileError("cuda.core returned an invalid eval cubin image", _compiler_log(log))

    return CompiledTileStaticEvalCubin(
        cubin=cubin,
        name_expressions=name_expressions,
        lowered_names=lowered_names,
        log=_compiler_log(log),
    )
