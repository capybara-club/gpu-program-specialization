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
from io import StringIO
from typing import Sequence

from cuda.core import Device, Program as CudaProgram, ProgramOptions

from .ast import Program, programs_bytes

try:
    import _secant_native as _native
except ImportError as error:
    raise ImportError(
        "the Secant nanobind extension is unavailable; build with SECANT_BUILD_PYTHON_BINDINGS=ON"
    ) from error


@dataclass(frozen=True, slots=True)
class MaterializeRecipe:
    """Static-column materialize shape. Patch capacity is the total shared island size in SASS instructions."""

    num_kernels: int = 1
    asts_per_kernel: int = 2
    num_inputs: int = 4
    patch_capacity_instructions: int = 64


@dataclass(frozen=True, slots=True)
class SSERecipe:
    """Static-column SSE shape. Patch capacity is the total shared island size in SASS instructions."""

    num_kernels: int = 1
    asts_per_kernel: int = 2
    num_inputs: int = 4
    num_targets: int = 2
    tile_rows: int = 128
    threads_per_block: int = 128
    patch_capacity_instructions: int = 64


@dataclass(frozen=True, slots=True)
class AffineStatsRecipe:
    """Static-column affine sufficient-statistics shape."""

    num_kernels: int = 1
    asts_per_kernel: int = 2
    num_inputs: int = 4
    num_targets: int = 2
    tile_rows: int = 128
    threads_per_block: int = 128
    patch_capacity_instructions: int = 64


@dataclass(frozen=True, slots=True)
class GramStatsRecipe:
    """Static-column cohort Gram sufficient-statistics shape."""

    num_kernels: int = 1
    asts_per_kernel: int = 2
    num_inputs: int = 4
    num_targets: int = 2
    tile_rows: int = 64
    threads_per_block: int = 64
    patch_capacity_instructions: int = 64


@dataclass(frozen=True, slots=True)
class DynamicConstantSSERecipe:
    """Dynamic-constant SSE shape. Patch capacity is the total shared island size in SASS instructions."""

    num_kernels: int = 1
    asts_per_kernel: int = 2
    num_input_columns: int = 4
    num_input_constants: int = 2
    num_targets: int = 2
    tile_rows: int = 128
    threads_per_block: int = 128
    patch_capacity_instructions: int = 64


@dataclass(frozen=True, slots=True)
class DynamicLeafSSERecipe:
    """Dynamic or mixed static/dynamic column-or-constant leaf SSE shape."""

    num_kernels: int = 1
    asts_per_kernel: int = 2
    num_input_columns: int = 4
    num_static_input_columns: int = 0
    num_dynamic_leaves: int = 8
    num_targets: int = 2
    tile_rows: int = 128
    threads_per_block: int = 128
    patch_capacity_instructions: int = 64


@dataclass(frozen=True, slots=True)
class CompiledTemplate:
    source: str
    cubin: bytes
    plan: object
    recipe: (
        MaterializeRecipe
        | SSERecipe
        | AffineStatsRecipe
        | GramStatsRecipe
        | DynamicConstantSSERecipe
        | DynamicLeafSSERecipe
    )
    shape: str
    arch: str
    log: str

    @property
    def num_asts(self) -> int:
        return self.recipe.num_kernels * self.recipe.asts_per_kernel

    def specialize(
        self,
        asts: Sequence[Program | bytes | bytearray | memoryview],
        *,
        routines: Sequence[Program | bytes | bytearray | memoryview] = (),
    ) -> bytes:
        packed_asts = programs_bytes(asts)
        packed_routines = tuple(programs_bytes(routines)) if routines else ()
        if len(packed_asts) != self.num_asts:
            raise ValueError(f"direct module specialization requires exactly {self.num_asts} AST programs")
        output = bytearray(self.cubin)
        self.plan.specialize_into(packed_routines, packed_asts, output)
        return bytes(output)


class CubinCompileError(RuntimeError):
    def __init__(self, message: str, log: str) -> None:
        super().__init__(message if not log else f"{message}\n{log.rstrip()}")
        self.log = log


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_materialize_recipe(recipe: MaterializeRecipe) -> None:
    _positive("num_kernels", recipe.num_kernels)
    _positive("asts_per_kernel", recipe.asts_per_kernel)
    _positive("num_inputs", recipe.num_inputs)
    _positive("patch_capacity_instructions", recipe.patch_capacity_instructions)


def _validate_sse_recipe(recipe: SSERecipe) -> None:
    _positive("num_kernels", recipe.num_kernels)
    _positive("asts_per_kernel", recipe.asts_per_kernel)
    _positive("num_inputs", recipe.num_inputs)
    _positive("num_targets", recipe.num_targets)
    _positive("tile_rows", recipe.tile_rows)
    _positive("threads_per_block", recipe.threads_per_block)
    _positive("patch_capacity_instructions", recipe.patch_capacity_instructions)
    if recipe.threads_per_block % 32 != 0 or recipe.threads_per_block > 512:
        raise ValueError("threads_per_block must be a multiple of 32 no larger than 512")
    if recipe.tile_rows < recipe.threads_per_block:
        raise ValueError("tile_rows must be at least threads_per_block")


def _validate_affine_stats_recipe(recipe: AffineStatsRecipe) -> None:
    _validate_sse_recipe(
        SSERecipe(
            recipe.num_kernels,
            recipe.asts_per_kernel,
            recipe.num_inputs,
            recipe.num_targets,
            recipe.tile_rows,
            recipe.threads_per_block,
            recipe.patch_capacity_instructions,
        )
    )


def _validate_gram_stats_recipe(recipe: GramStatsRecipe) -> None:
    _validate_sse_recipe(
        SSERecipe(
            recipe.num_kernels,
            recipe.asts_per_kernel,
            recipe.num_inputs,
            recipe.num_targets,
            recipe.tile_rows,
            recipe.threads_per_block,
            recipe.patch_capacity_instructions,
        )
    )
    if recipe.asts_per_kernel > 32:
        raise ValueError("asts_per_kernel must not exceed 32 for Gram statistics")
    shared_bytes = ((recipe.num_inputs + recipe.asts_per_kernel + 1) * recipe.tile_rows + 1) * 4
    if shared_bytes > 48 * 1024:
        raise ValueError("Gram statistics recipe exceeds 48 KiB of static shared memory")


def _validate_dynamic_recipe(recipe: DynamicConstantSSERecipe) -> None:
    _positive("num_kernels", recipe.num_kernels)
    _positive("asts_per_kernel", recipe.asts_per_kernel)
    _positive("num_input_columns", recipe.num_input_columns)
    _positive("num_input_constants", recipe.num_input_constants)
    _positive("num_targets", recipe.num_targets)
    _positive("tile_rows", recipe.tile_rows)
    _positive("threads_per_block", recipe.threads_per_block)
    _positive("patch_capacity_instructions", recipe.patch_capacity_instructions)
    if recipe.num_input_columns + recipe.num_input_constants > 128:
        raise ValueError("input columns and constants together must not exceed 128")
    if recipe.threads_per_block > 1024:
        raise ValueError("threads_per_block must not exceed 1024")


def _validate_dynamic_leaf_recipe(recipe: DynamicLeafSSERecipe) -> None:
    _positive("num_kernels", recipe.num_kernels)
    _positive("asts_per_kernel", recipe.asts_per_kernel)
    _positive("num_input_columns", recipe.num_input_columns)
    _positive("num_dynamic_leaves", recipe.num_dynamic_leaves)
    _positive("num_targets", recipe.num_targets)
    _positive("tile_rows", recipe.tile_rows)
    _positive("threads_per_block", recipe.threads_per_block)
    _positive("patch_capacity_instructions", recipe.patch_capacity_instructions)
    if recipe.num_dynamic_leaves > 32:
        raise ValueError("num_dynamic_leaves must not exceed 32")
    if recipe.num_static_input_columns not in (0, recipe.num_input_columns):
        raise ValueError("num_static_input_columns must be zero or num_input_columns")
    if recipe.num_static_input_columns + recipe.num_dynamic_leaves > 128:
        raise ValueError("static columns and dynamic leaves together must not exceed 128")
    if recipe.threads_per_block > 1024:
        raise ValueError("threads_per_block must not exceed 1024")


def _default_arch(device_id: int) -> str:
    device = Device(device_id)
    major, minor = device.compute_capability
    return f"sm_{major}{minor}"


def _compile_source(source: str, arch: str, name: str) -> tuple[bytes, str]:
    log = StringIO()
    options = ProgramOptions(
        name=name,
        arch=arch,
        std="c++11",
        ptxas_options=("--opt-level=1", "-warn-spills", "-Werror"),
        split_compile=1,
    )
    try:
        object_code = CudaProgram(source, "c++", options).compile("cubin", logs=log)
    except Exception as error:
        raise CubinCompileError(f"cuda.core failed to compile {name}", log.getvalue().rstrip("\0")) from error
    cubin = bytes(object_code.code)
    if not cubin.startswith(b"\x7fELF"):
        raise CubinCompileError("cuda.core returned an invalid CUBIN image", log.getvalue().rstrip("\0"))
    return cubin, log.getvalue().rstrip("\0")


def compile_materialize(
    recipe: MaterializeRecipe = MaterializeRecipe(),
    *,
    arch: str | None = None,
    device_id: int = 0,
) -> CompiledTemplate:
    _validate_materialize_recipe(recipe)
    selected_arch = _default_arch(device_id) if arch is None else arch
    source = _native.materialize_source_generate(
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.patch_capacity_instructions,
    )
    cubin, log = _compile_source(source, selected_arch, "secant_python_materialize.cu")
    plan = _native.CubinPlan.inspect_materialize(
        cubin,
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.patch_capacity_instructions,
    )
    return CompiledTemplate(source, cubin, plan, recipe, "materialize", selected_arch, log)


def compile_sse(
    recipe: SSERecipe = SSERecipe(),
    *,
    arch: str | None = None,
    device_id: int = 0,
) -> CompiledTemplate:
    _validate_sse_recipe(recipe)
    selected_arch = _default_arch(device_id) if arch is None else arch
    source = _native.sse_source_generate(
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    cubin, log = _compile_source(source, selected_arch, "secant_python_sse.cu")
    plan = _native.CubinPlan.inspect_sse(
        cubin,
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    return CompiledTemplate(source, cubin, plan, recipe, "sse", selected_arch, log)


def compile_affine_stats(
    recipe: AffineStatsRecipe = AffineStatsRecipe(),
    *,
    arch: str | None = None,
    device_id: int = 0,
) -> CompiledTemplate:
    _validate_affine_stats_recipe(recipe)
    selected_arch = _default_arch(device_id) if arch is None else arch
    source = _native.affine_stats_source_generate(
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    cubin, log = _compile_source(source, selected_arch, "secant_python_affine_stats.cu")
    plan = _native.CubinPlan.inspect_affine_stats(
        cubin,
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    return CompiledTemplate(source, cubin, plan, recipe, "affine_stats", selected_arch, log)


def compile_gram_stats(
    recipe: GramStatsRecipe = GramStatsRecipe(),
    *,
    arch: str | None = None,
    device_id: int = 0,
) -> CompiledTemplate:
    _validate_gram_stats_recipe(recipe)
    selected_arch = _default_arch(device_id) if arch is None else arch
    source = _native.gram_stats_source_generate(
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    cubin, log = _compile_source(source, selected_arch, "secant_python_gram_stats.cu")
    plan = _native.CubinPlan.inspect_gram_stats(
        cubin,
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_inputs,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    return CompiledTemplate(source, cubin, plan, recipe, "gram_stats", selected_arch, log)


def compile_dynamic_constant_sse(
    recipe: DynamicConstantSSERecipe = DynamicConstantSSERecipe(),
    *,
    arch: str | None = None,
    device_id: int = 0,
) -> CompiledTemplate:
    _validate_dynamic_recipe(recipe)
    selected_arch = _default_arch(device_id) if arch is None else arch
    source = _native.dynamic_constant_sse_source_generate(
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_input_columns,
        recipe.num_input_constants,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    cubin, log = _compile_source(source, selected_arch, "secant_python_dynamic_constant_sse.cu")
    plan = _native.CubinPlan.inspect_dynamic_constant_sse(
        cubin,
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_input_columns,
        recipe.num_input_constants,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    return CompiledTemplate(source, cubin, plan, recipe, "dynamic_constant_sse", selected_arch, log)


def compile_dynamic_leaf_sse(
    recipe: DynamicLeafSSERecipe = DynamicLeafSSERecipe(),
    *,
    arch: str | None = None,
    device_id: int = 0,
) -> CompiledTemplate:
    _validate_dynamic_leaf_recipe(recipe)
    selected_arch = _default_arch(device_id) if arch is None else arch
    source = _native.dynamic_leaf_sse_source_generate(
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_input_columns,
        recipe.num_static_input_columns,
        recipe.num_dynamic_leaves,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    cubin, log = _compile_source(source, selected_arch, "secant_python_dynamic_leaf_sse.cu")
    plan = _native.CubinPlan.inspect_dynamic_leaf_sse(
        cubin,
        recipe.num_kernels,
        recipe.asts_per_kernel,
        recipe.num_input_columns,
        recipe.num_static_input_columns,
        recipe.num_dynamic_leaves,
        recipe.num_targets,
        recipe.tile_rows,
        recipe.threads_per_block,
        recipe.patch_capacity_instructions,
    )
    return CompiledTemplate(source, cubin, plan, recipe, "dynamic_leaf_sse", selected_arch, log)
