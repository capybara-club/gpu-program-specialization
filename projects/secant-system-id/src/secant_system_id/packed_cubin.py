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
import struct
from typing import Sequence

from .cubin import (
    BPT_WORD0,
    INSTRUCTION_BYTES,
    OPCODE_FADD_IMM,
    OPCODE_FADD_REG,
    OPCODE_STS_RZ,
    OPCODE_STS_UR,
    SUPPORTED_SASS_ARCHITECTURES,
    WAIT_BARRIER_MASK,
    CubinFormatError,
    ElfImage,
    Function,
    _append_unique,
    _is_zero_umov,
    _word,
)
from .packed_template import PACKED_KERNEL_NAME
from .shape import KernelShape, PackedDispatch, PackedKernelSpec
from .template import FIRST_MARKER_BITS


@dataclass(frozen=True)
class PackedSitePlan:
    entry_offset: int
    dispatch_offsets: tuple[int, ...]
    dispatch_instructions: tuple[tuple[int, int], ...]
    arena_start_offset: int
    arena_instruction_count: int
    arena_end_offset: int
    continuation_offset: int
    incoming_wait_mask: int
    input_registers: tuple[int, ...]
    output_registers: tuple[int, ...]
    available_registers: tuple[int, ...]
    cleanup_offsets: tuple[int, ...]
    target_table_offsets: tuple[int, ...]
    original_target_values: tuple[int, ...]


@dataclass(frozen=True)
class PackedCubinPlan:
    cubin_size: int
    architecture: int
    input_count: int
    output_count: int
    genome_capacity: int
    register_count: int
    register_count_offsets: tuple[int, ...]
    register_count_header_offsets: tuple[int, ...]
    function: Function
    site: PackedSitePlan


@dataclass(frozen=True)
class PackedKernelPlan:
    spec: PackedKernelSpec
    cubin: PackedCubinPlan


@dataclass(frozen=True)
class PackedModulePlan:
    cubin_size: int
    architecture: int
    kernels: tuple[PackedKernelPlan, ...]

    @property
    def genome_capacity(self) -> int:
        return self.kernels[-1].spec.genome_end


def _target_section(elf: ElfImage, kernel_name: str):
    wanted = f".nv.constant2.{kernel_name}"
    matches = [section for section in elf.sections if section.name == wanted]
    if len(matches) != 1:
        raise CubinFormatError(
            f"expected one compact branch-target section {wanted!r}, found {len(matches)}"
        )
    return matches[0]


def inspect_packed_cubin(
    cubin: bytes,
    shape: KernelShape | None = None,
    dispatch: PackedDispatch | None = None,
    kernel_name: str = PACKED_KERNEL_NAME,
) -> PackedCubinPlan:
    """Recover the compact dispatch arena and its register contract."""

    elf = ElfImage(cubin)
    if elf.architecture not in SUPPORTED_SASS_ARCHITECTURES:
        raise CubinFormatError(
            f"the Python SASS writer supports sm_89, sm_90, and sm_120, not sm_{elf.architecture}"
        )
    function = elf.find_function(kernel_name)
    register_count, register_count_offsets, register_count_header_offsets = (
        elf.register_counts(function.symbol_index)
    )
    target_section = _target_section(elf, kernel_name)
    if target_section.size == 0 or target_section.size % 4:
        raise CubinFormatError("compact branch-target table is empty or misaligned")
    genome_capacity = target_section.size // 4
    original_targets = struct.unpack_from(
        f"<{genome_capacity}I", cubin, target_section.offset
    )
    if len(set(original_targets)) != len(original_targets):
        raise CubinFormatError("compact branch-target entries are not unique")
    if any(value % INSTRUCTION_BYTES or value >= function.size for value in original_targets):
        raise CubinFormatError("compact branch target is outside or misaligned in the kernel")
    arena_start = function.file_offset + min(original_targets)

    start = function.file_offset
    end = start + function.size
    marker_records: dict[int, tuple[int, int, int, int]] = {}
    bpts: list[int] = []
    for offset in range(start, end, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        word1 = _word(cubin, offset, 1)
        if word0 == BPT_WORD0:
            bpts.append(offset)
        if (word0 & 0xFFFF) != OPCODE_FADD_IMM:
            continue
        marker_index = (word0 >> 32) - FIRST_MARKER_BITS
        if not 0 <= marker_index < 255:
            continue
        if marker_index in marker_records:
            raise CubinFormatError("a packed marker immediate appears more than once")
        marker_records[marker_index] = (
            offset,
            (word0 >> 16) & 0xFF,
            (word0 >> 24) & 0xFF,
            (word1 >> 52) & WAIT_BARRIER_MASK,
        )
    if not marker_records or not bpts:
        raise CubinFormatError("packed marker scaffolding was not found")
    marker_count = max(marker_records) + 1
    if set(marker_records) != set(range(marker_count)):
        raise CubinFormatError("packed marker immediates are not contiguous")

    input_markers = sorted(
        index for index, record in marker_records.items() if record[0] < arena_start
    )
    output_markers = sorted(
        index for index, record in marker_records.items() if record[0] > arena_start
    )
    input_count = len(input_markers)
    output_count = len(output_markers)
    if (
        input_markers != list(range(input_count))
        or output_markers != list(range(input_count, marker_count))
        or input_count == 0
        or output_count == 0
    ):
        raise CubinFormatError("could not infer packed input/output marker boundaries")
    arena_end = min(marker_records[index][0] for index in output_markers)
    if arena_end <= arena_start or (arena_end - arena_start) % INSTRUCTION_BYTES:
        raise CubinFormatError("compact arena has an invalid continuation offset")

    if shape is not None and (
        shape.input_count != input_count or shape.ast_count != output_count
    ):
        raise CubinFormatError(
            "CUBIN-inferred packed shape does not match KernelShape: "
            f"inputs={input_count}, outputs={output_count}"
        )
    if dispatch is not None and dispatch.genome_capacity != genome_capacity:
        raise CubinFormatError(
            "CUBIN target-table capacity does not match PackedDispatch: "
            f"{genome_capacity} != {dispatch.genome_capacity}"
        )

    input_registers = tuple(marker_records[index][2] for index in input_markers)
    output_registers = tuple(marker_records[index][1] for index in output_markers)
    if len(set(input_registers)) != input_count:
        raise CubinFormatError("packed input markers alias physical registers")
    if len(set(output_registers)) != output_count:
        raise CubinFormatError("packed output markers alias physical registers")
    if set(input_registers) & set(output_registers):
        raise CubinFormatError("packed input and output registers overlap")

    scaffold_start = min(bpts)
    scaffold_end = max(bpts) + INSTRUCTION_BYTES
    available: list[int] = []
    cleanup: list[int] = []
    incoming_wait_mask = 0
    for index in input_markers:
        record = marker_records[index]
        incoming_wait_mask |= record[3]
        _append_unique(available, record[1])

    marker_offsets = {record[0] for record in marker_records.values()}
    for offset in range(scaffold_start, scaffold_end, INSTRUCTION_BYTES):
        if arena_start <= offset < arena_end:
            continue
        word0 = _word(cubin, offset, 0)
        opcode = word0 & 0xFFFF
        if offset in marker_offsets or word0 == BPT_WORD0:
            cleanup.append(offset)
        elif opcode == OPCODE_FADD_REG:
            cleanup.append(offset)
            _append_unique(available, (word0 >> 16) & 0xFF)
        elif opcode in {OPCODE_STS_RZ, OPCODE_STS_UR} or _is_zero_umov(
            word0, elf.architecture
        ):
            cleanup.append(offset)

    reserved = set(input_registers) | set(output_registers) | {255}
    available_registers = tuple(register for register in available if register not in reserved)
    if not available_registers:
        raise CubinFormatError("packed marker scaffolding exposed no temporary registers")

    cleanup_set = set(cleanup)
    dispatch_records = tuple(
        (offset, (_word(cubin, offset, 0), _word(cubin, offset, 1)))
        for offset in range(scaffold_start, arena_start, INSTRUCTION_BYTES)
        if offset not in cleanup_set
    )
    if len(dispatch_records) not in {4, 5}:
        raise CubinFormatError(
            "expected PTXAS's four- or five-instruction compact dispatch sequence, found "
            f"{len(dispatch_records)} instructions"
        )

    site = PackedSitePlan(
        entry_offset=scaffold_start,
        dispatch_offsets=tuple(record[0] for record in dispatch_records),
        dispatch_instructions=tuple(record[1] for record in dispatch_records),
        arena_start_offset=arena_start,
        arena_instruction_count=(arena_end - arena_start) // INSTRUCTION_BYTES,
        arena_end_offset=arena_end,
        continuation_offset=scaffold_end,
        incoming_wait_mask=incoming_wait_mask,
        input_registers=input_registers,
        output_registers=output_registers,
        available_registers=available_registers,
        cleanup_offsets=tuple(cleanup),
        target_table_offsets=tuple(
            target_section.offset + 4 * index for index in range(genome_capacity)
        ),
        original_target_values=tuple(original_targets),
    )
    return PackedCubinPlan(
        cubin_size=len(cubin),
        architecture=elf.architecture,
        input_count=input_count,
        output_count=output_count,
        genome_capacity=genome_capacity,
        register_count=register_count,
        register_count_offsets=register_count_offsets,
        register_count_header_offsets=register_count_header_offsets,
        function=function,
        site=site,
    )


def inspect_packed_module_cubin(
    cubin: bytes,
    shape: KernelShape,
    specs: Sequence[PackedKernelSpec],
) -> PackedModulePlan:
    """Inspect every independently specializable packed entry point in a CUBIN."""

    if not specs:
        raise CubinFormatError("a packed module must declare at least one kernel")
    kernels = tuple(
        PackedKernelPlan(
            spec,
            inspect_packed_cubin(cubin, shape, spec.dispatch, spec.name),
        )
        for spec in specs
    )
    architectures = {kernel.cubin.architecture for kernel in kernels}
    if len(architectures) != 1:
        raise CubinFormatError("packed module kernels disagree on architecture")
    function_ranges = sorted(
        (
            kernel.cubin.function.file_offset,
            kernel.cubin.function.file_offset + kernel.cubin.function.size,
        )
        for kernel in kernels
    )
    for previous, current in zip(function_ranges, function_ranges[1:]):
        if previous[1] > current[0]:
            raise CubinFormatError("packed module kernel functions overlap")
    return PackedModulePlan(len(cubin), kernels[0].cubin.architecture, kernels)
