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

from .cubin import INSTRUCTION_BYTES, WAIT_BARRIER_MASK
from .genome import SystemGenome
from .packed_cubin import PackedCubinPlan, PackedModulePlan
from .sass import (
    REGISTER_PAD,
    REGISTER_RZ,
    Assembler,
    SassAssemblyError,
    _branch,
    _nop,
)
from .shape import KernelShape


@dataclass(frozen=True)
class PackedSpecializationResult:
    cubin: bytes
    instruction_counts: tuple[tuple[int, ...], ...]
    body_instruction_counts: tuple[int, ...]
    body_offsets: tuple[int, ...]
    register_count: int


@dataclass(frozen=True)
class PackedKernelSpecialization:
    name: str
    genome_base: int
    genome_count: int
    instruction_counts: tuple[tuple[int, ...], ...]
    body_instruction_counts: tuple[int, ...]
    body_offsets: tuple[int, ...]
    register_count: int


@dataclass(frozen=True)
class PackedModuleSpecializationResult:
    cubin: bytes
    kernels: tuple[PackedKernelSpecialization, ...]


def specialize_packed_cubin(
    cubin: bytes,
    plan: PackedCubinPlan,
    genomes: Sequence[SystemGenome],
    shape: KernelShape,
) -> PackedSpecializationResult:
    """Pack complete multi-output system genomes into one compact SASS arena."""

    if len(cubin) != plan.cubin_size:
        raise SassAssemblyError("CUBIN size changed after packed inspection")
    if not genomes:
        raise SassAssemblyError("at least one system genome is required")
    if len(genomes) > plan.genome_capacity:
        raise SassAssemblyError(
            f"{len(genomes)} genomes exceed the CUBIN capacity of {plan.genome_capacity}"
        )
    if shape.input_count != plan.input_count or shape.ast_count != plan.output_count:
        raise SassAssemblyError("KernelShape does not match the packed CUBIN register ABI")

    packed: list[tuple[int, int]] = []
    all_instruction_counts: list[tuple[int, ...]] = []
    body_instruction_counts: list[int] = []
    body_offsets: list[int] = []
    expanded_register_count = plan.register_count

    for genome in genomes:
        genome.validate(shape)
        body: list[tuple[int, int]] = []
        genome_instruction_counts: list[int] = []
        for output_index, program in enumerate(genome.programs):
            assembler = Assembler(
                plan.site.input_registers,
                plan.site.output_registers,
                plan.site.available_registers,
                plan.site.output_registers[output_index],
                0,
                plan.register_count,
            )
            assembler.compile(program, plan.site.input_registers)
            body.extend(assembler.instructions)
            genome_instruction_counts.append(len(assembler.instructions))
            expanded_register_count = max(
                expanded_register_count, assembler.high_water_register
            )

        body_start = plan.site.arena_start_offset + len(packed) * INSTRUCTION_BYTES
        branch_offset = body_start + len(body) * INSTRUCTION_BYTES
        if branch_offset >= plan.site.arena_end_offset:
            raise SassAssemblyError("packed genome bodies exhausted the compact arena")
        remaining = (
            plan.site.continuation_offset - branch_offset
        ) // INSTRUCTION_BYTES
        body.append(_branch(remaining, plan.architecture))
        if len(packed) + len(body) > plan.site.arena_instruction_count:
            raise SassAssemblyError("packed genome bodies exhausted the compact arena")

        body_offsets.append(body_start - plan.function.file_offset)
        packed.extend(body)
        all_instruction_counts.append(tuple(genome_instruction_counts))
        body_instruction_counts.append(len(body))

    output = bytearray(cubin)
    nop = _nop()
    for offset in range(
        plan.site.entry_offset,
        plan.site.arena_start_offset,
        INSTRUCTION_BYTES,
    ):
        struct.pack_into("<QQ", output, offset, *nop)
    for instruction_index in range(plan.site.arena_instruction_count):
        struct.pack_into(
            "<QQ",
            output,
            plan.site.arena_start_offset + instruction_index * INSTRUCTION_BYTES,
            *nop,
        )
    for offset in plan.site.cleanup_offsets:
        struct.pack_into("<QQ", output, offset, *nop)

    dispatch_start = plan.site.dispatch_offsets[0]
    if dispatch_start <= plan.site.entry_offset:
        raise SassAssemblyError("packed dispatch scaffolding has no room for its fast-path branch")
    entry_remaining = (
        dispatch_start - plan.site.entry_offset
    ) // INSTRUCTION_BYTES
    entry_branch = _branch(entry_remaining, plan.architecture)
    entry_branch = (
        entry_branch[0],
        entry_branch[1] | ((plan.site.incoming_wait_mask & WAIT_BARRIER_MASK) << 52),
    )
    struct.pack_into(
        "<QQ",
        output,
        plan.site.entry_offset,
        *entry_branch,
    )
    for offset, instruction in zip(
        plan.site.dispatch_offsets, plan.site.dispatch_instructions
    ):
        struct.pack_into(
            "<QQ",
            output,
            offset,
            *instruction,
        )
    for instruction_index, instruction in enumerate(packed):
        struct.pack_into(
            "<QQ",
            output,
            plan.site.arena_start_offset + instruction_index * INSTRUCTION_BYTES,
            *instruction,
        )

    fallback = body_offsets[0]
    for index, table_offset in enumerate(plan.site.target_table_offsets):
        target = body_offsets[index] if index < len(body_offsets) else fallback
        struct.pack_into("<I", output, table_offset, target)

    final_register_count = plan.register_count
    if expanded_register_count > plan.register_count:
        if expanded_register_count > REGISTER_RZ - REGISTER_PAD:
            raise SassAssemblyError("packed expressions leave no register padding")
        final_register_count = expanded_register_count + REGISTER_PAD
        for offset in plan.register_count_offsets:
            struct.pack_into("<I", output, offset, final_register_count)
        for offset in plan.register_count_header_offsets:
            struct.pack_into("<B", output, offset, final_register_count)

    return PackedSpecializationResult(
        cubin=bytes(output),
        instruction_counts=tuple(all_instruction_counts),
        body_instruction_counts=tuple(body_instruction_counts),
        body_offsets=tuple(body_offsets),
        register_count=final_register_count,
    )


def specialize_packed_module_cubin(
    cubin: bytes,
    plan: PackedModulePlan,
    genomes: Sequence[SystemGenome],
    shape: KernelShape,
) -> PackedModuleSpecializationResult:
    """Patch every populated kernel arena while preserving one CUBIN image."""

    if len(cubin) != plan.cubin_size:
        raise SassAssemblyError("CUBIN size changed after packed module inspection")
    if not genomes:
        raise SassAssemblyError("at least one system genome is required")
    if len(genomes) > plan.genome_capacity:
        raise SassAssemblyError(
            f"{len(genomes)} genomes exceed the module capacity of {plan.genome_capacity}"
        )

    output = cubin
    kernel_results: list[PackedKernelSpecialization] = []
    for kernel in plan.kernels:
        start = kernel.spec.genome_base
        stop = min(start + kernel.spec.dispatch.genome_capacity, len(genomes))
        selected = genomes[start:stop] if start < len(genomes) else ()
        if selected:
            specialized = specialize_packed_cubin(output, kernel.cubin, selected, shape)
            output = specialized.cubin
            kernel_results.append(
                PackedKernelSpecialization(
                    name=kernel.spec.name,
                    genome_base=start,
                    genome_count=len(selected),
                    instruction_counts=specialized.instruction_counts,
                    body_instruction_counts=specialized.body_instruction_counts,
                    body_offsets=specialized.body_offsets,
                    register_count=specialized.register_count,
                )
            )
        else:
            kernel_results.append(
                PackedKernelSpecialization(
                    name=kernel.spec.name,
                    genome_base=start,
                    genome_count=0,
                    instruction_counts=(),
                    body_instruction_counts=(),
                    body_offsets=(),
                    register_count=kernel.cubin.register_count,
                )
            )
    return PackedModuleSpecializationResult(output, tuple(kernel_results))
