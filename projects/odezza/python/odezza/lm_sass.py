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
"""Python SASS specialization for the scalar trajectory-LM patch sites."""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .lm_ast import LMSystem, LMSystemDerivatives
from .lm_cse import compile_lm_program_group
from .lm_elf import LMCubinPlan, LMScalarSitePlan
from .lm_model import LMKernelShape
from .sass import (
    REGISTER_PAD,
    REGISTER_RZ,
    SassAssemblyError,
    _branch,
    _mov,
    _nop,
)
from .elf import INSTRUCTION_BYTES


@dataclass(frozen=True)
class LMSpecializationResult:
    cubin: bytes
    required_toggle_bits: int
    site_instruction_counts: tuple[int, ...]
    scalar_instruction_counts: tuple[tuple[int, ...], ...]
    register_count: int
    derivatives: LMSystemDerivatives


def _compile_site(
    plan: LMCubinPlan,
    site: LMScalarSitePlan,
    programs,
    shape: LMKernelShape,
) -> tuple[list[tuple[int, int]], tuple[int, ...], int]:
    instructions, scalar_counts, high_water = compile_lm_program_group(
        plan,
        site,
        programs,
        shape,
    )
    for output_register, kernel_output_register in zip(
        site.output_registers,
        site.kernel_output_registers,
    ):
        if output_register != kernel_output_register:
            instructions.append(_mov(kernel_output_register, output_register, 0))
    branch_offset = site.patch_start_offset + len(instructions) * INSTRUCTION_BYTES
    if branch_offset >= site.continuation_offset:
        raise SassAssemblyError(
            f"LM site {site.site_index} needs {len(instructions) + 1} instructions; capacity is {site.patch_instruction_count}"
        )
    remaining = (site.continuation_offset - branch_offset) // INSTRUCTION_BYTES
    instructions.append(_branch(remaining, plan.architecture))
    if len(instructions) > site.patch_instruction_count:
        raise SassAssemblyError(
            f"LM site {site.site_index} needs {len(instructions)} instructions; capacity is {site.patch_instruction_count}"
        )
    return instructions, scalar_counts, high_water


def specialize_lm_cubin(
    cubin: bytes,
    plan: LMCubinPlan,
    system: LMSystem,
) -> LMSpecializationResult:
    """Insert one complete system and its local derivatives into an LM template."""

    if len(cubin) != plan.cubin_size:
        raise SassAssemblyError("LM CUBIN size changed after inspection")
    shape = plan.shape
    system.validate(shape)
    derivatives = LMSystemDerivatives.from_system(system, shape)
    program_groups = derivatives.site_program_groups(shape)
    if len(program_groups) != len(plan.sites):
        raise SassAssemblyError("LM derivative programs do not match the inspected sites")

    output = bytearray(cubin)
    nop = _nop()
    instruction_counts: list[int] = []
    scalar_instruction_counts: list[tuple[int, ...]] = []
    expanded_register_count = plan.register_count
    for site, programs in zip(
        plan.sites,
        program_groups,
    ):
        input_count = len(site.input_registers)
        if any(
            len(values) != input_count
            for values in (
                site.input_materialization_offsets,
                site.input_source_registers,
                site.input_wait_masks,
            )
        ):
            raise SassAssemblyError(
                f"LM site {site.site_index} input-materialization plan is incomplete"
            )
        if any(
            offset not in site.cleanup_offsets
            for offset in site.input_materialization_offsets
        ):
            raise SassAssemblyError(
                f"LM site {site.site_index} input materialization is outside its scaffold"
            )
        if len(programs) != len(site.output_registers):
            raise SassAssemblyError(
                f"LM site {site.site_index} program/output count differs"
            )
        instructions, scalar_counts, high_water = _compile_site(
            plan,
            site,
            programs,
            shape,
        )
        expanded_register_count = max(expanded_register_count, high_water)
        instruction_counts.append(len(instructions))
        scalar_instruction_counts.append(scalar_counts)
        if site.entry_offset not in site.cleanup_offsets:
            raise SassAssemblyError(
                f"LM site {site.site_index} has no inspected entry instruction"
            )
        entry_distance = (site.patch_start_offset - site.entry_offset) // INSTRUCTION_BYTES
        if entry_distance <= 0:
            raise SassAssemblyError(
                f"LM site {site.site_index} patch does not follow its entry"
            )
        for offset in site.cleanup_offsets:
            struct.pack_into("<QQ", output, offset, *nop)
        for offset, source, destination, wait_mask in zip(
            site.input_materialization_offsets,
            site.input_source_registers,
            site.input_registers,
            site.input_wait_masks,
        ):
            struct.pack_into(
                "<QQ",
                output,
                offset,
                *_mov(destination, source, wait_mask),
            )
        struct.pack_into(
            "<QQ",
            output,
            site.entry_offset,
            *_branch(entry_distance, plan.architecture),
        )
        for index in range(site.patch_instruction_count):
            struct.pack_into(
                "<QQ",
                output,
                site.patch_start_offset + index * INSTRUCTION_BYTES,
                *nop,
            )
        for index, instruction in enumerate(instructions):
            struct.pack_into(
                "<QQ",
                output,
                site.patch_start_offset + index * INSTRUCTION_BYTES,
                *instruction,
            )

    final_register_count = plan.register_count
    if expanded_register_count > plan.register_count:
        if expanded_register_count > REGISTER_RZ - REGISTER_PAD:
            raise SassAssemblyError("LM specialization leaves no register padding")
        final_register_count = expanded_register_count + REGISTER_PAD
        for offset in plan.register_count_offsets:
            struct.pack_into("<I", output, offset, final_register_count)
        for offset in plan.register_count_header_offsets:
            struct.pack_into("<B", output, offset, final_register_count)

    return LMSpecializationResult(
        cubin=bytes(output),
        required_toggle_bits=system.required_toggle_bits,
        site_instruction_counts=tuple(instruction_counts),
        scalar_instruction_counts=tuple(scalar_instruction_counts),
        register_count=final_register_count,
        derivatives=derivatives,
    )
