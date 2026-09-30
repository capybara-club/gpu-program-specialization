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
"""Physical CUBIN inspection for scalar trajectory-LM patch sites."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any

from .elf import (
    BPT_WORD0,
    INSTRUCTION_BYTES,
    OPCODE_FADD_IMM,
    OPCODE_FADD_REG,
    OPCODE_FSEL,
    OPCODE_LOP3_IMM,
    WAIT_BARRIER_MASK,
    CubinFormatError,
    ElfImage,
    Function,
)
from .lm_manifest import lm_shape_from_manifest, parse_lm_cuda_manifest
from .lm_model import LM_KERNEL_NAME, LMKernelShape
from .model import SUPPORTED_ARCHITECTURES
from .lm_template import (
    LM_OUTPUT_MARKER_OFFSET,
    site_marker_bits,
)
from .manifest import TEMPLATE_ID_SYMBOL
from .sass import OPCODE_MOV


LM_INSPECTION_SCHEMA = "odezza.lm-module-inspection"
LM_INSPECTION_SCHEMA_VERSION = 1
OPCODES_IADD_IMM = frozenset((0x7810, 0x7835))


@dataclass(frozen=True)
class LMScalarSitePlan:
    site_index: int
    entry_offset: int
    scaffold_start_offset: int
    patch_start_offset: int
    patch_instruction_count: int
    continuation_offset: int
    incoming_wait_mask: int
    input_materialization_offsets: tuple[int, ...]
    input_source_registers: tuple[int, ...]
    input_wait_masks: tuple[int, ...]
    input_registers: tuple[int, ...]
    output_registers: tuple[int, ...]
    kernel_output_registers: tuple[int, ...]
    available_registers: tuple[int, ...]
    predicate_register: int
    permutation_register: int
    toggle_test_instruction: tuple[int, int]
    cleanup_offsets: tuple[int, ...]


@dataclass(frozen=True)
class LMCubinPlan:
    cubin_size: int
    architecture: int
    register_count: int
    register_count_offsets: tuple[int, ...]
    register_count_header_offsets: tuple[int, ...]
    function: Function
    shape: LMKernelShape
    sites: tuple[LMScalarSitePlan, ...]


@dataclass(frozen=True)
class VerifiedLMInspection:
    manifest: dict[str, Any]
    plan: LMCubinPlan
    document: dict[str, Any]


def _word(data: bytes, offset: int, index: int) -> int:
    return struct.unpack_from("<Q", data, offset + 8 * index)[0]


def _single_record(
    records: dict[int, tuple[int, int, int, int]],
    immediate: int,
    label: str,
) -> tuple[int, int, int, int]:
    try:
        return records[immediate]
    except KeyError as exc:
        raise CubinFormatError(f"LM site is missing its {label} marker") from exc


def _inspect_site(
    cubin: bytes,
    function: Function,
    shape: LMKernelShape,
    site_index: int,
) -> LMScalarSitePlan:
    start = function.file_offset
    end = start + function.size
    marker = site_marker_bits(site_index)
    input_count = shape.site_input_counts[site_index]
    output_count = len(shape.site_output_groups[site_index])
    wanted = {
        *(marker + index for index in range(input_count)),
        *(marker + LM_OUTPUT_MARKER_OFFSET + index for index in range(output_count)),
    }
    records: dict[int, tuple[int, int, int, int]] = {}
    for offset in range(start, end, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        if (word0 & 0xFFFF) != OPCODE_FADD_IMM:
            continue
        immediate = (word0 >> 32) & 0xFFFFFFFF
        if immediate not in wanted:
            continue
        if immediate in records:
            raise CubinFormatError("an LM scalar-site marker appears more than once")
        records[immediate] = (
            offset,
            (word0 >> 16) & 0xFF,
            (word0 >> 24) & 0xFF,
            (_word(cubin, offset, 1) >> 52) & WAIT_BARRIER_MASK,
        )
    if set(records) != wanted:
        raise CubinFormatError(
            f"LM scalar site {site_index} has missing or nonunique register markers"
        )

    inputs = tuple(records[marker + index] for index in range(input_count))
    outputs = tuple(
        _single_record(
            records,
            marker + LM_OUTPUT_MARKER_OFFSET + index,
            f"output {index}",
        )
        for index in range(output_count)
    )
    physical_input_start = min(record[0] for record in inputs)
    physical_input_end = max(record[0] for record in inputs)
    physical_output_start = min(record[0] for record in outputs)
    physical_output_end = max(record[0] for record in outputs)
    if not physical_input_end < physical_output_start:
        raise CubinFormatError("LM scalar-site marker order is invalid")

    patch_start = physical_output_end + INSTRUCTION_BYTES
    while patch_start < end and _word(cubin, patch_start, 0) != BPT_WORD0:
        patch_start += INSTRUCTION_BYTES
    if patch_start >= end:
        raise CubinFormatError("LM scalar site has no patch reserve")
    patch_offsets = tuple(
        range(
            patch_start,
            patch_start + shape.site_patch_capacity * INSTRUCTION_BYTES,
            INSTRUCTION_BYTES,
        )
    )
    if patch_offsets[-1] >= end or any(
        _word(cubin, offset, 0) != BPT_WORD0 for offset in patch_offsets
    ):
        raise CubinFormatError(
            f"LM site {site_index} does not preserve {shape.site_patch_capacity} contiguous patch instructions"
        )
    if patch_offsets != tuple(
        range(
            patch_start,
            patch_start + shape.site_patch_capacity * INSTRUCTION_BYTES,
            INSTRUCTION_BYTES,
        )
    ):
        raise CubinFormatError("LM scalar patch reserve is not contiguous")

    scaffold_candidates = [
        offset
        for offset in range(
            max(
                start,
                physical_input_start
                - (input_count + 4) * INSTRUCTION_BYTES,
            ),
            physical_input_start,
            INSTRUCTION_BYTES,
        )
        if _word(cubin, offset, 0) == BPT_WORD0
    ]
    if not scaffold_candidates:
        raise CubinFormatError(f"LM scalar site {site_index} has no leading breakpoint")
    scaffold_start = scaffold_candidates[-1]
    entry_candidates = [
        offset
        for offset in range(
            physical_input_end + INSTRUCTION_BYTES,
            physical_output_start,
            INSTRUCTION_BYTES,
        )
        if _word(cubin, offset, 0) == BPT_WORD0
    ]
    if len(entry_candidates) != 1:
        raise CubinFormatError(
            f"LM scalar site {site_index} has {len(entry_candidates)} post-input entry breakpoints"
        )
    entry = entry_candidates[0]

    predicate_register: int | None = None
    permutation_register: int | None = None
    salted_permutation_register: int | None = None
    toggle_template: tuple[int, int] | None = None
    salt_count = 0
    toggle_count = 0
    select_count = 0
    available = [record[2] for record in inputs]
    scaffold_end = patch_start
    for offset in range(scaffold_start, scaffold_end, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        word1 = _word(cubin, offset, 1)
        opcode = word0 & 0xFFFF
        if opcode in OPCODES_IADD_IMM and word0 >> 32 == site_index + 1:
            salt_count += 1
            salted_permutation_register = (word0 >> 16) & 0xFF
            permutation_register = (word0 >> 24) & 0xFF
        elif opcode == OPCODE_LOP3_IMM and (word0 & 0x00FFFFFF) == 0x00FF7812 and word0 >> 32 == 1:
            toggle_count += 1
            predicate_register = (word1 >> 17) & 0x7
            if salted_permutation_register is None or (word0 >> 24) & 0xFF != salted_permutation_register:
                raise CubinFormatError(
                    f"LM site {site_index} toggle test does not consume its site-specific salt"
                )
            toggle_template = (word0, word1)
        elif opcode == OPCODE_FSEL:
            select_count += 1
            predicate = (word1 >> 23) & 0x7
            if predicate_register is None or predicate != predicate_register:
                raise CubinFormatError("LM toggle compare and select use different predicates")
            available.append((word0 >> 16) & 0xFF)
        elif (opcode & 0x0FFF) == (OPCODE_FADD_REG & 0x0FFF) and opcode >> 12 < 7:
            select_count += 1
            predicate = opcode >> 12
            if predicate_register is None or predicate != predicate_register:
                raise CubinFormatError("LM folded toggle select uses the wrong predicate")
            available.append((word0 >> 16) & 0xFF)
    if (
        salt_count != 1
        or toggle_count != 1
        or select_count != 1
        or predicate_register is None
        or permutation_register is None
        or toggle_template is None
        or predicate_register >= 7
        or permutation_register == 255
    ):
        raise CubinFormatError(
            f"LM site {site_index} did not preserve one writable toggle predicate: "
            f"salt={salt_count}, toggle={toggle_count}, select={select_count}"
        )

    permutation_moves = []
    for offset in range(scaffold_start, scaffold_end, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        if word0 & 0xFFFF == OPCODE_MOV and (word0 >> 16) & 0xFF == permutation_register:
            permutation_moves.append((word0 >> 32) & 0xFF)
    if len(permutation_moves) > 1:
        raise CubinFormatError("LM permutation register has ambiguous move origins")
    if permutation_moves:
        permutation_register = permutation_moves[0]

    input_materialization_offsets = tuple(record[0] for record in inputs)
    input_source_registers = tuple(record[2] for record in inputs)
    input_wait_masks = tuple(record[3] for record in inputs)
    input_registers = tuple(record[1] for record in inputs)
    kernel_output_registers = tuple(record[1] for record in outputs)
    if len(set(input_registers)) != len(input_registers):
        raise CubinFormatError("LM input registers alias")
    if len(set(kernel_output_registers)) != len(kernel_output_registers):
        raise CubinFormatError("LM kernel output registers alias")
    if permutation_register in input_registers:
        raise CubinFormatError(
            f"LM site {site_index} scalar registers overlap: "
            f"inputs={input_registers}, outputs={kernel_output_registers}, "
            f"permutation={permutation_register}"
        )
    output_registers = kernel_output_registers
    reserved = (
        set(input_registers)
        | set(output_registers)
        | set(kernel_output_registers)
        | {permutation_register, 255}
    )
    available_registers = tuple(dict.fromkeys(value for value in available if value not in reserved))

    cleanup = tuple(range(scaffold_start, patch_start, INSTRUCTION_BYTES))
    return LMScalarSitePlan(
        site_index=site_index,
        entry_offset=entry,
        scaffold_start_offset=scaffold_start,
        patch_start_offset=patch_start,
        patch_instruction_count=shape.site_patch_capacity,
        continuation_offset=patch_start + shape.site_patch_capacity * INSTRUCTION_BYTES,
        incoming_wait_mask=0,
        input_materialization_offsets=input_materialization_offsets,
        input_source_registers=input_source_registers,
        input_wait_masks=input_wait_masks,
        input_registers=input_registers,
        output_registers=tuple(output_registers),
        kernel_output_registers=kernel_output_registers,
        available_registers=available_registers,
        predicate_register=predicate_register,
        permutation_register=permutation_register,
        toggle_test_instruction=toggle_template,
        cleanup_offsets=cleanup,
    )


def inspect_lm_cubin(
    cubin: bytes,
    shape: LMKernelShape,
) -> LMCubinPlan:
    elf = ElfImage(cubin)
    if elf.architecture not in SUPPORTED_ARCHITECTURES:
        raise CubinFormatError(
            f"LM CUBIN architecture sm_{elf.architecture} is unsupported"
        )
    function = elf.find_function(LM_KERNEL_NAME)
    register_count, register_offsets, header_offsets = elf.register_counts(
        function.symbol_index
    )
    sites = tuple(
        _inspect_site(cubin, function, shape, site_index)
        for site_index in range(shape.site_count)
    )
    if len({site.patch_start_offset for site in sites}) != len(sites):
        raise CubinFormatError("LM patch sites are not physically distinct")
    return LMCubinPlan(
        cubin_size=len(cubin),
        architecture=elf.architecture,
        register_count=register_count,
        register_count_offsets=register_offsets,
        register_count_header_offsets=header_offsets,
        function=function,
        shape=shape,
        sites=sites,
    )


def inspect_lm_module(source: str, cubin: bytes) -> VerifiedLMInspection:
    manifest, _payload = parse_lm_cuda_manifest(source)
    shape = lm_shape_from_manifest(manifest)
    expected_template_id = bytes.fromhex(manifest["template_id"])
    elf = ElfImage(cubin)
    symbol = elf.find_data_symbol(TEMPLATE_ID_SYMBOL, len(expected_template_id))
    actual_template_id = cubin[symbol.file_offset : symbol.file_offset + symbol.size]
    if actual_template_id != expected_template_id:
        raise CubinFormatError("LM source and CUBIN template IDs differ")
    plan = inspect_lm_cubin(cubin, shape)
    document: dict[str, Any] = {
        "schema": LM_INSPECTION_SCHEMA,
        "schema_version": LM_INSPECTION_SCHEMA_VERSION,
        "status": "verified",
        "identity": {
            "template_id": manifest["template_id"],
            "source_sha256": manifest["source_sha256"],
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        },
        "declared": manifest,
        "observed": {
            "architecture": f"sm_{plan.architecture}",
            "cubin_byte_size": len(cubin),
            "register_count": plan.register_count,
            "kernel": {
                "symbol_index": plan.function.symbol_index,
                "file_offset": plan.function.file_offset,
                "byte_size": plan.function.size,
            },
            "sites": [
                {
                    "site_index": site.site_index,
                    "entry_file_offset": site.entry_offset,
                    "patch_start_file_offset": site.patch_start_offset,
                    "patch_instruction_count": site.patch_instruction_count,
                    "continuation_file_offset": site.continuation_offset,
                    "input_registers": list(site.input_registers),
                    "input_source_registers": list(site.input_source_registers),
                    "input_materialization_file_offsets": list(site.input_materialization_offsets),
                    "scalar_output_indices": list(
                        plan.shape.site_output_groups[site.site_index]
                    ),
                    "output_registers": list(site.output_registers),
                    "kernel_output_registers": list(site.kernel_output_registers),
                    "output_relay_count": sum(
                        output != kernel
                        for output, kernel in zip(
                            site.output_registers,
                            site.kernel_output_registers,
                        )
                    ),
                    "available_registers": list(site.available_registers),
                    "predicate_register": site.predicate_register,
                    "permutation_register": site.permutation_register,
                }
                for site in plan.sites
            ],
        },
        "verification": {
            "source_payload_hash": "match",
            "source_template_id": "match",
            "cubin_template_id": "match",
            "declared_site_count": "match",
            "declared_patch_capacity": "match",
        },
    }
    return VerifiedLMInspection(manifest, plan, document)
