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
"""Python ELF and CUBIN inspection reference."""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .model import KERNEL_NAME, KernelShape, SUPPORTED_ARCHITECTURES
from .template import FIRST_MARKER_BITS, OUTPUT_LIVE_MARKER_BITS, SHARED_BEGIN_MARKER_BITS, SHARED_END_MARKER_BITS


INSTRUCTION_BYTES = 16
BPT_WORD0 = 0x000000040000795C
OPCODE_FADD_REG = 0x7221
OPCODE_FADD_IMM = 0x7421
OPCODE_FSEL = 0x7208
OPCODE_LOP3_IMM = 0x7812
OPCODE_STS_RZ = 0x7388
OPCODE_STS_UR = 0x7988
OPCODE_UMOV = 0x7C82
NVINFO_FORMAT_U32 = 0x04
NVINFO_ATTR_REGCOUNT = 0x2F
NVINFO_ATTR_SIZE = 8
WAIT_BARRIER_MASK = 0x3F


class CubinFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Section:
    name_offset: int
    kind: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int
    name: str = ""


@dataclass(frozen=True)
class Function:
    symbol_index: int
    file_offset: int
    size: int


@dataclass(frozen=True)
class DataSymbol:
    symbol_index: int
    section_index: int
    file_offset: int
    size: int


class ElfImage:
    def __init__(self, data: bytes):
        if len(data) < 64 or data[:4] != b"\x7fELF":
            raise CubinFormatError("input is not an ELF image")
        if data[4] != 2 or data[5] != 1:
            raise CubinFormatError("only little-endian ELF64 CUBINs are supported")
        machine = struct.unpack_from("<H", data, 18)[0]
        if machine != 190:
            raise CubinFormatError(f"ELF machine {machine} is not NVIDIA CUDA")
        self.data = data
        self.flags = struct.unpack_from("<I", data, 48)[0]
        section_offset = struct.unpack_from("<Q", data, 40)[0]
        section_entry_size, section_count, names_index = struct.unpack_from("<HHH", data, 58)
        if section_entry_size != 64 or section_count == 0 or names_index >= section_count:
            raise CubinFormatError("invalid ELF section table")
        if section_offset + section_count * section_entry_size > len(data):
            raise CubinFormatError("ELF section table is outside the file")
        raw: list[Section] = []
        for index in range(section_count):
            raw.append(Section(*struct.unpack_from("<IIQQQQIIQQ", data, section_offset + 64 * index)))
        names_section = raw[names_index]
        names = self._slice(names_section.offset, names_section.size)
        self.sections = tuple(
            Section(
                section.name_offset,
                section.kind,
                section.flags,
                section.address,
                section.offset,
                section.size,
                section.link,
                section.info,
                section.alignment,
                section.entry_size,
                self._string(names, section.name_offset),
            )
            for section in raw
        )

    @property
    def architecture(self) -> int:
        low = self.flags & 0xFF
        if low >= 30:
            return low
        high = (self.flags >> 8) & 0xFF
        return high if high >= 30 else 0

    def _slice(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise CubinFormatError("ELF range is outside the file")
        return self.data[offset : offset + size]

    @staticmethod
    def _string(table: bytes, offset: int) -> str:
        if offset < 0 or offset >= len(table):
            raise CubinFormatError("ELF string offset is outside its table")
        end = table.find(b"\0", offset)
        if end < 0:
            raise CubinFormatError("unterminated ELF string")
        return table[offset:end].decode("utf-8")

    def _symbols(self):
        for requested_kind in (2, 11):
            for table in self.sections:
                if table.kind != requested_kind:
                    continue
                if table.entry_size != 24 or table.size % 24 or table.link >= len(self.sections):
                    raise CubinFormatError("invalid ELF64 symbol table")
                strings_section = self.sections[table.link]
                strings = self._slice(strings_section.offset, strings_section.size)
                symbols = self._slice(table.offset, table.size)
                for index in range(table.size // 24):
                    fields = struct.unpack_from("<IBBHQQ", symbols, 24 * index)
                    yield index, self._string(strings, fields[0]), fields

    def find_function(self, name: str) -> Function:
        for symbol_index, symbol_name, fields in self._symbols():
            _name_offset, info, _other, section_index, value, size = fields
            if (info & 0x0F) != 2 or symbol_name != name:
                continue
            if section_index == 0 or section_index >= len(self.sections) or size == 0:
                raise CubinFormatError("kernel symbol has an invalid section")
            section = self.sections[section_index]
            if value < section.address:
                raise CubinFormatError("kernel symbol precedes its section")
            file_offset = section.offset + value - section.address
            self._slice(file_offset, size)
            if file_offset % INSTRUCTION_BYTES or size % INSTRUCTION_BYTES:
                raise CubinFormatError("kernel code is not instruction aligned")
            return Function(symbol_index, file_offset, size)
        raise CubinFormatError(f"kernel symbol {name!r} was not found")

    def find_data_symbol(self, name: str, expected_size: int | None = None) -> DataSymbol:
        matches: dict[tuple[int, int, int], DataSymbol] = {}
        for symbol_index, symbol_name, fields in self._symbols():
            _name_offset, info, _other, section_index, value, size = fields
            if (info & 0x0F) != 1 or symbol_name != name:
                continue
            if section_index == 0 or section_index >= len(self.sections) or size == 0:
                raise CubinFormatError("data symbol has an invalid section")
            section = self.sections[section_index]
            if value < section.address:
                raise CubinFormatError("data symbol precedes its section")
            file_offset = section.offset + value - section.address
            self._slice(file_offset, size)
            match = DataSymbol(symbol_index, section_index, file_offset, size)
            matches[(section_index, file_offset, size)] = match
        if len(matches) != 1:
            raise CubinFormatError(f"expected one data symbol {name!r}, found {len(matches)}")
        result = next(iter(matches.values()))
        if expected_size is not None and result.size != expected_size:
            raise CubinFormatError(f"data symbol {name!r} has size {result.size}, expected {expected_size}")
        return result

    def register_counts(self, symbol_index: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        values: list[int] = []
        offsets: list[int] = []
        for section in self.sections:
            if ".nv.info" not in section.name:
                continue
            raw = self._slice(section.offset, section.size)
            for relative in range(0, max(0, len(raw) - 11), 4):
                if (
                    raw[relative] == NVINFO_FORMAT_U32
                    and raw[relative + 1] == NVINFO_ATTR_REGCOUNT
                    and struct.unpack_from("<H", raw, relative + 2)[0] == NVINFO_ATTR_SIZE
                    and struct.unpack_from("<I", raw, relative + 4)[0] == symbol_index
                ):
                    values.append(struct.unpack_from("<I", raw, relative + 8)[0])
                    offsets.append(section.offset + relative + 8)
        header_offsets: list[int] = []
        section_table_offset = struct.unpack_from("<Q", self.data, 40)[0]
        for section_index, section in enumerate(self.sections):
            if section.name.startswith(".text.") and (section.info & 0x00FFFFFF) == symbol_index:
                count = section.info >> 24
                if count:
                    values.append(count)
                    header_offsets.append(section_table_offset + 64 * section_index + 47)
        if not values or len(set(values)) != 1:
            raise CubinFormatError("could not recover one consistent kernel register count")
        return values[0], tuple(offsets), tuple(header_offsets)


def _word(data: bytes, offset: int, index: int) -> int:
    return struct.unpack_from("<Q", data, offset + 8 * index)[0]


def _append_unique(values: list[int], value: int) -> None:
    if value not in values:
        values.append(value)


def _is_zero_umov(word0: int, architecture: int) -> bool:
    if (word0 & 0xFFFF) != OPCODE_UMOV:
        return False
    return architecture < 90 or ((word0 >> 32) & 0xFF) == (0x3F if architecture == 90 else 0xFF)


@dataclass(frozen=True)
class SitePlan:
    scaffold_start_offset: int
    scaffold_end_offset: int
    shared_start_offset: int
    shared_instruction_count: int
    dispatch_offsets: tuple[int, ...]
    dispatch_instructions: tuple[tuple[int, int], ...]
    arena_start_offset: int
    arena_instruction_count: int
    arena_end_offset: int
    continuation_offset: int
    incoming_wait_mask: int
    input_registers: tuple[int, ...]
    output_registers: tuple[int, ...]
    final_output_registers: tuple[int, ...]
    output_materialization_offsets: tuple[int, ...]
    available_registers: tuple[int, ...]
    predicate_register: int
    permutation_register: int
    toggle_test_instruction: tuple[int, int]
    cleanup_offsets: tuple[int, ...]
    target_table_offsets: tuple[int, ...]
    original_target_values: tuple[int, ...]


@dataclass(frozen=True)
class CubinPlan:
    cubin_size: int
    architecture: int
    input_count: int
    output_count: int
    system_capacity: int
    register_count: int
    register_count_offsets: tuple[int, ...]
    register_count_header_offsets: tuple[int, ...]
    function: Function
    site: SitePlan


def _target_section(elf: ElfImage, kernel_name: str):
    wanted = f".nv.constant2.{kernel_name}"
    matches = [section for section in elf.sections if section.name == wanted]
    if len(matches) != 1:
        raise CubinFormatError(f"expected one branch-target section {wanted!r}, found {len(matches)}")
    return matches[0]


def inspect_cubin(
    cubin: bytes,
    shape: KernelShape,
    kernel_name: str = KERNEL_NAME,
) -> CubinPlan:
    """Recover the common prelude, compact branch arena, and register ABI."""

    elf = ElfImage(cubin)
    if elf.architecture not in SUPPORTED_ARCHITECTURES:
        raise CubinFormatError(f"the Python SASS writer does not support sm_{elf.architecture}")
    function = elf.find_function(kernel_name)
    register_count, register_offsets, header_offsets = elf.register_counts(function.symbol_index)
    target_section = _target_section(elf, kernel_name)
    if target_section.size != 4 * shape.system_capacity:
        raise CubinFormatError(
            f"target table has {target_section.size // 4} entries; expected {shape.system_capacity}"
        )
    original_targets = struct.unpack_from(f"<{shape.system_capacity}I", cubin, target_section.offset)
    if len(set(original_targets)) != len(original_targets):
        raise CubinFormatError("branch-target entries are not unique")
    if any(value % INSTRUCTION_BYTES or value >= function.size for value in original_targets):
        raise CubinFormatError("branch target is outside or misaligned in the kernel")

    start = function.file_offset
    end = start + function.size
    arena_start = start + min(original_targets)
    bpts = [offset for offset in range(start, end, INSTRUCTION_BYTES) if _word(cubin, offset, 0) == BPT_WORD0]
    if len(bpts) < shape.shared_patch_capacity + 2 * shape.system_capacity + 3:
        raise CubinFormatError("marker scaffolding or code reserve is missing")

    marker_count = shape.input_count + shape.output_count
    marker_records: dict[int, tuple[int, int, int, int]] = {}
    special_records: dict[int, tuple[int, int, int, int]] = {}
    output_live_records: dict[int, tuple[int, int, int, int]] = {}
    for offset in range(start, end, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        if (word0 & 0xFFFF) != OPCODE_FADD_IMM:
            continue
        immediate = (word0 >> 32) & 0xFFFFFFFF
        record = (
            offset,
            (word0 >> 16) & 0xFF,
            (word0 >> 24) & 0xFF,
            (_word(cubin, offset, 1) >> 52) & WAIT_BARRIER_MASK,
        )
        marker_index = immediate - FIRST_MARKER_BITS
        if 0 <= marker_index < marker_count:
            if marker_index in marker_records:
                raise CubinFormatError("a register marker appears more than once")
            marker_records[marker_index] = record
        if immediate in {SHARED_BEGIN_MARKER_BITS, SHARED_END_MARKER_BITS}:
            if immediate in special_records:
                raise CubinFormatError("a shared-site marker appears more than once")
            special_records[immediate] = record
        live_index = immediate - OUTPUT_LIVE_MARKER_BITS
        if 0 <= live_index < shape.output_count:
            if live_index in output_live_records:
                raise CubinFormatError("a output-liveness marker appears more than once")
            output_live_records[live_index] = record
    if set(marker_records) != set(range(marker_count)):
        raise CubinFormatError("register markers are missing or noncontiguous")
    if set(special_records) != {SHARED_BEGIN_MARKER_BITS, SHARED_END_MARKER_BITS}:
        raise CubinFormatError("shared-site boundary markers are missing")
    if set(output_live_records) != set(range(shape.output_count)):
        raise CubinFormatError("output-liveness markers are missing")

    shared_begin = special_records[SHARED_BEGIN_MARKER_BITS][0]
    shared_boundary_end = special_records[SHARED_END_MARKER_BITS][0]
    shared_bpts = [offset for offset in bpts if shared_begin < offset < shared_boundary_end]
    if len(shared_bpts) != shape.shared_patch_capacity:
        raise CubinFormatError(
            f"shared reserve contains {len(shared_bpts)} patch instructions; expected {shape.shared_patch_capacity}"
        )
    shared_start = shared_bpts[0]
    shared_end = shared_bpts[-1] + INSTRUCTION_BYTES
    if tuple(shared_bpts) != tuple(range(shared_start, shared_end, INSTRUCTION_BYTES)):
        raise CubinFormatError("shared site has invalid boundaries")
    shared_instruction_count = (shared_end - shared_start) // INSTRUCTION_BYTES
    if not shared_end <= shared_boundary_end < arena_start:
        raise CubinFormatError("dispatch does not follow the shared specialization site")

    input_markers = [index for index in range(shape.input_count) if marker_records[index][0] < shared_begin]
    output_markers = [
        index
        for index in range(shape.input_count, marker_count)
        if marker_records[index][0] > arena_start
    ]
    if input_markers != list(range(shape.input_count)) or output_markers != list(range(shape.input_count, marker_count)):
        raise CubinFormatError("input/output marker boundaries do not match the declared shape")
    arena_end = min(marker_records[index][0] for index in output_markers)
    if arena_end <= arena_start or (arena_end - arena_start) % INSTRUCTION_BYTES:
        raise CubinFormatError("system arena has invalid boundaries")
    arena_instruction_count = (arena_end - arena_start) // INSTRUCTION_BYTES
    if arena_instruction_count not in {shape.arena_instruction_count - 1, shape.arena_instruction_count}:
        raise CubinFormatError(
            f"arena capacity is {arena_instruction_count}; expected {shape.arena_instruction_count} "
            "or one fewer when PTXAS elides the final fallthrough branch"
        )

    scaffold_start = min(bpts)
    scaffold_end = max(bpts) + INSTRUCTION_BYTES
    if scaffold_end <= arena_end or scaffold_end > end:
        raise CubinFormatError("continuation boundary is invalid")

    input_registers = tuple(marker_records[index][2] for index in input_markers)
    final_output_registers = tuple(marker_records[index][1] for index in output_markers)
    output_registers = tuple(output_live_records[index][1] for index in range(shape.output_count))
    output_materialization_offsets = tuple(marker_records[index][0] for index in output_markers)
    if (
        len(set(input_registers)) != len(input_registers)
        or len(set(output_registers)) != len(output_registers)
        or len(set(final_output_registers)) != len(final_output_registers)
    ):
        raise CubinFormatError("marker registers alias within the input or output ABI")
    if set(input_registers) & (set(output_registers) | set(final_output_registers)):
        raise CubinFormatError("input and output registers overlap")
    for index, output_register in enumerate(output_registers):
        live_record = output_live_records[index]
        final_record = marker_records[shape.input_count + index]
        if live_record[1] != output_register or final_record[2] != output_register:
            raise CubinFormatError("output register is not live across the system dispatch")

    available: list[int] = []
    cleanup: list[int] = []
    incoming_wait_mask = 0
    predicate_register: int | None = None
    permutation_register: int | None = None
    toggle_test_instruction: tuple[int, int] | None = None
    toggle_test_count = 0
    predicate_select_count = 0
    normal_marker_offsets = {record[0] for record in marker_records.values()}
    special_marker_offsets = {record[0] for record in special_records.values()}
    output_live_marker_offsets = {record[0] for record in output_live_records.values()}
    for index in input_markers:
        incoming_wait_mask |= marker_records[index][3]
        _append_unique(available, marker_records[index][1])

    for offset in range(scaffold_start, scaffold_end, INSTRUCTION_BYTES):
        if shared_start <= offset < shared_end or arena_start <= offset < arena_end:
            continue
        word0 = _word(cubin, offset, 0)
        word1 = _word(cubin, offset, 1)
        opcode = word0 & 0xFFFF
        if offset in normal_marker_offsets or offset in special_marker_offsets or offset in output_live_marker_offsets or word0 == BPT_WORD0:
            cleanup.append(offset)
        elif opcode == OPCODE_FADD_REG:
            cleanup.append(offset)
            _append_unique(available, (word0 >> 16) & 0xFF)
        elif opcode == OPCODE_LOP3_IMM and (word0 & 0x00FFFFFF) == 0x00FF7812 and word0 >> 32 == 1:
            toggle_test_count += 1
            predicate = (word1 >> 17) & 0x7
            predicate_mask = 0x7 << 17
            if ((word1 & ((1 << 40) - 1)) & ~predicate_mask) != 0x0780C0FF or toggle_test_count != 1:
                raise CubinFormatError("toggle test marker is not the required immediate bit test")
            predicate_register = predicate
            permutation_register = (word0 >> 24) & 0xFF
            toggle_test_instruction = (word0, word1)
            incoming_wait_mask |= (word1 >> 52) & WAIT_BARRIER_MASK
            cleanup.append(offset)
        elif opcode == OPCODE_FSEL:
            predicate_select_count += 1
            predicate = (word1 >> 23) & 0x7
            predicate_mask = 0x7 << 23
            if (word1 & ((1 << 40) - 1)) & ~predicate_mask:
                raise CubinFormatError("toggle select marker has an unexpected encoding")
            if predicate_register is None or predicate != predicate_register:
                raise CubinFormatError("toggle compare and select markers use different predicates")
            _append_unique(available, (word0 >> 16) & 0xFF)
            cleanup.append(offset)
        elif (opcode & 0x0FFF) == (OPCODE_FADD_REG & 0x0FFF) and opcode >> 12 < 7:
            predicate_select_count += 1
            predicate = opcode >> 12
            if (word1 & ((1 << 40) - 1)) != 0x00010000 or predicate_register is None or predicate != predicate_register:
                raise CubinFormatError("folded toggle select marker has an unexpected encoding")
            _append_unique(available, (word0 >> 16) & 0xFF)
            cleanup.append(offset)
        elif opcode in {OPCODE_STS_RZ, OPCODE_STS_UR} or _is_zero_umov(word0, elf.architecture):
            cleanup.append(offset)

    if (
        toggle_test_count != 1
        or predicate_select_count != 1
        or predicate_register is None
        or permutation_register is None
        or toggle_test_instruction is None
        or predicate_register >= 7
        or permutation_register == 255
    ):
        raise CubinFormatError("marker site did not reserve exactly one writable toggle predicate")
    if permutation_register in set(input_registers) | set(output_registers) | set(final_output_registers):
        raise CubinFormatError("permutation register overlaps the float input/output ABI")
    reserved = set(input_registers) | set(output_registers) | set(final_output_registers) | {permutation_register, 255}
    available_registers = tuple(register for register in available if register not in reserved)
    if not available_registers:
        raise CubinFormatError("marker scaffolding exposed no temporary registers")

    cleanup_set = set(cleanup)
    dispatch_records = tuple(
        (offset, (_word(cubin, offset, 0), _word(cubin, offset, 1)))
        for offset in range(shared_end, arena_start, INSTRUCTION_BYTES)
        if offset not in cleanup_set
    )
    if not 3 <= len(dispatch_records) <= 10:
        raise CubinFormatError(
            f"expected a compact system dispatch sequence, found {len(dispatch_records)} instructions"
        )

    return CubinPlan(
        cubin_size=len(cubin),
        architecture=elf.architecture,
        input_count=shape.input_count,
        output_count=shape.output_count,
        system_capacity=shape.system_capacity,
        register_count=register_count,
        register_count_offsets=register_offsets,
        register_count_header_offsets=header_offsets,
        function=function,
        site=SitePlan(
            scaffold_start_offset=scaffold_start,
            scaffold_end_offset=scaffold_end,
            shared_start_offset=shared_start,
            shared_instruction_count=shared_instruction_count,
            dispatch_offsets=tuple(record[0] for record in dispatch_records),
            dispatch_instructions=tuple(record[1] for record in dispatch_records),
            arena_start_offset=arena_start,
            arena_instruction_count=arena_instruction_count,
            arena_end_offset=arena_end,
            continuation_offset=arena_end,
            incoming_wait_mask=incoming_wait_mask,
            input_registers=input_registers,
            output_registers=output_registers,
            final_output_registers=final_output_registers,
            output_materialization_offsets=output_materialization_offsets,
            available_registers=available_registers,
            predicate_register=predicate_register,
            permutation_register=permutation_register,
            toggle_test_instruction=toggle_test_instruction,
            cleanup_offsets=tuple(cleanup),
            target_table_offsets=tuple(target_section.offset + 4 * index for index in range(shape.system_capacity)),
            original_target_values=tuple(original_targets),
        ),
    )
