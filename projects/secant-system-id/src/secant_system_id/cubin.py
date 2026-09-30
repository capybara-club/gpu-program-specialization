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

from .shape import KernelShape
from .template import FIRST_MARKER_BITS, KERNEL_NAME


INSTRUCTION_BYTES = 16
BPT_WORD0 = 0x000000040000795C
OPCODE_FADD_REG = 0x7221
OPCODE_FADD_IMM = 0x7421
OPCODE_STS_RZ = 0x7388
OPCODE_STS_UR = 0x7988
OPCODE_UMOV = 0x7C82
NVINFO_FORMAT_U32 = 0x04
NVINFO_ATTR_REGCOUNT = 0x2F
NVINFO_ATTR_SIZE = 8
SUPPORTED_SASS_ARCHITECTURES = frozenset((89, 90, 120))
WAIT_BARRIER_MASK = 0x3F


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


@dataclass(frozen=True)
class SitePlan:
    load_fence_offset: int
    start_offset: int
    instruction_count: int
    incoming_wait_mask: int
    input_registers: tuple[int, ...]
    output_registers: tuple[int, ...]
    available_registers: tuple[int, ...]


@dataclass(frozen=True)
class CubinPlan:
    cubin_size: int
    architecture: int
    input_count: int
    output_count: int
    patch_capacity: int
    register_count: int
    register_count_offsets: tuple[int, ...]
    register_count_header_offsets: tuple[int, ...]
    function: Function
    site: SitePlan


class CubinFormatError(ValueError):
    pass


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

        raw_sections: list[Section] = []
        for index in range(section_count):
            values = struct.unpack_from("<IIQQQQIIQQ", data, section_offset + 64 * index)
            raw_sections.append(Section(*values))
        names_section = raw_sections[names_index]
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
            for section in raw_sections
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

    def find_function(self, name: str) -> Function:
        for requested_kind in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            for symbol_table in self.sections:
                if symbol_table.kind != requested_kind:
                    continue
                if symbol_table.entry_size != 24 or symbol_table.size % 24:
                    raise CubinFormatError("invalid ELF64 symbol table")
                if symbol_table.link >= len(self.sections):
                    raise CubinFormatError("symbol table string link is invalid")
                strings_section = self.sections[symbol_table.link]
                strings = self._slice(strings_section.offset, strings_section.size)
                symbols = self._slice(symbol_table.offset, symbol_table.size)
                for symbol_index in range(symbol_table.size // 24):
                    name_offset, info, _other, section_index, value, size = struct.unpack_from(
                        "<IBBHQQ", symbols, 24 * symbol_index
                    )
                    if (info & 0x0F) != 2:
                        continue
                    symbol_name = self._string(strings, name_offset)
                    if symbol_name != name:
                        continue
                    if section_index == 0 or section_index >= len(self.sections) or size == 0:
                        raise CubinFormatError("kernel symbol has an invalid section")
                    section = self.sections[section_index]
                    if value < section.address:
                        raise CubinFormatError("kernel symbol precedes its section")
                    relative = value - section.address
                    file_offset = section.offset + relative
                    self._slice(file_offset, size)
                    if file_offset % INSTRUCTION_BYTES or size % INSTRUCTION_BYTES:
                        raise CubinFormatError("kernel code is not SASS-instruction aligned")
                    return Function(symbol_index, file_offset, size)
        raise CubinFormatError(f"kernel symbol {name!r} was not found")

    def find_data_symbol(self, name: str, expected_size: int | None = None) -> DataSymbol:
        matches: list[DataSymbol] = []
        for requested_kind in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            for symbol_table in self.sections:
                if symbol_table.kind != requested_kind:
                    continue
                if symbol_table.entry_size != 24 or symbol_table.size % 24:
                    raise CubinFormatError("invalid ELF64 symbol table")
                if symbol_table.link >= len(self.sections):
                    raise CubinFormatError("symbol table string link is invalid")
                strings_section = self.sections[symbol_table.link]
                strings = self._slice(strings_section.offset, strings_section.size)
                symbols = self._slice(symbol_table.offset, symbol_table.size)
                for symbol_index in range(symbol_table.size // 24):
                    name_offset, info, _other, section_index, value, size = struct.unpack_from(
                        "<IBBHQQ", symbols, 24 * symbol_index
                    )
                    if (info & 0x0F) != 1 or self._string(strings, name_offset) != name:
                        continue
                    if section_index == 0 or section_index >= len(self.sections) or size == 0:
                        raise CubinFormatError("data symbol has an invalid section or size")
                    section = self.sections[section_index]
                    if value < section.address:
                        raise CubinFormatError("data symbol precedes its section")
                    file_offset = section.offset + value - section.address
                    self._slice(file_offset, size)
                    matches.append(
                        DataSymbol(symbol_index, section_index, file_offset, size)
                    )
        unique = {
            (match.section_index, match.file_offset, match.size): match
            for match in matches
        }
        if len(unique) != 1:
            raise CubinFormatError(
                f"expected one data symbol {name!r}, found {len(unique)}"
            )
        result = next(iter(unique.values()))
        if expected_size is not None and result.size != expected_size:
            raise CubinFormatError(
                f"data symbol {name!r} has size {result.size}, expected {expected_size}"
            )
        return result

    def register_counts(
        self, symbol_index: int
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
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
            if not section.name.startswith(".text."):
                continue
            if (section.info & 0x00FFFFFF) != symbol_index:
                continue
            header_register_count = section.info >> 24
            if header_register_count != 0:
                values.append(header_register_count)
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


def inspect_cubin(
    cubin: bytes,
    shape: KernelShape | None = None,
    kernel_name: str = KERNEL_NAME,
) -> CubinPlan:
    elf = ElfImage(cubin)
    if elf.architecture not in SUPPORTED_SASS_ARCHITECTURES:
        raise CubinFormatError(
            f"the initial Python SASS writer supports sm_89, sm_90, and sm_120, not sm_{elf.architecture}"
        )
    function = elf.find_function(kernel_name)
    register_count, register_count_offsets, register_count_header_offsets = (
        elf.register_counts(function.symbol_index)
    )
    start = function.file_offset
    end = start + function.size
    bpts = [
        offset
        for offset in range(start, end, INSTRUCTION_BYTES)
        if _word(cubin, offset, 0) == BPT_WORD0
    ]
    if len(bpts) < 4:
        raise CubinFormatError("marker site does not contain its three fences and patch reserve")
    first_bpt, last_bpt = bpts[0], bpts[-1]
    patch_capacity = len(bpts) - 3
    second_bpt, third_bpt = bpts[1], bpts[2]
    marker_records: dict[int, tuple[int, int, int, int]] = {}
    for offset in range(first_bpt, last_bpt + INSTRUCTION_BYTES, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        word1 = _word(cubin, offset, 1)
        if (word0 & 0xFFFF) != OPCODE_FADD_IMM:
            continue
        marker_index = (word0 >> 32) - FIRST_MARKER_BITS
        if not 0 <= marker_index < 255:
            continue
        if marker_index in marker_records:
            raise CubinFormatError("a marker immediate appears more than once")
        marker_records[marker_index] = (
            offset,
            (word0 >> 16) & 0xFF,
            (word0 >> 24) & 0xFF,
            (word1 >> 52) & WAIT_BARRIER_MASK,
        )
    if not marker_records:
        raise CubinFormatError("no register markers were found")
    marker_count = max(marker_records) + 1
    if set(marker_records) != set(range(marker_count)):
        raise CubinFormatError("register marker immediates are not contiguous")
    input_markers = sorted(index for index, record in marker_records.items() if first_bpt < record[0] < second_bpt)
    output_markers = sorted(index for index, record in marker_records.items() if second_bpt < record[0] < third_bpt)
    input_count = len(input_markers)
    output_count = len(output_markers)
    if (
        input_markers != list(range(input_count))
        or output_markers != list(range(input_count, input_count + output_count))
        or input_count + output_count != marker_count
        or input_count == 0
        or output_count == 0
    ):
        raise CubinFormatError("could not infer a contiguous input/output marker boundary")
    if shape is not None and (
        shape.input_count != input_count
        or shape.ast_count != output_count
        or shape.patch_capacity != patch_capacity
    ):
        raise CubinFormatError(
            "CUBIN-inferred site shape does not match the supplied KernelShape: "
            f"inputs={input_count}, outputs={output_count}, patch={patch_capacity}"
        )
    marker_seen = [False] * marker_count
    input_registers = [0] * input_count
    output_registers = [0] * output_count
    output_marker_source: int | None = None
    available: list[int] = []
    incoming_wait_mask = 0
    reduction_count = 0
    keepalive_stores = 0

    for offset in range(first_bpt, last_bpt + INSTRUCTION_BYTES, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        word1 = _word(cubin, offset, 1)
        opcode = word0 & 0xFFFF
        immediate = word0 >> 32
        marker_index = immediate - FIRST_MARKER_BITS
        if opcode == OPCODE_FADD_IMM and 0 <= marker_index < marker_count:
            if marker_seen[marker_index]:
                raise CubinFormatError("a marker immediate appears more than once")
            marker_seen[marker_index] = True
            destination = (word0 >> 16) & 0xFF
            source = (word0 >> 24) & 0xFF
            if marker_index < input_count:
                input_registers[marker_index] = source
                incoming_wait_mask |= (word1 >> 52) & WAIT_BARRIER_MASK
                _append_unique(available, destination)
            else:
                if output_marker_source is not None and source != output_marker_source:
                    raise CubinFormatError("materialize outputs do not share their marker source")
                output_marker_source = source
                output_registers[marker_index - input_count] = destination
        elif opcode == OPCODE_FADD_REG:
            reduction_count += 1
            _append_unique(available, (word0 >> 16) & 0xFF)
        elif opcode in {OPCODE_STS_RZ, OPCODE_STS_UR}:
            keepalive_stores += 1
        elif word0 != BPT_WORD0 and not _is_zero_umov(word0, elf.architecture):
            raise CubinFormatError(f"unexpected SASS opcode 0x{opcode:04x} in marker site")

    if not all(marker_seen):
        raise CubinFormatError("one or more marker immediates were not found")
    expected_reductions = (input_count - 1) + (output_count + input_count - 1)
    if reduction_count != expected_reductions or keepalive_stores != 1:
        raise CubinFormatError(
            f"marker reduction shape mismatch: reductions={reduction_count}, stores={keepalive_stores}"
        )
    reserved = set(input_registers) | set(output_registers) | {255}
    available_registers = tuple(register for register in available if register not in reserved)
    if len(set(input_registers)) != len(input_registers):
        raise CubinFormatError("input markers alias physical registers")
    if len(set(output_registers)) != len(output_registers):
        raise CubinFormatError("output markers alias physical registers")
    if set(input_registers) & set(output_registers):
        raise CubinFormatError("input and output registers overlap")
    site_start = first_bpt + INSTRUCTION_BYTES
    site_end = last_bpt + INSTRUCTION_BYTES
    site = SitePlan(
        load_fence_offset=first_bpt,
        start_offset=site_start,
        instruction_count=(site_end - site_start) // INSTRUCTION_BYTES,
        incoming_wait_mask=incoming_wait_mask,
        input_registers=tuple(input_registers),
        output_registers=tuple(output_registers),
        available_registers=available_registers,
    )
    return CubinPlan(
        cubin_size=len(cubin),
        architecture=elf.architecture,
        input_count=input_count,
        output_count=output_count,
        patch_capacity=patch_capacity,
        register_count=register_count,
        register_count_offsets=register_count_offsets,
        register_count_header_offsets=register_count_header_offsets,
        function=function,
        site=site,
    )


def inspect_named_marker_site(
    cubin: bytes,
    input_count: int,
    output_count: int,
    patch_capacity: int,
    marker_bits: int,
    kernel_name: str = KERNEL_NAME,
) -> CubinPlan:
    """Inspect one explicitly-shaped marker when a function contains several.

    Each site receives a disjoint immediate-marker range. Its three fences are
    adjacent to the first input marker, the last input marker, and the last
    output marker. The patch reserve is the first run of patch_capacity BPT
    instructions following the output fence.
    """

    for name, value in (
        ("input_count", input_count),
        ("output_count", output_count),
        ("patch_capacity", patch_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CubinFormatError(f"{name} must be a positive integer")
    marker_count = input_count + output_count
    if marker_count >= 255:
        raise CubinFormatError("marker site cannot expose 255 or more values")

    elf = ElfImage(cubin)
    if elf.architecture not in SUPPORTED_SASS_ARCHITECTURES:
        raise CubinFormatError(
            f"the Python SASS writer does not support sm_{elf.architecture}"
        )
    function = elf.find_function(kernel_name)
    register_count, register_count_offsets, register_count_header_offsets = (
        elf.register_counts(function.symbol_index)
    )
    start = function.file_offset
    end = start + function.size
    marker_records: dict[int, tuple[int, int, int, int]] = {}
    for offset in range(start, end, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        if (word0 & 0xFFFF) != OPCODE_FADD_IMM:
            continue
        marker_index = (word0 >> 32) - marker_bits
        if not 0 <= marker_index < marker_count:
            continue
        if marker_index in marker_records:
            raise CubinFormatError("a named marker immediate appears more than once")
        word1 = _word(cubin, offset, 1)
        marker_records[marker_index] = (
            offset,
            (word0 >> 16) & 0xFF,
            (word0 >> 24) & 0xFF,
            (word1 >> 52) & WAIT_BARRIER_MASK,
        )
    if set(marker_records) != set(range(marker_count)):
        raise CubinFormatError("named marker immediates are missing or noncontiguous")

    first_bpt = marker_records[0][0] - INSTRUCTION_BYTES
    second_bpt = marker_records[input_count - 1][0] + INSTRUCTION_BYTES
    third_bpt = marker_records[marker_count - 1][0] + INSTRUCTION_BYTES
    for name, offset in (
        ("load", first_bpt),
        ("input", second_bpt),
        ("output", third_bpt),
    ):
        if not start <= offset < end or _word(cubin, offset, 0) != BPT_WORD0:
            raise CubinFormatError(f"named marker {name} fence was not preserved")

    reserve_start: int | None = None
    consecutive = 0
    for offset in range(third_bpt + INSTRUCTION_BYTES, end, INSTRUCTION_BYTES):
        if _word(cubin, offset, 0) == BPT_WORD0:
            consecutive += 1
            if consecutive == patch_capacity:
                reserve_start = offset - (patch_capacity - 1) * INSTRUCTION_BYTES
                break
        else:
            consecutive = 0
    if reserve_start is None:
        raise CubinFormatError("named marker patch reserve was not found")
    last_bpt = reserve_start + (patch_capacity - 1) * INSTRUCTION_BYTES
    bpt_count = sum(
        _word(cubin, offset, 0) == BPT_WORD0
        for offset in range(first_bpt, last_bpt + INSTRUCTION_BYTES, INSTRUCTION_BYTES)
    )
    if bpt_count != patch_capacity + 3:
        raise CubinFormatError("named marker site contains an unexpected BPT")

    marker_seen = [False] * marker_count
    input_registers = [0] * input_count
    output_registers = [0] * output_count
    output_marker_source: int | None = None
    available: list[int] = []
    incoming_wait_mask = 0
    reduction_count = 0
    keepalive_stores = 0
    for offset in range(first_bpt, last_bpt + INSTRUCTION_BYTES, INSTRUCTION_BYTES):
        word0 = _word(cubin, offset, 0)
        word1 = _word(cubin, offset, 1)
        opcode = word0 & 0xFFFF
        marker_index = (word0 >> 32) - marker_bits
        if opcode == OPCODE_FADD_IMM and 0 <= marker_index < marker_count:
            if marker_seen[marker_index]:
                raise CubinFormatError("a named marker immediate appears more than once")
            marker_seen[marker_index] = True
            destination = (word0 >> 16) & 0xFF
            source = (word0 >> 24) & 0xFF
            if marker_index < input_count:
                input_registers[marker_index] = source
                incoming_wait_mask |= (word1 >> 52) & WAIT_BARRIER_MASK
                _append_unique(available, destination)
            else:
                if output_marker_source is not None and source != output_marker_source:
                    raise CubinFormatError(
                        "named marker outputs do not share their source"
                    )
                output_marker_source = source
                output_registers[marker_index - input_count] = destination
        elif opcode == OPCODE_FADD_REG:
            reduction_count += 1
            _append_unique(available, (word0 >> 16) & 0xFF)
        elif opcode in {OPCODE_STS_RZ, OPCODE_STS_UR}:
            keepalive_stores += 1
        elif word0 != BPT_WORD0 and not _is_zero_umov(word0, elf.architecture):
            raise CubinFormatError(
                f"unexpected SASS opcode 0x{opcode:04x} in named marker site"
            )

    expected_reductions = (input_count - 1) + (
        output_count + input_count - 1
    )
    if (
        not all(marker_seen)
        or reduction_count != expected_reductions
        or keepalive_stores != 1
    ):
        raise CubinFormatError(
            "named marker reduction shape does not match its declared ABI"
        )
    if len(set(input_registers)) != input_count:
        raise CubinFormatError("named marker inputs alias physical registers")
    if len(set(output_registers)) != output_count:
        raise CubinFormatError("named marker outputs alias physical registers")
    if set(input_registers) & set(output_registers):
        raise CubinFormatError("named marker inputs and outputs overlap")
    reserved = set(input_registers) | set(output_registers) | {255}
    available_registers = tuple(
        register for register in available if register not in reserved
    )
    site_start = first_bpt + INSTRUCTION_BYTES
    site_end = last_bpt + INSTRUCTION_BYTES
    return CubinPlan(
        cubin_size=len(cubin),
        architecture=elf.architecture,
        input_count=input_count,
        output_count=output_count,
        patch_capacity=patch_capacity,
        register_count=register_count,
        register_count_offsets=register_count_offsets,
        register_count_header_offsets=register_count_header_offsets,
        function=function,
        site=SitePlan(
            load_fence_offset=first_bpt,
            start_offset=site_start,
            instruction_count=(site_end - site_start) // INSTRUCTION_BYTES,
            incoming_wait_mask=incoming_wait_mask,
            input_registers=tuple(input_registers),
            output_registers=tuple(output_registers),
            available_registers=available_registers,
        ),
    )
