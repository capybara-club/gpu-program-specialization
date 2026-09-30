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
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
import json
from pathlib import Path
import struct
import sys
from typing import TextIO

from elftools.common.exceptions import ELFError
from elftools.construct import ConstructError
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


_NAME_BYTES = 256
_INPUTS = 8
_MAX_OUTPUTS = 64
_MAX_SITE_INSTRUCTIONS = 8192
_INSTRUCTION_BYTES = 16
_WORD_BYTES = 8
_BPT_WORD0 = 0x000000040000795C
_FIRST_MARKER = 0x7FC0FFEE
_MARKER_STRIDE = 128
_MAX_PRE_MARKER_SETUP = 8
_TARGET_MARKER_OFFSET = 8
_SSE_MARKER_OFFSET = 16
_PAD_INSTRUCTIONS_PER_AST = 64

_OPCODE_FADD_IMM = 0x7421
_OPCODE_ULEA = 0x7291
_OPCODE_UMOV = 0x7882
_OPCODE_S2UR = 0x79C3
_OPCODE_STS_REG = 0x7388
_OPCODE_STS_UR = 0x7988

_NVINFO_FORMAT_U32 = 0x04
_NVINFO_ATTR_MAXREG_COUNT = 0x2F
_NVINFO_ATTR_SIZE_U32_PAIR = 8

_ELF64_HEADER_BYTES = 64
_ELF64_SECTION_BYTES = 64
_ELF64_SYMBOL_BYTES = 24
_SHT_SYMTAB = "SHT_SYMTAB"
_STT_FUNC = "STT_FUNC"
_SHN_UNDEF = "SHN_UNDEF"


class CubinInspectErrorCode(str, Enum):
    INVALID_VALUE = "CUSR_SASS_INSPECT_ERROR_INVALID_VALUE"
    OVERFLOW = "CUSR_SASS_INSPECT_ERROR_OVERFLOW"
    BAD_ELF = "CUSR_SASS_INSPECT_ERROR_BAD_ELF"
    FUNCTION_NOT_FOUND = "CUSR_SASS_INSPECT_ERROR_FUNCTION_NOT_FOUND"
    BAD_FUNCTION = "CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION"
    REGCOUNT_NOT_FOUND = "CUSR_SASS_INSPECT_ERROR_REGCOUNT_NOT_FOUND"
    SITE_NOT_FOUND = "CUSR_SASS_INSPECT_ERROR_SITE_NOT_FOUND"
    BAD_SITE = "CUSR_SASS_INSPECT_ERROR_BAD_SITE"


class CubinInspectError(ValueError):
    def __init__(self, code: CubinInspectErrorCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code


def _fail(code: CubinInspectErrorCode, message: str) -> None:
    raise CubinInspectError(code, message)


@dataclass(frozen=True, slots=True)
class RegcountRecord:
    kernel_index: int
    symbol_index: int
    section_name: str
    tag_file_offset: int
    value_file_offset: int
    value: int


@dataclass(frozen=True, slots=True)
class SassSite:
    kernel_index: int
    occurrence_index: int
    marker: int
    start_instruction: int
    end_instruction: int
    start_file_offset: int
    end_file_offset: int
    start_address: int
    end_address: int
    input_regs: tuple[int, ...]
    target_reg: int
    output_regs: tuple[int, ...]
    available_regs: tuple[int, ...]
    incoming_wait_mask: int


@dataclass(frozen=True, slots=True)
class SassKernel:
    name: str
    kernel_index: int
    symbol_index: int
    text_section_name: str
    start_address: int
    end_address: int
    start_file_offset: int
    end_file_offset: int
    first_instruction: int
    num_instructions: int
    first_regcount_record: int
    num_regcount_records: int
    first_site: int
    num_sites: int
    regcount_records: tuple[RegcountRecord, ...]
    sites: tuple[SassSite, ...]


@dataclass(frozen=True, slots=True)
class CubinInspection:
    cubin_size: int
    sass_arch: int
    num_kernels: int
    ast_capacity: int
    first_marker_bits: int
    expected_occurrences: int
    num_sites: int
    num_regcount_records: int
    kernels: tuple[SassKernel, ...]

    def to_dict(self) -> dict[str, object]:
        structured = _json_value(asdict(self))
        if not isinstance(structured, dict):
            raise TypeError("inspection did not convert to a dictionary")
        return structured


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class _Section:
    index: int
    address: int
    offset: int
    size: int
    name: str


@dataclass(frozen=True, slots=True)
class _Symbol:
    index: int
    symbol_type: str
    section_index: int | str
    value: int
    size: int
    name: str


@dataclass(frozen=True, slots=True)
class _Elf:
    data: bytes
    flags: int
    sections: tuple[_Section, ...]
    symbols: tuple[_Symbol, ...]


@dataclass(frozen=True, slots=True)
class _Function:
    name: str
    symbol_index: int
    text_section_name: str
    start_address: int
    end_address: int
    start_file_offset: int
    end_file_offset: int
    first_instruction: int
    num_instructions: int


def _range_ok(size: int, offset: int, count: int) -> bool:
    return 0 <= offset <= size and 0 <= count <= size - offset


def _ascii_elf_name(name: str, description: str) -> str:
    if not isinstance(name, str):
        _fail(CubinInspectErrorCode.BAD_ELF, f"{description} is not a string")
    try:
        name.encode("ascii")
    except UnicodeEncodeError as error:
        raise CubinInspectError(CubinInspectErrorCode.BAD_ELF, f"non-ASCII {description}") from error
    return name


def _parse_elf_with_pyelftools(data: bytes) -> _Elf:
    if not _range_ok(len(data), 0, _ELF64_HEADER_BYTES):
        _fail(CubinInspectErrorCode.BAD_ELF, "cubin is smaller than an ELF64 header")

    parsed = ELFFile(BytesIO(data))
    section_offset = int(parsed.header["e_shoff"])
    section_entry_size = int(parsed.header["e_shentsize"])
    section_count = parsed.num_sections()
    if parsed.elfclass != 64 or not parsed.little_endian:
        _fail(CubinInspectErrorCode.BAD_ELF, "cubin must be a little-endian ELF64 file")
    if section_entry_size != _ELF64_SECTION_BYTES or section_count == 0 or not _range_ok(
        len(data), section_offset, section_count * section_entry_size
    ):
        _fail(CubinInspectErrorCode.BAD_ELF, "unsupported or malformed ELF header")

    parsed_sections = tuple(parsed.iter_sections())
    if len(parsed_sections) != section_count:
        _fail(CubinInspectErrorCode.BAD_ELF, "ELF section table is incomplete")

    sections = []
    for index, section in enumerate(parsed_sections):
        header = section.header
        sections.append(
            _Section(
                index=index,
                address=int(header["sh_addr"]),
                offset=int(header["sh_offset"]),
                size=int(header["sh_size"]),
                name=_ascii_elf_name(section.name, "ELF section name"),
            )
        )

    symbol_section = next(
        (
            section
            for section in parsed_sections
            if isinstance(section, SymbolTableSection) and section.header["sh_type"] == _SHT_SYMTAB
        ),
        None,
    )
    if symbol_section is None:
        _fail(CubinInspectErrorCode.BAD_ELF, "ELF symbol table is missing or malformed")

    symbol_header = symbol_section.header
    symbol_offset = int(symbol_header["sh_offset"])
    symbol_size = int(symbol_header["sh_size"])
    symbol_entry_size = int(symbol_header["sh_entsize"])
    string_section_index = int(symbol_header["sh_link"])
    if (
        symbol_entry_size != _ELF64_SYMBOL_BYTES
        or symbol_size % symbol_entry_size != 0
        or not 0 <= string_section_index < section_count
    ):
        _fail(CubinInspectErrorCode.BAD_ELF, "ELF symbol table is missing or malformed")

    string_section = sections[string_section_index]
    if not _range_ok(len(data), symbol_offset, symbol_size) or not _range_ok(
        len(data), string_section.offset, string_section.size
    ):
        _fail(CubinInspectErrorCode.BAD_ELF, "symbol or string table is outside the cubin")

    symbols = []
    for index, symbol in enumerate(symbol_section.iter_symbols()):
        entry = symbol.entry
        symbols.append(
            _Symbol(
                index=index,
                symbol_type=str(entry["st_info"]["type"]),
                section_index=entry["st_shndx"],
                value=int(entry["st_value"]),
                size=int(entry["st_size"]),
                name=_ascii_elf_name(symbol.name, "ELF symbol name"),
            )
        )

    if len(symbols) != symbol_size // symbol_entry_size:
        _fail(CubinInspectErrorCode.BAD_ELF, "ELF symbol table is incomplete")

    return _Elf(
        data=data,
        flags=int(parsed.header["e_flags"]),
        sections=tuple(sections),
        symbols=tuple(symbols),
    )


def _parse_elf(data: bytes) -> _Elf:
    try:
        return _parse_elf_with_pyelftools(data)
    except CubinInspectError:
        raise
    except (ELFError, ConstructError, IndexError, KeyError, OverflowError, TypeError, UnicodeError, ValueError) as error:
        raise CubinInspectError(CubinInspectErrorCode.BAD_ELF, "malformed ELF cubin") from error


def _find_function(elf: _Elf, name: str) -> _Function:
    for symbol in elf.symbols:
        if symbol.name != name or symbol.symbol_type != _STT_FUNC:
            continue
        if (
            symbol.section_index == _SHN_UNDEF
            or not isinstance(symbol.section_index, int)
            or not 0 <= symbol.section_index < len(elf.sections)
        ):
            _fail(CubinInspectErrorCode.BAD_FUNCTION, f"{name} has an invalid text section")

        section = elf.sections[symbol.section_index]
        if symbol.size == 0 or symbol.size % _INSTRUCTION_BYTES != 0:
            _fail(CubinInspectErrorCode.BAD_FUNCTION, f"{name} has an invalid instruction size")
        if (
            symbol.value > section.size
            or symbol.size > section.size - symbol.value
            or not _range_ok(len(elf.data), section.offset + symbol.value, symbol.size)
        ):
            _fail(CubinInspectErrorCode.BAD_FUNCTION, f"{name} extends outside its text section")

        start_file_offset = section.offset + symbol.value
        if start_file_offset % _INSTRUCTION_BYTES != 0:
            _fail(CubinInspectErrorCode.BAD_FUNCTION, f"{name} is not instruction aligned")
        if len(name.encode("ascii")) + 1 > _NAME_BYTES:
            _fail(CubinInspectErrorCode.BAD_FUNCTION, f"{name} exceeds the inspector name capacity")

        start_address = section.address + symbol.value
        return _Function(
            name=name,
            symbol_index=symbol.index,
            text_section_name=section.name,
            start_address=start_address,
            end_address=start_address + symbol.size,
            start_file_offset=start_file_offset,
            end_file_offset=start_file_offset + symbol.size,
            first_instruction=start_file_offset // _INSTRUCTION_BYTES,
            num_instructions=symbol.size // _INSTRUCTION_BYTES,
        )

    _fail(CubinInspectErrorCode.FUNCTION_NOT_FOUND, f"ELF function not found: {name}")


def _word0(elf: _Elf, instruction: int) -> int:
    return struct.unpack_from("<Q", elf.data, instruction * _INSTRUCTION_BYTES)[0]


def _word1(elf: _Elf, instruction: int) -> int:
    return struct.unpack_from("<Q", elf.data, instruction * _INSTRUCTION_BYTES + _WORD_BYTES)[0]


def _opcode(word0: int) -> int:
    return word0 & 0xFFFF


def _dst_reg(word0: int) -> int:
    return (word0 >> 16) & 0xFF


def _src0_reg(word0: int) -> int:
    return (word0 >> 24) & 0xFF


def _immediate(word0: int) -> int:
    return (word0 >> 32) & 0xFFFFFFFF


def _wait_mask(word1: int) -> int:
    return (word1 >> 52) & 0xFFF


def _is_bpt(word0: int) -> bool:
    return word0 == _BPT_WORD0


def _is_sts(word0: int) -> bool:
    return _opcode(word0) in (_OPCODE_STS_REG, _OPCODE_STS_UR)


def _is_pre_bpt_anchor_setup(word0: int) -> bool:
    return _opcode(word0) in (_OPCODE_ULEA, _OPCODE_UMOV, _OPCODE_S2UR)


def _find_site_start(elf: _Elf, function_first: int, marker_instruction: int) -> int:
    if marker_instruction <= function_first:
        _fail(CubinInspectErrorCode.BAD_SITE, "marker precedes the patch-site opening breakpoint")

    instruction = marker_instruction
    while instruction > function_first:
        instruction -= 1
        word0 = _word0(elf, instruction)
        if _is_bpt(word0):
            return instruction
        if marker_instruction - instruction > _MAX_PRE_MARKER_SETUP or not _is_pre_bpt_anchor_setup(word0):
            _fail(CubinInspectErrorCode.BAD_SITE, "unexpected instruction before the first marker")

    _fail(CubinInspectErrorCode.BAD_SITE, "patch-site opening breakpoint was not found")


def _scan_one_site(
    elf: _Elf,
    function: _Function,
    kernel_index: int,
    marker: int,
    ast_capacity: int,
    found_instruction: int,
    occurrence_index: int,
) -> SassSite:
    function_first = function.first_instruction
    function_last = function_first + function.num_instructions
    if found_instruction < function_first or found_instruction >= function_last:
        _fail(CubinInspectErrorCode.BAD_SITE, "marker is outside the kernel")

    site_start = _find_site_start(elf, function_first, found_instruction)
    expected_trailing_bpts = ast_capacity * _PAD_INSTRUCTIONS_PER_AST + 1
    input_regs = [0] * _INPUTS
    input_seen = [False] * _INPUTS
    output_regs = [0] * ast_capacity
    output_seen = [False] * ast_capacity
    available_regs: list[int] = []
    target_reg = 0
    target_seen = False
    incoming_wait_mask = 0
    last_role_instruction = 0
    store_count = 0
    trailing_bpt_count = 0
    instruction = site_start
    instruction_limit = min(function_last, site_start + _MAX_SITE_INSTRUCTIONS)

    def add_available(reg: int) -> None:
        if reg not in available_regs:
            available_regs.append(reg)

    while instruction < instruction_limit:
        word0 = _word0(elf, instruction)
        opcode = _opcode(word0)

        if opcode == _OPCODE_FADD_IMM:
            immediate = _immediate(word0)
            if immediate >= marker and immediate - marker < _MARKER_STRIDE:
                role = immediate - marker
                dst = _dst_reg(word0)
                src = _src0_reg(word0)
                word1 = _word1(elf, instruction)

                if role < _INPUTS:
                    if input_seen[role]:
                        _fail(CubinInspectErrorCode.BAD_SITE, "duplicate input marker")
                    input_seen[role] = True
                    input_regs[role] = src
                    incoming_wait_mask |= _wait_mask(word1)
                    add_available(dst)
                elif role == _TARGET_MARKER_OFFSET:
                    if target_seen:
                        _fail(CubinInspectErrorCode.BAD_SITE, "duplicate target marker")
                    target_seen = True
                    target_reg = src
                    incoming_wait_mask |= _wait_mask(word1)
                    add_available(dst)
                elif _SSE_MARKER_OFFSET <= role < _SSE_MARKER_OFFSET + ast_capacity:
                    output_index = role - _SSE_MARKER_OFFSET
                    if output_seen[output_index] or dst != src:
                        _fail(CubinInspectErrorCode.BAD_SITE, "invalid SSE marker")
                    output_seen[output_index] = True
                    output_regs[output_index] = dst
                else:
                    _fail(CubinInspectErrorCode.BAD_SITE, "marker has an unknown patch-site role")

                last_role_instruction = instruction
                trailing_bpt_count = 0
                instruction += 1
                continue

        if _is_sts(word0):
            store_count += 1
            trailing_bpt_count = 0
            instruction += 1
            continue

        if _is_bpt(word0):
            if last_role_instruction != 0 and instruction > last_role_instruction:
                trailing_bpt_count += 1
            instruction += 1
            continue

        if _is_pre_bpt_anchor_setup(word0) and store_count < _INPUTS + 1:
            trailing_bpt_count = 0
            instruction += 1
            continue

        if last_role_instruction != 0 and trailing_bpt_count != 0:
            break

        _fail(CubinInspectErrorCode.BAD_SITE, "unexpected instruction inside the patch-site skeleton")

    if store_count != _INPUTS + 1 or not target_seen or trailing_bpt_count != expected_trailing_bpts:
        _fail(CubinInspectErrorCode.BAD_SITE, "patch-site anchors or trailing breakpoint pad are incomplete")
    if not all(input_seen) or not all(output_seen):
        _fail(CubinInspectErrorCode.BAD_SITE, "patch-site register markers are incomplete")

    if len(set(input_regs)) != len(input_regs):
        _fail(CubinInspectErrorCode.BAD_SITE, "input registers are not distinct")
    if len(set(output_regs)) != len(output_regs) or target_reg in output_regs:
        _fail(CubinInspectErrorCode.BAD_SITE, "output and target registers are not disjoint")
    if set(input_regs).intersection(output_regs):
        _fail(CubinInspectErrorCode.BAD_SITE, "input and output registers are not disjoint")

    reserved = set(input_regs)
    reserved.update(output_regs)
    reserved.add(target_reg)
    reserved.add(0xFF)
    available_regs = [reg for reg in available_regs if reg not in reserved]

    site_end = instruction
    if (
        site_end <= site_start
        or site_end > function_last
        or site_end - site_start > _MAX_SITE_INSTRUCTIONS
    ):
        _fail(CubinInspectErrorCode.BAD_SITE, "patch-site span is invalid")
    if site_end == site_start + _MAX_SITE_INSTRUCTIONS and site_end < function_last and _is_bpt(_word0(elf, site_end)):
        _fail(CubinInspectErrorCode.BAD_SITE, "patch-site breakpoint pad exceeds the supported limit")

    start_file_offset = site_start * _INSTRUCTION_BYTES
    end_file_offset = site_end * _INSTRUCTION_BYTES
    return SassSite(
        kernel_index=kernel_index,
        occurrence_index=occurrence_index,
        marker=marker,
        start_instruction=site_start,
        end_instruction=site_end,
        start_file_offset=start_file_offset,
        end_file_offset=end_file_offset,
        start_address=function.start_address + start_file_offset - function.start_file_offset,
        end_address=function.start_address + end_file_offset - function.start_file_offset,
        input_regs=tuple(input_regs),
        target_reg=target_reg,
        output_regs=tuple(output_regs),
        available_regs=tuple(available_regs),
        incoming_wait_mask=incoming_wait_mask,
    )


def _scan_sites(
    elf: _Elf,
    function: _Function,
    kernel_index: int,
    marker: int,
    ast_capacity: int,
    expected_occurrences: int,
) -> tuple[SassSite, ...]:
    first_instruction = function.first_instruction
    last_instruction = first_instruction + function.num_instructions
    search_instruction = first_instruction
    sites = []

    while search_instruction < last_instruction:
        found_instruction = None
        for instruction in range(search_instruction, last_instruction):
            word0 = _word0(elf, instruction)
            if _opcode(word0) == _OPCODE_FADD_IMM and _immediate(word0) == marker:
                found_instruction = instruction
                break
        if found_instruction is None:
            break

        site = _scan_one_site(
            elf,
            function,
            kernel_index,
            marker,
            ast_capacity,
            found_instruction,
            len(sites),
        )
        sites.append(site)
        search_instruction = site.end_instruction

    if not sites:
        _fail(CubinInspectErrorCode.SITE_NOT_FOUND, f"patch site not found in {function.name}")
    if expected_occurrences and len(sites) != expected_occurrences:
        _fail(CubinInspectErrorCode.BAD_SITE, f"unexpected patch-site occurrence count in {function.name}")
    return tuple(sites)


def _collect_regcounts(elf: _Elf, kernel_index: int, symbol_index: int) -> tuple[RegcountRecord, ...]:
    records = []
    for section in elf.sections:
        if ".nv.info" not in section.name:
            continue
        if not _range_ok(len(elf.data), section.offset, section.size):
            _fail(CubinInspectErrorCode.BAD_ELF, f"{section.name} extends outside the cubin")

        for section_offset in range(0, max(0, section.size - 11), 4):
            file_offset = section.offset + section_offset
            entry = elf.data[file_offset : file_offset + 12]
            if (
                entry[0] != _NVINFO_FORMAT_U32
                or entry[1] != _NVINFO_ATTR_MAXREG_COUNT
                or int.from_bytes(entry[2:4], "little") != _NVINFO_ATTR_SIZE_U32_PAIR
                or int.from_bytes(entry[4:8], "little") != symbol_index
            ):
                continue
            records.append(
                RegcountRecord(
                    kernel_index=kernel_index,
                    symbol_index=symbol_index,
                    section_name=section.name,
                    tag_file_offset=file_offset,
                    value_file_offset=file_offset + 8,
                    value=int.from_bytes(entry[8:12], "little"),
                )
            )

    if not records:
        _fail(CubinInspectErrorCode.REGCOUNT_NOT_FOUND, f"regcount records not found for symbol {symbol_index}")
    return tuple(records)


def _sass_arch(flags: int) -> int:
    arch = flags & 0xFF
    if arch >= 30:
        return arch
    arch = (flags >> 8) & 0xFF
    return arch if arch >= 30 else 0


def inspect_cubin(
    cubin: bytes | bytearray | memoryview,
    function_names: Sequence[str],
    *,
    ast_capacity: int,
    first_marker_bits: int = _FIRST_MARKER,
    expected_occurrences: int = 0,
) -> CubinInspection:
    try:
        data = bytes(cubin)
    except (TypeError, ValueError) as error:
        raise CubinInspectError(CubinInspectErrorCode.INVALID_VALUE, "cubin must be bytes-like") from error
    if isinstance(function_names, (str, bytes)):
        _fail(CubinInspectErrorCode.INVALID_VALUE, "function_names must be a sequence of complete names")
    function_names = tuple(function_names)

    if (
        isinstance(ast_capacity, bool)
        or not isinstance(ast_capacity, int)
        or isinstance(first_marker_bits, bool)
        or not isinstance(first_marker_bits, int)
        or isinstance(expected_occurrences, bool)
        or not isinstance(expected_occurrences, int)
    ):
        _fail(CubinInspectErrorCode.INVALID_VALUE, "inspection dimensions must be integers")
    if not data or not function_names or not 1 <= ast_capacity <= _MAX_OUTPUTS:
        _fail(CubinInspectErrorCode.INVALID_VALUE, "cubin, function names, and AST capacity must be non-empty")
    if any(not isinstance(name, str) or not name for name in function_names):
        _fail(CubinInspectErrorCode.INVALID_VALUE, "function names must be non-empty strings")
    try:
        for name in function_names:
            name.encode("ascii")
    except UnicodeEncodeError as error:
        raise CubinInspectError(CubinInspectErrorCode.INVALID_VALUE, "function names must be ASCII") from error
    if not 0 <= first_marker_bits <= 0xFFFFFFFF or expected_occurrences < 0:
        _fail(CubinInspectErrorCode.INVALID_VALUE, "marker and occurrence values are out of range")

    elf = _parse_elf(data)
    kernels = []
    all_regcounts: list[RegcountRecord] = []
    all_sites: list[SassSite] = []

    for kernel_index, function_name in enumerate(function_names):
        marker_offset = kernel_index * _MARKER_STRIDE
        if marker_offset > 0xFFFFFFFF - first_marker_bits:
            _fail(CubinInspectErrorCode.OVERFLOW, "kernel marker range exceeds uint32")

        function = _find_function(elf, function_name)
        regcount_records = _collect_regcounts(elf, kernel_index, function.symbol_index)
        sites = _scan_sites(
            elf,
            function,
            kernel_index,
            first_marker_bits + marker_offset,
            ast_capacity,
            expected_occurrences,
        )
        first_regcount_record = len(all_regcounts)
        first_site = len(all_sites)
        all_regcounts.extend(regcount_records)
        all_sites.extend(sites)
        kernels.append(
            SassKernel(
                name=function.name,
                kernel_index=kernel_index,
                symbol_index=function.symbol_index,
                text_section_name=function.text_section_name,
                start_address=function.start_address,
                end_address=function.end_address,
                start_file_offset=function.start_file_offset,
                end_file_offset=function.end_file_offset,
                first_instruction=function.first_instruction,
                num_instructions=function.num_instructions,
                first_regcount_record=first_regcount_record,
                num_regcount_records=len(regcount_records),
                first_site=first_site,
                num_sites=len(sites),
                regcount_records=regcount_records,
                sites=sites,
            )
        )

    return CubinInspection(
        cubin_size=len(data),
        sass_arch=_sass_arch(elf.flags),
        num_kernels=len(kernels),
        ast_capacity=ast_capacity,
        first_marker_bits=first_marker_bits,
        expected_occurrences=expected_occurrences,
        num_sites=len(all_sites),
        num_regcount_records=len(all_regcounts),
        kernels=tuple(kernels),
    )


def inspect_cubin_file(
    cubin_path: str | Path,
    function_names: Sequence[str],
    *,
    ast_capacity: int,
    first_marker_bits: int = 0x7FC0FFEE,
    expected_occurrences: int = 0,
) -> CubinInspection:
    return inspect_cubin(
        Path(cubin_path).read_bytes(),
        function_names,
        ast_capacity=ast_capacity,
        first_marker_bits=first_marker_bits,
        expected_occurrences=expected_occurrences,
    )


def print_cubin_inspection(inspection: CubinInspection, file: TextIO | None = None) -> None:
    stream = sys.stdout if file is None else file
    json.dump(inspection.to_dict(), stream, indent=2)
    stream.write("\n")
