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

import struct
import unittest

from secant_system_id.cubin import CubinFormatError, ElfImage


def _minimal_cubin(
    register_count: int = 72,
    symbol_index: int = 7,
    header_register_count: int | None = None,
) -> tuple[bytes, int, int]:
    section_table_offset = 64
    section_count = 4
    names = b"\0.shstrtab\0.text.test_kernel\0.nv.info.test_kernel\0"
    names_offset = section_table_offset + section_count * 64
    text_offset = (names_offset + len(names) + 15) & ~15
    nvinfo_offset = text_offset + 16
    image = bytearray(nvinfo_offset + 12)
    if header_register_count is None:
        header_register_count = register_count

    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        image,
        0,
        b"\x7fELF\x02\x01\x01" + bytes(9),
        2,
        190,
        1,
        0,
        0,
        section_table_offset,
        89,
        64,
        0,
        0,
        64,
        section_count,
        1,
    )
    image[names_offset : names_offset + len(names)] = names
    section_header_format = "<IIQQQQIIQQ"
    struct.pack_into(
        section_header_format,
        image,
        section_table_offset + 64,
        names.index(b".shstrtab"),
        3,
        0,
        0,
        names_offset,
        len(names),
        0,
        0,
        1,
        0,
    )
    struct.pack_into(
        section_header_format,
        image,
        section_table_offset + 128,
        names.index(b".text.test_kernel"),
        1,
        0,
        0,
        text_offset,
        16,
        0,
        (header_register_count << 24) | symbol_index,
        16,
        0,
    )
    struct.pack_into(
        section_header_format,
        image,
        section_table_offset + 192,
        names.index(b".nv.info.test_kernel"),
        1,
        0,
        0,
        nvinfo_offset,
        12,
        0,
        0,
        4,
        0,
    )
    struct.pack_into("<BBHII", image, nvinfo_offset, 0x04, 0x2F, 8, symbol_index, register_count)
    header_count_offset = section_table_offset + 128 + 47
    nvinfo_count_offset = nvinfo_offset + 8
    return bytes(image), nvinfo_count_offset, header_count_offset


class CubinRegisterCountTest(unittest.TestCase):
    def test_register_count_includes_text_section_header_byte_offset(self) -> None:
        cubin, nvinfo_offset, header_offset = _minimal_cubin()

        count, nvinfo_offsets, header_offsets = ElfImage(cubin).register_counts(7)

        self.assertEqual(count, 72)
        self.assertEqual(nvinfo_offsets, (nvinfo_offset,))
        self.assertEqual(header_offsets, (header_offset,))
        self.assertEqual(cubin[header_offset], count)

    def test_register_count_rejects_disagreement_between_metadata_locations(self) -> None:
        cubin, _nvinfo_offset, header_offset = _minimal_cubin()
        inconsistent = bytearray(cubin)
        inconsistent[header_offset] = 71

        with self.assertRaisesRegex(CubinFormatError, "consistent kernel register count"):
            ElfImage(bytes(inconsistent)).register_counts(7)

    def test_zero_text_section_count_uses_only_nvinfo_metadata(self) -> None:
        cubin, nvinfo_offset, _header_offset = _minimal_cubin(
            header_register_count=0
        )

        count, nvinfo_offsets, header_offsets = ElfImage(cubin).register_counts(7)

        self.assertEqual(count, 72)
        self.assertEqual(nvinfo_offsets, (nvinfo_offset,))
        self.assertEqual(header_offsets, ())


if __name__ == "__main__":
    unittest.main()
