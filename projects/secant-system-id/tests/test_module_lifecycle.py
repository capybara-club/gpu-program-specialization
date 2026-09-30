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

from secant_system_id.module_lifecycle import (
    KERNEL_NAME,
    NONCE_MAGIC,
    find_nonce_offset,
    generate_module_lifecycle_cuda,
)


class ModuleLifecycleBenchmarkTest(unittest.TestCase):
    def test_source_has_early_exit_verification_and_brkpt_padding(self) -> None:
        source = generate_module_lifecycle_cuda(13)
        self.assertIn(f'extern "C" __global__ void {KERNEL_NAME}', source)
        self.assertIn("const unsigned int gate_value = gate[0];", source)
        self.assertIn("unsigned long long wait_clocks", source)
        self.assertIn("while (clock64() - started < wait_clocks)", source)
        self.assertIn("sink[0] = secant_module_lifecycle_nonce;", source)
        self.assertIn("SSID_BRKPT_8", source)
        self.assertIn("SSID_BRKPT_4", source)
        self.assertIn("SSID_BRKPT_1", source)

    def test_zero_padding_keeps_a_reachable_runtime_tail(self) -> None:
        source = generate_module_lifecycle_cuda(0)
        self.assertIn('asm volatile("");', source)

    def test_global_padding_does_not_expand_source(self) -> None:
        source = generate_module_lifecycle_cuda(global_padding_bytes=3_500_000)
        self.assertIn("secant_module_lifecycle_payload[3500000] = {0x5a}", source)
        self.assertLess(len(source), 4096)

    def test_multiple_entry_points_split_the_total_padding(self) -> None:
        source = generate_module_lifecycle_cuda(padding_brkpts=10, kernel_count=4)
        for index in range(4):
            self.assertIn(f"void {KERNEL_NAME}_{index:03d}(", source)
        self.assertNotIn(f"void {KERNEL_NAME}(", source)

    def test_nonce_offset_requires_exactly_one_marker(self) -> None:
        marker = struct.pack("<Q", NONCE_MAGIC)
        self.assertEqual(find_nonce_offset(b"abc" + marker + b"xyz"), 3)
        with self.assertRaises(ValueError):
            find_nonce_offset(b"no marker")
        with self.assertRaises(ValueError):
            find_nonce_offset(marker + marker)

    def test_padding_count_validation(self) -> None:
        with self.assertRaises(ValueError):
            generate_module_lifecycle_cuda(-1)
        with self.assertRaises(TypeError):
            generate_module_lifecycle_cuda(True)
        with self.assertRaises(ValueError):
            generate_module_lifecycle_cuda(1, 1)
        with self.assertRaises(ValueError):
            generate_module_lifecycle_cuda(kernel_count=0)
        with self.assertRaises(ValueError):
            generate_module_lifecycle_cuda(padding_brkpts=65_537, kernel_count=1)
        generate_module_lifecycle_cuda(padding_brkpts=65_537, kernel_count=2)


if __name__ == "__main__":
    unittest.main()
