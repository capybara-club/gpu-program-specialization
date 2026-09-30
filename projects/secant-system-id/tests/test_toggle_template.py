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

import unittest

from secant_system_id.fed_batch import FED_BATCH_MODEL, FED_BATCH_SHAPE
from secant_system_id.toggle_fed_batch import benchmark_toggle_systems, make_toggle_population
from secant_system_id.toggle_template import (
    TOGGLE_KERNEL_NAME,
    generate_toggle_cuda,
    generate_toggle_cuda_module,
)


class ToggleTemplateTest(unittest.TestCase):
    def test_cuda_packs_systems_and_maps_configuration_bits(self) -> None:
        systems = benchmark_toggle_systems(4)
        source = generate_toggle_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE, systems, 5)
        self.assertIn(f'extern "C" __global__ void {TOGGLE_KERNEL_NAME}', source)
        self.assertIn("#define SSID_PACKED_SYSTEM_COUNT 4", source)
        self.assertIn("#define SSID_TOGGLE_MASK 31u", source)
        self.assertIn("const unsigned int system_index = blockIdx.y;", source)
        self.assertIn("configuration >> SSID_TOGGLE_BIT_COUNT", source)
        self.assertEqual(source.count("case 0u:"), 1)
        self.assertEqual(source.count("case 3u:"), 1)

    def test_dynamic_bank_is_removed_but_reference_reuse_remains(self) -> None:
        source = generate_toggle_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            benchmark_toggle_systems(2),
            5,
        )
        self.assertIn("extern __shared__ float reference[];", source)
        self.assertIn("const float constant0 = constant_banks[constant_base + 0ull];", source)
        self.assertIn("stage_state[1]", source)
        self.assertIn("__fmul_rn(", source)
        self.assertIn("__fadd_rn(", source)
        self.assertIn("__fdividef(", source)
        self.assertNotIn("leaf_bindings", source)
        self.assertNotIn("volatile float *bank", source)
        self.assertNotIn("bank[", source)

    def test_five_bits_and_four_constant_banks_make_128_configurations(self) -> None:
        source = generate_toggle_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            benchmark_toggle_systems(1),
            5,
        )
        self.assertIn(
            "available_configuration_count = (unsigned long long)num_constant_banks << SSID_TOGGLE_BIT_COUNT",
            source,
        )
        self.assertIn("unsigned int configuration_count,", source)
        self.assertIn("(toggle_bits >> 3u) & 3u", source)

    def test_extra_constant_banks_are_unique_and_bounded(self) -> None:
        systems = benchmark_toggle_systems(1)
        population = make_toggle_population(systems, 8)
        banks = [population.constants(0, bank) for bank in range(8)]
        self.assertEqual(len(set(banks)), 8)
        incumbent = banks[0]
        for bank in banks[1:]:
            for original, candidate in zip(incumbent, bank):
                self.assertLessEqual(abs(candidate - original), 0.105 * (abs(original) + 1.0) + 1.0e-6)

    def test_module_contains_equal_independent_entry_points(self) -> None:
        module = generate_toggle_cuda_module(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            benchmark_toggle_systems(8),
            2,
            5,
        )
        self.assertEqual(module.systems_per_kernel, 2)
        self.assertEqual(len(module.kernel_names), 4)
        for name in module.kernel_names:
            self.assertEqual(module.source.count(f"void {name}("), 1)


if __name__ == "__main__":
    unittest.main()
