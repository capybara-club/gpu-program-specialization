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

import math
import unittest

from odezza.example import example_bundle, lotka_volterra_group
from odezza.manifest import ManifestError, parse_cuda_manifest, shape_from_manifest
from odezza.model import KernelShape
from odezza.template import generate_cuda


class TemplateTests(unittest.TestCase):
    def test_manifest_round_trip_and_shape(self) -> None:
        shape = KernelShape(trajectory_count=3, observation_count=12, system_capacity=4)
        source = generate_cuda(shape)
        manifest, payload = parse_cuda_manifest(source)
        self.assertEqual(shape_from_manifest(manifest), shape)
        self.assertIn("odezza_scoring", payload)
        self.assertIn("const unsigned int system_index = blockIdx.y;", payload)
        self.assertIn("and.b32 odezza_toggle_masked", payload)
        self.assertIn('"r"(permutation)', payload)
        self.assertIn("const unsigned int permutation = (unsigned int)configuration;", payload)
        self.assertIn("configuration >> active_toggle_count", payload)
        self.assertNotIn("ODEZZA_TOGGLE_PERMUTATION_COUNT", payload)
        self.assertNotIn("TOGGLE_CAPACITY", payload)
        self.assertNotIn("(permutation >> 4u) & 1u", payload)
        self.assertEqual(manifest["configuration"]["toggle_permutations"], "runtime power of two")
        self.assertEqual(manifest["configuration"]["permutation_word_bits"], 32)

    def test_payload_tampering_is_rejected(self) -> None:
        source = generate_cuda()
        with self.assertRaisesRegex(ManifestError, "payload hash"):
            parse_cuda_manifest(source.replace("fused RK4", "modified RK4", 1))

    def test_shape_uses_the_cross_language_identifier_and_float_contract(self) -> None:
        shape = KernelShape(observation_interval=0.123456789123)
        self.assertEqual(shape.observation_interval, 0.123456789)
        with self.assertRaisesRegex(ValueError, "ASCII C identifiers"):
            KernelShape(state_names=("x", "λ"))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            KernelShape(observation_interval=math.nan)

    def test_example_has_complete_reference_layout(self) -> None:
        shape = KernelShape(observation_count=8, trajectory_count=2)
        bundle = example_bundle(shape)
        self.assertEqual(len(bundle["reference_data"]), shape.reference_float_count)
        group = lotka_volterra_group(shape)
        self.assertEqual(group.required_toggle_bits, 0)
        self.assertEqual(bundle["ground_truth"]["configuration"], 0)


if __name__ == "__main__":
    unittest.main()
