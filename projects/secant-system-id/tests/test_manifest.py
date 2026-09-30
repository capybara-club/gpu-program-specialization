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
from secant_system_id.manifest import (
    ManifestError,
    dispatch_from_manifest,
    kernel_specs_from_manifest,
    parse_cuda_manifest,
    shape_from_manifest,
)
from secant_system_id.packed_template import generate_packed_cuda, generate_packed_cuda_module
from secant_system_id.shape import PackedDispatch
from secant_system_id.template import generate_cuda


class ManifestTest(unittest.TestCase):
    def test_direct_source_is_self_describing_and_deterministic(self) -> None:
        first = generate_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE)
        second = generate_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE)
        self.assertEqual(first, second)
        manifest, payload = parse_cuda_manifest(first)
        self.assertEqual(manifest["kernel"]["topology"], "direct")
        self.assertEqual(manifest["model"]["name"], "astaxanthin_fed_batch")
        self.assertEqual(manifest["shape"]["trajectory_layout"], "dense_aligned")
        self.assertIn("ssid_template_id[8]", payload)
        self.assertEqual(manifest["template_id_algorithm"], "sha256")
        self.assertEqual(len(manifest["template_id"]), 64)
        self.assertEqual(shape_from_manifest(manifest), FED_BATCH_SHAPE)
        self.assertIsNone(dispatch_from_manifest(manifest))

    def test_packed_source_records_dispatch(self) -> None:
        dispatch = PackedDispatch(8, 4)
        source = generate_packed_cuda(
            FED_BATCH_MODEL, FED_BATCH_SHAPE, dispatch
        )
        manifest, _payload = parse_cuda_manifest(source)
        self.assertEqual(manifest["kernel"]["topology"], "packed")
        self.assertEqual(dispatch_from_manifest(manifest), dispatch)
        self.assertEqual(manifest["shape"]["output_count"], 2)

    def test_single_kernel_warp_source_records_ownership(self) -> None:
        source = generate_packed_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            PackedDispatch(8, 8),
            warp_per_system=True,
        )
        manifest, _payload = parse_cuda_manifest(source)
        self.assertEqual(manifest["kernel"]["work_ownership"], "warp_per_system")

    def test_payload_edit_is_detected(self) -> None:
        source = generate_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE)
        modified = source.replace("-2.58f", "-2.59f", 1)
        self.assertNotEqual(source, modified)
        with self.assertRaisesRegex(ManifestError, "payload hash"):
            parse_cuda_manifest(modified)

    def test_multi_kernel_module_records_every_entry_point(self) -> None:
        dispatch = PackedDispatch(8, 4)
        source = generate_packed_cuda_module(
            FED_BATCH_MODEL, FED_BATCH_SHAPE, 3, dispatch
        )
        manifest, _payload = parse_cuda_manifest(source)
        specs = kernel_specs_from_manifest(manifest)
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["module"]["kernel_count"], 3)
        self.assertEqual(manifest["module"]["genome_capacity"], 24)
        self.assertEqual([spec.name for spec in specs], [
            "ssid_score_packed_0",
            "ssid_score_packed_1",
            "ssid_score_packed_2",
        ])
        self.assertEqual([spec.genome_base for spec in specs], [0, 8, 16])
        self.assertTrue(all(spec.dispatch == dispatch for spec in specs))


if __name__ == "__main__":
    unittest.main()
