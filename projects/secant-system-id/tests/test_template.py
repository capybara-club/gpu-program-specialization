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

from secant_system_id.shape import KernelShape
from secant_system_id.models import ONE_SITE_MODEL, THREE_SITE_MODEL
from secant_system_id.template import FIRST_MARKER_BITS, generate_cuda, generate_fedbatch_cuda


class TemplateTest(unittest.TestCase):
    def test_marker_and_shared_bank_shape(self) -> None:
        shape = KernelShape()
        source = generate_fedbatch_cuda(shape)
        self.assertEqual(source.count("brkpt;"), shape.patch_capacity + 3)
        self.assertIn(f"0f{FIRST_MARKER_BITS:08x}", source)
        self.assertIn("volatile float *bank", source)
        self.assertIn("missing_output0", source)
        self.assertIn("missing_output1", source)
        self.assertIn(
            "for (unsigned int index = threadIdx.x; index < SSID_REFERENCE_FLOAT_COUNT; index += blockDim.x)",
            source,
        )
        self.assertNotIn("for (unsigned int index = threadIdx.x;\n", source)
        self.assertIn(
            "const unsigned int binding0_raw = leaf_bindings[0ull * bindings_leading_dimension + setting];",
            source,
        )
        self.assertEqual(shape.winner_shared_bytes(256), shape.shared_bytes(256) + 2048)

    def test_missing_site_count_is_model_driven(self) -> None:
        one_shape = ONE_SITE_MODEL.make_shape(
            constant_count=3, trajectory_count=2, observation_count=5
        )
        one_source = generate_cuda(ONE_SITE_MODEL, one_shape)
        self.assertIn("#define SSID_STATE_COUNT 2", one_source)
        self.assertIn("missing_output0", one_source)
        self.assertNotIn("missing_output1", one_source)
        self.assertIn("const float derivative1 = (missing_output0 - (0.1f * stage_state[1]));", one_source)

        three_shape = THREE_SITE_MODEL.make_shape(
            constant_count=4, trajectory_count=2, observation_count=5
        )
        three_source = generate_cuda(THREE_SITE_MODEL, three_shape)
        self.assertIn("missing_output2", three_source)
        self.assertEqual(three_shape.ast_leaf_counts, (2, 3, 4))
        self.assertEqual(three_shape.input_count, 9)


if __name__ == "__main__":
    unittest.main()
