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

from secant_system_id.c_runtime import GPCandidate
from secant_system_id.fed_batch import PLANTED_BINDINGS, PLANTED_CONSTANTS, planted_programs
from secant_system_id.trajectory_lm import make_lm_population, render_trajectory_lm_source


class TrajectoryLMTest(unittest.TestCase):
    def _candidate(self) -> GPCandidate:
        return GPCandidate(
            3,
            0.25,
            0.2501,
            7,
            23,
            tuple(program.data for program in planted_programs()),
            PLANTED_CONSTANTS,
            PLANTED_BINDINGS,
        )

    def test_first_fit_is_exact_incumbent(self) -> None:
        candidate = self._candidate()
        population = make_lm_population(candidate, 32, 4, 19)
        self.assertEqual(population.num_fits, 128)
        for constant, expected in enumerate(candidate.constants):
            self.assertAlmostEqual(population.starts[constant * 128], expected, places=5)
        for leaf, expected in enumerate(candidate.bindings):
            self.assertEqual(population.bindings[leaf * 32], expected)

    def test_population_is_deterministic_and_mutates(self) -> None:
        first = make_lm_population(self._candidate(), 16, 2, 29, 0.0)
        second = make_lm_population(self._candidate(), 16, 2, 29, 0.0)
        self.assertEqual(first.starts, second.starts)
        self.assertEqual(first.bindings, second.bindings)
        mutated = tuple(first.bindings[leaf * 16 + 1] for leaf in range(16))
        self.assertNotEqual(mutated, PLANTED_BINDINGS)

    def test_rendered_source_embeds_both_sites(self) -> None:
        source = render_trajectory_lm_source()
        self.assertIn("#define SSID_SPECIALIZED_GRADIENT 1", source)
        self.assertIn("ssid_primal_marked_0", source)
        self.assertIn("ssid_partial_marked_0", source)
        self.assertNotIn("fedbatch_primal_site.inc", source)
        self.assertNotIn("fedbatch_partial_site.inc", source)


if __name__ == "__main__":
    unittest.main()
