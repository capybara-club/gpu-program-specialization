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

from secant_system_id.fed_batch import (
    DIVERSE_INITIAL_STATES,
    FED_BATCH_MODEL,
    INVALID_MSE,
    PLANTED_BINDINGS,
    PLANTED_CONSTANTS,
    PRODUCT_PAIRED_INITIAL_STATES,
    benchmark_genomes,
    make_population,
    pack_population,
    packed_setting_values,
    planted_programs,
    reference_data,
    score_packed_population,
    score_setting,
    setting_values,
)


class FedBatchTest(unittest.TestCase):
    def test_product_paired_design_varies_only_product_within_pairs(self) -> None:
        self.assertEqual(len(PRODUCT_PAIRED_INITIAL_STATES), 16)
        for first, second in zip(
            PRODUCT_PAIRED_INITIAL_STATES[::2], PRODUCT_PAIRED_INITIAL_STATES[1::2]
        ):
            self.assertEqual(first[:3], second[:3])
            self.assertEqual((first[3], second[3]), (0.0, 4.0))

    def test_diverse_reference_design_matches_the_planted_laws(self) -> None:
        shape = FED_BATCH_MODEL.make_shape(
            constant_count=8,
            trajectory_count=len(DIVERSE_INITIAL_STATES),
            observation_count=12,
        )
        reference = reference_data(4, DIVERSE_INITIAL_STATES, shape.observation_count)
        self.assertEqual(len(reference), shape.reference_float_count)
        self.assertLess(
            score_setting(
                planted_programs(shape),
                PLANTED_CONSTANTS,
                PLANTED_BINDINGS,
                reference,
                4,
                shape,
                1.0,
            ),
            1.0e-12,
        )

    def test_planted_setting_reproduces_reference(self) -> None:
        steps = 8
        population = make_population(4)
        constants, bindings = setting_values(population, 0)
        score = score_setting(planted_programs(), constants, bindings, reference_data(steps), steps)
        self.assertLess(score, 1.0e-12)

    def test_cpu_reference_scores_every_packed_setting(self) -> None:
        steps = 2
        base = make_population(3)
        genomes = benchmark_genomes(2)
        population = pack_population(base, len(genomes))
        reference = reference_data(steps)
        scores = score_packed_population(genomes, population, reference, steps)
        self.assertEqual(len(scores), 6)
        self.assertEqual(scores.typecode, "f")
        for genome_index, genome in enumerate(genomes):
            for setting in range(population.num_settings):
                constants, bindings = packed_setting_values(
                    population, genome_index, setting
                )
                expected = score_setting(
                    genome.programs, constants, bindings, reference, steps
                )
                self.assertAlmostEqual(
                    scores[genome_index * population.num_settings + setting],
                    expected,
                    delta=max(1.0e-7, abs(expected) * 1.0e-6),
                )

    def test_invalid_cpu_trajectory_uses_gpu_sentinel(self) -> None:
        score = score_setting(
            planted_programs(),
            PLANTED_CONSTANTS,
            [3] * 16,
            reference_data(2),
            2,
        )
        self.assertEqual(score, INVALID_MSE)


if __name__ == "__main__":
    unittest.main()
