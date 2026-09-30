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

from array import array
import unittest

from secant_system_id.bindings import constant_leaf, state_leaf
from secant_system_id.fed_batch import FED_BATCH_SHAPE
from secant_system_id.fed_batch import PLANTED_BINDINGS, PLANTED_CONSTANTS
from secant_system_id.toggle_ast import (
    ToggleInstructionType,
    ToggleProgram,
    ToggleSystem,
    source_input,
    toggle1,
    toggle2,
)
from secant_system_id.toggle_fed_batch import (
    benchmark_diverse_toggle_systems,
    benchmark_toggle_systems,
    make_toggle_population,
    materialize_toggle_population,
)


class ToggleAstTest(unittest.TestCase):
    def test_one_and_two_bit_leaves_encode_source_kinds(self) -> None:
        expression = (
            toggle1(0, state_leaf(1), constant_leaf(0))
            + toggle2(
                1,
                state_leaf(0),
                state_leaf(2),
                constant_leaf(1),
                constant_leaf(2),
            )
        )
        program = ToggleProgram.from_expression(expression, FED_BATCH_SHAPE)
        instructions = tuple(program.instructions())
        self.assertEqual(instructions[0].kind, ToggleInstructionType.TOGGLE_1BIT_INPUT_F32)
        self.assertEqual(instructions[0].operands, (0, 1, 4))
        self.assertEqual(instructions[1].kind, ToggleInstructionType.TOGGLE_2BIT_INPUT_F32)
        self.assertEqual(instructions[1].operands, (1, 0, 2, 5, 6))
        self.assertEqual(program.required_toggle_bits, 3)

    def test_permutation_resolves_direct_register_sources(self) -> None:
        expression = toggle1(0, state_leaf(1), constant_leaf(0)) * toggle2(
            1,
            state_leaf(0),
            state_leaf(2),
            constant_leaf(1),
            constant_leaf(2),
        )
        program = ToggleProgram.from_expression(expression, FED_BATCH_SHAPE)
        state = (2.0, 3.0, 5.0, 7.0)
        constants = (11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0)
        self.assertEqual(program.evaluate(state, constants, 0, FED_BATCH_SHAPE), 6.0)
        self.assertEqual(program.evaluate(state, constants, 1, FED_BATCH_SHAPE), 22.0)
        self.assertEqual(program.evaluate(state, constants, 4, FED_BATCH_SHAPE), 39.0)
        self.assertEqual(program.evaluate(state, constants, 7, FED_BATCH_SHAPE), 187.0)

    def test_materialized_program_and_bindings_are_equivalent(self) -> None:
        first_expression = toggle1(0, state_leaf(1), constant_leaf(0)) + source_input(state_leaf(0))
        second_expression = toggle2(
            1,
            state_leaf(2),
            constant_leaf(1),
            state_leaf(3),
            constant_leaf(2),
        ) * source_input(constant_leaf(3))
        system = ToggleSystem(
            (
                ToggleProgram.from_expression(first_expression, FED_BATCH_SHAPE),
                ToggleProgram.from_expression(second_expression, FED_BATCH_SHAPE),
            )
        )
        shape = FED_BATCH_SHAPE.__class__(
            ast_leaf_counts=(2, 2),
            state_count=FED_BATCH_SHAPE.state_count,
            constant_count=FED_BATCH_SHAPE.constant_count,
            trajectory_count=FED_BATCH_SHAPE.trajectory_count,
            observation_count=FED_BATCH_SHAPE.observation_count,
        )
        # Programs are source-shape independent until validation, so rebuild them
        # against the four-leaf test shape.
        system = ToggleSystem(
            (
                ToggleProgram.from_expression(first_expression, shape),
                ToggleProgram.from_expression(second_expression, shape),
            )
        )
        genome = system.materialized_genome(shape)
        state = (2.0, 3.0, 5.0, 7.0)
        constants = (11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0)
        for permutation in range(8):
            bindings = system.resolved_bindings(permutation, shape)
            bank = state + constants
            leaves = tuple(bank[index] for index in bindings)
            observed = tuple(program.evaluate(leaves) for program in genome.programs)
            expected = tuple(
                program.evaluate(state, constants, permutation, shape)
                for program in system.programs
            )
            self.assertEqual(observed, expected)

    def test_cuda_render_uses_register_names_without_shared_bank_indices(self) -> None:
        program = ToggleProgram.from_expression(
            toggle1(0, state_leaf(1), constant_leaf(0))
            + toggle2(1, state_leaf(0), constant_leaf(1), state_leaf(2), constant_leaf(2)),
            FED_BATCH_SHAPE,
        )
        rendered = program.render_cuda(FED_BATCH_SHAPE)
        self.assertIn("stage_state[1]", rendered)
        self.assertIn("constant0", rendered)
        self.assertIn("(toggle_bits >> 0u)", rendered)
        self.assertNotIn("bank[", rendered)

    def test_fedbatch_zero_permutation_is_the_planted_configuration(self) -> None:
        systems = benchmark_toggle_systems(2)
        population = make_toggle_population(systems, 4)
        materialized = materialize_toggle_population(systems, population)
        self.assertEqual(population.toggle_bit_count, 5)
        self.assertEqual(population.configurations_per_system, 128)
        for observed, expected in zip(population.constants(0, 0), PLANTED_CONSTANTS):
            self.assertAlmostEqual(observed, expected, places=5)
        self.assertEqual(systems[0].resolved_bindings(0, FED_BATCH_SHAPE), PLANTED_BINDINGS)
        self.assertEqual(
            materialized.bindings[0 :: materialized.num_settings][: FED_BATCH_SHAPE.input_count],
            array("I", PLANTED_BINDINGS),
        )

    def test_diverse_family_has_unique_structures_and_fixed_cost(self) -> None:
        systems = benchmark_diverse_toggle_systems(32)
        programs = [program for system in systems for program in system.programs]
        signatures = {
            tuple(int(instruction.kind) for instruction in program.instructions())
            for program in programs
        }
        self.assertEqual(len(signatures), 64)
        self.assertTrue(all(program.required_toggle_bits == 5 for program in programs))
        self.assertTrue(all(program.leaf_count == 8 for program in programs))
        self.assertTrue(all(len(tuple(program.instructions())) == 18 for program in programs))

    def test_diverse_toggle_programs_match_materialized_leaf_programs(self) -> None:
        state = (2.0, 3.0, 5.0, 7.0)
        constants = (11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0)
        bank = state + constants
        for system in benchmark_diverse_toggle_systems(8):
            genome = system.materialized_genome(FED_BATCH_SHAPE)
            for permutation in range(32):
                bindings = system.resolved_bindings(permutation, FED_BATCH_SHAPE)
                leaves = tuple(bank[index] for index in bindings)
                observed = tuple(program.evaluate(leaves) for program in genome.programs)
                expected = tuple(
                    program.evaluate(state, constants, permutation, FED_BATCH_SHAPE)
                    for program in system.programs
                )
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
