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

import json
import math
import unittest

from secant_system_id.ast import Program, absolute, input_slot
from secant_system_id.autodiff import (
    DerivativeTapeError,
    PostorderADTape,
    PostorderDerivativeBundle,
    SystemADPlan,
    SystemDerivativeBundle,
)
from secant_system_id.cubin import CubinPlan, Function, SitePlan
from secant_system_id.derivative_sass import (
    _SiteCSEAssembler,
    _compile_program_group,
)
from secant_system_id.fed_batch import FED_BATCH_SHAPE, PLANTED_BINDINGS, PLANTED_CONSTANTS, planted_programs
from secant_system_id.genome import SystemGenome
from secant_system_id.gpu_gradient_check import KERNEL_NAME as GRADIENT_CHECK_KERNEL, render_gradient_check_source
from secant_system_id.lm_specialization import (
    PARTIAL_MARKER_BITS,
    PRIMAL_MARKER_BITS,
    partial_output_count,
    primal_output_count,
    render_partial_site,
    render_primal_site,
)


class AutodiffTapeTest(unittest.TestCase):
    def test_postorder_tape_round_trips_and_is_inspectable(self) -> None:
        expression = (input_slot(0) + 2.0 * input_slot(1)) / input_slot(2)
        tape = PostorderADTape.from_program(Program.from_expression(expression), 3)
        decoded = PostorderADTape.from_bytes(tape.to_bytes())
        self.assertEqual(decoded, tape)
        inspection = tape.inspection()
        self.assertEqual(inspection["schema"], "secant-system-id.postorder-ad-tape")
        self.assertEqual(inspection["sha256"], tape.sha256)
        self.assertEqual(len(inspection["postorder"]), tape.node_count)
        self.assertEqual(len(inspection["reverse"]), tape.node_count)
        json.dumps(inspection)

    def test_reverse_partials_match_finite_differences(self) -> None:
        expression = (
            input_slot(0) * input_slot(1)
            + input_slot(2) / input_slot(3)
            - input_slot(0)
        )
        tape = PostorderADTape.from_program(Program.from_expression(expression), 4)
        inputs = [1.7, -0.4, 2.3, 1.2]
        value, partials = tape.evaluate(inputs)
        self.assertAlmostEqual(value, Program.from_expression(expression).evaluate(inputs), places=12)
        epsilon = 1.0e-5
        for index in range(4):
            positive = inputs.copy()
            negative = inputs.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            observed = (
                Program.from_expression(expression).evaluate(positive)
                - Program.from_expression(expression).evaluate(negative)
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(partials[index], observed, places=6)

    def test_repeated_leaf_occurrences_accumulate(self) -> None:
        expression = input_slot(0) * input_slot(0) + input_slot(0) * input_slot(1)
        tape = PostorderADTape.from_program(Program.from_expression(expression), 2)
        value, partials = tape.evaluate((3.0, 4.0))
        self.assertEqual(value, 21.0)
        self.assertEqual(partials, (10.0, 3.0))

    def test_nonsmooth_operations_are_rejected_for_lm(self) -> None:
        program = Program.from_expression(absolute(input_slot(0)))
        with self.assertRaisesRegex(DerivativeTapeError, "ABS_F32"):
            PostorderADTape.from_program(program, 1)

    def test_planted_rate_partials_follow_dynamic_leaf_slots(self) -> None:
        first, _second = planted_programs()
        tape = PostorderADTape.from_program(first, FED_BATCH_SHAPE.input_count)
        state = (0.1, 10.0, 5.0, 0.0)
        bank = state + PLANTED_CONSTANTS
        leaves = [0.0] * FED_BATCH_SHAPE.input_count
        for leaf in range(8):
            leaves[leaf] = bank[PLANTED_BINDINGS[leaf]]
        value, partials = tape.evaluate(leaves)
        expected = first.evaluate(leaves)
        self.assertAlmostEqual(value, expected)
        epsilon = 1.0e-4
        for leaf in range(8):
            positive = leaves.copy()
            negative = leaves.copy()
            positive[leaf] += epsilon
            negative[leaf] -= epsilon
            finite_difference = (first.evaluate(positive) - first.evaluate(negative)) / (2.0 * epsilon)
            self.assertTrue(math.isfinite(partials[leaf]))
            self.assertAlmostEqual(partials[leaf], finite_difference, delta=2.0e-5)
        self.assertEqual(partials[8:], (0.0,) * 8)

    def test_system_workspace_reuses_the_largest_site(self) -> None:
        genome = SystemGenome(planted_programs())
        plan = SystemADPlan.from_genome(genome, FED_BATCH_SHAPE)
        self.assertEqual(plan.scratch_node_capacity, max(tape.node_count for tape in plan.tapes))
        expected = (
            4 * FED_BATCH_SHAPE.reference_float_count
            + 4 * FED_BATCH_SHAPE.bank_slot_count * 256
            + 8 * plan.scratch_node_capacity * 256
        )
        self.assertEqual(plan.shared_bytes_per_cta(FED_BATCH_SHAPE, 256), expected)
        self.assertLessEqual(plan.shared_bytes_per_cta(FED_BATCH_SHAPE, 256), 99 * 1024)

    def test_scalar_sass_lowering_remains_postorder_and_matches_tape(self) -> None:
        first, _second = planted_programs()
        bundle = PostorderDerivativeBundle.from_program(
            first,
            FED_BATCH_SHAPE.input_count,
            range(8),
        )
        state = (0.1, 10.0, 5.0, 0.0)
        bank = state + PLANTED_CONSTANTS
        leaves = [0.0] * FED_BATCH_SHAPE.input_count
        for leaf in range(8):
            leaves[leaf] = bank[PLANTED_BINDINGS[leaf]]
        value, partials = bundle.tape.evaluate(leaves)
        self.assertAlmostEqual(bundle.primal.evaluate(leaves), value)
        for leaf, program in enumerate(bundle.partials):
            self.assertAlmostEqual(program.evaluate(leaves), partials[leaf], places=7)
        inspection = bundle.inspection()
        self.assertEqual(
            inspection["schema"],
            "secant-system-id.postorder-derivative-bundle",
        )
        self.assertEqual(len(inspection["programs"]), 9)
        self.assertLess(max(bundle.instruction_counts), 64)

    def test_system_derivative_output_layout_is_site_major(self) -> None:
        bundle = SystemDerivativeBundle.from_genome(
            SystemGenome(planted_programs()),
            FED_BATCH_SHAPE,
        )
        self.assertEqual(bundle.output_count, 18)
        self.assertEqual(len(bundle.programs), 18)
        self.assertEqual(bundle.instruction_counts[0][0], 16)
        self.assertEqual(bundle.instruction_counts[1][0], 16)
        inspection = bundle.inspection()
        self.assertEqual(
            inspection["output_layout"],
            "site-major: primal, then one partial per site leaf",
        )

    def test_lm_markers_split_primal_from_statistics_partials(self) -> None:
        primal = render_primal_site(FED_BATCH_SHAPE, patch_capacity=32)
        partial = render_partial_site(FED_BATCH_SHAPE, patch_capacity=48)
        self.assertEqual(primal_output_count(FED_BATCH_SHAPE), 2)
        self.assertEqual(partial_output_count(FED_BATCH_SHAPE), 16)
        self.assertEqual(primal.count("brkpt;"), 35)
        self.assertEqual(partial.count("brkpt;"), 51)
        self.assertIn(f"0f{PRIMAL_MARKER_BITS:08x}", primal)
        self.assertIn(f"0f{PARTIAL_MARKER_BITS:08x}", partial)
        self.assertNotIn("leaf_partials", primal)
        self.assertIn("float leaf_partials_1[8]", partial)
        self.assertIn("float leaf_partials_2[8]", partial)
        self.assertIn("postorder AD tape v1", primal)
        self.assertIn("postorder AD tape v1", partial)

    def test_gpu_gradient_check_uses_the_production_split_marker_abi(self) -> None:
        source = render_gradient_check_source(FED_BATCH_SHAPE)
        self.assertIn(f"void {GRADIENT_CHECK_KERNEL}(", source)
        self.assertIn(f"0f{PRIMAL_MARKER_BITS:08x}", source)
        self.assertIn(f"0f{PARTIAL_MARKER_BITS:08x}", source)
        self.assertIn("specific_growth_1", source)
        self.assertIn("specific_growth_2", source)
        self.assertIn("leaf_partials_1[8]", source)
        self.assertIn("leaf_partials_2[8]", source)

    def test_cse_relocates_a_live_value_before_pinning_its_output(self) -> None:
        plan = CubinPlan(
            4096,
            89,
            2,
            2,
            64,
            30,
            (),
            (),
            Function(0, 0, 1024),
            SitePlan(0, 0, 64, 0, (10, 11), (20, 21), ()),
        )
        shared = input_slot(0) * input_slot(1)
        programs = (
            Program.from_expression(shared + input_slot(0)),
            Program.from_expression(shared + input_slot(1)),
        )
        assembler = _SiteCSEAssembler(plan, plan.site.output_registers, 0)
        counts, _high_water = assembler.compile(programs)
        self.assertEqual(len(counts), 2)
        self.assertIn(20, assembler.pinned)
        self.assertIn(21, assembler.pinned)

    def test_register_pressure_fallback_compiles_completed_ada_winner(self) -> None:
        programs = (
            Program(
                bytes.fromhex(
                    "b606b605b606b60287b600858788b606b601b603b6078888"
                    "b6068888b605858883"
                )
            ),
            Program(bytes.fromhex("b60ab60cb60a87b60eb60f87858883")),
        )
        bundle = SystemDerivativeBundle.from_genome(
            SystemGenome(programs),
            FED_BATCH_SHAPE,
        )
        self.assertEqual([site.tape.node_count for site in bundle.sites], [21, 9])
        plan = CubinPlan(
            65536,
            89,
            16,
            16,
            1024,
            252,
            (),
            (),
            Function(0, 0, 32768),
            SitePlan(
                0,
                0,
                1024,
                0,
                tuple(range(170, 202, 2)),
                (219, 210, 204, 168, 162, 158, 160, 156),
                (212, 218, 217),
            ),
        )
        assembler, counts, high_water, used_fallback = _compile_program_group(
            plan,
            plan.site.output_registers,
            0,
            bundle.sites[0].partials,
        )
        self.assertTrue(used_fallback)
        self.assertEqual(len(counts), 8)
        self.assertEqual(len(assembler.pinned), 8)
        self.assertLess(sum(counts), plan.patch_capacity)
        self.assertLessEqual(high_water, 253)

    def test_low_pressure_group_keeps_cross_output_cse(self) -> None:
        plan = CubinPlan(
            4096,
            89,
            2,
            2,
            64,
            30,
            (),
            (),
            Function(0, 0, 1024),
            SitePlan(0, 0, 64, 0, (10, 11), (20, 21), (22, 23)),
        )
        programs = (
            Program.from_expression(input_slot(0) + input_slot(1)),
            Program.from_expression(input_slot(0) * input_slot(1)),
        )
        _assembler, counts, _high_water, used_fallback = _compile_program_group(
            plan,
            plan.site.output_registers,
            0,
            programs,
        )
        self.assertFalse(used_fallback)
        self.assertEqual(len(counts), 2)


if __name__ == "__main__":
    unittest.main()
