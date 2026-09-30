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

from secant_system_id.ast import Program, input_slot
from secant_system_id.c_runtime import GPCandidate
from secant_system_id.lm_selection import (
    SELECTION_NOVELTY_RANDOM,
    SELECTION_RANDOM,
    lm_run_diagnostics,
    lm_selection_summary,
    select_lm_candidates,
    structure_sha256,
)


class LMSelectionTest(unittest.TestCase):
    def _program(self, tag: int, offset: int) -> bytes:
        expression = input_slot(offset)
        for step in range(tag):
            leaf = input_slot(offset + 1 + step % 3)
            expression = expression + leaf if step % 2 == 0 else expression * leaf
        return Program.from_expression(expression).data

    def _candidate(self, index: int, tag: int, objective: float | None = None) -> GPCandidate:
        return GPCandidate(
            index,
            0.1 + index * 0.01,
            0.1 + index * 0.01 if objective is None else objective,
            25,
            8,
            (self._program(tag, 0), self._program(tag + 1, 8)),
            (float(index),) * 8,
            (index,) * 16,
        )

    def test_structure_hash_excludes_constants_and_bindings(self) -> None:
        first = self._candidate(0, 1)
        second = self._candidate(9, 1)
        changed = self._candidate(0, 2)
        self.assertEqual(structure_sha256(first.programs), structure_sha256(second.programs))
        self.assertNotEqual(structure_sha256(first.programs), structure_sha256(changed.programs))
        renamed = (
            Program.from_expression(input_slot(4) + input_slot(7)).data,
            Program.from_expression(input_slot(12) + input_slot(15)).data,
        )
        canonical = (
            Program.from_expression(input_slot(0) + input_slot(1)).data,
            Program.from_expression(input_slot(8) + input_slot(9)).data,
        )
        self.assertEqual(structure_sha256(renamed), structure_sha256(canonical))

    def test_novelty_random_selection_is_deterministic_and_tracks_overlap(self) -> None:
        candidates = tuple(self._candidate(index, index + 1) for index in range(6))
        selection = select_lm_candidates(
            candidates, set(), SELECTION_NOVELTY_RANDOM, 4, 2, 2, 6, 101, 24
        )
        repeated = select_lm_candidates(
            candidates, set(), SELECTION_NOVELTY_RANDOM, 4, 2, 2, 6, 101, 24
        )
        self.assertEqual(selection, repeated)
        record = selection.boundary_record()
        self.assertEqual(record["novelty_selected"], 2)
        self.assertEqual(record["random_selected"], 2)
        self.assertEqual(
            record["union_selected"],
            record["novelty_only"] + record["random_only"] + record["overlap"],
        )
        self.assertEqual(record["selection_pool_structures"], record["union_selected"] + record["neither"])

    def test_novelty_means_first_appearance_in_the_promising_pool(self) -> None:
        candidates = tuple(self._candidate(index, index + 1) for index in range(6))
        first = select_lm_candidates(
            candidates, set(), SELECTION_NOVELTY_RANDOM, 4, 2, 2, 4, 113, 24
        )
        seen = set(first.observed_structure_ids)
        repeated = select_lm_candidates(
            candidates, seen, SELECTION_NOVELTY_RANDOM, 4, 2, 2, 4, 113, 49
        )
        self.assertEqual(repeated.novel_pool_structures, 0)
        self.assertFalse(any(item.selected_by_novelty for item in repeated.selected))
        new_best = self._candidate(20, 9, 0.001)
        changed = select_lm_candidates(
            (new_best,) + candidates, seen, SELECTION_NOVELTY_RANDOM, 4, 2, 2, 4, 113, 74
        )
        novelty = [item for item in changed.selected if item.selected_by_novelty]
        self.assertEqual([item.candidate.index for item in novelty], [20])

    def test_random_mode_uses_a_deterministic_probability_gate(self) -> None:
        candidates = tuple(self._candidate(index, index + 1) for index in range(8))
        disabled = select_lm_candidates(
            candidates, set(), SELECTION_RANDOM, 4, 2, 2, 8, 127, 24, 0.0
        )
        self.assertFalse(disabled.random_triggered)
        self.assertEqual(disabled.selected, ())
        enabled = select_lm_candidates(
            candidates, set(), SELECTION_RANDOM, 4, 2, 2, 8, 127, 24, 1.0
        )
        repeated = select_lm_candidates(
            candidates, set(), SELECTION_RANDOM, 4, 2, 2, 8, 127, 24, 1.0
        )
        self.assertTrue(enabled.random_triggered)
        self.assertEqual(enabled, repeated)
        self.assertEqual(len(enabled.selected), 4)
        self.assertTrue(all(item.selected_by_random for item in enabled.selected))

    def test_summary_reports_venn_and_arm_outcomes(self) -> None:
        boundaries = (
            {
                "novelty_selected": 2,
                "random_selected": 2,
                "union_selected": 3,
                "eligible_union": 3,
                "novelty_only": 1,
                "random_only": 1,
                "overlap": 1,
                "neither": 5,
            },
        )
        events = (
            {
                "before_mse": 1.0,
                "after_mse": 0.5,
                "accepted": True,
                "selected_by_novelty": True,
                "selected_by_random": False,
                "selected_by_novelty_only": True,
                "selected_by_random_only": False,
                "selected_by_both": False,
            },
            {
                "before_mse": 1.0,
                "after_mse": 0.95,
                "accepted": True,
                "selected_by_novelty": True,
                "selected_by_random": True,
                "selected_by_novelty_only": False,
                "selected_by_random_only": False,
                "selected_by_both": True,
            },
            {
                "before_mse": 1.0,
                "after_mse": 1.0,
                "accepted": False,
                "selected_by_novelty": False,
                "selected_by_random": True,
                "selected_by_novelty_only": False,
                "selected_by_random_only": True,
                "selected_by_both": False,
            },
        )
        summary = lm_selection_summary(boundaries, events)
        self.assertEqual(summary["venn"], {"novelty_only": 1, "random_only": 1, "overlap": 1, "neither": 5})
        self.assertEqual(summary["outcomes"]["novelty"]["evaluated"], 2)
        self.assertEqual(summary["outcomes"]["novelty"]["improved_over_1_percent"], 2)
        self.assertEqual(summary["outcomes"]["random"]["evaluated"], 2)
        self.assertEqual(summary["outcomes"]["random"]["improved_over_1_percent"], 1)

    def test_diagnostics_make_complete_specialization_failure_explicit(self) -> None:
        boundaries = (
            {
                "generation": 100,
                "union_selected": 972,
                "eligible_union": 0,
                "skipped_count": 972,
                "skipped": [
                    {
                        "candidate_index": 7,
                        "candidate_complexity": 30,
                        "ast_node_counts": [21, 9],
                        "structure_sha256": "deadbeef",
                        "reason": "partial LM specialization ran out of registers",
                    }
                ],
            },
        )
        diagnostics = lm_run_diagnostics(boundaries, ())
        self.assertEqual(diagnostics["status"], "failed")
        self.assertEqual(diagnostics["specialization"]["attempted"], 972)
        self.assertEqual(diagnostics["specialization"]["rejected"], 972)
        self.assertEqual(
            diagnostics["specialization"]["rejection_categories"],
            {"register_pressure": 1},
        )
        self.assertEqual(diagnostics["selection"]["boundaries_with_all_selected_rejected"], 1)
        self.assertIn(
            "lm_all_specializations_rejected",
            {flag["code"] for flag in diagnostics["health_flags"]},
        )

    def test_diagnostics_do_not_multiply_shared_batch_time_per_candidate(self) -> None:
        boundaries = (
            {
                "generation": 25,
                "random_triggered": True,
                "union_selected": 2,
                "eligible_union": 2,
                "skipped_count": 0,
                "skipped": [],
                "gp_population_snapshot": {
                    "candidates": 100,
                    "valid_mse_candidates": 90,
                    "invalid_mse_candidates": 10,
                    "median_complexity": 18,
                    "maximum_complexity": 30,
                },
                "lm_batch": {
                    "candidate_count": 2,
                    "fits": 200,
                    "wall_seconds": 0.5,
                    "upload_seconds": 0.01,
                    "download_seconds": 0.02,
                    "sum_ticket_seconds": 0.8,
                    "sum_queue_wait_seconds": 0.1,
                    "sum_image_copy_seconds": 0.03,
                    "sum_module_load_seconds": 0.2,
                },
            },
        )
        base_event = {
            "before_mse": 1.0,
            "after_mse": 0.5,
            "accepted": True,
            "candidate_complexity": 30,
            "ast_node_counts": [21, 9],
            "fits": 100,
            "finite_fits": 99,
            "invalid_mse_fits": 1,
            "total_lm_iterations": 300,
            "total_accepted_steps": 120,
            "fits_with_accepted_steps": 80,
            "fits_hitting_iteration_limit": 5,
            "register_count": 252,
            "primal_sass_instructions": [10, 20],
            "partial_sass_instructions": [30, 40],
            "primal_pressure_fallback_groups": [],
            "partial_pressure_fallback_groups": [0],
            "promotion_batch_wall_seconds": 0.5,
        }
        events = (base_event, {**base_event, "accepted": False, "after_mse": 1.0})
        diagnostics = lm_run_diagnostics(boundaries, events)
        self.assertEqual(diagnostics["timing"]["wall_seconds"], 0.5)
        self.assertEqual(diagnostics["timing"]["fits"], 200)
        self.assertEqual(diagnostics["optimization"]["accepted_candidates"], 1)
        self.assertEqual(diagnostics["optimization"]["invalid_mse_fits"], 2)
        self.assertEqual(diagnostics["gp_population"]["invalid_mse_observation_rate"], 0.1)
        self.assertEqual(diagnostics["gp_population"]["maximum_observed_complexity"], 30.0)
        self.assertEqual(
            diagnostics["pressure_fallback"]["evaluated_candidates_using_fallback"], 2
        )
        self.assertEqual(diagnostics["kernel"]["evaluated_candidate_complexity"]["maximum"], 30.0)
        self.assertEqual(diagnostics["kernel"]["evaluated_ast_nodes_by_site"][0]["median"], 21.0)
        self.assertIn(
            "lm_invalid_fit_scores",
            {flag["code"] for flag in diagnostics["health_flags"]},
        )


if __name__ == "__main__":
    unittest.main()
