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

import numpy as np

import secant
from secant.routines import DEFAULT_ROUTINES


class CubinPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(0x5EC47)
        cls.rows = 257
        cls.inputs = cls.rng.uniform(0.5, 3.0, size=(4, cls.rows)).astype(np.float32)
        cls.targets = cls.rng.uniform(-2.0, 2.0, size=(2, cls.rows)).astype(np.float32)
        cls.constants = cls.rng.uniform(0.5, 2.0, size=(2, 7)).astype(np.float32)
        cls.programs = (
            secant.Program(secant.sin(secant.input(0)) * secant.cos(secant.input(1))),
            secant.Program(secant.safe_div(secant.input(2), secant.input(3))),
        )
        cls.dynamic_programs = (
            secant.Program(secant.sin(secant.input(0)) + secant.dynamic_constant(0)),
            secant.Program(secant.safe_div(secant.input(1), secant.dynamic_constant(1))),
        )

    def test_materialize_pipeline_matches_numpy(self) -> None:
        recipe = secant.MaterializeRecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_inputs=4,
            patch_capacity_instructions=64,
        )
        template = secant.compile_materialize(recipe)
        cubin = template.specialize(self.programs, routines=DEFAULT_ROUTINES)
        actual = secant.MaterializeModule(cubin, recipe).run(self.inputs)
        expected = secant.materialize(self.programs, self.inputs, routines=DEFAULT_ROUTINES)
        np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-4)

    def test_sse_pipeline_matches_numpy(self) -> None:
        recipe = secant.SSERecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_inputs=4,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        template = secant.compile_sse(recipe)
        cubin = template.specialize(self.programs, routines=DEFAULT_ROUTINES)
        module = secant.SSEModule(cubin, recipe)
        with module.resident(self.inputs, self.targets, num_streams=2) as execution:
            result = execution.run(iterations=2, warmups=1)
        actual = result.output
        expected = secant.sse(self.programs, self.inputs, self.targets, routines=DEFAULT_ROUTINES)
        self.assertGreater(result.gpu_seconds, 0.0)
        self.assertGreater(result.wall_seconds, result.gpu_seconds)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=4.0e-3)

    def test_sse_bulk_runner_accepts_packed_ast_bytes(self) -> None:
        recipe = secant.SSERecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_inputs=4,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        template = secant.compile_sse(recipe)
        programs = self.programs * 2
        ast_data, ast_offsets = secant.pack_programs(programs)
        with secant.SSEBulkRunner(template, num_workers=2, num_streams=2) as runner:
            result = runner.run_all(
                ast_data,
                ast_offsets,
                self.inputs,
                self.targets,
                routines=DEFAULT_ROUTINES,
            )
        expected = secant.sse(programs, self.inputs, self.targets, routines=DEFAULT_ROUTINES)
        self.assertEqual(result.stats["num_modules"], 2)
        self.assertEqual(result.stats["num_asts"], 4)
        np.testing.assert_allclose(result.output, expected, rtol=5.0e-4, atol=4.0e-3)

    def test_sse_bulk_runner_accepts_partial_module_and_targets(self) -> None:
        recipe = secant.SSERecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_inputs=4,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        template = secant.compile_sse(recipe)
        programs = self.programs + self.programs[:1]
        ast_data, ast_offsets = secant.pack_programs(programs)
        targets = self.targets[:1]
        with secant.SSEBulkRunner(template, num_workers=2, num_streams=2) as runner:
            result = runner.run_all(
                ast_data,
                ast_offsets,
                self.inputs,
                targets,
                routines=DEFAULT_ROUTINES,
            )
        expected = secant.sse(programs, self.inputs, targets, routines=DEFAULT_ROUTINES)
        self.assertEqual(result.stats["num_modules"], 2)
        self.assertEqual(result.stats["num_asts"], 3)
        self.assertEqual(result.output.shape, (3, 1))
        np.testing.assert_allclose(result.output, expected, rtol=5.0e-4, atol=4.0e-3)

    def test_affine_stats_pipeline_matches_numpy(self) -> None:
        recipe = secant.AffineStatsRecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_inputs=4,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        template = secant.compile_affine_stats(recipe)
        cubin = template.specialize(self.programs, routines=DEFAULT_ROUTINES)
        actual = secant.AffineStatsModule(cubin, recipe).run(self.inputs, self.targets)
        expected = secant.affine_stats(self.programs, self.inputs, self.targets, routines=DEFAULT_ROUTINES)
        np.testing.assert_allclose(actual, expected, rtol=6.0e-4, atol=6.0e-3)

    def test_gram_stats_pipeline_matches_numpy(self) -> None:
        recipe = secant.GramStatsRecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_inputs=4,
            num_targets=2,
            tile_rows=64,
            threads_per_block=64,
            patch_capacity_instructions=64,
        )
        template = secant.compile_gram_stats(recipe)
        cubin = template.specialize(self.programs, routines=DEFAULT_ROUTINES)
        actual = secant.GramStatsModule(cubin, recipe).run(self.inputs, self.targets)
        expected = secant.gram_stats(
            self.programs,
            self.inputs,
            self.targets,
            recipe.asts_per_kernel,
            routines=DEFAULT_ROUTINES,
        )
        np.testing.assert_allclose(actual, expected, rtol=6.0e-4, atol=6.0e-3)

    def test_dynamic_constant_sse_pipeline_matches_numpy(self) -> None:
        recipe = secant.DynamicConstantSSERecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_input_columns=4,
            num_input_constants=2,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        template = secant.compile_dynamic_constant_sse(recipe)
        cubin = template.specialize(self.dynamic_programs, routines=DEFAULT_ROUTINES)
        actual = secant.DynamicConstantSSEModule(cubin, recipe).run(self.inputs, self.constants, self.targets)
        expected = secant.dynamic_constant_sse(
            self.dynamic_programs,
            self.inputs,
            self.constants,
            self.targets,
            routines=DEFAULT_ROUTINES,
        )
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=4.0e-3)

    def test_dynamic_leaf_sse_pipeline_matches_numpy(self) -> None:
        recipe = secant.DynamicLeafSSERecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_dynamic_leaves=4,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        programs = (
            secant.Program(secant.dynamic_column(0) + secant.dynamic_constant(1)),
            secant.Program(secant.sin(secant.dynamic_constant_or_column(2)) * secant.dynamic_column(3)),
        )
        masks = np.asarray((0b1101, 0b1001, 0b1101), dtype=np.uint32)
        words = np.empty((3, 4), dtype=np.uint32)
        words[:, 0] = 0
        words[:, 1] = np.asarray((0.5, 0.75, 1.0), dtype=np.float32).view(np.uint32)
        words[:, 2] = np.asarray((2, np.float32(-0.25).view(np.uint32), 1), dtype=np.uint32)
        words[:, 3] = 3

        template = secant.compile_dynamic_leaf_sse(recipe)
        cubin = template.specialize(programs)
        actual = secant.DynamicLeafSSEModule(cubin, recipe).run(self.inputs, masks, words, self.targets)
        expected = secant.dynamic_leaf_sse(programs, self.inputs, masks, words, self.targets)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=4.0e-3)

    def test_mixed_static_dynamic_leaf_sse_pipeline_matches_numpy(self) -> None:
        recipe = secant.DynamicLeafSSERecipe(
            num_kernels=1,
            asts_per_kernel=2,
            num_input_columns=self.inputs.shape[0],
            num_static_input_columns=self.inputs.shape[0],
            num_dynamic_leaves=4,
            num_targets=2,
            tile_rows=128,
            threads_per_block=128,
            patch_capacity_instructions=64,
        )
        programs = (
            secant.Program(secant.input(0) + secant.dynamic_constant(1)),
            secant.Program(secant.sin(secant.input(2)) * secant.dynamic_column(3)),
        )
        masks = np.asarray((0b1101, 0b1001, 0b1101), dtype=np.uint32)
        words = np.empty((3, 4), dtype=np.uint32)
        words[:, 0] = 0
        words[:, 1] = np.asarray((0.5, 0.75, 1.0), dtype=np.float32).view(np.uint32)
        words[:, 2] = np.asarray((2, np.float32(-0.25).view(np.uint32), 1), dtype=np.uint32)
        words[:, 3] = 1

        template = secant.compile_dynamic_leaf_sse(recipe)
        cubin = template.specialize(programs)
        active_inputs = self.inputs[:3]
        actual = secant.DynamicLeafSSEModule(cubin, recipe).run(active_inputs, masks, words, self.targets)
        expected = secant.dynamic_leaf_sse(
            programs,
            active_inputs,
            masks,
            words,
            self.targets,
            num_static_input_columns=active_inputs.shape[0],
        )
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=4.0e-3)


if __name__ == "__main__":
    unittest.main()
