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
    FED_BATCH_MODEL,
    FED_BATCH_SHAPE,
    benchmark_genomes,
    make_population,
    pack_population,
)
from secant_system_id.packed_template import generate_packed_cuda, generate_packed_cuda_module
from secant_system_id.models import THREE_SITE_MODEL
from secant_system_id.shape import PackedDispatch


class PackedKernelTest(unittest.TestCase):
    def test_warp_per_system_uses_one_setting_per_lane(self) -> None:
        source = generate_packed_cuda_module(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            2,
            PackedDispatch(8, 8),
            winner_output=True,
            warp_per_system=True,
        )
        self.assertIn("one system per warp, one setting per lane", source)
        self.assertIn("const unsigned int local_warp = threadIdx.x >> 5u;", source)
        self.assertIn("blockIdx.y * SSID_GENOMES_PER_CTA + local_warp", source)
        self.assertIn("num_settings > 32u", source)
        self.assertIn("__shfl_down_sync", source)
        self.assertNotIn("score_scratch[threadIdx.x] = setting_score", source)
        self.assertIn('"work_ownership": "warp_per_system"', source)

    def test_relative_state_loss_is_explicit_and_optional(self) -> None:
        raw = generate_packed_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE)
        relative = generate_packed_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            relative_error_floor=1.0,
        )
        self.assertNotIn("const float target0_value", raw)
        self.assertIn("relative state-space MSE with denominator floor 1.0", relative)
        self.assertIn("fmaxf(fabsf(target0_value), 1.0f)", relative)

    def test_exact_zero_guard_is_recovery_opt_in(self) -> None:
        ordinary = generate_packed_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE)
        guarded = generate_packed_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            reject_exact_zero_score=True,
        )
        self.assertIn(
            "mse_out[output_index] = valid && isfinite(squared_error) ?",
            ordinary,
        )
        self.assertIn(
            "mse_out[output_index] = valid && isfinite(squared_error) && squared_error > 0.0f ?",
            guarded,
        )

    def test_winner_reducer_cannot_emit_a_sentinel_setting(self) -> None:
        source = generate_packed_cuda(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            winner_output=True,
        )
        self.assertIn("candidate_setting < num_settings && isfinite(candidate_score)", source)
        self.assertIn(
            "setting_scratch[0] < num_settings ? setting_scratch[0] : 0u",
            source,
        )

    def test_one_dispatch_site_selects_complete_genomes(self) -> None:
        dispatch = PackedDispatch(genome_capacity=8, genomes_per_cta=8)
        source = generate_packed_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE, dispatch)
        self.assertEqual(source.count("brx.idx.uni"), 1)
        self.assertEqual(source.count("ssid_genome_targets: .branchtargets"), 1)
        self.assertEqual(source.count("ssid_genome_0:"), 1)
        self.assertEqual(source.count("ssid_genome_7:"), 1)
        self.assertIn("const unsigned int first_genome = 0u + blockIdx.y", source)
        self.assertIn("local_genome < SSID_GENOMES_PER_CTA", source)
        self.assertIn("for (int experiment = 0;", source)
        self.assertIn("(unsigned long long)genome * num_settings + setting", source)
        self.assertIn(
            "for (unsigned int local_genome = 0; local_genome < SSID_GENOMES_PER_CTA; ++local_genome)",
            source,
        )

    def test_module_has_independent_kernel_sites_and_global_genome_ranges(self) -> None:
        dispatch = PackedDispatch(genome_capacity=8, genomes_per_cta=4)
        source = generate_packed_cuda_module(
            FED_BATCH_MODEL, FED_BATCH_SHAPE, 2, dispatch
        )
        self.assertEqual(source.count("extern \"C\" __global__ void ssid_score_packed_"), 2)
        self.assertEqual(source.count("brx.idx.uni"), 2)
        self.assertIn("void ssid_score_packed_0(", source)
        self.assertIn("void ssid_score_packed_1(", source)
        self.assertIn("ssid_k0_genome_targets: .branchtargets", source)
        self.assertIn("ssid_k1_genome_targets: .branchtargets", source)
        self.assertIn("const unsigned int first_genome = 0u + blockIdx.y", source)
        self.assertIn("const unsigned int first_genome = 8u + blockIdx.y", source)
        self.assertIn("const unsigned int dispatch_index = genome - 8u;", source)
        self.assertEqual(source.count("#define SSID_STATE_COUNT"), 1)
        self.assertNotIn("for (unsigned int local_genome = 0;\n", source)
        self.assertIn(
            "const unsigned int binding0_raw = leaf_bindings[((unsigned long long)genome * SSID_INPUT_COUNT + 0ull) * bindings_leading_dimension + setting];",
            source,
        )

    def test_population_is_genome_major(self) -> None:
        population = make_population(4)
        packed = pack_population(population, 3)
        self.assertEqual(packed.num_genomes, 3)
        self.assertEqual(packed.num_settings, 4)
        self.assertEqual(
            packed.settings.tolist(), population.settings.tolist() * 3
        )
        self.assertEqual(
            packed.bindings.tolist(), population.bindings.tolist() * 3
        )

    def test_benchmark_genomes_are_multi_output_and_distinct(self) -> None:
        genomes = benchmark_genomes(8)
        self.assertTrue(all(len(genome.programs) == 2 for genome in genomes))
        self.assertEqual(len({genome.programs[0].data for genome in genomes}), 8)

    def test_benchmark_genome_variants_preserve_shape_but_change_bytes(self) -> None:
        first = benchmark_genomes(8, variant=1)
        second = benchmark_genomes(8, variant=2)
        self.assertEqual(
            [len(program.data) for genome in first for program in genome.programs],
            [len(program.data) for genome in second for program in genome.programs],
        )
        self.assertNotEqual(
            [program.data for genome in first for program in genome.programs],
            [program.data for genome in second for program in genome.programs],
        )

    def test_packed_source_supports_three_missing_sites(self) -> None:
        shape = THREE_SITE_MODEL.make_shape(
            constant_count=4, trajectory_count=2, observation_count=5
        )
        source = generate_packed_cuda(
            THREE_SITE_MODEL, shape, PackedDispatch(8, 8)
        )
        self.assertIn("float missing_output0;", source)
        self.assertIn("float missing_output1;", source)
        self.assertIn("float missing_output2;", source)
        self.assertIn("const float derivative2", source)

    def test_winner_source_reduces_ctas_and_genomes_on_device(self) -> None:
        source = generate_packed_cuda_module(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            1,
            PackedDispatch(8, 1),
            winner_output=True,
        )
        self.assertIn("float *__restrict__ cta_best_score_out", source)
        self.assertIn("unsigned int *__restrict__ cta_best_setting_out", source)
        self.assertIn("void ssid_reduce_winners(", source)
        self.assertIn("best_score_out[genome]", source)
        self.assertNotIn("float *__restrict__ mse_out", source)

    def test_hashed_settings_source_keeps_only_per_genome_incumbents(self) -> None:
        source = generate_packed_cuda_module(
            FED_BATCH_MODEL,
            FED_BATCH_SHAPE,
            1,
            PackedDispatch(8, 1),
            winner_output=True,
            hashed_settings=True,
            constant_mutation_scale=0.5,
            binding_keep_probability=0.5,
        )
        self.assertIn("ssid_gp_mix64", source)
        self.assertIn("random_generation = bindings_leading_dimension >> 32u", source)
        self.assertIn("settings[(unsigned long long)genome * SSID_CONSTANT_COUNT", source)
        self.assertNotIn("* settings_leading_dimension + setting]", source)


if __name__ == "__main__":
    unittest.main()
