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
from dataclasses import replace
from pathlib import Path
import subprocess
import unittest

from secant_system_id.ast import (
    InstructionType,
    Program,
    absolute,
    fma,
    input_slot,
    maximum,
    minimum,
    operation,
)
from secant_system_id.autodiff import PostorderADTape
from secant_system_id.c_runtime import (
    C99Library,
    CTrajectoryLMLaunch,
    FlatGenomeBatch,
    GPConfig,
    SSID_SETTINGS_HASHED_INCUMBENT,
    benchmark_specialization,
    default_c99_library_path,
)
from secant_system_id.cubin import Function
from secant_system_id.fed_batch import FED_BATCH_SHAPE, benchmark_genomes
from secant_system_id.genome import SystemGenome
from secant_system_id.packed_cubin import (
    PackedCubinPlan,
    PackedKernelPlan,
    PackedModulePlan,
    PackedSitePlan,
)
from secant_system_id.packed_sass import specialize_packed_module_cubin
from secant_system_id.shape import PackedDispatch, PackedKernelSpec


def _kernel_plan(index: int, genome_base: int, architecture: int = 89) -> PackedKernelPlan:
    region = 512 + index * 4096
    function = Function(symbol_index=index + 1, file_offset=region, size=3584)
    dispatch_offsets = tuple(region + 64 + 16 * offset for offset in range(4))
    arena_start = region + 128
    target_base = 64 + index * 64
    cubin = PackedCubinPlan(
        cubin_size=8704,
        architecture=architecture,
        input_count=FED_BATCH_SHAPE.input_count,
        output_count=FED_BATCH_SHAPE.ast_count,
        genome_capacity=2,
        register_count=64,
        register_count_offsets=(target_base + 32,),
        register_count_header_offsets=(target_base + 40,),
        function=function,
        site=PackedSitePlan(
            entry_offset=region,
            dispatch_offsets=dispatch_offsets,
            dispatch_instructions=tuple(
                (0x1000 + index * 16 + offset, 0x2000 + index * 16 + offset)
                for offset in range(4)
            ),
            arena_start_offset=arena_start,
            arena_instruction_count=192,
            arena_end_offset=arena_start + 192 * 16,
            continuation_offset=arena_start + 192 * 16 + 32,
            incoming_wait_mask=0x5,
            input_registers=tuple(range(8, 24)),
            output_registers=(24, 25),
            available_registers=tuple(range(26, 48)),
            cleanup_offsets=(region + 16, region + 32),
            target_table_offsets=(target_base, target_base + 4),
            original_target_values=(128, 256),
        ),
    )
    return PackedKernelPlan(
        PackedKernelSpec(
            name=f"ssid_score_packed_{index}",
            genome_base=genome_base,
            dispatch=PackedDispatch(genome_capacity=2, genomes_per_cta=2),
        ),
        cubin,
    )


def _module_plan(architecture: int = 89) -> PackedModulePlan:
    return PackedModulePlan(
        8704,
        architecture,
        (
            _kernel_plan(0, 0, architecture),
            _kernel_plan(1, 2, architecture),
        ),
    )


class C99RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not default_c99_library_path().is_file():
            root = Path(__file__).resolve().parents[1]
            subprocess.run(["make", "c99"], cwd=root, check=True)
        cls.api = C99Library()

    def test_flat_batch_is_genome_and_ast_contiguous(self) -> None:
        genomes = benchmark_genomes(4)
        batch = FlatGenomeBatch.from_genomes(genomes, FED_BATCH_SHAPE)
        self.assertEqual(batch.genome_count, 4)
        self.assertEqual(batch.ast_count, 8)
        self.assertEqual(batch.c_value.genomes[3].first_ast, 6)
        self.assertEqual(batch.c_value.asts[7].site_index, 1)
        self.assertGreater(batch.program_byte_count, 0)

    def test_sm89_hashed_settings_are_available_after_dependency_fix(self) -> None:
        plan = _module_plan(89)
        cubin = bytes((index * 29 + 17) & 0xFF for index in range(plan.cubin_size))
        with self.api.template(
            cubin,
            plan,
            FED_BATCH_SHAPE,
            settings_mode=SSID_SETTINGS_HASHED_INCUMBENT,
        ):
            pass

    def test_sm90_hashed_settings_remain_available(self) -> None:
        plan = _module_plan(90)
        cubin = bytes((index * 31 + 19) & 0xFF for index in range(plan.cubin_size))
        with self.api.template(
            cubin,
            plan,
            FED_BATCH_SHAPE,
            settings_mode=SSID_SETTINGS_HASHED_INCUMBENT,
        ):
            pass

    def test_c99_ad_tape_matches_python_byte_for_byte(self) -> None:
        for program in benchmark_genomes(4)[2].programs:
            expected = PostorderADTape.from_program(
                program, FED_BATCH_SHAPE.input_count
            )
            encoded, stats = self.api.build_ad_tape(
                program.data, FED_BATCH_SHAPE.input_count
            )
            self.assertEqual(encoded, expected.to_bytes())
            self.assertEqual(stats.node_count, expected.node_count)
            self.assertEqual(
                stats.maximum_stack_depth, expected.maximum_stack_depth
            )
            self.assertEqual(
                stats.scratch_floats_per_thread,
                expected.scratch_floats_per_thread,
            )

    def test_c99_specializer_matches_python_byte_for_byte(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 37 + 11) & 0xFF for index in range(plan.cubin_size))
        genomes = benchmark_genomes(4)
        expected = specialize_packed_module_cubin(
            cubin, plan, genomes, FED_BATCH_SHAPE
        ).cubin
        batch = FlatGenomeBatch.from_genomes(genomes, FED_BATCH_SHAPE)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            observed, stats = template.specialize(batch)
        self.assertEqual(observed, expected)
        self.assertEqual(stats.genome_count, 4)
        self.assertEqual(stats.populated_kernel_count, 2)
        self.assertGreater(stats.sass_instruction_count, 0)

    def test_c99_specializer_can_grow_from_an_empty_scratch_pool(self) -> None:
        original = _module_plan()
        plan = replace(
            original,
            kernels=tuple(
                replace(
                    kernel,
                    cubin=replace(
                        kernel.cubin,
                        site=replace(kernel.cubin.site, available_registers=()),
                    ),
                )
                for kernel in original.kernels
            ),
        )
        cubin = bytes((index * 43 + 13) & 0xFF for index in range(plan.cubin_size))
        genomes = benchmark_genomes(4)
        expected = specialize_packed_module_cubin(
            cubin, plan, genomes, FED_BATCH_SHAPE
        ).cubin
        batch = FlatGenomeBatch.from_genomes(genomes, FED_BATCH_SHAPE)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            observed, stats = template.specialize(batch)
        self.assertEqual(observed, expected)
        self.assertGreater(stats.maximum_register_count, 64)

    def test_sm90_c99_specializer_matches_python_byte_for_byte(self) -> None:
        plan = _module_plan(90)
        cubin = bytes((index * 41 + 7) & 0xFF for index in range(plan.cubin_size))
        genomes = benchmark_genomes(4)
        expected = specialize_packed_module_cubin(
            cubin, plan, genomes, FED_BATCH_SHAPE
        ).cubin
        batch = FlatGenomeBatch.from_genomes(genomes, FED_BATCH_SHAPE)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            observed, stats = template.specialize(batch)
        self.assertEqual(observed, expected)
        self.assertEqual(stats.genome_count, 4)

    def test_c99_specializer_matches_every_supported_operation(self) -> None:
        first = [input_slot(index) for index in range(8)]
        second = [input_slot(index) for index in range(8, 16)]
        first_expression = fma(
            maximum(absolute(-first[0]), minimum(first[1], first[2])),
            operation(InstructionType.SQRT_F32, first[3] * first[3] + 1.0),
            first[4] - first[5] / (first[6] + 1.0),
        )
        second_expression = (
            operation(InstructionType.SIN_F32, second[0])
            + operation(InstructionType.COS_F32, second[1])
            + operation(InstructionType.EXP_F32, second[2])
            + operation(InstructionType.LOG_F32, absolute(second[3]) + 1.0)
            + operation(InstructionType.EX2_F32, second[4])
            + operation(InstructionType.LG2_F32, absolute(second[5]) + 1.0)
            + operation(InstructionType.RSQRT_F32, second[6] * second[6] + 1.0)
            + operation(InstructionType.TANH_F32, second[7])
            + operation(InstructionType.RCP_F32, absolute(second[0]) + 1.0)
        )
        genome = SystemGenome(
            (
                Program.from_expression(first_expression),
                Program.from_expression(second_expression),
            )
        )
        genomes = (genome, genome, genome, genome)
        plan = _module_plan()
        cubin = bytes((index * 19 + 5) & 0xFF for index in range(plan.cubin_size))
        expected = specialize_packed_module_cubin(cubin, plan, genomes, FED_BATCH_SHAPE).cubin
        batch = FlatGenomeBatch.from_genomes(genomes, FED_BATCH_SHAPE)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            observed, _stats = template.specialize(batch)
        self.assertEqual(observed, expected)

    def test_specialization_only_pipeline_uses_configurable_workers(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 13 + 3) & 0xFF for index in range(plan.cubin_size))
        batch = FlatGenomeBatch.from_genomes(benchmark_genomes(4), FED_BATCH_SHAPE)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            results = [
                benchmark_specialization(
                    template, batch, workers, submissions=12
                )
                for workers in (1, 2, 4)
            ]
        self.assertEqual([result.workers for result in results], [1, 2, 4])
        self.assertTrue(all(result.submissions == 12 for result in results))
        self.assertTrue(all(result.asts_per_second > 0 for result in results))

    def test_raw_specialization_benchmark_uses_worker_owned_cubins(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 7 + 17) & 0xFF for index in range(plan.cubin_size))
        batch = FlatGenomeBatch.from_genomes(benchmark_genomes(4), FED_BATCH_SHAPE)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            result = template.benchmark_raw(batch, threads=4, modules=100)
        self.assertEqual(result.thread_count, 4)
        self.assertEqual(result.module_count, 100)
        self.assertEqual(result.ast_count, 800)
        self.assertEqual(result.cubin_bytes_copied, 100 * len(cubin))
        self.assertGreater(result.asts_per_second, 0)

    def test_raw_module_queue_copies_prespecialized_lm_cubins(self) -> None:
        cubin = bytes((index * 17 + 9) & 0xFF for index in range(4096))
        launch = CTrajectoryLMLaunch(
            1, 8, 2, 4, 4, 2, 3, 16, 4, 3, 1.0e-3,
            4, 8, 5, 6, 7, 128,
        )
        with self.api.module_pipeline(
            len(cubin), workers=4, queue_capacity=16, enable_cuda=False,
            execution_streams=4,
        ) as pipeline:
            tickets = [pipeline.submit_trajectory_lm(cubin, launch) for _ in range(12)]
            results = [ticket.wait() for ticket in tickets]
            for ticket in tickets:
                ticket.close()
        self.assertEqual(len(results), 12)
        self.assertTrue(all(result.status == 0 for result in results))
        self.assertTrue(all(result.launched_kernel_count == 0 for result in results))

    def test_c99_gp_generates_deterministic_multi_site_batches(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 23 + 29) & 0xFF for index in range(plan.cubin_size))
        config = GPConfig(
            population_size=10,
            settings_per_genome=8,
            max_nodes_per_ast=20,
            max_program_bytes_per_ast=96,
            initial_max_nodes=12,
            elite_count=2,
            seed=41,
        )
        unary = (InstructionType.NEG_F32, InstructionType.ABS_F32)
        binary = (
            InstructionType.ADD_F32,
            InstructionType.SUB_F32,
            InstructionType.MUL_F32,
            InstructionType.DIV_F32,
        )
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            with template.gp(config, FED_BATCH_SHAPE, unary, binary) as first:
                first_batches = [first.batch_programs(index) for index in range(first.batch_count)]
            with template.gp(config, FED_BATCH_SHAPE, unary, binary) as second:
                second_batches = [second.batch_programs(index) for index in range(second.batch_count)]
            for batch in first_batches:
                genomes = tuple(SystemGenome(tuple(Program(program) for program in programs)) for programs in batch)
                template.specialize(FlatGenomeBatch.from_genomes(genomes, FED_BATCH_SHAPE))
        self.assertEqual([len(batch) for batch in first_batches], [4, 4, 2])
        self.assertEqual(first_batches, second_batches)
        for batch in first_batches:
            for programs in batch:
                genome = SystemGenome(tuple(Program(program) for program in programs))
                genome.validate(FED_BATCH_SHAPE)

    def test_c99_gp_inherits_winning_settings_and_advances(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 31 + 13) & 0xFF for index in range(plan.cubin_size))
        config = GPConfig(
            population_size=8,
            settings_per_genome=8,
            max_nodes_per_ast=20,
            max_program_bytes_per_ast=96,
            initial_max_nodes=12,
            elite_count=2,
            seed=53,
        )
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            with template.gp(
                config,
                FED_BATCH_SHAPE,
                (InstructionType.NEG_F32, InstructionType.ABS_F32),
                (InstructionType.ADD_F32, InstructionType.SUB_F32, InstructionType.MUL_F32, InstructionType.DIV_F32),
            ) as gp:
                settings, bindings = gp.settings_generate()
                selected = 3
                expected_constants = [
                    settings[((genome * FED_BATCH_SHAPE.constant_count + constant) * config.settings_per_genome) + selected]
                    for genome in range(config.population_size)
                    for constant in range(FED_BATCH_SHAPE.constant_count)
                ]
                expected_bindings = [
                    bindings[((genome * FED_BATCH_SHAPE.input_count + leaf) * config.settings_per_genome) + selected]
                    for genome in range(config.population_size)
                    for leaf in range(FED_BATCH_SHAPE.input_count)
                ]
                gp.winners_apply(
                    array("f", [float(config.population_size - genome) for genome in range(config.population_size)]),
                    array("I", [selected]) * config.population_size,
                    settings,
                    bindings,
                )
                inherited_settings, inherited_bindings = gp.settings_generate()
                for genome in range(config.population_size):
                    for constant in range(FED_BATCH_SHAPE.constant_count):
                        self.assertEqual(
                            inherited_settings[(genome * FED_BATCH_SHAPE.constant_count + constant) * config.settings_per_genome],
                            expected_constants[genome * FED_BATCH_SHAPE.constant_count + constant],
                        )
                    for leaf in range(FED_BATCH_SHAPE.input_count):
                        self.assertEqual(
                            inherited_bindings[(genome * FED_BATCH_SHAPE.input_count + leaf) * config.settings_per_genome],
                            expected_bindings[genome * FED_BATCH_SHAPE.input_count + leaf],
                        )
                best = gp.best()
                self.assertEqual(best.mse, 1.0)
                self.assertEqual(best.generation, 0)
                self.assertEqual(best.constants, tuple(expected_constants[-FED_BATCH_SHAPE.constant_count :]))
                self.assertEqual(best.bindings, tuple(expected_bindings[-FED_BATCH_SHAPE.input_count :]))
                gp.advance()
                self.assertEqual(gp.generation, 1)
                for batch_index in range(gp.batch_count):
                    for programs in gp.batch_programs(batch_index):
                        SystemGenome(tuple(Program(program) for program in programs)).validate(FED_BATCH_SHAPE)

    def test_c99_gp_external_promotion_only_accepts_improvements(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 47 + 31) & 0xFF for index in range(plan.cubin_size))
        config = GPConfig(population_size=8, settings_per_genome=4, elite_count=2, seed=59)
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            with template.gp(
                config,
                FED_BATCH_SHAPE,
                (InstructionType.NEG_F32,),
                (InstructionType.ADD_F32, InstructionType.SUB_F32, InstructionType.MUL_F32, InstructionType.DIV_F32),
            ) as gp:
                settings, bindings = gp.settings_generate()
                gp.winners_apply(
                    array("f", [10.0 + genome for genome in range(config.population_size)]),
                    array("I", [0]) * config.population_size,
                    settings,
                    bindings,
                )
                before = gp.candidate(3)
                promoted_constants = tuple(value + 0.25 for value in before.constants)
                self.assertFalse(
                    gp.candidate_improve(3, before.mse + 1.0, promoted_constants, before.bindings)
                )
                self.assertEqual(gp.candidate(3), before)
                nonfinite_constants = (float("nan"),) + promoted_constants[1:]
                with self.assertRaisesRegex(RuntimeError, "not finite"):
                    gp.candidate_improve(
                        3, before.mse * 0.5, nonfinite_constants, before.bindings
                    )
                self.assertTrue(
                    gp.candidate_improve(3, before.mse * 0.25, promoted_constants, before.bindings)
                )
                after = gp.candidate(3)
                self.assertAlmostEqual(after.mse, before.mse * 0.25)
                self.assertEqual(after.programs, before.programs)
                for observed, expected in zip(after.constants, promoted_constants):
                    self.assertAlmostEqual(observed, expected)
                self.assertEqual(after.bindings, before.bindings)
                self.assertEqual(gp.best().mse, after.mse)

    def test_c99_gp_variation_stays_valid_across_generations(self) -> None:
        plan = _module_plan()
        cubin = bytes((index * 43 + 3) & 0xFF for index in range(plan.cubin_size))
        config = GPConfig(
            population_size=32,
            settings_per_genome=8,
            max_nodes_per_ast=30,
            max_program_bytes_per_ast=128,
            initial_max_nodes=15,
            elite_count=4,
            seed=67,
        )
        with self.api.template(cubin, plan, FED_BATCH_SHAPE) as template:
            with template.gp(
                config,
                FED_BATCH_SHAPE,
                (InstructionType.NEG_F32, InstructionType.ABS_F32),
                (InstructionType.ADD_F32, InstructionType.SUB_F32, InstructionType.MUL_F32, InstructionType.DIV_F32),
            ) as gp:
                for generation in range(50):
                    settings, bindings = gp.settings_generate()
                    gp.winners_apply(
                        array("f", [float((genome * 17 + generation * 13) % 101) for genome in range(config.population_size)]),
                        array("I", [(genome + generation) % config.settings_per_genome for genome in range(config.population_size)]),
                        settings,
                        bindings,
                    )
                    if generation != 49:
                        gp.advance()
                self.assertEqual(gp.generation, 49)
                for batch_index in range(gp.batch_count):
                    for programs in gp.batch_programs(batch_index):
                        SystemGenome(tuple(Program(program) for program in programs)).validate(FED_BATCH_SHAPE)


if __name__ == "__main__":
    unittest.main()
