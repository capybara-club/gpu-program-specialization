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

from dataclasses import replace
import random
import struct
import unittest

from odezza import (
    LMKernelShape,
    LMSystem,
    absolute,
    cosine,
    exponential,
    fma,
    logarithm,
    optimized_constant,
    sine,
    source,
    square_root,
    state_source,
    toggle2,
    toggle4,
)
from odezza.elf import (
    INSTRUCTION_BYTES,
    Function,
)
from odezza.ast import InstructionType, operation
from odezza.lm_ast import LMAstError, LMSystemDerivatives, lm_program_shape
from odezza.lm_model import LM_GROUPED_LAYOUT, LM_TWO_SITE_LAYOUT
from odezza.lm_elf import LMCubinPlan, LMScalarSitePlan
from odezza.lm_manifest import parse_lm_cuda_manifest
from odezza.lm_sass import specialize_lm_cubin
from odezza.lm_template import generate_lm_cuda
from odezza.sass import (
    OPCODE_BRA,
    OPCODE_FSEL,
    OPCODE_LOP3_IMM,
    OPCODE_MOV,
    SassAssemblyError,
    _branch,
    _mov,
)


def example_system(shape: LMKernelShape) -> LMSystem:
    x = source(state_source(0))
    y = source(state_source(1))
    k0 = optimized_constant(0)
    k1 = optimized_constant(1)
    k2 = optimized_constant(2)
    return LMSystem.from_expressions(
        (
            k0 * toggle2(0, state_source(0), state_source(1)) - k1 * x * y,
            k1 * x * y - k2 * y,
        ),
        shape,
    )


def synthetic_plan(shape: LMKernelShape) -> tuple[bytes, LMCubinPlan]:
    scaffold_instruction_count = max(shape.site_input_counts) + 1
    site_stride = (shape.site_patch_capacity + scaffold_instruction_count + 2) * 16
    cubin = bytes(shape.site_count * site_stride + 16)
    sites = []
    register_base = max(shape.site_input_counts) + 4
    permutation_register = register_base - 1
    for site_index in range(shape.site_count):
        scaffold_start = site_index * site_stride
        input_count = shape.site_input_counts[site_index]
        materialization_offsets = tuple(
            scaffold_start + index * 16
            for index in range(input_count)
        )
        entry = scaffold_start + input_count * 16
        patch_start = entry + 16
        output_count = len(shape.site_output_groups[site_index])
        output_registers = tuple(range(register_base, register_base + output_count))
        input_source_registers = tuple(range(64 - input_count, 64))
        input_wait_masks = tuple(1 << (index % 6) for index in range(input_count))
        sites.append(
            LMScalarSitePlan(
                site_index=site_index,
                entry_offset=entry,
                scaffold_start_offset=scaffold_start,
                patch_start_offset=patch_start,
                patch_instruction_count=shape.site_patch_capacity,
                continuation_offset=patch_start + (shape.site_patch_capacity + 1) * 16,
                incoming_wait_mask=0,
                input_materialization_offsets=materialization_offsets,
                input_source_registers=input_source_registers,
                input_wait_masks=input_wait_masks,
                input_registers=tuple(range(input_count)),
                output_registers=output_registers,
                kernel_output_registers=output_registers,
                available_registers=tuple(range(register_base + output_count, 64)),
                predicate_register=0,
                permutation_register=permutation_register,
                toggle_test_instruction=(
                    (1 << 32) | (permutation_register << 24) | (255 << 16) | OPCODE_LOP3_IMM,
                    0x000FC0000780C0FF,
                ),
                cleanup_offsets=tuple(range(scaffold_start, patch_start, 16)),
            )
        )
    return cubin, LMCubinPlan(
        cubin_size=len(cubin),
        architecture=120,
        register_count=64,
        register_count_offsets=(),
        register_count_header_offsets=(),
        function=Function(1, 0, len(cubin)),
        shape=shape,
        sites=tuple(sites),
    )


class LMAstTests(unittest.TestCase):
    def test_zero_denominator_is_not_host_constant_folded(self) -> None:
        from odezza.ast import constant, NodeKind
        from odezza.lm_ast import _divide
        for numerator in (0.0, 1.0, -1.0):
            for denominator in (0.0, -0.0):
                expression = _divide(constant(numerator), constant(denominator))
                self.assertEqual(expression.kind, NodeKind.OPERATION)
                self.assertEqual(expression.value, InstructionType.DIV_F32)
        shape = LMKernelShape(state_count=1, optimized_constant_count=3)
        x = source(state_source(0))
        system = LMSystem.from_expressions((optimized_constant(0) + x / constant(0.0),), shape)
        derivatives = LMSystemDerivatives.from_system(system, shape)
        self.assertTrue(any(i.kind == InstructionType.DIV_F32
                            for i in derivatives.partials[0][0].instructions()))

    def setUp(self) -> None:
        self.shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )

    def test_repeated_optimized_index_has_one_shared_derivative(self) -> None:
        x = source(state_source(0))
        y = source(state_source(1))
        shared = optimized_constant(0)
        system = LMSystem.from_expressions(
            (shared * x + shared * y, optimized_constant(1) * y),
            self.shape,
        )
        bundle = LMSystemDerivatives.from_system(system, self.shape)
        base = lm_program_shape(self.shape)
        derivative = bundle.partials[0][self.shape.state_count]
        observed = derivative.evaluate(
            (2.0, 5.0),
            (0.0,),
            0,
            base,
            (3.0, 7.0, 11.0),
        )
        self.assertEqual(observed, 7.0)

    def test_toggle_is_preserved_in_primal_and_local_partials(self) -> None:
        system = example_system(self.shape)
        bundle = LMSystemDerivatives.from_system(system, self.shape)
        self.assertEqual(system.required_toggle_bits, 1)
        self.assertTrue(
            any(program.required_toggle_bits == 1 for program in bundle.site_programs(self.shape))
        )

    def test_four_way_toggle_can_select_an_optimized_constant(self) -> None:
        x = source(state_source(0))
        y = source(state_source(1))
        selected = toggle4(0, 1, x, y, optimized_constant(0), 1.0)
        system = LMSystem.from_expressions(
            (selected, optimized_constant(1) * y),
            self.shape,
        )
        bundle = LMSystemDerivatives.from_system(system, self.shape)
        base = lm_program_shape(self.shape)
        parameter_partial = bundle.partials[0][self.shape.state_count]
        self.assertEqual(system.required_toggle_bits, 2)
        self.assertEqual(
            parameter_partial.evaluate(
                (2.0, 5.0),
                (0.0,),
                2,
                base,
                (3.0, 7.0, 11.0),
            ),
            1.0,
        )

    def test_nonsmooth_operator_requires_an_explicit_policy(self) -> None:
        x = source(state_source(0))
        system = LMSystem.from_expressions(
            (absolute(x), optimized_constant(0) * source(state_source(1))),
            self.shape,
        )
        with self.assertRaisesRegex(LMAstError, "nonsmooth"):
            LMSystemDerivatives.from_system(system, self.shape)

    def test_smooth_local_partials_match_finite_differences(self) -> None:
        x = source(state_source(0))
        y = source(state_source(1))
        k0 = optimized_constant(0)
        k1 = optimized_constant(1)
        k2 = optimized_constant(2)
        positive = x * x + 1.5
        denominator = k2 * k2 + 1.2
        system = LMSystem.from_expressions(
            (
                k0 * toggle2(0, x, y) + k0 * y + k1 * x * y
                + sine(k2 * x) + logarithm(positive) + square_root(y * y + 1.0),
                fma(k1, x, y) / denominator + cosine(x) + exponential(0.1 * k0),
            ),
            self.shape,
        )
        bundle = LMSystemDerivatives.from_system(system, self.shape)
        base = lm_program_shape(self.shape)
        state = [0.7, -0.4]
        parameters = [0.3, -0.2, 0.6]
        epsilon = 1.0e-4
        for permutation in (0, 1):
            for rhs_index, primal in enumerate(bundle.primal):
                for target, partial in enumerate(bundle.partials[rhs_index]):
                    plus_state = list(state)
                    minus_state = list(state)
                    plus_parameters = list(parameters)
                    minus_parameters = list(parameters)
                    if target < self.shape.state_count:
                        plus_state[target] += epsilon
                        minus_state[target] -= epsilon
                    else:
                        parameter = target - self.shape.state_count
                        plus_parameters[parameter] += epsilon
                        minus_parameters[parameter] -= epsilon
                    plus = primal.evaluate(
                        plus_state,
                        (0.0,),
                        permutation,
                        base,
                        plus_parameters,
                    )
                    minus = primal.evaluate(
                        minus_state,
                        (0.0,),
                        permutation,
                        base,
                        minus_parameters,
                    )
                    finite_difference = (plus - minus) / (2.0 * epsilon)
                    analytic = partial.evaluate(
                        state,
                        (0.0,),
                        permutation,
                        base,
                        parameters,
                    )
                    self.assertAlmostEqual(analytic, finite_difference, delta=2.0e-4)

    def test_random_smooth_asts_match_finite_differences(self) -> None:
        rng = random.Random(0x0DEA)
        leaves = (
            source(state_source(0)),
            source(state_source(1)),
            optimized_constant(0),
            optimized_constant(1),
            optimized_constant(2),
        )

        def expression(depth):
            if depth == 0 or rng.random() < 0.25:
                return rng.choice(leaves)
            kind = rng.randrange(9)
            if kind == 0:
                return expression(depth - 1) + expression(depth - 1)
            if kind == 1:
                return expression(depth - 1) - expression(depth - 1)
            if kind == 2:
                return expression(depth - 1) * expression(depth - 1)
            if kind == 3:
                denominator = expression(depth - 1)
                return expression(depth - 1) / (1.0 + denominator * denominator)
            if kind == 4:
                return sine(expression(depth - 1))
            if kind == 5:
                return cosine(expression(depth - 1))
            if kind == 6:
                value = expression(depth - 1)
                return logarithm(1.0 + value * value)
            if kind == 7:
                value = expression(depth - 1)
                return square_root(1.0 + value * value)
            return operation(InstructionType.TANH_F32, expression(depth - 1))

        base = lm_program_shape(self.shape)
        epsilon = 1.0e-4
        for _ in range(128):
            system = LMSystem.from_expressions(
                (expression(3), expression(3)),
                self.shape,
            )
            bundle = LMSystemDerivatives.from_system(system, self.shape)
            state = [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)]
            parameters = [rng.uniform(-0.8, 0.8) for _ in range(3)]
            for rhs_index, primal in enumerate(bundle.primal):
                for target, partial in enumerate(bundle.partials[rhs_index]):
                    plus_state = list(state)
                    minus_state = list(state)
                    plus_parameters = list(parameters)
                    minus_parameters = list(parameters)
                    if target < self.shape.state_count:
                        plus_state[target] += epsilon
                        minus_state[target] -= epsilon
                    else:
                        parameter = target - self.shape.state_count
                        plus_parameters[parameter] += epsilon
                        minus_parameters[parameter] -= epsilon
                    plus = primal.evaluate(
                        plus_state,
                        (0.0,),
                        0,
                        base,
                        plus_parameters,
                    )
                    minus = primal.evaluate(
                        minus_state,
                        (0.0,),
                        0,
                        base,
                        minus_parameters,
                    )
                    finite_difference = (plus - minus) / (2.0 * epsilon)
                    analytic = partial.evaluate(
                        state,
                        (0.0,),
                        0,
                        base,
                        parameters,
                    )
                    self.assertAlmostEqual(analytic, finite_difference, delta=5.0e-4)

    def test_coupled_rk4_sensitivities_match_finite_differences(self) -> None:
        shape = LMKernelShape(
            state_count=2,
            optimized_constant_count=1,
            site_patch_capacity=96,
        )
        x = source(state_source(0))
        y = source(state_source(1))
        parameter = optimized_constant(0)
        system = LMSystem.from_expressions(
            (
                parameter * x + 2.0 * y,
                3.0 * x - y,
            ),
            shape,
        )
        derivatives = LMSystemDerivatives.from_system(system, shape)
        program_shape = lm_program_shape(shape)

        def evaluate(state, parameter_value):
            parameters = (parameter_value,)
            primal = tuple(
                program.evaluate(state, (0.0,), 0, program_shape, parameters)
                for program in derivatives.primal
            )
            partials = tuple(
                tuple(
                    program.evaluate(state, (0.0,), 0, program_shape, parameters)
                    for program in programs
                )
                for programs in derivatives.partials
            )
            return primal, partials

        def state_step(state, parameter_value, step_size):
            base = tuple(state)
            stage = tuple(state)
            total = [0.0, 0.0]
            for rk_stage in range(4):
                primal, _ = evaluate(stage, parameter_value)
                weight = 1.0 if rk_stage in (0, 3) else 2.0
                for output in range(2):
                    total[output] += weight * primal[output]
                if rk_stage != 3:
                    stage_h = step_size if rk_stage == 2 else 0.5 * step_size
                    stage = tuple(
                        base[output] + stage_h * primal[output]
                        for output in range(2)
                    )
            return tuple(
                base[output] + step_size * total[output] / 6.0
                for output in range(2)
            )

        def sensitivity_step(state, parameter_value, step_size):
            base_state = tuple(state)
            stage_state = tuple(state)
            total_state = [0.0, 0.0]
            base_sensitivity = (0.0, 0.0)
            stage_sensitivity = base_sensitivity
            total_sensitivity = [0.0, 0.0]
            for rk_stage in range(4):
                primal, partials = evaluate(stage_state, parameter_value)
                weight = 1.0 if rk_stage in (0, 3) else 2.0
                stage_derivative = tuple(
                    partials[output][2]
                    + partials[output][0] * stage_sensitivity[0]
                    + partials[output][1] * stage_sensitivity[1]
                    for output in range(2)
                )
                for output in range(2):
                    total_state[output] += weight * primal[output]
                    total_sensitivity[output] += weight * stage_derivative[output]
                if rk_stage != 3:
                    stage_h = step_size if rk_stage == 2 else 0.5 * step_size
                    stage_state = tuple(
                        base_state[output] + stage_h * primal[output]
                        for output in range(2)
                    )
                    stage_sensitivity = tuple(
                        base_sensitivity[output] + stage_h * stage_derivative[output]
                        for output in range(2)
                    )
            return tuple(
                base_sensitivity[output] + step_size * total_sensitivity[output] / 6.0
                for output in range(2)
            )

        state = (1.0, 0.5)
        parameter_value = 0.7
        step_size = 0.1
        epsilon = 1.0e-5
        analytic = sensitivity_step(state, parameter_value, step_size)
        plus = state_step(state, parameter_value + epsilon, step_size)
        minus = state_step(state, parameter_value - epsilon, step_size)
        finite_difference = tuple(
            (plus[index] - minus[index]) / (2.0 * epsilon)
            for index in range(2)
        )
        for observed, expected in zip(analytic, finite_difference):
            self.assertAlmostEqual(observed, expected, delta=1.0e-8)


class LMTemplateTests(unittest.TestCase):
    def test_cooperative_shape_partitions_derivatives_and_optimizer_state(self) -> None:
        shape = LMKernelShape(
            state_count=8,
            optimized_constant_count=8,
            site_patch_capacity=64,
            site_layout=LM_TWO_SITE_LAYOUT,
            fit_thread_count=4,
        )
        self.assertEqual(shape.primal_site_count, 1)
        self.assertEqual(shape.site_count, 17)
        self.assertEqual(tuple(map(len, shape.site_output_groups)), (8,) + (12, 4) * 8)
        self.assertEqual(shape.local_parameter_count, 2)
        self.assertEqual(shape.local_optimizer_count, 11)
        self.assertEqual(shape.fits_per_warp, 8)

    def test_sixteen_state_shape_fits_the_operand_budget(self) -> None:
        shape = LMKernelShape(
            state_count=16,
            optimized_constant_count=8,
            site_patch_capacity=8,
            site_layout=LM_TWO_SITE_LAYOUT,
            fit_thread_count=8,
        )
        self.assertEqual(shape.maximum_site_output_count, 4)
        self.assertEqual(shape.primal_site_count, 4)
        self.assertEqual(shape.site_count, 100)
        self.assertEqual(shape.local_parameter_count, 1)
        self.assertEqual(shape.local_optimizer_count, 6)
        self.assertEqual(shape.inline_asm_operand_count, 30)

    def test_cooperative_source_replicates_sites_and_shuffles_optimizer_state(self) -> None:
        shape = LMKernelShape(
            state_count=8,
            optimized_constant_count=8,
            site_patch_capacity=64,
            site_layout=LM_TWO_SITE_LAYOUT,
            fit_thread_count=8,
        )
        _, payload = parse_lm_cuda_manifest(generate_lm_cuda(shape))
        self.assertNotIn("lm_partial_workspace", payload)
        self.assertNotIn("lm_partial_address", payload)
        self.assertIn("float factor[ODEZZA_LM_PARAMETER_COUNT * ODEZZA_LM_PARAMETER_COUNT]", payload)
        self.assertIn("local_optimizer[", payload)
        self.assertIn("local_optimizer[0]", payload)
        self.assertNotIn("__shfl_sync(lm_fit_mask, lm_partial_", payload)
        self.assertNotIn("if (lm_fit_lane == 0u) {\n                        asm volatile", payload)

    def test_cooperative_width_must_be_supported_and_use_the_streamed_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            LMKernelShape(fit_thread_count=3)
        with self.assertRaisesRegex(ValueError, "requires two_site_partials"):
            LMKernelShape(fit_thread_count=4)

    def test_sensitivity_stage_updates_are_not_in_place(self) -> None:
        for layout in (LM_GROUPED_LAYOUT, LM_TWO_SITE_LAYOUT):
            shape = LMKernelShape(
                state_count=2,
                optimized_constant_count=3,
                site_patch_capacity=128,
                site_layout=layout,
            )
            source_text = generate_lm_cuda(shape)
            _, payload = parse_lm_cuda_manifest(source_text)
            self.assertNotIn("base_sensitivity", payload)
            if layout == LM_GROUPED_LAYOUT:
                self.assertIn("stage_derivative[output] = sensitivity_derivative;", payload)
                self.assertIn("stage_derivative[output],", payload)
            else:
                self.assertIn("stage_derivative_0[local_parameter] = fmaf(", payload)
                self.assertIn("stage_derivative_0[local_parameter],", payload)
            self.assertIn("sensitivities[sensitivity_index]", payload)

    def test_two_site_layout_streams_local_partials_through_register_sites(self) -> None:
        shape = LMKernelShape(
            state_count=2,
            optimized_constant_count=3,
            site_patch_capacity=128,
            site_layout=LM_TWO_SITE_LAYOUT,
            fit_thread_count=4,
        )
        source_text = generate_lm_cuda(shape)
        manifest, payload = parse_lm_cuda_manifest(source_text)
        self.assertEqual(shape.site_count, 3)
        self.assertEqual(shape.site_input_counts, (5, 5, 5))
        self.assertEqual(manifest["generator_abi_version"], 4)
        self.assertEqual(manifest["kernel"]["sensitivity_integrator"], "simultaneous-stage-rk4-v2")
        self.assertEqual(manifest["abi"]["layout"], "partitioned-primal-and-replicated-partials-v8")
        self.assertNotIn("float local_partials[", payload)
        self.assertNotIn("lm_partial_workspace", payload)
        self.assertNotIn("lm_partial_address", payload)
        self.assertNotIn("__shfl_sync(lm_fit_mask, lm_partial_", payload)

    def test_two_site_layout_specializes_primal_and_all_partials(self) -> None:
        shape = LMKernelShape(
            state_count=2,
            optimized_constant_count=3,
            site_patch_capacity=128,
            site_layout=LM_TWO_SITE_LAYOUT,
        )
        cubin, plan = synthetic_plan(shape)
        result = specialize_lm_cubin(cubin, plan, example_system(shape))
        self.assertEqual(len(result.site_instruction_counts), 3)
        self.assertEqual(len(result.scalar_instruction_counts[0]), shape.state_count)
        self.assertEqual(
            sum(len(counts) for counts in result.scalar_instruction_counts[1:]),
            shape.local_partial_count,
        )

    def test_group_capacity_partitions_primals_and_partials_separately(self) -> None:
        automatic = LMKernelShape(
            state_count=4,
            optimized_constant_count=6,
            site_patch_capacity=192,
        )
        limited = replace(automatic, site_output_capacity=4)
        self.assertEqual(automatic.maximum_site_output_count, 8)
        self.assertEqual(tuple(map(len, automatic.site_output_groups)), (4, 8, 8, 8, 8, 8))
        self.assertEqual(tuple(map(len, limited.site_output_groups)), (4,) * 11)

    def test_manifest_and_source_describe_starts_toggles_and_ragged_data(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=64,
        )
        source_text = generate_lm_cuda(shape)
        manifest, payload = parse_lm_cuda_manifest(source_text)
        self.assertEqual(manifest["shape"]["site_count"], shape.site_count)
        self.assertIn("start_index = fit >> active_toggle_count", payload)
        self.assertIn("trajectory_offsets", payload)
        self.assertIn("reference_weights", payload)
        self.assertIn("max_damping_attempts", payload)
        self.assertIn("factorization_attempts_out", payload)
        self.assertIn("++factorization_attempts", payload)
        primal_keepalive = shape.input_count + len(shape.site_output_groups[0])
        self.assertIn(
            f"add.rn.ftz.f32 %{primal_keepalive}, %{primal_keepalive}, %0;",
            payload,
        )
        self.assertIn("local_partials", payload)
        barrier = payload.index("__syncthreads();")
        partial_cta_exit = payload.index("if (fit >= fit_count) return;")
        self.assertLess(barrier, partial_cta_exit)

    def test_synthetic_specialization_emits_toggle_selects_and_all_sites(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )
        cubin, plan = synthetic_plan(shape)
        result = specialize_lm_cubin(cubin, plan, example_system(shape))
        self.assertEqual(len(result.site_instruction_counts), shape.site_count)
        self.assertEqual(result.required_toggle_bits, 1)
        opcodes = []
        for site, count in zip(plan.sites, result.site_instruction_counts):
            opcodes.extend(
                struct.unpack_from(
                    "<Q",
                    result.cubin,
                    site.patch_start_offset + index * 16,
                )[0]
                & 0xFFFF
                for index in range(count)
            )
        self.assertIn(OPCODE_FSEL, opcodes)

    def test_grouped_specialization_reuses_an_identical_rhs(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )
        x = source(state_source(0))
        shared_rhs = optimized_constant(0) * x + sine(x)
        system = LMSystem.from_expressions((shared_rhs, shared_rhs), shape)
        cubin, plan = synthetic_plan(shape)
        result = specialize_lm_cubin(cubin, plan, system)
        self.assertEqual(result.scalar_instruction_counts[0][1], 1)

    def test_synthetic_specialization_branches_over_scaffold(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )
        cubin, plan = synthetic_plan(shape)
        result = specialize_lm_cubin(cubin, plan, example_system(shape))
        for site in plan.sites:
            branch = struct.unpack_from("<QQ", result.cubin, site.entry_offset)
            distance = (site.patch_start_offset - site.entry_offset) // 16
            self.assertEqual(branch, _branch(distance, plan.architecture))
            self.assertEqual(branch[0] & 0xFFFF, OPCODE_BRA)

    def test_synthetic_specialization_materializes_physical_inputs(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )
        cubin, plan = synthetic_plan(shape)
        result = specialize_lm_cubin(cubin, plan, example_system(shape))
        for site in plan.sites:
            for offset, source_register, input_register, wait_mask in zip(
                site.input_materialization_offsets,
                site.input_source_registers,
                site.input_registers,
                site.input_wait_masks,
            ):
                self.assertEqual(
                    struct.unpack_from("<QQ", result.cubin, offset),
                    _mov(input_register, source_register, wait_mask),
                )

    def test_synthetic_specialization_rejects_incomplete_input_materialization(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )
        cubin, plan = synthetic_plan(shape)
        first = replace(
            plan.sites[0],
            input_wait_masks=plan.sites[0].input_wait_masks[:-1],
        )
        plan = replace(plan, sites=(first, *plan.sites[1:]))
        with self.assertRaisesRegex(SassAssemblyError, "materialization plan is incomplete"):
            specialize_lm_cubin(cubin, plan, example_system(shape))

    def test_synthetic_specialization_can_expand_register_count(self) -> None:
        shape = LMKernelShape(
            optimized_constant_count=3,
            site_patch_capacity=96,
        )
        cubin, plan = synthetic_plan(shape)
        plan = replace(plan, register_count=20)
        result = specialize_lm_cubin(cubin, plan, example_system(shape))
        self.assertGreater(result.register_count, plan.register_count)

    def test_synthetic_specialization_relays_an_overlapping_kernel_output(self) -> None:
        shape = LMKernelShape(optimized_constant_count=3, site_patch_capacity=96)
        cubin, plan = synthetic_plan(shape)
        first = replace(
            plan.sites[0],
            input_registers=(0, *plan.sites[0].input_registers[1:]),
            output_registers=plan.sites[0].output_registers,
            kernel_output_registers=(0, *plan.sites[0].kernel_output_registers[1:]),
        )
        plan = replace(plan, sites=(first, *plan.sites[1:]))
        result = specialize_lm_cubin(cubin, plan, example_system(shape))
        relay_offset = first.patch_start_offset + (result.site_instruction_counts[0] - 2) * 16
        relay_word = struct.unpack_from("<Q", result.cubin, relay_offset)[0]
        self.assertEqual(relay_word & 0xFFFF, OPCODE_MOV)
        self.assertEqual((relay_word >> 16) & 0xFF, 0)
        self.assertEqual((relay_word >> 32) & 0xFF, first.output_registers[0])

if __name__ == "__main__":
    unittest.main()
