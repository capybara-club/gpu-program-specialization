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

import math
import random
import unittest

from secant_system_id.ast import (
    Expression,
    InstructionType,
    Program,
    constant,
    input_slot,
    operation,
)
from secant_system_id.autodiff import PostorderADTape, PostorderDerivativeBundle, SystemDerivativeBundle
from secant_system_id.genome import SystemGenome
from secant_system_id.shape import KernelShape


_SMOOTH_OPERATIONS = (
    InstructionType.ADD_F32,
    InstructionType.SUB_F32,
    InstructionType.MUL_F32,
    InstructionType.DIV_F32,
    InstructionType.NEG_F32,
    InstructionType.SQRT_F32,
    InstructionType.RCP_F32,
    InstructionType.FMA_F32,
    InstructionType.SIN_F32,
    InstructionType.COS_F32,
    InstructionType.EX2_F32,
    InstructionType.LG2_F32,
    InstructionType.RSQRT_F32,
    InstructionType.TANH_F32,
    InstructionType.EXP_F32,
    InstructionType.LOG_F32,
)


def _bounded(value: Expression) -> Expression:
    return operation(InstructionType.TANH_F32, value)


def _positive(value: Expression) -> Expression:
    bounded = _bounded(value)
    return bounded * bounded + 0.75


def _random_leaf(random_source: random.Random, input_indices: tuple[int, ...]) -> Expression:
    if random_source.random() < 0.82:
        return input_slot(random_source.choice(input_indices))
    return constant(random_source.choice((-1.0, -0.5, -0.125, 0.125, 0.5, 1.0)))


def _random_subexpression(
    random_source: random.Random,
    input_indices: tuple[int, ...],
    depth: int,
) -> Expression:
    if depth <= 0 or random_source.random() < 0.22:
        return _random_leaf(random_source, input_indices)
    kind = random_source.randrange(7)
    if kind == 0:
        return 0.5 * (
            _random_subexpression(random_source, input_indices, depth - 1)
            + _random_subexpression(random_source, input_indices, depth - 1)
        )
    if kind == 1:
        return 0.5 * (
            _random_subexpression(random_source, input_indices, depth - 1)
            - _random_subexpression(random_source, input_indices, depth - 1)
        )
    if kind == 2:
        return (
            _bounded(_random_subexpression(random_source, input_indices, depth - 1))
            * _bounded(_random_subexpression(random_source, input_indices, depth - 1))
        )
    if kind == 3:
        return operation(
            InstructionType.SIN_F32,
            _random_subexpression(random_source, input_indices, depth - 1),
        )
    if kind == 4:
        return operation(
            InstructionType.COS_F32,
            _random_subexpression(random_source, input_indices, depth - 1),
        )
    if kind == 5:
        return _bounded(_random_subexpression(random_source, input_indices, depth - 1))
    return operation(
        InstructionType.FMA_F32,
        0.5 * _bounded(_random_subexpression(random_source, input_indices, depth - 1)),
        _bounded(_random_subexpression(random_source, input_indices, depth - 1)),
        0.25 * _random_subexpression(random_source, input_indices, depth - 1),
    )


def _random_expression(
    random_source: random.Random,
    root: InstructionType,
    input_indices: tuple[int, ...],
) -> Expression:
    first = _random_subexpression(random_source, input_indices, 2)
    second = _random_subexpression(random_source, input_indices, 2)
    third = _random_subexpression(random_source, input_indices, 2)
    if root in {InstructionType.ADD_F32, InstructionType.SUB_F32, InstructionType.MUL_F32}:
        return operation(root, first, second)
    if root == InstructionType.DIV_F32:
        return operation(root, first, _positive(second))
    if root in {
        InstructionType.SQRT_F32,
        InstructionType.RCP_F32,
        InstructionType.LG2_F32,
        InstructionType.RSQRT_F32,
        InstructionType.LOG_F32,
    }:
        return operation(root, _positive(first))
    if root in {InstructionType.EX2_F32, InstructionType.EXP_F32}:
        return operation(root, 0.5 * _bounded(first))
    if root == InstructionType.FMA_F32:
        return operation(root, first, second, third)
    return operation(root, first)


def _assert_close(
    test: unittest.TestCase,
    observed: float,
    expected: float,
    *,
    relative: float,
    absolute: float,
    message: str,
) -> None:
    test.assertTrue(math.isfinite(observed), f"non-finite observed value: {message}")
    test.assertTrue(math.isfinite(expected), f"non-finite expected value: {message}")
    tolerance = absolute + relative * max(abs(observed), abs(expected))
    test.assertLessEqual(abs(observed - expected), tolerance, message)


def _site_rates_and_partials(
    bundle: SystemDerivativeBundle,
    leaves: tuple[float, ...],
) -> tuple[tuple[float, float], tuple[tuple[float, ...], tuple[float, ...]]]:
    rates: list[float] = []
    partials: list[tuple[float, ...]] = []
    for site in bundle.sites:
        rates.append(site.primal.evaluate_validated(leaves))
        partials.append(
            tuple(program.evaluate_validated(leaves) for program in site.partials)
        )
    return (rates[0], rates[1]), (partials[0], partials[1])


def _state_rhs(state: tuple[float, ...], rates: tuple[float, float]) -> tuple[float, ...]:
    growth_1 = rates[0] * state[0]
    growth_2 = rates[1] * state[0]
    return (
        growth_1 + growth_2 - 0.0055 * state[0],
        -2.58 * growth_1,
        -1.71 * growth_2,
        0.21 * state[0] - 0.0466 * state[0] * state[0],
    )


def _evaluate_stage(
    bundle: SystemDerivativeBundle,
    bindings: tuple[int, ...],
    parameters: tuple[float, ...],
    fixed_constants: tuple[float, ...],
    state: tuple[float, ...],
    sensitivities: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    parameter_count = len(parameters)
    bank = state + parameters + fixed_constants
    leaves = tuple(bank[index] for index in bindings)
    rates, leaf_partials = _site_rates_and_partials(bundle, leaves)
    rate_state = [[0.0] * 4 for _ in range(2)]
    rate_parameter = [[0.0] * parameter_count for _ in range(2)]
    for site, derivative_values in enumerate(leaf_partials):
        offset = bundle.sites[site].leaf_inputs[0]
        for local_leaf, derivative in enumerate(derivative_values):
            binding = bindings[offset + local_leaf]
            if binding < 4:
                rate_state[site][binding] += derivative
            elif binding < 4 + parameter_count:
                rate_parameter[site][binding - 4] += derivative

    growth_state = [[0.0] * 4 for _ in range(2)]
    for component in range(4):
        growth_state[0][component] = state[0] * rate_state[0][component]
        growth_state[1][component] = state[0] * rate_state[1][component]
        if component == 0:
            growth_state[0][component] += rates[0]
            growth_state[1][component] += rates[1]

    system_jacobian = [[0.0] * 4 for _ in range(4)]
    for component in range(4):
        system_jacobian[0][component] = (
            growth_state[0][component]
            + growth_state[1][component]
            - (0.0055 if component == 0 else 0.0)
        )
        system_jacobian[1][component] = -2.58 * growth_state[0][component]
        system_jacobian[2][component] = -1.71 * growth_state[1][component]
    system_jacobian[3][0] = 0.21 - 2.0 * 0.0466 * state[0]

    parameter_jacobian = [[0.0] * parameter_count for _ in range(4)]
    for parameter in range(parameter_count):
        growth_1 = state[0] * rate_parameter[0][parameter]
        growth_2 = state[0] * rate_parameter[1][parameter]
        parameter_jacobian[0][parameter] = growth_1 + growth_2
        parameter_jacobian[1][parameter] = -2.58 * growth_1
        parameter_jacobian[2][parameter] = -1.71 * growth_2

    sensitivity_rhs = [[0.0] * parameter_count for _ in range(4)]
    for component in range(4):
        for parameter in range(parameter_count):
            sensitivity_rhs[component][parameter] = parameter_jacobian[component][parameter] + sum(
                system_jacobian[component][source] * sensitivities[source][parameter]
                for source in range(4)
            )
    return _state_rhs(state, rates), tuple(tuple(row) for row in sensitivity_rhs)


def _rk4_step_with_sensitivities(
    bundle: SystemDerivativeBundle,
    bindings: tuple[int, ...],
    parameters: tuple[float, ...],
    fixed_constants: tuple[float, ...],
    state: tuple[float, ...],
    sensitivities: tuple[tuple[float, ...], ...],
    step_size: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    state_stages: list[tuple[float, ...]] = []
    sensitivity_stages: list[tuple[tuple[float, ...], ...]] = []
    for stage in range(4):
        if stage == 0:
            stage_state = state
            stage_sensitivity = sensitivities
        else:
            source = stage - 1
            scale = step_size if stage == 3 else 0.5 * step_size
            stage_state = tuple(
                state[component] + scale * state_stages[source][component]
                for component in range(4)
            )
            stage_sensitivity = tuple(
                tuple(
                    sensitivities[component][parameter]
                    + scale * sensitivity_stages[source][component][parameter]
                    for parameter in range(len(parameters))
                )
                for component in range(4)
            )
        state_rhs, sensitivity_rhs = _evaluate_stage(
            bundle,
            bindings,
            parameters,
            fixed_constants,
            stage_state,
            stage_sensitivity,
        )
        state_stages.append(state_rhs)
        sensitivity_stages.append(sensitivity_rhs)

    weights = (1.0, 2.0, 2.0, 1.0)
    next_state = tuple(
        state[component]
        + step_size
        / 6.0
        * sum(weights[stage] * state_stages[stage][component] for stage in range(4))
        for component in range(4)
    )
    next_sensitivity = tuple(
        tuple(
            sensitivities[component][parameter]
            + step_size
            / 6.0
            * sum(
                weights[stage] * sensitivity_stages[stage][component][parameter]
                for stage in range(4)
            )
            for parameter in range(len(parameters))
        )
        for component in range(4)
    )
    return next_state, next_sensitivity


def _trajectory_observations(
    bundle: SystemDerivativeBundle,
    bindings: tuple[int, ...],
    parameters: tuple[float, ...],
    fixed_constants: tuple[float, ...],
    initial_state: tuple[float, ...],
    observation_count: int,
    steps_per_observation: int,
    step_size: float,
) -> tuple[tuple[float, ...], ...]:
    state = initial_state
    sensitivities = tuple((0.0,) * len(parameters) for _ in range(4))
    observations: list[tuple[float, ...]] = []
    for _observation in range(observation_count):
        for _step in range(steps_per_observation):
            state, sensitivities = _rk4_step_with_sensitivities(
                bundle,
                bindings,
                parameters,
                fixed_constants,
                state,
                sensitivities,
                step_size,
            )
        observations.append(state)
    return tuple(observations)


def _trajectory_loss_and_gradient(
    bundle: SystemDerivativeBundle,
    bindings: tuple[int, ...],
    parameters: tuple[float, ...],
    fixed_constants: tuple[float, ...],
    initial_state: tuple[float, ...],
    targets: tuple[tuple[float, ...], ...],
    steps_per_observation: int,
    step_size: float,
) -> tuple[float, tuple[float, ...]]:
    state = initial_state
    sensitivities = tuple((0.0,) * len(parameters) for _ in range(4))
    loss = 0.0
    gradient = [0.0] * len(parameters)
    for target in targets:
        for _step in range(steps_per_observation):
            state, sensitivities = _rk4_step_with_sensitivities(
                bundle,
                bindings,
                parameters,
                fixed_constants,
                state,
                sensitivities,
                step_size,
            )
        for component in range(4):
            scale = 1.0 / max(abs(target[component]), 1.0)
            residual = (state[component] - target[component]) * scale
            loss += residual * residual
            for parameter in range(len(parameters)):
                gradient[parameter] += (
                    2.0 * residual * sensitivities[component][parameter] * scale
                )
    return loss, tuple(gradient)


class RandomGradientTest(unittest.TestCase):
    def test_derivative_bundle_preserves_primal_bytecode(self) -> None:
        expression = operation(
            InstructionType.FMA_F32,
            input_slot(0),
            input_slot(1),
            constant(0.0),
        )
        program = Program.from_expression(expression)
        bundle = PostorderDerivativeBundle.from_program(program, 2, range(2))
        self.assertEqual(bundle.primal.data, program.data)
        self.assertIn(
            InstructionType.FMA_F32,
            tuple(instruction.kind for instruction in bundle.primal.instructions()),
        )

    def test_random_smooth_ast_gradients_match_finite_differences(self) -> None:
        random_source = random.Random(0x5EC4A7)
        input_indices = tuple(range(8))
        for case in range(512):
            root = _SMOOTH_OPERATIONS[case % len(_SMOOTH_OPERATIONS)]
            expression = _random_expression(random_source, root, input_indices)
            if case % 4 == 0:
                repeated = input_slot(case % len(input_indices))
                expression = expression + 0.125 * repeated * repeated
            program = Program.from_expression(expression)
            tape = PostorderADTape.from_program(program, len(input_indices))
            bundle = PostorderDerivativeBundle.from_program(
                program,
                len(input_indices),
                input_indices,
            )
            for point in range(2):
                inputs = [random_source.uniform(-0.75, 0.75) for _ in input_indices]
                value, reverse_partials = tape.evaluate(inputs)
                _assert_close(
                    self,
                    value,
                    program.evaluate(inputs),
                    relative=2.0e-12,
                    absolute=2.0e-12,
                    message=f"case={case} root={root.name} point={point} primal",
                )
                for leaf in input_indices:
                    epsilon = 1.0e-6 * max(1.0, abs(inputs[leaf]))
                    positive = inputs.copy()
                    negative = inputs.copy()
                    positive[leaf] += epsilon
                    negative[leaf] -= epsilon
                    finite_difference = (
                        program.evaluate(positive) - program.evaluate(negative)
                    ) / (2.0 * epsilon)
                    symbolic = bundle.partials[leaf].evaluate(inputs)
                    message = f"case={case} root={root.name} point={point} leaf={leaf}"
                    _assert_close(
                        self,
                        reverse_partials[leaf],
                        finite_difference,
                        relative=3.0e-5,
                        absolute=2.0e-7,
                        message=message + " reverse",
                    )
                    _assert_close(
                        self,
                        symbolic,
                        finite_difference,
                        relative=4.0e-5,
                        absolute=3.0e-7,
                        message=message + " symbolic",
                    )

    def test_random_system_trajectory_gradients_include_rk4_chain_rule(self) -> None:
        random_source = random.Random(0x7A4EC70)
        shape = KernelShape(
            ast_leaf_counts=(8, 8),
            state_count=4,
            constant_count=8,
            trajectory_count=1,
            observation_count=3,
        )
        parameter_count = 6
        fixed_constants = (1.0, 0.0)
        for case in range(32):
            expressions = []
            for site, offset in enumerate(shape.ast_input_offsets):
                root = _SMOOTH_OPERATIONS[(2 * case + site) % len(_SMOOTH_OPERATIONS)]
                raw = _random_expression(
                    random_source,
                    root,
                    tuple(range(offset, offset + shape.ast_leaf_counts[site])),
                )
                expressions.append(0.01 * (1.0 + _bounded(raw)))
            genome = SystemGenome(tuple(Program.from_expression(value) for value in expressions))
            bundle = SystemDerivativeBundle.from_genome(genome, shape)

            bindings = list(range(4, 4 + parameter_count)) + list(range(4))
            bindings.extend(random_source.randrange(shape.bank_slot_count) for _ in range(6))
            random_source.shuffle(bindings)
            binding_tuple = tuple(bindings)
            parameters = tuple(random_source.uniform(0.08, 0.45) for _ in range(parameter_count))
            target_parameters = tuple(
                value * (1.03 + 0.01 * ((index + case) % 3))
                for index, value in enumerate(parameters)
            )
            initial_state = (
                random_source.uniform(0.08, 0.24),
                random_source.uniform(0.8, 1.4),
                random_source.uniform(0.6, 1.2),
                random_source.uniform(0.0, 0.2),
            )
            targets = _trajectory_observations(
                bundle,
                binding_tuple,
                target_parameters,
                fixed_constants,
                initial_state,
                shape.observation_count,
                3,
                0.02,
            )
            loss, gradient = _trajectory_loss_and_gradient(
                bundle,
                binding_tuple,
                parameters,
                fixed_constants,
                initial_state,
                targets,
                3,
                0.02,
            )
            self.assertTrue(math.isfinite(loss), f"case={case} trajectory loss")
            for parameter in range(parameter_count):
                epsilon = 2.0e-6 * max(1.0, abs(parameters[parameter]))
                positive = list(parameters)
                negative = list(parameters)
                positive[parameter] += epsilon
                negative[parameter] -= epsilon
                positive_loss = _trajectory_loss_and_gradient(
                    bundle,
                    binding_tuple,
                    tuple(positive),
                    fixed_constants,
                    initial_state,
                    targets,
                    3,
                    0.02,
                )[0]
                negative_loss = _trajectory_loss_and_gradient(
                    bundle,
                    binding_tuple,
                    tuple(negative),
                    fixed_constants,
                    initial_state,
                    targets,
                    3,
                    0.02,
                )[0]
                finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon)
                _assert_close(
                    self,
                    gradient[parameter],
                    finite_difference,
                    relative=3.0e-4,
                    absolute=2.0e-9,
                    message=f"case={case} parameter={parameter} trajectory gradient",
                )


if __name__ == "__main__":
    unittest.main()
