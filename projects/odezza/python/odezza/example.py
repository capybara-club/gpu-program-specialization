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
"""Python example-system fixtures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from .ast import OutputPrograms, Program, SystemGroup, constant_source, source, state_source, toggle2
from .model import KernelShape


def lotka_volterra_group(shape: KernelShape = KernelShape()) -> SystemGroup:
    if shape.state_count != 2 or shape.constant_count < 4:
        raise ValueError("the Lotka-Volterra example requires two states and four constants")
    prey = source(state_source(0))
    predator = source(state_source(1))
    alpha = source(constant_source(0))
    beta = source(constant_source(1))
    predator_growth = source(constant_source(2))
    predator_loss = source(constant_source(3))
    shared_prey = Program.from_expression(alpha * prey - beta * prey * predator, shape)
    correct_predator = Program.from_expression(predator_growth * prey * predator - predator_loss * predator, shape)
    alternate_predator = Program.from_expression(predator_growth * prey - predator_loss * predator, shape)
    return SystemGroup(
        shared=OutputPrograms((0,), (shared_prey,)),
        systems=(
            OutputPrograms((1,), (correct_predator,)),
            OutputPrograms((1,), (alternate_predator,)),
        ),
    )


def lotka_volterra_full_branch_group(shape: KernelShape = KernelShape()) -> SystemGroup:
    shared_group = lotka_volterra_group(shape)
    shared_prey = shared_group.shared.programs[0]
    return SystemGroup(
        shared=OutputPrograms((), ()),
        systems=tuple(
            OutputPrograms((0, 1), (shared_prey, system.programs[0]))
            for system in shared_group.systems
        ),
    )


def lotka_volterra_benchmark_group(shape: KernelShape = KernelShape()) -> SystemGroup:
    """Fill all eight branch slots with distinct, valid predator RHS candidates."""
    if shape.system_capacity < 8:
        raise ValueError("the benchmark requires eight system slots")
    if shape.state_count != 2 or shape.constant_count < 4:
        raise ValueError("the Lotka-Volterra benchmark requires two states and four constants")
    prey = source(state_source(0))
    predator = source(state_source(1))
    alpha = source(constant_source(0))
    beta = source(constant_source(1))
    predator_growth = source(constant_source(2))
    predator_loss = source(constant_source(3))
    shared_prey = Program.from_expression(alpha * prey - beta * prey * predator, shape)
    candidates = (
        predator_growth * prey * predator - predator_loss * predator,
        predator_growth * prey - predator_loss * predator,
        predator_growth * predator - predator_loss * prey,
        predator_growth * prey * predator - predator_loss * prey,
        predator_growth * prey * predator - predator_loss * toggle2(0, state_source(1), state_source(0)),
        predator_growth * toggle2(1, state_source(0), state_source(1)) - predator_loss * predator,
        predator_growth * toggle2(2, state_source(0), state_source(1)) * predator
        - predator_loss * toggle2(3, state_source(1), state_source(0)),
        predator_growth * prey * toggle2(0, state_source(1), state_source(0))
        - predator_loss * toggle2(4, state_source(1), state_source(0)),
    )
    return SystemGroup(
        shared=OutputPrograms((0,), (shared_prey,)),
        systems=tuple(
            OutputPrograms((1,), (Program.from_expression(candidate, shape),))
            for candidate in candidates
        ),
    )


def _complete_rhs(group: SystemGroup, system_index: int, shape: KernelShape) -> tuple[Program, ...]:
    programs: list[Program | None] = [None] * shape.output_count
    for output_index, program in zip(group.shared.output_indices, group.shared.programs):
        programs[output_index] = program
    system = group.systems[system_index]
    for output_index, program in zip(system.output_indices, system.programs):
        programs[output_index] = program
    if any(program is None for program in programs):
        raise ValueError("system group does not provide a complete RHS")
    return tuple(program for program in programs if program is not None)


def _rk4_step(
    programs: Sequence[Program],
    state: Sequence[float],
    constants: Sequence[float],
    permutation: int,
    h: float,
    shape: KernelShape,
) -> list[float]:
    base = [float(value) for value in state]
    current = list(base)
    summed = [0.0] * shape.state_count
    for stage in range(4):
        derivatives = [
            program.evaluate(current, constants, permutation, shape)
            for program in programs
        ]
        weight = 1.0 if stage in (0, 3) else 2.0
        for index in range(shape.state_count):
            summed[index] += weight * derivatives[index]
        if stage != 3:
            stage_h = h if stage == 2 else 0.5 * h
            current = [
                base[index] + stage_h * derivatives[index]
                for index in range(shape.state_count)
            ]
    return [
        base[index] + h * summed[index] / 6.0
        for index in range(shape.state_count)
    ]


def lotka_volterra_reference(
    shape: KernelShape = KernelShape(),
    constants: Sequence[float] = (1.5, 1.0, 0.75, 1.0),
    permutation: int = 0,
    steps_per_observation: int = 4,
) -> list[float]:
    group = lotka_volterra_group(shape)
    programs = _complete_rhs(group, 0, shape)
    if len(constants) != shape.constant_count:
        raise ValueError("constant vector does not match the kernel shape")
    if steps_per_observation <= 0:
        raise ValueError("steps_per_observation must be positive")
    states = [
        [10.0 + trajectory, 5.0 + 0.25 * trajectory]
        for trajectory in range(shape.trajectory_count)
    ]
    initial = [
        states[trajectory][component]
        for component in range(shape.state_count)
        for trajectory in range(shape.trajectory_count)
    ]
    targets = [
        [[0.0 for _ in range(shape.observation_count)] for _ in range(shape.trajectory_count)]
        for _ in range(shape.state_count)
    ]
    h = shape.observation_interval / steps_per_observation
    for observation in range(shape.observation_count):
        for trajectory in range(shape.trajectory_count):
            state = states[trajectory]
            for _ in range(steps_per_observation):
                state = _rk4_step(programs, state, constants, permutation, h, shape)
            if not all(math.isfinite(value) for value in state):
                raise ValueError("example trajectory became non-finite")
            states[trajectory] = state
            for component in range(shape.state_count):
                targets[component][trajectory][observation] = state[component]
    flattened = list(initial)
    flattened.extend(
        targets[component][trajectory][observation]
        for component in range(shape.state_count)
        for trajectory in range(shape.trajectory_count)
        for observation in range(shape.observation_count)
    )
    return flattened


def _constant_banks(system_index: int, bank_count: int) -> list[list[float]]:
    if isinstance(bank_count, bool) or not isinstance(bank_count, int) or bank_count <= 0:
        raise ValueError("bank_count must be a positive integer")
    base = (1.5, 1.0, 0.75, 1.0)
    result: list[list[float]] = []
    for bank_index in range(bank_count):
        if bank_index == 0:
            result.append(list(base))
            continue
        result.append(
            [
                value * (
                    1.0
                    + ((((bank_index + 1) * (constant_index + 3) + 5 * system_index) % 17) - 8)
                    / 200.0
                )
                for constant_index, value in enumerate(base)
            ]
        )
    return result


def benchmark_bundle(
    shape: KernelShape = KernelShape(),
    bank_count: int = 64,
) -> dict[str, object]:
    group = lotka_volterra_benchmark_group(shape)
    constants = [1.5, 1.0, 0.75, 1.0]
    return {
        "schema": "odezza.example-bundle",
        "schema_version": 1,
        "name": "lotka_volterra_eight_system_throughput_benchmark",
        "shape": shape.to_dict(),
        "group": group.to_document(shape),
        "system_count": len(group.systems),
        "constant_banks": [
            _constant_banks(system_index, bank_count)
            for system_index in range(len(group.systems))
        ],
        "ground_truth": {
            "system": 0,
            "constant_bank": 0,
            "permutation": 0,
            "configuration": 0,
        },
        "steps_per_observation": 4,
        "reference_data": lotka_volterra_reference(shape, constants, 0, 4),
    }


def example_bundle(shape: KernelShape = KernelShape()) -> dict[str, object]:
    group = lotka_volterra_group(shape)
    constants = [1.5, 1.0, 0.75, 1.0]
    return {
        "schema": "odezza.example-bundle",
        "schema_version": 1,
        "name": "lotka_volterra_shared_prey_rhs",
        "shape": shape.to_dict(),
        "group": group.to_document(shape),
        "system_count": len(group.systems),
        "constant_banks": [[constants], [constants]],
        "ground_truth": {
            "system": 0,
            "constant_bank": 0,
            "permutation": 0,
            "configuration": 0,
        },
        "steps_per_observation": 4,
        "reference_data": lotka_volterra_reference(shape, constants, 0, 4),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write an example multi-system scoring document.")
    parser.add_argument("-o", "--output", default="-")
    parser.add_argument("--group-only", action="store_true")
    parser.add_argument("--full-branches", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--banks", type=int, default=64)
    arguments = parser.parse_args(argv)
    shape = KernelShape()
    if arguments.full_branches and arguments.benchmark:
        parser.error("--full-branches and --benchmark are mutually exclusive")
    if arguments.full_branches:
        value = lotka_volterra_full_branch_group(shape).to_document(shape)
    elif arguments.benchmark:
        value = (
            lotka_volterra_benchmark_group(shape).to_document(shape)
            if arguments.group_only
            else benchmark_bundle(shape, arguments.banks)
        )
    else:
        value = (
            lotka_volterra_group(shape).to_document(shape)
            if arguments.group_only
            else example_bundle(shape)
        )
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if arguments.output == "-":
        print(rendered, end="")
    else:
        Path(arguments.output).write_text(rendered)


if __name__ == "__main__":
    main()
