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
from dataclasses import dataclass
import math
import random
from typing import Sequence

from .ast import InstructionType
from .bindings import constant_leaf, state_leaf
from .fed_batch import FED_BATCH_SHAPE, PLANTED_CONSTANTS, PackedPopulation
from .shape import KernelShape
from .toggle_ast import (
    ToggleProgram,
    ToggleSystem,
    source_input,
    toggle1,
    toggle2,
    toggle_literal,
    toggle_operation,
)


_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


@dataclass(frozen=True)
class TogglePopulation:
    """Per-system constant banks paired with low-bit AST permutations."""

    num_systems: int
    constant_bank_count: int
    toggle_bit_count: int
    constant_banks: array

    @property
    def permutations_per_bank(self) -> int:
        return 1 << self.toggle_bit_count

    @property
    def configurations_per_system(self) -> int:
        return self.constant_bank_count * self.permutations_per_bank

    def constants(
        self,
        system_index: int,
        bank_index: int,
        shape: KernelShape = FED_BATCH_SHAPE,
    ) -> tuple[float, ...]:
        if not 0 <= system_index < self.num_systems:
            raise IndexError("toggle system index is outside the population")
        if not 0 <= bank_index < self.constant_bank_count:
            raise IndexError("constant bank index is outside the population")
        start = (
            (system_index * self.constant_bank_count + bank_index)
            * shape.constant_count
        )
        return tuple(self.constant_banks[start : start + shape.constant_count])


def benchmark_toggle_systems(
    count: int,
    shape: KernelShape = FED_BATCH_SHAPE,
) -> tuple[ToggleSystem, ...]:
    """Create distinct fed-batch systems with five bits of leaf variation."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("toggle-system count must be positive")
    if shape.ast_leaf_counts != (8, 8):
        raise ValueError("the fed-batch toggle family requires two eight-leaf sites")

    first_leaves = (
        toggle1(0, constant_leaf(0), constant_leaf(3)),
        toggle1(1, state_leaf(1), state_leaf(2)),
        toggle1(2, state_leaf(1), state_leaf(0)),
        toggle2(
            3,
            constant_leaf(1),
            constant_leaf(4),
            constant_leaf(7),
            state_leaf(1),
        ),
        source_input(state_leaf(0)),
        source_input(constant_leaf(6)),
        source_input(constant_leaf(2)),
        source_input(state_leaf(2)),
    )
    second_leaves = (
        source_input(constant_leaf(3)),
        source_input(state_leaf(2)),
        source_input(state_leaf(2)),
        source_input(constant_leaf(4)),
        source_input(state_leaf(0)),
        source_input(constant_leaf(6)),
        source_input(constant_leaf(5)),
        source_input(state_leaf(1)),
    )

    systems: list[ToggleSystem] = []
    for index in range(count):
        first_scale = 1.0 + index / 8192.0
        second_scale = 1.0 - index / 16384.0
        a = first_leaves
        b = second_leaves
        first_rate = first_scale * a[0] * a[1] / (
            (a[2] + a[3] * a[4]) * (a[5] + a[6] * a[7])
        )
        second_rate = second_scale * b[0] * b[1] / (
            (b[2] + b[3] * b[4]) * (b[5] + b[6] * b[7])
        )
        system = ToggleSystem(
            (
                ToggleProgram.from_expression(first_rate, shape),
                ToggleProgram.from_expression(second_rate, shape),
            )
        )
        system.validate(shape)
        systems.append(system)
    return tuple(systems)


def benchmark_diverse_toggle_systems(
    count: int,
    shape: KernelShape = FED_BATCH_SHAPE,
    seed: int = 0x5EC4A7,
) -> tuple[ToggleSystem, ...]:
    """Create structurally distinct, fixed-cost systems for dispatch benchmarks."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("diverse toggle-system count must be positive")
    if shape.ast_leaf_counts != (8, 8):
        raise ValueError("the diverse fed-batch family requires two eight-leaf sites")
    sources = tuple(
        [state_leaf(index) for index in range(shape.state_count)]
        + [constant_leaf(index) for index in range(shape.constant_count)]
    )
    signatures: set[tuple[int, ...]] = set()

    def make_program(program_seed: int) -> ToggleProgram:
        attempt = 0
        while True:
            rng = random.Random(program_seed + attempt * 0x9E3779B9)
            offset = rng.randrange(len(sources))

            def choice(delta: int):
                return sources[(offset + delta) % len(sources)]

            leaves = [
                toggle1(0, choice(0), choice(5)),
                toggle1(1, choice(1), choice(8)),
                toggle1(2, choice(2), choice(10)),
                toggle2(3, choice(3), choice(6), choice(9), choice(11)),
                source_input(choice(4)),
                source_input(choice(7)),
                source_input(choice(10)),
                source_input(choice(1)),
            ]
            rng.shuffle(leaves)
            operations = [
                InstructionType.MUL_F32,
                InstructionType.MUL_F32,
                InstructionType.MUL_F32,
                InstructionType.MUL_F32,
                InstructionType.ADD_F32,
                InstructionType.ADD_F32,
                InstructionType.DIV_F32,
            ]
            rng.shuffle(operations)
            nodes = list(leaves)
            for operation_kind in operations:
                lhs = nodes.pop(rng.randrange(len(nodes)))
                rhs = nodes.pop(rng.randrange(len(nodes)))
                combined = toggle_operation(operation_kind, lhs, rhs)
                nodes.insert(rng.randrange(len(nodes) + 1), combined)
            expression = toggle_operation(
                InstructionType.MUL_F32,
                toggle_literal(0.75 + 0.5 * rng.random()),
                nodes[0],
            )
            program = ToggleProgram.from_expression(expression, shape)
            signature = tuple(int(instruction.kind) for instruction in program.instructions())
            if signature not in signatures:
                signatures.add(signature)
                return program
            attempt += 1

    systems = tuple(
        ToggleSystem(
            (
                make_program(seed + system_index * 2),
                make_program(seed + system_index * 2 + 1),
            )
        )
        for system_index in range(count)
    )
    for system in systems:
        system.validate(shape)
    return systems


def make_toggle_population(
    systems: Sequence[ToggleSystem],
    constant_bank_count: int,
    shape: KernelShape = FED_BATCH_SHAPE,
) -> TogglePopulation:
    if not systems:
        raise ValueError("a toggle population requires at least one system")
    if isinstance(constant_bank_count, bool) or not isinstance(constant_bank_count, int):
        raise TypeError("constant-bank count must be an integer")
    if constant_bank_count <= 0:
        raise ValueError("constant-bank count must be positive")
    for system in systems:
        system.validate(shape)
    toggle_bit_count = max(system.required_toggle_bits for system in systems)
    values = array("f")
    for system_index in range(len(systems)):
        for bank_index in range(constant_bank_count):
            for constant_index, incumbent in enumerate(PLANTED_CONSTANTS[: shape.constant_count]):
                if bank_index == 0:
                    value = incumbent
                else:
                    key = (
                        (system_index + 1) * 0xD6E8FEB86659FD93
                        ^ (bank_index + 1) * 0xA0761D6478BD642F
                        ^ (constant_index + 1) * 0xE7037ED1A0B428DB
                    ) & _MASK64
                    unit = (_mix64(key) >> 40) * (1.0 / 16777216.0)
                    value = incumbent + (2.0 * unit - 1.0) * 0.105 * (abs(incumbent) + 1.0)
                if not math.isfinite(value):
                    raise AssertionError("generated constant bank is non-finite")
                values.append(value)
    return TogglePopulation(
        len(systems),
        constant_bank_count,
        toggle_bit_count,
        values,
    )


def materialize_toggle_population(
    systems: Sequence[ToggleSystem],
    population: TogglePopulation,
    shape: KernelShape = FED_BATCH_SHAPE,
) -> PackedPopulation:
    """Expand toggles into the existing materialized settings/bindings layout."""

    if len(systems) != population.num_systems:
        raise ValueError("toggle systems and population have different sizes")
    configurations = population.configurations_per_system
    permutation_mask = population.permutations_per_bank - 1
    settings = array("f", [0.0]) * (
        population.num_systems * shape.constant_count * configurations
    )
    bindings = array("I", [0]) * (
        population.num_systems * shape.input_count * configurations
    )
    for system_index, system in enumerate(systems):
        for configuration in range(configurations):
            bank_index = configuration >> population.toggle_bit_count
            permutation = configuration & permutation_mask
            constants = population.constants(system_index, bank_index, shape)
            resolved = system.resolved_bindings(permutation, shape)
            for constant_index, value in enumerate(constants):
                settings[
                    ((system_index * shape.constant_count + constant_index) * configurations)
                    + configuration
                ] = value
            for leaf_index, encoded in enumerate(resolved):
                bindings[
                    ((system_index * shape.input_count + leaf_index) * configurations)
                    + configuration
                ] = encoded
    return PackedPopulation(population.num_systems, configurations, settings, bindings)
