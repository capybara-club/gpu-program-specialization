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

from dataclasses import dataclass
import math
import struct
from typing import Sequence

from .ast import InstructionType, Program, input_slot
from .fed_batch import INVALID_MSE, PLANTED_BINDINGS, PLANTED_CONSTANTS, planted_programs
from .shape import KernelShape


HELD_OUT_INITIAL_STATES = (
    (0.12, 7.5, 3.5, 0.0),
    (0.15, 12.0, 4.0, 0.0),
    (0.18, 16.0, 3.0, 0.0),
    (0.08, 8.0, 6.0, 0.0),
    (0.22, 6.0, 2.0, 0.0),
    (0.14, 18.0, 6.5, 0.0),
)

RATE_SURFACE_STATES = tuple(
    (biomass, glucose, sucrose, 0.0)
    for biomass in (0.05, 0.1, 0.2, 0.4, 0.8)
    for glucose in (1.0, 5.0, 10.0, 20.0, 35.0)
    for sucrose in (0.5, 2.5, 5.0, 8.0)
)


@dataclass(frozen=True)
class RateSurfaceMetrics:
    joint_nrmse: float
    site_nrmse: tuple[float, ...]
    maximum_absolute_error: float
    valid_points: int


_COMMUTATIVE = {
    InstructionType.ADD_F32: "add",
    InstructionType.MUL_F32: "mul",
    InstructionType.MIN_F32: "min",
    InstructionType.MAX_F32: "max",
}

_BINARY = {
    InstructionType.SUB_F32: "sub",
    InstructionType.DIV_F32: "div",
}

_UNARY = {
    InstructionType.NEG_F32: "neg",
    InstructionType.ABS_F32: "abs",
    InstructionType.SQRT_F32: "sqrt",
    InstructionType.RCP_F32: "rcp",
    InstructionType.SIN_F32: "sin",
    InstructionType.COS_F32: "cos",
    InstructionType.EX2_F32: "ex2",
    InstructionType.LG2_F32: "lg2",
    InstructionType.RSQRT_F32: "rsqrt",
    InstructionType.TANH_F32: "tanh",
    InstructionType.EXP_F32: "exp",
    InstructionType.LOG_F32: "log",
}


def _canonical_commutative(name: str, lhs: tuple, rhs: tuple) -> tuple:
    values: list[tuple] = []
    for value in (lhs, rhs):
        if value and value[0] == name:
            values.extend(value[1])
        else:
            values.append(value)
    return name, tuple(sorted(values, key=repr))


def canonical_structure(
    program: Program,
    bindings: Sequence[int],
    shape: KernelShape,
) -> tuple:
    """Resolve leaves and canonicalize a program while anonymizing parameters."""

    program.validate(shape.input_count)
    if len(bindings) != shape.input_count:
        raise ValueError("binding count does not match the kernel shape")
    stack: list[tuple] = []
    for instruction in program.instructions():
        kind = instruction.kind
        if kind == InstructionType.STATIC_COLUMN_INPUT_F32:
            source = bindings[int(instruction.operand)]
            stack.append(("state", source) if source < shape.state_count else ("parameter",))
        elif kind == InstructionType.CONSTANT_BITS_F32:
            stack.append(("literal", int(instruction.operand)))
        elif kind == InstructionType.RETURN_F32:
            return stack[-1]
        elif kind in _UNARY:
            stack.append((_UNARY[kind], stack.pop()))
        elif kind in _COMMUTATIVE:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(_canonical_commutative(_COMMUTATIVE[kind], lhs, rhs))
        elif kind in _BINARY:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append((_BINARY[kind], lhs, rhs))
        else:
            raise ValueError(f"structural recovery does not support {kind.name}")
    raise AssertionError("validated program did not return")


def strict_structure_match(
    programs: Sequence[Program],
    bindings: Sequence[int],
    shape: KernelShape,
) -> bool:
    return all(site_structure_matches(programs, bindings, shape))


def site_structure_matches(
    programs: Sequence[Program],
    bindings: Sequence[int],
    shape: KernelShape,
) -> tuple[bool, ...]:
    if len(programs) != shape.ast_count:
        return tuple(False for _ in range(shape.ast_count))
    target = planted_programs(shape)
    alternatives: list[tuple[Program, ...]] = [(target[0],), (target[1],)]
    if shape.ast_count == 2 and shape.ast_leaf_counts[1] >= 5:
        offset = shape.ast_input_offsets[1]
        leaves = [input_slot(offset + index) for index in range(5)]
        simplified_second = Program.from_expression(
            leaves[0] * leaves[1] / (leaves[2] + leaves[3] * leaves[4])
        )
        alternatives[1] = alternatives[1] + (simplified_second,)
    return tuple(
        any(
            canonical_structure(candidate, bindings, shape)
            == canonical_structure(planted, PLANTED_BINDINGS, shape)
            for planted in site_alternatives
        )
        for candidate, site_alternatives in zip(programs, alternatives)
    )


def _evaluate_rates(
    programs: Sequence[Program],
    constants: Sequence[float],
    bindings: Sequence[int],
    state: Sequence[float],
) -> tuple[float, ...]:
    bank = tuple(state) + tuple(constants)
    leaves = tuple(bank[index] for index in bindings)
    return tuple(program.evaluate_validated(leaves) for program in programs)


def _planted_rates(state: Sequence[float]) -> tuple[float, float]:
    biomass, glucose, sucrose, _product = state
    mu_m1, kc1, k1, mu_m2, kc2, k2 = PLANTED_CONSTANTS[:6]
    return (
        mu_m1 * glucose / ((glucose + kc1 * biomass) * (1.0 + k1 * sucrose)),
        mu_m2 * sucrose / ((sucrose + kc2 * biomass) * (1.0 + k2 * glucose)),
    )


def rate_surface_metrics(
    programs: Sequence[Program],
    constants: Sequence[float],
    bindings: Sequence[int],
    shape: KernelShape,
    states: Sequence[Sequence[float]] = RATE_SURFACE_STATES,
) -> RateSurfaceMetrics:
    if len(programs) != shape.ast_count or len(constants) != shape.constant_count or len(bindings) != shape.input_count:
        raise ValueError("candidate dimensions do not match the kernel shape")
    for program in programs:
        program.validate(shape.input_count)
    squared_error = [0.0] * shape.ast_count
    squared_target = [0.0] * shape.ast_count
    maximum_error = 0.0
    valid = 0
    try:
        for state in states:
            observed = _evaluate_rates(programs, constants, bindings, state)
            expected = _planted_rates(state)
            if not all(math.isfinite(value) for value in observed):
                raise ArithmeticError("non-finite candidate rate")
            for site, (candidate, planted) in enumerate(zip(observed, expected)):
                error = candidate - planted
                squared_error[site] += error * error
                squared_target[site] += planted * planted
                maximum_error = max(maximum_error, abs(error))
            valid += 1
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        return RateSurfaceMetrics(math.inf, tuple(math.inf for _ in range(shape.ast_count)), math.inf, valid)
    site_nrmse = tuple(
        math.sqrt(error / max(target, 1.0e-30))
        for error, target in zip(squared_error, squared_target)
    )
    return RateSurfaceMetrics(
        math.sqrt(sum(squared_error) / max(sum(squared_target), 1.0e-30)),
        site_nrmse,
        maximum_error,
        valid,
    )


def _rhs(state: Sequence[float], rates: Sequence[float]) -> tuple[float, float, float, float]:
    biomass = state[0]
    growth1 = rates[0] * biomass
    growth2 = rates[1] * biomass
    return (
        growth1 + growth2 - 0.0055 * biomass,
        -2.58 * growth1,
        -1.71 * growth2,
        0.21 * biomass - 0.0466 * biomass * biomass,
    )


def _rk4_step(state: Sequence[float], step: float, rate_function) -> tuple[float, ...]:
    first = _rhs(state, rate_function(state))
    stage = tuple(state[index] + 0.5 * step * first[index] for index in range(4))
    second = _rhs(stage, rate_function(stage))
    stage = tuple(state[index] + 0.5 * step * second[index] for index in range(4))
    third = _rhs(stage, rate_function(stage))
    stage = tuple(state[index] + step * third[index] for index in range(4))
    fourth = _rhs(stage, rate_function(stage))
    return tuple(
        state[index] + step * (first[index] + 2.0 * second[index] + 2.0 * third[index] + fourth[index]) / 6.0
        for index in range(4)
    )


def held_out_trajectory_mse(
    programs: Sequence[Program],
    constants: Sequence[float],
    bindings: Sequence[int],
    shape: KernelShape,
    initial_states: Sequence[Sequence[float]] = HELD_OUT_INITIAL_STATES,
    steps_per_observation: int = 16,
    observations: int = 12,
) -> float:
    if steps_per_observation <= 0 or observations <= 0:
        raise ValueError("trajectory integration counts must be positive")
    for program in programs:
        program.validate(shape.input_count)
    squared_error = 0.0
    values = 0
    step = 8.0 / steps_per_observation
    try:
        for initial in initial_states:
            candidate = tuple(initial)
            planted = tuple(initial)
            for _observation in range(observations):
                for _integration in range(steps_per_observation):
                    candidate = _rk4_step(
                        candidate,
                        step,
                        lambda state: _evaluate_rates(programs, constants, bindings, state),
                    )
                    planted = _rk4_step(planted, step, _planted_rates)
                    if not all(math.isfinite(value) for value in candidate):
                        return INVALID_MSE
                for candidate_value, planted_value in zip(candidate, planted):
                    error = candidate_value - planted_value
                    squared_error += error * error
                    values += 1
        result = squared_error / values
        return result if math.isfinite(result) and result < INVALID_MSE else INVALID_MSE
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        return INVALID_MSE


def resolved_expression(
    program: Program,
    constants: Sequence[float],
    bindings: Sequence[int],
    shape: KernelShape,
) -> str:
    """Render a compact expression after resolving each dynamic leaf binding."""

    stack: list[str] = []
    binary_symbols = {
        InstructionType.ADD_F32: "+",
        InstructionType.SUB_F32: "-",
        InstructionType.MUL_F32: "*",
        InstructionType.DIV_F32: "/",
    }
    for instruction in program.instructions():
        kind = instruction.kind
        if kind == InstructionType.STATIC_COLUMN_INPUT_F32:
            source = bindings[int(instruction.operand)]
            if source < shape.state_count:
                stack.append(shape_state_name(source))
            else:
                constant_index = source - shape.state_count
                stack.append(f"c{constant_index}[{constants[constant_index]:.7g}]")
        elif kind == InstructionType.CONSTANT_BITS_F32:
            value = struct.unpack("<f", struct.pack("<I", int(instruction.operand)))[0]
            stack.append(f"{value:.7g}")
        elif kind == InstructionType.RETURN_F32:
            return stack[-1]
        elif kind in binary_symbols:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(f"({lhs} {binary_symbols[kind]} {rhs})")
        elif kind in _COMMUTATIVE:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(f"{_COMMUTATIVE[kind]}({lhs}, {rhs})")
        elif kind in _UNARY:
            stack.append(f"{_UNARY[kind]}({stack.pop()})")
        else:
            raise ValueError(f"expression rendering does not support {kind.name}")
    raise AssertionError("validated program did not return")


def shape_state_name(index: int) -> str:
    names = ("X", "G", "S", "P")
    return names[index] if index < len(names) else f"state{index}"
