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
"""LM system validation and symbolic forward-derivative lowering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Sequence

from .ast import (
    Expression,
    InstructionType,
    NodeKind,
    Program,
    _ARITY,
    constant,
    operation,
    optimized_constant,
    source,
    state_source,
)
from .lm_model import LMKernelShape
from .model import KernelShape


LM_SYSTEM_SCHEMA = "odezza.lm-system"
LM_SYSTEM_SCHEMA_VERSION = 1


class LMAstError(ValueError):
    pass


def lm_program_shape(shape: LMKernelShape) -> KernelShape:
    return KernelShape(
        state_names=tuple(f"state_{index}" for index in range(shape.state_count)),
        constant_count=1,
        trajectory_count=1,
        observation_count=1,
        observation_interval=1.0,
        system_capacity=1,
        shared_patch_capacity=8,
        system_patch_capacity=8,
    )


def lm_sass_shape(shape: LMKernelShape) -> KernelShape:
    return KernelShape(
        state_names=tuple(f"state_{index}" for index in range(shape.state_count)),
        constant_count=shape.optimized_constant_count,
        trajectory_count=1,
        observation_count=1,
        observation_interval=1.0,
        system_capacity=1,
        shared_patch_capacity=8,
        system_patch_capacity=shape.site_patch_capacity,
    )


@dataclass(frozen=True)
class LMSystem:
    rhs: tuple[Program, ...]

    @classmethod
    def from_expressions(
        cls,
        expressions: Sequence[Expression],
        shape: LMKernelShape,
    ) -> "LMSystem":
        base = lm_program_shape(shape)
        return cls(
            tuple(
                Program.from_expression(
                    expression,
                    base,
                    shape.optimized_constant_count,
                )
                for expression in expressions
            )
        )

    def validate(self, shape: LMKernelShape) -> None:
        if len(self.rhs) != shape.state_count:
            raise LMAstError(
                f"LM system contains {len(self.rhs)} right-hand sides; expected {shape.state_count}"
            )
        base = lm_program_shape(shape)
        for program in self.rhs:
            program.validate(base, shape.optimized_constant_count)
            for instruction in program.instructions():
                if instruction.kind == InstructionType.CONSTANT_F32:
                    raise LMAstError(
                        "the first LM shape supports state, optimized-constant, and literal leaves; indexed scoring constants are not part of this ABI"
                    )

    @property
    def required_toggle_bits(self) -> int:
        return max((program.required_toggle_bits for program in self.rhs), default=0)

    def to_document(self, shape: LMKernelShape) -> dict[str, Any]:
        self.validate(shape)
        semantic = {
            "schema": LM_SYSTEM_SCHEMA,
            "schema_version": LM_SYSTEM_SCHEMA_VERSION,
            "required_toggle_bits": self.required_toggle_bits,
            "rhs": [program.data.hex() for program in self.rhs],
        }
        digest = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        base = lm_program_shape(shape)
        return {
            **semantic,
            "system_id": digest,
            "decoded": [
                program.decoded(base, shape.optimized_constant_count)
                for program in self.rhs
            ],
        }

    @classmethod
    def from_document(
        cls,
        value: dict[str, Any],
        shape: LMKernelShape,
    ) -> "LMSystem":
        if (
            value.get("schema") != LM_SYSTEM_SCHEMA
            or value.get("schema_version") != LM_SYSTEM_SCHEMA_VERSION
        ):
            raise LMAstError("unsupported LM-system document")
        raw_rhs = value.get("rhs")
        if not isinstance(raw_rhs, list):
            raise LMAstError("LM-system document must contain an rhs array")
        try:
            result = cls(tuple(Program(bytes.fromhex(item)) for item in raw_rhs))
        except (TypeError, ValueError) as exc:
            raise LMAstError(f"LM-system program bytes are invalid: {exc}") from exc
        result.validate(shape)
        expected = result.to_document(shape)
        if value.get("required_toggle_bits") != result.required_toggle_bits:
            raise LMAstError("LM-system toggle declaration disagrees with its programs")
        if value.get("system_id") != expected["system_id"]:
            raise LMAstError("LM-system identity disagrees with its programs")
        return result


@dataclass(frozen=True)
class LMSystemDerivatives:
    primal: tuple[Program, ...]
    partials: tuple[tuple[Program, ...], ...]

    @classmethod
    def from_system(
        cls,
        system: LMSystem,
        shape: LMKernelShape,
    ) -> "LMSystemDerivatives":
        system.validate(shape)
        base = lm_program_shape(shape)
        expressions = tuple(_program_expression(program) for program in system.rhs)
        partials: list[tuple[Program, ...]] = []
        for expression in expressions:
            rhs_partials = tuple(
                Program.from_expression(
                    _differentiate(expression, target, shape),
                    base,
                    shape.optimized_constant_count,
                )
                for target in range(shape.input_count)
            )
            partials.append(rhs_partials)
        return cls(system.rhs, tuple(partials))

    def site_programs(self, shape: LMKernelShape) -> tuple[Program, ...]:
        if len(self.primal) != shape.state_count or len(self.partials) != shape.state_count:
            raise LMAstError("LM derivative bundle does not match its kernel shape")
        result = list(self.primal)
        for input_index in range(shape.input_count):
            for rhs_index in range(shape.state_count):
                result.append(self.partials[rhs_index][input_index])
        if len(result) != shape.scalar_output_count:
            raise AssertionError("LM derivative scalar-output ordering is inconsistent")
        return tuple(result)

    def site_program_groups(
        self,
        shape: LMKernelShape,
    ) -> tuple[tuple[Program, ...], ...]:
        programs = self.site_programs(shape)
        return tuple(
            tuple(programs[index] for index in group)
            for group in shape.site_output_groups
        )

    def inspection(self, shape: LMKernelShape) -> dict[str, Any]:
        base = lm_program_shape(shape)
        return {
            "schema": "odezza.lm-derivative-bundle",
            "schema_version": 1,
            "site_order": "all primal RHS values, then input-major local partials by RHS",
            "sites": [
                {
                    "site_index": index,
                    "byte_count": len(program.data),
                    "instruction_count": sum(1 for _ in program.instructions()),
                    "instructions": program.decoded(
                        base,
                        shape.optimized_constant_count,
                    ),
                }
                for index, program in enumerate(self.site_programs(shape))
            ],
        }


def lower_lm_program(
    program: Program,
    shape: LMKernelShape,
) -> Program:
    program.validate(lm_program_shape(shape), shape.optimized_constant_count)
    lowered = bytearray(program.data)
    offset = 0
    for instruction in program.instructions():
        if instruction.kind == InstructionType.OPTIMIZED_CONSTANT_F32:
            lowered[offset] = InstructionType.CONSTANT_F32
        offset += _instruction_width(instruction.kind)
    result = Program(bytes(lowered))
    result.validate(lm_sass_shape(shape))
    return result


def _instruction_width(kind: InstructionType) -> int:
    if kind in {
        InstructionType.STATE_F32,
        InstructionType.CONSTANT_F32,
        InstructionType.OPTIMIZED_CONSTANT_F32,
        InstructionType.TOGGLE2_F32,
    }:
        return 2
    if kind == InstructionType.TOGGLE4_F32:
        return 3
    if kind == InstructionType.LITERAL_F32:
        return 5
    return 1


def _program_expression(program: Program) -> Expression:
    stack: list[Expression] = []
    for instruction in program.instructions():
        kind = instruction.kind
        if kind == InstructionType.STATE_F32:
            stack.append(source(state_source(instruction.operands[0])))
        elif kind == InstructionType.OPTIMIZED_CONSTANT_F32:
            stack.append(optimized_constant(instruction.operands[0]))
        elif kind == InstructionType.LITERAL_F32:
            stack.append(
                constant(
                    struct.unpack(
                        "<f",
                        struct.pack("<I", instruction.operands[0]),
                    )[0]
                )
            )
        elif kind in {InstructionType.TOGGLE2_F32, InstructionType.TOGGLE4_F32}:
            choice_count = 1 << len(instruction.operands)
            choices = tuple(stack[-choice_count:])
            del stack[-choice_count:]
            stack.append(Expression(NodeKind.TOGGLE, instruction.operands, choices))
        elif kind == InstructionType.RETURN_F32:
            break
        elif kind == InstructionType.CONSTANT_F32:
            raise LMAstError("indexed scoring constants are not supported by the first LM shape")
        elif kind in _ARITY:
            arity = _ARITY[kind]
            arguments = tuple(stack[-arity:])
            del stack[-arity:]
            stack.append(operation(kind, *arguments))
        else:
            raise LMAstError(f"LM expression reconstruction does not support {kind.name}")
    if len(stack) != 1:
        raise LMAstError("LM postorder program did not reconstruct one expression")
    return stack[0]


def _literal_value(expression: Expression) -> float | None:
    return float(expression.value) if expression.kind == NodeKind.LITERAL else None


def _is_literal(expression: Expression, value: float) -> bool:
    observed = _literal_value(expression)
    return observed is not None and observed == value


def _negate(value: Expression) -> Expression:
    if _is_literal(value, 0.0):
        return value
    literal = _literal_value(value)
    if literal is not None:
        return constant(-literal)
    if value.kind == NodeKind.OPERATION and value.value == InstructionType.NEG_F32:
        return value.arguments[0]
    return operation(InstructionType.NEG_F32, value)


def _add(lhs: Expression, rhs: Expression) -> Expression:
    if _is_literal(lhs, 0.0):
        return rhs
    if _is_literal(rhs, 0.0):
        return lhs
    left = _literal_value(lhs)
    right = _literal_value(rhs)
    if left is not None and right is not None:
        return constant(left + right)
    return operation(InstructionType.ADD_F32, lhs, rhs)


def _subtract(lhs: Expression, rhs: Expression) -> Expression:
    if _is_literal(rhs, 0.0):
        return lhs
    if _is_literal(lhs, 0.0):
        return _negate(rhs)
    left = _literal_value(lhs)
    right = _literal_value(rhs)
    if left is not None and right is not None:
        return constant(left - right)
    return operation(InstructionType.SUB_F32, lhs, rhs)


def _multiply(lhs: Expression, rhs: Expression) -> Expression:
    if _is_literal(lhs, 0.0) or _is_literal(rhs, 0.0):
        return constant(0.0)
    if _is_literal(lhs, 1.0):
        return rhs
    if _is_literal(rhs, 1.0):
        return lhs
    if _is_literal(lhs, -1.0):
        return _negate(rhs)
    if _is_literal(rhs, -1.0):
        return _negate(lhs)
    left = _literal_value(lhs)
    right = _literal_value(rhs)
    if left is not None and right is not None:
        return constant(left * right)
    return operation(InstructionType.MUL_F32, lhs, rhs)


def _divide(lhs: Expression, rhs: Expression) -> Expression:
    # Keep singular arithmetic in the AST. The rollout marks it invalid; host
    # constant folding must not abort preparation of the other candidates.
    if _is_literal(rhs, 0.0):
        return operation(InstructionType.DIV_F32, lhs, rhs)
    if _is_literal(lhs, 0.0):
        return constant(0.0)
    if _is_literal(rhs, 1.0):
        return lhs
    left = _literal_value(lhs)
    right = _literal_value(rhs)
    if left is not None and right is not None:
        return constant(left / right)
    return operation(InstructionType.DIV_F32, lhs, rhs)


def _same_toggle(bit_indices: object, choices: Sequence[Expression]) -> Expression:
    if not choices:
        raise AssertionError("toggle derivative has no choices")
    if all(choice == choices[0] for choice in choices[1:]):
        return choices[0]
    return Expression(NodeKind.TOGGLE, bit_indices, tuple(choices))


def _differentiate(
    expression: Expression,
    target: int,
    shape: LMKernelShape,
) -> Expression:
    if expression.kind == NodeKind.SOURCE:
        leaf = expression.value
        if leaf.kind.name != "STATE":
            raise LMAstError("indexed scoring constants are not supported by trajectory LM")
        return constant(1.0 if leaf.index == target else 0.0)
    if expression.kind == NodeKind.OPTIMIZED_CONSTANT:
        input_index = shape.state_count + int(expression.value)
        return constant(1.0 if input_index == target else 0.0)
    if expression.kind == NodeKind.LITERAL:
        return constant(0.0)
    if expression.kind == NodeKind.TOGGLE:
        return _same_toggle(
            expression.value,
            tuple(_differentiate(choice, target, shape) for choice in expression.arguments),
        )
    if expression.kind != NodeKind.OPERATION:
        raise LMAstError("unknown LM expression node")

    kind = expression.value
    arguments = expression.arguments
    derivatives = tuple(_differentiate(value, target, shape) for value in arguments)
    value = expression
    if kind == InstructionType.ADD_F32:
        return _add(derivatives[0], derivatives[1])
    if kind == InstructionType.SUB_F32:
        return _subtract(derivatives[0], derivatives[1])
    if kind == InstructionType.MUL_F32:
        return _add(
            _multiply(derivatives[0], arguments[1]),
            _multiply(arguments[0], derivatives[1]),
        )
    if kind == InstructionType.DIV_F32:
        return _divide(
            _subtract(derivatives[0], _multiply(value, derivatives[1])),
            arguments[1],
        )
    if kind == InstructionType.NEG_F32:
        return _negate(derivatives[0])
    if kind == InstructionType.SQRT_F32:
        return _divide(derivatives[0], _multiply(constant(2.0), value))
    if kind == InstructionType.RCP_F32:
        return _negate(
            _divide(
                derivatives[0],
                _multiply(arguments[0], arguments[0]),
            )
        )
    if kind == InstructionType.FMA_F32:
        return _add(
            _add(
                _multiply(derivatives[0], arguments[1]),
                _multiply(arguments[0], derivatives[1]),
            ),
            derivatives[2],
        )
    if kind == InstructionType.SIN_F32:
        return _multiply(
            derivatives[0],
            operation(InstructionType.COS_F32, arguments[0]),
        )
    if kind == InstructionType.COS_F32:
        return _negate(
            _multiply(
                derivatives[0],
                operation(InstructionType.SIN_F32, arguments[0]),
            )
        )
    if kind == InstructionType.EX2_F32:
        return _multiply(
            _multiply(derivatives[0], constant(math.log(2.0))),
            value,
        )
    if kind == InstructionType.LG2_F32:
        return _divide(
            derivatives[0],
            _multiply(arguments[0], constant(math.log(2.0))),
        )
    if kind == InstructionType.RSQRT_F32:
        return _multiply(
            _multiply(constant(-0.5), derivatives[0]),
            _multiply(value, _multiply(value, value)),
        )
    if kind == InstructionType.TANH_F32:
        return _multiply(
            derivatives[0],
            _subtract(constant(1.0), _multiply(value, value)),
        )
    if kind == InstructionType.EXP_F32:
        return _multiply(derivatives[0], value)
    if kind == InstructionType.LOG_F32:
        return _divide(derivatives[0], arguments[0])
    if kind in {
        InstructionType.ABS_F32,
        InstructionType.MIN_F32,
        InstructionType.MAX_F32,
    }:
        raise LMAstError(
            f"{kind.name} requires an explicit nonsmooth derivative policy before it can be used in trajectory LM"
        )
    raise LMAstError(f"no trajectory-LM derivative rule is defined for {kind.name}")
