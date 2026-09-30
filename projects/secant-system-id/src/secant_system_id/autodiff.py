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

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import sys
from typing import Sequence

from .ast import Expression, InstructionType, NodeKind, Program, constant, input_slot, operation
from .genome import SystemGenome
from .shape import KernelShape


TAPE_MAGIC = b"SSIDAT01"
TAPE_SCHEMA = "secant-system-id.postorder-ad-tape"
TAPE_SCHEMA_VERSION = 1
_HEADER = struct.Struct("<8sIIII")
_NODE = struct.Struct("<BBHIHHHH")
_NO_ARGUMENT = 0xFFFF


class DerivativeTapeError(ValueError):
    pass


_ARITY = {
    InstructionType.ADD_F32: 2,
    InstructionType.SUB_F32: 2,
    InstructionType.MUL_F32: 2,
    InstructionType.DIV_F32: 2,
    InstructionType.NEG_F32: 1,
    InstructionType.SQRT_F32: 1,
    InstructionType.RCP_F32: 1,
    InstructionType.FMA_F32: 3,
    InstructionType.SIN_F32: 1,
    InstructionType.COS_F32: 1,
    InstructionType.EX2_F32: 1,
    InstructionType.LG2_F32: 1,
    InstructionType.RSQRT_F32: 1,
    InstructionType.TANH_F32: 1,
    InstructionType.EXP_F32: 1,
    InstructionType.LOG_F32: 1,
}

_NONSMOOTH = {
    InstructionType.ABS_F32,
    InstructionType.MIN_F32,
    InstructionType.MAX_F32,
}


@dataclass(frozen=True)
class ADNode:
    kind: InstructionType
    operand: int | None
    arguments: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind in {
            InstructionType.STATIC_COLUMN_INPUT_F32,
            InstructionType.CONSTANT_BITS_F32,
        }:
            if self.operand is None or self.arguments:
                raise DerivativeTapeError("leaf AD nodes require one operand and no arguments")
            return
        expected = _ARITY.get(self.kind)
        if expected is None or self.operand is not None or len(self.arguments) != expected:
            raise DerivativeTapeError("operation AD node does not match its derivative rule")


@dataclass(frozen=True)
class ReverseInstruction:
    node: int
    kind: InstructionType
    arguments: tuple[int, ...]


@dataclass(frozen=True)
class PostorderADTape:
    input_count: int
    nodes: tuple[ADNode, ...]
    root: int
    maximum_stack_depth: int

    @classmethod
    def from_program(cls, program: Program, input_count: int) -> "PostorderADTape":
        program.validate(input_count)
        nodes: list[ADNode] = []
        stack: list[int] = []
        maximum_stack_depth = 0
        for instruction in program.instructions():
            kind = instruction.kind
            if kind == InstructionType.RETURN_F32:
                continue
            if kind in _NONSMOOTH:
                raise DerivativeTapeError(
                    f"{kind.name} is not differentiable enough for trajectory LM"
                )
            if kind in {
                InstructionType.STATIC_COLUMN_INPUT_F32,
                InstructionType.CONSTANT_BITS_F32,
            }:
                node = ADNode(kind, int(instruction.operand), ())
            else:
                arity = _ARITY.get(kind)
                if arity is None:
                    raise DerivativeTapeError(f"no AD rule is defined for {kind.name}")
                arguments = tuple(stack[-arity:])
                del stack[-arity:]
                node = ADNode(kind, None, arguments)
            nodes.append(node)
            stack.append(len(nodes) - 1)
            maximum_stack_depth = max(maximum_stack_depth, len(stack))
        if len(stack) != 1:
            raise DerivativeTapeError("postorder AD tape did not produce one root")
        return cls(input_count, tuple(nodes), stack[0], maximum_stack_depth)

    def __post_init__(self) -> None:
        if self.input_count <= 0:
            raise DerivativeTapeError("AD tape input count must be positive")
        if not self.nodes or self.root != len(self.nodes) - 1:
            raise DerivativeTapeError("AD tape root must be its final postorder node")
        if self.maximum_stack_depth <= 0:
            raise DerivativeTapeError("AD tape stack depth must be positive")
        for index, node in enumerate(self.nodes):
            if node.kind == InstructionType.STATIC_COLUMN_INPUT_F32:
                if node.operand is None or not 0 <= node.operand < self.input_count:
                    raise DerivativeTapeError("AD input node is outside the tape ABI")
            for argument in node.arguments:
                if not 0 <= argument < index:
                    raise DerivativeTapeError("AD node arguments must precede their consumer")

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def scratch_floats_per_thread(self) -> int:
        return 2 * self.node_count

    @property
    def scratch_bytes_per_thread(self) -> int:
        return 4 * self.scratch_floats_per_thread

    @property
    def reverse_instructions(self) -> tuple[ReverseInstruction, ...]:
        return tuple(
            ReverseInstruction(index, node.kind, node.arguments)
            for index, node in reversed(tuple(enumerate(self.nodes)))
        )

    def to_bytes(self) -> bytes:
        encoded = bytearray(
            _HEADER.pack(
                TAPE_MAGIC,
                self.input_count,
                self.node_count,
                self.root,
                self.maximum_stack_depth,
            )
        )
        for node in self.nodes:
            arguments = node.arguments + (_NO_ARGUMENT,) * (3 - len(node.arguments))
            encoded.extend(
                _NODE.pack(
                    int(node.kind),
                    len(node.arguments),
                    0,
                    0 if node.operand is None else node.operand,
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    0,
                )
            )
        return bytes(encoded)

    @classmethod
    def from_bytes(cls, encoded: bytes) -> "PostorderADTape":
        if len(encoded) < _HEADER.size:
            raise DerivativeTapeError("truncated AD tape header")
        magic, input_count, node_count, root, maximum_stack_depth = _HEADER.unpack_from(encoded)
        if magic != TAPE_MAGIC:
            raise DerivativeTapeError("unknown AD tape magic or schema version")
        expected = _HEADER.size + node_count * _NODE.size
        if len(encoded) != expected:
            raise DerivativeTapeError("AD tape byte count does not match its header")
        nodes: list[ADNode] = []
        offset = _HEADER.size
        for _index in range(node_count):
            opcode, arity, flags, operand, first, second, third, reserved = _NODE.unpack_from(
                encoded, offset
            )
            offset += _NODE.size
            if flags != 0 or reserved != 0 or arity > 3:
                raise DerivativeTapeError("AD tape contains unsupported record flags")
            try:
                kind = InstructionType(opcode)
            except ValueError as exc:
                raise DerivativeTapeError(f"unknown AD opcode 0x{opcode:02x}") from exc
            raw_arguments = (first, second, third)
            if any(value != _NO_ARGUMENT for value in raw_arguments[arity:]):
                raise DerivativeTapeError("AD tape has nonempty unused arguments")
            node_operand = operand if kind in {
                InstructionType.STATIC_COLUMN_INPUT_F32,
                InstructionType.CONSTANT_BITS_F32,
            } else None
            nodes.append(ADNode(kind, node_operand, raw_arguments[:arity]))
        return cls(input_count, tuple(nodes), root, maximum_stack_depth)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def inspection(self) -> dict[str, object]:
        return {
            "schema": TAPE_SCHEMA,
            "schema_version": TAPE_SCHEMA_VERSION,
            "sha256": self.sha256,
            "input_count": self.input_count,
            "node_count": self.node_count,
            "root": self.root,
            "maximum_stack_depth": self.maximum_stack_depth,
            "scratch": {
                "layout": "thread_column_node_major",
                "value_float_count_per_thread": self.node_count,
                "adjoint_float_count_per_thread": self.node_count,
                "total_bytes_per_thread": self.scratch_bytes_per_thread,
            },
            "postorder": [
                {
                    "node": index,
                    "opcode": node.kind.name,
                    "operand": node.operand,
                    "arguments": list(node.arguments),
                }
                for index, node in enumerate(self.nodes)
            ],
            "reverse": [
                {
                    "node": instruction.node,
                    "rule": instruction.kind.name,
                    "arguments": list(instruction.arguments),
                }
                for instruction in self.reverse_instructions
            ],
        }

    def evaluate(self, inputs: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        if len(inputs) != self.input_count:
            raise DerivativeTapeError("input values do not match the AD tape ABI")
        values = [0.0] * self.node_count
        for index, node in enumerate(self.nodes):
            kind = node.kind
            if kind == InstructionType.STATIC_COLUMN_INPUT_F32:
                values[index] = float(inputs[node.operand])
            elif kind == InstructionType.CONSTANT_BITS_F32:
                values[index] = struct.unpack("<f", struct.pack("<I", node.operand))[0]
            else:
                arguments = [values[value] for value in node.arguments]
                values[index] = _evaluate_operation(kind, arguments)

        adjoints = [0.0] * self.node_count
        partials = [0.0] * self.input_count
        adjoints[self.root] = 1.0
        for instruction in self.reverse_instructions:
            node = self.nodes[instruction.node]
            adjoint = adjoints[instruction.node]
            if node.kind == InstructionType.STATIC_COLUMN_INPUT_F32:
                partials[node.operand] += adjoint
            elif node.kind != InstructionType.CONSTANT_BITS_F32:
                arguments = [values[value] for value in node.arguments]
                contributions = _reverse_operation(node.kind, arguments, adjoint)
                for argument, contribution in zip(node.arguments, contributions):
                    adjoints[argument] += contribution
        return values[self.root], tuple(partials)


@dataclass(frozen=True)
class SystemADPlan:
    tapes: tuple[PostorderADTape, ...]
    scratch_node_capacity: int

    @classmethod
    def from_genome(cls, genome: SystemGenome, shape: KernelShape) -> "SystemADPlan":
        genome.validate(shape)
        tapes = tuple(
            PostorderADTape.from_program(program, shape.input_count)
            for program in genome.programs
        )
        return cls(tapes, max(tape.node_count for tape in tapes))

    @property
    def scratch_floats_per_thread(self) -> int:
        return 2 * self.scratch_node_capacity

    def shared_bytes_per_cta(self, shape: KernelShape, threads_per_block: int) -> int:
        if threads_per_block <= 0:
            raise DerivativeTapeError("threads per block must be positive")
        reference_bytes = 4 * shape.reference_float_count
        bank_bytes = 4 * shape.bank_slot_count * threads_per_block
        scratch_bytes = 4 * self.scratch_floats_per_thread * threads_per_block
        return reference_bytes + bank_bytes + scratch_bytes

    def inspection(self, shape: KernelShape, threads_per_block: int) -> dict[str, object]:
        return {
            "schema": "secant-system-id.system-ad-plan",
            "schema_version": 1,
            "scratch_reuse": "one workspace reused sequentially across missing sites",
            "scratch_node_capacity": self.scratch_node_capacity,
            "scratch_floats_per_thread": self.scratch_floats_per_thread,
            "threads_per_block": threads_per_block,
            "shared_bytes_per_cta": self.shared_bytes_per_cta(shape, threads_per_block),
            "sites": [tape.inspection() for tape in self.tapes],
        }


@dataclass(frozen=True)
class PostorderDerivativeBundle:
    """Inspectable scalar programs lowered from one compact AD tape.

    The tape remains the canonical derivative representation. These programs
    are an initial lowering for the existing scalar SASS writer: one primal
    followed by one partial per requested leaf slot. A later reverse-tape SASS
    backend can consume the same tape without changing the serialized ABI.
    """

    tape: PostorderADTape
    primal: Program
    leaf_inputs: tuple[int, ...]
    partials: tuple[Program, ...]

    @classmethod
    def from_program(
        cls,
        program: Program,
        input_count: int,
        leaf_inputs: Sequence[int],
    ) -> "PostorderDerivativeBundle":
        tape = PostorderADTape.from_program(program, input_count)
        requested = tuple(int(value) for value in leaf_inputs)
        if len(set(requested)) != len(requested) or any(
            not 0 <= value < input_count for value in requested
        ):
            raise DerivativeTapeError("derivative leaf inputs must be unique tape inputs")
        # The derivative lowering is allowed to simplify its own expressions,
        # but the primal is part of the genome ABI and must remain byte-for-byte
        # identical. Reconstructing it through the symbolic helpers rewrites
        # FMA and folds constants, which incorrectly rejects valid GP programs.
        primal = program
        partials = tuple(
            Program.from_expression(_differentiate_expression(tape, target))
            for target in requested
        )
        return cls(tape, primal, requested, partials)

    @property
    def programs(self) -> tuple[Program, ...]:
        return (self.primal,) + self.partials

    @property
    def instruction_counts(self) -> tuple[int, ...]:
        return tuple(sum(1 for _ in program.instructions()) for program in self.programs)

    def inspection(self) -> dict[str, object]:
        return {
            "schema": "secant-system-id.postorder-derivative-bundle",
            "schema_version": 1,
            "canonical_tape_sha256": self.tape.sha256,
            "leaf_inputs": list(self.leaf_inputs),
            "programs": [
                {
                    "kind": "primal" if index == 0 else "leaf_partial",
                    "leaf_input": None if index == 0 else self.leaf_inputs[index - 1],
                    "byte_count": len(program.data),
                    "instruction_count": self.instruction_counts[index],
                    "hex": program.data.hex(),
                    "postorder": [
                        {
                            "opcode": instruction.kind.name,
                            "operand": instruction.operand,
                        }
                        for instruction in program.instructions()
                    ],
                }
                for index, program in enumerate(self.programs)
            ],
        }


@dataclass(frozen=True)
class SystemDerivativeBundle:
    """Canonical AD tapes and their initial scalar lowering for one genome.

    Output order is stable and part of the specialization ABI. Each missing
    site contributes its primal first, followed by partials with respect to
    that site's leaf slots. Global input indices remain visible so repeated or
    dynamically-bound leaves can be accumulated by trajectory sensitivities.
    """

    sites: tuple[PostorderDerivativeBundle, ...]

    @classmethod
    def from_genome(
        cls,
        genome: SystemGenome,
        shape: KernelShape,
    ) -> "SystemDerivativeBundle":
        genome.validate(shape)
        sites = tuple(
            PostorderDerivativeBundle.from_program(
                program,
                shape.input_count,
                range(offset, offset + leaf_count),
            )
            for program, offset, leaf_count in zip(
                genome.programs,
                shape.ast_input_offsets,
                shape.ast_leaf_counts,
            )
        )
        return cls(sites)

    @property
    def programs(self) -> tuple[Program, ...]:
        return tuple(program for site in self.sites for program in site.programs)

    @property
    def output_count(self) -> int:
        return sum(1 + len(site.leaf_inputs) for site in self.sites)

    @property
    def instruction_counts(self) -> tuple[tuple[int, ...], ...]:
        return tuple(site.instruction_counts for site in self.sites)

    def inspection(self) -> dict[str, object]:
        return {
            "schema": "secant-system-id.system-derivative-bundle",
            "schema_version": 1,
            "output_layout": "site-major: primal, then one partial per site leaf",
            "output_count": self.output_count,
            "sites": [site.inspection() for site in self.sites],
        }


def _constant_bits(expression: Expression) -> int | None:
    return expression.value if expression.kind == NodeKind.CONSTANT else None


def _constant_value(expression: Expression) -> float | None:
    bits = _constant_bits(expression)
    return None if bits is None else struct.unpack("<f", struct.pack("<I", bits))[0]


def _is_constant(expression: Expression, expected: float) -> bool:
    value = _constant_value(expression)
    return value is not None and value == expected


def _negate_expression(value: Expression) -> Expression:
    if _is_constant(value, 0.0):
        return value
    if value.kind == NodeKind.OPERATION and value.value == int(InstructionType.NEG_F32):
        return value.arguments[0]
    constant_value = _constant_value(value)
    return constant(-constant_value) if constant_value is not None else operation(InstructionType.NEG_F32, value)


def _add_expression(lhs: Expression, rhs: Expression) -> Expression:
    if _is_constant(lhs, 0.0):
        return rhs
    if _is_constant(rhs, 0.0):
        return lhs
    left_value = _constant_value(lhs)
    right_value = _constant_value(rhs)
    if left_value is not None and right_value is not None:
        return constant(left_value + right_value)
    return operation(InstructionType.ADD_F32, lhs, rhs)


def _subtract_expression(lhs: Expression, rhs: Expression) -> Expression:
    if _is_constant(rhs, 0.0):
        return lhs
    if _is_constant(lhs, 0.0):
        return _negate_expression(rhs)
    left_value = _constant_value(lhs)
    right_value = _constant_value(rhs)
    if left_value is not None and right_value is not None:
        return constant(left_value - right_value)
    return operation(InstructionType.SUB_F32, lhs, rhs)


def _multiply_expression(lhs: Expression, rhs: Expression) -> Expression:
    if _is_constant(lhs, 0.0) or _is_constant(rhs, 0.0):
        return constant(0.0)
    if _is_constant(lhs, 1.0):
        return rhs
    if _is_constant(rhs, 1.0):
        return lhs
    if _is_constant(lhs, -1.0):
        return _negate_expression(rhs)
    if _is_constant(rhs, -1.0):
        return _negate_expression(lhs)
    left_value = _constant_value(lhs)
    right_value = _constant_value(rhs)
    if left_value is not None and right_value is not None:
        return constant(left_value * right_value)
    return operation(InstructionType.MUL_F32, lhs, rhs)


def _divide_expression(lhs: Expression, rhs: Expression) -> Expression:
    if _is_constant(lhs, 0.0):
        return constant(0.0)
    if _is_constant(rhs, 1.0):
        return lhs
    left_value = _constant_value(lhs)
    right_value = _constant_value(rhs)
    if left_value is not None and right_value is not None:
        return constant(left_value / right_value)
    return operation(InstructionType.DIV_F32, lhs, rhs)


def _apply_expression(kind: InstructionType, arguments: Sequence[Expression]) -> Expression:
    if kind == InstructionType.ADD_F32:
        return _add_expression(arguments[0], arguments[1])
    if kind == InstructionType.SUB_F32:
        return _subtract_expression(arguments[0], arguments[1])
    if kind == InstructionType.MUL_F32:
        return _multiply_expression(arguments[0], arguments[1])
    if kind == InstructionType.DIV_F32:
        return _divide_expression(arguments[0], arguments[1])
    if kind == InstructionType.NEG_F32:
        return _negate_expression(arguments[0])
    if kind == InstructionType.FMA_F32:
        return _add_expression(
            _multiply_expression(arguments[0], arguments[1]), arguments[2]
        )
    return operation(kind, *arguments)


def _tape_expressions(tape: PostorderADTape) -> list[Expression]:
    expressions: list[Expression] = []
    for node in tape.nodes:
        if node.kind == InstructionType.STATIC_COLUMN_INPUT_F32:
            expression = input_slot(node.operand)
        elif node.kind == InstructionType.CONSTANT_BITS_F32:
            expression = Expression(NodeKind.CONSTANT, node.operand)
        else:
            expression = _apply_expression(
                node.kind, [expressions[index] for index in node.arguments]
            )
        expressions.append(expression)
    return expressions


def _tape_expression(tape: PostorderADTape) -> Expression:
    return _tape_expressions(tape)[tape.root]


def _differentiate_expression(tape: PostorderADTape, target: int) -> Expression:
    values = _tape_expressions(tape)
    derivatives: list[Expression] = []
    for node, value in zip(tape.nodes, values):
        if node.kind == InstructionType.STATIC_COLUMN_INPUT_F32:
            derivative = constant(1.0 if node.operand == target else 0.0)
        elif node.kind == InstructionType.CONSTANT_BITS_F32:
            derivative = constant(0.0)
        else:
            arguments = [values[index] for index in node.arguments]
            argument_derivatives = [derivatives[index] for index in node.arguments]
            kind = node.kind
            if kind == InstructionType.ADD_F32:
                derivative = _add_expression(argument_derivatives[0], argument_derivatives[1])
            elif kind == InstructionType.SUB_F32:
                derivative = _subtract_expression(argument_derivatives[0], argument_derivatives[1])
            elif kind == InstructionType.MUL_F32:
                derivative = _add_expression(
                    _multiply_expression(argument_derivatives[0], arguments[1]),
                    _multiply_expression(arguments[0], argument_derivatives[1]),
                )
            elif kind == InstructionType.DIV_F32:
                # Express the quotient rule through the already-computed
                # primal value. Besides being algebraically equivalent to
                # (du*v-u*dv)/(v*v), this form lets the multi-output SASS
                # lowering share the quotient and denominator across every
                # leaf partial.
                derivative = _divide_expression(
                    _subtract_expression(
                        argument_derivatives[0],
                        _multiply_expression(value, argument_derivatives[1]),
                    ),
                    arguments[1],
                )
            elif kind == InstructionType.NEG_F32:
                derivative = _negate_expression(argument_derivatives[0])
            elif kind == InstructionType.SQRT_F32:
                derivative = _divide_expression(
                    argument_derivatives[0],
                    _multiply_expression(constant(2.0), value),
                )
            elif kind == InstructionType.RCP_F32:
                derivative = _negate_expression(
                    _divide_expression(
                        argument_derivatives[0],
                        _multiply_expression(arguments[0], arguments[0]),
                    )
                )
            elif kind == InstructionType.FMA_F32:
                derivative = _add_expression(
                    _add_expression(
                        _multiply_expression(argument_derivatives[0], arguments[1]),
                        _multiply_expression(arguments[0], argument_derivatives[1]),
                    ),
                    argument_derivatives[2],
                )
            elif kind == InstructionType.SIN_F32:
                derivative = _multiply_expression(
                    argument_derivatives[0],
                    operation(InstructionType.COS_F32, arguments[0]),
                )
            elif kind == InstructionType.COS_F32:
                derivative = _negate_expression(
                    _multiply_expression(
                        argument_derivatives[0],
                        operation(InstructionType.SIN_F32, arguments[0]),
                    )
                )
            elif kind == InstructionType.EX2_F32:
                derivative = _multiply_expression(
                    _multiply_expression(argument_derivatives[0], constant(math.log(2.0))),
                    value,
                )
            elif kind == InstructionType.LG2_F32:
                derivative = _divide_expression(
                    argument_derivatives[0],
                    _multiply_expression(arguments[0], constant(math.log(2.0))),
                )
            elif kind == InstructionType.RSQRT_F32:
                derivative = _multiply_expression(
                    _multiply_expression(constant(-0.5), argument_derivatives[0]),
                    _multiply_expression(value, _multiply_expression(value, value)),
                )
            elif kind == InstructionType.TANH_F32:
                derivative = _multiply_expression(
                    argument_derivatives[0],
                    _subtract_expression(constant(1.0), _multiply_expression(value, value)),
                )
            elif kind == InstructionType.EXP_F32:
                derivative = _multiply_expression(argument_derivatives[0], value)
            elif kind == InstructionType.LOG_F32:
                derivative = _divide_expression(argument_derivatives[0], arguments[0])
            else:
                raise DerivativeTapeError(f"symbolic lowering is missing for {kind.name}")
        derivatives.append(derivative)
    return derivatives[tape.root]


def _evaluate_operation(kind: InstructionType, arguments: Sequence[float]) -> float:
    if kind == InstructionType.ADD_F32:
        return arguments[0] + arguments[1]
    if kind == InstructionType.SUB_F32:
        return arguments[0] - arguments[1]
    if kind == InstructionType.MUL_F32:
        return arguments[0] * arguments[1]
    if kind == InstructionType.DIV_F32:
        return arguments[0] / arguments[1]
    if kind == InstructionType.NEG_F32:
        return -arguments[0]
    if kind == InstructionType.SQRT_F32:
        return math.sqrt(arguments[0])
    if kind == InstructionType.RCP_F32:
        return 1.0 / arguments[0]
    if kind == InstructionType.FMA_F32:
        return arguments[0] * arguments[1] + arguments[2]
    if kind == InstructionType.SIN_F32:
        return math.sin(arguments[0])
    if kind == InstructionType.COS_F32:
        return math.cos(arguments[0])
    if kind == InstructionType.EX2_F32:
        return 2.0 ** arguments[0]
    if kind == InstructionType.LG2_F32:
        return math.log2(arguments[0])
    if kind == InstructionType.RSQRT_F32:
        return 1.0 / math.sqrt(arguments[0])
    if kind == InstructionType.TANH_F32:
        return math.tanh(arguments[0])
    if kind == InstructionType.EXP_F32:
        return math.exp(arguments[0])
    if kind == InstructionType.LOG_F32:
        return math.log(arguments[0])
    raise DerivativeTapeError(f"evaluation rule is missing for {kind.name}")


def _reverse_operation(
    kind: InstructionType,
    arguments: Sequence[float],
    adjoint: float,
) -> tuple[float, ...]:
    if kind == InstructionType.ADD_F32:
        return adjoint, adjoint
    if kind == InstructionType.SUB_F32:
        return adjoint, -adjoint
    if kind == InstructionType.MUL_F32:
        return adjoint * arguments[1], adjoint * arguments[0]
    if kind == InstructionType.DIV_F32:
        inverse = 1.0 / arguments[1]
        return adjoint * inverse, -adjoint * arguments[0] * inverse * inverse
    if kind == InstructionType.NEG_F32:
        return (-adjoint,)
    if kind == InstructionType.SQRT_F32:
        return (0.5 * adjoint / math.sqrt(arguments[0]),)
    if kind == InstructionType.RCP_F32:
        inverse = 1.0 / arguments[0]
        return (-adjoint * inverse * inverse,)
    if kind == InstructionType.FMA_F32:
        return adjoint * arguments[1], adjoint * arguments[0], adjoint
    if kind == InstructionType.SIN_F32:
        return (adjoint * math.cos(arguments[0]),)
    if kind == InstructionType.COS_F32:
        return (-adjoint * math.sin(arguments[0]),)
    if kind == InstructionType.EX2_F32:
        return (adjoint * math.log(2.0) * (2.0 ** arguments[0]),)
    if kind == InstructionType.LG2_F32:
        return (adjoint / (arguments[0] * math.log(2.0)),)
    if kind == InstructionType.RSQRT_F32:
        inverse_root = 1.0 / math.sqrt(arguments[0])
        return (-0.5 * adjoint * inverse_root * inverse_root * inverse_root,)
    if kind == InstructionType.TANH_F32:
        value = math.tanh(arguments[0])
        return (adjoint * (1.0 - value * value),)
    if kind == InstructionType.EXP_F32:
        return (adjoint * math.exp(arguments[0]),)
    if kind == InstructionType.LOG_F32:
        return (adjoint / arguments[0],)
    raise DerivativeTapeError(f"reverse rule is missing for {kind.name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a Secant postorder program as a versioned reverse-AD tape."
    )
    parser.add_argument(
        "program_hex",
        help="postorder program bytes as hexadecimal, or '-' to read hexadecimal from stdin",
    )
    parser.add_argument("--input-count", type=int, required=True)
    parser.add_argument(
        "--tape-output",
        help="optional explicit path for the binary tape; no files are written by default",
    )
    arguments = parser.parse_args(argv)
    source = sys.stdin.read() if arguments.program_hex == "-" else arguments.program_hex
    try:
        program = Program(bytes.fromhex("".join(source.split())))
        tape = PostorderADTape.from_program(program, arguments.input_count)
    except (ValueError, DerivativeTapeError) as exc:
        parser.error(str(exc))
    if arguments.tape_output:
        with open(arguments.tape_output, "wb") as output:
            output.write(tape.to_bytes())
    json.dump(tape.inspection(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
