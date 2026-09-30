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
"""Python postorder-program reference model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Sequence

from .model import MAX_TOGGLE_BITS, KernelShape


MAX_PROGRAM_INSTRUCTIONS = 2048
MAX_STACK_DEPTH = 128
SYSTEM_GROUP_SCHEMA = "odezza.system-group"
SYSTEM_GROUP_SCHEMA_VERSION = 2


class InstructionType(IntEnum):
    RETURN_F32 = 0x80
    STATE_F32 = 0x81
    CONSTANT_F32 = 0x82
    LITERAL_F32 = 0x83
    TOGGLE2_F32 = 0x84
    TOGGLE4_F32 = 0x85
    OPTIMIZED_CONSTANT_F32 = 0x86
    ADD_F32 = 0x90
    SUB_F32 = 0x91
    MUL_F32 = 0x92
    DIV_F32 = 0x93
    NEG_F32 = 0x94
    SQRT_F32 = 0x95
    RCP_F32 = 0x96
    ABS_F32 = 0x97
    MIN_F32 = 0x98
    MAX_F32 = 0x99
    FMA_F32 = 0x9A
    SIN_F32 = 0x9B
    COS_F32 = 0x9C
    EX2_F32 = 0x9D
    LG2_F32 = 0x9E
    RSQRT_F32 = 0x9F
    TANH_F32 = 0xA0
    EXP_F32 = 0xA1
    LOG_F32 = 0xA2


_ARITY: dict[InstructionType, int] = {
    InstructionType.ADD_F32: 2,
    InstructionType.SUB_F32: 2,
    InstructionType.MUL_F32: 2,
    InstructionType.DIV_F32: 2,
    InstructionType.NEG_F32: 1,
    InstructionType.SQRT_F32: 1,
    InstructionType.RCP_F32: 1,
    InstructionType.ABS_F32: 1,
    InstructionType.MIN_F32: 2,
    InstructionType.MAX_F32: 2,
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


class LeafKind(IntEnum):
    STATE = 0
    CONSTANT = 1


@dataclass(frozen=True)
class LeafSource:
    kind: LeafKind
    index: int

    def encode(self, shape: KernelShape) -> int:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("leaf index must be a non-negative integer")
        if self.kind == LeafKind.STATE:
            if self.index >= shape.state_count:
                raise ValueError("state leaf is outside the state vector")
            return self.index
        if self.kind == LeafKind.CONSTANT:
            if self.index >= shape.constant_count:
                raise ValueError("constant leaf is outside the constant bank")
            return shape.state_count + self.index
        raise ValueError("unknown leaf kind")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.name.lower(), "index": self.index}


def state_source(index: int) -> LeafSource:
    return LeafSource(LeafKind.STATE, index)


def constant_source(index: int) -> LeafSource:
    return LeafSource(LeafKind.CONSTANT, index)


def decode_source(encoded: int, shape: KernelShape) -> LeafSource:
    if isinstance(encoded, bool) or not isinstance(encoded, int) or not 0 <= encoded < shape.source_count:
        raise ValueError("encoded leaf is outside the state/constant register bank")
    if encoded < shape.state_count:
        return state_source(encoded)
    return constant_source(encoded - shape.state_count)


class NodeKind(IntEnum):
    SOURCE = 0
    TOGGLE = 1
    LITERAL = 2
    OPERATION = 3
    OPTIMIZED_CONSTANT = 4


@dataclass(frozen=True)
class Expression:
    kind: NodeKind
    value: object
    arguments: tuple["Expression", ...] = ()

    def __add__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.ADD_F32, self, other)

    def __radd__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.ADD_F32, other, self)

    def __sub__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.SUB_F32, self, other)

    def __rsub__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.SUB_F32, other, self)

    def __mul__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.MUL_F32, self, other)

    def __rmul__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.MUL_F32, other, self)

    def __truediv__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.DIV_F32, self, other)

    def __rtruediv__(self, other: "Expression | float | int") -> "Expression":
        return operation(InstructionType.DIV_F32, other, self)

    def __neg__(self) -> "Expression":
        return operation(InstructionType.NEG_F32, self)


def _expression(value: Expression | float | int) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("expression operands must be expressions or real numbers")
    return constant(float(value))


def source(value: LeafSource) -> Expression:
    if not isinstance(value, LeafSource):
        raise TypeError("source requires a LeafSource")
    return Expression(NodeKind.SOURCE, value)


def optimized_constant(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index > 255:
        raise ValueError("optimized-constant index must fit in one byte")
    return Expression(NodeKind.OPTIMIZED_CONSTANT, index)


def toggle2(
    bit_index: int,
    first: Expression | LeafSource | float | int,
    second: Expression | LeafSource | float | int,
) -> Expression:
    return _toggle((bit_index,), (first, second))


def toggle4(
    first_bit_index: int,
    second_bit_index: int,
    first: Expression | LeafSource | float | int,
    second: Expression | LeafSource | float | int,
    third: Expression | LeafSource | float | int,
    fourth: Expression | LeafSource | float | int,
) -> Expression:
    return _toggle((first_bit_index, second_bit_index), (first, second, third, fourth))


def _toggle_choice(value: Expression | LeafSource | float | int) -> Expression:
    if isinstance(value, LeafSource):
        return source(value)
    result = _expression(value)
    if result.kind not in {NodeKind.SOURCE, NodeKind.LITERAL, NodeKind.OPTIMIZED_CONSTANT}:
        raise TypeError("toggle choices must be direct state, constant, optimized-constant, or literal leaves")
    return result


def _toggle(bit_indices: tuple[int, ...], choices: tuple[Expression | LeafSource | float | int, ...]) -> Expression:
    if len(bit_indices) not in {1, 2} or len(choices) != 1 << len(bit_indices):
        raise ValueError("a toggle must have two or four choices")
    if any(isinstance(bit, bool) or not isinstance(bit, int) or not 0 <= bit < MAX_TOGGLE_BITS for bit in bit_indices):
        raise ValueError("toggle bit index exceeds the 32-bit permutation ABI")
    if len(set(bit_indices)) != len(bit_indices):
        raise ValueError("multiway toggles require distinct bit indices")
    return Expression(NodeKind.TOGGLE, bit_indices, tuple(_toggle_choice(choice) for choice in choices))


def constant(value: float | int) -> Expression:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("literal constants must be finite")
    return Expression(NodeKind.LITERAL, converted)


def operation(kind: InstructionType, *arguments: Expression | float | int) -> Expression:
    if kind not in _ARITY:
        raise ValueError(f"{kind.name} is not an FP32 expression operation")
    if len(arguments) != _ARITY[kind]:
        raise ValueError(f"{kind.name} requires {_ARITY[kind]} arguments")
    return Expression(NodeKind.OPERATION, kind, tuple(_expression(value) for value in arguments))


def absolute(value: Expression | float | int) -> Expression:
    return operation(InstructionType.ABS_F32, value)


def minimum(lhs: Expression | float | int, rhs: Expression | float | int) -> Expression:
    return operation(InstructionType.MIN_F32, lhs, rhs)


def maximum(lhs: Expression | float | int, rhs: Expression | float | int) -> Expression:
    return operation(InstructionType.MAX_F32, lhs, rhs)


def fma(lhs: Expression | float | int, rhs: Expression | float | int, addend: Expression | float | int) -> Expression:
    return operation(InstructionType.FMA_F32, lhs, rhs, addend)


def square_root(value: Expression | float | int) -> Expression:
    return operation(InstructionType.SQRT_F32, value)


def sine(value: Expression | float | int) -> Expression:
    return operation(InstructionType.SIN_F32, value)


def cosine(value: Expression | float | int) -> Expression:
    return operation(InstructionType.COS_F32, value)


def exponential(value: Expression | float | int) -> Expression:
    return operation(InstructionType.EXP_F32, value)


def logarithm(value: Expression | float | int) -> Expression:
    return operation(InstructionType.LOG_F32, value)


@dataclass(frozen=True)
class DecodedInstruction:
    kind: InstructionType
    operands: tuple[int, ...] = ()


@dataclass(frozen=True)
class Program:
    data: bytes

    @classmethod
    def from_expression(
        cls,
        expression: Expression,
        shape: KernelShape,
        optimized_constant_count: int = 0,
    ) -> "Program":
        encoded = bytearray()

        def visit(node: Expression) -> None:
            for argument in node.arguments:
                visit(argument)
            if node.kind == NodeKind.SOURCE:
                leaf = node.value
                leaf.encode(shape)
                opcode = InstructionType.STATE_F32 if leaf.kind == LeafKind.STATE else InstructionType.CONSTANT_F32
                encoded.extend((opcode, leaf.index))
            elif node.kind == NodeKind.OPTIMIZED_CONSTANT:
                encoded.extend((InstructionType.OPTIMIZED_CONSTANT_F32, int(node.value)))
            elif node.kind == NodeKind.TOGGLE:
                opcode = {
                    2: InstructionType.TOGGLE2_F32,
                    4: InstructionType.TOGGLE4_F32,
                }[len(node.arguments)]
                encoded.append(opcode)
                encoded.extend(node.value)
            elif node.kind == NodeKind.LITERAL:
                encoded.append(InstructionType.LITERAL_F32)
                encoded.extend(struct.pack("<f", float(node.value)))
            elif node.kind == NodeKind.OPERATION:
                encoded.append(int(node.value))
            else:
                raise ValueError("unknown expression node")

        visit(expression)
        encoded.append(InstructionType.RETURN_F32)
        result = cls(bytes(encoded))
        result.validate(shape, optimized_constant_count)
        return result

    def instructions(self) -> Iterable[DecodedInstruction]:
        offset = 0
        count = 0
        while offset < len(self.data):
            count += 1
            if count > MAX_PROGRAM_INSTRUCTIONS:
                raise ValueError("program contains too many instructions")
            try:
                kind = InstructionType(self.data[offset])
            except ValueError as exc:
                raise ValueError(f"unknown opcode 0x{self.data[offset]:02x}") from exc
            offset += 1
            if kind in {
                InstructionType.STATE_F32,
                InstructionType.CONSTANT_F32,
                InstructionType.OPTIMIZED_CONSTANT_F32,
                InstructionType.TOGGLE2_F32,
            }:
                width = 1
            elif kind == InstructionType.TOGGLE4_F32:
                width = 2
            elif kind == InstructionType.LITERAL_F32:
                width = 4
            else:
                width = 0
            if offset + width > len(self.data):
                raise ValueError(f"truncated {kind.name} instruction")
            if kind == InstructionType.LITERAL_F32:
                operands = (struct.unpack_from("<I", self.data, offset)[0],)
            else:
                operands = tuple(self.data[offset : offset + width])
            offset += width
            yield DecodedInstruction(kind, operands)

    def validate(self, shape: KernelShape, optimized_constant_count: int = 0) -> None:
        if isinstance(optimized_constant_count, bool) or not isinstance(optimized_constant_count, int) or optimized_constant_count < 0 or optimized_constant_count > 256:
            raise ValueError("optimized-constant count must be between zero and 256")
        stack: list[bool] = []
        returned = False
        for instruction in self.instructions():
            if returned:
                raise ValueError("instructions follow RETURN_F32")
            kind = instruction.kind
            if kind == InstructionType.STATE_F32:
                if instruction.operands[0] >= shape.state_count:
                    raise ValueError("state leaf is outside the state vector")
                stack.append(True)
            elif kind == InstructionType.CONSTANT_F32:
                if instruction.operands[0] >= shape.constant_count:
                    raise ValueError("constant leaf is outside the constant bank")
                stack.append(True)
            elif kind == InstructionType.OPTIMIZED_CONSTANT_F32:
                if instruction.operands[0] >= optimized_constant_count:
                    raise ValueError("optimized-constant leaf is outside the LM parameter vector")
                stack.append(True)
            elif kind in {InstructionType.TOGGLE2_F32, InstructionType.TOGGLE4_F32}:
                arity = {
                    InstructionType.TOGGLE2_F32: 2,
                    InstructionType.TOGGLE4_F32: 4,
                }[kind]
                bits = instruction.operands
                if any(bit >= MAX_TOGGLE_BITS for bit in bits):
                    raise ValueError("toggle bit index exceeds the 32-bit permutation ABI")
                if len(set(bits)) != len(bits):
                    raise ValueError("multiway toggles require distinct bit indices")
                if len(stack) < arity:
                    raise ValueError(f"stack underflow at {kind.name}")
                if not all(stack[-arity:]):
                    raise ValueError("toggle choices must be direct state, constant, optimized-constant, or literal leaves")
                del stack[-arity:]
                stack.append(False)
            elif kind == InstructionType.LITERAL_F32:
                value = struct.unpack("<f", struct.pack("<I", instruction.operands[0]))[0]
                if not math.isfinite(value):
                    raise ValueError("literal constants must be finite")
                stack.append(True)
            elif kind == InstructionType.RETURN_F32:
                if len(stack) != 1:
                    raise ValueError("RETURN_F32 requires exactly one stack value")
                returned = True
            elif kind in _ARITY:
                arity = _ARITY[kind]
                if len(stack) < arity:
                    raise ValueError(f"stack underflow at {kind.name}")
                del stack[-arity:]
                stack.append(False)
            else:
                raise ValueError(f"{kind.name} is not valid in an ODE program")
            if len(stack) > MAX_STACK_DEPTH:
                raise ValueError("program exceeds the postorder stack limit")
        if not returned:
            raise ValueError("program does not terminate with RETURN_F32")

    @property
    def required_toggle_bits(self) -> int:
        required = 0
        for instruction in self.instructions():
            if instruction.kind in {InstructionType.TOGGLE2_F32, InstructionType.TOGGLE4_F32}:
                required = max(required, *(bit + 1 for bit in instruction.operands))
        return required

    def evaluate(
        self,
        states: Sequence[float],
        constants: Sequence[float],
        permutation: int,
        shape: KernelShape,
        optimized_constants: Sequence[float] = (),
    ) -> float:
        self.validate(shape, len(optimized_constants))
        if len(states) != shape.state_count or len(constants) != shape.constant_count:
            raise ValueError("state or constant vector does not match the kernel shape")
        if not 0 <= permutation < 1 << MAX_TOGGLE_BITS:
            raise ValueError("toggle permutation is outside the 32-bit ABI")
        stack: list[float] = []
        for instruction in self.instructions():
            kind = instruction.kind
            if kind == InstructionType.STATE_F32:
                stack.append(float(states[instruction.operands[0]]))
            elif kind == InstructionType.CONSTANT_F32:
                stack.append(float(constants[instruction.operands[0]]))
            elif kind == InstructionType.OPTIMIZED_CONSTANT_F32:
                stack.append(float(optimized_constants[instruction.operands[0]]))
            elif kind == InstructionType.TOGGLE2_F32:
                choices = stack[-2:]
                del stack[-2:]
                stack.append(choices[(permutation >> instruction.operands[0]) & 1])
            elif kind == InstructionType.TOGGLE4_F32:
                choices = stack[-4:]
                del stack[-4:]
                index = ((permutation >> instruction.operands[0]) & 1) | (((permutation >> instruction.operands[1]) & 1) << 1)
                stack.append(choices[index])
            elif kind == InstructionType.LITERAL_F32:
                stack.append(struct.unpack("<f", struct.pack("<I", instruction.operands[0]))[0])
            elif kind == InstructionType.RETURN_F32:
                return stack[-1]
            else:
                _evaluate_operation(stack, kind)
        raise AssertionError("validated program did not return")

    def decoded(self, shape: KernelShape, optimized_constant_count: int = 0) -> list[dict[str, Any]]:
        self.validate(shape, optimized_constant_count)
        result: list[dict[str, Any]] = []
        for instruction in self.instructions():
            record: dict[str, Any] = {"opcode": instruction.kind.name}
            if instruction.kind == InstructionType.STATE_F32:
                record["source"] = state_source(instruction.operands[0]).to_dict()
            elif instruction.kind == InstructionType.CONSTANT_F32:
                record["source"] = constant_source(instruction.operands[0]).to_dict()
            elif instruction.kind == InstructionType.OPTIMIZED_CONSTANT_F32:
                record["optimized_constant_index"] = instruction.operands[0]
            elif instruction.kind in {InstructionType.TOGGLE2_F32, InstructionType.TOGGLE4_F32}:
                record["bit_indices"] = list(instruction.operands)
            elif instruction.kind == InstructionType.LITERAL_F32:
                record["value"] = struct.unpack("<f", struct.pack("<I", instruction.operands[0]))[0]
            result.append(record)
        return result


@dataclass(frozen=True)
class OutputPrograms:
    """Postorder programs paired with the derivative registers they populate."""

    output_indices: tuple[int, ...]
    programs: tuple[Program, ...]

    def validate(self, shape: KernelShape, *, allow_empty: bool = True) -> None:
        if len(self.output_indices) != len(self.programs):
            raise ValueError("output indices and programs must have identical lengths")
        if not allow_empty and not self.programs:
            raise ValueError("at least one output program is required")
        if len(set(self.output_indices)) != len(self.output_indices):
            raise ValueError("an output derivative is assigned more than once")
        for index in self.output_indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < shape.output_count:
                raise ValueError("output derivative index is outside the state vector")
        for program in self.programs:
            program.validate(shape)

    @property
    def required_toggle_bits(self) -> int:
        return max((program.required_toggle_bits for program in self.programs), default=0)

    def records(self) -> list[dict[str, Any]]:
        return [
            {"output_index": output_index, "program": program.data.hex()}
            for output_index, program in zip(self.output_indices, self.programs)
        ]

    @classmethod
    def from_records(cls, records: object) -> "OutputPrograms":
        if not isinstance(records, list):
            raise ValueError("output programs must be a JSON array")
        indices: list[int] = []
        programs: list[Program] = []
        try:
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("each output program must be a JSON object")
                indices.append(int(record["output_index"]))
                programs.append(Program(bytes.fromhex(record["program"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"output-program bytes are invalid: {exc}") from exc
        return cls(tuple(indices), tuple(programs))


@dataclass(frozen=True)
class SystemGroup:
    """One common derivative prelude and several dispatched candidate systems.

    Shared programs populate derivatives that are identical for every candidate.
    Each branch must populate exactly the complementary derivative set. Together,
    the shared prelude and any one branch always produce one complete RHS vector.
    """

    shared: OutputPrograms
    systems: tuple[OutputPrograms, ...]

    def validate(self, shape: KernelShape) -> None:
        self.shared.validate(shape)
        if self.shared.required_toggle_bits:
            raise ValueError("shared fixed RHS programs may not contain toggles")
        if not self.systems:
            raise ValueError("a system group requires at least one candidate system")
        if len(self.systems) > shape.system_capacity:
            raise ValueError(
                f"system group has {len(self.systems)} systems; template capacity is {shape.system_capacity}"
            )
        shared_indices = set(self.shared.output_indices)
        required_branch_indices = set(range(shape.output_count)) - shared_indices
        for system in self.systems:
            system.validate(shape, allow_empty=not required_branch_indices)
            branch_indices = set(system.output_indices)
            if shared_indices & branch_indices:
                raise ValueError("a candidate branch overwrites a shared derivative")
            if branch_indices != required_branch_indices:
                raise ValueError(
                    "each candidate branch must populate exactly the derivatives omitted by the shared prelude"
                )

    @property
    def required_toggle_bits(self) -> int:
        return max(
            self.shared.required_toggle_bits,
            *(system.required_toggle_bits for system in self.systems),
        )

    def programs(self) -> Iterable[Program]:
        yield from self.shared.programs
        for system in self.systems:
            yield from system.programs

    def to_document(self, shape: KernelShape) -> dict[str, Any]:
        self.validate(shape)
        semantic = {
            "schema": SYSTEM_GROUP_SCHEMA,
            "schema_version": SYSTEM_GROUP_SCHEMA_VERSION,
            "required_toggle_bits": self.required_toggle_bits,
            "shared": self.shared.records(),
            "systems": [system.records() for system in self.systems],
        }
        digest = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            **semantic,
            "group_id": digest,
            "decoded": {
                "shared": self._decoded(self.shared, shape),
                "systems": [self._decoded(system, shape) for system in self.systems],
            },
        }

    @staticmethod
    def _decoded(outputs: OutputPrograms, shape: KernelShape) -> list[dict[str, Any]]:
        return [
            {"output_index": index, "instructions": program.decoded(shape)}
            for index, program in zip(outputs.output_indices, outputs.programs)
        ]

    @classmethod
    def from_document(cls, value: dict[str, Any], shape: KernelShape) -> "SystemGroup":
        if value.get("schema") != SYSTEM_GROUP_SCHEMA or value.get("schema_version") != SYSTEM_GROUP_SCHEMA_VERSION:
            raise ValueError("unsupported system-group document")
        systems_value = value.get("systems")
        if not isinstance(systems_value, list):
            raise ValueError("system-group document must contain a systems array")
        result = cls(
            OutputPrograms.from_records(value.get("shared")),
            tuple(OutputPrograms.from_records(system) for system in systems_value),
        )
        result.validate(shape)
        semantic = {
            "schema": SYSTEM_GROUP_SCHEMA,
            "schema_version": SYSTEM_GROUP_SCHEMA_VERSION,
            "required_toggle_bits": result.required_toggle_bits,
            "shared": result.shared.records(),
            "systems": [system.records() for system in result.systems],
        }
        expected = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if value.get("required_toggle_bits") != result.required_toggle_bits:
            raise ValueError("system-group toggle-bit declaration does not match its programs")
        if value.get("group_id") != expected:
            raise ValueError("system-group identity does not match its encoded programs")
        return result


def _source_value(encoded: int, states: Sequence[float], constants: Sequence[float], shape: KernelShape) -> float:
    value = decode_source(encoded, shape)
    return float(states[value.index] if value.kind == LeafKind.STATE else constants[value.index])


def _evaluate_operation(stack: list[float], kind: InstructionType) -> None:
    arity = _ARITY[kind]
    arguments = stack[-arity:]
    del stack[-arity:]
    if kind == InstructionType.ADD_F32:
        result = arguments[0] + arguments[1]
    elif kind == InstructionType.SUB_F32:
        result = arguments[0] - arguments[1]
    elif kind == InstructionType.MUL_F32:
        result = arguments[0] * arguments[1]
    elif kind == InstructionType.DIV_F32:
        result = arguments[0] / arguments[1]
    elif kind == InstructionType.NEG_F32:
        result = -arguments[0]
    elif kind == InstructionType.SQRT_F32:
        result = math.sqrt(arguments[0])
    elif kind == InstructionType.RCP_F32:
        result = 1.0 / arguments[0]
    elif kind == InstructionType.ABS_F32:
        result = abs(arguments[0])
    elif kind == InstructionType.MIN_F32:
        result = min(arguments)
    elif kind == InstructionType.MAX_F32:
        result = max(arguments)
    elif kind == InstructionType.FMA_F32:
        result = arguments[0] * arguments[1] + arguments[2]
    elif kind == InstructionType.SIN_F32:
        result = math.sin(arguments[0])
    elif kind == InstructionType.COS_F32:
        result = math.cos(arguments[0])
    elif kind == InstructionType.EX2_F32:
        result = 2.0 ** arguments[0]
    elif kind == InstructionType.LG2_F32:
        result = math.log2(arguments[0])
    elif kind == InstructionType.RSQRT_F32:
        result = 1.0 / math.sqrt(arguments[0])
    elif kind == InstructionType.TANH_F32:
        result = math.tanh(arguments[0])
    elif kind == InstructionType.EXP_F32:
        result = math.exp(arguments[0])
    elif kind == InstructionType.LOG_F32:
        result = math.log(arguments[0])
    else:
        raise ValueError(f"evaluation is not implemented for {kind.name}")
    stack.append(result)
