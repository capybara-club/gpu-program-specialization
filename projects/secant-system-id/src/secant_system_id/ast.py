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
from enum import IntEnum
import math
import struct
from typing import Iterable, Sequence


MAX_INPUTS = 128
MAX_PROGRAM_INSTRUCTIONS = 1024
MAX_STACK_DEPTH = 128


class InstructionType(IntEnum):
    """Secant's post-order instruction ABI.

    The numeric values intentionally match ``SecantAstInstructionType`` so
    serialized programs can later move to the C99 writer without translation.
    """

    NONE = 0x80
    CONSTANT_BITS_F32 = 0x81
    ROUTINE_F32 = 0x82
    RETURN_F32 = 0x83
    ROUTINE_ARG_F32 = 0x84
    ADD_F32 = 0x85
    SUB_F32 = 0x86
    MUL_F32 = 0x87
    DIV_F32 = 0x88
    NEG_F32 = 0x89
    SQRT_F32 = 0x8A
    RCP_F32 = 0x8B
    ABS_F32 = 0x8C
    MIN_F32 = 0x8D
    MAX_F32 = 0x8E
    FMA_F32 = 0x8F
    SIN_F32 = 0x90
    COS_F32 = 0x91
    EX2_F32 = 0x92
    LG2_F32 = 0x93
    RSQRT_F32 = 0x94
    TANH_F32 = 0x95
    INPUT_S32 = 0x96
    CONSTANT_S32 = 0x97
    ROUTINE_S32 = 0x98
    RETURN_S32 = 0x99
    ROUTINE_ARG_S32 = 0x9A
    ADD_S32 = 0x9B
    SUB_S32 = 0x9C
    MUL_S32 = 0x9D
    DIV_S32 = 0x9E
    NEG_S32 = 0x9F
    ABS_S32 = 0xA0
    MIN_S32 = 0xA1
    MAX_S32 = 0xA2
    FMA_S32 = 0xA3
    INPUT_U32 = 0xA4
    CONSTANT_U32 = 0xA5
    ROUTINE_U32 = 0xA6
    RETURN_U32 = 0xA7
    ROUTINE_ARG_U32 = 0xA8
    ADD_U32 = 0xA9
    SUB_U32 = 0xAA
    MUL_U32 = 0xAB
    DIV_U32 = 0xAC
    MIN_U32 = 0xAD
    MAX_U32 = 0xAE
    FMA_U32 = 0xAF
    AND_U32 = 0xB0
    OR_U32 = 0xB1
    XOR_U32 = 0xB2
    NOT_U32 = 0xB3
    SHL_U32 = 0xB4
    SHR_U32 = 0xB5
    STATIC_COLUMN_INPUT_F32 = 0xB6
    DYNAMIC_COLUMN_INPUT_F32 = 0xB7
    DYNAMIC_CONSTANT_INPUT_F32 = 0xB8
    DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32 = 0xB9
    EXP_F32 = 0xBA
    LOG_F32 = 0xBB


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


class NodeKind(IntEnum):
    INPUT = 0
    CONSTANT = 1
    OPERATION = 2


@dataclass(frozen=True)
class Expression:
    kind: NodeKind
    value: int
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


def operation(kind: InstructionType, *arguments: Expression | float | int) -> Expression:
    if kind not in _ARITY:
        raise ValueError(f"{kind.name} is not an FP32 expression operation")
    if len(arguments) != _ARITY[kind]:
        raise ValueError(f"{kind.name} requires {_ARITY[kind]} arguments")
    return Expression(NodeKind.OPERATION, int(kind), tuple(_expression(arg) for arg in arguments))


def input_slot(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_INPUTS:
        raise ValueError(f"input index must be in [0, {MAX_INPUTS})")
    return Expression(NodeKind.INPUT, index)


def constant(value: float) -> Expression:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return Expression(NodeKind.CONSTANT, bits)


def fma(lhs: Expression | float | int, rhs: Expression | float | int, addend: Expression | float | int) -> Expression:
    return operation(InstructionType.FMA_F32, lhs, rhs, addend)


def absolute(value: Expression | float | int) -> Expression:
    return operation(InstructionType.ABS_F32, value)


def minimum(lhs: Expression | float | int, rhs: Expression | float | int) -> Expression:
    return operation(InstructionType.MIN_F32, lhs, rhs)


def maximum(lhs: Expression | float | int, rhs: Expression | float | int) -> Expression:
    return operation(InstructionType.MAX_F32, lhs, rhs)


@dataclass(frozen=True)
class DecodedInstruction:
    kind: InstructionType
    operand: int | None = None


@dataclass(frozen=True)
class Program:
    data: bytes

    @classmethod
    def from_expression(cls, expression: Expression) -> "Program":
        encoded = bytearray()

        def visit(node: Expression) -> None:
            for argument in node.arguments:
                visit(argument)
            if node.kind == NodeKind.INPUT:
                encoded.extend((InstructionType.STATIC_COLUMN_INPUT_F32, node.value))
            elif node.kind == NodeKind.CONSTANT:
                encoded.append(InstructionType.CONSTANT_BITS_F32)
                encoded.extend(struct.pack("<I", node.value))
            else:
                encoded.append(node.value)

        visit(expression)
        encoded.append(InstructionType.RETURN_F32)
        program = cls(bytes(encoded))
        program.validate()
        return program

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
                InstructionType.STATIC_COLUMN_INPUT_F32,
                InstructionType.DYNAMIC_COLUMN_INPUT_F32,
                InstructionType.DYNAMIC_CONSTANT_INPUT_F32,
                InstructionType.DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
                InstructionType.ROUTINE_F32,
                InstructionType.ROUTINE_ARG_F32,
            }:
                if offset >= len(self.data):
                    raise ValueError("truncated indexed instruction")
                operand = self.data[offset]
                offset += 1
            elif kind == InstructionType.CONSTANT_BITS_F32:
                if offset + 4 > len(self.data):
                    raise ValueError("truncated FP32 constant")
                operand = struct.unpack_from("<I", self.data, offset)[0]
                offset += 4
            else:
                operand = None
            yield DecodedInstruction(kind, operand)

    def validate(self, input_count: int = MAX_INPUTS) -> None:
        depth = 0
        returned = False
        for instruction in self.instructions():
            kind = instruction.kind
            if returned:
                raise ValueError("instructions follow RETURN_F32")
            if kind == InstructionType.STATIC_COLUMN_INPUT_F32:
                if instruction.operand is None or instruction.operand >= input_count:
                    raise ValueError("input index is outside the kernel ABI")
                depth += 1
            elif kind == InstructionType.CONSTANT_BITS_F32:
                depth += 1
            elif kind == InstructionType.RETURN_F32:
                if depth != 1:
                    raise ValueError("RETURN_F32 requires exactly one stack value")
                returned = True
            elif kind in _ARITY:
                arity = _ARITY[kind]
                if depth < arity:
                    raise ValueError(f"stack underflow at {kind.name}")
                depth = depth - arity + 1
            else:
                raise ValueError(f"{kind.name} is not supported in a standalone expression")
            if depth > MAX_STACK_DEPTH:
                raise ValueError("expression exceeds the post-order stack limit")
        if not returned:
            raise ValueError("program does not terminate with RETURN_F32")

    def input_indices(self) -> frozenset[int]:
        return frozenset(
            int(instruction.operand)
            for instruction in self.instructions()
            if instruction.kind == InstructionType.STATIC_COLUMN_INPUT_F32
        )

    def evaluate(self, inputs: Sequence[float]) -> float:
        self.validate(len(inputs))
        return self.evaluate_validated(inputs)

    def evaluate_validated(self, inputs: Sequence[float]) -> float:
        """Evaluate after the caller has already validated this program's ABI."""

        stack: list[float] = []
        for instruction in self.instructions():
            kind = instruction.kind
            if kind == InstructionType.STATIC_COLUMN_INPUT_F32:
                stack.append(float(inputs[instruction.operand]))
            elif kind == InstructionType.CONSTANT_BITS_F32:
                stack.append(struct.unpack("<f", struct.pack("<I", instruction.operand))[0])
            elif kind == InstructionType.RETURN_F32:
                return stack[-1]
            elif kind == InstructionType.NEG_F32:
                stack.append(-stack.pop())
            elif kind == InstructionType.ABS_F32:
                stack.append(abs(stack.pop()))
            elif kind in {InstructionType.ADD_F32, InstructionType.SUB_F32, InstructionType.MUL_F32, InstructionType.DIV_F32,
                          InstructionType.MIN_F32, InstructionType.MAX_F32}:
                rhs = stack.pop()
                lhs = stack.pop()
                if kind == InstructionType.ADD_F32:
                    stack.append(lhs + rhs)
                elif kind == InstructionType.SUB_F32:
                    stack.append(lhs - rhs)
                elif kind == InstructionType.MUL_F32:
                    stack.append(lhs * rhs)
                elif kind == InstructionType.DIV_F32:
                    stack.append(lhs / rhs)
                elif kind == InstructionType.MIN_F32:
                    stack.append(min(lhs, rhs))
                else:
                    stack.append(max(lhs, rhs))
            elif kind == InstructionType.FMA_F32:
                addend = stack.pop()
                rhs = stack.pop()
                lhs = stack.pop()
                stack.append(lhs * rhs + addend)
            else:
                value = stack.pop()
                if kind == InstructionType.SQRT_F32:
                    stack.append(math.sqrt(value))
                elif kind == InstructionType.RCP_F32:
                    stack.append(1.0 / value)
                elif kind == InstructionType.SIN_F32:
                    stack.append(math.sin(value))
                elif kind == InstructionType.COS_F32:
                    stack.append(math.cos(value))
                elif kind == InstructionType.EX2_F32:
                    stack.append(2.0**value)
                elif kind == InstructionType.LG2_F32:
                    stack.append(math.log2(value))
                elif kind == InstructionType.RSQRT_F32:
                    stack.append(1.0 / math.sqrt(value))
                elif kind == InstructionType.TANH_F32:
                    stack.append(math.tanh(value))
                elif kind == InstructionType.EXP_F32:
                    stack.append(math.exp(value))
                elif kind == InstructionType.LOG_F32:
                    stack.append(math.log(value))
                else:
                    raise ValueError(f"evaluation is not implemented for {kind.name}")
        raise AssertionError("validated program did not return")
