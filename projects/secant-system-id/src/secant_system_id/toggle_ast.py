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

from .ast import InstructionType, Program, constant, input_slot, operation
from .bindings import LeafKind, LeafSource, decode_leaf
from .genome import SystemGenome
from .shape import KernelShape


MAX_TOGGLE_BITS = 31


class ToggleInstructionType(IntEnum):
    """CUDA-prototype leaf instructions outside the established Secant ABI."""

    SOURCE_INPUT_F32 = 0xBC
    TOGGLE_1BIT_INPUT_F32 = 0xBD
    TOGGLE_2BIT_INPUT_F32 = 0xBE


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


class ToggleNodeKind(IntEnum):
    SOURCE = 0
    TOGGLE = 1
    LITERAL = 2
    OPERATION = 3


@dataclass(frozen=True)
class ToggleExpression:
    kind: ToggleNodeKind
    value: object
    arguments: tuple["ToggleExpression", ...] = ()

    def __add__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.ADD_F32, self, other)

    def __radd__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.ADD_F32, other, self)

    def __sub__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.SUB_F32, self, other)

    def __rsub__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.SUB_F32, other, self)

    def __mul__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.MUL_F32, self, other)

    def __rmul__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.MUL_F32, other, self)

    def __truediv__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.DIV_F32, self, other)

    def __rtruediv__(self, other: "ToggleExpression | float | int") -> "ToggleExpression":
        return toggle_operation(InstructionType.DIV_F32, other, self)

    def __neg__(self) -> "ToggleExpression":
        return toggle_operation(InstructionType.NEG_F32, self)


def _toggle_expression(value: ToggleExpression | float | int) -> ToggleExpression:
    if isinstance(value, ToggleExpression):
        return value
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("toggle-expression operands must be expressions or real numbers")
    return toggle_literal(float(value))


def toggle_operation(
    kind: InstructionType,
    *arguments: ToggleExpression | float | int,
) -> ToggleExpression:
    if kind not in _ARITY:
        raise ValueError(f"{kind.name} is not a supported FP32 toggle operation")
    if len(arguments) != _ARITY[kind]:
        raise ValueError(f"{kind.name} requires {_ARITY[kind]} arguments")
    return ToggleExpression(
        ToggleNodeKind.OPERATION,
        kind,
        tuple(_toggle_expression(argument) for argument in arguments),
    )


def toggle_literal(value: float) -> ToggleExpression:
    if not math.isfinite(float(value)):
        raise ValueError("toggle AST literals must be finite")
    return ToggleExpression(ToggleNodeKind.LITERAL, float(value))


def source_input(source: LeafSource) -> ToggleExpression:
    if not isinstance(source, LeafSource):
        raise TypeError("source_input requires a LeafSource")
    return ToggleExpression(ToggleNodeKind.SOURCE, source)


def toggle1(bit_offset: int, first: LeafSource, second: LeafSource) -> ToggleExpression:
    return _toggle_leaf(bit_offset, (first, second))


def toggle2(
    bit_offset: int,
    first: LeafSource,
    second: LeafSource,
    third: LeafSource,
    fourth: LeafSource,
) -> ToggleExpression:
    return _toggle_leaf(bit_offset, (first, second, third, fourth))


def _toggle_leaf(bit_offset: int, choices: tuple[LeafSource, ...]) -> ToggleExpression:
    width = 1 if len(choices) == 2 else 2 if len(choices) == 4 else 0
    if width == 0:
        raise ValueError("toggle leaves require exactly two or four choices")
    if isinstance(bit_offset, bool) or not isinstance(bit_offset, int):
        raise TypeError("toggle bit offset must be an integer")
    if bit_offset < 0 or bit_offset + width > MAX_TOGGLE_BITS:
        raise ValueError(f"toggle bits must fit in the low {MAX_TOGGLE_BITS} bits")
    if any(not isinstance(choice, LeafSource) for choice in choices):
        raise TypeError("toggle choices must be state or constant LeafSource values")
    return ToggleExpression(ToggleNodeKind.TOGGLE, (bit_offset, choices))


@dataclass(frozen=True)
class DecodedToggleInstruction:
    kind: InstructionType | ToggleInstructionType
    operands: tuple[int, ...] = ()


@dataclass(frozen=True)
class ToggleProgram:
    """Postorder program whose leaf instructions select register-resident values."""

    data: bytes

    @classmethod
    def from_expression(
        cls,
        expression: ToggleExpression,
        shape: KernelShape,
    ) -> "ToggleProgram":
        encoded = bytearray()

        def visit(node: ToggleExpression) -> None:
            for argument in node.arguments:
                visit(argument)
            if node.kind == ToggleNodeKind.SOURCE:
                encoded.extend((ToggleInstructionType.SOURCE_INPUT_F32, node.value.encode(shape)))
            elif node.kind == ToggleNodeKind.TOGGLE:
                bit_offset, choices = node.value
                opcode = (
                    ToggleInstructionType.TOGGLE_1BIT_INPUT_F32
                    if len(choices) == 2
                    else ToggleInstructionType.TOGGLE_2BIT_INPUT_F32
                )
                encoded.extend((opcode, bit_offset))
                encoded.extend(choice.encode(shape) for choice in choices)
            elif node.kind == ToggleNodeKind.LITERAL:
                encoded.append(InstructionType.CONSTANT_BITS_F32)
                encoded.extend(struct.pack("<f", float(node.value)))
            elif node.kind == ToggleNodeKind.OPERATION:
                encoded.append(int(node.value))
            else:
                raise ValueError("unknown toggle-expression node")

        visit(expression)
        encoded.append(InstructionType.RETURN_F32)
        program = cls(bytes(encoded))
        program.validate(shape)
        return program

    def instructions(self) -> Iterable[DecodedToggleInstruction]:
        offset = 0
        while offset < len(self.data):
            raw = self.data[offset]
            offset += 1
            if raw == ToggleInstructionType.SOURCE_INPUT_F32:
                if offset >= len(self.data):
                    raise ValueError("truncated source input")
                yield DecodedToggleInstruction(ToggleInstructionType(raw), (self.data[offset],))
                offset += 1
            elif raw == ToggleInstructionType.TOGGLE_1BIT_INPUT_F32:
                if offset + 3 > len(self.data):
                    raise ValueError("truncated one-bit toggle input")
                yield DecodedToggleInstruction(ToggleInstructionType(raw), tuple(self.data[offset : offset + 3]))
                offset += 3
            elif raw == ToggleInstructionType.TOGGLE_2BIT_INPUT_F32:
                if offset + 5 > len(self.data):
                    raise ValueError("truncated two-bit toggle input")
                yield DecodedToggleInstruction(ToggleInstructionType(raw), tuple(self.data[offset : offset + 5]))
                offset += 5
            else:
                try:
                    kind = InstructionType(raw)
                except ValueError as exc:
                    raise ValueError(f"unknown toggle opcode 0x{raw:02x}") from exc
                if kind == InstructionType.CONSTANT_BITS_F32:
                    if offset + 4 > len(self.data):
                        raise ValueError("truncated FP32 literal")
                    yield DecodedToggleInstruction(kind, (struct.unpack_from("<I", self.data, offset)[0],))
                    offset += 4
                else:
                    yield DecodedToggleInstruction(kind)

    def validate(self, shape: KernelShape) -> None:
        depth = 0
        returned = False
        for instruction in self.instructions():
            if returned:
                raise ValueError("instructions follow RETURN_F32")
            kind = instruction.kind
            if kind == ToggleInstructionType.SOURCE_INPUT_F32:
                self._validate_source(instruction.operands[0], shape)
                depth += 1
            elif kind in {
                ToggleInstructionType.TOGGLE_1BIT_INPUT_F32,
                ToggleInstructionType.TOGGLE_2BIT_INPUT_F32,
            }:
                width = 1 if kind == ToggleInstructionType.TOGGLE_1BIT_INPUT_F32 else 2
                bit_offset, *choices = instruction.operands
                if bit_offset + width > MAX_TOGGLE_BITS:
                    raise ValueError("toggle bit range exceeds the supported mask")
                for choice in choices:
                    self._validate_source(choice, shape)
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
                raise ValueError(f"{kind.name} is not valid in a toggle program")
        if not returned:
            raise ValueError("toggle program does not terminate with RETURN_F32")

    @staticmethod
    def _validate_source(encoded: int, shape: KernelShape) -> None:
        decode_leaf(encoded, shape)

    @property
    def required_toggle_bits(self) -> int:
        required = 0
        for instruction in self.instructions():
            if instruction.kind == ToggleInstructionType.TOGGLE_1BIT_INPUT_F32:
                required = max(required, instruction.operands[0] + 1)
            elif instruction.kind == ToggleInstructionType.TOGGLE_2BIT_INPUT_F32:
                required = max(required, instruction.operands[0] + 2)
        return required

    @property
    def leaf_count(self) -> int:
        return sum(
            instruction.kind
            in {
                ToggleInstructionType.SOURCE_INPUT_F32,
                ToggleInstructionType.TOGGLE_1BIT_INPUT_F32,
                ToggleInstructionType.TOGGLE_2BIT_INPUT_F32,
            }
            for instruction in self.instructions()
        )

    def resolved_sources(self, permutation: int, shape: KernelShape) -> tuple[int, ...]:
        if permutation < 0:
            raise ValueError("toggle permutation must be non-negative")
        self.validate(shape)
        result: list[int] = []
        for instruction in self.instructions():
            if instruction.kind == ToggleInstructionType.SOURCE_INPUT_F32:
                result.append(instruction.operands[0])
            elif instruction.kind == ToggleInstructionType.TOGGLE_1BIT_INPUT_F32:
                bit_offset, first, second = instruction.operands
                result.append((first, second)[(permutation >> bit_offset) & 1])
            elif instruction.kind == ToggleInstructionType.TOGGLE_2BIT_INPUT_F32:
                bit_offset, *choices = instruction.operands
                result.append(choices[(permutation >> bit_offset) & 3])
        return tuple(result)

    def to_dynamic_program(self, input_offset: int, shape: KernelShape) -> Program:
        self.validate(shape)
        stack: list[object] = []
        leaf_index = 0
        for instruction in self.instructions():
            kind = instruction.kind
            if kind in {
                ToggleInstructionType.SOURCE_INPUT_F32,
                ToggleInstructionType.TOGGLE_1BIT_INPUT_F32,
                ToggleInstructionType.TOGGLE_2BIT_INPUT_F32,
            }:
                stack.append(input_slot(input_offset + leaf_index))
                leaf_index += 1
            elif kind == InstructionType.CONSTANT_BITS_F32:
                value = struct.unpack("<f", struct.pack("<I", instruction.operands[0]))[0]
                stack.append(constant(value))
            elif kind == InstructionType.RETURN_F32:
                return Program.from_expression(stack[-1])
            elif kind in _ARITY:
                arguments = stack[-_ARITY[kind] :]
                del stack[-_ARITY[kind] :]
                stack.append(operation(kind, *arguments))
        raise AssertionError("validated toggle program did not return")

    def evaluate(
        self,
        state: Sequence[float],
        constants: Sequence[float],
        permutation: int,
        shape: KernelShape,
    ) -> float:
        self.validate(shape)
        stack: list[float] = []
        for instruction in self.instructions():
            kind = instruction.kind
            if kind in {
                ToggleInstructionType.SOURCE_INPUT_F32,
                ToggleInstructionType.TOGGLE_1BIT_INPUT_F32,
                ToggleInstructionType.TOGGLE_2BIT_INPUT_F32,
            }:
                if kind == ToggleInstructionType.SOURCE_INPUT_F32:
                    encoded = instruction.operands[0]
                elif kind == ToggleInstructionType.TOGGLE_1BIT_INPUT_F32:
                    bit_offset, first, second = instruction.operands
                    encoded = (first, second)[(permutation >> bit_offset) & 1]
                else:
                    bit_offset, *choices = instruction.operands
                    encoded = choices[(permutation >> bit_offset) & 3]
                source = decode_leaf(encoded, shape)
                stack.append(float(state[source.index] if source.kind == LeafKind.STATE else constants[source.index]))
            elif kind == InstructionType.CONSTANT_BITS_F32:
                stack.append(struct.unpack("<f", struct.pack("<I", instruction.operands[0]))[0])
            elif kind == InstructionType.RETURN_F32:
                return stack[-1]
            else:
                _evaluate_operation(stack, kind)
        raise AssertionError("validated toggle program did not return")

    def render_cuda(self, shape: KernelShape, toggle_name: str = "toggle_bits") -> str:
        self.validate(shape)
        stack: list[str] = []
        for instruction in self.instructions():
            kind = instruction.kind
            if kind == ToggleInstructionType.SOURCE_INPUT_F32:
                stack.append(_render_source(instruction.operands[0], shape))
            elif kind == ToggleInstructionType.TOGGLE_1BIT_INPUT_F32:
                bit_offset, first, second = instruction.operands
                stack.append(
                    f"((({toggle_name} >> {bit_offset}u) & 1u) != 0u ? "
                    f"{_render_source(second, shape)} : {_render_source(first, shape)})"
                )
            elif kind == ToggleInstructionType.TOGGLE_2BIT_INPUT_F32:
                bit_offset, *choices = instruction.operands
                selector = f"(({toggle_name} >> {bit_offset}u) & 3u)"
                rendered = [_render_source(choice, shape) for choice in choices]
                stack.append(
                    f"({selector} == 0u ? {rendered[0]} : {selector} == 1u ? {rendered[1]} : "
                    f"{selector} == 2u ? {rendered[2]} : {rendered[3]})"
                )
            elif kind == InstructionType.CONSTANT_BITS_F32:
                value = struct.unpack("<f", struct.pack("<I", instruction.operands[0]))[0]
                stack.append(_cuda_float(value))
            elif kind == InstructionType.RETURN_F32:
                return stack[-1]
            else:
                _render_operation(stack, kind)
        raise AssertionError("validated toggle program did not return")


@dataclass(frozen=True)
class ToggleSystem:
    programs: tuple[ToggleProgram, ...]

    def validate(self, shape: KernelShape) -> None:
        if len(self.programs) != shape.ast_count:
            raise ValueError("toggle system requires one program per missing site")
        for site, (program, expected_leaves) in enumerate(zip(self.programs, shape.ast_leaf_counts)):
            program.validate(shape)
            if program.leaf_count != expected_leaves:
                raise ValueError(
                    f"toggle AST {site} has {program.leaf_count} leaves; expected {expected_leaves}"
                )

    @property
    def required_toggle_bits(self) -> int:
        return max((program.required_toggle_bits for program in self.programs), default=0)

    def materialized_genome(self, shape: KernelShape) -> SystemGenome:
        self.validate(shape)
        return SystemGenome(
            tuple(
                program.to_dynamic_program(offset, shape)
                for program, offset in zip(self.programs, shape.ast_input_offsets)
            )
        )

    def resolved_bindings(self, permutation: int, shape: KernelShape) -> tuple[int, ...]:
        self.validate(shape)
        return tuple(
            source
            for program in self.programs
            for source in program.resolved_sources(permutation, shape)
        )


def _render_source(encoded: int, shape: KernelShape) -> str:
    source = decode_leaf(encoded, shape)
    return (
        f"stage_state[{source.index}]"
        if source.kind == LeafKind.STATE
        else f"constant{source.index}"
    )


def _cuda_float(value: float) -> str:
    rendered = format(float(value), ".9g")
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "f"


def _render_operation(stack: list[str], kind: InstructionType) -> None:
    arity = _ARITY.get(kind)
    if arity is None or len(stack) < arity:
        raise ValueError(f"cannot render malformed {kind.name}")
    arguments = stack[-arity:]
    del stack[-arity:]
    if kind in {
        InstructionType.ADD_F32,
        InstructionType.SUB_F32,
        InstructionType.MUL_F32,
        InstructionType.DIV_F32,
    }:
        intrinsic = {
            InstructionType.ADD_F32: "__fadd_rn",
            InstructionType.SUB_F32: "__fsub_rn",
            InstructionType.MUL_F32: "__fmul_rn",
            InstructionType.DIV_F32: "__fdividef",
        }[kind]
        # Preserve the encoded postorder tree. Ordinary CUDA expressions may be
        # reassociated or fused under fast math, while Secant's SASS path emits
        # one operation for each AST instruction.
        stack.append(f"{intrinsic}({arguments[0]}, {arguments[1]})")
    elif kind == InstructionType.NEG_F32:
        stack.append(f"(-{arguments[0]})")
    elif kind == InstructionType.FMA_F32:
        stack.append(f"fmaf({arguments[0]}, {arguments[1]}, {arguments[2]})")
    else:
        function = {
            InstructionType.SQRT_F32: "sqrtf",
            InstructionType.RCP_F32: None,
            InstructionType.ABS_F32: "fabsf",
            InstructionType.MIN_F32: "fminf",
            InstructionType.MAX_F32: "fmaxf",
            InstructionType.SIN_F32: "sinf",
            InstructionType.COS_F32: "cosf",
            InstructionType.EX2_F32: "exp2f",
            InstructionType.LG2_F32: "log2f",
            InstructionType.RSQRT_F32: "rsqrtf",
            InstructionType.TANH_F32: "tanhf",
            InstructionType.EXP_F32: "expf",
            InstructionType.LOG_F32: "logf",
        }.get(kind)
        if kind == InstructionType.RCP_F32:
            stack.append(f"(1.0f / {arguments[0]})")
        elif function is None:
            raise ValueError(f"CUDA rendering is not implemented for {kind.name}")
        else:
            stack.append(f"{function}({', '.join(arguments)})")


def _evaluate_operation(stack: list[float], kind: InstructionType) -> None:
    arity = _ARITY.get(kind)
    if arity is None or len(stack) < arity:
        raise ValueError(f"cannot evaluate malformed {kind.name}")
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
