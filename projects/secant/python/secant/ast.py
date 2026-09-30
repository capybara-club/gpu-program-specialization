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
import struct
from typing import Iterable, Sequence

import numpy as np


MAX_INPUTS = 128
MAX_PROGRAM_INSTRUCTIONS = 1024
MAX_PROGRAM_BYTES = 5 * MAX_PROGRAM_INSTRUCTIONS
MAX_STACK_DEPTH = 128
MAX_ROUTINES = 255
MAX_ROUTINE_ARGUMENTS = 4
MAX_ROUTINE_DEPTH = 8
MAX_DYNAMIC_LEAVES = 32


class InstructionType(IntEnum):
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
    STATIC_COLUMN_INPUT_F32 = 0xB6
    DYNAMIC_COLUMN_INPUT_F32 = 0xB7
    DYNAMIC_CONSTANT_INPUT_F32 = 0xB8
    DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32 = 0xB9
    EXP_F32 = 0xBA
    LOG_F32 = 0xBB


_ARITY = {
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


class _NodeKind(IntEnum):
    STATIC_COLUMN_INPUT = 0
    DYNAMIC_COLUMN_INPUT = 1
    DYNAMIC_CONSTANT_INPUT = 2
    DYNAMIC_CONSTANT_OR_COLUMN_INPUT = 3
    CONSTANT = 4
    ROUTINE_ARG = 5
    ROUTINE = 6
    OP = 7


@dataclass(frozen=True, slots=True)
class Expression:
    kind: _NodeKind
    value: int
    arguments: tuple[Expression, ...] = ()

    def __add__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.ADD_F32, self, other)

    def __radd__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.ADD_F32, other, self)

    def __sub__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.SUB_F32, self, other)

    def __rsub__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.SUB_F32, other, self)

    def __mul__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.MUL_F32, self, other)

    def __rmul__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.MUL_F32, other, self)

    def __truediv__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.DIV_F32, self, other)

    def __rtruediv__(self, other: Expression | float | int) -> Expression:
        return _operation(InstructionType.DIV_F32, other, self)

    def __neg__(self) -> Expression:
        return _operation(InstructionType.NEG_F32, self)


def _expression(value: Expression | float | int) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("expression operands must be Expression, float, or int")
    return constant(float(value))


def _operation(instruction_type: InstructionType, *arguments: Expression | float | int) -> Expression:
    expected = _ARITY[instruction_type]
    if len(arguments) != expected:
        raise ValueError(f"{instruction_type.name} requires {expected} arguments")
    return Expression(_NodeKind.OP, int(instruction_type), tuple(_expression(argument) for argument in arguments))


def input(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_INPUTS:
        raise ValueError(f"input index must be in [0, {MAX_INPUTS})")
    return Expression(_NodeKind.STATIC_COLUMN_INPUT, index)


def dynamic_column(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_DYNAMIC_LEAVES:
        raise ValueError(f"dynamic column index must be in [0, {MAX_DYNAMIC_LEAVES})")
    return Expression(_NodeKind.DYNAMIC_COLUMN_INPUT, index)


def dynamic_constant(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_DYNAMIC_LEAVES:
        raise ValueError(f"dynamic constant index must be in [0, {MAX_DYNAMIC_LEAVES})")
    return Expression(_NodeKind.DYNAMIC_CONSTANT_INPUT, index)


def dynamic_constant_or_column(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_DYNAMIC_LEAVES:
        raise ValueError(f"dynamic leaf index must be in [0, {MAX_DYNAMIC_LEAVES})")
    return Expression(_NodeKind.DYNAMIC_CONSTANT_OR_COLUMN_INPUT, index)


def constant(value: float) -> Expression:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return Expression(_NodeKind.CONSTANT, bits)


def constant_bits(bits: int) -> Expression:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("constant bits must fit in uint32")
    return Expression(_NodeKind.CONSTANT, bits)


def routine_arg(index: int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_ROUTINE_ARGUMENTS:
        raise ValueError(f"routine argument index must be in [0, {MAX_ROUTINE_ARGUMENTS})")
    return Expression(_NodeKind.ROUTINE_ARG, index)


def routine(index: int, *arguments: Expression | float | int) -> Expression:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_ROUTINES:
        raise ValueError(f"routine index must be in [0, {MAX_ROUTINES})")
    if len(arguments) > MAX_ROUTINE_ARGUMENTS:
        raise ValueError(f"routine calls support at most {MAX_ROUTINE_ARGUMENTS} arguments")
    return Expression(_NodeKind.ROUTINE, index, tuple(_expression(argument) for argument in arguments))


def absolute(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.ABS_F32, value)


def minimum(lhs: Expression | float | int, rhs: Expression | float | int) -> Expression:
    return _operation(InstructionType.MIN_F32, lhs, rhs)


def maximum(lhs: Expression | float | int, rhs: Expression | float | int) -> Expression:
    return _operation(InstructionType.MAX_F32, lhs, rhs)


def fma(
    lhs: Expression | float | int,
    rhs: Expression | float | int,
    addend: Expression | float | int,
) -> Expression:
    return _operation(InstructionType.FMA_F32, lhs, rhs, addend)


def sqrt(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.SQRT_F32, value)


def rcp(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.RCP_F32, value)


def sin(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.SIN_F32, value)


def cos(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.COS_F32, value)


def exp2(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.EX2_F32, value)


def log2(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.LG2_F32, value)


def rsqrt(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.RSQRT_F32, value)


def tanh(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.TANH_F32, value)


def exp(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.EXP_F32, value)


def log(value: Expression | float | int) -> Expression:
    return _operation(InstructionType.LOG_F32, value)


def _encode_expression(expression: Expression, output: bytearray, instruction_count: list[int]) -> None:
    for argument in expression.arguments:
        _encode_expression(argument, output, instruction_count)
    if expression.kind is _NodeKind.STATIC_COLUMN_INPUT:
        output.extend((InstructionType.STATIC_COLUMN_INPUT_F32, expression.value))
    elif expression.kind is _NodeKind.DYNAMIC_COLUMN_INPUT:
        output.extend((InstructionType.DYNAMIC_COLUMN_INPUT_F32, expression.value))
    elif expression.kind is _NodeKind.DYNAMIC_CONSTANT_INPUT:
        output.extend((InstructionType.DYNAMIC_CONSTANT_INPUT_F32, expression.value))
    elif expression.kind is _NodeKind.DYNAMIC_CONSTANT_OR_COLUMN_INPUT:
        output.extend((InstructionType.DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32, expression.value))
    elif expression.kind is _NodeKind.CONSTANT:
        output.append(InstructionType.CONSTANT_BITS_F32)
        output.extend(struct.pack("<I", expression.value))
    elif expression.kind is _NodeKind.ROUTINE_ARG:
        output.extend((InstructionType.ROUTINE_ARG_F32, expression.value))
    elif expression.kind is _NodeKind.ROUTINE:
        output.extend((InstructionType.ROUTINE_F32, expression.value))
    elif expression.kind is _NodeKind.OP:
        output.append(expression.value)
    else:
        raise ValueError("invalid expression node")
    instruction_count[0] += 1
    if instruction_count[0] >= MAX_PROGRAM_INSTRUCTIONS or len(output) >= MAX_PROGRAM_BYTES:
        raise ValueError("AST program exceeds the Secant instruction limit")


class Program:
    __slots__ = ("_bytecode", "_root")

    def __init__(self, root: Expression | float | int) -> None:
        expression = _expression(root)
        output = bytearray()
        instruction_count = [0]

        _encode_expression(expression, output, instruction_count)
        output.append(InstructionType.RETURN_F32)
        instruction_count[0] += 1
        self._root = expression
        self._bytecode = bytes(output)

    @property
    def root(self) -> Expression:
        return self._root

    @property
    def bytecode(self) -> bytes:
        return self._bytecode

    def __bytes__(self) -> bytes:
        return self._bytecode

    def __repr__(self) -> str:
        return f"Program(bytecode={self._bytecode.hex()!r})"


def program_bytes(program: Program | bytes | bytearray | memoryview) -> bytes:
    if isinstance(program, Program):
        return program.bytecode
    if isinstance(program, bytes):
        return program
    if isinstance(program, (bytearray, memoryview)):
        return bytes(program)
    raise TypeError("AST programs must be Program or bytes-like objects")


def programs_bytes(programs: Iterable[Program | bytes | bytearray | memoryview]) -> tuple[bytes, ...]:
    result = tuple(program_bytes(program) for program in programs)
    if not result:
        raise ValueError("at least one AST program is required")
    return result


def pack_programs(
    programs: Iterable[Program | bytes | bytearray | memoryview],
) -> tuple[bytearray, np.ndarray]:
    """Pack variable-length programs contiguously and return uint64 start/end offsets."""

    output = bytearray()
    offsets = [0]
    for program in programs:
        output.extend(program_bytes(program))
        offsets.append(len(output))
    if len(offsets) == 1:
        raise ValueError("at least one AST program is required")
    return output, np.asarray(offsets, dtype=np.uint64)


def _instruction_size(code: int) -> int:
    if code == InstructionType.CONSTANT_BITS_F32:
        return 5
    if code in (
        InstructionType.ROUTINE_F32,
        InstructionType.ROUTINE_ARG_F32,
        InstructionType.STATIC_COLUMN_INPUT_F32,
        InstructionType.DYNAMIC_COLUMN_INPUT_F32,
        InstructionType.DYNAMIC_CONSTANT_INPUT_F32,
        InstructionType.DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
    ):
        return 2
    try:
        InstructionType(code)
    except ValueError:
        return 0
    return 1


def validate_program(program: Program | bytes | bytearray | memoryview) -> bytes:
    data = program_bytes(program)
    offset = 0

    for _ in range(MAX_PROGRAM_INSTRUCTIONS):
        if offset >= len(data):
            raise ValueError("AST program is not return terminated")
        size = _instruction_size(data[offset])
        if size == 0 or offset + size > len(data):
            raise ValueError("AST program contains an invalid or truncated instruction")
        if data[offset] == InstructionType.RETURN_F32:
            if offset + 1 != len(data):
                raise ValueError("AST program contains bytes after return")
            return data
        offset += size
    raise ValueError("AST program exceeds the instruction limit")


def _validate_dynamic_leaf_program(
    program: Program | bytes | bytearray | memoryview,
    num_dynamic_leaves: int,
    num_static_input_columns: int,
) -> bytes:
    data = validate_program(program)
    offset = 0

    while True:
        code = data[offset]
        if code == InstructionType.STATIC_COLUMN_INPUT_F32 and data[offset + 1] >= num_static_input_columns:
            raise ValueError(f"static column index {data[offset + 1]} is out of range")
        if code in (
            InstructionType.DYNAMIC_COLUMN_INPUT_F32,
            InstructionType.DYNAMIC_CONSTANT_INPUT_F32,
            InstructionType.DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
        ) and data[offset + 1] >= num_dynamic_leaves:
            raise ValueError(f"dynamic leaf index {data[offset + 1]} is out of range")
        if code == InstructionType.RETURN_F32:
            return data
        offset += _instruction_size(code)


def _routine_arities(routines: Sequence[bytes]) -> tuple[int, ...]:
    result: list[int] = []
    for routine_program in routines:
        offset = 0
        arity = 0
        for _ in range(MAX_PROGRAM_INSTRUCTIONS):
            code = routine_program[offset]
            if code == InstructionType.ROUTINE_ARG_F32:
                arity = max(arity, routine_program[offset + 1] + 1)
            if code == InstructionType.RETURN_F32:
                break
            offset += _instruction_size(code)
        result.append(arity)
    return tuple(result)


def _as_f32(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _evaluate_frame(
    program: bytes,
    values: np.ndarray,
    routines: tuple[bytes, ...],
    routine_arities: tuple[int, ...],
    arguments: tuple[np.ndarray, ...],
    dynamic_constant_values: np.ndarray | None,
    dynamic_leaf_mask: int | None,
    dynamic_leaf_words: np.ndarray | None,
    depth: int,
) -> np.ndarray:
    if depth > MAX_ROUTINE_DEPTH:
        raise ValueError("AST routine depth exceeded")
    stack: list[np.ndarray] = []
    offset = 0

    for instruction_index in range(MAX_PROGRAM_INSTRUCTIONS):
        code = program[offset]
        if code == InstructionType.STATIC_COLUMN_INPUT_F32:
            input_index = program[offset + 1]
            if input_index >= values.shape[-1]:
                raise ValueError(f"static column index {input_index} is out of range")
            stack.append(values[..., input_index])
        elif code in (
            InstructionType.DYNAMIC_COLUMN_INPUT_F32,
            InstructionType.DYNAMIC_CONSTANT_INPUT_F32,
            InstructionType.DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
        ):
            input_index = program[offset + 1]
            if code == InstructionType.DYNAMIC_CONSTANT_INPUT_F32 and dynamic_constant_values is not None:
                if input_index >= dynamic_constant_values.shape[-1]:
                    raise ValueError(f"dynamic constant index {input_index} is out of range")
                stack.append(dynamic_constant_values[..., input_index])
            else:
                if dynamic_leaf_mask is None or dynamic_leaf_words is None or input_index >= len(dynamic_leaf_words):
                    raise ValueError("dynamic leaf instruction is not valid for this execution shape")
                is_column = (dynamic_leaf_mask & (1 << input_index)) != 0
                if code == InstructionType.DYNAMIC_COLUMN_INPUT_F32 and not is_column:
                    raise ValueError(f"dynamic column leaf {input_index} contains a constant")
                if code == InstructionType.DYNAMIC_CONSTANT_INPUT_F32 and is_column:
                    raise ValueError(f"dynamic constant leaf {input_index} contains a column")
                word = int(dynamic_leaf_words[input_index])
                if is_column:
                    if word >= values.shape[-1]:
                        raise ValueError(f"dynamic column index {word} is out of range")
                    stack.append(values[..., word])
                else:
                    stack.append(np.asarray(word, dtype=np.uint32).view(np.float32))
        elif code == InstructionType.CONSTANT_BITS_F32:
            stack.append(np.asarray(struct.unpack_from("<f", program, offset + 1)[0], dtype=np.float32))
        elif code == InstructionType.ROUTINE_ARG_F32:
            argument_index = program[offset + 1]
            if argument_index >= len(arguments):
                raise ValueError(f"routine argument index {argument_index} is out of range")
            stack.append(arguments[argument_index])
        elif code == InstructionType.ROUTINE_F32:
            routine_index = program[offset + 1]
            if routine_index >= len(routines):
                raise ValueError(f"routine index {routine_index} is out of range")
            arity = routine_arities[routine_index]
            if len(stack) < arity:
                raise ValueError(f"stack underflow at instruction {instruction_index}")
            routine_arguments = tuple(stack[-arity:]) if arity else ()
            if arity:
                del stack[-arity:]
            stack.append(
                _evaluate_frame(
                    routines[routine_index],
                    values,
                    routines,
                    routine_arities,
                    routine_arguments,
                    dynamic_constant_values,
                    dynamic_leaf_mask,
                    dynamic_leaf_words,
                    depth + 1,
                )
            )
        elif code == InstructionType.RETURN_F32:
            if len(stack) != 1:
                raise ValueError("return requires exactly one live value")
            return _as_f32(stack[0])
        else:
            instruction_type = InstructionType(code)
            arity = _ARITY.get(instruction_type)
            if arity is None or len(stack) < arity:
                raise ValueError(f"stack underflow or unsupported operation at instruction {instruction_index}")
            operands = stack[-arity:]
            del stack[-arity:]
            with np.errstate(all="ignore"):
                if instruction_type is InstructionType.ADD_F32:
                    value = operands[0] + operands[1]
                elif instruction_type is InstructionType.SUB_F32:
                    value = operands[0] - operands[1]
                elif instruction_type is InstructionType.MUL_F32:
                    value = operands[0] * operands[1]
                elif instruction_type is InstructionType.DIV_F32:
                    value = operands[0] / operands[1]
                elif instruction_type is InstructionType.NEG_F32:
                    value = -operands[0]
                elif instruction_type is InstructionType.SQRT_F32:
                    value = np.sqrt(operands[0])
                elif instruction_type is InstructionType.RCP_F32:
                    value = np.float32(1.0) / operands[0]
                elif instruction_type is InstructionType.ABS_F32:
                    value = np.abs(operands[0])
                elif instruction_type is InstructionType.MIN_F32:
                    value = np.fmin(operands[0], operands[1])
                elif instruction_type is InstructionType.MAX_F32:
                    value = np.fmax(operands[0], operands[1])
                elif instruction_type is InstructionType.FMA_F32:
                    value = operands[0] * operands[1] + operands[2]
                elif instruction_type is InstructionType.SIN_F32:
                    value = np.sin(operands[0])
                elif instruction_type is InstructionType.COS_F32:
                    value = np.cos(operands[0])
                elif instruction_type is InstructionType.EX2_F32:
                    value = np.exp2(operands[0])
                elif instruction_type is InstructionType.LG2_F32:
                    value = np.log2(operands[0])
                elif instruction_type is InstructionType.RSQRT_F32:
                    value = np.float32(1.0) / np.sqrt(operands[0])
                elif instruction_type is InstructionType.TANH_F32:
                    value = np.tanh(operands[0])
                elif instruction_type is InstructionType.EXP_F32:
                    value = np.exp(operands[0])
                elif instruction_type is InstructionType.LOG_F32:
                    value = np.log(operands[0])
                else:
                    raise ValueError(f"unsupported operation {instruction_type.name}")
            stack.append(_as_f32(value))
        offset += _instruction_size(code)
    raise ValueError("AST program did not return")


def evaluate(
    program: Program | bytes | bytearray | memoryview,
    values: np.ndarray,
    *,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    packed_program = validate_program(program)
    packed_routines = tuple(validate_program(routine_program) for routine_program in routines)
    packed_values = np.asarray(values, dtype=np.float32)
    if packed_values.ndim == 0:
        raise ValueError("values must have an input dimension")
    return _evaluate_frame(
        packed_program,
        packed_values,
        packed_routines,
        _routine_arities(packed_routines),
        (),
        None,
        None,
        None,
        0,
    )


def materialize(
    programs: Sequence[Program | bytes | bytearray | memoryview],
    input_columns: np.ndarray,
    *,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    packed_input = np.asarray(input_columns, dtype=np.float32)
    if packed_input.ndim != 2 or packed_input.shape[0] == 0 or packed_input.shape[1] == 0:
        raise ValueError("input_columns must be nonempty [input, row] data")
    return np.stack(
        tuple(evaluate(program, packed_input.T, routines=routines) for program in programs),
        axis=0,
    ).astype(np.float32, copy=False)


def sse(
    programs: Sequence[Program | bytes | bytearray | memoryview],
    input_columns: np.ndarray,
    targets: np.ndarray,
    *,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    values = materialize(programs, input_columns, routines=routines)
    packed_targets = np.asarray(targets, dtype=np.float32)
    if packed_targets.ndim != 2 or packed_targets.shape[1] != values.shape[1]:
        raise ValueError("targets must be [target, row] with the same row count as input")
    errors = values[:, None, :] - packed_targets[None, :, :]
    return np.sum(errors * errors, axis=-1, dtype=np.float32)


def affine_stats(
    programs: Sequence[Program | bytes | bytearray | memoryview],
    input_columns: np.ndarray,
    targets: np.ndarray,
    *,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    """Return `[ast][prediction sum, square sum, target cross sums...]`."""

    values = materialize(programs, input_columns, routines=routines)
    packed_targets = np.asarray(targets, dtype=np.float32)
    if packed_targets.ndim != 2 or packed_targets.shape[1] != values.shape[1]:
        raise ValueError("targets must be [target, row] with the same row count as input")
    output = np.empty((values.shape[0], 2 + packed_targets.shape[0]), dtype=np.float32)
    output[:, 0] = np.sum(values, axis=1, dtype=np.float32)
    output[:, 1] = np.sum(values * values, axis=1, dtype=np.float32)
    output[:, 2:] = values @ packed_targets.T
    return output


def gram_stats(
    programs: Sequence[Program | bytes | bytearray | memoryview],
    input_columns: np.ndarray,
    targets: np.ndarray,
    asts_per_cohort: int,
    *,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    """Return padded feature sums, Gram matrices, and target cross sums by cohort."""

    if isinstance(asts_per_cohort, bool) or not isinstance(asts_per_cohort, int) or not 0 < asts_per_cohort <= 32:
        raise ValueError("asts_per_cohort must be an integer in [1, 32]")
    values = materialize(programs, input_columns, routines=routines)
    packed_targets = np.asarray(targets, dtype=np.float32)
    if packed_targets.ndim != 2 or packed_targets.shape[1] != values.shape[1]:
        raise ValueError("targets must be [target, row] with the same row count as input")
    num_targets = packed_targets.shape[0]
    num_cohorts = (values.shape[0] + asts_per_cohort - 1) // asts_per_cohort
    statistic_count = asts_per_cohort + asts_per_cohort * asts_per_cohort + asts_per_cohort * num_targets
    output = np.zeros((num_cohorts, statistic_count), dtype=np.float32)
    gram_begin = asts_per_cohort
    cross_begin = gram_begin + asts_per_cohort * asts_per_cohort

    for cohort in range(num_cohorts):
        begin = cohort * asts_per_cohort
        active = values[begin : begin + asts_per_cohort]
        active_count = active.shape[0]
        output[cohort, :active_count] = np.sum(active, axis=1, dtype=np.float32)
        gram = output[cohort, gram_begin:cross_begin].reshape(asts_per_cohort, asts_per_cohort)
        gram[:active_count, :active_count] = active @ active.T
        cross = output[cohort, cross_begin:].reshape(asts_per_cohort, num_targets)
        cross[:active_count, :] = active @ packed_targets.T
    return output


def dynamic_constant_sse(
    programs: Sequence[Program | bytes | bytearray | memoryview],
    input_columns: np.ndarray,
    constant_settings: np.ndarray,
    targets: np.ndarray,
    *,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    packed_input = np.asarray(input_columns, dtype=np.float32)
    packed_constants = np.asarray(constant_settings, dtype=np.float32)
    packed_targets = np.asarray(targets, dtype=np.float32)
    if packed_input.ndim != 2 or packed_constants.ndim != 2 or packed_targets.ndim != 2:
        raise ValueError("input_columns, constant_settings, and targets must be two-dimensional")
    if packed_input.shape[1] == 0 or packed_targets.shape[1] != packed_input.shape[1]:
        raise ValueError("input and target row counts must match and be nonzero")

    packed_programs = tuple(validate_program(program) for program in programs)
    packed_routines = tuple(validate_program(routine_program) for routine_program in routines)
    routine_arities = _routine_arities(packed_routines)
    output = np.empty((len(packed_programs), packed_targets.shape[0], packed_constants.shape[1]), dtype=np.float32)
    for setting in range(packed_constants.shape[1]):
        predictions = np.stack(
            tuple(
                _evaluate_frame(
                    program,
                    packed_input.T,
                    packed_routines,
                    routine_arities,
                    (),
                    packed_constants[:, setting],
                    None,
                    None,
                    0,
                )
                for program in packed_programs
            ),
            axis=0,
        )
        errors = predictions[:, None, :] - packed_targets[None, :, :]
        output[:, :, setting] = np.sum(errors * errors, axis=-1, dtype=np.float32)
    return output


def dynamic_leaf_sse(
    programs: Sequence[Program | bytes | bytearray | memoryview],
    input_columns: np.ndarray,
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    targets: np.ndarray,
    *,
    num_static_input_columns: int = 0,
    routines: Sequence[Program | bytes | bytearray | memoryview] = (),
) -> np.ndarray:
    packed_input = np.asarray(input_columns, dtype=np.float32)
    packed_masks = np.asarray(leaf_masks, dtype=np.uint32)
    packed_words = np.asarray(leaf_words, dtype=np.uint32)
    packed_targets = np.asarray(targets, dtype=np.float32)
    if packed_input.ndim != 2 or packed_targets.ndim != 2:
        raise ValueError("input_columns and targets must be two-dimensional")
    if packed_masks.ndim != 1 or packed_words.ndim != 2 or packed_words.shape[0] != packed_masks.shape[0]:
        raise ValueError("leaf_masks and leaf_words must be [setting] and [setting, leaf]")
    if not 0 < packed_words.shape[1] <= MAX_DYNAMIC_LEAVES:
        raise ValueError(f"dynamic leaf count must be in [1, {MAX_DYNAMIC_LEAVES}]")
    if packed_input.shape[0] == 0 or packed_input.shape[1] == 0 or packed_targets.shape[1] != packed_input.shape[1]:
        raise ValueError("input and target row counts must match and be nonzero")
    if num_static_input_columns not in (0, packed_input.shape[0]):
        raise ValueError("num_static_input_columns must be zero or the input column count")

    packed_programs = tuple(
        _validate_dynamic_leaf_program(program, packed_words.shape[1], num_static_input_columns)
        for program in programs
    )
    packed_routines = tuple(
        _validate_dynamic_leaf_program(routine_program, packed_words.shape[1], num_static_input_columns)
        for routine_program in routines
    )
    routine_arities = _routine_arities(packed_routines)
    output = np.empty((len(packed_programs), packed_targets.shape[0], packed_masks.shape[0]), dtype=np.float32)
    for setting, mask in enumerate(packed_masks):
        predictions = np.stack(
            tuple(
                _evaluate_frame(
                    program,
                    packed_input.T,
                    packed_routines,
                    routine_arities,
                    (),
                    None,
                    int(mask),
                    packed_words[setting],
                    0,
                )
                for program in packed_programs
            ),
            axis=0,
        )
        errors = predictions[:, None, :] - packed_targets[None, :, :]
        output[:, :, setting] = np.sum(errors * errors, axis=-1, dtype=np.float32)
    return output
