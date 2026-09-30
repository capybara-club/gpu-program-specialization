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

from collections.abc import Sequence

import numpy as np

from .ast import InstructionType, MAX_PROGRAM_INSTRUCTIONS, Op, decode_instruction, pack_program, pack_programs


def _f32(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _constant_from_bits(bits: int) -> np.float32:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)[()]


def _evaluate_frame(
    program: np.ndarray,
    inputs: np.ndarray,
    routines: np.ndarray | None,
    routine_args: tuple[np.ndarray, ...],
    depth: int,
) -> np.ndarray:
    if depth > 8:
        raise ValueError("AST routine depth exceeded")

    stack: list[np.ndarray] = []
    for instruction_idx, word in enumerate(program[:MAX_PROGRAM_INSTRUCTIONS]):
        instruction_type, aux, payload = decode_instruction(word)

        if instruction_type is InstructionType.INPUT:
            if payload >= inputs.shape[-1]:
                raise ValueError(f"input index {payload} is out of range")
            stack.append(inputs[..., payload])
        elif instruction_type is InstructionType.CONSTANT_BITS:
            stack.append(_constant_from_bits(payload))
        elif instruction_type is InstructionType.ROUTINE_ARG:
            if payload >= len(routine_args):
                raise ValueError(f"routine argument index {payload} is out of range")
            stack.append(routine_args[payload])
        elif instruction_type is InstructionType.OP:
            op = Op(payload)
            if len(stack) < aux:
                raise ValueError(f"stack underflow at instruction {instruction_idx}")

            with np.errstate(all="ignore"):
                if op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MIN, Op.MAX):
                    if aux != 2:
                        raise ValueError(f"{op.name} requires two arguments")
                    rhs = stack.pop()
                    lhs = stack.pop()
                    if op is Op.ADD:
                        value = lhs + rhs
                    elif op is Op.SUB:
                        value = lhs - rhs
                    elif op is Op.MUL:
                        value = lhs * rhs
                    elif op is Op.DIV:
                        value = lhs / rhs
                    elif op is Op.MIN:
                        value = np.fmin(lhs, rhs)
                    else:
                        value = np.fmax(lhs, rhs)
                elif op is Op.FMA:
                    if aux != 3:
                        raise ValueError("FMA requires three arguments")
                    c = stack.pop()
                    b = stack.pop()
                    a = stack.pop()
                    value = a * b + c
                else:
                    if aux != 1:
                        raise ValueError(f"{op.name} requires one argument")
                    operand = stack.pop()
                    if op is Op.NEG:
                        value = -operand
                    elif op is Op.SQRT:
                        value = np.sqrt(operand)
                    elif op is Op.RCP:
                        value = np.float32(1.0) / operand
                    elif op is Op.ABS:
                        value = np.abs(operand)
                    elif op is Op.SIN:
                        value = np.sin(operand)
                    elif op is Op.COS:
                        value = np.cos(operand)
                    elif op is Op.EX2:
                        value = np.exp2(operand)
                    elif op is Op.LG2:
                        value = np.log2(operand)
                    elif op is Op.RSQRT:
                        value = np.float32(1.0) / np.sqrt(operand)
                    elif op is Op.TANH:
                        value = np.tanh(operand)
                    else:
                        raise ValueError(f"unsupported AST operation {op.name}")
                stack.append(_f32(value))
        elif instruction_type is InstructionType.ROUTINE:
            if routines is None or payload >= routines.shape[0] or len(stack) < aux:
                raise ValueError(f"routine index {payload} or its arguments are invalid")
            args = tuple(stack[-aux:]) if aux else ()
            if aux:
                del stack[-aux:]
            stack.append(_evaluate_frame(routines[payload], inputs, routines, args, depth + 1))
        elif instruction_type is InstructionType.RETURN:
            if len(stack) != 1:
                raise ValueError("return requires exactly one live value")
            return _f32(stack[0])
        else:
            raise ValueError(f"invalid AST instruction type {int(instruction_type)}")

    raise ValueError("AST program did not return")


def evaluate_program(
    program: Sequence[int | np.uint64] | np.ndarray,
    inputs: np.ndarray,
    *,
    routines: np.ndarray | None = None,
) -> np.ndarray:
    packed_program = pack_program(program)
    packed_inputs = np.asarray(inputs, dtype=np.float32)
    if packed_inputs.ndim == 0:
        raise ValueError("inputs must have an input dimension")
    packed_routines = None if routines is None else pack_programs(routines)
    return _evaluate_frame(packed_program, packed_inputs, packed_routines, (), 0)


def evaluate_programs(
    programs: Sequence[Sequence[int | np.uint64] | np.ndarray] | np.ndarray,
    inputs: np.ndarray,
    *,
    routines: np.ndarray | None = None,
) -> np.ndarray:
    packed_programs = pack_programs(programs)
    packed_inputs = np.asarray(inputs, dtype=np.float32)
    if packed_inputs.ndim == 0:
        raise ValueError("inputs must have an input dimension")
    packed_routines = None if routines is None else pack_programs(routines)
    return np.stack(
        tuple(_evaluate_frame(program, packed_inputs, packed_routines, (), 0) for program in packed_programs),
        axis=0,
    )


def evaluate_sse(
    programs: Sequence[Sequence[int | np.uint64] | np.ndarray] | np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    *,
    routines: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if x.ndim != 2 or target.ndim != 1 or x.shape[1] < target.size:
        raise ValueError("x must be [column, row] and target must be [row]")

    predictions = evaluate_values(
        programs,
        x,
        leaf_masks,
        leaf_words,
        num_rows=target.size,
        routines=routines,
    )
    errors = np.asarray(predictions - target[None, None, :], dtype=np.float32)
    return np.sum(errors * errors, axis=-1, dtype=np.float32)


def evaluate_values(
    programs: Sequence[Sequence[int | np.uint64] | np.ndarray] | np.ndarray,
    x: np.ndarray,
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    *,
    num_rows: int | None = None,
    routines: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    leaf_masks = np.asarray(leaf_masks, dtype=np.uint8)
    leaf_words = np.asarray(leaf_words, dtype=np.uint32)

    if x.ndim != 2:
        raise ValueError("x must be [column, row]")
    if num_rows is None:
        num_rows = x.shape[1]
    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or not 1 <= num_rows <= x.shape[1]:
        raise ValueError("num_rows must select a positive prefix of x")
    if leaf_masks.ndim != 1 or leaf_words.ndim != 2 or leaf_words.shape[0] != leaf_masks.size or leaf_words.shape[1] < 8:
        raise ValueError("settings must provide one mask and eight words per setting")

    num_settings = leaf_masks.size
    inputs = np.empty((num_settings, num_rows, 8), dtype=np.float32)
    for input_idx in range(8):
        words = leaf_words[:, input_idx]
        inputs[:, :, input_idx] = words.view(np.float32)[:, None]
        active = ((leaf_masks >> input_idx) & 1).astype(bool)
        if np.any(active):
            columns = words[active].astype(np.intp)
            if np.any(columns >= x.shape[0]):
                raise ValueError("active setting column is out of range")
            inputs[active, :, input_idx] = x[columns, :num_rows]

    return evaluate_programs(programs, inputs, routines=routines)
