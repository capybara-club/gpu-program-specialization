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
from enum import IntEnum
import struct

import numpy as np


INSTRUCTION_SIZE = 8
MAX_PROGRAM_INSTRUCTIONS = 1024


class InstructionType(IntEnum):
    NONE = 0
    INPUT = 1
    CONSTANT_BITS = 2
    OP = 3
    ROUTINE = 4
    RETURN = 5
    ROUTINE_ARG = 6


class Op(IntEnum):
    NONE = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    NEG = 5
    SQRT = 6
    RCP = 7
    ABS = 8
    MIN = 9
    MAX = 10
    FMA = 11
    SIN = 12
    COS = 13
    EX2 = 14
    LG2 = 15
    RSQRT = 16
    TANH = 17


def _encode(instruction_type: InstructionType, aux: int, payload: int) -> np.uint64:
    if not 0 <= aux <= 0xFFFF:
        raise ValueError("aux must fit in uint16")
    if not 0 <= payload <= 0xFFFFFFFF:
        raise ValueError("payload must fit in uint32")
    return np.uint64(int(instruction_type) | (aux << 16) | (payload << 32))


def encode_input(input_idx: int) -> np.uint64:
    return _encode(InstructionType.INPUT, 0, input_idx)


def encode_constant_bits(bits: int) -> np.uint64:
    return _encode(InstructionType.CONSTANT_BITS, 0, bits)


def encode_constant(value: float) -> np.uint64:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return encode_constant_bits(bits)


def encode_op(op: Op, num_args: int) -> np.uint64:
    return _encode(InstructionType.OP, num_args, int(op))


def encode_routine(routine_idx: int, num_args: int) -> np.uint64:
    return _encode(InstructionType.ROUTINE, num_args, routine_idx)


def encode_routine_arg(arg_idx: int) -> np.uint64:
    return _encode(InstructionType.ROUTINE_ARG, 0, arg_idx)


RETURN = _encode(InstructionType.RETURN, 0, 0)
ADD = encode_op(Op.ADD, 2)
SUB = encode_op(Op.SUB, 2)
MUL = encode_op(Op.MUL, 2)
DIV = encode_op(Op.DIV, 2)
NEG = encode_op(Op.NEG, 1)
SQRT = encode_op(Op.SQRT, 1)
RCP = encode_op(Op.RCP, 1)
ABS = encode_op(Op.ABS, 1)
MIN = encode_op(Op.MIN, 2)
MAX = encode_op(Op.MAX, 2)
FMA = encode_op(Op.FMA, 3)
SIN = encode_op(Op.SIN, 1)
COS = encode_op(Op.COS, 1)
EX2 = encode_op(Op.EX2, 1)
LG2 = encode_op(Op.LG2, 1)
RSQRT = encode_op(Op.RSQRT, 1)
TANH = encode_op(Op.TANH, 1)


def decode_instruction(instruction: int | np.uint64) -> tuple[InstructionType, int, int]:
    word = int(instruction)
    instruction_type = InstructionType(word & 0xFFFF)
    aux = (word >> 16) & 0xFFFF
    payload = (word >> 32) & 0xFFFFFFFF
    return instruction_type, aux, payload


def _require_terminated(program: np.ndarray, label: str) -> None:
    if program.ndim != 1 or program.size == 0 or program.size > MAX_PROGRAM_INSTRUCTIONS:
        raise ValueError(f"{label} must be a non-empty one-dimensional AST program")
    instruction_types = np.bitwise_and(program, np.uint64(0xFFFF))
    if not np.any(instruction_types == int(InstructionType.RETURN)):
        raise ValueError(f"{label} must contain a return instruction")


def pack_program(instructions: Sequence[int | np.uint64] | np.ndarray) -> np.ndarray:
    program = np.ascontiguousarray(instructions, dtype=np.uint64)
    _require_terminated(program, "program")
    return program


def pack_programs(
    programs: Sequence[Sequence[int | np.uint64] | np.ndarray] | np.ndarray,
    *,
    stride: int | None = None,
) -> np.ndarray:
    if isinstance(programs, np.ndarray) and programs.ndim == 2:
        packed = np.ascontiguousarray(programs, dtype=np.uint64)
        for program_idx, program in enumerate(packed):
            _require_terminated(program, f"programs[{program_idx}]")
        if stride is not None and stride != packed.shape[1]:
            raise ValueError("stride does not match the supplied program matrix")
        return packed

    rows = tuple(pack_program(program) for program in programs)
    if not rows:
        raise ValueError("programs must not be empty")
    output_stride = max(program.size for program in rows) if stride is None else stride
    if output_stride <= 0 or output_stride > MAX_PROGRAM_INSTRUCTIONS:
        raise ValueError("stride must be in [1, 1024]")
    if any(program.size > output_stride for program in rows):
        raise ValueError("stride is shorter than at least one AST program")

    packed = np.zeros((len(rows), output_stride), dtype=np.uint64)
    for program_idx, program in enumerate(rows):
        packed[program_idx, : program.size] = program
    return packed


def verify_native_abi() -> None:
    try:
        import _cusr_native as _native
    except ImportError as error:
        raise RuntimeError("the cuSR nanobind extension is not available") from error

    probes = (
        (encode_input(7), _native.encode_input(7)),
        (encode_constant_bits(0xDEADBEEF), _native.encode_constant_bits(0xDEADBEEF)),
        (encode_constant(1.25), _native.encode_constant(1.25)),
        (encode_op(Op.FMA, 3), _native.encode_op(_native.Op.FMA, 3)),
        (encode_routine(9, 2), _native.encode_routine(9, 2)),
        (encode_routine_arg(3), _native.encode_routine_arg(3)),
        (RETURN, _native.encode_return()),
    )
    if _native.INSTRUCTION_SIZE != INSTRUCTION_SIZE or any(int(lhs) != int(rhs) for lhs, rhs in probes):
        raise RuntimeError("Python and native CusrAstInstruction encodings do not match")
