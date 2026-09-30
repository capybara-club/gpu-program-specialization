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

import math

import numpy as np

import secant
from secant.routines import SAFE_RSQRT_ROUTINE_INDEX, SAFE_SQRT_ROUTINE_INDEX


_MASK64 = (1 << 64) - 1
_BINARY_ENCODINGS = (
    (secant.InstructionType.ADD_F32,),
    (secant.InstructionType.MUL_F32,),
    (secant.InstructionType.MIN_F32,),
    (secant.InstructionType.MAX_F32,),
)
_UNARY_ENCODINGS = {
    "alu": (
        (secant.InstructionType.NEG_F32,),
        (secant.InstructionType.ABS_F32,),
    ),
    "mufu": (
        (secant.InstructionType.SIN_F32,),
        (secant.InstructionType.COS_F32,),
        (secant.InstructionType.TANH_F32,),
        (secant.InstructionType.ROUTINE_F32, SAFE_SQRT_ROUTINE_INDEX),
        (secant.InstructionType.ROUTINE_F32, SAFE_RSQRT_ROUTINE_INDEX),
    ),
}


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, value ^ (value >> 31)


def _affine_permutation(total: int, seed: int) -> tuple[int, int]:
    state, value = _splitmix64(seed & _MASK64)
    offset = value % total
    state, value = _splitmix64(state)
    stride = value % (total - 1) + 1
    while math.gcd(stride, total) != 1:
        state, value = _splitmix64(state)
        stride = value % (total - 1) + 1
    return offset, stride


def _balanced_postorder(num_leaves: int) -> tuple[tuple[bool, int], ...]:
    result: list[tuple[bool, int]] = []
    binary_index = 0

    def visit(first_leaf: int, leaf_count: int) -> None:
        nonlocal binary_index
        if leaf_count == 1:
            result.append((True, first_leaf))
            return
        left_count = leaf_count // 2
        visit(first_leaf, left_count)
        visit(first_leaf + left_count, leaf_count - left_count)
        result.append((False, binary_index))
        binary_index += 1

    visit(0, num_leaves)
    return tuple(result)


def generate_unique_balanced_programs(
    count: int,
    mode: str,
    seed: int,
    num_inputs: int = 8,
    num_leaves: int = 8,
) -> tuple[bytearray, np.ndarray]:
    """Generate unique, seeded, balanced postorder programs in packed form.

    Every leaf independently selects an input and unary operation. Every
    internal node independently selects a binary operation. The generated
    bytecode sequences are unique, although algebraically equivalent trees
    can still exist because Secant does not simplify expressions here.
    """

    if mode not in _UNARY_ENCODINGS:
        raise ValueError(f"unknown AST mode {mode!r}")
    if count <= 0:
        raise ValueError("count must be positive")
    if not 1 <= num_inputs <= 128:
        raise ValueError("num_inputs must be in [1, 128]")
    if num_leaves < 2:
        raise ValueError("num_leaves must be at least two")

    unary_encodings = _UNARY_ENCODINGS[mode]
    num_binary = num_leaves - 1
    total = (
        num_inputs**num_leaves *
        len(unary_encodings)**num_leaves *
        len(_BINARY_ENCODINGS)**num_binary
    )
    if count > total:
        raise ValueError(f"requested {count} programs from a space of {total}")

    postorder = _balanced_postorder(num_leaves)
    max_program_size = (
        num_leaves * (2 + max(len(encoding) for encoding in unary_encodings)) +
        num_binary +
        1
    )
    output = bytearray(count * max_program_size)
    offsets = np.empty(count + 1, dtype=np.uint64)
    leaf_choices = [0] * num_leaves
    unary_choices = [0] * num_leaves
    binary_choices = [0] * num_binary
    offset, stride = _affine_permutation(total, seed)
    combination = offset
    write_offset = 0
    offsets[0] = 0

    for program_index in range(count):
        value = combination
        for index in range(num_leaves):
            leaf_choices[index] = value % num_inputs
            value //= num_inputs
        for index in range(num_leaves):
            unary_choices[index] = value % len(unary_encodings)
            value //= len(unary_encodings)
        for index in range(num_binary):
            binary_choices[index] = value % len(_BINARY_ENCODINGS)
            value //= len(_BINARY_ENCODINGS)

        for is_leaf, index in postorder:
            if is_leaf:
                output[write_offset] = secant.InstructionType.STATIC_COLUMN_INPUT_F32
                output[write_offset + 1] = leaf_choices[index]
                write_offset += 2
                encoding = unary_encodings[unary_choices[index]]
            else:
                encoding = _BINARY_ENCODINGS[binary_choices[index]]
            for byte in encoding:
                output[write_offset] = byte
                write_offset += 1

        output[write_offset] = secant.InstructionType.RETURN_F32
        write_offset += 1
        offsets[program_index + 1] = write_offset
        combination = (combination + stride) % total

    del output[write_offset:]
    return output, offsets
