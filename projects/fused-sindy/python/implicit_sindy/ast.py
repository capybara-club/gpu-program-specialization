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
from typing import Iterable, Sequence

import numpy as np


NUM_INPUTS = 8
NUM_UNARY_OPS = 15
NUM_BINARY_OPS = 7


class UnaryOp(IntEnum):
    IDENTITY = 0
    SQUARE_F32 = 1
    CUBE_F32 = 2
    NEG_FTZ_F32 = 3
    ABS_FTZ_F32 = 4
    RCP_APPROX_FTZ_F32 = 5
    SQRT_APPROX_FTZ_F32 = 6
    RSQRT_APPROX_FTZ_F32 = 7
    SIN_APPROX_FTZ_F32 = 8
    COS_APPROX_FTZ_F32 = 9
    EX2_APPROX_FTZ_F32 = 10
    EXP_APPROX_FTZ_F32 = 11
    LOG2_APPROX_FTZ_F32 = 12
    LOG10_APPROX_FTZ_F32 = 13
    SAFE_RCP_F32 = 14
    SAFE_SQRT_F32 = 15
    SAFE_RSQRT_F32 = 16
    SAFE_EX2_F32 = 17
    SAFE_EXP_F32 = 18
    SAFE_LOG2_F32 = 19
    SAFE_LOG10_F32 = 20
    ZERO_F32 = 21


class BinaryOp(IntEnum):
    ADD_FTZ_F32 = 0
    SUB_FTZ_F32 = 1
    MUL_FTZ_F32 = 2
    KEEP_LEFT = 3
    KEEP_RIGHT = 4
    DIV_APPROX_FTZ_F32 = 5
    MIN_FTZ_F32 = 6
    MAX_FTZ_F32 = 7
    SAFE_DIV_F32 = 8


@dataclass(frozen=True)
class BinaryAst:
    unary: tuple[int, ...]
    binary: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.unary) != NUM_UNARY_OPS:
            raise ValueError(f"unary must contain {NUM_UNARY_OPS} ops")
        if len(self.binary) != NUM_BINARY_OPS:
            raise ValueError(f"binary must contain {NUM_BINARY_OPS} ops")
        for op in self.unary:
            UnaryOp(int(op))
        for op in self.binary:
            BinaryOp(int(op))


def binary_ast(
    unary: Sequence[int | UnaryOp] | None = None,
    binary: Sequence[int | BinaryOp] | None = None,
) -> BinaryAst:
    if unary is None:
        unary = (UnaryOp.IDENTITY,) * NUM_UNARY_OPS
    if binary is None:
        binary = (BinaryOp.ADD_FTZ_F32,) * NUM_BINARY_OPS
    return BinaryAst(
        tuple(int(UnaryOp(int(op))) for op in unary),
        tuple(int(BinaryOp(int(op))) for op in binary),
    )


def padding_ast() -> BinaryAst:
    return binary_ast(
        unary=(UnaryOp.ZERO_F32,) * NUM_UNARY_OPS,
        binary=(BinaryOp.KEEP_LEFT,) * NUM_BINARY_OPS,
    )


def ast_arrays(asts: Iterable[BinaryAst]) -> tuple[np.ndarray, np.ndarray]:
    ast_list = list(asts)
    if not ast_list:
        raise ValueError("at least one AST is required")
    unary = np.empty((len(ast_list), NUM_UNARY_OPS), dtype=np.int32)
    binary = np.empty((len(ast_list), NUM_BINARY_OPS), dtype=np.int32)
    for idx, ast in enumerate(ast_list):
        if not isinstance(ast, BinaryAst):
            ast = binary_ast(ast.unary, ast.binary)  # type: ignore[attr-defined]
        unary[idx, :] = np.asarray(ast.unary, dtype=np.int32)
        binary[idx, :] = np.asarray(ast.binary, dtype=np.int32)
    return unary, binary


def f32_constant_word(value: float) -> np.int32:
    return np.asarray([value], dtype=np.float32).view(np.int32)[0]


def deterministic_test_cohort(count: int = 32) -> list[BinaryAst]:
    cohort: list[BinaryAst] = []
    binary_ops = list(BinaryOp)
    for idx in range(count):
        unary = []
        binary = []
        for op_idx in range(NUM_UNARY_OPS):
            if (idx + op_idx) % 7 == 0:
                unary.append(UnaryOp.CUBE_F32)
            elif (idx + op_idx) % 3 == 0:
                unary.append(UnaryOp.SQUARE_F32)
            else:
                unary.append(UnaryOp.IDENTITY)
        for op_idx in range(NUM_BINARY_OPS):
            binary.append(binary_ops[(idx + op_idx) % len(binary_ops)])
        cohort.append(binary_ast(unary, binary))
    return cohort


def random_ast_cohort(
    count: int = 32,
    seed: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    unary_ops: Sequence[int | UnaryOp] | None = None,
    binary_ops: Sequence[int | BinaryOp] | None = None,
) -> list[BinaryAst]:
    if count <= 0:
        raise ValueError("count must be positive")
    if rng is None:
        rng = np.random.default_rng(seed)

    if unary_ops is None:
        unary_choices = np.asarray([int(op) for op in UnaryOp], dtype=np.int32)
    else:
        unary_choices = np.asarray([int(UnaryOp(int(op))) for op in unary_ops], dtype=np.int32)
    if binary_ops is None:
        binary_choices = np.asarray([int(op) for op in BinaryOp], dtype=np.int32)
    else:
        binary_choices = np.asarray([int(BinaryOp(int(op))) for op in binary_ops], dtype=np.int32)
    if unary_choices.size == 0 or binary_choices.size == 0:
        raise ValueError("unary_ops and binary_ops must not be empty")

    cohort: list[BinaryAst] = []
    for _ in range(count):
        unary = rng.choice(unary_choices, size=NUM_UNARY_OPS, replace=True)
        binary = rng.choice(binary_choices, size=NUM_BINARY_OPS, replace=True)
        cohort.append(binary_ast(unary.tolist(), binary.tolist()))
    return cohort
