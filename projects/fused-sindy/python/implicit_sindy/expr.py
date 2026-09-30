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
from typing import Sequence

import numpy as np

from .ast import (
    NUM_BINARY_OPS,
    NUM_INPUTS,
    NUM_UNARY_OPS,
    BinaryAst,
    BinaryOp,
    UnaryOp,
    binary_ast,
    f32_constant_word,
)


@dataclass(frozen=True)
class LeafBinding:
    is_feature: bool
    word: int


class Expr:
    def __add__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.ADD_FTZ_F32, self, as_expr(other))

    def __radd__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.ADD_FTZ_F32, as_expr(other), self)

    def __sub__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.SUB_FTZ_F32, self, as_expr(other))

    def __rsub__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.SUB_FTZ_F32, as_expr(other), self)

    def __mul__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.MUL_FTZ_F32, self, as_expr(other))

    def __rmul__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.MUL_FTZ_F32, as_expr(other), self)

    def __truediv__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.DIV_APPROX_FTZ_F32, self, as_expr(other))

    def __rtruediv__(self, other: object) -> Expr:
        return BinaryExpr(BinaryOp.DIV_APPROX_FTZ_F32, as_expr(other), self)

    def __neg__(self) -> Expr:
        return UnaryExpr(UnaryOp.NEG_FTZ_F32, self)

    def __abs__(self) -> Expr:
        return UnaryExpr(UnaryOp.ABS_FTZ_F32, self)

    def to_ast_and_leaf_settings(self) -> tuple[BinaryAst, np.int32, np.ndarray]:
        return expression_to_ast_and_leaf_settings(self)


@dataclass(frozen=True)
class FeatureExpr(Expr):
    column: int


@dataclass(frozen=True)
class ConstantExpr(Expr):
    value: float


@dataclass(frozen=True)
class UnaryExpr(Expr):
    op: UnaryOp
    child: Expr


@dataclass(frozen=True)
class BinaryExpr(Expr):
    op: BinaryOp
    left: Expr
    right: Expr


def feature(column: int) -> Expr:
    if column < 0:
        raise ValueError("feature column must be non-negative")
    return FeatureExpr(int(column))


def const(value: float) -> Expr:
    return ConstantExpr(float(value))


def square(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SQUARE_F32, as_expr(expr))


def cube(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.CUBE_F32, as_expr(expr))


def neg(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.NEG_FTZ_F32, as_expr(expr))


def abs_f32(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.ABS_FTZ_F32, as_expr(expr))


def rcp(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.RCP_APPROX_FTZ_F32, as_expr(expr))


def safe_rcp(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_RCP_F32, as_expr(expr))


def sqrt(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SQRT_APPROX_FTZ_F32, as_expr(expr))


def safe_sqrt(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_SQRT_F32, as_expr(expr))


def rsqrt(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.RSQRT_APPROX_FTZ_F32, as_expr(expr))


def safe_rsqrt(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_RSQRT_F32, as_expr(expr))


def sin(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SIN_APPROX_FTZ_F32, as_expr(expr))


def cos(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.COS_APPROX_FTZ_F32, as_expr(expr))


def ex2(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.EX2_APPROX_FTZ_F32, as_expr(expr))


def safe_ex2(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_EX2_F32, as_expr(expr))


def exp(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.EXP_APPROX_FTZ_F32, as_expr(expr))


def safe_exp(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_EXP_F32, as_expr(expr))


def log2(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.LOG2_APPROX_FTZ_F32, as_expr(expr))


def safe_log2(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_LOG2_F32, as_expr(expr))


def log10(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.LOG10_APPROX_FTZ_F32, as_expr(expr))


def safe_log10(expr: object) -> Expr:
    return UnaryExpr(UnaryOp.SAFE_LOG10_F32, as_expr(expr))


def safe_div(left: object, right: object) -> Expr:
    return BinaryExpr(BinaryOp.SAFE_DIV_F32, as_expr(left), as_expr(right))


def minimum(left: object, right: object) -> Expr:
    return BinaryExpr(BinaryOp.MIN_FTZ_F32, as_expr(left), as_expr(right))


def maximum(left: object, right: object) -> Expr:
    return BinaryExpr(BinaryOp.MAX_FTZ_F32, as_expr(left), as_expr(right))


def as_expr(value: object) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return const(float(value))
    raise TypeError(f"cannot convert {type(value).__name__} to Expr")


def expression_to_ast_and_leaf_settings(expr: object) -> tuple[BinaryAst, np.int32, np.ndarray]:
    lowerer = _Lowerer()
    lowerer.lower(as_expr(expr), 14)
    return (
        binary_ast(lowerer.unary, lowerer.binary),
        np.int32(lowerer.leaf_mask),
        np.asarray(lowerer.leaf_words, dtype=np.int32),
    )


def expressions_to_asts_and_settings(
    expressions: Sequence[object],
) -> tuple[list[BinaryAst], np.ndarray, np.ndarray]:
    if len(expressions) == 0:
        raise ValueError("at least one expression is required")
    asts: list[BinaryAst] = []
    leaf_masks = np.zeros((1, len(expressions)), dtype=np.int32)
    leaf_words = np.zeros((1, len(expressions), NUM_INPUTS), dtype=np.int32)
    for idx, expr in enumerate(expressions):
        ast, mask, words = expression_to_ast_and_leaf_settings(expr)
        asts.append(ast)
        leaf_masks[0, idx] = mask
        leaf_words[0, idx, :] = words
    return asts, leaf_masks, leaf_words


class _Lowerer:
    def __init__(self) -> None:
        self.unary = [int(UnaryOp.IDENTITY)] * NUM_UNARY_OPS
        self.binary = [int(BinaryOp.KEEP_LEFT)] * NUM_BINARY_OPS
        self.leaf_mask = 0
        self.leaf_words = [int(f32_constant_word(0.0))] * NUM_INPUTS

    def lower(self, expr: Expr, node_idx: int) -> None:
        if isinstance(expr, UnaryExpr):
            self.lower(expr.child, node_idx)
            if self.unary[node_idx] != int(UnaryOp.IDENTITY):
                raise ValueError("expression has stacked unary ops at one fixed AST node")
            self.unary[node_idx] = int(expr.op)
            return

        if isinstance(expr, BinaryExpr):
            if node_idx < NUM_INPUTS:
                raise ValueError("expression exceeds fixed depth-3 binary AST")
            binary_idx, left_idx, right_idx = _children(node_idx)
            self.binary[binary_idx] = int(expr.op)
            self.lower(expr.left, left_idx)
            self.lower(expr.right, right_idx)
            return

        if isinstance(expr, (FeatureExpr, ConstantExpr)):
            if node_idx < NUM_INPUTS:
                self.set_leaf(expr, node_idx)
                return
            binary_idx, left_idx, right_idx = _children(node_idx)
            self.binary[binary_idx] = int(BinaryOp.KEEP_LEFT)
            self.lower(expr, left_idx)
            self.lower(ConstantExpr(0.0), right_idx)
            return

        raise TypeError(f"unsupported expression node {type(expr).__name__}")

    def set_leaf(self, expr: FeatureExpr | ConstantExpr, leaf_idx: int) -> None:
        if isinstance(expr, FeatureExpr):
            self.leaf_mask |= 1 << leaf_idx
            self.leaf_words[leaf_idx] = int(expr.column)
        else:
            self.leaf_words[leaf_idx] = int(f32_constant_word(expr.value))


def _children(ast_idx: int) -> tuple[int, int, int]:
    if 8 <= ast_idx < 12:
        local = ast_idx - 8
        return local, local * 2, local * 2 + 1
    if 12 <= ast_idx < 14:
        local = ast_idx - 12
        return 4 + local, 8 + local * 2, 8 + local * 2 + 1
    if ast_idx == 14:
        return 6, 12, 13
    raise ValueError(f"node {ast_idx} does not have children")
