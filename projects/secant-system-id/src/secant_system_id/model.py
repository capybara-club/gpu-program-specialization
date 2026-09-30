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
from typing import Iterable

from .shape import KernelShape


class CudaExpressionKind(IntEnum):
    LITERAL = 0
    STATE = 1
    MISSING = 2
    ADD = 3
    SUBTRACT = 4
    MULTIPLY = 5
    DIVIDE = 6
    NEGATE = 7
    CALL = 8


@dataclass(frozen=True)
class CudaExpression:
    """A small typed expression tree rendered into the known CUDA backbone."""

    kind: CudaExpressionKind
    value: float | int | str
    arguments: tuple["CudaExpression", ...] = ()

    def __add__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.ADD, self, other)

    def __radd__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.ADD, other, self)

    def __sub__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.SUBTRACT, self, other)

    def __rsub__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.SUBTRACT, other, self)

    def __mul__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.MULTIPLY, self, other)

    def __rmul__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.MULTIPLY, other, self)

    def __truediv__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.DIVIDE, self, other)

    def __rtruediv__(self, other: "CudaExpression | float | int") -> "CudaExpression":
        return _binary(CudaExpressionKind.DIVIDE, other, self)

    def __neg__(self) -> "CudaExpression":
        return CudaExpression(CudaExpressionKind.NEGATE, 0, (self,))

    def references(self) -> Iterable[tuple[CudaExpressionKind, int]]:
        if self.kind in {CudaExpressionKind.STATE, CudaExpressionKind.MISSING}:
            yield self.kind, int(self.value)
        for argument in self.arguments:
            yield from argument.references()

    def render(self) -> str:
        if self.kind == CudaExpressionKind.LITERAL:
            value = float(self.value)
            if not math.isfinite(value):
                raise ValueError("known CUDA literals must be finite")
            rendered = format(value, ".9g")
            if "." not in rendered and "e" not in rendered.lower():
                rendered += ".0"
            return rendered + "f"
        if self.kind == CudaExpressionKind.STATE:
            return f"stage_state[{int(self.value)}]"
        if self.kind == CudaExpressionKind.MISSING:
            return f"missing_output{int(self.value)}"
        if self.kind == CudaExpressionKind.NEGATE:
            return f"(-{self.arguments[0].render()})"
        if self.kind == CudaExpressionKind.CALL:
            return f"{self.value}({', '.join(argument.render() for argument in self.arguments)})"
        operator = {
            CudaExpressionKind.ADD: "+",
            CudaExpressionKind.SUBTRACT: "-",
            CudaExpressionKind.MULTIPLY: "*",
            CudaExpressionKind.DIVIDE: "/",
        }.get(self.kind)
        if operator is None or len(self.arguments) != 2:
            raise ValueError("malformed known CUDA expression")
        return f"({self.arguments[0].render()} {operator} {self.arguments[1].render()})"


def _expression(value: CudaExpression | float | int) -> CudaExpression:
    if isinstance(value, CudaExpression):
        return value
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("known-equation operands must be CUDA expressions or real numbers")
    return literal(value)


def _binary(
    kind: CudaExpressionKind,
    lhs: CudaExpression | float | int,
    rhs: CudaExpression | float | int,
) -> CudaExpression:
    return CudaExpression(kind, 0, (_expression(lhs), _expression(rhs)))


def literal(value: float | int) -> CudaExpression:
    return CudaExpression(CudaExpressionKind.LITERAL, float(value))


def state(index: int) -> CudaExpression:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("state index must be a non-negative integer")
    return CudaExpression(CudaExpressionKind.STATE, index)


def missing(index: int) -> CudaExpression:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("missing-site index must be a non-negative integer")
    return CudaExpression(CudaExpressionKind.MISSING, index)


def cuda_call(name: str, *arguments: CudaExpression | float | int) -> CudaExpression:
    allowed = {"fabsf", "fmaxf", "fminf", "expf", "logf", "sqrtf", "tanhf"}
    if name not in allowed:
        raise ValueError(f"CUDA call {name!r} is not in the known-equation whitelist")
    return CudaExpression(CudaExpressionKind.CALL, name, tuple(_expression(arg) for arg in arguments))


@dataclass(frozen=True)
class MissingSite:
    name: str
    leaf_count: int

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("missing-site names must be non-empty identifiers")
        if isinstance(self.leaf_count, bool) or not isinstance(self.leaf_count, int) or self.leaf_count <= 0:
            raise ValueError("missing-site leaf_count must be positive")


@dataclass(frozen=True)
class SystemModel:
    name: str
    state_names: tuple[str, ...]
    missing_sites: tuple[MissingSite, ...]
    derivatives: tuple[CudaExpression, ...]
    observation_interval: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("model name must be a non-empty identifier")
        if not self.state_names or len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state names must be non-empty and unique")
        if any(not name.isidentifier() for name in self.state_names):
            raise ValueError("state names must be valid identifiers")
        if not self.missing_sites or len({site.name for site in self.missing_sites}) != len(self.missing_sites):
            raise ValueError("a model requires uniquely named missing sites")
        if len(self.derivatives) != len(self.state_names):
            raise ValueError("one known derivative expression is required per state")
        if not math.isfinite(self.observation_interval) or self.observation_interval <= 0.0:
            raise ValueError("observation_interval must be finite and positive")
        for derivative in self.derivatives:
            for kind, index in derivative.references():
                if kind == CudaExpressionKind.STATE and index >= self.state_count:
                    raise ValueError("known derivative references a nonexistent state")
                if kind == CudaExpressionKind.MISSING and index >= self.missing_count:
                    raise ValueError("known derivative references a nonexistent missing site")

    @property
    def state_count(self) -> int:
        return len(self.state_names)

    @property
    def missing_count(self) -> int:
        return len(self.missing_sites)

    @property
    def ast_leaf_counts(self) -> tuple[int, ...]:
        return tuple(site.leaf_count for site in self.missing_sites)

    def make_shape(
        self,
        *,
        constant_count: int,
        trajectory_count: int,
        observation_count: int,
        patch_capacity: int = 192,
    ) -> KernelShape:
        return KernelShape(
            ast_leaf_counts=self.ast_leaf_counts,
            state_count=self.state_count,
            constant_count=constant_count,
            trajectory_count=trajectory_count,
            observation_count=observation_count,
            patch_capacity=patch_capacity,
        )
