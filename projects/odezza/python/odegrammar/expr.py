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
"""Small, safe expression parser: Python syntax is inspected, never executed."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Iterator, Mapping


class ExpressionError(ValueError):
    """An expression is outside the supported scalar grammar."""


@dataclass(frozen=True)
class Node:
    op: str
    value: object = None
    children: tuple["Node", ...] = ()


_NAMESPACES = {
    "leaf": "leaf", "const": "const", "rng": "rng",
    "theta": "param", "param": "param", "shape": "shape",
}
_UNARY_FUNCTIONS = {"sin", "cos", "tanh", "exp", "log", "sqrt", "abs"}
_BINARY = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


def _integer_literal(node: ast.AST) -> int:
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return sign * node.value
    raise ExpressionError("pow and ** require a literal integer exponent")


def _convert(node: ast.AST) -> Node:
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ExpressionError("Only finite real numeric literals are allowed")
        try:
            finite = math.isfinite(node.value)
        except OverflowError:
            finite = False
        if not finite:
            raise ExpressionError("Numeric literals must be finite and representable")
        return Node("literal", node.value)
    if isinstance(node, ast.Name):
        return Node("symbol", node.id)
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id not in _NAMESPACES:
            raise ExpressionError("References must be leaf.name, const.name, rng.name, theta.name, or shape.name")
        return Node(_NAMESPACES[node.value.id], node.attr)
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ExpressionError("Only unary + and - are supported")
        operand = _convert(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if operand.op == "literal":
            return Node("literal", -operand.value)
        return Node("neg", children=(operand,))
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            return Node("powi", _integer_literal(node.right), (_convert(node.left),))
        op = _BINARY.get(type(node.op))
        if op is None:
            raise ExpressionError("Supported operators are +, -, *, /, and ** (use pow(x, 2), not x^2)")
        return Node(op, children=(_convert(node.left), _convert(node.right)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ExpressionError("Only named calls with positional arguments are supported")
        name = node.func.id
        if name in _UNARY_FUNCTIONS:
            if len(node.args) != 1:
                raise ExpressionError(f"{name} requires exactly one argument")
            return Node(name, children=(_convert(node.args[0]),))
        if name == "pow":
            if len(node.args) != 2:
                raise ExpressionError("pow requires two arguments")
            return Node("powi", _integer_literal(node.args[1]), (_convert(node.args[0]),))
        if name == "hole":
            if not node.args or not isinstance(node.args[0], ast.Name):
                raise ExpressionError("Use hole(RuleName, argument, ...) with a bare rule name")
            return Node("hole", node.args[0].id, tuple(_convert(a) for a in node.args[1:]))
        # A user rule can be written as R(x0) as well as hole(R, x0).
        # The expander validates the rule name and its parameter count.
        return Node("hole", name, tuple(_convert(a) for a in node.args))
    raise ExpressionError(f"Unsupported expression syntax: {type(node).__name__}")


def parse_expr(text: str) -> Node:
    """Parse safe infix syntax into an immutable expression tree."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("An expression must be a non-empty string")
    try:
        return _convert(ast.parse(text.strip(), mode="eval").body)
    except (SyntaxError, RecursionError) as exc:
        raise ExpressionError(f"Invalid expression: {exc}") from exc


def substitute(node: Node, mapping: Mapping[str, Node]) -> Node:
    """Substitute formal parameters (bare symbols), leaving namespaces intact."""
    if node.op == "symbol" and node.value in mapping:
        return mapping[node.value]
    if not node.children:
        return node
    return Node(node.op, node.value, tuple(substitute(c, mapping) for c in node.children))


def rename_refs(node: Node, mapping: Mapping[tuple[str, str], str]) -> Node:
    """Rename qualified slots without changing bare state/formal symbols."""
    value = mapping.get((node.op, node.value), node.value) if isinstance(node.value, str) else node.value
    return Node(node.op, value, tuple(rename_refs(c, mapping) for c in node.children))


def walk(node: Node) -> Iterator[Node]:
    """Walk in preorder; tag wrappers remain visible to consumers."""
    yield node
    for child in node.children:
        yield from walk(child)


def node_count(node: Node) -> int:
    """Count executable expression nodes; provenance tags have no cost."""
    return (0 if node.op == "tag" else 1) + sum(node_count(c) for c in node.children)


def node_depth(node: Node) -> int:
    return (0 if node.op == "tag" else 1) + max((node_depth(c) for c in node.children), default=0)
