#!/usr/bin/env python3
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
"""SRBench v2.0-compatible symbolic assessment for Secant expressions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sympy
from sympy import Float, Integer, Symbol, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.traversal import preorder_traversal
import yaml


@dataclass(frozen=True)
class SymbolicAssessment:
    symbolic_solution: bool
    symbolic_error_is_zero: bool
    symbolic_error_is_constant: bool
    symbolic_fraction_is_constant: bool
    simplified_expression: str
    symbolic_error: str
    symbolic_fraction: str
    assessment_error: str


def _round_floats(expression):
    rounded = expression

    for value in preorder_traversal(expression):
        if isinstance(value, Float):
            replacement = Integer(0) if abs(value) < 0.0001 else Float(round(value, 3), 3)
            rounded = rounded.subs(value, replacement)
    return rounded


def _metadata_get(source_path: Path) -> tuple[list[str], str]:
    metadata_path = source_path.with_name("metadata.yaml")

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    feature_names = [feature["name"] for feature in metadata["features"]]
    formula_lines = [line for line in metadata["description"].splitlines() if "=" in line]
    if not formula_lines:
        raise ValueError(f"{metadata_path} does not contain a formula")
    return feature_names, formula_lines[0].split("=", 1)[1].strip()


def _neg(value):
    return -value


def _rcp(value):
    return 1 / value


def _rsqrt(value):
    return 1 / sympy.sqrt(value)


def _square(value):
    return value * value


def _cube(value):
    return value * value * value


def _exp2(value):
    return 2 ** value


def _log2(value):
    return sympy.log(value, 2)


def _expressions_get(source_path: Path, candidate_text: str):
    feature_names, true_text = _metadata_get(source_path)
    symbols = {name: Symbol(name) for name in feature_names}
    candidate_locals = {
        **{f"x{index}": symbols[name] for index, name in enumerate(feature_names)},
        "abs": sympy.Abs,
        "cos": sympy.cos,
        "cube": _cube,
        "exp": sympy.exp,
        "exp2": _exp2,
        "log": sympy.log,
        "log2": _log2,
        "max": sympy.Max,
        "min": sympy.Min,
        "neg": _neg,
        "rcp": _rcp,
        "rsqrt": _rsqrt,
        "sin": sympy.sin,
        "square": _square,
        "sqrt": sympy.sqrt,
        "tanh": sympy.tanh,
    }
    true_expression = parse_expr(true_text.replace("pi", "3.1415926535"), local_dict=symbols)
    candidate_expression = parse_expr(candidate_text, local_dict=candidate_locals)
    true_expression = _round_floats(true_expression)
    candidate_expression = simplify(_round_floats(candidate_expression), ratio=1)
    return true_expression, candidate_expression


def assess_expression(source_path: Path, candidate_text: str, validation_r2: float) -> SymbolicAssessment:
    """Apply the exact symbolic-solution rules used by SRBench v2.0."""
    try:
        if validation_r2 <= 0.5:
            raise ValueError("model is not accurate enough for symbolic assessment")
        true_expression, candidate_expression = _expressions_get(source_path, candidate_text)
        candidate_string = str(candidate_expression)
        if candidate_string in ("0", "nan"):
            raise ValueError(f"candidate simplifies to {candidate_string}")

        symbolic_error = _round_floats(true_expression - candidate_expression)
        symbolic_fraction = _round_floats(candidate_expression / true_expression)
        if not symbolic_error.is_constant() or symbolic_fraction.is_constant():
            symbolic_error = _round_floats(simplify(symbolic_error, ratio=1))

        error_is_zero = str(symbolic_error) == "0"
        error_is_constant = bool(symbolic_error.is_constant())
        fraction_is_constant = bool(symbolic_fraction.is_constant())
        return SymbolicAssessment(
            symbolic_solution=error_is_zero or error_is_constant or fraction_is_constant,
            symbolic_error_is_zero=error_is_zero,
            symbolic_error_is_constant=error_is_constant,
            symbolic_fraction_is_constant=fraction_is_constant,
            simplified_expression=candidate_string,
            symbolic_error=str(symbolic_error),
            symbolic_fraction=str(symbolic_fraction),
            assessment_error="",
        )
    except Exception as error:
        return SymbolicAssessment(False, False, False, False, "", "", "", str(error))
