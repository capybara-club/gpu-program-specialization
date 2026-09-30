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
"""Audit current secant-sr grammar coverage of SRBench Feynman formulas."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import pandas as pd
import sympy as sp


# Minimum postorder node costs using the fixed constant vocabulary in app/search.c.
CONSTANT_NODE_COST = {
    sp.pi: 1,
    sp.Rational(-32, 5): 13,
    sp.Rational(-5): 3,
    sp.Rational(-4): 3,
    sp.Rational(-3): 1,
    sp.Rational(-2): 1,
    sp.Rational(-3, 2): 1,
    sp.Rational(-1): 1,
    sp.Rational(-1, 2): 1,
    sp.Rational(-1, 3): 3,
    sp.Rational(-1, 4): 1,
    sp.Rational(-1, 8): 3,
    sp.Rational(1, 16): 3,
    sp.Rational(1, 8): 3,
    sp.Rational(3, 20): 7,
    sp.Rational(1, 6): 3,
    sp.Rational(1, 4): 1,
    sp.Rational(1, 3): 3,
    sp.Rational(3, 8): 3,
    sp.Rational(1, 2): 1,
    sp.Rational(3, 4): 3,
    sp.Rational(1): 1,
    sp.Rational(4, 3): 5,
    sp.Rational(3, 2): 1,
    sp.Rational(2): 1,
    sp.Rational(8, 3): 7,
    sp.Rational(3): 1,
    sp.Rational(4): 3,
    sp.Rational(8): 5,
}


def _formula_get(metadata_path: Path) -> str:
    for line in metadata_path.read_text().splitlines():
        if "=" in line:
            return line.strip().split("=", 1)[1].strip()
    raise ValueError(f"no formula in {metadata_path}")


def _features_get(dataset_path: Path) -> list[str]:
    with gzip.open(dataset_path, "rt") as source:
        columns = source.readline().strip().split("\t")
    if len(columns) < 2 or columns[-1] != "target":
        raise ValueError(f"invalid target column in {dataset_path}")
    return columns[:-1]


def _lowered_node_count(expression: sp.Expr) -> tuple[int, list[str]]:
    if isinstance(expression, sp.Symbol):
        return 1, []
    if expression == sp.pi:
        return CONSTANT_NODE_COST[sp.pi], []
    if expression.is_Number:
        if expression in CONSTANT_NODE_COST:
            return CONSTANT_NODE_COST[expression], []
        return 0, [f"constant:{expression}"]
    if expression.func in (sp.Add, sp.Mul):
        lowered = [_lowered_node_count(argument) for argument in expression.args]
        return (
            sum(count for count, _ in lowered) + len(lowered) - 1,
            sum((issues for _, issues in lowered), []),
        )
    if expression.func == sp.Pow:
        base, exponent = expression.args
        base_count, issues = _lowered_node_count(base)
        if not exponent.is_Rational or int(exponent.q) not in (1, 2):
            return 0, issues + [f"power:{exponent}"]
        numerator = int(exponent.p)
        denominator = int(exponent.q)
        magnitude = abs(numerator)
        if denominator == 1:
            count = 1 if numerator == 0 else base_count * magnitude + magnitude - 1
        else:
            integer_factors = (magnitude - 1) // 2
            count = (integer_factors + 1) * base_count + integer_factors + 1
        if numerator < 0:
            count += 1
        return count, issues * max(1, (magnitude + 1) // 2)
    if expression.func in (sp.sin, sp.cos, sp.tanh, sp.Abs, sp.exp, sp.log):
        count, issues = _lowered_node_count(expression.args[0])
        return count + 1, issues
    if expression.func in (sp.Min, sp.Max):
        lowered = [_lowered_node_count(argument) for argument in expression.args]
        return (
            sum(count for count, _ in lowered) + len(lowered) - 1,
            sum((issues for _, issues in lowered), []),
        )
    return 0, [f"function:{expression.func.__name__}"]


def audit(pmlb_root: Path, srbench_root: Path) -> list[dict[str, str | int]]:
    manifest = pd.read_csv(srbench_root / "docs" / "csv" / "groundtruth.csv")
    names = sorted(manifest.loc[manifest["data_group"] == "Feynman", "dataset"].unique())
    records: list[dict[str, str | int]] = []
    for name in names:
        dataset_dir = pmlb_root / "datasets" / name
        dataset_path = dataset_dir / f"{name}.tsv.gz"
        features = _features_get(dataset_path)
        formula = _formula_get(dataset_dir / "metadata.yaml")
        expression = sp.sympify(
            formula.replace("^", "**"),
            locals={**{feature: sp.Symbol(feature) for feature in features}, "ln": sp.log},
        )
        node_count, issues = _lowered_node_count(expression)
        unique_issues = sorted(set(issues))
        if unique_issues:
            status = "unsupported"
        elif node_count <= 15:
            status = "dynamic_leaf_node_limit"
        elif node_count <= 72:
            status = "ordinary_node_limit"
        else:
            status = "over_node_limit"
        records.append({
            "problem": name,
            "num_inputs": len(features),
            "lowered_nodes": node_count,
            "status": status,
            "issues": ";".join(unique_issues),
            "formula": formula,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmlb-root", type=Path, required=True)
    parser.add_argument("--srbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-problems", choices=("all", "dynamic", "ordinary", "unsupported"), default=None)
    args = parser.parse_args()
    records = audit(args.pmlb_root, args.srbench_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    if args.print_problems is not None:
        accepted = {
            "all": {"dynamic_leaf_node_limit", "ordinary_node_limit", "unsupported", "over_node_limit"},
            "dynamic": {"dynamic_leaf_node_limit"},
            "ordinary": {"dynamic_leaf_node_limit", "ordinary_node_limit"},
            "unsupported": {"unsupported", "over_node_limit"},
        }[args.print_problems]
        print(" ".join(str(record["problem"]) for record in records if record["status"] in accepted))
    else:
        counts = {status: sum(record["status"] == status for record in records) for status in (
            "dynamic_leaf_node_limit", "ordinary_node_limit", "unsupported", "over_node_limit"
        )}
        print(f"datasets={len(records)} " + " ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
