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
"""Extract per-trial targets from the published SRBench v2 result artifact."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


TARGET_FIELDS = (
    "algorithm",
    "dataset",
    "random_state",
    "target_noise",
    "training_seconds",
    "r2_test",
    "model_size",
    "symbolic_solution",
    "symbolic_model",
)


def targets_select(
    results: pd.DataFrame,
    algorithm: str,
    data_group: str,
    target_noise: float,
    problems: set[str] | None,
) -> list[dict[str, object]]:
    required = {
        "algorithm",
        "dataset",
        "random_state",
        "target_noise",
        "data_group",
        "training time (s)",
        "r2_test",
        "model_size",
        "symbolic_solution",
        "symbolic_model",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"result artifact is missing columns: {', '.join(sorted(missing))}")

    selected = results[
        (results["algorithm"] == algorithm)
        & (results["data_group"] == data_group)
        & (results["target_noise"] == target_noise)
    ]
    if problems is not None:
        selected = selected[selected["dataset"].isin(problems)]

    records = []
    for _, row in selected.sort_values(["dataset", "random_state"]).iterrows():
        records.append({
            "algorithm": row["algorithm"],
            "dataset": row["dataset"],
            "random_state": int(row["random_state"]),
            "target_noise": float(row["target_noise"]),
            "training_seconds": float(row["training time (s)"]),
            "r2_test": float(row["r2_test"]),
            "model_size": int(row["model_size"]),
            "symbolic_solution": int(row["symbolic_solution"]),
            "symbolic_model": row["symbolic_model"],
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", default="Operon")
    parser.add_argument("--data-group", default="Feynman")
    parser.add_argument("--target-noise", type=float, default=0.0)
    parser.add_argument("--problems", nargs="+", default=None)
    args = parser.parse_args()

    records = targets_select(
        pd.read_feather(args.results),
        args.algorithm,
        args.data_group,
        args.target_noise,
        set(args.problems) if args.problems is not None else None,
    )
    if not records:
        parser.error("no matching result rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TARGET_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"targets={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
