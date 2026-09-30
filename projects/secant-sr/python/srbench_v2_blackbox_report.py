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
"""Summarize Secant-SR predictive results on SRBench v2.0 black-box data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def _number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"missing {name!r} in result row for {row.get('problem', '<unknown>')}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--official-results", type=Path, default=None)
    parser.add_argument("--max-inputs", type=int, default=0,
        help="exclude Secant rows above this input count and match official rows to the retained datasets")
    args = parser.parse_args()
    if args.max_inputs < 0:
        parser.error("--max-inputs must be nonnegative")

    rows: list[dict[str, str]] = []
    for path in args.results:
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    excluded_datasets: set[str] = set()
    if args.max_inputs != 0:
        excluded_datasets = {
            row["problem"] for row in rows if int(_number(row, "num_inputs")) > args.max_inputs
        }
        rows = [row for row in rows if int(_number(row, "num_inputs")) <= args.max_inputs]
    if not rows:
        parser.error("result files contain no records")

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["problem"]].append(row)
    median_r2 = [statistics.median(_number(row, "validation_r2") for row in group)
                 for group in by_dataset.values()]
    median_nodes = [statistics.median(_number(row, "nodes") for row in group)
                    for group in by_dataset.values()]
    elapsed = [_number(row, "elapsed_seconds") for row in rows]

    if args.max_inputs != 0:
        print(
            f"Scope: Secant datasets with at most {args.max_inputs} inputs "
            f"({len(excluded_datasets)} represented wider datasets excluded).\n"
        )

    print("| Trials | Datasets | Mean dataset-median R2 | Median dataset-median R2 | "
          "Mean floored R2 | Mean dataset-median nodes | Total fit time | Mean fit time |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {len(rows)} | {len(by_dataset)} | {statistics.mean(median_r2):.6f} | "
        f"{statistics.median(median_r2):.6f} | {statistics.mean(max(value, 0.0) for value in median_r2):.6f} | "
        f"{statistics.mean(median_nodes):.2f} | {sum(elapsed):.3f} s | {statistics.mean(elapsed):.3f} s |"
    )

    if args.official_results is not None:
        with args.official_results.open(newline="") as source:
            official = list(csv.DictReader(source))
        if args.max_inputs != 0:
            retained_datasets = set(by_dataset)
            official = [row for row in official if row["dataset"] in retained_datasets]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in official:
            grouped[row["algorithm"]].append(row)
        print("\nPublished SRBench v2.0 aggregates on the same represented dataset subset:\n")
        print("| Algorithm | Datasets | Mean R2 | Median R2 | Mean floored R2 | Mean model size | Mean fit time |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for algorithm, group in sorted(
            grouped.items(),
            key=lambda item: statistics.mean(max(_number(row, "r2_test"), 0.0) for row in item[1]),
            reverse=True,
        ):
            r2 = [_number(row, "r2_test") for row in group]
            sizes = [_number(row, "model_size") for row in group if row.get("model_size", "") != ""]
            times = [_number(row, "training time (s)") for row in group]
            print(
                f"| {algorithm} | {len(group)} | {statistics.mean(r2):.6f} | {statistics.median(r2):.6f} | "
                f"{statistics.mean(max(value, 0.0) for value in r2):.6f} | "
                f"{statistics.mean(sizes):.2f} | {statistics.mean(times):.3f} s |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
