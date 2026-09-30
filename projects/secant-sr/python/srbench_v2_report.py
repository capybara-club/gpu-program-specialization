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
"""Summarize Secant-SR results under the SRBench v2.0 protocol."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--official-results", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for path in args.results:
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    if not rows:
        parser.error("result files contain no records")

    print(
        "| Target noise | Trials | Datasets | Median-trial R2 > 0.999 | "
        "Trial R2 > 0.999 | Symbolic recovery | Total time | Mean fit time |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for noise in sorted({float(row["target_noise"]) for row in rows}):
        group = [row for row in rows if float(row["target_noise"]) == noise]
        datasets = {row["problem"] for row in group}
        seeds = {int(row["seed"]) for row in group}
        dataset_accuracy = statistics.mean(
            statistics.median(
                int(row["accuracy_solution"])
                for row in group
                if row["problem"] == dataset
            )
            for dataset in datasets
        )
        trial_accuracy = statistics.mean(int(row["accuracy_solution"]) for row in group)
        symbolic_values = [int(row["symbolic_solution"]) for row in group if row["symbolic_solution"] != ""]
        symbolic_text = (
            f"{100.0 * statistics.mean(symbolic_values):.2f}%"
            if len(symbolic_values) == len(group)
            else "not assessed"
        )
        elapsed = [float(row["elapsed_seconds"]) for row in group]
        print(
            f"| {noise:g} | {len(seeds)} | {len(datasets)} | {100.0 * dataset_accuracy:.2f}% | "
            f"{100.0 * trial_accuracy:.2f}% | {symbolic_text} | {sum(elapsed):.3f} s | "
            f"{statistics.mean(elapsed):.3f} s |"
        )
    if args.official_results is not None:
        result_noises = {float(row["target_noise"]) for row in rows}
        with args.official_results.open(newline="") as source:
            official_rows = [
                row for row in csv.DictReader(source)
                if row["data_group"] == "Feynman" and float(row["target_noise"]) in result_noises
            ]
        grouped = defaultdict(list)
        for row in official_rows:
            grouped[(float(row["target_noise"]), row["algorithm"])].append(row)
        print("\nPublished SRBench v2.0 aggregates:\n")
        print("| Target noise | Algorithm | Datasets | Median-trial R2 > 0.999 | Mean symbolic recovery |")
        print("|---:|---|---:|---:|---:|")
        ordered = sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0],
                -statistics.mean(float(row["symbolic_solution_rate_(%)"]) for row in item[1]),
            ),
        )
        for (noise, algorithm), group in ordered:
            accuracy = statistics.mean(float(row["accuracy_solution"]) for row in group)
            symbolic = statistics.mean(float(row["symbolic_solution_rate_(%)"]) for row in group)
            print(f"| {noise:g} | {algorithm} | {len(group)} | {100.0 * accuracy:.2f}% | {symbolic:.2f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
