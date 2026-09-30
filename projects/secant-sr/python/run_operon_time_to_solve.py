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
"""Bracket Operon time-to-solution using increasing wall-clock budgets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_operon_suite import DEFAULT_PROBLEMS, PROBLEM_INPUTS, run_one


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(1, 11)))
    parser.add_argument("--budgets", type=int, nargs="+", default=(1, 2, 5, 10, 30, 60))
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--max-evaluations", type=int, default=100_000_000)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--optimizer-iterations", type=int, default=5)
    parser.add_argument("--optimize-probability", type=float, default=1.0)
    parser.add_argument("--early-stop-error", type=float, default=1.0e-7)
    parser.add_argument("--stop-r2", type=float, default=0.999999)
    parser.add_argument("--output", type=Path, default=Path("scratch/suite/operon_time_to_solve.csv"))
    args = parser.parse_args()

    if not args.budgets or any(value <= 0 for value in args.budgets):
        raise ValueError("budgets must contain positive integer seconds")
    args.budgets = tuple(sorted(set(args.budgets)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "problem", "seed", "solved", "solve_budget_seconds", "fit_seconds_total", "attempts",
        "train_r2", "validation_r2", "complexity", "evaluations", "generations", "best_expression",
    )
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for problem in args.problems:
            if problem not in PROBLEM_INPUTS:
                raise ValueError(f"unknown problem: {problem}")
            for seed in args.seeds:
                fit_seconds_total = 0.0
                record = None
                attempts = 0
                for budget in args.budgets:
                    args.seconds = budget
                    record = run_one(args, problem, seed)
                    attempts += 1
                    fit_seconds_total += float(record["fit_seconds"])
                    if int(record["solved"]):
                        break
                assert record is not None
                output_record = {
                    "problem": problem,
                    "seed": seed,
                    "solved": record["solved"],
                    "solve_budget_seconds": args.seconds,
                    "fit_seconds_total": fit_seconds_total,
                    "attempts": attempts,
                    "train_r2": record["train_r2"],
                    "validation_r2": record["validation_r2"],
                    "complexity": record["complexity"],
                    "evaluations": record["evaluations"],
                    "generations": record["generations"],
                    "best_expression": record["best_expression"],
                }
                writer.writerow(output_record)
                output.flush()
                print(
                    f"problem={problem} seed={seed} validation_r2={record['validation_r2']:.9g} "
                    f"solved={record['solved']} solve_budget_seconds={args.seconds} "
                    f"fit_seconds_total={fit_seconds_total:.3f} attempts={attempts}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
