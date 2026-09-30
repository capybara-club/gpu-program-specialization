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
"""Run Operon over the same deterministic problems and splits as secant-sr."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from run_pysr_suite import DEFAULT_PROBLEMS, PROBLEM_INPUTS, problem_data, r2_get


def run_one(args: argparse.Namespace, problem: str, seed: int) -> dict[str, int | float | str]:
    from pyoperon.sklearn import SymbolicRegressor

    train_x, train_y = problem_data(problem, seed ^ 0x747261696E, args.rows)
    validation_x, validation_y = problem_data(problem, seed ^ 0x76616C6964617465, args.validation_rows)
    model = SymbolicRegressor(
        allowed_symbols="add,sub,mul,div,fmin,fmax,abs,sqrt,sin,cos,tanh,exp,log,constant,variable",
        objectives=["r2"],
        optimizer="lm",
        optimizer_iterations=args.optimizer_iterations,
        local_search_probability=args.optimize_probability,
        lamarckian_probability=1.0,
        population_size=args.population_size,
        pool_size=args.population_size,
        generations=1_000_000,
        max_evaluations=args.max_evaluations,
        max_length=args.max_length,
        max_depth=args.max_depth,
        epsilon=args.early_stop_error,
        model_selection_criterion="mean_squared_error",
        n_threads=args.threads,
        max_time=args.seconds,
        random_state=seed,
    )
    begin = time.monotonic()
    model.fit(train_x.astype(np.float64), train_y.astype(np.float64))
    elapsed = time.monotonic() - begin
    train_r2 = r2_get(train_y, model.predict(train_x.astype(np.float64)))
    validation_r2 = r2_get(validation_y, model.predict(validation_x.astype(np.float64)))
    best = model.pareto_front_[0]
    return {
        "problem": problem,
        "seed": seed,
        "train_r2": train_r2,
        "validation_r2": validation_r2,
        "complexity": int(best["complexity"]),
        "solved": int(validation_r2 > args.stop_r2),
        "fit_seconds": elapsed,
        "evaluations": int(model.stats_["evaluation_count"]),
        "generations": int(model.stats_["generations"]),
        "best_expression": str(best["model"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(1, 11)))
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--max-evaluations", type=int, default=100_000_000)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--optimizer-iterations", type=int, default=5)
    parser.add_argument("--optimize-probability", type=float, default=1.0)
    parser.add_argument("--early-stop-error", type=float, default=1.0e-7)
    parser.add_argument("--stop-r2", type=float, default=0.999999)
    parser.add_argument("--output", type=Path, default=Path("scratch/suite/operon_results.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "problem", "seed", "train_r2", "validation_r2", "complexity", "solved", "fit_seconds",
        "evaluations", "generations", "best_expression",
    )
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for problem in args.problems:
            if problem not in PROBLEM_INPUTS:
                raise ValueError(f"unknown problem: {problem}")
            for seed in args.seeds:
                record = run_one(args, problem, seed)
                writer.writerow(record)
                output.flush()
                print(
                    f"problem={problem} seed={seed} validation_r2={record['validation_r2']:.9g} "
                    f"solved={record['solved']} fit_seconds={record['fit_seconds']:.3f} "
                    f"evaluations={record['evaluations']} expression={record['best_expression']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
