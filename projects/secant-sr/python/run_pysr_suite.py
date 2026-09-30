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
"""Run PySR over the same deterministic problems and splits as secant-sr."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import numpy as np


MASK_U64 = (1 << 64) - 1
DEFAULT_PROBLEMS = (
    "nguyen1",
    "nguyen5",
    "rational2",
    "distance2",
    "interaction3",
    "oscillator2",
)
PROBLEM_INPUTS = {
    "nguyen1": ((-1.0, 1.0),),
    "nguyen5": ((-2.0, 2.0),),
    "rational2": ((-2.0, 2.0), (-2.0, 2.0)),
    "distance2": ((-2.0, 2.0), (-2.0, 2.0)),
    "interaction3": ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
    "oscillator2": ((-np.pi, np.pi), (-np.pi, np.pi)),
}


def hash_u64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK_U64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK_U64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK_U64
    return value ^ (value >> 31)


def unit_f32(value: int) -> np.float32:
    return np.float32((hash_u64(value) >> 11) * (1.0 / 9007199254740992.0))


def problem_target(problem: str, x: np.ndarray) -> np.ndarray:
    if problem == "nguyen1":
        return x[:, 0] * x[:, 0] * x[:, 0] + x[:, 0] * x[:, 0] + x[:, 0]
    if problem == "nguyen5":
        return np.sin(x[:, 0] * x[:, 0]) * np.cos(x[:, 0]) - np.float32(1.0)
    if problem == "rational2":
        return x[:, 0] / (np.float32(1.0) + x[:, 1] * x[:, 1])
    if problem == "distance2":
        return np.sqrt(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1])
    if problem == "interaction3":
        return x[:, 0] * x[:, 1] + x[:, 2] * x[:, 2] - x[:, 2]
    if problem == "oscillator2":
        return np.sin(x[:, 0]) + np.cos(x[:, 1])
    raise ValueError(f"unknown problem: {problem}")


def problem_data(problem: str, split_seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    ranges = PROBLEM_INPUTS[problem]
    x = np.empty((rows, len(ranges)), dtype=np.float32)
    for row in range(rows):
        for input_idx, (lower, upper) in enumerate(ranges):
            coordinate = (
                split_seed
                ^ (((row + 1) * 0xD1B54A32D192ED03) & MASK_U64)
                ^ (((input_idx + 1) * 0x94D049BB133111EB) & MASK_U64)
            )
            x[row, input_idx] = np.float32(lower) + unit_f32(coordinate) * np.float32(upper - lower)
    return x, problem_target(problem, x).astype(np.float32)


def r2_get(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = target.astype(np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    residual = target64 - prediction64
    return 1.0 - float(np.sum(residual * residual)) / float(np.sum((target64 - target64.mean()) ** 2))


def run_one(args: argparse.Namespace, problem: str, seed: int) -> dict[str, int | float | str]:
    from pysr import PySRRegressor
    import sympy as sp

    train_x, train_y = problem_data(problem, seed ^ 0x747261696E, args.rows)
    validation_x, validation_y = problem_data(problem, seed ^ 0x76616C6964617465, args.validation_rows)
    target_variance = float(np.mean((train_y.astype(np.float64) - float(np.mean(train_y))) ** 2))
    early_stop_condition = None
    if args.early_stop:
        early_stop_condition = target_variance * args.early_stop_nmse
    model = PySRRegressor(
        model_selection="accuracy",
        binary_operators=["+", "-", "*", "/", "min", "max"],
        unary_operators=["neg", "abs", "sqrt", "inv", "sin", "cos", "tanh", "exp", "log"],
        niterations=1_000_000,
        timeout_in_seconds=args.seconds,
        early_stop_condition=early_stop_condition,
        populations=args.populations,
        population_size=args.population_size,
        ncycles_per_iteration=args.ncycles_per_iteration,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        precision=32,
        batching=False,
        parallelism="multithreading",
        random_state=seed,
        progress=False,
        verbosity=0,
        update_verbosity=0,
        should_optimize_constants=True,
        extra_sympy_mappings={
            "min": sp.Min,
            "max": sp.Max,
            "inv": lambda value: 1 / value,
            "neg": lambda value: -value,
        },
        output_jax_format=False,
        output_torch_format=False,
        temp_equation_file=True,
        delete_tempfiles=True,
    )
    begin = time.monotonic()
    model.fit(train_x, train_y, variable_names=[f"x{i}" for i in range(train_x.shape[1])])
    elapsed = time.monotonic() - begin
    best = model.get_best()
    best_idx = int(best.name)
    train_r2 = r2_get(train_y, model.predict(train_x, index=best_idx))
    validation_r2 = r2_get(validation_y, model.predict(validation_x, index=best_idx))
    return {
        "problem": problem,
        "seed": seed,
        "train_r2": train_r2,
        "validation_r2": validation_r2,
        "complexity": int(best["complexity"]),
        "solved": int(validation_r2 > args.stop_r2),
        "fit_seconds": elapsed,
        "best_expression": str(model.sympy(index=best_idx)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--populations", type=int, default=24)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--ncycles-per-iteration", type=int, default=50)
    parser.add_argument("--maxsize", type=int, default=40)
    parser.add_argument("--maxdepth", type=int, default=16)
    parser.add_argument("--stop-r2", type=float, default=0.999999)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-nmse", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path, default=Path("scratch/suite/pysr_results.csv"))
    args = parser.parse_args()

    os.environ.setdefault("JULIA_NUM_THREADS", str(args.threads))
    os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "problem", "seed", "train_r2", "validation_r2", "complexity", "solved", "fit_seconds",
        "best_expression",
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
                    f"expression={record['best_expression']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
