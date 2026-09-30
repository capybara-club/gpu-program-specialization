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

"""Run the paired setting-distribution/persistence ablation."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path


POLICIES = (
    "legacy",
    "legacy-rotating",
    "virtual-bank-fixed",
    "virtual-bank",
)

DEFAULT_PROBLEMS = (
    "nguyen1",
    "nguyen5",
    "rational2",
    "distance2",
    "interaction3",
    "oscillator2",
)


def rows_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def summary_write(output_dir: Path, policies: tuple[str, ...]) -> None:
    summary_rows: list[dict[str, object]] = []
    paired_rows: dict[tuple[str, str], dict[str, object]] = {}
    trajectory_rows: list[dict[str, object]] = []
    time_quality_rows: list[dict[str, object]] = []

    for policy in policies:
        result_rows = rows_read(output_dir / f"{policy}.csv")
        generation_rows = rows_read(output_dir / f"{policy}_generations.csv")
        problems = sorted({row["problem"] for row in result_rows})

        for problem in problems + ["ALL"]:
            selected = result_rows if problem == "ALL" else [row for row in result_rows if row["problem"] == problem]
            selected_generations = generation_rows if problem == "ALL" else [
                row for row in generation_rows if row["problem"] == problem
            ]
            solved = [int(row["solved"]) for row in selected]
            train_r2 = [float(row["train_r2"]) for row in selected]
            elapsed = [float(row["search_elapsed_seconds"]) for row in selected]
            first_hits = [float(row["first_hit_seconds"]) for row in selected if row["first_hit_seconds"]]
            update_seconds = [float(row["leaf_setting_update_seconds"]) for row in selected_generations]

            summary_rows.append({
                "policy": policy,
                "problem": problem,
                "trials": len(selected),
                "solved": sum(solved),
                "solve_rate": sum(solved) / len(selected),
                "mean_train_r2": statistics.fmean(train_r2),
                "median_elapsed_seconds": statistics.median(elapsed),
                "median_first_hit_seconds": statistics.median(first_hits) if first_hits else "",
                "setting_update_total_seconds": sum(update_seconds),
                "mean_setting_update_milliseconds": 1000.0 * statistics.fmean(update_seconds),
            })

            trials: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in selected_generations:
                trials.setdefault((row["problem"], row["seed"]), []).append(row)
            for checkpoint in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
                normalized_sses: list[float] = []

                for rows in trials.values():
                    observed = [row for row in rows if float(row["elapsed_seconds"]) <= checkpoint]
                    if observed:
                        normalized_sses.append(1.0 - float(observed[-1]["train_r2"]))
                if normalized_sses:
                    time_quality_rows.append({
                        "policy": policy,
                        "problem": problem,
                        "seconds": checkpoint,
                        "trials_observed": len(normalized_sses),
                        "solved": sum(value < 1.0e-6 for value in normalized_sses),
                        "mean_normalized_sse": statistics.fmean(normalized_sses),
                        "median_normalized_sse": statistics.median(normalized_sses),
                        "mean_log10_normalized_sse": statistics.fmean(
                            math.log10(max(value, 1.0e-15)) for value in normalized_sses
                        ),
                    })

        for row in result_rows:
            key = (row["problem"], row["seed"])
            paired = paired_rows.setdefault(key, {"problem": key[0], "seed": key[1]})
            paired[f"{policy}_solved"] = row["solved"]
            paired[f"{policy}_train_r2"] = row["train_r2"]
            paired[f"{policy}_elapsed_seconds"] = row["search_elapsed_seconds"]
            paired[f"{policy}_first_hit_seconds"] = row["first_hit_seconds"]

        for row in generation_rows:
            trajectory_rows.append({
                "policy": policy,
                "problem": row["problem"],
                "seed": row["seed"],
                "generation": row["generation"],
                "elapsed_seconds": row["elapsed_seconds"],
                "leaf_setting_update_seconds": row["leaf_setting_update_seconds"],
                "train_r2": row["train_r2"],
                "normalized_sse": 1.0 - float(row["train_r2"]),
                "best_sse_after_static": row["best_sse_after_static"],
                "best_sse_after_full": row["best_sse_after_full"],
                "best_sse_after_mixed": row["best_sse_after_mixed"],
                "best_sse_after_constant": row["best_sse_after_constant"],
            })

    with (output_dir / "summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    paired_fields = ("problem", "seed", *(
        f"{policy}_{field}"
        for policy in policies
        for field in ("solved", "train_r2", "elapsed_seconds", "first_hit_seconds")
    ))
    with (output_dir / "paired.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows[key] for key in sorted(paired_rows))

    with (output_dir / "trajectories.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=tuple(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with (output_dir / "time_quality.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=tuple(time_quality_rows[0]))
        writer.writeheader()
        writer.writerows(time_quality_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/secant_sr_search"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(23654, 15795, 860))
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--population", type=int, default=8192)
    parser.add_argument("--leaf-settings", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_script = Path(__file__).with_name("run_suite.py")

    for policy in () if args.summarize_only else args.policies:
        command = [
            sys.executable,
            str(suite_script),
            "--executable", str(args.executable),
            "--backend", "cubin-maturity",
            "--problems", *args.problems,
            "--seeds", *(str(seed) for seed in args.seeds),
            "--population", str(args.population),
            "--generations", "100000",
            "--rows", "1024",
            "--validation-rows", "4096",
            "--workers", "24",
            "--streams", "8",
            "--leaf-settings", str(args.leaf_settings),
            "--leaf-setting-policy", policy,
            "--constant-settings", "8192",
            "--constant-optimize-budget", "512",
            "--constant-optimize-interval", "1",
            "--constant-optimize-random-fraction", "0.25",
            "--constant-optimizer-iterations", "4",
            "--constant-optimizer-scale", "1",
            "--constant-optimizer-decay", "0.5",
            "--constant-sweep-phase", "each",
            "--dynamic-leaves", "8",
            "--mixed-dynamic-leaves", "4",
            "--mixed-refine-probability", "0.25",
            "--dynamic-max-nodes", "30",
            "--kernels", "64",
            "--asts-per-kernel", "32",
            "--static-tile-rows", "1024",
            "--patch-instructions-per-ast", "64",
            "--operator-profile", "scientific",
            "--final-cpu-optimize", "0",
            "--validation-mode", "final",
            "--stop-metric", "train",
            "--stop-r2", "0.999999",
            "--time-limit-seconds", str(args.time_limit_seconds),
            "--persistent-process",
            "--skip-symbolic-assessment",
            "--output", str(output_dir / f"{policy}.csv"),
            "--generation-output", str(output_dir / f"{policy}_generations.csv"),
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)

    summary_write(output_dir, tuple(args.policies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
