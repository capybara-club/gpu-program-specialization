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
"""Benchmark the complete dynamic-leaf SR evaluation path on a large dataset."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


def generation_parse(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/secant_sr_search"))
    parser.add_argument("--problem", default="nguyen1")
    parser.add_argument("--population", type=int, default=8192)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1_048_576)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--settings", type=int, default=4096)
    parser.add_argument("--dynamic-leaves", type=int, default=8)
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--asts-per-kernel", type=int, default=32)
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--patch-instructions-per-ast", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("scratch/dynamic_leaf_1m.csv"))
    args = parser.parse_args()

    command = [
        str(args.executable),
        "--backend", "cubin-dynamic-leaf",
        "--problem", args.problem,
        "--population", str(args.population),
        "--generations", str(args.generations),
        "--rows", str(args.rows),
        "--validation-rows", str(args.validation_rows),
        "--leaf-settings", str(args.settings),
        "--dynamic-leaves", str(args.dynamic_leaves),
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(args.asts_per_kernel),
        "--tile-rows", str(args.tile_rows),
        "--workers", str(args.workers),
        "--streams", str(args.streams),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--seed", str(args.seed),
        "--stop-r2", "2",
    ]
    environment = os.environ.copy()
    environment["CUDA_MODULE_LOADING"] = "EAGER"
    begin = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    elapsed_seconds = time.monotonic() - begin
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        return completed.returncode

    records = []
    for line in completed.stdout.splitlines():
        if not line.startswith(f"problem={args.problem} generation="):
            continue
        generation = generation_parse(line)
        records.append({
            "problem": args.problem,
            "generation": generation["generation"],
            "population": args.population,
            "rows": args.rows,
            "settings": args.settings,
            "dynamic_leaves": args.dynamic_leaves,
            "tile_rows": args.tile_rows,
            "evaluation_seconds": generation["evaluation_seconds"],
            "row_evals_per_second": generation["row_evals_per_second"],
            "compile_seconds": generation["compile_seconds"],
            "module_load_seconds": generation["module_load_seconds"],
            "device_runtime_seconds": generation["device_runtime_seconds"],
            "train_r2": generation["train_r2"],
            "validation_r2": generation["validation_r2"],
            "process_seconds": f"{elapsed_seconds:.9f}",
        })
    if len(records) != args.generations:
        raise RuntimeError(f"expected {args.generations} generation records, found {len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"benchmark_csv={args.output} process_seconds={elapsed_seconds:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
