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

import argparse
import csv
import os
from pathlib import Path
import subprocess


DEFAULT_KERNELS = (8, 16, 32, 64, 128, 256)


def default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo / "build" / "secant_constant_optimizer_sse_bench",
        repo.parent / "build" / "secant_constant_optimizer_sse_bench",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def result_parse(output):
    for line in output.splitlines():
        if line.startswith("backend=cubin ") and "shape=constant_optimizer_sse" in line:
            return dict(item.split("=", 1) for item in line.split() if "=" in item)
    raise RuntimeError("benchmark output did not contain a constant-optimizer result")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep constant-optimizer kernels per CUBIN module at fixed total AST count.")
    parser.add_argument("--benchmark", type=Path, default=default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", type=int, default=list(DEFAULT_KERNELS))
    parser.add_argument("--total-asts", type=int, default=4096)
    parser.add_argument("--ast-mode", choices=("simple", "alu", "mufu"), default="alu")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--constants", type=int, default=4)
    parser.add_argument("--settings", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--tile-rows", type=int, choices=(128, 256), default=128)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--optimizer-iterations", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = []
    environment = os.environ.copy()
    environment.setdefault("CUDA_MODULE_LOADING", "EAGER")
    for kernels in args.kernels:
        if kernels <= 0 or args.total_asts % kernels != 0:
            raise RuntimeError(f"total AST count {args.total_asts} is not divisible by {kernels} kernels")
        modules = args.total_asts // kernels
        command = [
            str(args.benchmark),
            "--ast-mode", args.ast_mode,
            "--modules", str(modules),
            "--workers", str(args.workers),
            "--streams", str(args.streams),
            "--kernels", str(kernels),
            "--columns", str(args.columns),
            "--constants", str(args.constants),
            "--settings", str(args.settings),
            "--rows", str(args.rows),
            "--tile-rows", str(args.tile_rows),
            "--threads", str(args.threads),
            "--optimizer-iterations", str(args.optimizer_iterations),
            "--warmups", str(args.warmups),
            "--iterations", str(args.iterations),
            "--seed", str(args.seed),
        ]
        process = subprocess.run(command, text=True, capture_output=True, env=environment)
        if process.returncode != 0:
            raise RuntimeError(
                f"command failed: {' '.join(command)}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}")
        result = result_parse(process.stdout)
        if result.get("verify") != "pass" or int(result["asts"]) != args.total_asts:
            raise RuntimeError(f"invalid benchmark result: {result}")
        rows.append(result)
        print(process.stdout.strip(), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
