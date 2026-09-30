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
import os
from pathlib import Path
import subprocess
import sys

from common import parse_csv_record, positive_int, resume_records, write_csv


SHAPES = ("materialize", "sse")


def default_benchmark():
    repo = Path(__file__).resolve().parents[2]
    candidates = (
        repo.parent / "build" / "secant" / "secant_native_avx_bench",
        repo / "build" / "secant_native_avx_bench",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def run_case(
    benchmark,
    shape,
    rows,
    asts,
    workers,
    warmups,
    min_repeats,
    min_seconds,
    seed,
    omp_proc_bind,
    omp_places,
):
    command = [
        str(benchmark),
        "--shape", shape,
        "--rows", str(rows),
        "--asts", str(asts),
        "--workers", str(workers),
        "--warmups", str(warmups),
        "--min-repeats", str(min_repeats),
        "--min-seconds", str(min_seconds),
        "--seed", str(seed),
    ]
    environment = os.environ.copy()
    environment["OMP_PROC_BIND"] = omp_proc_bind
    environment["OMP_PLACES"] = omp_places
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"native AVX benchmark failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    record = parse_csv_record(process.stdout)
    expected = (shape, rows, asts, workers)
    actual = (
        record["shape"],
        int(record["rows"]),
        int(record["asts"]),
        int(record["workers"]),
    )
    if actual != expected:
        raise RuntimeError(f"native AVX benchmark returned {actual}, expected {expected}")
    return record


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect reproducible native AVX2/FMA baseline data.")
    parser.add_argument("--benchmark", type=Path, default=default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    parser.add_argument(
        "--rows",
        type=positive_int,
        nargs="+",
        default=[1_024, 16_384, 262_144],
    )
    parser.add_argument("--asts", type=positive_int, nargs="+", default=[256])
    parser.add_argument(
        "--workers",
        type=positive_int,
        nargs="+",
        default=[1, 12, 24],
    )
    parser.add_argument("--warmups", type=positive_int, default=2)
    parser.add_argument("--min-repeats", type=positive_int, default=5)
    parser.add_argument("--min-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=positive_int, default=1)
    parser.add_argument("--omp-proc-bind", default="spread")
    parser.add_argument("--omp-places", default="cores")
    args = parser.parse_args(argv)
    if args.min_seconds <= 0.0:
        parser.error("--min-seconds must be positive")
    for values, label in (
        (args.shapes, "shapes"),
        (args.rows, "rows"),
        (args.asts, "asts"),
        (args.workers, "workers"),
    ):
        if len(values) != len(set(values)):
            parser.error(f"--{label} contains duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    key_fields = ("shape", "rows", "asts", "workers")
    records, completed = resume_records(
        args.output,
        key_fields,
        {
            "system": ("native_avx2",),
            "backend": ("cpu",),
            "shape": args.shapes,
            "rows": args.rows,
            "asts": args.asts,
            "workers": args.workers,
            "seed": (args.seed,),
        },
    )
    for shape in args.shapes:
        for rows in args.rows:
            for asts in args.asts:
                for workers in args.workers:
                    key = tuple(str(value) for value in (
                        shape,
                        rows,
                        asts,
                        workers,
                    ))
                    if key in completed:
                        print(
                            f"skip native_avx2 {shape} rows={rows} "
                            f"asts={asts} workers={workers}",
                            flush=True,
                        )
                        continue
                    record = run_case(
                        args.benchmark,
                        shape,
                        rows,
                        asts,
                        workers,
                        args.warmups,
                        args.min_repeats,
                        args.min_seconds,
                        args.seed,
                        args.omp_proc_bind,
                        args.omp_places,
                    )
                    records.append(record)
                    completed.add(key)
                    write_csv(args.output, records)
                    print(
                        f"{record['system']} {shape} rows={rows} asts={asts} "
                        f"workers={workers} row_evals_per_second="
                        f"{float(record['row_evals_per_second']):.3e}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
