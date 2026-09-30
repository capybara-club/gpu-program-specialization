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
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common import parse_csv_records, positive_int, resume_records, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from corpus import DEFAULT_CORPUS, julia_expression, load_corpus


SHAPES = ("materialize", "sse")
EVALUATION_MODES = ("standard", "turbo")


def discover_julia():
    sandbox_julia = (
        Path(sys.prefix)
        / "julia_env"
        / "pyjuliapkg"
        / "install"
        / "bin"
        / "julia"
    )
    if sandbox_julia.is_file():
        return sandbox_julia
    executable = shutil.which("julia")
    return Path(executable) if executable else sandbox_julia


def discover_project():
    project = Path(sys.prefix) / "julia_env"
    return project if project.is_dir() else None


def run_worker(
    julia,
    project,
    worker,
    rows,
    asts,
    shapes,
    evaluation_modes,
    workers,
    warmups,
    min_repeats,
    min_seconds,
    seed,
    corpus,
    expression_file,
    target_file,
):
    command = [str(julia), f"--threads={workers}"]
    if project is not None:
        command.append(f"--project={project}")
    command.extend((
        str(worker),
        "--rows", ",".join(str(value) for value in rows),
        "--asts", ",".join(str(value) for value in asts),
        "--shapes", ",".join(shapes),
        "--evaluation-modes", ",".join(evaluation_modes),
        "--workers", str(workers),
        "--warmups", str(warmups),
        "--min-repeats", str(min_repeats),
        "--min-seconds", str(min_seconds),
        "--seed", str(seed),
        "--corpus-name", corpus["name"],
        "--corpus-hash", corpus["hash"],
        "--corpus-profile", corpus["profile"],
        "--num-inputs", str(corpus["num_inputs"]),
        "--expression-file", str(expression_file),
        "--target-file", str(target_file),
    ))
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"PySR Julia worker failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    records = parse_csv_records(process.stdout)
    if any(int(record["workers"]) != workers for record in records):
        raise RuntimeError("PySR worker returned the wrong thread count")
    return records


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect PySR/SymbolicRegression AST runtime data.")
    parser.add_argument("--julia", type=Path, default=discover_julia())
    parser.add_argument("--project", type=Path, default=discover_project())
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("pysr_worker.jl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    parser.add_argument(
        "--evaluation-modes",
        choices=EVALUATION_MODES,
        nargs="+",
        default=list(EVALUATION_MODES),
    )
    parser.add_argument(
        "--rows",
        type=positive_int,
        nargs="+",
        default=[1_024, 16_384, 262_144],
    )
    parser.add_argument("--asts", type=positive_int, nargs="+", default=[256])
    parser.add_argument("--workers", type=positive_int, nargs="+", default=[1, 24])
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--min-repeats", type=positive_int, default=3)
    parser.add_argument("--min-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=positive_int, default=1)
    args = parser.parse_args(argv)
    if args.min_seconds <= 0.0:
        parser.error("--min-seconds must be positive")
    if not args.julia.is_file():
        parser.error(
            "Julia was not found; run this script from a PySR sandbox or pass --julia")
    if args.project is not None and not args.project.is_dir():
        parser.error("--project does not exist")
    for values, label in (
        (args.shapes, "shapes"),
        (args.evaluation_modes, "evaluation-modes"),
        (args.rows, "rows"),
        (args.asts, "asts"),
        (args.workers, "workers"),
    ):
        if len(values) != len(set(values)):
            parser.error(f"--{label} contains duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    corpus = load_corpus(args.corpus)
    key_fields = ("shape", "rows", "asts", "workers", "execution_mode")
    records, completed = resume_records(
        args.output,
        key_fields,
        {
            "system": ("pysr_symbolicregression",),
            "backend": ("cpu_symbolicregression_julia",),
            "shape": args.shapes,
            "rows": args.rows,
            "asts": args.asts,
            "workers": args.workers,
            "seed": (args.seed,),
            "corpus": (corpus["name"],),
            "corpus_hash": (corpus["hash"],),
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        expression_file = Path(directory) / "expressions.jl.txt"
        target_file = Path(directory) / "target.jl.txt"
        expression_count = min(max(args.asts), len(corpus["expressions"]))
        expression_file.write_text(
            "\n".join(
                julia_expression(expression)
                for expression in corpus["expressions"][:expression_count]) + "\n",
            encoding="utf-8")
        target_file.write_text(
            julia_expression(
                corpus["expressions"][corpus["target_expression"]]) + "\n",
            encoding="utf-8")
        for workers in args.workers:
            expected = {
                tuple(str(value) for value in (
                    shape,
                    rows,
                    asts,
                    workers,
                    (
                        "serial_" if workers == 1
                        else "julia_threads_"
                    ) + mode,
                ))
                for shape in args.shapes
                for rows in args.rows
                for asts in args.asts
                for mode in args.evaluation_modes
            }
            if expected.issubset(completed):
                print(
                    f"skip pysr_symbolicregression workers={workers}",
                    flush=True,
                )
                continue
            worker_records = run_worker(
                args.julia,
                args.project,
                args.worker,
                args.rows,
                args.asts,
                args.shapes,
                args.evaluation_modes,
                workers,
                args.warmups,
                args.min_repeats,
                args.min_seconds,
                args.seed,
                corpus,
                expression_file,
                target_file,
            )
            for record in worker_records:
                key = tuple(record[field] for field in key_fields)
                if key not in completed:
                    records.append(record)
                    completed.add(key)
            write_csv(args.output, records)
            for record in worker_records:
                print(
                    f"{record['system']} {record['shape']} rows={record['rows']} "
                    f"asts={record['asts']} workers={record['workers']} "
                    f"mode={record['execution_mode']} "
                    f"row_evals_per_second="
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
