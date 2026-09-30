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
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile


AST_MODES = ("simple", "alu", "mufu")
TOPOLOGIES = ("dynamic", "mixed")
CSE_MODES = ("distinct", "shared")
DYNAMIC_VALUES = ("mixed", "columns")
RESULT_FIELDS = (
    "asts",
    "leaves_used",
    "static_columns",
    "static_column_capacity",
    "template_seconds",
    "compile_seconds",
    "compile_asts_per_second",
    "module_load_seconds",
    "runtime_seconds",
    "runner_pipeline_seconds",
    "pipeline_seconds",
    "runtime_row_evals_per_second",
    "pipeline_row_evals_per_second",
    "verify",
)
FIELDNAMES = (
    "topology",
    "ast_mode",
    "cse",
    "dynamic_values",
    "ast_nodes",
    "asts_per_kernel",
    "columns",
    "column_capacity",
    "leaves",
    "dynamic_sites",
    "settings",
    "trial",
    "modules",
    "workers",
    "streams",
    "kernels",
    "targets",
    "rows",
    "tile_rows",
    "threads",
    "warmups",
    "iterations",
    "patch_instructions_per_ast",
    "seed",
    "target_sm",
    *RESULT_FIELDS,
)


def find_default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo / "build" / "secant_dynamic_leaf_sse_bench",
        repo.parent / "build" / "secant_dynamic_leaf_sse_bench",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def input_configuration(text):
    pieces = text.split(":", 1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("input configuration must be ACTIVE:CAPACITY")
    active = positive_int(pieces[0])
    capacity = positive_int(pieces[1])
    if active > capacity:
        raise argparse.ArgumentTypeError("active input count cannot exceed capacity")
    return active, capacity


def parse_result(output):
    result = None
    for line in output.splitlines():
        if not line.startswith("backend=cubin shape=dynamic_leaf_sse "):
            continue
        result = {}
        for item in line.split():
            if "=" in item:
                name, value = item.split("=", 1)
                result[name] = value
    if result is None:
        raise RuntimeError("benchmark output did not contain a dynamic-leaf SSE result")
    return result


def row_key(row):
    return (
        row["topology"],
        row["ast_mode"],
        row["cse"],
        row.get("dynamic_values") or "mixed",
        int(row.get("ast_nodes") or "0"),
        int(row["asts_per_kernel"]),
        int(row["columns"]),
        int(row["column_capacity"]),
        int(row["leaves"]),
        int(row["dynamic_sites"]),
        int(row["settings"]),
        int(row["trial"]),
        int(row["modules"]),
        int(row["workers"]),
        int(row["streams"]),
        int(row["kernels"]),
        int(row["targets"]),
        int(row["rows"]),
        int(row["tile_rows"]),
        int(row["threads"]),
        int(row["warmups"]),
        int(row["iterations"]),
        int(row["patch_instructions_per_ast"]),
        int(row["seed"]),
        int(row["target_sm"]),
    )


def read_rows(path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row_key(row): row for row in csv.DictReader(file)}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            for key in sorted(rows):
                writer.writerow(rows[key])
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def cases_get(args):
    for topology, ast_mode, cse, dynamic_values, ast_nodes, asts_per_kernel, input_counts, leaves, settings, modules, \
            trial in \
            itertools.product(
            args.topologies,
            args.ast_modes,
            args.cse_modes,
            args.dynamic_values,
            args.ast_nodes,
            args.asts_per_kernel,
            args.input_configurations,
            args.dynamic_leaves,
            args.settings,
            args.modules,
            range(args.trials)):
        if ast_nodes != 0 and ast_mode == "simple":
            continue
        if topology == "dynamic":
            dynamic_sites_values = (leaves,)
        elif ast_mode == "simple":
            dynamic_sites_values = (1,) if 1 in args.dynamic_sites else ()
        else:
            dynamic_sites_values = tuple(value for value in args.dynamic_sites if value <= leaves)
        for dynamic_sites in dynamic_sites_values:
            yield topology, ast_mode, cse, dynamic_values, ast_nodes, asts_per_kernel, input_counts, leaves, \
                dynamic_sites, settings, modules, trial


def case_key(args, case):
    topology, ast_mode, cse, dynamic_values, ast_nodes, asts_per_kernel, input_counts, leaves, dynamic_sites, \
        settings, modules, trial = case
    columns, column_capacity = input_counts
    return (
        topology,
        ast_mode,
        cse,
        dynamic_values,
        ast_nodes,
        asts_per_kernel,
        columns,
        column_capacity,
        leaves,
        dynamic_sites,
        settings,
        trial,
        modules,
        args.workers,
        args.streams,
        args.kernels,
        args.targets,
        args.rows,
        args.tile_rows,
        args.threads,
        args.warmups,
        args.iterations,
        args.patch_instructions_per_ast,
        args.seed + trial,
        args.target_sm,
    )


def run_case(args, case):
    topology, ast_mode, cse, dynamic_values, ast_nodes, asts_per_kernel, input_counts, leaves, dynamic_sites, \
        settings, modules, trial = case
    columns, column_capacity = input_counts
    static_columns = column_capacity if topology == "mixed" else 0
    command = [
        str(args.benchmark),
        "--ast-mode", ast_mode,
        "--cse", cse,
        "--dynamic-values", dynamic_values,
        "--modules", str(modules),
        "--workers", str(args.workers),
        "--streams", str(args.streams),
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(asts_per_kernel),
        "--leaves", str(leaves),
        "--dynamic-sites", str(dynamic_sites),
        "--columns", str(columns),
        "--column-capacity", str(column_capacity),
        "--static-columns", str(static_columns),
        "--targets", str(args.targets),
        "--settings", str(settings),
        "--rows", str(args.rows),
        "--tile-rows", str(args.tile_rows),
        "--threads", str(args.threads),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--warmups", str(args.warmups),
        "--iterations", str(args.iterations),
        "--seed", str(args.seed + trial),
        "--target-sm", str(args.target_sm),
    ]
    if ast_nodes != 0:
        command.extend(("--ast-nodes", str(ast_nodes)))
    environment = os.environ.copy()
    environment.setdefault("CUDA_MODULE_LOADING", "EAGER")
    process = subprocess.run(command, text=True, capture_output=True, env=environment)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    fields = parse_result(process.stdout)
    expected = {
        "ast_mode": ast_mode,
        "cse": cse,
        "dynamic_values": dynamic_values,
        "ast_nodes": str(ast_nodes),
        "asts_per_kernel": str(asts_per_kernel),
        "leaves": str(leaves),
        "dynamic_sites": str(dynamic_sites),
        "columns": str(columns),
        "column_capacity": str(column_capacity),
        "static_columns": str(columns if topology == "mixed" else 0),
        "static_column_capacity": str(static_columns),
        "settings": str(settings),
        "modules": str(modules),
        "verify": "pass",
    }
    for name, value in expected.items():
        if fields.get(name) != value:
            raise RuntimeError(f"benchmark returned {name}={fields.get(name)!r}, expected {value!r}")

    row = {
        "topology": topology,
        "ast_mode": ast_mode,
        "cse": cse,
        "dynamic_values": dynamic_values,
        "ast_nodes": ast_nodes,
        "asts_per_kernel": asts_per_kernel,
        "columns": columns,
        "column_capacity": column_capacity,
        "leaves": leaves,
        "dynamic_sites": dynamic_sites,
        "settings": settings,
        "trial": trial,
        "modules": modules,
        "workers": args.workers,
        "streams": args.streams,
        "kernels": args.kernels,
        "targets": args.targets,
        "rows": args.rows,
        "tile_rows": args.tile_rows,
        "threads": args.threads,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "patch_instructions_per_ast": args.patch_instructions_per_ast,
        "seed": args.seed + trial,
        "target_sm": args.target_sm,
    }
    for name in RESULT_FIELDS:
        if name not in fields:
            raise RuntimeError(f"benchmark result omitted {name}")
        row[name] = fields[name]
    return row


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run a correctness-enforced dynamic-leaf SSE characteristic grid.")
    parser.add_argument("--benchmark", type=Path, default=find_default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topologies", nargs="+", choices=TOPOLOGIES, default=list(TOPOLOGIES))
    parser.add_argument("--ast-modes", nargs="+", choices=AST_MODES, default=list(AST_MODES))
    parser.add_argument("--cse-modes", nargs="+", choices=CSE_MODES, default=["distinct"])
    parser.add_argument("--dynamic-values", nargs="+", choices=DYNAMIC_VALUES, default=["mixed"])
    parser.add_argument("--ast-nodes", nargs="+", type=nonnegative_int, default=[0])
    parser.add_argument("--asts-per-kernel", nargs="+", type=positive_int, default=[1, 8, 32])
    parser.add_argument(
        "--input-configurations",
        nargs="+",
        type=input_configuration,
        default=[(4, 4), (4, 16), (16, 16), (16, 32), (32, 32)],
        metavar="ACTIVE:CAPACITY",
    )
    parser.add_argument("--dynamic-leaves", nargs="+", type=positive_int, default=[8])
    parser.add_argument("--dynamic-sites", nargs="+", type=positive_int, default=[1, 2, 4, 8])
    parser.add_argument("--settings", nargs="+", type=positive_int, default=[1024])
    parser.add_argument("--trials", type=positive_int, default=1)
    parser.add_argument("--modules", nargs="+", type=positive_int, default=[1])
    parser.add_argument("--workers", type=positive_int, default=2)
    parser.add_argument("--streams", type=positive_int, default=4)
    parser.add_argument("--kernels", type=positive_int, default=8)
    parser.add_argument("--targets", type=positive_int, default=1)
    parser.add_argument("--rows", type=positive_int, default=65536)
    parser.add_argument("--tile-rows", type=positive_int, default=64)
    parser.add_argument("--threads", type=positive_int, default=128)
    parser.add_argument("--patch-instructions-per-ast", type=positive_int, default=64)
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=1)
    parser.add_argument("--seed", type=nonnegative_int, default=1)
    parser.add_argument("--target-sm", type=nonnegative_int, default=0)
    parser.add_argument("--max-cases", type=nonnegative_int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.streams > args.kernels:
        parser.error("streams cannot exceed kernels")
    for name in ("topologies", "ast_modes", "cse_modes", "dynamic_values", "ast_nodes", "asts_per_kernel",
                 "input_configurations", "dynamic_leaves", "dynamic_sites", "settings", "modules"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"{name.replace('_', '-')} entries must be unique")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.force and args.output.exists():
        args.output.unlink()
    rows = read_rows(args.output)
    cases = list(cases_get(args))
    pending = [case for case in cases if case_key(args, case) not in rows]
    if args.max_cases != 0:
        pending = pending[:args.max_cases]
    print(f"seeded={len(rows)} total={len(cases)} pending={len(pending)}", flush=True)

    for case in pending:
        row = run_case(args, case)
        rows[row_key(row)] = row
        write_rows(args.output, rows)
        print(
            f"{len(rows)}/{len(cases)} {row['topology']} {row['ast_mode']} "
            f"values={row['dynamic_values']} nodes={row['ast_nodes']} asts/kernel={row['asts_per_kernel']} "
            f"inputs={row['columns']}:{row['column_capacity']} "
            f"leaves={row['leaves']} dynamic-sites={row['dynamic_sites']} settings={row['settings']} "
            f"runtime={float(row['runtime_row_evals_per_second']):.3e} row-evals/s",
            flush=True,
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
