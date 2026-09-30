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
import sys
import tempfile


BACKENDS = ("cuda", "ptx", "cubin")
AST_MODES = ("alu", "mufu")
CSE_MODES = ("shared", "distinct")
DEFAULT_SETTINGS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)
RESULT_FIELDS = (
    "template_prepare_seconds",
    "compile_seconds",
    "compile_asts_per_second",
    "module_load_seconds",
    "runtime_seconds",
    "row_evals",
    "row_evals_per_second",
    "atomic_updates",
    "atomic_updates_per_second",
    "registers",
    "static_shared_bytes",
    "local_bytes",
    "active_blocks_per_sm",
    "verify",
)
FIELDNAMES = (
    "backend",
    "ast_mode",
    "cse",
    "settings",
    "trial",
    "kernels",
    "asts_per_kernel",
    "asts",
    "columns",
    "constants",
    "targets",
    "rows",
    "tile_rows",
    "threads",
    "streams",
    "iterations",
    "patch_instructions_per_ast",
    "seed",
    "source_sm",
    "target_sm",
    "opt_level",
    *RESULT_FIELDS,
)


def find_default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo / "build" / "secant_dynamic_constant_sse_bench",
        repo.parent / "build" / "secant_dynamic_constant_sse_bench",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def parse_result(output):
    fields = None
    for line in output.splitlines():
        if not line.startswith("backend=") or "shape=dynamic_constant_sse" not in line:
            continue
        fields = {}
        for item in line.split():
            if "=" in item:
                name, value = item.split("=", 1)
                fields[name] = value
    if fields is None:
        raise RuntimeError("benchmark output did not contain a result")
    return fields


def row_key(row):
    return (
        row["backend"],
        row["ast_mode"],
        row["cse"],
        int(row["settings"]),
        int(row["trial"]),
    )


def read_rows(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row.setdefault(
                "cse",
                row.get(
                    "mufu_cse",
                    "distinct" if row.get("ast_mode") in ("alu", "mufu") else "not_applicable",
                ),
            )
            rows[row_key(row)] = row
    return rows


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


def run_case(args, backend, ast_mode, cse, settings):
    command = [
        str(args.benchmark),
        "--backend", backend,
        "--ast-mode", ast_mode,
        "--cse", cse,
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(args.asts_per_kernel),
        "--columns", str(args.columns),
        "--constants", str(args.constants),
        "--targets", str(args.targets),
        "--settings", str(settings),
        "--rows", str(args.rows),
        "--tile-rows", str(args.tile_rows),
        "--threads", str(args.threads),
        "--streams", str(args.streams),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--warmups", str(args.warmups),
        "--iterations", str(args.iterations),
        "--seed", str(args.seed),
        "--source-sm", str(args.source_sm),
        "--target-sm", str(args.target_sm),
        "--opt-level", str(args.opt_level),
    ]
    environment = os.environ.copy()
    environment.setdefault("CUDA_MODULE_LOADING", "EAGER")
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    fields = parse_result(process.stdout)
    expected = {
        "backend": backend,
        "shape": "dynamic_constant_sse",
        "ast_mode": ast_mode,
        "cse": cse if ast_mode in ("alu", "mufu") else "not_applicable",
        "settings": str(settings),
        "verify": "pass",
    }
    for name, value in expected.items():
        if fields.get(name) != value:
            raise RuntimeError(
                f"benchmark returned {name}={fields.get(name)!r}, expected {value!r}")
    return fields


def normalized_row(args, fields, backend, ast_mode, cse, settings, trial):
    row = {
        "backend": backend,
        "ast_mode": ast_mode,
        "cse": cse if ast_mode in ("alu", "mufu") else "not_applicable",
        "settings": settings,
        "trial": trial,
        "patch_instructions_per_ast": args.patch_instructions_per_ast,
        "seed": args.seed,
        "source_sm": args.source_sm,
        "target_sm": args.target_sm,
        "opt_level": args.opt_level,
    }
    for name in (
        "kernels",
        "asts_per_kernel",
        "asts",
        "columns",
        "constants",
        "targets",
        "rows",
        "tile_rows",
        "threads",
        "streams",
        "iterations",
        *RESULT_FIELDS,
    ):
        if name not in fields:
            raise RuntimeError(f"benchmark result omitted {name}")
        row[name] = fields[name]
    return row


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Compare CUDA, PTX, and direct-CUBIN dynamic-constant SSE "
            "compilation and runtime over setting count."
        ))
    parser.add_argument("--benchmark", type=Path, default=find_default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--ast-modes", nargs="+", choices=AST_MODES, default=list(AST_MODES))
    parser.add_argument(
        "--cse-modes",
        nargs="+",
        choices=CSE_MODES,
        default=list(CSE_MODES),
    )
    parser.add_argument("--settings", nargs="+", type=int, default=list(DEFAULT_SETTINGS))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--kernels", type=int, default=1)
    parser.add_argument("--asts-per-kernel", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--constants", type=int, default=4)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--patch-instructions-per-ast", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-sm", type=int, default=80)
    parser.add_argument("--target-sm", type=int, default=0)
    parser.add_argument("--opt-level", type=int, choices=(0, 1), default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    numeric_values = (
        args.trials,
        args.kernels,
        args.asts_per_kernel,
        args.columns,
        args.constants,
        args.targets,
        args.rows,
        args.tile_rows,
        args.threads,
        args.streams,
        args.patch_instructions_per_ast,
        args.iterations,
        args.source_sm,
        *args.settings,
    )
    if args.warmups < 0 or args.target_sm < 0 or any(value <= 0 for value in numeric_values):
        parser.error("numeric benchmark arguments must be positive")
    if len(set(args.backends)) != len(args.backends):
        parser.error("backends must be unique")
    if len(set(args.ast_modes)) != len(args.ast_modes):
        parser.error("AST modes must be unique")
    if len(set(args.cse_modes)) != len(args.cse_modes):
        parser.error("CSE modes must be unique")
    if len(set(args.settings)) != len(args.settings):
        parser.error("settings must be unique")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.force and args.output.exists():
        args.output.unlink()
    rows = read_rows(args.output)
    mode_cases = len(args.ast_modes) * len(args.cse_modes)
    total = len(args.backends) * mode_cases * len(args.settings) * args.trials
    print(f"seeded={len(rows)} total={total}", flush=True)

    for backend in args.backends:
        for ast_mode in args.ast_modes:
            for cse in args.cse_modes:
                for settings in args.settings:
                    for trial in range(args.trials):
                        key = (backend, ast_mode, cse, settings, trial)
                        if key in rows:
                            continue
                        fields = run_case(args, backend, ast_mode, cse, settings)
                        rows[key] = normalized_row(
                            args,
                            fields,
                            backend,
                            ast_mode,
                            cse,
                            settings,
                            trial,
                        )
                        write_rows(args.output, rows)
                        print(
                            f"{len(rows)}/{total} {backend} {ast_mode} "
                            f"cse={cse} settings={settings} "
                            f"compile={float(fields['compile_asts_per_second']):.3e} AST/s "
                            f"runtime={float(fields['row_evals_per_second']):.3e} row-evals/s",
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
