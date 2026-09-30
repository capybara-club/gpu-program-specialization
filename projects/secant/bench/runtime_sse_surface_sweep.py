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

from backend_csv import BACKENDS, backend_csv_path, backend_opt_levels

OPT_LEVELS = (0, 1)
AST_MODES = ("alu", "mufu")
DEFAULT_AST_COUNTS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
DEFAULT_TILE_ROWS = (128, 256, 512, 1024, 2048, 4096, 8192)
FIELDNAMES = (
    "backend",
    "opt_level",
    "ast_mode",
    "tile_rows",
    "asts_per_kernel",
    "row_evals_per_second",
    "kernels",
    "threads",
    "run_rows",
    "run_iterations",
    "patch_instructions_per_ast",
)


def find_default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo.parent / "build" / "secant_runtime_bench",
        repo / "build" / "secant_runtime_bench",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def row_key(row):
    return (
        row["backend"],
        int(row["opt_level"]),
        row["ast_mode"],
        int(row["tile_rows"]),
        int(row["asts_per_kernel"]),
    )


def normalized_row(
    backend,
    opt_level,
    ast_mode,
    tile_rows,
    asts_per_kernel,
    rate,
    args,
):
    return {
        "backend": backend,
        "opt_level": opt_level,
        "ast_mode": ast_mode,
        "tile_rows": tile_rows,
        "asts_per_kernel": asts_per_kernel,
        "row_evals_per_second": f"{rate:.9g}",
        "kernels": args.kernels,
        "threads": args.threads,
        "run_rows": args.run_rows,
        "run_iterations": args.run_iterations,
        "patch_instructions_per_ast": args.patch_instructions_per_ast,
    }


def validate_metadata(row, args, source):
    expected = {
        "kernels": args.kernels,
        "threads": args.threads,
        "run_rows": args.run_rows,
        "run_iterations": args.run_iterations,
        "patch_instructions_per_ast": args.patch_instructions_per_ast,
    }
    for name, value in expected.items():
        if name not in row or row[name] == "":
            row[name] = str(value)
            continue
        try:
            actual = int(row[name])
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                f"{source} has invalid {name} metadata") from error
        if actual != value:
            raise RuntimeError(
                f"{source} has {name}={actual}, expected {value}")


def read_output(path, args):
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            validate_metadata(row, args, path)
            rows[row_key(row)] = row
    return rows


def import_packing(paths, args, rows):
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["kernel_shape"] != "sse":
                    continue
                validate_metadata(row, args, path)
                backend = row["backend"]
                opt_level = int(row["opt_level"])
                ast_mode = row["ast_mode"]
                tile_rows = int(row["tile_rows"])
                asts_per_kernel = int(row["asts_per_kernel"])
                if tile_rows not in args.tile_rows or asts_per_kernel not in args.ast_counts:
                    continue
                rows[(
                    backend,
                    opt_level,
                    ast_mode,
                    tile_rows,
                    asts_per_kernel,
                )] = normalized_row(
                    backend,
                    opt_level,
                    ast_mode,
                    tile_rows,
                    asts_per_kernel,
                    float(row["row_evals_per_second"]),
                    args,
                )


def import_tiles(paths, args, rows):
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                validate_metadata(row, args, path)
                backend = row["backend"]
                opt_level = int(row["opt_level"])
                ast_mode = row["ast_mode"]
                tile_rows = int(row["tile_rows"])
                asts_per_kernel = int(row["asts_per_kernel"])
                if tile_rows not in args.tile_rows or asts_per_kernel not in args.ast_counts:
                    continue
                rows[(
                    backend,
                    opt_level,
                    ast_mode,
                    tile_rows,
                    asts_per_kernel,
                )] = normalized_row(
                    backend,
                    opt_level,
                    ast_mode,
                    tile_rows,
                    asts_per_kernel,
                    float(row["row_evals_per_second"]),
                    args,
                )


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


def parse_result(
    output,
    backend,
    opt_level,
    ast_mode,
    tile_rows,
    asts_per_kernel,
):
    fields = None
    for line in output.splitlines():
        if not line.startswith("runtime "):
            continue
        fields = {}
        for item in line.split()[1:]:
            if "=" in item:
                name, value = item.split("=", 1)
                fields[name] = value
    if fields is None:
        raise RuntimeError("benchmark output did not contain a runtime result")

    expected = {
        "backend": backend,
        "shape": "sse",
        "opt_level": str(opt_level),
        "ast_mode": ast_mode,
        "tile_rows": str(tile_rows),
        "asts_per_kernel": str(asts_per_kernel),
    }
    for name, value in expected.items():
        if fields.get(name) != value:
            raise RuntimeError(
                f"benchmark returned {name}={fields.get(name)!r}, "
                f"expected {value!r}")
    try:
        return float(fields["row_evals_per_second"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("benchmark returned an invalid row-evals/s value") from error


def run_case(
    args,
    backend,
    opt_level,
    ast_mode,
    tile_rows,
    asts_per_kernel,
):
    command = [
        str(args.benchmark),
        "--backend", backend,
        "--shape", "sse",
        "--ast-mode", ast_mode,
        "--warmups", str(args.warmups),
        "--run-iterations", str(args.run_iterations),
        "--run-rows", str(args.run_rows),
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(asts_per_kernel),
        "--tile-rows", str(tile_rows),
        "--threads", str(args.threads),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--check-rows", str(args.check_rows),
        "--source-sm", str(args.source_sm),
        "--target-sm", str(args.target_sm),
        "--opt-level", str(opt_level),
    ]
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    return parse_result(
        process.stdout,
        backend,
        opt_level,
        ast_mode,
        tile_rows,
        asts_per_kernel,
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Sweep SSE runtime over AST packing and logical tile size.")
    parser.add_argument("--benchmark", type=Path, default=find_default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument(
        "--tile-rows",
        nargs="+",
        type=int,
        default=list(DEFAULT_TILE_ROWS))
    parser.add_argument(
        "--ast-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_AST_COUNTS))
    parser.add_argument("--packing-csv", type=Path)
    parser.add_argument("--tile-csv", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--run-iterations", type=int, default=100)
    parser.add_argument("--run-rows", type=int, default=1048576)
    parser.add_argument("--kernels", type=int, default=16)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--patch-instructions-per-ast", type=int, default=64)
    parser.add_argument("--check-rows", type=int, default=257)
    parser.add_argument("--source-sm", type=int, default=80)
    parser.add_argument("--target-sm", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    numeric_values = (
        args.run_iterations,
        args.run_rows,
        args.kernels,
        args.threads,
        args.patch_instructions_per_ast,
        args.check_rows,
        args.source_sm,
        args.target_sm,
        *args.tile_rows,
        *args.ast_counts,
    )
    if args.warmups < 0 or any(value <= 0 for value in numeric_values):
        parser.error("numeric benchmark arguments must be positive")
    if len(set(args.backends)) != len(args.backends):
        parser.error("backends must be unique")
    if len(set(args.tile_rows)) != len(args.tile_rows):
        parser.error("tile rows must be unique")
    if len(set(args.ast_counts)) != len(args.ast_counts):
        parser.error("AST counts must be unique")
    return args


def main(argv=None):
    args = parse_args(argv)
    output_paths = {
        backend: backend_csv_path(args.output, backend)
        for backend in args.backends
    }
    if args.force:
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    rows = {}
    for path in args.import_csv:
        rows.update(read_output(path, args))
    for path in output_paths.values():
        rows.update(read_output(path, args))
    packing_paths = (
        [backend_csv_path(args.packing_csv, backend) for backend in args.backends]
        if args.packing_csv is not None
        else [])
    tile_paths = (
        [backend_csv_path(args.tile_csv, backend) for backend in args.backends]
        if args.tile_csv is not None
        else [])
    import_packing(packing_paths, args, rows)
    import_tiles(tile_paths, args, rows)
    for backend, path in output_paths.items():
        backend_rows = {
            row_key_value: row
            for row_key_value, row in rows.items()
            if row_key_value[0] == backend
        }
        write_rows(path, backend_rows)

    total = sum(
        len(backend_opt_levels(backend, OPT_LEVELS)) *
        len(AST_MODES) *
        len(args.tile_rows) *
        len(args.ast_counts)
        for backend in args.backends)
    print(f"seeded={len(rows)} total={total}", flush=True)

    for backend in args.backends:
        for opt_level in backend_opt_levels(backend, OPT_LEVELS):
            for ast_mode in AST_MODES:
                for tile_rows in args.tile_rows:
                    for asts_per_kernel in args.ast_counts:
                        key = (
                            backend,
                            opt_level,
                            ast_mode,
                            tile_rows,
                            asts_per_kernel,
                        )
                        if key in rows:
                            continue
                        rate = run_case(
                            args,
                            backend,
                            opt_level,
                            ast_mode,
                            tile_rows,
                            asts_per_kernel,
                        )
                        rows[key] = normalized_row(
                            backend,
                            opt_level,
                            ast_mode,
                            tile_rows,
                            asts_per_kernel,
                            rate,
                            args,
                        )
                        backend_rows = {
                            row_key_value: row
                            for row_key_value, row in rows.items()
                            if row_key_value[0] == backend
                        }
                        write_rows(output_paths[backend], backend_rows)
                        print(
                            f"{len(rows)}/{total} {backend} O{opt_level} "
                            f"{ast_mode} tile={tile_rows} asts={asts_per_kernel} "
                            f"row-evals/s={rate:.3e}",
                            flush=True,
                        )
    for backend, path in output_paths.items():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
