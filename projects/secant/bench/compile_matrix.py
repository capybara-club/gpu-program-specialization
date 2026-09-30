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
import shlex
import subprocess
import sys
import tempfile

from backend_csv import BACKENDS, backend_csv_path, backend_opt_levels

DEFAULT_MODULE_SHAPES = (
    "8x24",
    "16x24",
    "32x24",
    "64x8",
    "64x16",
    "64x64",
)

KERNEL_SHAPES = ("materialize", "sse")
OPT_LEVELS = (0, 1)
AST_MODES = ("alu", "mufu")
WORKERS = (1, 12, 24)

IDENTITY_COLUMNS = (
    "backend",
    "kernel_shape",
    "opt_level",
    "ast_mode",
    "metric",
    "workers",
    "warmup_modules",
    "timed_modules",
    "source_sm",
    "target_sm",
    "tile_rows",
    "threads",
    "patch_instructions_per_ast",
)


def parse_module_shape(text):
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"invalid module shape: {text}")
    try:
        kernels = int(parts[0], 10)
        asts = int(parts[1], 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid module shape: {text}") from error
    if kernels <= 0 or asts <= 0:
        raise argparse.ArgumentTypeError(f"invalid module shape: {text}")
    return kernels, asts


def module_shape_name(shape):
    return f"{shape[0]}x{shape[1]}"


def find_default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo.parent / "build" / "secant_compile_bench",
        repo / "build" / "secant_compile_bench",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def configurations(args):
    for workers in args.workers:
        for backend in args.backends:
            for kernel_shape in args.kernel_shapes:
                for opt_level in backend_opt_levels(backend, args.opt_levels):
                    for ast_mode in args.ast_modes:
                        yield workers, backend, kernel_shape, opt_level, ast_mode


def result_key(workers, backend, kernel_shape, opt_level, ast_mode):
    return int(workers), backend, kernel_shape, int(opt_level), ast_mode


def metadata(args):
    return {
        "metric": "aggregate_asts_per_second",
        "source_sm": str(args.source_sm),
        "target_sm": str(args.target_sm),
        "tile_rows": str(args.tile_rows),
        "threads": str(args.threads),
        "patch_instructions_per_ast": str(args.patch_instructions_per_ast),
    }


def load_results(path, args, shape_names):
    results = {}
    if not path.exists():
        return results

    expected_columns = list(IDENTITY_COLUMNS) + shape_names
    legacy_columns = [
        name for name in IDENTITY_COLUMNS
        if name != "patch_instructions_per_ast"
    ] + shape_names
    expected_metadata = metadata(args)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames not in (expected_columns, legacy_columns):
            raise RuntimeError(
                f"{path} has incompatible columns; use --force or another output path")
        for row in reader:
            for name, expected in expected_metadata.items():
                if name not in row:
                    continue
                if row[name] != expected:
                    raise RuntimeError(
                        f"{path} was generated with different {name}; "
                        "use --force or another output path")
            key = result_key(
                row["workers"],
                row["backend"],
                row["kernel_shape"],
                row["opt_level"],
                row["ast_mode"])
            results[key] = {
                shape_name: row[shape_name]
                for shape_name in shape_names
                if row[shape_name] != ""
            }
    return results


def write_results(path, args, backend, module_shapes, results):
    shape_names = [module_shape_name(shape) for shape in module_shapes]
    fieldnames = list(IDENTITY_COLUMNS) + shape_names
    common_metadata = metadata(args)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False)
    try:
        with temporary:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            for workers, row_backend, kernel_shape, opt_level, ast_mode in configurations(args):
                if row_backend != backend:
                    continue
                key = result_key(workers, row_backend, kernel_shape, opt_level, ast_mode)
                row = {
                    "backend": row_backend,
                    "kernel_shape": kernel_shape,
                    "opt_level": str(opt_level),
                    "ast_mode": ast_mode,
                    "workers": str(workers),
                    "warmup_modules": str(workers * args.warmup_waves),
                    "timed_modules": str(workers * args.timed_waves),
                    **common_metadata,
                }
                row.update(results.get(key, {}))
                writer.writerow(row)
        os.replace(temporary.name, path)
    except BaseException:
        try:
            os.unlink(temporary.name)
        except FileNotFoundError:
            pass
        raise


def benchmark_command(
    args,
    workers,
    backend,
    kernel_shape,
    opt_level,
    ast_mode,
    module_shape,
    seed
):
    kernels, asts_per_kernel = module_shape
    return [
        str(args.benchmark),
        "--backend", backend,
        "--shape", kernel_shape,
        "--ast-mode", ast_mode,
        "--opt-level", str(opt_level),
        "--kernels", str(kernels),
        "--asts-per-kernel", str(asts_per_kernel),
        "--workers", str(workers),
        "--warmups", str(workers * args.warmup_waves),
        "--iterations", str(workers * args.timed_waves),
        "--check-modules", "0",
        "--tile-rows", str(args.tile_rows),
        "--threads", str(args.threads),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--compile-scratch-bytes", str(args.compile_scratch_bytes),
        "--source-sm", str(args.source_sm),
        "--target-sm", str(args.target_sm),
        "--seed", str(seed),
    ]


def parse_compile_result(output, expected):
    fields = None
    for line in output.splitlines():
        if not line.startswith("compile "):
            continue
        fields = {}
        for item in line.split()[1:]:
            if "=" in item:
                name, value = item.split("=", 1)
                fields[name] = value
    if fields is None:
        raise RuntimeError("benchmark output did not contain a compile result")

    workers, backend, kernel_shape, opt_level, ast_mode, module_shape = expected
    expected_fields = {
        "backend": backend,
        "shape": kernel_shape,
        "opt_level": str(opt_level),
        "ast_mode": ast_mode,
        "workers": str(workers),
        "kernels_per_module": str(module_shape[0]),
        "asts_per_kernel": str(module_shape[1]),
    }
    for name, value in expected_fields.items():
        if fields.get(name) != value:
            raise RuntimeError(
                f"benchmark returned {name}={fields.get(name)!r}, expected {value!r}")
    try:
        return float(fields["asts_per_second"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("benchmark returned an invalid AST/s value") from error


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate a compile-throughput matrix from secant_compile_bench.")
    parser.add_argument("--benchmark", type=Path, default=find_default_benchmark())
    parser.add_argument("--output", type=Path, default=Path("secant_compile_matrix.csv"))
    parser.add_argument("--import-csv", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--module-shapes",
        nargs="+",
        type=parse_module_shape,
        default=[parse_module_shape(shape) for shape in DEFAULT_MODULE_SHAPES],
        metavar="KxA")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument(
        "--kernel-shapes",
        nargs="+",
        choices=KERNEL_SHAPES,
        default=list(KERNEL_SHAPES))
    parser.add_argument("--opt-levels", nargs="+", type=int, choices=OPT_LEVELS, default=list(OPT_LEVELS))
    parser.add_argument("--ast-modes", nargs="+", choices=AST_MODES, default=list(AST_MODES))
    parser.add_argument("--workers", nargs="+", type=int, default=list(WORKERS))
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--timed-waves", type=int, default=2)
    parser.add_argument("--source-sm", type=int, default=80)
    parser.add_argument("--target-sm", type=int, default=120)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--patch-instructions-per-ast", type=int, default=64)
    parser.add_argument("--compile-scratch-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    numeric_values = (
        args.timed_waves,
        args.source_sm,
        args.target_sm,
        args.tile_rows,
        args.threads,
        args.patch_instructions_per_ast,
        args.compile_scratch_bytes,
    )
    if (
        any(value <= 0 for value in numeric_values) or
        any(workers <= 0 for workers in args.workers) or
        args.warmup_waves < 0 or
        args.seed < 0
    ):
        parser.error("numeric benchmark arguments must be positive")
    if len(set(args.workers)) != len(args.workers):
        parser.error("worker counts must be unique")
    if len(set(args.module_shapes)) != len(args.module_shapes):
        parser.error("module shapes must be unique")
    if len(set(args.backends)) != len(args.backends):
        parser.error("backends must be unique")
    if any(
        len(set(values)) != len(values)
        for values in (
            args.kernel_shapes,
            args.opt_levels,
            args.ast_modes,
        )
    ):
        parser.error("configuration lists must contain unique values")
    if not any(
        backend_opt_levels(backend, args.opt_levels)
        for backend in args.backends
    ):
        parser.error("the selected backends have no supported optimization levels")
    return args


def main(argv=None):
    args = parse_args(argv)
    shape_names = [module_shape_name(shape) for shape in args.module_shapes]
    output_paths = {
        backend: backend_csv_path(args.output, backend)
        for backend in args.backends
    }

    if args.force and not args.dry_run:
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    results = {}
    if not args.dry_run:
        for path in args.import_csv:
            results.update(load_results(path, args, shape_names))
        for path in output_paths.values():
            results.update(load_results(path, args, shape_names))
    total = sum(
        len(backend_opt_levels(backend, args.opt_levels)) *
        len(args.kernel_shapes) *
        len(args.ast_modes) *
        len(args.workers) *
        len(args.module_shapes)
        for backend in args.backends)
    completed = 0

    for workers, backend, kernel_shape, opt_level, ast_mode in configurations(args):
        key = result_key(workers, backend, kernel_shape, opt_level, ast_mode)
        row_results = results.setdefault(key, {})
        for module_shape in args.module_shapes:
            completed += 1
            shape_name = module_shape_name(module_shape)
            seed = args.seed + completed - 1
            command = benchmark_command(
                args,
                workers,
                backend,
                kernel_shape,
                opt_level,
                ast_mode,
                module_shape,
                seed)
            if args.dry_run:
                print(shlex.join(command))
                continue
            if shape_name in row_results:
                print(
                    f"[{completed}/{total}] resume {backend} {kernel_shape} "
                    f"O{opt_level} {ast_mode} {workers}w {shape_name}",
                    flush=True)
                continue

            print(
                f"[{completed}/{total}] run {backend} {kernel_shape} "
                f"O{opt_level} {ast_mode} {workers}w {shape_name}",
                flush=True)
            process = subprocess.run(command, text=True, capture_output=True)
            if process.stdout:
                print(process.stdout, end="")
            if process.stderr:
                print(process.stderr, end="", file=sys.stderr)
            if process.returncode != 0:
                raise RuntimeError(
                    f"benchmark failed with exit code {process.returncode}: "
                    f"{shlex.join(command)}")
            value = parse_compile_result(
                process.stdout,
                (workers, backend, kernel_shape, opt_level, ast_mode, module_shape))
            row_results[shape_name] = f"{value:.3f}"
            write_results(
                output_paths[backend],
                args,
                backend,
                args.module_shapes,
                results)

    if not args.dry_run:
        for backend, path in output_paths.items():
            write_results(path, args, backend, args.module_shapes, results)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
