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
from pathlib import Path
import sys

from backend_csv import BACKENDS

KERNEL_SHAPES = ("materialize", "sse")
OPT_LEVELS = (0, 1)
AST_MODES = ("alu", "mufu")
KERNEL_SHAPE_LABELS = {
    "materialize": "Materialize",
    "sse": "SSE",
}
IDENTITY_COLUMNS = {
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
}


def read_matrix(paths):
    shape_names = None
    matrix = {}
    workers = set()
    backends = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise RuntimeError(f"{path} has no header")
            file_shape_names = [
                name for name in reader.fieldnames
                if name not in IDENTITY_COLUMNS
            ]
            if shape_names is None:
                shape_names = file_shape_names
            elif shape_names != file_shape_names:
                raise RuntimeError(f"{path} has incompatible module-shape columns")
            rows = list(reader)

        for row in rows:
            try:
                key = (
                    int(row["workers"]),
                    row["kernel_shape"],
                    int(row["opt_level"]),
                    row["ast_mode"],
                    row["backend"],
                )
            except (KeyError, ValueError) as error:
                raise RuntimeError(f"{path} has invalid row identity") from error
            if key in matrix:
                raise RuntimeError(f"CSV inputs contain duplicate row: {key}")
            if row["backend"] not in BACKENDS:
                continue
            try:
                matrix[key] = {
                    shape_name: float(row[shape_name])
                    for shape_name in shape_names
                }
            except (KeyError, ValueError) as error:
                raise RuntimeError(f"{path} has invalid values for row: {key}") from error
            workers.add(key[0])
            backends.add(key[4])

    if not shape_names or not matrix:
        raise RuntimeError("CSV inputs contain no matrix data")
    ordered_backends = [backend for backend in BACKENDS if backend in backends]
    return shape_names, sorted(workers), ordered_backends, matrix


def format_rate(value, fastest):
    text = f"{value:,.0f}"
    return f"**{text}**" if fastest else text


def render_report(input_paths, graph_path, shape_names, worker_counts, backends, matrix):
    source_links = ", ".join(
        f"[{path.name}]({path.name})"
        for path in input_paths)
    lines = [
        "# SECANT Compile Matrix",
        "",
        f"Sources: {source_links}",
        "",
        "Each cell reports aggregate AST/s. Module-shape columns are",
        "`kernels/module x ASTs/kernel`. The faster backend for each module shape",
        "is shown in bold.",
        "",
        "For the PTX backend, NVRTC reduces the generated CUDA template to",
        "optimized PTX once during handle creation, outside the measured hot path.",
        "The timed compile therefore starts from this simpler template and covers",
        "AST injection plus nvPTXCompiler. This preprocessing is also why",
        "nvPTXCompiler O1 can occasionally compile SSE faster than O0: its cleanup",
        "can reduce later lowering, register-allocation, and emission work by more",
        "than the optimization passes cost.",
        "",
    ]
    if graph_path is not None:
        lines.extend((
            f"![SECANT compile throughput]({graph_path.name})",
            "",
        ))

    for workers in worker_counts:
        worker_label = "Worker" if workers == 1 else "Workers"
        lines.extend((
            f"## {workers} {worker_label}",
            "",
            "| Configuration | Backend | " + " | ".join(shape_names) + " |",
            "|---|---|" + "|".join("---:" for _ in shape_names) + "|",
        ))
        for kernel_shape in KERNEL_SHAPES:
            for opt_level in OPT_LEVELS:
                for ast_mode in AST_MODES:
                    configuration = (
                        f"{KERNEL_SHAPE_LABELS[kernel_shape]} "
                        f"O{opt_level} {ast_mode.upper()}")
                    rows = []
                    for backend in backends:
                        key = (
                            workers,
                            kernel_shape,
                            opt_level,
                            ast_mode,
                            backend,
                        )
                        if key in matrix:
                            rows.append((backend, matrix[key]))
                    if not rows:
                        continue

                    for row_index, (backend, values) in enumerate(rows):
                        cells = []
                        for shape_name in shape_names:
                            fastest = values[shape_name] == max(
                                row_values[shape_name]
                                for _, row_values in rows)
                            cells.append(format_rate(values[shape_name], fastest))
                        configuration_cell = (
                            configuration if row_index == 0 else "")
                        lines.append(
                            f"| {configuration_cell} | {backend.upper()} | "
                            + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Render a SECANT compile matrix CSV as Markdown tables.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--graph", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    shape_names, workers, backends, matrix = read_matrix(args.inputs)
    report = render_report(
        args.inputs,
        args.graph,
        shape_names,
        workers,
        backends,
        matrix,
    )

    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
