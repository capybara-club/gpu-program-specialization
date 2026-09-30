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
from collections import defaultdict
from pathlib import Path
import sys

from common import read_csv


SHAPES = ("materialize", "sse")
METRICS = {
    "row-evals": ("row_evals_per_second", "Row evaluations / second"),
    "asts": ("asts_per_second", "AST evaluations / second"),
}


def series_label(record, all_configurations):
    system = record["system"].replace("_", " ")
    backend = record["backend"].replace("_", " ")
    if not all_configurations:
        return f"{system} | {backend}"
    mode = record["execution_mode"].replace("_", " ")
    workers = int(record["workers"])
    parts = [system]
    if backend not in system:
        parts.append(backend)
    if mode not in ("", backend):
        parts.append(mode)
    if workers > 1:
        parts.append(f"{workers} workers")
    return " | ".join(parts)


def select_best(
    records,
    shape,
    ast_mode,
    workers,
    metric,
    all_configurations,
):
    selected = {}
    for record in records:
        if record["shape"] != shape or record["ast_mode"] != ast_mode:
            continue
        if workers and int(record["workers"]) not in workers:
            continue
        key = (series_label(record, all_configurations), int(record["rows"]))
        if key not in selected or float(record[metric]) > float(selected[key][metric]):
            selected[key] = record
    grouped = defaultdict(list)
    for (label, _), record in selected.items():
        grouped[label].append(record)
    for values in grouped.values():
        values.sort(key=lambda record: int(record["rows"]))
    return grouped


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Graph normalized SECANT baseline CSV files together.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    parser.add_argument("--ast-mode", default="alu")
    parser.add_argument("--workers", type=int, nargs="*")
    parser.add_argument("--metric", choices=METRICS, default="row-evals")
    parser.add_argument(
        "--all-configurations",
        action="store_true",
        help="draw each worker and execution mode instead of the best per system",
    )
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument(
        "--connect-points",
        action="store_true",
        help="connect measured row counts with lines",
    )
    parser.add_argument("--title", default="Expression evaluation throughput")
    args = parser.parse_args(argv)
    if args.workers and any(value <= 0 for value in args.workers):
        parser.error("--workers must be positive")
    if len(args.shapes) != len(set(args.shapes)):
        parser.error("--shapes contains duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter

    records = []
    for path in args.inputs:
        records.extend(read_csv(path))
    corpus_keys = {
        (record["corpus"], record["corpus_hash"], record["seed"])
        for record in records
    }
    if len(corpus_keys) != 1:
        raise RuntimeError(
            "comparison inputs do not use one corpus, corpus hash, and seed: "
            f"{sorted(corpus_keys)}")

    metric, y_label = METRICS[args.metric]
    populated = []
    for shape in args.shapes:
        grouped = select_best(
            records,
            shape,
            args.ast_mode,
            set(args.workers or ()),
            metric,
            args.all_configurations,
        )
        if grouped:
            populated.append((shape, grouped))
    if not populated:
        raise RuntimeError("no CSV records matched the graph filters")

    figure, axes = plt.subplots(
        1,
        len(populated),
        figsize=(7.2 * len(populated), 5.4),
        squeeze=False,
    )
    for axis, (shape, grouped) in zip(axes[0], populated):
        for label, values in sorted(grouped.items()):
            x_values = [int(record["rows"]) for record in values]
            y_values = [float(record[metric]) for record in values]
            if args.connect_points:
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2.0,
                    label=label,
                )
            else:
                axis.scatter(x_values, y_values, s=45.0, label=label)
        if not args.linear_x:
            axis.set_xscale("log", base=2)
        if args.log_y:
            axis.set_yscale("log")
        else:
            axis.yaxis.set_major_formatter(EngFormatter())
        axis.set_xlabel("Rows per AST evaluation")
        axis.set_ylabel(y_label)
        axis.set_title(shape.upper())
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
