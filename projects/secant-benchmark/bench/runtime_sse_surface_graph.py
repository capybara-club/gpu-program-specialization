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

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from backend_csv import BACKENDS, select_backends

OPT_LEVELS = (0, 1)
AST_MODES = ("alu", "mufu")
TILE_COLORS = (
    "#332288",
    "#117733",
    "#44AA99",
    "#3B74B2",
    "#B38B00",
    "#CC6677",
    "#AA4499",
)
TILE_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def rate_label(value, _position):
    if value >= 1.0e12:
        return f"{value / 1.0e12:g}T"
    if value >= 1.0e9:
        return f"{value / 1.0e9:g}G"
    return f"{value:g}"


def read_sweep(paths):
    data = {}
    metadata = None
    backends = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key = (
                    row["backend"],
                    int(row["opt_level"]),
                    row["ast_mode"],
                    int(row["tile_rows"]),
                    int(row["asts_per_kernel"]),
                )
                if key in data:
                    raise RuntimeError(f"CSV inputs contain duplicate row: {key}")
                data[key] = float(row["row_evals_per_second"])
                backends.add(key[0])
                row_metadata = (
                    int(row["kernels"]),
                    int(row["threads"]),
                    int(row["run_rows"]),
                    int(row["run_iterations"]),
                )
                if metadata is None:
                    metadata = row_metadata
                elif metadata != row_metadata:
                    raise RuntimeError("CSV inputs mix incompatible runtime configurations")
    if not data or metadata is None:
        raise RuntimeError("CSV contains no runtime data")
    ast_counts = sorted({key[4] for key in data})
    tile_rows = sorted({key[3] for key in data})
    ordered_backends = [backend for backend in BACKENDS if backend in backends]
    for backend in ordered_backends:
        opt_levels = sorted({
            key[1]
            for key in data
            if key[0] == backend
        })
        for opt_level in opt_levels:
            for ast_mode in AST_MODES:
                for tile_row_count in tile_rows:
                    backend_ast_counts = sorted({
                        key[4]
                        for key in data
                        if key[0] == backend
                    })
                    for ast_count in backend_ast_counts:
                        key = (
                            backend,
                            opt_level,
                            ast_mode,
                            tile_row_count,
                            ast_count,
                        )
                        if key not in data:
                            raise RuntimeError(f"CSV is missing row: {key}")
    return ast_counts, tile_rows, ordered_backends, data, metadata


def draw_graph(
    input_paths,
    output_path,
    requested_backends=None,
    log_scale=False,
):
    ast_counts, tile_rows, available_backends, data, metadata = read_sweep(input_paths)
    backends = select_backends(available_backends, requested_backends)
    opt_levels = [
        opt_level for opt_level in OPT_LEVELS
        if any(
            key[0] in backends and key[1] == opt_level
            for key in data
        )
    ]
    if len(tile_rows) > len(TILE_COLORS):
        raise RuntimeError("graph does not have enough tile styles")
    kernels, threads, run_rows, run_iterations = metadata
    row_configs = [
        (backend, ast_mode)
        for ast_mode in AST_MODES
        for backend in backends
    ]
    opt_maximum = {
        opt_level: max(
            value
            for key, value in data.items()
            if key[0] in backends and key[1] == opt_level
        )
        for opt_level in opt_levels
    }
    opt_minimum = {
        opt_level: min(
            value
            for key, value in data.items()
            if key[0] in backends and key[1] == opt_level
        )
        for opt_level in opt_levels
    }
    x_values = list(range(len(ast_counts)))

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": "#444444",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-runtime-sse-surface",
    })
    figure, axes = pyplot.subplots(
        len(row_configs),
        len(opt_levels),
        figsize=(16.0, 14.0),
        sharex=True,
        squeeze=False,
    )

    for row_idx, (backend, ast_mode) in enumerate(row_configs):
        for column_idx, opt_level in enumerate(opt_levels):
            axis = axes[row_idx][column_idx]
            for tile_idx, tile_row_count in enumerate(tile_rows):
                points = [
                    (index, data[(
                        backend,
                        opt_level,
                        ast_mode,
                        tile_row_count,
                        ast_count,
                    )])
                    for index, ast_count in enumerate(ast_counts)
                    if (
                        backend,
                        opt_level,
                        ast_mode,
                        tile_row_count,
                        ast_count,
                    ) in data
                ]
                if not points:
                    continue
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=TILE_COLORS[tile_idx],
                    marker=TILE_MARKERS[tile_idx],
                    markersize=3.8,
                    linewidth=1.7,
                )

            if log_scale:
                axis.set_yscale("log")
                axis.set_ylim(
                    opt_minimum[opt_level] / 1.15,
                    opt_maximum[opt_level] * 1.15,
                )
            else:
                axis.set_ylim(0.0, opt_maximum[opt_level] * 1.08)
            axis.set_xticks(
                x_values,
                [str(ast_count) for ast_count in ast_counts],
                rotation=30,
                ha="right",
            )
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_xlabel("ASTs/kernel")
            axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
            axis.set_axisbelow(True)
            if row_idx == 0:
                axis.set_title(f"O{opt_level}", fontsize=13, pad=10)
            if column_idx == 0:
                axis.set_ylabel(
                    f"{backend.upper()} {ast_mode.upper()}\nrow-evals/s",
                    labelpad=10,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TILE_COLORS[tile_idx],
            marker=TILE_MARKERS[tile_idx],
            linewidth=1.7,
            markersize=4.5,
            label=f"{tile_row_count:,}",
        )
        for tile_idx, tile_row_count in enumerate(tile_rows)
    ]
    figure.legend(
        handles=legend_handles,
        title="Logical rows/CTA",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        frameon=False,
        ncol=len(tile_rows),
    )
    figure.suptitle(
        "SECANT SSE Runtime by AST Packing and Logical Tile Size",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.950,
        f"{'Logarithmic' if log_scale else 'Linear'} Y axes; each "
        "optimization column has its own scale; higher is better",
        ha="center",
        color="#444444",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.032,
        f"{kernels} kernels; {threads} threads; {run_rows:,} rows; "
        f"{run_iterations} timed iterations",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.015,
        "O0/O1 = final ptxas or nvPTXCompiler optimization level; "
        "X axis = ASTs/kernel",
        ha="center",
        color="#444444",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.08,
        top=0.82,
        hspace=0.35,
        wspace=0.13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "svg",
        metadata={"Date": None, "Creator": "SECANT SSE surface graph"},
    )
    pyplot.close(figure)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Graph SSE runtime over AST packing and tile size.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--backends",
        choices=BACKENDS,
        nargs="+",
    )
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    draw_graph(
        args.inputs,
        args.output,
        requested_backends=args.backends,
        log_scale=args.log_scale,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
