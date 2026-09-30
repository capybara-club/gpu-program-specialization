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
BACKEND_COLORS = {
    "ptx": "#0072B2",
    "cuda": "#D55E00",
    "cubin": "#009E73",
}
BACKEND_MARKERS = {
    "ptx": "o",
    "cuda": "s",
    "cubin": "^",
}
BACKEND_LINESTYLES = {
    "ptx": "-",
    "cuda": "--",
    "cubin": "-.",
}


def rate_label(value, _position):
    if value >= 1.0e12:
        return f"{value / 1.0e12:g}T"
    if value >= 1.0e9:
        return f"{value / 1.0e9:g}G"
    return f"{value:g}"


def read_sweep(paths):
    data = {}
    metadata = {}
    backends = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key = (
                    row["backend"],
                    int(row["opt_level"]),
                    row["ast_mode"],
                    int(row["tile_rows"]),
                )
                if key in data:
                    raise RuntimeError(f"CSV inputs contain duplicate row: {key}")
                data[key] = float(row["row_evals_per_second"])
                backends.add(key[0])
                row_metadata = (
                    int(row["kernels"]),
                    int(row["asts_per_kernel"]),
                    int(row["threads"]),
                    int(row["run_rows"]),
                    int(row["run_iterations"]),
                )
                if key[0] not in metadata:
                    metadata[key[0]] = row_metadata
                elif metadata[key[0]] != row_metadata:
                    raise RuntimeError(
                        f"CSV inputs mix incompatible {key[0]} runtime configurations")
    if not data or not metadata:
        raise RuntimeError("CSV contains no runtime data")
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
                    key = (backend, opt_level, ast_mode, tile_row_count)
                    if key not in data:
                        raise RuntimeError(f"CSV is missing row: {key}")
    return tile_rows, ordered_backends, data, metadata


def draw_graph(
    input_paths,
    output_path,
    requested_backends=None,
    log_scale=False,
):
    tile_rows, available_backends, data, metadata = read_sweep(input_paths)
    backends = select_backends(available_backends, requested_backends)
    opt_levels = [
        opt_level for opt_level in OPT_LEVELS
        if any(
            key[0] in backends and key[1] == opt_level
            for key in data
        )
    ]
    metadata_values = set(metadata.values())
    x_values = list(range(len(tile_rows)))

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#444444",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-runtime-tile-sweep",
    })
    figure, axes = pyplot.subplots(
        len(AST_MODES),
        len(opt_levels),
        figsize=(13.0, 9.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    selected_values = [
        value
        for key, value in data.items()
        if key[0] in backends
    ]
    minimum = min(selected_values)
    maximum = max(selected_values)

    for row_idx, ast_mode in enumerate(AST_MODES):
        for column_idx, opt_level in enumerate(opt_levels):
            axis = axes[row_idx][column_idx]
            for backend in backends:
                points = [
                    (index, data[(backend, opt_level, ast_mode, tile_row_count)])
                    for index, tile_row_count in enumerate(tile_rows)
                    if (backend, opt_level, ast_mode, tile_row_count) in data
                ]
                if not points:
                    continue
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=BACKEND_COLORS[backend],
                    linestyle=BACKEND_LINESTYLES[backend],
                    marker=BACKEND_MARKERS[backend],
                    markersize=5.0,
                    linewidth=2.1,
                )

            if log_scale:
                axis.set_yscale("log")
                axis.set_ylim(minimum / 1.15, maximum * 1.15)
            else:
                axis.set_ylim(0.0, maximum * 1.08)
            axis.set_xticks(
                x_values,
                [f"{tile_row_count:,}" for tile_row_count in tile_rows],
                rotation=30,
                ha="right",
            )
            axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
            axis.set_axisbelow(True)
            if row_idx == 0:
                axis.set_title(f"O{opt_level}", fontsize=13, pad=10)
            if column_idx == 0:
                axis.set_ylabel(f"{ast_mode.upper()}\nrow-evals/s", labelpad=10)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BACKEND_COLORS[backend],
            linestyle=BACKEND_LINESTYLES[backend],
            marker=BACKEND_MARKERS[backend],
            linewidth=2.1,
            markersize=5.0,
            label=backend.upper(),
        )
        for backend in backends
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        frameon=False,
        ncol=2,
    )
    figure.suptitle(
        "SECANT SSE Runtime Throughput by Logical Tile Size",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.940,
        f"{'Logarithmic' if log_scale else 'Linear'} Y axis; higher is better",
        ha="center",
        color="#444444",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.050,
        (
            f"{next(iter(metadata_values))[0]} kernels; "
            f"{next(iter(metadata_values))[1]} ASTs/kernel; "
            f"{next(iter(metadata_values))[2]} threads; "
            f"{next(iter(metadata_values))[3]:,} rows; "
            f"{next(iter(metadata_values))[4]} timed iterations"
            if len(metadata_values) == 1
            else "Backend-specific AST packing; see the source CSV metadata"
        ),
        ha="center",
        color="#444444",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.030,
        "Tile size = logical rows per CTA; O0/O1 = final ptxas or "
        "nvPTXCompiler optimization level",
        ha="center",
        color="#444444",
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.012,
        "ALU = balanced 8-leaf add/mul/min/max trees; "
        "MUFU adds sin/cos/ex2, safe sqrt/rsqrt, and occasional safe divide",
        ha="center",
        color="#444444",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.04, 0.10, 0.99, 0.81), h_pad=2.0, w_pad=1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "svg",
        metadata={"Date": None, "Creator": "SECANT runtime tile graph"},
    )
    pyplot.close(figure)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Graph a SECANT SSE runtime tile sweep.")
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
