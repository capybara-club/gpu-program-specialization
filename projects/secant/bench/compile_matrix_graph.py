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
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from compile_matrix_report import (
    AST_MODES,
    KERNEL_SHAPES,
    KERNEL_SHAPE_LABELS,
    OPT_LEVELS,
    read_matrix,
)
from backend_csv import BACKENDS, select_backends


BACKEND_COLORS = {
    "ptx": "#0072B2",
    "cuda": "#D55E00",
    "cubin": "#009E73",
}
WORKER_LINESTYLES = {
    1: ":",
    12: "--",
    24: "-",
}
WORKER_MARKERS = {
    1: "o",
    12: "^",
    24: "s",
}


def rate_label(value, _position):
    if value >= 1.0e6:
        return f"{value / 1.0e6:g}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:g}k"
    return f"{value:g}"


def draw_graph(
    input_paths,
    output_path,
    requested_backends=None,
    log_scale=False,
):
    shape_names, worker_counts, available_backends, matrix = read_matrix(input_paths)
    backends = select_backends(available_backends, requested_backends)
    unsupported_workers = [
        workers
        for workers in worker_counts
        if workers not in WORKER_LINESTYLES
    ]
    if unsupported_workers:
        raise RuntimeError(
            f"graph has no style for worker count: {unsupported_workers[0]}")

    column_configs = [
        (opt_level, ast_mode)
        for opt_level in OPT_LEVELS
        for ast_mode in AST_MODES
        if any(
            key[2] == opt_level and
            key[3] == ast_mode and
            key[4] in backends
            for key in matrix
        )
    ]
    row_count = len(KERNEL_SHAPES)
    column_count = len(column_configs)

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-compile-matrix",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })
    figure, axes = pyplot.subplots(
        row_count,
        column_count,
        figsize=(16.0, 9.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    x_values = list(range(len(shape_names)))

    for row_idx, kernel_shape in enumerate(KERNEL_SHAPES):
        row_values = [
            value
            for (workers, shape, opt_level, ast_mode, backend), values
            in matrix.items()
            if shape == kernel_shape and backend in backends
            for value in values.values()
        ]
        row_min = min(row_values)
        row_max = max(row_values)
        for column_idx, (opt_level, ast_mode) in enumerate(column_configs):
            axis = axes[row_idx][column_idx]
            for backend in backends:
                for workers in worker_counts:
                    values = matrix.get((
                        workers,
                        kernel_shape,
                        opt_level,
                        ast_mode,
                        backend,
                    ))
                    if values is None:
                        continue
                    rates = [values[shape_name] for shape_name in shape_names]
                    axis.plot(
                        x_values,
                        rates,
                        color=BACKEND_COLORS[backend],
                        linestyle=WORKER_LINESTYLES[workers],
                        marker=WORKER_MARKERS[workers],
                        markersize=4.5,
                        linewidth=1.8,
                    )

            if log_scale:
                axis.set_yscale("log")
                axis.set_ylim(row_min / 1.15, row_max * 1.15)
            else:
                axis.set_ylim(0.0, row_max * 1.08)
            axis.set_xticks(x_values, shape_names, rotation=35, ha="right")
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_xlabel("Module shape")
            axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
            axis.grid(
                axis="y",
                which="major",
                color="#D8D8D8",
                linewidth=0.7,
            )
            axis.set_axisbelow(True)

            if row_idx == 0:
                axis.set_title(f"O{opt_level} {ast_mode.upper()}", pad=10)
            if column_idx == 0:
                axis.set_ylabel(
                    f"{KERNEL_SHAPE_LABELS[kernel_shape]}\nAST/s",
                    labelpad=10,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BACKEND_COLORS[backend],
            linewidth=2.0,
            label=backend.upper(),
        )
        for backend in backends
    ]
    worker_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=WORKER_LINESTYLES[workers],
            marker=WORKER_MARKERS[workers],
            linewidth=1.8,
            markersize=5.0,
            label=f"{workers} worker" if workers == 1 else f"{workers} workers",
        )
        for workers in worker_counts
    ]
    figure.legend(
        handles=legend_handles + worker_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        frameon=False,
        ncol=5,
    )
    figure.suptitle(
        "SECANT Compile Throughput by Module Shape",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        f"Aggregate AST/s; {'logarithmic' if log_scale else 'linear'} "
        "Y axis; higher is better",
        ha="center",
        color="#444444",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.050,
        "Module shape = kernels/module x ASTs/kernel",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.030,
        "O0/O1 = final backend optimization level "
        "(CUDA ptxas; PTX nvPTXCompiler); NVRTC remains optimized",
        ha="center",
        color="#444444",
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.012,
        "ALU = balanced 8-leaf add/mul/min/max trees; "
        "MUFU = ALU trees plus sin/cos/ex2, safe sqrt/rsqrt, "
        "and occasional safe divide",
        ha="center",
        color="#444444",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.03, 0.10, 0.99, 0.84), h_pad=2.0, w_pad=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "svg",
        metadata={"Date": None, "Creator": "SECANT compile matrix graph"},
    )
    pyplot.close(figure)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Graph a SECANT compile matrix CSV.")
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
