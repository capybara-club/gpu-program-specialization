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


BACKEND_COLORS = {
    "cuda": "#E69F00",
    "ptx": "#0072B2",
    "cubin": "#009E73",
    "hip": "#D55E00",
    "hsaco": "#CC79A7",
}
OPT_MARKERS = {
    0: "o",
    1: "s",
}


def rate_label(value, _position):
    if value >= 1.0e12:
        return f"{value / 1.0e12:g}T"
    if value >= 1.0e9:
        return f"{value / 1.0e9:g}G"
    if value >= 1.0e6:
        return f"{value / 1.0e6:g}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:g}K"
    return f"{value:g}"


def read_data(paths):
    rows = []
    systems = set()
    metadata = set()
    keys = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                backend = row["backend"]
                opt_level = int(row["opt_level"])
                if backend not in BACKEND_COLORS:
                    raise RuntimeError(f"unsupported backend in CSV: {backend}")
                if opt_level not in OPT_MARKERS:
                    raise RuntimeError(f"unsupported optimization level: {opt_level}")
                key = (backend, opt_level, int(row["rows"]))
                if key in keys:
                    raise RuntimeError(f"CSV inputs contain duplicate row: {key}")
                keys.add(key)
                systems.add(row["system"])
                metadata.add((
                    row["shape"],
                    row["ast_mode"],
                    int(row["kernels"]),
                    int(row["asts_per_kernel"]),
                    int(row["asts"]),
                    int(row["tile_rows"]),
                    int(row["threads"]),
                    int(row["streams"]),
                    int(row["iterations"]),
                    row["corpus"],
                    row["corpus_hash"],
                ))
                rows.append({
                    "backend": backend,
                    "opt_level": opt_level,
                    "rows": int(row["rows"]),
                    "runtime_rate": float(row["runtime_row_evals_per_second"]),
                    "pipeline_rate": float(row["pipeline_row_evals_per_second"]),
                })
    if not rows:
        raise RuntimeError("CSV contains no end-to-end data")
    if len(systems) != 1:
        raise RuntimeError("graph inputs must describe exactly one system")
    if len(metadata) != 1:
        raise RuntimeError("graph inputs mix incompatible benchmark configurations")
    return next(iter(systems)), next(iter(metadata)), rows


def draw_graph(paths, output, linear_y):
    system, metadata, rows = read_data(paths)
    (
        shape,
        ast_mode,
        kernels,
        asts_per_kernel,
        asts,
        tile_rows,
        threads,
        streams,
        iterations,
        corpus,
        corpus_hash,
    ) = metadata
    configurations = sorted({
        (row["backend"], row["opt_level"])
        for row in rows
    })

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#444444",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-end-to-end-frontier",
    })
    figure, axis = pyplot.subplots(figsize=(12.5, 7.8))

    for backend, opt_level in configurations:
        points = sorted(
            (
                row["rows"],
                row["runtime_rate"],
                row["pipeline_rate"],
            )
            for row in rows
            if row["backend"] == backend and row["opt_level"] == opt_level
        )
        color = BACKEND_COLORS[backend]
        marker = OPT_MARKERS[opt_level]
        axis.plot(
            [point[0] for point in points],
            [point[2] for point in points],
            color=color,
            linestyle="-",
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=6.5,
            linewidth=2.4,
            zorder=3,
        )
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            linestyle=":",
            marker=marker,
            markerfacecolor="#FFFFFF",
            markeredgecolor=color,
            markeredgewidth=1.4,
            markersize=6.5,
            linewidth=2.0,
            zorder=2,
        )

    axis.set_xscale("log", base=2)
    if not linear_y:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0.0)
    row_counts = sorted({row["rows"] for row in rows})
    axis.set_xticks(row_counts, [f"{value:,}" for value in row_counts])
    axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
    axis.grid(which="major", color="#D8D8D8", linewidth=0.75)
    axis.grid(which="minor", color="#EEEEEE", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.set_xlabel("Rows evaluated per AST")
    axis.set_ylabel("Effective row evaluations per second")

    configuration_handles = [
        Line2D(
            [0],
            [0],
            color=BACKEND_COLORS[backend],
            linestyle="-",
            marker=OPT_MARKERS[opt_level],
            markerfacecolor=BACKEND_COLORS[backend],
            markeredgecolor=BACKEND_COLORS[backend],
            linewidth=2.4,
            markersize=6.5,
            label=f"{backend.upper()} O{opt_level}",
        )
        for backend, opt_level in configurations
    ]
    configuration_legend = axis.legend(
        handles=configuration_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        title="Backend",
        ncol=1 if len(configuration_handles) <= 3 else 2,
    )
    axis.add_artist(configuration_legend)
    phase_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="-",
            linewidth=2.4,
            label="Compile + module load + run",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=":",
            linewidth=2.0,
            label="Run only",
        ),
    ]
    axis.legend(
        handles=phase_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        title="Timing boundary",
    )

    figure.suptitle(
        f"SECANT Compile-to-Execution Frontier on {system}",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"{shape.upper()} · {ast_mode.upper()} · {asts} ASTs "
            f"({kernels} kernels × {asts_per_kernel} ASTs) · "
            f"{streams} streams · one pinned CPU core"
        ),
        ha="center",
        color="#444444",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.040,
        (
            f"Inputs are resident on device; transfers, cold template creation, "
            f"template inspection, and correctness checks are outside timing. "
            f"Tile={tile_rows}; threads={threads}; iterations={iterations}."
        ),
        ha="center",
        color="#444444",
        fontsize=8.8,
    )
    figure.text(
        0.5,
        0.017,
        (
            f"Every marker is a measured run. Corpus={corpus}; "
            f"definition hash={corpus_hash}. "
            f"{'Linear' if linear_y else 'Logarithmic'} Y axis."
        ),
        ha="center",
        color="#555555",
        fontsize=8.3,
    )
    figure.tight_layout(rect=(0.04, 0.075, 0.99, 0.89))

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format=output.suffix.lstrip(".") or "svg",
        metadata={"Date": None, "Creator": "SECANT end-to-end graph"},
    )
    pyplot.close(figure)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Graph SECANT compile + load + run throughput against runtime-only throughput.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-y", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    draw_graph(args.inputs, args.output, args.linear_y)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
