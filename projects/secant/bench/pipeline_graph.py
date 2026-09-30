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
AST_MODES = ("alu", "mufu")
SHAPES = ("materialize", "sse")


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


def read_data(path):
    rows = []
    systems = set()
    metadata = set()
    keys = set()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            backend = row["backend"]
            opt_level = int(row["opt_level"])
            shape = row["shape"]
            ast_mode = row["ast_mode"]
            if backend not in BACKEND_COLORS:
                raise RuntimeError(f"unsupported backend in CSV: {backend}")
            if opt_level not in OPT_MARKERS:
                raise RuntimeError(f"unsupported optimization level: {opt_level}")
            if ast_mode not in AST_MODES:
                raise RuntimeError(f"unsupported AST mode: {ast_mode}")
            if shape not in SHAPES:
                raise RuntimeError(f"unsupported kernel shape: {shape}")
            key = (
                backend,
                opt_level,
                shape,
                ast_mode,
                int(row["rows"]),
            )
            if key in keys:
                raise RuntimeError(f"CSV contains duplicate row: {key}")
            keys.add(key)
            systems.add(row["system"])
            metadata.add((
                int(row["modules"]),
                int(row["workers"]),
                int(row["kernels_per_module"]),
                int(row["asts_per_kernel"]),
                int(row["asts_per_module"]),
                int(row["asts"]),
                int(row["tile_rows"]),
                int(row["threads"]),
                int(row["streams"]),
                int(row["run_iterations"]),
            ))
            rows.append({
                "backend": backend,
                "opt_level": opt_level,
                "shape": shape,
                "ast_mode": ast_mode,
                "rows": int(row["rows"]),
                "runtime_rate":
                    float(row["runtime_row_evals_per_second"]),
                "pipeline_rate":
                    float(row["pipeline_row_evals_per_second"]),
            })
    if not rows:
        raise RuntimeError("CSV contains no pipeline data")
    if len(systems) != 1:
        raise RuntimeError("CSV must describe exactly one system")
    if len(metadata) != 1:
        raise RuntimeError("CSV mixes incompatible benchmark dimensions")
    return next(iter(systems)), next(iter(metadata)), rows


def draw_graph(input_path, output_path, linear_y):
    system, metadata, rows = read_data(input_path)
    (
        modules,
        workers,
        kernels,
        asts_per_kernel,
        asts_per_module,
        asts,
        tile_rows,
        threads,
        streams,
        run_iterations,
    ) = metadata
    modes = [mode for mode in AST_MODES if any(
        row["ast_mode"] == mode for row in rows)]
    shapes = [shape for shape in SHAPES if any(
        row["shape"] == shape for row in rows)]
    configurations = sorted({
        (row["backend"], row["opt_level"])
        for row in rows
    })
    row_counts = sorted({row["rows"] for row in rows})
    if len(row_counts) < 2:
        raise RuntimeError(
            "pipeline graph requires at least two measured row counts")
    expected_points = {
        (backend, opt_level, shape, ast_mode, row_count)
        for backend, opt_level in configurations
        for shape in shapes
        for ast_mode in modes
        for row_count in row_counts
    }
    actual_points = {
        (
            row["backend"],
            row["opt_level"],
            row["shape"],
            row["ast_mode"],
            row["rows"],
        )
        for row in rows
    }
    missing_points = sorted(expected_points - actual_points)
    if missing_points:
        preview = ", ".join(str(point) for point in missing_points[:4])
        if len(missing_points) > 4:
            preview += ", ..."
        raise RuntimeError(
            f"pipeline CSV is missing {len(missing_points)} point(s): "
            f"{preview}")

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#444444",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-overlapped-pipeline",
    })
    figure, axes = pyplot.subplots(
        len(shapes),
        len(modes),
        figsize=(
            14.0 if len(modes) == 2 else 8.0,
            10.2 if len(shapes) == 2 else 7.6),
        sharey=True,
        squeeze=False,
    )

    for shape_idx, shape in enumerate(shapes):
        for mode_idx, ast_mode in enumerate(modes):
            axis = axes[shape_idx][mode_idx]
            for backend, opt_level in configurations:
                points = sorted(
                    (
                        row["rows"],
                        row["runtime_rate"],
                        row["pipeline_rate"],
                    )
                    for row in rows
                    if row["backend"] == backend and
                       row["opt_level"] == opt_level and
                       row["shape"] == shape and
                       row["ast_mode"] == ast_mode
                )
                if not points:
                    continue
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
                    markersize=6.0,
                    linewidth=2.2,
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
                    markeredgewidth=1.3,
                    markersize=6.0,
                    linewidth=1.9,
                    zorder=2,
                )
            axis.set_xscale("log", base=2)
            if not linear_y:
                axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0.0)
            axis.set_xticks(
                row_counts,
                [f"{value:,}" for value in row_counts])
            axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
            axis.grid(which="major", color="#D8D8D8", linewidth=0.75)
            axis.grid(which="minor", color="#EEEEEE", linewidth=0.45)
            axis.set_axisbelow(True)
            axis.set_xlabel("Rows evaluated per AST")
            axis.set_title(
                f"{shape.upper()} · {ast_mode.upper()}",
                fontsize=12,
                fontweight="bold")
    for shape_idx in range(len(shapes)):
        axes[shape_idx][0].set_ylabel(
            "Effective row evaluations per second")

    configuration_handles = [
        Line2D(
            [0],
            [0],
            color=BACKEND_COLORS[backend],
            linestyle="-",
            marker=OPT_MARKERS[opt_level],
            markerfacecolor=BACKEND_COLORS[backend],
            markeredgecolor=BACKEND_COLORS[backend],
            linewidth=2.2,
            markersize=6.0,
            label=f"{backend.upper()} O{opt_level}",
        )
        for backend, opt_level in configurations
    ]
    phase_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="-",
            linewidth=2.2,
            label="Compile/run pipeline wall time",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=":",
            linewidth=1.9,
            label="GPU-event runtime only",
        ),
    ]
    figure.legend(
        handles=configuration_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=len(configuration_handles),
        frameon=True,
        framealpha=0.94,
        title="Backend and native compiler optimization",
    )
    figure.legend(
        handles=phase_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.805),
        ncol=2,
        frameon=True,
        framealpha=0.94,
        title="Timing boundary",
    )
    figure.suptitle(
        f"SECANT Concurrent Compile and GPU Execution on {system}",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"{workers} OpenMP workers · {modules} modules · "
            f"{asts_per_module:,} ASTs/module "
            f"({kernels} kernels × {asts_per_kernel} ASTs) · "
            f"{asts:,} ASTs total"
        ),
        ha="center",
        color="#444444",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.045,
        (
            "Completed modules are consumed in ready order. Each transition "
            "waits on the all-stream stop event, unloads the previous module, "
            "loads the next module, and launches across pre-created streams."
        ),
        ha="center",
        color="#444444",
        fontsize=8.6,
    )
    figure.text(
        0.5,
        0.019,
        (
            f"Inputs remain device-resident; transfers and cold template work "
            f"are excluded. Tile={tile_rows}; threads={threads}; "
            f"streams={streams}; run iterations={run_iterations}. "
            f"{'Linear' if linear_y else 'Logarithmic'} Y axis."
        ),
        ha="center",
        color="#555555",
        fontsize=8.2,
    )
    figure.tight_layout(rect=(0.035, 0.075, 0.995, 0.735))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "svg",
        metadata={"Date": None, "Creator": "SECANT pipeline graph"},
    )
    pyplot.close(figure)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Graph runtime-only and concurrent compile/run pipeline "
            "throughput for ALU and MUFU ASTs."))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-y", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    draw_graph(args.input, args.output, args.linear_y)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
