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
import statistics
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROWS = (1024, 4096, 16384, 65536, 131072, 262144)
SHAPES = ("materialize", "sse")

PIPELINE_LABELS = {
    ("cuda", 0): "CUDA O0",
    ("cuda", 1): "CUDA O1",
    ("ptx", 0): "PTX O0",
    ("ptx", 1): "PTX O1",
    ("cubin", 1): "CUBIN",
    ("hip", 0): "HIP O0",
    ("hip", 1): "HIP O1",
    ("hsaco", 1): "HSACO",
}

BASELINE_LABELS = {
    "native_avx2": "Static Native AVX2",
    "pysr_symbolicregression": "PySR backend",
    "evogp": "EvoGP",
    "kozax": "Kozax",
    "operon": "Operon",
}

DIRECT_BACKENDS = {
    "nvidia": "CUBIN",
    "amd": "HSACO",
}

PLATFORM_TITLES = {
    "nvidia": "RTX 5090 + Ryzen 9 9900X",
    "amd": "RX 9070 XT + Ryzen 9 7900X",
}

COLORS = {
    "CUDA O0": "#E69F00",
    "CUDA O1": "#A65600",
    "PTX O0": "#56B4E9",
    "PTX O1": "#0072B2",
    "CUBIN": "#009E73",
    "HIP O0": "#E69F00",
    "HIP O1": "#A65600",
    "HSACO": "#CC79A7",
    "Static Native AVX2": "#4D4D4D",
    "PySR backend": "#7B61A8",
    "EvoGP": "#D6A900",
    "Kozax": "#8C8C00",
    "Operon": "#6B8E23",
}

MARKERS = {
    "CUDA O0": "o",
    "CUDA O1": "s",
    "PTX O0": "^",
    "PTX O1": "D",
    "CUBIN": "P",
    "HIP O0": "o",
    "HIP O1": "s",
    "HSACO": "P",
    "Static Native AVX2": "X",
    "PySR backend": "v",
    "EvoGP": "<",
    "Kozax": ">",
    "Operon": "h",
}

COMBINED_LABELS = {
    "CUBIN": "SECANT",
    "HSACO": "SECANT",
    "CUDA O0": "CUDA",
    "CUDA O1": "CUDA",
    "PTX O0": "PTX",
    "PTX O1": "PTX",
    "HIP O0": "HIP",
    "HIP O1": "HIP",
}

COMBINED_COLORS = {
    "SECANT": "#009E73",
    "CUDA": "#A65600",
    "PTX": "#0072B2",
    "HIP": "#A65600",
}

COMBINED_MARKERS = {
    "SECANT": "P",
    "CUDA": "s",
    "PTX": "D",
    "HIP": "s",
}

COMBINED_LABEL_ORDER = {
    label: index for index, label in enumerate((
        "SECANT",
        "CUDA",
        "PTX",
        "HIP",
        "Static Native AVX2",
        "EvoGP",
        "Kozax",
        "Operon",
        "PySR backend",
    ))
}

POINT_FIELDS = (
    "platform",
    "label",
    "family",
    "backend",
    "opt_level",
    "shape",
    "ast_mode",
    "rows",
    "asts",
    "workers",
    "runtime_row_evals_per_second",
    "pipeline_row_evals_per_second",
    "compile_asts_per_second",
    "compile_window_seconds",
    "module_load_seconds",
    "module_unload_seconds",
    "runtime_seconds",
    "pipeline_seconds",
    "execution_mode",
    "corpus",
    "corpus_hash",
)


def rate_label(value, _position=None):
    if value >= 1.0e12:
        return f"{value / 1.0e12:g}T"
    if value >= 1.0e9:
        return f"{value / 1.0e9:g}G"
    if value >= 1.0e6:
        return f"{value / 1.0e6:g}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:g}K"
    return f"{value:g}"


def seconds_label(value, _position=None):
    if value >= 1.0:
        return f"{value:g}s"
    if value >= 1.0e-3:
        return f"{value * 1.0e3:g}ms"
    return f"{value * 1.0e6:g}us"


def row_label(value):
    return str(value)


def configure_matplotlib():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": "#444444",
        "axes.titleweight": "bold",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-overnight-runtime-report-2026-07-27",
    })


def read_pipeline(path, platform):
    points = []
    with path.open(newline="", encoding="utf-8") as file:
        for source in csv.DictReader(file):
            if source["platform"] != platform:
                raise RuntimeError(
                    f"{path} contains platform={source['platform']}, "
                    f"expected {platform}")
            if source["ast_mode"] != "alu":
                continue
            key = (source["backend"], int(source["opt_level"]))
            label = PIPELINE_LABELS.get(key)
            if label is None:
                raise RuntimeError(f"unsupported pipeline backend: {key}")
            points.append({
                "platform": platform,
                "label": label,
                "family": "secant",
                "backend": source["backend"],
                "opt_level": int(source["opt_level"]),
                "shape": source["shape"],
                "ast_mode": source["ast_mode"],
                "rows": int(source["rows"]),
                "asts": int(source["asts"]),
                "workers": int(source["workers"]),
                "runtime_row_evals_per_second":
                    float(source["runtime_row_evals_per_second"]),
                "pipeline_row_evals_per_second":
                    float(source["pipeline_row_evals_per_second"]),
                "compile_asts_per_second":
                    float(source["compile_asts_per_second"]),
                "compile_window_seconds":
                    float(source["compile_window_seconds"]),
                "module_load_seconds":
                    float(source["module_load_seconds"]),
                "module_unload_seconds":
                    float(source["module_unload_seconds"]),
                "runtime_seconds": float(source["runtime_seconds"]),
                "pipeline_seconds": float(source["pipeline_seconds"]),
                "execution_mode": (
                    f"{source['modules']} modules, "
                    f"{source['kernels_per_module']} kernels/module, "
                    f"{source['asts_per_kernel']} ASTs/kernel"),
                "corpus": source["corpus"],
                "corpus_hash": source["corpus_hash"],
            })
    return points


def read_baselines(directory, platform):
    best = {}
    for path in sorted(directory.glob("baseline_*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            for source in csv.DictReader(file):
                if source["ast_mode"] != "alu":
                    continue
                label = BASELINE_LABELS.get(source["system"])
                if label is None:
                    raise RuntimeError(
                        f"unsupported baseline system in {path}: "
                        f"{source['system']}")
                key = (label, source["shape"], int(source["rows"]))
                rate = float(source["row_evals_per_second"])
                if key in best and (
                    best[key]["runtime_row_evals_per_second"] >= rate
                ):
                    continue
                best[key] = {
                    "platform": platform,
                    "label": label,
                    "family": "baseline",
                    "backend": source["backend"],
                    "opt_level": "",
                    "shape": source["shape"],
                    "ast_mode": source["ast_mode"],
                    "rows": int(source["rows"]),
                    "asts": int(source["asts"]),
                    "workers": int(source["workers"]),
                    "runtime_row_evals_per_second": rate,
                    "pipeline_row_evals_per_second": "",
                    "compile_asts_per_second": "",
                    "compile_window_seconds": "",
                    "module_load_seconds": "",
                    "module_unload_seconds": "",
                    "runtime_seconds": float(source["median_seconds"]),
                    "pipeline_seconds": "",
                    "execution_mode": source["execution_mode"],
                    "corpus": source["corpus"],
                    "corpus_hash": source["corpus_hash"],
                }
    return list(best.values())


def validate_points(points):
    identities = {
        (point["corpus"], point["corpus_hash"])
        for point in points
    }
    if len(identities) != 1:
        text = ", ".join(
            f"{name}:{digest}" for name, digest in sorted(identities))
        raise RuntimeError(f"mixed benchmark corpora: {text}")
    for platform in PLATFORM_TITLES:
        platform_points = [
            point for point in points
            if point["platform"] == platform and point["family"] == "secant"
        ]
        if not platform_points:
            raise RuntimeError(f"no SECANT points for {platform}")
        for label in {
            point["label"] for point in platform_points
        }:
            for shape in SHAPES:
                found = {
                    point["rows"] for point in platform_points
                    if point["label"] == label and point["shape"] == shape
                }
                if found != set(ROWS):
                    raise RuntimeError(
                        f"incomplete {platform} {label} {shape}: "
                        f"{sorted(found)}")


def write_points(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=POINT_FIELDS)
        writer.writeheader()
        for point in sorted(points, key=lambda item: (
            item["platform"],
            item["shape"],
            item["family"],
            item["label"],
            item["rows"],
        )):
            writer.writerow(point)


def save_figure(figure, path):
    figure.savefig(
        path,
        format="svg",
        metadata={
            "Date": None,
            "Creator": "SECANT overnight runtime report",
        },
    )
    figure.savefig(
        path.with_suffix(".png"),
        format="png",
        dpi=160,
        metadata={"Date": None},
    )
    pyplot.close(figure)


def style_rate_axis(axis, log_scale):
    axis.set_xscale("log", base=2)
    if log_scale:
        axis.set_yscale("log")
    axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
    axis.set_xticks(ROWS)
    axis.set_xticklabels([row_label(value) for value in ROWS])
    axis.grid(which="major", color="#D8D8D8", linewidth=0.75)
    if log_scale:
        axis.grid(which="minor", color="#EEEEEE", linewidth=0.4)
    axis.set_axisbelow(True)
    axis.set_xlabel("Rows evaluated per AST")


def plot_series(axis, points, label, metric, linestyle="-"):
    selected = sorted(
        (
            point["rows"],
            float(point[metric]),
        )
        for point in points
        if point["label"] == label and point[metric] != ""
    )
    if not selected:
        return
    axis.plot(
        [item[0] for item in selected],
        [item[1] for item in selected],
        color=COLORS[label],
        marker=MARKERS[label],
        markerfacecolor=COLORS[label],
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.7,
        markersize=5.5,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )


def combined_label(label):
    return COMBINED_LABELS.get(label, label)


def combined_color(label):
    if label in COMBINED_COLORS:
        return COMBINED_COLORS[label]
    return COLORS[label]


def combined_marker(label):
    if label in COMBINED_MARKERS:
        return COMBINED_MARKERS[label]
    return MARKERS[label]


def plot_combined_series(axis, points, label, metric, linestyle="-"):
    best = {}
    for point in points:
        if combined_label(point["label"]) != label or point[metric] == "":
            continue
        rows = point["rows"]
        rate = float(point[metric])
        if rows not in best or rate > best[rows]:
            best[rows] = rate
    if not best:
        return
    selected = sorted(best.items())
    axis.plot(
        [item[0] for item in selected],
        [item[1] for item in selected],
        color=combined_color(label),
        marker=combined_marker(label),
        markerfacecolor=combined_color(label),
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.7,
        markersize=5.5,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )


def runtime_overview(points, platform, path):
    platform_points = [
        point for point in points if point["platform"] == platform
    ]
    labels = list(dict.fromkeys(
        point["label"] for point in platform_points
    ))
    figure, axes = pyplot.subplots(
        1,
        2,
        figsize=(15.0, 7.3),
        sharex=True,
    )
    for axis, shape in zip(axes, SHAPES):
        shape_points = [
            point for point in platform_points
            if point["shape"] == shape
        ]
        for label in labels:
            label_points = [
                point for point in shape_points
                if point["label"] == label
            ]
            if any(
                point["family"] == "secant"
                for point in label_points
            ):
                plot_series(
                    axis,
                    label_points,
                    label,
                    "pipeline_row_evals_per_second",
                    "-")
                plot_series(
                    axis,
                    label_points,
                    label,
                    "runtime_row_evals_per_second",
                    "--")
            else:
                plot_series(
                    axis,
                    label_points,
                    label,
                    "runtime_row_evals_per_second",
                    ":")
        style_rate_axis(axis, True)
        axis.set_title(
            "Materialize every row" if shape == "materialize"
            else "Reduce rows to SSE")
    axes[0].set_ylabel("Row evaluations per second")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=2.0,
            label=label,
        )
        for label in labels
    ]
    handles.extend((
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.0,
            linestyle="-",
            label="24-worker compile + load + run + unload",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.0,
            linestyle="--",
            label="Compiled backend runtime only",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.0,
            linestyle=":",
            label="External warm runtime",
        ),
    ))
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=min(5, len(handles)),
        frameon=True,
    )
    figure.suptitle(
        (
            "Runtime and End-to-End Pipeline Throughput - "
            f"{PLATFORM_TITLES[platform]}"
        ),
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Logarithmic throughput axis. Solid SECANT lines use the measured "
            "24-worker concurrent pipeline wall time; dashed lines use GPU "
            "event runtime.\n"
            "Static Native AVX2 is runtime-only because its AOT build is "
            "outside the benchmark. Other external systems show their fastest "
            "measured warm runtime with zero dynamic compilation cost."
            + (
                "\nEvoGP is absent because the AMD overnight adapter did not "
                "run it; this is not an AMD hardware limitation."
                if platform == "amd"
                else ""
            )
        ),
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#444444",
    )
    figure.tight_layout(rect=(0.095, 0.09, 0.995, 0.77))
    save_figure(figure, path)


def combined_runtime_overview(points, path, include_compile):
    platform_labels = {}
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=(17.0, 12.2),
        sharex=True,
    )
    for row, platform in enumerate(("nvidia", "amd")):
        platform_points = [
            point for point in points
            if (
                point["platform"] == platform and
                (
                    not include_compile or
                    point["label"] != "Static Native AVX2"
                )
            )
        ]
        labels = dict.fromkeys(
            combined_label(point["label"]) for point in platform_points
        )
        platform_labels[platform] = sorted(
            labels,
            key=lambda label: COMBINED_LABEL_ORDER.get(
                label,
                len(COMBINED_LABEL_ORDER)),
        )
        for column, shape in enumerate(SHAPES):
            axis = axes[row][column]
            shape_points = [
                point for point in platform_points
                if point["shape"] == shape
            ]
            for label in platform_labels[platform]:
                label_points = [
                    point for point in shape_points
                    if combined_label(point["label"]) == label
                ]
                if any(
                    point["family"] == "secant"
                    for point in label_points
                ):
                    plot_combined_series(
                        axis,
                        label_points,
                        label,
                        (
                            "pipeline_row_evals_per_second"
                            if include_compile
                            else "runtime_row_evals_per_second"
                        ))
                else:
                    plot_combined_series(
                        axis,
                        label_points,
                        label,
                        "runtime_row_evals_per_second")
            style_rate_axis(axis, True)
            axis.tick_params(axis="x", labelbottom=True)
            shape_title = (
                "Materialize every row"
                if shape == "materialize"
                else "Reduce rows to SSE"
            )
            axis.set_title(shape_title)
        axes[row][0].set_ylabel("Row evaluations per second")
    platform_legend_positions = {
        "nvidia": (0.5, 0.94),
        "amd": (0.5, 0.475),
    }
    for platform in ("nvidia", "amd"):
        backend_handles = [
            Line2D(
                [0],
                [0],
                color=combined_color(label),
                marker=combined_marker(label),
                linewidth=2.0,
                label=label,
            )
            for label in platform_labels[platform]
        ]
        figure.legend(
            handles=backend_handles,
            title=PLATFORM_TITLES[platform],
            title_fontproperties={"weight": "bold"},
            loc="upper center",
            bbox_to_anchor=platform_legend_positions[platform],
            ncol=min(6, len(backend_handles)),
            frameon=True,
        )
    figure.suptitle(
        (
            "End-to-End Compile + Load + Run Throughput"
            if include_compile
            else "Runtime Throughput"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.015,
        (
            (
                "CUDA, PTX, and HIP show the faster O0/O1 end-to-end result "
                "independently at each measured row count. SECANT, CUDA, PTX, "
                "and HIP use measured 24-worker compile, module load, run, and "
                "unload wall time.\nEvoGP, Kozax, Operon, and PySR use measured "
                "backend runtime with zero dynamic compilation cost. Static "
                "Native AVX2 is omitted because its AOT build is unmeasured."
                if include_compile
                else
                "Every line is runtime only. CUDA, PTX, and HIP show the faster "
                "O0/O1 runtime independently at each measured row count.\n"
                "Static Native AVX2 excludes its AOT build."
            ) +
            "\nEvoGP is absent on AMD because that overnight adapter did not "
            "run it, not because of an AMD hardware limitation."
        ),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.085,
        top=0.79,
        hspace=0.55,
        wspace=0.11,
    )
    save_figure(figure, path)


def gpu_runtime_detail(points, path):
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        sharex=True,
    )
    for platform_idx, platform in enumerate(("nvidia", "amd")):
        allowed_families = {"secant"}
        allowed_labels = {"EvoGP", "Kozax"}
        for shape_idx, shape in enumerate(SHAPES):
            axis = axes[platform_idx][shape_idx]
            selected = [
                point for point in points
                if point["platform"] == platform and
                point["shape"] == shape and
                (
                    point["family"] in allowed_families or
                    point["label"] in allowed_labels
                )
            ]
            labels = list(dict.fromkeys(
                point["label"] for point in selected
            ))
            for label in labels:
                plot_series(
                    axis,
                    selected,
                    label,
                    "runtime_row_evals_per_second")
            style_rate_axis(axis, False)
            axis.set_ylim(bottom=0.0)
            axis.set_title(
                f"{PLATFORM_TITLES[platform]} - "
                f"{'Materialize' if shape == 'materialize' else 'SSE'}")
            if shape_idx == 0:
                axis.set_ylabel("Runtime row evaluations per second")
    labels = list(dict.fromkeys(
        point["label"] for point in points
        if point["family"] == "secant" or
        point["label"] in {"EvoGP", "Kozax"}
    ))
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=2.0,
            markersize=5.5,
            label=label,
        )
        for label in labels
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=min(6, len(handles)),
        frameon=True,
    )
    figure.suptitle(
        "GPU Runtime Detail",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Linear throughput axes make absolute GPU differences visible. "
            "Axis ranges are independent because materialization and SSE have "
            "different output and reduction costs."
        ),
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    figure.tight_layout(rect=(0.025, 0.055, 0.995, 0.86))
    save_figure(figure, path)


def pipeline_frontier(points, platform, path):
    selected = [
        point for point in points
        if point["platform"] == platform and point["family"] == "secant"
    ]
    labels = list(dict.fromkeys(point["label"] for point in selected))
    figure, axes = pyplot.subplots(
        1,
        2,
        figsize=(15.0, 6.8),
        sharex=True,
    )
    for axis, shape in zip(axes, SHAPES):
        shape_points = [
            point for point in selected if point["shape"] == shape
        ]
        for label in labels:
            plot_series(
                axis,
                shape_points,
                label,
                "pipeline_row_evals_per_second",
                "-")
            plot_series(
                axis,
                shape_points,
                label,
                "runtime_row_evals_per_second",
                "--")
        style_rate_axis(axis, True)
        axis.set_title(
            "Materialize every row" if shape == "materialize"
            else "Reduce rows to SSE")
    axes[0].set_ylabel("Row evaluations per second")
    backend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=2.0,
            label=label,
        )
        for label in labels
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.0,
            linestyle="-",
            label="24-worker compile + load + run + unload",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.0,
            linestyle="--",
            label="Runtime only",
        ),
    ]
    figure.legend(
        handles=backend_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=min(5, len(backend_handles) + len(style_handles)),
        frameon=True,
    )
    figure.suptitle(
        f"Runtime and Full Pipeline - {PLATFORM_TITLES[platform]}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        (
            "The solid rate uses measured wall time from 24-worker concurrent "
            "compilation start through module execution and unload. "
            "The dashed rate uses GPU event time only."
        ),
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    figure.tight_layout(rect=(0.05, 0.055, 0.995, 0.82))
    save_figure(figure, path)


def direct_lifecycle(points, path):
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        sharex=True,
    )
    component_colors = {
        "Runtime": "#009E73",
        "Non-GPU pipeline wall": "#A0A0A0",
    }
    for platform_idx, platform in enumerate(("nvidia", "amd")):
        direct_label = DIRECT_BACKENDS[platform]
        for shape_idx, shape in enumerate(SHAPES):
            axis = axes[platform_idx][shape_idx]
            selected = sorted(
                (
                    point for point in points
                    if point["platform"] == platform and
                    point["label"] == direct_label and
                    point["shape"] == shape
                ),
                key=lambda point: point["rows"],
            )
            positions = list(range(len(selected)))
            runtime = [point["runtime_seconds"] for point in selected]
            overhead = [
                max(
                    0.0,
                    point["pipeline_seconds"] -
                    point["runtime_seconds"])
                for point in selected
            ]
            axis.bar(
                positions,
                runtime,
                color=component_colors["Runtime"],
                label="Runtime")
            axis.bar(
                positions,
                overhead,
                bottom=runtime,
                color=component_colors["Non-GPU pipeline wall"],
                label="Non-GPU pipeline wall")
            axis.set_xticks(positions)
            axis.set_xticklabels([
                row_label(point["rows"]) for point in selected
            ])
            axis.yaxis.set_major_formatter(FuncFormatter(seconds_label))
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.75)
            axis.set_axisbelow(True)
            axis.set_title(
                f"{PLATFORM_TITLES[platform]} - {direct_label} "
                f"{'Materialize' if shape == 'materialize' else 'SSE'}")
            axis.set_xlabel("Rows evaluated per AST")
            if shape_idx == 0:
                axis.set_ylabel("Full campaign wall time")
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=8.0,
            label=label,
        )
        for label, color in component_colors.items()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
    )
    figure.suptitle(
        "Direct Binary Pipeline Wall-Time Decomposition",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.02,
        (
            "GPU-event runtime is compared with total pipeline wall time. "
            "The residual includes startup, scheduling, compilation not "
            "hidden by execution, unloading, and loader tails. Module-load "
            "call time is not stacked because eager CUDA loading overlaps "
            "GPU execution."
        ),
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    figure.tight_layout(rect=(0.025, 0.055, 0.995, 0.88))
    save_figure(figure, path)


def compile_throughput(points, path):
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
    )
    for platform_idx, platform in enumerate(("nvidia", "amd")):
        platform_points = [
            point for point in points
            if point["platform"] == platform and point["family"] == "secant"
        ]
        labels = list(dict.fromkeys(
            point["label"] for point in platform_points
        ))
        for shape_idx, shape in enumerate(SHAPES):
            axis = axes[platform_idx][shape_idx]
            rates = [
                statistics.median([
                    point["compile_asts_per_second"]
                    for point in platform_points
                    if point["label"] == label and
                    point["shape"] == shape
                ])
                for label in labels
            ]
            positions = list(range(len(labels)))
            axis.bar(
                positions,
                rates,
                color=[COLORS[label] for label in labels],
            )
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20, ha="right")
            axis.grid(axis="y", which="major", color="#D8D8D8")
            axis.grid(axis="y", which="minor", color="#EEEEEE")
            axis.set_axisbelow(True)
            axis.set_title(
                f"{PLATFORM_TITLES[platform]} - "
                f"{'Materialize' if shape == 'materialize' else 'SSE'}")
            if shape_idx == 0:
                axis.set_ylabel("ASTs compiled per second")
    figure.suptitle(
        "Measured 24-Worker Compiler-Capacity Throughput",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Rates use the maximum cumulative compile-callback time of any "
            "worker, excluding bounded-slot backpressure. Bars are medians "
            "across six row-count runs; row count does not change the AST "
            "corpus."
        ),
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    figure.tight_layout(rect=(0.025, 0.055, 0.995, 0.94))
    save_figure(figure, path)


def speedup_over_external(points, path):
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        sharex=True,
    )
    for platform_idx, platform in enumerate(("nvidia", "amd")):
        direct_label = DIRECT_BACKENDS[platform]
        for shape_idx, shape in enumerate(SHAPES):
            axis = axes[platform_idx][shape_idx]
            direct = {
                point["rows"]: point
                for point in points
                if point["platform"] == platform and
                point["label"] == direct_label and
                point["shape"] == shape
            }
            external = {}
            for point in points:
                if point["platform"] != platform or \
                   point["family"] != "baseline" or \
                   point["shape"] != shape:
                    continue
                external[point["rows"]] = max(
                    external.get(point["rows"], 0.0),
                    point["runtime_row_evals_per_second"],
                )
            common_rows = sorted(set(direct) & set(external))
            runtime_speedup = [
                direct[row]["runtime_row_evals_per_second"] / external[row]
                for row in common_rows
            ]
            pipeline_speedup = [
                direct[row]["pipeline_row_evals_per_second"] / external[row]
                for row in common_rows
            ]
            axis.plot(
                common_rows,
                pipeline_speedup,
                color=COLORS[direct_label],
                marker="s",
                linewidth=2.2,
                label="Full pipeline")
            axis.plot(
                common_rows,
                runtime_speedup,
                color=COLORS[direct_label],
                marker="o",
                linewidth=2.2,
                linestyle="--",
                label="Runtime only")
            axis.axhline(1.0, color="#555555", linewidth=1.0)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(ROWS)
            axis.set_xticklabels([row_label(value) for value in ROWS])
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _position: f"{value:g}x"))
            axis.grid(which="major", color="#D8D8D8", linewidth=0.75)
            axis.grid(which="minor", color="#EEEEEE", linewidth=0.4)
            axis.set_axisbelow(True)
            axis.set_title(
                f"{PLATFORM_TITLES[platform]} - {direct_label} "
                f"{'Materialize' if shape == 'materialize' else 'SSE'}")
            axis.set_xlabel("Rows evaluated per AST")
            if shape_idx == 0:
                axis.set_ylabel("Speedup over fastest external baseline")
    handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="s",
            linewidth=2.2,
            label="Compile + load + run + unload"),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            linewidth=2.2,
            linestyle="--",
            label="Runtime only"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
    )
    figure.suptitle(
        "Direct Binary Speedup Over the Fastest External Runtime",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.02,
        (
            "The reference is selected independently at each row count from "
            "Static Native AVX2, PySR, Operon, Kozax, and EvoGP where "
            "available. Static Native AVX2 excludes its AOT build; other "
            "external systems assume zero dynamic compilation cost."
        ),
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    figure.tight_layout(rect=(0.025, 0.055, 0.995, 0.88))
    save_figure(figure, path)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate the SECANT overnight runtime report graphs.")
    parser.add_argument("--nvidia-dir", type=Path, required=True)
    parser.add_argument("--amd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    points = []
    for platform, directory in (
        ("nvidia", args.nvidia_dir),
        ("amd", args.amd_dir),
    ):
        points.extend(read_pipeline(directory / "pipeline.csv", platform))
        points.extend(read_baselines(directory, platform))
    validate_points(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_points(args.output_dir / "overnight_points.csv", points)
    configure_matplotlib()
    runtime_overview(
        points,
        "nvidia",
        args.output_dir / "runtime_overview_rtx5090.svg")
    runtime_overview(
        points,
        "amd",
        args.output_dir / "runtime_overview_rx9070xt.svg")
    combined_runtime_overview(
        points,
        args.output_dir / "runtime_overview.svg",
        False)
    combined_runtime_overview(
        points,
        args.output_dir / "end_to_end_overview.svg",
        True)
    gpu_runtime_detail(
        points,
        args.output_dir / "runtime_gpu_detail.svg")
    pipeline_frontier(
        points,
        "nvidia",
        args.output_dir / "pipeline_frontier_rtx5090.svg")
    pipeline_frontier(
        points,
        "amd",
        args.output_dir / "pipeline_frontier_rx9070xt.svg")
    direct_lifecycle(
        points,
        args.output_dir / "direct_binary_lifecycle.svg")
    compile_throughput(
        points,
        args.output_dir / "compile_throughput.svg")
    speedup_over_external(
        points,
        args.output_dir / "direct_binary_speedup.svg")
    for path in sorted(args.output_dir.glob("*.svg")):
        print(f"wrote {path}")
        print(f"wrote {path.with_suffix('.png')}")
    print(f"wrote {args.output_dir / 'overnight_points.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
