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


OUTPUT_FIELDS = (
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
    "execution_mode",
    "runtime_row_evals_per_second",
    "compile_inclusive_row_evals_per_second",
    "compile_assumption",
    "corpus",
    "corpus_hash",
    "source",
)

LABELS = {
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

COLORS = {
    "CUDA O0": "#E69F00",
    "CUDA O1": "#A65600",
    "PTX O0": "#56B4E9",
    "PTX O1": "#0072B2",
    "CUBIN": "#009E73",
    "HIP O0": "#D55E00",
    "HIP O1": "#9C3B00",
    "HSACO": "#CC79A7",
    "Static Native AVX2": "#4D4D4D",
    "PySR backend": "#7B61A8",
    "EvoGP": "#F0E442",
    "Kozax": "#8C8C00",
    "Operon": "#6B8E23",
}

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


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


def read_pipeline(path, platform, ast_mode):
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for source in csv.DictReader(file):
            if source["platform"] != platform:
                raise RuntimeError(
                    f"{path} contains platform={source['platform']}, "
                    f"expected {platform}")
            if source["ast_mode"] != ast_mode:
                continue
            backend = source["backend"]
            opt_level = int(source["opt_level"])
            label = LABELS.get((backend, opt_level))
            if label is None:
                raise RuntimeError(
                    f"unsupported pipeline configuration: "
                    f"{backend} O{opt_level}")
            rows.append({
                "platform": platform,
                "label": label,
                "family": "secant",
                "backend": backend,
                "opt_level": opt_level,
                "shape": source["shape"],
                "ast_mode": source["ast_mode"],
                "rows": int(source["rows"]),
                "asts": int(source["asts"]),
                "workers": int(source["workers"]),
                "execution_mode": (
                    f"measured_pipeline_{source['modules']}modules_"
                    f"{source['kernels_per_module']}kernels_"
                    f"{source['asts_per_kernel']}asts"),
                "runtime_row_evals_per_second":
                    float(source["runtime_row_evals_per_second"]),
                "compile_inclusive_row_evals_per_second":
                    float(source["pipeline_row_evals_per_second"]),
                "compile_assumption": "measured_compile_load_run_pipeline",
                "corpus": source["corpus"],
                "corpus_hash": source["corpus_hash"],
                "source": str(path),
            })
    if not rows:
        raise RuntimeError(f"{path} contains no {ast_mode} pipeline rows")
    return rows


def read_baselines(paths, platform, ast_mode):
    candidates = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for source in csv.DictReader(file):
                if source["ast_mode"] != ast_mode:
                    continue
                label = BASELINE_LABELS.get(source["system"])
                if label is None:
                    raise RuntimeError(
                        f"unsupported baseline system in {path}: "
                        f"{source['system']}")
                rate = float(source["row_evals_per_second"])
                key = (
                    label,
                    source["shape"],
                    source["ast_mode"],
                    int(source["rows"]),
                )
                current = candidates.get(key)
                if current is not None and (
                    float(current["runtime_row_evals_per_second"]) >= rate
                ):
                    continue
                is_static_native = source["system"] == "native_avx2"
                candidates[key] = {
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
                    "execution_mode": source["execution_mode"],
                    "runtime_row_evals_per_second": rate,
                    "compile_inclusive_row_evals_per_second":
                        "" if is_static_native else rate,
                    "compile_assumption":
                        "static_aot_build_excluded"
                        if is_static_native
                        else "zero_compile_cost_assumed",
                    "corpus": source["corpus"],
                    "corpus_hash": source["corpus_hash"],
                    "source": str(path),
                }
    return list(candidates.values())


def validate_corpus(rows):
    identities = {
        (row["corpus"], row["corpus_hash"])
        for row in rows
    }
    if len(identities) != 1:
        formatted = ", ".join(
            f"{name}:{digest}" for name, digest in sorted(identities))
        raise RuntimeError(
            f"comparison mixes incompatible corpora: {formatted}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: (
                    f"{row[name]:.9g}"
                    if name.endswith("_per_second") and row[name] != ""
                    else row[name]
                )
                for name in OUTPUT_FIELDS
            })
    temporary.replace(path)


def draw(path, rows, title):
    shapes = [
        shape for shape in ("materialize", "sse")
        if any(row["shape"] == shape for row in rows)
    ]
    labels = list(dict.fromkeys(row["label"] for row in rows))
    markers = {
        label: MARKERS[idx % len(MARKERS)]
        for idx, label in enumerate(labels)
    }
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": "#444444",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "svg.hashsalt": "secant-comprehensive-campaign",
    })
    figure, axes = pyplot.subplots(
        2,
        len(shapes),
        figsize=(14.0 if len(shapes) == 2 else 8.0, 9.2),
        squeeze=False,
        sharex="col",
    )
    metrics = (
        (
            "runtime_row_evals_per_second",
            "Runtime only",
        ),
        (
            "compile_inclusive_row_evals_per_second",
            "Compilation included",
        ),
    )
    for metric_idx, (metric, metric_title) in enumerate(metrics):
        for shape_idx, shape in enumerate(shapes):
            axis = axes[metric_idx][shape_idx]
            for label in labels:
                points = sorted(
                    (
                        row["rows"],
                        row[metric],
                    )
                    for row in rows
                    if (
                        row["shape"] == shape and
                        row["label"] == label and
                        row[metric] != ""
                    )
                )
                if not points:
                    continue
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=COLORS[label],
                    marker=markers[label],
                    markerfacecolor=COLORS[label],
                    markeredgecolor="#FFFFFF",
                    markeredgewidth=0.7,
                    markersize=5.8,
                    linewidth=2.0,
                    label=label,
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            row_ticks = sorted({
                row["rows"] for row in rows if row["shape"] == shape
            })
            axis.set_xticks(row_ticks)
            axis.set_xticklabels([str(value) for value in row_ticks])
            axis.yaxis.set_major_formatter(FuncFormatter(rate_label))
            axis.grid(which="major", color="#D8D8D8", linewidth=0.75)
            axis.grid(which="minor", color="#EEEEEE", linewidth=0.45)
            axis.set_axisbelow(True)
            axis.set_title(
                f"{shape.upper()} · {metric_title}",
                fontsize=11.5,
                fontweight="bold")
            axis.set_xlabel("Rows evaluated per AST")
        axes[metric_idx][0].set_ylabel("Row evaluations per second")

    legend = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            marker=markers[label],
            markerfacecolor=COLORS[label],
            markeredgecolor="#FFFFFF",
            linewidth=2.0,
            markersize=5.8,
            label=label,
        )
        for label in labels
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=min(7, len(legend)),
        frameon=True,
        framealpha=0.95,
    )
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.025,
        (
            "Every marker is measured. Static Native AVX2 appears only in the "
            "runtime panels because its AOT build is outside the benchmark. "
            "Other baselines are assigned zero dynamic compile cost; SECANT "
            "uses measured concurrent compile, module transition, and "
            "execution wall time."
        ),
        ha="center",
        color="#444444",
        fontsize=8.8,
    )
    figure.tight_layout(rect=(0.035, 0.055, 0.995, 0.84))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format=path.suffix.lstrip(".") or "svg",
        metadata={"Date": None, "Creator": "SECANT comprehensive campaign"},
    )
    pyplot.close(figure)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Combine a measured SECANT compile/run pipeline with runtime-only "
            "backend baselines."))
    parser.add_argument("--platform", choices=("nvidia", "amd"), required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, nargs="*", default=[])
    parser.add_argument("--ast-mode", choices=("alu", "mufu"), default="alu")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--title")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = read_pipeline(args.pipeline, args.platform, args.ast_mode)
    rows.extend(read_baselines(args.baselines, args.platform, args.ast_mode))
    validate_corpus(rows)
    rows.sort(key=lambda row: (
        row["shape"],
        row["label"],
        row["rows"],
    ))
    write_csv(args.output_csv, rows)
    title = args.title or (
        f"SECANT Runtime and Compile-Inclusive Throughput ({args.platform})")
    draw(args.output_svg, rows, title)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_svg}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
