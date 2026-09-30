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

from common import read_csv


SHAPES = ("materialize", "sse")
SHAPE_LABELS = {
    "materialize": "Materialize",
    "sse": "SSE",
}
SYSTEM_LABELS = {
    "secant:gpu_direct_cubin": "SECANT CUBIN",
    "secant:gpu_nvrtc_cuda": "SECANT CUDA",
    "secant:gpu_ptx_inject_nvptxcompiler": "SECANT PTX",
    "native_avx2": "Static Native AVX2",
    "pysr_symbolicregression": "PySR",
    "evogp": "EvoGP",
    "kozax": "Kozax",
    "operon": "Operon",
}


def series_id(record):
    if record["system"] == "secant":
        return f"secant:{record['backend']}"
    return record["system"]


def select_best(records, shape):
    selected = {}
    for record in records:
        if record["shape"] != shape or record["ast_mode"] != "alu":
            continue
        key = (series_id(record), int(record["rows"]))
        if key not in selected:
            selected[key] = record
        elif float(record["row_evals_per_second"]) > float(
            selected[key]["row_evals_per_second"]
        ):
            selected[key] = record
    return selected


def format_rate(value):
    return f"{value:.3e}"


def table_for_shape(records, shape, systems):
    selected = select_best(records, shape)
    rows = sorted({row_count for _, row_count in selected})
    lines = [
        f"## {SHAPE_LABELS[shape]}",
        "",
        "| Rows | " + " | ".join(SYSTEM_LABELS.get(system, system) for system in systems) + " |",
        "|---:|" + "|".join("---:" for _ in systems) + "|",
    ]
    for row_count in rows:
        values = []
        for system in systems:
            record = selected.get((system, row_count))
            values.append(
                format_rate(float(record["row_evals_per_second"]))
                if record is not None else "-")
        lines.append(f"| {row_count:,} | " + " | ".join(values) + " |")
    lines.extend(("", "Best measured configuration for each series and row count.", ""))
    return lines, selected


def configuration_table(shape, selected, systems):
    lines = [
        f"## {SHAPE_LABELS[shape]} Selected Configurations",
        "",
        "| Shape | Rows | System | ASTs | Workers | Mode |",
        "|---|---:|---|---:|---:|---|",
    ]
    for (series, rows), record in sorted(
        selected.items(),
        key=lambda item: (item[1]["shape"], item[0][1], item[0][0]),
    ):
        if series not in systems:
            continue
        lines.append(
            f"| {record['shape']} | {rows:,} | "
            f"{SYSTEM_LABELS.get(series, series)} | {int(record['asts']):,} | "
            f"{int(record['workers'])} | `{record['execution_mode']}` |")
    lines.append("")
    return lines


def secant_comparison_table(selected_by_shape, systems):
    lines = [
        "## SECANT Relative Throughput",
        "",
        "| SECANT series | Rows | Fastest external backend | SECANT / external |",
        "|---|---:|---|---:|",
    ]
    for shape in SHAPES:
        selected = selected_by_shape[shape]
        row_counts = sorted({
            rows for series, rows in selected
            if series.startswith("secant:")})
        for rows in row_counts:
            external = max(
                (
                    record for (series, record_rows), record in selected.items()
                    if not series.startswith("secant:") and record_rows == rows
                ),
                key=lambda record: float(record["row_evals_per_second"]),
            )
            for series in systems:
                if not series.startswith("secant:"):
                    continue
                secant = selected.get((series, rows))
                if secant is None:
                    continue
                ratio = (
                    float(secant["row_evals_per_second"]) /
                    float(external["row_evals_per_second"]))
                lines.append(
                    f"| {shape} / {SYSTEM_LABELS[series]} | {rows:,} | "
                    f"{SYSTEM_LABELS.get(series_id(external), external['system'])} | "
                    f"{ratio:.2f}x |")
    lines.append("")
    return lines


def low_row_interpretation(selected_by_shape):
    sse = selected_by_shape["sse"]
    materialize = selected_by_shape["materialize"]
    row_counts = sorted({
        rows for series, rows in sse
        if series.startswith("secant:")
        and ("native_avx2", rows) in sse
    })
    if not row_counts:
        return []
    rows = row_counts[0]
    native_sse = float(sse[("native_avx2", rows)]["row_evals_per_second"])
    best_secant_sse = max(
        float(record["row_evals_per_second"])
        for (series, record_rows), record in sse.items()
        if series.startswith("secant:") and record_rows == rows
    )
    native_materialize = float(
        materialize[("native_avx2", rows)]["row_evals_per_second"])
    best_secant_materialize = max(
        float(record["row_evals_per_second"])
        for (series, record_rows), record in materialize.items()
        if series.startswith("secant:") and record_rows == rows
    )
    asts = int(next(
        record["asts"]
        for (series, record_rows), record in sse.items()
        if series.startswith("secant:") and record_rows == rows
    ))
    return [
        "## Low-row Interpretation",
        "",
        f"At {rows:,} rows, Static Native AVX2 SSE is "
        f"{native_sse / best_secant_sse:.2f}x faster than the best SECANT "
        f"GPU SSE result. The complete batch contains only "
        f"{asts * rows:,} row evaluations. GPU SSE still pays kernel scheduling, "
        "CTA shuffle/shared-memory reduction, and atomic accumulation costs, while "
        "the 12-core AVX2 path works from CPU cache. This is a low-workload "
        "crossover rather than evidence of timed host-to-device transfers.",
        "",
        f"Materialize does not pay the SSE reduction epilogue: at the same row "
        f"count, the best SECANT GPU result is "
        f"{best_secant_materialize / native_materialize:.2f}x faster than "
        "Static Native AVX2. The GPU SSE path overtakes Static Native AVX2 at "
        "the next measured "
        "row count.",
        "",
    ]


def secant_sample_summary(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    measurements = [row for row in rows if row["phase"] == "measurement"]
    if not measurements:
        raise RuntimeError(f"SECANT sample CSV has no measurements: {path}")
    kernel_launches = 0
    row_evals = 0
    for row in measurements:
        timed_batches = int(row["timed_batches"])
        launches = int(row["kernel_launches"])
        evaluations = int(row["row_evals"])
        if launches != timed_batches * int(row["kernels"]):
            raise RuntimeError("SECANT sample kernel-launch accounting is invalid")
        if evaluations != timed_batches * int(row["asts"]) * int(row["rows"]):
            raise RuntimeError("SECANT sample row-evaluation accounting is invalid")
        kernel_launches += launches
        row_evals += evaluations
    return (
        len(measurements),
        kernel_launches,
        row_evals,
        min(float(row["seconds"]) for row in measurements),
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Write a compact report from normalized fixed-corpus CSVs.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--secant-samples", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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
    corpus, corpus_hash, seed = corpus_keys.pop()
    systems = [
        series for series in SYSTEM_LABELS
        if any(series_id(record) == series for record in records)
    ]
    lines = [
        "# Fixed-Corpus Runtime Backend Comparison",
        "",
        f"- Corpus: `{corpus}`",
        f"- Corpus hash: `{corpus_hash}`",
        f"- Input seed: `{seed}`",
        "- Workload: first 256 unique balanced eight-leaf ALU expressions",
        "- Metric: complete AST row evaluations per second",
        "",
    ]
    if args.graph is not None:
        lines.extend((f"![Runtime comparison]({args.graph.name})", ""))
    selected_by_shape = {}
    for shape in SHAPES:
        table, selected = table_for_shape(records, shape, systems)
        lines.extend(table)
        selected_by_shape[shape] = selected
    lines.extend(secant_comparison_table(selected_by_shape, systems))
    lines.extend(low_row_interpretation(selected_by_shape))
    for shape in SHAPES:
        lines.extend(configuration_table(shape, selected_by_shape[shape], systems))
    lines.extend((
        "## Timing Boundary",
        "",
        "All programs and inputs are prepared before timing. SECANT uses CUDA "
        "events around kernel launches after CUBIN specialization, module load, "
        "allocation, host-to-device copies, and CPU verification. Input columns "
        "and SSE targets are already resident in device memory. Output clearing "
        "is ordered before the start event. External collectors use "
        "their warmed backend evaluation APIs and synchronize GPU work where "
        "applicable. Search, mutation, and expression construction are excluded.",
        "",
        "The Static Native AVX2 collector calibrates multiple complete "
        "evaluations into "
        "each long timed sample. A compiler memory barrier follows every evaluation "
        "so repeated writes cannot be collapsed. Its CSV notes record the actual "
        "iterations per sample.",
        "",
        "Every plotted marker is a measured row count. The graph does not connect "
        "markers by default, and therefore does not visually interpolate between "
        "measurements. SECANT throughput is calculated from the exact number of "
        "timed batches, kernel launches, and row evaluations reported by the "
        "runtime executable. Each timed batch executes the same fixed set of 256 "
        "unique expressions against the resident data, which isolates steady-state "
        "evaluation from AST specialization and module loading.",
        "",
        "Materialize writes every AST-row result. SSE/MSE rows use each backend's "
        "available fused or materialize-then-reduce path; inspect each CSV's "
        "`execution_mode` and `notes` fields before treating reduction ratios as "
        "identical kernel contracts.",
        "",
    ))
    if args.secant_samples is not None:
        samples, launches, row_evals, minimum_seconds = secant_sample_summary(
            args.secant_samples)
        lines.extend((
            f"The SECANT raw audit contains {samples:,} accepted timing samples, "
            f"{launches:,} executed kernel launches, and "
            f"{row_evals:,} executed row evaluations. The shortest accepted "
            f"sample ran for {minimum_seconds * 1.0e3:.3f} ms.",
            "",
        ))
    lines.extend(("## Raw Data", ""))
    lines.extend(f"- [`{path.name}`]({path.name})" for path in args.inputs)
    if args.secant_samples is not None:
        lines.append(
            f"- [`{args.secant_samples.name}`]({args.secant_samples.name}) "
            "(every SECANT probe, calibration run, and accepted timing sample)")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
