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
import math
from pathlib import Path
import statistics
import subprocess
import sys

from common import positive_int, write_csv


SHAPES = ("materialize", "sse")
BACKENDS = ("cubin", "cuda", "ptx")
BACKEND_NAMES = {
    "cubin": "gpu_direct_cubin",
    "cuda": "gpu_nvrtc_cuda",
    "ptx": "gpu_ptx_inject_nvptxcompiler",
}
SAMPLE_FIELDS = (
    "backend",
    "shape",
    "ast_mode",
    "corpus",
    "corpus_hash",
    "expanded_corpus_hash",
    "seed",
    "rows",
    "asts",
    "kernels",
    "asts_per_kernel",
    "tile_rows",
    "threads",
    "streams",
    "source_sm",
    "target_sm",
    "opt_level",
    "phase",
    "sample_index",
    "timed_batches",
    "kernel_launches",
    "row_evals",
    "seconds",
    "batch_seconds",
    "row_evals_per_second",
)


def default_benchmark():
    repo = Path(__file__).resolve().parents[2]
    candidates = (
        repo.parent / "build" / "secant" / "secant_runtime_bench",
        repo / "build" / "secant_runtime_bench",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_runtime_output(text):
    lines = [line for line in text.splitlines() if line.startswith("runtime ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one SECANT runtime result, found {len(lines)}")
    fields = {}
    for item in lines[0].split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise RuntimeError(f"invalid SECANT runtime field: {item}")
        fields[key] = value
    required = {
        "backend",
        "shape",
        "ast_mode",
        "seed",
        "kernels",
        "asts_per_kernel",
        "tile_rows",
        "threads",
        "streams",
        "rows",
        "device_inputs_resident",
        "timed_transfers",
        "iterations",
        "timed_batches",
        "kernel_launches",
        "seconds",
        "row_evals",
        "row_evals_per_second",
        "corpus",
        "corpus_hash",
        "expanded_corpus_hash",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise RuntimeError(f"SECANT runtime result is missing {missing[0]}")
    return fields


def run_once(
    benchmark,
    backend,
    shape,
    rows,
    kernels,
    asts_per_kernel,
    tile_rows,
    threads,
    streams,
    warmups,
    iterations,
    check_rows,
    source_sm,
    target_sm,
    opt_level,
    seed,
):
    command = [
        str(benchmark),
        "--backend", backend,
        "--shape", shape,
        "--ast-mode", "alu",
        "--warmups", str(warmups),
        "--run-iterations", str(iterations),
        "--run-rows", str(rows),
        "--kernels", str(kernels),
        "--asts-per-kernel", str(asts_per_kernel),
        "--inputs", "8",
        "--targets", "1",
        "--tile-rows", str(tile_rows),
        "--threads", str(threads),
        "--streams", str(streams),
        "--check-rows", str(min(check_rows, rows)),
        "--source-sm", str(source_sm),
        "--target-sm", str(target_sm),
        "--opt-level", str(opt_level),
        "--seed", str(seed),
    ]
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"SECANT runtime benchmark failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    if "status=pass" not in process.stdout:
        raise RuntimeError("SECANT runtime benchmark did not report CPU verification")
    return parse_runtime_output(process.stdout)


def validate_result(
    result,
    backend,
    shape,
    rows,
    kernels,
    asts_per_kernel,
    iterations,
    streams,
    seed,
):
    expected = {
        "backend": backend,
        "shape": shape,
        "ast_mode": "alu",
        "rows": str(rows),
        "kernels": str(kernels),
        "asts_per_kernel": str(asts_per_kernel),
        "iterations": str(iterations),
        "streams": str(streams),
        "seed": str(seed),
    }
    for key, value in expected.items():
        if result[key] != value:
            raise RuntimeError(
                f"SECANT runtime returned {key}={result[key]}, expected {value}")
    if float(result["seconds"]) <= 0.0:
        raise RuntimeError("SECANT runtime returned a nonpositive duration")
    if result["device_inputs_resident"] != "1":
        raise RuntimeError("SECANT runtime did not report resident device inputs")
    if result["timed_transfers"] != "0":
        raise RuntimeError("SECANT runtime included transfers in its timed region")
    timed_batches = int(result["timed_batches"])
    kernel_launches = int(float(result["kernel_launches"]))
    row_evals = int(float(result["row_evals"]))
    expected_kernel_launches = iterations * kernels
    expected_row_evals = iterations * kernels * asts_per_kernel * rows
    if timed_batches != iterations:
        raise RuntimeError(
            f"SECANT runtime counted {timed_batches} timed batches, "
            f"expected {iterations}")
    if kernel_launches != expected_kernel_launches:
        raise RuntimeError(
            f"SECANT runtime counted {kernel_launches} kernel launches, "
            f"expected {expected_kernel_launches}")
    if row_evals != expected_row_evals:
        raise RuntimeError(
            f"SECANT runtime counted {row_evals} row evaluations, "
            f"expected {expected_row_evals}")


def sample_record(
    result,
    backend,
    shape,
    rows,
    asts,
    kernels,
    asts_per_kernel,
    tile_rows,
    threads,
    streams,
    source_sm,
    target_sm,
    opt_level,
    phase,
    sample_index,
):
    timed_batches = int(result["timed_batches"])
    seconds = float(result["seconds"])
    return {
        "backend": BACKEND_NAMES[backend],
        "shape": shape,
        "ast_mode": result["ast_mode"],
        "corpus": result["corpus"],
        "corpus_hash": result["corpus_hash"],
        "expanded_corpus_hash": result["expanded_corpus_hash"],
        "seed": result["seed"],
        "rows": rows,
        "asts": asts,
        "kernels": kernels,
        "asts_per_kernel": asts_per_kernel,
        "tile_rows": tile_rows,
        "threads": threads,
        "streams": streams,
        "source_sm": source_sm,
        "target_sm": target_sm,
        "opt_level": opt_level,
        "phase": phase,
        "sample_index": sample_index,
        "timed_batches": timed_batches,
        "kernel_launches": int(float(result["kernel_launches"])),
        "row_evals": int(float(result["row_evals"])),
        "seconds": f"{seconds:.9f}",
        "batch_seconds": f"{seconds / timed_batches:.12f}",
        "row_evals_per_second": result["row_evals_per_second"],
    }


def write_samples_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise RuntimeError("refusing to write an empty SECANT sample CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect_case(
    benchmark,
    backend,
    shape,
    rows,
    asts,
    asts_per_kernel,
    tile_rows,
    threads,
    streams,
    warmups,
    repeats,
    probe_iterations,
    min_seconds,
    max_iterations,
    check_rows,
    source_sm,
    target_sm,
    opt_level,
    seed,
):
    if asts % asts_per_kernel != 0:
        raise RuntimeError(
            f"AST count {asts} is not divisible by packing {asts_per_kernel}")
    kernels = asts // asts_per_kernel
    probe = run_once(
        benchmark,
        backend,
        shape,
        rows,
        kernels,
        asts_per_kernel,
        tile_rows,
        threads,
        streams,
        warmups,
        probe_iterations,
        check_rows,
        source_sm,
        target_sm,
        opt_level,
        seed,
    )
    validate_result(
        probe,
        backend,
        shape,
        rows,
        kernels,
        asts_per_kernel,
        probe_iterations,
        streams,
        seed,
    )
    seconds_per_iteration = float(probe["seconds"]) / probe_iterations
    iterations = max(
        probe_iterations,
        math.ceil(min_seconds / seconds_per_iteration),
    )
    iterations = min(iterations, max_iterations)

    results = []
    raw_results = [(probe, "probe", 0)]
    calibration_index = 0
    while len(results) < repeats:
        result = run_once(
            benchmark,
            backend,
            shape,
            rows,
            kernels,
            asts_per_kernel,
            tile_rows,
            threads,
            streams,
            warmups,
            iterations,
            check_rows,
            source_sm,
            target_sm,
            opt_level,
            seed,
        )
        validate_result(
            result,
            backend,
            shape,
            rows,
            kernels,
            asts_per_kernel,
            iterations,
            streams,
            seed,
        )
        if (
            not results
            and float(result["seconds"]) < min_seconds
            and iterations < max_iterations
        ):
            raw_results.append(
                (result, "calibration", calibration_index))
            calibration_index += 1
            iterations = min(
                max_iterations,
                max(
                    iterations + 1,
                    math.ceil(
                        iterations
                        * min_seconds
                        / float(result["seconds"])
                        * 1.1),
                ),
            )
            continue
        results.append(result)
        raw_results.append(
            (result, "measurement", len(results) - 1))

    sample_rows = [
        sample_record(
            result,
            backend,
            shape,
            rows,
            asts,
            kernels,
            asts_per_kernel,
            tile_rows,
            threads,
            streams,
            source_sm,
            target_sm,
            opt_level,
            phase,
            sample_index,
        )
        for result, phase, sample_index in raw_results
    ]

    corpus_keys = {
        (result["corpus"], result["corpus_hash"], result["expanded_corpus_hash"])
        for result in results
    }
    if len(corpus_keys) != 1:
        raise RuntimeError(f"SECANT runtime corpus identity changed: {corpus_keys}")

    batch_seconds = [
        float(result["seconds"]) / int(result["iterations"])
        for result in results
    ]
    best = min(batch_seconds)
    median = statistics.median(batch_seconds)
    corpus, corpus_hash, expanded_hash = corpus_keys.pop()
    record = {
        "system": "secant",
        "backend": BACKEND_NAMES[backend],
        "shape": shape,
        "ast_mode": "alu",
        "corpus": corpus,
        "corpus_hash": corpus_hash,
        "seed": seed,
        "rows": rows,
        "asts": asts,
        "workers": 1,
        "execution_mode": (
            f"kernel_only_k{kernels}_a{asts_per_kernel}"
            + (f"_tile{tile_rows}" if shape == "sse" else "")
            + f"_streams{streams}"),
        "best_seconds": f"{best:.9f}",
        "median_seconds": f"{median:.9f}",
        "asts_per_second": f"{asts / median:.3f}",
        "row_evals_per_second": f"{asts * rows / median:.3f}",
        "repeats": len(results),
        "checksum": "0",
        "notes": (
            f"expanded_corpus_hash={expanded_hash};"
            f"cuda_events;kernel_only;cpu_verified;iterations={iterations};"
            "device_inputs_resident=1;timed_transfers=0;"
            f"timed_batches={iterations};"
            f"kernel_launches={iterations * kernels};"
            f"timed_row_evals={iterations * asts * rows};"
            f"streams={streams}"),
    }
    return record, sample_rows


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect fixed-corpus SECANT GPU backend runtime data.")
    parser.add_argument("--benchmark", type=Path, default=default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--samples-output",
        type=Path,
        help="write every probe and timed process sample to this CSV",
    )
    parser.add_argument(
        "--backends",
        choices=BACKENDS,
        nargs="+",
        default=["cubin"],
    )
    parser.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    parser.add_argument(
        "--rows",
        type=positive_int,
        nargs="+",
        default=[1_024, 16_384, 262_144],
    )
    parser.add_argument("--asts", type=positive_int, nargs="+", default=[256])
    parser.add_argument(
        "--asts-per-kernel",
        type=positive_int,
        nargs="+",
        default=[2, 8, 32, 128],
    )
    parser.add_argument(
        "--sse-tile-rows",
        type=positive_int,
        nargs="+",
        default=[128, 256, 1_024, 4_096],
    )
    parser.add_argument("--threads", type=positive_int, default=128)
    parser.add_argument("--streams", type=positive_int, nargs="+", default=[1])
    parser.add_argument("--warmups", type=positive_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--probe-iterations", type=positive_int, default=10)
    parser.add_argument("--min-seconds", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=positive_int, default=100_000)
    parser.add_argument("--check-rows", type=positive_int, default=257)
    parser.add_argument("--source-sm", type=positive_int, default=80)
    parser.add_argument("--target-sm", type=positive_int, default=120)
    parser.add_argument("--opt-level", choices=(0, 1), type=int, default=1)
    parser.add_argument("--seed", type=positive_int, default=1)
    args = parser.parse_args(argv)
    if args.min_seconds <= 0.0:
        parser.error("--min-seconds must be positive")
    if not args.benchmark.is_file():
        parser.error(f"benchmark executable does not exist: {args.benchmark}")
    if any(tile_rows < args.threads for tile_rows in args.sse_tile_rows):
        parser.error("every --sse-tile-rows value must be at least --threads")
    for values, label in (
        (args.shapes, "shapes"),
        (args.backends, "backends"),
        (args.rows, "rows"),
        (args.asts, "asts"),
        (args.asts_per_kernel, "asts-per-kernel"),
        (args.sse_tile_rows, "sse-tile-rows"),
        (args.streams, "streams"),
    ):
        if len(values) != len(set(values)):
            parser.error(f"--{label} contains duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    records = []
    sample_rows = []
    for backend in args.backends:
        for shape in args.shapes:
            for rows in args.rows:
                for asts in args.asts:
                    for asts_per_kernel in args.asts_per_kernel:
                        kernels = asts // asts_per_kernel
                        tile_rows_values = (
                            [1] if shape == "materialize"
                            else [
                                tile_rows for tile_rows in args.sse_tile_rows
                                if tile_rows <= rows
                            ]
                        )
                        if not tile_rows_values:
                            raise RuntimeError(
                                f"no SSE tile size is valid for {rows} rows")
                        for tile_rows in tile_rows_values:
                            for streams in args.streams:
                                if streams > kernels:
                                    continue
                                record, case_samples = collect_case(
                                    args.benchmark,
                                    backend,
                                    shape,
                                    rows,
                                    asts,
                                    asts_per_kernel,
                                    tile_rows,
                                    args.threads,
                                    streams,
                                    args.warmups,
                                    args.repeats,
                                    args.probe_iterations,
                                    args.min_seconds,
                                    args.max_iterations,
                                    args.check_rows,
                                    args.source_sm,
                                    args.target_sm,
                                    args.opt_level,
                                    args.seed,
                                )
                                records.append(record)
                                sample_rows.extend(case_samples)
                                write_csv(args.output, records)
                                if args.samples_output is not None:
                                    write_samples_csv(
                                        args.samples_output,
                                        sample_rows)
                                print(
                                    f"secant {backend} {shape} rows={rows} "
                                    f"asts={asts} "
                                    f"asts_per_kernel={asts_per_kernel} "
                                    f"tile_rows={tile_rows} streams={streams} "
                                    f"row_evals_per_second="
                                    f"{float(record['row_evals_per_second']):.3e}",
                                    flush=True,
                                )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
