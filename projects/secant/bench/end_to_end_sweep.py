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
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


NVIDIA_CONFIGS = (
    ("cuda", 0),
    ("cuda", 1),
    ("ptx", 0),
    ("ptx", 1),
    ("cubin", 1),
)
AMD_CONFIGS = (
    ("hip", 0),
    ("hip", 1),
    ("hsaco", 1),
)
DEFAULT_ROWS = (
    1024,
    4096,
    16384,
    65536,
    131072,
    262144,
)
SAMPLE_FIELDS = (
    "system",
    "platform",
    "backend",
    "opt_level",
    "shape",
    "ast_mode",
    "sample",
    "seed",
    "cpu",
    "rows",
    "kernels",
    "asts_per_kernel",
    "asts",
    "tile_rows",
    "threads",
    "streams",
    "iterations",
    "compile_seconds",
    "load_seconds",
    "run_seconds",
    "pipeline_seconds",
    "compile_asts_per_second",
    "load_asts_per_second",
    "row_evals",
    "runtime_row_evals_per_second",
    "pipeline_row_evals_per_second",
    "corpus",
    "corpus_hash",
    "expanded_corpus_hash",
)
SUMMARY_FIELDS = (
    "system",
    "platform",
    "backend",
    "opt_level",
    "shape",
    "ast_mode",
    "cpu",
    "rows",
    "kernels",
    "asts_per_kernel",
    "asts",
    "tile_rows",
    "threads",
    "streams",
    "iterations",
    "samples",
    "seed_first",
    "seed_last",
    "compile_seconds",
    "load_seconds",
    "run_seconds",
    "pipeline_seconds",
    "compile_asts_per_second",
    "load_asts_per_second",
    "row_evals",
    "runtime_row_evals_per_second",
    "pipeline_row_evals_per_second",
    "corpus",
    "corpus_hash",
)


def find_default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo.parent / "build" / "secant_end_to_end_bench",
        repo / "build" / "secant_end_to_end_bench",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def parse_key_values(line):
    fields = {}
    for item in line.split()[1:]:
        if "=" in item:
            name, value = item.split("=", 1)
            fields[name] = value
    return fields


def parse_result(output, expected):
    result = None
    verified = False
    for line in output.splitlines():
        if line.startswith("verify "):
            fields = parse_key_values(line)
            verified = fields.get("status") == "pass"
        elif line.startswith("end_to_end "):
            result = parse_key_values(line)
    if not verified:
        raise RuntimeError("benchmark did not report a successful verification")
    if result is None:
        raise RuntimeError("benchmark output did not contain an end_to_end record")
    for name, value in expected.items():
        if result.get(name) != str(value):
            raise RuntimeError(
                f"benchmark returned {name}={result.get(name)!r}, "
                f"expected {value!r}")
    if result.get("device_inputs_resident") != "1":
        raise RuntimeError("benchmark did not keep inputs resident")
    if result.get("timed_transfers") != "0":
        raise RuntimeError("benchmark included data transfers")
    if result.get("cold_template_prepare_timed") != "0":
        raise RuntimeError("benchmark included cold template preparation")
    if result.get("template_copy_timed") != "0":
        raise RuntimeError("benchmark included template copying")

    numeric_names = (
        "compile_seconds",
        "load_seconds",
        "run_seconds",
        "pipeline_seconds",
        "compile_asts_per_second",
        "load_asts_per_second",
        "row_evals",
        "runtime_row_evals_per_second",
        "pipeline_row_evals_per_second",
    )
    try:
        numeric = {name: float(result[name]) for name in numeric_names}
    except (KeyError, ValueError) as error:
        raise RuntimeError("benchmark returned an invalid numeric field") from error
    phase_sum = (
        numeric["compile_seconds"] +
        numeric["load_seconds"] +
        numeric["run_seconds"]
    )
    tolerance = max(1.0e-9, phase_sum * 1.0e-6)
    if abs(numeric["pipeline_seconds"] - phase_sum) > tolerance:
        raise RuntimeError("pipeline time does not equal compile + load + run")
    return result, numeric


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_case(args, backend, opt_level, rows, seed, cpu):
    check_rows = min(args.check_rows, rows)
    command = [
        str(args.benchmark),
        "--backend", backend,
        "--shape", args.shape,
        "--ast-mode", args.ast_mode,
        "--warmups", "0",
        "--run-iterations", str(args.run_iterations),
        "--run-rows", str(rows),
        "--check-rows", str(check_rows),
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(args.asts_per_kernel),
        "--inputs", str(args.inputs),
        "--targets", str(args.targets),
        "--tile-rows", str(args.tile_rows),
        "--threads", str(args.threads),
        "--streams", str(args.streams),
        "--patch-instructions-per-ast", str(args.patch_instructions_per_ast),
        "--compile-scratch-bytes", str(args.compile_scratch_bytes),
        "--source-sm", str(args.source_sm),
        "--target-sm", str(args.target_sm),
        "--opt-level", str(opt_level),
        "--seed", str(seed),
        "--device", str(args.device),
    ]
    if args.hip_arch:
        command.extend(("--hip-arch", args.hip_arch))

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"

    def pin_child():
        os.sched_setaffinity(0, {cpu})

    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        preexec_fn=pin_child,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    expected = {
        "backend": backend,
        "shape": args.shape,
        "ast_mode": args.ast_mode,
        "seed": seed,
        "opt_level": opt_level,
        "rows": rows,
        "kernels": args.kernels,
        "asts_per_kernel": args.asts_per_kernel,
        "tile_rows": args.tile_rows,
        "threads": args.threads,
        "streams": args.streams,
        "iterations": args.run_iterations,
    }
    return parse_result(process.stdout, expected)


def sample_row(args, backend, opt_level, rows, sample, seed, cpu, result, numeric):
    return {
        "system": args.system,
        "platform": args.platform,
        "backend": backend,
        "opt_level": opt_level,
        "shape": args.shape,
        "ast_mode": args.ast_mode,
        "sample": sample,
        "seed": seed,
        "cpu": cpu,
        "rows": rows,
        "kernels": args.kernels,
        "asts_per_kernel": args.asts_per_kernel,
        "asts": result["asts"],
        "tile_rows": args.tile_rows,
        "threads": args.threads,
        "streams": args.streams,
        "iterations": args.run_iterations,
        "compile_seconds": f"{numeric['compile_seconds']:.9g}",
        "load_seconds": f"{numeric['load_seconds']:.9g}",
        "run_seconds": f"{numeric['run_seconds']:.9g}",
        "pipeline_seconds": f"{numeric['pipeline_seconds']:.9g}",
        "compile_asts_per_second": f"{numeric['compile_asts_per_second']:.9g}",
        "load_asts_per_second": f"{numeric['load_asts_per_second']:.9g}",
        "row_evals": f"{numeric['row_evals']:.0f}",
        "runtime_row_evals_per_second": f"{numeric['runtime_row_evals_per_second']:.9g}",
        "pipeline_row_evals_per_second": f"{numeric['pipeline_row_evals_per_second']:.9g}",
        "corpus": result["corpus"],
        "corpus_hash": result["corpus_hash"],
        "expanded_corpus_hash": result["expanded_corpus_hash"],
    }


def summarize(args, samples, cpu):
    rows = []
    grouped = {}
    for sample in samples:
        key = (
            sample["backend"],
            int(sample["opt_level"]),
            int(sample["rows"]),
        )
        grouped.setdefault(key, []).append(sample)

    for key in sorted(grouped):
        backend, opt_level, row_count = key
        group = grouped[key]
        asts = int(float(group[0]["asts"]))
        row_evals = int(float(group[0]["row_evals"]))
        phase_names = ("compile_seconds", "load_seconds", "run_seconds")
        medians = {
            name: statistics.median(float(sample[name]) for sample in group)
            for name in phase_names
        }
        pipeline_seconds = sum(medians.values())
        seeds = [int(sample["seed"]) for sample in group]
        corpus_names = {sample["corpus"] for sample in group}
        corpus_hashes = {sample["corpus_hash"] for sample in group}
        if len(corpus_names) != 1 or len(corpus_hashes) != 1:
            raise RuntimeError("samples mix incompatible AST corpus definitions")
        rows.append({
            "system": args.system,
            "platform": args.platform,
            "backend": backend,
            "opt_level": opt_level,
            "shape": args.shape,
            "ast_mode": args.ast_mode,
            "cpu": cpu,
            "rows": row_count,
            "kernels": args.kernels,
            "asts_per_kernel": args.asts_per_kernel,
            "asts": asts,
            "tile_rows": args.tile_rows,
            "threads": args.threads,
            "streams": args.streams,
            "iterations": args.run_iterations,
            "samples": len(group),
            "seed_first": min(seeds),
            "seed_last": max(seeds),
            "compile_seconds": f"{medians['compile_seconds']:.9g}",
            "load_seconds": f"{medians['load_seconds']:.9g}",
            "run_seconds": f"{medians['run_seconds']:.9g}",
            "pipeline_seconds": f"{pipeline_seconds:.9g}",
            "compile_asts_per_second": f"{asts / medians['compile_seconds']:.9g}",
            "load_asts_per_second": f"{asts / medians['load_seconds']:.9g}",
            "row_evals": row_evals,
            "runtime_row_evals_per_second": f"{row_evals / medians['run_seconds']:.9g}",
            "pipeline_row_evals_per_second": f"{row_evals / pipeline_seconds:.9g}",
            "corpus": next(iter(corpus_names)),
            "corpus_hash": next(iter(corpus_hashes)),
        })
    return rows


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Measure AST compile, module load, and execution over real row counts.")
    parser.add_argument("--benchmark", type=Path, default=find_default_benchmark())
    parser.add_argument("--platform", choices=("nvidia", "amd"), required=True)
    parser.add_argument("--system")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path)
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-processes", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--shape", choices=("materialize", "sse"), default="sse")
    parser.add_argument("--ast-mode", choices=("simple", "alu", "mufu"), default="mufu")
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--asts-per-kernel", type=int, default=32)
    parser.add_argument("--inputs", type=int, default=8)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--run-iterations", type=int, default=1)
    parser.add_argument("--check-rows", type=int, default=257)
    parser.add_argument("--patch-instructions-per-ast", type=int, default=64)
    parser.add_argument("--compile-scratch-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--source-sm", type=int, default=80)
    parser.add_argument("--target-sm", type=int, default=120)
    parser.add_argument("--hip-arch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)

    defaults = {
        "nvidia": ("NVIDIA GeForce RTX 5090", NVIDIA_CONFIGS),
        "amd": ("Radeon RX 9070 XT", AMD_CONFIGS),
    }
    default_system, default_configs = defaults[args.platform]
    if args.system is None:
        args.system = default_system
    if args.backends is None:
        configs = default_configs
    else:
        requested = set(args.backends)
        unknown = requested - {backend for backend, _ in default_configs}
        if unknown:
            parser.error(f"backend is not valid for {args.platform}: {sorted(unknown)[0]}")
        configs = tuple(
            config for config in default_configs
            if config[0] in requested
        )
    args.configs = configs
    if args.samples_output is None:
        args.samples_output = args.output.with_name(
            args.output.stem + "_samples" + (args.output.suffix or ".csv"))
    numeric = (
        *args.rows,
        args.repeats,
        args.kernels,
        args.asts_per_kernel,
        args.inputs,
        args.targets,
        args.tile_rows,
        args.threads,
        args.streams,
        args.run_iterations,
        args.check_rows,
        args.patch_instructions_per_ast,
        args.compile_scratch_bytes,
        args.source_sm,
        args.target_sm,
    )
    if any(value <= 0 for value in numeric) or args.warmup_processes < 0:
        parser.error("benchmark dimensions must be positive")
    if args.streams > args.kernels:
        parser.error("streams cannot exceed kernels")
    if len(set(args.rows)) != len(args.rows):
        parser.error("row counts must be unique")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        parser.error("one-core benchmark requires Linux CPU affinity support")
    available_cpus = sorted(os.sched_getaffinity(0))
    if not available_cpus:
        parser.error("current process has no available CPUs")
    if args.cpu is None:
        args.cpu = available_cpus[0]
    elif args.cpu not in available_cpus:
        parser.error(f"CPU {args.cpu} is outside the current affinity mask")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.benchmark.is_file():
        raise RuntimeError(f"benchmark executable not found: {args.benchmark}")

    for config_idx, (backend, opt_level) in enumerate(args.configs):
        for warmup in range(args.warmup_processes):
            seed = args.seed + 1000000 + config_idx * 1000 + warmup
            run_case(args, backend, opt_level, args.rows[0], seed, args.cpu)

    samples = []
    for row_idx, rows in enumerate(args.rows):
        for sample in range(args.repeats):
            rotation = sample % len(args.configs)
            ordered_configs = (
                args.configs[rotation:] +
                args.configs[:rotation]
            )
            for backend, opt_level in ordered_configs:
                seed = args.seed + row_idx * args.repeats + sample
                result, numeric = run_case(
                    args,
                    backend,
                    opt_level,
                    rows,
                    seed,
                    args.cpu,
                )
                samples.append(sample_row(
                    args,
                    backend,
                    opt_level,
                    rows,
                    sample,
                    seed,
                    args.cpu,
                    result,
                    numeric,
                ))
                print(
                    f"{backend} O{opt_level} rows={rows:,} sample={sample + 1}/"
                    f"{args.repeats} compile={numeric['compile_seconds']:.6f}s "
                    f"load={numeric['load_seconds']:.6f}s "
                    f"run={numeric['run_seconds']:.6f}s "
                    f"pipeline={numeric['pipeline_row_evals_per_second']:.3e} rows/s",
                    flush=True,
                )

    summaries = summarize(args, samples, args.cpu)
    write_csv(args.samples_output, SAMPLE_FIELDS, samples)
    write_csv(args.output, SUMMARY_FIELDS, summaries)
    print(f"wrote {args.samples_output}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
