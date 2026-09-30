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
DEFAULT_AST_MODES = ("alu", "mufu")
SAMPLE_FIELDS = (
    "system",
    "platform",
    "backend",
    "opt_level",
    "shape",
    "ast_mode",
    "sample",
    "seed",
    "rows",
    "modules",
    "workers",
    "kernels_per_module",
    "asts_per_kernel",
    "asts_per_module",
    "asts",
    "tile_rows",
    "threads",
    "streams",
    "run_iterations",
    "module_transition",
    "cuda_module_loading",
    "compile_window_seconds",
    "compile_critical_seconds",
    "compile_work_seconds",
    "module_load_seconds",
    "completion_wait_seconds",
    "module_unload_seconds",
    "runtime_seconds",
    "pipeline_seconds",
    "compile_asts_per_second",
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
    "rows",
    "modules",
    "workers",
    "kernels_per_module",
    "asts_per_kernel",
    "asts_per_module",
    "asts",
    "tile_rows",
    "threads",
    "streams",
    "run_iterations",
    "module_transition",
    "cuda_module_loading",
    "samples",
    "seed_first",
    "seed_last",
    "compile_window_seconds",
    "compile_critical_seconds",
    "compile_work_seconds",
    "module_load_seconds",
    "completion_wait_seconds",
    "module_unload_seconds",
    "runtime_seconds",
    "pipeline_seconds",
    "compile_asts_per_second",
    "row_evals",
    "runtime_row_evals_per_second",
    "pipeline_row_evals_per_second",
    "corpus",
    "corpus_hash",
)
NUMERIC_FIELDS = (
    "compile_window_seconds",
    "compile_critical_seconds",
    "compile_work_seconds",
    "module_load_seconds",
    "completion_wait_seconds",
    "module_unload_seconds",
    "runtime_seconds",
    "pipeline_seconds",
    "compile_asts_per_second",
    "row_evals",
    "runtime_row_evals_per_second",
    "pipeline_row_evals_per_second",
)
TIME_FIELDS = (
    "compile_window_seconds",
    "compile_critical_seconds",
    "compile_work_seconds",
    "module_load_seconds",
    "completion_wait_seconds",
    "module_unload_seconds",
    "runtime_seconds",
    "pipeline_seconds",
)


def find_default_benchmark():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo.parent / "build" / "secant_pipeline_bench",
        repo / "build" / "secant_pipeline_bench",
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
        elif line.startswith("pipeline "):
            result = parse_key_values(line)
    if not verified:
        raise RuntimeError("benchmark did not report successful CPU verification")
    if result is None:
        raise RuntimeError("benchmark output did not contain a pipeline record")
    for name, value in expected.items():
        if result.get(name) != str(value):
            raise RuntimeError(
                f"benchmark returned {name}={result.get(name)!r}, "
                f"expected {value!r}")
    expected_completion = (
        "hip_events"
        if expected["backend"] in ("hip", "hsaco")
        else "cuda_events"
    )
    nvidia_backend = expected["backend"] in ("cuda", "ptx", "cubin")
    contracts = {
        "gpu_completion": expected_completion,
        "module_transition": (
            "eager_load_overlap_wait_unload"
            if nvidia_backend
            else "event_wait_unload_load"
        ),
        "cuda_module_loading": (
            "eager" if nvidia_backend else "not_applicable"
        ),
        "completion_order": "compile_ready_queue",
        "device_inputs_resident": "1",
        "timed_transfers": "0",
        "cold_template_prepare_timed": "0",
        "template_copy_timed": "0",
    }
    for name, value in contracts.items():
        if result.get(name) != value:
            raise RuntimeError(
                f"benchmark violated {name}: {result.get(name)!r}")
    try:
        numeric = {name: float(result[name]) for name in NUMERIC_FIELDS}
    except (KeyError, ValueError) as error:
        raise RuntimeError("benchmark returned an invalid numeric field") from error
    if any(numeric[name] <= 0.0 for name in (
            "compile_window_seconds",
            "compile_critical_seconds",
            "compile_work_seconds",
            "runtime_seconds",
            "pipeline_seconds",
            "compile_asts_per_second",
            "row_evals",
            "runtime_row_evals_per_second",
            "pipeline_row_evals_per_second")):
        raise RuntimeError("benchmark returned a non-positive primary metric")
    tolerance = max(1.0e-6, numeric["pipeline_seconds"] * 1.0e-5)
    if numeric["pipeline_seconds"] + tolerance < numeric["compile_window_seconds"]:
        raise RuntimeError("pipeline wall time is shorter than its compile window")
    if numeric["pipeline_seconds"] + tolerance < numeric["runtime_seconds"]:
        raise RuntimeError("pipeline wall time is shorter than GPU event runtime")
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


def read_samples(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(SAMPLE_FIELDS):
            raise RuntimeError(
                f"existing sample CSV has incompatible fields: {path}")
        return list(reader)


def sample_key(
    backend,
    opt_level,
    shape,
    ast_mode,
    rows,
    sample,
):
    return (
        backend,
        int(opt_level),
        shape,
        ast_mode,
        int(rows),
        int(sample),
    )


def run_case(args, backend, opt_level, shape, ast_mode, rows, seed):
    check_rows = min(args.check_rows, rows)
    command = [
        str(args.benchmark),
        "--backend", backend,
        "--shape", shape,
        "--ast-mode", ast_mode,
        "--modules", str(args.modules),
        "--workers", str(args.workers),
        "--kernels", str(args.kernels),
        "--asts-per-kernel", str(args.asts_per_kernel),
        "--inputs", str(args.inputs),
        "--targets", str(args.targets),
        "--tile-rows", str(args.tile_rows),
        "--threads", str(args.threads),
        "--streams", str(args.streams),
        "--run-rows", str(rows),
        "--run-iterations", str(args.run_iterations),
        "--check-rows", str(check_rows),
        "--check-modules", "1",
        "--patch-instructions-per-ast",
        str(args.patch_instructions_per_ast),
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
    environment["OMP_NUM_THREADS"] = str(args.workers)
    environment["OMP_DYNAMIC"] = "FALSE"
    if backend in ("cuda", "ptx", "cubin"):
        environment["CUDA_MODULE_LOADING"] = "EAGER"
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed:\n{' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    expected = {
        "backend": backend,
        "shape": shape,
        "ast_mode": ast_mode,
        "seed": seed,
        "opt_level": opt_level,
        "workers": args.workers,
        "modules": args.modules,
        "kernels_per_module": args.kernels,
        "asts_per_kernel": args.asts_per_kernel,
        "rows": rows,
        "run_iterations": args.run_iterations,
        "streams": args.streams,
        "tile_rows": args.tile_rows,
        "threads": args.threads,
    }
    return parse_result(process.stdout, expected)


def make_sample(
    args,
    backend,
    opt_level,
    shape,
    ast_mode,
    rows,
    sample,
    seed,
    result,
    numeric,
):
    row = {
        "system": args.system,
        "platform": args.platform,
        "backend": backend,
        "opt_level": opt_level,
        "shape": shape,
        "ast_mode": ast_mode,
        "sample": sample,
        "seed": seed,
        "rows": rows,
        "modules": args.modules,
        "workers": args.workers,
        "kernels_per_module": args.kernels,
        "asts_per_kernel": args.asts_per_kernel,
        "asts_per_module": result["asts_per_module"],
        "asts": result["asts"],
        "tile_rows": args.tile_rows,
        "threads": args.threads,
        "streams": args.streams,
        "run_iterations": args.run_iterations,
        "module_transition": result["module_transition"],
        "cuda_module_loading": result["cuda_module_loading"],
        "corpus": result["corpus"],
        "corpus_hash": result["corpus_hash"],
        "expanded_corpus_hash": result["expanded_corpus_hash"],
    }
    for name in NUMERIC_FIELDS:
        row[name] = (
            f"{numeric[name]:.0f}"
            if name == "row_evals"
            else f"{numeric[name]:.9g}"
        )
    return row


def summarize(samples):
    grouped = {}
    for sample in samples:
        key = (
            sample["backend"],
            int(sample["opt_level"]),
            sample["shape"],
            sample["ast_mode"],
            int(sample["rows"]),
        )
        grouped.setdefault(key, []).append(sample)

    rows = []
    for key in sorted(grouped):
        backend, opt_level, shape, ast_mode, row_count = key
        group = grouped[key]
        medians = {
            name: statistics.median(float(sample[name]) for sample in group)
            for name in TIME_FIELDS
        }
        asts = int(group[0]["asts"])
        row_evals = int(group[0]["row_evals"])
        seeds = [int(sample["seed"]) for sample in group]
        corpus_names = {sample["corpus"] for sample in group}
        corpus_hashes = {sample["corpus_hash"] for sample in group}
        if len(corpus_names) != 1 or len(corpus_hashes) != 1:
            raise RuntimeError("samples mix incompatible AST corpus definitions")
        row = {
            "system": group[0]["system"],
            "platform": group[0]["platform"],
            "backend": backend,
            "opt_level": opt_level,
            "shape": shape,
            "ast_mode": ast_mode,
            "rows": row_count,
            "modules": group[0]["modules"],
            "workers": group[0]["workers"],
            "kernels_per_module": group[0]["kernels_per_module"],
            "asts_per_kernel": group[0]["asts_per_kernel"],
            "asts_per_module": group[0]["asts_per_module"],
            "asts": asts,
            "tile_rows": group[0]["tile_rows"],
            "threads": group[0]["threads"],
            "streams": group[0]["streams"],
            "run_iterations": group[0]["run_iterations"],
            "module_transition": group[0]["module_transition"],
            "cuda_module_loading": group[0]["cuda_module_loading"],
            "samples": len(group),
            "seed_first": min(seeds),
            "seed_last": max(seeds),
            "compile_asts_per_second":
                f"{asts / medians['compile_critical_seconds']:.9g}",
            "row_evals": row_evals,
            "runtime_row_evals_per_second":
                f"{row_evals / medians['runtime_seconds']:.9g}",
            "pipeline_row_evals_per_second":
                f"{row_evals / medians['pipeline_seconds']:.9g}",
            "corpus": next(iter(corpus_names)),
            "corpus_hash": next(iter(corpus_hashes)),
        }
        for name in TIME_FIELDS:
            row[name] = f"{medians[name]:.9g}"
        rows.append(row)
    return rows


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Measure concurrent AST compilation and sequential GPU module "
            "execution with compile/run overlap."))
    parser.add_argument("--benchmark", type=Path, default=find_default_benchmark())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path)
    parser.add_argument("--platform", choices=("nvidia", "amd"), required=True)
    parser.add_argument("--system")
    parser.add_argument("--backends", nargs="+")
    parser.add_argument(
        "--opt-levels",
        type=int,
        nargs="+",
        choices=(0, 1),
        default=(0, 1))
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=("materialize", "sse"),
        default=("materialize", "sse"))
    parser.add_argument(
        "--ast-modes",
        nargs="+",
        choices=DEFAULT_AST_MODES,
        default=list(DEFAULT_AST_MODES))
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--modules", type=int, default=48)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--asts-per-kernel", type=int, default=128)
    parser.add_argument("--inputs", type=int, default=8)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=8192)
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
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard an existing raw-sample CSV instead of resuming it")
    args = parser.parse_args(argv)

    platform_defaults = {
        "nvidia": (
            "NVIDIA GeForce RTX 5090 + AMD Ryzen 9 9900X",
            NVIDIA_CONFIGS,
        ),
        "amd": (
            "Radeon RX 9070 XT + AMD Ryzen 9 7900X",
            AMD_CONFIGS,
        ),
    }
    default_system, platform_configs = platform_defaults[args.platform]
    if args.system is None:
        args.system = default_system
    requested = set(args.backends) if args.backends else None
    requested_opt_levels = set(args.opt_levels)
    available_backends = {backend for backend, _ in platform_configs}
    if requested is not None and not requested <= available_backends:
        parser.error(
            f"backend is not valid for {args.platform}: "
            f"{sorted(requested - available_backends)[0]}")
    args.configs = tuple(
        config for config in platform_configs
        if (requested is None or config[0] in requested) and
           config[1] in requested_opt_levels
    )
    if args.samples_output is None:
        suffix = args.output.suffix or ".csv"
        args.samples_output = args.output.with_name(
            args.output.stem + "_samples" + suffix)
    numeric = (
        *args.rows,
        args.repeats,
        args.modules,
        args.workers,
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
    )
    if any(value <= 0 for value in numeric):
        parser.error("benchmark dimensions must be positive")
    if args.target_sm < 0:
        parser.error("--target-sm must be nonnegative")
    if not args.configs:
        parser.error("at least one backend must be selected")
    if len(set(args.rows)) != len(args.rows):
        parser.error("row counts must be unique")
    if len(set(args.ast_modes)) != len(args.ast_modes):
        parser.error("AST modes must be unique")
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("kernel shapes must be unique")
    if len(requested_opt_levels) != len(args.opt_levels):
        parser.error("optimization levels must be unique")
    if args.streams > args.kernels:
        parser.error("streams cannot exceed kernels per module")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.benchmark.is_file():
        raise RuntimeError(f"benchmark executable not found: {args.benchmark}")

    samples = [] if args.force else read_samples(args.samples_output)
    allowed_configs = set(args.configs)
    allowed_shapes = set(args.shapes)
    allowed_modes = set(args.ast_modes)
    allowed_rows = set(args.rows)
    expected_metadata = {
        "system": args.system,
        "platform": args.platform,
        "modules": str(args.modules),
        "workers": str(args.workers),
        "kernels_per_module": str(args.kernels),
        "asts_per_kernel": str(args.asts_per_kernel),
        "tile_rows": str(args.tile_rows),
        "threads": str(args.threads),
        "streams": str(args.streams),
        "run_iterations": str(args.run_iterations),
    }
    for existing in samples:
        if any(
                existing.get(name) != value
                for name, value in expected_metadata.items()):
            raise RuntimeError(
                "existing sample CSV has incompatible benchmark dimensions")
        if (
            (existing["backend"], int(existing["opt_level"]))
                not in allowed_configs or
            existing["shape"] not in allowed_shapes or
            existing["ast_mode"] not in allowed_modes or
            int(existing["rows"]) not in allowed_rows or
            int(existing["sample"]) >= args.repeats
        ):
            raise RuntimeError(
                "existing sample CSV contains a case outside this sweep")
    completed = {
        sample_key(
            sample["backend"],
            sample["opt_level"],
            sample["shape"],
            sample["ast_mode"],
            sample["rows"],
            sample["sample"],
        )
        for sample in samples
    }
    total_groups = (
        len(args.shapes) *
        len(args.ast_modes) *
        len(args.rows) *
        args.repeats)
    group_idx = 0
    for shape_idx, shape in enumerate(args.shapes):
        for mode_idx, ast_mode in enumerate(args.ast_modes):
            for row_idx, rows in enumerate(args.rows):
                for sample in range(args.repeats):
                    rotation = (
                        shape_idx + mode_idx + row_idx + sample
                    ) % len(args.configs)
                    configurations = (
                        args.configs[rotation:] +
                        args.configs[:rotation]
                    )
                    group_idx += 1
                    for backend, opt_level in configurations:
                        key = sample_key(
                            backend,
                            opt_level,
                            shape,
                            ast_mode,
                            rows,
                            sample,
                        )
                        if key in completed:
                            continue
                        seed = (
                            args.seed +
                            shape_idx *
                                len(args.ast_modes) *
                                len(args.rows) *
                                args.repeats +
                            mode_idx *
                                len(args.rows) *
                                args.repeats +
                            row_idx * args.repeats +
                            sample)
                        result, numeric = run_case(
                            args,
                            backend,
                            opt_level,
                            shape,
                            ast_mode,
                            rows,
                            seed,
                        )
                        samples.append(make_sample(
                            args,
                            backend,
                            opt_level,
                            shape,
                            ast_mode,
                            rows,
                            sample,
                            seed,
                            result,
                            numeric,
                        ))
                        completed.add(key)
                        write_csv(
                            args.samples_output,
                            SAMPLE_FIELDS,
                            samples)
                        print(
                            f"[{group_idx}/{total_groups}] "
                            f"{backend} O{opt_level} {shape} "
                            f"{ast_mode} rows={rows:,} "
                            f"compile="
                            f"{numeric['compile_asts_per_second']:.3e} AST/s "
                            f"runtime="
                            f"{numeric['runtime_row_evals_per_second']:.3e} rows/s "
                            f"pipeline="
                            f"{numeric['pipeline_row_evals_per_second']:.3e} rows/s",
                            flush=True,
                        )

    summaries = summarize(samples)
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
