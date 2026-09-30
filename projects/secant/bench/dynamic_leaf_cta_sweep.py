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
import itertools
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


FIELDNAMES = (
    "workload",
    "backend",
    "setting_owner",
    "asts_per_kernel",
    "rows",
    "settings",
    "settings_per_cta",
    "setting_ctas",
    "tile_rows",
    "row_ctas",
    "total_ctas",
    "threads",
    "parameters",
    "columns",
    "ast_mode",
    "trial",
    "milliseconds",
    "row_evals_per_second",
    "billion_row_evals_per_second",
    "registers",
    "local_bytes",
    "verify",
    "verified_settings",
)


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def executable_default(name):
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo / "build" / name,
        repo.parent / "build" / name,
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def parse_key_values(output, shape):
    result = None
    for line in output.splitlines():
        fields = {}
        for item in line.split():
            if "=" in item:
                name, value = item.split("=", 1)
                fields[name] = value
        if fields.get("shape") == shape:
            result = fields
    if result is None:
        raise RuntimeError(f"benchmark output did not contain shape={shape}")
    return result


def run_process(command):
    environment = os.environ.copy()
    environment["CUDA_MODULE_LOADING"] = "EAGER"
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}):\n{' '.join(map(str, command))}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    return process.stdout


def common_row(args, workload, backend, owner, asts_per_kernel, settings,
               settings_per_cta, tile_rows, threads, trial, milliseconds,
               rate, fields):
    setting_ctas = (settings + settings_per_cta - 1) // settings_per_cta
    row_ctas = (args.rows + tile_rows - 1) // tile_rows
    return {
        "workload": workload,
        "backend": backend,
        "setting_owner": owner,
        "asts_per_kernel": asts_per_kernel,
        "rows": args.rows,
        "settings": settings,
        "settings_per_cta": settings_per_cta,
        "setting_ctas": setting_ctas,
        "tile_rows": tile_rows,
        "row_ctas": row_ctas,
        "total_ctas": setting_ctas * row_ctas,
        "threads": threads,
        "parameters": args.parameters,
        "columns": args.columns,
        "ast_mode": args.sse_ast_mode if workload == "sse" else args.lm_ast_mode,
        "trial": trial,
        "milliseconds": f"{milliseconds:.9g}",
        "row_evals_per_second": f"{rate:.9g}",
        "billion_row_evals_per_second": f"{rate / 1.0e9:.9g}",
        "registers": fields.get("registers", ""),
        "local_bytes": fields.get("local_bytes", ""),
        "verify": fields.get("verify", fields.get("checked", "")),
        "verified_settings": fields.get("checked_settings", ""),
    }


def run_lm(args, backend, owner, settings, settings_per_cta,
           tile_rows, threads, trial):
    command = [
        str(args.lm_benchmark),
        "--backend", backend,
        "--setting-owner", owner,
        "--ast-mode", args.lm_ast_mode,
        "--bindings", args.lm_bindings,
        "--rows", str(args.rows),
        "--settings", str(settings),
        "--settings-per-cta", str(settings_per_cta),
        "--parameters", str(args.parameters),
        "--columns", str(args.columns),
        "--tile-rows", str(tile_rows),
        "--threads", str(threads),
        "--verify-settings", str(args.lm_verify_settings),
        "--warmups", str(args.lm_warmups),
        "--iterations", str(args.lm_iterations),
        "--device", str(args.device),
    ]
    fields = parse_key_values(run_process(command), "dynamic_leaf_lm")
    milliseconds = float(fields["milliseconds"])
    rate = float(fields["grow_evals_per_s"]) * 1.0e9
    return common_row(
        args, "lm", backend, owner, 1, settings, settings_per_cta,
        tile_rows, threads, trial, milliseconds, rate, fields)


def run_sse(args, backend, owner, asts_per_kernel, settings,
            settings_per_cta, tile_rows, threads, trial):
    command = [
        str(args.sse_benchmark),
        "--backend", backend,
        "--setting-owner", owner,
        "--ast-mode", args.sse_ast_mode,
        "--modules", "1",
        "--workers", "1",
        "--streams", "1",
        "--kernels", "1",
        "--asts-per-kernel", str(asts_per_kernel),
        "--leaves", str(args.parameters),
        "--dynamic-sites", str(args.parameters),
        "--columns", str(args.columns),
        "--column-capacity", str(args.columns),
        "--targets", "1",
        "--settings", str(settings),
        "--settings-per-cta", str(settings_per_cta),
        "--rows", str(args.rows),
        "--tile-rows", str(tile_rows),
        "--threads", str(threads),
        "--warmups", str(args.sse_warmups),
        "--iterations", str(args.sse_iterations),
        "--device", str(args.device),
    ]
    fields = parse_key_values(run_process(command), "dynamic_leaf_sse")
    rate = float(fields["runtime_row_evals_per_second"])
    milliseconds = args.rows * settings * asts_per_kernel * 1000.0 / rate
    return common_row(
        args, "sse", backend, owner, asts_per_kernel, settings,
        settings_per_cta,
        tile_rows, threads, trial, milliseconds, rate, fields)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def print_summary(rows, count):
    grouped = {}
    for row in rows:
        key = (
            row["workload"], row["backend"], row["setting_owner"],
            int(row["settings"]), int(row["asts_per_kernel"]),
            int(row["settings_per_cta"]),
            int(row["tile_rows"]), int(row["threads"]),
        )
        grouped.setdefault(key, []).append(float(row["row_evals_per_second"]))
    partitions = {}
    for key, rates in grouped.items():
        partition = key[:4]
        partitions.setdefault(partition, []).append(
            (statistics.median(rates), key))
    print("\nBest median hot-kernel configurations:", file=sys.stderr)
    for partition in sorted(partitions):
        workload, backend, owner, settings = partition
        ranked = sorted(partitions[partition], reverse=True)
        print(f"  {workload} {backend} {owner} settings={settings}:", file=sys.stderr)
        for rate, key in ranked[:count]:
            _, _, _, _, asts_per_kernel, settings_per_cta, tile_rows, threads = key
            print(
                f"    asts/kernel={asts_per_kernel} settings/cta={settings_per_cta} "
                f"tile={tile_rows} "
                f"threads={threads} rate={rate / 1.0e9:.3f}B row-evals/s",
                file=sys.stderr,
            )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Sweep dynamic-leaf SSE and LM setting/CTA topology.")
    parser.add_argument(
        "--sse-benchmark", type=Path,
        default=executable_default("secant_dynamic_leaf_sse_bench"))
    parser.add_argument(
        "--lm-benchmark", type=Path,
        default=executable_default("secant_dynamic_leaf_lm_bench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", choices=("sse", "lm"), default=("sse", "lm"))
    parser.add_argument("--backends", nargs="+", choices=("cuda", "ptx"), default=("ptx",))
    parser.add_argument("--owners", nargs="+", choices=("thread", "warp"), default=("thread",))
    parser.add_argument("--rows", type=positive_int, default=16384)
    parser.add_argument(
        "--sse-asts-per-kernel", nargs="+", type=positive_int, default=(1,))
    parser.add_argument("--settings", nargs="+", type=positive_int, default=(4096, 8192))
    parser.add_argument(
        "--settings-per-cta", nargs="+", type=positive_int,
        default=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192))
    parser.add_argument(
        "--tile-rows", nargs="+", type=positive_int,
        default=(32, 64, 128, 256, 512))
    parser.add_argument("--threads", nargs="+", type=positive_int, default=(128,))
    parser.add_argument("--parameters", type=positive_int, default=8)
    parser.add_argument("--columns", type=positive_int, default=8)
    parser.add_argument("--trials", type=positive_int, default=1)
    parser.add_argument("--sse-ast-mode", choices=("simple", "alu", "mufu"), default="alu")
    parser.add_argument("--lm-ast-mode", choices=("linear", "square-cube"), default="square-cube")
    parser.add_argument("--lm-bindings", choices=("constants", "mixed"), default="mixed")
    parser.add_argument("--sse-warmups", type=positive_int, default=2)
    parser.add_argument("--sse-iterations", type=positive_int, default=10)
    parser.add_argument("--lm-warmups", type=positive_int, default=5)
    parser.add_argument("--lm-iterations", type=positive_int, default=50)
    parser.add_argument("--lm-verify-settings", type=positive_int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--summary-count", type=positive_int, default=3)
    args = parser.parse_args(argv)
    if args.parameters > 8:
        parser.error("--parameters must not exceed 8")
    if args.columns + args.parameters > 32:
        parser.error("--columns plus --parameters must not exceed 32")
    if any(value > 1024 for value in args.threads):
        parser.error("--threads must not exceed 1024")
    if "warp" in args.owners and any(value % 32 for value in args.threads):
        parser.error("warp ownership requires thread counts divisible by 32")
    return args


def main(argv=None):
    args = parse_args(argv)
    cases = []
    for workload, backend, owner, settings, settings_per_cta, tile_rows, threads in itertools.product(
            args.workloads, args.backends, args.owners, args.settings,
            args.settings_per_cta, args.tile_rows, args.threads):
        ast_counts = args.sse_asts_per_kernel if workload == "sse" else (1,)
        if settings_per_cta <= settings:
            for asts_per_kernel in ast_counts:
                cases.append((
                    workload, backend, owner, asts_per_kernel, settings,
                    settings_per_cta, tile_rows, threads))
    total = len(cases) * args.trials
    rows = []
    completed = 0
    for case in cases:
        workload, backend, owner, asts_per_kernel, settings, settings_per_cta, tile_rows, threads = case
        for trial in range(args.trials):
            completed += 1
            print(
                f"[{completed}/{total}] {workload} {backend} {owner} settings={settings} "
                f"asts/kernel={asts_per_kernel} settings/cta={settings_per_cta} "
                f"tile={tile_rows} threads={threads}",
                file=sys.stderr,
                flush=True,
            )
            if workload == "sse":
                row = run_sse(
                    args, backend, owner, asts_per_kernel, settings,
                    settings_per_cta, tile_rows, threads, trial)
            else:
                row = run_lm(
                    args, backend, owner, settings, settings_per_cta,
                    tile_rows, threads, trial)
            rows.append(row)
            write_rows(args.output, rows)
    print_summary(rows, args.summary_count)
    print(f"wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
