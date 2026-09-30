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

import csv
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


def write_fake_benchmark(path):
    path.write_text(
        """#!/usr/bin/env python3
import sys

def value(name):
    return sys.argv[sys.argv.index(name) + 1]

backend = value("--backend")
shape = value("--shape")
ast_mode = value("--ast-mode")
opt_level = value("--opt-level")
kernels = value("--kernels")
asts = value("--asts-per-kernel")
workers = value("--workers")
modules = value("--iterations")
rate = (
    int(kernels) * 100000 +
    int(asts) * 1000 +
    int(workers) * 100 +
    int(opt_level) * 100 +
    (10 if ast_mode == "mufu" else 0) +
    (1 if backend == "cuda" else 0))
print(
    f"compile backend={backend} shape={shape} ast_mode={ast_mode} "
    f"opt_level={opt_level} workers={workers} modules={modules} "
    f"kernels_per_module={kernels} asts_per_kernel={asts} asts=1 "
    f"seconds=1 asts_per_second={rate}.0 us_per_ast=1")
""",
        encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run(command):
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed:\n{' '.join(str(arg) for arg in command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    return process


def main():
    matrix_script = Path(sys.argv[1]).resolve()
    report_script = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        benchmark = directory / "fake_benchmark.py"
        output = directory / "matrix.csv"
        backend_outputs = {
            backend: directory / f"matrix_{backend}.csv"
            for backend in ("ptx", "cuda", "cubin")
        }
        write_fake_benchmark(benchmark)

        command = [
            sys.executable,
            str(matrix_script),
            "--benchmark", str(benchmark),
            "--output", str(output),
            "--module-shapes", "2x3", "4x5",
            "--workers", "1", "3",
            "--warmup-waves", "1",
            "--timed-waves", "2",
            "--source-sm", "80",
            "--target-sm", "120",
        ]
        run(command)

        rows = []
        backend_rows = {}
        for backend, path in backend_outputs.items():
            with path.open(newline="", encoding="utf-8") as file:
                backend_rows[backend] = list(csv.DictReader(file))
            rows.extend(backend_rows[backend])
        if len(rows) != 40:
            raise RuntimeError(f"expected 40 matrix rows, found {len(rows)}")
        if len(backend_rows["ptx"]) != 16 or len(backend_rows["cuda"]) != 16:
            raise RuntimeError("PTX/CUDA output contains the wrong row count")
        if len(backend_rows["cubin"]) != 8:
            raise RuntimeError("CUBIN output contains the wrong row count")
        if any(row["opt_level"] != "1" for row in backend_rows["cubin"]):
            raise RuntimeError("CUBIN output contains a non-O1 row")
        expected_header = [
            "backend",
            "kernel_shape",
            "opt_level",
            "ast_mode",
            "metric",
            "workers",
            "warmup_modules",
            "timed_modules",
            "source_sm",
            "target_sm",
            "tile_rows",
            "threads",
            "patch_instructions_per_ast",
            "2x3",
            "4x5",
        ]
        if list(rows[0]) != expected_header:
            raise RuntimeError(f"unexpected CSV header: {list(rows[0])}")
        for row in rows:
            if not row["2x3"] or not row["4x5"]:
                raise RuntimeError("matrix contains an empty benchmark cell")
            if int(row["warmup_modules"]) != int(row["workers"]):
                raise RuntimeError("matrix stored the wrong warmup module count")
            if int(row["timed_modules"]) != int(row["workers"]) * 2:
                raise RuntimeError("matrix stored the wrong timed module count")
            expected = (
                200000 +
                3000 +
                int(row["workers"]) * 100 +
                int(row["opt_level"]) * 100 +
                (10 if row["ast_mode"] == "mufu" else 0) +
                (1 if row["backend"] == "cuda" else 0))
            if float(row["2x3"]) != expected:
                raise RuntimeError("matrix stored the wrong AST/s value")

        benchmark.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
        benchmark.chmod(benchmark.stat().st_mode | stat.S_IXUSR)
        resumed = run(command)
        if resumed.stdout.count("resume ") != 80:
            raise RuntimeError("resume did not skip every populated cell")

        dry_run = run(command + ["--dry-run"])
        command_lines = [
            line for line in dry_run.stdout.splitlines()
            if line.startswith(str(benchmark))
        ]
        if len(command_lines) != 80:
            raise RuntimeError(
                f"expected 80 dry-run commands, found {len(command_lines)}")

        report = directory / "matrix.md"
        run([
            sys.executable,
            str(report_script),
            *(str(path) for path in backend_outputs.values()),
            "--output",
            str(report),
        ])
        report_text = report.read_text(encoding="utf-8")
        if report_text.count("## 1 Worker") != 1:
            raise RuntimeError("report is missing the one-worker section")
        if report_text.count("## 3 Workers") != 1:
            raise RuntimeError("report is missing the three-worker section")
        if report_text.count("| Configuration | Backend |") != 2:
            raise RuntimeError("report does not contain one table per worker count")
        if report_text.count("| PTX |") != 16 or report_text.count("| CUDA |") != 16:
            raise RuntimeError("report tables do not contain every backend subrow")
        if report_text.count("| CUBIN |") != 8:
            raise RuntimeError("report tables do not contain every CUBIN subrow")
        if "|  | CUDA | **" not in report_text:
            raise RuntimeError("report did not highlight the faster backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
