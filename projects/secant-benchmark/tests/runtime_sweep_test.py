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
tile_rows = value("--tile-rows")
rate = (
    int(kernels) * 100000 +
    int(asts) * 1000 +
    int(tile_rows) +
    int(opt_level) * 10 +
    (2 if ast_mode == "mufu" else 0) +
    (1 if backend == "cuda" else 0))
print(
    f"runtime backend={backend} shape={shape} ast_mode={ast_mode} "
    f"opt_level={opt_level} kernels={kernels} asts_per_kernel={asts} "
    f"tile_rows={tile_rows} row_evals_per_second={rate}.0")
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


def require_failure(command, expected_text):
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode == 0:
        raise RuntimeError(
            f"command unexpectedly succeeded:\n"
            f"{' '.join(str(arg) for arg in command)}")
    if expected_text not in process.stderr:
        raise RuntimeError(
            f"command failed without {expected_text!r}:\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}")


def read_rows(base, backend):
    path = base.with_name(f"{base.stem}_{backend}.csv")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def require_cubin_o1(rows):
    if not rows or any(row["backend"] != "cubin" for row in rows):
        raise RuntimeError("CUBIN CSV contains invalid backend rows")
    if any(row["opt_level"] != "1" for row in rows):
        raise RuntimeError("CUBIN CSV contains a non-O1 row")


def main():
    tile_script = Path(sys.argv[1]).resolve()
    packing_script = Path(sys.argv[2]).resolve()
    surface_script = Path(sys.argv[3]).resolve()

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        benchmark = directory / "fake_runtime_benchmark.py"
        tile_output = directory / "tile.csv"
        packing_output = directory / "packing.csv"
        surface_output = directory / "surface.csv"
        write_fake_benchmark(benchmark)

        common = [
            "--benchmark", str(benchmark),
            "--backends", "ptx", "cuda", "cubin",
            "--warmups", "0",
            "--run-iterations", "1",
            "--run-rows", "1024",
            "--kernels", "2",
            "--threads", "64",
            "--check-rows", "1",
        ]
        run([
            sys.executable,
            str(tile_script),
            "--output", str(tile_output),
            "--tile-rows", "64", "128",
            "--asts-per-kernel", "2",
            *common,
        ])
        tile_ptx = read_rows(tile_output, "ptx")
        tile_cuda = read_rows(tile_output, "cuda")
        tile_cubin = read_rows(tile_output, "cubin")
        if len(tile_ptx) != 8 or len(tile_cuda) != 8 or len(tile_cubin) != 4:
            raise RuntimeError("tile sweep produced the wrong per-backend row counts")
        require_cubin_o1(tile_cubin)
        require_failure([
            sys.executable,
            str(tile_script),
            "--output", str(tile_output),
            "--tile-rows", "64", "128",
            "--asts-per-kernel", "4",
            *common,
        ], "asts_per_kernel=2, expected 4")

        run([
            sys.executable,
            str(packing_script),
            "--output", str(packing_output),
            "--ast-counts", "2", "4",
            "--tile-rows", "64",
            *common,
        ])
        packing_ptx = read_rows(packing_output, "ptx")
        packing_cuda = read_rows(packing_output, "cuda")
        packing_cubin = read_rows(packing_output, "cubin")
        if len(packing_ptx) != 16 or len(packing_cuda) != 16 or len(packing_cubin) != 8:
            raise RuntimeError("packing sweep produced the wrong per-backend row counts")
        require_cubin_o1(packing_cubin)

        run([
            sys.executable,
            str(surface_script),
            "--output", str(surface_output),
            "--packing-csv", str(packing_output),
            "--tile-csv", str(tile_output),
            "--backends", "cubin",
            "--tile-rows", "64", "128",
            "--ast-counts", "2", "4",
            "--warmups", "0",
            "--run-iterations", "1",
            "--run-rows", "1024",
            "--kernels", "2",
            "--threads", "64",
            "--check-rows", "1",
            "--benchmark", str(benchmark),
        ])
        surface_cubin = read_rows(surface_output, "cubin")
        if len(surface_cubin) != 8:
            raise RuntimeError("surface sweep produced the wrong CUBIN row count")
        require_cubin_o1(surface_cubin)

    return 0


if __name__ == "__main__":
    sys.exit(main())
