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
# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
"""Run selected existing host checks. No installs, downloads, GPU or credentials."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*command, cwd=ROOT):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["secant", "secant-sr", "odezza", "mm-ptx"])
    args = parser.parse_args()
    required = ["cc", "c++", "cmake", "ctest", "make"]
    missing = [x for x in required if shutil.which(x) is None]
    if missing:
        raise SystemExit("Missing prerequisites (nothing installed): " + ", ".join(missing))
    for name, option in [("secant", "SECANT_ENABLE_CUDA_INTEGRATION"), ("secant-sr", "SECANT_SR_ENABLE_CUDA")]:
        if args.only and args.only != name:
            continue
        source, build = ROOT / "projects" / name, ROOT / "build" / name
        run("cmake", "-S", source, "-B", build, f"-D{option}=OFF", "-DCMAKE_BUILD_TYPE=Release")
        run("cmake", "--build", build, "--parallel", min(os.cpu_count() or 2, 8))
        run("ctest", "--test-dir", build, "--output-on-failure")
    if args.only in (None, "odezza"):
        run("make", "test", "test-cache", cwd=ROOT / "projects/odezza/frontend")
    if args.only in (None, "mm-ptx"):
        source = ROOT / "projects/mm-ptx"
        build = ROOT / "build/mm-ptx-host"
        build.mkdir(parents=True, exist_ok=True)
        includes = [arg for path in [source, source / "common", source / "tools", source / "thirdparty/boost_subset"] for arg in ["-I", path]]
        cases = [
            ("ptx_inject_inconsistent_duplicate_args", "c"),
            ("stack_ptx_zero_frame_depth_empty_string", "c"),
            ("stack_ptx_meta_dup_empty_stack", "c"),
            ("stack_ptx_meta_reverse_empty_stack", "c"),
            ("ast_ptx_expected_output", "c"),
            ("ast_ptx_cpp", "cpp"),
        ]
        for name, ext in cases:
            binary = build / name
            run("c++" if ext == "cpp" else "cc", "-std=c++20" if ext == "cpp" else "-std=c99", "-O2", *includes, source / "tests" / f"test_{name}.{ext}", "-lm", "-o", binary)
            run(binary)
        print(f"MM PTX: {len(cases)} host tests passed; CUDA injection tests not run.")
    print("Selected host checks passed. This does not validate GPU execution.")


if __name__ == "__main__":
    main()
