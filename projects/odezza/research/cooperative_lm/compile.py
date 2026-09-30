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
from pathlib import Path
import time

from cuda.core import Device, Program, ProgramOptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--fit-threads", type=int, required=True)
    parser.add_argument("--distributed-solve", action="store_true")
    parser.add_argument("--leader-evaluation", action="store_true")
    parser.add_argument("--arch", default="sm_120")
    arguments = parser.parse_args()

    device = Device(0)
    device.set_current()
    options = ProgramOptions(
        name=f"cooperative-lm-t{arguments.fit_threads}",
        std="c++17",
        arch=arguments.arch,
        use_fast_math=True,
        define_macro=[
            f"ODEZZA_FIT_THREAD_COUNT={arguments.fit_threads}",
            f"ODEZZA_DISTRIBUTED_SOLVE={int(arguments.distributed_solve)}",
            f"ODEZZA_LEADER_EVALUATION={int(arguments.leader_evaluation)}",
        ],
        ptxas_options=["-O3"],
    )
    program = Program(Path(arguments.source).read_text(), code_type="c++", options=options)
    started = time.perf_counter()
    try:
        object_code = program.compile("cubin")
        cubin = bytes(object_code.code)
        elapsed = time.perf_counter() - started
        close = getattr(object_code, "close", None)
        if close is not None:
            close()
    finally:
        program.close()
    Path(arguments.output).write_bytes(cubin)
    print(
        f"fit_threads={arguments.fit_threads} "
        f"distributed_solve={int(arguments.distributed_solve)} "
        f"leader_evaluation={int(arguments.leader_evaluation)} "
        f"compile_seconds={elapsed:.6f} "
        f"cubin_bytes={len(cubin)}"
    )


if __name__ == "__main__":
    main()
