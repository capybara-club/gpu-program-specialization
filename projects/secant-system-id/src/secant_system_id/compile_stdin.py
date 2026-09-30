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
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .compiler import compile_cuda


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile CUDA source from stdin with the installed NVRTC library."
    )
    parser.add_argument("--arch", default="sm_120")
    parser.add_argument("--no-fast-math", action="store_true")
    parser.add_argument("-o", "--output", default="-", help="output CUBIN path, or - for stdout")
    arguments = parser.parse_args()
    result = compile_cuda(
        sys.stdin.read(),
        arguments.arch,
        fast_math=not arguments.no_fast_math,
    )
    if arguments.output == "-":
        sys.stdout.buffer.write(result.cubin)
    else:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(result.cubin)


if __name__ == "__main__":
    main()
