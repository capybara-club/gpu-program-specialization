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
"""Python CUDA-generator command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .model import KernelShape
from .template import generate_cuda


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the multi-system CUDA score template.")
    parser.add_argument("--shape", help="optional KernelShape JSON file")
    parser.add_argument("-o", "--output", default="-")
    arguments = parser.parse_args(argv)
    shape = (
        KernelShape()
        if arguments.shape is None
        else KernelShape.from_dict(json.loads(Path(arguments.shape).read_text()))
    )
    source = generate_cuda(shape)
    if arguments.output == "-":
        sys.stdout.write(source)
    else:
        Path(arguments.output).write_text(source)


if __name__ == "__main__":
    main()
