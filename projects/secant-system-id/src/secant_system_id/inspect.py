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
import json
from pathlib import Path
import sys

from .cli import main as cli_main
from .inspection import inspect_module


def _read_source(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text()


def _read_cubin(path: str) -> bytes:
    return sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()


def main() -> None:
    # Preserve the original CUBIN-only command while adding the bound module
    # inspection form whenever --source is present.
    if "--source" not in sys.argv[1:]:
        cli_main(["inspect", *sys.argv[1:]])
        return
    parser = argparse.ArgumentParser(
        description="Verify generated CUDA source against its CUBIN and emit JSON."
    )
    parser.add_argument("cubin", help="CUBIN path, or - for binary stdin")
    parser.add_argument(
        "--source", required=True, help="generated CUDA source path, or - for text stdin"
    )
    parser.add_argument("-o", "--output", default="-", help="JSON path, or - for stdout")
    arguments = parser.parse_args()
    if arguments.source == "-" and arguments.cubin == "-":
        parser.error("source and CUBIN cannot both use stdin")
    result = inspect_module(
        _read_source(arguments.source),
        _read_cubin(arguments.cubin),
    )
    rendered = json.dumps(result.document, indent=2, sort_keys=True) + "\n"
    if arguments.output == "-":
        sys.stdout.write(rendered)
    else:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)


if __name__ == "__main__":
    main()
