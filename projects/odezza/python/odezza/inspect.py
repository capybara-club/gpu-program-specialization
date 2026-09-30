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
"""Python CUBIN-inspection command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .inspection import inspect_module


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify generated CUDA against a CUBIN and emit its physical layout.")
    parser.add_argument("cubin")
    parser.add_argument("--source", required=True)
    parser.add_argument("-o", "--output", default="-")
    arguments = parser.parse_args(argv)
    result = inspect_module(Path(arguments.source).read_text(), Path(arguments.cubin).read_bytes())
    rendered = json.dumps(result.document, indent=2, sort_keys=True) + "\n"
    if arguments.output == "-":
        sys.stdout.write(rendered)
    else:
        Path(arguments.output).write_text(rendered)


if __name__ == "__main__":
    main()
