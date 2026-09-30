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
"""Python command dispatcher."""

from __future__ import annotations

import argparse
import importlib


COMMANDS = {
    "generate": "odezza.generate",
    "compile": "odezza.compiler",
    "inspect": "odezza.inspect",
    "example": "odezza.example",
    "specialize": "odezza.specialize",
    "run": "odezza.runner",
    "lm-generate": "odezza.lm_generate",
    "lm-inspect": "odezza.lm_inspect",
    "lm-specialize": "odezza.lm_specialize",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="odezza",
        description="Generate, inspect, specialize, and run ODE scoring kernels.",
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    module = importlib.import_module(COMMANDS[parsed.command])
    module.main(parsed.arguments)


if __name__ == "__main__":
    main()
