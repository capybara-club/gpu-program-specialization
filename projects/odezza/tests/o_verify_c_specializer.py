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
import hashlib
import json
from pathlib import Path

from odezza.ast import OutputPrograms, Program, SystemGroup, constant_source, source, state_source, toggle2
from odezza.inspection import plan_from_inspection
from odezza.manifest import shape_from_manifest
from odezza.sass import specialize_cubin


def expected_group(shape) -> SystemGroup:
    prey = source(state_source(0))
    predator = source(state_source(1))
    alpha = source(constant_source(0))
    beta = source(constant_source(1))
    growth = source(constant_source(2))
    loss = source(constant_source(3))
    fixed = Program.from_expression(alpha * prey - beta * prey * predator, shape)
    first = Program.from_expression(growth * prey * predator - loss * predator, shape)
    second = Program.from_expression(growth / toggle2(0, state_source(0), 1.0) - loss * predator, shape)
    return SystemGroup(
        OutputPrograms((0,), (fixed,)),
        (
            OutputPrograms((1,), (first,)),
            OutputPrograms((1,), (second,)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare C99 and Python scoring-CUBIN specialization byte for byte.")
    parser.add_argument("template")
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--c-specialized", required=True)
    arguments = parser.parse_args()
    template = Path(arguments.template).read_bytes()
    inspection = json.loads(Path(arguments.inspection).read_text())
    manifest, plan = plan_from_inspection(inspection, template)
    shape = shape_from_manifest(manifest)
    expected = specialize_cubin(template, plan, expected_group(shape), shape).cubin
    actual = Path(arguments.c_specialized).read_bytes()
    if expected != actual:
        differing = next((index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), None)
        raise SystemExit(
            f"C/Python specialization differs at byte {differing}; "
            f"expected {hashlib.sha256(expected).hexdigest()}, actual {hashlib.sha256(actual).hexdigest()}"
        )
    print(f"C/Python specialization parity: exact; {len(actual)} bytes; sha256 {hashlib.sha256(actual).hexdigest()}")


if __name__ == "__main__":
    main()
