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
import json
from pathlib import Path

from odezza import (
    LMKernelShape,
    LMSystem,
    optimized_constant,
    source,
    state_source,
    toggle2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the example 2x8 LM system document.")
    parser.add_argument("-o", "--output")
    arguments = parser.parse_args()
    shape = LMKernelShape(
        state_count=2,
        optimized_constant_count=8,
        site_patch_capacity=256,
    )
    x = source(state_source(0))
    y = source(state_source(1))
    parameter = tuple(optimized_constant(index) for index in range(8))
    selected_state = toggle2(0, y, x)
    rhs_0 = -parameter[0] * x + parameter[1] * selected_state
    rhs_0 += parameter[2] * x * y + parameter[3]
    rhs_1 = parameter[4] * x - parameter[5] * y
    rhs_1 += parameter[2] * x * y + parameter[6] + parameter[7] * x * x
    system = LMSystem.from_expressions(
        (rhs_0, rhs_1),
        shape,
    )
    rendered = json.dumps(system.to_document(shape), indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
