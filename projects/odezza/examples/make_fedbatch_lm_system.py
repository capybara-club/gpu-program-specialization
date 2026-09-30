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

from odezza import LMKernelShape, LMSystem, optimized_constant, source, state_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the fixed fed-batch LM system document.")
    parser.add_argument("-o", "--output", required=True)
    arguments = parser.parse_args()

    shape = LMKernelShape(state_count=4, optimized_constant_count=6, site_patch_capacity=256)
    biomass = source(state_source(0))
    glucose = source(state_source(1))
    sucrose = source(state_source(2))
    parameter = tuple(optimized_constant(index) for index in range(6))

    growth_1 = parameter[0] * glucose * biomass
    growth_1 /= (glucose + parameter[1] * biomass) * (1.0 + parameter[2] * sucrose)
    growth_2 = parameter[3] * sucrose * biomass
    growth_2 /= (sucrose + parameter[4] * biomass) * (1.0 + parameter[5] * glucose)
    system = LMSystem.from_expressions(
        (
            growth_1 + growth_2 - 0.0055 * biomass,
            -2.58 * growth_1,
            -1.71 * growth_2,
            0.21 * biomass - 0.0466 * biomass * biomass,
        ),
        shape,
    )
    Path(arguments.output).write_text(json.dumps(system.to_document(shape), indent=2) + "\n")


if __name__ == "__main__":
    main()
