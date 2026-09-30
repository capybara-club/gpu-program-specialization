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
"""Command-line specialization for one trajectory-LM system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lm_ast import LMSystem
from .lm_elf import inspect_lm_module
from .lm_sass import specialize_lm_cubin


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Specialize one LM system and its derivatives into a verified CUBIN."
    )
    parser.add_argument("source")
    parser.add_argument("template_cubin")
    parser.add_argument("system")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--report", default="-")
    arguments = parser.parse_args(argv)
    source = Path(arguments.source).read_text()
    cubin = Path(arguments.template_cubin).read_bytes()
    inspection = inspect_lm_module(source, cubin)
    system = LMSystem.from_document(
        json.loads(Path(arguments.system).read_text()),
        inspection.plan.shape,
    )
    result = specialize_lm_cubin(cubin, inspection.plan, system)
    Path(arguments.output).write_bytes(result.cubin)
    system_document = system.to_document(inspection.plan.shape)
    report = {
        "schema": "odezza.lm-specialization",
        "schema_version": 1,
        "template_id": inspection.manifest["template_id"],
        "system_id": system_document["system_id"],
        "required_toggle_bits": result.required_toggle_bits,
        "register_count": result.register_count,
        "site_instruction_counts": list(result.site_instruction_counts),
        "scalar_instruction_counts": [
            list(counts) for counts in result.scalar_instruction_counts
        ],
        "derivatives": result.derivatives.inspection(inspection.plan.shape),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report == "-":
        print(rendered, end="")
    else:
        Path(arguments.report).write_text(rendered)


if __name__ == "__main__":
    main()
