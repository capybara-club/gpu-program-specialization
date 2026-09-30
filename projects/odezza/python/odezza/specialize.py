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
"""Python specialization command."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import shape_from_manifest
from .model import MAX_TOGGLE_BITS
from .ast import SystemGroup
from .inspection import plan_from_inspection
from .sass import SpecializationResult, specialize_cubin


SPECIALIZATION_SCHEMA = "odezza.specialization"
SPECIALIZATION_SCHEMA_VERSION = 1


def specialize_from_documents(
    cubin: bytes,
    inspection: dict[str, Any],
    group_document: dict[str, Any],
) -> tuple[SpecializationResult, dict[str, Any]]:
    manifest, plan = plan_from_inspection(inspection, cubin)
    shape = shape_from_manifest(manifest)
    group = SystemGroup.from_document(group_document, shape)
    result = specialize_cubin(cubin, plan, group, shape)
    report = {
        "schema": SPECIALIZATION_SCHEMA,
        "schema_version": SPECIALIZATION_SCHEMA_VERSION,
        "status": "specialized",
        "template_id": manifest["template_id"],
        "template_cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        "group_id": group_document["group_id"],
        "system_count": len(group.systems),
        "system_capacity": shape.system_capacity,
        "required_toggle_bits": group.required_toggle_bits,
        "available_toggle_bits": MAX_TOGGLE_BITS,
        "shared_instruction_counts_by_output": list(result.shared_instruction_counts),
        "system_instruction_counts_by_output": [list(value) for value in result.system_instruction_counts],
        "body_instruction_counts_including_branch": list(result.body_instruction_counts),
        "body_offsets": list(result.body_offsets),
        "template_register_count": plan.register_count,
        "specialized_register_count": result.register_count,
        "specialized_cubin_sha256": hashlib.sha256(result.cubin).hexdigest(),
    }
    return result, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Specialize the template with one shared prelude and candidate branches.")
    parser.add_argument("cubin")
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--report", default="-")
    arguments = parser.parse_args(argv)
    result, report = specialize_from_documents(
        Path(arguments.cubin).read_bytes(),
        json.loads(Path(arguments.inspection).read_text()),
        json.loads(Path(arguments.group).read_text()),
    )
    Path(arguments.output).write_bytes(result.cubin)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report == "-":
        print(rendered, end="")
    else:
        Path(arguments.report).write_text(rendered)


if __name__ == "__main__":
    main()
