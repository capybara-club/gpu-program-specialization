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
"""Compare the C99 arena inspection with the Python physical-layout reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from odezza.inspection import inspect_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("c_test", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("cubin", type=Path)
    arguments = parser.parse_args()
    completed = subprocess.run(
        [str(arguments.c_test), str(arguments.cubin)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout)
    expected = inspect_module(arguments.source.read_text(), arguments.cubin.read_bytes())
    observed = expected.document["observed"]
    site = observed["site"]
    function = observed["function"]
    declared_shape = expected.manifest["shape"]
    comparisons = {
        "cubin_size": observed["cubin_byte_size"],
        "architecture": int(observed["architecture"].removeprefix("sm_")),
        "register_count": observed["register_count"],
        "input_count": declared_shape["input_count"],
        "output_count": declared_shape["output_count"],
        "system_capacity": observed["system_capacity"],
        "function_symbol_index": function["symbol_index"],
        "function_file_offset": function["file_offset"],
        "function_byte_size": function["byte_size"],
        "scaffold_start": site["scaffold_start_file_offset"],
        "scaffold_end": site["scaffold_end_file_offset"],
        "shared_start": site["shared_start_file_offset"],
        "shared_instruction_count": site["shared_instruction_count"],
        "arena_start": site["arena_start_file_offset"],
        "arena_instruction_count": site["arena_instruction_count"],
        "requested_arena_instruction_count": site["requested_arena_instruction_count"],
        "system_patch_capacity": declared_shape["system_patch_capacity"],
        "final_fallthrough_branch_elided": site["final_fallthrough_branch_elided"],
        "arena_end": site["arena_end_file_offset"],
        "continuation": site["continuation_file_offset"],
        "incoming_wait_mask": site["incoming_wait_mask"],
        "predicate_register": site["predicate_register"],
        "register_count_offsets": observed["register_count_file_offsets"],
        "register_count_header_offsets": observed["register_count_header_file_offsets"],
        "dispatch_offsets": site["dispatch_file_offsets"],
        "dispatch_instructions": [
            [f"{word0:016x}", f"{word1:016x}"]
            for word0, word1 in site["dispatch_instructions"]
        ],
        "input_registers": site["input_registers"],
        "output_registers": site["output_registers"],
        "final_output_registers": site["final_output_registers"],
        "output_materialization_offsets": site["output_materialization_file_offsets"],
        "available_registers": site["available_registers"],
        "cleanup_offsets": site["cleanup_file_offsets"],
        "target_table_offsets": site["target_table_file_offsets"],
        "original_target_values": site["original_target_values"],
        "template_id_file_offset": observed["template_id_symbol"]["file_offset"],
        "template_id": observed["template_id_symbol"]["value"],
    }
    mismatches = {
        key: {"c": actual.get(key), "python": value}
        for key, value in comparisons.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise SystemExit(json.dumps(mismatches, indent=2, sort_keys=True))
    print(
        "C/Python CUBIN inspection parity: exact; "
        f"arena {actual['arena_bytes']} bytes; "
        f"sm_{actual['architecture']}; {actual['register_count']} registers"
    )


if __name__ == "__main__":
    main()
