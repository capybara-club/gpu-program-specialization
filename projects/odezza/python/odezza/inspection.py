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
"""Serializable Python inspection model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from .manifest import TEMPLATE_ID_SYMBOL, ManifestError, shape_from_manifest, parse_cuda_manifest
from .elf import CubinPlan, ElfImage, Function, SitePlan, inspect_cubin


INSPECTION_SCHEMA = "odezza.module-inspection"
INSPECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VerifiedInspection:
    manifest: dict[str, Any]
    plan: CubinPlan
    document: dict[str, Any]


def inspect_module(source: str, cubin: bytes) -> VerifiedInspection:
    manifest, _ = parse_cuda_manifest(source)
    shape = shape_from_manifest(manifest)
    source_template_id = bytes.fromhex(manifest["template_id"])
    elf = ElfImage(cubin)
    symbol = elf.find_data_symbol(TEMPLATE_ID_SYMBOL, len(source_template_id))
    cubin_template_id = cubin[symbol.file_offset : symbol.file_offset + symbol.size]
    if cubin_template_id != source_template_id:
        raise ManifestError("CUDA source and CUBIN template identities differ")
    plan = inspect_cubin(cubin, shape, manifest["kernel"]["name"])
    site = plan.site
    document = {
        "schema": INSPECTION_SCHEMA,
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "verified",
        "identity": {
            "template_id": manifest["template_id"],
            "source_sha256": manifest["source_sha256"],
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        },
        "declared": manifest,
        "observed": {
            "architecture": f"sm_{plan.architecture}",
            "cubin_byte_size": len(cubin),
            "register_count": plan.register_count,
            "register_count_file_offsets": list(plan.register_count_offsets),
            "register_count_header_file_offsets": list(plan.register_count_header_offsets),
            "system_capacity": plan.system_capacity,
            "function": {
                "symbol_index": plan.function.symbol_index,
                "file_offset": plan.function.file_offset,
                "byte_size": plan.function.size,
            },
            "site": {
                "scaffold_start_file_offset": site.scaffold_start_offset,
                "scaffold_end_file_offset": site.scaffold_end_offset,
                "shared_start_file_offset": site.shared_start_offset,
                "shared_instruction_count": site.shared_instruction_count,
                "dispatch_file_offsets": list(site.dispatch_offsets),
                "dispatch_instructions": [[word0, word1] for word0, word1 in site.dispatch_instructions],
                "arena_start_file_offset": site.arena_start_offset,
                "arena_instruction_count": site.arena_instruction_count,
                "requested_arena_instruction_count": shape.arena_instruction_count,
                "final_fallthrough_branch_elided": site.arena_instruction_count == shape.arena_instruction_count - 1,
                "arena_end_file_offset": site.arena_end_offset,
                "continuation_file_offset": site.continuation_offset,
                "incoming_wait_mask": site.incoming_wait_mask,
                "input_registers": list(site.input_registers),
                "output_registers": list(site.output_registers),
                "final_output_registers": list(site.final_output_registers),
                "output_materialization_file_offsets": list(site.output_materialization_offsets),
                "available_registers": list(site.available_registers),
                "predicate_register": site.predicate_register,
                "cleanup_file_offsets": list(site.cleanup_offsets),
                "target_table_file_offsets": list(site.target_table_offsets),
                "original_target_values": list(site.original_target_values),
                "input_register_order": manifest["abi"]["input_register_order"],
                "output_register_order": manifest["abi"]["output_register_order"],
            },
            "template_id_symbol": {
                "name": TEMPLATE_ID_SYMBOL,
                "file_offset": symbol.file_offset,
                "byte_size": symbol.size,
                "value": cubin_template_id.hex(),
            },
        },
        "verification": {
            "source_payload_hash": "match",
            "source_template_id": "match",
            "cubin_template_id": "match",
            "declared_shape": "match",
            "shared_prelude_layout": "match",
            "compact_system_dispatch": "match",
            "toggle_predicate_reservation": f"P{site.predicate_register}",
        },
    }
    return VerifiedInspection(manifest, plan, document)


def plan_from_inspection(
    document: dict[str, Any], cubin: bytes
) -> tuple[dict[str, Any], CubinPlan]:
    if document.get("schema") != INSPECTION_SCHEMA or document.get("schema_version") != INSPECTION_SCHEMA_VERSION:
        raise ManifestError("unsupported stored inspection document")
    if document.get("status") != "verified":
        raise ManifestError("stored inspection is not verified")
    if document.get("identity", {}).get("cubin_sha256") != hashlib.sha256(cubin).hexdigest():
        raise ManifestError("stored inspection is bound to a different CUBIN")
    manifest = document.get("declared")
    if not isinstance(manifest, dict):
        raise ManifestError("stored inspection does not contain its manifest")
    shape = shape_from_manifest(manifest)
    observed = document.get("observed")
    try:
        architecture = int(str(observed["architecture"]).removeprefix("sm_"))
        function_value = observed["function"]
        site_value = observed["site"]
        plan = CubinPlan(
            cubin_size=int(observed["cubin_byte_size"]),
            architecture=architecture,
            input_count=shape.input_count,
            output_count=shape.output_count,
            system_capacity=int(observed["system_capacity"]),
            register_count=int(observed["register_count"]),
            register_count_offsets=tuple(int(value) for value in observed["register_count_file_offsets"]),
            register_count_header_offsets=tuple(int(value) for value in observed["register_count_header_file_offsets"]),
            function=Function(
                int(function_value["symbol_index"]),
                int(function_value["file_offset"]),
                int(function_value["byte_size"]),
            ),
            site=SitePlan(
                scaffold_start_offset=int(site_value["scaffold_start_file_offset"]),
                scaffold_end_offset=int(site_value["scaffold_end_file_offset"]),
                shared_start_offset=int(site_value["shared_start_file_offset"]),
                shared_instruction_count=int(site_value["shared_instruction_count"]),
                dispatch_offsets=tuple(int(value) for value in site_value["dispatch_file_offsets"]),
                dispatch_instructions=tuple((int(value[0]), int(value[1])) for value in site_value["dispatch_instructions"]),
                arena_start_offset=int(site_value["arena_start_file_offset"]),
                arena_instruction_count=int(site_value["arena_instruction_count"]),
                arena_end_offset=int(site_value["arena_end_file_offset"]),
                continuation_offset=int(site_value["continuation_file_offset"]),
                incoming_wait_mask=int(site_value["incoming_wait_mask"]),
                input_registers=tuple(int(value) for value in site_value["input_registers"]),
                output_registers=tuple(int(value) for value in site_value["output_registers"]),
                final_output_registers=tuple(int(value) for value in site_value["final_output_registers"]),
                output_materialization_offsets=tuple(
                    int(value) for value in site_value["output_materialization_file_offsets"]
                ),
                available_registers=tuple(int(value) for value in site_value["available_registers"]),
                predicate_register=int(site_value["predicate_register"]),
                cleanup_offsets=tuple(int(value) for value in site_value["cleanup_file_offsets"]),
                target_table_offsets=tuple(int(value) for value in site_value["target_table_file_offsets"]),
                original_target_values=tuple(int(value) for value in site_value["original_target_values"]),
            ),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"stored inspection is malformed: {exc}") from exc
    if len(cubin) != plan.cubin_size:
        raise ManifestError("stored inspection CUBIN size is inconsistent")
    elf = ElfImage(cubin)
    if elf.architecture != plan.architecture:
        raise ManifestError("stored inspection architecture disagrees with its CUBIN")
    register_count, offsets, header_offsets = elf.register_counts(plan.function.symbol_index)
    if register_count != plan.register_count or set(offsets) != set(plan.register_count_offsets) or set(header_offsets) != set(plan.register_count_header_offsets):
        raise ManifestError("stored inspection register metadata disagrees with its CUBIN")
    return manifest, plan
