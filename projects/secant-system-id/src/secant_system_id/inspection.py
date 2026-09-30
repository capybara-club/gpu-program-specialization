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

from dataclasses import dataclass
import hashlib
from typing import Any

from .cubin import CubinPlan, ElfImage, Function, SitePlan, inspect_cubin
from .manifest import (
    LEGACY_MANIFEST_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    MARKER_ABI_VERSION,
    POSTORDER_ABI,
    TEMPLATE_ID_SYMBOL,
    ManifestError,
    dispatch_from_manifest,
    kernel_specs_from_manifest,
    parse_cuda_manifest,
    shape_from_manifest,
)
from .packed_cubin import (
    PackedCubinPlan,
    PackedKernelPlan,
    PackedModulePlan,
    PackedSitePlan,
    inspect_packed_cubin,
    inspect_packed_module_cubin,
)
from .shape import PackedDispatch, PackedKernelSpec


INSPECTION_SCHEMA = "secant-system-id.module-inspection"
INSPECTION_SCHEMA_VERSION = 3
LEGACY_INSPECTION_SCHEMA_VERSION = 1
PREVIOUS_INSPECTION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class VerifiedInspection:
    manifest: dict[str, Any]
    plan: CubinPlan | PackedCubinPlan | PackedModulePlan
    document: dict[str, Any]


def _function_document(plan: CubinPlan | PackedCubinPlan) -> dict[str, Any]:
    return {
        "symbol_index": plan.function.symbol_index,
        "file_offset": plan.function.file_offset,
        "byte_size": plan.function.size,
    }


def _direct_document(plan: CubinPlan) -> dict[str, Any]:
    return {
        "topology": "direct",
        "input_count": plan.input_count,
        "output_count": plan.output_count,
        "patch_capacity": plan.patch_capacity,
        "register_count": plan.register_count,
        "register_count_file_offsets": list(plan.register_count_offsets),
        "register_count_header_file_offsets": list(plan.register_count_header_offsets),
        "function": _function_document(plan),
        "site": {
            "load_fence_file_offset": plan.site.load_fence_offset,
            "start_file_offset": plan.site.start_offset,
            "instruction_count": plan.site.instruction_count,
            "end_file_offset": (
                plan.site.start_offset + 16 * plan.site.instruction_count
            ),
            "incoming_wait_mask": plan.site.incoming_wait_mask,
            "input_registers": list(plan.site.input_registers),
            "output_registers": list(plan.site.output_registers),
            "available_registers": list(plan.site.available_registers),
        },
    }


def _packed_document(plan: PackedCubinPlan) -> dict[str, Any]:
    return {
        "topology": "packed",
        "input_count": plan.input_count,
        "output_count": plan.output_count,
        "genome_capacity": plan.genome_capacity,
        "register_count": plan.register_count,
        "register_count_file_offsets": list(plan.register_count_offsets),
        "register_count_header_file_offsets": list(plan.register_count_header_offsets),
        "function": _function_document(plan),
        "site": {
            "entry_file_offset": plan.site.entry_offset,
            "dispatch_instruction_file_offsets": list(plan.site.dispatch_offsets),
            "dispatch_instruction_words": [
                [f"0x{word0:016x}", f"0x{word1:016x}"]
                for word0, word1 in plan.site.dispatch_instructions
            ],
            "arena_start_file_offset": plan.site.arena_start_offset,
            "arena_instruction_count": plan.site.arena_instruction_count,
            "arena_end_file_offset": plan.site.arena_end_offset,
            "continuation_file_offset": plan.site.continuation_offset,
            "incoming_wait_mask": plan.site.incoming_wait_mask,
            "input_registers": list(plan.site.input_registers),
            "output_registers": list(plan.site.output_registers),
            "available_registers": list(plan.site.available_registers),
            "cleanup_file_offsets": list(plan.site.cleanup_offsets),
            "target_table_file_offsets": list(plan.site.target_table_offsets),
            "original_target_values": list(plan.site.original_target_values),
        },
    }


def inspect_module(source: str, cubin: bytes) -> VerifiedInspection:
    """Verify a generated source/CUBIN pair and describe the physical module."""

    manifest, _payload = parse_cuda_manifest(source)
    try:
        abi = manifest["abi"]
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"manifest is missing its ABI declaration: {exc}") from exc
    if abi.get("marker_version") != MARKER_ABI_VERSION:
        raise ManifestError("manifest marker ABI is not supported")
    if abi.get("postorder") != POSTORDER_ABI:
        raise ManifestError("manifest post-order ABI is not supported")
    if abi.get("template_id_symbol") != TEMPLATE_ID_SYMBOL:
        raise ManifestError("manifest template-ID symbol is not supported")

    source_template_id = bytes.fromhex(manifest["template_id"])
    elf = ElfImage(cubin)
    symbol = elf.find_data_symbol(TEMPLATE_ID_SYMBOL, len(source_template_id))
    cubin_template_id = cubin[
        symbol.file_offset : symbol.file_offset + symbol.size
    ]
    if cubin_template_id != source_template_id:
        raise ManifestError(
            "CUDA source and CUBIN template IDs differ; the artifacts are not a pair"
        )

    shape = shape_from_manifest(manifest)
    if manifest["schema_version"] == MANIFEST_SCHEMA_VERSION:
        specs = kernel_specs_from_manifest(manifest)
        plan: CubinPlan | PackedCubinPlan | PackedModulePlan = inspect_packed_module_cubin(
            cubin, shape, specs
        )
        observed = {
            "topology": "packed_module",
            "kernel_count": len(plan.kernels),
            "genome_capacity": plan.genome_capacity,
            "kernels": [
                {
                    "name": kernel.spec.name,
                    "genome_base": kernel.spec.genome_base,
                    "genomes_per_cta": kernel.spec.dispatch.genomes_per_cta,
                    **_packed_document(kernel.cubin),
                }
                for kernel in plan.kernels
            ],
        }
    else:
        if manifest["schema_version"] != LEGACY_MANIFEST_SCHEMA_VERSION:
            raise ManifestError("unsupported manifest schema version")
        try:
            kernel = manifest["kernel"]
            topology = kernel["topology"]
            kernel_name = kernel["name"]
        except (KeyError, TypeError) as exc:
            raise ManifestError(f"manifest is missing its kernel declaration: {exc}") from exc
        if topology not in {"direct", "packed"}:
            raise ManifestError(f"unsupported kernel topology {topology!r}")
        if not isinstance(kernel_name, str) or not kernel_name:
            raise ManifestError("manifest kernel name must be non-empty")
        dispatch = dispatch_from_manifest(manifest)
        if topology == "direct":
            if dispatch is not None:
                raise ManifestError("direct kernel manifest unexpectedly declares packed dispatch")
            plan = inspect_cubin(cubin, shape, kernel_name)
            observed = _direct_document(plan)
        else:
            if dispatch is None:
                raise ManifestError("packed kernel manifest does not declare packed dispatch")
            plan = inspect_packed_cubin(cubin, shape, dispatch, kernel_name)
            observed = _packed_document(plan)

    document: dict[str, Any] = {
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
            "template_id_symbol": {
                "name": TEMPLATE_ID_SYMBOL,
                "file_offset": symbol.file_offset,
                "byte_size": symbol.size,
                "value": cubin_template_id.hex(),
            },
            **observed,
        },
        "verification": {
            "source_payload_hash": "match",
            "source_template_id": "match",
            "cubin_template_id": "match",
            "declared_shape": "match",
            "declared_dispatch": "match",
        },
    }
    return VerifiedInspection(manifest, plan, document)


def _kernel_plan_from_observed(
    observed: dict[str, Any],
    cubin: bytes,
    architecture: int,
) -> CubinPlan | PackedCubinPlan:
    function_value = observed["function"]
    function = Function(
        symbol_index=int(function_value["symbol_index"]),
        file_offset=int(function_value["file_offset"]),
        size=int(function_value["byte_size"]),
    )
    topology = observed["topology"]
    site = observed["site"]
    observed_register_count = int(observed["register_count"])
    observed_register_offsets = tuple(
        int(value) for value in observed["register_count_file_offsets"]
    )
    (
        cubin_register_count,
        cubin_register_offsets,
        cubin_register_header_offsets,
    ) = ElfImage(cubin).register_counts(function.symbol_index)
    if cubin_register_count != observed_register_count:
        raise ManifestError("stored inspection register count disagrees with its CUBIN")
    if set(cubin_register_offsets) != set(observed_register_offsets):
        raise ManifestError("stored inspection register-count offsets disagree with its CUBIN")
    if "register_count_header_file_offsets" in observed:
        observed_header_offsets = tuple(
            int(value) for value in observed["register_count_header_file_offsets"]
        )
        if set(cubin_register_header_offsets) != set(observed_header_offsets):
            raise ManifestError(
                "stored inspection register-count header offsets disagree with its CUBIN"
            )
    else:
        observed_header_offsets = cubin_register_header_offsets
    common = {
        "cubin_size": len(cubin),
        "architecture": architecture,
        "input_count": int(observed["input_count"]),
        "output_count": int(observed["output_count"]),
        "register_count": observed_register_count,
        "register_count_offsets": observed_register_offsets,
        "register_count_header_offsets": observed_header_offsets,
        "function": function,
    }
    if topology == "direct":
        return CubinPlan(
            patch_capacity=int(observed["patch_capacity"]),
            site=SitePlan(
                load_fence_offset=int(site["load_fence_file_offset"]),
                start_offset=int(site["start_file_offset"]),
                instruction_count=int(site["instruction_count"]),
                incoming_wait_mask=int(site["incoming_wait_mask"]),
                input_registers=tuple(int(value) for value in site["input_registers"]),
                output_registers=tuple(int(value) for value in site["output_registers"]),
                available_registers=tuple(int(value) for value in site["available_registers"]),
            ),
            **common,
        )
    if topology == "packed":
        return PackedCubinPlan(
            genome_capacity=int(observed["genome_capacity"]),
            site=PackedSitePlan(
                entry_offset=int(site["entry_file_offset"]),
                dispatch_offsets=tuple(int(value) for value in site["dispatch_instruction_file_offsets"]),
                dispatch_instructions=tuple(
                    (int(words[0], 16), int(words[1], 16))
                    for words in site["dispatch_instruction_words"]
                ),
                arena_start_offset=int(site["arena_start_file_offset"]),
                arena_instruction_count=int(site["arena_instruction_count"]),
                arena_end_offset=int(site["arena_end_file_offset"]),
                continuation_offset=int(site["continuation_file_offset"]),
                incoming_wait_mask=int(site["incoming_wait_mask"]),
                input_registers=tuple(int(value) for value in site["input_registers"]),
                output_registers=tuple(int(value) for value in site["output_registers"]),
                available_registers=tuple(int(value) for value in site["available_registers"]),
                cleanup_offsets=tuple(int(value) for value in site["cleanup_file_offsets"]),
                target_table_offsets=tuple(int(value) for value in site["target_table_file_offsets"]),
                original_target_values=tuple(int(value) for value in site["original_target_values"]),
            ),
            **common,
        )
    raise ManifestError(f"stored inspection has unknown kernel topology {topology!r}")


def plan_from_inspection(
    document: dict[str, Any],
    cubin: bytes,
) -> CubinPlan | PackedCubinPlan | PackedModulePlan:
    """Reconstruct a plan after binding stored inspection JSON to its CUBIN."""

    if document.get("schema") != INSPECTION_SCHEMA:
        raise ManifestError("unsupported stored inspection schema")
    if document.get("schema_version") not in {
        LEGACY_INSPECTION_SCHEMA_VERSION,
        PREVIOUS_INSPECTION_SCHEMA_VERSION,
        INSPECTION_SCHEMA_VERSION,
    }:
        raise ManifestError("unsupported stored inspection schema version")
    if document.get("status") != "verified":
        raise ManifestError("stored inspection is not marked verified")
    try:
        expected_cubin_hash = document["identity"]["cubin_sha256"]
        observed = document["observed"]
        expected_cubin_size = int(observed["cubin_byte_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"stored inspection identity is malformed: {exc}") from exc
    if len(cubin) != expected_cubin_size:
        raise ManifestError("stored inspection CUBIN size does not match the supplied CUBIN")
    if hashlib.sha256(cubin).hexdigest() != expected_cubin_hash:
        raise ManifestError("stored inspection CUBIN hash does not match the supplied CUBIN")
    try:
        architecture_text = observed["architecture"]
        if not isinstance(architecture_text, str) or not architecture_text.startswith("sm_"):
            raise ValueError("architecture is not formatted as sm_NN")
        architecture = int(architecture_text[3:])
        if observed["topology"] != "packed_module":
            return _kernel_plan_from_observed(observed, cubin, architecture)

        kernels = []
        for kernel_value in observed["kernels"]:
            spec = PackedKernelSpec(
                name=kernel_value["name"],
                genome_base=int(kernel_value["genome_base"]),
                dispatch=PackedDispatch(
                    genome_capacity=int(kernel_value["genome_capacity"]),
                    genomes_per_cta=int(kernel_value["genomes_per_cta"]),
                ),
            )
            kernel_plan = _kernel_plan_from_observed(
                kernel_value, cubin, architecture
            )
            if not isinstance(kernel_plan, PackedCubinPlan):
                raise ManifestError("packed module contains a non-packed kernel plan")
            kernels.append(PackedKernelPlan(spec, kernel_plan))
        plan = PackedModulePlan(expected_cubin_size, architecture, tuple(kernels))
        if int(observed["kernel_count"]) != len(plan.kernels):
            raise ValueError("stored module kernel_count disagrees with kernels")
        if int(observed["genome_capacity"]) != plan.genome_capacity:
            raise ValueError("stored module genome_capacity disagrees with kernels")
        return plan
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"stored inspection document is malformed: {exc}") from exc
