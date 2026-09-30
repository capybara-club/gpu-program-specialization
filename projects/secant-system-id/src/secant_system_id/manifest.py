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

import hashlib
import json
import re
from typing import Any, Sequence

from .model import SystemModel
from .shape import KernelShape, PackedDispatch, PackedKernelSpec


MANIFEST_SCHEMA = "secant-system-id.module-manifest"
MANIFEST_SCHEMA_VERSION = 3
GENERATOR_ABI_VERSION = 3
LEGACY_MANIFEST_SCHEMA_VERSION = 2
LEGACY_GENERATOR_ABI_VERSION = 2
MARKER_ABI_VERSION = 1
POSTORDER_ABI = "secant-c99-postorder-v1"
TEMPLATE_ID_SYMBOL = "ssid_template_id"
MANIFEST_BEGIN = "/* SECANT_SYSTEM_ID_MANIFEST_BEGIN"
MANIFEST_END = "SECANT_SYSTEM_ID_MANIFEST_END */"
TEMPLATE_ID_ALGORITHM = "sha256"
TEMPLATE_ID_BYTES = 32
ZERO_TEMPLATE_ID = bytes(TEMPLATE_ID_BYTES)
MAX_MANIFEST_BYTES = 1 << 20


class ManifestError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _template_words(template_id: bytes) -> tuple[int, ...]:
    if len(template_id) != TEMPLATE_ID_BYTES:
        raise ValueError(
            f"template ID must contain exactly {TEMPLATE_ID_BYTES} bytes"
        )
    return tuple(
        int.from_bytes(template_id[offset : offset + 4], "little")
        for offset in range(0, TEMPLATE_ID_BYTES, 4)
    )


def template_id_declaration(template_id: bytes) -> str:
    words = _template_words(template_id)
    values = ", ".join(f"0x{word:08x}u" for word in words)
    return (
        f'extern "C" __device__ __constant__ unsigned int '
        f"{TEMPLATE_ID_SYMBOL}[8] = {{{values}}};"
    )


def make_manifest(
    model: SystemModel,
    shape: KernelShape,
    kernel_name: str,
    topology: str,
    dispatch: PackedDispatch | None = None,
    work_ownership: str = "cta_serial",
) -> dict[str, Any]:
    if topology not in {"direct", "packed"}:
        raise ValueError("topology must be direct or packed")
    if (topology == "packed") != (dispatch is not None):
        raise ValueError("packed topology requires PackedDispatch and direct topology forbids it")
    if work_ownership not in {"cta_serial", "warp_per_system"}:
        raise ValueError("unknown kernel work ownership")
    if topology == "direct" and work_ownership != "cta_serial":
        raise ValueError("direct kernels only support CTA-serial ownership")
    missing_sites = []
    for index, (site, input_offset) in enumerate(
        zip(model.missing_sites, shape.ast_input_offsets)
    ):
        missing_sites.append(
            {
                "index": index,
                "name": site.name,
                "leaf_count": site.leaf_count,
                "input_offset": input_offset,
            }
        )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": LEGACY_MANIFEST_SCHEMA_VERSION,
        "generator_abi_version": LEGACY_GENERATOR_ABI_VERSION,
        "kernel": {
            "name": kernel_name,
            "topology": topology,
            "work_ownership": work_ownership,
        },
        "model": {
            "name": model.name,
            "state_names": list(model.state_names),
            "missing_sites": missing_sites,
            "observation_interval": model.observation_interval,
        },
        "shape": {
            "state_count": shape.state_count,
            "constant_count": shape.constant_count,
            "input_count": shape.input_count,
            "output_count": shape.ast_count,
            "trajectory_count": shape.trajectory_count,
            "observation_count": shape.observation_count,
            "patch_capacity_per_genome": shape.patch_capacity,
            "trajectory_layout": "dense_aligned",
        },
        "dispatch": None
        if dispatch is None
        else {
            "genome_capacity": dispatch.genome_capacity,
            "genomes_per_cta": dispatch.genomes_per_cta,
        },
        "abi": {
            "marker_version": MARKER_ABI_VERSION,
            "postorder": POSTORDER_ABI,
            "template_id_symbol": TEMPLATE_ID_SYMBOL,
        },
        "template_id_algorithm": TEMPLATE_ID_ALGORITHM,
    }
    return manifest


def make_packed_module_manifest(
    model: SystemModel,
    shape: KernelShape,
    kernels: Sequence[PackedKernelSpec],
    settings_mode: str = "materialized",
    constant_mutation_scale: float | None = None,
    binding_keep_probability: float | None = None,
    work_ownership: str = "cta_serial",
) -> dict[str, Any]:
    if not kernels:
        raise ValueError("a packed module must contain at least one kernel")
    if settings_mode not in {"materialized", "hashed_incumbent"}:
        raise ValueError("unknown packed-module settings mode")
    if work_ownership not in {"cta_serial", "warp_per_system"}:
        raise ValueError("unknown packed-module work ownership")
    names = [kernel.name for kernel in kernels]
    if len(set(names)) != len(names):
        raise ValueError("packed module kernel names must be unique")
    expected_base = 0
    kernel_records = []
    for index, kernel in enumerate(kernels):
        if kernel.genome_base != expected_base:
            raise ValueError("packed module kernel genome ranges must be contiguous")
        kernel_records.append(
            {
                "index": index,
                "name": kernel.name,
                "topology": "packed",
                "genome_base": kernel.genome_base,
                "dispatch": {
                    "genome_capacity": kernel.dispatch.genome_capacity,
                    "genomes_per_cta": kernel.dispatch.genomes_per_cta,
                },
            }
        )
        expected_base = kernel.genome_end

    missing_sites = []
    for index, (site, input_offset) in enumerate(
        zip(model.missing_sites, shape.ast_input_offsets)
    ):
        missing_sites.append(
            {
                "index": index,
                "name": site.name,
                "leaf_count": site.leaf_count,
                "input_offset": input_offset,
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_abi_version": GENERATOR_ABI_VERSION,
        "module": {
            "topology": "packed",
            "kernel_count": len(kernel_records),
            "genome_capacity": expected_base,
            "work_ownership": work_ownership,
        },
        "kernels": kernel_records,
        "model": {
            "name": model.name,
            "state_names": list(model.state_names),
            "missing_sites": missing_sites,
            "observation_interval": model.observation_interval,
        },
        "shape": {
            "state_count": shape.state_count,
            "constant_count": shape.constant_count,
            "input_count": shape.input_count,
            "output_count": shape.ast_count,
            "trajectory_count": shape.trajectory_count,
            "observation_count": shape.observation_count,
            "patch_capacity_per_genome": shape.patch_capacity,
            "trajectory_layout": "dense_aligned",
        },
        "settings": {
            "mode": settings_mode,
            "constant_mutation_scale": constant_mutation_scale,
            "binding_keep_probability": binding_keep_probability,
        },
        "abi": {
            "marker_version": MARKER_ABI_VERSION,
            "postorder": POSTORDER_ABI,
            "template_id_symbol": TEMPLATE_ID_SYMBOL,
        },
        "template_id_algorithm": TEMPLATE_ID_ALGORITHM,
    }


def finalize_cuda_source(body: str, manifest: dict[str, Any]) -> str:
    """Attach a self-verifying manifest and CUBIN-resident template ID."""

    if not body.endswith("\n"):
        body += "\n"
    normalized_payload = template_id_declaration(ZERO_TEMPLATE_ID) + "\n\n" + body
    template_id = hashlib.sha256(
        _canonical(manifest) + b"\0" + normalized_payload.encode("utf-8")
    ).digest()
    payload = template_id_declaration(template_id) + "\n\n" + body
    completed = dict(manifest)
    completed["template_id"] = template_id.hex()
    completed["source_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rendered_manifest = json.dumps(completed, indent=2, sort_keys=True)
    return f"{MANIFEST_BEGIN}\n{rendered_manifest}\n{MANIFEST_END}\n{payload}"


def parse_cuda_manifest(source: str) -> tuple[dict[str, Any], str]:
    if not source.startswith(MANIFEST_BEGIN + "\n"):
        raise ManifestError("CUDA source does not begin with a Secant System ID manifest")
    end = source.find("\n" + MANIFEST_END)
    if end < 0:
        raise ManifestError("CUDA source manifest is not terminated")
    raw_json = source[len(MANIFEST_BEGIN) + 1 : end]
    if len(raw_json.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ManifestError("CUDA source manifest exceeds the supported size")
    payload_start = end + 1 + len(MANIFEST_END)
    if payload_start >= len(source) or source[payload_start] != "\n":
        raise ManifestError("CUDA source manifest is not followed by a payload")
    payload = source[payload_start + 1 :]
    try:
        manifest = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"CUDA source manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("CUDA source manifest must be a JSON object")
    validate_manifest(manifest, payload)
    return manifest, payload


def validate_manifest(manifest: dict[str, Any], payload: str) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError("unsupported CUDA source manifest schema")
    schema_version = manifest.get("schema_version")
    if schema_version not in {LEGACY_MANIFEST_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION}:
        raise ManifestError("unsupported CUDA source manifest schema version")
    expected_generator = (
        LEGACY_GENERATOR_ABI_VERSION
        if schema_version == LEGACY_MANIFEST_SCHEMA_VERSION
        else GENERATOR_ABI_VERSION
    )
    if manifest.get("generator_abi_version") != expected_generator:
        raise ManifestError("unsupported CUDA generator ABI version")
    if manifest.get("template_id_algorithm") != TEMPLATE_ID_ALGORITHM:
        raise ManifestError("unsupported template-ID algorithm")
    required_objects = (
        ("kernel", "model", "shape", "abi")
        if schema_version == LEGACY_MANIFEST_SCHEMA_VERSION
        else ("module", "model", "shape", "abi")
    )
    for field in required_objects:
        if not isinstance(manifest.get(field), dict):
            raise ManifestError(f"manifest {field} must be a JSON object")
    if schema_version == LEGACY_MANIFEST_SCHEMA_VERSION:
        if manifest.get("dispatch") is not None and not isinstance(
            manifest.get("dispatch"), dict
        ):
            raise ManifestError("manifest dispatch must be null or a JSON object")
    elif not isinstance(manifest.get("kernels"), list) or not manifest["kernels"]:
        raise ManifestError("module manifest kernels must be a non-empty JSON array")
    if schema_version == MANIFEST_SCHEMA_VERSION:
        kernel_specs_from_manifest(manifest)
    template_hex = manifest.get("template_id")
    source_hash = manifest.get("source_sha256")
    if not isinstance(template_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", template_hex):
        raise ManifestError("manifest template_id must be a lowercase SHA-256 digest")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ManifestError("manifest source_sha256 must be lowercase hexadecimal")
    actual_source_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual_source_hash != source_hash:
        raise ManifestError("CUDA source payload hash does not match its manifest")

    template_id = bytes.fromhex(template_hex)
    declaration = template_id_declaration(template_id)
    if not payload.startswith(declaration + "\n\n"):
        raise ManifestError("CUDA source payload does not contain its declared template ID")
    normalized_payload = (
        template_id_declaration(ZERO_TEMPLATE_ID) + payload[len(declaration) :]
    )
    semantic_manifest = dict(manifest)
    del semantic_manifest["template_id"]
    del semantic_manifest["source_sha256"]
    expected_template_id = hashlib.sha256(
        _canonical(semantic_manifest) + b"\0" + normalized_payload.encode("utf-8")
    ).digest()
    if expected_template_id != template_id:
        raise ManifestError("CUDA source template ID does not match its manifest and payload")


def shape_from_manifest(manifest: dict[str, Any]) -> KernelShape:
    try:
        shape = manifest["shape"]
        sites = manifest["model"]["missing_sites"]
        result = KernelShape(
            ast_leaf_counts=tuple(int(site["leaf_count"]) for site in sites),
            state_count=int(shape["state_count"]),
            constant_count=int(shape["constant_count"]),
            trajectory_count=int(shape["trajectory_count"]),
            observation_count=int(shape["observation_count"]),
            patch_capacity=int(shape["patch_capacity_per_genome"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"manifest does not contain a valid kernel shape: {exc}") from exc
    if result.input_count != shape.get("input_count"):
        raise ManifestError("manifest input_count disagrees with its missing-site leaves")
    if result.ast_count != shape.get("output_count"):
        raise ManifestError("manifest output_count disagrees with its missing-site count")
    return result


def dispatch_from_manifest(manifest: dict[str, Any]) -> PackedDispatch | None:
    value = manifest.get("dispatch")
    if value is None:
        return None
    try:
        return PackedDispatch(
            genome_capacity=int(value["genome_capacity"]),
            genomes_per_cta=int(value["genomes_per_cta"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"manifest does not contain a valid packed dispatch: {exc}") from exc


def kernel_specs_from_manifest(manifest: dict[str, Any]) -> tuple[PackedKernelSpec, ...]:
    """Return the packed entry points declared by a version-3 module manifest."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("manifest does not describe a multi-kernel module")
    try:
        module = manifest["module"]
        records = manifest["kernels"]
        if module["topology"] != "packed":
            raise ValueError("only packed multi-kernel modules are supported")
        if module.get("work_ownership", "cta_serial") not in {
            "cta_serial",
            "warp_per_system",
        }:
            raise ValueError("module work_ownership is not supported")
        specs = tuple(
            PackedKernelSpec(
                name=record["name"],
                genome_base=int(record["genome_base"]),
                dispatch=PackedDispatch(
                    genome_capacity=int(record["dispatch"]["genome_capacity"]),
                    genomes_per_cta=int(record["dispatch"]["genomes_per_cta"]),
                ),
            )
            for record in records
        )
        if int(module["kernel_count"]) != len(specs):
            raise ValueError("module kernel_count disagrees with kernels")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("module kernel names are not unique")
        expected_base = 0
        for index, spec in enumerate(specs):
            if int(records[index]["index"]) != index:
                raise ValueError("module kernel indices are not contiguous")
            if spec.genome_base != expected_base:
                raise ValueError("module genome ranges are not contiguous")
            expected_base = spec.genome_end
        if int(module["genome_capacity"]) != expected_base:
            raise ValueError("module genome_capacity disagrees with kernels")
        return specs
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"manifest does not contain valid packed kernels: {exc}") from exc
