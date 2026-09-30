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
"""CUDA-template manifest and identity reference."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .model import KERNEL_NAME, KernelShape


MANIFEST_SCHEMA = "odezza.cuda-template"
MANIFEST_SCHEMA_VERSION = 2
GENERATOR_ABI_VERSION = 3
MARKER_ABI_VERSION = 2
POSTORDER_ABI = "odezza-postorder-f32-v2"
SASS_ABI = "odezza-sm89-sm90-sm120-v2"
TEMPLATE_ID_SYMBOL = "odezza_template_id"
TEMPLATE_ID_BYTES = 32
TEMPLATE_ID_ALGORITHM = "sha256"
ZERO_TEMPLATE_ID = bytes(TEMPLATE_ID_BYTES)
MANIFEST_BEGIN = "/* ODEZZA_MANIFEST_BEGIN"
MANIFEST_END = "ODEZZA_MANIFEST_END */"
MAX_MANIFEST_BYTES = 1 << 20


class ManifestError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _template_words(template_id: bytes) -> tuple[int, ...]:
    if len(template_id) != TEMPLATE_ID_BYTES:
        raise ValueError(f"template ID must contain {TEMPLATE_ID_BYTES} bytes")
    return tuple(int.from_bytes(template_id[offset : offset + 4], "little") for offset in range(0, TEMPLATE_ID_BYTES, 4))


def template_id_declaration(template_id: bytes) -> str:
    values = ", ".join(f"0x{word:08x}u" for word in _template_words(template_id))
    return f'extern "C" __device__ __constant__ unsigned int {TEMPLATE_ID_SYMBOL}[8] = {{{values}}};'


def make_manifest(shape: KernelShape) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_abi_version": GENERATOR_ABI_VERSION,
        "kernel": {
            "name": KERNEL_NAME,
            "shape": "scoring",
            "ownership": "one candidate system per grid-y coordinate and one configuration per thread",
            "integrator": "rk4",
            "score": "dense aligned mean squared trajectory error",
        },
        "shape": shape.to_dict(),
        "configuration": {
            "toggle_permutations": "runtime power of two",
            "toggle_bit_positions": "specialized AST immediates",
            "permutation_word_bits": 32,
            "constant_banks": "runtime system-major",
            "ordering": "grid-y selects system; specialized masks read low configuration bits; runtime active-toggle count shifts to the constant bank",
        },
        "data": {
            "trajectory_layout": "dense_aligned_shared_across_systems",
            "reference_layout": ["initial[state,trajectory]", "target[state,trajectory,observation]"],
            "constant_layout": "constant[system,bank,index]",
            "output_layout": "mse[system,configuration]",
        },
        "specialization": {
            "shared": "common derivative programs execute before uniform system dispatch",
            "branches": "one complete complementary derivative program set per candidate system",
            "system_capacity": shape.system_capacity,
            "shared_patch_capacity": shape.shared_patch_capacity,
            "system_patch_capacity": shape.system_patch_capacity,
        },
        "abi": {
            "marker_version": MARKER_ABI_VERSION,
            "postorder": POSTORDER_ABI,
            "sass": SASS_ABI,
            "layout": "shared-prelude-then-compact-system-dispatch-v1",
            "input_register_order": ["states", "system_constants", "permutation_u32"],
            "output_register_order": "rhs_by_state",
            "template_id_symbol": TEMPLATE_ID_SYMBOL,
        },
        "template_id_algorithm": TEMPLATE_ID_ALGORITHM,
    }


def finalize_cuda_source(body: str, manifest: dict[str, Any]) -> str:
    if not body.endswith("\n"):
        body += "\n"
    normalized_payload = template_id_declaration(ZERO_TEMPLATE_ID) + "\n\n" + body
    template_id = hashlib.sha256(_canonical(manifest) + b"\0" + normalized_payload.encode()).digest()
    payload = template_id_declaration(template_id) + "\n\n" + body
    completed = dict(manifest)
    completed["template_id"] = template_id.hex()
    completed["source_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    rendered = json.dumps(completed, indent=2, sort_keys=True)
    return f"{MANIFEST_BEGIN}\n{rendered}\n{MANIFEST_END}\n{payload}"


def parse_cuda_manifest(source: str) -> tuple[dict[str, Any], str]:
    if not source.startswith(MANIFEST_BEGIN + "\n"):
        raise ManifestError("CUDA source does not begin with an Odezza manifest")
    end = source.find("\n" + MANIFEST_END)
    if end < 0:
        raise ManifestError("CUDA source manifest is not terminated")
    raw = source[len(MANIFEST_BEGIN) + 1 : end]
    if len(raw.encode()) > MAX_MANIFEST_BYTES:
        raise ManifestError("CUDA source manifest is too large")
    payload_start = end + 1 + len(MANIFEST_END)
    if payload_start >= len(source) or source[payload_start] != "\n":
        raise ManifestError("CUDA source manifest is not followed by a payload")
    payload = source[payload_start + 1 :]
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"CUDA manifest is invalid JSON: {exc}") from exc
    validate_manifest(manifest, payload)
    return manifest, payload


def validate_manifest(manifest: dict[str, Any], payload: str) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError("unsupported CUDA manifest schema")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("unsupported CUDA manifest schema version")
    if manifest.get("generator_abi_version") != GENERATOR_ABI_VERSION:
        raise ManifestError("unsupported CUDA generator ABI version")
    for field in ("kernel", "shape", "configuration", "data", "abi"):
        if not isinstance(manifest.get(field), dict):
            raise ManifestError(f"manifest {field} must be a JSON object")
    abi = manifest["abi"]
    if abi.get("marker_version") != MARKER_ABI_VERSION or abi.get("postorder") != POSTORDER_ABI or abi.get("sass") != SASS_ABI:
        raise ManifestError("unsupported specialization ABI")
    if abi.get("template_id_symbol") != TEMPLATE_ID_SYMBOL:
        raise ManifestError("unsupported template-ID symbol")
    if manifest.get("template_id_algorithm") != TEMPLATE_ID_ALGORITHM:
        raise ManifestError("unsupported template-ID algorithm")
    template_hex = manifest.get("template_id")
    source_hash = manifest.get("source_sha256")
    if not isinstance(template_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", template_hex):
        raise ManifestError("manifest template_id must be a SHA-256 digest")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ManifestError("manifest source_sha256 must be a SHA-256 digest")
    if hashlib.sha256(payload.encode()).hexdigest() != source_hash:
        raise ManifestError("CUDA payload hash does not match the manifest")
    declaration = template_id_declaration(bytes.fromhex(template_hex))
    if not payload.startswith(declaration + "\n\n"):
        raise ManifestError("CUDA payload does not contain its template identity")
    normalized_payload = template_id_declaration(ZERO_TEMPLATE_ID) + payload[len(declaration) :]
    semantic = dict(manifest)
    del semantic["template_id"]
    del semantic["source_sha256"]
    expected = hashlib.sha256(_canonical(semantic) + b"\0" + normalized_payload.encode()).hexdigest()
    if expected != template_hex:
        raise ManifestError("template identity does not match the manifest and CUDA payload")
    kernel_shape = manifest["kernel"].get("shape")
    if kernel_shape != "scoring":
        raise ManifestError(f"unsupported CUDA kernel shape {kernel_shape!r}")
    shape = shape_from_manifest(manifest)
    if manifest["configuration"].get("toggle_permutations") != "runtime power of two" or manifest["configuration"].get("permutation_word_bits") != 32:
        raise ManifestError("configuration metadata disagrees with the runtime toggle ABI")


def shape_from_manifest(manifest: dict[str, Any]) -> KernelShape:
    if manifest.get("kernel", {}).get("shape") != "scoring":
        raise ManifestError("manifest does not describe the scoring kernel")
    try:
        shape = KernelShape.from_dict(manifest["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"manifest kernel shape is invalid: {exc}") from exc
    declared = manifest["shape"]
    for key, actual in (
        ("state_count", shape.state_count),
        ("input_count", shape.input_count),
        ("output_count", shape.output_count),
        ("reference_float_count", shape.reference_float_count),
        ("arena_instruction_count", shape.arena_instruction_count),
    ):
        if declared.get(key) != actual:
            raise ManifestError(f"manifest {key} disagrees with its base dimensions")
    specialization = manifest.get("specialization")
    if not isinstance(specialization, dict):
        raise ManifestError("manifest specialization must be a JSON object")
    for key, actual in (
        ("system_capacity", shape.system_capacity),
        ("shared_patch_capacity", shape.shared_patch_capacity),
        ("system_patch_capacity", shape.system_patch_capacity),
    ):
        if specialization.get(key) != actual:
            raise ManifestError(f"manifest {key} disagrees with its kernel shape")
    if manifest.get("abi", {}).get("layout") != "shared-prelude-then-compact-system-dispatch-v1":
        raise ManifestError("unsupported specialization layout")
    return shape
