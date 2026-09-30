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
"""Identity-bound manifest for the Python trajectory-LM template."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .lm_model import (
    LM_TWO_SITE_LAYOUT,
    LM_KERNEL_NAME,
    LM_MANIFEST_SHAPE,
    LM_POSTORDER_ABI,
    LMKernelShape,
)
from .manifest import (
    MANIFEST_BEGIN,
    MANIFEST_END,
    MAX_MANIFEST_BYTES,
    TEMPLATE_ID_ALGORITHM,
    TEMPLATE_ID_SYMBOL,
    ZERO_TEMPLATE_ID,
    ManifestError,
    finalize_cuda_source,
    template_id_declaration,
)


LM_MANIFEST_SCHEMA = "odezza.lm-cuda-template"
LM_MANIFEST_SCHEMA_VERSION = 1
LM_GENERATOR_ABI_VERSION = 4


def make_lm_manifest(shape: LMKernelShape) -> dict[str, Any]:
    two_site = shape.site_layout == LM_TWO_SITE_LAYOUT
    return {
        "schema": LM_MANIFEST_SCHEMA,
        "schema_version": LM_MANIFEST_SCHEMA_VERSION,
        "generator_abi_version": LM_GENERATOR_ABI_VERSION,
        "kernel": {
            "name": LM_KERNEL_NAME,
            "shape": LM_MANIFEST_SHAPE,
            "ownership": (
                "one start and toggle permutation per thread"
                if shape.fit_thread_count == 1
                else f"one start and toggle permutation per {shape.fit_thread_count}-thread warp subgroup"
            ),
            "integrator": "resident fixed-step RK4 with coupled-stage forward sensitivities",
            "sensitivity_integrator": "simultaneous-stage-rk4-v2",
            "optimizer": (
                "thread-owned damped Levenberg-Marquardt"
                if shape.fit_thread_count == 1
                else "subgroup-owned damped Levenberg-Marquardt with parameter-column sensitivity partitioning"
            ),
        },
        "shape": shape.to_dict(),
        "configuration": {
            "ordering": "low fit bits choose the toggle permutation; high fit bits choose the start",
            "fit_mapping": (
                "one CUDA thread per fit"
                if shape.fit_thread_count == 1
                else "contiguous power-of-two warp subgroups own consecutive fits"
            ),
            "toggle_bit_positions": "specialized AST immediates",
            "permutation_word_bits": 32,
            "starts": "runtime start-major optimized constants",
        },
        "data": {
            "trajectory_layout": "runtime ragged irregular dense-state",
            "reference_layout": "reference[state,concatenated_point]",
            "reference_weight_layout": "reference_weights[state,concatenated_point]",
            "time_layout": "time[concatenated_point]",
            "offset_layout": "trajectory_offsets[trajectory+1]",
            "output_layout": {
                "constants": "constants[fit,optimized_constant]",
                "initial_mse": "initial_mse[fit]",
                "mse": "mse[fit]",
                "iterations": "iterations[fit]",
                "accepted_steps": "accepted_steps[fit]",
                "factorization_attempts": "factorization_attempts[fit]",
            },
        },
        "specialization": {
            "site_order": (
                "operand-bounded primal sites followed by output-major derivative sites evaluated independently by each fit lane"
                if two_site
                else "operand-bounded primal groups, then operand-bounded input-major local-partial groups"
            ),
            "derivative_backend": (
                "register-output symbolic derivative groups evaluated independently by each fit lane"
                if two_site
                else "symbolic forward partials with CSE inside each multi-output site"
            ),
            "site_count": shape.site_count,
            "site_patch_capacity": shape.site_patch_capacity,
            "candidate_systems_per_module": 1,
            "nonsmooth_policy": "reject ABS, MIN, and MAX",
        },
        "abi": {
            "marker_version": shape.marker_abi_version,
            "postorder": LM_POSTORDER_ABI,
            "sass": shape.specialization_abi,
            "layout": (
                "partitioned-primal-and-replicated-partials-v8"
                if two_site
                else "operand-bounded-multi-output-sites-v4"
            ),
            "input_register_order": (
                ["states", "optimized_constants", "permutation_u32"]
                if two_site
                else ["states", "optimized_constants", "permutation_u32"]
            ),
            "output_register_order": (
                "right-hand sides at primal sites; output-major local derivative groups at derivative sites"
                if two_site
                else "the declared scalar outputs for each grouped site"
            ),
            "template_id_symbol": TEMPLATE_ID_SYMBOL,
        },
        "template_id_algorithm": TEMPLATE_ID_ALGORITHM,
    }


def finalize_lm_cuda_source(body: str, shape: LMKernelShape) -> str:
    return finalize_cuda_source(body, make_lm_manifest(shape))


def parse_lm_cuda_manifest(source: str) -> tuple[dict[str, Any], str]:
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
    validate_lm_manifest(manifest, payload)
    return manifest, payload


def validate_lm_manifest(manifest: dict[str, Any], payload: str) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != LM_MANIFEST_SCHEMA:
        raise ManifestError("unsupported LM CUDA manifest schema")
    if manifest.get("schema_version") != LM_MANIFEST_SCHEMA_VERSION:
        raise ManifestError("unsupported LM CUDA manifest schema version")
    if manifest.get("generator_abi_version") != LM_GENERATOR_ABI_VERSION:
        raise ManifestError("unsupported LM CUDA generator ABI version")
    for field in ("kernel", "shape", "configuration", "data", "specialization", "abi"):
        if not isinstance(manifest.get(field), dict):
            raise ManifestError(f"LM manifest {field} must be a JSON object")
    if manifest["kernel"].get("name") != LM_KERNEL_NAME or manifest["kernel"].get("shape") != LM_MANIFEST_SHAPE:
        raise ManifestError("LM manifest names an unsupported kernel")
    try:
        shape = LMKernelShape.from_dict(manifest["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"LM manifest kernel shape is invalid: {exc}") from exc
    abi = manifest["abi"]
    if (
        abi.get("marker_version") != shape.marker_abi_version
        or abi.get("postorder") != LM_POSTORDER_ABI
        or abi.get("sass") != shape.specialization_abi
        or abi.get("layout")
        != (
            "partitioned-primal-and-replicated-partials-v8"
            if shape.site_layout == LM_TWO_SITE_LAYOUT
            else "operand-bounded-multi-output-sites-v4"
        )
        or abi.get("template_id_symbol") != TEMPLATE_ID_SYMBOL
    ):
        raise ManifestError("unsupported LM specialization ABI")
    if manifest.get("template_id_algorithm") != TEMPLATE_ID_ALGORITHM:
        raise ManifestError("unsupported LM template-ID algorithm")
    template_hex = manifest.get("template_id")
    source_hash = manifest.get("source_sha256")
    if not isinstance(template_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", template_hex):
        raise ManifestError("LM template_id must be a SHA-256 digest")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ManifestError("LM source_sha256 must be a SHA-256 digest")
    if hashlib.sha256(payload.encode()).hexdigest() != source_hash:
        raise ManifestError("LM CUDA payload hash does not match its manifest")
    declaration = template_id_declaration(bytes.fromhex(template_hex))
    if not payload.startswith(declaration + "\n\n"):
        raise ManifestError("LM CUDA payload does not contain its template identity")
    normalized = template_id_declaration(ZERO_TEMPLATE_ID) + payload[len(declaration) :]
    semantic = dict(manifest)
    del semantic["template_id"]
    del semantic["source_sha256"]
    canonical = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    expected = hashlib.sha256(canonical + b"\0" + normalized.encode()).hexdigest()
    if expected != template_hex:
        raise ManifestError("LM template identity does not match its manifest and payload")
    lm_shape_from_manifest(manifest)


def lm_shape_from_manifest(manifest: dict[str, Any]) -> LMKernelShape:
    if manifest.get("kernel", {}).get("shape") != LM_MANIFEST_SHAPE:
        raise ManifestError("manifest does not describe the trajectory-LM kernel")
    try:
        shape = LMKernelShape.from_dict(manifest["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"LM manifest kernel shape is invalid: {exc}") from exc
    specialization = manifest.get("specialization", {})
    if specialization.get("site_count") != shape.site_count:
        raise ManifestError("LM manifest site count disagrees with its shape")
    if specialization.get("site_patch_capacity") != shape.site_patch_capacity:
        raise ManifestError("LM manifest patch capacity disagrees with its shape")
    return shape
