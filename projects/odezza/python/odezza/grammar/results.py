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
"""Compact report views and manifest-bound retained-candidate reconstruction."""
from copy import deepcopy
import re

from .lowering import CompatibilityError, Layout, identities

ADDRESS_VERSION = "odezza.grammar-address.v1"


def decimal_index(value, bits):
    # Strings preserve uint64 addresses through JavaScript/MCP clients.
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("Indices must be canonical nonnegative decimal strings")
    if len(value) > 20 or int(value) >= 1 << bits:
        raise ValueError(f"Index exceeds uint{bits} range")
    return int(value)


def origin(plan, manifest, candidate):
    if "addressing" not in manifest:
        raise CompatibilityError("This older job has no ordinal manifest; use the full report and candidate_id replay")
    return dict(version=ADDRESS_VERSION, manifest_id=manifest["manifest_id"],
                variant_index=str(plan.variant_index(candidate["variant_id"])),
                configuration_index=str(candidate["configuration_index"]))


def compact_report(report, manifest, job_id, plan):
    result = {key: deepcopy(value) for key, value in report.items() if key != "winners"}
    result.update(job_id=job_id, result_format="odezza.grammar-results.compact.v1",
                  ranking=deepcopy(manifest["retention"]), leaderboards={}, candidates={})

    def references(rows):
        ids = []
        for row in rows:
            identifier = row["id"]
            ids.append(identifier)
            if identifier not in result["candidates"]:
                candidate = {key: deepcopy(row[key]) for key in
                    ("id", "mse", "structure_id", "numeric_id", "equations", "named_values", "families", "tags") if key in row}
                candidate.update(phase="score", origin=origin(plan, manifest, row))
                result["candidates"][identifier] = candidate
        return ids

    winners = report.get("winners", {})
    if "global" in winners:
        result["leaderboards"]["global"] = references(winners["global"])
    for kind in ("families", "tags"):
        if kind in winners:
            result["leaderboards"][kind] = {name: references(rows) for name, rows in winners[kind].items()}
    return result


def reconstruct(plan, manifest, address):
    if not isinstance(address, dict) or set(address) != {"version", "manifest_id", "variant_index", "configuration_index"}:
        raise ValueError("Address requires version, manifest_id, variant_index and configuration_index")
    if address["version"] != ADDRESS_VERSION or "addressing" not in manifest:
        raise CompatibilityError("Unsupported or unavailable address manifest")
    if address["manifest_id"] != manifest.get("manifest_id"):
        raise ValueError("Address belongs to a different manifest")
    ordinal = decimal_index(address["variant_index"], 32)
    index = decimal_index(address["configuration_index"], 64)
    variant = plan.variant_at(ordinal)
    layout = Layout.from_pools(variant["pools"])
    native = layout.native_index(index)
    bank, permutation = divmod(native, layout.permutation_count)
    # Existing index narrows to one structural binding, with at most retention-k
    # snapshots. No scan or regeneration of earlier ASTs or RNG values is needed.
    candidate = plan.candidate_at(variant["id"], permutation, index)
    if candidate["bank_index"] != bank or candidate["slots"] != [list(pair) for pair in layout.slots]:
        raise ValueError("Retained snapshot does not match its configuration address")
    if len(candidate["values"]) != len(layout.slots):
        raise ValueError("Retained coefficient snapshot has the wrong size")
    skeleton = plan.skeleton(variant["skeleton_id"])
    numeric_id, structure_id, programs = identities(skeleton, variant, manifest["states"], layout,
                                                    candidate["values"], permutation)
    if (numeric_id, structure_id, programs) != (candidate["numeric_id"], candidate["structure_id"], candidate["resolved_programs"]):
        raise ValueError("Reconstructed RHS differs from the retained snapshot")
    return dict(candidate, resolved_programs=programs, origin=origin(plan, manifest, candidate))
