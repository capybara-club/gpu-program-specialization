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
"""Optional compact transport for the compiler's JSON record stream.

``compact_records`` emits content-addressed ``pool_resource`` records before
references to them.  Constant arrays, raw RNG bank descriptors, and numeric
plans are interned separately.  A compact variant retains its original fields
and identity, replaces ``pools`` with ``numeric_plan_id``, and carries only its
``toggles``.  ``expand_records`` restores the original record stream exactly
(dictionary key order aside) and consumes the additional resource records.

The compactor remembers at most ``max_seen`` resource IDs, never prior payloads
or ASTs.  Eviction can produce repeated, identical resource declarations, which
readers must accept.  The reference expander caches resource payloads; very
large consumers can replace that cache with an on-disk content-addressed store.

Resource wire format::

    {"type": "pool_resource", "format": "odegrammar.compact.v1",
     "kind": "constant_bank" | "rng_bank" | "numeric_plan",
     "id": "<kind>_<sha256>", "data": <kind-specific payload>}

Constant-bank data is the numeric values array.  RNG-bank data is the original
bank request without ``count``; a plan's ``rng_bank_refs`` stores each required
count separately, preserving prefix-stable RNG addressing.  Numeric-plan data
contains ``pools`` without toggles, leaf axes, or configuration_count, plus
``rng_bank_refs``.  Bank constants replace ``values`` with ``values_ref``.
Leaf axes and the Cartesian configuration count reconstruct from toggles and
the ordered numeric axes.  This adapter accepts canonical PoolPlanner records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Iterator


COMPACT_FORMAT = "odegrammar.compact.v1"
RESOURCE_KINDS = {"constant_bank", "rng_bank", "numeric_plan"}


class StreamError(ValueError):
    """An invalid compact stream or a noncanonical input pool record."""


def _resource_id(kind: str, data) -> str:
    digest = hashlib.sha256((COMPACT_FORMAT + ":" + kind + ":").encode("ascii"))
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    try:
        # Avoid allocating a second full serialized copy of a large bank.
        for chunk in encoder.iterencode(data):
            digest.update(chunk.encode("ascii"))
    except (TypeError, ValueError) as exc:
        raise StreamError(f"resource {kind} must contain finite JSON data") from exc
    return kind + "_" + digest.hexdigest()


def _leaf_axes(toggles: dict) -> list[dict]:
    if not isinstance(toggles, dict):
        raise StreamError("variant toggles must be an object")
    if any(not isinstance(name, str) for name in toggles):
        raise StreamError("toggle names must be strings")
    result = []
    for name in sorted(toggles):
        group = toggles[name]
        if (not isinstance(group, list) or len(group) not in (2, 4)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in group)
                or len(set(group)) != len(group)):
            raise StreamError(f"toggle {name!r} requires 2 or 4 distinct state indices")
        result.append({"id": f"leaf:{name}", "kind": "leaf", "count": len(group),
                       "slots": [name], "states": list(group)})
    return result


def _configuration_count(axes: list[dict]) -> int:
    counts = []
    ids = set()
    for axis in axes:
        if not isinstance(axis, dict) or not isinstance(axis.get("id"), str):
            raise StreamError("pool axes require string IDs")
        if axis["id"] in ids:
            raise StreamError("pool axes must have distinct IDs")
        ids.add(axis["id"])
        count = axis.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise StreamError("pool axis counts must be positive integers")
        counts.append(count)
    return math.prod(counts)


def compact_records(records: Iterable[dict], *, max_seen: int = 100000) -> Iterator[dict]:
    """Compact a compiler record iterator, with bounded ID-only deduplication.

    Non-variant records pass through unchanged.  ``max_seen=0`` disables
    interning across references while retaining the same compact wire format;
    it is useful for consumers seeking a strict constant-memory transport.
    """
    if isinstance(max_seen, bool) or not isinstance(max_seen, int) or max_seen < 0:
        raise StreamError("max_seen must be a nonnegative integer")
    seen: OrderedDict[str, None] = OrderedDict()

    def resource(kind: str, data) -> tuple[str, dict | None]:
        ident = _resource_id(kind, data)
        if ident in seen:
            seen.move_to_end(ident)
            return ident, None
        if max_seen:
            seen[ident] = None
            if len(seen) > max_seen:
                seen.popitem(last=False)
        return ident, {"type": "pool_resource", "format": COMPACT_FORMAT,
                       "kind": kind, "id": ident, "data": data}

    for record in records:
        if not isinstance(record, dict):
            raise StreamError("stream records must be objects")
        if record.get("type") == "pool_resource":
            raise StreamError("input is already compact; expand it before recompacting")
        if record.get("type") != "variant":
            yield record
            continue
        pools = record.get("pools")
        if not isinstance(pools, dict):
            raise StreamError("uncompressed variant requires pools")
        if "numeric_plan_id" in record or "toggles" in record:
            raise StreamError("variant has reserved compact-stream fields")
        required = {"toggles", "pool_axes", "constants", "rng_bindings", "prelude",
                    "rng_bank_requests", "parameter_initials", "configuration_count"}
        if not required <= set(pools):
            raise StreamError(f"variant pools missing fields: {sorted(required - set(pools))}")
        toggles = pools["toggles"]
        leaf_axes = _leaf_axes(toggles)
        axes = pools["pool_axes"]
        if not isinstance(axes, list) or axes[:len(leaf_axes)] != leaf_axes:
            raise StreamError("noncanonical leaf axes cannot be compacted losslessly")
        numeric_axes = axes[len(leaf_axes):]
        if any(not isinstance(axis, dict) or axis.get("kind") == "leaf" for axis in numeric_axes):
            raise StreamError("leaf axes must precede numeric axes")
        expected_count = _configuration_count(axes)
        if isinstance(pools["configuration_count"], bool) or pools["configuration_count"] != expected_count:
            raise StreamError("configuration_count does not match Cartesian axes")

        excluded = {"toggles", "pool_axes", "configuration_count", "constants", "rng_bank_requests"}
        numeric_pools = {key: copy.deepcopy(value) for key, value in pools.items() if key not in excluded}
        numeric_pools["pool_axes"] = copy.deepcopy(numeric_axes)
        numeric_pools["constants"] = {}
        if not isinstance(pools["constants"], dict):
            raise StreamError("pool constants must be an object")
        for name, spec in pools["constants"].items():
            if not isinstance(spec, dict):
                raise StreamError("constant bindings must be objects")
            if "values_ref" in spec:
                raise StreamError("constant binding has reserved values_ref field")
            compact_spec = {key: copy.deepcopy(value) for key, value in spec.items() if key != "values"}
            if "values" in spec:
                if not isinstance(spec["values"], list):
                    raise StreamError("constant values must be an array")
                ident, declaration = resource("constant_bank", spec["values"])
                if declaration is not None:
                    yield declaration
                compact_spec["values_ref"] = ident
            numeric_pools["constants"][name] = compact_spec

        if not isinstance(pools["rng_bank_requests"], list):
            raise StreamError("rng_bank_requests must be an array")
        bank_refs = []
        for request in pools["rng_bank_requests"]:
            if not isinstance(request, dict) or "count" not in request:
                raise StreamError("RNG requests require count")
            count = request["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise StreamError("RNG bank count must be a positive integer")
            descriptor = {key: copy.deepcopy(value) for key, value in request.items() if key != "count"}
            ident, declaration = resource("rng_bank", descriptor)
            if declaration is not None:
                yield declaration
            bank_refs.append({"id": ident, "count": count})

        data = {"pools": numeric_pools, "rng_bank_refs": bank_refs}
        ident, declaration = resource("numeric_plan", data)
        if declaration is not None:
            yield declaration
        output = {key: value for key, value in record.items() if key != "pools"}
        output["numeric_plan_id"] = ident
        output["toggles"] = copy.deepcopy(toggles)
        yield output


def expand_records(records: Iterable[dict]) -> Iterator[dict]:
    """Restore compact records, checking resource identity and references.

    Both ordinary and compact variants are accepted.  Resource declarations
    are consumed, not returned.  This reference reader retains resources in
    memory and therefore is intended for inspection or bounded workloads.
    """
    resources: dict[str, tuple[str, object]] = {}

    def get(ident, kind: str):
        if not isinstance(ident, str) or ident not in resources:
            raise StreamError(f"missing {kind} resource {ident!r}")
        stored_kind, data = resources[ident]
        if stored_kind != kind:
            raise StreamError(f"resource {ident!r} has kind {stored_kind}, expected {kind}")
        return copy.deepcopy(data)

    for record in records:
        if not isinstance(record, dict):
            raise StreamError("stream records must be objects")
        if record.get("type") == "pool_resource":
            if record.get("format") != COMPACT_FORMAT:
                raise StreamError("unsupported compact resource format")
            kind = record.get("kind")
            if not isinstance(kind, str) or kind not in RESOURCE_KINDS or "data" not in record:
                raise StreamError("unknown or incomplete pool resource")
            ident = record.get("id")
            if ident != _resource_id(kind, record["data"]):
                raise StreamError("pool resource content does not match its ID")
            if ident not in resources:
                resources[ident] = (kind, copy.deepcopy(record["data"]))
            continue
        if record.get("type") != "variant" or "numeric_plan_id" not in record:
            yield record
            continue
        if "pools" in record:
            raise StreamError("variant cannot have both pools and numeric_plan_id")
        data = get(record["numeric_plan_id"], "numeric_plan")
        if not isinstance(data, dict) or not isinstance(data.get("pools"), dict) or not isinstance(data.get("rng_bank_refs"), list):
            raise StreamError("malformed numeric plan")
        pools = data["pools"]
        if not isinstance(pools.get("constants"), dict) or not isinstance(pools.get("pool_axes"), list):
            raise StreamError("malformed numeric pool bindings")
        for name, spec in pools["constants"].items():
            if not isinstance(spec, dict):
                raise StreamError("malformed constant binding")
            if "values_ref" in spec:
                if "values" in spec:
                    raise StreamError("constant cannot have values and values_ref")
                spec["values"] = get(spec.pop("values_ref"), "constant_bank")
        pools["rng_bank_requests"] = []
        for ref in data["rng_bank_refs"]:
            if (not isinstance(ref, dict) or isinstance(ref.get("count"), bool)
                    or not isinstance(ref.get("count"), int) or ref["count"] < 1):
                raise StreamError("malformed RNG bank reference")
            request = get(ref.get("id"), "rng_bank")
            if not isinstance(request, dict) or "count" in request:
                raise StreamError("malformed RNG bank resource")
            request["count"] = ref["count"]
            pools["rng_bank_requests"].append(request)
        toggles = record.get("toggles")
        leaf_axes = _leaf_axes(toggles)
        if any(not isinstance(axis, dict) or axis.get("kind") == "leaf" for axis in pools["pool_axes"]):
            raise StreamError("numeric plan cannot contain leaf axes")
        pools["toggles"] = copy.deepcopy(toggles)
        pools["pool_axes"] = leaf_axes + pools["pool_axes"]
        pools["configuration_count"] = _configuration_count(pools["pool_axes"])
        output = {key: value for key, value in record.items() if key not in ("numeric_plan_id", "toggles")}
        output["pools"] = pools
        yield output
