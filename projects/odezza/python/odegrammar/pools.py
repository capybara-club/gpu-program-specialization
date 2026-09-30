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
"""Lazy configuration planning for lowered ODE expression programs.

This module does not run an ODE or generate GPU data.  RNG descriptors specify
stable, counter-addressed *reference* banks (SHA-256, not Philox).  An execution
backend can replace that generator with a versioned, documented implementation.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
from collections.abc import Iterator, Mapping
from typing import Any


class PoolError(ValueError):
    """An invalid constant, state selection, or random-bank declaration."""


def _object(value: Any, context: str) -> dict:
    if not isinstance(value, dict):
        raise PoolError(f"{context} must be an object")
    return value


def _keys(value: dict, allowed: set[str], context: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise PoolError(f"{context}: unknown fields {sorted(extra)}")


def _number(value: Any, context: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoolError(f"{context} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise PoolError(f"{context} must be a finite number")
    return value


def _integer(value: Any, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PoolError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise PoolError(f"{context} must be >= {minimum}")
    return value


def _name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PoolError(f"{context} must be a nonempty string")
    return value


def _numbers(value: Any, context: str) -> list[float | int]:
    if not isinstance(value, list) or not value:
        raise PoolError(f"{context} must be a nonempty list of finite numbers")
    for item in value:
        _number(item, context)
    # Declarations already belong to the planner's defensive deep copy.  Do
    # not duplicate a potentially large numeric bank merely to validate it.
    return value


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf8")).hexdigest()


def _unrank_combination(n: int, k: int, rank: int) -> tuple[int, ...]:
    """Unrank the lexicographic order used by itertools.combinations."""
    result = []
    start = 0
    for remaining in range(k, 0, -1):
        # Binary search skips potentially enormous candidate-state domains.
        total = math.comb(n - start, remaining)
        lo, hi = start, n - remaining
        while lo < hi:
            mid = (lo + hi + 1) // 2
            preceding = total - math.comb(n - mid, remaining)
            if preceding <= rank:
                lo = mid
            else:
                hi = mid - 1
        chosen = lo
        rank -= total - math.comb(n - chosen, remaining)
        result.append(chosen)
        start = chosen + 1
    return tuple(result)


def _sample_ranks(total: int, count: int, seed: int) -> Iterator[int]:
    """Partial Fisher-Yates without materializing range(total)."""
    generator = random.Random(seed)
    swaps: dict[int, int] = {}
    for i in range(count):
        j = generator.randrange(i, total)
        selected = swaps.get(j, j)
        swaps[j] = swaps.get(i, i)
        swaps.pop(i, None)
        yield selected


class PoolPlanner:
    """Validate declarations and lazily produce binary/quad toggle variants.

    Distinct named leaf slots are independent Cartesian axes.  Repeated
    references to the same slot share one choice.  Each yielded variant has
    exactly one 2- or 4-state group per active leaf slot.  Exhaustive groups can
    overlap: ``configuration_count`` counts visits, including those repeats.
    """

    def __init__(self, states, leaves=None, constants=None, constant_banks=None,
                 rng=None, rng_banks=None, parameters=None, seed=0):
        if not isinstance(states, list) or not states:
            raise PoolError("states must be a nonempty list")
        self.states = [_name(v, "state") for v in states]
        if len(set(self.states)) != len(self.states):
            raise PoolError("state names must be distinct")
        self.state_indices = {name: i for i, name in enumerate(self.states)}
        self.seed = _integer(seed, "seed")
        self.leaves = copy.deepcopy(_object(leaves if leaves is not None else {}, "leaves"))
        self.constants = copy.deepcopy(_object(constants if constants is not None else {}, "constants"))
        self.constant_banks = copy.deepcopy(_object(constant_banks if constant_banks is not None else {}, "constant_banks"))
        self.rng = copy.deepcopy(_object(rng if rng is not None else {}, "rng"))
        self.rng_banks = copy.deepcopy(_object(rng_banks if rng_banks is not None else {}, "rng_banks"))
        self.parameters = copy.deepcopy(_object(parameters if parameters is not None else {}, "parameters"))
        self.leaf_arities: dict[str, int] = {}
        self._leaf_specs: dict[str, dict] = {}
        self._constant_specs: dict[str, dict] = {}
        self._rng_specs: dict[str, dict] = {}
        self._bank_specs: dict[str, dict] = {}
        self._parameter_specs: dict[str, float | int] = {}
        self.validate()

    def validate(self) -> None:
        """Validate all declarations, even those absent from a generated RHS."""
        self.leaf_arities.clear()
        self._leaf_specs.clear()
        self._constant_specs.clear()
        self._rng_specs.clear()
        self._bank_specs.clear()
        self._parameter_specs.clear()
        for name, values in self.constant_banks.items():
            _name(name, "constant bank name")
            _numbers(values, f"constant_banks.{name}")
        for name, spec in self.constants.items():
            _name(name, "constant slot name")
            context = f"constants.{name}"
            _object(spec, context)
            _keys(spec, {"value", "values", "bank"}, context)
            if len(spec) != 1:
                raise PoolError(f"{context} requires exactly one of value, values, bank")
            if "value" in spec:
                self._constant_specs[name] = {"kind": "fixed", "value": _number(spec["value"], context)}
            elif "values" in spec:
                self._constant_specs[name] = {"kind": "bank", "axis": f"const:{name}",
                                              "values": _numbers(spec["values"], context)}
            else:
                bank = _name(spec["bank"], context)
                if bank not in self.constant_banks:
                    raise PoolError(f"{context}: unknown constant bank {bank!r}")
                self._constant_specs[name] = {"kind": "bank", "axis": f"const:{name}", "bank": bank,
                                              "values": self.constant_banks[bank]}

        for name, spec in self.leaves.items():
            _name(name, "leaf slot name")
            context = f"leaves.{name}"
            _object(spec, context)
            _keys(spec, {"states", "arity", "coverage", "samples", "seed", "groups"}, context)
            arity = spec.get("arity", 2)
            if isinstance(arity, bool) or not isinstance(arity, int) or arity not in (2, 4):
                raise PoolError(f"{context}.arity must be 2 or 4")
            candidate_names = spec.get("states")
            if not isinstance(candidate_names, list) or any(not isinstance(s, str) for s in candidate_names):
                raise PoolError(f"{context}.states must be a list of declared state names")
            if len(candidate_names) != len(set(candidate_names)):
                raise PoolError(f"{context}.states must be distinct")
            unknown = set(candidate_names) - set(self.states)
            if unknown:
                raise PoolError(f"{context}: unknown states {sorted(unknown)}")
            if len(candidate_names) < arity:
                raise PoolError(f"{context}: {arity}-way toggle needs at least {arity} distinct states")
            candidates = sorted(self.state_indices[s] for s in candidate_names)
            coverage = spec.get("coverage", "all")
            if coverage not in ("all", "sample", "explicit"):
                raise PoolError(f"{context}.coverage must be all, sample, or explicit")
            if "seed" in spec:
                _integer(spec["seed"], f"{context}.seed")
            if "samples" in spec:
                _integer(spec["samples"], f"{context}.samples", 1)
            if coverage == "sample" and "samples" not in spec:
                raise PoolError(f"{context}: sample coverage requires samples")
            if coverage != "sample" and "samples" in spec:
                raise PoolError(f"{context}: samples is only valid for sample coverage")
            if coverage != "explicit" and "groups" in spec:
                raise PoolError(f"{context}: groups is only valid for explicit coverage")
            groups = None
            if coverage == "explicit":
                raw_groups = spec.get("groups")
                if not isinstance(raw_groups, list) or not raw_groups:
                    raise PoolError(f"{context}.groups must be a nonempty list")
                groups = []
                seen = set()
                for raw in raw_groups:
                    if not isinstance(raw, list) or len(raw) != arity or any(not isinstance(s, str) for s in raw):
                        raise PoolError(f"{context}: each explicit group requires {arity} distinct states")
                    if len(set(raw)) != arity or not set(raw) <= set(candidate_names):
                        raise PoolError(f"{context}: explicit group must contain distinct candidate states")
                    group = tuple(sorted(self.state_indices[s] for s in raw))
                    if group in seen:
                        raise PoolError(f"{context}: duplicate explicit toggle group")
                    seen.add(group)
                    groups.append(group)
            self.leaf_arities[name] = arity
            self._leaf_specs[name] = {"states": candidates, "arity": arity, "coverage": coverage,
                                      "samples": spec.get("samples"), "seed": spec.get("seed", self.seed),
                                      "groups": groups}

        for name, spec in self.rng_banks.items():
            _name(name, "RNG bank name")
            context = f"rng_banks.{name}"
            _object(spec, context)
            _keys(spec, {"base", "count", "seed", "scope"}, context)
            base = spec.get("base")
            if base not in ("uniform01", "normal01"):
                raise PoolError(f"{context}.base must be uniform01 or normal01")
            scope = spec.get("scope", "run")
            if scope not in ("run", "skeleton"):
                raise PoolError(f"{context}.scope must be run or skeleton")
            self._bank_specs[name] = {"base": base, "count": _integer(spec.get("count"), f"{context}.count", 1),
                                      "seed": _integer(spec.get("seed", self.seed), f"{context}.seed"), "scope": scope}

        shared_counts = {}
        for name, spec in self.rng.items():
            _name(name, "RNG slot name")
            context = f"rng.{name}"
            _object(spec, context)
            _keys(spec, {"bank", "axis", "stream", "transform"}, context)
            bank = _name(spec.get("bank"), f"{context}.bank")
            if bank not in self._bank_specs:
                raise PoolError(f"{context}: unknown RNG bank {bank!r}")
            bank_spec = self._bank_specs[bank]
            if "axis" in spec:
                axis_name = _name(spec["axis"], f"{context}.axis")
                axis = f"rng:shared:{axis_name}"
                count = bank_spec["count"]
                if axis in shared_counts and shared_counts[axis] != count:
                    raise PoolError(f"{context}: shared RNG axis {axis_name!r} requires equal bank counts")
                shared_counts[axis] = count
            else:
                axis = f"rng:slot:{name}"
            stream = _name(spec.get("stream", name), f"{context}.stream")
            transform = self._transform(spec.get("transform", {"kind": "identity"}), bank_spec["base"], context)
            self._rng_specs[name] = {"bank": bank, "axis": axis, "stream": stream, "transform": transform}

        for name, spec in self.parameters.items():
            _name(name, "parameter name")
            context = f"parameters.{name}"
            _object(spec, context)
            _keys(spec, {"initial"}, context)
            self._parameter_specs[name] = _number(spec.get("initial"), f"{context}.initial")

    def _transform(self, transform: Any, base: str, context: str) -> dict:
        _object(transform, f"{context}.transform")
        kind = transform.get("kind", "identity")
        fields = {"identity": (), "affine": ("scale", "shift"), "uniform": ("low", "high"),
                  "normal": ("mean", "std"), "log_uniform": ("low", "high")}
        if not isinstance(kind, str) or kind not in fields:
            raise PoolError(f"{context}: unknown RNG transform {kind!r}")
        _keys(transform, {"kind", *fields[kind]}, f"{context}.transform")
        if kind in ("uniform", "log_uniform") and base != "uniform01":
            raise PoolError(f"{context}: {kind} transform requires a uniform01 bank")
        if kind == "normal" and base != "normal01":
            raise PoolError(f"{context}: normal transform requires a normal01 bank")
        result = {"kind": kind}
        defaults = {"scale": 1, "shift": 0, "mean": 0, "std": 1}
        for field in fields[kind]:
            value = transform.get(field, defaults.get(field))
            if isinstance(value, str) and value.startswith("const."):
                slot = value[6:]
                if slot not in self._constant_specs:
                    raise PoolError(f"{context}: unknown transform constant {slot!r}")
                result[field] = {"const": slot}
            else:
                result[field] = _number(value, f"{context}.transform.{field}")
        if kind == "normal" and self._range(result["std"])[0] <= 0:
            raise PoolError(f"{context}: normal std must be positive for every configuration")
        if kind in ("uniform", "log_uniform"):
            if self._range(result["low"])[1] >= self._range(result["high"])[0]:
                raise PoolError(f"{context}: low must be below high for every Cartesian configuration")
            if kind == "log_uniform" and self._range(result["low"])[0] <= 0:
                raise PoolError(f"{context}: log_uniform bounds must be positive")
        return result

    def _range(self, value: Any) -> tuple[float | int, float | int]:
        if isinstance(value, dict):
            spec = self._constant_specs[value["const"]]
            if spec["kind"] == "fixed":
                return spec["value"], spec["value"]
            return min(spec["values"]), max(spec["values"])
        return value, value

    def resolve_active(self, active: Mapping[str, set[str]]) -> dict[str, set[str]]:
        """Include constants referenced only by a pre-integration transform."""
        if not isinstance(active, Mapping) or set(active) - {"leaf", "const", "rng", "param"}:
            raise PoolError("active must map leaf, const, rng, param to collections of names")
        definitions = {"leaf": self._leaf_specs, "const": self._constant_specs,
                       "rng": self._rng_specs, "param": self._parameter_specs}
        result = {}
        for kind, specs in definitions.items():
            values = active.get(kind, set())
            if isinstance(values, str):
                raise PoolError(f"active.{kind} must be a collection of names")
            try:
                result[kind] = set(values)
            except TypeError as exc:
                raise PoolError(f"active.{kind} must be a collection of names") from exc
            if any(not isinstance(v, str) for v in result[kind]):
                raise PoolError(f"active.{kind} must contain string names")
            missing = result[kind] - set(specs)
            if missing:
                raise PoolError(f"unknown {kind} slots: {sorted(missing)}")
        for name in result["rng"]:
            for value in self._rng_specs[name]["transform"].values():
                if isinstance(value, dict):
                    result["const"].add(value["const"])
        return result

    def _groups(self, name: str) -> Iterator[tuple[int, ...]]:
        spec = self._leaf_specs[name]
        states, arity = spec["states"], spec["arity"]
        if spec["coverage"] == "explicit":
            yield from spec["groups"]
        elif spec["coverage"] == "all":
            yield from itertools.combinations(states, arity)
        else:
            total = math.comb(len(states), arity)
            count = min(spec["samples"], total)
            if count == total:
                yield from itertools.combinations(states, arity)
            else:
                seed = int(_stable_digest(["leaf_groups_v1", spec["seed"], name, states, arity]), 16)
                for rank in _sample_ranks(total, count, seed):
                    yield tuple(states[i] for i in _unrank_combination(len(states), arity, rank))

    def variant_count(self, active: Mapping[str, set[str]]) -> int:
        active = self.resolve_active(active)
        result = 1
        for name in active["leaf"]:
            result *= self._group_count(name)
        return result

    def _group_count(self, name: str) -> int:
        spec = self._leaf_specs[name]
        count = math.comb(len(spec["states"]), spec["arity"])
        if spec["coverage"] == "explicit":
            return len(spec["groups"])
        if spec["coverage"] == "sample":
            return min(count, spec["samples"])
        return count

    def _bank_request(self, bank: str, stream: str, skeleton_id: str) -> dict:
        spec = self._bank_specs[bank]
        address = {"version": 1, "bank": bank, "seed": spec["seed"], "scope": spec["scope"],
                   "stream": stream, "base": spec["base"]}
        if spec["scope"] == "skeleton":
            address["skeleton_id"] = skeleton_id
        key = _stable_digest(address)
        return {"id": "bank_" + key, "algorithm": "sha256_reference_v1", "address": address,
                "key": key, "base": spec["base"], "count": spec["count"],
                "generation": "once_before_integration", "counter": "axis_index"}

    def iter_variants(self, active: Mapping[str, set[str]], skeleton_id: str,
                      group_sampling: dict | None = None) -> Iterator[dict]:
        """Yield all group combinations or N sampled *joint* combinations.

        ``group_sampling={"count": N, "seed": S}`` samples without replacement
        from the Cartesian product of each active leaf's allowed groups.  It
        emits at most N variants, rather than sampling N groups independently
        for every leaf and emitting their Cartesian product.  Exhaustive and
        explicit leaf domains support direct random access.  A leaf whose
        own coverage is ``sample`` keeps only the sampled-domain prefix needed
        to retrieve requested indices; very large such domains are better
        expressed as coverage=all with joint group_sampling.
        """
        active = self.resolve_active(active)
        _name(skeleton_id, "skeleton_id")
        if group_sampling is not None:
            _object(group_sampling, "group_sampling")
            _keys(group_sampling, {"count", "seed"}, "group_sampling")
            _integer(group_sampling.get("count"), "group_sampling.count", 1)
            _integer(group_sampling.get("seed", self.seed), "group_sampling.seed")
        leaf_names = sorted(active["leaf"])
        base = {"constants": {}, "rng_bindings": {}, "prelude": [], "rng_bank_requests": [],
                "parameter_initials": {n: self._parameter_specs[n] for n in sorted(active["param"])}}
        numeric_axes = []
        for name in sorted(active["const"]):
            spec = copy.deepcopy(self._constant_specs[name])
            base["constants"][name] = spec
            if spec["kind"] == "bank":
                numeric_axes.append({"id": spec["axis"], "kind": "constant", "count": len(spec["values"]),
                                     "slots": [name]})
        bank_requests, rng_axes = {}, {}
        for name in sorted(active["rng"]):
            spec = self._rng_specs[name]
            request = self._bank_request(spec["bank"], spec["stream"], skeleton_id)
            bank_requests[request["id"]] = request
            axis = spec["axis"]
            rng_axes.setdefault(axis, {"id": axis, "kind": "rng", "count": request["count"], "slots": []})["slots"].append(name)
            binding = {"bank_id": request["id"], "axis": axis, "base": request["base"],
                       "transform": copy.deepcopy(spec["transform"])}
            base["rng_bindings"][name] = binding
            base["prelude"].append({"op": "RNG_BIND", "slot": name, **copy.deepcopy(binding),
                                     "frequency": "once_per_configuration", "output": "RNG_VALUE"})
        base["rng_bank_requests"] = list(bank_requests.values())
        numeric_axes += [rng_axes[name] for name in sorted(rng_axes)]

        def emit(toggles: dict[str, list[int]]) -> dict:
            output = copy.deepcopy(base)
            output["toggles"] = copy.deepcopy(toggles)
            axes = [{"id": f"leaf:{name}", "kind": "leaf", "count": self.leaf_arities[name],
                     "slots": [name], "states": list(toggles[name])} for name in leaf_names]
            axes += copy.deepcopy(numeric_axes)
            output["pool_axes"] = axes
            output["configuration_count"] = math.prod(axis["count"] for axis in axes)
            return output

        def visit(index: int, toggles: dict[str, list[int]]) -> Iterator[dict]:
            if index == len(leaf_names):
                yield emit(toggles)
                return
            name = leaf_names[index]
            for group in self._groups(name):
                toggles[name] = list(group)
                yield from visit(index + 1, toggles)
            toggles.pop(name, None)

        total = self.variant_count(active)
        if group_sampling is None or group_sampling["count"] >= total:
            yield from visit(0, {})
            return

        # Mixed-radix rank decoding gives a uniform sample of the *joint*
        # finite space without enumerating its product or its combinations.
        counts = {name: self._group_count(name) for name in leaf_names}
        sampled_prefixes: dict[str, list[tuple[int, ...]]] = {}
        sampled_iterators: dict[str, Iterator[tuple[int, ...]]] = {}

        def group_at(name: str, index: int) -> tuple[int, ...]:
            spec = self._leaf_specs[name]
            if spec["coverage"] == "explicit":
                return spec["groups"][index]
            n, arity = len(spec["states"]), spec["arity"]
            if spec["coverage"] == "all" or counts[name] == math.comb(n, arity):
                return tuple(spec["states"][i] for i in _unrank_combination(n, arity, index))
            cache = sampled_prefixes.setdefault(name, [])
            if name not in sampled_iterators:
                sampled_iterators[name] = self._groups(name)
            while len(cache) <= index:
                cache.append(next(sampled_iterators[name]))
            return cache[index]

        seed = int(_stable_digest(["joint_leaf_groups_v1", group_sampling.get("seed", self.seed),
                                   {name: self._leaf_specs[name] for name in leaf_names}]), 16)
        for rank in _sample_ranks(total, group_sampling["count"], seed):
            indices = {}
            for name in reversed(leaf_names):
                rank, indices[name] = divmod(rank, counts[name])
            toggles = {name: list(group_at(name, indices[name])) for name in leaf_names}
            yield emit(toggles)


def bank_value(request: dict, index: int) -> float:
    """Random-access reference unit sample; explicitly not a Philox generator.

    The bank key and sample index are the complete address.  Evaluation order,
    chunking, thread assignment, and requested bank length cannot change it.
    Floating-point normal transforms may differ by platform libm rounding.
    """
    if request.get("algorithm") != "sha256_reference_v1":
        raise PoolError("unsupported reference bank algorithm")
    _integer(index, "bank index", 0)
    if index >= request["count"]:
        raise PoolError("bank index out of bounds")

    def bits(component: int) -> int:
        raw = hashlib.sha256(f"{request['key']}:{index}:{component}".encode("ascii")).digest()
        return int.from_bytes(raw[:8], "big") >> 11

    if request["base"] == "uniform01":
        return bits(0) / (1 << 53)
    if request["base"] == "normal01":
        # Positive first uniform avoids log(0); second uniform is in [0,1).
        u = (bits(0) + 1) / ((1 << 53) + 1)
        v = bits(1) / (1 << 53)
        return math.sqrt(-2.0 * math.log(u)) * math.cos(math.tau * v)
    raise PoolError("unsupported reference bank base")


def materialize_bank(request: dict, *, max_count: int = 1000000) -> list[float]:
    """Materialize a bounded reference bank; use bank_value for larger banks."""
    _integer(max_count, "max_count", 1)
    if request["count"] > max_count:
        raise PoolError(f"reference materialization exceeds max_count={max_count}; use bank_value")
    return [bank_value(request, i) for i in range(request["count"])]


def configuration_indices(variant: dict, linear_index: int) -> dict[str, int]:
    """Decode a flat Cartesian index without constructing the Cartesian pool.

    ``pool_axes`` order is canonical, with the last listed axis varying
    fastest (the convention used by itertools.product).  Python integers
    support pools larger than 64-bit indices; a downstream GPU backend must
    independently validate its own index-width limit.
    """
    _integer(linear_index, "configuration index", 0)
    if linear_index >= variant["configuration_count"]:
        raise PoolError("configuration index out of bounds")
    indices = {}
    remaining = linear_index
    for axis in reversed(variant["pool_axes"]):
        remaining, indices[axis["id"]] = divmod(remaining, axis["count"])
    return {axis["id"]: indices[axis["id"]] for axis in variant["pool_axes"]}


def evaluate_prelude(variant: dict, axis_indices: Mapping[str, int] | None = None) -> dict:
    """Resolve one configuration for tests/inspection, without an ODE solve.

    Omitted axes select index zero.  RNG values are computed once here and
    should remain fixed across every RHS evaluation for this configuration.
    """
    if axis_indices is None:
        axis_indices = {}
    known = {axis["id"] for axis in variant["pool_axes"]}
    if set(axis_indices) - known:
        raise PoolError(f"unknown configuration axes: {sorted(set(axis_indices) - known)}")
    selected = {}
    for axis in variant["pool_axes"]:
        index = _integer(axis_indices.get(axis["id"], 0), f"axis {axis['id']}", 0)
        if index >= axis["count"]:
            raise PoolError(f"axis {axis['id']} index out of bounds")
        selected[axis["id"]] = index
    constants = {}
    for name, spec in variant["constants"].items():
        constants[name] = spec["value"] if spec["kind"] == "fixed" else spec["values"][selected[spec["axis"]]]
    banks = {request["id"]: request for request in variant["rng_bank_requests"]}
    values = {}
    for name, binding in variant["rng_bindings"].items():
        raw = bank_value(banks[binding["bank_id"]], selected[binding["axis"]])
        transform = binding["transform"]

        def param(field: str) -> float | int:
            value = transform[field]
            return constants[value["const"]] if isinstance(value, dict) else value

        kind = transform["kind"]
        if kind == "identity":
            value = raw
        elif kind == "affine":
            value = param("scale") * raw + param("shift")
        elif kind == "uniform":
            # Convex form avoids overflow of high-low for extreme finite bounds.
            value = (1.0 - raw) * param("low") + raw * param("high")
        elif kind == "normal":
            value = param("mean") + param("std") * raw
        elif kind == "log_uniform":
            value = math.exp((1.0 - raw) * math.log(param("low")) + raw * math.log(param("high")))
        else:
            raise PoolError(f"unknown transform {kind!r}")
        _number(value, f"transformed RNG slot {name}")
        values[name] = value
    return {"leaf": {name: group[selected[f"leaf:{name}"]] for name, group in variant["toggles"].items()},
            "const": constants, "rng": values, "param": copy.deepcopy(variant["parameter_initials"])}
