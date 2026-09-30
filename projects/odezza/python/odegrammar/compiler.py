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
"""Streaming compiler from constrained ODE grammars to postorder programs.

This module does not integrate an ODE, fit coefficients, or execute GPU code.
The emitted pools and setup descriptors form a backend-independent interface.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator

from .expr import Node, parse_expr, rename_refs, walk
from .expansion import GrammarExpander
from .pools import PoolPlanner


FORMAT = "odegrammar.postorder.v1"
NAMESPACES = {"leaf": "leaves", "const": "constants", "rng": "rng", "param": "parameters"}
PREFIXES = {"leaf": "leaf", "const": "const", "rng": "rng", "param": "theta"}
OPERATORS = {"add", "sub", "mul", "div", "neg", "powi", "sin", "cos", "tanh", "exp", "log", "sqrt", "abs"}
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class CompileError(ValueError):
    def __init__(self, message: str, code: str = "INVALID_REQUEST", path: str = ""):
        super().__init__((path + ": " if path else "") + message)
        self.code = code
        self.path = path

    def as_dict(self) -> dict:
        return {"type": "error", "code": self.code, "path": self.path, "message": str(self)}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def content_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise CompileError("expected an object", path=path)
    if any(not isinstance(key, str) for key in value):
        raise CompileError("object keys must be strings", path=path)
    return value


def _keys(value: dict, allowed: set[str], path: str):
    unknown = set(value) - allowed
    if unknown:
        raise CompileError("unknown fields: " + ", ".join(sorted(unknown)), path=path)


def _integer(value: Any, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CompileError(f"expected an integer >= {minimum}", path=path)
    return value


def _finite(value: Any, path: str, positive: bool = False) -> float:
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise CompileError("expected a finite number", path=path)
    if positive and value <= 0:
        raise CompileError("expected a positive number", path=path)
    return float(value)


def _states(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CompileError("provide a nonempty ordered state list", path="states")
    if any(not isinstance(x, str) or not IDENTIFIER.fullmatch(x) or x == "t" for x in value):
        raise CompileError("state names must be identifiers other than reserved time symbol t", path="states")
    if len(value) != len(set(value)):
        raise CompileError("state names must be unique", path="states")
    return value


def _tags(value: Any, path: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(t, str) or not t for t in value):
        raise CompileError("expected nonempty tag strings", path=path)
    return sorted(set(value))


def validate_integration(value: Any, annotation_only: bool = False) -> dict:
    settings = deepcopy(_object(value, "integration"))
    _keys(settings, {"method", "dt", "stiff", "rtol", "atol", "max_steps"}, "integration")
    method = settings.setdefault("method", "rk4")
    if not isinstance(method, str) or not IDENTIFIER.fullmatch(method):
        raise CompileError("method must be an identifier", path="integration.method")
    stiff = settings.setdefault("stiff", False)
    if not isinstance(stiff, bool):
        raise CompileError("expected a boolean", path="integration.stiff")
    for name in ("dt", "rtol", "atol"):
        if name in settings:
            _finite(settings[name], "integration." + name, positive=True)
    if "max_steps" in settings:
        _integer(settings["max_steps"], "integration.max_steps")
    reasons = []
    if method != "rk4":
        reasons.append("integration_method_unavailable")
    if method == "rk4" and stiff:
        reasons.append("rk4_is_not_a_stiff_solver")
    if method == "rk4" and ("rtol" in settings or "atol" in settings):
        reasons.append("fixed_step_rk4_does_not_implement_tolerance_control")
    if reasons and not annotation_only:
        raise CompileError("; ".join(reasons) + ". Use --allow-unsupported-integrator for annotation-only output; no fallback will occur.", code="UNSUPPORTED_INTEGRATOR", path="integration")
    return {"settings": settings, "backend_supported": not reasons, "annotation_only": bool(reasons), "reasons": reasons, "implemented_methods": ["rk4"], "runtime_dt_required": method == "rk4" and "dt" not in settings}


def normalize_request(raw: dict) -> dict:
    """Accept the earlier generate/rhs example as well as the native request."""
    request = deepcopy(_object(raw, "request"))
    if "generate" in request:
        generated = _object(request.pop("generate"), "generate")
        _keys(generated, {"rules", "shapes", "rhs", "expansion", "target_unique_skeletons", "deduplicate", "max_nodes", "max_depth"}, "generate")
        for name in ("rules", "shapes", "rhs"):
            if name in generated:
                if name in request:
                    raise CompileError(f"declare {name} either at top level or in generate")
                request[name] = generated[name]
        expansion = _object(request.setdefault("expansion", {}), "expansion")
        mode = generated.get("expansion", "cartesian")
        if mode not in ("cartesian", "enumerate", "sample"):
            raise CompileError("expected cartesian, enumerate, or sample", path="generate.expansion")
        expansion.setdefault("strategy", "enumerate" if mode == "cartesian" else mode)
        for name in ("max_nodes", "max_depth"):
            if name in generated:
                expansion.setdefault(name, generated[name])
        if "target_unique_skeletons" in generated:
            _object(request.setdefault("limits", {}), "limits").setdefault("max_skeletons", generated["target_unique_skeletons"])
        if generated.get("deduplicate", "lowered_system") != "lowered_system":
            raise CompileError("only lowered_system deduplication is supported", path="generate.deduplicate")
    if "budget" in request:
        budget = _object(request.pop("budget"), "budget")
        aliases = {"max_generation_attempts": "max_derivations", "max_wall_seconds": "max_seconds", "max_unique_skeletons": "max_skeletons", "max_configurations": "max_configurations", "max_variants": "max_variants"}
        _keys(budget, set(aliases) | {"on_limit"}, "budget")
        if budget.get("on_limit", "return_partial") != "return_partial":
            raise CompileError("only return_partial is supported", path="budget.on_limit")
        limits = _object(request.setdefault("limits", {}), "limits")
        for old, new in aliases.items():
            if old in budget:
                limits.setdefault(new, budget[old])
    if "family_id" in request:
        if "families" in request:
            raise CompileError("family_id cannot accompany families")
        request["families"] = [{"id": request.pop("family_id")}]
    return request


def _retention(raw: Any) -> dict:
    raw = deepcopy(_object(raw, "retain"))
    if "global_top_k" in raw:
        if "global" in raw:
            raise CompileError("global and global_top_k are mutually exclusive", path="retain")
        raw["global"] = raw.pop("global_top_k")
    _keys(raw, {"global", "per_family", "by_tag"}, "retain")
    def entry(value, path):
        if isinstance(value, int) and not isinstance(value, bool):
            value = {"k": value}
        value = _object(value, path)
        _keys(value, {"k", "unit"}, path)
        k = _integer(value.get("k", 20), path + ".k", minimum=0)
        unit = value.get("unit", "numeric_candidate")
        if unit not in ("numeric_candidate", "resolved_structure", "variant"):
            raise CompileError("unknown retention unit", path=path + ".unit")
        return {"k": k, "unit": unit}
    result = {"global": entry(raw.get("global", 20), "retain.global")}
    if "per_family" in raw:
        result["per_family"] = entry(raw["per_family"], "retain.per_family")
    by_tag = _object(raw.get("by_tag", {}), "retain.by_tag")
    result["by_tag"] = {}
    for name, value in by_tag.items():
        if not isinstance(name, str) or not name:
            raise CompileError("tag names must be nonempty strings", path="retain.by_tag")
        result["by_tag"][name] = entry(value, "retain.by_tag." + name)
    return result


class MemoryDedup:
    """Store IDs, never the expanded AST population."""
    def __init__(self):
        self.ids: dict[str, set[str]] = {}

    def add(self, kind: str, identifier: str) -> bool:
        ids = self.ids.setdefault(kind, set())
        if identifier in ids:
            return False
        ids.add(identifier)
        return True

    def contains(self, kind: str, identifier: str) -> bool:
        return identifier in self.ids.get(kind, ())

    def close(self):
        pass


class SQLiteDedup:
    """Optional disk-backed exact dedup for million-program streams.

    A new database is required for each output stream: reusing it would omit
    records required to interpret the new output.
    """
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.exists():
            raise CompileError("dedup database already exists; choose a fresh path", path=str(path))
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("CREATE TABLE ids (kind TEXT, id TEXT, PRIMARY KEY(kind,id)) WITHOUT ROWID")
        self.count = 0

    def add(self, kind: str, identifier: str) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO ids VALUES (?,?)", (kind, identifier))
        self.count += 1
        if self.count % 10000 == 0:
            self.connection.commit()
        return cursor.rowcount == 1

    def contains(self, kind: str, identifier: str) -> bool:
        return self.connection.execute("SELECT 1 FROM ids WHERE kind=? AND id=?", (kind, identifier)).fetchone() is not None

    def close(self):
        try:
            self.connection.commit()
        finally:
            self.connection.close()


def _rewrite_strings(value: Any, aliases: dict[tuple[str, str], str]) -> Any:
    if isinstance(value, str):
        for (namespace, name), replacement in aliases.items():
            if value == PREFIXES[namespace] + "." + name:
                return PREFIXES[namespace] + "." + replacement
        return value
    if isinstance(value, list):
        return [_rewrite_strings(v, aliases) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_strings(v, aliases) for k, v in value.items()}
    return value


def normalize_locals(rhs: dict[str, Node], local_bindings: dict[str, dict]) -> tuple[dict[str, Node], dict[str, dict]]:
    """Alpha-rename generated local slots while preserving their ties."""
    aliases: dict[tuple[str, str], str] = {}
    counts = {namespace: 0 for namespace in NAMESPACES}
    def assign(namespace, name):
        key = (namespace, name)
        if name in local_bindings.get(NAMESPACES[namespace], {}) and key not in aliases:
            aliases[key] = "__local_" + namespace + "_" + str(counts[namespace])
            counts[namespace] += 1
    for root in rhs.values():
        for node in walk(root):
            if node.op in NAMESPACES:
                assign(node.op, node.value)
    # Transform parameters can depend on constants absent from the RHS.
    def dependencies(value):
        if isinstance(value, str):
            for namespace, prefix in PREFIXES.items():
                if value.startswith(prefix + "."):
                    assign(namespace, value[len(prefix) + 1:])
        elif isinstance(value, dict):
            for key in sorted(value):
                dependencies(value[key])
        elif isinstance(value, list):
            for item in value:
                dependencies(item)
    cursor = 0
    while cursor < len(aliases):
        key = list(aliases)[cursor]
        dependencies(local_bindings[NAMESPACES[key[0]]][key[1]])
        cursor += 1
    normalized = {section: {} for section in NAMESPACES.values()}
    for (namespace, old), new in aliases.items():
        normalized[NAMESPACES[namespace]][new] = _rewrite_strings(local_bindings[NAMESPACES[namespace]][old], aliases)
    return {state: rename_refs(root, aliases) for state, root in rhs.items()}, normalized


def lower_system(rhs: dict[str, Node], states: list[str], bindings: dict[str, dict]) -> tuple[dict, dict, list, list]:
    active = {namespace: set() for namespace in NAMESPACES}
    annotations = []
    all_tags = set()
    state_indices = {state: index for index, state in enumerate(states)}
    lowered = {}
    for state in states:
        program = []
        def visit(node):
            start = len(program)
            if node.op == "tag":
                visit(node.children[0])
                tags = _tags(list(node.value), "expression.tags")
                all_tags.update(tags)
                annotations.append({"state": state, "start": start, "end": len(program), "tags": tags})
                return
            for child in node.children:
                visit(child)
            if node.op == "literal":
                _finite(node.value, "literal")
                instruction = {"op": "LITERAL", "value": node.value}
            elif node.op == "symbol":
                if node.value == "t":
                    instruction = {"op": "TIME"}
                elif node.value in state_indices:
                    instruction = {"op": "STATE", "index": state_indices[node.value]}
                else:
                    raise CompileError("unresolved symbol " + repr(node.value), path="rhs." + state)
            elif node.op in NAMESPACES:
                if node.value not in bindings[NAMESPACES[node.op]]:
                    raise CompileError(f"undeclared {PREFIXES[node.op]}.{node.value}", path="rhs." + state)
                active[node.op].add(node.value)
                opcode = {"leaf": "TOGGLE", "const": "CONSTANT", "rng": "RNG_VALUE", "param": "PARAMETER"}[node.op]
                instruction = {"op": opcode, "slot": node.value}
                if node.op == "leaf":
                    instruction["arity"] = bindings["leaves"][node.value].get("arity", 2)
            elif node.op in OPERATORS:
                instruction = {"op": node.op.upper()}
                if node.op == "powi":
                    instruction["exponent"] = node.value
            else:
                raise CompileError("unexpanded expression node " + node.op, path="rhs." + state)
            program.append(instruction)
        visit(rhs[state])
        lowered[state] = program
    annotations.sort(key=lambda x: (x["state"], x["start"], x["end"], x["tags"]))
    return lowered, active, sorted(all_tags), annotations


DEFAULT_LIMITS = {"max_skeletons": 10000, "max_variants": 100000, "max_configurations": 1000000000, "max_derivations": 1000000, "max_expansion_steps": 10000000, "max_seconds": 60.0}


class Compiler:
    def __init__(self, request: dict, *, annotation_only: bool = False, dedup=None):
        self.request = normalize_request(request)
        request = self.request
        _keys(request, {"version", "kind", "states", "problem_id", "description", "integration", "rules", "shapes", "rhs", "families", "leaves", "constant_banks", "constants", "rng_banks", "rng", "parameters", "expansion", "toggle_sampling", "limits", "retain"}, "request")
        if type(request.get("version", 1)) is not int or request.get("version", 1) != 1:
            raise CompileError("supported grammar version is integer 1", path="version")
        if request.get("kind", "compile") != "compile":
            raise CompileError("use the separate lm command for an LM request", path="kind")
        self.states = _states(request.get("states"))
        self.integration = validate_integration(request.get("integration", {"method": "rk4"}), annotation_only)
        self.retention = _retention(request.get("retain", {}))
        expansion = _object(request.get("expansion", {}), "expansion")
        _keys(expansion, {"strategy", "seed", "max_nodes", "max_depth", "max_expansion_depth"}, "expansion")
        self.expansion = {"strategy": "enumerate", "seed": 0, "max_nodes": 63, "max_depth": 12, "max_expansion_depth": 20, **expansion}
        if self.expansion["strategy"] not in ("enumerate", "sample"):
            raise CompileError("expected enumerate or sample", path="expansion.strategy")
        for name in ("max_nodes", "max_depth", "max_expansion_depth"):
            _integer(self.expansion[name], "expansion." + name)
        _integer(self.expansion["seed"], "expansion.seed", minimum=0)
        limits = _object(request.get("limits", {}), "limits")
        _keys(limits, set(DEFAULT_LIMITS), "limits")
        self.limits = {**DEFAULT_LIMITS, **limits}
        for name, value in self.limits.items():
            if name == "max_seconds":
                _finite(value, "limits." + name, positive=True)
            else:
                _integer(value, "limits." + name)
        self.dedup = dedup if dedup is not None else MemoryDedup()
        self._owns_dedup = dedup is None
        self._used = False
        self.families = self._families()
        self.summary = {}

    def _families(self) -> list[dict]:
        request = self.request
        families = request.get("families", [{"id": "default"}])
        if not isinstance(families, list) or not families:
            raise CompileError("expected a nonempty family list", path="families")
        ids = set()
        result = []
        inherited = ("rules", "shapes", "leaves", "constants", "rng", "parameters")
        for index, raw in enumerate(families):
            path = f"families[{index}]"
            raw = _object(raw, path)
            _keys(raw, {"id", "tags", "description", "rhs", "toggle_sampling", *inherited}, path)
            identifier = raw.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise CompileError("family id must be a nonempty unique string", path=path + ".id")
            ids.add(identifier)
            family = {"id": identifier, "tags": _tags(raw.get("tags", []), path + ".tags")}
            toggle_sampling = raw.get("toggle_sampling", request.get("toggle_sampling"))
            if toggle_sampling is not None:
                toggle_sampling = _object(toggle_sampling, path + ".toggle_sampling")
                _keys(toggle_sampling, {"count", "seed"}, path + ".toggle_sampling")
                _integer(toggle_sampling.get("count"), path + ".toggle_sampling.count")
                seed = toggle_sampling.get("seed", self.expansion["seed"])
                _integer(seed, path + ".toggle_sampling.seed", minimum=0)
                toggle_sampling = {"count": toggle_sampling["count"], "seed": seed}
            family["toggle_sampling"] = toggle_sampling
            for name in inherited:
                family[name] = {**_object(request.get(name, {}), name), **_object(raw.get(name, {}), path + "." + name)}
                if name in NAMESPACES.values() and any(key.startswith("__local_") for key in family[name]):
                    raise CompileError("__local_ prefix is reserved for hygienic grammar bindings", path=path + "." + name)
            family["rhs"] = _object(raw.get("rhs", request.get("rhs", {})), path + ".rhs")
            if set(family["rhs"]) != set(self.states):
                raise CompileError("provide exactly one RHS for every declared state", path=path + ".rhs")
            family["rhs"] = {state: family["rhs"][state] for state in self.states}
            if any(not isinstance(expr, str) for expr in family["rhs"].values()):
                raise CompileError("RHS values must be expression strings", path=path + ".rhs")
            # Validate all global pools even if a branch never references them.
            PoolPlanner(self.states, family["leaves"], family["constants"], request.get("constant_banks", {}), family["rng"], request.get("rng_banks", {}), family["parameters"], seed=self.expansion["seed"])
            self._validate_references(family)
            result.append(family)
        return result

    def _validate_references(self, family):
        """Catch misspellings in branches that may not be reached by a cap."""
        rule_names = set()
        signatures = {}
        for signature in family["rules"]:
            match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s*", signature)
            if not match:
                raise CompileError("invalid rule signature " + repr(signature))
            name = match.group(1)
            if name in self.states or name == "t":
                raise CompileError("rule names cannot shadow a state or reserved time symbol t", path="rules." + signature)
            formals = [x.strip() for x in (match.group(2) or "").split(",") if x.strip()]
            if name in signatures:
                raise CompileError("duplicate rule name " + name)
            signatures[name] = formals
            rule_names.add(name)
        def check(text, formals=(), local=None, path="expression"):
            root = parse_expr(text)
            local = local or {}
            for node in walk(root):
                if node.op == "symbol" and node.value not in set(self.states) | {"t"} | rule_names | set(formals):
                    raise CompileError("unknown symbol " + repr(node.value), path=path)
                if node.op == "symbol" and node.value in signatures and node.value not in formals and signatures[node.value]:
                    raise CompileError(f"rule {node.value} requires arguments; use hole({node.value}, ...)", path=path)
                if node.op in NAMESPACES:
                    section = NAMESPACES[node.op]
                    if node.value not in family[section] and node.value not in local.get(section, {}):
                        raise CompileError(f"undeclared {PREFIXES[node.op]}.{node.value}", path=path)
                if node.op == "shape" and node.value not in family["shapes"]:
                    raise CompileError("unknown shape " + node.value, path=path)
                if node.op == "hole":
                    if node.value not in signatures:
                        raise CompileError("unknown rule " + node.value, path=path)
                    if len(node.children) != len(signatures[node.value]):
                        raise CompileError(f"rule {node.value} expects {len(signatures[node.value])} arguments", path=path)
            return root
        for state, expression in family["rhs"].items():
            check(expression, path="rhs." + state)
        def check_alternative(alternative, formals, path):
            if isinstance(alternative, str):
                check(alternative, formals, path=path)
                return
            alternative = _object(alternative, path)
            _keys(alternative, {"expr", "tags", "locals"}, path)
            _tags(alternative.get("tags", []), path + ".tags")
            local = _object(alternative.get("locals", {}), path + ".locals")
            _keys(local, set(NAMESPACES.values()), path + ".locals")
            check(alternative.get("expr"), formals, local, path)
            merged = {section: {**family[section], **_object(local.get(section, {}), path + ".locals." + section)} for section in NAMESPACES.values()}
            PoolPlanner(self.states, merged["leaves"], merged["constants"], self.request.get("constant_banks", {}), merged["rng"], self.request.get("rng_banks", {}), merged["parameters"], seed=self.expansion["seed"])
        for name, definition in family["shapes"].items():
            if isinstance(definition, dict) and set(definition) == {"choices"}:
                definition = definition["choices"]
            alternatives = definition if isinstance(definition, list) else [definition]
            if not alternatives:
                raise CompileError("named shape choices cannot be empty", path="shapes." + name)
            for index, alternative in enumerate(alternatives):
                check_alternative(alternative, (), f"shapes.{name}[{index}]")
        for signature, alternatives in family["rules"].items():
            name = re.match(r"\s*([A-Za-z_]\w*)", signature).group(1)
            if not isinstance(alternatives, list) or not alternatives:
                raise CompileError("rule alternatives must be a nonempty list", path="rules." + signature)
            for index, alternative in enumerate(alternatives):
                path = f"rules.{signature}[{index}]"
                check_alternative(alternative, signatures[name], path)
        # The expander performs signature/reserved-name/argument checks even
        # during validate, where no iterator is consumed.
        GrammarExpander(family["rules"], family["shapes"])

    def records(self) -> Iterator[dict]:
        if self._used:
            raise CompileError("Compiler streams are single use; construct a new Compiler to replay")
        self._used = True
        started = time.monotonic()
        counts = {"derivations": 0, "unique_skeletons": 0, "unique_variants": 0, "configuration_visits": 0, "duplicate_program_derivations": 0, "duplicate_variants": 0, "provenance_events": 0}
        family_counts = {}
        expansion_stats = {}
        stop = None
        expansion_steps = 0
        yield {"type": "manifest", "format": FORMAT, "request_id": content_id(self.request), "problem_id": self.request.get("problem_id"), "states": self.states, "integration": self.integration, "retention": self.retention, "expansion": self.expansion, "limits": self.limits, "rng_bank_definitions": self.request.get("rng_banks", {}), "family_ids": [f["id"] for f in self.families], "family_policies": [{"id": f["id"], "toggle_sampling": f["toggle_sampling"]} for f in self.families], "notes": ["Compiler only: no ODE solve or LM is executed.", "Configuration counts include overlaps between exhaustive toggle groups.", "RNG setup runs once per configuration before the integration loop."]}
        for family_index, family in enumerate(self.families):
            if stop:
                break
            family_counts[family["id"]] = {"derivations": 0, "variant_memberships": 0}
            remaining = self.limits["max_derivations"] - counts["derivations"]
            if remaining <= 0:
                stop = "max_derivations"
                break
            if expansion_steps >= self.limits["max_expansion_steps"]:
                stop = "max_expansion_steps"
                break
            expander = GrammarExpander(family["rules"], family["shapes"], max_nodes=self.expansion["max_nodes"], max_depth=self.expansion["max_depth"], max_expansion_depth=self.expansion["max_expansion_depth"], max_steps=self.limits["max_expansion_steps"] - expansion_steps, deadline=started + self.limits["max_seconds"])
            family_seed = int(content_id({"seed": self.expansion["seed"], "family": family["id"]})[:16], 16)
            iterator = expander.expand(family["rhs"], strategy=self.expansion["strategy"], seed=family_seed, max_derivations=remaining)
            for expanded in iterator:
                if time.monotonic() - started >= self.limits["max_seconds"]:
                    stop = "max_seconds"
                    break
                counts["derivations"] += 1
                family_counts[family["id"]]["derivations"] += 1
                rhs, locals_ = normalize_locals(expanded.rhs, expanded.locals)
                bindings = {section: {**family[section], **locals_[section]} for section in NAMESPACES.values()}
                lowered, active, tags, annotations = lower_system(rhs, self.states, bindings)
                program_identity = {"format": FORMAT, "states": self.states, "rhs": lowered}
                skeleton_id = content_id(program_identity)
                known_skeleton = self.dedup.contains("skeleton", skeleton_id)
                if not known_skeleton and counts["unique_skeletons"] >= self.limits["max_skeletons"]:
                    stop = "max_skeletons"
                    break
                planner = PoolPlanner(self.states, bindings["leaves"], bindings["constants"], self.request.get("constant_banks", {}), bindings["rng"], self.request.get("rng_banks", {}), bindings["parameters"], seed=self.expansion["seed"])
                emitted_skeleton = known_skeleton
                if known_skeleton:
                    counts["duplicate_program_derivations"] += 1
                for pool in planner.iter_variants(active, skeleton_id, group_sampling=family["toggle_sampling"]):
                    if time.monotonic() - started >= self.limits["max_seconds"]:
                        stop = "max_seconds"
                        break
                    variant_identity = {"skeleton_id": skeleton_id, "pools": pool, "integration": self.integration["settings"]}
                    variant_id = content_id(variant_identity)
                    provenance = {"variant_id": variant_id, "family_id": family["id"], "tags": sorted(set(tags) | set(family["tags"])), "annotations": annotations}
                    provenance_id = content_id(provenance)
                    if self.dedup.contains("variant", variant_id):
                        counts["duplicate_variants"] += 1
                        if self.dedup.add("provenance", provenance_id):
                            counts["provenance_events"] += 1
                            family_counts[family["id"]]["variant_memberships"] += 1
                            yield {"type": "provenance", **provenance}
                        continue
                    if counts["unique_variants"] >= self.limits["max_variants"]:
                        stop = "max_variants"
                        break
                    visits = pool["configuration_count"]
                    if counts["configuration_visits"] + visits > self.limits["max_configurations"]:
                        stop = "max_configurations"
                        break
                    if not emitted_skeleton:
                        self.dedup.add("skeleton", skeleton_id)
                        counts["unique_skeletons"] += 1
                        emitted_skeleton = True
                        yield {"type": "skeleton", "id": skeleton_id, "rhs": lowered, "active_slots": {name: sorted(values) for name, values in active.items()}, "node_count": sum(len(p) for p in lowered.values()), "parameter_count": len(active["param"])}
                    self.dedup.add("variant", variant_id)
                    self.dedup.add("provenance", provenance_id)
                    counts["unique_variants"] += 1
                    counts["configuration_visits"] += visits
                    family_counts[family["id"]]["variant_memberships"] += 1
                    yield {"type": "variant", "id": variant_id, "skeleton_id": skeleton_id, "pools": pool, **provenance}
                if stop:
                    break
            expansion_stats[family["id"]] = dict(expander.stats)
            expansion_steps += expander.stats.get("steps", 0)
            if not stop and expander.stop_reason not in (None, "exhausted"):
                stop = expander.stop_reason
        self.summary = {"type": "summary", **counts, "complete": stop is None, "stop_reason": stop or "exhausted", "elapsed_seconds": time.monotonic() - started, "families": family_counts, "expansion_stats": expansion_stats}
        yield self.summary


def compile_request(request: dict, *, annotation_only: bool = False, dedup=None) -> Iterator[dict]:
    yield from Compiler(request, annotation_only=annotation_only, dedup=dedup).records()


def compile_lm_request(raw: dict, *, annotation_only: bool = False) -> dict:
    request = deepcopy(_object(raw, "lm_request"))
    _keys(request, {"version", "kind", "states", "candidate_ids", "parameters", "settings", "integration"}, "lm_request")
    if request.get("kind") != "lm":
        raise CompileError("kind must be lm", path="kind")
    if type(request.get("version", 1)) is not int or request.get("version", 1) != 1:
        raise CompileError("supported grammar version is integer 1", path="version")
    states = _states(request.get("states"))
    if len(states) > 8:
        raise CompileError("LM supports at most 8 states", code="LM_STATE_LIMIT", path="states")
    parameters = request.get("parameters")
    if not isinstance(parameters, list) or not parameters or any(not isinstance(p, str) or not IDENTIFIER.fullmatch(p) for p in parameters):
        raise CompileError("provide distinct fitted RHS parameter names", path="parameters")
    if len(set(parameters)) != len(parameters):
        raise CompileError("duplicate names are shared parameters; list each once", path="parameters")
    if len(parameters) > 8:
        raise CompileError("LM supports at most 8 fitted scalars; fixed constants do not consume this limit", code="LM_PARAMETER_LIMIT", path="parameters")
    ids = request.get("candidate_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise CompileError("provide nonempty unique candidate IDs", path="candidate_ids")
    settings = _object(request.get("settings", {}), "settings")
    _keys(settings, {"max_iterations", "damping", "tolerance"}, "settings")
    settings = {"max_iterations": 50, "damping": 0.001, "tolerance": 1e-8, **settings}
    _integer(settings["max_iterations"], "settings.max_iterations")
    for name in ("damping", "tolerance"):
        _finite(settings[name], "settings." + name, positive=True)
    result = {"type": "lm_request", "format": FORMAT, "states": states, "candidate_ids": ids, "parameters": parameters, "fitted_dimension": len(parameters), "settings": settings, "optimizer_executed": False, "candidate_resolution": "deferred_to_execution_backend", "parameter_scope": "shared_rhs_scalars", "supports_initial_condition_fitting": False}
    if "integration" in request:
        result["integration"] = validate_integration(request["integration"], annotation_only)
    return result


def evaluate_postorder(program: list[dict], state_values: list[float], *, constants: dict | None = None, random_values: dict | None = None, parameters: dict | None = None, toggles: dict | None = None, choices: dict | None = None, t: float = 0.0) -> float:
    """Small strict reference evaluator for emitted RHS programs, not a solver."""
    stack = []
    constants, random_values, parameters = constants or {}, random_values or {}, parameters or {}
    toggles, choices = toggles or {}, choices or {}
    unary = {"SIN": math.sin, "COS": math.cos, "TANH": math.tanh, "EXP": math.exp, "LOG": math.log, "SQRT": math.sqrt, "ABS": abs, "NEG": lambda x: -x}
    for instruction in program:
        op = instruction["op"]
        if op == "LITERAL":
            stack.append(instruction["value"])
        elif op == "STATE":
            stack.append(state_values[instruction["index"]])
        elif op == "TIME":
            stack.append(t)
        elif op == "TOGGLE":
            slot = instruction["slot"]
            selection = choices.get(slot, 0)
            if isinstance(selection, bool) or not isinstance(selection, int) or not 0 <= selection < instruction["arity"]:
                raise ValueError("invalid toggle selection for " + slot)
            stack.append(state_values[toggles[slot][selection]])
        elif op in ("CONSTANT", "RNG_VALUE", "PARAMETER"):
            stack.append({"CONSTANT": constants, "RNG_VALUE": random_values, "PARAMETER": parameters}[op][instruction["slot"]])
        elif op in ("ADD", "SUB", "MUL", "DIV"):
            right, left = stack.pop(), stack.pop()
            stack.append({"ADD": lambda: left + right, "SUB": lambda: left - right, "MUL": lambda: left * right, "DIV": lambda: left / right}[op]())
        elif op == "POWI":
            stack.append(stack.pop() ** instruction["exponent"])
        elif op in unary:
            stack.append(unary[op](stack.pop()))
        else:
            raise ValueError("unknown opcode " + str(op))
    if len(stack) != 1:
        raise ValueError("invalid postorder stack")
    return stack[0]
