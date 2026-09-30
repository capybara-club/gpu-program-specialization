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
"""Streaming grammar expansion with deterministic sampling and lexical locals."""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from .expr import Node, node_count, node_depth, parse_expr, rename_refs, substitute, walk


class GrammarError(ValueError):
    """An invalid rule, shape, or generation request."""


class _StepLimit(Exception):
    pass


class _DeadlineLimit(Exception):
    pass


@dataclass
class ExpandedSystem:
    rhs: dict[str, Node]
    locals: dict[str, dict]


@dataclass(frozen=True)
class _Alternative:
    expr: Node
    tags: tuple[str, ...] = ()
    locals: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Rule:
    name: str
    parameters: tuple[str, ...]
    alternatives: tuple[_Alternative, ...]


@dataclass(frozen=True)
class _Context:
    shapes: dict[str, Node] = field(default_factory=dict)
    locals: dict[str, dict] = field(default_factory=lambda: {k: {} for k in _LOCAL_KINDS})


_LOCAL_KINDS = {"leaves": "leaf", "constants": "const", "rng": "rng", "parameters": "param"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUALIFIED_REF = re.compile(r"\b(leaf|const|rng|theta|param)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def _positive_int(value, label: str, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise GrammarError(f"{label} must be {'non-negative' if allow_zero else 'positive'} integer")
    return value


def _signature(text: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(text, str):
        raise GrammarError("Rule names/signatures must be strings")
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise GrammarError(f"Invalid rule signature {text!r}") from exc
    if isinstance(node, ast.Name):
        name, parameters = node.id, ()
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if any(not isinstance(a, ast.Name) for a in node.args):
            raise GrammarError(f"Rule signature {text!r} requires bare parameter names")
        name, parameters = node.func.id, tuple(a.id for a in node.args)
    else:
        raise GrammarError(f"Invalid rule signature {text!r}; expected R or R(z, v)")
    if name in {"hole", "pow", "sin", "cos", "tanh", "exp", "log", "sqrt", "abs"}:
        raise GrammarError(f"Rule name {name!r} is reserved")
    if len(parameters) != len(set(parameters)):
        raise GrammarError(f"Rule {name!r} repeats a formal parameter")
    return name, parameters


def _alternative(raw) -> _Alternative:
    if isinstance(raw, (str, Node)):
        return _Alternative(parse_expr(raw) if isinstance(raw, str) else raw)
    if not isinstance(raw, dict) or "expr" not in raw:
        raise GrammarError("A production must be an expression string or {expr, tags, locals}")
    unexpected = set(raw) - {"expr", "tags", "locals"}
    if unexpected:
        raise GrammarError(f"Unknown production fields: {', '.join(sorted(unexpected))}")
    tags = raw.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)) or any(not isinstance(t, str) or not t for t in tags):
        raise GrammarError("Production tags must be non-empty strings")
    local_defs = raw.get("locals", {})
    if not isinstance(local_defs, dict) or set(local_defs) - set(_LOCAL_KINDS):
        raise GrammarError("Production locals may contain leaves, constants, rng, and parameters")
    for category, definitions in local_defs.items():
        if not isinstance(definitions, dict):
            raise GrammarError(f"Local {category} must be a name-to-definition object")
        for name in definitions:
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise GrammarError(f"Invalid local binding name {name!r}")
    expr = raw["expr"]
    if not isinstance(expr, (str, Node)):
        raise GrammarError("A production expr must be an expression string")
    return _Alternative(parse_expr(expr) if isinstance(expr, str) else expr, tuple(dict.fromkeys(tags)), copy.deepcopy(local_defs))


def _rewrite_definition(value, renames):
    if isinstance(value, str):
        def replace(match):
            namespace, name = match.groups()
            op = "param" if namespace in ("theta", "param") else namespace
            return namespace + "." + renames.get((op, name), name)
        return _QUALIFIED_REF.sub(replace, value)
    if isinstance(value, list):
        return [_rewrite_definition(v, renames) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_definition(v, renames) for k, v in value.items()}
    return value


class GrammarExpander:
    """Expand whole-system grammars without materializing the Cartesian product.

    ``max_nodes`` applies to the whole system; ``max_depth`` to each RHS.
    Recursive rules are bounded by ``max_expansion_depth`` and global work steps.
    ``deadline`` is an optional absolute ``time.monotonic()`` deadline, checked
    before the first expansion step and at intervals of at most 128 steps.
    Sampling is with replacement at derivation level; deduplication belongs to
    the compiler so it can retain all family/tag provenance.
    """

    def __init__(self, rules, shapes, max_nodes=63, max_depth=12,
                 max_expansion_depth=20, max_steps=1000000, deadline=None):
        self.max_nodes = _positive_int(max_nodes, "max_nodes")
        self.max_depth = _positive_int(max_depth, "max_depth")
        self.max_expansion_depth = _positive_int(max_expansion_depth, "max_expansion_depth")
        self.max_steps = _positive_int(max_steps, "max_steps")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise GrammarError("deadline must be a finite absolute time.monotonic value or None")
        self.deadline = deadline
        if not isinstance(rules, dict) or not isinstance(shapes, dict):
            raise GrammarError("rules and shapes must be objects")
        self.rules: dict[str, _Rule] = {}
        for signature, alternatives in rules.items():
            name, parameters = _signature(signature)
            if name in self.rules:
                raise GrammarError(f"Duplicate rule name {name!r}")
            if not isinstance(alternatives, list) or not alternatives:
                raise GrammarError(f"Rule {name!r} needs a non-empty production list")
            self.rules[name] = _Rule(name, parameters, tuple(_alternative(a) for a in alternatives))
        self.shapes: dict[str, tuple[_Alternative, ...]] = {}
        for name, definition in shapes.items():
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise GrammarError(f"Invalid shape name {name!r}")
            if isinstance(definition, dict) and set(definition) == {"choices"}:
                definition = definition["choices"]
            alternatives = definition if isinstance(definition, list) else [definition]
            if not alternatives:
                raise GrammarError(f"Shape {name!r} needs at least one choice")
            self.shapes[name] = tuple(_alternative(a) for a in alternatives)
        for rule in self.rules.values():
            for alternative in rule.alternatives:
                self._validate_references(alternative.expr, set(rule.parameters))
        for alternatives in self.shapes.values():
            for alternative in alternatives:
                self._validate_references(alternative.expr)
        self.stats: dict = {}
        self.stop_reason: str | None = None

    def _validate_references(self, node: Node, formals=frozenset()):
        for item in walk(node):
            if item.op == "hole":
                if item.value not in self.rules:
                    raise GrammarError(f"Unknown grammar rule {item.value!r}")
                expected = len(self.rules[item.value].parameters)
                if len(item.children) != expected:
                    raise GrammarError(f"Rule {item.value!r} expects {expected} argument(s), got {len(item.children)}")
            elif item.op == "shape" and item.value not in self.shapes:
                raise GrammarError(f"Unknown named shape {item.value!r}")
            elif item.op == "symbol" and item.value in self.rules and item.value not in formals:
                if self.rules[item.value].parameters:
                    raise GrammarError(f"Rule {item.value!r} requires arguments; use hole({item.value}, ...)")

    def _step(self):
        if self.deadline is not None and self.stats["steps"] % 128 == 0 and time.monotonic() >= self.deadline:
            raise _DeadlineLimit
        if self.stats["steps"] >= self.max_steps:
            raise _StepLimit
        self.stats["steps"] += 1

    def _valid(self, node):
        if node_count(node) > self.max_nodes or node_depth(node) > self.max_depth:
            self.stats["pruned_size"] += 1
            return False
        return True

    def _instantiate(self, alternative, context, path):
        if not alternative.locals:
            return alternative.expr, context
        # Stable occurrence paths give lexical names, independent of how many
        # unrelated branches have already been visited.
        token = hashlib.sha256(repr(path).encode("utf-8")).hexdigest()[:20]
        renames = {
            (op, name): f"__local_{token}_{name}"
            for category, op in _LOCAL_KINDS.items()
            for name in alternative.locals.get(category, {})
        }
        definitions = {k: dict(v) for k, v in context.locals.items()}
        for category, entries in alternative.locals.items():
            op = _LOCAL_KINDS[category]
            for name, definition in entries.items():
                definitions[category][renames[(op, name)]] = _rewrite_definition(definition, renames)
        return rename_refs(alternative.expr, renames), _Context(context.shapes, definitions)

    def _alternatives(self, alternatives):
        if self._strategy == "sample":
            index = self._random.randrange(len(alternatives))
            return ((index, alternatives[index]),)
        return enumerate(alternatives)

    def _children(self, children, context, path, expansion_depth, active_shapes, index=0, selected=()):
        if index == len(children):
            yield selected, context
            return
        for expanded, updated in self._node(children[index], context, path + ("child", index), expansion_depth, active_shapes):
            if sum(node_count(c) for c in selected) + node_count(expanded) > self.max_nodes:
                self.stats["pruned_size"] += 1
                continue
            yield from self._children(children, updated, path, expansion_depth, active_shapes, index + 1, selected + (expanded,))

    def _node(self, node, context, path, expansion_depth, active_shapes):
        self._step()
        if node.op == "symbol" and node.value in self.rules:
            node = Node("hole", node.value)
        if node.op == "shape":
            name = node.value
            if name in context.shapes:
                yield context.shapes[name], context
                return
            if name in active_shapes:
                raise GrammarError(f"Cyclic named shape {name!r}; use a bounded recursive rule instead")
            if expansion_depth >= self.max_expansion_depth:
                self.stats["pruned_recursion"] += 1
                return
            for index, alternative in self._alternatives(self.shapes[name]):
                self._step()
                expr, bound = self._instantiate(alternative, context, ("shape", name, index))
                for result, updated in self._node(expr, bound, ("shape", name, index), expansion_depth + 1, active_shapes | {name}):
                    if alternative.tags:
                        result = Node("tag", alternative.tags, (result,))
                    shapes = dict(updated.shapes)
                    shapes[name] = result
                    yield result, _Context(shapes, updated.locals)
            return
        if node.op == "hole":
            if expansion_depth >= self.max_expansion_depth:
                self.stats["pruned_recursion"] += 1
                return
            rule = self.rules[node.value]
            for arguments, arg_context in self._children(node.children, context, path + ("args",), expansion_depth, active_shapes):
                for index, alternative in self._alternatives(rule.alternatives):
                    self._step()
                    alt_path = path + ("rule", rule.name, index)
                    expr, bound = self._instantiate(alternative, arg_context, alt_path)
                    expr = substitute(expr, dict(zip(rule.parameters, arguments)))
                    for result, updated in self._node(expr, bound, alt_path, expansion_depth + 1, active_shapes):
                        if alternative.tags:
                            result = Node("tag", alternative.tags, (result,))
                        yield result, updated
            return
        if not node.children:
            yield node, context
            return
        for children, updated in self._children(node.children, context, path, expansion_depth, active_shapes):
            result = Node(node.op, node.value, children)
            if self._valid(result):
                yield result, updated

    def _system(self, expressions, context, index=0, selected=None, count=0):
        selected = {} if selected is None else selected
        if index == len(expressions):
            yield ExpandedSystem(dict(selected), {k: dict(v) for k, v in context.locals.items()})
            return
        state, expression = expressions[index]
        for expanded, updated in self._node(expression, context, ("rhs", state), 0, frozenset()):
            size = node_count(expanded)
            if count + size > self.max_nodes or node_depth(expanded) > self.max_depth:
                self.stats["pruned_size"] += 1
                continue
            selected[state] = expanded
            yield from self._system(expressions, updated, index + 1, selected, count + size)
            del selected[state]

    def expand(self, rhs, strategy="enumerate", seed=0, max_derivations=1000000) -> Iterator[ExpandedSystem]:
        if strategy not in ("enumerate", "sample"):
            raise GrammarError("Expansion strategy must be 'enumerate' or 'sample'")
        _positive_int(max_derivations, "max_derivations", allow_zero=True)
        if type(seed) is not int:
            raise GrammarError("Expansion seed must be an integer")
        if not isinstance(rhs, dict) or not rhs:
            raise GrammarError("rhs must be a non-empty state-to-expression object")
        expressions = []
        for state, value in rhs.items():
            if not isinstance(state, str) or not isinstance(value, (str, Node)):
                raise GrammarError("rhs maps state names to expression strings or Nodes")
            expr = parse_expr(value) if isinstance(value, str) else value
            self._validate_references(expr)
            expressions.append((state, expr))
        self._strategy, self._random = strategy, random.Random(seed)
        self.stats = {"steps": 0, "derivations": 0, "attempts": 0, "pruned_size": 0, "pruned_recursion": 0}
        self.stop_reason = None
        if not max_derivations:
            self.stop_reason = "max_derivations"
            return
        try:
            if strategy == "sample":
                for _ in range(max_derivations):
                    self.stats["attempts"] += 1
                    for system in self._system(expressions, _Context()):
                        self.stats["derivations"] += 1
                        yield system
                self.stop_reason = "max_derivations"
            else:
                for system in self._system(expressions, _Context()):
                    self.stats["attempts"] += 1
                    self.stats["derivations"] += 1
                    yield system
                    if self.stats["derivations"] >= max_derivations:
                        # We intentionally do not generate one extra candidate
                        # to discover whether the cap coincides with exhaustion:
                        # that lookahead could traverse a large invalid space.
                        self.stop_reason = "max_derivations"
                        return
                self.stop_reason = "exhausted"
        except _StepLimit:
            self.stop_reason = "max_expansion_steps"
        except _DeadlineLimit:
            self.stop_reason = "max_seconds"
        except RecursionError as exc:
            raise GrammarError("Expansion exceeded Python's recursion limit; lower max_expansion_depth/max_depth or simplify the grammar") from exc
