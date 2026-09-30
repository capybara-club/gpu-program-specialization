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
import hashlib
import json
import math
from pathlib import Path
import struct


SCHEMA = "secant-benchmark-corpus-v1"
DEFAULT_CORPUS = Path(__file__).with_name("portable_alu_v1.json")
SAFE_EPSILON = 0.01

BINARY_OPERATIONS = (
    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
    "safe_div",
)
UNARY_OPERATIONS = (
    "neg",
    "abs",
    "sin",
    "cos",
    "ex2",
    "lg2",
    "sqrt",
    "rsqrt",
    "tanh",
    "safe_sqrt",
    "safe_rsqrt",
)
COMMUTATIVE_OPERATIONS = frozenset(("add", "mul", "min", "max"))
PROFILE_OPERATIONS = {
    "balanced-alu": ("add", "mul", "min", "max"),
    "balanced-mufu": (
        "add",
        "mul",
        "min",
        "max",
        "sin",
        "cos",
        "tanh",
        "safe_sqrt",
        "safe_rsqrt",
    ),
    "mixed-depth": ("add", "sub", "mul", "min", "max"),
    "constants": ("add", "mul", "min", "max"),
    "safe-math": (
        "add",
        "mul",
        "min",
        "max",
        "safe_div",
        "abs",
        "tanh",
        "safe_sqrt",
        "safe_rsqrt",
    ),
}


def _merge_stats(operation, child_stats):
    inputs = []
    constants = 0
    operations = {}
    leaves = 0
    nodes = 1
    depth = 0
    for stats in child_stats:
        inputs.extend(stats["inputs"])
        constants += stats["constants"]
        leaves += stats["leaves"]
        nodes += stats["nodes"]
        depth = max(depth, stats["depth"])
        for name, count in stats["operations"].items():
            operations[name] = operations.get(name, 0) + count
    operations[operation] = operations.get(operation, 0) + 1
    return {
        "inputs": inputs,
        "constants": constants,
        "operations": operations,
        "leaves": leaves,
        "nodes": nodes,
        "depth": depth + 1,
    }


def _validate_expression(expression, num_inputs):
    if isinstance(expression, int) and not isinstance(expression, bool):
        if expression < 0 or expression >= num_inputs:
            raise ValueError(f"input index is out of range: {expression}")
        return {
            "inputs": [expression],
            "constants": 0,
            "operations": {},
            "leaves": 1,
            "nodes": 1,
            "depth": 0,
        }
    if not isinstance(expression, list) or not expression:
        raise ValueError(f"invalid expression node: {expression!r}")

    operation = expression[0]
    if operation == "constant":
        if (len(expression) != 2 or
                isinstance(expression[1], bool) or
                not isinstance(expression[1], (int, float)) or
                not math.isfinite(expression[1])):
            raise ValueError(f"invalid f32 constant: {expression!r}")
        try:
            struct.pack("<f", expression[1])
        except OverflowError as error:
            raise ValueError(f"constant does not fit in f32: {expression[1]}") from error
        return {
            "inputs": [],
            "constants": 1,
            "operations": {},
            "leaves": 1,
            "nodes": 1,
            "depth": 0,
        }
    if operation in UNARY_OPERATIONS:
        if len(expression) != 2:
            raise ValueError(f"unary operation has the wrong arity: {expression!r}")
        return _merge_stats(
            operation,
            (_validate_expression(expression[1], num_inputs),),
        )
    if operation in BINARY_OPERATIONS:
        if len(expression) != 3:
            raise ValueError(f"binary operation has the wrong arity: {expression!r}")
        return _merge_stats(
            operation,
            (
                _validate_expression(expression[1], num_inputs),
                _validate_expression(expression[2], num_inputs),
            ),
        )
    raise ValueError(f"unsupported corpus operation: {operation}")


def _expression_key(expression, canonicalize_commutative):
    if isinstance(expression, int):
        return f"i{expression}"
    operation = expression[0]
    if operation == "constant":
        bits = struct.unpack("<I", struct.pack("<f", expression[1]))[0]
        return f"c{bits:08x}"
    arguments = [
        _expression_key(argument, canonicalize_commutative)
        for argument in expression[1:]
    ]
    if canonicalize_commutative and operation in COMMUTATIVE_OPERATIONS:
        arguments.sort()
    return f"{operation}({','.join(arguments)})"


def _base_shape(expression):
    if isinstance(expression, int) or expression[0] == "constant":
        return None
    if expression[0] in UNARY_OPERATIONS:
        return _base_shape(expression[1])
    return _base_shape(expression[1]), _base_shape(expression[2])


def _balanced_shape(num_leaves):
    if num_leaves == 1:
        return None
    left_leaves = num_leaves // 2
    return (
        _balanced_shape(left_leaves),
        _balanced_shape(num_leaves - left_leaves),
    )


def _validate_profile(profile, stats, expression):
    operations = stats["operations"]
    binary_count = sum(operations.get(name, 0) for name in BINARY_OPERATIONS)
    unary_count = sum(operations.get(name, 0) for name in UNARY_OPERATIONS)

    if profile in ("balanced-alu", "balanced-mufu", "safe-math"):
        if (sorted(stats["inputs"]) != list(range(8)) or
                stats["constants"] != 0 or
                stats["leaves"] != 8 or
                binary_count != 7 or
                _base_shape(expression) != _balanced_shape(8)):
            raise ValueError(
                f"{profile} expressions must be balanced reductions over "
                "inputs 0 through 7 exactly once")
        expected_unary = 0 if profile == "balanced-alu" else 8
        if unary_count != expected_unary:
            raise ValueError(
                f"{profile} expressions require {expected_unary} unary operations")
        return
    if profile == "constants":
        if (stats["leaves"] != 8 or
                stats["constants"] == 0 or
                not stats["inputs"] or
                binary_count != 7 or
                unary_count != 0 or
                _base_shape(expression) != _balanced_shape(8)):
            raise ValueError(
                "constants expressions require a balanced eight-leaf tree "
                "containing both inputs and constants")
        return
    if profile == "mixed-depth":
        if (stats["leaves"] < 2 or
                stats["leaves"] > 8 or
                stats["constants"] != 0 or
                binary_count != stats["leaves"] - 1 or
                unary_count != 0):
            raise ValueError(
                "mixed-depth expressions require two to eight input leaves "
                "and binary operations only")
        return
    raise ValueError(f"unsupported benchmark corpus profile: {profile}")


def load_corpus(path=DEFAULT_CORPUS):
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        corpus = json.load(file)
    if corpus.get("schema") != SCHEMA:
        raise ValueError("unsupported benchmark corpus schema")
    if not isinstance(corpus.get("name"), str) or not corpus["name"]:
        raise ValueError("benchmark corpus has no name")
    profile = corpus.get("profile")
    if profile not in PROFILE_OPERATIONS:
        raise ValueError("benchmark corpus profile is invalid")
    num_inputs = corpus.get("num_inputs")
    expressions = corpus.get("expressions")
    if not isinstance(num_inputs, int) or num_inputs != 8:
        raise ValueError("benchmark corpus input count must be eight")
    if corpus.get("operations") != list(PROFILE_OPERATIONS[profile]):
        raise ValueError("benchmark corpus operation list is not canonical")
    if corpus.get("input_generator") != "secant_hash32_f32_v1":
        raise ValueError("benchmark corpus input generator is not canonical")
    if not isinstance(expressions, list) or not expressions:
        raise ValueError("benchmark corpus has no expressions")

    generation = corpus.get("generation")
    if (not isinstance(generation, dict) or
            generation.get("algorithm") !=
                "splitmix64_affine_profile_permutation_v1" or
            generation.get("profile") != profile or
            generation.get("uniqueness") not in
                ("commutative", "serialized") or
            not isinstance(generation.get("seed"), int) or
            generation["seed"] < 0 or
            generation["seed"] > 0xffffffffffffffff):
        raise ValueError("benchmark corpus generation metadata is invalid")

    target_expression = corpus.get("target_expression")
    if (not isinstance(target_expression, int) or
            target_expression < 0 or
            target_expression >= len(expressions)):
        raise ValueError("benchmark corpus target expression is invalid")

    canonicalize = generation["uniqueness"] == "commutative"
    encoded_expressions = set()
    allowed_operations = frozenset(PROFILE_OPERATIONS[profile])
    for expression in expressions:
        stats = _validate_expression(expression, num_inputs)
        if not frozenset(stats["operations"]).issubset(allowed_operations):
            raise ValueError(
                f"expression uses operations outside the {profile} profile")
        _validate_profile(profile, stats, expression)
        encoded = _expression_key(expression, canonicalize)
        if encoded in encoded_expressions:
            raise ValueError(
                f"benchmark corpus expressions must be unique by "
                f"{generation['uniqueness']} identity")
        encoded_expressions.add(encoded)

    canonical = json.dumps(
        corpus,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    corpus["hash"] = hashlib.sha256(canonical).hexdigest()[:16]
    corpus["path"] = path
    return corpus


def postorder(expression):
    if isinstance(expression, int):
        return [("input", expression)]
    operation = expression[0]
    if operation == "constant":
        return [("constant", expression[1])]
    instructions = []
    for argument in expression[1:]:
        instructions.extend(postorder(argument))
    instructions.append((operation, None))
    return instructions


def lower_safe_operations(expression):
    if isinstance(expression, int):
        return expression
    operation = expression[0]
    if operation == "constant":
        return expression
    arguments = [
        lower_safe_operations(argument)
        for argument in expression[1:]
    ]
    epsilon = ["constant", SAFE_EPSILON]
    if operation == "neg":
        return ["mul", ["constant", -1.0], arguments[0]]
    if operation == "safe_div":
        return [
            "div",
            ["mul", arguments[0], arguments[1]],
            ["add", ["mul", arguments[1], arguments[1]], epsilon],
        ]
    if operation == "safe_sqrt":
        return ["sqrt", ["add", ["abs", arguments[0]], epsilon]]
    if operation == "safe_rsqrt":
        return [
            "div",
            ["constant", 1.0],
            ["sqrt", ["add", ["abs", arguments[0]], epsilon]],
        ]
    return [operation, *arguments]


def _julia_f32(value):
    text = format(float(value), ".9g").lower()
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}f{int(exponent)}"
    if "." not in text:
        text += ".0"
    return f"{text}f0"


def julia_expression(expression):
    if isinstance(expression, int):
        return f"x{expression + 1}"
    operation = expression[0]
    if operation == "constant":
        return _julia_f32(expression[1])
    arguments = [julia_expression(argument) for argument in expression[1:]]
    if operation == "add":
        return f"({arguments[0]} + {arguments[1]})"
    if operation == "sub":
        return f"({arguments[0]} - {arguments[1]})"
    if operation == "mul":
        return f"({arguments[0]} * {arguments[1]})"
    if operation == "div":
        return f"({arguments[0]} / {arguments[1]})"
    if operation in ("min", "max"):
        return f"{operation}({arguments[0]}, {arguments[1]})"
    if operation == "safe_div":
        return f"secant_safe_div({arguments[0]}, {arguments[1]})"
    if operation == "neg":
        return f"(-{arguments[0]})"
    if operation in ("abs", "sin", "cos", "sqrt", "tanh"):
        return f"{operation}({arguments[0]})"
    if operation == "ex2":
        return f"exp2({arguments[0]})"
    if operation == "lg2":
        return f"log2({arguments[0]})"
    if operation == "rsqrt":
        return f"inv(sqrt({arguments[0]}))"
    if operation == "safe_sqrt":
        return f"secant_safe_sqrt({arguments[0]})"
    if operation == "safe_rsqrt":
        return f"secant_safe_rsqrt({arguments[0]})"
    raise ValueError(f"unsupported Julia operation: {operation}")


def hash32(value):
    value &= 0xffffffff
    value ^= value >> 16
    value = (value * 0x7feb352d) & 0xffffffff
    value ^= value >> 15
    value = (value * 0x846ca68b) & 0xffffffff
    value ^= value >> 16
    return value


def input_value(column, row, seed):
    bits = hash32(
        (column * 0x9e3779b9) ^
        (row * 0x85ebca6b) ^
        (seed * 0xc2b2ae35) ^
        0x51ed270b)
    value = struct.unpack("<f", struct.pack("<f", (bits & 0xffff) / 32767.5))[0]
    value = struct.unpack("<f", struct.pack("<f", value - 1.0))[0]
    return struct.unpack("<f", struct.pack("<f", value * 1.5))[0]


def make_numpy_input(numpy, corpus, rows, seed):
    columns = []
    row_indices = numpy.arange(rows, dtype=numpy.uint64)
    mask32 = numpy.uint64(0xffffffff)
    for column in range(corpus["num_inputs"]):
        values = numpy.asarray((
            numpy.uint64(column) * numpy.uint64(0x9e3779b9) ^
            row_indices * numpy.uint64(0x85ebca6b) ^
            numpy.uint64(seed) * numpy.uint64(0xc2b2ae35) ^
            numpy.uint64(0x51ed270b)
        ) & mask32, dtype=numpy.uint32)
        values ^= values >> numpy.uint32(16)
        values = numpy.asarray(
            values.astype(numpy.uint64) * numpy.uint64(0x7feb352d) & mask32,
            dtype=numpy.uint32)
        values ^= values >> numpy.uint32(15)
        values = numpy.asarray(
            values.astype(numpy.uint64) * numpy.uint64(0x846ca68b) & mask32,
            dtype=numpy.uint32)
        values ^= values >> numpy.uint32(16)
        column_values = (
            (values & numpy.uint32(0xffff)).astype(numpy.float32) /
            numpy.float32(32767.5) -
            numpy.float32(1.0)
        ) * numpy.float32(1.5)
        columns.append(column_values)
    return numpy.stack(columns, axis=1)


def make_torch_input(torch, corpus, rows, seed, device):
    row_indices = torch.arange(rows, dtype=torch.int64, device=device)
    columns = []
    for column in range(corpus["num_inputs"]):
        values = (
            column * 0x9e3779b9 ^
            row_indices * 0x85ebca6b ^
            seed * 0xc2b2ae35 ^
            0x51ed270b
        ) & 0xffffffff
        values ^= values >> 16
        values = (values * 0x7feb352d) & 0xffffffff
        values ^= values >> 15
        values = (values * 0x846ca68b) & 0xffffffff
        values ^= values >> 16
        column_values = (
            (values & 0xffff).to(torch.float32) /
            torch.tensor(32767.5, dtype=torch.float32, device=device) -
            1.0
        ) * 1.5
        columns.append(column_values)
    return torch.stack(columns, dim=1)


def evaluate_numpy(numpy, expression, values):
    if isinstance(expression, int):
        return values[:, expression]
    operation = expression[0]
    if operation == "constant":
        return numpy.full(values.shape[0], expression[1], dtype=numpy.float32)
    arguments = [
        evaluate_numpy(numpy, argument, values)
        for argument in expression[1:]
    ]
    if operation == "add":
        return arguments[0] + arguments[1]
    if operation == "sub":
        return arguments[0] - arguments[1]
    if operation == "mul":
        return arguments[0] * arguments[1]
    if operation == "div":
        return arguments[0] / arguments[1]
    if operation == "min":
        return numpy.minimum(arguments[0], arguments[1])
    if operation == "max":
        return numpy.maximum(arguments[0], arguments[1])
    if operation == "safe_div":
        return (
            arguments[0] * arguments[1] /
            (arguments[1] * arguments[1] + numpy.float32(SAFE_EPSILON))
        )
    if operation == "neg":
        return -arguments[0]
    if operation == "abs":
        return numpy.abs(arguments[0])
    if operation == "sin":
        return numpy.sin(arguments[0])
    if operation == "cos":
        return numpy.cos(arguments[0])
    if operation == "ex2":
        return numpy.exp2(arguments[0])
    if operation == "lg2":
        return numpy.log2(arguments[0])
    if operation == "sqrt":
        return numpy.sqrt(arguments[0])
    if operation == "rsqrt":
        return numpy.float32(1.0) / numpy.sqrt(arguments[0])
    if operation == "tanh":
        return numpy.tanh(arguments[0])
    if operation == "safe_sqrt":
        return numpy.sqrt(
            numpy.abs(arguments[0]) + numpy.float32(SAFE_EPSILON))
    if operation == "safe_rsqrt":
        return numpy.float32(1.0) / numpy.sqrt(
            numpy.abs(arguments[0]) + numpy.float32(SAFE_EPSILON))
    raise ValueError(f"unsupported NumPy operation: {operation}")
