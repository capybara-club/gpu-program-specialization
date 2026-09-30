#!/usr/bin/env python3
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

import argparse
import filecmp
import functools
import itertools
import json
import math
from pathlib import Path
import sys

from corpus import PROFILE_OPERATIONS, SCHEMA


NUM_INPUTS = 8
MASK64 = (1 << 64) - 1
ALU_BINARY = ("add", "mul", "min", "max")
MIXED_BINARY = ("add", "sub", "mul", "min", "max")
MUFU_UNARY = ("sin", "cos", "tanh", "safe_sqrt", "safe_rsqrt")
SAFE_BINARY = ("add", "mul", "min", "max", "safe_div")
SAFE_UNARY = ("abs", "tanh", "safe_sqrt", "safe_rsqrt")
CONSTANTS = (-2.0, -1.0, -0.5, -0.125, 0.125, 0.5, 1.0, 2.0)


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + 0x9e3779b97f4a7c15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK64
        return value ^ (value >> 31)


def make_input_layout(leaves):
    def build(first, count):
        if count == 1:
            return leaves[first]
        left_count = count // 2
        left = build(first, left_count)
        right = build(first + left_count, count - left_count)
        if left > right:
            left, right = right, left
        return left, right

    return build(0, NUM_INPUTS)


@functools.lru_cache(maxsize=1)
def make_input_layouts():
    return sorted({
        make_input_layout(permutation)
        for permutation in itertools.permutations(range(NUM_INPUTS))
    })


def make_balanced_shape(num_leaves):
    if num_leaves == 1:
        return None
    left_leaves = num_leaves // 2
    return (
        make_balanced_shape(left_leaves),
        make_balanced_shape(num_leaves - left_leaves),
    )


@functools.lru_cache(maxsize=None)
def make_binary_shapes(num_leaves):
    if num_leaves == 1:
        return (None,)
    return tuple(
        (left, right)
        for left_leaves in range(1, num_leaves)
        for left in make_binary_shapes(left_leaves)
        for right in make_binary_shapes(num_leaves - left_leaves)
    )


def decode_choices(code, count, choices):
    decoded = []
    for _ in range(count):
        decoded.append(choices[code % len(choices)])
        code //= len(choices)
    return decoded, code


def build_from_input_layout(layout, unary_operations, binary_operations):
    unary_idx = 0
    binary_idx = 0

    def build(node):
        nonlocal unary_idx, binary_idx
        if isinstance(node, int):
            expression = node
            if unary_operations:
                expression = [unary_operations[unary_idx], expression]
                unary_idx += 1
            return expression
        left = build(node[0])
        right = build(node[1])
        operation = binary_operations[binary_idx]
        binary_idx += 1
        return [operation, left, right]

    return build(layout)


def build_from_shape(shape, leaves, binary_operations):
    leaf_idx = 0
    binary_idx = 0

    def build(node):
        nonlocal leaf_idx, binary_idx
        if node is None:
            expression = leaves[leaf_idx]
            leaf_idx += 1
            return expression
        left = build(node[0])
        right = build(node[1])
        operation = binary_operations[binary_idx]
        binary_idx += 1
        return [operation, left, right]

    return build(shape)


def balanced_profile_space(num_unary, num_binary):
    return (
        len(make_input_layouts()) *
        num_unary ** NUM_INPUTS *
        num_binary ** (NUM_INPUTS - 1)
    )


def constants_profile_space():
    leaf_values = NUM_INPUTS + len(CONSTANTS)
    valid_leaves = (
        leaf_values ** NUM_INPUTS -
        NUM_INPUTS ** NUM_INPUTS -
        len(CONSTANTS) ** NUM_INPUTS
    )
    return valid_leaves * len(ALU_BINARY) ** (NUM_INPUTS - 1)


@functools.lru_cache(maxsize=1)
def mixed_profile_ranges():
    endings = []
    total = 0
    for num_leaves in range(2, NUM_INPUTS + 1):
        total += (
            len(make_binary_shapes(num_leaves)) *
            NUM_INPUTS ** num_leaves *
            len(MIXED_BINARY) ** (num_leaves - 1)
        )
        endings.append(total)
    return endings


def profile_space(profile):
    if profile == "balanced-alu":
        return balanced_profile_space(1, len(ALU_BINARY))
    if profile == "balanced-mufu":
        return balanced_profile_space(len(MUFU_UNARY), len(ALU_BINARY))
    if profile == "safe-math":
        return balanced_profile_space(len(SAFE_UNARY), len(SAFE_BINARY))
    if profile == "constants":
        return constants_profile_space()
    if profile == "mixed-depth":
        return mixed_profile_ranges()[-1]
    raise ValueError(f"unknown profile: {profile}")


def decode_balanced(combination, unary_choices, binary_choices):
    layouts = make_input_layouts()
    layout = layouts[combination % len(layouts)]
    combination //= len(layouts)
    unary_operations, combination = decode_choices(
        combination,
        NUM_INPUTS if unary_choices else 0,
        unary_choices or (None,),
    )
    binary_operations, combination = decode_choices(
        combination,
        NUM_INPUTS - 1,
        binary_choices,
    )
    if combination != 0:
        raise RuntimeError("balanced expression code exceeds its profile")
    return build_from_input_layout(
        layout,
        unary_operations,
        binary_operations,
    )


def decode_constants(combination):
    leaf_choices = tuple(range(NUM_INPUTS)) + tuple(
        ["constant", value] for value in CONSTANTS)
    candidate_space = (
        len(leaf_choices) ** NUM_INPUTS *
        len(ALU_BINARY) ** (NUM_INPUTS - 1)
    )
    candidate = combination % candidate_space
    leaves, candidate = decode_choices(candidate, NUM_INPUTS, leaf_choices)
    binary_operations, candidate = decode_choices(
        candidate,
        NUM_INPUTS - 1,
        ALU_BINARY,
    )
    if candidate != 0:
        raise RuntimeError("constant expression code exceeds its profile")
    has_input = any(isinstance(leaf, int) for leaf in leaves)
    has_constant = any(isinstance(leaf, list) for leaf in leaves)
    if not has_input or not has_constant:
        return None
    return build_from_shape(
        make_balanced_shape(NUM_INPUTS),
        leaves,
        binary_operations,
    )


def mixed_space(num_leaves):
    return (
        len(make_binary_shapes(num_leaves)) *
        NUM_INPUTS ** num_leaves *
        len(MIXED_BINARY) ** (num_leaves - 1)
    )


def decode_mixed(num_leaves, combination):
    shapes = make_binary_shapes(num_leaves)
    shape = shapes[combination % len(shapes)]
    combination //= len(shapes)
    leaves, combination = decode_choices(
        combination,
        num_leaves,
        tuple(range(NUM_INPUTS)),
    )
    binary_operations, combination = decode_choices(
        combination,
        num_leaves - 1,
        MIXED_BINARY,
    )
    if combination != 0:
        raise RuntimeError("mixed-depth expression code exceeds its profile")
    return build_from_shape(shape, leaves, binary_operations)


def decode_expression(profile, combination):
    if profile == "balanced-alu":
        return decode_balanced(combination, (), ALU_BINARY)
    if profile == "balanced-mufu":
        return decode_balanced(combination, MUFU_UNARY, ALU_BINARY)
    if profile == "safe-math":
        return decode_balanced(combination, SAFE_UNARY, SAFE_BINARY)
    if profile == "constants":
        return decode_constants(combination)
    raise ValueError(f"unknown profile: {profile}")


def candidate_space(profile):
    if profile != "constants":
        return profile_space(profile)
    return (
        (NUM_INPUTS + len(CONSTANTS)) ** NUM_INPUTS *
        len(ALU_BINARY) ** (NUM_INPUTS - 1)
    )


def affine_parameters(total, seed):
    random = SplitMix64(seed)
    offset = random.next() % total
    stride = random.next() % (total - 1) + 1
    while math.gcd(stride, total) != 1:
        stride = random.next() % (total - 1) + 1
    return offset, stride


def expression_strings(profile, count, seed):
    if profile == "mixed-depth":
        yield from mixed_expression_strings(count, seed)
        return
    total = candidate_space(profile)
    offset, stride = affine_parameters(total, seed)
    emitted = 0
    candidate_idx = 0
    while emitted < count and candidate_idx < total:
        combination = (offset + candidate_idx * stride) % total
        candidate_idx += 1
        expression = decode_expression(profile, combination)
        if expression is None:
            continue
        emitted += 1
        yield json.dumps(expression, separators=(",", ":"), ensure_ascii=True)
    if emitted != count:
        raise RuntimeError(
            f"profile exhausted after generating {emitted} expressions")


def mixed_expression_strings(count, seed):
    leaf_counts = list(range(2, NUM_INPUTS + 1))
    rotation = seed % len(leaf_counts)
    leaf_counts = leaf_counts[rotation:] + leaf_counts[:rotation]
    states = {}
    for num_leaves in leaf_counts:
        total = mixed_space(num_leaves)
        offset, stride = affine_parameters(
            total,
            seed ^ (num_leaves * 0x9e3779b97f4a7c15),
        )
        states[num_leaves] = [0, total, offset, stride]

    emitted = 0
    while emitted < count and leaf_counts:
        next_counts = []
        for num_leaves in leaf_counts:
            if emitted == count:
                break
            state = states[num_leaves]
            sequence_idx, total, offset, stride = state
            if sequence_idx >= total:
                continue
            combination = (offset + sequence_idx * stride) % total
            state[0] += 1
            expression = decode_mixed(num_leaves, combination)
            emitted += 1
            yield json.dumps(
                expression,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            if state[0] < total:
                next_counts.append(num_leaves)
        leaf_counts = next_counts
    if emitted != count:
        raise RuntimeError(
            f"mixed-depth profile exhausted after {emitted} expressions")


def profile_uniqueness(profile):
    return (
        "commutative"
        if profile in ("balanced-alu", "balanced-mufu", "safe-math")
        else "serialized"
    )


def write_corpus(path, profile, count, seed, name):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("{\n")
        file.write(f'  "schema": {json.dumps(SCHEMA)},\n')
        file.write(f'  "name": {json.dumps(name)},\n')
        file.write(f'  "profile": {json.dumps(profile)},\n')
        file.write(f'  "num_inputs": {NUM_INPUTS},\n')
        file.write('  "input_generator": "secant_hash32_f32_v1",\n')
        file.write('  "target_expression": 0,\n')
        file.write(
            f'  "operations": '
            f'{json.dumps(list(PROFILE_OPERATIONS[profile]))},\n')
        file.write('  "generation": {\n')
        file.write(
            '    "algorithm": '
            '"splitmix64_affine_profile_permutation_v1",\n')
        file.write(f'    "profile": {json.dumps(profile)},\n')
        file.write(
            f'    "uniqueness": '
            f'{json.dumps(profile_uniqueness(profile))},\n')
        file.write(f'    "seed": {seed}\n')
        file.write('  },\n')
        file.write('  "expressions": [\n')
        for expression_idx, expression in enumerate(
                expression_strings(profile, count, seed)):
            suffix = "," if expression_idx + 1 < count else ""
            file.write(f"    {expression}{suffix}\n")
        file.write("  ]\n")
        file.write("}\n")


def parse_nonnegative_int(text):
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def parse_positive_int(text):
    value = int(text, 0)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate deterministic expression corpora.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_OPERATIONS),
        default="balanced-alu",
    )
    parser.add_argument("--count", type=parse_positive_int, default=1024)
    parser.add_argument("--seed", type=parse_nonnegative_int, default=1)
    parser.add_argument("--name")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    maximum = profile_space(args.profile)
    if args.count > maximum:
        parser.error(
            f"--count exceeds the {maximum} expressions in "
            f"the {args.profile} profile")
    if args.seed > MASK64:
        parser.error("--seed must fit in uint64")
    if args.name is None:
        profile_name = args.profile.replace("-", "_")
        args.name = (
            f"portable_{profile_name}_{NUM_INPUTS}x{args.count}"
            f"_seed{args.seed}_v1")
    if not args.name:
        parser.error("--name must not be empty")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        write_corpus(
            temporary,
            args.profile,
            args.count,
            args.seed,
            args.name,
        )
        if args.check:
            if (not args.output.is_file() or
                    not filecmp.cmp(temporary, args.output, shallow=False)):
                raise RuntimeError(f"generated corpus is stale: {args.output}")
            temporary.unlink()
        else:
            temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"{args.output}: profile={args.profile} expressions={args.count} "
        f"inputs={NUM_INPUTS} seed={args.seed} name={args.name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
