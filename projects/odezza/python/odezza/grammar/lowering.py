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
"""Lossless grammar-to-native lowering and explicit configuration addressing."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from odegrammar.compiler import content_id


class CompatibilityError(ValueError):
    """Valid grammar that cannot execute under the selected backend contract."""


def f32(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CompatibilityError("Expected a finite numeric FP32 value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        raise CompatibilityError("Value exceeds FP32 range") from None
    if not math.isfinite(result):
        raise CompatibilityError("Value exceeds FP32 range")
    return result


OPS = {"ADD": 0x90, "SUB": 0x91, "MUL": 0x92, "DIV": 0x93,
       "NEG": 0x94, "SQRT": 0x95, "ABS": 0x97, "SIN": 0x9b,
       "COS": 0x9c, "TANH": 0xa0, "EXP": 0xa1, "LOG": 0xa2}
NAMES = {"CONSTANT": "const", "RNG_VALUE": "rng", "PARAMETER": "param"}


def literal(value):
    return bytes([0x83]) + struct.pack("<f", f32(value))


@dataclass
class Layout:
    slots: list[tuple[str, str]]
    bits: dict[str, tuple[int, ...]]
    numeric_count: int
    permutation_count: int

    @classmethod
    def from_pools(cls, pools):
        slots = [(namespace, name) for namespace, section in
                 (("const", "constants"), ("rng", "rng_bindings"), ("param", "parameter_initials"))
                 for name in sorted(pools[section])]
        bits, shift, numeric = {}, 0, 1
        for axis in reversed(pools["pool_axes"]):
            if axis["kind"] == "leaf":
                width = {2: 1, 4: 2}.get(axis["count"])
                if width is None:
                    raise CompatibilityError("Native toggles require exactly two or four states")
                bits[axis["slots"][0]] = tuple(range(shift, shift + width))
                shift += width
            else:
                numeric *= axis["count"]
        if shift > 32:
            raise CompatibilityError("Grammar requires more than 32 native toggle bits")
        if numeric * (1 << shift) > 2**64 - 1:
            raise CompatibilityError("Configuration product exceeds the native uint64 address space")
        if len(slots) > 253:
            raise CompatibilityError("Too many native coefficient slots")
        return cls(slots, bits, numeric, 1 << shift)

    @property
    def toggle_bits(self):
        return sum(map(len, self.bits.values()))

    def grammar_index(self, bank, permutation):
        if not 0 <= bank < self.numeric_count or not 0 <= permutation < self.permutation_count:
            raise IndexError("Configuration outside variant")
        return permutation * self.numeric_count + bank

    def native_index(self, grammar_index):
        permutation, bank = divmod(grammar_index, self.numeric_count)
        self.grammar_index(bank, permutation)
        return bank * self.permutation_count + permutation

    def bindings(self, pools, permutation):
        return {name: pools["toggles"][name][(permutation >> bits[0]) & ((1 << len(bits)) - 1)]
                for name, bits in self.bits.items()}


def lower(program, pools, layout, *, fixed=None, fitted=None, permutation=None):
    """Fixed values become literals for replay/LM; fitted names remain slots."""
    slots = {pair: i for i, pair in enumerate(layout.slots)}
    fitted = {} if fitted is None else fitted
    stack = []
    for instruction in program:
        op = instruction["op"]
        if op == "TIME":
            raise CompatibilityError("TIME/t is not supported by the native scoring or LM ABI")
        if op == "LITERAL":
            stack.append(literal(instruction["value"]))
        elif op == "STATE":
            stack.append(bytes([0x81, instruction["index"]]))
        elif op == "TOGGLE":
            name = instruction["slot"]
            if permutation is not None:
                stack.append(bytes([0x81, layout.bindings(pools, permutation)[name]]))
            else:
                leaves = b"".join(bytes([0x81, s]) for s in pools["toggles"][name])
                stack.append(leaves + bytes([0x84 if instruction["arity"] == 2 else 0x85, *layout.bits[name]]))
        elif op in NAMES:
            pair = (NAMES[op], instruction["slot"])
            if pair in fitted:
                stack.append(bytes([0x82, fitted[pair]]))
            elif fixed is not None:
                stack.append(literal(fixed[pair]))
            else:
                stack.append(bytes([0x82, slots[pair]]))
        elif op == "POWI":
            exponent = instruction["exponent"]
            if abs(exponent) > 16:
                raise CompatibilityError("Integer powers beyond [-16,16] exceed this adapter's lowering budget")
            arg = stack.pop()
            value = literal(1) if exponent == 0 else arg
            for _ in range(1, abs(exponent)):
                value += arg + bytes([OPS["MUL"]])
            stack.append(literal(1) + value + bytes([OPS["DIV"]]) if exponent < 0 else value)
        elif op in OPS:
            right = stack.pop()
            left = stack.pop() if op in ("ADD", "SUB", "MUL", "DIV") else b""
            stack.append(left + right + bytes([OPS[op]]))
        else:
            raise CompatibilityError("Unsupported instruction: " + str(op))
        if stack and len(stack[-1]) > 65536:
            raise CompatibilityError("Expanded native RHS exceeds 64 KiB")
    if len(stack) != 1:
        raise CompatibilityError("Malformed postorder program")
    return stack[0] + bytes([0x80])


def lower_system(skeleton, variant, states, **kwargs):
    layout = Layout.from_pools(variant["pools"])
    validate_pools(variant["pools"])
    if 2 * len(states) + len(layout.slots) + 1 >= 255:
        raise CompatibilityError("Native marker ABI requires 2*states + coefficient_slots + 1 < 255")
    return layout, [lower(skeleton["rhs"][s], variant["pools"], layout, **kwargs) for s in states]


def validate_pools(pools):
    ranges = {}
    for name, spec in pools["constants"].items():
        values = [f32(spec["value"])] if spec["kind"] == "fixed" else [f32(x) for x in spec["values"]]
        ranges[name] = min(values), max(values)
    for value in pools["parameter_initials"].values(): f32(value)
    for binding in pools["rng_bindings"].values():
        transform = binding["transform"]
        operands = {k: ranges[v["const"]] if isinstance(v, dict) else (f32(v), f32(v))
                    for k, v in transform.items() if k != "kind"}
        kind = transform["kind"]
        if kind == "normal" and operands["std"][0] <= 0:
            raise CompatibilityError("Normal standard deviation must remain positive in FP32")
        if kind in ("uniform", "log_uniform") and operands["low"][1] >= operands["high"][0]:
            raise CompatibilityError("RNG bounds collapse or reverse after FP32 conversion")
        if kind == "log_uniform" and operands["low"][0] <= 0:
            raise CompatibilityError("Log-uniform lower bound must remain positive in FP32")


def identities(skeleton, variant, states, layout, values, permutation):
    fixed = dict(zip(layout.slots, values))
    resolved = [lower(skeleton["rhs"][s], variant["pools"], layout,
                      fixed=fixed, permutation=permutation).hex() for s in states]
    # Structure retains parameter sharing, but removes user slot spelling and values.
    structure = [lower(skeleton["rhs"][s], variant["pools"], layout,
                       permutation=permutation).hex() for s in states]
    remapping = {}
    canonical = []
    for code in structure:
        data = bytearray.fromhex(code)
        i = 0
        while i < len(data):
            op = data[i]; i += 1
            if op == 0x82:
                index = data[i]
                if index not in remapping: remapping[index] = len(remapping)
                data[i] = remapping[index]; i += 1
            elif op == 0x81: i += 1
            elif op == 0x83: i += 4
        canonical.append(data.hex())
    return content_id(resolved), content_id(canonical), resolved
