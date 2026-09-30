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
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .shape import KernelShape


class LeafKind(IntEnum):
    STATE = 0
    CONSTANT = 1


@dataclass(frozen=True)
class LeafSource:
    kind: LeafKind
    index: int

    def encode(self, shape: KernelShape = KernelShape()) -> int:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("leaf-source index must be a non-negative integer")
        if self.kind == LeafKind.STATE:
            if self.index >= shape.state_count:
                raise ValueError("state leaf is outside the state vector")
            return self.index
        if self.kind == LeafKind.CONSTANT:
            if self.index >= shape.constant_count:
                raise ValueError("constant leaf is outside the setting's constant bank")
            return shape.state_count + self.index
        raise ValueError("unknown leaf-source kind")


def state_leaf(index: int) -> LeafSource:
    return LeafSource(LeafKind.STATE, index)


def constant_leaf(index: int) -> LeafSource:
    return LeafSource(LeafKind.CONSTANT, index)


def decode_leaf(encoded: int, shape: KernelShape = KernelShape()) -> LeafSource:
    if isinstance(encoded, bool) or not isinstance(encoded, int) or not 0 <= encoded < shape.bank_slot_count:
        raise ValueError("encoded leaf is outside the shared bank")
    if encoded < shape.state_count:
        return state_leaf(encoded)
    return constant_leaf(encoded - shape.state_count)
