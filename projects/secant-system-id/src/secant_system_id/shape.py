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


@dataclass(frozen=True)
class KernelShape:
    """Compile-time dimensions of the first thread-owned trajectory kernel."""

    ast_leaf_counts: tuple[int, ...] = (8, 8)
    state_count: int = 4
    constant_count: int = 8
    trajectory_count: int = 3
    observation_count: int = 12
    patch_capacity: int = 192

    def __post_init__(self) -> None:
        values = (
            self.state_count,
            self.constant_count,
            self.trajectory_count,
            self.observation_count,
            self.patch_capacity,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("all kernel-shape dimensions must be positive integers")
        if not self.ast_leaf_counts or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.ast_leaf_counts
        ):
            raise ValueError("ast_leaf_counts must contain one positive count per missing site")
        if self.input_count + self.ast_count + 1 > 255:
            raise ValueError("marker ABI cannot use more than 255 registers")

    @property
    def ast_count(self) -> int:
        return len(self.ast_leaf_counts)

    @property
    def input_count(self) -> int:
        return sum(self.ast_leaf_counts)

    @property
    def ast_input_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        offset = 0
        for count in self.ast_leaf_counts:
            offsets.append(offset)
            offset += count
        return tuple(offsets)

    @property
    def bank_slot_count(self) -> int:
        return self.state_count + self.constant_count

    @property
    def reference_float_count(self) -> int:
        return self.state_count * self.trajectory_count * (self.observation_count + 1)

    def shared_bytes(self, threads_per_block: int) -> int:
        if threads_per_block <= 0:
            raise ValueError("threads_per_block must be positive")
        return 4 * (self.reference_float_count + self.bank_slot_count * threads_per_block)

    def winner_shared_bytes(self, threads_per_block: int) -> int:
        """Shared storage for the trajectory bank plus score/index reduction."""

        return self.shared_bytes(threads_per_block) + 8 * threads_per_block


@dataclass(frozen=True)
class PackedDispatch:
    """Compile-time shape of the compact system-genome dispatch arena.

    ``genome_capacity`` is the number of target-table entries in the CUBIN.
    ``genomes_per_cta`` is the number of entries one CTA evaluates in turn.
    Each entry is a complete system genome and therefore contains one AST for
    every missing mathematical site in ``KernelShape``.
    """

    genome_capacity: int = 8
    genomes_per_cta: int = 8

    def __post_init__(self) -> None:
        for name, value in (
            ("genome_capacity", self.genome_capacity),
            ("genomes_per_cta", self.genomes_per_cta),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.genomes_per_cta > self.genome_capacity:
            raise ValueError("genomes_per_cta cannot exceed genome_capacity")
        if self.genome_capacity > 128:
            raise ValueError("the compact PTX branch-target table supports at most 128 genomes")


@dataclass(frozen=True)
class PackedKernelSpec:
    """One packed entry point and the global genome range it owns."""

    name: str
    genome_base: int
    dispatch: PackedDispatch

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("packed kernel name must be non-empty")
        if isinstance(self.genome_base, bool) or not isinstance(self.genome_base, int):
            raise ValueError("genome_base must be a non-negative integer")
        if self.genome_base < 0:
            raise ValueError("genome_base must be a non-negative integer")
        if not isinstance(self.dispatch, PackedDispatch):
            raise ValueError("dispatch must be a PackedDispatch")

    @property
    def genome_end(self) -> int:
        return self.genome_base + self.dispatch.genome_capacity
