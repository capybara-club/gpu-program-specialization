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
"""Compile-time shape for the Python trajectory-LM research kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


LM_KERNEL_NAME = "odezza_trajectory_lm"
LM_MANIFEST_SHAPE = "trajectory_lm"
LM_POSTORDER_ABI = "odezza-lm-postorder-f32-v1"
LM_GROUPED_MARKER_ABI_VERSION = 4
LM_GROUPED_SPECIALIZATION_ABI = "odezza-lm-grouped-sites-v4"
LM_TWO_SITE_MARKER_ABI_VERSION = 8
LM_TWO_SITE_SPECIALIZATION_ABI = "odezza-lm-replicated-sites-v8"
LM_GROUPED_LAYOUT = "grouped_partials"
LM_TWO_SITE_LAYOUT = "two_site_partials"


@dataclass(frozen=True)
class LMKernelShape:
    """Exact dimensions retained in a resident LM fit."""

    state_count: int = 2
    optimized_constant_count: int = 8
    site_patch_capacity: int = 256
    site_output_capacity: int = 0
    site_layout: str = LM_GROUPED_LAYOUT
    fit_thread_count: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.state_count, bool)
            or not isinstance(self.state_count, int)
            or self.state_count < 1
        ):
            raise ValueError("state_count must be positive")
        if (
            isinstance(self.optimized_constant_count, bool)
            or not isinstance(self.optimized_constant_count, int)
            or not 1 <= self.optimized_constant_count <= 16
        ):
            raise ValueError("optimized_constant_count must be between one and sixteen")
        if (
            isinstance(self.site_patch_capacity, bool)
            or not isinstance(self.site_patch_capacity, int)
            or self.site_patch_capacity < 8
        ):
            raise ValueError("site_patch_capacity must be at least eight instructions")
        if self.site_layout not in {LM_GROUPED_LAYOUT, LM_TWO_SITE_LAYOUT}:
            raise ValueError("site_layout must be grouped_partials or two_site_partials")
        if self.fit_thread_count not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("fit_thread_count must be a power of two between one and thirty-two")
        if self.fit_thread_count > 1 and self.site_layout != LM_TWO_SITE_LAYOUT:
            raise ValueError("cooperative LM ownership currently requires two_site_partials")
        if self.site_layout == LM_GROUPED_LAYOUT and self.maximum_site_output_count < 1:
            raise ValueError(
                "the LM marker leaves no output within the conservative 30-operand inline-assembly budget"
            )
        if (
            isinstance(self.site_output_capacity, bool)
            or not isinstance(self.site_output_capacity, int)
            or self.site_output_capacity < 0
            or (
                self.site_layout == LM_GROUPED_LAYOUT
                and self.site_output_capacity > self.maximum_site_output_count
            )
        ):
            raise ValueError(
                "site_output_capacity must be zero for automatic selection or fit the inline-assembly budget"
            )
        if self.site_layout == LM_TWO_SITE_LAYOUT and self.site_output_capacity != 0:
            raise ValueError("two_site_partials selects its fixed outputs automatically")
        if self.site_layout == LM_TWO_SITE_LAYOUT and self.maximum_site_output_count < 1:
            raise ValueError(
                "two_site_partials leaves no primal output within the inline-assembly operand budget"
            )
        if self.site_layout == LM_TWO_SITE_LAYOUT and self.two_site_inline_asm_operand_count > 30:
            raise ValueError(
                "two_site_partials exceeds the conservative 30-operand inline-assembly budget"
            )

    @property
    def input_count(self) -> int:
        return self.state_count + self.optimized_constant_count

    @property
    def local_partial_count(self) -> int:
        return self.state_count * self.input_count

    @property
    def scalar_output_count(self) -> int:
        return self.state_count + self.local_partial_count

    @property
    def maximum_site_output_count(self) -> int:
        if self.site_layout == LM_TWO_SITE_LAYOUT:
            return 28 - self.input_count
        return 28 - 2 * self.input_count

    @property
    def active_site_output_capacity(self) -> int:
        return self.site_output_capacity or self.maximum_site_output_count

    @property
    def site_output_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.site_layout == LM_TWO_SITE_LAYOUT:
            outputs = tuple(range(self.state_count))
            capacity = self.maximum_site_output_count
            groups = [
                outputs[index : index + capacity]
                for index in range(0, len(outputs), capacity)
            ]
            for rhs_index in range(self.state_count):
                partials = tuple(
                    self.state_count + input_index * self.state_count + rhs_index
                    for input_index in range(self.input_count)
                )
                groups.extend(
                    partials[index : index + capacity]
                    for index in range(0, len(partials), capacity)
                )
            return tuple(groups)
        capacity = self.active_site_output_capacity
        primals = tuple(range(self.state_count))
        partials = tuple(range(self.state_count, self.scalar_output_count))
        groups = [
            primals[index : index + capacity]
            for index in range(0, len(primals), capacity)
        ]
        groups.extend(
            partials[index : index + capacity]
            for index in range(0, len(partials), capacity)
        )
        return tuple(groups)

    @property
    def primal_site_count(self) -> int:
        if self.site_layout == LM_TWO_SITE_LAYOUT:
            return (
                self.state_count
                + self.maximum_site_output_count
                - 1
            ) // self.maximum_site_output_count
        return sum(group[0] < self.state_count for group in self.site_output_groups)

    @property
    def site_count(self) -> int:
        return len(self.site_output_groups)

    @property
    def gram_count(self) -> int:
        count = self.optimized_constant_count
        return count * (count + 1) // 2

    @property
    def local_parameter_count(self) -> int:
        return (self.optimized_constant_count + self.fit_thread_count - 1) // self.fit_thread_count

    @property
    def fits_per_warp(self) -> int:
        return 32 // self.fit_thread_count

    @property
    def local_optimizer_count(self) -> int:
        return (
            self.optimized_constant_count
            + self.gram_count
            + self.fit_thread_count
            - 1
        ) // self.fit_thread_count

    @property
    def inline_asm_operand_count(self) -> int:
        if self.site_layout == LM_TWO_SITE_LAYOUT:
            return self.two_site_inline_asm_operand_count
        largest_group = max(map(len, self.site_output_groups))
        return 2 * self.input_count + largest_group + 2

    @property
    def site_input_counts(self) -> tuple[int, ...]:
        if self.site_layout == LM_TWO_SITE_LAYOUT:
            return (self.input_count,) * self.site_count
        return (self.input_count,) * self.site_count

    @property
    def two_site_inline_asm_operand_count(self) -> int:
        primal = self.input_count + self.maximum_site_output_count + 2
        partials = self.input_count + self.maximum_site_output_count + 2
        return max(primal, partials)

    @property
    def marker_abi_version(self) -> int:
        return (
            LM_TWO_SITE_MARKER_ABI_VERSION
            if self.site_layout == LM_TWO_SITE_LAYOUT
            else LM_GROUPED_MARKER_ABI_VERSION
        )

    @property
    def specialization_abi(self) -> str:
        return (
            LM_TWO_SITE_SPECIALIZATION_ABI
            if self.site_layout == LM_TWO_SITE_LAYOUT
            else LM_GROUPED_SPECIALIZATION_ABI
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "input_count": self.input_count,
                "local_partial_count": self.local_partial_count,
                "scalar_output_count": self.scalar_output_count,
                "maximum_site_output_count": self.maximum_site_output_count,
                "active_site_output_capacity": self.active_site_output_capacity,
                "site_output_groups": [list(group) for group in self.site_output_groups],
                "primal_site_count": self.primal_site_count,
                "site_count": self.site_count,
                "gram_count": self.gram_count,
                "local_parameter_count": self.local_parameter_count,
                "fits_per_warp": self.fits_per_warp,
                "local_optimizer_count": self.local_optimizer_count,
                "inline_asm_operand_count": self.inline_asm_operand_count,
                "site_input_counts": list(self.site_input_counts),
                "marker_abi_version": self.marker_abi_version,
                "specialization_abi": self.specialization_abi,
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LMKernelShape":
        if not isinstance(value, dict):
            raise ValueError("LM kernel shape must be a JSON object")
        result = cls(
            state_count=int(value["state_count"]),
            optimized_constant_count=int(value["optimized_constant_count"]),
            site_patch_capacity=int(value["site_patch_capacity"]),
            site_output_capacity=int(value.get("site_output_capacity", 0)),
            site_layout=str(value.get("site_layout", LM_GROUPED_LAYOUT)),
            fit_thread_count=int(value.get("fit_thread_count", 1)),
        )
        for name, actual in (
            ("input_count", result.input_count),
            ("local_partial_count", result.local_partial_count),
            ("scalar_output_count", result.scalar_output_count),
            ("maximum_site_output_count", result.maximum_site_output_count),
            ("active_site_output_capacity", result.active_site_output_capacity),
            ("primal_site_count", result.primal_site_count),
            ("site_count", result.site_count),
            ("gram_count", result.gram_count),
            ("local_parameter_count", result.local_parameter_count),
            ("fits_per_warp", result.fits_per_warp),
            ("local_optimizer_count", result.local_optimizer_count),
            ("inline_asm_operand_count", result.inline_asm_operand_count),
            ("marker_abi_version", result.marker_abi_version),
        ):
            if name in value and int(value[name]) != actual:
                raise ValueError(f"LM shape {name} disagrees with its base dimensions")
        if "site_output_groups" in value and value["site_output_groups"] != [
            list(group) for group in result.site_output_groups
        ]:
            raise ValueError("LM shape site_output_groups disagrees with its base dimensions")
        if "site_input_counts" in value and value["site_input_counts"] != list(result.site_input_counts):
            raise ValueError("LM shape site_input_counts disagrees with its base dimensions")
        if "specialization_abi" in value and value["specialization_abi"] != result.specialization_abi:
            raise ValueError("LM shape specialization_abi disagrees with its site layout")
        return result
