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
"""Python kernel-shape reference model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


KERNEL_NAME = "odezza_scoring"
SUPPORTED_ARCHITECTURES = frozenset((89, 90, 120))
MAX_TOGGLE_BITS = 32


@dataclass(frozen=True)
class KernelShape:
    """Compile-time dimensions for the grid-dispatched multi-system scorer."""

    state_names: tuple[str, ...] = ("prey", "predator")
    constant_count: int = 4
    trajectory_count: int = 1
    observation_count: int = 64
    observation_interval: float = 0.05
    system_capacity: int = 8
    shared_patch_capacity: int = 384
    system_patch_capacity: int = 384

    def __post_init__(self) -> None:
        if not self.state_names or len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state names must be nonempty and unique")
        if any(
            not isinstance(name, str)
            or not name.isascii()
            or not name.isidentifier()
            for name in self.state_names
        ):
            raise ValueError("state names must be ASCII C identifiers")
        for name, value in (
            ("constant_count", self.constant_count),
            ("trajectory_count", self.trajectory_count),
            ("observation_count", self.observation_count),
            ("system_capacity", self.system_capacity),
            ("shared_patch_capacity", self.shared_patch_capacity),
            ("system_patch_capacity", self.system_patch_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.system_capacity > 65535:
            raise ValueError("system_capacity must fit in a CUDA grid-y dimension")
        if (
            isinstance(self.observation_interval, bool)
            or not isinstance(self.observation_interval, (int, float))
            or not math.isfinite(self.observation_interval)
            or self.observation_interval <= 0.0
        ):
            raise ValueError("observation_interval must be finite and positive")
        object.__setattr__(
            self,
            "observation_interval",
            float(format(float(self.observation_interval), ".9g")),
        )
        if self.input_count + self.output_count + 1 >= 255:
            raise ValueError("the marker ABI cannot expose 255 or more values")

    @property
    def state_count(self) -> int:
        return len(self.state_names)

    @property
    def output_count(self) -> int:
        return self.state_count

    @property
    def input_count(self) -> int:
        return self.state_count + self.constant_count

    @property
    def source_count(self) -> int:
        return self.state_count + self.constant_count

    @property
    def reference_float_count(self) -> int:
        return self.state_count * self.trajectory_count * (self.observation_count + 1)

    @property
    def shared_bytes(self) -> int:
        return 4 * self.reference_float_count

    @property
    def arena_instruction_count(self) -> int:
        return self.system_capacity * self.system_patch_capacity

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "state_count": self.state_count,
                "input_count": self.input_count,
                "output_count": self.output_count,
                "reference_float_count": self.reference_float_count,
                "arena_instruction_count": self.arena_instruction_count,
            }
        )
        result["state_names"] = list(self.state_names)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KernelShape":
        if not isinstance(value, dict):
            raise ValueError("kernel shape must be a JSON object")
        return cls(
            state_names=tuple(value["state_names"]),
            constant_count=int(value["constant_count"]),
            trajectory_count=int(value["trajectory_count"]),
            observation_count=int(value["observation_count"]),
            observation_interval=float(value["observation_interval"]),
            system_capacity=int(value["system_capacity"]),
            shared_patch_capacity=int(value["shared_patch_capacity"]),
            system_patch_capacity=int(value["system_patch_capacity"]),
        )
