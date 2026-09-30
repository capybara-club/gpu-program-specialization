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

from array import array
from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Mapping, Sequence

from .model import SystemModel
from .shape import KernelShape


State = tuple[float, ...]
RhsFunction = Callable[[float, Sequence[float], Mapping[str, float]], State]


class RecoveryProtocol(str, Enum):
    """The quantity a benchmark asks a method to recover."""

    TRAJECTORY = "trajectory"
    DERIVATIVE_REGRESSION = "derivative_regression"
    PARAMETER_FIT = "parameter_fit"


@dataclass(frozen=True)
class NoiseSpec:
    kind: str = "none"
    level: float = 0.0
    seed: int | None = None
    target_semantics: str = "observed_trajectory"

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("noise kind must be non-empty")
        if not math.isfinite(self.level) or self.level < 0.0:
            raise ValueError("noise level must be finite and non-negative")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("noise seed must be an integer or None")
        if not self.target_semantics:
            raise ValueError("noise target semantics must be non-empty")


@dataclass(frozen=True)
class ObservationSeries:
    """One trajectory or experiment, including sparse/irregular observations.

    A ``None`` state value is deliberately legal here. The current CUDA trajectory
    kernel cannot consume it, but keeping missingness in the problem object prevents
    adapters such as PEtab from silently densifying measurements.
    """

    name: str
    times: tuple[float, ...]
    values: tuple[tuple[float | None, ...], ...]
    initial_state: State
    derivatives: tuple[tuple[float | None, ...], ...] | None = None
    noise: NoiseSpec = NoiseSpec()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation-series name must be non-empty")
        if not self.times or len(self.times) != len(self.values):
            raise ValueError("times and values must have the same non-zero length")
        if not self.initial_state:
            raise ValueError("initial_state must be non-empty")
        state_count = len(self.initial_state)
        if any(not math.isfinite(value) for value in self.initial_state):
            raise ValueError("initial_state values must be finite")
        previous = None
        for time, row in zip(self.times, self.values):
            if not math.isfinite(time) or (previous is not None and time <= previous):
                raise ValueError("observation times must be finite and strictly increasing")
            if len(row) != state_count:
                raise ValueError("every observation row must match initial_state width")
            if any(value is not None and not math.isfinite(value) for value in row):
                raise ValueError("observations must be finite or None")
            previous = time
        if self.derivatives is not None:
            if len(self.derivatives) != len(self.times):
                raise ValueError("derivatives must align with observation times")
            for row in self.derivatives:
                if len(row) != state_count:
                    raise ValueError("derivative rows must match initial_state width")
                if any(value is not None and not math.isfinite(value) for value in row):
                    raise ValueError("derivatives must be finite or None")

    @property
    def state_count(self) -> int:
        return len(self.initial_state)

    @property
    def is_complete(self) -> bool:
        return all(value is not None for row in self.values for value in row)

    @property
    def has_complete_derivatives(self) -> bool:
        return self.derivatives is not None and all(
            value is not None for row in self.derivatives for value in row
        )

    @property
    def is_uniform(self) -> bool:
        if len(self.times) < 3:
            return True
        interval = self.times[1] - self.times[0]
        tolerance = max(1.0e-12, abs(interval) * 1.0e-9)
        return all(
            abs((self.times[index] - self.times[index - 1]) - interval) <= tolerance
            for index in range(2, len(self.times))
        )


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    nominal: float
    lower: float | None = None
    upper: float | None = None
    estimated: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("parameter names must be valid identifiers")
        if not math.isfinite(self.nominal):
            raise ValueError("parameter nominal values must be finite")
        if self.lower is not None and not math.isfinite(self.lower):
            raise ValueError("parameter lower bounds must be finite or None")
        if self.upper is not None and not math.isfinite(self.upper):
            raise ValueError("parameter upper bounds must be finite or None")
        if self.lower is not None and self.nominal < self.lower:
            raise ValueError("parameter nominal value is below its lower bound")
        if self.upper is not None and self.nominal > self.upper:
            raise ValueError("parameter nominal value is above its upper bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("parameter lower bound exceeds upper bound")


@dataclass(frozen=True)
class StructureSearchSpec:
    operators: tuple[str, ...]
    max_nodes: int
    constant_count: int
    patch_capacity: int = 192

    def __post_init__(self) -> None:
        if not self.operators or any(not operator for operator in self.operators):
            raise ValueError("at least one non-empty search operator is required")
        for name, value in (
            ("max_nodes", self.max_nodes),
            ("constant_count", self.constant_count),
            ("patch_capacity", self.patch_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class IntegrationSpec:
    method: str
    start: float
    stop: float
    sample_count: int
    reference_solver: str
    reference_tolerances: str

    def __post_init__(self) -> None:
        if not self.method or not self.reference_solver or not self.reference_tolerances:
            raise ValueError("integration method and reference details must be non-empty")
        if not math.isfinite(self.start) or not math.isfinite(self.stop) or self.stop <= self.start:
            raise ValueError("integration interval must be finite and increasing")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise ValueError("sample_count must be an integer")
        if self.sample_count < 2:
            raise ValueError("integration sample_count must be at least two")


@dataclass(frozen=True)
class ProblemProvenance:
    corpus: str
    system_id: str
    source_url: str
    data_protocol: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.corpus or not self.system_id or not self.source_url or not self.data_protocol:
            raise ValueError("problem provenance fields must be non-empty")


@dataclass(frozen=True)
class RecoveryProblem:
    """Complete, benchmark-aware input to a Secant system-ID run."""

    name: str
    protocol: RecoveryProtocol
    model: SystemModel
    parameters: tuple[ParameterSpec, ...]
    search: StructureSearchSpec | None
    integration: IntegrationSpec
    training: tuple[ObservationSeries, ...]
    validation: tuple[ObservationSeries, ...] = ()
    test: tuple[ObservationSeries, ...] = ()
    truth_equations: tuple[str, ...] = ()
    rhs: RhsFunction | None = None
    provenance: ProblemProvenance | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("recovery-problem name must be a valid identifier")
        if not self.training:
            raise ValueError("a recovery problem requires at least one training series")
        if len({parameter.name for parameter in self.parameters}) != len(self.parameters):
            raise ValueError("parameter names must be unique")
        if self.truth_equations and len(self.truth_equations) != self.model.state_count:
            raise ValueError("truth equations must contain one expression per state")
        for series in self.training + self.validation + self.test:
            if series.state_count != self.model.state_count:
                raise ValueError("all series must match the system-model state count")

    @property
    def nominal_parameters(self) -> dict[str, float]:
        return {parameter.name: parameter.nominal for parameter in self.parameters}

    def series(self, split: str = "training") -> tuple[ObservationSeries, ...]:
        try:
            return {"training": self.training, "validation": self.validation, "test": self.test}[split]
        except KeyError as error:
            raise ValueError("split must be training, validation, or test") from error

    def trajectory_kernel_blockers(self, split: str = "training") -> tuple[str, ...]:
        selected = self.series(split)
        if not selected:
            return (f"{split} split is empty",)
        blockers: list[str] = []
        first_times = selected[0].times
        if any(not series.is_complete for series in selected):
            blockers.append("missing state observations")
        if any(not series.is_uniform for series in selected):
            blockers.append("irregular observation times")
        if any(series.times != first_times for series in selected[1:]):
            blockers.append("trajectory-specific time grids")
        if len(first_times) < 2:
            blockers.append("fewer than two time points")
        return tuple(blockers)

    def make_trajectory_shape(self, split: str = "training") -> KernelShape:
        blockers = self.trajectory_kernel_blockers(split)
        if blockers:
            raise ValueError("current dense trajectory kernel cannot represent: " + ", ".join(blockers))
        selected = self.series(split)
        if self.search is None:
            raise ValueError("trajectory shape requires a structure-search specification")
        return self.model.make_shape(
            constant_count=self.search.constant_count,
            trajectory_count=len(selected),
            observation_count=len(selected[0].times) - 1,
            patch_capacity=self.search.patch_capacity,
        )

    def dense_reference_array(self, split: str = "training") -> array:
        shape = self.make_trajectory_shape(split)
        selected = self.series(split)
        result = array("f")
        for component in range(shape.state_count):
            for series in selected:
                result.append(series.initial_state[component])
        for component in range(shape.state_count):
            for series in selected:
                for row in series.values[1:]:
                    value = row[component]
                    if value is None:
                        raise AssertionError("kernel blocker failed to reject missing value")
                    result.append(value)
        return result


def uniform_times(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("uniform time grids require at least two points")
    interval = (stop - start) / (count - 1)
    return tuple(start + interval * index for index in range(count))


def simulate_rk4(
    rhs: RhsFunction,
    initial_state: Sequence[float],
    times: Sequence[float],
    parameters: Mapping[str, float],
    *,
    maximum_step: float,
) -> tuple[State, ...]:
    """Small dependency-free reference integrator used only for local validation."""

    if not times or maximum_step <= 0.0 or not math.isfinite(maximum_step):
        raise ValueError("times must be non-empty and maximum_step finite and positive")
    state_value = tuple(float(value) for value in initial_state)
    result = [state_value]
    current_time = float(times[0])
    for target_time_raw in times[1:]:
        target_time = float(target_time_raw)
        if target_time <= current_time:
            raise ValueError("integration times must be strictly increasing")
        step_count = max(1, math.ceil((target_time - current_time) / maximum_step))
        step = (target_time - current_time) / step_count
        for _ in range(step_count):
            k1 = rhs(current_time, state_value, parameters)
            stage = tuple(state_value[i] + 0.5 * step * k1[i] for i in range(len(state_value)))
            k2 = rhs(current_time + 0.5 * step, stage, parameters)
            stage = tuple(state_value[i] + 0.5 * step * k2[i] for i in range(len(state_value)))
            k3 = rhs(current_time + 0.5 * step, stage, parameters)
            stage = tuple(state_value[i] + step * k3[i] for i in range(len(state_value)))
            k4 = rhs(current_time + step, stage, parameters)
            state_value = tuple(
                state_value[i] + step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                for i in range(len(state_value))
            )
            if any(not math.isfinite(value) for value in state_value):
                raise FloatingPointError("reference RK4 trajectory became non-finite")
            current_time += step
        current_time = target_time
        result.append(state_value)
    return tuple(result)
