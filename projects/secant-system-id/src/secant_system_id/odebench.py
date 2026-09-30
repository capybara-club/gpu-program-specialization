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
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from .model import MissingSite, SystemModel, missing
from .problem import (
    IntegrationSpec,
    NoiseSpec,
    ObservationSeries,
    ParameterSpec,
    ProblemProvenance,
    RecoveryProblem,
    RecoveryProtocol,
    RhsFunction,
    StructureSearchSpec,
    simulate_rk4,
    uniform_times,
)


ODEBENCH_SOURCE = "https://github.com/GPBench/ODEBench"
ODEFORMER_PAPER = "https://arxiv.org/abs/2310.05573"
DEFAULT_SEARCH = StructureSearchSpec(
    operators=("add", "subtract", "multiply", "square"),
    max_nodes=30,
    constant_count=8,
)
ODEBENCH_INTEGRATION = IntegrationSpec(
    method="rk4_secant_candidate",
    start=0.0,
    stop=10.0,
    sample_count=150,
    reference_solver="ODEBench LSODA",
    reference_tolerances="rtol=1e-5, atol=1e-7, first_step=1e-6, min_step=1e-10",
)


@dataclass(frozen=True)
class ODEBenchDefinition:
    system_id: int
    name: str
    description: str
    equations: tuple[str, ...]
    constants: tuple[float, ...]
    initial_conditions: tuple[tuple[float, ...], ...]
    rhs: RhsFunction

    def __post_init__(self) -> None:
        if self.system_id <= 0 or not self.name.isidentifier():
            raise ValueError("ODEBench definitions require a positive id and identifier name")
        if not self.equations or len(self.initial_conditions) != 2:
            raise ValueError("ODEBench definitions require equations and two initial conditions")
        if any(len(initial) != len(self.equations) for initial in self.initial_conditions):
            raise ValueError("initial conditions must match equation count")

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(
            ParameterSpec(f"c{index}", value)
            for index, value in enumerate(self.constants)
        )

    @property
    def nominal_parameters(self) -> dict[str, float]:
        return {parameter.name: parameter.nominal for parameter in self.parameter_specs}


def _lotka_volterra(
    _time: float, state: Sequence[float], parameters: Mapping[str, float]
) -> tuple[float, float]:
    x, y = state
    return (
        x * (parameters["c0"] - parameters["c1"] * y),
        -y * (parameters["c2"] - parameters["c3"] * x),
    )


def _van_der_pol(
    _time: float, state: Sequence[float], parameters: Mapping[str, float]
) -> tuple[float, float]:
    x, velocity = state
    return velocity, -x - parameters["c0"] * (x * x - 1.0) * velocity


def _lorenz_63(
    _time: float, state: Sequence[float], parameters: Mapping[str, float]
) -> tuple[float, float, float]:
    x, y, z = state
    return (
        parameters["c0"] * (y - x),
        parameters["c1"] * x - y - x * z,
        x * y - parameters["c2"] * z,
    )


def _rossler(
    _time: float, state: Sequence[float], parameters: Mapping[str, float]
) -> tuple[float, float, float]:
    x, y, z = state
    time_scale = parameters["c3"]
    return (
        time_scale * (-y - z),
        time_scale * (x + parameters["c0"] * y),
        time_scale * (parameters["c1"] + z * (x - parameters["c2"])),
    )


ODEBENCH_REFERENCE_SYSTEMS = (
    ODEBenchDefinition(
        system_id=27,
        name="lotka_volterra",
        description="Lotka-Volterra model (simple)",
        equations=("x0*(c0-c1*x1)", "-x1*(c2-c3*x0)"),
        constants=(1.84, 1.45, 3.0, 1.62),
        initial_conditions=((8.3, 3.4), (0.4, 0.65)),
        rhs=_lotka_volterra,
    ),
    ODEBenchDefinition(
        system_id=37,
        name="van_der_pol",
        description="Van der Pol oscillator",
        equations=("x1", "-x0-c0*(x0^2-1)*x1"),
        constants=(0.43,),
        initial_conditions=((2.2, 0.0), (0.1, 3.2)),
        rhs=_van_der_pol,
    ),
    ODEBenchDefinition(
        system_id=56,
        name="lorenz_63",
        description="Lorenz system (chaotic)",
        equations=("c0*(x1-x0)", "c1*x0-x1-x0*x2", "x0*x1-c2*x2"),
        constants=(10.0, 28.0, 8.0 / 3.0),
        initial_conditions=((2.3, 8.1, 12.4), (10.0, 20.0, 30.0)),
        rhs=_lorenz_63,
    ),
    ODEBenchDefinition(
        system_id=59,
        name="rossler",
        description="Rossler system (chaotic, ODEBench time scale)",
        equations=("c3*(-x1-x2)", "c3*(x0+c0*x1)", "c3*(c1+x2*(x0-c2))"),
        constants=(0.2, 0.2, 5.7, 5.0),
        initial_conditions=((2.3, 1.1, 0.8), (-0.1, 4.1, -2.1)),
        rhs=_rossler,
    ),
)


def definition_by_name(name: str) -> ODEBenchDefinition:
    for definition in ODEBENCH_REFERENCE_SYSTEMS:
        if definition.name == name:
            return definition
    raise KeyError(name)


def _series(
    definition: ODEBenchDefinition,
    index: int,
    *,
    maximum_step: float,
) -> ObservationSeries:
    times = uniform_times(
        ODEBENCH_INTEGRATION.start,
        ODEBENCH_INTEGRATION.stop,
        ODEBENCH_INTEGRATION.sample_count,
    )
    initial = definition.initial_conditions[index]
    parameters = definition.nominal_parameters
    values = simulate_rk4(
        definition.rhs,
        initial,
        times,
        parameters,
        maximum_step=maximum_step,
    )
    derivatives = tuple(definition.rhs(time, state, parameters) for time, state in zip(times, values))
    return ObservationSeries(
        name=f"{definition.name}_initial_{index}",
        times=times,
        values=values,
        initial_state=initial,
        derivatives=derivatives,
    )


def make_reference_problem(
    definition: ODEBenchDefinition,
    *,
    protocol: RecoveryProtocol = RecoveryProtocol.TRAJECTORY,
    maximum_step: float = 0.0025,
) -> RecoveryProblem:
    """Build a dependency-free local reference, not the official LSODA artifact."""

    model = SystemModel(
        name=definition.name,
        state_names=tuple(f"x{index}" for index in range(len(definition.equations))),
        missing_sites=tuple(
            MissingSite(f"rhs{index}", 8) for index in range(len(definition.equations))
        ),
        derivatives=tuple(missing(index) for index in range(len(definition.equations))),
        observation_interval=(
            ODEBENCH_INTEGRATION.stop - ODEBENCH_INTEGRATION.start
        ) / (ODEBENCH_INTEGRATION.sample_count - 1),
    )
    first = _series(definition, 0, maximum_step=maximum_step)
    second = _series(definition, 1, maximum_step=maximum_step)
    if protocol == RecoveryProtocol.DERIVATIVE_REGRESSION:
        training = (first,)
        validation: tuple[ObservationSeries, ...] = ()
        test: tuple[ObservationSeries, ...] = ()
        data_protocol = "MDBench ODE derivative-regression layout; local RK4 stand-in"
    elif protocol == RecoveryProtocol.TRAJECTORY:
        training = (first,)
        validation = ()
        test = (second,)
        data_protocol = "local two-initial-condition trajectory validation stand-in"
    else:
        raise ValueError("ODEBench reference systems support trajectory or derivative regression")
    return RecoveryProblem(
        name=definition.name,
        protocol=protocol,
        model=model,
        parameters=definition.parameter_specs,
        search=DEFAULT_SEARCH,
        integration=ODEBENCH_INTEGRATION,
        training=training,
        validation=validation,
        test=test,
        truth_equations=definition.equations,
        rhs=definition.rhs,
        provenance=ProblemProvenance(
            corpus="ODEBench/MDBench",
            system_id=str(definition.system_id),
            source_url=ODEBENCH_SOURCE,
            data_protocol=data_protocol,
            notes=(
                "ODEBench has 63 systems; this module initially validates four named systems.",
                "Official comparison data must come from the LSODA-generated benchmark artifact.",
                f"Direct ODEFormer comparison protocol is described by {ODEFORMER_PAPER}.",
            ),
        ),
    )


def _identifier(text: str, system_id: int) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", text.lower()).strip("_")
    if not value or value[0].isdigit():
        value = f"system_{system_id}_{value}"
    return value


def load_solutions_json(
    path: str | Path,
    *,
    require_all_63: bool = True,
) -> tuple[RecoveryProblem, ...]:
    """Load the official ODEBench ``solutions.json`` without SciPy or SymPy.

    The official generator has already performed integration and embeds all
    metadata needed by this adapter. Each returned problem preserves all
    parameter/initial-condition solutions instead of deciding yet whether a
    direct comparison should fit them jointly or independently.
    """

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("ODEBench solutions JSON must contain a list")
    if require_all_63 and len(records) != 63:
        raise ValueError(f"expected all 63 ODEBench systems, found {len(records)}")
    problems: list[RecoveryProblem] = []
    seen_ids: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each ODEBench record must be an object")
        system_id = record.get("id")
        dimension = record.get("dim")
        equation_text = record.get("eq")
        if isinstance(system_id, bool) or not isinstance(system_id, int) or system_id <= 0:
            raise ValueError("ODEBench system id must be a positive integer")
        if system_id in seen_ids:
            raise ValueError(f"duplicate ODEBench system id {system_id}")
        seen_ids.add(system_id)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"ODEBench system {system_id} has invalid dimension")
        if not isinstance(equation_text, str):
            raise ValueError(f"ODEBench system {system_id} is missing equations")
        equations = tuple(part.strip() for part in equation_text.split("|"))
        if len(equations) != dimension:
            raise ValueError(f"ODEBench system {system_id} equation count disagrees with dimension")
        nested_solutions = record.get("solutions")
        if not isinstance(nested_solutions, list) or not nested_solutions:
            raise ValueError(f"ODEBench system {system_id} has no generated solutions")
        series: list[ObservationSeries] = []
        nominal_constants: tuple[float, ...] | None = None
        first_start = math.inf
        last_stop = -math.inf
        maximum_samples = 0
        for parameter_index, solutions in enumerate(nested_solutions):
            if not isinstance(solutions, list) or not solutions:
                raise ValueError(f"ODEBench system {system_id} has an empty parameter solution set")
            for initial_index, solution in enumerate(solutions):
                if not isinstance(solution, dict) or not solution.get("success", False):
                    raise ValueError(
                        f"ODEBench system {system_id} parameter {parameter_index} initial {initial_index} failed"
                    )
                times = tuple(float(value) for value in solution.get("t", ()))
                component_values = solution.get("y")
                initial = tuple(float(value) for value in solution.get("init", ()))
                constants = tuple(float(value) for value in solution.get("consts", ()))
                if nominal_constants is None:
                    nominal_constants = constants
                if not times or not isinstance(component_values, list) or len(component_values) != dimension:
                    raise ValueError(f"ODEBench system {system_id} has malformed trajectory data")
                if len(initial) != dimension or any(len(component) != len(times) for component in component_values):
                    raise ValueError(f"ODEBench system {system_id} trajectory dimensions disagree")
                rows = tuple(
                    tuple(float(component_values[component][sample]) for component in range(dimension))
                    for sample in range(len(times))
                )
                noise_amplitude = float(solution.get("noise_amplitude", 0.0))
                series.append(
                    ObservationSeries(
                        name=f"odebench_{system_id}_p{parameter_index}_i{initial_index}",
                        times=times,
                        values=rows,
                        initial_state=initial,
                        noise=NoiseSpec(
                            kind="none" if noise_amplitude == 0.0 else "multiplicative_gaussian_amplitude",
                            level=noise_amplitude,
                            seed=int(solution.get("random_seed", 42)),
                            target_semantics="ODEBench observed trajectory",
                        ),
                    )
                )
                first_start = min(first_start, times[0])
                last_stop = max(last_stop, times[-1])
                maximum_samples = max(maximum_samples, len(times))
        assert nominal_constants is not None
        description = str(record.get("eq_description", f"ODEBench system {system_id}"))
        model_name = f"odebench_{system_id}_{_identifier(description, system_id)}"
        nominal_interval = (last_stop - first_start) / max(1, maximum_samples - 1)
        model = SystemModel(
            name=model_name,
            state_names=tuple(f"x{index}" for index in range(dimension)),
            missing_sites=tuple(MissingSite(f"rhs{index}", 8) for index in range(dimension)),
            derivatives=tuple(missing(index) for index in range(dimension)),
            observation_interval=nominal_interval,
        )
        problems.append(
            RecoveryProblem(
                name=model_name,
                protocol=RecoveryProtocol.TRAJECTORY,
                model=model,
                parameters=tuple(
                    ParameterSpec(f"c{index}", value)
                    for index, value in enumerate(nominal_constants)
                ),
                search=DEFAULT_SEARCH,
                integration=IntegrationSpec(
                    method="rk4_secant_candidate",
                    start=first_start,
                    stop=last_stop,
                    sample_count=maximum_samples,
                    reference_solver="official ODEBench generated solution",
                    reference_tolerances=ODEBENCH_INTEGRATION.reference_tolerances,
                ),
                training=tuple(series),
                truth_equations=equations,
                provenance=ProblemProvenance(
                    corpus="ODEBench",
                    system_id=str(system_id),
                    source_url=ODEBENCH_SOURCE,
                    data_protocol="official solutions.json; comparison grouping not yet selected",
                    notes=(
                        "Preserves all parameter and initial-condition trajectories.",
                        "Use one trajectory per fit for a direct ODEFormer protocol unless a joint fit is declared.",
                    ),
                ),
            )
        )
    return tuple(problems)
