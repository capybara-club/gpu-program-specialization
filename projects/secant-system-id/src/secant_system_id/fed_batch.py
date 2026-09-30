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
import math
import random
from typing import Sequence

from .ast import Program, input_slot
from .bindings import constant_leaf, state_leaf
from .genome import SystemGenome
from .model import MissingSite, SystemModel, missing, state
from .problem import (
    IntegrationSpec,
    ObservationSeries,
    ParameterSpec,
    ProblemProvenance,
    RecoveryProblem,
    RecoveryProtocol,
    StructureSearchSpec,
)
from .shape import KernelShape


INITIAL_STATES = (
    (0.1, 10.0, 5.0, 0.0),
    (0.1, 20.0, 5.0, 0.0),
    (0.2, 5.0, 2.5, 0.0),
)

DIVERSE_INITIAL_STATES = (
    (0.08, 4.0, 1.0, 0.0),
    (0.08, 4.0, 8.0, 0.0),
    (0.08, 22.0, 1.0, 0.0),
    (0.08, 22.0, 8.0, 0.0),
    (0.24, 4.0, 1.0, 0.0),
    (0.24, 4.0, 8.0, 0.0),
    (0.24, 22.0, 1.0, 0.0),
    (0.24, 22.0, 8.0, 0.0),
    (0.12, 8.0, 3.0, 0.0),
    (0.12, 17.0, 6.0, 0.0),
    (0.19, 9.0, 7.0, 0.0),
    (0.19, 18.0, 2.0, 0.0),
)

# Paired product levels expose spurious dependence on product without forbidding
# that state as a candidate leaf. The planted rate laws are invariant to P.
PRODUCT_PAIRED_INITIAL_STATES = tuple(
    (biomass, glucose, sucrose, product)
    for biomass, glucose, sucrose in (
        (0.08, 4.0, 1.0),
        (0.08, 4.0, 8.0),
        (0.08, 22.0, 1.0),
        (0.08, 22.0, 8.0),
        (0.24, 4.0, 1.0),
        (0.24, 4.0, 8.0),
        (0.24, 22.0, 1.0),
        (0.24, 22.0, 8.0),
    )
    for product in (0.0, 4.0)
)

# mu_m1, K_c1, k_1, mu_m2, K_c2, k_2, one, zero
PLANTED_CONSTANTS = (0.43, 63.7, 5.8, 0.132, 3.68, 0.0, 1.0, 0.0)
INVALID_MSE = 3.402823466e38

_x = state(0)
_rate1 = missing(0)
_rate2 = missing(1)
FED_BATCH_MODEL = SystemModel(
    name="astaxanthin_fed_batch",
    state_names=("biomass", "glucose", "sucrose", "product"),
    missing_sites=(
        MissingSite("specific_growth_glucose", 8),
        MissingSite("specific_growth_sucrose", 8),
    ),
    derivatives=(
        _rate1 * _x + _rate2 * _x - 0.0055 * _x,
        -2.58 * _rate1 * _x,
        -1.71 * _rate2 * _x,
        0.21 * _x - 0.0466 * _x * _x,
    ),
    observation_interval=8.0,
)
FED_BATCH_SHAPE = FED_BATCH_MODEL.make_shape(
    constant_count=8,
    trajectory_count=3,
    observation_count=12,
)

# Each value indexes [X, S1, S2, P, c0, ..., c7] in the per-thread shared bank.
PLANTED_BINDINGS = tuple(source.encode(FED_BATCH_SHAPE) for source in (
    constant_leaf(0), state_leaf(1), state_leaf(1), constant_leaf(1),
    state_leaf(0), constant_leaf(6), constant_leaf(2), state_leaf(2),
    constant_leaf(3), state_leaf(2), state_leaf(2), constant_leaf(4),
    state_leaf(0), constant_leaf(6), constant_leaf(5), state_leaf(1),
))


@dataclass(frozen=True)
class Population:
    num_settings: int
    settings: array
    bindings: array


@dataclass(frozen=True)
class PackedPopulation:
    num_genomes: int
    num_settings: int
    settings: array
    bindings: array


def _planted_expressions(shape: KernelShape):
    if shape.ast_leaf_counts != (8, 8):
        raise ValueError("the planted fed-batch programs require two eight-leaf missing sites")
    first_offset, second_offset = shape.ast_input_offsets
    a = [input_slot(first_offset + index) for index in range(8)]
    b = [input_slot(second_offset + index) for index in range(8)]
    glucose_rate = a[0] * a[1] / ((a[2] + a[3] * a[4]) * (a[5] + a[6] * a[7]))
    sucrose_rate = b[0] * b[1] / ((b[2] + b[3] * b[4]) * (b[5] + b[6] * b[7]))
    return glucose_rate, sucrose_rate


def planted_programs(shape: KernelShape = FED_BATCH_SHAPE) -> tuple[Program, Program]:
    """Two generic rational ASTs whose leaf meanings come entirely from settings."""

    glucose_rate, sucrose_rate = _planted_expressions(shape)
    return Program.from_expression(glucose_rate), Program.from_expression(sucrose_rate)


def benchmark_genomes(
    count: int = 8,
    shape: KernelShape = FED_BATCH_SHAPE,
    variant: int = 0,
) -> tuple[SystemGenome, ...]:
    """Distinct, deterministic genomes for compact-dispatch correctness tests."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(variant, bool) or not isinstance(variant, int) or variant < 0:
        raise ValueError("variant must be a non-negative integer")
    glucose_rate, sucrose_rate = _planted_expressions(shape)
    scales = (1.0, 0.94, 1.06, 0.85, 1.15, 0.72, 1.28, 0.55)
    glucose_variant_scale = 1.0 + variant / 8192.0
    sucrose_variant_scale = 1.0 - variant / 16384.0
    if sucrose_variant_scale <= 0.0:
        raise ValueError("variant is too large for the finite benchmark family")
    genomes: list[SystemGenome] = []
    for index in range(count):
        scale = scales[index % len(scales)] * (1.0 + 0.01 * (index // len(scales)))
        genome = SystemGenome(
            (
                Program.from_expression(glucose_rate * (scale * glucose_variant_scale)),
                Program.from_expression(sucrose_rate * (scale * sucrose_variant_scale)),
            )
        )
        genome.validate(shape)
        genomes.append(genome)
    return tuple(genomes)


def _rates(state: Sequence[float], constants: Sequence[float]) -> tuple[float, float]:
    biomass, glucose, sucrose, _product = state
    mu_m1, kc1, k1, mu_m2, kc2, k2 = constants[:6]
    rate1 = mu_m1 * glucose / ((glucose + kc1 * biomass) * (1.0 + k1 * sucrose))
    rate2 = mu_m2 * sucrose / ((sucrose + kc2 * biomass) * (1.0 + k2 * glucose))
    return rate1, rate2


def _rhs(state: Sequence[float], rates: tuple[float, float]) -> tuple[float, float, float, float]:
    biomass = state[0]
    growth1 = rates[0] * biomass
    growth2 = rates[1] * biomass
    return (
        growth1 + growth2 - 0.0055 * biomass,
        -2.58 * growth1,
        -1.71 * growth2,
        0.21 * biomass - 0.0466 * biomass * biomass,
    )


def _rk4_step(state: Sequence[float], h: float, rate_function) -> tuple[float, ...]:
    k1 = _rhs(state, rate_function(state))
    stage = tuple(state[i] + 0.5 * h * k1[i] for i in range(4))
    k2 = _rhs(stage, rate_function(stage))
    stage = tuple(state[i] + 0.5 * h * k2[i] for i in range(4))
    k3 = _rhs(stage, rate_function(stage))
    stage = tuple(state[i] + h * k3[i] for i in range(4))
    k4 = _rhs(stage, rate_function(stage))
    return tuple(state[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(4))


def reference_data(
    steps_per_observation: int = 16,
    initial_states: Sequence[Sequence[float]] = INITIAL_STATES,
    observation_count: int = 12,
) -> array:
    return make_recovery_problem(
        initial_states,
        observation_count=observation_count,
        steps_per_observation=steps_per_observation,
    ).dense_reference_array()


def _fed_batch_rhs(
    _time: float,
    state_value: Sequence[float],
    parameters: dict[str, float],
) -> tuple[float, float, float, float]:
    constants = tuple(
        parameters[name]
        for name in ("mu_m1", "kc1", "k1", "mu_m2", "kc2", "k2", "one", "zero")
    )
    return _rhs(state_value, _rates(state_value, constants))


def make_recovery_problem(
    initial_states: Sequence[Sequence[float]] = INITIAL_STATES,
    *,
    observation_count: int = 12,
    steps_per_observation: int = 16,
) -> RecoveryProblem:
    """Describe the legacy fed-batch experiment through the generic problem ABI."""

    if not initial_states or any(len(initial) != 4 for initial in initial_states):
        raise ValueError("fed-batch recovery requires four-state initial conditions")
    if observation_count <= 0 or steps_per_observation <= 0:
        raise ValueError("fed-batch observation and RK4 step counts must be positive")
    parameter_names = ("mu_m1", "kc1", "k1", "mu_m2", "kc2", "k2", "one", "zero")
    parameters = tuple(
        ParameterSpec(name, value, estimated=index < 6)
        for index, (name, value) in enumerate(zip(parameter_names, PLANTED_CONSTANTS))
    )
    parameter_values = {parameter.name: parameter.nominal for parameter in parameters}
    interval = FED_BATCH_MODEL.observation_interval
    times = tuple(interval * index for index in range(observation_count + 1))
    series: list[ObservationSeries] = []
    step = interval / steps_per_observation
    for experiment, initial_raw in enumerate(initial_states):
        initial = tuple(float(value) for value in initial_raw)
        current = initial
        values = [current]
        derivatives = [_fed_batch_rhs(0.0, current, parameter_values)]
        for observation in range(observation_count):
            for _ in range(steps_per_observation):
                current = _rk4_step(
                    current,
                    step,
                    lambda value: _rates(value, PLANTED_CONSTANTS),
                )
            values.append(current)
            derivatives.append(_fed_batch_rhs(times[observation + 1], current, parameter_values))
        series.append(
            ObservationSeries(
                name=f"fed_batch_{experiment}",
                times=times,
                values=tuple(values),
                initial_state=initial,
                derivatives=tuple(derivatives),
            )
        )
    return RecoveryProblem(
        name=FED_BATCH_MODEL.name,
        protocol=RecoveryProtocol.TRAJECTORY,
        model=FED_BATCH_MODEL,
        parameters=parameters,
        search=StructureSearchSpec(
            operators=("add", "subtract", "multiply", "divide", "square"),
            max_nodes=30,
            constant_count=8,
        ),
        integration=IntegrationSpec(
            method="rk4_secant_candidate",
            start=0.0,
            stop=interval * observation_count,
            sample_count=observation_count + 1,
            reference_solver="fixed-step RK4 reproduction",
            reference_tolerances=f"{steps_per_observation} steps per {interval:g}-hour observation",
        ),
        training=tuple(series),
        truth_equations=(
            "(mu1+mu2-0.0055)*biomass",
            "-2.58*mu1*biomass",
            "-1.71*mu2*biomass",
            "0.21*biomass-0.0466*biomass^2",
        ),
        rhs=_fed_batch_rhs,
        provenance=ProblemProvenance(
            corpus="astaxanthin fed-batch reproduction",
            system_id="fed_batch_astaxanthin",
            source_url="https://doi.org/10.1002/bit.70328",
            data_protocol="dense aligned full-state trajectories",
            notes=(
                "Two specific-growth-rate sites are structurally hidden.",
                "The one and zero bank slots are fixed helpers, not fitted physical parameters.",
            ),
        ),
    )


FED_BATCH_PROBLEM = make_recovery_problem()


def make_population(
    num_settings: int,
    shape: KernelShape = FED_BATCH_SHAPE,
    seed: int = 7,
) -> Population:
    if num_settings <= 0:
        raise ValueError("num_settings must be positive")
    if shape.constant_count != len(PLANTED_CONSTANTS) or shape.input_count != len(PLANTED_BINDINGS):
        raise ValueError("the initial planted population requires 8 constants and 16 leaves")
    generator = random.Random(seed)
    settings = array("f", [0.0]) * (shape.constant_count * num_settings)
    bindings = array("I", [0]) * (shape.input_count * num_settings)
    for constant_index, planted in enumerate(PLANTED_CONSTANTS):
        for setting in range(num_settings):
            if setting == 0:
                value = planted
            elif constant_index == 5:
                value = generator.uniform(0.0, 0.5)
            elif constant_index == 6:
                value = generator.uniform(0.25, 2.0)
            elif constant_index == 7:
                value = generator.uniform(-1.0, 1.0)
            else:
                value = planted * math.exp(generator.uniform(-2.0, 2.0))
            settings[constant_index * num_settings + setting] = value
    for leaf_index, planted in enumerate(PLANTED_BINDINGS):
        for setting in range(num_settings):
            bindings[leaf_index * num_settings + setting] = (
                planted if setting == 0 else generator.randrange(shape.bank_slot_count)
            )
    return Population(num_settings, settings, bindings)


def pack_population(
    population: Population,
    num_genomes: int,
    shape: KernelShape = FED_BATCH_SHAPE,
) -> PackedPopulation:
    """Repeat a setting population in genome-major, plane-major layout."""

    if isinstance(num_genomes, bool) or not isinstance(num_genomes, int) or num_genomes <= 0:
        raise ValueError("num_genomes must be a positive integer")
    expected_settings = shape.constant_count * population.num_settings
    expected_bindings = shape.input_count * population.num_settings
    if len(population.settings) != expected_settings or len(population.bindings) != expected_bindings:
        raise ValueError("population dimensions do not match KernelShape")
    settings = array("f")
    bindings = array("I")
    for _genome in range(num_genomes):
        settings.extend(population.settings)
        bindings.extend(population.bindings)
    return PackedPopulation(num_genomes, population.num_settings, settings, bindings)


def setting_values(population: Population, setting: int, shape: KernelShape = FED_BATCH_SHAPE) -> tuple[list[float], list[int]]:
    constants = [population.settings[index * population.num_settings + setting] for index in range(shape.constant_count)]
    bindings = [population.bindings[index * population.num_settings + setting] for index in range(shape.input_count)]
    return constants, bindings


def packed_setting_values(
    population: PackedPopulation,
    genome: int,
    setting: int,
    shape: KernelShape = FED_BATCH_SHAPE,
) -> tuple[list[float], list[int]]:
    if not 0 <= genome < population.num_genomes:
        raise IndexError("genome is outside the packed population")
    if not 0 <= setting < population.num_settings:
        raise IndexError("setting is outside the packed population")
    constants = [
        population.settings[
            (genome * shape.constant_count + index) * population.num_settings + setting
        ]
        for index in range(shape.constant_count)
    ]
    bindings = [
        population.bindings[
            (genome * shape.input_count + index) * population.num_settings + setting
        ]
        for index in range(shape.input_count)
    ]
    return constants, bindings


def score_setting(
    programs: Sequence[Program],
    constants: Sequence[float],
    bindings: Sequence[int],
    reference: Sequence[float],
    steps_per_observation: int,
    shape: KernelShape = FED_BATCH_SHAPE,
    relative_error_floor: float | None = None,
) -> float:
    if len(programs) != 2 or len(constants) != shape.constant_count or len(bindings) != shape.input_count:
        raise ValueError("setting dimensions do not match the fed-batch shape")
    if steps_per_observation <= 0:
        raise ValueError("steps_per_observation must be positive")
    if relative_error_floor is not None and (
        not math.isfinite(relative_error_floor) or relative_error_floor <= 0.0
    ):
        raise ValueError("relative_error_floor must be finite and positive")
    for program in programs:
        program.validate(shape.input_count)
    target_offset = shape.state_count * shape.trajectory_count
    target_plane = shape.trajectory_count * shape.observation_count
    squared_error = 0.0
    h = 8.0 / steps_per_observation

    def dynamic_rates(state: Sequence[float]) -> tuple[float, float]:
        bank = tuple(state) + tuple(constants)
        leaves = [bank[index] for index in bindings]
        return (
            programs[0].evaluate_validated(leaves),
            programs[1].evaluate_validated(leaves),
        )

    try:
        for experiment in range(shape.trajectory_count):
            state = tuple(reference[component * shape.trajectory_count + experiment] for component in range(4))
            for observation in range(shape.observation_count):
                for _step in range(steps_per_observation):
                    state = _rk4_step(state, h, dynamic_rates)
                    if not all(math.isfinite(value) for value in state):
                        return INVALID_MSE
                target = experiment * shape.observation_count + observation
                for component in range(4):
                    target_value = reference[target_offset + component * target_plane + target]
                    error = state[component] - target_value
                    if relative_error_floor is not None:
                        error /= max(abs(target_value), relative_error_floor)
                    squared_error += error * error
        result = squared_error / (shape.state_count * shape.trajectory_count * shape.observation_count)
        return result if math.isfinite(result) and result < INVALID_MSE else INVALID_MSE
    except (ArithmeticError, ValueError):
        return INVALID_MSE


def score_packed_population(
    genomes: Sequence[SystemGenome],
    population: PackedPopulation,
    reference: Sequence[float],
    steps_per_observation: int,
    shape: KernelShape = FED_BATCH_SHAPE,
) -> array:
    """CPU-score every genome-major `(genome, setting)` configuration."""

    if len(genomes) != population.num_genomes:
        raise ValueError("genome count does not match the packed population")
    expected_settings = population.num_genomes * shape.constant_count * population.num_settings
    expected_bindings = population.num_genomes * shape.input_count * population.num_settings
    if len(population.settings) != expected_settings or len(population.bindings) != expected_bindings:
        raise ValueError("packed population dimensions do not match KernelShape")
    if len(reference) != shape.reference_float_count:
        raise ValueError("reference data does not match KernelShape")
    scores = array("f", [0.0]) * (population.num_genomes * population.num_settings)
    for genome_index, genome in enumerate(genomes):
        genome.validate(shape)
        for setting in range(population.num_settings):
            constants, bindings = packed_setting_values(
                population, genome_index, setting, shape
            )
            scores[genome_index * population.num_settings + setting] = score_setting(
                genome.programs,
                constants,
                bindings,
                reference,
                steps_per_observation,
                shape,
            )
    return scores
