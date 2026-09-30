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

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import struct
from typing import Mapping, Sequence
import zipfile

from .problem import NoiseSpec, ObservationSeries, RecoveryProblem, RhsFunction


MDBENCH_SOURCE = "https://github.com/gryaklab/mdbench"
MDBENCH_DATA_DOI = "https://doi.org/10.5281/zenodo.17611099"


@dataclass(frozen=True)
class NumericArray:
    shape: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class MDBenchSplit:
    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]
    approximate_derivatives: tuple[tuple[float, ...], ...]
    true_derivatives: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class MDBenchDataset:
    name: str
    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]
    true_derivatives: tuple[tuple[float, ...], ...]
    approximate_derivatives: tuple[tuple[float, ...], ...]
    noise: NoiseSpec = NoiseSpec()

    def __post_init__(self) -> None:
        if not self.name or len(self.times) < 3:
            raise ValueError("MDBench datasets require a name and at least three time points")
        if len(self.states) != len(self.times) or len(self.true_derivatives) != len(self.times):
            raise ValueError("MDBench arrays must align on the time dimension")
        if len(self.approximate_derivatives) != len(self.times):
            raise ValueError("approximate derivatives must align on the time dimension")
        width = len(self.states[0])
        if width <= 0 or any(len(row) != width for row in self.states):
            raise ValueError("MDBench state data must be a non-empty rectangular matrix")
        if any(len(row) != width for row in self.true_derivatives + self.approximate_derivatives):
            raise ValueError("MDBench derivative matrices must match state width")
        if any(not math.isfinite(value) for value in self.times):
            raise ValueError("MDBench times must be finite")
        if any(self.times[index] <= self.times[index - 1] for index in range(1, len(self.times))):
            raise ValueError("MDBench times must be strictly increasing")
        matrices = self.states + self.true_derivatives + self.approximate_derivatives
        if any(not math.isfinite(value) for row in matrices for value in row):
            raise ValueError("MDBench state and derivative values must be finite")

    @property
    def state_count(self) -> int:
        return len(self.states[0])

    def split(self, test_ratio: float = 0.2) -> tuple[MDBenchSplit, MDBenchSplit]:
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("test_ratio must lie strictly between zero and one")
        test_count = int(len(self.times) * test_ratio)
        if test_count <= 0 or test_count >= len(self.times):
            raise ValueError("test split would be empty")
        cutoff = len(self.times) - test_count

        def section(start: int, stop: int) -> MDBenchSplit:
            return MDBenchSplit(
                self.times[start:stop],
                self.states[start:stop],
                self.approximate_derivatives[start:stop],
                self.true_derivatives[start:stop],
            )

        return section(0, cutoff), section(cutoff, len(self.times))

    def as_observation_series(self) -> ObservationSeries:
        return ObservationSeries(
            name=self.name,
            times=self.times,
            values=self.states,
            initial_state=self.states[0],
            derivatives=self.true_derivatives,
            noise=self.noise,
        )


def _decode_npy(payload: bytes) -> NumericArray:
    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise ValueError("NPZ member is not a NumPy NPY array")
    major, minor = payload[6], payload[7]
    if major == 1:
        header_size, prefix_size = struct.unpack_from("<H", payload, 8)[0], 10
    elif major in {2, 3}:
        header_size, prefix_size = struct.unpack_from("<I", payload, 8)[0], 12
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")
    header_end = prefix_size + header_size
    try:
        header = ast.literal_eval(payload[prefix_size:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid NPY header") from error
    if header.get("fortran_order"):
        raise ValueError("Fortran-order NPY arrays are not supported")
    shape = header.get("shape")
    descriptor = header.get("descr")
    if not isinstance(shape, tuple) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
    ):
        raise ValueError("invalid NPY shape")
    formats = {
        "<f4": "<f",
        "=f4": "=f",
        "|f4": "=f",
        "<f8": "<d",
        "=f8": "=d",
        "|f8": "=d",
        ">f4": ">f",
        ">f8": ">d",
    }
    code = formats.get(descriptor)
    if code is None:
        raise ValueError(f"unsupported NPY dtype {descriptor!r}; expected float32 or float64")
    count = math.prod(shape)
    item_size = struct.calcsize(code)
    body = payload[header_end:]
    if len(body) != count * item_size:
        raise ValueError("NPY payload length does not match its shape")
    values = tuple(value[0] for value in struct.iter_unpack(code, body))
    return NumericArray(shape, values)


def _matrix(array: NumericArray, name: str) -> tuple[tuple[float, ...], ...]:
    if len(array.shape) != 2 or min(array.shape) <= 0:
        raise ValueError(f"MDBench {name} must be a non-empty two-dimensional array")
    rows, columns = array.shape
    return tuple(
        array.values[row * columns : (row + 1) * columns]
        for row in range(rows)
    )


def finite_difference(
    times: Sequence[float], states: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    """Second-order finite differences for adapter validation.

    MDBench itself uses ``findiff``. This dependency-free version is not claimed
    to be byte-for-byte identical; official benchmark runs should use MDBench's
    supplied training derivatives or its own evaluator.
    """

    if len(times) != len(states) or len(times) < 3:
        raise ValueError("finite differences require at least three aligned samples")
    width = len(states[0])
    if any(len(row) != width for row in states):
        raise ValueError("state rows must have equal width")
    result: list[tuple[float, ...]] = []
    for index in range(len(times)):
        if index == 0:
            left, right = 0, 1
        elif index == len(times) - 1:
            left, right = index - 1, index
        else:
            left, right = index - 1, index + 1
        duration = times[right] - times[left]
        result.append(tuple((states[right][column] - states[left][column]) / duration for column in range(width)))
    return tuple(result)


def load_npz(path: str | Path) -> MDBenchDataset:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = {name.rsplit("/", 1)[-1]: name for name in archive.namelist()}
        missing = {name for name in ("t.npy", "u.npy", "du.npy") if name not in members}
        if missing:
            raise ValueError("MDBench NPZ is missing: " + ", ".join(sorted(missing)))
        times_array = _decode_npy(archive.read(members["t.npy"]))
        states_array = _decode_npy(archive.read(members["u.npy"]))
        derivatives_array = _decode_npy(archive.read(members["du.npy"]))
    if len(times_array.shape) != 1:
        raise ValueError("MDBench t must be one-dimensional")
    states = _matrix(states_array, "u")
    derivatives = _matrix(derivatives_array, "du")
    times = times_array.values
    if len(times) != len(states) or len(times) != len(derivatives):
        raise ValueError("MDBench t, u, and du disagree on sample count")
    return MDBenchDataset(
        name=path.stem,
        times=times,
        states=states,
        true_derivatives=derivatives,
        approximate_derivatives=finite_difference(times, states),
    )


def dataset_from_problem(problem: RecoveryProblem) -> MDBenchDataset:
    if len(problem.training) != 1:
        raise ValueError("MDBench ODE protocol uses one trajectory per dataset")
    series = problem.training[0]
    if not series.has_complete_derivatives or not series.is_complete:
        raise ValueError("MDBench conversion requires complete states and true derivatives")
    states = tuple(tuple(float(value) for value in row) for row in series.values)
    assert series.derivatives is not None
    derivatives = tuple(tuple(float(value) for value in row) for row in series.derivatives)
    return MDBenchDataset(
        name=series.name,
        times=series.times,
        states=states,
        true_derivatives=derivatives,
        approximate_derivatives=finite_difference(series.times, states),
        noise=series.noise,
    )


def add_multiplicative_snr_noise(
    dataset: MDBenchDataset, snr_db: float, *, seed: int
) -> MDBenchDataset:
    """Match MDBench's ``u * (1 + noise)`` semantics with a deterministic local RNG."""

    if not math.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    generator = random.Random(seed)
    standard_deviation = math.sqrt(10.0 ** (-snr_db / 10.0))
    states = tuple(
        tuple(value * (1.0 + generator.gauss(0.0, standard_deviation)) for value in row)
        for row in dataset.states
    )
    return MDBenchDataset(
        name=f"{dataset.name}_snr_{snr_db:g}",
        times=dataset.times,
        states=states,
        true_derivatives=dataset.true_derivatives,
        approximate_derivatives=finite_difference(dataset.times, states),
        noise=NoiseSpec(
            kind="multiplicative_gaussian_snr_db",
            level=snr_db,
            seed=seed,
            target_semantics="exact derivative of clean trajectory retained by MDBench",
        ),
    )


def nmse(expected: Sequence[Sequence[float]], actual: Sequence[Sequence[float]]) -> float:
    if len(expected) != len(actual) or any(len(a) != len(b) for a, b in zip(expected, actual)):
        raise ValueError("NMSE inputs must have identical matrix shapes")
    squared_error = sum(
        (target - prediction) ** 2
        for target_row, prediction_row in zip(expected, actual)
        for target, prediction in zip(target_row, prediction_row)
    )
    squared_target = sum(value * value for row in expected for value in row)
    return squared_error / (squared_target + 1.0e-10)


def oracle_test_nmse(
    dataset: MDBenchDataset,
    rhs: RhsFunction,
    parameters: Mapping[str, float],
) -> float:
    _train, test = dataset.split()
    predictions = tuple(
        rhs(time, state, parameters) for time, state in zip(test.times, test.states)
    )
    return nmse(test.true_derivatives, predictions)


def run_reference_pilot(*, snr_db: float = 20.0, seed: int = 0) -> dict[str, object]:
    """Exercise clean/noisy MDBench semantics with known RHS functions.

    This is an adapter/correctness pilot, not a symbolic-search benchmark.
    """

    from .odebench import ODEBENCH_REFERENCE_SYSTEMS, make_reference_problem
    from .problem import RecoveryProtocol

    systems: list[dict[str, object]] = []
    for index, definition in enumerate(ODEBENCH_REFERENCE_SYSTEMS):
        problem = make_reference_problem(
            definition,
            protocol=RecoveryProtocol.DERIVATIVE_REGRESSION,
        )
        clean = dataset_from_problem(problem)
        noisy = add_multiplicative_snr_noise(clean, snr_db, seed=seed + index)
        assert problem.rhs is not None
        systems.append(
            {
                "system_id": definition.system_id,
                "name": definition.name,
                "samples": len(clean.times),
                "states": clean.state_count,
                "clean_oracle_test_nmse": oracle_test_nmse(clean, problem.rhs, problem.nominal_parameters),
                "noisy_oracle_test_nmse": oracle_test_nmse(noisy, problem.rhs, problem.nominal_parameters),
            }
        )
    return {
        "kind": "adapter_correctness_pilot_not_search",
        "corpus": "MDBench ODE / ODEBench shared systems",
        "snr_db": snr_db,
        "noise_rng": "Python random.Random; semantics match MDBench but samples differ from NumPy",
        "official_artifact_used": False,
        "systems": systems,
    }


def main() -> None:
    print(json.dumps(run_reference_pilot(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
