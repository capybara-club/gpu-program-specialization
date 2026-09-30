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
import ctypes
import math
from pathlib import Path
import random
import time
from typing import Sequence

from .c_runtime import C99Library, C99Pipeline, CTrajectoryLMLaunch, GPCandidate
from .fed_batch import FED_BATCH_SHAPE
from .lm_specialization import render_partial_site, render_primal_site
from .runtime import Driver, _host_address


KERNEL_NAME = "secant_cubin_materialize_000"


@dataclass(frozen=True)
class TrajectoryLMPopulation:
    num_settings: int
    starts_per_setting: int
    starts: array
    bindings: array

    @property
    def num_fits(self) -> int:
        return self.num_settings * self.starts_per_setting


@dataclass(frozen=True)
class TrajectoryLMResult:
    constants: array
    mse: array
    iterations: array
    accepted_steps: array
    seconds: float
    module_load_seconds: float
    best_fit: int

    @property
    def best_mse(self) -> float:
        return float(self.mse[self.best_fit])


@dataclass(frozen=True)
class TrajectoryLMBatchResult:
    results: tuple[TrajectoryLMResult, ...]
    wall_seconds: float
    upload_seconds: float
    download_seconds: float
    ticket_total_seconds: tuple[float, ...]
    queue_wait_seconds: tuple[float, ...]
    image_copy_seconds: tuple[float, ...]
    module_load_seconds: tuple[float, ...]


def render_trajectory_lm_source(template_path: str | Path | None = None) -> str:
    """Render the fixed fed-batch LM control with both split SASS patch sites."""

    path = (
        Path(template_path)
        if template_path is not None
        else Path(__file__).resolve().parents[2]
        / "experiments"
        / "trajectory_lm"
        / "fedbatch_thread_lm.cu"
    )
    source = path.read_text()
    source = source.replace(
        '#include "fedbatch_primal_site.inc"', render_primal_site(FED_BATCH_SHAPE)
    )
    source = source.replace(
        '#include "fedbatch_partial_site.inc"', render_partial_site(FED_BATCH_SHAPE)
    )
    return "#define SSID_SPECIALIZED_GRADIENT 1\n" + source


def make_lm_population(
    candidate: GPCandidate,
    num_settings: int,
    starts_per_setting: int,
    seed: int,
    binding_keep_probability: float = 0.75,
    start_scale: float = 0.5,
) -> TrajectoryLMPopulation:
    """Create binding settings and parameter starts around one GP incumbent."""

    if num_settings <= 0 or starts_per_setting <= 0:
        raise ValueError("LM settings and starts must be positive")
    if len(candidate.constants) != FED_BATCH_SHAPE.constant_count:
        raise ValueError("trajectory LM requires eight GP constants")
    if len(candidate.bindings) != FED_BATCH_SHAPE.input_count:
        raise ValueError("trajectory LM requires sixteen GP leaf bindings")
    if not 0.0 <= binding_keep_probability <= 1.0 or start_scale < 0.0:
        raise ValueError("invalid trajectory-LM mutation controls")

    rng = random.Random(seed)
    num_fits = num_settings * starts_per_setting
    starts = array("f", [0.0]) * (FED_BATCH_SHAPE.constant_count * num_fits)
    bindings = array("I", [0]) * (FED_BATCH_SHAPE.input_count * num_settings)
    for setting in range(num_settings):
        for leaf, incumbent in enumerate(candidate.bindings):
            value = incumbent
            if setting != 0 and rng.random() >= binding_keep_probability:
                value = rng.randrange(FED_BATCH_SHAPE.bank_slot_count)
            bindings[leaf * num_settings + setting] = value
        for start_index in range(starts_per_setting):
            fit = setting * starts_per_setting + start_index
            for constant, incumbent in enumerate(candidate.constants):
                value = incumbent
                if start_index != 0 and constant < 6:
                    radius = (abs(incumbent) + 1.0) * start_scale
                    value += rng.uniform(-radius, radius)
                starts[constant * num_fits + fit] = value
    return TrajectoryLMPopulation(num_settings, starts_per_setting, starts, bindings)


class TrajectoryLMQueue:
    """Persistent C99 loader and fixed device buffers for promotion batches."""

    def __init__(
        self,
        api: C99Library,
        cubin_byte_count: int,
        reference: array,
        max_candidates: int,
        num_settings: int,
        starts_per_setting: int,
        workers: int = 2,
        maximum_loaded_modules: int = 4,
        device_ordinal: int = 0,
    ):
        if len(reference) != 832:
            raise ValueError("trajectory LM queue requires the product_paired16 reference")
        if min(
            cubin_byte_count,
            max_candidates,
            num_settings,
            starts_per_setting,
            workers,
            maximum_loaded_modules,
        ) <= 0:
            raise ValueError("trajectory LM queue dimensions must be positive")
        self.max_candidates = max_candidates
        self.num_settings = num_settings
        self.starts_per_setting = starts_per_setting
        self.num_fits = num_settings * starts_per_setting
        self.cubin_byte_count = cubin_byte_count
        self.pipeline: C99Pipeline | None = api.module_pipeline(
            cubin_byte_count,
            workers,
            max(max_candidates, maximum_loaded_modules),
            maximum_loaded_modules,
            device_ordinal,
            enable_cuda=True,
        )
        self.allocations: list[int] = []
        try:
            self.reference_device = self._allocate(len(reference) * reference.itemsize)
            self.starts_device = self._allocate(
                max_candidates * FED_BATCH_SHAPE.constant_count * self.num_fits * 4
            )
            self.bindings_device = self._allocate(
                max_candidates * FED_BATCH_SHAPE.input_count * num_settings * 4
            )
            self.constants_device = self._allocate(max_candidates * 6 * self.num_fits * 4)
            self.mse_device = self._allocate(max_candidates * self.num_fits * 4)
            self.iterations_device = self._allocate(max_candidates * self.num_fits * 4)
            self.accepted_steps_device = self._allocate(max_candidates * self.num_fits * 4)
            self.pipeline.upload(self.reference_device, reference)
        except Exception:
            self.close()
            raise

    def _allocate(self, byte_count: int) -> int:
        assert self.pipeline is not None
        pointer = self.pipeline.allocate(byte_count)
        self.allocations.append(pointer)
        return pointer

    def close(self) -> None:
        if self.pipeline is None:
            return
        for pointer in reversed(self.allocations):
            self.pipeline.free(pointer)
        self.allocations.clear()
        self.pipeline.close()
        self.pipeline = None

    def __enter__(self) -> "TrajectoryLMQueue":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def run(
        self,
        cubins: Sequence[bytes],
        populations: Sequence[TrajectoryLMPopulation],
        steps_per_observation: int = 16,
        max_lm_iterations: int = 20,
        max_damping_attempts: int = 8,
        initial_damping: float = 1.0e-3,
        threads_per_block: int = 128,
    ) -> TrajectoryLMBatchResult:
        if self.pipeline is None:
            raise RuntimeError("trajectory LM queue is closed")
        count = len(cubins)
        if count == 0 or count != len(populations) or count > self.max_candidates:
            raise ValueError("trajectory LM batch size is invalid")
        if any(len(cubin) != self.cubin_byte_count for cubin in cubins):
            raise ValueError("trajectory LM batch CUBIN sizes do not match the queue")
        if any(
            population.num_settings != self.num_settings
            or population.starts_per_setting != self.starts_per_setting
            for population in populations
        ):
            raise ValueError("trajectory LM batch populations do not match the queue")
        if (
            steps_per_observation <= 0
            or max_lm_iterations <= 0
            or max_damping_attempts <= 0
            or not math.isfinite(initial_damping)
            or initial_damping <= 0.0
            or threads_per_block <= 0
            or threads_per_block > 1024
        ):
            raise ValueError("invalid trajectory LM queue launch controls")

        starts = array("f")
        bindings = array("I")
        for population in populations:
            starts.extend(population.starts)
            bindings.extend(population.bindings)
        wall_started = time.perf_counter()
        upload_started = time.perf_counter()
        self.pipeline.upload(self.starts_device, starts)
        self.pipeline.upload(self.bindings_device, bindings)
        upload_seconds = time.perf_counter() - upload_started

        starts_stride_bytes = FED_BATCH_SHAPE.constant_count * self.num_fits * 4
        bindings_stride_bytes = FED_BATCH_SHAPE.input_count * self.num_settings * 4
        constants_stride_bytes = 6 * self.num_fits * 4
        fit_stride_bytes = self.num_fits * 4
        tickets = []
        for index, cubin in enumerate(cubins):
            launch = CTrajectoryLMLaunch(
                self.starts_device + index * starts_stride_bytes,
                self.num_fits,
                self.bindings_device + index * bindings_stride_bytes,
                self.num_settings,
                self.num_settings,
                self.starts_per_setting,
                self.reference_device,
                steps_per_observation,
                max_lm_iterations,
                max_damping_attempts,
                initial_damping,
                self.constants_device + index * constants_stride_bytes,
                self.num_fits,
                self.mse_device + index * fit_stride_bytes,
                self.iterations_device + index * fit_stride_bytes,
                self.accepted_steps_device + index * fit_stride_bytes,
                threads_per_block,
            )
            tickets.append(self.pipeline.submit_trajectory_lm(cubin, launch))

        ticket_results = []
        first_error: Exception | None = None
        for ticket in tickets:
            try:
                ticket_results.append(ticket.wait())
            except Exception as error:
                if first_error is None:
                    first_error = error
        for ticket in tickets:
            ticket.close()
        if first_error is not None:
            raise first_error

        download_started = time.perf_counter()
        constants = array("f", [0.0]) * (count * 6 * self.num_fits)
        mse = array("f", [0.0]) * (count * self.num_fits)
        iterations = array("I", [0]) * (count * self.num_fits)
        accepted_steps = array("I", [0]) * (count * self.num_fits)
        self.pipeline.download(self.constants_device, constants)
        self.pipeline.download(self.mse_device, mse)
        self.pipeline.download(self.iterations_device, iterations)
        self.pipeline.download(self.accepted_steps_device, accepted_steps)
        download_seconds = time.perf_counter() - download_started
        wall_seconds = time.perf_counter() - wall_started

        results = []
        for index, ticket_result in enumerate(ticket_results):
            constants_begin = index * 6 * self.num_fits
            fit_begin = index * self.num_fits
            candidate_constants = constants[
                constants_begin : constants_begin + 6 * self.num_fits
            ]
            candidate_mse = mse[fit_begin : fit_begin + self.num_fits]
            candidate_iterations = iterations[fit_begin : fit_begin + self.num_fits]
            candidate_accepted = accepted_steps[fit_begin : fit_begin + self.num_fits]
            finite = [
                fit for fit, value in enumerate(candidate_mse) if math.isfinite(value)
            ]
            best_fit = min(finite, key=candidate_mse.__getitem__) if finite else 0
            results.append(
                TrajectoryLMResult(
                    candidate_constants,
                    candidate_mse,
                    candidate_iterations,
                    candidate_accepted,
                    float(ticket_result.total_seconds),
                    float(ticket_result.module_load_seconds),
                    best_fit,
                )
            )
        return TrajectoryLMBatchResult(
            tuple(results),
            wall_seconds,
            upload_seconds,
            download_seconds,
            tuple(float(result.total_seconds) for result in ticket_results),
            tuple(float(result.queue_wait_seconds) for result in ticket_results),
            tuple(float(result.specialization_seconds) for result in ticket_results),
            tuple(float(result.module_load_seconds) for result in ticket_results),
        )


def run_trajectory_lm(
    cubin: bytes,
    population: TrajectoryLMPopulation,
    reference: array,
    steps_per_observation: int = 16,
    max_lm_iterations: int = 20,
    max_damping_attempts: int = 8,
    initial_damping: float = 1.0e-3,
    threads_per_block: int = 128,
    device_ordinal: int = 0,
) -> TrajectoryLMResult:
    """Run all thread-owned fits for one specialized two-site genome."""

    num_fits = population.num_fits
    if len(population.starts) != FED_BATCH_SHAPE.constant_count * num_fits:
        raise ValueError("trajectory-LM starts have the wrong layout")
    if len(population.bindings) != FED_BATCH_SHAPE.input_count * population.num_settings:
        raise ValueError("trajectory-LM bindings have the wrong layout")
    if len(reference) != 832:
        raise ValueError("trajectory LM currently requires the dense product_paired16 reference")
    if (
        steps_per_observation <= 0
        or max_lm_iterations <= 0
        or max_damping_attempts <= 0
        or threads_per_block <= 0
        or threads_per_block > 1024
        or not math.isfinite(initial_damping)
        or initial_damping <= 0.0
    ):
        raise ValueError("invalid trajectory-LM launch controls")

    driver = Driver()
    driver.check(driver.library.cuInit(0), "cuInit")
    device = ctypes.c_int()
    driver.check(driver.library.cuDeviceGet(ctypes.byref(device), device_ordinal), "cuDeviceGet")
    context = ctypes.c_void_p()
    driver.check(driver.library.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
    module = ctypes.c_void_p()
    allocations: list[int] = []
    try:
        image = ctypes.create_string_buffer(cubin)
        module_started = time.perf_counter()
        driver.check(driver.library.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        module_load_seconds = time.perf_counter() - module_started
        function = ctypes.c_void_p()
        driver.check(
            driver.library.cuModuleGetFunction(
                ctypes.byref(function), module, KERNEL_NAME.encode()
            ),
            f"cuModuleGetFunction({KERNEL_NAME})",
        )

        def allocate(values: array | None, byte_count: int | None = None) -> int:
            size = byte_count if byte_count is not None else len(values) * values.itemsize
            pointer = ctypes.c_uint64()
            driver.check(driver.library.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
            allocations.append(pointer.value)
            if values is not None:
                driver.check(
                    driver.library.cuMemcpyHtoD_v2(pointer.value, _host_address(values), size),
                    "cuMemcpyHtoD",
                )
            return pointer.value

        starts_device = allocate(population.starts)
        bindings_device = allocate(population.bindings)
        reference_device = allocate(reference)
        constants_device = allocate(None, 6 * num_fits * 4)
        mse_device = allocate(None, num_fits * 4)
        iterations_device = allocate(None, num_fits * 4)
        accepted_device = allocate(None, num_fits * 4)
        holders = [
            ctypes.c_uint64(starts_device),
            ctypes.c_uint64(num_fits),
            ctypes.c_uint64(bindings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint32(population.starts_per_setting),
            ctypes.c_uint64(reference_device),
            ctypes.c_uint32(steps_per_observation),
            ctypes.c_uint32(max_lm_iterations),
            ctypes.c_uint32(max_damping_attempts),
            ctypes.c_float(initial_damping),
            ctypes.c_uint64(constants_device),
            ctypes.c_uint64(num_fits),
            ctypes.c_uint64(mse_device),
            ctypes.c_uint64(iterations_device),
            ctypes.c_uint64(accepted_device),
        ]
        parameters = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
        )
        ctas = (num_fits + threads_per_block - 1) // threads_per_block
        shared_bytes = 12 * threads_per_block * 4
        started = time.perf_counter()
        driver.check(
            driver.library.cuLaunchKernel(
                function,
                1,
                ctas,
                1,
                threads_per_block,
                1,
                1,
                shared_bytes,
                None,
                parameters,
                None,
            ),
            "cuLaunchKernel(trajectory LM)",
        )
        driver.check(driver.library.cuCtxSynchronize(), "trajectory-LM synchronization")
        seconds = time.perf_counter() - started

        constants = array("f", [0.0]) * (6 * num_fits)
        mse = array("f", [0.0]) * num_fits
        iterations = array("I", [0]) * num_fits
        accepted_steps = array("I", [0]) * num_fits
        for output, pointer in (
            (constants, constants_device),
            (mse, mse_device),
            (iterations, iterations_device),
            (accepted_steps, accepted_device),
        ):
            driver.check(
                driver.library.cuMemcpyDtoH_v2(
                    _host_address(output), pointer, len(output) * output.itemsize
                ),
                "cuMemcpyDtoH",
            )
        finite = [index for index, value in enumerate(mse) if math.isfinite(value)]
        best_fit = min(finite, key=mse.__getitem__) if finite else 0
        return TrajectoryLMResult(
            constants,
            mse,
            iterations,
            accepted_steps,
            seconds,
            module_load_seconds,
            best_fit,
        )
    finally:
        if context.value:
            driver.library.cuCtxSetCurrent(context)
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        if module.value:
            driver.library.cuModuleUnload(module)
        if context.value:
            driver.library.cuCtxDestroy_v2(context)


def promoted_values(
    population: TrajectoryLMPopulation,
    result: TrajectoryLMResult,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    fit = result.best_fit
    setting = fit // population.starts_per_setting
    constants = tuple(
        float(result.constants[parameter * population.num_fits + fit])
        if parameter < 6
        else float(population.starts[parameter * population.num_fits + fit])
        for parameter in range(FED_BATCH_SHAPE.constant_count)
    )
    bindings = tuple(
        int(population.bindings[leaf * population.num_settings + setting])
        for leaf in range(FED_BATCH_SHAPE.input_count)
    )
    return constants, bindings
