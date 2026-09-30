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
import ctypes
import time

from .runtime import Driver, RunResult, _host_address
from .shape import KernelShape
from .toggle_fed_batch import TogglePopulation
from .toggle_template import TOGGLE_KERNEL_NAME


def run_toggle_cuda(
    cubin: bytes,
    population: TogglePopulation,
    reference: array,
    shape: KernelShape,
    steps_per_observation: int = 16,
    threads_per_block: int = 128,
    repetitions: int = 20,
    device_ordinal: int = 0,
    configuration_count: int | None = None,
) -> RunResult:
    """Run the CUDA-compiled toggle prototype with one system per CTA row."""

    if repetitions <= 0 or threads_per_block <= 0 or steps_per_observation <= 0:
        raise ValueError("repetitions, threads, and integration steps must be positive")
    if len(reference) != shape.reference_float_count:
        raise ValueError("reference data does not match the toggle kernel shape")
    expected_constants = (
        population.num_systems * population.constant_bank_count * shape.constant_count
    )
    if len(population.constant_banks) != expected_constants:
        raise ValueError("constant-bank data does not match the toggle population")
    available_configurations = population.configurations_per_system
    configurations = (
        available_configurations if configuration_count is None else configuration_count
    )
    if not 0 < configurations <= available_configurations:
        raise ValueError("configuration count exceeds the available toggle population")

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
        module_load_started = time.perf_counter()
        driver.check(driver.library.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        module_load_seconds = time.perf_counter() - module_load_started

        function = ctypes.c_void_p()
        lookup_started = time.perf_counter()
        driver.check(
            driver.library.cuModuleGetFunction(
                ctypes.byref(function), module, TOGGLE_KERNEL_NAME.encode()
            ),
            "cuModuleGetFunction",
        )
        function_lookup_seconds = time.perf_counter() - lookup_started
        registers_per_thread = ctypes.c_int()
        local_bytes_per_thread = ctypes.c_int()
        driver.check(
            driver.library.cuFuncGetAttribute(
                ctypes.byref(registers_per_thread), 4, function
            ),
            "cuFuncGetAttribute(NUM_REGS)",
        )
        driver.check(
            driver.library.cuFuncGetAttribute(
                ctypes.byref(local_bytes_per_thread), 3, function
            ),
            "cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
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

        constants_device = allocate(population.constant_banks)
        reference_device = allocate(reference)
        output_count = population.num_systems * configurations
        mse_device = allocate(None, 4 * output_count)
        holders = [
            ctypes.c_uint64(constants_device),
            ctypes.c_uint32(population.constant_bank_count),
            ctypes.c_uint32(configurations),
            ctypes.c_uint64(reference_device),
            ctypes.c_uint32(steps_per_observation),
            ctypes.c_uint64(mse_device),
        ]
        parameters = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
        )
        configuration_tiles = (
            configurations + threads_per_block - 1
        ) // threads_per_block
        shared_bytes = 4 * shape.reference_float_count

        def launch() -> None:
            driver.check(
                driver.library.cuLaunchKernel(
                    function,
                    configuration_tiles,
                    population.num_systems,
                    1,
                    threads_per_block,
                    1,
                    1,
                    shared_bytes,
                    None,
                    parameters,
                    None,
                ),
                "cuLaunchKernel",
            )

        launch()
        driver.check(driver.library.cuCtxSynchronize(), "warm-up synchronization")
        started = time.perf_counter()
        for _ in range(repetitions):
            launch()
        driver.check(driver.library.cuCtxSynchronize(), "timed synchronization")
        elapsed = time.perf_counter() - started
        mse = array("f", [0.0]) * output_count
        driver.check(
            driver.library.cuMemcpyDtoH_v2(
                _host_address(mse), mse_device, 4 * output_count
            ),
            "cuMemcpyDtoH",
        )
        return RunResult(
            mse,
            elapsed / repetitions,
            repetitions,
            module_load_seconds,
            function_lookup_seconds,
            1,
            registers_per_thread.value,
            local_bytes_per_thread.value,
        )
    finally:
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        if module.value:
            driver.library.cuModuleUnload(module)
        driver.library.cuCtxDestroy_v2(context)
