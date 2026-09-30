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
from dataclasses import dataclass
import time
from typing import Sequence

from .runtime import Driver, _host_address
from .shape import KernelShape
from .toggle_fed_batch import TogglePopulation


@dataclass(frozen=True)
class ToggleModuleRunResult:
    mse: array
    seconds_per_sweep: float
    sweeps: int
    module_load_seconds: float
    function_lookup_seconds: float
    kernel_count: int
    execution_streams: int
    maximum_registers_per_thread: int
    maximum_local_bytes_per_thread: int


def run_toggle_cuda_module(
    cubin: bytes,
    kernel_names: Sequence[str],
    systems_per_kernel: int,
    population: TogglePopulation,
    reference: array,
    shape: KernelShape,
    steps_per_observation: int = 16,
    threads_per_block: int = 32,
    repetitions: int = 100,
    execution_streams: int = 16,
    device_ordinal: int = 0,
    configuration_count: int | None = None,
) -> ToggleModuleRunResult:
    """Run every independent toggle entry point as one resident module sweep."""

    if not kernel_names or systems_per_kernel <= 0:
        raise ValueError("toggle module requires kernels and systems per kernel")
    if population.num_systems != len(kernel_names) * systems_per_kernel:
        raise ValueError("toggle population does not match the module entry points")
    if repetitions <= 0 or not 0 < threads_per_block <= 1024:
        raise ValueError("repetitions and CUDA block size must be positive")
    if execution_streams <= 0 or steps_per_observation <= 0:
        raise ValueError("execution streams and integration steps must be positive")
    if len(reference) != shape.reference_float_count:
        raise ValueError("reference data does not match the toggle module shape")
    expected_constants = (
        population.num_systems * population.constant_bank_count * shape.constant_count
    )
    if len(population.constant_banks) != expected_constants:
        raise ValueError("constant banks do not match the toggle module population")
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
    streams: list[ctypes.c_void_p] = []
    try:
        image = ctypes.create_string_buffer(cubin)
        started = time.perf_counter()
        driver.check(driver.library.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        module_load_seconds = time.perf_counter() - started

        functions: list[ctypes.c_void_p] = []
        registers: list[int] = []
        local_bytes: list[int] = []
        started = time.perf_counter()
        for name in kernel_names:
            function = ctypes.c_void_p()
            driver.check(
                driver.library.cuModuleGetFunction(ctypes.byref(function), module, name.encode()),
                f"cuModuleGetFunction({name})",
            )
            register_count = ctypes.c_int()
            local_byte_count = ctypes.c_int()
            driver.check(
                driver.library.cuFuncGetAttribute(ctypes.byref(register_count), 4, function),
                f"cuFuncGetAttribute(NUM_REGS, {name})",
            )
            driver.check(
                driver.library.cuFuncGetAttribute(ctypes.byref(local_byte_count), 3, function),
                f"cuFuncGetAttribute(LOCAL_SIZE_BYTES, {name})",
            )
            functions.append(function)
            registers.append(register_count.value)
            local_bytes.append(local_byte_count.value)
        function_lookup_seconds = time.perf_counter() - started

        for _ in range(min(execution_streams, len(functions))):
            stream = ctypes.c_void_p()
            driver.check(driver.library.cuStreamCreate(ctypes.byref(stream), 1), "cuStreamCreate")
            streams.append(stream)

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
        configuration_tiles = (configurations + threads_per_block - 1) // threads_per_block
        shared_bytes = 4 * shape.reference_float_count
        parameter_sets = []
        for kernel_index in range(len(functions)):
            system_base = kernel_index * systems_per_kernel
            constants_offset = (
                system_base
                * population.constant_bank_count
                * shape.constant_count
                * 4
            )
            output_offset = system_base * configurations * 4
            holders = (
                ctypes.c_uint64(constants_device + constants_offset),
                ctypes.c_uint32(population.constant_bank_count),
                ctypes.c_uint32(configurations),
                ctypes.c_uint64(reference_device),
                ctypes.c_uint32(steps_per_observation),
                ctypes.c_uint64(mse_device + output_offset),
            )
            parameters = (ctypes.c_void_p * len(holders))(
                *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
            )
            parameter_sets.append((holders, parameters))

        def launch_sweep() -> None:
            for kernel_index, function in enumerate(functions):
                driver.check(
                    driver.library.cuLaunchKernel(
                        function,
                        configuration_tiles,
                        systems_per_kernel,
                        1,
                        threads_per_block,
                        1,
                        1,
                        shared_bytes,
                        streams[kernel_index % len(streams)],
                        parameter_sets[kernel_index][1],
                        None,
                    ),
                    f"cuLaunchKernel({kernel_names[kernel_index]})",
                )

        launch_sweep()
        driver.check(driver.library.cuCtxSynchronize(), "warm-up synchronization")
        started = time.perf_counter()
        for _ in range(repetitions):
            launch_sweep()
        driver.check(driver.library.cuCtxSynchronize(), "timed synchronization")
        elapsed = time.perf_counter() - started
        mse = array("f", [0.0]) * output_count
        driver.check(
            driver.library.cuMemcpyDtoH_v2(_host_address(mse), mse_device, 4 * output_count),
            "cuMemcpyDtoH",
        )
        return ToggleModuleRunResult(
            mse,
            elapsed / repetitions,
            repetitions,
            module_load_seconds,
            function_lookup_seconds,
            len(functions),
            len(streams),
            max(registers),
            max(local_bytes),
        )
    finally:
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        for stream in streams:
            driver.library.cuStreamDestroy_v2(stream)
        if module.value:
            driver.library.cuModuleUnload(module)
        driver.library.cuCtxDestroy_v2(context)
