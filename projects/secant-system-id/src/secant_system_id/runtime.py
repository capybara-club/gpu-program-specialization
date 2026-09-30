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
import ctypes.util
from dataclasses import dataclass
import time

from .fed_batch import PackedPopulation, Population
from .packed_cubin import PackedModulePlan
from .packed_template import PACKED_KERNEL_NAME
from .shape import KernelShape, PackedDispatch
from .template import KERNEL_NAME


class CudaDriverError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunResult:
    mse: array
    seconds_per_launch: float
    launches: int
    module_load_seconds: float = 0.0
    function_lookup_seconds: float = 0.0
    kernel_count: int = 1
    registers_per_thread: int | None = None
    local_bytes_per_thread: int | None = None


class Driver:
    def __init__(self):
        name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.library = ctypes.CDLL(name)
        except OSError as exc:
            raise CudaDriverError("the NVIDIA CUDA driver library was not found") from exc
        self._bind()

    def _bind(self) -> None:
        lib = self.library
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cuCtxSynchronize.argtypes = []
        for function in (
            lib.cuInit,
            lib.cuDeviceGet,
            lib.cuCtxCreate_v2,
            lib.cuCtxSetCurrent,
            lib.cuCtxDestroy_v2,
            lib.cuModuleLoadData,
            lib.cuModuleUnload,
            lib.cuModuleGetFunction,
            lib.cuFuncGetAttribute,
            lib.cuStreamCreate,
            lib.cuStreamDestroy_v2,
            lib.cuMemAlloc_v2,
            lib.cuMemFree_v2,
            lib.cuMemcpyHtoD_v2,
            lib.cuMemcpyDtoH_v2,
            lib.cuLaunchKernel,
            lib.cuCtxSynchronize,
        ):
            function.restype = ctypes.c_int

    @staticmethod
    def check(result: int, operation: str) -> None:
        if result != 0:
            raise CudaDriverError(f"{operation} failed with CUDA driver result {result}")


def _host_address(values: array) -> ctypes.c_void_p:
    if len(values) == 0:
        raise ValueError("cannot transfer an empty array")
    return ctypes.cast((ctypes.c_char * (len(values) * values.itemsize)).from_buffer(values), ctypes.c_void_p)


def run_fedbatch(
    cubin: bytes,
    population: Population,
    reference: array,
    shape: KernelShape = KernelShape(),
    steps_per_observation: int = 16,
    threads_per_block: int = 128,
    repetitions: int = 10,
    device_ordinal: int = 0,
) -> RunResult:
    if repetitions <= 0 or threads_per_block <= 0:
        raise ValueError("repetitions and threads_per_block must be positive")
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
        function_lookup_started = time.perf_counter()
        driver.check(
            driver.library.cuModuleGetFunction(ctypes.byref(function), module, KERNEL_NAME.encode()),
            "cuModuleGetFunction",
        )
        function_lookup_seconds = time.perf_counter() - function_lookup_started

        def allocate(values: array | None, byte_count: int | None = None) -> int:
            size = byte_count if byte_count is not None else len(values) * values.itemsize
            pointer = ctypes.c_uint64()
            driver.check(driver.library.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
            allocations.append(pointer.value)
            if values is not None:
                driver.check(driver.library.cuMemcpyHtoD_v2(pointer.value, _host_address(values), size), "cuMemcpyHtoD")
            return pointer.value

        settings_device = allocate(population.settings)
        bindings_device = allocate(population.bindings)
        reference_device = allocate(reference)
        mse_device = allocate(None, 4 * population.num_settings)
        holders = [
            ctypes.c_uint64(settings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(bindings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(reference_device),
            ctypes.c_uint32(steps_per_observation),
            ctypes.c_uint64(mse_device),
        ]
        parameters = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
        )
        blocks = (population.num_settings + threads_per_block - 1) // threads_per_block
        shared_bytes = shape.shared_bytes(threads_per_block)

        def launch() -> None:
            driver.check(
                driver.library.cuLaunchKernel(
                    function,
                    blocks, 1, 1,
                    threads_per_block, 1, 1,
                    shared_bytes, None,
                    parameters, None,
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
        mse = array("f", [0.0]) * population.num_settings
        driver.check(
            driver.library.cuMemcpyDtoH_v2(_host_address(mse), mse_device, 4 * population.num_settings),
            "cuMemcpyDtoH",
        )
        return RunResult(
            mse,
            elapsed / repetitions,
            repetitions,
            module_load_seconds,
            function_lookup_seconds,
            1,
        )
    finally:
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        if module.value:
            driver.library.cuModuleUnload(module)
        driver.library.cuCtxDestroy_v2(context)


def run_packed_fedbatch(
    cubin: bytes,
    population: PackedPopulation,
    reference: array,
    shape: KernelShape = KernelShape(),
    dispatch: PackedDispatch = PackedDispatch(),
    steps_per_observation: int = 16,
    threads_per_block: int = 128,
    repetitions: int = 10,
    device_ordinal: int = 0,
) -> RunResult:
    if repetitions <= 0 or threads_per_block <= 0:
        raise ValueError("repetitions and threads_per_block must be positive")
    if population.num_genomes > dispatch.genome_capacity:
        raise ValueError("population exceeds the packed CUBIN genome capacity")
    expected_settings = (
        population.num_genomes * shape.constant_count * population.num_settings
    )
    expected_bindings = (
        population.num_genomes * shape.input_count * population.num_settings
    )
    if len(population.settings) != expected_settings:
        raise ValueError("packed settings do not match the genome-major kernel layout")
    if len(population.bindings) != expected_bindings:
        raise ValueError("packed bindings do not match the genome-major kernel layout")
    if len(reference) != shape.reference_float_count:
        raise ValueError("reference data does not match KernelShape")
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
        function_lookup_started = time.perf_counter()
        driver.check(
            driver.library.cuModuleGetFunction(
                ctypes.byref(function), module, PACKED_KERNEL_NAME.encode()
            ),
            "cuModuleGetFunction",
        )
        function_lookup_seconds = time.perf_counter() - function_lookup_started

        def allocate(values: array | None, byte_count: int | None = None) -> int:
            size = byte_count if byte_count is not None else len(values) * values.itemsize
            pointer = ctypes.c_uint64()
            driver.check(driver.library.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
            allocations.append(pointer.value)
            if values is not None:
                driver.check(
                    driver.library.cuMemcpyHtoD_v2(
                        pointer.value, _host_address(values), size
                    ),
                    "cuMemcpyHtoD",
                )
            return pointer.value

        settings_device = allocate(population.settings)
        bindings_device = allocate(population.bindings)
        reference_device = allocate(reference)
        output_count = population.num_genomes * population.num_settings
        mse_device = allocate(None, 4 * output_count)
        holders = [
            ctypes.c_uint64(settings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(bindings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint32(population.num_genomes),
            ctypes.c_uint64(reference_device),
            ctypes.c_uint32(steps_per_observation),
            ctypes.c_uint64(mse_device),
        ]
        parameters = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
        )
        setting_tiles = (
            population.num_settings + threads_per_block - 1
        ) // threads_per_block
        genome_groups = (
            population.num_genomes + dispatch.genomes_per_cta - 1
        ) // dispatch.genomes_per_cta
        shared_bytes = shape.shared_bytes(threads_per_block)

        def launch() -> None:
            driver.check(
                driver.library.cuLaunchKernel(
                    function,
                    setting_tiles, genome_groups, 1,
                    threads_per_block, 1, 1,
                    shared_bytes, None,
                    parameters, None,
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
        )
    finally:
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        if module.value:
            driver.library.cuModuleUnload(module)
        driver.library.cuCtxDestroy_v2(context)


def run_packed_module_fedbatch(
    cubin: bytes,
    plan: PackedModulePlan,
    population: PackedPopulation,
    reference: array,
    shape: KernelShape = KernelShape(),
    steps_per_observation: int = 16,
    threads_per_block: int = 128,
    repetitions: int = 10,
    device_ordinal: int = 0,
) -> RunResult:
    """Load one CUBIN once and execute every populated packed entry point."""

    if repetitions <= 0 or threads_per_block <= 0:
        raise ValueError("repetitions and threads_per_block must be positive")
    if population.num_genomes > plan.genome_capacity:
        raise ValueError("population exceeds the packed module genome capacity")
    expected_settings = population.num_genomes * shape.constant_count * population.num_settings
    expected_bindings = population.num_genomes * shape.input_count * population.num_settings
    if len(population.settings) != expected_settings:
        raise ValueError("packed settings do not match the genome-major kernel layout")
    if len(population.bindings) != expected_bindings:
        raise ValueError("packed bindings do not match the genome-major kernel layout")
    if len(reference) != shape.reference_float_count:
        raise ValueError("reference data does not match KernelShape")
    for kernel in plan.kernels:
        if kernel.cubin.input_count != shape.input_count or kernel.cubin.output_count != shape.ast_count:
            raise ValueError("packed module kernel does not match KernelShape")

    active_kernels = [
        kernel for kernel in plan.kernels if kernel.spec.genome_base < population.num_genomes
    ]
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

        function_lookup_started = time.perf_counter()
        functions = []
        for kernel in active_kernels:
            function = ctypes.c_void_p()
            driver.check(
                driver.library.cuModuleGetFunction(
                    ctypes.byref(function), module, kernel.spec.name.encode()
                ),
                f"cuModuleGetFunction({kernel.spec.name})",
            )
            functions.append((kernel, function))
        function_lookup_seconds = time.perf_counter() - function_lookup_started

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

        settings_device = allocate(population.settings)
        bindings_device = allocate(population.bindings)
        reference_device = allocate(reference)
        output_count = population.num_genomes * population.num_settings
        mse_device = allocate(None, 4 * output_count)
        holders = [
            ctypes.c_uint64(settings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(bindings_device),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint64(population.num_settings),
            ctypes.c_uint32(population.num_genomes),
            ctypes.c_uint64(reference_device),
            ctypes.c_uint32(steps_per_observation),
            ctypes.c_uint64(mse_device),
        ]
        parameters = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
        )
        setting_tiles = (population.num_settings + threads_per_block - 1) // threads_per_block
        shared_bytes = shape.shared_bytes(threads_per_block)

        def launch_module() -> None:
            for kernel, function in functions:
                active_genomes = min(
                    kernel.spec.dispatch.genome_capacity,
                    population.num_genomes - kernel.spec.genome_base,
                )
                genome_groups = (
                    active_genomes + kernel.spec.dispatch.genomes_per_cta - 1
                ) // kernel.spec.dispatch.genomes_per_cta
                driver.check(
                    driver.library.cuLaunchKernel(
                        function,
                        setting_tiles, genome_groups, 1,
                        threads_per_block, 1, 1,
                        shared_bytes, None,
                        parameters, None,
                    ),
                    f"cuLaunchKernel({kernel.spec.name})",
                )

        launch_module()
        driver.check(driver.library.cuCtxSynchronize(), "warm-up synchronization")
        started = time.perf_counter()
        for _ in range(repetitions):
            launch_module()
        driver.check(driver.library.cuCtxSynchronize(), "timed synchronization")
        elapsed = time.perf_counter() - started
        mse = array("f", [0.0]) * output_count
        driver.check(
            driver.library.cuMemcpyDtoH_v2(_host_address(mse), mse_device, 4 * output_count),
            "cuMemcpyDtoH",
        )
        return RunResult(
            mse,
            elapsed / repetitions,
            repetitions,
            module_load_seconds,
            function_lookup_seconds,
            len(functions),
        )
    finally:
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        if module.value:
            driver.library.cuModuleUnload(module)
        driver.library.cuCtxDestroy_v2(context)
