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
"""Python CUDA execution tooling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from .compiler import CudaPythonUnavailable
from .elf import ElfImage
from .inspection import INSPECTION_SCHEMA, INSPECTION_SCHEMA_VERSION
from .manifest import TEMPLATE_ID_SYMBOL, shape_from_manifest
from .model import KERNEL_NAME, MAX_TOGGLE_BITS, KernelShape


RUN_REQUEST_SCHEMA = "odezza.run-request"
RUN_REQUEST_SCHEMA_VERSION = 1
RUN_REPORT_SCHEMA = "odezza.run-report"
RUN_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModuleInput:
    name: str
    cubin: bytes


@dataclass(frozen=True)
class ValidatedRun:
    manifest: dict[str, Any]
    shape: KernelShape
    system_count: int
    constant_bank_count: int
    active_toggle_count: int
    configuration_count: int
    flattened_constants: tuple[float, ...]
    reference_data: tuple[float, ...]


def _cuda_core():
    try:
        import numpy as np
        from cuda.core import Device, LaunchConfig, ObjectCode, PinnedMemoryResource, launch
    except (ImportError, ModuleNotFoundError) as exc:
        raise CudaPythonUnavailable(
            "CUDA Python and NumPy are required to load and run specialized CUBINs"
        ) from exc
    return np, Device, LaunchConfig, ObjectCode, PinnedMemoryResource, launch


def _finite_floats(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain real numbers") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def validate_run(
    modules: Sequence[ModuleInput],
    inspection: dict[str, Any],
    constant_banks: Sequence[Sequence[Sequence[float]]],
    reference_data: Sequence[float],
    active_toggle_count: int,
    steps_per_observation: int,
    threads_per_block: int,
    stream_count: int,
) -> ValidatedRun:
    if inspection.get("schema") != INSPECTION_SCHEMA or inspection.get("schema_version") != INSPECTION_SCHEMA_VERSION:
        raise ValueError("run inspection has an unsupported schema")
    if inspection.get("status") != "verified" or not isinstance(inspection.get("declared"), dict):
        raise ValueError("run inspection is not verified")
    if not modules:
        raise ValueError("at least one specialized module is required")
    manifest = inspection["declared"]
    shape = shape_from_manifest(manifest)
    if not constant_banks:
        raise ValueError("at least one system of constant banks is required")
    if len(constant_banks) > shape.system_capacity:
        raise ValueError("constant-bank system count exceeds the template capacity")

    flattened: list[float] = []
    bank_count: int | None = None
    for system_index, system_banks in enumerate(constant_banks):
        if not system_banks:
            raise ValueError(f"system {system_index} requires at least one constant bank")
        if bank_count is None:
            bank_count = len(system_banks)
        elif len(system_banks) != bank_count:
            raise ValueError("every system must have the same number of constant banks")
        for bank_index, bank in enumerate(system_banks):
            converted = _finite_floats(
                bank,
                f"system {system_index} constant bank {bank_index}",
            )
            if len(converted) != shape.constant_count:
                raise ValueError(
                    f"system {system_index} constant bank {bank_index} contains "
                    f"{len(converted)} values; expected {shape.constant_count}"
                )
            flattened.extend(converted)
    assert bank_count is not None
    if isinstance(active_toggle_count, bool) or not isinstance(active_toggle_count, int) or not 0 <= active_toggle_count <= MAX_TOGGLE_BITS:
        raise ValueError("active_toggle_count must be within the 32-bit permutation ABI")

    reference = _finite_floats(reference_data, "reference_data")
    if len(reference) != shape.reference_float_count:
        raise ValueError(
            f"reference_data contains {len(reference)} values; "
            f"expected {shape.reference_float_count}"
        )
    for label, value in (
        ("steps_per_observation", steps_per_observation),
        ("threads_per_block", threads_per_block),
        ("stream_count", stream_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if threads_per_block > 1024:
        raise ValueError("threads_per_block exceeds the CUDA architectural limit")

    configuration_count = bank_count * (1 << active_toggle_count)
    if configuration_count > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("configuration count exceeds the kernel's 64-bit index space")

    expected_template_id = bytes.fromhex(manifest["template_id"])
    expected_architecture = int(str(inspection["observed"]["architecture"]).removeprefix("sm_"))
    for module in modules:
        if not module.name:
            raise ValueError("module names must be nonempty")
        elf = ElfImage(module.cubin)
        if elf.architecture != expected_architecture:
            raise ValueError(
                f"module {module.name!r} architecture does not match the template inspection"
            )
        symbol = elf.find_data_symbol(TEMPLATE_ID_SYMBOL, len(expected_template_id))
        actual = module.cubin[symbol.file_offset : symbol.file_offset + symbol.size]
        if actual != expected_template_id:
            raise ValueError(f"module {module.name!r} belongs to a different CUDA template")

    return ValidatedRun(
        manifest,
        shape,
        len(constant_banks),
        bank_count,
        active_toggle_count,
        configuration_count,
        tuple(flattened),
        reference,
    )


def _close(value: object, *arguments: object) -> None:
    close = getattr(value, "close", None)
    if close is not None:
        close(*arguments)


def run_modules(
    modules: Sequence[ModuleInput],
    inspection: dict[str, Any],
    constant_banks: Sequence[Sequence[Sequence[float]]],
    reference_data: Sequence[float],
    *,
    active_toggle_count: int,
    steps_per_observation: int,
    threads_per_block: int = 128,
    stream_count: int = 8,
    device_id: int = 0,
) -> dict[str, Any]:
    validated = validate_run(
        modules,
        inspection,
        constant_banks,
        reference_data,
        active_toggle_count,
        steps_per_observation,
        threads_per_block,
        stream_count,
    )
    np, Device, LaunchConfig, ObjectCode, PinnedMemoryResource, launch = _cuda_core()
    device = Device(device_id)
    device.set_current()
    observed_architecture = int(str(inspection["observed"]["architecture"]).removeprefix("sm_"))
    if int(device.arch) != observed_architecture:
        raise ValueError(
            f"device {device_id} is sm_{int(device.arch)}, "
            f"but the specialized modules are sm_{observed_architecture}"
        )

    streams = [device.create_stream() for _ in range(min(stream_count, len(modules)))]
    setup_stream = streams[0]
    pinned = PinnedMemoryResource()
    float_size = np.dtype(np.float32).itemsize
    constant_host = pinned.allocate(
        len(validated.flattened_constants) * float_size,
        stream=setup_stream,
    )
    reference_host = pinned.allocate(
        len(validated.reference_data) * float_size,
        stream=setup_stream,
    )
    np.from_dlpack(constant_host).view(np.float32)[:] = validated.flattened_constants
    np.from_dlpack(reference_host).view(np.float32)[:] = validated.reference_data
    constant_device = device.allocate(constant_host.size, stream=setup_stream)
    reference_device = device.allocate(reference_host.size, stream=setup_stream)
    constant_device.copy_from(constant_host, stream=setup_stream)
    reference_device.copy_from(reference_host, stream=setup_stream)
    setup_stream.sync()

    loaded: list[object] = []
    output_device: list[object] = []
    output_host: list[object] = []
    names: list[str] = []
    output_count = validated.system_count * validated.configuration_count
    grid_x = (
        validated.configuration_count + threads_per_block - 1
    ) // threads_per_block
    launch_config = LaunchConfig(
        grid=(grid_x, validated.system_count, 1),
        block=threads_per_block,
        shmem_size=validated.shape.shared_bytes,
    )
    dispatch_started = time.perf_counter()
    try:
        for index, module_input in enumerate(modules):
            stream = streams[index % len(streams)]
            host = pinned.allocate(output_count * float_size, stream=stream)
            device_output = device.allocate(host.size, stream=stream)
            module = ObjectCode.from_cubin(module_input.cubin, name=module_input.name)
            kernel = module.get_kernel(KERNEL_NAME)
            launch(
                stream,
                launch_config,
                kernel,
                constant_device,
                np.uint32(validated.constant_bank_count),
                reference_device,
                np.uint32(validated.active_toggle_count),
                np.uint32(validated.system_count),
                np.uint32(steps_per_observation),
                device_output,
            )
            device_output.copy_to(host, stream=stream)
            loaded.append(module)
            output_device.append(device_output)
            output_host.append(host)
            names.append(module_input.name)
        dispatch_elapsed = time.perf_counter() - dispatch_started
        for stream in streams:
            stream.sync()
        total_elapsed = time.perf_counter() - dispatch_started
        flat_scores = [
            [float(value) for value in np.from_dlpack(host).view(np.float32)]
            for host in output_host
        ]
    finally:
        for module in loaded:
            _close(module)
        for index, buffer in enumerate(output_device):
            _close(buffer, streams[index % len(streams)])
        for index, buffer in enumerate(output_host):
            _close(buffer, streams[index % len(streams)])
        _close(constant_device, setup_stream)
        _close(reference_device, setup_stream)
        _close(constant_host, setup_stream)
        _close(reference_host, setup_stream)
        for stream in streams:
            _close(stream)
        _close(pinned)

    module_scores = [
        [
            values[
                system_index * validated.configuration_count
                : (system_index + 1) * validated.configuration_count
            ]
            for system_index in range(validated.system_count)
        ]
        for values in flat_scores
    ]
    return {
        "schema": RUN_REPORT_SCHEMA,
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "template_id": validated.manifest["template_id"],
        "architecture": f"sm_{observed_architecture}",
        "device_id": device_id,
        "module_count": len(modules),
        "system_count": validated.system_count,
        "constant_bank_count": validated.constant_bank_count,
        "active_toggle_count": validated.active_toggle_count,
        "toggle_permutations": 1 << validated.active_toggle_count,
        "configuration_count_per_system": validated.configuration_count,
        "threads_per_block": threads_per_block,
        "stream_count": len(streams),
        "dispatch_seconds": dispatch_elapsed,
        "total_seconds": total_elapsed,
        "modules": [
            {
                "name": names[module_index],
                "cubin_sha256": hashlib.sha256(
                    modules[module_index].cubin
                ).hexdigest(),
                "systems": [
                    {
                        "system": system_index,
                        "best_configuration": min(
                            range(len(values)),
                            key=values.__getitem__,
                        ),
                        "best_mse": min(values),
                        "mse": values,
                    }
                    for system_index, values in enumerate(
                        module_scores[module_index]
                    )
                ],
            }
            for module_index in range(len(modules))
        ],
    }


def load_run_request(path: Path) -> tuple[dict[str, Any], list[ModuleInput], dict[str, Any]]:
    document = json.loads(path.read_text())
    if document.get("schema") != RUN_REQUEST_SCHEMA or document.get("schema_version") != RUN_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported run-request document")
    root = path.resolve().parent
    inspection = json.loads((root / document["inspection"]).read_text())
    raw_modules = document.get("modules")
    if not isinstance(raw_modules, list):
        raise ValueError("run request modules must be an array")
    modules = [
        ModuleInput(str(item["name"]), (root / str(item["cubin"])).read_bytes())
        for item in raw_modules
    ]
    return document, modules, inspection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load and run multi-system scoring CUBINs."
    )
    parser.add_argument("request")
    parser.add_argument("-o", "--output", default="-")
    arguments = parser.parse_args(argv)
    request, modules, inspection = load_run_request(Path(arguments.request))
    report = run_modules(
        modules,
        inspection,
        request["constant_banks"],
        request["reference_data"],
        active_toggle_count=int(request["active_toggle_count"]),
        steps_per_observation=int(request["steps_per_observation"]),
        threads_per_block=int(request.get("threads_per_block", 128)),
        stream_count=int(request.get("stream_count", 8)),
        device_id=int(request.get("device_id", 0)),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output == "-":
        print(rendered, end="")
    else:
        Path(arguments.output).write_text(rendered)


if __name__ == "__main__":
    main()
