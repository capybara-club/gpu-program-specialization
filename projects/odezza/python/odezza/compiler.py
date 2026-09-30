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
"""Python CUDA compilation tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from .lm_manifest import parse_lm_cuda_manifest
from .manifest import ManifestError, parse_cuda_manifest
from .model import SUPPORTED_ARCHITECTURES


class CudaPythonUnavailable(RuntimeError):
    pass


def _cuda_core():
    try:
        from cuda.core import Device, Program, ProgramOptions
    except (ImportError, ModuleNotFoundError) as exc:
        raise CudaPythonUnavailable(
            "CUDA Python is not installed; install the cuda13 optional dependency on a CUDA host"
        ) from exc
    return Device, Program, ProgramOptions


def _architecture(value: str | None, detected: int) -> str:
    result = value or f"sm_{detected}"
    match = re.fullmatch(r"sm_([0-9]+)", result)
    if match is None:
        raise ValueError("architecture must have the form sm_89, sm_90, or sm_120")
    numeric = int(match.group(1))
    if numeric not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(f"sm_{item}" for item in sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(f"the physical specializer supports {supported}, not {result}")
    return result


def compile_cuda_source(
    source: str,
    architecture: str | None = None,
    device_id: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    try:
        manifest, _ = parse_cuda_manifest(source)
    except ManifestError:
        manifest, _ = parse_lm_cuda_manifest(source)
    Device, Program, ProgramOptions = _cuda_core()
    device = Device(device_id)
    device.set_current()
    selected_architecture = _architecture(architecture, int(device.arch))
    options = ProgramOptions(
        name=f"odezza-{manifest['template_id'][:16]}",
        std="c++17",
        arch=selected_architecture,
        use_fast_math=True,
        ptxas_options=["-O3"],
    )
    program = Program(source, code_type="c++", options=options)
    started = time.perf_counter()
    try:
        object_code = program.compile("cubin")
        cubin = bytes(object_code.code)
        elapsed = time.perf_counter() - started
        close = getattr(object_code, "close", None)
        if close is not None:
            close()
    finally:
        program.close()
    report = {
        "schema": "odezza.cuda-compilation",
        "schema_version": 1,
        "template_id": manifest["template_id"],
        "architecture": selected_architecture,
        "device_id": device_id,
        "options": {
            "std": "c++17",
            "use_fast_math": True,
            "ptxas_options": ["-O3"],
        },
        "elapsed_seconds": elapsed,
        "cubin_byte_size": len(cubin),
        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
    }
    return cubin, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile a generated Odezza CUDA template to a CUBIN.")
    parser.add_argument("source")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--arch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", default="-")
    arguments = parser.parse_args(argv)
    cubin, report = compile_cuda_source(
        Path(arguments.source).read_text(),
        architecture=arguments.arch,
        device_id=arguments.device,
    )
    Path(arguments.output).write_bytes(cubin)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report == "-":
        print(rendered, end="")
    else:
        Path(arguments.report).write_text(rendered)


if __name__ == "__main__":
    main()
