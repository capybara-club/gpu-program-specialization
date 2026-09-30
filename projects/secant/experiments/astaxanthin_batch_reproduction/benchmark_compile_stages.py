#!/usr/bin/env python3
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
"""Measure NVRTC CUDA-to-CUBIN and CUDA-to-PTX-to-CUBIN compilation."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time


class Nvrtc:
    def __init__(self, library: str):
        self.library = ctypes.CDLL(library)
        self.library.nvrtcVersion.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.nvrtcVersion.restype = ctypes.c_int
        self.library.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        self.library.nvrtcGetErrorString.restype = ctypes.c_char_p
        self.library.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.nvrtcCreateProgram.restype = ctypes.c_int
        self.library.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.nvrtcCompileProgram.restype = ctypes.c_int
        self.library.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.nvrtcGetProgramLogSize.restype = ctypes.c_int
        self.library.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.nvrtcGetProgramLog.restype = ctypes.c_int
        self.library.nvrtcGetCUBINSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.nvrtcGetCUBINSize.restype = ctypes.c_int
        self.library.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.nvrtcGetCUBIN.restype = ctypes.c_int
        self.library.nvrtcGetPTXSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.nvrtcGetPTXSize.restype = ctypes.c_int
        self.library.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.nvrtcGetPTX.restype = ctypes.c_int
        self.library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvrtcDestroyProgram.restype = ctypes.c_int

    def check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        error = self.library.nvrtcGetErrorString(result).decode("utf-8", "replace")
        raise RuntimeError(f"{operation}: {error}")

    def version(self) -> str:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self.check(self.library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)),
                   "nvrtcVersion")
        return f"{major.value}.{minor.value}"

    def program_log(self, program: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        self.check(self.library.nvrtcGetProgramLogSize(program, ctypes.byref(size)),
                   "nvrtcGetProgramLogSize")
        if size.value <= 1:
            return ""
        buffer = ctypes.create_string_buffer(size.value)
        self.check(self.library.nvrtcGetProgramLog(program, buffer),
                   "nvrtcGetProgramLog")
        return buffer.value.decode("utf-8", "replace")

    def compile(self, source: bytes, name: str, options: list[str], artifact: str) -> tuple[bytes, dict[str, float]]:
        program = ctypes.c_void_p()
        encoded_options = [option.encode() for option in options]
        option_array = (ctypes.c_char_p * len(encoded_options))(*encoded_options)
        total_begin = time.perf_counter_ns()
        create_begin = total_begin
        self.check(
            self.library.nvrtcCreateProgram(
                ctypes.byref(program), source, name.encode(), 0, None, None
            ),
            "nvrtcCreateProgram",
        )
        create_end = time.perf_counter_ns()
        try:
            compile_begin = time.perf_counter_ns()
            result = self.library.nvrtcCompileProgram(
                program, len(encoded_options), option_array
            )
            compile_end = time.perf_counter_ns()
            if result != 0:
                log = self.program_log(program)
                self.check(result, f"nvrtcCompileProgram\n{log}")

            artifact_begin = time.perf_counter_ns()
            size = ctypes.c_size_t()
            if artifact == "cubin":
                self.check(self.library.nvrtcGetCUBINSize(program, ctypes.byref(size)),
                           "nvrtcGetCUBINSize")
                buffer = ctypes.create_string_buffer(size.value)
                self.check(self.library.nvrtcGetCUBIN(program, buffer), "nvrtcGetCUBIN")
                output = buffer.raw
            elif artifact == "ptx":
                self.check(self.library.nvrtcGetPTXSize(program, ctypes.byref(size)),
                           "nvrtcGetPTXSize")
                buffer = ctypes.create_string_buffer(size.value)
                self.check(self.library.nvrtcGetPTX(program, buffer), "nvrtcGetPTX")
                output = buffer.raw[:-1]
            else:
                raise ValueError(f"unsupported artifact: {artifact}")
            artifact_end = time.perf_counter_ns()
        finally:
            self.check(self.library.nvrtcDestroyProgram(ctypes.byref(program)),
                       "nvrtcDestroyProgram")
        total_end = time.perf_counter_ns()
        scale = 1.0 / 1_000_000.0
        return output, {
            "create_ms": (create_end - create_begin) * scale,
            "compile_ms": (compile_end - compile_begin) * scale,
            "artifact_ms": (artifact_end - artifact_begin) * scale,
            "total_ms": (total_end - total_begin) * scale,
        }


def find_nvrtc(requested: str | None) -> str:
    candidates = [
        requested,
        "/usr/local/cuda/lib64/libnvrtc.so",
        "/usr/local/cuda/lib64/libnvrtc.so.13",
        "/usr/opt/cuda/lib64/libnvrtc.so",
        "libnvrtc.so",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate)
            del library
            return candidate
        except OSError:
            pass
    raise RuntimeError("could not find libnvrtc; pass --libnvrtc")


def command_version(command: str) -> str:
    result = subprocess.run(
        [command, "--version"], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True
    )
    return result.stdout.strip()


def duration_ms(command: list[str]) -> float:
    begin = time.perf_counter_ns()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return (time.perf_counter_ns() - begin) / 1_000_000.0


def distribution(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for field in records[0]:
        ordered = sorted(record[field] for record in records)
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        result[field] = {
            "minimum": ordered[0],
            "median": statistics.median(ordered),
            "mean": statistics.fmean(ordered),
            "p95": ordered[p95_index],
            "maximum": ordered[-1],
        }
    return result


def scalar_distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "minimum_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[p95_index],
        "maximum_ms": ordered[-1],
    }


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor()
    for line in cpuinfo.read_text(errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def varied_source(source: bytes, index: int) -> bytes:
    """Change a live model constant so compiler caches cannot reuse the CUBIN."""
    needle = b"const float mu_d = 0.0055f;"
    if source.count(needle) != 1:
        raise RuntimeError("--vary-source requires exactly one mu_d declaration")
    value = 0.0055 + (index + 1) * 1.0e-8
    replacement = f"const float mu_d = {value:.10f}f;".encode()
    return source.replace(needle, replacement, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("architecture", help="real architecture such as sm_89")
    parser.add_argument("--libnvrtc")
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--opt-level", type=int, choices=range(4), default=3)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--define", action="append", default=[])
    parser.add_argument(
        "--no-line-info",
        action="store_true",
        help="omit generated line information for production-style CUBIN sizing",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable NVRTC compiler caching",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--vary-source",
        action="store_true",
        help="vary a live ODE constant on every sample to defeat compile caches",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    if not source_path.is_file():
        parser.error(f"source does not exist: {source_path}")
    if not args.architecture.startswith("sm_"):
        parser.error("architecture must use the sm_XX form")
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")

    source = source_path.read_bytes()
    ptxas = shutil.which(args.ptxas) or args.ptxas
    nvrtc_path = find_nvrtc(args.libnvrtc)
    nvrtc = Nvrtc(nvrtc_path)
    virtual_architecture = "compute_" + args.architecture.removeprefix("sm_")
    definitions = [f"-D{definition}" for definition in args.define]
    common = ["--std=c++17", "--use_fast_math"]
    if not args.no_line_info:
        common.append("--generate-line-info")
    if args.no_cache:
        common.append("--no-cache")
    common.extend(definitions)
    cubin_options = [
        *common,
        f"--gpu-architecture={args.architecture}",
        f"--ptxas-options=--opt-level={args.opt_level}",
    ]
    ptx_options = [*common, f"--gpu-architecture={virtual_architecture}"]

    with tempfile.TemporaryDirectory(prefix="secant-ode-compile-") as directory:
        temporary = Path(directory)
        ptx_path = temporary / "staged.ptx"
        cubin_path = temporary / "staged.cubin"

        for warmup in range(args.warmups):
            sample_source = varied_source(source, warmup) if args.vary_source else source
            nvrtc.compile(sample_source, source_path.name, cubin_options, "cubin")
            ptx, _ = nvrtc.compile(sample_source, source_path.name, ptx_options, "ptx")
            ptx_path.write_bytes(ptx)
            duration_ms([ptxas, f"-O{args.opt_level}", f"-arch={args.architecture}",
                         str(ptx_path), "-o", str(cubin_path)])

        cubin_records: list[dict[str, float]] = []
        ptx_records: list[dict[str, float]] = []
        ptxas_times: list[float] = []
        direct_cubin = b""
        staged_ptx = b""
        for repetition in range(args.repetitions):
            sample_index = args.warmups + repetition
            sample_source = varied_source(source, sample_index) if args.vary_source else source
            direct_cubin, cubin_record = nvrtc.compile(
                sample_source, source_path.name, cubin_options, "cubin"
            )
            staged_ptx, ptx_record = nvrtc.compile(
                sample_source, source_path.name, ptx_options, "ptx"
            )
            ptx_path.write_bytes(staged_ptx)
            ptxas_time = duration_ms(
                [ptxas, f"-O{args.opt_level}", f"-arch={args.architecture}", str(ptx_path),
                 "-o", str(cubin_path)]
            )
            cubin_records.append(cubin_record)
            ptx_records.append(ptx_record)
            ptxas_times.append(ptxas_time)

        verbose = subprocess.run(
            [ptxas, f"-O{args.opt_level}", "-v", f"-arch={args.architecture}", str(ptx_path),
             "-o", str(cubin_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
        staged_cubin = cubin_path.read_bytes()

        artifact_paths: dict[str, str] = {}
        if args.output_directory is not None:
            output_directory = args.output_directory.resolve()
            output_directory.mkdir(parents=True, exist_ok=True)
            prefix = f"{source_path.stem}_{args.architecture}"
            outputs = {
                "ptx": output_directory / f"{prefix}.ptx",
                "direct_cubin": output_directory / f"{prefix}.direct.cubin",
                "staged_cubin": output_directory / f"{prefix}.staged.cubin",
            }
            outputs["ptx"].write_bytes(staged_ptx)
            outputs["direct_cubin"].write_bytes(direct_cubin)
            outputs["staged_cubin"].write_bytes(staged_cubin)
            artifact_paths = {name: str(path) for name, path in outputs.items()}

        result = {
            "schema": "secant.ode_compile_stages.v1",
            "host": platform.node(),
            "cpu": cpu_model(),
            "machine": platform.machine(),
            "source": str(source_path),
            "source_bytes": len(source),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "architecture": args.architecture,
            "opt_level": args.opt_level,
            "definitions": args.define,
            "varied_source": args.vary_source,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "nvrtc_library": nvrtc_path,
            "nvrtc_version": nvrtc.version(),
            "ptxas_version": command_version(ptxas),
            "artifacts": {
                "ptx_bytes": len(staged_ptx),
                "direct_cubin_bytes": len(direct_cubin),
                "staged_cubin_bytes": len(staged_cubin),
                "cubins_byte_identical": direct_cubin == staged_cubin,
                "paths": artifact_paths,
            },
            "cuda_to_cubin_ms": distribution(cubin_records),
            "cuda_to_ptx_ms": distribution(ptx_records),
            "ptx_to_cubin": scalar_distribution(ptxas_times),
            "ptxas_verbose": verbose,
        }
        print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
