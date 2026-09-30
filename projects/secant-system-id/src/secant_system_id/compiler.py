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
from pathlib import Path
import ctypes
import ctypes.util
import platform
import shutil
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class CompilationResult:
    cubin: bytes
    elapsed_seconds: float
    command: tuple[str, ...]
    log: str


def find_nvcc(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    discovered = shutil.which("nvcc")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        Path(path)
        for path in (
            "/usr/local/cuda/bin/nvcc",
            "/opt/cuda/bin/nvcc",
            "/usr/opt/cuda/bin/nvcc",
        )
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise FileNotFoundError(
        "nvcc was not found; install the supported CUDA toolkit or pass --nvcc. "
        "This project will not download or build a compiler as a workaround."
    )


def compile_cuda(
    source: str,
    architecture: str,
    nvcc: str | Path | None = None,
    fast_math: bool = True,
) -> CompilationResult:
    if not architecture.startswith("sm_") or not architecture[3:].isdigit():
        raise ValueError("architecture must look like sm_89, sm_90, or sm_120")
    if nvcc is None:
        return _compile_nvrtc(source, architecture, fast_math)
    compiler = find_nvcc(nvcc)
    with tempfile.TemporaryDirectory(prefix="secant-system-id-") as temporary:
        directory = Path(temporary)
        source_path = directory / "kernel.cu"
        cubin_path = directory / "kernel.cubin"
        source_path.write_text(source)
        command = [
            str(compiler),
            "--cubin",
            f"--gpu-architecture={architecture}",
            "-O3",
            "--std=c++17",
        ]
        if fast_math:
            command.append("--use_fast_math")
        command.extend((str(source_path), "-o", str(cubin_path)))
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True)
        elapsed = time.perf_counter() - started
        log = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"CUDA compilation failed:\n{log}")
        return CompilationResult(cubin_path.read_bytes(), elapsed, tuple(command), log)


def _find_nvrtc() -> str:
    discovered = ctypes.util.find_library("nvrtc")
    if discovered:
        return discovered
    machine = platform.machine()
    target = "aarch64-linux" if machine in {"aarch64", "arm64"} else "x86_64-linux"
    candidates = (
        Path(f"/usr/local/cuda/targets/{target}/lib/libnvrtc.so"),
        Path("/usr/local/cuda/lib64/libnvrtc.so"),
        Path("/opt/cuda/lib64/libnvrtc.so"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "the installed CUDA NVRTC library was not found; install the supported CUDA toolkit "
        "or pass --nvcc for an existing compatible nvcc"
    )


def _compile_nvrtc(source: str, architecture: str, fast_math: bool) -> CompilationResult:
    library_name = _find_nvrtc()
    library = ctypes.CDLL(library_name)
    program_type = ctypes.c_void_p
    library.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(program_type), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
    ]
    library.nvrtcCompileProgram.argtypes = [program_type, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    library.nvrtcGetProgramLogSize.argtypes = [program_type, ctypes.POINTER(ctypes.c_size_t)]
    library.nvrtcGetProgramLog.argtypes = [program_type, ctypes.c_char_p]
    library.nvrtcGetCUBINSize.argtypes = [program_type, ctypes.POINTER(ctypes.c_size_t)]
    library.nvrtcGetCUBIN.argtypes = [program_type, ctypes.c_void_p]
    library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(program_type)]
    library.nvrtcGetErrorString.argtypes = [ctypes.c_int]
    library.nvrtcGetErrorString.restype = ctypes.c_char_p

    def error_string(result: int) -> str:
        value = library.nvrtcGetErrorString(result)
        return value.decode() if value else f"NVRTC result {result}"

    options = [
        "--std=c++17",
        f"--gpu-architecture={architecture}",
        "--ptxas-options=--verbose",
        "--ptxas-options=--opt-level=3",
    ]
    if fast_math:
        options.append("--use_fast_math")
    encoded_options = [option.encode() for option in options]
    option_array = (ctypes.c_char_p * len(encoded_options))(*encoded_options)
    program = program_type()
    started = time.perf_counter()
    result = library.nvrtcCreateProgram(
        ctypes.byref(program), source.encode(), b"fedbatch_template.cu", 0, None, None
    )
    if result != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {error_string(result)}")
    try:
        result = library.nvrtcCompileProgram(program, len(options), option_array)
        log_size = ctypes.c_size_t()
        log = ""
        if library.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)) == 0 and log_size.value:
            log_buffer = ctypes.create_string_buffer(log_size.value)
            if library.nvrtcGetProgramLog(program, log_buffer) == 0:
                log = log_buffer.value.decode(errors="replace")
        if result != 0:
            raise RuntimeError(f"CUDA compilation failed: {error_string(result)}\n{log}")
        cubin_size = ctypes.c_size_t()
        result = library.nvrtcGetCUBINSize(program, ctypes.byref(cubin_size))
        if result != 0 or cubin_size.value == 0:
            raise RuntimeError(f"nvrtcGetCUBINSize failed: {error_string(result)}")
        cubin_buffer = ctypes.create_string_buffer(cubin_size.value)
        result = library.nvrtcGetCUBIN(program, cubin_buffer)
        if result != 0:
            raise RuntimeError(f"nvrtcGetCUBIN failed: {error_string(result)}")
        elapsed = time.perf_counter() - started
        return CompilationResult(cubin_buffer.raw, elapsed, tuple(["nvrtc", *options]), log)
    finally:
        library.nvrtcDestroyProgram(ctypes.byref(program))
