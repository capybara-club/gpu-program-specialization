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

import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import time

from .compiler import CompilationResult, compile_cuda


KERNEL_NAME = "secant_module_lifecycle_kernel"
NONCE_MAGIC = 0xD43C7A91E5B6028F
MAX_SAFE_BRKPTS_PER_KERNEL = 65536


def generate_module_lifecycle_cuda(
    padding_brkpts: int = 0,
    global_padding_bytes: int = 0,
    kernel_count: int = 1,
) -> str:
    """Generate a trivial early-exit kernel with unreachable SASS-size padding."""

    if isinstance(padding_brkpts, bool) or not isinstance(padding_brkpts, int):
        raise TypeError("BRKPT padding count must be an integer")
    if padding_brkpts < 0:
        raise ValueError("BRKPT padding count cannot be negative")
    if isinstance(global_padding_bytes, bool) or not isinstance(global_padding_bytes, int):
        raise TypeError("global padding byte count must be an integer")
    if global_padding_bytes < 0:
        raise ValueError("global padding byte count cannot be negative")
    if padding_brkpts and global_padding_bytes:
        raise ValueError("select BRKPT code padding or global-data padding, not both")
    if isinstance(kernel_count, bool) or not isinstance(kernel_count, int):
        raise TypeError("kernel count must be an integer")
    if kernel_count <= 0:
        raise ValueError("kernel count must be positive")
    padding_counts = tuple(
        padding_brkpts // kernel_count + (index < padding_brkpts % kernel_count)
        for index in range(kernel_count)
    )
    if max(padding_counts, default=0) > MAX_SAFE_BRKPTS_PER_KERNEL:
        raise ValueError(
            "BRKPT padding exceeds the safe 65,536-instruction per-kernel build limit; "
            "increase --kernels, use --padding-mode global, or benchmark a real CUBIN"
        )
    macro_lines = ['#define SSID_BRKPT_1 asm volatile("brkpt;");']
    power = 1
    while power * 2 <= max(1, *padding_counts):
        macro_lines.append(
            f"#define SSID_BRKPT_{power * 2} SSID_BRKPT_{power} SSID_BRKPT_{power}"
        )
        power *= 2
    def render_padding(count: int) -> str:
        invocations: list[str] = []
        remaining = count
        while remaining:
            selected = 1 << (remaining.bit_length() - 1)
            invocations.append(f"    SSID_BRKPT_{selected}")
            remaining -= selected
        return "\n".join(invocations) if invocations else "    asm volatile(\"\");"

    macros = "\n".join(macro_lines)
    global_padding = (
        f'extern "C" __device__ unsigned char secant_module_lifecycle_payload[{global_padding_bytes}] = {{0x5a}};'
        if global_padding_bytes
        else ""
    )
    kernels: list[str] = []
    for index, padding_count in enumerate(padding_counts):
        name = KERNEL_NAME if kernel_count == 1 else f"{KERNEL_NAME}_{index:03d}"
        kernels.append(f'''extern "C" __global__ void {name}(
    const volatile unsigned int *__restrict__ gate,
    unsigned long long *__restrict__ sink,
    unsigned long long wait_clocks)
{{
    const unsigned int gate_value = gate[0];
    if ((gate_value & 1u) != 0u) {{
        if (wait_clocks != 0ull) {{
            const unsigned long long started = clock64();
            while (clock64() - started < wait_clocks) asm volatile("");
        }}
        return;
    }}
    if ((gate_value & 2u) != 0u) {{
        if (blockIdx.x == 0u && threadIdx.x == 0u) sink[0] = secant_module_lifecycle_nonce;
        return;
    }}
{render_padding(padding_count)}
}}
''')
    return f'''#define SSID_NONCE_MAGIC 0x{NONCE_MAGIC:016x}ULL
{macros}

extern "C" __device__ __constant__ unsigned long long secant_module_lifecycle_nonce = SSID_NONCE_MAGIC;
{global_padding}

{''.join(kernels)}'''


def find_nonce_offset(cubin: bytes, nonce: int = NONCE_MAGIC) -> int:
    marker = struct.pack("<Q", nonce)
    offsets: list[int] = []
    start = 0
    while True:
        offset = cubin.find(marker, start)
        if offset < 0:
            break
        offsets.append(offset)
        start = offset + 1
    if len(offsets) != 1:
        raise ValueError(
            f"expected exactly one lifecycle nonce in CUBIN, found {len(offsets)}"
        )
    return offsets[0]


def _compile(
    padding_amount: int,
    padding_mode: str,
    architecture: str,
    nvcc: str | Path | None,
    kernel_count: int,
) -> tuple[str, CompilationResult]:
    if padding_mode == "brkpt":
        return _compile_brkpt_ptx(padding_amount, architecture, nvcc, kernel_count)
    elif padding_mode == "global":
        source = generate_module_lifecycle_cuda(
            global_padding_bytes=padding_amount,
            kernel_count=kernel_count,
        )
    else:
        raise ValueError("padding mode must be 'brkpt' or 'global'")
    return source, compile_cuda(source, architecture, nvcc, fast_math=False)


def _nvrtc_library_name() -> str:
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
        "the installed NVRTC library was not found; this benchmark will not download or build a compiler"
    )


def _compile_cuda_to_ptx(source: str, architecture: str) -> tuple[bytes, str, float]:
    library = ctypes.CDLL(_nvrtc_library_name())
    program_type = ctypes.c_void_p
    library.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(program_type),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.nvrtcCompileProgram.argtypes = [
        program_type,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.nvrtcGetProgramLogSize.argtypes = [program_type, ctypes.POINTER(ctypes.c_size_t)]
    library.nvrtcGetProgramLog.argtypes = [program_type, ctypes.c_char_p]
    library.nvrtcGetPTXSize.argtypes = [program_type, ctypes.POINTER(ctypes.c_size_t)]
    library.nvrtcGetPTX.argtypes = [program_type, ctypes.c_void_p]
    library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(program_type)]
    library.nvrtcGetErrorString.argtypes = [ctypes.c_int]
    library.nvrtcGetErrorString.restype = ctypes.c_char_p

    virtual_architecture = f"compute_{architecture.removeprefix('sm_')}"
    options = ("--std=c++17", f"--gpu-architecture={virtual_architecture}")
    encoded_options = [option.encode() for option in options]
    option_array = (ctypes.c_char_p * len(encoded_options))(*encoded_options)
    program = program_type()
    started = time.perf_counter()
    result = library.nvrtcCreateProgram(
        ctypes.byref(program), source.encode(), b"module_lifecycle.cu", 0, None, None
    )
    if result != 0:
        message = library.nvrtcGetErrorString(result)
        raise RuntimeError(f"nvrtcCreateProgram failed: {message.decode() if message else result}")
    try:
        result = library.nvrtcCompileProgram(program, len(options), option_array)
        log_size = ctypes.c_size_t()
        log = ""
        if library.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)) == 0 and log_size.value:
            log_buffer = ctypes.create_string_buffer(log_size.value)
            if library.nvrtcGetProgramLog(program, log_buffer) == 0:
                log = log_buffer.value.decode(errors="replace")
        if result != 0:
            message = library.nvrtcGetErrorString(result)
            raise RuntimeError(
                f"NVRTC PTX compilation failed: {message.decode() if message else result}\n{log}"
            )
        ptx_size = ctypes.c_size_t()
        result = library.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
        if result != 0 or ptx_size.value == 0:
            raise RuntimeError(f"nvrtcGetPTXSize failed with result {result}")
        ptx_buffer = ctypes.create_string_buffer(ptx_size.value)
        result = library.nvrtcGetPTX(program, ptx_buffer)
        if result != 0:
            raise RuntimeError(f"nvrtcGetPTX failed with result {result}")
        return ptx_buffer.raw.rstrip(b"\x00"), log, time.perf_counter() - started
    finally:
        library.nvrtcDestroyProgram(ctypes.byref(program))


def _find_ptxas(nvcc: str | Path | None) -> Path:
    candidates: list[Path] = []
    if nvcc is not None:
        candidates.append(Path(nvcc).resolve().with_name("ptxas"))
    discovered = shutil.which("ptxas")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        Path(path)
        for path in (
            "/usr/local/cuda/bin/ptxas",
            "/opt/cuda/bin/ptxas",
            "/usr/opt/cuda/bin/ptxas",
        )
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise FileNotFoundError(
        "the installed ptxas executable was not found; this benchmark will not download or build one"
    )


def _compile_brkpt_ptx(
    padding_brkpts: int,
    architecture: str,
    nvcc: str | Path | None,
    kernel_count: int,
) -> tuple[str, CompilationResult]:
    marker_source = generate_module_lifecycle_cuda(
        padding_brkpts=kernel_count,
        kernel_count=kernel_count,
    )
    ptx, nvrtc_log, nvrtc_seconds = _compile_cuda_to_ptx(marker_source, architecture)
    marker = b"brkpt;"
    if ptx.count(marker) != kernel_count:
        raise RuntimeError(
            f"expected {kernel_count} PTX BRKPT markers, found {ptx.count(marker)}"
        )
    padding_counts = tuple(
        padding_brkpts // kernel_count + (index < padding_brkpts % kernel_count)
        for index in range(kernel_count)
    )
    pieces = ptx.split(marker)
    expanded = bytearray(pieces[0])
    for index, padding_count in enumerate(padding_counts):
        expanded.extend(b"\n\tbrkpt;" * padding_count)
        expanded.extend(pieces[index + 1])
    expanded_ptx = bytes(expanded)
    ptxas = _find_ptxas(nvcc)
    with tempfile.TemporaryDirectory(prefix="secant-module-lifecycle-") as temporary:
        directory = Path(temporary)
        ptx_path = directory / "module_lifecycle.ptx"
        cubin_path = directory / "module_lifecycle.cubin"
        ptx_path.write_bytes(expanded_ptx)
        command = (
            str(ptxas),
            f"--gpu-name={architecture}",
            "--opt-level=3",
            "--verbose",
            str(ptx_path),
            "--output-file",
            str(cubin_path),
        )
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True)
        ptxas_seconds = time.perf_counter() - started
        log = nvrtc_log + completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"PTX BRKPT padding assembly failed:\n{log}")
        cubin = cubin_path.read_bytes()
    source = (
        marker_source
        + f"\n/* The generator expanded the single PTX BRKPT marker to {padding_brkpts} instructions before ptxas. */\n"
    )
    return source, CompilationResult(
        cubin,
        nvrtc_seconds + ptxas_seconds,
        ("nvrtc-to-ptx", *command),
        log,
    )


def build_sized_module(
    target_cubin_bytes: int,
    architecture: str,
    nvcc: str | Path | None,
    padding_mode: str = "brkpt",
    kernel_count: int = 1,
) -> tuple[str, CompilationResult, int, list[dict[str, str | float | int]]]:
    """Compile a module near a target byte size with at most one correction."""

    if target_cubin_bytes <= 0:
        raise ValueError("target CUBIN size must be positive")
    base_source, base = _compile(0, padding_mode, architecture, nvcc, kernel_count)
    attempts: list[dict[str, str | float | int]] = [
        {
            "padding_mode": padding_mode,
            "padding_units": 0,
            "cubin_bytes": len(base.cubin),
            "compile_seconds": base.elapsed_seconds,
        }
    ]
    if target_cubin_bytes <= len(base.cubin):
        return base_source, base, 0, attempts
    estimated_bytes_per_unit = 16 if padding_mode == "brkpt" else 1
    count = max(1, (target_cubin_bytes - len(base.cubin)) // estimated_bytes_per_unit)
    source, result = _compile(count, padding_mode, architecture, nvcc, kernel_count)
    attempts.append(
        {
            "padding_mode": padding_mode,
            "padding_units": count,
            "cubin_bytes": len(result.cubin),
            "compile_seconds": result.elapsed_seconds,
        }
    )
    per_instruction = (len(result.cubin) - len(base.cubin)) / count
    tolerance = max(32768, target_cubin_bytes // 50)
    if per_instruction > 0 and abs(len(result.cubin) - target_cubin_bytes) > tolerance:
        corrected = max(1, round((target_cubin_bytes - len(base.cubin)) / per_instruction))
        if corrected != count:
            count = corrected
            source, result = _compile(count, padding_mode, architecture, nvcc, kernel_count)
            attempts.append(
                {
                    "padding_mode": padding_mode,
                    "padding_units": count,
                    "cubin_bytes": len(result.cubin),
                    "compile_seconds": result.elapsed_seconds,
                }
            )
    return source, result, count, attempts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an adjustable-size CUDA CUBIN for module lifecycle benchmarking."
    )
    sizing = parser.add_mutually_exclusive_group(required=True)
    sizing.add_argument("--target-cubin-bytes", type=int)
    sizing.add_argument("--padding-brkpts", type=int)
    sizing.add_argument("--global-padding-bytes", type=int)
    parser.add_argument("--padding-mode", choices=("brkpt", "global"), default="brkpt")
    parser.add_argument("--kernels", type=int, default=1)
    parser.add_argument("--arch", default="sm_120")
    parser.add_argument("--nvcc")
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args(argv)
    if arguments.kernels <= 0:
        raise ValueError("--kernels must be positive")

    if arguments.target_cubin_bytes is not None:
        source, compilation, padding_units, attempts = build_sized_module(
            arguments.target_cubin_bytes,
            arguments.arch,
            arguments.nvcc,
            arguments.padding_mode,
            arguments.kernels,
        )
    else:
        padding_mode = "global" if arguments.global_padding_bytes is not None else "brkpt"
        padding_units = (
            arguments.global_padding_bytes
            if arguments.global_padding_bytes is not None
            else arguments.padding_brkpts
        )
        source, compilation = _compile(
            padding_units,
            padding_mode,
            arguments.arch,
            arguments.nvcc,
            arguments.kernels,
        )
        attempts = [
            {
                "padding_mode": padding_mode,
                "padding_units": padding_units,
                "cubin_bytes": len(compilation.cubin),
                "compile_seconds": compilation.elapsed_seconds,
            }
        ]
    nonce_offset = find_nonce_offset(compilation.cubin)
    output = Path(arguments.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "module_lifecycle.cu"
    cubin_path = output / "module_lifecycle.cubin"
    log_path = output / "compile.log"
    manifest_path = output / "manifest.json"
    source_path.write_text(source)
    cubin_path.write_bytes(compilation.cubin)
    log_path.write_text(compilation.log)
    manifest = {
        "schema": "secant.module_lifecycle_fixture.v1",
        "architecture": arguments.arch,
        "kernel_name": KERNEL_NAME,
        "kernel_count": arguments.kernels,
        "kernel_names": [
            KERNEL_NAME
            if arguments.kernels == 1
            else f"{KERNEL_NAME}_{index:03d}"
            for index in range(arguments.kernels)
        ],
        "target_cubin_bytes": arguments.target_cubin_bytes,
        "actual_cubin_bytes": len(compilation.cubin),
        "padding_mode": arguments.padding_mode if arguments.target_cubin_bytes is not None else padding_mode,
        "padding_units": padding_units,
        "padding_brkpts": padding_units if (arguments.padding_mode if arguments.target_cubin_bytes is not None else padding_mode) == "brkpt" else 0,
        "global_padding_bytes": padding_units if (arguments.padding_mode if arguments.target_cubin_bytes is not None else padding_mode) == "global" else 0,
        "nonce_magic": NONCE_MAGIC,
        "nonce_magic_hex": f"0x{NONCE_MAGIC:016x}",
        "nonce_offset": nonce_offset,
        "sha256": hashlib.sha256(compilation.cubin).hexdigest(),
        "compile_seconds": compilation.elapsed_seconds,
        "compile_command": compilation.command,
        "calibration_attempts": attempts,
        "kernel_behavior": "volatile global gate read; gate bit 0 optionally waits for runtime clock cycles and returns; gate bit 1 verifies the module nonce; zero reaches BRKPT padding",
        "material_deviations": [
            "CUBIN size is approximate and includes ELF metadata plus the selected unreachable BRKPT code or initialized global-data padding.",
            "Patching the nonce makes module bytes and GPU-visible constant data unique without changing SASS instruction bytes.",
            "Global-data padding measures a different loader regime from code-heavy System ID CUBINs and may include device allocation/copy work absent from a code-only module.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
