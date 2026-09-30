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
"""Compile postorder ODE ASTs directly as CUDA and inspect the resulting CUBIN.

This is deliberately non-production research tooling.  It keeps the current
four-state Odezza benchmark trajectory shape while replacing the SASS patch
arena with CUDA source generated from concrete ASTs.
"""

from __future__ import annotations

import argparse
from array import array
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import tempfile
import time
from typing import Iterable


MAGIC = b"ODEZZAAS"
HEADER_BYTES = 24
STATE_COUNT = 4
CONSTANT_COUNT = 4
TRAJECTORY_COUNT = 3
OBSERVATION_COUNT = 8
POINTS_PER_TRAJECTORY = OBSERVATION_COUNT + 1
TRAJECTORY_POINT_COUNT = TRAJECTORY_COUNT * POINTS_PER_TRAJECTORY
BLOCK_SIZE = 128

OP_RETURN = 0x80
OP_STATE = 0x81
OP_CONSTANT = 0x82
OP_LITERAL = 0x83
OP_TOGGLE2 = 0x84
OP_TOGGLE4 = 0x85
OP_ADD = 0x90
OP_SUB = 0x91
OP_MUL = 0x92
OP_DIV = 0x93
OP_NEG = 0x94
OP_SQRT = 0x95
OP_RCP = 0x96
OP_ABS = 0x97
OP_MIN = 0x98
OP_MAX = 0x99
OP_FMA = 0x9A
OP_SIN = 0x9B
OP_COS = 0x9C
OP_EX2 = 0x9D
OP_LG2 = 0x9E
OP_RSQRT = 0x9F
OP_TANH = 0xA0
OP_EXP = 0xA1
OP_LOG = 0xA2

IMMEDIATE_BYTES = {
    OP_STATE: 1,
    OP_CONSTANT: 1,
    OP_LITERAL: 4,
    OP_TOGGLE2: 1,
    OP_TOGGLE4: 2,
}


class ScratchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Program:
    encoded: bytes
    statements: tuple[str, ...]
    result: str
    toggle_count: int
    operator_count: int


def _read_program(data: bytes, offset: int) -> tuple[bytes, int]:
    begin = offset
    while True:
        if offset >= len(data):
            raise ScratchError("truncated AST corpus")
        opcode = data[offset]
        offset += 1
        if opcode == OP_RETURN:
            return data[begin:offset], offset
        immediate = IMMEDIATE_BYTES.get(opcode, 0)
        if offset + immediate > len(data):
            raise ScratchError("truncated AST immediate")
        offset += immediate


def read_corpus(path: Path, count: int, first: int) -> tuple[list[bytes], int]:
    data = path.read_bytes()
    if len(data) < HEADER_BYTES or data[:8] != MAGIC:
        raise ScratchError("not an Odezza AST corpus")
    version, maximum_bytes, record_count = struct.unpack_from("<IIQ", data, 8)
    if version != 1 or maximum_bytes == 0:
        raise ScratchError("unsupported AST corpus header")
    if first < 0 or first >= record_count:
        raise ScratchError("first AST is outside the corpus")
    selected = min(count, record_count - first)
    programs: list[bytes] = []
    offset = HEADER_BYTES
    for index in range(record_count):
        encoded, offset = _read_program(data, offset)
        if first <= index < first + selected:
            programs.append(encoded)
        if index >= first + selected:
            break
    if len(programs) != selected:
        raise ScratchError("AST corpus ended before the requested selection")
    return programs, record_count


def _pop(stack: list[str], count: int) -> list[str]:
    if len(stack) < count:
        raise ScratchError("malformed postorder AST stack")
    values = stack[-count:]
    del stack[-count:]
    return values


def decode_program(encoded: bytes) -> Program:
    stack: list[str] = []
    statements: list[str] = []
    offset = 0
    temporary = 0
    toggles = 0
    operators = 0

    def emit(expression: str) -> str:
        nonlocal temporary
        name = f"value{temporary}"
        temporary += 1
        statements.append(f"float {name} = {expression};")
        return name

    while offset < len(encoded):
        opcode = encoded[offset]
        offset += 1
        if opcode == OP_RETURN:
            if len(stack) != 1 or offset != len(encoded):
                raise ScratchError("malformed AST return")
            return Program(encoded, tuple(statements), stack[0], toggles, operators)
        if opcode == OP_STATE:
            index = encoded[offset]
            offset += 1
            if index >= STATE_COUNT:
                raise ScratchError("AST state index exceeds scratch shape")
            stack.append(f"stage_state[{index}]")
            continue
        if opcode == OP_CONSTANT:
            index = encoded[offset]
            offset += 1
            if index >= CONSTANT_COUNT:
                raise ScratchError("AST constant index exceeds scratch shape")
            stack.append(f"constant{index}")
            continue
        if opcode == OP_LITERAL:
            bits = struct.unpack_from("<I", encoded, offset)[0]
            offset += 4
            stack.append(f"__uint_as_float(0x{bits:08x}u)")
            continue
        if opcode == OP_TOGGLE2:
            first_value, second_value = _pop(stack, 2)
            bit = encoded[offset]
            offset += 1
            if bit >= 32:
                raise ScratchError("toggle bit exceeds the configuration word")
            stack.append(emit(f"((permutation >> {bit}u) & 1u) != 0u ? {second_value} : {first_value}"))
            toggles += 1
            continue
        if opcode == OP_TOGGLE4:
            values = _pop(stack, 4)
            low_bit = encoded[offset]
            high_bit = encoded[offset + 1]
            offset += 2
            if low_bit >= 32 or high_bit >= 32:
                raise ScratchError("toggle bit exceeds the configuration word")
            selection = f"(((permutation >> {low_bit}u) & 1u) | (((permutation >> {high_bit}u) & 1u) << 1u))"
            stack.append(
                emit(
                    f"{selection} == 0u ? {values[0]} : ({selection} == 1u ? {values[1]} : "
                    f"({selection} == 2u ? {values[2]} : {values[3]}))"
                )
            )
            toggles += 1
            continue

        operators += 1
        if opcode in (OP_NEG, OP_SQRT, OP_RCP, OP_ABS, OP_SIN, OP_COS, OP_EX2, OP_LG2, OP_RSQRT, OP_TANH, OP_EXP, OP_LOG):
            value = _pop(stack, 1)[0]
            expression = {
                OP_NEG: f"-({value})",
                OP_SQRT: f"sqrtf({value})",
                OP_RCP: f"1.0f / ({value})",
                OP_ABS: f"fabsf({value})",
                OP_SIN: f"sinf({value})",
                OP_COS: f"cosf({value})",
                OP_EX2: f"exp2f({value})",
                OP_LG2: f"log2f({value})",
                OP_RSQRT: f"rsqrtf({value})",
                OP_TANH: f"tanhf({value})",
                OP_EXP: f"expf({value})",
                OP_LOG: f"logf({value})",
            }[opcode]
            stack.append(emit(expression))
            continue
        if opcode in (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MIN, OP_MAX):
            lhs, rhs = _pop(stack, 2)
            expression = {
                OP_ADD: f"({lhs}) + ({rhs})",
                OP_SUB: f"({lhs}) - ({rhs})",
                OP_MUL: f"({lhs}) * ({rhs})",
                OP_DIV: f"({lhs}) / ({rhs})",
                OP_MIN: f"fminf({lhs}, {rhs})",
                OP_MAX: f"fmaxf({lhs}, {rhs})",
            }[opcode]
            stack.append(emit(expression))
            continue
        if opcode == OP_FMA:
            lhs, rhs, addend = _pop(stack, 3)
            stack.append(emit(f"fmaf({lhs}, {rhs}, {addend})"))
            continue
        raise ScratchError(f"unsupported AST opcode 0x{opcode:02x}")
    raise ScratchError("AST has no return")


def _candidate_switch(programs: list[Program]) -> str:
    lines = ["                    float rhs3 = 0.0f;", "                    switch (system_index) {"]
    for index, program in enumerate(programs):
        lines.append(f"                    case {index}u: {{")
        lines.append(f'#line 1 "odezza_candidate_{index}.ast"')
        for statement in program.statements:
            lines.append(f"                        {statement}")
        lines.append(f"                        rhs3 = {program.result};")
        lines.append('#line 1 "cuda_ast_scoring.cu"')
        lines.append("                        break;")
        lines.append("                    }")
    lines.extend(["                    default: return;", "                    }"])
    return "\n".join(lines)


def generate_cuda(programs: list[Program], compile_tag: str = "") -> str:
    candidate = _candidate_switch(programs)
    return f'''/* Scratch CUDA AST specialization experiment. tag={compile_tag} */
#define SYSTEM_CAPACITY {len(programs)}u
#define STATE_COUNT 4u
#define CONSTANT_COUNT 4u

extern "C" __global__ void cuda_ast_scoring(
    const float *__restrict__ constant_banks,
    unsigned int constant_bank_count,
    const unsigned int *__restrict__ trajectory_offsets,
    const float *__restrict__ trajectory_times,
    const float *__restrict__ reference_data,
    unsigned int trajectory_count,
    unsigned int trajectory_point_count,
    unsigned int active_toggle_count,
    unsigned int steps_per_observation,
    float *__restrict__ mse_out
) {{
    if (constant_bank_count == 0u || trajectory_count == 0u || active_toggle_count > 32u || steps_per_observation == 0u) return;
    extern __shared__ unsigned int shared_words[];
    float *reference = reinterpret_cast<float *>(shared_words);
    float *times = reference + STATE_COUNT * trajectory_point_count;
    unsigned int *offsets = reinterpret_cast<unsigned int *>(times + trajectory_point_count);
    for (unsigned int index = threadIdx.x; index < STATE_COUNT * trajectory_point_count; index += blockDim.x) reference[index] = reference_data[index];
    for (unsigned int index = threadIdx.x; index < trajectory_point_count; index += blockDim.x) times[index] = trajectory_times[index];
    for (unsigned int index = threadIdx.x; index <= trajectory_count; index += blockDim.x) offsets[index] = trajectory_offsets[index];
    __syncthreads();

    const unsigned int system_index = blockIdx.y;
    const unsigned long long configuration = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long configuration_count = (unsigned long long)constant_bank_count << active_toggle_count;
    if (system_index >= SYSTEM_CAPACITY || configuration >= configuration_count) return;
    const unsigned int permutation = (unsigned int)configuration;
    const unsigned int constant_bank = (unsigned int)(configuration >> active_toggle_count);
    const unsigned long long constant_base = ((unsigned long long)system_index * constant_bank_count + constant_bank) * CONSTANT_COUNT;
    const float constant0 = constant_banks[constant_base + 0u];
    const float constant1 = constant_banks[constant_base + 1u];
    const float constant2 = constant_banks[constant_base + 2u];
    const float constant3 = constant_banks[constant_base + 3u];

    float squared_error = 0.0f;
    unsigned int scored_point_count = 0u;
    unsigned int expected_start = 0u;
    int valid = 1;
#pragma unroll 1
    for (unsigned int trajectory = 0u; trajectory < trajectory_count; ++trajectory) {{
        const unsigned int point_begin = offsets[trajectory];
        const unsigned int point_end = offsets[trajectory + 1u];
        if (point_begin != expected_start || point_end <= point_begin || point_end > trajectory_point_count) return;
        expected_start = point_end;
        float state[STATE_COUNT];
#pragma unroll
        for (unsigned int component = 0u; component < STATE_COUNT; ++component) state[component] = reference[component * trajectory_point_count + point_begin];
#pragma unroll 1
        for (unsigned int point = point_begin + 1u; point < point_end; ++point) {{
            const float interval = times[point] - times[point - 1u];
            valid = valid && isfinite(interval) && interval > 0.0f;
            const float h = interval / (float)steps_per_observation;
            const float half_h = 0.5f * h;
            const float sixth_h = h / 6.0f;
#pragma unroll 1
            for (unsigned int step = 0u; step < steps_per_observation; ++step) {{
                float base_state[STATE_COUNT];
                float stage_state[STATE_COUNT];
                float sum_state[STATE_COUNT];
#pragma unroll
                for (unsigned int component = 0u; component < STATE_COUNT; ++component) {{
                    base_state[component] = state[component];
                    stage_state[component] = state[component];
                    sum_state[component] = 0.0f;
                }}
#pragma unroll 1
                for (unsigned int rk_stage = 0u; rk_stage < 4u; ++rk_stage) {{
                    const float rhs0 = constant0 * stage_state[0];
                    const float rhs1 = -(constant1 * stage_state[1]);
                    const float rhs2 = constant2 * stage_state[0] - constant3 * stage_state[2];
{candidate}
                    const float rhs[STATE_COUNT] = {{rhs0, rhs1, rhs2, rhs3}};
                    const float weight = (rk_stage == 0 || rk_stage == 3) ? 1.0f : 2.0f;
#pragma unroll
                    for (unsigned int component = 0u; component < STATE_COUNT; ++component) sum_state[component] = fmaf(weight, rhs[component], sum_state[component]);
                    if (rk_stage != 3) {{
                        const float stage_h = rk_stage == 2 ? h : half_h;
#pragma unroll
                        for (unsigned int component = 0u; component < STATE_COUNT; ++component) stage_state[component] = fmaf(stage_h, rhs[component], base_state[component]);
                    }}
                }}
#pragma unroll
                for (unsigned int component = 0u; component < STATE_COUNT; ++component) state[component] = fmaf(sixth_h, sum_state[component], base_state[component]);
            }}
            ++scored_point_count;
#pragma unroll
            for (unsigned int component = 0u; component < STATE_COUNT; ++component) {{
                valid = valid && isfinite(state[component]);
                const float error = state[component] - reference[component * trajectory_point_count + point];
                squared_error = fmaf(error, error, squared_error);
            }}
        }}
    }}
    const unsigned long long output_index = (unsigned long long)system_index * configuration_count + configuration;
    mse_out[output_index] = valid && expected_start == trajectory_point_count && scored_point_count != 0u && isfinite(squared_error)
        ? squared_error / (float)(STATE_COUNT * scored_point_count)
        : 3.402823466e+38F;
}}
'''


class Nvrtc:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("libnvrtc.so")
        self._configure()

    def _configure(self) -> None:
        lib = self.library
        lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    def compile(self, source: str, sm: int, fast_math: bool) -> tuple[bytes, str, float]:
        program = ctypes.c_void_p()
        encoded = source.encode()
        result = self.library.nvrtcCreateProgram(ctypes.byref(program), encoded, b"cuda_ast_scoring.cu", 0, None, None)
        if result != 0:
            raise ScratchError(f"nvrtcCreateProgram failed with {result}")
        options = [f"--gpu-architecture=sm_{sm}", "--std=c++17", "--fmad=true", "--generate-line-info"]
        if fast_math:
            options.append("--use_fast_math")
        encoded_options = [value.encode() for value in options]
        option_array = (ctypes.c_char_p * len(encoded_options))(*encoded_options)
        started = time.perf_counter()
        result = self.library.nvrtcCompileProgram(program, len(encoded_options), option_array)
        elapsed = time.perf_counter() - started
        log_size = ctypes.c_size_t()
        self.library.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log_buffer = ctypes.create_string_buffer(max(1, log_size.value))
        self.library.nvrtcGetProgramLog(program, log_buffer)
        log = log_buffer.value.decode(errors="replace")
        if result != 0:
            self.library.nvrtcDestroyProgram(ctypes.byref(program))
            raise ScratchError(f"NVRTC compilation failed with {result}:\n{log}")
        cubin_size = ctypes.c_size_t()
        if self.library.nvrtcGetCUBINSize(program, ctypes.byref(cubin_size)) != 0:
            self.library.nvrtcDestroyProgram(ctypes.byref(program))
            raise ScratchError("NVRTC did not return a CUBIN")
        cubin_buffer = ctypes.create_string_buffer(cubin_size.value)
        result = self.library.nvrtcGetCUBIN(program, cubin_buffer)
        self.library.nvrtcDestroyProgram(ctypes.byref(program))
        if result != 0:
            raise ScratchError(f"nvrtcGetCUBIN failed with {result}")
        return bytes(cubin_buffer.raw), log, elapsed


def _tool_path(name: str) -> str:
    candidates = [Path("/usr/local/cuda/bin") / name, Path("/opt/cuda/bin") / name]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return name


def analyze_cubin(cubin_path: Path) -> dict[str, object]:
    nvdisasm = _tool_path("nvdisasm")
    cuobjdump = _tool_path("cuobjdump")
    command = [
        nvdisasm,
        "--print-code",
        "--print-line-info-inline",
        "--print-instruction-encoding",
        str(cubin_path),
    ]
    output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    resource_output = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(cubin_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    register_matches = [int(value) for value in re.findall(r"\bREG:(\d+)", resource_output)]
    opcode_pattern = re.compile(r"/\*[0-9a-fA-F]+\*/\s+([A-Za-z0-9_.]+)")
    encoding_pattern = re.compile(r"0x([0-9a-fA-F]{16})")
    source_pattern = re.compile(r'//## File "([^"]+)", line')
    current_source = ""
    pending_opcode: str | None = None
    pending_source = ""
    pending_words: list[str] = []
    instructions: list[tuple[str, int, int, str]] = []
    for line in output.splitlines():
        source_match = source_pattern.search(line)
        if source_match:
            current_source = source_match.group(1)
        opcode_match = opcode_pattern.search(line)
        if opcode_match:
            pending_opcode = opcode_match.group(1)
            pending_source = current_source
            pending_words = []
        words = encoding_pattern.findall(line)
        if pending_opcode is not None and words:
            pending_words.extend(words)
        if pending_opcode is not None and len(pending_words) >= 2:
            instructions.append((pending_opcode, int(pending_words[0], 16), int(pending_words[1], 16), pending_source))
            pending_opcode = None
            pending_source = ""
            pending_words = []
    if not instructions:
        raise ScratchError("nvdisasm output contained no decodable instruction encodings")
    def schedule(values: Iterable[tuple[str, int, int, str]]) -> dict[str, object]:
        selected = list(values)
        stall_histogram: dict[str, int] = {}
        yield_histogram: dict[str, int] = {}
        opcode_counts: dict[str, int] = {}
        opcode_stalls: dict[str, list[int]] = {}
        for opcode, _word0, word1, _source in selected:
            stall = (word1 >> 40) & 0xF
            yield_bit = (word1 >> 44) & 0x1
            stall_histogram[str(stall)] = stall_histogram.get(str(stall), 0) + 1
            yield_histogram[str(yield_bit)] = yield_histogram.get(str(yield_bit), 0) + 1
            opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1
            opcode_stalls.setdefault(opcode, []).append(stall)
        top_opcodes = sorted(opcode_counts.items(), key=lambda item: (-item[1], item[0]))[:24]
        return {
            "instruction_count": len(selected),
            "encoded_stall_histogram": stall_histogram,
            "encoded_stall_mean": (
                sum(int(stall) * count for stall, count in stall_histogram.items()) / len(selected)
                if selected
                else None
            ),
            "encoded_yield_bit_histogram": yield_histogram,
            "top_opcodes": [
                {
                    "opcode": opcode,
                    "count": count,
                    "mean_encoded_stall": sum(opcode_stalls[opcode]) / count,
                }
                for opcode, count in top_opcodes
            ],
        }

    complete_schedule = schedule(instructions)
    candidate_schedule = schedule(
        instruction for instruction in instructions if Path(instruction[3]).name.startswith("odezza_candidate_")
    )
    return {
        "register_count": max(register_matches) if register_matches else None,
        "sass_instruction_count": len(instructions),
        "encoded_stall_histogram": complete_schedule["encoded_stall_histogram"],
        "encoded_stall_mean": complete_schedule["encoded_stall_mean"],
        "encoded_yield_bit_histogram": complete_schedule["encoded_yield_bit_histogram"],
        "yield_bit_note": "raw instruction word bit 44; semantic polarity is intentionally not inferred",
        "top_opcodes": complete_schedule["top_opcodes"],
        "candidate_schedule": candidate_schedule,
    }


class CudaDriver:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcuda.so.1")
        self._configure()

    def _function(self, base: str, versioned: bool = False):
        names = [f"{base}_v2", base] if versioned else [base, f"{base}_v2"]
        for name in names:
            try:
                return getattr(self.library, name)
            except AttributeError:
                pass
        raise ScratchError(f"CUDA driver symbol {base} is unavailable")

    def _configure(self) -> None:
        self.cuInit = self._function("cuInit")
        self.cuDeviceGet = self._function("cuDeviceGet")
        self.cuDeviceGetAttribute = self._function("cuDeviceGetAttribute")
        self.cuDevicePrimaryCtxRetain = self._function("cuDevicePrimaryCtxRetain")
        self.cuDevicePrimaryCtxRelease = self._function("cuDevicePrimaryCtxRelease", True)
        self.cuCtxSetCurrent = self._function("cuCtxSetCurrent")
        self.cuCtxSynchronize = self._function("cuCtxSynchronize")
        self.cuModuleLoadData = self._function("cuModuleLoadData")
        self.cuModuleUnload = self._function("cuModuleUnload")
        self.cuModuleGetFunction = self._function("cuModuleGetFunction")
        self.cuMemAlloc = self._function("cuMemAlloc", True)
        self.cuMemFree = self._function("cuMemFree", True)
        self.cuMemcpyHtoD = self._function("cuMemcpyHtoD", True)
        self.cuMemcpyDtoH = self._function("cuMemcpyDtoH", True)
        self.cuLaunchKernel = self._function("cuLaunchKernel")
        self.cuOccupancyMaxActiveBlocksPerMultiprocessor = self._function("cuOccupancyMaxActiveBlocksPerMultiprocessor")
        self.cuEventCreate = self._function("cuEventCreate")
        self.cuEventRecord = self._function("cuEventRecord")
        self.cuEventSynchronize = self._function("cuEventSynchronize")
        self.cuEventElapsedTime = self._function("cuEventElapsedTime")
        self.cuEventDestroy = self._function("cuEventDestroy", True)

    @staticmethod
    def check(result: int, operation: str) -> None:
        if result != 0:
            raise ScratchError(f"{operation} failed with CUDA result {result}")


def _reference_data(steps_per_observation: int) -> tuple[array, array, array]:
    initial = (
        (0.75, 1.25, 0.5, 0.9),
        (1.0, 1.0, 0.8, 1.2),
        (1.25, 0.75, 1.1, 1.5),
    )
    constants = (0.05, 0.04, 0.03, 0.02)
    offsets = array("I", [index * POINTS_PER_TRAJECTORY for index in range(TRAJECTORY_COUNT + 1)])
    times = array("f", [0.0] * TRAJECTORY_POINT_COUNT)
    planes = [[0.0] * TRAJECTORY_POINT_COUNT for _ in range(STATE_COUNT)]

    def rhs(state: list[float]) -> tuple[float, float, float, float]:
        return (
            constants[0] * state[0],
            -(constants[1] * state[1]),
            constants[2] * state[0] - constants[3] * state[2],
            state[0] - state[3],
        )

    for trajectory in range(TRAJECTORY_COUNT):
        point_begin = offsets[trajectory]
        state = list(initial[trajectory])
        for component in range(STATE_COUNT):
            planes[component][point_begin] = state[component]
        for observation in range(OBSERVATION_COUNT):
            point = point_begin + observation + 1
            times[point] = 0.05 * (observation + 1)
            h = 0.05 / steps_per_observation
            for _ in range(steps_per_observation):
                base = state[:]
                stage = state[:]
                sums = [0.0] * STATE_COUNT
                for rk_stage in range(4):
                    values = rhs(stage)
                    weight = 1.0 if rk_stage in (0, 3) else 2.0
                    for component in range(STATE_COUNT):
                        sums[component] += weight * values[component]
                    if rk_stage != 3:
                        stage_h = h if rk_stage == 2 else 0.5 * h
                        stage = [base[index] + stage_h * values[index] for index in range(STATE_COUNT)]
                state = [base[index] + h * sums[index] / 6.0 for index in range(STATE_COUNT)]
            for component in range(STATE_COUNT):
                planes[component][point] = state[component]
    reference = array("f")
    for plane in planes:
        reference.extend(plane)
    return offsets, times, reference


def benchmark_cubin(
    cubin: bytes,
    system_count: int,
    constant_bank_count: int,
    toggle_bits: int,
    steps_per_observation: int,
    run_count: int,
    device_ordinal: int,
) -> dict[str, object]:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    driver = CudaDriver()
    driver.check(driver.cuInit(0), "cuInit")
    device = ctypes.c_int()
    driver.check(driver.cuDeviceGet(ctypes.byref(device), device_ordinal), "cuDeviceGet")
    context = ctypes.c_void_p()
    driver.check(driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
    driver.check(driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    allocations: list[int] = []
    try:
        cubin_buffer = ctypes.create_string_buffer(cubin)
        driver.check(driver.cuModuleLoadData(ctypes.byref(module), cubin_buffer), "cuModuleLoadData")
        driver.check(driver.cuModuleGetFunction(ctypes.byref(function), module, b"cuda_ast_scoring"), "cuModuleGetFunction")
        offsets, times, reference = _reference_data(steps_per_observation)
        constants = array("f", [0.05, 0.04, 0.03, 0.02]) * (system_count * constant_bank_count)
        configuration_count = constant_bank_count << toggle_bits
        score_count = system_count * configuration_count

        def upload(values: array) -> int:
            pointer = ctypes.c_uint64()
            size = len(values) * values.itemsize
            driver.check(driver.cuMemAlloc(ctypes.byref(pointer), size), "cuMemAlloc")
            allocations.append(pointer.value)
            source = (ctypes.c_ubyte * size).from_buffer(values)
            driver.check(driver.cuMemcpyHtoD(pointer, source, size), "cuMemcpyHtoD")
            return pointer.value

        constants_device = upload(constants)
        offsets_device = upload(offsets)
        times_device = upload(times)
        reference_device = upload(reference)
        scores_device_value = ctypes.c_uint64()
        driver.check(driver.cuMemAlloc(ctypes.byref(scores_device_value), score_count * 4), "cuMemAlloc scores")
        allocations.append(scores_device_value.value)

        arguments = [
            ctypes.c_uint64(constants_device),
            ctypes.c_uint(constant_bank_count),
            ctypes.c_uint64(offsets_device),
            ctypes.c_uint64(times_device),
            ctypes.c_uint64(reference_device),
            ctypes.c_uint(TRAJECTORY_COUNT),
            ctypes.c_uint(TRAJECTORY_POINT_COUNT),
            ctypes.c_uint(toggle_bits),
            ctypes.c_uint(steps_per_observation),
            ctypes.c_uint64(scores_device_value.value),
        ]
        parameter_array = (ctypes.c_void_p * len(arguments))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in arguments]
        )
        grid_x = (configuration_count + BLOCK_SIZE - 1) // BLOCK_SIZE
        shared_bytes = (STATE_COUNT * TRAJECTORY_POINT_COUNT + TRAJECTORY_POINT_COUNT + TRAJECTORY_COUNT + 1) * 4
        active_blocks = ctypes.c_int()
        maximum_threads = ctypes.c_int()
        driver.check(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active_blocks),
                function,
                BLOCK_SIZE,
                shared_bytes,
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        driver.check(driver.cuDeviceGetAttribute(ctypes.byref(maximum_threads), 39, device), "maximum threads per SM")

        def launch() -> None:
            driver.check(
                driver.cuLaunchKernel(
                    function,
                    grid_x,
                    system_count,
                    1,
                    BLOCK_SIZE,
                    1,
                    1,
                    shared_bytes,
                    None,
                    parameter_array,
                    None,
                ),
                "cuLaunchKernel",
            )

        launch()
        driver.check(driver.cuCtxSynchronize(), "warmup synchronize")
        driver.check(driver.cuEventCreate(ctypes.byref(start_event), 0), "cuEventCreate start")
        driver.check(driver.cuEventCreate(ctypes.byref(stop_event), 0), "cuEventCreate stop")
        milliseconds: list[float] = []
        for _ in range(run_count):
            driver.check(driver.cuEventRecord(start_event, None), "cuEventRecord start")
            launch()
            driver.check(driver.cuEventRecord(stop_event, None), "cuEventRecord stop")
            driver.check(driver.cuEventSynchronize(stop_event), "cuEventSynchronize")
            elapsed = ctypes.c_float()
            driver.check(driver.cuEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event), "cuEventElapsedTime")
            milliseconds.append(float(elapsed.value))
        first_score = ctypes.c_float()
        driver.check(driver.cuMemcpyDtoH(ctypes.byref(first_score), scores_device_value, 4), "cuMemcpyDtoH score")
        median_ms = statistics.median(milliseconds)
        return {
            "run_count": run_count,
            "minimum_kernel_ms": min(milliseconds),
            "median_kernel_ms": median_ms,
            "mean_kernel_ms": statistics.mean(milliseconds),
            "configurations_per_second": score_count / (median_ms * 1.0e-3),
            "individual_trajectories_per_second": TRAJECTORY_COUNT * score_count / (median_ms * 1.0e-3),
            "first_score": first_score.value,
            "score_count": score_count,
            "dynamic_shared_bytes": shared_bytes,
            "active_blocks_per_sm": active_blocks.value,
            "active_threads_per_sm": active_blocks.value * BLOCK_SIZE,
            "theoretical_thread_occupancy": active_blocks.value * BLOCK_SIZE / maximum_threads.value,
        }
    finally:
        if start_event.value:
            driver.cuEventDestroy(start_event)
        if stop_event.value:
            driver.cuEventDestroy(stop_event)
        for allocation in reversed(allocations):
            driver.cuMemFree(ctypes.c_uint64(allocation))
        if module.value:
            driver.cuModuleUnload(module)
        driver.cuCtxSetCurrent(None)
        driver.cuDevicePrimaryCtxRelease(device)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--system-count", type=int, default=32)
    parser.add_argument("--first-system", type=int, default=0)
    parser.add_argument("--constant-banks", type=int, default=64)
    parser.add_argument("--toggle-bits", type=int, default=5)
    parser.add_argument("--steps-per-observation", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sm", type=int, default=120)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fast-math", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--compile-tag", default="")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--artifact-directory", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.system_count <= 0 or arguments.constant_banks <= 0 or not 0 <= arguments.toggle_bits <= 30:
        raise ScratchError("counts must be positive and toggle bits must be 0 through 30")
    encoded, corpus_count = read_corpus(arguments.corpus, arguments.system_count, arguments.first_system)
    programs = [decode_program(value) for value in encoded]
    source = generate_cuda(programs, arguments.compile_tag)
    if arguments.disable_cache:
        os.environ["CUDA_CACHE_DISABLE"] = "1"
    compiled, log, compile_seconds = Nvrtc().compile(source, arguments.sm, arguments.fast_math)
    if arguments.artifact_directory is not None:
        arguments.artifact_directory.mkdir(parents=True, exist_ok=True)
        source_path = arguments.artifact_directory / f"cuda_ast_{len(programs)}.cu"
        cubin_path = arguments.artifact_directory / f"cuda_ast_{len(programs)}.cubin"
        source_path.write_text(source)
        cubin_path.write_bytes(compiled)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="odezza-cuda-ast-")
        cubin_path = Path(temporary.name) / "cuda_ast.cubin"
        cubin_path.write_bytes(compiled)
        source_path = None
    analysis = analyze_cubin(cubin_path)
    benchmark = None
    if not arguments.no_run:
        benchmark = benchmark_cubin(
            compiled,
            len(programs),
            arguments.constant_banks,
            arguments.toggle_bits,
            arguments.steps_per_observation,
            arguments.runs,
            arguments.device,
        )
    report = {
        "schema": "odezza.scratch.cuda-ast-specialization-v1",
        "comparison_scope": "direct CUDA AST branches in the current four-state, three-trajectory RK4 benchmark shape",
        "known_rhs_realization": "three benchmark RHS expressions compiled directly as CUDA",
        "candidate_rhs_realization": "one corpus AST per uniform CUDA switch branch",
        "corpus": str(arguments.corpus),
        "corpus_ast_count": corpus_count,
        "first_system": arguments.first_system,
        "system_count": len(programs),
        "mean_ast_operators": statistics.mean(program.operator_count for program in programs),
        "mean_ast_toggles": statistics.mean(program.toggle_count for program in programs),
        "source_bytes": len(source.encode()),
        "cubin_bytes": len(compiled),
        "nvrtc_compile_seconds": compile_seconds,
        "nvrtc_log": log,
        "sm": arguments.sm,
        "fast_math": arguments.fast_math,
        "compiler_cache_disabled": arguments.disable_cache,
        "compile_tag": arguments.compile_tag,
        "constant_banks": arguments.constant_banks,
        "toggle_bits": arguments.toggle_bits,
        "configurations_per_ast": arguments.constant_banks << arguments.toggle_bits,
        "analysis": analysis,
        "benchmark": benchmark,
        "artifacts": {
            "cuda": str(source_path) if source_path is not None else None,
            "cubin": str(cubin_path) if arguments.artifact_directory is not None else None,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScratchError as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(1)
