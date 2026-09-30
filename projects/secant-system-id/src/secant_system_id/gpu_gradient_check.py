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
from array import array
import ctypes
from dataclasses import dataclass
import json
import math
import random
from typing import Sequence

from .ast import Expression, InstructionType, Program, constant, input_slot, operation
from .compiler import compile_cuda
from .genome import SystemGenome
from .lm_specialization import (
    DEFAULT_PARTIAL_PATCH_CAPACITY,
    DEFAULT_PRIMAL_PATCH_CAPACITY,
    render_partial_site,
    render_primal_site,
    specialize_split_lm_cubin,
)
from .runtime import Driver, _host_address
from .shape import KernelShape


KERNEL_NAME = "ssid_random_gradient_check"
SMOOTH_OPERATIONS = (
    InstructionType.ADD_F32,
    InstructionType.SUB_F32,
    InstructionType.MUL_F32,
    InstructionType.DIV_F32,
    InstructionType.NEG_F32,
    InstructionType.SQRT_F32,
    InstructionType.RCP_F32,
    InstructionType.FMA_F32,
    InstructionType.SIN_F32,
    InstructionType.COS_F32,
    InstructionType.EX2_F32,
    InstructionType.LG2_F32,
    InstructionType.RSQRT_F32,
    InstructionType.TANH_F32,
    InstructionType.EXP_F32,
    InstructionType.LOG_F32,
)


@dataclass(frozen=True)
class GradientCheckResult:
    architecture: str
    seed: int
    genome_count: int
    points_per_genome: int
    scalar_comparisons: int
    maximum_primal_absolute_error: float
    maximum_primal_relative_error: float
    maximum_partial_absolute_error: float
    maximum_partial_relative_error: float
    maximum_finite_difference_absolute_error: float
    maximum_finite_difference_relative_error: float
    maximum_register_count: int
    maximum_primal_instructions: int
    maximum_partial_instructions: int
    compile_seconds: float


def render_gradient_check_source(
    shape: KernelShape,
    primal_patch_capacity: int = DEFAULT_PRIMAL_PATCH_CAPACITY,
    partial_patch_capacity: int = DEFAULT_PARTIAL_PATCH_CAPACITY,
) -> str:
    if shape.ast_count != 2 or shape.ast_leaf_counts != (8, 8):
        raise ValueError("the initial GPU gradient check expects two eight-leaf sites")
    primal = render_primal_site(shape, primal_patch_capacity)
    partial = render_partial_site(shape, partial_patch_capacity)
    primal_stores = "\n".join(
        f"    output[(unsigned long long){site} * output_leading_dimension + point] = specific_growth_{site + 1};"
        for site in range(shape.ast_count)
    )
    partial_stores: list[str] = []
    output = shape.ast_count
    for site, leaf_count in enumerate(shape.ast_leaf_counts):
        for leaf in range(leaf_count):
            partial_stores.append(
                f"    output[(unsigned long long){output} * output_leading_dimension + point] = "
                f"leaf_partials_{site + 1}[{leaf}];"
            )
            output += 1
    return f'''extern "C" __global__
void {KERNEL_NAME}(
    const float* __restrict__ input,
    unsigned long long input_leading_dimension,
    unsigned int point_count,
    float* __restrict__ output,
    unsigned long long output_leading_dimension) {{
    extern __shared__ float marker_keepalive[];
    const unsigned int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float leaves[{shape.input_count}];
#pragma unroll
    for (unsigned int leaf = 0; leaf < {shape.input_count}u; ++leaf) {{
        leaves[leaf] = input[(unsigned long long)leaf * input_leading_dimension + point];
    }}
{primal}
{partial}
{primal_stores}
{chr(10).join(partial_stores)}
    if (marker_keepalive == nullptr) output[point] = 0.0f;
}}
'''


def _bounded(value: Expression) -> Expression:
    return operation(InstructionType.TANH_F32, value)


def _positive(value: Expression) -> Expression:
    bounded = _bounded(value)
    return bounded * bounded + 0.75


def _random_leaf(random_source: random.Random, inputs: tuple[int, ...]) -> Expression:
    if random_source.random() < 0.84:
        return input_slot(random_source.choice(inputs))
    return constant(random_source.choice((-1.0, -0.5, -0.125, 0.125, 0.5, 1.0)))


def _random_subexpression(
    random_source: random.Random,
    inputs: tuple[int, ...],
    depth: int,
) -> Expression:
    if depth <= 0 or random_source.random() < 0.24:
        return _random_leaf(random_source, inputs)
    kind = random_source.randrange(7)
    if kind == 0:
        return 0.5 * (
            _random_subexpression(random_source, inputs, depth - 1)
            + _random_subexpression(random_source, inputs, depth - 1)
        )
    if kind == 1:
        return 0.5 * (
            _random_subexpression(random_source, inputs, depth - 1)
            - _random_subexpression(random_source, inputs, depth - 1)
        )
    if kind == 2:
        return (
            _bounded(_random_subexpression(random_source, inputs, depth - 1))
            * _bounded(_random_subexpression(random_source, inputs, depth - 1))
        )
    if kind == 3:
        return operation(
            InstructionType.SIN_F32,
            _random_subexpression(random_source, inputs, depth - 1),
        )
    if kind == 4:
        return operation(
            InstructionType.COS_F32,
            _random_subexpression(random_source, inputs, depth - 1),
        )
    if kind == 5:
        return _bounded(_random_subexpression(random_source, inputs, depth - 1))
    return operation(
        InstructionType.FMA_F32,
        0.5 * _bounded(_random_subexpression(random_source, inputs, depth - 1)),
        _bounded(_random_subexpression(random_source, inputs, depth - 1)),
        0.25 * _random_subexpression(random_source, inputs, depth - 1),
    )


def _random_expression(
    random_source: random.Random,
    root: InstructionType,
    inputs: tuple[int, ...],
) -> Expression:
    first = _random_subexpression(random_source, inputs, 2)
    second = _random_subexpression(random_source, inputs, 2)
    third = _random_subexpression(random_source, inputs, 2)
    if root in {InstructionType.ADD_F32, InstructionType.SUB_F32, InstructionType.MUL_F32}:
        result = operation(root, first, second)
    elif root == InstructionType.DIV_F32:
        result = operation(root, first, _positive(second))
    elif root in {
        InstructionType.SQRT_F32,
        InstructionType.RCP_F32,
        InstructionType.LG2_F32,
        InstructionType.RSQRT_F32,
        InstructionType.LOG_F32,
    }:
        result = operation(root, _positive(first))
    elif root in {InstructionType.EX2_F32, InstructionType.EXP_F32}:
        result = operation(root, 0.5 * _bounded(first))
    elif root == InstructionType.FMA_F32:
        result = operation(root, first, second, third)
    else:
        result = operation(root, first)
    repeated = input_slot(inputs[random_source.randrange(len(inputs))])
    return result + 0.125 * repeated * repeated


def random_genome(random_source: random.Random, shape: KernelShape, case: int) -> SystemGenome:
    programs: list[Program] = []
    for site, (offset, leaf_count) in enumerate(
        zip(shape.ast_input_offsets, shape.ast_leaf_counts)
    ):
        root = SMOOTH_OPERATIONS[(case * shape.ast_count + site) % len(SMOOTH_OPERATIONS)]
        expression = _random_expression(
            random_source,
            root,
            tuple(range(offset, offset + leaf_count)),
        )
        programs.append(Program.from_expression(expression))
    return SystemGenome(tuple(programs))


def _error(observed: float, expected: float) -> tuple[float, float]:
    absolute = abs(observed - expected)
    relative = absolute / max(1.0e-6, abs(expected))
    return absolute, relative


def run_random_gradient_check(
    architecture: str,
    genome_count: int = 64,
    points_per_genome: int = 32,
    seed: int = 0x51A55,
    nvcc: str | None = None,
    shape: KernelShape = KernelShape(),
    threads_per_block: int = 128,
) -> GradientCheckResult:
    if genome_count <= 0 or points_per_genome <= 0 or threads_per_block <= 0:
        raise ValueError("gradient-check dimensions must be positive")
    source = render_gradient_check_source(shape)
    compilation = compile_cuda(source, architecture, nvcc)
    random_source = random.Random(seed)
    output_count = shape.ast_count + shape.input_count
    input_host = array("f", [0.0]) * (shape.input_count * points_per_genome)
    output_host = array("f", [0.0]) * (output_count * points_per_genome)

    driver = Driver()
    driver.check(driver.library.cuInit(0), "cuInit")
    device = ctypes.c_int()
    driver.check(driver.library.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    driver.check(driver.library.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
    input_device = ctypes.c_uint64()
    output_device = ctypes.c_uint64()
    driver.check(
        driver.library.cuMemAlloc_v2(ctypes.byref(input_device), len(input_host) * 4),
        "cuMemAlloc(input)",
    )
    driver.check(
        driver.library.cuMemAlloc_v2(ctypes.byref(output_device), len(output_host) * 4),
        "cuMemAlloc(output)",
    )

    maximum_primal_absolute = 0.0
    maximum_primal_relative = 0.0
    maximum_partial_absolute = 0.0
    maximum_partial_relative = 0.0
    maximum_finite_absolute = 0.0
    maximum_finite_relative = 0.0
    maximum_register_count = 0
    maximum_primal_instructions = 0
    maximum_partial_instructions = 0
    comparisons = 0
    try:
        for case in range(genome_count):
            genome = random_genome(random_source, shape, case)
            specialization, bundle = specialize_split_lm_cubin(
                compilation.cubin,
                genome,
                shape,
                kernel_name=KERNEL_NAME,
            )
            maximum_register_count = max(maximum_register_count, specialization.register_count)
            maximum_primal_instructions = max(
                maximum_primal_instructions,
                sum(specialization.primal_instruction_counts),
            )
            maximum_partial_instructions = max(
                maximum_partial_instructions,
                sum(specialization.partial_instruction_counts),
            )
            for point in range(points_per_genome):
                for leaf in range(shape.input_count):
                    input_host[leaf * points_per_genome + point] = random_source.uniform(-0.75, 0.75)
            driver.check(
                driver.library.cuMemcpyHtoD_v2(
                    input_device.value,
                    _host_address(input_host),
                    len(input_host) * 4,
                ),
                "cuMemcpyHtoD(input)",
            )
            module = ctypes.c_void_p()
            image = ctypes.create_string_buffer(specialization.cubin)
            driver.check(
                driver.library.cuModuleLoadData(ctypes.byref(module), image),
                "cuModuleLoadData",
            )
            try:
                function = ctypes.c_void_p()
                driver.check(
                    driver.library.cuModuleGetFunction(
                        ctypes.byref(function), module, KERNEL_NAME.encode()
                    ),
                    "cuModuleGetFunction",
                )
                holders = [
                    ctypes.c_uint64(input_device.value),
                    ctypes.c_uint64(points_per_genome),
                    ctypes.c_uint32(points_per_genome),
                    ctypes.c_uint64(output_device.value),
                    ctypes.c_uint64(points_per_genome),
                ]
                parameters = (ctypes.c_void_p * len(holders))(
                    *(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders)
                )
                blocks = (points_per_genome + threads_per_block - 1) // threads_per_block
                driver.check(
                    driver.library.cuLaunchKernel(
                        function,
                        blocks,
                        1,
                        1,
                        threads_per_block,
                        1,
                        1,
                        4,
                        None,
                        parameters,
                        None,
                    ),
                    "cuLaunchKernel",
                )
                driver.check(driver.library.cuCtxSynchronize(), "cuCtxSynchronize")
                driver.check(
                    driver.library.cuMemcpyDtoH_v2(
                        _host_address(output_host),
                        output_device.value,
                        len(output_host) * 4,
                    ),
                    "cuMemcpyDtoH(output)",
                )
            finally:
                driver.library.cuModuleUnload(module)

            for point in range(points_per_genome):
                inputs = tuple(
                    float(input_host[leaf * points_per_genome + point])
                    for leaf in range(shape.input_count)
                )
                partial_output = shape.ast_count
                for site, derivative_site in enumerate(bundle.sites):
                    expected_primal = derivative_site.primal.evaluate_validated(inputs)
                    observed_primal = float(output_host[site * points_per_genome + point])
                    absolute, relative = _error(observed_primal, expected_primal)
                    maximum_primal_absolute = max(maximum_primal_absolute, absolute)
                    maximum_primal_relative = max(maximum_primal_relative, relative)
                    if absolute > 3.0e-5 + 3.0e-4 * abs(expected_primal):
                        raise AssertionError(
                            f"GPU primal mismatch case={case} point={point} site={site}: "
                            f"observed={observed_primal:.9g} expected={expected_primal:.9g}"
                        )
                    comparisons += 1
                    for local_leaf, partial_program in enumerate(derivative_site.partials):
                        observed = float(
                            output_host[partial_output * points_per_genome + point]
                        )
                        expected = partial_program.evaluate_validated(inputs)
                        absolute, relative = _error(observed, expected)
                        maximum_partial_absolute = max(maximum_partial_absolute, absolute)
                        maximum_partial_relative = max(maximum_partial_relative, relative)
                        if absolute > 8.0e-5 + 2.0e-3 * abs(expected):
                            raise AssertionError(
                                f"GPU partial mismatch case={case} point={point} site={site} "
                                f"leaf={local_leaf}: observed={observed:.9g} expected={expected:.9g}"
                            )
                        global_leaf = derivative_site.leaf_inputs[local_leaf]
                        epsilon = 1.0e-5 * max(1.0, abs(inputs[global_leaf]))
                        positive = list(inputs)
                        negative = list(inputs)
                        positive[global_leaf] += epsilon
                        negative[global_leaf] -= epsilon
                        finite_difference = (
                            derivative_site.primal.evaluate_validated(positive)
                            - derivative_site.primal.evaluate_validated(negative)
                        ) / (2.0 * epsilon)
                        finite_absolute, finite_relative = _error(observed, finite_difference)
                        maximum_finite_absolute = max(maximum_finite_absolute, finite_absolute)
                        maximum_finite_relative = max(maximum_finite_relative, finite_relative)
                        if finite_absolute > 1.2e-4 + 3.0e-3 * abs(finite_difference):
                            raise AssertionError(
                                f"GPU finite-difference mismatch case={case} point={point} "
                                f"site={site} leaf={local_leaf}: observed={observed:.9g} "
                                f"finite={finite_difference:.9g}"
                            )
                        comparisons += 2
                        partial_output += 1
    finally:
        if output_device.value:
            driver.library.cuMemFree_v2(output_device.value)
        if input_device.value:
            driver.library.cuMemFree_v2(input_device.value)
        driver.library.cuCtxDestroy_v2(context)

    return GradientCheckResult(
        architecture,
        seed,
        genome_count,
        points_per_genome,
        comparisons,
        maximum_primal_absolute,
        maximum_primal_relative,
        maximum_partial_absolute,
        maximum_partial_relative,
        maximum_finite_absolute,
        maximum_finite_relative,
        maximum_register_count,
        maximum_primal_instructions,
        maximum_partial_instructions,
        compilation.elapsed_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare randomized split-SASS primal/partials with CPU and finite differences"
    )
    parser.add_argument("--arch", required=True)
    parser.add_argument("--genomes", type=int, default=64)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x51A55)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--nvcc")
    arguments = parser.parse_args(argv)
    result = run_random_gradient_check(
        arguments.arch,
        arguments.genomes,
        arguments.points,
        arguments.seed,
        arguments.nvcc,
        threads_per_block=arguments.threads,
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
