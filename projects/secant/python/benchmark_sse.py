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
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import secant
from secant.routines import DEFAULT_ROUTINES


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_PATHS = {
    "alu": _REPO_ROOT / "bench" / "corpus" / "portable_alu_v1.json",
    "mufu": _REPO_ROOT / "bench" / "corpus" / "portable_mufu_v1.json",
}


def _expression(node):
    if isinstance(node, int):
        return secant.input(node)
    operation = node[0]
    if operation == "constant":
        return secant.constant(node[1])
    arguments = tuple(_expression(argument) for argument in node[1:])
    if operation == "add":
        return arguments[0] + arguments[1]
    if operation == "sub":
        return arguments[0] - arguments[1]
    if operation == "mul":
        return arguments[0] * arguments[1]
    if operation == "div":
        return arguments[0] / arguments[1]
    if operation == "min":
        return secant.minimum(*arguments)
    if operation == "max":
        return secant.maximum(*arguments)
    if operation == "safe_div":
        return secant.safe_div(*arguments)
    if operation == "neg":
        return -arguments[0]
    if operation == "abs":
        return secant.absolute(arguments[0])
    if operation == "sin":
        return secant.sin(arguments[0])
    if operation == "cos":
        return secant.cos(arguments[0])
    if operation == "ex2":
        return secant.exp2(arguments[0])
    if operation == "lg2":
        return secant.log2(arguments[0])
    if operation == "sqrt":
        return secant.sqrt(arguments[0])
    if operation == "rsqrt":
        return secant.rsqrt(arguments[0])
    if operation == "tanh":
        return secant.tanh(arguments[0])
    if operation == "safe_sqrt":
        return secant.safe_sqrt(arguments[0])
    if operation == "safe_rsqrt":
        return secant.safe_rsqrt(arguments[0])
    raise ValueError(f"unsupported corpus operation {operation!r}")


def _corpus_programs(mode: str) -> tuple[secant.Program, ...]:
    with _CORPUS_PATHS[mode].open(encoding="utf-8") as file:
        corpus = json.load(file)
    return tuple(secant.Program(_expression(expression)) for expression in corpus["expressions"])


def _expand_programs(programs: tuple[secant.Program, ...], count: int) -> tuple[secant.Program, ...]:
    return tuple(programs[index % len(programs)] for index in range(count))


def _expected_sse(
    unique_programs: tuple[secant.Program, ...],
    total_programs: int,
    columns: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    unique_output = secant.sse(unique_programs, columns, targets, routines=DEFAULT_ROUTINES)
    indices = np.arange(total_programs) % len(unique_programs)
    return unique_output[indices]


def _run_mode(
    mode: str,
    template: secant.CompiledTemplate,
    columns: np.ndarray,
    targets: np.ndarray,
    check_columns: np.ndarray,
    check_targets: np.ndarray,
    num_streams: int,
    warmups: int,
    iterations: int,
) -> None:
    unique_programs = _corpus_programs(mode)
    programs = _expand_programs(unique_programs, template.num_asts)

    start = perf_counter()
    cubin = template.specialize(programs, routines=DEFAULT_ROUTINES)
    specialize_seconds = perf_counter() - start

    start = perf_counter()
    module = secant.SSEModule(cubin, template.recipe)
    module_load_seconds = perf_counter() - start

    expected = _expected_sse(unique_programs, template.num_asts, check_columns, check_targets)
    with module.resident(check_columns, check_targets, num_streams=num_streams) as check_execution:
        check = check_execution.run(warmups=1)
    np.testing.assert_allclose(check.output, expected, rtol=5.0e-4, atol=2.0e-2)

    with module.resident(columns, targets, num_streams=num_streams) as execution:
        result = execution.run(warmups=warmups, iterations=iterations)
        upload_seconds = execution.upload_seconds

    row_evals = template.num_asts * columns.shape[1] * iterations
    absolute_error = np.abs(check.output - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), np.float32(1.0))
    print(
        f"mode={mode} asts={template.num_asts} rows={columns.shape[1]} iterations={iterations} "
        f"streams={num_streams} specialize_seconds={specialize_seconds:.9f} "
        f"specialize_asts_per_second={template.num_asts / specialize_seconds:.3f} "
        f"module_load_seconds={module_load_seconds:.9f} upload_seconds={upload_seconds:.9f} "
        f"gpu_seconds={result.gpu_seconds:.9f} wall_seconds={result.wall_seconds:.9f} "
        f"gpu_row_evals_per_second={row_evals / result.gpu_seconds:.6e} "
        f"resident_wall_row_evals_per_second={row_evals / result.wall_seconds:.6e} "
        f"max_abs_error={np.max(absolute_error):.9g} max_relative_error={np.max(relative_error):.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark resident multi-stream Secant CUBIN SSE execution")
    parser.add_argument("--kernels", type=int, default=32)
    parser.add_argument("--asts-per-kernel", type=int, default=128)
    parser.add_argument("--rows", type=int, default=1_048_576)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--patch-capacity", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--check-rows", type=int, default=4097)
    parser.add_argument("--seed", type=int, default=0x5EC47)
    parser.add_argument("--mode", choices=("alu", "mufu", "both"), default="both")
    args = parser.parse_args()

    recipe = secant.SSERecipe(
        num_kernels=args.kernels,
        asts_per_kernel=args.asts_per_kernel,
        num_inputs=8,
        num_targets=args.targets,
        tile_rows=args.tile_rows,
        threads_per_block=args.threads,
        patch_capacity_instructions=args.patch_capacity,
    )

    start = perf_counter()
    template = secant.compile_sse(recipe)
    template_seconds = perf_counter() - start
    print(
        f"template arch={template.arch} compile_seconds={template_seconds:.9f} "
        f"source_bytes={len(template.source.encode('utf-8'))} cubin_bytes={len(template.cubin)} "
        f"kernels={recipe.num_kernels} asts_per_kernel={recipe.asts_per_kernel} "
        f"patch_capacity={recipe.patch_capacity_instructions}"
    )

    rng = np.random.default_rng(args.seed)
    columns = rng.uniform(0.5, 3.0, size=(recipe.num_inputs, args.rows)).astype(np.float32)
    targets = rng.uniform(-2.0, 2.0, size=(recipe.num_targets, args.rows)).astype(np.float32)
    check_columns = columns[:, : args.check_rows]
    check_targets = targets[:, : args.check_rows]

    modes = ("alu", "mufu") if args.mode == "both" else (args.mode,)
    for mode in modes:
        _run_mode(
            mode,
            template,
            columns,
            targets,
            check_columns,
            check_targets,
            args.streams,
            args.warmups,
            args.iterations,
        )


if __name__ == "__main__":
    main()
