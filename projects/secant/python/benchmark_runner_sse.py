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
from time import perf_counter

import numpy as np

import secant
from benchmark_programs import generate_unique_balanced_programs
from benchmark_sse import _corpus_programs
from secant.routines import DEFAULT_ROUTINES


def _pack_corpus(programs: tuple[secant.Program, ...], count: int) -> tuple[bytearray, np.ndarray]:
    return secant.pack_programs(programs[index % len(programs)] for index in range(count))


def _program_at(ast_data: bytearray, ast_offsets: np.ndarray, index: int) -> bytes:
    return bytes(ast_data[int(ast_offsets[index]) : int(ast_offsets[index + 1])])


def _check_indices(num_asts: int) -> np.ndarray:
    return np.unique(
        np.asarray(
            (0, 1, 1023, 1024, num_asts // 2, num_asts - 1),
            dtype=np.int64,
        )
    )


def _run_mode(
    mode: str,
    runner: secant.SSEBulkRunner,
    num_asts: int,
    columns: np.ndarray,
    targets: np.ndarray,
    ast_source: str,
    seed: int,
) -> None:
    start = perf_counter()
    if ast_source == "unique":
        ast_data, ast_offsets = generate_unique_balanced_programs(num_asts, mode, seed)
    else:
        corpus = _corpus_programs(mode)
        ast_data, ast_offsets = _pack_corpus(corpus, num_asts)
    pack_seconds = perf_counter() - start

    start = perf_counter()
    result = runner.run_all(
        ast_data,
        ast_offsets,
        columns,
        targets,
        routines=DEFAULT_ROUTINES,
    )
    python_wall_seconds = perf_counter() - start

    check_indices = _check_indices(num_asts)
    check_programs = tuple(_program_at(ast_data, ast_offsets, int(index)) for index in check_indices)
    expected = secant.sse(check_programs, columns, targets, routines=DEFAULT_ROUTINES)
    actual = result.output[check_indices]
    np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=2.0)

    stats = result.stats
    row_evals = num_asts * columns.shape[1]
    absolute_error = np.abs(actual - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), np.float32(1.0))
    print(
        f"mode={mode} ast_source={ast_source} asts={num_asts} packed_bytes={len(ast_data)} "
        f"offsets_bytes={ast_offsets.nbytes} "
        f"pack_seconds={pack_seconds:.9f} pack_asts_per_second={num_asts / pack_seconds:.3f} "
        f"modules={stats['num_modules']} modules_loaded={stats['modules_loaded']} "
        f"compile_window_seconds={stats['compile_window_seconds']:.9f} "
        f"compile_asts_per_second={num_asts / stats['compile_window_seconds']:.3f} "
        f"compile_critical_seconds={stats['compile_critical_seconds']:.9f} "
        f"compile_critical_asts_per_second={num_asts / stats['compile_critical_seconds']:.3f} "
        f"compile_work_seconds={stats['compile_work_seconds']:.9f} "
        f"module_load_seconds={stats['module_load_seconds']:.9f} "
        f"completion_wait_seconds={stats['completion_wait_seconds']:.9f} "
        f"module_unload_seconds={stats['module_unload_seconds']:.9f} runtime_seconds={stats['runtime_seconds']:.9f} "
        f"pipeline_seconds={stats['total_seconds']:.9f} python_wall_seconds={python_wall_seconds:.9f} "
        f"runtime_row_evals_per_second={row_evals / stats['runtime_seconds']:.6e} "
        f"pipeline_row_evals_per_second={row_evals / stats['total_seconds']:.6e} "
        f"python_wall_row_evals_per_second={row_evals / python_wall_seconds:.6e} "
        f"max_abs_error={np.max(absolute_error):.9g} max_relative_error={np.max(relative_error):.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run large packed AST byte arrays through the native CUBIN SSE runner")
    parser.add_argument("--modules", type=int, default=128)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--kernels", type=int, default=32)
    parser.add_argument("--asts-per-kernel", type=int, default=128)
    parser.add_argument("--rows", type=int, default=16_384)
    parser.add_argument("--tile-rows", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--patch-capacity", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0x5EC47)
    parser.add_argument("--mode", choices=("alu", "mufu", "both"), default="both")
    parser.add_argument("--ast-source", choices=("unique", "portable"), default="unique")
    args = parser.parse_args()

    recipe = secant.SSERecipe(
        num_kernels=args.kernels,
        asts_per_kernel=args.asts_per_kernel,
        num_inputs=8,
        num_targets=1,
        tile_rows=args.tile_rows,
        threads_per_block=args.threads,
        patch_capacity_instructions=args.patch_capacity,
    )
    num_asts = args.modules * recipe.num_kernels * recipe.asts_per_kernel

    start = perf_counter()
    template = secant.compile_sse(recipe)
    template_seconds = perf_counter() - start

    rng = np.random.default_rng(args.seed)
    columns = rng.uniform(0.5, 3.0, size=(recipe.num_inputs, args.rows)).astype(np.float32)
    targets = rng.uniform(-2.0, 2.0, size=(recipe.num_targets, args.rows)).astype(np.float32)

    start = perf_counter()
    runner = secant.SSEBulkRunner(
        template,
        num_workers=args.workers,
        num_streams=args.streams,
    )
    runner_create_seconds = perf_counter() - start
    print(
        f"template arch={template.arch} compile_seconds={template_seconds:.9f} cubin_bytes={len(template.cubin)} "
        f"runner_create_seconds={runner_create_seconds:.9f} modules={args.modules} workers={args.workers} "
        f"streams={args.streams} kernels={recipe.num_kernels} asts_per_kernel={recipe.asts_per_kernel} "
        f"rows={args.rows}"
    )

    try:
        modes = ("alu", "mufu") if args.mode == "both" else (args.mode,)
        for mode in modes:
            _run_mode(mode, runner, num_asts, columns, targets, args.ast_source, args.seed)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
